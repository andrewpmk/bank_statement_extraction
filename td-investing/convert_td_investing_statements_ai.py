#!/usr/bin/env python3
"""Convert TD Direct Investing (TD Waterhouse) statement PDFs into CSVs using OpenRouter.

TD Direct Investing statements bundle several sub-accounts into one PDF (non-registered
CDN/US cash, Self-Directed RSP CDN/US, TFSA). The file format has changed several times
over the account's history (an old "TD Waterhouse President's Account" layout with an
"Activities" table using separate Charged/Credited columns, and a newer "TD Direct
Investing" layout with an "Activity in your account this period" table using a single
signed Amount column), and one physical PDF sometimes bundles several months of
statements concatenated together (observed historically as one PDF per year, one PDF per
4-month chunk, and one PDF per quarter, in addition to one PDF per month). December
statements sometimes append an extra annual "performance report" / "fees and charges
report" / disclosures per account.

None of that is assumed from the filename or folder: every page's statement period
(month/year) and layout are determined from the page's own text content, so a file can be
processed correctly regardless of what it happens to be named or how many months/accounts
it bundles. Filenames are only used to discover candidate statement files.

Only files matching TDWaterhouse*.pdf are considered, excluding annual tax-summary files
like TDWaterhouse_Tax2015.pdf (these contain a SIN, not statement data); Tax-Document*
slips, trade-summary CSVs, and other non-statement files under td-investing/ are ignored.

Each statement page is classified (via extracted text, not the AI) into one of 5 canonical
accounts - Canadian Cash, US Cash, Canadian RRSP, USA RRSP, Canada TFSA - or excluded as a
boundary page (disclosures, performance/fees reports, address filler, overall summary).
The page's own statement-period text (e.g. "Statement for January 1 to January 31, 2012"
or "June 1, 2025 to June 30, 2025") is read only from pages that pass this classification,
so an unrelated date range on an excluded page (e.g. an annual performance report's
Jan-Dec range) can never be mistaken for the statement's own month. Contiguous pages for
the same (period, account) are grouped into a segment; each segment's pages are rendered
as images and sent to a single OpenRouter vision call that returns both the holdings and
the period's transactions as JSON.

For each statement period this produces two combined (all-accounts) CSVs next to the
source PDFs:
  td_investing_balances_YYYYMM.csv     - Year,Month,Account,Currency,Holding,Quantity,Price,MarketValue,BookCost,UnrealizedGainLoss
  td_investing_transactions_YYYYMM.csv - Year,Month,Account,Currency,Date,Activity,Description,Quantity,Price,Amount,Commission,Balance

Default model: openai/gpt-4o. Override with --model if needed.
"""

from __future__ import annotations

import argparse
import base64
import csv
import io
import json
import os
import re
import sys
import urllib.error
import urllib.request
from dataclasses import dataclass, field
from pathlib import Path

try:
    import pypdfium2 as pdfium
except Exception:
    pdfium = None

try:
    import pdfplumber
except Exception:
    pdfplumber = None

MODEL = "openai/gpt-4o"

MONTH_ABBR_TO_NUM = {
    "JAN": 1, "FEB": 2, "MAR": 3, "APR": 4, "MAY": 5, "JUN": 6,
    "JUL": 7, "AUG": 8, "SEP": 9, "SEPT": 9, "OCT": 10, "NOV": 11, "DEC": 12,
}

MONTH_TOKEN = r"[A-Za-z]{3,9}\.?"

# Statement period is detected from page content, never from the filename or folder name -
# statements get filed under whatever year/month the user happens to save them as, and old-format
# PDFs bundle several months in one file, so the filename cannot be trusted either way. Patterns
# are tried in order on each account-content page (never on excluded boundary pages - see
# classify_account - so an annual performance-report date range can't be mistaken for the
# statement's own month):
#   1. Old format running header: "Statement for January 1 to January 31, 2012"
#   2. New format's first-page range: "June 1, 2025 to June 30, 2025"
#   3. New format's continuation-page header: "Your investment account statement: Jun 30, 2025"
#   4. Generic fallback: "... on June 30, 2025" (e.g. "Holdings in your account ... on June 30, 2025")
#   5. Earliest (2006-2007) layout, whose PDF text extraction often drops spaces entirely:
#      "AUG1,2006TOAUG31,2006" - same "<date> TO <date>" shape as #2 but tolerant of zero
#      whitespace anywhere in the token.
PERIOD_PATTERNS = [
    re.compile(
        rf"Statement\s+for\s+{MONTH_TOKEN}\s+\d{{1,2}}\s+to\s+(?P<month>{MONTH_TOKEN})\s+\d{{1,2}},\s*(?P<year>\d{{4}})",
        re.IGNORECASE,
    ),
    re.compile(
        rf"{MONTH_TOKEN}\s+\d{{1,2}},\s*\d{{4}}\s+to\s+(?P<month>{MONTH_TOKEN})\s+\d{{1,2}},\s*(?P<year>\d{{4}})",
        re.IGNORECASE,
    ),
    re.compile(
        rf"(?:account statement|RSP statement|TFSA statement)\s*:\s*(?P<month>{MONTH_TOKEN})\s+\d{{1,2}},\s*(?P<year>\d{{4}})",
        re.IGNORECASE,
    ),
    re.compile(
        rf"\bon\s+(?P<month>{MONTH_TOKEN})\s+\d{{1,2}},\s*(?P<year>\d{{4}})\b",
        re.IGNORECASE,
    ),
    re.compile(
        rf"{MONTH_TOKEN}\s*\d{{1,2}},\s*\d{{4}}\s*TO\s*(?P<month>{MONTH_TOKEN})\s*\d{{1,2}},\s*(?P<year>\d{{4}})",
        re.IGNORECASE,
    ),
]

# Boundary phrases: pages containing any of these are never account content, even if they
# also happen to contain "Account type:"/"Account number:" text (performance & fee reports
# and the address/marketing filler page repeat the account header).
BOUNDARY_PHRASES = (
    "your performance report",
    "your fees and charges report",
    "avoid delays in your money getting to you",
    "for the period ending",
)

ACCOUNT_TYPE_RE = re.compile(r"Account type:\s*([^\n]+)", re.IGNORECASE)
OLD_ACCOUNT_HEADER_RE = re.compile(r"^TD Waterhouse President'?s ?Account\s*(.*)$", re.IGNORECASE)
# Earliest (2006-2007) layout: first page header is the bare word "ACCOUNT" with the account
# type on the next line (e.g. "DIRECTTRADING-CDN"); continuation pages repeat both on one line
# ("ACCOUNT ACCOUNT NUMBER STATEMENT PERIOD PAGE" / "DIRECTTRADING-CDN 510H45 ..."). Either way
# the type token is the first whitespace-separated word of the second line.
EARLIEST_ACCOUNT_HEADER_RE = re.compile(r"^ACCOUNT(\s+ACCOUNT\s+NUMBER\b.*)?$", re.IGNORECASE)
# Transitional (2015-2016) layout: no "Account type:" label and no "TD Waterhouse President's
# Account" prefix - the header line IS the bare account-type label itself, optionally followed by
# " - <account number>" on continuation pages (e.g. "Direct Trading - CDN", "Direct Trading - CDN -
# 510H45", "Tax-Free Savings Account - CDN* - 510H45-J").
BARE_LABEL_ACCOUNT_HEADER_RE = re.compile(
    r"^(Direct Trading|Self-Directed RSP|Self-Directed RRSP|Self-Directed RIF|Tax-Free Savings Account|Tax Free Savings Account)\b.*$",
    re.IGNORECASE,
)


@dataclass(frozen=True)
class Period:
    year: int
    month: int


@dataclass
class Segment:
    period: Period
    account: str
    currency: str
    pages: list[int] = field(default_factory=list)


CANONICAL_ACCOUNTS = {
    ("Cash", "CAD"): "Canadian Cash",
    ("Cash", "USD"): "US Cash",
    ("RRSP", "CAD"): "Canadian RRSP",
    ("RRSP", "USD"): "USA RRSP",
    ("TFSA", "CAD"): "Canada TFSA",
}


def load_dotenv(dotenv_path: Path) -> None:
    if not dotenv_path.exists():
        return
    for line in dotenv_path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, value = line.split("=", 1)
        key = key.strip()
        if not key or key in os.environ:
            continue
        value = value.strip().strip("\"'")
        os.environ[key] = value


def is_statement_pdf(path: Path) -> bool:
    name = path.name.lower()
    if not name.startswith("tdwaterhouse") or path.suffix.lower() != ".pdf":
        return False
    # "TDWaterhouse_Tax2015.pdf" / "TDWaterhouse_Tax2015-trading.pdf" are annual tax-summary
    # documents (containing a SIN), not monthly/quarterly statements - exclude like Tax-Document_*.
    if re.search(r"tax\d{4}", name):
        return False
    return True


def find_pdfs(root: Path) -> list[Path]:
    if root.is_file():
        return [root] if is_statement_pdf(root) else []
    return sorted(p for p in root.rglob("*.pdf") if p.is_file() and is_statement_pdf(p))


def extract_page_texts(pdf_path: Path) -> list[str]:
    if pdfplumber is None:
        raise RuntimeError("Missing dependency pdfplumber. Install requirements before running.")
    with pdfplumber.open(pdf_path) as pdf:
        return [p.extract_text() or "" for p in pdf.pages]


def classify_account(text: str) -> tuple[str, str] | None:
    """Return (category, currency) for an account-content page, or None for a boundary page."""
    lower = text.lower()
    if any(phrase in lower for phrase in BOUNDARY_PHRASES):
        return None

    lines = text.splitlines()
    first_line = lines[0].strip() if lines else ""
    if first_line.strip().lower() == "disclosures":
        return None

    type_label = None
    m = ACCOUNT_TYPE_RE.search(text)
    if m:
        type_label = m.group(1)
    else:
        m2 = OLD_ACCOUNT_HEADER_RE.match(first_line)
        if m2:
            type_label = m2.group(1)
        elif BARE_LABEL_ACCOUNT_HEADER_RE.match(first_line):
            type_label = first_line
        elif EARLIEST_ACCOUNT_HEADER_RE.match(first_line):
            second_line = lines[1].strip() if len(lines) > 1 else ""
            type_label = second_line.split()[0] if second_line else None

    if not type_label:
        return None

    label_up = type_label.upper()
    if "TFSA" in label_up or "TAX-FREE" in label_up:
        category = "TFSA"
    elif re.search(r"\bR[R]?SP\b", label_up) or "LIRA" in label_up or "\bRIF\b" in label_up:
        category = "RRSP"
    else:
        category = "Cash"

    currency = "USD" if re.search(r"\bUS\b", label_up) else "CAD"
    return category, currency


def month_name_to_num(name: str) -> int | None:
    abbr = re.sub(r"[^A-Z]", "", name.upper())[:3]
    return MONTH_ABBR_TO_NUM.get(abbr)


def detect_period(text: str) -> Period | None:
    for pattern in PERIOD_PATTERNS:
        m = pattern.search(text)
        if not m:
            continue
        month_num = month_name_to_num(m.group("month"))
        if month_num:
            return Period(year=int(m.group("year")), month=month_num)
    return None


def build_segments(pdf_path: Path, page_texts: list[str]) -> list[Segment]:
    segments: list[Segment] = []
    current: Segment | None = None
    current_period: Period | None = None

    for i, text in enumerate(page_texts):
        info = classify_account(text)
        if info is None:
            if current is not None:
                segments.append(current)
                current = None
            continue

        # Only ever read the period off a page that already passed classify_account, i.e. real
        # account content - never off an excluded boundary page (this is what keeps an annual
        # performance-report's Jan-Dec date range from clobbering the statement's own month).
        detected = detect_period(text)
        if detected is not None:
            current_period = detected

        if current_period is None:
            # Can't place this page's content in a period yet; drop it rather than guess.
            continue

        category, currency = info
        if (
            current is not None
            and current.period == current_period
            and current.account == category
            and current.currency == currency
        ):
            current.pages.append(i)
        else:
            if current is not None:
                segments.append(current)
            current = Segment(period=current_period, account=category, currency=currency, pages=[i])

    if current is not None:
        segments.append(current)

    # Merge any non-contiguous segments that share the same (period, account, currency).
    merged: dict[tuple[Period, str, str], Segment] = {}
    order: list[tuple[Period, str, str]] = []
    for seg in segments:
        key = (seg.period, seg.account, seg.currency)
        if key not in merged:
            merged[key] = Segment(period=seg.period, account=seg.account, currency=seg.currency, pages=list(seg.pages))
            order.append(key)
        else:
            merged[key].pages.extend(seg.pages)

    result = []
    for key in order:
        seg = merged[key]
        seg.pages = sorted(set(seg.pages))
        result.append(seg)
    return result


def render_pages_as_data_urls(pdf_path: Path, page_indices: list[int], dpi: int = 300) -> list[str]:
    if pdfium is None:
        raise RuntimeError("Missing dependency pypdfium2. Install requirements before running AI image extraction.")

    data_urls: list[str] = []
    scale = dpi / 72.0

    doc = pdfium.PdfDocument(str(pdf_path))
    try:
        for page_index in page_indices:
            page = doc[page_index]
            try:
                bitmap = page.render(scale=scale)
                try:
                    image = bitmap.to_pil()
                    buf = io.BytesIO()
                    image.save(buf, format="PNG")
                    encoded = base64.b64encode(buf.getvalue()).decode("ascii")
                    data_urls.append(f"data:image/png;base64,{encoded}")
                finally:
                    bitmap.close()
            finally:
                page.close()
    finally:
        doc.close()

    if not data_urls:
        raise RuntimeError(f"No pages rendered from PDF: {pdf_path}")

    return data_urls


def build_prompt(period: Period, account_label: str, currency: str) -> str:
    # Fixed instruction text first and the per-call variables (account/currency/period) appended
    # at the very end, so every call shares one identical, cacheable prompt prefix.
    rules = (
        "You are given page images from a TD Direct Investing statement. Each image is preceded by "
        "the PDF's own raw extracted text for that same page - those characters are always exactly "
        "correct (it is not OCR), but the extractor's reading order and line-wrapping can jumble the "
        "table layout, so use the image to see which numbers/ticker belong to which row and column, "
        "and use the raw text to double check the exact digits and ticker letters, especially small "
        "print like ticker symbols in parentheses. Where the image and raw text disagree on a "
        "character, trust the raw text. "
        "Extract two things and return ONLY a JSON object with keys 'holdings' and 'transactions'.\n\n"
        "holdings: one entry per row in the 'Holdings in your account' (or 'Holdings') table(s), "
        "including continuation pages. Include the plain 'Cash' row as its own holding (Holding='Cash', "
        "Quantity and Price blank, MarketValue/BookCost equal to the cash amount). Include money market "
        "fund rows such as 'ISA-TDB@x% PA /NL (TDB8150C)' or '(TDB8152C)' as ordinary holdings exactly "
        "like any other security (they have Price 10.000) - do not merge them into the Cash row. "
        "A holding's name sometimes wraps onto the next printed line with the numeric columns sandwiched "
        "in between (e.g. 'BMO S&P/TSX CAPP <numbers>' then 'COMP ETF (ZCN)' on the next line); join the "
        "wrapped name into one Holding string in reading order, e.g. 'BMO S&P/TSX CAPP COMP ETF (ZCN)'. "
        "Do NOT include section headers (e.g. 'Equities', 'Cash & cash equivalents', 'Canadian common "
        "shares & equivalents') or 'Total ...' subtotal rows as holdings. "
        "The Quantity column sometimes has a 'Seg' marker printed between the number and the Holding's "
        "wrapped name (e.g. '2,500 SEG 35.890') - that marker denotes segregated custody, it is not part "
        "of the quantity or the name; drop it from both. Quantity must be a plain number only. "
        "Each holdings object must have EXACT keys: Holding, Quantity, Price, MarketValue, BookCost, "
        "UnrealizedGainLoss. Leave a key blank ('') if the statement does not show it for that row.\n\n"
        "transactions: one entry per row in the 'Activity in your account this period' (or 'Activities', "
        "or 'Transactions during period') table for THIS account only. Do NOT include rows from a "
        "'Pending activity in your account this period' section (trades not yet settled) - skip that "
        "section entirely. Do NOT include 'Beginning/Ending cash balance' or 'Cash-opening/closing "
        "balance'/'Cash - opening/closing balance' bookkeeping rows. "
        "The oldest-format statements have no plain-English Activity column at all - instead they print "
        "separate 'Bought or Received'/'Sold or Delivered' quantity columns and a short reference code "
        "(e.g. 'REC99', 'SFK38', 'WBD') in place of Activity; in that case use the reference code as "
        "Activity, and make Quantity positive for a Bought/Received row or negative for a Sold/Delivered "
        "row. Still include these rows even when Debit and Credit are both 0.00 (e.g. an in-kind "
        "transfer-in of existing shares) - do not skip a row just because it has no cash impact. "
        "Join any wrapped continuation text (extra notes like 'WE ARE RELATED TO ISSUER...' or a transfer "
        "reference like 'TSF TO 6337953') into the Description of the row above it. "
        "Older-format statements print two separate columns instead of one signed Amount - labelled "
        "either 'Charged'/'Credited' or 'Debit'/'Credit' - with exactly one of the two populated per "
        "row; in that case set Amount to the Credited/Credit value, or to the negative of the "
        "Charged/Debit value. Newer-format statements already print one signed Amount "
        "column (negative for Buy/Withdrawal/Withholding Tax, positive for Sell/Distribution/Dividend/"
        "Deposit) - use it unchanged. "
        "If a row shows an explicit commission/fee charge separate from Amount, put it in Commission, "
        "otherwise leave Commission blank. Balance is the running cash balance printed after the row. "
        "Date should be exactly as printed (e.g. 'Jun 30' or 'Jan 03'), without a year. "
        "Each transactions object must have EXACT keys: Date, Activity, Description, Quantity, Price, "
        "Amount, Commission, Balance.\n\n"
        "Return ONLY the JSON object, no surrounding text.\n\n"
        f"Account: {account_label}\nCurrency: {currency}\nStatement period: {period.year:04d}-{period.month:02d}"
    )
    return rules


def build_user_content(prompt: str, page_data_urls: list[str], page_raw_texts: list[str]) -> list[dict[str, object]]:
    content: list[dict[str, object]] = [{"type": "text", "text": prompt}]
    for idx, (data_url, raw_text) in enumerate(zip(page_data_urls, page_raw_texts), start=1):
        content.append({
            "type": "text",
            "text": (
                f"Statement page {idx}. Raw text extracted directly from the PDF for this same "
                "page (character-accurate, but reading order/line-wrapping may be jumbled - "
                "cross-reference it against the image below to resolve any digit or ticker you "
                "are unsure of from the image alone, but use the image for row/column structure):\n"
                f"{raw_text}"
            ),
        })
        content.append({"type": "image_url", "image_url": {"url": data_url}})
    return content


def call_openrouter(model: str, api_key: str, prompt: str, page_data_urls: list[str], page_raw_texts: list[str], timeout_s: int) -> str:
    url = "https://openrouter.ai/api/v1/chat/completions"
    user_content = build_user_content(prompt, page_data_urls, page_raw_texts)
    body = {
        "model": model,
        "messages": [
            {"role": "system", "content": "You output strict JSON only."},
            {"role": "user", "content": user_content},
        ],
        "temperature": 0,
    }

    req = urllib.request.Request(
        url,
        data=json.dumps(body).encode("utf-8"),
        headers={"Authorization": f"Bearer {api_key}", "Content-Type": "application/json"},
        method="POST",
    )

    try:
        with urllib.request.urlopen(req, timeout=timeout_s) as resp:
            data = json.loads(resp.read().decode("utf-8"))
    except urllib.error.HTTPError as exc:
        detail = exc.read().decode("utf-8", errors="replace")
        raise RuntimeError(f"OpenRouter HTTP {exc.code}: {detail}") from exc
    except urllib.error.URLError as exc:
        raise RuntimeError(f"OpenRouter request failed: {exc}") from exc

    try:
        return data["choices"][0]["message"]["content"]
    except (KeyError, IndexError, TypeError) as exc:
        raise RuntimeError(f"Unexpected OpenRouter response: {data}") from exc


def extract_json_object(text: str) -> dict[str, list]:
    text = text.strip()
    if text.startswith("{"):
        return json.loads(text)
    match = re.search(r"\{.*\}", text, re.DOTALL)
    if not match:
        raise ValueError("Model response did not contain a JSON object.")
    return json.loads(match.group(0))


def _field(row: dict[str, object], key: str) -> str:
    value = row.get(key)
    if value is None:
        return ""
    return str(value).strip()


def normalize_amount(value: str) -> str:
    value = value.strip()
    if not value:
        return ""
    value = value.replace(",", "").replace("$", "")
    value = re.sub(r"\bSEG\b", "", value, flags=re.IGNORECASE).strip()
    if value.startswith("(") and value.endswith(")"):
        value = "-" + value[1:-1]
    try:
        return f"{float(value):.3f}".rstrip("0").rstrip(".") if "." in value else value
    except ValueError:
        return value


def normalize_money(value: str) -> str:
    value = value.strip()
    if not value:
        return ""
    value = value.replace(",", "").replace("$", "")
    if value.startswith("(") and value.endswith(")"):
        value = "-" + value[1:-1]
    try:
        return f"{float(value):.2f}"
    except ValueError:
        return value


def canonical_key(value: str) -> str:
    return re.sub(r"[^A-Z0-9]", "", value.upper())


def clean_holdings_rows(ai_rows: list[dict[str, object]], period: Period, account_label: str, currency: str) -> list[list[str]]:
    out: list[list[str]] = []
    for row in ai_rows:
        holding = re.sub(r"\s{2,}", " ", _field(row, "Holding"))
        if not holding:
            continue
        compact = canonical_key(holding)
        if compact.startswith("TOTAL") or compact in {"EQUITIES", "CASHCASHEQUIVALENTS", "FIXEDINCOME"}:
            continue

        quantity = normalize_amount(_field(row, "Quantity"))
        price = normalize_amount(_field(row, "Price"))
        market_value = normalize_money(_field(row, "MarketValue"))
        book_cost = normalize_money(_field(row, "BookCost"))
        gain_loss = normalize_money(_field(row, "UnrealizedGainLoss"))

        if not quantity and not price and not market_value:
            continue

        out.append([
            str(period.year), f"{period.month:02d}", account_label, currency,
            holding, quantity, price, market_value, book_cost, gain_loss,
        ])
    return out


def parse_transaction_date(token: str, period: Period) -> str:
    token = token.strip()
    # Earliest (2006-2007) layout prints the settlement date as YY/MM/DD (e.g. "06/08/22"),
    # not "Mon D" - it already carries its own year, so use it directly.
    m_numeric = re.match(r"^(\d{2})/(\d{2})/(\d{2})$", token)
    if m_numeric:
        yy, mm, dd = (int(g) for g in m_numeric.groups())
        if 1 <= mm <= 12:
            return f"{mm:02d}/{dd:02d}/{2000 + yy:04d}"

    token = token.upper()
    m = re.match(r"([A-Z]+)\.?\s+(\d{1,2})", token)
    if not m:
        return ""
    mon_abbr = m.group(1)[:3]
    day = int(m.group(2))
    month_num = MONTH_ABBR_TO_NUM.get(mon_abbr)
    if not month_num:
        return ""
    return f"{month_num:02d}/{day:02d}/{period.year:04d}"


def clean_transaction_rows(ai_rows: list[dict[str, object]], period: Period, account_label: str, currency: str) -> list[list[str]]:
    out: list[list[str]] = []
    for row in ai_rows:
        date_token = _field(row, "Date")
        activity = _field(row, "Activity")
        desc = re.sub(r"\s{2,}", " ", _field(row, "Description"))
        compact_activity = canonical_key(activity)

        if compact_activity in {"", "BEGINNINGCASHBALANCE", "ENDINGCASHBALANCE", "CASHOPENINGBALANCE", "CASHCLOSINGBALANCE", "OPENINGBALANCE", "CLOSINGBALANCE"}:
            continue
        if not date_token:
            continue

        date = parse_transaction_date(date_token, period)
        if not date:
            continue

        quantity = normalize_amount(_field(row, "Quantity"))
        price = normalize_amount(_field(row, "Price"))
        amount = normalize_money(_field(row, "Amount"))
        commission = normalize_money(_field(row, "Commission"))
        balance = normalize_money(_field(row, "Balance"))

        out.append([
            str(period.year), f"{period.month:02d}", account_label, currency,
            date, activity, desc, quantity, price, amount, commission, balance,
        ])
    return out


def output_csv_paths(pdf_path: Path, period: Period, output_dir: Path | None) -> tuple[Path, Path]:
    base_dir = output_dir if output_dir else pdf_path.parent
    balances = base_dir / f"td_investing_balances_{period.year:04d}{period.month:02d}.csv"
    transactions = base_dir / f"td_investing_transactions_{period.year:04d}{period.month:02d}.csv"
    return balances, transactions


def write_csv(path: Path, header: list[str], rows: list[list[str]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(header)
        writer.writerows(rows)


def process_segment(pdf_path: Path, segment: Segment, page_texts: list[str], model: str, api_key: str, timeout_s: int, dpi: int) -> tuple[list[list[str]], list[list[str]]]:
    account_label = CANONICAL_ACCOUNTS.get((segment.account, segment.currency), f"{segment.account} {segment.currency}")
    page_data_urls = render_pages_as_data_urls(pdf_path, segment.pages, dpi=dpi)
    page_raw_texts = [page_texts[i] for i in segment.pages]
    prompt = build_prompt(segment.period, account_label, segment.currency)
    model_text = call_openrouter(model, api_key, prompt, page_data_urls, page_raw_texts, timeout_s)
    payload = extract_json_object(model_text)

    holdings = clean_holdings_rows(payload.get("holdings", []), segment.period, account_label, segment.currency)
    transactions = clean_transaction_rows(payload.get("transactions", []), segment.period, account_label, segment.currency)
    return holdings, transactions


def convert_file(pdf_path: Path, output_dir: Path | None, model: str, api_key: str, timeout_s: int, dpi: int) -> list[Path]:
    page_texts = extract_page_texts(pdf_path)
    segments = build_segments(pdf_path, page_texts)
    if not segments:
        raise RuntimeError(f"No recognizable account segments found in PDF: {pdf_path}")

    by_period: dict[Period, list[Segment]] = {}
    for seg in segments:
        by_period.setdefault(seg.period, []).append(seg)

    written: list[Path] = []
    for period, segs in sorted(by_period.items(), key=lambda kv: (kv[0].year, kv[0].month)):
        all_holdings: list[list[str]] = []
        all_transactions: list[list[str]] = []
        for seg in segs:
            holdings, transactions = process_segment(pdf_path, seg, page_texts, model, api_key, timeout_s, dpi)
            all_holdings.extend(holdings)
            all_transactions.extend(transactions)

        balances_path, transactions_path = output_csv_paths(pdf_path, period, output_dir)
        write_csv(
            balances_path,
            ["Year", "Month", "Account", "Currency", "Holding", "Quantity", "Price", "MarketValue", "BookCost", "UnrealizedGainLoss"],
            all_holdings,
        )
        write_csv(
            transactions_path,
            ["Year", "Month", "Account", "Currency", "Date", "Activity", "Description", "Quantity", "Price", "Amount", "Commission", "Balance"],
            all_transactions,
        )
        written.extend([balances_path, transactions_path])

    return written


def main() -> int:
    parser = argparse.ArgumentParser(description="Convert TD Direct Investing statement PDFs with OpenRouter-assisted extraction (default: openai/gpt-4o).")
    parser.add_argument("--input", default=".", help="PDF file or root folder to scan for TDWaterhouse*.pdf statement PDFs.")
    parser.add_argument("--output-dir", default=None, help="Optional output folder for CSV files (default: alongside each source PDF).")
    parser.add_argument("--model", default=MODEL, help="OpenRouter model id.")
    parser.add_argument("--timeout", type=int, default=180, help="OpenRouter timeout in seconds.")
    parser.add_argument("--dpi", type=int, default=450, help="Page render resolution for the vision model (higher helps with small print like ticker symbols).")
    parser.add_argument("--dotenv", default=".env", help="Path to .env file containing OPENROUTER_API_KEY.")
    parser.add_argument("--fail-fast", action="store_true", help="Stop on first file conversion error.")
    args = parser.parse_args()

    root = Path(args.input).resolve()
    output_dir = Path(args.output_dir).resolve() if args.output_dir else None

    load_dotenv(Path(args.dotenv).resolve())
    api_key = os.environ.get("OPENROUTER_API_KEY", "").strip()
    if not api_key:
        print("Missing OPENROUTER_API_KEY. Put it in your .env file or environment.", file=sys.stderr)
        return 2

    pdfs = find_pdfs(root)
    if not pdfs:
        print(f"No TDWaterhouse*.pdf statements found under: {root}")
        return 1

    converted = 0
    failed = 0

    for pdf in pdfs:
        try:
            outputs = convert_file(pdf, output_dir, args.model, api_key, args.timeout, args.dpi)
            for out in outputs:
                print(f"OK   {pdf} -> {out}")
            converted += 1
        except Exception as exc:
            print(f"FAIL {pdf}: {exc}")
            failed += 1
            if args.fail_fast:
                return 2

    print(f"Done. Converted={converted}, Failed={failed}")
    return 0 if failed == 0 else 2


if __name__ == "__main__":
    raise SystemExit(main())
