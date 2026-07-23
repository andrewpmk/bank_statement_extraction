#!/usr/bin/env python3
"""Convert Questrade FHSA account statement PDFs into CSVs using OpenRouter.

This folder holds statements for a single account (an Individual Tax-Free First Home
Savings Account, account #53067946) - there is no multi-account bundling or currency
segmentation to do, unlike TD Direct Investing. One PDF is one calendar month.

Statement filenames have varied over time ("statement_YYYY_MM.pdf", "Questrade
FHSA-<acct> Statement YYYY-MM.pdf", "Individual FHSA_YYYY-MM-DD.pdf"), so every PDF
under the input root is treated as a candidate statement; the statement's own month is
read from the "Current month: <Month> <Day>, <Year>" text repeated in every page's
footer, never from the filename.

Questrade's own PDF text extraction order gets jumbled on pages with a right-hand
footnote sidebar (a template change starting ~2026-03 made this worse, splitting
headings like "03. INVESTMENT DETAILS" and "Exchange-traded funds (ETFs) owned" across
several out-of-order lines). Page classification therefore only looks for short,
still-intact marker lines that survive the jumbling ("Cash", "owned"/"<Type> owned",
"Transactions" + "Trans Date"), never for a full multi-word heading or a fixed line
position - see classify_page.

Each statement period yields two pages of interest:
  - the "Cash" + "<Type> owned" (e.g. "Exchange-traded funds (ETFs) owned") pages under
    03. INVESTMENT DETAILS, rendered as images for one OpenRouter call that returns the
    holdings.
  - the "Transactions" page under 04. ACTIVITY DETAILS, rendered as images for one
    OpenRouter call that returns the period's transactions.

Output CSVs, written next to the source PDFs:
  questrade_investing_balances_YYYYMM.csv     - Year,Month,Account,Currency,Holding,Quantity,Price,MarketValue,BookCost,UnrealizedGainLoss
  questrade_investing_transactions_YYYYMM.csv - Year,Month,Account,Currency,Date,Activity,Description,Quantity,Price,Amount,Commission,Balance

Account is always "FHSA" and Balance is always blank (Questrade's transactions table has
no running per-row balance, only Opening/Closing balance marker rows, which are excluded).

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
from dataclasses import dataclass
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
ACCOUNT = "FHSA"

MONTH_ABBR_TO_NUM = {
    "JAN": 1, "FEB": 2, "MAR": 3, "APR": 4, "MAY": 5, "JUN": 6,
    "JUL": 7, "AUG": 8, "SEP": 9, "SEPT": 9, "OCT": 10, "NOV": 11, "DEC": 12,
}

# "Current month: January 31, 2024" is repeated in the footer of every page of every
# statement, regardless of layout version, so it is always safe to read the period from.
PERIOD_RE = re.compile(
    r"Current month:\s*(?P<month>[A-Za-z]+)\.?\s+\d{1,2},\s*(?P<year>\d{4})",
    re.IGNORECASE,
)


@dataclass(frozen=True)
class Period:
    year: int
    month: int


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


def find_pdfs(root: Path) -> list[Path]:
    if root.is_file():
        return [root] if root.suffix.lower() == ".pdf" else []
    return sorted(p for p in root.rglob("*.pdf") if p.is_file())


def extract_page_texts(pdf_path: Path) -> list[str]:
    if pdfplumber is None:
        raise RuntimeError("Missing dependency pdfplumber. Install requirements before running.")
    with pdfplumber.open(pdf_path) as pdf:
        return [p.extract_text() or "" for p in pdf.pages]


def month_name_to_num(name: str) -> int | None:
    abbr = re.sub(r"[^A-Z]", "", name.upper())[:3]
    return MONTH_ABBR_TO_NUM.get(abbr)


def detect_period(text: str) -> Period | None:
    m = PERIOD_RE.search(text)
    if not m:
        return None
    month_num = month_name_to_num(m.group("month"))
    if not month_num:
        return None
    return Period(year=int(m.group("year")), month=month_num)


def classify_page(text: str) -> str | None:
    """Return 'cash', 'holding_detail', 'transactions', or None for a page we don't need.

    Deliberately matches only short marker lines rather than a full heading or a fixed
    line position - see module docstring for why (reading-order jumbling on sidebar
    pages splits headings unpredictably across lines).
    """
    lines = [l.strip() for l in text.splitlines()]
    lower = [l.lower() for l in lines]

    if "transactions" in lower and "trans date" in text.lower():
        return "transactions"
    if "cash" in lower:
        return "cash"
    for l in lower:
        if l == "owned" or (l.endswith(" owned") and l != "securities owned"):
            return "holding_detail"
    return None


def build_period_pages(page_texts: list[str]) -> dict[Period, dict[str, list[int]]]:
    """Group each classified page's index under its statement period.

    The period is only ever read off a page that already passed classify_page (real
    account content), never off an excluded boundary page (annual reports, glossary,
    disclosures) - the same guard TD's converter uses to avoid misreading an unrelated
    date range.
    """
    result: dict[Period, dict[str, list[int]]] = {}
    current_period: Period | None = None

    for i, text in enumerate(page_texts):
        kind = classify_page(text)
        if kind is None:
            continue

        detected = detect_period(text)
        if detected is not None:
            current_period = detected
        if current_period is None:
            continue

        buckets = result.setdefault(current_period, {"holdings": [], "transactions": []})
        if kind == "transactions":
            buckets["transactions"].append(i)
        else:
            buckets["holdings"].append(i)

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


HOLDINGS_PROMPT = (
    "You are given page images from a Questrade account statement (FHSA account), covering "
    "the 'Cash' page and the '<Type> owned' page(s) (e.g. 'Exchange-traded funds (ETFs) owned', "
    "'Stocks owned', 'Mutual funds owned') from the '03. INVESTMENT DETAILS' section. Each image "
    "is preceded by the PDF's own raw extracted text for that same page - those characters are "
    "always exactly correct (it is not OCR), but the extractor's reading order can jumble the "
    "table layout (a heading or a security's wrapped name/description can end up split across "
    "non-adjacent lines, sometimes with unrelated numbers or footnote text in between), so use "
    "the image to see which numbers/ticker belong to which row and column, and use the raw text "
    "to double check the exact digits and ticker letters. Where the image and raw text disagree "
    "on a character, trust the raw text.\n\n"
    "Return ONLY a JSON object with key 'holdings': one entry per row, as follows.\n\n"
    "From the 'Cash' page: one entry with Holding='Cash', Quantity and Price blank, MarketValue "
    "= the 'Owned' row's 'Combined in CAD' amount for the current month, BookCost and "
    "UnrealizedGainLoss blank.\n\n"
    "From each '<Type> owned' table: one entry per security row (there is one row per security "
    "under a 'Securities held in CAD' or 'Securities held in USD' sub-heading). Columns are "
    "Symbol, Description, Cost basis, Qty, Segr., Cost/share, Pos. cost, Mkt. price, Mkt. value, "
    "P&L, % return, % port. Set Holding to the Description with the Symbol appended in "
    "parentheses, e.g. 'BMO S&P/TSX CAPPED COMPOSITE INDEX ETF CAD UNITS (ZCN)' - drop any "
    "leading '.' from the symbol. The security's Description sometimes wraps onto a different "
    "printed line than its numeric row, with unrelated footnote text or numbers sandwiched in "
    "between in the raw text; join the wrapped Description into one string in reading order using "
    "the image. Quantity=Qty, Price=Mkt. price, MarketValue=Mkt. value, BookCost=Pos. cost, "
    "UnrealizedGainLoss=P&L. Do NOT include the 'Securities held in CAD/USD' sub-heading rows or "
    "any 'Market Value ($)' subtotal row as holdings.\n\n"
    "Each holdings object must have EXACT keys: Holding, Quantity, Price, MarketValue, BookCost, "
    "UnrealizedGainLoss. Leave a key blank ('') if the statement does not show it for that row.\n\n"
    "Return ONLY the JSON object, no surrounding text."
)

TRANSACTIONS_PROMPT = (
    "You are given page image(s) of the 'Transactions' table from a Questrade account statement "
    "(FHSA account), under '04. ACTIVITY DETAILS'. Each image is preceded by the PDF's own raw "
    "extracted text for that same page - those characters are always exactly correct (it is not "
    "OCR), but the extractor's reading order can jumble the table layout: a transaction's "
    "Description frequently wraps onto a different printed line than its Trans Date/numbers row, "
    "sometimes appearing one line ABOVE the row it belongs to, with unrelated text in between. Use "
    "the image to see which Description belongs to which dated row, and use the raw text to double "
    "check exact digits/ticker letters. Where they disagree on a character, trust the raw text.\n\n"
    "The table has columns: Trans Date, Settle Date, Activity type, Symbol, Description, then a "
    "CAD group (Qty, Price, Gross, Com., Net) and a USD group (Price, Gross, Com., Net). For each "
    "row exactly one of the CAD/USD groups is populated (the other shows all dashes '-').\n\n"
    "Return ONLY a JSON object with key 'transactions': one entry per data row, skipping the "
    "'Opening balance' and 'Closing balance' bookkeeping rows entirely. For each remaining row:\n"
    "- Date = Trans Date (as printed, e.g. '01-03-2024').\n"
    "- Activity = the Activity type cell. It is sometimes blank (e.g. for a distribution/dividend "
    "row) - when blank, infer a short label from the Description instead, e.g. 'Distribution' when "
    "the Description contains 'DIST ON', 'Dividend' for a dividend payment, 'Transfer' for a "
    "transfer, 'Fee' for a fee. Never leave Activity blank.\n"
    "- Description = the Symbol + Description cell joined into one string, e.g. 'BMO S&P/TSX "
    "CAPPED COMPOSITE INDEX ETF CAD UNITS (ZCN) WE ACTED AS AGENT' - drop a leading '.' from the "
    "symbol before appending it in parentheses. If the Symbol+Description cell is plain free text "
    "with no real ticker (e.g. 'FHSA CONTRIBUTION', '53067946 Visa Debit BSN ORIG DEP 10/05/23'), "
    "use it as-is with no parenthetical.\n"
    "- Currency = 'CAD' if the CAD group is populated for this row, else 'USD'.\n"
    "- Quantity = Qty from the populated currency group (blank if not a security trade).\n"
    "- Price = Price from the populated currency group.\n"
    "- Amount = Net from the populated currency group (this already includes the commission).\n"
    "- Commission = Com. from the populated currency group.\n\n"
    "Each transactions object must have EXACT keys: Date, Activity, Description, Currency, "
    "Quantity, Price, Amount, Commission. Leave a key blank ('') if the statement does not show it "
    "for that row.\n\n"
    "Return ONLY the JSON object, no surrounding text."
)


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


def clean_holdings_rows(ai_rows: list[dict[str, object]], period: Period) -> list[list[str]]:
    out: list[list[str]] = []
    for row in ai_rows:
        holding = re.sub(r"\s{2,}", " ", _field(row, "Holding"))
        if not holding:
            continue
        compact = canonical_key(holding)
        if compact.startswith("TOTAL") or compact in {"MARKETVALUE", "SECURITIESHELDINCAD", "SECURITIESHELDINUSD"}:
            continue

        quantity = normalize_amount(_field(row, "Quantity"))
        price = normalize_amount(_field(row, "Price"))
        market_value = normalize_money(_field(row, "MarketValue"))
        book_cost = normalize_money(_field(row, "BookCost"))
        gain_loss = normalize_money(_field(row, "UnrealizedGainLoss"))

        if not quantity and not price and not market_value:
            continue

        out.append([
            str(period.year), f"{period.month:02d}", ACCOUNT, "CAD",
            holding, quantity, price, market_value, book_cost, gain_loss,
        ])
    return out


def parse_transaction_date(token: str) -> str:
    token = token.strip()
    m = re.match(r"^(\d{2})-(\d{2})-(\d{4})$", token)
    if not m:
        return ""
    mm, dd, yyyy = (int(g) for g in m.groups())
    if not (1 <= mm <= 12):
        return ""
    return f"{mm:02d}/{dd:02d}/{yyyy:04d}"


def clean_transaction_rows(ai_rows: list[dict[str, object]], period: Period) -> list[list[str]]:
    out: list[list[str]] = []
    for row in ai_rows:
        date_token = _field(row, "Date")
        activity = _field(row, "Activity")
        desc = re.sub(r"\s{2,}", " ", _field(row, "Description"))
        currency = _field(row, "Currency").upper() or "CAD"

        compact_activity = canonical_key(activity)
        if compact_activity in {"", "OPENINGBALANCE", "CLOSINGBALANCE"}:
            continue

        date = parse_transaction_date(date_token)
        if not date:
            continue

        quantity = normalize_amount(_field(row, "Quantity"))
        price = normalize_amount(_field(row, "Price"))
        amount = normalize_money(_field(row, "Amount"))
        commission = normalize_money(_field(row, "Commission"))

        out.append([
            str(period.year), f"{period.month:02d}", ACCOUNT, currency,
            date, activity, desc, quantity, price, amount, commission, "",
        ])
    return out


def output_csv_paths(pdf_path: Path, period: Period, output_dir: Path | None) -> tuple[Path, Path]:
    base_dir = output_dir if output_dir else pdf_path.parent
    balances = base_dir / f"questrade_investing_balances_{period.year:04d}{period.month:02d}.csv"
    transactions = base_dir / f"questrade_investing_transactions_{period.year:04d}{period.month:02d}.csv"
    return balances, transactions


def write_csv(path: Path, header: list[str], rows: list[list[str]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(header)
        writer.writerows(rows)


def process_period(pdf_path: Path, period: Period, pages: dict[str, list[int]], page_texts: list[str], model: str, api_key: str, timeout_s: int, dpi: int) -> tuple[list[list[str]], list[list[str]]]:
    holdings: list[list[str]] = []
    transactions: list[list[str]] = []

    if pages["holdings"]:
        page_indices = sorted(pages["holdings"])
        page_data_urls = render_pages_as_data_urls(pdf_path, page_indices, dpi=dpi)
        page_raw_texts = [page_texts[i] for i in page_indices]
        model_text = call_openrouter(model, api_key, HOLDINGS_PROMPT, page_data_urls, page_raw_texts, timeout_s)
        payload = extract_json_object(model_text)
        holdings = clean_holdings_rows(payload.get("holdings", []), period)

    if pages["transactions"]:
        page_indices = sorted(pages["transactions"])
        page_data_urls = render_pages_as_data_urls(pdf_path, page_indices, dpi=dpi)
        page_raw_texts = [page_texts[i] for i in page_indices]
        model_text = call_openrouter(model, api_key, TRANSACTIONS_PROMPT, page_data_urls, page_raw_texts, timeout_s)
        payload = extract_json_object(model_text)
        transactions = clean_transaction_rows(payload.get("transactions", []), period)

    return holdings, transactions


def convert_file(pdf_path: Path, output_dir: Path | None, model: str, api_key: str, timeout_s: int, dpi: int) -> list[Path]:
    page_texts = extract_page_texts(pdf_path)
    period_pages = build_period_pages(page_texts)
    if not period_pages:
        raise RuntimeError(f"No recognizable statement content found in PDF: {pdf_path}")

    written: list[Path] = []
    for period, pages in sorted(period_pages.items(), key=lambda kv: (kv[0].year, kv[0].month)):
        holdings, transactions = process_period(pdf_path, period, pages, page_texts, model, api_key, timeout_s, dpi)

        balances_path, transactions_path = output_csv_paths(pdf_path, period, output_dir)
        write_csv(
            balances_path,
            ["Year", "Month", "Account", "Currency", "Holding", "Quantity", "Price", "MarketValue", "BookCost", "UnrealizedGainLoss"],
            holdings,
        )
        write_csv(
            transactions_path,
            ["Year", "Month", "Account", "Currency", "Date", "Activity", "Description", "Quantity", "Price", "Amount", "Commission", "Balance"],
            transactions,
        )
        written.extend([balances_path, transactions_path])

    return written


def main() -> int:
    parser = argparse.ArgumentParser(description="Convert Questrade FHSA statement PDFs with OpenRouter-assisted extraction (default: openai/gpt-4o).")
    parser.add_argument("--input", default=".", help="PDF file or root folder to scan for Questrade statement PDFs.")
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
        print(f"No statement PDFs found under: {root}")
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
