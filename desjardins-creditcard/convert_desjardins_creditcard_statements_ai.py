#!/usr/bin/env python3
"""Convert Desjardins Visa credit card statement PDFs into CSVs using OpenRouter (model default: openai/gpt-4o-mini).

Supported filename pattern:
  VISADesjardins_YYYY-MM.pdf → statement date is read from the PDF text
                                 ("STATEMENT DATE Day D Month M Year Y"); YYYY-MM
                                 in the filename is the year/month of that date.

Output files are named: desjardins_YYYYMMDD_YYYYMMDD.csv (start/end dates from PDF).

Columns (headers): Date,Posted Date,Description,Category,Amount,Currency,Exchange Rate,Amount CAD
Date format in CSV: MM/DD/YYYY
Amount: original-currency amount (positive for purchases, negative for payments/credits)
Amount CAD: CAD amount (positive for purchases, negative for payments/credits)
"""

from __future__ import annotations

import argparse
import calendar
import csv
import io
import json
import os
import re
import sys
from datetime import datetime
from pathlib import Path

try:
    import pypdfium2 as pdfium
except Exception:
    pdfium = None

try:
    import pdfplumber
except Exception:
    pdfplumber = None

MONTHS_RE = r"Jan|Feb|Mar|Apr|May|Jun|Jul|Aug|Sep|Sept|Oct|Nov|Dec"
_MON = {
    "jan": 1, "feb": 2, "mar": 3, "apr": 4, "may": 5, "jun": 6,
    "jul": 7, "aug": 8, "sep": 9, "oct": 10, "nov": 11, "dec": 12
}

# Currency names as printed on Desjardins statements (no space before "XRT")
_CURRENCY_NAME_TO_CODE = {
    "EURO": "EUR",
    "US DOLLAR": "USD",
    "USD": "USD",
    "BRITISH POUND": "GBP",
    "POUND STERLING": "GBP",
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


def render_pdf_pages_as_data_urls(pdf_path: Path, dpi: int = 300) -> list[str]:
    if pdfium is None:
        raise RuntimeError("Missing dependency pypdfium2. Install requirements before running AI image extraction.")

    data_urls: list[str] = []
    scale = dpi / 72.0

    doc = pdfium.PdfDocument(str(pdf_path))
    try:
        page_count = len(doc)
        for page_index in range(page_count):
            page = doc[page_index]
            try:
                bitmap = page.render(scale=scale)
                try:
                    image = bitmap.to_pil()
                    buf = io.BytesIO()
                    image.save(buf, format="PNG")
                    import base64

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


def extract_text_from_pdf(pdf_path: Path) -> str:
    if pdfplumber is not None:
        try:
            with pdfplumber.open(pdf_path) as pdf:
                pages = [p.extract_text() or "" for p in pdf.pages]
            return "\n".join(pages)
        except Exception:
            pass

    try:
        from pypdf import PdfReader

        reader = PdfReader(str(pdf_path))
        texts: list[str] = []
        for p in reader.pages:
            try:
                texts.append(p.extract_text() or "")
            except Exception:
                texts.append("")
        return "\n".join(texts)
    except Exception:
        return ""


def _prev_month_same_day(year: int, month: int, day: int) -> datetime:
    prev_month = month - 1 or 12
    prev_year = year if month != 1 else year - 1
    last_day = calendar.monthrange(prev_year, prev_month)[1]
    return datetime(prev_year, prev_month, min(day, last_day))


def find_statement_period_in_text(text: str, pdf_path: Path | None = None) -> tuple[datetime, datetime] | None:
    # 1) Primary: "STATEMENT DATE Day D Month M Year Y" printed on every page
    m = re.search(r"STATEMENT DATE\s+Day\s+(\d{1,2})\s+Month\s+(\d{1,2})\s+Year\s+(\d{4})", text, re.I)
    if m:
        try:
            day, month, year = int(m.group(1)), int(m.group(2)), int(m.group(3))
            end = datetime(year, month, day)
            start = _prev_month_same_day(year, month, day)
            return start, end
        except Exception:
            pass

    # 2) Fallback: "Date DD MM YYYY Date DD MM YYYY" (statement date, then due date)
    m = re.search(r"\bDate\s+(\d{1,2})\s+(\d{1,2})\s+(\d{4})\s+Date\s+(\d{1,2})\s+(\d{1,2})\s+(\d{4})", text)
    if m:
        try:
            day, month, year = int(m.group(1)), int(m.group(2)), int(m.group(3))
            end = datetime(year, month, day)
            start = _prev_month_same_day(year, month, day)
            return start, end
        except Exception:
            pass

    # 3) Fallback: filename VISADesjardins_YYYY-MM.pdf, day unknown so assume day 1
    if pdf_path is not None:
        fm = re.search(r"VISADesjardins[_-](\d{4})-(\d{2})\b", pdf_path.name, re.I)
        if fm:
            try:
                year, month = int(fm.group(1)), int(fm.group(2))
                last_day = calendar.monthrange(year, month)[1]
                end = datetime(year, month, last_day)
                start = _prev_month_same_day(year, month, last_day)
                return start, end
            except Exception:
                pass

    return None


def build_prompt(start: datetime, end: datetime) -> str:
    rules = (
        "You are given images of a Desjardins Visa credit card statement "
        "together with the machine-extracted text from the same PDF. "
        "Extract ALL posted transactions from BOTH the 'REGULAR TRANSACTION DETAILS' section "
        "(rows under 'Transactions made with the card of: ...', these are purchases/debits) "
        "AND the 'Account Operations' section (payments and credits) as a single JSON array. "
        "Each transaction detail row has the form: "
        "'<Transaction Date D> <Transaction Date M> <Posting Date D> <Posting Date M> <line number> <Description> <Amount>[CR]'. "
        "Each object must have EXACT keys: Date, PostedDate, Description, Category, OriginalAmount, Currency, ExchangeRate, AmountCAD. "
        "Date = the Transaction Date, PostedDate = the Posting Date, both in MM/DD/YYYY format "
        "(the D M values on the statement are Day then Month, with no year; use the statement year range "
        f"{start.year}–{end.year} to resolve the year, accounting for statements that span a December/January boundary). "
        "IMPORTANT: copy Description values character-for-character from the machine-extracted text "
        "provided below — do NOT re-read them from the images. The extracted text is authoritative; "
        "the images are provided only for layout context. "
        "Category does not exist on this statement — always set Category to ''. "
        "For foreign currency transactions (indicated by a second line immediately below the transaction, "
        "e.g. '17.90EURO XRT: 1.527374' or '7.00US DOLLAR XRT: 1.360000', with NO space between the amount "
        "and the currency name): "
        "  OriginalAmount = the foreign currency amount (e.g. 17.90), "
        "  Currency = the ISO currency code (EURO -> EUR, US DOLLAR -> USD), "
        "  ExchangeRate = the rate shown after 'XRT:' (e.g. 1.527374), "
        "  AmountCAD = the CAD amount shown on the main transaction line. "
        "For domestic CAD transactions: OriginalAmount = AmountCAD, Currency = '', ExchangeRate = ''. "
        "Amounts followed by the suffix 'CR' (e.g. '460.13CR') are credits/payments/refunds — output them "
        "as NEGATIVE numbers. Amounts with no 'CR' suffix are purchases/debits — output them as POSITIVE numbers. "
        "Do NOT include the 'CR' text itself in any numeric field. "
        "Do NOT include section headers, column headers, totals, the rewards program table, the interest-rate "
        "or credit-limit summary boxes, page numbers/footers, or the MESSAGE section. "
        "Return ONLY a JSON array (no surrounding text)."
    )
    meta = {"statement_period": {"start": start.strftime("%Y-%m-%d"), "end": end.strftime("%Y-%m-%d")}}
    return rules + "\n\nInput JSON:\n" + json.dumps(meta)


def build_user_content(prompt: str, page_data_urls: list[str], pdf_text: str = "") -> list[dict[str, object]]:
    content: list[dict[str, object]] = [{"type": "text", "text": prompt}]
    if pdf_text.strip():
        content.append({"type": "text", "text": "Machine-extracted PDF text (authoritative source for descriptions):\n" + pdf_text})
    for idx, data_url in enumerate(page_data_urls, start=1):
        content.append({"type": "text", "text": f"Statement page {idx}"})
        content.append({"type": "image_url", "image_url": {"url": data_url}})
    return content


def call_openrouter(model: str, api_key: str, prompt: str, page_data_urls: list[str], timeout_s: int, pdf_text: str = "") -> str:
    url = "https://openrouter.ai/api/v1/chat/completions"
    user_content = build_user_content(prompt, page_data_urls, pdf_text)
    body = {
        "model": model,
        "messages": [
            {"role": "system", "content": "You output strict JSON only."},
            {"role": "user", "content": user_content},
        ],
        "temperature": 0,
    }

    import urllib.request, urllib.error

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


def extract_json_array(text: str) -> list[dict[str, object]]:
    text = text.strip()
    if text.startswith("["):
        return json.loads(text)
    m = re.search(r"\[.*\]", text, re.DOTALL)
    if not m:
        raise ValueError("Model response did not contain a JSON array.")
    return json.loads(m.group(0))


def normalize_amount(value: object) -> str:
    if value is None:
        return ""
    v = str(value).strip()
    v = v.replace(",", "")
    v = re.sub(r"CR$", "", v, flags=re.I).strip()
    if v.startswith("(") and v.endswith(")"):
        v = "-" + v[1:-1]
    try:
        f = float(v)
        return f"{f:.2f}"
    except Exception:
        return v


def normalize_rate(value: str) -> str:
    if not value:
        return ""
    return value.strip().replace(",", "").rstrip("*")


def normalize_description(value: object) -> str:
    if value is None:
        return ""
    return re.sub(r"\s+", " ", str(value).strip())


def normalize_currency_code(value: object) -> str:
    if not value:
        return ""
    raw = str(value).strip().upper()
    if raw in _CURRENCY_NAME_TO_CODE:
        return _CURRENCY_NAME_TO_CODE[raw]
    code = re.sub(r"[^A-Z]", "", raw)
    return _CURRENCY_NAME_TO_CODE.get(code, code[:3] if code else "")


_FX_ARTIFACT_DESC_RE = re.compile(r'^[\d,]+\.\d{2}\s*[A-Za-z]+(?:\s[A-Za-z]+)*$')


def _is_fx_artifact_description(desc: str) -> bool:
    """True if desc looks like a bare FX detail line ("17.90EURO", "7.00 US DOLLAR")
    rather than a real merchant name. The vision model occasionally hallucinates a
    spurious extra transaction row using the FX detail line's own text as the
    description — this identifies those so they can be dropped.
    """
    return bool(_FX_ARTIFACT_DESC_RE.match(desc.strip()))


def clean_ai_rows(ai_rows: list[dict[str, object]]) -> list[list[str]]:
    out: list[list[str]] = []
    for row in ai_rows:
        try:
            date = str(row.get("Date", "")).strip()
            posted = str(row.get("PostedDate", row.get("Posted Date", ""))).strip()
            desc = normalize_description(row.get("Description", ""))
            category = normalize_description(row.get("Category", ""))

            orig_raw = row.get("OriginalAmount", row.get("Amount", ""))
            currency_raw = row.get("Currency", "")
            ex_raw = row.get("ExchangeRate", row.get("Exchange Rate", ""))
            cad_raw = row.get("AmountCAD", row.get("Amount CAD", orig_raw))

            orig = normalize_amount(orig_raw)
            currency = normalize_currency_code(currency_raw) if currency_raw else ""
            ex = normalize_rate(str(ex_raw).strip()) if ex_raw else ""
            amt_cad = normalize_amount(cad_raw)

            if currency == "CAD":
                currency = ""
                ex = ""

            if not date or not desc or not amt_cad:
                continue

            if _is_fx_artifact_description(desc):
                continue

            out.append([date, posted, desc, category, orig, currency, ex, amt_cad])
        except Exception:
            continue
    return out


_TXN_LINE_RE = re.compile(
    r'^\s*(\d{1,2})\s+(\d{1,2})\s+(\d{1,2})\s+(\d{1,2})\s+\d{3}\s+(.+?)\s+([\d,]+\.\d{2})(CR)?\s*$',
    re.I,
)


def _normalize_match_key(desc: str) -> str:
    return re.sub(r"\s+", " ", desc.strip()).upper()


def _resolve_txn_year(day: int, month: int, start: datetime, end: datetime) -> datetime | None:
    """Resolve a bare Day/Month (no year, as printed on Desjardins statements) to a full
    date using the statement's start/end window, handling December/January rollover.
    Mirrors the resolution the AI is instructed to perform, so results line up with it.
    """
    from datetime import timedelta

    lo = start - timedelta(days=3)
    hi = end + timedelta(days=5)
    for year in {start.year, end.year}:
        try:
            candidate = datetime(year, month, day)
        except ValueError:
            continue
        if lo <= candidate <= hi:
            return candidate
    return None


def parse_fx_from_text(text: str, start: datetime, end: datetime) -> list[dict[str, str]]:
    """Scan extracted PDF text for foreign currency detail lines.

    Desjardins prints a second line immediately after a foreign currency transaction,
    with NO space between the amount and the currency name:
        17.90EURO XRT: 1.527374
        7.00US DOLLAR XRT: 1.360000
    This function finds those lines and pairs them with the transaction/posting dates
    and CAD amount on the preceding transaction detail line (authoritative — the AI
    sometimes swaps OriginalAmount/AmountCAD, invents a value, or garbles the
    Description for foreign currency rows, e.g. copying "17.90EURO" as the description).
    """
    results: list[dict[str, str]] = []
    lines = text.splitlines()
    fx_line_re = re.compile(r'^\s*([\d.]+)([A-Z][A-Za-z ]+?)\s+XRT:\s*([\d.]+)\s*$')
    for i, line in enumerate(lines):
        m = fx_line_re.match(line)
        if m and i > 0:
            orig = m.group(1)
            currency = normalize_currency_code(m.group(2))
            rate = m.group(3)
            prev = lines[i - 1].strip()
            txn_m = _TXN_LINE_RE.match(prev)
            if not txn_m:
                continue
            txn_d, txn_m_, post_d, post_m, desc, cad_amount, cr = txn_m.groups()
            txn_date = _resolve_txn_year(int(txn_d), int(txn_m_), start, end)
            post_date = _resolve_txn_year(int(post_d), int(post_m), start, end)
            if not txn_date or not post_date:
                continue
            signed_cad = f"-{cad_amount}" if cr else cad_amount
            results.append({
                'date': txn_date.strftime('%m/%d/%Y'),
                'posted_date': post_date.strftime('%m/%d/%Y'),
                'description': normalize_description(desc),
                'description_key': _normalize_match_key(desc),
                'cad_amount': signed_cad,
                'orig_amount': orig,
                'currency': currency,
                'exchange_rate': rate,
            })
    return results


def apply_fx_to_rows(out_rows: list[list[str]], fx_list: list[dict[str, str]]) -> list[list[str]]:
    """Overwrite Description/OriginalAmount/Currency/ExchangeRate/AmountCAD for rows
    matching a foreign currency entry parsed directly from the PDF text. The text-derived
    values are authoritative since the AI is unreliable on foreign currency rows — it has
    been observed to swap OriginalAmount/AmountCAD, invent a value, or copy the FX detail
    line ("17.90EURO") into the Description field instead of the real merchant name.

    Matching is keyed on (Date, PostedDate) rather than Description or amount, since those
    plain D/M-derived dates are the fields the AI has consistently gotten right even when
    it garbled everything else about a foreign currency row. Falls back to matching on
    Description for any leftover fx entries (e.g. if a date was miscomputed).

    Row layout: [date, posted, desc, category, orig_amount, currency, exchange_rate, amt_cad]
    """
    if not fx_list:
        return out_rows

    fx_by_date: dict[tuple[str, str], list[dict[str, str]]] = {}
    for fx in fx_list:
        fx_by_date.setdefault((fx['date'], fx['posted_date']), []).append(fx)

    used_date: dict[tuple[str, str], int] = {}
    matched_ids: set[int] = set()
    for row in out_rows:
        key = (row[0], row[1])
        candidates = fx_by_date.get(key, [])
        idx = used_date.get(key, 0)
        if idx < len(candidates):
            fx = candidates[idx]
            used_date[key] = idx + 1
            matched_ids.add(id(fx))
            row[2] = fx['description']
            row[4] = fx['orig_amount']
            row[5] = fx['currency']
            row[6] = fx['exchange_rate']
            row[7] = fx['cad_amount']

    leftover = [fx for fx in fx_list if id(fx) not in matched_ids]
    if leftover:
        fx_by_desc: dict[str, list[dict[str, str]]] = {}
        for fx in leftover:
            fx_by_desc.setdefault(fx['description_key'], []).append(fx)
        used_desc: dict[str, int] = {}
        for row in out_rows:
            key = _normalize_match_key(row[2])
            candidates = fx_by_desc.get(key, [])
            idx = used_desc.get(key, 0)
            if idx < len(candidates):
                fx = candidates[idx]
                used_desc[key] = idx + 1
                matched_ids.add(id(fx))
                row[4] = fx['orig_amount']
                row[5] = fx['currency']
                row[6] = fx['exchange_rate']
                row[7] = fx['cad_amount']

    # Any fx entries still unmatched mean the AI dropped that transaction entirely
    # (e.g. it only emitted the spurious FX-artifact row, which clean_ai_rows removes).
    # Reconstruct the row from scratch using the authoritative text-derived values.
    for fx in fx_list:
        if id(fx) in matched_ids:
            continue
        out_rows.append([
            fx['date'], fx['posted_date'], fx['description'], "",
            fx['orig_amount'], fx['currency'], fx['exchange_rate'], fx['cad_amount'],
        ])

    return out_rows


def output_csv_path(pdf_path: Path, start: datetime, end: datetime, output_dir: Path | None) -> Path:
    filename = f"desjardins_{start.strftime('%Y%m%d')}_{end.strftime('%Y%m%d')}.csv"
    if output_dir:
        return output_dir / filename
    return pdf_path.parent / filename


def find_pdfs(root: Path) -> list[Path]:
    if root.is_file() and root.suffix.lower() == ".pdf":
        return [root]
    return sorted(p for p in root.rglob("*.pdf") if p.is_file())


def convert_file(pdf: Path, output_dir: Path | None, model: str, api_key: str, timeout_s: int) -> Path:
    text = extract_text_from_pdf(pdf)
    period = find_statement_period_in_text(text, pdf_path=pdf)
    if not period:
        raise RuntimeError(f"Unable to determine statement period from PDF: {pdf}")
    start, end = period

    page_data_urls = render_pdf_pages_as_data_urls(pdf)
    prompt = build_prompt(start, end)
    model_text = call_openrouter(model, api_key, prompt, page_data_urls, timeout_s, pdf_text=text)
    ai_rows = extract_json_array(model_text)
    out_rows = clean_ai_rows(ai_rows)
    fx_list = parse_fx_from_text(text, start, end)
    out_rows = apply_fx_to_rows(out_rows, fx_list)

    out_path = output_csv_path(pdf, start, end, output_dir)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with out_path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["Date", "Posted Date", "Description", "Category", "Amount", "Currency", "Exchange Rate", "Amount CAD"])
        writer.writerows(out_rows)
    return out_path


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Convert Desjardins Visa credit card statement PDFs with OpenRouter-assisted parsing (default: openai/gpt-4o-mini)."
    )
    parser.add_argument("--input", default=".", help="PDF file or root folder to scan for statement PDFs.")
    parser.add_argument("--output-dir", default=None, help="Optional output folder for CSV files.")
    parser.add_argument("--model", default="openai/gpt-4o-mini", help="OpenRouter model id (default: openai/gpt-4o-mini).")
    parser.add_argument("--timeout", type=int, default=120, help="OpenRouter timeout in seconds.")
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
        print(f"No PDFs found under: {root}")
        return 1

    converted = 0
    failed = 0

    for pdf in pdfs:
        try:
            out = convert_file(pdf, output_dir, args.model, api_key, args.timeout)
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
    sys.exit(main())
