#!/usr/bin/env python3
"""Convert Capital One credit card statement PDFs into CSVs using OpenRouter (model default: openai/gpt-5-mini).

Output files are named: capitalone_YYYYMMDD_YYYYMMDD.csv (start/end dates from PDF or filename).

Columns (headers): Date,Posted Date,Description,Amount,Currency,Exchange Rate,Amount CAD,Balance
Date format in CSV: MM/DD/YYYY
Amount: original-currency amount (positive for purchases, negative for refunds/payments)
Amount CAD: amount converted to CAD (positive for purchases, negative for refunds/payments)
Balance: cumulative balance of posted transactions
"""

from __future__ import annotations

import argparse
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

try:
    from dateutil.parser import parse as parse_date
except Exception:
    parse_date = None

MONTHS_RE = r"Jan|Feb|Mar|Apr|May|Jun|Jul|Aug|Sep|Sept|Oct|Nov|Dec"


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


def find_statement_period_in_text(text: str, pdf_path: Path | None = None) -> tuple[datetime, datetime] | None:
    def adjust_years(s: datetime, e: datetime) -> tuple[datetime, datetime]:
        # Adjust start year to make start <= end while preserving month/day when possible.
        try:
            if s <= e:
                return s, e
            # Try aligning start year to end's year
            s_same_year = s.replace(year=e.year)
            if s_same_year <= e:
                return s_same_year, e
            # fallback: set start to end.year-1
            return s.replace(year=e.year - 1), e
        except Exception:
            return s, e

    # 1) Explicit Statement Period in text
    for pat in [
        re.compile(r"Statement\s+Period[:\s]*([A-Za-z0-9 ,./-]+?)\s*(?:to|-|\u2013)\s*([A-Za-z0-9 ,./-]+)", re.I),
        re.compile(r"Period[:\s]*([A-Za-z0-9 ,./-]+?)\s*(?:to|-|\u2013)\s*([A-Za-z0-9 ,./-]+)", re.I),
    ]:
        m = pat.search(text)
        if m and parse_date:
            try:
                start = parse_date(m.group(1).strip(), fuzzy=True)
                end = parse_date(m.group(2).strip(), fuzzy=True)
                start, end = adjust_years(start, end)
                return start, end
            except Exception:
                pass

    # 2) Filename-derived heuristics: look for YYYY_MM or YYYYMM patterns
    if pdf_path is not None:
        name = pdf_path.name
        # Capital One filename pattern like Stmnt_MMYYYY...: most statements use
        # billing cycle 15th(prev month) -> 14th(this month). However the first
        # statement `Stmnt_032014_5041.pdf` is an exception (20th -> 14th).
        m_st = re.search(r"Stmnt[_-]?(\d{2})(\d{4})", name, re.I)
        if m_st:
            try:
                mo = int(m_st.group(1))
                y = int(m_st.group(2))
                end = datetime(y, mo, 14)
                prev_month = mo - 1 or 12
                prev_year = y if mo != 1 else y - 1
                # default start = 15th of previous month
                start_day = 15
                # special-case known first-statement filename prefix
                if re.match(r"Stmnt[_-]?032014", name, re.I):
                    start_day = 20
                start = datetime(prev_year, prev_month, start_day)
                return start, end
            except Exception:
                pass
        # Look for any YYYYMMDD occurrences and prefer the rightmost (likely the end date)
        matches = re.findall(r"(\d{4})[_-]?(\d{2})[_-]?(\d{2})", name)
        if matches:
            try:
                y, mo, d = map(int, matches[-1])
                # assume found date (rightmost) is statement end
                end = datetime(y, mo, d)
                # fallback to 1-month window: start = 1st of previous month
                prev_month = mo - 1 or 12
                prev_year = y if mo != 1 else y - 1
                start = datetime(prev_year, prev_month, 1)
                return start, end
            except Exception:
                pass
        # Prefer the rightmost YYYY_MM occurrence if present
        matches2 = re.findall(r"(\d{4})[_-]?(\d{2})", name)
        if matches2:
            try:
                y, mo = map(int, matches2[-1])
                # assume end on last day of that month (approx)
                end = datetime(y, mo, 28)
                prev_month = mo - 1 or 12
                prev_year = y if mo != 1 else y - 1
                start = datetime(prev_year, prev_month, 1)
                return start, end
            except Exception:
                pass

    # 3) Header fallback: search for month-name dates in first 800 chars
    header = text[:800]
    date_regex = re.compile(r"\b(?:" + MONTHS_RE + r")[a-z]*\s+\d{1,2},?\s+\d{4}\b", re.I)
    candidates_raw = date_regex.findall(header)
    if candidates_raw and parse_date:
        parsed = []
        for tok in candidates_raw:
            try:
                parsed.append(parse_date(tok, fuzzy=True))
            except Exception:
                continue
        if parsed:
            s, e = min(parsed), max(parsed)
            s, e = adjust_years(s, e)
            return s, e

    return None


def build_prompt(start: datetime, end: datetime) -> str:
    rules = (
        "You are given images of a Capital One credit card statement. Extract all posted transactions as a JSON array."
        " Each object must have EXACT keys: Date, Posted Date, Description, OriginalAmount, Currency, ExchangeRate, AmountCAD, Balance."
        " Date and Posted Date must be in MM/DD/YYYY format."
        " OriginalAmount should be the transaction amount in the ORIGINAL currency (decimal, e.g. 12.34)."
        " AmountCAD should be the transaction amount converted to Canadian dollars (decimal with two fraction digits, e.g. 12.34)."
        " For both OriginalAmount and AmountCAD: use POSITIVE numbers for purchases and NEGATIVE numbers for refunds or payments."
        " Do NOT alter the Balance sign or convention — Balance should be the cumulative balance after the posted transaction (decimal with two fraction digits)."
        " Currency should be the ISO or common code for the original currency (e.g. USD, EUR). Leave it blank if the original currency is Canadian dollars."
        " ExchangeRate should be the rate used to convert original currency to CAD (decimal). Leave blank if Currency is Canadian."
        " Only include real transaction rows; exclude headers, footers, page numbers, and summary totals."
        " Return ONLY a JSON array (no surrounding text)."
    )
    meta = {"statement_period": {"start": start.strftime("%Y-%m-%d"), "end": end.strftime("%Y-%m-%d")}}
    return rules + "\n\nInput JSON:\n" + json.dumps(meta)


def build_user_content(prompt: str, page_data_urls: list[str]) -> list[dict[str, object]]:
    content: list[dict[str, object]] = [{"type": "text", "text": prompt}]
    for idx, data_url in enumerate(page_data_urls, start=1):
        content.append({"type": "text", "text": f"Statement page {idx}"})
        content.append({"type": "image_url", "image_url": {"url": data_url}})
    return content


def call_openrouter(model: str, api_key: str, prompt: str, page_data_urls: list[str], timeout_s: int) -> str:
    url = "https://openrouter.ai/api/v1/chat/completions"
    user_content = build_user_content(prompt, page_data_urls)
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


def normalize_amount(value: str) -> str:
    if value is None:
        return ""
    v = str(value).strip()
    v = v.replace(",", "")
    if v.startswith("(") and v.endswith(")"):
        v = "-" + v[1:-1]
    try:
        f = float(v)
        return f"{f:.2f}"
    except Exception:
        return v


def normalize_rate(value: str) -> str:
    if value is None:
        return ""
    v = str(value).strip()
    v = v.replace(",", "")
    return v


def normalize_description(value: str) -> str:
    if value is None:
        return ""
    return re.sub(r"\s+", " ", str(value).strip())


def clean_ai_rows(ai_rows: list[dict[str, object]]) -> list[list[str]]:
    out: list[list[str]] = []
    for row in ai_rows:
        try:
            date = str(row.get("Date", "")).strip()
            posted = str(row.get("Posted Date", "")).strip()
            desc = normalize_description(row.get("Description", ""))
            orig_amt_raw = row.get("OriginalAmount", row.get("Original Amount", ""))
            currency_raw = row.get("Currency", "")
            ex_raw = row.get("ExchangeRate", row.get("Exchange Rate", ""))
            amt_cad_raw = row.get("AmountCAD", row.get("Amount CAD", row.get("Amount", "")))
            bal = normalize_amount(str(row.get("Balance", "")).strip())

            orig_amt = normalize_amount(str(orig_amt_raw).strip()) if orig_amt_raw is not None else ""
            currency = str(currency_raw).strip() if currency_raw is not None else ""
            ex = normalize_rate(str(ex_raw).strip()) if ex_raw is not None else ""
            amt_cad = normalize_amount(str(amt_cad_raw).strip()) if amt_cad_raw is not None else ""

            if currency:
                cur_up = re.sub(r"[^A-Z0-9]", "", currency.upper())
                if cur_up in {"CAD", "C", "CA", "CA$", "C$", "CAN"}:
                    currency = ""
                    ex = ""

            if not date or not desc or not amt_cad:
                continue

            out.append([date, posted, desc, orig_amt, currency, ex, amt_cad, bal])
        except Exception:
            continue
    return out


def output_csv_path(pdf_path: Path, start: datetime, end: datetime, output_dir: Path | None) -> Path:
    filename = f"capitalone_{start.strftime('%Y%m%d')}_{end.strftime('%Y%m%d')}.csv"
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
    model_text = call_openrouter(model, api_key, prompt, page_data_urls, timeout_s)
    ai_rows = extract_json_array(model_text)
    out_rows = clean_ai_rows(ai_rows)

    out_path = output_csv_path(pdf, start, end, output_dir)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with out_path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["Date", "Posted Date", "Description", "Amount", "Currency", "Exchange Rate", "Amount CAD", "Balance"])
        writer.writerows(out_rows)
    return out_path


def main() -> int:
    parser = argparse.ArgumentParser(description="Convert Capital One credit card statement PDFs with OpenRouter-assisted parsing (default: openai/gpt-5-mini).")
    parser.add_argument("--input", default=".", help="PDF file or root folder to scan for statement PDFs.")
    parser.add_argument("--output-dir", default=None, help="Optional output folder for CSV files.")
    parser.add_argument("--model", default="openai/gpt-5-mini", help="OpenRouter model id (default: openai/gpt-5-mini).")
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
    main()
