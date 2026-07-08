#!/usr/bin/env python3
"""Convert CIBC credit card statement PDFs into CSVs using OpenRouter (model default: openai/gpt-4o-mini).

Supported filename patterns:
  CIBC_Visa_YYYY_MM.pdf         → billing cycle ends on the 24th of YYYY/MM
  onlineStatement_YYYY-MM-DD.pdf → billing cycle ends on the date in the filename

Output files are named: cibc_YYYYMMDD_YYYYMMDD.csv (start/end dates from PDF or filename).

Columns (headers): Date,Posted Date,Description,Category,Amount,Currency,Exchange Rate,Amount CAD
Date format in CSV: MM/DD/YYYY
Amount: original-currency amount (positive for purchases, negative for payments/credits)
Amount CAD: CAD amount (positive for purchases, negative for payments/credits)
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
_MON = {
    "jan": 1, "feb": 2, "mar": 3, "apr": 4, "may": 5, "jun": 6,
    "jul": 7, "aug": 8, "sep": 9, "oct": 10, "nov": 11, "dec": 12
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


def _billing_cycle_from_end_cibc(end_year: int, end_month: int, end_day: int = 24) -> tuple[datetime, datetime]:
    """CIBC statements run from the 25th of the previous month to the 24th of end_month."""
    end = datetime(end_year, end_month, end_day)
    prev_month = end_month - 1 or 12
    prev_year = end_year if end_month != 1 else end_year - 1
    start = datetime(prev_year, prev_month, 25)
    return start, end


def find_statement_period_in_text(text: str, pdf_path: Path | None = None) -> tuple[datetime, datetime] | None:
    if pdf_path is not None:
        name = pdf_path.name

        # 1a) Pattern: onlineStatement_YYYY-MM-DD.pdf — date IS the statement end date
        m = re.search(r"onlineStatement[_-](\d{4})-(\d{2})-(\d{2})\b", name, re.I)
        if m:
            try:
                y, mo, day = int(m.group(1)), int(m.group(2)), int(m.group(3))
                return _billing_cycle_from_end_cibc(y, mo, day)
            except Exception:
                pass

        # 1b) Pattern: CIBC_Visa_YYYY_MM.pdf — end = 24th of YYYY/MM
        m = re.search(r"CIBC[_-]Visa[_-](\d{4})[_-](\d{2})\b", name, re.I)
        if m:
            try:
                y, mo = int(m.group(1)), int(m.group(2))
                return _billing_cycle_from_end_cibc(y, mo, 24)
            except Exception:
                pass

    # 2) Parse "Month DD[,] [YYYY] to Month DD, YYYY" from PDF text
    #    e.g. "December 25,2023to January 24, 2024" or "July 25to August 24, 2024"
    pat = re.compile(
        r"([A-Za-z]{3,9})\s+(\d{1,2}),?\s*(?:(\d{4})\s*)?to\s*([A-Za-z]{3,9})\s+(\d{1,2}),?\s*(\d{4})",
        re.I,
    )
    m = pat.search(text)
    if m:
        try:
            m1 = _MON.get(m.group(1).lower()[:3])
            m2 = _MON.get(m.group(4).lower()[:3])
            end_y = int(m.group(6))
            start_y = int(m.group(3)) if m.group(3) else (end_y if m2 and m1 and m1 <= m2 else end_y - 1)
            if m1 and m2:
                start = datetime(start_y, m1, int(m.group(2)))
                end = datetime(end_y, m2, int(m.group(5)))
                return start, end
        except Exception:
            pass

    # 3) Fallback: look for "MONTH statement period" heading + year from filename
    m2 = re.search(r"([A-Za-z]{3,9})\s+statement\s+period", text, re.I)
    if m2:
        mo_name = m2.group(1).lower()[:3]
        mo = _MON.get(mo_name)
        year = None
        if pdf_path is not None:
            ym = re.search(r"(\d{4})", pdf_path.name)
            if ym:
                year = int(ym.group(1))
        if mo and year:
            try:
                return _billing_cycle_from_end_cibc(year, mo, 24)
            except Exception:
                pass

    return None


def build_prompt(start: datetime, end: datetime) -> str:
    rules = (
        "You are given images of a CIBC Dividend Visa credit card statement "
        "together with the machine-extracted text from the same PDF. "
        "Extract ALL posted transactions — both from the 'Your payments' section and the "
        "'Your new charges and credits' section — as a single JSON array. "
        "Each object must have EXACT keys: Date, PostedDate, Description, Category, OriginalAmount, Currency, ExchangeRate, AmountCAD. "
        "Date and PostedDate must be in MM/DD/YYYY format (use the statement year range "
        f"{start.year}\u2013{end.year} to resolve ambiguous months). "
        "IMPORTANT: copy Description and Category values character-for-character from the "
        "machine-extracted text provided below \u2014 do NOT re-read them from the images. "
        "The extracted text is authoritative; the images are provided only for layout context. "
        "For foreign currency transactions (indicated by a second line like '13.56 USD @ 1.421091445**'): "
        "  OriginalAmount = the foreign currency amount (e.g. 13.56), "
        "  Currency = the ISO currency code (e.g. USD), "
        "  ExchangeRate = the rate shown (e.g. 1.421091445), "
        "  AmountCAD = the CAD amount shown on the main transaction line. "
        "For domestic CAD transactions: OriginalAmount = AmountCAD, Currency = '', ExchangeRate = ''. "
        "For payments (from 'Your payments' section): Category = '', Currency = '', ExchangeRate = '', "
        "  OriginalAmount = AmountCAD = the payment amount. "
        "Use POSITIVE numbers for purchases/charges and NEGATIVE numbers for payments, credits, or refunds. "
        "Ignore the '\u00dd' cash back marker symbol \u2014 it is not part of the transaction data. "
        "Do NOT include section headers, totals, page numbers, spend report rows, or summary lines. "
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
            currency = str(currency_raw).strip() if currency_raw else ""
            ex = normalize_rate(str(ex_raw).strip()) if ex_raw else ""
            amt_cad = normalize_amount(cad_raw)

            # Treat CAD-labelled currencies as domestic
            if currency:
                cur_up = re.sub(r"[^A-Z0-9]", "", currency.upper())
                if cur_up in {"CAD", "C", "CA", "CA$", "C$", "CAN"}:
                    currency = ""
                    ex = ""

            if not date or not desc or not amt_cad:
                continue

            out.append([date, posted, desc, category, orig, currency, ex, amt_cad])
        except Exception:
            continue
    return out


def parse_fx_from_text(text: str) -> list[dict[str, str]]:
    """Scan extracted PDF text for foreign currency detail lines.

    CIBC prints a second line immediately after a foreign currency transaction:
        13.56 USD @ 1.421091445**
    This function finds those lines and pairs them with the CAD amount on the
    preceding transaction line.
    """
    results: list[dict[str, str]] = []
    lines = text.splitlines()
    fx_line_re = re.compile(r'^\s*([\d.]+)\s+([A-Z]{3})\s+@\s+([\d.]+)\*{0,2}\s*$')
    for i, line in enumerate(lines):
        m = fx_line_re.match(line)
        if m and i > 0:
            orig = m.group(1)
            currency = m.group(2)
            rate = m.group(3)
            prev = lines[i - 1].strip()
            amt_m = re.search(r'([\d.]+)\s*$', prev)
            if amt_m:
                results.append({
                    'cad_amount': amt_m.group(1),
                    'orig_amount': orig,
                    'currency': currency,
                    'exchange_rate': rate,
                })
    return results


def apply_fx_to_rows(out_rows: list[list[str]], fx_list: list[dict[str, str]]) -> list[list[str]]:
    """Fill in OriginalAmount/Currency/ExchangeRate for rows where Currency is blank
    but whose Amount CAD matches a foreign currency entry parsed from the text.

    Row layout: [date, posted, desc, category, orig_amount, currency, exchange_rate, amt_cad]
    """
    if not fx_list:
        return out_rows
    fx_by_cad: dict[str, list[dict[str, str]]] = {}
    for fx in fx_list:
        fx_by_cad.setdefault(fx['cad_amount'], []).append(fx)
    used: dict[str, int] = {}
    for row in out_rows:
        if row[5]:  # currency already set by AI
            continue
        amt_cad = row[7]
        candidates = fx_by_cad.get(amt_cad, [])
        idx = used.get(amt_cad, 0)
        if idx < len(candidates):
            fx = candidates[idx]
            used[amt_cad] = idx + 1
            row[4] = fx['orig_amount']
            row[5] = fx['currency']
            row[6] = fx['exchange_rate']
    return out_rows


def output_csv_path(pdf_path: Path, start: datetime, end: datetime, output_dir: Path | None) -> Path:
    filename = f"cibc_{start.strftime('%Y%m%d')}_{end.strftime('%Y%m%d')}.csv"
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
    fx_list = parse_fx_from_text(text)
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
        description="Convert CIBC credit card statement PDFs with OpenRouter-assisted parsing (default: openai/gpt-4o-mini)."
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
