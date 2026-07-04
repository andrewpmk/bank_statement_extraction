#!/usr/bin/env python3
"""Convert TD credit card statement PDFs into CSVs using an AI model (sonnet 4.6).

Output files are named: tdaeroplan_YYYYMMDD_YYYYMMDD.csv (start/end dates from PDF).

Columns (headers): Date,Posted Date,Description,Amount,Balance
Date format in CSV: MM/DD/YYYY
Amount: positive for purchase, negative for refund or bill pay
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
from dataclasses import dataclass
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
    # Prefer pdfplumber for more reliable text extraction
    if pdfplumber is not None:
        try:
            with pdfplumber.open(pdf_path) as pdf:
                pages = [p.extract_text() or "" for p in pdf.pages]
            return "\n".join(pages)
        except Exception:
            pass

    # Fallback to pypdf (installed as pypdf)
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


def find_statement_period_in_text(text: str) -> tuple[datetime, datetime] | None:
    # Try common explicit patterns like 'Statement Period: January 1, 2023 to January 31, 2023'
    pat = re.compile(r"Statement\s+Period[:\s]*([A-Za-z0-9 ,./-]+?)\s*(?:to|-|\u2013)\s*([A-Za-z0-9 ,./-]+)", re.I)
    m = pat.search(text)
    if not m:
        pat2 = re.compile(r"Period[:\s]*([A-Za-z0-9 ,./-]+?)\s*(?:to|-|\u2013)\s*([A-Za-z0-9 ,./-]+)", re.I)
        m = pat2.search(text)
    if m and parse_date:
        a = m.group(1).strip()
        b = m.group(2).strip()
        try:
            start = parse_date(a, fuzzy=True)
            end = parse_date(b, fuzzy=True)
            # If parsed start is after end (e.g., statement in Jan where start is Dec previous year
            # but parsed with the same year), adjust start to previous year.
            try:
                if start > end:
                    start = start.replace(year=end.year - 1)
            except Exception:
                pass
            return start, end
        except Exception:
            pass

    # Fallback: collect all date-like tokens on first page and take min/max
    # look for Month DD, YYYY and other common formats
    date_regex = re.compile(r"\b(?:" + MONTHS_RE + r")[a-z]*\s+\d{1,2},?\s+\d{4}\b", re.I)
    candidates = date_regex.findall(text)
    if candidates and parse_date:
        parsed = []
        for tok in candidates:
            try:
                parsed.append(parse_date(tok, fuzzy=True))
            except Exception:
                continue
        if parsed:
            s = min(parsed)
            e = max(parsed)
            try:
                if s > e:
                    s = s.replace(year=e.year - 1)
            except Exception:
                pass
            return s, e

    # Last resort: find all yyyy-mm-dd or mm/dd/yyyy tokens
    nums = re.findall(r"\b\d{4}-\d{2}-\d{2}\b|\b\d{1,2}/\d{1,2}/\d{4}\b", text)
    if nums and parse_date:
        parsed = []
        for tok in nums:
            try:
                parsed.append(parse_date(tok, fuzzy=True))
            except Exception:
                continue
        if parsed:
            s = min(parsed)
            e = max(parsed)
            try:
                if s > e:
                    s = s.replace(year=e.year - 1)
            except Exception:
                pass
            return s, e

    return None


def build_prompt(start: datetime, end: datetime) -> str:
    # Request explicit keys including original-currency fields.
    rules = (
        "You are given images of a TD credit card statement. Extract all posted transactions as a JSON array."
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
    # handle parentheses like (12.34) meaning negative
    if v.startswith("(") and v.endswith(")"):
        v = "-" + v[1:-1]
    # ensure two decimals
    try:
        f = float(v)
        return f"{f:.2f}"
    except Exception:
        return v


def clean_ai_rows(ai_rows: list[dict[str, object]]) -> list[list[str]]:
    out: list[list[str]] = []
    for row in ai_rows:
        try:
            date = str(row.get("Date", "")).strip()
            posted = str(row.get("Posted Date", "")).strip()
            desc = str(row.get("Description", "")).strip()
            orig_amt_raw = row.get("OriginalAmount", row.get("Original Amount", ""))
            currency_raw = row.get("Currency", "")
            ex_raw = row.get("ExchangeRate", row.get("Exchange Rate", ""))
            amt_cad_raw = row.get("AmountCAD", row.get("Amount CAD", row.get("Amount", "")))
            bal = normalize_amount(str(row.get("Balance", "")).strip())

            orig_amt = normalize_amount(str(orig_amt_raw).strip()) if orig_amt_raw is not None else ""
            currency = str(currency_raw).strip() if currency_raw is not None else ""
            ex = normalize_amount(str(ex_raw).strip()) if ex_raw is not None else ""
            amt_cad = normalize_amount(str(amt_cad_raw).strip()) if amt_cad_raw is not None else ""

            # If currency indicates CAD, leave Currency and ExchangeRate blank as requested
            if currency:
                cur_up = re.sub(r"[^A-Z0-9]", "", currency.upper())
                if cur_up in {"CAD", "C", "CA", "CA$", "C$", "CAN"}:
                    currency = ""
                    ex = ""

            if not date or not desc or not amt_cad:
                continue

            # CSV columns: Date, Posted Date, Description, Amount(original), Currency, Exchange Rate, Amount CAD, Balance
            out.append([date, posted, desc, orig_amt, currency, ex, amt_cad, bal])
        except Exception:
            continue
    return out


def output_csv_path(pdf_path: Path, start: datetime, end: datetime, output_dir: Path | None) -> Path:
    filename = f"tdaeroplan_{start.strftime('%Y%m%d')}_{end.strftime('%Y%m%d')}.csv"
    if output_dir:
        return output_dir / filename
    return pdf_path.parent / filename


def find_pdfs(root: Path) -> list[Path]:
    if root.is_file() and root.suffix.lower() == ".pdf":
        return [root]
    return sorted(p for p in root.rglob("*.pdf") if p.is_file())


def convert_file(pdf: Path, output_dir: Path | None, model: str, api_key: str, timeout_s: int) -> Path:
    text = extract_text_from_pdf(pdf)
    period = find_statement_period_in_text(text)
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
    parser = argparse.ArgumentParser(description="Convert TD credit card statement PDFs with OpenRouter-assisted parsing (anthropic/claude-sonnet-5).")
    parser.add_argument("--input", default=".", help="PDF file or root folder to scan for statement PDFs.")
    parser.add_argument("--output-dir", default=None, help="Optional output folder for CSV files.")
    parser.add_argument("--model", default="anthropic/claude-sonnet-5", help="OpenRouter model id (default: anthropic/claude-sonnet-5).")
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
    raise SystemExit(main())
