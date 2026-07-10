#!/usr/bin/env python3
"""Convert Meridian Credit Union savings statement PDFs into accountactivity CSVs using OpenRouter.

This is adapted from the TD/Tangerine/Desjardins converters. Meridian statements print
one "Deposit Accounts" table per account (this account only ever has one: Online
Advantage Savings) with columns Date, Account Activity, Withdrawals, Deposits, Balance.
Transaction dates are printed as full dates (DD-Mon-YYYY), so no month/year inference is
needed the way TD/Tangerine's short "MONDD" dates require.

Supported filename patterns (statement period is read from the filename):
  MESTMT-MMDDYYYY-<account>-<seq>.pdf   (statement period ending date)
  Meridian_YYYY-MM.pdf

Output files are named: accountactivity_meridian_YYYY_MM.csv
in the same row format as the other converters: Date, Description, Withdrawals, Deposits, Balance

Default model: openai/gpt-4o-mini (cheap). Override with --model if needed.
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

MONTH_TO_NUM = {
    "JAN": 1,
    "FEB": 2,
    "MAR": 3,
    "APR": 4,
    "MAY": 5,
    "JUN": 6,
    "JUL": 7,
    "AUG": 8,
    "SEP": 9,
    "OCT": 10,
    "NOV": 11,
    "DEC": 12,
}

MESTMT_RE = re.compile(
    r"MESTMT-(?P<month>\d{2})(?P<day>\d{2})(?P<year>\d{4})-\d+-\d+\.pdf$", re.IGNORECASE
)
MERIDIAN_NAME_MONTH_RE = re.compile(r"Meridian_(?P<year>\d{4})-(?P<month>\d{2})\.pdf$", re.IGNORECASE)

MODEL = "openai/gpt-4o-mini"


@dataclass
class StatementPeriod:
    end_month: int
    end_year: int


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


def normalize_description(value: str) -> str:
    value = re.sub(r"\s*\n\s*", " ", value.strip())
    return re.sub(r"\s{2,}", " ", value)


def normalize_amount(value: str) -> str:
    value = value.strip()
    if not value:
        return ""
    value = value.replace(",", "")
    # Meridian prints Withdrawals with a leading "-" as a visual cue; the column already
    # means "money out", so the sign is not a real negative and must not be kept.
    return value.lstrip("-")


def canonical_desc_key(value: str) -> str:
    return re.sub(r"[^A-Z0-9]", "", value.upper())


def parse_statement_period(pdf_path: Path) -> StatementPeriod:
    match = MESTMT_RE.search(pdf_path.name) or MERIDIAN_NAME_MONTH_RE.search(pdf_path.name)
    if not match:
        raise ValueError(f"Unable to parse statement period from filename: {pdf_path.name}")
    return StatementPeriod(end_month=int(match.group("month")), end_year=int(match.group("year")))


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


def parse_full_date(token: str) -> str:
    token = token.strip().upper()
    m = re.fullmatch(r"(\d{1,2})-([A-Z]{3})-(\d{4})", token)
    if not m:
        raise ValueError(f"Unrecognized transaction date token: {token}")
    day = int(m.group(1))
    mon_abbr = m.group(2)
    year = int(m.group(3))
    if mon_abbr not in MONTH_TO_NUM:
        raise ValueError(f"Unrecognized month abbreviation in date token: {token}")
    return f"{MONTH_TO_NUM[mon_abbr]:02d}/{day:02d}/{year:04d}"


def build_prompt() -> str:
    rules = (
        "You extract Meridian Credit Union savings statement transactions from page images. "
        "The table is under a 'Deposit Accounts' heading with a "
        "'Date Account Activity Withdrawals Deposits Balance' header row. "
        "Each visible transaction row must become one output row. Do not merge adjacent rows. "
        "A row's description sometimes wraps onto the next printed line with no date of its own "
        "(e.g. a payee name like 'Tangerine' or 'PAYPAL PTE LTD' under a 'Pre-Authorized # ...' line); "
        "join that continuation line into the Description of the row above it, it is not its own row. "
        "Description always populated for real transactions. "
        "Exactly one of Withdrawals/Deposits must be populated in output. "
        "Withdrawals is always a positive magnitude: the statement sometimes prints a leading '-' before a "
        "Withdrawals amount as a visual cue, but that is not a negative number, do not include the sign. "
        "Date is always populated for output rows; it is printed as 'DD-Mon-YYYY', e.g. '31-Mar-2022'; "
        "output it unchanged in that same 'DD-Mon-YYYY' format. "
        "If Balance is missing, compute running balance from prior known balance and amount. "
        "Include the 'Balance Forward' row when visible, using Description 'Balance Forward'. "
        "Ignore 'Account Totals' summary rows. "
        "Preserve spacing and punctuation in Description; do not rewrite text. "
        "Return ONLY JSON array. Each object must contain keys exactly: "
        "Description, Withdrawals, Deposits, Date, Balance. "
        "Amounts use plain decimals without commas."
    )
    return rules


def extract_json_array(text: str) -> list[dict[str, str]]:
    text = text.strip()
    if text.startswith("["):
        return json.loads(text)
    match = re.search(r"\[.*\]", text, re.DOTALL)
    if not match:
        raise ValueError("Model response did not contain a JSON array.")
    return json.loads(match.group(0))


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


def _field(row: dict[str, object], key: str) -> str:
    value = row.get(key)
    if value is None:
        return ""
    return str(value).strip()


def clean_ai_rows(ai_rows: list[dict[str, str]]) -> list[list[str]]:
    output_rows: list[list[str]] = []
    prev_balance: float | None = None

    for row in ai_rows:
        desc = normalize_description(_field(row, "Description"))
        wd = normalize_amount(_field(row, "Withdrawals"))
        dep = normalize_amount(_field(row, "Deposits"))
        date_token = _field(row, "Date")
        bal = normalize_amount(_field(row, "Balance"))

        if not desc:
            continue

        compact = canonical_desc_key(desc)
        if "TOTAL" in compact:
            continue

        if compact in {"STARTINGBALANCE", "BALANCEFORWARD"}:
            cur_balance: float | None = None
            if bal:
                try:
                    cur_balance = round(float(bal), 2)
                except ValueError:
                    cur_balance = None
            if cur_balance is None:
                continue
            prev_balance = cur_balance
            continue

        if not date_token:
            continue

        if bool(wd) == bool(dep):
            continue

        mmddyyyy = parse_full_date(date_token)

        cur_balance = None
        if bal:
            try:
                cur_balance = round(float(bal), 2)
            except ValueError:
                cur_balance = None

        if cur_balance is None and prev_balance is not None:
            try:
                if wd:
                    cur_balance = round(prev_balance - float(wd), 2)
                else:
                    cur_balance = round(prev_balance + float(dep), 2)
            except ValueError:
                cur_balance = None

        if cur_balance is None:
            continue

        output_rows.append([mmddyyyy, desc, wd, dep, f"{cur_balance:.2f}"])
        prev_balance = cur_balance

    return output_rows


def output_csv_path(pdf_path: Path, period: StatementPeriod, output_dir: Path | None) -> Path:
    filename = f"accountactivity_meridian_{period.end_year:04d}_{period.end_month:02d}.csv"
    if output_dir:
        return output_dir / filename
    return pdf_path.parent / filename


def convert_file(pdf_path: Path, output_dir: Path | None, model: str, api_key: str, timeout_s: int) -> Path:
    period = parse_statement_period(pdf_path)
    page_data_urls = render_pdf_pages_as_data_urls(pdf_path)
    prompt = build_prompt()
    model_text = call_openrouter(model, api_key, prompt, page_data_urls, timeout_s)
    ai_rows = extract_json_array(model_text)
    out_rows = clean_ai_rows(ai_rows)

    out_path = output_csv_path(pdf_path, period, output_dir)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with out_path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerows(out_rows)
    return out_path


def find_pdfs(root: Path) -> list[Path]:
    return sorted(p for p in root.rglob("*.pdf") if p.is_file())


def main() -> int:
    parser = argparse.ArgumentParser(description="Convert Meridian statement PDFs with OpenRouter-assisted table normalization.")
    parser.add_argument("--input", default=".", help="Root folder to scan for statement PDFs.")
    parser.add_argument("--output-dir", default=None, help="Optional output folder for CSV files.")
    parser.add_argument("--model", default=MODEL, help="OpenRouter model id.")
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
