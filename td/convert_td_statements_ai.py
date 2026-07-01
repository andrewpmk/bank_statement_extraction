#!/usr/bin/env python3
"""Convert TD chequing statement PDFs into accountactivity-YYYY-MM.csv using OpenRouter.

This script extracts raw table rows with pdfplumber, then uses an AI model to
normalize transaction rows and compute missing balances.
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

STATEMENT_FILENAME_RE = re.compile(
    r"_(?P<start_mon>[A-Za-z]{3})_(?P<start_day>\d{2})-(?P<end_mon>[A-Za-z]{3})_(?P<end_day>\d{2})_(?P<end_year>\d{4})\.pdf$",
    re.IGNORECASE,
)


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


def normalize_cell(value: str | None) -> str:
    if value is None:
        return ""
    parts = [part.strip() for part in str(value).split("\n")]
    parts = [part for part in parts if part]
    return " ".join(parts)


def normalize_description(value: str) -> str:
    # Normalize spacing only: collapse multiple spaces while preserving content.
    return re.sub(r"\s{2,}", " ", value).strip()


def normalize_amount(value: str) -> str:
    value = value.strip()
    if not value:
        return ""
    return value.replace(",", "")


def canonical_desc_key(value: str) -> str:
    return re.sub(r"[^A-Z0-9]", "", value.upper())


def parse_statement_period(pdf_path: Path) -> StatementPeriod:
    match = STATEMENT_FILENAME_RE.search(pdf_path.name)
    if not match:
        raise ValueError(f"Unable to parse statement period from filename: {pdf_path.name}")

    end_mon = match.group("end_mon").upper()
    end_year = int(match.group("end_year"))
    if end_mon not in MONTH_TO_NUM:
        raise ValueError(f"Unrecognized end month in filename: {pdf_path.name}")

    return StatementPeriod(end_month=MONTH_TO_NUM[end_mon], end_year=end_year)


def render_pdf_pages_as_data_urls(pdf_path: Path, dpi: int = 300) -> list[str]:
    if pdfium is None:
        raise RuntimeError(
            "Missing dependency pypdfium2. Install requirements before running AI image extraction."
        )

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


def month_day_to_mmddyyyy(token: str, period: StatementPeriod) -> str:
    token = token.strip().upper()
    m = re.fullmatch(r"([A-Z]{3})(\d{2})", token)
    if not m:
        raise ValueError(f"Unrecognized transaction date token: {token}")

    mon_abbr = m.group(1)
    day = int(m.group(2))
    if mon_abbr not in MONTH_TO_NUM:
        raise ValueError(f"Unrecognized month abbreviation in date token: {token}")

    month_num = MONTH_TO_NUM[mon_abbr]
    year = period.end_year if month_num <= period.end_month else period.end_year - 1
    return f"{month_num:02d}/{day:02d}/{year:04d}"


def build_prompt(period: StatementPeriod) -> str:
    rules = (
        "You extract TD statement transactions from page images. "
        "Columns are Description, Withdrawals, Deposits, Date, Balance. "
        "Rules: Each visible transaction row must become one output row. Do not merge adjacent rows. "
        "If a description wraps across multiple lines within the same visible row, join only those lines. "
        "Description always populated for real transactions. "
        "Exactly one of Withdrawals/Deposits must be populated in output. "
        "Date is always populated for output rows. "
        "If Balance is missing, compute running balance from prior known balance and amount. "
        "Include STARTING BALANCE / BALANCE FORWARD rows when visible. "
        "Ignore TOTAL summary rows. "
        "Preserve spacing and punctuation in Description; do not rewrite text. "
        "Return ONLY JSON array. Each object must contain keys exactly: "
        "Description, Withdrawals, Deposits, Date, Balance. "
        "Date must stay in MONDD format, e.g. MAY07. Amounts use plain decimals without commas."
    )
    payload = {
        "statement_period": {
            "end_month": period.end_month,
            "end_year": period.end_year,
        }
    }
    return f"{rules}\n\nInput JSON:\n{json.dumps(payload, ensure_ascii=True)}"


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
            {
                "role": "system",
                "content": "You output strict JSON only.",
            },
            {
                "role": "user",
                "content": user_content,
            },
        ],
        "temperature": 0,
    }

    req = urllib.request.Request(
        url,
        data=json.dumps(body).encode("utf-8"),
        headers={
            "Authorization": f"Bearer {api_key}",
            "Content-Type": "application/json",
        },
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


def clean_ai_rows(
    ai_rows: list[dict[str, str]],
    period: StatementPeriod,
) -> list[list[str]]:
    output_rows: list[list[str]] = []
    prev_balance: float | None = None

    for row in ai_rows:
        desc = normalize_description(str(row.get("Description", "")))
        wd = normalize_amount(str(row.get("Withdrawals", "")).strip())
        dep = normalize_amount(str(row.get("Deposits", "")).strip())
        date_token = str(row.get("Date", "")).strip().upper()
        bal = normalize_amount(str(row.get("Balance", "")).strip())

        if not desc or not date_token:
            continue

        compact = canonical_desc_key(desc)
        if compact.startswith("TOTAL"):
            continue

        if compact in {"STARTINGBALANCE", "BALANCEFORWARD"}:
            mmddyyyy = month_day_to_mmddyyyy(date_token, period)

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

        if bool(wd) == bool(dep):
            continue

        mmddyyyy = month_day_to_mmddyyyy(date_token, period)

        cur_balance: float | None = None
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

        output_rows.append(
            [
                mmddyyyy,
                desc,
                wd,
                dep,
                f"{cur_balance:.2f}",
            ]
        )
        prev_balance = cur_balance

    return output_rows


def output_csv_path(pdf_path: Path, period: StatementPeriod, output_dir: Path | None) -> Path:
    filename = f"accountactivity-{period.end_year:04d}-{period.end_month:02d}.csv"
    if output_dir:
        return output_dir / filename
    return pdf_path.parent / filename


def convert_file(pdf_path: Path, output_dir: Path | None, model: str, api_key: str, timeout_s: int) -> Path:
    period = parse_statement_period(pdf_path)
    page_data_urls = render_pdf_pages_as_data_urls(pdf_path)
    prompt = build_prompt(period)
    model_text = call_openrouter(model, api_key, prompt, page_data_urls, timeout_s)
    ai_rows = extract_json_array(model_text)
    out_rows = clean_ai_rows(ai_rows, period)

    out_path = output_csv_path(pdf_path, period, output_dir)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with out_path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerows(out_rows)
    return out_path


def find_pdfs(root: Path) -> list[Path]:
    return sorted(p for p in root.rglob("*.pdf") if p.is_file())


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Convert TD chequing statement PDFs with OpenRouter-assisted table normalization.",
    )
    parser.add_argument("--input", default=".", help="Root folder to scan for statement PDFs.")
    parser.add_argument("--output-dir", default=None, help="Optional output folder for CSV files.")
    parser.add_argument("--model", default="anthropic/claude-sonnet-5", help="OpenRouter model id.")
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
