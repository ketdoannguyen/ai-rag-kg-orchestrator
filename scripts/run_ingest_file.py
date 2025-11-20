#!/usr/bin/env python
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT_DIR = Path(__file__).resolve().parents[1]
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from config.settings import get_settings
from ingest_file import start_ingest


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run the ingest_file workflow on a PDF.")
    parser.add_argument(
        "--pdf_path",
        help="Path to the input PDF file.",
    )
    parser.add_argument(
        "--ensure-exists",
        action="store_true",
        help="Fail early if the provided file path does not exist.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    pdf_path = Path(args.pdf_path).expanduser()
    if args.ensure_exists and not pdf_path.exists():
        raise FileNotFoundError(f"Input file not found: {pdf_path}")

    settings = get_settings()
    result = start_ingest(file_path=str(pdf_path), database_url=settings.DATABASE_URL)
    print(json.dumps(result, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()

