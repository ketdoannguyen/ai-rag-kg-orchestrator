#!/usr/bin/env python3
"""Script to run web ingestion workflow."""

import argparse
import json
import sys
from pathlib import Path

from loguru import logger

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from ingest_web import start_ingest


def main() -> None:
    parser = argparse.ArgumentParser(description="Ingest web URLs")
    parser.add_argument(
        "urls",
        nargs="+",
        help="One or more URLs to ingest",
    )
    parser.add_argument(
        "--enable-ocr",
        action="store_true",
        help="Enable OCR processing (disabled by default)",
    )
    parser.add_argument(
        "--output",
        type=str,
        help="Output file path for JSON result (optional)",
    )

    args = parser.parse_args()

    try:
        logger.info(f"Starting web ingestion for {len(args.urls)} URL(s)")
        result = start_ingest(urls=args.urls, enable_ocr=args.enable_ocr)

        # Print result
        print("\n" + "=" * 80)
        print("INGESTION RESULT")
        print("=" * 80)
        print(json.dumps(result, indent=2, ensure_ascii=False))
        print("=" * 80 + "\n")

        # Save to file if requested
        if args.output:
            output_path = Path(args.output)
            output_path.parent.mkdir(parents=True, exist_ok=True)
            output_path.write_text(
                json.dumps(result, indent=2, ensure_ascii=False), encoding="utf-8"
            )
            logger.info(f"Result saved to: {output_path}")

        logger.success("Web ingestion completed successfully")
    except Exception as e:
        logger.error(f"Error during web ingestion: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()




