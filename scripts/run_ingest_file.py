#!/usr/bin/env python3
"""Script to run file ingestion workflow."""

import argparse
import json
import sys
from pathlib import Path

from loguru import logger

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from workflows.ingest_file import FileIngestionInput, start_ingest


def main() -> None:
    parser = argparse.ArgumentParser(description="Ingest PDF file using Marker SDK")
    parser.add_argument(
        "file_path",
        type=str,
        help="Path to the input PDF file",
    )
    parser.add_argument(
        "--enable-ocr",
        action="store_true",
        help="Enable OCR processing (disabled by default)",
    )
    parser.add_argument(
        "--max-retries",
        type=int,
        default=2,
        help="Maximum number of retries for operations (default: 2, max: 5)",
    )
    parser.add_argument(
        "--timeout-seconds",
        type=float,
        default=300.0,
        help="Timeout for conversion in seconds (default: 300, max: 1800)",
    )
    parser.add_argument(
        "--output",
        type=str,
        help="Output file path for JSON result (optional)",
    )

    args = parser.parse_args()

    try:
        # Validate file path
        file_path = Path(args.file_path).expanduser().resolve()
        if not file_path.exists():
            raise FileNotFoundError(f"Input file not found: {file_path}")
        if not file_path.is_file():
            raise ValueError(f"Path is not a file: {file_path}")
        if file_path.suffix.lower() != ".pdf":
            raise ValueError(f"Only PDF files are supported, got: {file_path.suffix}")

        # Create input payload
        input_payload = FileIngestionInput(
            file_path=file_path,
            enable_ocr=args.enable_ocr,
            max_retries=args.max_retries,
            timeout_seconds=args.timeout_seconds,
        )

        logger.info(f"Starting file ingestion for {file_path.name}")
        result = start_ingest(payload=input_payload)

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

        logger.success("File ingestion completed successfully")
    except Exception as e:
        logger.error(f"Error during file ingestion: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()

