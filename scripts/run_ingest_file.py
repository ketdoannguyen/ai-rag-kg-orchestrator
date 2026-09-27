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
    parser = argparse.ArgumentParser(
        description="Ingest an uploaded DataSource with the vision OCR workflow"
    )
    parser.add_argument(
        "data_source_id",
        type=str,
        help="UUID of the DataSource created by the upload_file workflow",
    )
    parser.add_argument(
        "--max-retries",
        type=int,
        default=2,
        help="Maximum retries per page (default: 2, max: 5)",
    )
    parser.add_argument(
        "--timeout-seconds",
        type=float,
        default=300.0,
        help="Timeout per vision request in seconds (default: 300, max: 1800)",
    )
    parser.add_argument(
        "--output",
        type=str,
        help="Output file path for JSON result (optional)",
    )

    args = parser.parse_args()

    try:
        input_payload = FileIngestionInput(
            data_source_id=args.data_source_id,
            max_retries=args.max_retries,
            timeout_seconds=args.timeout_seconds,
        )

        logger.info(f"Starting file ingestion for DataSource {args.data_source_id}")
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

        if result.get("status") != "complete":
            logger.error(f"Ingestion failed: {result.get('error')}")
            sys.exit(1)

        logger.success("File ingestion completed successfully")
    except Exception as e:
        logger.error(f"Error during file ingestion: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()
