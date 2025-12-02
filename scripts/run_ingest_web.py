#!/usr/bin/env python3
"""Script to run web ingestion workflow."""

import argparse
import asyncio
import json
import sys
from pathlib import Path
from uuid import UUID

from loguru import logger

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from workflows.ingest_web import WebIngestionInput, start_ingest_web


def main() -> None:
    parser = argparse.ArgumentParser(description="Ingest web URLs")
    parser.add_argument(
        "urls",
        nargs="+",
        help="One or more URLs to ingest",
    )
    parser.add_argument(
        "--created-by",
        type=str,
        required=True,
        help="User ID (UUID) who created this ingestion",
    )
    parser.add_argument(
        "--output",
        type=str,
        help="Output file path for JSON result (optional)",
    )
    parser.add_argument(
        "--max-concurrent-urls",
        type=int,
        default=4,
        help="Maximum number of URLs to crawl concurrently (default: 4)",
    )
    parser.add_argument(
        "--max-retries",
        type=int,
        default=3,
        help="Maximum number of retries per URL (default: 3)",
    )
    parser.add_argument(
        "--timeout",
        type=float,
        default=90.0,
        help="Timeout per URL in seconds (default: 90.0)",
    )

    args = parser.parse_args()

    try:
        created_by = UUID(args.created_by)
    except ValueError:
        logger.error(f"Invalid UUID format for --created-by: {args.created_by}")
        sys.exit(1)

    try:
        logger.info(f"Starting web ingestion for {len(args.urls)} URL(s)")

        payload = WebIngestionInput(
            urls=args.urls,
            created_by=created_by,
            max_concurrent_urls=args.max_concurrent_urls,
            max_url_retries=args.max_retries,
            per_url_timeout_seconds=args.timeout,
        )

        result = asyncio.run(start_ingest_web(payload))

        # Print summary
        print("\n" + "=" * 80)
        print("INGESTION RESULT SUMMARY")
        print("=" * 80)
        total = len(result["results"])
        succeeded = sum(1 for r in result["results"] if r["status"] == "success")
        failed = sum(1 for r in result["results"] if r["status"] == "error")
        skipped = sum(1 for r in result["results"] if r["status"] == "skipped")
        print(f"Total URLs: {total}")
        print(f"  ✓ Succeeded: {succeeded}")
        print(f"  ✗ Failed: {failed}")
        print(f"  ⊘ Skipped: {skipped}")
        print("=" * 80)

        # Print detailed result
        print("\n" + "=" * 80)
        print("DETAILED RESULTS")
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
        import traceback
        traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main()
