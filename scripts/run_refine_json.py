#!/usr/bin/env python3
"""Script to run refine_json workflow from command line."""

import argparse
import asyncio
import json
import sys
from pathlib import Path

from loguru import logger

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from config.settings import settings
from config.llm_gateway import build_llm_gateway_client
from workflows.refine_json import TextbookRefiner


async def main() -> None:
    parser = argparse.ArgumentParser(description="Refine textbook JSON content using LLM")
    parser.add_argument(
        "input_path",
        type=str,
        help="Path to input JSON file (processed textbook)",
    )
    parser.add_argument(
        "output_path",
        type=str,
        help="Path to output JSON file (refined textbook)",
    )
    parser.add_argument(
        "--timeout",
        type=float,
        default=60.0,
        help="Timeout for LLM calls in seconds (default: 60.0)",
    )

    args = parser.parse_args()

    input_path = Path(args.input_path)
    output_path = Path(args.output_path)

    if not input_path.exists():
        logger.error(f"Input file not found: {input_path}")
        sys.exit(1)

    # Ensure output directory exists
    output_path.parent.mkdir(parents=True, exist_ok=True)

    try:
        logger.info(f"Starting refinement: {input_path} -> {output_path}")

        # Initialize LLM Client
        llm_client = build_llm_gateway_client(settings)

        # Create Refiner
        refiner = TextbookRefiner(llm_client=llm_client, model=settings.LLM_GATEWAY_CHAT_MODEL)

        # Run Refinement
        result = await refiner.refine(
            input_path=str(input_path), output_path=str(output_path), llm_timeout=args.timeout
        )

        # Print result
        print("\n" + "=" * 80)
        print("REFINEMENT RESULT")
        print("=" * 80)
        print(json.dumps(result.model_dump(), indent=2, ensure_ascii=False))
        print("=" * 80 + "\n")

        if result.success:
            logger.success(f"✅ Refined {result.paragraphs_refined} paragraphs")
            logger.info(f"📁 Output saved to: {result.output_path}")
            if result.minio_object:
                logger.info(f"☁️  Uploaded to MinIO: {result.minio_object}")
        else:
            logger.error(f"❌ Refinement failed: {result.error}")
            sys.exit(1)

    except Exception as e:
        logger.exception(f"Error during refinement: {e}")
        sys.exit(1)


if __name__ == "__main__":
    asyncio.run(main())
