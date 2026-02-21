#!/usr/bin/env python3
"""Script to run extraction on unreviewed nodes in ArangoDB.

This script delegates to the extract_relations workflow which implements:
1. Phase 1: Intra-batch deduplication (Union-Find algorithm)
2. Phase 2: Cross-batch processing with multi-node handling
3. Phase 3: Intra-batch relation creation

Key features:
- Proper Phase 1 deduplication prevents race conditions
- Workflow checks node existence before merge operations
- Supports continuous mode to process ALL unreviewed nodes
- Debug mode generates detailed MD reports
- Real-time progress display

Usage:
    python scripts/run_extraction.py
    python scripts/run_extraction.py --continuous
    python scripts/run_extraction.py --use-llm
    python scripts/run_extraction.py --continuous --use-llm --batch-size 50
"""

from __future__ import annotations

import argparse
import asyncio
import sys
import time
from pathlib import Path

from loguru import logger

# Add project root to path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from arango_client.client import ArangoClient
from arango_client.collections.schema import Collections, EdgeCollections

from config.extraction import get_extraction_settings
from config.ontologies import get_current_ontology
from workflows.dependencies import init_arango_client
from workflows.extract_relations import (
    ExtractionMode,
    ExtractionTier,
    ExtractRelationsInput,
    ExtractRelationsWorkflow,
)

# =============================================================================
# Progress Display
# =============================================================================


def print_progress(message: str, tier: str | None = None, **kwargs) -> None:
    """Print progress message immediately to stdout.

    This function uses print with flush=True to ensure immediate display,
    bypassing any logging buffering issues.

    Why print() instead of logger?
    1. Logger buffers output for I/O optimization - delays display
    2. Logger needs structured format serialization overhead
    3. Logger may filter based on log levels
    4. print(flush=True) writes directly to stdout with immediate flush
    """
    prefix = f"[{tier}] " if tier else ""

    # Format different types of progress messages
    if "batch" in kwargs and "total_batches" in kwargs:
        # LLM batch progress - show batch number and pairs count
        batch = kwargs.get("batch", "?")
        total = kwargs.get("total_batches", "?")
        pairs = kwargs.get("pairs", "?")
        print(f"  {prefix}[LLM] Batch {batch}/{total} ({pairs} pairs)", flush=True)
    elif "leaders" in kwargs and "merged" in kwargs:
        # Phase completion - show key metrics
        leaders = kwargs.get("leaders", 0)
        merged = kwargs.get("merged", 0)
        print(f"  {prefix}[DONE] {message}: leaders={leaders}, merged={merged}", flush=True)
    elif "nodes" in kwargs:
        # Phase start with node count
        nodes = kwargs.get("nodes", 0)
        print(f"  {prefix}[START] {message} ({nodes} nodes)", flush=True)
    elif "leaders" in kwargs:
        # Cross-batch processing start
        leaders = kwargs.get("leaders", 0)
        print(f"  {prefix}[START] {message} ({leaders} leaders)", flush=True)
    elif "auto_merge" in kwargs or "llm_pairs" in kwargs:
        # Collection result
        auto_merge = kwargs.get("auto_merge", 0)
        llm_pairs = kwargs.get("llm_pairs", 0)
        print(
            f"  {prefix}[COLLECT] {message}: auto_merge={auto_merge}, llm_pairs={llm_pairs}",
            flush=True,
        )
    elif "result" in kwargs:
        # LLM result summary
        result = kwargs.get("result", {})
        merged = result.get("merged", 0)
        relations = result.get("relations", 0)
        kept = result.get("kept_separate", 0)
        print(
            f"  {prefix}[RESULT] {message}: merged={merged}, "
            f"relations={relations}, kept_separate={kept}",
            flush=True,
        )
    # Generic progress
    elif kwargs:
        details = ", ".join(f"{k}={v}" for k, v in kwargs.items())
        print(f"  {prefix}[PROGRESS] {message} ({details})", flush=True)
    else:
        print(f"  {prefix}[PROGRESS] {message}", flush=True)


# =============================================================================
# Database Utilities
# =============================================================================


def get_total_unreviewed(client: ArangoClient, tier: str) -> int:
    """Get total count of unreviewed nodes."""
    total = 0

    if tier in ("all", "content"):
        query = """
        FOR doc IN @@collection
            FILTER doc.is_reviewed != true
            FILTER doc.embedding != null
            COLLECT WITH COUNT INTO count
            RETURN count
        """
        cursor = client.db.aql.execute(query, bind_vars={"@collection": Collections.CONTENT_NODES})
        total += next(cursor, 0)

    if tier in ("all", "structure"):
        query = """
        FOR doc IN @@collection
            FILTER doc.is_reviewed != true
            FILTER doc.embedding != null
            COLLECT WITH COUNT INTO count
            RETURN count
        """
        cursor = client.db.aql.execute(
            query, bind_vars={"@collection": Collections.STRUCTURE_NODES}
        )
        total += next(cursor, 0)

    return total


def get_node_counts(client: ArangoClient) -> dict[str, int]:
    """Get node counts by collection."""
    counts = {}
    for coll in [Collections.CONTENT_NODES, Collections.STRUCTURE_NODES]:
        if client.db.has_collection(coll):
            counts[coll] = client.db.collection(coll).count()
    return counts


def get_edge_stats(client: ArangoClient) -> dict[str, int]:
    """Get edge statistics by relation type."""
    query = """
    FOR e IN @@edge_collection
        COLLECT type = e.type WITH COUNT INTO count
        RETURN {type: type, count: count}
    """
    cursor = client.db.aql.execute(
        query, bind_vars={"@edge_collection": EdgeCollections.KNOWLEDGE_EDGES}
    )
    return {item["type"]: item["count"] for item in cursor}


def reset_all_nodes(client: ArangoClient, tier: str) -> tuple[int, int]:
    """Reset all nodes to is_reviewed=false.

    Returns:
        Tuple of (content_count, structure_count) nodes reset
    """
    content_count = 0
    structure_count = 0

    if tier in ("all", "content"):
        query = """
        FOR doc IN @@collection
            FILTER doc.is_reviewed == true
            UPDATE doc WITH { is_reviewed: false } IN @@collection
            RETURN 1
        """
        cursor = client.db.aql.execute(query, bind_vars={"@collection": Collections.CONTENT_NODES})
        content_count = sum(1 for _ in cursor)
        logger.info(f"Reset {content_count} content nodes to is_reviewed=false")

    if tier in ("all", "structure"):
        query = """
        FOR doc IN @@collection
            FILTER doc.is_reviewed == true
            UPDATE doc WITH { is_reviewed: false } IN @@collection
            RETURN 1
        """
        cursor = client.db.aql.execute(
            query, bind_vars={"@collection": Collections.STRUCTURE_NODES}
        )
        structure_count = sum(1 for _ in cursor)
        logger.info(f"Reset {structure_count} structure nodes to is_reviewed=false")

    return content_count, structure_count


# =============================================================================
# Report Printing
# =============================================================================


def print_report(
    result: dict,
    edge_stats_before: dict[str, int],
    edge_stats_after: dict[str, int],
    node_counts_before: dict[str, int],
    node_counts_after: dict[str, int],
    settings: object,
    duration: float,
    batches_completed: int,
) -> None:
    """Print extraction report."""
    print("\n" + "=" * 60)
    print("EXTRACTION REPORT")
    print("=" * 60)

    # Processing stats
    print("\n[Processing]")
    print(f"  Batches completed: {batches_completed}")
    print(f"  Duration: {duration:.1f}s")
    print(f"  Nodes processed: {result.get('nodes_processed', 0)}")
    print(f"  Nodes merged: {result.get('nodes_merged', 0)}")
    print(f"  Relations created: {result.get('relations_created', 0)}")
    print(f"  Nodes kept separate: {result.get('nodes_kept_separate', 0)}")
    print(f"  LLM calls used: {result.get('llm_calls_used', 0)}")

    # New relation stats
    if result.get("new_relations_proposed", 0) > 0:
        print("\n[New Relations]")
        print(f"  Proposed: {result.get('new_relations_proposed', 0)}")
        print(f"  Merged with existing: {result.get('new_relations_merged', 0)}")
        print(f"  Approved as new: {result.get('new_relations_approved', 0)}")

    # Errors
    if result.get("errors"):
        print(f"\n[Errors]: {len(result['errors'])}")
        for err in result["errors"][:5]:
            print(f"  - {err}")
        if len(result["errors"]) > 5:
            print(f"  ... and {len(result['errors']) - 5} more")

    # Edge delta
    print("\n[Edge Changes]")
    all_types = set(edge_stats_before.keys()) | set(edge_stats_after.keys())
    for rel_type in sorted(all_types):
        before = edge_stats_before.get(rel_type, 0)
        after = edge_stats_after.get(rel_type, 0)
        delta = after - before
        if delta != 0:
            sign = "+" if delta > 0 else ""
            print(f"  {rel_type}: {before} -> {after} ({sign}{delta})")

    # Node counts
    print("\n[Node Counts]")
    for coll in sorted(node_counts_before.keys()):
        before = node_counts_before.get(coll, 0)
        after = node_counts_after.get(coll, 0)
        delta = after - before
        if delta != 0:
            sign = "+" if delta > 0 else ""
            print(f"  {coll}: {before} -> {after} ({sign}{delta})")
        else:
            print(f"  {coll}: {after}")

    # Thresholds used
    print("\n[Thresholds]")
    print(f"  Auto-merge: >= {settings.EXTRACTION_AUTO_MERGE_THRESHOLD}")
    print(
        f"  LLM zone: {settings.EXTRACTION_LLM_THRESHOLD} - {settings.EXTRACTION_AUTO_MERGE_THRESHOLD}"
    )
    print(f"  Fallback relation: {settings.EXTRACTION_FALLBACK_RELATION}")

    print("\n" + "=" * 60)


# =============================================================================
# Main Async Function
# =============================================================================


async def async_main(
    *,
    tier: str = "all",
    dry_run: bool = False,
    batch_size: int = 100,
    continuous: bool = False,
    use_llm: bool = True,
    max_batches: int | None = None,
    debug_mode: bool = False,
    reset_nodes: bool = False,
) -> None:
    """Main async entry point - delegates to workflow."""
    settings = get_extraction_settings()
    ontology = get_current_ontology()

    # Initialize client
    client = init_arango_client()

    try:
        # Reset nodes if requested
        if reset_nodes:
            logger.warning("RESET MODE: Resetting all nodes to is_reviewed=false")
            content_reset, structure_reset = reset_all_nodes(client, tier)
            logger.info(f"Reset complete: {content_reset} content, {structure_reset} structure")

        # Get initial stats
        edge_stats_before = get_edge_stats(client)
        node_counts_before = get_node_counts(client)
        total_unreviewed = get_total_unreviewed(client, tier)

        logger.info(f"Total unreviewed nodes: {total_unreviewed}")

        if total_unreviewed == 0:
            logger.info("No unreviewed nodes to process")
            return

        # Log mode
        if dry_run:
            logger.warning("DRY-RUN MODE: No changes will be made to database")

        if continuous:
            logger.info(f"CONTINUOUS MODE: Processing ALL nodes in batches of {batch_size}")

        # Convert tier string to enum
        tier_enum = ExtractionTier(tier)

        # Track overall stats
        start_time = time.time()
        batches_completed = 0
        total_result: dict = {
            "nodes_processed": 0,
            "nodes_merged": 0,
            "relations_created": 0,
            "nodes_kept_separate": 0,
            "llm_calls_used": 0,
            "new_relations_proposed": 0,
            "new_relations_merged": 0,
            "new_relations_approved": 0,
            "errors": [],
        }

        # Create workflow instance with progress callback
        workflow = ExtractRelationsWorkflow(
            db=client.db,
            settings=settings,
            ontology=ontology,
            progress_callback=print_progress,
        )

        if debug_mode:
            workflow.enable_debug_logging()

        # Process batches
        batch_num = 0
        while True:
            batch_num += 1

            # Check batch limit
            if max_batches and batch_num > max_batches:
                print(f"Reached max batches limit ({max_batches})", flush=True)
                break

            # Check remaining
            remaining = get_total_unreviewed(client, tier)
            if remaining == 0:
                print("No more unreviewed nodes to process", flush=True)
                break

            print(f"\n{'=' * 50}", flush=True)
            print(f"BATCH {batch_num} ({remaining} remaining)", flush=True)
            print(f"{'=' * 50}", flush=True)

            # Create input for this batch
            input_data = ExtractRelationsInput(
                mode=ExtractionMode.BATCH,
                tier=tier_enum,
                batch_size=batch_size,
                max_llm_calls=999999 if use_llm else 0,  # No limit when LLM enabled
                dry_run=dry_run,
                cross_source=settings.EXTRACTION_CROSS_SOURCE,
            )

            # Run workflow
            result = await workflow.run(input_data)

            # Aggregate results
            total_result["nodes_processed"] += result.nodes_processed
            total_result["nodes_merged"] += result.nodes_merged
            total_result["relations_created"] += result.relations_created
            total_result["nodes_kept_separate"] += result.nodes_kept_separate
            total_result["llm_calls_used"] += result.llm_calls_used
            total_result["new_relations_proposed"] += result.new_relations_proposed
            total_result["new_relations_merged"] += result.new_relations_merged
            total_result["new_relations_approved"] += result.new_relations_approved
            total_result["errors"].extend(result.errors)

            batches_completed += 1

            # Log batch result - use print for immediate display
            print(
                f"  Batch {batch_num} complete: "
                f"processed={result.nodes_processed}, "
                f"merged={result.nodes_merged}, "
                f"relations={result.relations_created}, "
                f"llm_calls={result.llm_calls_used}",
                flush=True,
            )

            # Check if we should continue
            if result.nodes_processed == 0:
                print("No nodes processed in this batch", flush=True)
                break

            if not continuous:
                print("Single batch mode, stopping", flush=True)
                break

        # End timing
        duration = time.time() - start_time

        # Get final stats
        edge_stats_after = get_edge_stats(client)
        node_counts_after = get_node_counts(client)

        # Print report
        print_report(
            result=total_result,
            edge_stats_before=edge_stats_before,
            edge_stats_after=edge_stats_after,
            node_counts_before=node_counts_before,
            node_counts_after=node_counts_after,
            settings=settings,
            duration=duration,
            batches_completed=batches_completed,
        )

        # Debug report path
        if debug_mode and hasattr(workflow, "_debug_logger") and workflow._debug_logger:
            report_path = workflow._debug_logger.finish_run()
            if report_path:
                print(f"\n[DEBUG REPORT]\n  Path: {report_path}")

    finally:
        client.close()


# =============================================================================
# CLI Entry Point
# =============================================================================


def main() -> None:
    """Main entry point."""
    parser = argparse.ArgumentParser(
        description="Run extraction on unreviewed nodes (delegates to workflow)",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Run single batch of 100 nodes
  python scripts/run_extraction.py

  # Run until ALL unreviewed nodes are processed
  python scripts/run_extraction.py --continuous

  # Use LLM to extract semantic relations (default: enabled)
  python scripts/run_extraction.py --use-llm

  # Continuous + LLM with custom batch size
  python scripts/run_extraction.py --continuous --use-llm --batch-size 50

  # Dry run (preview without changes)
  python scripts/run_extraction.py --dry-run --continuous

  # Process only content nodes
  python scripts/run_extraction.py --tier content --continuous

  # Limit to 5 batches max
  python scripts/run_extraction.py --continuous --max-batches 5

  # Reset all nodes to unreviewed and run extraction (for testing)
  python scripts/run_extraction.py --reset --use-llm --debug
        """,
    )
    parser.add_argument(
        "--tier",
        choices=["all", "content", "structure"],
        default="all",
        help="Which tier to process (default: all)",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Preview changes without modifying database",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=200,
        help="Number of nodes per batch (default: 100)",
    )
    parser.add_argument(
        "--continuous",
        action="store_true",
        help="Process ALL unreviewed nodes (loop until none left)",
    )
    parser.add_argument(
        "--use-llm",
        action="store_true",
        default=True,
        help="Use LLM to extract semantic relations (default: enabled)",
    )
    parser.add_argument(
        "--no-llm",
        action="store_true",
        help="Disable LLM extraction (use only SIMILAR_TO)",
    )
    parser.add_argument(
        "--max-batches",
        type=int,
        default=None,
        help="Maximum number of batches to run (for testing)",
    )
    parser.add_argument(
        "--verbose",
        "-v",
        action="store_true",
        help="Enable verbose logging (show debug messages)",
    )
    parser.add_argument(
        "--debug",
        action="store_true",
        help="Enable debug logging - generate MD report with full decision log",
    )
    parser.add_argument(
        "--reset",
        action="store_true",
        help="Reset all nodes to is_reviewed=false before running (for testing)",
    )

    args = parser.parse_args()

    # Configure logging
    logger.remove()
    if args.verbose:
        logger.add(sys.stderr, level="DEBUG")
    else:
        logger.add(sys.stderr, level="INFO")

    # Determine LLM usage (--no-llm overrides default)
    use_llm = args.use_llm and not args.no_llm

    # Run
    asyncio.run(
        async_main(
            tier=args.tier,
            dry_run=args.dry_run,
            batch_size=args.batch_size,
            continuous=args.continuous,
            use_llm=use_llm,
            max_batches=args.max_batches,
            debug_mode=args.debug,
            reset_nodes=args.reset,
        )
    )


if __name__ == "__main__":
    main()
