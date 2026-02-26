#!/usr/bin/env python3
"""Generate a benchmark dataset with quality reports.

End-to-end pipeline: ArangoDB → subgraph extraction → LLM query generation →
direction validation → QA checks → triage → reports.

Usage:
    uv run python scripts/benchmark_generate_dataset.py --target 100
    uv run python scripts/benchmark_generate_dataset.py --target 50 --skip-qa
    uv run python scripts/benchmark_generate_dataset.py --target 100 --max-subgraphs 20
    uv run python scripts/benchmark_generate_dataset.py --target 100 --force-regen
"""

from __future__ import annotations

import argparse
import asyncio
import json
import logging
import sys
import time
from datetime import UTC, datetime
from pathlib import Path

# Add project root and scripts/ to sys.path
_SCRIPT_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(_SCRIPT_DIR.parent))
sys.path.insert(0, str(_SCRIPT_DIR))

logging.basicConfig(level=logging.INFO, format="%(levelname)s | %(message)s")
logger = logging.getLogger(__name__)


# ------------------------------------------------------------------ #
# Step 1: Connect ArangoDB + collect valid_node_ids                    #
# ------------------------------------------------------------------ #


def step_connect_arango():
    """Connect to ArangoDB and collect all valid content node IDs."""
    from workflows.dependencies import init_arango_client

    logger.info("Step 1: Connecting to ArangoDB")
    client = init_arango_client()

    # Collect valid_node_ids for QA check 1
    cursor = client.db.aql.execute(
        "FOR doc IN @@coll RETURN doc._id",
        bind_vars={"@coll": "content_nodes"},
    )
    valid_node_ids = set(cursor)
    logger.info("  Connected. %d valid content node IDs collected", len(valid_node_ids))
    return client, valid_node_ids


# ------------------------------------------------------------------ #
# Migration guard                                                       #
# ------------------------------------------------------------------ #


def check_migration_guard(output_dir: str) -> None:
    """Warn if existing checkpoint files lack LLM-assigned directions.

    Pre-v4.0 checkpoints have direction=null (or missing) on all queries,
    meaning they were generated before direction was LLM-assigned.
    Running without --force-regen would silently produce invalid data.
    """
    out = Path(output_dir)
    if not out.exists():
        return

    legacy_files: list[str] = []
    for path in out.glob("*.json"):
        if path.name.startswith("_") or path.name == "all_queries.json":
            continue
        try:
            with open(path) as f:
                data = json.load(f)
            queries = data.get("queries", [])
            if queries and all(q.get("direction") is None for q in queries):
                legacy_files.append(path.name)
        except (json.JSONDecodeError, OSError):
            pass

    if legacy_files:
        logger.warning(
            "MIGRATION WARNING: %d checkpoint file(s) in %s have no LLM-assigned "
            "directions (legacy format). Use --force-regen to regenerate them.",
            len(legacy_files),
            output_dir,
        )
        for f in legacy_files[:5]:
            logger.warning("  %s", f)
        if len(legacy_files) > 5:
            logger.warning("  ... and %d more", len(legacy_files) - 5)


# ------------------------------------------------------------------ #
# Force regeneration                                                    #
# ------------------------------------------------------------------ #


def step_force_regen(output_dir: str) -> None:
    """Remove all per-subgraph checkpoint files to force fresh generation."""
    out = Path(output_dir)
    if not out.exists():
        return
    removed = 0
    for path in out.glob("*.json"):
        if path.name.startswith("_") or path.name == "all_queries.json":
            continue
        path.unlink()
        removed += 1
    if removed:
        logger.info("  Removed %d checkpoint file(s) from %s", removed, output_dir)


# ------------------------------------------------------------------ #
# Step 2-3: Section listing + subgraph extraction                      #
# ------------------------------------------------------------------ #


def step_extract_subgraphs(client, max_subgraphs: int, target_total: int):
    """List candidates, extract subgraphs, rescale allocations."""
    from workflows.benchmark.query_allocator import balance_allocations
    from workflows.benchmark.section_lister import list_candidate_sections
    from workflows.benchmark.subgraph_selector import select_and_extract

    logger.info("Step 2: Listing candidate sections")
    candidates = list_candidate_sections(client)
    logger.info("  Found %d candidates", len(candidates))
    if not candidates:
        logger.error("No candidate sections found. Is the KG populated?")
        sys.exit(1)

    logger.info("Step 3: Extracting subgraphs (max=%d)", max_subgraphs)
    subgraphs, report = select_and_extract(client, candidates, max_subgraphs)
    logger.info(
        "  Extracted %d subgraphs, initial allocation=%d",
        report.selected_count,
        report.total_allocation,
    )
    if not subgraphs:
        logger.error("No subgraphs extracted!")
        sys.exit(1)

    # Rescale allocations from default 1500 to target_total
    subgraphs = balance_allocations(subgraphs, target_total=target_total)
    total_alloc = sum(sg.query_allocation for sg in subgraphs)
    logger.info("  Rescaled allocations to target=%d (actual sum=%d)", target_total, total_alloc)

    for sg in subgraphs:
        logger.info(
            "    %s: %s — %d nodes, %d rels, alloc=%d, tier=%s",
            sg.subgraph_id[:8],
            sg.anchor_title[:40],
            len(sg.nodes),
            len(sg.relations),
            sg.query_allocation,
            sg.tier,
        )

    return subgraphs, report


# ------------------------------------------------------------------ #
# Step 4: Run full generation                                          #
# ------------------------------------------------------------------ #


async def step_generate_queries(
    subgraphs,
    output_dir: str,
    seed: int,
    per_subgraph_allocations: dict | None = None,
    max_concurrency: int = 5,
):
    """Run full generation pipeline (LLM calls + distractors + validation)."""
    from workflows.benchmark.runner import run_full_generation

    logger.info("Step 4: Running full query generation")
    summary = await run_full_generation(
        subgraphs=subgraphs,
        output_dir=output_dir,
        distractor_count=2,
        seed=seed,
        per_subgraph_allocations=per_subgraph_allocations,
        max_concurrency=max_concurrency,
    )
    logger.info(
        "  Generated %d queries (%d validation issues)",
        summary["total_queries"],
        summary["validation_issues"],
    )
    return summary


# ------------------------------------------------------------------ #
# Step 5: Direction validation                                         #
# ------------------------------------------------------------------ #


def step_validate_directions(dataset):
    """Validate LLM-assigned directions vs rule-inferred; log distribution."""
    from collections import Counter

    from workflows.benchmark.direction_classifier import validate_direction

    logger.info("Step 5: Validating direction consistency (LLM-assigned)")
    inconsistent = [q for q in dataset.queries if not validate_direction(q)[0]]
    if inconsistent:
        logger.warning("  %d queries have LLM/rule direction mismatch", len(inconsistent))
        for q in inconsistent[:5]:
            _, msg = validate_direction(q)
            logger.warning("    %s: %s", q.query_id[:8], msg)
    else:
        logger.info("  All directions consistent with rules")

    distribution = dict(Counter(q.direction for q in dataset.queries))
    for direction, count in sorted(distribution.items(), key=lambda x: (x[0] is None, x[0])):
        logger.info("  %s: %d", direction, count)

    return distribution


# ------------------------------------------------------------------ #
# Step 6: QA checks                                                    #
# ------------------------------------------------------------------ #


def step_qa_checks(dataset, valid_node_ids, target_total: int):
    """Run QA checks on classified queries."""
    from workflows.benchmark.qa_checks import run_qa_checks

    # Scale min_per_direction: default 80 is for 1500 queries
    min_per_direction = max(1, target_total * 80 // 1500)

    logger.info("Step 6: Running QA checks (min_per_direction=%d)", min_per_direction)
    expected_sg_ids = {sg.subgraph_id for sg in dataset.subgraphs}

    clean, flagged, qa_report = run_qa_checks(
        queries=dataset.queries,
        valid_node_ids=valid_node_ids,
        query_embeddings=None,
        node_embeddings=None,
        min_per_direction=min_per_direction,
        expected_subgraph_ids=expected_sg_ids,
        dedup_embeddings=None,
    )

    logger.info(
        "  QA: %d clean, %d flagged, %d removed",
        len(clean),
        len(flagged),
        qa_report.total_removed,
    )
    if qa_report.warnings:
        for w in qa_report.warnings:
            logger.warning("  QA Warning: %s", w)

    return clean, flagged, qa_report


# ------------------------------------------------------------------ #
# Step 7: Triage                                                       #
# ------------------------------------------------------------------ #


def step_triage(clean_queries, seed: int):
    """Run triage on clean queries."""
    from workflows.benchmark.triage import triage_queries

    logger.info("Step 7: Running triage")
    triage_result = triage_queries(clean_queries, seed=seed)

    for tier, count in sorted(triage_result.tier_counts.items()):
        sample_count = len(triage_result.tier_details.get(tier, []))
        logger.info("  %s: %d queries, %d for review", tier, count, sample_count)

    return triage_result


# ------------------------------------------------------------------ #
# Step 8: Save final dataset                                           #
# ------------------------------------------------------------------ #


def step_save_dataset(clean_queries, flagged_queries, subgraphs, output_dir, target):
    """Save final benchmark dataset JSON."""
    from workflows.benchmark.models import BenchmarkDataset

    output_path = Path(output_dir) / f"benchmark_dataset_{target}q.json"

    dataset = BenchmarkDataset(
        version="4.0",
        created_at=datetime.now(UTC).isoformat(),
        queries=clean_queries + flagged_queries,
        subgraphs=subgraphs,
        metadata={
            "target": target,
            "clean_count": len(clean_queries),
            "flagged_count": len(flagged_queries),
            "total_count": len(clean_queries) + len(flagged_queries),
        },
    )

    with open(output_path, "w") as f:
        json.dump(dataset.model_dump(mode="json"), f, indent=2, ensure_ascii=False)

    logger.info("Step 8: Saved dataset → %s", output_path)
    return dataset, output_path


# ------------------------------------------------------------------ #
# Main Pipeline                                                        #
# ------------------------------------------------------------------ #


async def async_main(
    target: int,
    max_subgraphs: int,
    output_dir: str,
    skip_qa: bool,
    seed: int,
    force_regen: bool = False,
    concurrency: int = 5,
) -> bool:
    """Run the full benchmark dataset generation pipeline."""
    from benchmark_reports import generate_direction_analysis, generate_quality_report

    start_time = time.time()
    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)

    # Migration guard: warn if existing checkpoints lack LLM-assigned directions
    check_migration_guard(output_dir)

    # Force-regen: wipe checkpoints before proceeding
    if force_regen:
        step_force_regen(output_dir)

    # Step 1: Connect + collect valid node IDs
    client, valid_node_ids = step_connect_arango()

    try:
        # Steps 2-3: Extract subgraphs with rescaled allocations
        subgraphs, selection_report = step_extract_subgraphs(client, max_subgraphs, target)

        # Compute per-subgraph direction allocations (capability-aware, exact-sum)
        from workflows.benchmark.balance import compute_subgraph_allocation

        per_subgraph_allocations = {
            sg.subgraph_id: compute_subgraph_allocation(sg, sg.query_allocation)
            for sg in subgraphs
            if sg.query_allocation > 0
        }
        logger.info(
            "  Computed per-subgraph allocations for %d subgraphs", len(per_subgraph_allocations)
        )

        # Step 4: Generate queries via LLM
        summary = await step_generate_queries(
            subgraphs,
            output_dir,
            seed,
            per_subgraph_allocations=per_subgraph_allocations,
            max_concurrency=concurrency,
        )

        # Load dataset from runner output
        dataset_path = Path(summary["dataset_path"])
        if not dataset_path.exists():
            logger.error("Runner did not produce %s", dataset_path)
            return False

        from workflows.benchmark.models import BenchmarkDataset

        with open(dataset_path) as f:
            raw = json.load(f)
        dataset = BenchmarkDataset(**raw)
        logger.info("  Loaded %d queries from runner output", len(dataset.queries))

        if not dataset.queries:
            logger.warning("0 queries generated — continuing with empty dataset")

        # Step 5: Validate LLM-assigned directions (no reclassification)
        distribution = step_validate_directions(dataset)

        # Step 6: QA checks
        if skip_qa:
            logger.info("Step 6: SKIPPED (--skip-qa)")
            clean_queries = dataset.queries
            flagged_queries = []
            from workflows.benchmark.qa_checks import QAReport

            qa_report = QAReport(
                total_input=len(clean_queries),
                total_output=len(clean_queries),
                total_removed=0,
                total_flagged=0,
                check_stats={},
            )
        else:
            clean_queries, flagged_queries, qa_report = step_qa_checks(
                dataset, valid_node_ids, target
            )

        # Step 7: Triage (on clean queries only)
        triage_result = step_triage(clean_queries, seed)

        # Step 8: Save final dataset
        dataset, final_path = step_save_dataset(
            clean_queries, flagged_queries, subgraphs, output_dir, target
        )

        duration = time.time() - start_time

        # Step 9: Quality report
        generate_quality_report(
            dataset,
            selection_report,
            distribution,
            qa_report,
            triage_result,
            clean_queries,
            flagged_queries,
            output_dir,
            duration,
        )

        # Step 10: Direction analysis
        generate_direction_analysis(dataset, output_dir)

        # Final summary
        logger.info("\n" + "=" * 60)
        logger.info("PIPELINE COMPLETE")
        logger.info("=" * 60)
        logger.info("  Total queries: %d", len(dataset.queries))
        logger.info(
            "  Clean: %d, Flagged: %d, Removed: %d",
            len(clean_queries),
            len(flagged_queries),
            qa_report.total_removed,
        )
        logger.info("  Duration: %.0fs", duration)
        logger.info("  Output: %s", output_dir)
        logger.info("=" * 60)

        return len(clean_queries) > 0

    finally:
        client.close()


def main():
    parser = argparse.ArgumentParser(
        description="Generate benchmark dataset with quality reports",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument(
        "--target",
        type=int,
        default=100,
        help="Total query target (default: 100)",
    )
    parser.add_argument(
        "--max-subgraphs",
        type=int,
        default=15,
        help="Max subgraphs to extract (default: 15)",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default="data/benchmark/dataset_100q",
        help="Output directory (default: data/benchmark/dataset_100q)",
    )
    parser.add_argument(
        "--skip-qa",
        action="store_true",
        help="Skip QA checks for rapid iteration",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="RNG seed (default: 42)",
    )
    parser.add_argument(
        "--force-regen",
        action="store_true",
        help="Delete existing checkpoint files and regenerate from scratch",
    )
    parser.add_argument(
        "--concurrency",
        type=int,
        default=10,
        metavar="N",
        help="Max concurrent LLM calls (default: 5, min: 1)",
    )

    args = parser.parse_args()

    success = asyncio.run(
        async_main(
            target=args.target,
            max_subgraphs=args.max_subgraphs,
            output_dir=args.output_dir,
            skip_qa=args.skip_qa,
            seed=args.seed,
            force_regen=args.force_regen,
            concurrency=max(1, args.concurrency),
        )
    )
    sys.exit(0 if success else 1)


if __name__ == "__main__":
    main()
