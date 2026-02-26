"""Full-Scale Generation Orchestrator for Benchmark v4.0 (Task 15).

Coordinates the end-to-end pipeline: list sections → select subgraphs →
extract → allocate → generate queries → inject distractors → export.
Supports checkpoint resume via per-subgraph JSON files + _progress.json.
"""

from __future__ import annotations

import json
import logging
import random
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from workflows.benchmark.models import (
    BenchmarkDataset,
    BenchmarkQuery,
    FactNode,
    Subgraph,
)
from workflows.benchmark.query_generator import (
    build_benchmark_llm_client,
    generate_batch,
)

logger = logging.getLogger(__name__)

DEFAULT_OUTPUT_DIR = "data/benchmark/queries"
DEFAULT_DISTRACTOR_COUNT = 2


# ------------------------------------------------------------------ #
# Distractor Injection                                                #
# ------------------------------------------------------------------ #


def inject_distractors(
    query: BenchmarkQuery,
    subgraph: Subgraph,
    count: int = DEFAULT_DISTRACTOR_COUNT,
    rng: random.Random | None = None,
) -> BenchmarkQuery:
    """Add grade-0 distractor nodes from different sections.

    Selects content nodes from parent sections that do NOT contain
    any grade-2 fact nodes. Falls back to any non-existing node
    if insufficient cross-section candidates.

    Args:
        query: The query to augment.
        subgraph: Source subgraph with all available nodes.
        count: Number of distractors to add.
        rng: Optional Random instance for reproducibility.

    Returns:
        New BenchmarkQuery with distractor FactNodes appended.
    """
    if rng is None:
        rng = random.Random()

    existing_ids = {fn.node_id for fn in query.fact_nodes}

    # Sections containing grade-2 nodes — distractors should come from OTHER sections.
    # FactNode.parent_section holds human-readable titles (from LLM output),
    # so compare against SubgraphNode.parent_section_title (same space).
    grade2_sections = {fn.parent_section for fn in query.fact_nodes if fn.grade == 2}

    # Prefer nodes from different sections than grade-2 nodes
    cross_section = [
        n
        for n in subgraph.nodes
        if n.node_id not in existing_ids
        and n.parent_section_title is not None
        and n.parent_section_title not in grade2_sections
    ]

    # Fallback: any node not already in the query
    all_candidates = [n for n in subgraph.nodes if n.node_id not in existing_ids]

    if not cross_section and not all_candidates:
        return query

    # Prioritise cross-section distractors, then fill remaining slots
    # from same-section candidates so we never drop available cross-section nodes.
    if len(cross_section) >= count:
        selected = rng.sample(cross_section, count)
    else:
        selected = list(cross_section)  # take all cross-section first
        same_section = [c for c in all_candidates if c not in cross_section]
        remaining = count - len(selected)
        if same_section and remaining > 0:
            selected += rng.sample(same_section, min(remaining, len(same_section)))
    distractors = [
        FactNode(
            node_id=n.node_id,
            grade=0,
            parent_section=n.parent_section_title or n.parent_section_key or "unknown",
        )
        for n in selected
    ]

    return query.model_copy(update={"fact_nodes": list(query.fact_nodes) + distractors})


def inject_distractors_batch(
    queries: list[BenchmarkQuery],
    subgraph: Subgraph,
    count: int = DEFAULT_DISTRACTOR_COUNT,
    seed: int = 42,
) -> list[BenchmarkQuery]:
    """Inject distractors into all queries for a given subgraph.

    Uses a seeded RNG for reproducibility.
    """
    rng = random.Random(seed)
    return [inject_distractors(q, subgraph, count, rng) for q in queries]


# ------------------------------------------------------------------ #
# Progress Tracking                                                   #
# ------------------------------------------------------------------ #


def _load_progress(output_dir: Path) -> dict[str, Any]:
    """Load progress file, or return empty state."""
    progress_path = output_dir / "_progress.json"
    if progress_path.exists():
        with open(progress_path) as f:
            return json.load(f)
    return {"completed": [], "started_at": datetime.now(UTC).isoformat()}


def _save_progress(output_dir: Path, progress: dict[str, Any]) -> None:
    """Save progress to _progress.json."""
    progress_path = output_dir / "_progress.json"
    progress["updated_at"] = datetime.now(UTC).isoformat()
    with open(progress_path, "w") as f:
        json.dump(progress, f, indent=2, ensure_ascii=False)


_NON_SUBGRAPH_FILES = {"all_queries.json", "benchmark_dataset_100q.json"}


def _get_completed_subgraphs(output_dir: Path) -> set[str]:
    """Scan output directory for completed subgraph checkpoint files."""
    completed = set()
    for path in output_dir.glob("*.json"):
        if path.name.startswith("_") or path.name in _NON_SUBGRAPH_FILES:
            continue
        completed.add(path.stem)
    return completed


# ------------------------------------------------------------------ #
# Validation                                                          #
# ------------------------------------------------------------------ #


def validate_queries(
    queries: list[BenchmarkQuery],
    subgraph: Subgraph,
) -> tuple[list[BenchmarkQuery], list[dict]]:
    """Validate generated queries against subgraph node inventory.

    Checks:
        - All fact_node IDs exist in subgraph
        - At least one grade-2 node
        - At least one grade-0 node (distractor)
        - Confidence score present (overall >= 0)

    Returns:
        (valid_queries, issues) where issues contains per-query failure details.
    """
    valid_ids = {n.node_id for n in subgraph.nodes}
    valid = []
    issues = []

    for q in queries:
        problems = []

        # Check node existence
        invalid_nodes = [fn.node_id for fn in q.fact_nodes if fn.node_id not in valid_ids]
        if invalid_nodes:
            problems.append(f"invalid_node_ids={invalid_nodes}")

        # Check grade-2 presence
        if not any(fn.grade == 2 for fn in q.fact_nodes):
            problems.append("no_grade_2")

        # Check grade-0 presence (distractor)
        if not any(fn.grade == 0 for fn in q.fact_nodes):
            problems.append("no_grade_0_distractor")

        # Check confidence
        if q.confidence.overall < 0:
            problems.append("negative_confidence")

        if problems:
            issues.append(
                {
                    "query_id": q.query_id,
                    "problems": problems,
                }
            )
        else:
            valid.append(q)

    return valid, issues


# ------------------------------------------------------------------ #
# Full Pipeline Runner                                                #
# ------------------------------------------------------------------ #


async def run_full_generation(
    subgraphs: list[Subgraph],
    output_dir: str = DEFAULT_OUTPUT_DIR,
    distractor_count: int = DEFAULT_DISTRACTOR_COUNT,
    seed: int = 42,
    per_subgraph_allocations: dict[str, dict[str, int]] | None = None,
    context_mode: str = "full",
    max_concurrency: int = 5,
) -> dict[str, Any]:
    """Run the full query generation pipeline with distractors.

    Pipeline:
        1. Build LLM client
        2. Generate queries per subgraph (with checkpoint resume)
        3. Inject distractors into each query
        4. Validate all queries
        5. Export BenchmarkDataset

    Args:
        subgraphs: Pre-extracted subgraphs with query allocations.
        output_dir: Directory for checkpoints and final output.
        distractor_count: Number of grade-0 distractors per query.
        seed: RNG seed for reproducibility.
        per_subgraph_allocations: Mapping of subgraph_id → direction allocation dict.
            If None, generation uses default even distribution.
        context_mode: Context format mode passed to ``generate_batch``.
        max_concurrency: Maximum number of concurrent LLM calls.

    Returns:
        Summary statistics dict.
    """
    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)

    # Step 1: Generate queries (checkpoint-resumable)
    llm_client = build_benchmark_llm_client()
    gen_stats = await generate_batch(
        llm_client,
        subgraphs,
        output_dir,
        per_subgraph_allocations=per_subgraph_allocations,
        context_mode=context_mode,
        max_concurrency=max_concurrency,
    )

    logger.info(
        "Generation complete: %d valid, %d rejected across %d subgraphs",
        gen_stats["total_valid"],
        gen_stats["total_rejected"],
        gen_stats["subgraphs_processed"],
    )

    # Step 2: Load all generated queries and inject distractors
    subgraph_map = {sg.subgraph_id: sg for sg in subgraphs}
    all_queries: list[BenchmarkQuery] = []
    all_issues: list[dict] = []

    # Known aggregate/report files to skip when scanning checkpoints
    _SKIP_FILES = {"all_queries.json"}

    for checkpoint_path in sorted(out.glob("*.json")):
        if checkpoint_path.name.startswith("_") or checkpoint_path.name in _SKIP_FILES:
            continue

        with open(checkpoint_path) as f:
            data = json.load(f)

        sg_id = data.get("subgraph_id", checkpoint_path.stem)
        sg = subgraph_map.get(sg_id)
        if sg is None:
            logger.warning("No subgraph found for checkpoint %s", sg_id)
            continue

        raw_queries = [BenchmarkQuery(**q) for q in data.get("queries", [])]

        # Inject distractors
        augmented = inject_distractors_batch(raw_queries, sg, distractor_count, seed)

        # Validate
        valid, issues = validate_queries(augmented, sg)
        all_queries.extend(valid)
        all_issues.extend(issues)

    # Step 3: Export as BenchmarkDataset
    dataset = BenchmarkDataset(
        version="4.0",
        created_at=datetime.now(UTC).isoformat(),
        queries=all_queries,
        subgraphs=subgraphs,
        metadata={
            "total_queries": len(all_queries),
            "total_validation_issues": len(all_issues),
            "distractor_count": distractor_count,
            "generation_stats": gen_stats,
        },
    )

    dataset_path = out / "all_queries.json"
    with open(dataset_path, "w") as f:
        json.dump(dataset.model_dump(mode="json"), f, indent=2, ensure_ascii=False)

    # Save progress
    progress = _load_progress(out)
    progress["completed"] = list(_get_completed_subgraphs(out))
    progress["total_queries"] = len(all_queries)
    progress["validation_issues"] = len(all_issues)
    _save_progress(out, progress)

    summary = {
        "total_queries": len(all_queries),
        "validation_issues": len(all_issues),
        "subgraphs_processed": gen_stats["subgraphs_processed"],
        "dataset_path": str(dataset_path),
        "issues": all_issues[:20],  # First 20 for debugging
    }

    logger.info(
        "Pipeline complete: %d queries, %d issues, saved to %s",
        len(all_queries),
        len(all_issues),
        dataset_path,
    )

    return summary
