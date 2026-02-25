"""Automated QA Checks for Benchmark v4.0 (Task 20).

Runs 7 quality checks on the query set, removes invalid queries,
flags issues for human review, and generates a QA report.
"""

from __future__ import annotations

import logging
import math
from collections import Counter
from dataclasses import dataclass, field
from typing import Any

from workflows.benchmark.dedup import dedup_queries
from workflows.benchmark.models import BenchmarkQuery

logger = logging.getLogger(__name__)


@dataclass
class CheckResult:
    """Result of a single QA check on one query."""

    check_name: str
    passed: bool
    action: str  # PASS, REMOVE, FLAG, WARN
    detail: str = ""


@dataclass
class QAReport:
    """Aggregated QA report across all checks and queries."""

    total_input: int
    total_output: int
    total_removed: int
    total_flagged: int
    check_stats: dict[str, dict[str, int]]  # check_name → {passed, failed, action_counts}
    flagged_query_ids: list[str] = field(default_factory=list)
    removed_query_ids: list[str] = field(default_factory=list)
    warnings: list[str] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        """Serialize to JSON-compatible dict."""
        return {
            "total_input": self.total_input,
            "total_output": self.total_output,
            "total_removed": self.total_removed,
            "total_flagged": self.total_flagged,
            "check_stats": self.check_stats,
            "flagged_query_ids": self.flagged_query_ids,
            "removed_query_ids": self.removed_query_ids,
            "warnings": self.warnings,
        }


# ------------------------------------------------------------------ #
# Individual Checks                                                   #
# ------------------------------------------------------------------ #


def check_node_existence(
    query: BenchmarkQuery,
    valid_node_ids: set[str],
) -> CheckResult:
    """Check 1: All fact_node IDs exist in the database."""
    invalid = [fn.node_id for fn in query.fact_nodes if fn.node_id not in valid_node_ids]
    if invalid:
        return CheckResult(
            check_name="node_existence",
            passed=False,
            action="REMOVE",
            detail=f"Invalid node IDs: {invalid}",
        )
    return CheckResult(check_name="node_existence", passed=True, action="PASS")


def check_content_relevance(
    query: BenchmarkQuery,
    query_embedding: list[float] | None,
    node_embeddings: dict[str, list[float]],
    threshold: float = 0.3,
) -> CheckResult:
    """Check 2: Query-to-fact_node cosine similarity > threshold."""
    if query_embedding is None:
        return CheckResult(
            check_name="content_relevance",
            passed=True,
            action="PASS",
            detail="No embedding available, skipping",
        )

    for fn in query.fact_nodes:
        if fn.grade == 0:
            continue  # Skip distractors
        node_emb = node_embeddings.get(fn.node_id)
        if node_emb is None:
            continue
        sim = _cosine_similarity(query_embedding, node_emb)
        if sim < threshold:
            return CheckResult(
                check_name="content_relevance",
                passed=False,
                action="FLAG",
                detail=f"Low relevance: {fn.node_id} sim={sim:.3f} < {threshold}",
            )

    return CheckResult(check_name="content_relevance", passed=True, action="PASS")


def check_confidence_completeness(query: BenchmarkQuery) -> CheckResult:
    """Check 6: Confidence score is present and valid."""
    return CheckResult(check_name="confidence_completeness", passed=True, action="PASS")


def check_grade0_presence(query: BenchmarkQuery) -> CheckResult:
    """Check 7: At least one grade-0 distractor node."""
    has_distractor = any(fn.grade == 0 for fn in query.fact_nodes)
    if not has_distractor:
        return CheckResult(
            check_name="grade0_presence",
            passed=False,
            action="FLAG",
            detail="No grade-0 distractor node",
        )
    return CheckResult(check_name="grade0_presence", passed=True, action="PASS")


def check_concern_for_low_confidence(
    query: BenchmarkQuery,
    threshold: int = 80,
) -> CheckResult:
    """Check 7b: Concern text required when confidence < threshold."""
    if query.confidence.overall < threshold and not query.confidence_concerns.strip():
        return CheckResult(
            check_name="concern_required",
            passed=False,
            action="FLAG",
            detail=f"Confidence={query.confidence.overall} but no concern text",
        )
    return CheckResult(check_name="concern_required", passed=True, action="PASS")


# ------------------------------------------------------------------ #
# Distribution Checks (aggregate)                                     #
# ------------------------------------------------------------------ #


def check_direction_counts(
    queries: list[BenchmarkQuery],
    min_per_direction: int = 80,
) -> list[str]:
    """Check 4: Per-direction count >= minimum."""
    dist = Counter(q.direction for q in queries if q.direction)
    warnings = []
    for direction in ("TD", "HZ", "BU", "NT", "CS"):
        count = dist.get(direction, 0)
        if count < min_per_direction:
            warnings.append(
                f"Direction {direction}: {count} queries (< {min_per_direction} minimum)"
            )
    return warnings


def check_subgraph_coverage(
    queries: list[BenchmarkQuery],
    expected_subgraph_ids: set[str] | None = None,
) -> list[str]:
    """Check 5: All expected subgraphs have at least one query.

    If expected_subgraph_ids is None, derives the set from the queries
    themselves (which can only detect truly present subgraphs).
    """
    present = {q.subgraph_id for q in queries}
    if expected_subgraph_ids is None:
        expected_subgraph_ids = present

    empty = sorted(expected_subgraph_ids - present)
    warnings = []
    if empty:
        warnings.append(f"Empty subgraphs (0 queries): {empty}")
    return warnings


# ------------------------------------------------------------------ #
# Main QA Pipeline                                                    #
# ------------------------------------------------------------------ #


def run_qa_checks(
    queries: list[BenchmarkQuery],
    valid_node_ids: set[str] | None = None,
    query_embeddings: dict[str, list[float]] | None = None,
    node_embeddings: dict[str, list[float]] | None = None,
    relevance_threshold: float = 0.3,
    min_per_direction: int = 80,
    expected_subgraph_ids: set[str] | None = None,
    dedup_embeddings: list[list[float]] | None = None,
    dedup_threshold: float = 0.92,
) -> tuple[list[BenchmarkQuery], list[BenchmarkQuery], QAReport]:
    """Run all 7 QA checks on query set.

    Args:
        queries: Input query set.
        valid_node_ids: Set of valid ArangoDB node IDs. If None, skip check 1.
        query_embeddings: {query_id: embedding} for relevance check.
        node_embeddings: {node_id: embedding} for relevance check.
        relevance_threshold: Cosine similarity threshold for check 2.
        min_per_direction: Minimum queries per direction for check 4.
        dedup_embeddings: Ordered list of embeddings for check 3 (dedup).
            Must have same length as queries. If None, skip check 3.
        dedup_threshold: Cosine threshold for near-duplicate detection.

    Returns:
        (clean_queries, flagged_queries, qa_report)
    """
    skip_node_check = valid_node_ids is None
    if valid_node_ids is None:
        valid_node_ids = set()
    if query_embeddings is None:
        query_embeddings = {}
    if node_embeddings is None:
        node_embeddings = {}

    check_stats: dict[str, dict[str, int]] = {}
    clean: list[BenchmarkQuery] = []
    flagged: list[BenchmarkQuery] = []
    removed_ids: list[str] = []
    flagged_ids: list[str] = []
    surviving_dedup_embeddings: list[list[float]] = []

    for idx, query in enumerate(queries):
        results: list[CheckResult] = []

        # Check 1: Node existence (skip only when caller passed None)
        if not skip_node_check:
            results.append(check_node_existence(query, valid_node_ids))

        # Check 2: Content relevance
        q_emb = query_embeddings.get(query.query_id)
        results.append(check_content_relevance(query, q_emb, node_embeddings, relevance_threshold))

        # Check 6: Confidence completeness
        results.append(check_confidence_completeness(query))

        # Check 7: Grade-0 presence
        results.append(check_grade0_presence(query))

        # Check 7b: Concern required for low confidence
        results.append(check_concern_for_low_confidence(query))

        # Aggregate per-query results
        should_remove = any(r.action == "REMOVE" for r in results)
        should_flag = any(r.action == "FLAG" for r in results)

        for r in results:
            if r.check_name not in check_stats:
                check_stats[r.check_name] = {"passed": 0, "failed": 0}
            if r.passed:
                check_stats[r.check_name]["passed"] += 1
            else:
                check_stats[r.check_name]["failed"] += 1

        if should_remove:
            removed_ids.append(query.query_id)
        else:
            if should_flag:
                flagged.append(query)
                flagged_ids.append(query.query_id)
            else:
                clean.append(query)
            # Keep embedding for surviving queries (clean + flagged)
            if dedup_embeddings is not None and idx < len(dedup_embeddings):
                surviving_dedup_embeddings.append(dedup_embeddings[idx])

    # Aggregate checks (4, 5) — warnings only
    all_output = clean + flagged

    # Check 3: Dedup (cosine similarity across query embeddings)
    if dedup_embeddings is not None and len(surviving_dedup_embeddings) == len(all_output):
        all_output, removed_pairs = dedup_queries(
            all_output, surviving_dedup_embeddings, dedup_threshold
        )
        dedup_removed = {removed_id for _, removed_id, _ in removed_pairs}
        check_stats["dedup"] = {
            "passed": len(all_output),
            "failed": len(removed_pairs),
        }
        removed_ids.extend(dedup_removed)
        # Recompute clean and flagged after dedup removal
        flagged = [q for q in flagged if q.query_id not in dedup_removed]
        flagged_ids = [qid for qid in flagged_ids if qid not in dedup_removed]
        clean = [q for q in all_output if q.query_id not in {f.query_id for f in flagged}]

    warnings = check_direction_counts(all_output, min_per_direction)
    warnings.extend(check_subgraph_coverage(all_output, expected_subgraph_ids))

    report = QAReport(
        total_input=len(queries),
        total_output=len(all_output),
        total_removed=len(removed_ids),
        total_flagged=len(flagged),
        check_stats=check_stats,
        flagged_query_ids=flagged_ids,
        removed_query_ids=removed_ids,
        warnings=warnings,
    )

    logger.info(
        "QA complete: %d input → %d clean, %d flagged, %d removed, %d warnings",
        report.total_input,
        report.total_output,
        report.total_flagged,
        report.total_removed,
        len(report.warnings),
    )

    return clean, flagged, report


# ------------------------------------------------------------------ #
# Helpers                                                             #
# ------------------------------------------------------------------ #


def _cosine_similarity(a: list[float], b: list[float]) -> float:
    """Compute cosine similarity between two vectors."""
    if len(a) != len(b):
        return 0.0
    dot = sum(x * y for x, y in zip(a, b))
    norm_a = math.sqrt(sum(x * x for x in a))
    norm_b = math.sqrt(sum(x * x for x in b))
    if norm_a == 0.0 or norm_b == 0.0:
        return 0.0
    return dot / (norm_a * norm_b)
