"""Evaluation Report Generator for Benchmark v4.0 (Task 27).

Generates comprehensive markdown report with dual-granularity tables,
per-direction breakdown, significance tests, and hypothesis validation.
"""

from __future__ import annotations

import logging
from collections import Counter
from datetime import UTC, datetime
from pathlib import Path

from workflows.benchmark.models import BenchmarkQuery
from workflows.benchmark.run_evaluation import EvaluationRunSummary
from workflows.benchmark.statistics import StatisticsReport

logger = logging.getLogger(__name__)


def generate_report(
    summary: EvaluationRunSummary,
    stats: StatisticsReport,
    queries: list[BenchmarkQuery],
    output_path: str = "data/benchmark/reports/evaluation_report.md",
) -> str:
    """Generate the full evaluation report as markdown.

    Sections:
        1. Dataset Statistics
        2. Method Performance (content-level)
        3. Per-Direction Breakdown
        4. Statistical Significance
        5. Hypothesis Validation (H1-H8)
        6. Sanity Checks
    """
    sections = [
        _header(),
        _dataset_statistics(queries),
        _method_performance_table(summary),
        _per_direction_table(stats),
        _significance_table(stats),
        _hypothesis_validation(stats),
        _sanity_checks(summary),
    ]

    report = "\n\n".join(sections)

    path = Path(output_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        f.write(report)

    logger.info("Report saved to %s", path)
    return str(path)


# ------------------------------------------------------------------ #
# Report Sections                                                     #
# ------------------------------------------------------------------ #


def _header() -> str:
    timestamp = datetime.now(UTC).strftime("%Y-%m-%d %H:%M UTC")
    return f"# Benchmark Evaluation Report v4.0\n\n**Generated:** {timestamp}\n\n---"


def _dataset_statistics(queries: list[BenchmarkQuery]) -> str:
    total = len(queries)

    directions = Counter(q.direction for q in queries if q.direction)
    subgraphs = len({q.subgraph_id for q in queries})

    # Grade distribution across all fact_nodes
    grade_counts = Counter(fn.grade for q in queries for fn in q.fact_nodes)

    lines = [
        "## 1. Dataset Statistics\n",
        "| Metric | Value |",
        "|--------|-------|",
        f"| Total queries | {total} |",
        f"| Unique subgraphs | {subgraphs} |",
        f"| Avg fact nodes/query | {sum(len(q.fact_nodes) for q in queries) / max(total, 1):.1f} |",
        f"| Grade-0 nodes | {grade_counts.get(0, 0)} |",
        f"| Grade-1 nodes | {grade_counts.get(1, 0)} |",
        f"| Grade-2 nodes | {grade_counts.get(2, 0)} |",
        "",
        "### Direction Distribution\n",
        "| Direction | Count | Percentage |",
        "|-----------|-------|------------|",
    ]
    for d in ("TD", "HZ", "BU", "NT", "CS", "AMBIGUOUS"):
        count = directions.get(d, 0)
        pct = (count / total * 100) if total > 0 else 0.0
        lines.append(f"| {d} | {count} | {pct:.1f}% |")

    return "\n".join(lines)


def _method_performance_table(summary: EvaluationRunSummary) -> str:
    lines = [
        "## 2. Method Performance (Content-Level)\n",
        "| Method | NDCG@10 | Recall@10 | Precision@10 | MRR | Queries |",
        "|--------|---------|-----------|-------------|-----|---------|",
    ]
    for mr in summary.method_results:
        lines.append(
            f"| {mr.method_id} ({mr.method_name}) "
            f"| {mr.avg_ndcg:.4f} | {mr.avg_recall:.4f} "
            f"| {mr.avg_precision:.4f} | {mr.avg_mrr:.4f} | {mr.query_count} |"
        )
    return "\n".join(lines)


def _per_direction_table(stats: StatisticsReport) -> str:
    if not stats.per_direction:
        return "## 3. Per-Direction Breakdown\n\nNo per-direction data available."

    lines = ["## 3. Per-Direction Breakdown\n"]

    for direction in sorted(stats.per_direction.keys()):
        methods = stats.per_direction[direction]
        lines.append(f"\n### {direction}\n")
        lines.append("| Method | NDCG@10 |")
        lines.append("|--------|---------|")
        for mid, avg in sorted(methods.items()):
            lines.append(f"| {mid} | {avg:.4f} |")

    return "\n".join(lines)


def _significance_table(stats: StatisticsReport) -> str:
    lines = [
        "## 4. Statistical Significance\n",
        f"**Friedman test:** χ²={stats.friedman_statistic:.4f}, "
        f"p={stats.friedman_p_value:.6f} "
        f"({'significant' if stats.friedman_significant else 'not significant'})\n",
    ]

    if stats.pairwise_results:
        sig_pairs = [p for p in stats.pairwise_results if p.significant]
        lines.append(
            f"**Significant pairs (BH-corrected):** {len(sig_pairs)}/{len(stats.pairwise_results)}\n"
        )
        lines.append("| Method A | Method B | p-adj | Cliff's δ | Effect |")
        lines.append("|----------|----------|-------|-----------|--------|")
        for pr in sig_pairs[:30]:  # Limit to top 30
            lines.append(
                f"| {pr.method_a} | {pr.method_b} "
                f"| {pr.p_adjusted:.4f} | {pr.cliff_delta:.3f} | {pr.effect_size} |"
            )
    else:
        lines.append("No pairwise tests performed (Friedman not significant).")

    return "\n".join(lines)


def _hypothesis_validation(stats: StatisticsReport) -> str:
    lines = [
        "## 5. Hypothesis Validation\n",
        "| ID | Hypothesis | Supported | Evidence |",
        "|----|-----------|-----------|----------|",
    ]
    for h in stats.hypothesis_results:
        icon = "YES" if h.supported else "NO"
        lines.append(f"| {h.hypothesis_id} | {h.description} | {icon} | {h.evidence[:80]}... |")

    return "\n".join(lines)


def _sanity_checks(summary: EvaluationRunSummary) -> str:
    lines = ["## 6. Sanity Checks\n"]

    # Find B1 and graph-enhanced methods
    method_avgs = {mr.method_id: mr.avg_ndcg for mr in summary.method_results}
    b1 = method_avgs.get("B1", 0.0)

    checks = [
        ("B1 NDCG > 0", b1 > 0.0),
        (
            "B1 < best E* method",
            b1 < max((v for k, v in method_avgs.items() if k.startswith("E")), default=0.0),
        ),
        ("All methods produce results", all(mr.query_count > 0 for mr in summary.method_results)),
    ]

    for desc, passed in checks:
        status = "PASS" if passed else "FAIL"
        lines.append(f"- [{status}] {desc}")

    return "\n".join(lines)
