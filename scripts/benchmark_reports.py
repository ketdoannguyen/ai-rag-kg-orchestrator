"""Report generators for benchmark dataset pipeline.

Generates quality_report.md and direction_analysis.md from pipeline results.
"""

from __future__ import annotations

import logging
from collections import Counter
from datetime import UTC, datetime
from pathlib import Path

logger = logging.getLogger(__name__)


def generate_quality_report(
    dataset,
    selection_report,
    distribution,
    qa_report,
    triage_result,
    clean_queries,
    flagged_queries,
    output_dir: str,
    duration: float,
) -> Path:
    """Generate quality_report.md with 9 sections."""
    from workflows.benchmark.balance import DIRECTION_TARGETS

    all_queries = dataset.queries
    total = len(all_queries)
    lines = ["# Benchmark Dataset Quality Report\n"]
    lines.append(f"Generated: {datetime.now(UTC).strftime('%Y-%m-%d %H:%M UTC')}\n")

    # Section 1: Overview
    lines.append("## 1. Overview\n")
    lines.append("| Metric | Value |")
    lines.append("|--------|-------|")
    lines.append(f"| Total queries | {total} |")
    lines.append(f"| Clean queries | {len(clean_queries)} |")
    lines.append(f"| Flagged queries | {len(flagged_queries)} |")
    lines.append(f"| Removed queries | {qa_report.total_removed} |")
    lines.append(f"| Subgraphs | {len(dataset.subgraphs)} |")
    lines.append(f"| Avg relation coverage | {selection_report.avg_relation_coverage:.1%} |")
    lines.append(f"| Pipeline duration | {duration:.0f}s |")
    lines.append("")

    # Section 2: Subgraph Table
    lines.append("## 2. Subgraph Summary\n")
    lines.append("| Subgraph | Tier | Nodes | Relations | Alloc | Generated |")
    lines.append("|----------|------|-------|-----------|-------|-----------|")
    sg_query_counts = Counter(q.subgraph_id for q in all_queries)
    for sg in dataset.subgraphs:
        gen_count = sg_query_counts.get(sg.subgraph_id, 0)
        lines.append(
            f"| {sg.anchor_title[:35]} | {sg.tier} | {len(sg.nodes)} "
            f"| {len(sg.relations)} | {sg.query_allocation} | {gen_count} |"
        )
    lines.append("")

    # Section 3: Direction Distribution
    lines.append("## 3. Direction Distribution\n")
    lines.append("| Direction | Count | Actual % | Target % | Status |")
    lines.append("|-----------|-------|----------|----------|--------|")
    for d in ("TD", "HZ", "BU", "NT", "CS", "AMBIGUOUS"):
        count = distribution.get(d, 0)
        actual_pct = (count / total * 100) if total > 0 else 0
        target_pct = DIRECTION_TARGETS.get(d, 0) * 100
        threshold = max(1, total * DIRECTION_TARGETS.get(d, 0) * 0.8)
        status = "OK" if count >= threshold else "DEFICIT"
        if d == "AMBIGUOUS":
            status = "—"
        lines.append(f"| {d} | {count} | {actual_pct:.1f}% | {target_pct:.0f}% | {status} |")
    lines.append("")

    # Section 4: QA Check Results
    lines.append("## 4. QA Check Results\n")
    lines.append("| Check | Passed | Failed |")
    lines.append("|-------|--------|--------|")
    for check_name, stats in sorted(qa_report.check_stats.items()):
        passed = stats.get("passed", 0)
        failed = stats.get("failed", 0)
        lines.append(f"| {check_name} | {passed} | {failed} |")
    if qa_report.warnings:
        lines.append("\n**Warnings:**\n")
        for w in qa_report.warnings:
            lines.append(f"- {w}")
    lines.append("")

    # Section 5: Confidence Distribution
    lines.append("## 5. Confidence Distribution\n")
    buckets = {"85-100": 0, "75-84": 0, "50-74": 0, "0-49": 0}
    for q in all_queries:
        c = q.confidence.overall
        if c >= 85:
            buckets["85-100"] += 1
        elif c >= 75:
            buckets["75-84"] += 1
        elif c >= 50:
            buckets["50-74"] += 1
        else:
            buckets["0-49"] += 1
    lines.append("| Bucket | Count | % |")
    lines.append("|--------|-------|---|")
    for bucket, count in buckets.items():
        pct = (count / total * 100) if total > 0 else 0
        lines.append(f"| {bucket} | {count} | {pct:.1f}% |")
    lines.append("")

    # Section 6: Grade Distribution
    lines.append("## 6. Grade Distribution\n")
    grade_counts: Counter = Counter()
    for q in all_queries:
        for fn in q.fact_nodes:
            grade_counts[fn.grade] += 1
    total_facts = sum(grade_counts.values())
    lines.append("| Grade | Count | % |")
    lines.append("|-------|-------|---|")
    for g in (0, 1, 2):
        count = grade_counts.get(g, 0)
        pct = (count / total_facts * 100) if total_facts > 0 else 0
        lines.append(f"| {g} | {count} | {pct:.1f}% |")
    lines.append("")

    # Section 7: Triage Tiers
    lines.append("## 7. Triage Tiers\n")
    lines.append("| Tier | Count | Review Sample |")
    lines.append("|------|-------|---------------|")
    for tier in ("high", "medium", "low"):
        count = triage_result.tier_counts.get(tier, 0)
        sample = len(triage_result.tier_details.get(tier, []))
        lines.append(f"| {tier} | {count} | {sample} |")
    lines.append("")

    # Section 8: Sample Queries
    lines.append("## 8. Sample Queries (up to 20)\n")
    for i, q in enumerate(all_queries[:20], 1):
        lines.append(f"### Query {i}: `{q.query_id}`\n")
        lines.append(f"- **Text:** {q.query_text}")
        lines.append(f"- **Direction:** {q.direction}")
        lines.append(f"- **Confidence:** {q.confidence.overall}")
        lines.append(f"- **Subgraph:** {q.subgraph_id[:12]}")
        lines.append(f"- **Fact nodes:** {len(q.fact_nodes)}")
        grades = Counter(fn.grade for fn in q.fact_nodes)
        lines.append(f"- **Grade distribution:** {dict(grades)}")
        lines.append(f"- **Relations used:** {q.relations_used}")
        lines.append(f"- **Reasoning:** {q.reasoning[:150]}...")
        lines.append("")

    # Section 9: Known Issues
    lines.append("## 9. Known Issues\n")
    lines.append("- **Enrichment gap:** 0 enriched nodes across subgraphs (cross-L5 Tier2 issue)")
    lines.append("- **Direction skew:** BU/NT/CS directions typically underrepresented")
    lines.append("- **All subgraphs 'large' tier:** uniform tier distribution expected")
    lines.append("")

    report_path = Path(output_dir) / "quality_report.md"
    report_path.write_text("\n".join(lines), encoding="utf-8")
    logger.info("Step 9: Quality report → %s", report_path)
    return report_path


def generate_direction_analysis(dataset, output_dir: str) -> Path:
    """Generate direction_analysis.md with distribution and rebalancing analysis."""
    from workflows.benchmark.balance import (
        DIRECTION_TARGETS,
        analyze_distribution,
        identify_rebalance_candidates,
    )

    lines = ["# Direction Distribution Analysis\n"]
    lines.append(f"Generated: {datetime.now(UTC).strftime('%Y-%m-%d %H:%M UTC')}\n")

    balance_report = analyze_distribution(dataset.queries)

    # Distribution table
    lines.append("## Distribution vs Target\n")
    lines.append("| Direction | Actual | Target | Delta | Status |")
    lines.append("|-----------|--------|--------|-------|--------|")
    for d in ("TD", "HZ", "BU", "NT", "CS"):
        actual = balance_report.distribution.get(d, 0)
        target = balance_report.targets.get(d, 0)
        delta = actual - target
        status = (
            "OK" if d not in balance_report.deficits else f"DEFICIT ({balance_report.deficits[d]})"
        )
        lines.append(f"| {d} | {actual} | {target} | {delta:+d} | {status} |")
    amb = balance_report.ambiguous_count
    lines.append(f"| AMBIGUOUS | {amb} | — | — | {balance_report.ambiguous_pct:.1f}% |")
    lines.append(f"\n**Overall balanced:** {'Yes' if balance_report.is_balanced else 'No'}\n")

    # Deficit analysis
    if balance_report.deficits:
        lines.append("## Deficit Analysis\n")
        for direction, deficit in balance_report.deficits.items():
            lines.append(f"### {direction} (need {deficit} more)\n")
            candidates = identify_rebalance_candidates(dataset.queries, direction)
            if candidates:
                lines.append(f"Candidate subgraphs for rebalancing: {len(candidates)}\n")
                for sg_id in candidates[:10]:
                    lines.append(f"- `{sg_id}`")
            else:
                lines.append("No candidate subgraphs found for rebalancing.")
            lines.append("")

    # Per-subgraph direction breakdown
    lines.append("## Per-Subgraph Direction Breakdown\n")
    sg_dirs: dict[str, Counter] = {}
    for q in dataset.queries:
        sg_dirs.setdefault(q.subgraph_id, Counter())[q.direction] += 1

    lines.append("| Subgraph | TD | HZ | BU | NT | CS | AMB | Total |")
    lines.append("|----------|----|----|----|----|----|----|-------|")
    for sg in dataset.subgraphs:
        counts = sg_dirs.get(sg.subgraph_id, Counter())
        total_q = sum(counts.values())
        lines.append(
            f"| {sg.anchor_title[:30]} "
            f"| {counts.get('TD', 0)} | {counts.get('HZ', 0)} "
            f"| {counts.get('BU', 0)} | {counts.get('NT', 0)} "
            f"| {counts.get('CS', 0)} | {counts.get('AMBIGUOUS', 0)} "
            f"| {total_q} |"
        )
    lines.append("")

    # Recommendations
    lines.append("## Rebalancing Recommendations\n")
    if balance_report.is_balanced:
        lines.append("Distribution is balanced. No action needed.\n")
    else:
        lines.append("Consider the following to improve balance:\n")
        for direction, deficit in balance_report.deficits.items():
            pct = DIRECTION_TARGETS[direction] * 100
            lines.append(
                f"- **{direction}** (target {pct:.0f}%): "
                f"Generate {deficit} more queries from identified candidate subgraphs"
            )
    lines.append("")

    report_path = Path(output_dir) / "direction_analysis.md"
    report_path.write_text("\n".join(lines), encoding="utf-8")
    logger.info("Step 10: Direction analysis → %s", report_path)
    return report_path
