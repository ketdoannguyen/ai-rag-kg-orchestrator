"""E3 Ablation Study for Benchmark v4.0 (Task 28).

Compares E3 scoring strategies: parent-level (E3a-c) vs content-aware (E3d-e).
Tests hypothesis H7 and quantifies the dilution effect.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Any

from workflows.benchmark.run_evaluation import MethodRunResult
from workflows.benchmark.statistics import classify_effect_size, cliffs_delta

logger = logging.getLogger(__name__)

# H7 minimum effect-size threshold (must exceed CLIFF_NEGLIGIBLE = 0.147)
H7_MIN_EFFECT = 0.15

E3_PARENT_LEVEL = ["E3a", "E3b", "E3c"]
E3_CONTENT_AWARE = ["E3d", "E3e"]
ALL_E3 = E3_PARENT_LEVEL + E3_CONTENT_AWARE


@dataclass
class AblationEntry:
    """Per-method ablation metrics."""

    method_id: str
    method_name: str
    avg_ndcg: float
    avg_recall: float
    avg_precision: float
    avg_mrr: float
    avg_result_count: float
    query_count: int


@dataclass
class AblationReport:
    """Complete E3 ablation study report."""

    entries: list[AblationEntry]
    h7_result: dict[str, Any]
    precision_recall_tradeoff: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return {
            "e3_comparison": [
                {
                    "method_id": e.method_id,
                    "method_name": e.method_name,
                    "ndcg_at_10": round(e.avg_ndcg, 4),
                    "recall_at_10": round(e.avg_recall, 4),
                    "precision_at_10": round(e.avg_precision, 4),
                    "mrr": round(e.avg_mrr, 4),
                    "avg_result_count": round(e.avg_result_count, 1),
                }
                for e in self.entries
            ],
            "h7": self.h7_result,
            "precision_recall_tradeoff": self.precision_recall_tradeoff,
        }


def build_ablation_entries(
    method_results: list[MethodRunResult],
) -> list[AblationEntry]:
    """Extract E3 method results into ablation entries."""
    entries = []
    for mr in method_results:
        if mr.method_id not in ALL_E3:
            continue

        avg_results = 0.0
        if mr.evaluations:
            avg_results = sum(e.result_count for e in mr.evaluations) / len(mr.evaluations)

        entries.append(
            AblationEntry(
                method_id=mr.method_id,
                method_name=mr.method_name,
                avg_ndcg=mr.avg_ndcg,
                avg_recall=mr.avg_recall,
                avg_precision=mr.avg_precision,
                avg_mrr=mr.avg_mrr,
                avg_result_count=avg_results,
                query_count=mr.query_count,
            )
        )

    return sorted(entries, key=lambda e: e.method_id)


def test_h7(
    method_results: list[MethodRunResult],
) -> dict[str, Any]:
    """Test H7: Content-aware E3 scoring improves over parent-level.

    Compares E3d (FILTERED_AVG_LOG) vs E3b (AVG_LOG) using Cliff's delta.
    """
    e3d = None
    e3b = None
    for mr in method_results:
        if mr.method_id == "E3d":
            e3d = mr
        elif mr.method_id == "E3b":
            e3b = mr

    if e3d is None or e3b is None:
        return {
            "supported": False,
            "evidence": "E3d or E3b not found in results",
            "cliff_delta": 0.0,
            "effect_size": "N/A",
        }

    # Build query-aligned metric vectors
    ndcg_d_map = {e.query_id: e.ndcg_at_10 for e in e3d.evaluations}
    ndcg_b_map = {e.query_id: e.ndcg_at_10 for e in e3b.evaluations}
    common_ids = sorted(set(ndcg_d_map) & set(ndcg_b_map))

    if not common_ids:
        return {
            "supported": False,
            "evidence": "No overlapping queries between E3d and E3b",
            "cliff_delta": 0.0,
            "effect_size": "N/A",
        }

    ndcg_d = [ndcg_d_map[qid] for qid in common_ids]
    ndcg_b = [ndcg_b_map[qid] for qid in common_ids]

    delta = cliffs_delta(ndcg_d, ndcg_b)
    effect = classify_effect_size(delta)

    supported = delta > 0 and abs(delta) >= H7_MIN_EFFECT

    return {
        "supported": supported,
        "evidence": (
            f"E3d avg_ndcg={e3d.avg_ndcg:.4f} vs E3b avg_ndcg={e3b.avg_ndcg:.4f}, "
            f"Cliff's δ={delta:.4f} ({effect})"
        ),
        "cliff_delta": delta,
        "effect_size": effect,
    }


def analyze_precision_recall_tradeoff(
    method_results: list[MethodRunResult],
) -> dict[str, Any]:
    """Analyze precision vs recall tradeoff for filtered vs unfiltered E3."""
    filtered = {}
    unfiltered = {}

    for mr in method_results:
        data = {
            "precision": mr.avg_precision,
            "recall": mr.avg_recall,
            "ndcg": mr.avg_ndcg,
        }
        if mr.method_id in E3_CONTENT_AWARE:
            filtered[mr.method_id] = data
        elif mr.method_id in E3_PARENT_LEVEL:
            unfiltered[mr.method_id] = data

    return {
        "content_aware": filtered,
        "parent_level": unfiltered,
        "interpretation": (
            "Content-aware methods (E3d-e) are expected to have higher precision "
            "but potentially lower recall due to filtering. Parent-level methods "
            "(E3a-c) may have higher recall but diluted precision."
        ),
    }


def run_ablation_study(
    method_results: list[MethodRunResult],
) -> AblationReport:
    """Run the complete E3 ablation study.

    Pipeline:
        1. Build E3 comparison table
        2. Test H7 (content-aware vs parent-level)
        3. Analyze precision-recall tradeoff
    """
    entries = build_ablation_entries(method_results)
    h7 = test_h7(method_results)
    tradeoff = analyze_precision_recall_tradeoff(method_results)

    report = AblationReport(
        entries=entries,
        h7_result=h7,
        precision_recall_tradeoff=tradeoff,
    )

    logger.info(
        "Ablation: %d E3 methods compared, H7 %s",
        len(entries),
        "supported" if h7["supported"] else "not supported",
    )

    return report
