"""Statistical Tests for Benchmark v4.0 (Task 26).

Implements Friedman → pairwise Wilcoxon → BH FDR correction → Cliff's delta.
Per-direction sub-analysis and hypothesis testing (H1-H8).
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from itertools import combinations
from typing import Any

from workflows.benchmark.run_evaluation import MethodRunResult

logger = logging.getLogger(__name__)

# Cliff's delta thresholds (Romano et al., 2006)
CLIFF_NEGLIGIBLE = 0.147
CLIFF_SMALL = 0.330
CLIFF_MEDIUM = 0.474


@dataclass
class PairwiseResult:
    """Result of a single pairwise comparison."""

    method_a: str
    method_b: str
    p_value: float
    p_adjusted: float
    significant: bool
    cliff_delta: float
    effect_size: str  # negligible, small, medium, large


@dataclass
class HypothesisResult:
    """Result of testing one hypothesis."""

    hypothesis_id: str
    description: str
    supported: bool
    evidence: str
    p_value: float | None = None
    effect_size: float | None = None


@dataclass
class StatisticsReport:
    """Complete statistical analysis report."""

    friedman_statistic: float
    friedman_p_value: float
    friedman_significant: bool
    pairwise_results: list[PairwiseResult] = field(default_factory=list)
    hypothesis_results: list[HypothesisResult] = field(default_factory=list)
    per_direction: dict[str, dict[str, Any]] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return {
            "friedman": {
                "statistic": round(self.friedman_statistic, 4),
                "p_value": self.friedman_p_value,
                "significant": self.friedman_significant,
            },
            "pairwise_count": len(self.pairwise_results),
            "significant_pairs": sum(1 for p in self.pairwise_results if p.significant),
            "hypotheses": [
                {
                    "id": h.hypothesis_id,
                    "description": h.description,
                    "supported": h.supported,
                    "evidence": h.evidence,
                }
                for h in self.hypothesis_results
            ],
            "per_direction": self.per_direction,
        }


# ------------------------------------------------------------------ #
# Core Statistical Functions                                          #
# ------------------------------------------------------------------ #


def cliffs_delta(x: list[float], y: list[float]) -> float:
    """Compute Cliff's delta effect size.

    δ = (#{x_i > y_j} - #{x_i < y_j}) / (n_x * n_y)

    Returns value in [-1, 1].
    """
    n_x = len(x)
    n_y = len(y)
    if n_x == 0 or n_y == 0:
        return 0.0

    more = 0
    less = 0
    for xi in x:
        for yj in y:
            if xi > yj:
                more += 1
            elif xi < yj:
                less += 1

    return (more - less) / (n_x * n_y)


def classify_effect_size(delta: float) -> str:
    """Classify Cliff's delta into effect size category."""
    abs_delta = abs(delta)
    if abs_delta < CLIFF_NEGLIGIBLE:
        return "negligible"
    if abs_delta < CLIFF_SMALL:
        return "small"
    if abs_delta < CLIFF_MEDIUM:
        return "medium"
    return "large"


def extract_metric_vectors(
    method_results: list[MethodRunResult],
    metric: str = "ndcg_at_10",
) -> dict[str, list[float]]:
    """Extract per-query metric values for each method, aligned by query ID.

    Only includes queries present in ALL methods so that vectors are
    positionally aligned for paired statistical tests (Friedman, Wilcoxon).

    Args:
        method_results: Results from all methods.
        metric: Which metric to extract (ndcg_at_10, recall_at_10, etc.)

    Returns:
        {method_id: [metric_value_per_query]} — aligned by shared query IDs.
    """
    if not method_results:
        return {}

    # Build per-method lookup: {method_id: {query_id: metric_value}}
    # Skip methods with zero evaluations so they don't zero out the intersection.
    method_query_map: dict[str, dict[str, float]] = {}
    for mr in method_results:
        qmap = {e.query_id: getattr(e, metric) for e in mr.evaluations}
        if qmap:
            method_query_map[mr.method_id] = qmap

    if not method_query_map:
        return {mr.method_id: [] for mr in method_results}

    # Find query IDs present in ALL non-empty methods
    query_sets = [set(qmap.keys()) for qmap in method_query_map.values()]
    common_ids = sorted(set.intersection(*query_sets)) if query_sets else []

    if not common_ids:
        return {mr.method_id: [] for mr in method_results}

    # Build aligned vectors using common query ID order
    vectors: dict[str, list[float]] = {}
    for mid, qmap in method_query_map.items():
        vectors[mid] = [qmap[qid] for qid in common_ids]

    return vectors


def run_friedman_test(
    method_results: list[MethodRunResult],
    metric: str = "ndcg_at_10",
) -> tuple[float, float]:
    """Run Friedman test across all methods.

    Requires scipy. Returns (statistic, p_value).
    Falls back to (0.0, 1.0) if scipy unavailable or insufficient data.
    """
    try:
        from scipy.stats import friedmanchisquare
    except ImportError:
        logger.warning("scipy not available, skipping Friedman test")
        return 0.0, 1.0

    vectors = extract_metric_vectors(method_results, metric)
    method_ids = sorted(vectors.keys())

    if len(method_ids) < 3:
        logger.warning("Need ≥3 methods for Friedman test, got %d", len(method_ids))
        return 0.0, 1.0

    # Vectors are already aligned by extract_metric_vectors
    vec_len = len(vectors[method_ids[0]]) if method_ids else 0
    if vec_len < 2:
        return 0.0, 1.0

    aligned = [vectors[m] for m in method_ids]
    stat, p = friedmanchisquare(*aligned)
    return float(stat), float(p)


def run_pairwise_wilcoxon(
    method_results: list[MethodRunResult],
    metric: str = "ndcg_at_10",
    alpha: float = 0.05,
) -> list[PairwiseResult]:
    """Run pairwise Wilcoxon signed-rank tests with BH correction.

    Returns list of PairwiseResult for all C(n,2) pairs.
    """
    try:
        from scipy.stats import wilcoxon
        from statsmodels.stats.multitest import multipletests
    except ImportError:
        logger.warning("scipy/statsmodels not available, skipping Wilcoxon tests")
        return []

    vectors = extract_metric_vectors(method_results, metric)
    method_ids = sorted(vectors.keys())
    pairs = list(combinations(method_ids, 2))

    if not pairs:
        return []

    # Compute raw p-values
    raw_p_values: list[float] = []
    pair_data: list[tuple[str, str, list[float], list[float]]] = []

    # Vectors are already aligned by extract_metric_vectors
    vec_len = len(vectors[method_ids[0]]) if method_ids else 0
    if vec_len < 5:
        logger.warning("Need ≥5 observations for Wilcoxon, got %d", vec_len)
        return []

    for ma, mb in pairs:
        va = vectors[ma]
        vb = vectors[mb]

        # Check if all differences are zero
        diffs = [a - b for a, b in zip(va, vb)]
        if all(d == 0.0 for d in diffs):
            raw_p_values.append(1.0)
        else:
            try:
                _, p = wilcoxon(va, vb)
                raw_p_values.append(float(p))
            except ValueError:
                raw_p_values.append(1.0)

        pair_data.append((ma, mb, va, vb))

    # BH FDR correction
    _, p_adjusted, _, _ = multipletests(raw_p_values, alpha=alpha, method="fdr_bh")

    # Build results with Cliff's delta
    results: list[PairwiseResult] = []
    for i, (ma, mb, va, vb) in enumerate(pair_data):
        delta = cliffs_delta(va, vb)
        results.append(
            PairwiseResult(
                method_a=ma,
                method_b=mb,
                p_value=raw_p_values[i],
                p_adjusted=float(p_adjusted[i]),
                significant=float(p_adjusted[i]) < alpha,
                cliff_delta=delta,
                effect_size=classify_effect_size(delta),
            )
        )

    return results


# ------------------------------------------------------------------ #
# Hypothesis Testing                                                  #
# ------------------------------------------------------------------ #

# H1-H8 from the design doc §10.5
HYPOTHESES = [
    (
        "H1",
        "Graph-enhanced methods outperform baselines",
        ["E1a", "E2a", "E3a"],
        ["B1", "B2", "B3"],
    ),
    ("H2", "RRF fusion improves over single-signal scoring", ["E1b", "E2b"], ["E1a", "E2a"]),
    (
        "H3",
        "Horizontal (E2) outperforms Top-Down (E1) on HZ queries",
        ["E2a", "E2b"],
        ["E1a", "E1b"],
    ),
    ("H4", "Top-Down (E1) outperforms others on TD queries", ["E1a", "E1b"], ["E2a", "B1"]),
    ("H5", "Bottom-Up (E3) competitive on BU queries", ["E3a", "E3b"], ["E1a", "E2a"]),
    ("H6", "Multi-signal RRF (E2d) is best overall E2 variant", ["E2d"], ["E2a", "E2b", "E2c"]),
    (
        "H7",
        "Content-aware E3 scoring improves over parent-level",
        ["E3d", "E3e"],
        ["E3a", "E3b", "E3c"],
    ),
    ("H8", "Baselines perform adequately on NT (neutral) queries", ["B1"], ["E1a", "E2a"]),
]


def test_hypotheses(
    pairwise_results: list[PairwiseResult],
    method_results: list[MethodRunResult],
    metric: str = "ndcg_at_10",
) -> list[HypothesisResult]:
    """Test all 8 hypotheses using pairwise comparison results.

    Each hypothesis compares a set of 'better' methods against 'baseline' methods.
    """
    # Build lookup for pairwise results
    pw_lookup: dict[tuple[str, str], PairwiseResult] = {}
    for pr in pairwise_results:
        pw_lookup[(pr.method_a, pr.method_b)] = pr
        pw_lookup[(pr.method_b, pr.method_a)] = pr

    # Build lookup for method averages
    avg_lookup: dict[str, float] = {}
    for mr in method_results:
        if metric == "ndcg_at_10":
            avg_lookup[mr.method_id] = mr.avg_ndcg
        elif metric == "recall_at_10":
            avg_lookup[mr.method_id] = mr.avg_recall
        elif metric == "precision_at_10":
            avg_lookup[mr.method_id] = mr.avg_precision
        else:
            avg_lookup[mr.method_id] = mr.avg_mrr

    results: list[HypothesisResult] = []

    for hid, desc, better_methods, base_methods in HYPOTHESES:
        # Check if better methods have higher average than base
        better_avg = _safe_mean([avg_lookup.get(m, 0.0) for m in better_methods])
        base_avg = _safe_mean([avg_lookup.get(m, 0.0) for m in base_methods])

        # Check significance in any relevant pair (with directional check)
        sig_pairs = []
        for bm in better_methods:
            for basem in base_methods:
                pair = pw_lookup.get((bm, basem))
                if pair and pair.significant:
                    # cliff_delta is computed as cliffs_delta(method_a, method_b),
                    # positive means method_a > method_b.
                    bm_wins = pair.cliff_delta > 0 if pair.method_a == bm else pair.cliff_delta < 0
                    if bm_wins:
                        sig_pairs.append(f"{bm}>{basem}")

        supported = better_avg > base_avg and len(sig_pairs) > 0
        evidence = (
            f"avg({','.join(better_methods)})={better_avg:.4f} vs "
            f"avg({','.join(base_methods)})={base_avg:.4f}; "
            f"significant pairs: {sig_pairs or 'none'}"
        )

        results.append(
            HypothesisResult(
                hypothesis_id=hid,
                description=desc,
                supported=supported,
                evidence=evidence,
            )
        )

    return results


# ------------------------------------------------------------------ #
# Per-Direction Analysis                                              #
# ------------------------------------------------------------------ #


def analyze_per_direction(
    method_results: list[MethodRunResult],
    queries: list[Any],
    metric: str = "ndcg_at_10",
) -> dict[str, dict[str, float]]:
    """Compute per-direction average metrics for each method.

    Args:
        method_results: Results from all methods.
        queries: Full query set (used for direction lookup).
        metric: Which metric to analyze.

    Returns:
        {direction: {method_id: avg_metric}}
    """
    # Build query direction lookup
    direction_map = {q.query_id: q.direction for q in queries if q.direction}

    per_direction: dict[str, dict[str, list[float]]] = {}

    for mr in method_results:
        for ev in mr.evaluations:
            direction = direction_map.get(ev.query_id)
            if not direction:
                continue
            if direction not in per_direction:
                per_direction[direction] = {}
            if mr.method_id not in per_direction[direction]:
                per_direction[direction][mr.method_id] = []
            per_direction[direction][mr.method_id].append(getattr(ev, metric))

    # Compute averages
    result: dict[str, dict[str, float]] = {}
    for direction, methods in sorted(per_direction.items()):
        result[direction] = {mid: _safe_mean(values) for mid, values in sorted(methods.items())}

    return result


# ------------------------------------------------------------------ #
# Helpers                                                             #
# ------------------------------------------------------------------ #


def _safe_mean(values: list[float]) -> float:
    """Compute mean, returning 0.0 for empty lists."""
    if not values:
        return 0.0
    return sum(values) / len(values)
