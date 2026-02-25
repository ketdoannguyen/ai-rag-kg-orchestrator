"""Inter-Annotator Agreement Measurement for Benchmark v4.0 (Task 23).

Computes Cohen's kappa between two annotators on a shared sample.
Provides disagreement analysis and reporting.
"""

from __future__ import annotations

import logging
import random
from collections import Counter
from dataclasses import dataclass, field
from typing import Any

from workflows.benchmark.models import BenchmarkQuery

logger = logging.getLogger(__name__)

TARGET_KAPPA = 0.60
DEFAULT_SAMPLE_SIZE = 100


@dataclass
class IAAReport:
    """Inter-Annotator Agreement report."""

    sample_size: int
    kappa: float
    meets_threshold: bool
    observed_agreement: float
    expected_agreement: float
    disagreements: list[dict[str, Any]] = field(default_factory=list)
    label_distribution_a: dict[str, int] = field(default_factory=dict)
    label_distribution_b: dict[str, int] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return {
            "sample_size": self.sample_size,
            "kappa": round(self.kappa, 4),
            "meets_threshold": self.meets_threshold,
            "observed_agreement": round(self.observed_agreement, 4),
            "expected_agreement": round(self.expected_agreement, 4),
            "disagreement_count": len(self.disagreements),
            "label_distribution_a": self.label_distribution_a,
            "label_distribution_b": self.label_distribution_b,
        }


def cohens_kappa(
    labels_a: list[str],
    labels_b: list[str],
) -> tuple[float, float, float]:
    """Compute Cohen's kappa for two annotators.

    Args:
        labels_a: Labels from annotator A.
        labels_b: Labels from annotator B (same length as labels_a).

    Returns:
        (kappa, observed_agreement, expected_agreement)

    Raises:
        ValueError: If label lists have different lengths.
    """
    if len(labels_a) != len(labels_b):
        raise ValueError(f"Label lists must have same length: {len(labels_a)} vs {len(labels_b)}")

    n = len(labels_a)
    if n == 0:
        return 0.0, 0.0, 0.0

    # Observed agreement
    agree = sum(1 for a, b in zip(labels_a, labels_b) if a == b)
    p_o = agree / n

    # Expected agreement by chance
    all_labels = set(labels_a) | set(labels_b)
    count_a = Counter(labels_a)
    count_b = Counter(labels_b)

    p_e = sum((count_a.get(label, 0) / n) * (count_b.get(label, 0) / n) for label in all_labels)

    # Cohen's kappa
    kappa = 1.0 if p_e == 1.0 else (p_o - p_e) / (1.0 - p_e)

    return kappa, p_o, p_e


def select_iaa_sample(
    queries: list[BenchmarkQuery],
    sample_size: int = DEFAULT_SAMPLE_SIZE,
    seed: int = 42,
) -> list[BenchmarkQuery]:
    """Select a stratified random sample for IAA measurement.

    Stratifies by direction to ensure diverse representation.
    """
    rng = random.Random(seed)

    # Group by direction
    by_direction: dict[str, list[BenchmarkQuery]] = {}
    for q in queries:
        d = q.direction or "UNKNOWN"
        by_direction.setdefault(d, []).append(q)

    # Proportional allocation
    total = len(queries)
    if total == 0:
        return []

    sample: list[BenchmarkQuery] = []
    selected_ids: set[str] = set()
    remaining = sample_size

    directions = sorted(by_direction.keys())
    for direction in directions:
        group = by_direction[direction]
        n_select = max(1, int(sample_size * len(group) / total))
        n_select = min(n_select, remaining, len(group))

        selected = rng.sample(group, n_select)
        sample.extend(selected)
        selected_ids.update(q.query_id for q in selected)
        remaining -= len(selected)

    # Fill leftover slots proportionally across groups that still have capacity
    if remaining > 0:
        candidates = [q for q in queries if q.query_id not in selected_ids]
        extra = rng.sample(candidates, min(remaining, len(candidates)))
        sample.extend(extra)

    return sample[:sample_size]


def compute_iaa(
    query_ids: list[str],
    labels_a: dict[str, str],
    labels_b: dict[str, str],
) -> IAAReport:
    """Compute IAA for a set of double-annotated queries.

    Args:
        query_ids: IDs of queries in the IAA sample.
        labels_a: {query_id: decision} from annotator A.
        labels_b: {query_id: decision} from annotator B.

    Returns:
        IAAReport with kappa, agreement stats, and disagreement details.
    """
    # Align labels
    aligned_a: list[str] = []
    aligned_b: list[str] = []
    disagreements: list[dict[str, Any]] = []

    for qid in query_ids:
        la = labels_a.get(qid)
        lb = labels_b.get(qid)
        if la is None or lb is None:
            continue
        aligned_a.append(la)
        aligned_b.append(lb)
        if la != lb:
            disagreements.append(
                {
                    "query_id": qid,
                    "annotator_a": la,
                    "annotator_b": lb,
                }
            )

    kappa, p_o, p_e = cohens_kappa(aligned_a, aligned_b)

    return IAAReport(
        sample_size=len(aligned_a),
        kappa=kappa,
        meets_threshold=kappa >= TARGET_KAPPA,
        observed_agreement=p_o,
        expected_agreement=p_e,
        disagreements=disagreements,
        label_distribution_a=dict(Counter(aligned_a)),
        label_distribution_b=dict(Counter(aligned_b)),
    )
