"""Confidence-Based Triage for Benchmark v4.0 (Task 21).

Routes queries into review tiers based on confidence scores.
High-confidence queries get spot-checked; low-confidence get full review.
"""

from __future__ import annotations

import logging
import random
from dataclasses import dataclass, field
from typing import Any

from workflows.benchmark.models import BenchmarkQuery

logger = logging.getLogger(__name__)

# Triage thresholds (using raw confidence 0-100 scale)
HIGH_THRESHOLD = 85
MEDIUM_THRESHOLD = 75

# Sample rates per tier
SAMPLE_RATES: dict[str, float] = {
    "high": 0.10,  # 10% spot-check
    "medium": 0.50,  # 50% review
    "low": 1.00,  # 100% review
}


@dataclass
class TriageResult:
    """Result of triage assignment."""

    tier_counts: dict[str, int]
    review_queries: list[BenchmarkQuery]
    review_count: int
    tier_details: dict[str, list[str]] = field(default_factory=dict)  # tier → [query_ids]

    def to_dict(self) -> dict[str, Any]:
        return {
            "tier_counts": self.tier_counts,
            "review_count": self.review_count,
            "tier_details": dict(self.tier_details),
        }


def classify_tier(confidence: int) -> str:
    """Classify a query into a review tier based on raw confidence."""
    if confidence >= HIGH_THRESHOLD:
        return "high"
    if confidence >= MEDIUM_THRESHOLD:
        return "medium"
    return "low"


def triage_queries(
    queries: list[BenchmarkQuery],
    seed: int = 42,
) -> TriageResult:
    """Assign queries to review tiers and select review samples.

    Args:
        queries: Classified and QA-checked queries.
        seed: RNG seed for reproducible sampling.

    Returns:
        TriageResult with tier assignment and selected review queries.
    """
    rng = random.Random(seed)

    # Group by tier
    tiers: dict[str, list[BenchmarkQuery]] = {"high": [], "medium": [], "low": []}
    for q in queries:
        tier = classify_tier(q.confidence.overall)
        tiers[tier].append(q)

    tier_counts = {tier: len(qs) for tier, qs in tiers.items()}

    # Sample per tier
    review_queries: list[BenchmarkQuery] = []
    tier_details: dict[str, list[str]] = {}

    for tier, tier_queries in tiers.items():
        rate = SAMPLE_RATES[tier]
        sample_size = max(1, int(len(tier_queries) * rate)) if tier_queries else 0

        if sample_size >= len(tier_queries):
            selected = tier_queries
        else:
            selected = rng.sample(tier_queries, sample_size)

        review_queries.extend(selected)
        tier_details[tier] = [q.query_id for q in selected]

    logger.info(
        "Triage: high=%d (sample %d), medium=%d (sample %d), low=%d (sample %d), total review=%d",
        tier_counts["high"],
        len(tier_details.get("high", [])),
        tier_counts["medium"],
        len(tier_details.get("medium", [])),
        tier_counts["low"],
        len(tier_details.get("low", [])),
        len(review_queries),
    )

    return TriageResult(
        tier_counts=tier_counts,
        review_queries=review_queries,
        review_count=len(review_queries),
        tier_details=tier_details,
    )
