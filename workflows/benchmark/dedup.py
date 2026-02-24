"""Query Deduplication Pipeline for Benchmark v4.0.

Removes near-duplicate queries using cosine similarity on query embeddings.
When two queries exceed the threshold, the one with lower confidence is removed.
Tie-breaking: fewer fact_nodes → remove; then higher index → remove.
"""

from __future__ import annotations

import math

from workflows.benchmark.models import BenchmarkQuery


def _cosine_similarity(a: list[float], b: list[float]) -> float:
    """Compute cosine similarity between two embedding vectors."""
    dot = sum(x * y for x, y in zip(a, b, strict=True))
    norm_a = math.sqrt(sum(x * x for x in a))
    norm_b = math.sqrt(sum(x * x for x in b))
    if norm_a == 0.0 or norm_b == 0.0:
        return 0.0
    return dot / (norm_a * norm_b)


def _should_remove_first(q1: BenchmarkQuery, q2: BenchmarkQuery) -> bool:
    """Return True if q1 should be removed (q2 wins the tie-break).

    Priority (remove the weaker query):
      1. Lower confidence.overall → remove
      2. Fewer fact_nodes → remove
      3. Higher index in original list → remove (caller passes index via query_id order)
    """
    conf1 = q1.confidence.overall
    conf2 = q2.confidence.overall
    if conf1 != conf2:
        return conf1 < conf2
    # Tie-break: fewer fact_nodes → remove
    fn1 = len(q1.fact_nodes)
    fn2 = len(q2.fact_nodes)
    if fn1 != fn2:
        return fn1 < fn2
    # Final tie-break: q2 wins (lower original index assumed for q1 by caller)
    return False


def dedup_queries(
    queries: list[BenchmarkQuery],
    embeddings: list[list[float]],
    threshold: float = 0.92,
) -> tuple[list[BenchmarkQuery], list[tuple[str, str, float]]]:
    """Remove near-duplicate queries using pairwise cosine similarity.

    Args:
        queries: List of BenchmarkQuery objects (ordered by original index).
        embeddings: Corresponding list of embedding vectors (same length as queries).
        threshold: Cosine similarity threshold above which queries are considered
            duplicates. Default 0.92.

    Returns:
        (unique_queries, removed_pairs) where removed_pairs is a list of
        (kept_id, removed_id, similarity) tuples for audit purposes.
    """
    if len(queries) != len(embeddings):
        raise ValueError(
            f"queries ({len(queries)}) and embeddings ({len(embeddings)}) must have equal length"
        )

    removed_indices: set[int] = set()
    removed_pairs: list[tuple[str, str, float]] = []

    n = len(queries)
    for i in range(n):
        if i in removed_indices:
            continue
        for j in range(i + 1, n):
            if j in removed_indices:
                continue
            sim = _cosine_similarity(embeddings[i], embeddings[j])
            if sim > threshold:
                # Determine which to remove
                # i < j so q_i has lower original index (q_i wins final tie-break)
                if _should_remove_first(queries[i], queries[j]):
                    removed_indices.add(i)
                    removed_pairs.append((queries[j].query_id, queries[i].query_id, sim))
                    break  # i removed, stop comparing i against others
                else:
                    removed_indices.add(j)
                    removed_pairs.append((queries[i].query_id, queries[j].query_id, sim))

    unique_queries = [q for idx, q in enumerate(queries) if idx not in removed_indices]
    return unique_queries, removed_pairs
