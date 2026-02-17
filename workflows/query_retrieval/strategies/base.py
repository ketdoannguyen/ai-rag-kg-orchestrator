"""Base utilities for retrieval strategies.

Shared code between E1 (TopDownRetrieval) and E2 (HorizontalRetrieval) strategies:
- Vectorized cosine similarity computation
- ContentWithContext conversion
- Async executor for database operations
"""

from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor

import numpy as np

from workflows.query_retrieval.models import ContentWithContext

# Shared executor for async database operations
_EXECUTOR = ThreadPoolExecutor(max_workers=4)


def get_executor() -> ThreadPoolExecutor:
    """Get the shared ThreadPoolExecutor for async DB operations."""
    return _EXECUTOR


def cosine_similarity_batch(
    query_embedding: list[float],
    embeddings: list[list[float]],
) -> np.ndarray:
    """Compute cosine similarity between query and a batch of embeddings.

    Args:
        query_embedding: Query embedding vector
        embeddings: List of embedding vectors to compare against

    Returns:
        1-D numpy array of cosine scores clipped to [0, 1]
    """
    query_vec = np.array(query_embedding, dtype=np.float32)
    query_norm = np.linalg.norm(query_vec)

    if query_norm == 0 or not embeddings:
        return np.zeros(len(embeddings), dtype=np.float32)

    emb_matrix = np.array(embeddings, dtype=np.float32)
    row_norms = np.linalg.norm(emb_matrix, axis=1)

    scores = np.zeros(len(embeddings), dtype=np.float32)
    valid_mask = row_norms > 0

    if valid_mask.any():
        dot_products = emb_matrix @ query_vec
        scores[valid_mask] = dot_products[valid_mask] / (row_norms[valid_mask] * query_norm)
        scores = np.clip(scores, 0, 1)

    return scores


def compute_ranks_descending(scores: np.ndarray) -> np.ndarray:
    """Compute 1-indexed ranks (higher score = lower rank number).

    Args:
        scores: Array of scores (higher = better)

    Returns:
        Array of ranks (1 = best)
    """
    n = len(scores)
    if n == 0:
        return np.array([], dtype=np.float32)

    order = np.argsort(-scores)  # Descending
    ranks = np.zeros(n, dtype=np.float32)
    for rank, idx in enumerate(order, start=1):
        ranks[idx] = rank
    return ranks


def compute_ranks_with_ties(scores: np.ndarray) -> np.ndarray:
    """Compute ranks with average rank for ties.

    When items have same score, they get average of occupied ranks.
    Example: [0.8, 0.6, 0.8] -> [1.5, 3.0, 1.5]

    Args:
        scores: Array of scores (higher = better)

    Returns:
        Array of ranks (1-indexed, lower = better)
    """
    n = len(scores)
    if n == 0:
        return np.array([], dtype=np.float32)

    ranks = np.zeros(n, dtype=np.float32)
    order = np.argsort(-scores)
    sorted_scores = scores[order]

    i = 0
    while i < n:
        j = i
        while j < n and sorted_scores[j] == sorted_scores[i]:
            j += 1
        avg_rank = (i + 1 + j) / 2.0
        for idx in range(i, j):
            ranks[order[idx]] = avg_rank
        i = j

    return ranks


def rrf_fusion_2way(
    scores_a: np.ndarray,
    scores_b: np.ndarray,
    k: int = 60,
    use_tie_handling: bool = False,
) -> np.ndarray:
    """Apply 2-way Reciprocal Rank Fusion.

    Formula: RRF(d) = 1/(k + rank_a(d)) + 1/(k + rank_b(d))

    Args:
        scores_a: First signal scores
        scores_b: Second signal scores
        k: RRF smoothing constant (default: 60)
        use_tie_handling: Use average ranks for ties

    Returns:
        RRF scores (higher = better combined rank)
    """
    n = len(scores_a)
    if n == 0:
        return np.array([], dtype=np.float32)
    if n == 1:
        return np.array([2.0 / (k + 1)], dtype=np.float32)

    rank_fn = compute_ranks_with_ties if use_tie_handling else compute_ranks_descending
    ranks_a = rank_fn(scores_a)
    ranks_b = rank_fn(scores_b)

    return (1.0 / (k + ranks_a) + 1.0 / (k + ranks_b)).astype(np.float32)


def rrf_fusion_3way(
    scores_a: np.ndarray,
    scores_b: np.ndarray,
    scores_c: np.ndarray,
    k: int = 60,
    use_tie_handling: bool = False,
) -> np.ndarray:
    """Apply 3-way Reciprocal Rank Fusion.

    Formula: RRF(d) = 1/(k + rank_a) + 1/(k + rank_b) + 1/(k + rank_c)

    Args:
        scores_a, scores_b, scores_c: Three signal scores
        k: RRF smoothing constant
        use_tie_handling: Use average ranks for ties

    Returns:
        RRF scores (higher = better combined rank)
    """
    n = len(scores_a)
    if n == 0:
        return np.array([], dtype=np.float32)
    if n == 1:
        return np.array([3.0 / (k + 1)], dtype=np.float32)

    rank_fn = compute_ranks_with_ties if use_tie_handling else compute_ranks_descending
    ranks_a = rank_fn(scores_a)
    ranks_b = rank_fn(scores_b)
    ranks_c = rank_fn(scores_c)

    return (1.0 / (k + ranks_a) + 1.0 / (k + ranks_b) + 1.0 / (k + ranks_c)).astype(np.float32)


def to_content_with_context(
    node_data: dict,
    score: float,
    found_via_relation: str | None = None,
) -> ContentWithContext:
    """Convert a node dict to ContentWithContext.

    Args:
        node_data: Node document from ArangoDB
        score: Final relevance score
        found_via_relation: Relation type used to find this node (optional)

    Returns:
        ContentWithContext instance
    """
    return ContentWithContext(
        node_id=node_data.get("_key", node_data.get("_id", "")),
        text=node_data.get("text", ""),
        score=score,
        node_type=node_data.get("node_type", "paragraph"),
        found_via_relation=found_via_relation,
    )
