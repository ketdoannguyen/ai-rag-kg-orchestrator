"""Multi-signal relation scorer for query retrieval.

This module provides the RelationScorer class that computes final scores
as a weighted combination of:
- Embedding similarity (cosine)
- Family affinity (query_type → relation families)
- Keyword matching (exact substring match)

Uses vectorized NumPy for performance.
"""

from __future__ import annotations

import numpy as np
from loguru import logger

from config.ontologies.families import DEFAULT_AFFINITY, get_affinity

# =============================================================================
# Scoring Weights (must sum to 1.0)
# =============================================================================

EMBEDDING_WEIGHT = 0.4  # Weight for embedding similarity signal
AFFINITY_WEIGHT = 0.4  # Weight for family affinity signal
KEYWORD_WEIGHT = 0.2  # Weight for keyword matching signal

# =============================================================================
# Query-type-based K defaults
# =============================================================================

QUERY_TYPE_BASE_K: dict[str, int] = {
    "factual": 5,
    "causal": 8,
    "comparison": 10,
    "figure": 8,
    "concept": 7,
    "lesson_synthesis": 12,
    "structural": 3,
    "other": 5,
}

# Adaptive K bounds
MIN_K = 3
MAX_K = 15
GAP_THRESHOLD = 0.1


def compute_adaptive_k(
    scores: list[float],
    query_type: str,
    gap_threshold: float = GAP_THRESHOLD,
    min_k: int = MIN_K,
    max_k: int = MAX_K,
) -> int:
    """Compute adaptive K based on query type and score distribution.

    This function determines the optimal number of relations to return
    by combining:
    1. Query-type-based default K
    2. Gap-based cutoff (find natural breaks in score distribution)

    Args:
        scores: List of scores sorted descending
        query_type: The query type (causal, factual, etc.)
        gap_threshold: Minimum gap to trigger cutoff (default: 0.1)
        min_k: Minimum K to return (default: 3)
        max_k: Maximum K to return (default: 15)

    Returns:
        Optimal K value
    """
    if not scores:
        return min_k

    # Step 1: Get base K from query type
    base_k = QUERY_TYPE_BASE_K.get(query_type.lower(), 7)

    # Step 2: Find gap-based cutoff
    gap_k = len(scores)  # Default: no cutoff

    # Start from index 2 (need at least 3 items)
    # Stop at min(len-1, max_k) to avoid out of bounds
    search_end = min(len(scores) - 1, max_k)

    for i in range(2, search_end):
        gap = scores[i] - scores[i + 1]
        if gap > gap_threshold:
            gap_k = i + 1  # Include index i, so k = i + 1
            break

    # Step 3: Take minimum of base_k and gap_k
    final_k = min(base_k, gap_k)

    # Step 4: Clamp to bounds
    final_k = max(min_k, min(max_k, final_k))

    logger.debug(
        "Adaptive K computed",
        base_k=base_k,
        gap_cutoff=gap_k,
        final_k=final_k,
    )

    return final_k


class RelationScorer:
    """Multi-signal relation scorer.

    Computes final score as weighted combination of:
    - Embedding similarity (cosine)
    - Family affinity (query_type → relation families)
    - Keyword matching (exact substring match)

    Uses vectorized NumPy for performance.

    Attributes:
        embedding_weight: Weight for embedding similarity signal (default: 0.4)
        affinity_weight: Weight for family affinity signal (default: 0.4)
        keyword_weight: Weight for keyword matching signal (default: 0.2)
    """

    def __init__(
        self,
        embedding_weight: float = EMBEDDING_WEIGHT,
        affinity_weight: float = AFFINITY_WEIGHT,
        keyword_weight: float = KEYWORD_WEIGHT,
    ):
        """Initialize scorer with weights.

        Args:
            embedding_weight: Embedding signal weight
            affinity_weight: Affinity signal weight
            keyword_weight: Keyword signal weight

        Raises:
            ValueError: If weights don't sum to 1.0
        """
        total = embedding_weight + affinity_weight + keyword_weight
        if not (0.99 <= total <= 1.01):
            raise ValueError(f"Weights must sum to 1.0, got {total}")
        self.embedding_weight = embedding_weight
        self.affinity_weight = affinity_weight
        self.keyword_weight = keyword_weight

    def score_all_vectorized(
        self,
        query: str,
        query_embedding: list[float],
        query_type: str,
        relations: list,
    ) -> list[tuple[str, float, float, float, float]]:
        """Score all relations using vectorized operations.

        Returns list of (relation_id, final_score, emb_score, aff_score, kw_score)
        sorted by final_score descending.

        Args:
            query: User query text
            query_embedding: Query embedding vector
            query_type: Primary query type (causal, factual, etc.)
            relations: List of OntologyRelation objects

        Returns:
            Sorted list of (id, final, emb, aff, kw) tuples
        """
        if not relations:
            return []

        n = len(relations)

        # SIGNAL 1: EMBEDDING SCORES (Fully Vectorized)
        embedding_scores = self._compute_embedding_scores(query_embedding, relations, n)

        # SIGNAL 2: AFFINITY SCORES (Fast dictionary lookup)
        affinity_scores = self._compute_affinity_scores(query_type, relations, n)

        # SIGNAL 3: KEYWORD SCORES (String matching)
        keyword_scores = self._compute_keyword_scores(query, relations, n)

        # FINAL SCORES (Vectorized)
        final_scores = (
            self.embedding_weight * embedding_scores
            + self.affinity_weight * affinity_scores
            + self.keyword_weight * keyword_scores
        )

        # Build result tuples
        results = [
            (
                getattr(relations[i], "id", f"unknown_{i}"),
                float(final_scores[i]),
                float(embedding_scores[i]),
                float(affinity_scores[i]),
                float(keyword_scores[i]),
            )
            for i in range(n)
        ]

        # Sort by final score descending
        results.sort(key=lambda x: x[1], reverse=True)

        return results

    def _compute_embedding_scores(
        self,
        query_embedding: list[float],
        relations: list,
        n: int,
    ) -> np.ndarray:
        """Compute embedding similarity scores using vectorized operations."""
        embedding_scores = np.zeros(n)

        # Collect embeddings and track indices
        embeddings_list = []
        valid_indices = []

        for i, rel in enumerate(relations):
            emb = getattr(rel, "embedding", None)
            if emb is not None and len(emb) > 0:
                embeddings_list.append(emb)
                valid_indices.append(i)

        if embeddings_list:
            query_vec = np.array(query_embedding, dtype=np.float32)
            query_norm = np.linalg.norm(query_vec)

            if query_norm > 0:
                # Build matrix and compute all at once
                rel_matrix = np.array(embeddings_list, dtype=np.float32)
                rel_norms = np.linalg.norm(rel_matrix, axis=1)

                # Mask for valid norms
                valid_norm_mask = rel_norms > 0

                if valid_norm_mask.any():
                    # Compute dot products for valid entries
                    dot_products = np.dot(rel_matrix, query_vec)

                    # Compute similarities only for valid norms
                    similarities = np.zeros(len(embeddings_list))
                    similarities[valid_norm_mask] = dot_products[valid_norm_mask] / (
                        rel_norms[valid_norm_mask] * query_norm
                    )

                    # Clamp to [0, 1]
                    similarities = np.clip(similarities, 0, 1)

                    # Assign back to original indices
                    for idx, sim in zip(valid_indices, similarities):
                        embedding_scores[idx] = sim

        return embedding_scores

    def _compute_affinity_scores(
        self,
        query_type: str,
        relations: list,
        n: int,
    ) -> np.ndarray:
        """Compute family affinity scores using dictionary lookup."""
        affinity_scores = np.zeros(n)
        query_lower = query_type.lower() if query_type else ""

        for i, rel in enumerate(relations):
            families = getattr(rel, "families", None) or []
            if families:
                # Get max affinity across all families
                max_aff = max(get_affinity(query_lower, f) for f in families)
                affinity_scores[i] = max_aff
            else:
                affinity_scores[i] = DEFAULT_AFFINITY

        return affinity_scores

    def _compute_keyword_scores(
        self,
        query: str,
        relations: list,
        n: int,
    ) -> np.ndarray:
        """Compute keyword matching scores."""
        keyword_scores = np.zeros(n)
        query_text_lower = query.lower() if query else ""

        for i, rel in enumerate(relations):
            keywords = getattr(rel, "keywords", None) or []
            # Filter empty keywords
            valid_keywords = [kw for kw in keywords if kw and kw.strip()]
            if valid_keywords:
                matched = sum(1 for kw in valid_keywords if kw.lower() in query_text_lower)
                keyword_scores[i] = matched / len(valid_keywords)

        return keyword_scores
