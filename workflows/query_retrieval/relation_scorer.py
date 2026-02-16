"""Hybrid BM25 + Embedding relation scorer with RRF fusion.

This module provides the HybridRelationScorer class that computes final scores
using Reciprocal Rank Fusion (RRF) of:
- BM25 lexical matching (Robertson et al. 1994)
- Embedding similarity (cosine)

RRF fusion (Cormack et al. SIGIR 2009) is used instead of weighted sums because:
- It uses ranks instead of scores (scale-invariant)
- It has only one hyperparameter (k=60, literature standard)
- It doesn't require arbitrary weight tuning

References:
    Robertson, S. E., & Walker, S. (1994). "Some simple effective approximations
    to the 2-Poisson model for probabilistic weighted retrieval." SIGIR '94.

    Cormack, G. V., Clarke, C. L., & Buettcher, S. (2009). "Reciprocal rank fusion
    outperforms condorcet and individual rank learning methods." SIGIR '09.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np
from loguru import logger
from rank_bm25 import BM25Okapi

if TYPE_CHECKING:
    from config.ontologies.relations import OntologyRelation


# =============================================================================
# RRF Constant (literature standard from Cormack et al. 2009)
# =============================================================================

RRF_K = 60  # Standard RRF smoothing constant


class HybridRelationScorer:
    """Hybrid relation scorer using BM25 + Embedding with RRF fusion.

    Unlike weighted-sum approaches with arbitrary weights (e.g., 0.4 + 0.4 + 0.2),
    RRF fusion:
    - Uses ranks instead of scores (scale-invariant)
    - Has theoretical backing (SIGIR 2009)
    - Has only one hyperparameter (k=60, standard)

    The scoring process:
    1. Score all relations using BM25 (lexical matching)
    2. Score all relations using embedding similarity (semantic matching)
    3. Rank relations by each signal independently
    4. Combine ranks using RRF: score(d) = 1/(k + bm25_rank) + 1/(k + emb_rank)

    Attributes:
        rrf_k: RRF smoothing constant (default: 60, from literature)
    """

    def __init__(self, rrf_k: int = RRF_K):
        """Initialize scorer.

        Args:
            rrf_k: RRF smoothing constant. Higher values give more weight
                   to lower-ranked items. Default 60 is from literature.
        """
        self.rrf_k = rrf_k

    def score_all(
        self,
        query: str,
        query_embedding: list[float],
        relations: list[OntologyRelation],
    ) -> list[tuple[str, float, float, float]]:
        """Score all relations using BM25 + Embedding hybrid with RRF.

        Args:
            query: User query text
            query_embedding: Query embedding vector
            relations: List of OntologyRelation objects

        Returns:
            List of (relation_id, rrf_score, bm25_score, emb_score)
            sorted by rrf_score descending
        """
        if not relations:
            return []

        n = len(relations)

        # Signal 1: BM25 scores
        bm25_scores = self._compute_bm25_scores(query, relations)

        # Signal 2: Embedding scores
        emb_scores = self._compute_embedding_scores(query_embedding, relations)

        # RRF fusion
        rrf_scores = self._rrf_fusion(bm25_scores, emb_scores)

        # Build results with relation IDs
        results = [
            (
                getattr(relations[i], "id", f"rel_{i}"),
                float(rrf_scores[i]),
                float(bm25_scores[i]),
                float(emb_scores[i]),
            )
            for i in range(n)
        ]

        # Sort by RRF score descending
        results.sort(key=lambda x: x[1], reverse=True)

        logger.debug(
            "HybridRelationScorer scored relations",
            n_relations=n,
            top_score=results[0][1] if results else 0,
            score_range=(results[-1][1], results[0][1]) if results else (0, 0),
        )

        return results

    def _compute_bm25_scores(
        self,
        query: str,
        relations: list[OntologyRelation],
    ) -> np.ndarray:
        """Compute BM25 scores for query against relation documents.

        BM25 is a probabilistic retrieval model that considers:
        - Term frequency (with saturation)
        - Inverse document frequency (rare terms weighted higher)
        - Document length normalization

        Each relation document is constructed from:
        - relation.description (primary text)
        - relation.keywords (joined as additional text)

        Args:
            query: User query text
            relations: List of OntologyRelation objects

        Returns:
            Array of BM25 scores (higher = more lexically relevant)
        """
        # Build corpus: each relation's description + keywords
        corpus = []
        for rel in relations:
            doc_text = getattr(rel, "description", "") or ""
            keywords = getattr(rel, "keywords", []) or []
            if keywords:
                doc_text += " " + " ".join(str(kw) for kw in keywords if kw)
            # Simple whitespace tokenization
            # Note: For Vietnamese, consider pyvi tokenizer for better results
            tokens = doc_text.lower().split()
            corpus.append(tokens if tokens else [""])

        # Build BM25 index
        bm25 = BM25Okapi(corpus)

        # Tokenize query
        query_tokens = query.lower().split() if query else [""]

        # Get scores
        scores = bm25.get_scores(query_tokens)

        return np.array(scores, dtype=np.float32)

    def _compute_embedding_scores(
        self,
        query_embedding: list[float],
        relations: list[OntologyRelation],
    ) -> np.ndarray:
        """Compute cosine similarity scores (vectorized).

        Args:
            query_embedding: Query embedding vector
            relations: List of OntologyRelation objects

        Returns:
            Array of cosine similarities in [0, 1]
        """
        n = len(relations)
        scores = np.zeros(n, dtype=np.float32)

        # Collect valid embeddings
        embeddings_list = []
        valid_indices = []

        for i, rel in enumerate(relations):
            emb = getattr(rel, "embedding", None)
            if emb is not None and len(emb) > 0:
                embeddings_list.append(emb)
                valid_indices.append(i)

        if not embeddings_list:
            return scores

        # Vectorized cosine similarity
        query_vec = np.array(query_embedding, dtype=np.float32)
        query_norm = np.linalg.norm(query_vec)

        if query_norm == 0:
            return scores

        rel_matrix = np.array(embeddings_list, dtype=np.float32)
        rel_norms = np.linalg.norm(rel_matrix, axis=1)

        # Compute similarities for valid entries
        valid_mask = rel_norms > 0
        if valid_mask.any():
            dot_products = np.dot(rel_matrix, query_vec)
            similarities = np.zeros(len(embeddings_list), dtype=np.float32)
            similarities[valid_mask] = dot_products[valid_mask] / (
                rel_norms[valid_mask] * query_norm
            )
            # Clamp to [0, 1] (negative similarities treated as 0)
            similarities = np.clip(similarities, 0, 1)

            for idx, sim in zip(valid_indices, similarities):
                scores[idx] = sim

        return scores

    def _rrf_fusion(
        self,
        bm25_scores: np.ndarray,
        emb_scores: np.ndarray,
    ) -> np.ndarray:
        """Apply Reciprocal Rank Fusion.

        RRF combines multiple ranking signals using ranks instead of raw scores:
            RRF(d) = sum_i(1 / (k + rank_i(d)))

        where:
        - k is the smoothing constant (default 60)
        - rank_i(d) is the rank of document d in ranking system i (1-indexed)

        Properties:
        - Scale-invariant: works regardless of score ranges
        - Robust: outlier scores don't dominate
        - Simple: no weight tuning required

        Args:
            bm25_scores: BM25 scores array
            emb_scores: Embedding similarity scores array

        Returns:
            RRF scores array (higher = better combined rank)
        """
        n = len(bm25_scores)

        if n == 0:
            return np.array([], dtype=np.float32)

        if n == 1:
            # Single item: max possible RRF score
            return np.array([2.0 / (self.rrf_k + 1)], dtype=np.float32)

        # Compute ranks (1-indexed, higher score = lower rank number)
        # argsort gives indices that would sort ascending
        # We want descending order, so negate scores
        bm25_order = np.argsort(-bm25_scores)
        emb_order = np.argsort(-emb_scores)

        # Create rank arrays (1-indexed)
        bm25_ranks = np.zeros(n, dtype=np.int32)
        emb_ranks = np.zeros(n, dtype=np.int32)

        for rank, idx in enumerate(bm25_order, start=1):
            bm25_ranks[idx] = rank
        for rank, idx in enumerate(emb_order, start=1):
            emb_ranks[idx] = rank

        # RRF formula: 1/(k + rank)
        rrf_scores = (1.0 / (self.rrf_k + bm25_ranks) + 1.0 / (self.rrf_k + emb_ranks)).astype(
            np.float32
        )

        return rrf_scores


# =============================================================================
# Backward Compatibility Alias (deprecated)
# =============================================================================


class RelationScorer(HybridRelationScorer):
    """Deprecated: Use HybridRelationScorer instead.

    This class is kept for backward compatibility but will be removed
    in a future version.
    """

    def __init__(self, *args, **kwargs):
        """Initialize with deprecation warning."""
        import warnings

        warnings.warn(
            "RelationScorer is deprecated. Use HybridRelationScorer instead.",
            DeprecationWarning,
            stacklevel=2,
        )
        # Ignore old-style weights if passed
        super().__init__()

    def score_all_vectorized(
        self,
        query: str,
        query_embedding: list[float],
        query_type: str,  # Ignored - no longer used
        relations: list,
    ) -> list[tuple[str, float, float, float, float]]:
        """Backward-compatible interface.

        Note: query_type is ignored. The new scorer doesn't use query types.

        Returns 5-tuples for compatibility: (id, rrf, bm25, emb, 0.0)
        The last element is always 0.0 (was keyword score, now unused).
        """
        results = self.score_all(query, query_embedding, relations)
        # Convert 4-tuples to 5-tuples for compatibility
        return [(r[0], r[1], r[2], r[3], 0.0) for r in results]
