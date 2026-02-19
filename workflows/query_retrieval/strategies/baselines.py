"""B2 (BM25) and B3 (Hybrid RRF) baseline retrieval strategies.

Baselines for benchmarking against experimental strategies (E1/E2/E3).
- B2: Pure BM25 full-text search via ArangoSearch
- B3: Hybrid vector + BM25 with Reciprocal Rank Fusion
"""

from __future__ import annotations

import asyncio
import time
from dataclasses import dataclass
from typing import TYPE_CHECKING

import numpy as np
from arango_client.handlers.search_handler import SearchHandler, SearchResult
from loguru import logger

from workflows.query_retrieval.models import ContentWithContext
from workflows.query_retrieval.strategies.base import get_executor, rrf_fusion_2way

if TYPE_CHECKING:
    from arango.database import StandardDatabase


# =============================================================================
# Metrics
# =============================================================================


@dataclass
class BaselineMetrics:
    """Metrics collected during baseline retrieval execution."""

    strategy: str = ""
    search_time_ms: float = 0.0
    merge_time_ms: float = 0.0
    total_time_ms: float = 0.0
    total_candidates: int = 0
    vector_candidates: int = 0
    bm25_candidates: int = 0

    def to_dict(self) -> dict:
        """Convert to dict for inclusion in QueryRetrievalOutput."""
        return {
            "strategy": self.strategy,
            "search_time_ms": round(self.search_time_ms, 1),
            "merge_time_ms": round(self.merge_time_ms, 1),
            "total_time_ms": round(self.total_time_ms, 1),
            "total_candidates": self.total_candidates,
            "vector_candidates": self.vector_candidates,
            "bm25_candidates": self.bm25_candidates,
        }


# =============================================================================
# Helpers
# =============================================================================


def _search_result_to_content(result: SearchResult, score: float) -> ContentWithContext:
    """Convert a SearchResult to ContentWithContext."""
    node = result.node
    text = getattr(node, "text", None) or ""
    if not text and node.attributes:
        text = node.attributes.get("text", "") if isinstance(node.attributes, dict) else ""
    return ContentWithContext(
        node_id=str(node.id) if node.id else "",
        text=text,
        score=score,
        node_type=node.node_type if isinstance(node.node_type, str) else node.node_type.value,
    )


# =============================================================================
# B2: BM25 Retrieval
# =============================================================================


class BM25Retrieval:
    """B2: Pure BM25 full-text search baseline.

    Uses ArangoSearch TOKENS() ALL IN with BM25 scoring.
    No vector embeddings involved.
    """

    def __init__(self, db: StandardDatabase):
        self.db = db
        self.search_handler = SearchHandler(db)
        self.metrics = BaselineMetrics(strategy="bm25")
        self._executor = get_executor()
        self._log = logger.bind(strategy="bm25")

    async def execute(
        self,
        query_text: str,
        top_k: int = 10,
        source_id: str | None = None,
    ) -> list[ContentWithContext]:
        """Execute B2 BM25 retrieval.

        Args:
            query_text: Search query text
            top_k: Number of results to return
            source_id: Optional source ID filter

        Returns:
            List of ContentWithContext sorted by BM25 score
        """
        total_start = time.perf_counter()
        self.metrics = BaselineMetrics(strategy="bm25")

        # BM25 search
        t0 = time.perf_counter()
        results = await asyncio.get_event_loop().run_in_executor(
            self._executor,
            lambda: self.search_handler.bm25_search_content(
                query_text=query_text,
                limit=top_k,
                source_id=source_id,
            ),
        )
        self.metrics.search_time_ms = (time.perf_counter() - t0) * 1000
        self.metrics.bm25_candidates = len(results)
        self.metrics.total_candidates = len(results)

        # Convert to ContentWithContext
        content_results = [_search_result_to_content(r, r.score) for r in results]

        self.metrics.total_time_ms = (time.perf_counter() - total_start) * 1000

        self._log.info(
            "BM25 retrieval completed",
            candidates=self.metrics.bm25_candidates,
            returned=len(content_results),
            latency_ms=f"{self.metrics.total_time_ms:.0f}",
        )

        return content_results


# =============================================================================
# B3: Hybrid (Vector + BM25) Retrieval with RRF
# =============================================================================


class HybridRetrieval:
    """B3: Hybrid Vector + BM25 retrieval with RRF fusion.

    Retrieves candidates from both vector search and BM25 search,
    then merges via Reciprocal Rank Fusion for final ranking.
    """

    def __init__(self, db: StandardDatabase):
        self.db = db
        self.search_handler = SearchHandler(db)
        self.metrics = BaselineMetrics(strategy="hybrid")
        self._executor = get_executor()
        self._log = logger.bind(strategy="hybrid")

    async def execute(
        self,
        query_text: str,
        query_embedding: list[float],
        top_k: int = 10,
        source_id: str | None = None,
    ) -> list[ContentWithContext]:
        """Execute B3 Hybrid retrieval with RRF fusion.

        Args:
            query_text: Search query text (for BM25)
            query_embedding: Query embedding vector (for vector search)
            top_k: Number of final results to return
            source_id: Optional source ID filter

        Returns:
            List of ContentWithContext sorted by RRF score
        """
        total_start = time.perf_counter()
        self.metrics = BaselineMetrics(strategy="hybrid")

        # Over-retrieve from both sources for better fusion
        retrieve_k = top_k * 2

        # Run both searches in parallel
        t0 = time.perf_counter()
        loop = asyncio.get_event_loop()
        vector_future = loop.run_in_executor(
            self._executor,
            lambda: self.search_handler.vector_search_content(
                query_vector=query_embedding,
                limit=retrieve_k,
                source_id=source_id,
            ),
        )
        bm25_future = loop.run_in_executor(
            self._executor,
            lambda: self.search_handler.bm25_search_content(
                query_text=query_text,
                limit=retrieve_k,
                source_id=source_id,
            ),
        )
        vector_results, bm25_results = await asyncio.gather(vector_future, bm25_future)
        self.metrics.search_time_ms = (time.perf_counter() - t0) * 1000
        self.metrics.vector_candidates = len(vector_results)
        self.metrics.bm25_candidates = len(bm25_results)

        # Merge via RRF
        t1 = time.perf_counter()
        content_results = self._rrf_merge(vector_results, bm25_results, top_k)
        self.metrics.merge_time_ms = (time.perf_counter() - t1) * 1000
        self.metrics.total_candidates = (
            self.metrics.vector_candidates + self.metrics.bm25_candidates
        )

        self.metrics.total_time_ms = (time.perf_counter() - total_start) * 1000

        self._log.info(
            "Hybrid retrieval completed",
            vector_candidates=self.metrics.vector_candidates,
            bm25_candidates=self.metrics.bm25_candidates,
            returned=len(content_results),
            latency_ms=f"{self.metrics.total_time_ms:.0f}",
        )

        return content_results

    def _rrf_merge(
        self,
        vector_results: list[SearchResult],
        bm25_results: list[SearchResult],
        top_k: int,
    ) -> list[ContentWithContext]:
        """Merge vector and BM25 results via RRF fusion.

        Builds a union of all unique nodes, constructs aligned score arrays
        with 0 for missing nodes in each signal, then applies RRF.

        Args:
            vector_results: Results from vector search
            bm25_results: Results from BM25 search
            top_k: Number of final results

        Returns:
            Top-K ContentWithContext sorted by RRF score
        """
        if not vector_results and not bm25_results:
            return []

        # Build union set keyed by node_id
        # Store: {node_id: (SearchResult, vector_score, bm25_score)}
        node_map: dict[str, tuple[SearchResult, float, float]] = {}

        for r in vector_results:
            node_id = str(r.node.id) if r.node.id else ""
            if node_id not in node_map:
                node_map[node_id] = (r, r.score, 0.0)
            else:
                existing = node_map[node_id]
                node_map[node_id] = (existing[0], r.score, existing[2])

        for r in bm25_results:
            node_id = str(r.node.id) if r.node.id else ""
            if node_id not in node_map:
                node_map[node_id] = (r, 0.0, r.score)
            else:
                existing = node_map[node_id]
                node_map[node_id] = (existing[0], existing[1], r.score)

        if not node_map:
            return []

        # Build aligned score arrays
        node_ids = list(node_map.keys())
        vector_scores = np.array([node_map[nid][1] for nid in node_ids], dtype=np.float32)
        bm25_scores = np.array([node_map[nid][2] for nid in node_ids], dtype=np.float32)

        # RRF fusion
        rrf_scores = rrf_fusion_2way(vector_scores, bm25_scores)

        # Sort by RRF score descending
        sorted_indices = np.argsort(-rrf_scores)[:top_k]

        results = []
        for idx in sorted_indices:
            nid = node_ids[idx]
            search_result = node_map[nid][0]
            results.append(_search_result_to_content(search_result, float(rrf_scores[idx])))

        return results
