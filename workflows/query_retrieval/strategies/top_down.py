"""E1: Top-Down Retrieval — search structure nodes, descend to content.

This strategy uses structure nodes as an "index" to find relevant regions
in the knowledge graph, then harvests content nodes from those regions.

Algorithm:
    1. PRE-FILTER: Build pool of content-rich structure nodes
       (where HAS_CONTENT count >= HAS_SECTION count)
    2. STRUCTURE SEARCH: Cosine search on pool to find top-K sections
    3. CONTENT HARVEST: Descend through HAS_SECTION edges (max 2 levels),
       collect all HAS_CONTENT nodes
    4. RE-RANK: Cosine re-rank harvested content, return top-K final
"""

from __future__ import annotations

import asyncio
import time
from dataclasses import dataclass
from typing import TYPE_CHECKING

import numpy as np
from loguru import logger

from workflows.query_retrieval.models import ContentWithContext
from workflows.query_retrieval.strategies.base import (
    cosine_similarity_batch,
    get_executor,
    rrf_fusion_2way,
    to_content_with_context,
)
from workflows.query_retrieval.strategies.scoring import RRF_K, E1ScoringMethod

if TYPE_CHECKING:
    from arango.database import StandardDatabase

# =============================================================================
# Constants
# =============================================================================

DEFAULT_K_STRUCTURE = 10
DEFAULT_K_FINAL = 15
MAX_DESCEND_DEPTH = 2
MAX_COLLECTED_NODES = 100

# Pool size warning thresholds
POOL_MIN_WARNING = 10
POOL_MAX_WARNING = 500

# =============================================================================
# AQL Queries
# =============================================================================

AQL_CONTENT_RICH_POOL = """
FOR s IN structure_nodes
    FILTER @source_id == null OR s.source_id == @source_id
    LET has_content_count = LENGTH(
        FOR v, e IN 1..1 OUTBOUND s knowledge_edges
            FILTER e.type == "HAS_CONTENT"
            RETURN 1
    )
    LET has_section_count = LENGTH(
        FOR v, e IN 1..1 OUTBOUND s knowledge_edges
            FILTER e.type == "HAS_SECTION"
            RETURN 1
    )
    FILTER has_content_count > 0
    FILTER has_content_count >= has_section_count
    RETURN {
        _id: s._id,
        _key: s._key,
        embedding: s.embedding,
        level: s.level,
        summary: s.summary,
        has_content_count: has_content_count
    }
"""

AQL_CONTENT_HARVEST = """
LET direct_content = (
    FOR start_id IN @structure_node_ids
        FOR v, e IN 1..1 OUTBOUND DOCUMENT(start_id) knowledge_edges
            FILTER e.type == "HAS_CONTENT"
            RETURN { _id: v._id, _key: v._key, text: v.text,
                     embedding: v.embedding, node_type: v.node_type,
                     source: start_id, depth: 0 }
)

LET descend_content = (
    FOR start_id IN @structure_node_ids
        FOR v, e, p IN 2..3 OUTBOUND DOCUMENT(start_id) knowledge_edges
            FILTER p.edges[*].type ALL IN ["HAS_SECTION", "HAS_CONTENT"]
            FILTER LAST(p.edges).type == "HAS_CONTENT"
            LET depth = LENGTH(p.edges) - 1
            FILTER depth <= @max_descend_depth
            RETURN { _id: v._id, _key: v._key, text: v.text,
                     embedding: v.embedding, node_type: v.node_type,
                     source: start_id, depth: depth }
)

FOR item IN UNION_DISTINCT(direct_content, descend_content)
    RETURN item
"""


# =============================================================================
# Metrics
# =============================================================================


@dataclass
class TopDownMetrics:
    """Metrics collected during E1 Top-Down Retrieval execution."""

    pool_size: int = 0
    structure_nodes_matched: int = 0
    content_nodes_harvested: int = 0
    content_nodes_returned: int = 0
    descend_depth_actual: int = 0
    latency_pool_ms: float = 0.0
    latency_search_ms: float = 0.0
    latency_harvest_ms: float = 0.0
    latency_rerank_ms: float = 0.0
    latency_total_ms: float = 0.0

    def to_dict(self) -> dict:
        """Convert to dict for inclusion in QueryRetrievalOutput timing."""
        return {
            "pool_size": self.pool_size,
            "structure_nodes_matched": self.structure_nodes_matched,
            "content_nodes_harvested": self.content_nodes_harvested,
            "content_nodes_returned": self.content_nodes_returned,
            "descend_depth_actual": self.descend_depth_actual,
            "latency_pool_ms": round(self.latency_pool_ms, 1),
            "latency_search_ms": round(self.latency_search_ms, 1),
            "latency_harvest_ms": round(self.latency_harvest_ms, 1),
            "latency_rerank_ms": round(self.latency_rerank_ms, 1),
            "latency_total_ms": round(self.latency_total_ms, 1),
        }


# =============================================================================
# Top-Down Retrieval Strategy
# =============================================================================


class TopDownRetrieval:
    """E1: Top-Down Retrieval strategy.

    Searches structure nodes (sections/subsections) by embedding similarity,
    then descends to harvest content nodes from matched regions.

    Attributes:
        db: ArangoDB database connection
        metrics: Collected metrics from last execution
    """

    def __init__(
        self,
        db: StandardDatabase,
    ):
        self.db = db
        self.metrics = TopDownMetrics()
        self._pool_cache: dict[str, list[dict]] = {}
        self._log = logger.bind(strategy="top_down")
        self._executor = get_executor()

    async def execute(
        self,
        query_embedding: list[float],
        top_k: int = DEFAULT_K_FINAL,
        source_id: str | None = None,
        k_structure: int = DEFAULT_K_STRUCTURE,
        scoring_method: E1ScoringMethod = E1ScoringMethod.COSINE,
    ) -> list[ContentWithContext]:
        """Execute E1 Top-Down Retrieval pipeline.

        Args:
            query_embedding: Query embedding vector
            top_k: Number of final content nodes to return
            source_id: Optional source ID to scope the search
            k_structure: Number of top structure nodes to match
            scoring_method: Scoring method to use (default: COSINE for backward compat)

        Returns:
            List of ContentWithContext results ranked by relevance
        """
        total_start = time.perf_counter()
        self.metrics = TopDownMetrics()

        # Step 0: Build content-rich pool
        t0 = time.perf_counter()
        pool = await self._build_content_rich_pool(source_id)
        self.metrics.latency_pool_ms = (time.perf_counter() - t0) * 1000
        self.metrics.pool_size = len(pool)

        if not pool:
            self._log.warning("Empty content-rich pool, returning no results")
            self.metrics.latency_total_ms = (time.perf_counter() - total_start) * 1000
            return []

        # Step 1: Structure cosine search
        t1 = time.perf_counter()
        top_structures = self._structure_cosine_search(query_embedding, pool, k_structure)
        self.metrics.latency_search_ms = (time.perf_counter() - t1) * 1000
        self.metrics.structure_nodes_matched = len(top_structures)

        if not top_structures:
            self._log.warning("No structure nodes matched")
            self.metrics.latency_total_ms = (time.perf_counter() - total_start) * 1000
            return []

        # Step 2: Content harvest
        t2 = time.perf_counter()
        harvested = await self._content_harvest([s["_id"] for s in top_structures])
        self.metrics.latency_harvest_ms = (time.perf_counter() - t2) * 1000
        self.metrics.content_nodes_harvested = len(harvested)

        if harvested:
            self.metrics.descend_depth_actual = max(h["depth"] for h in harvested)

        # Store seed scores for RRF scoring (structure_id -> cosine score)
        seed_scores = {s["_id"]: s["structure_score"] for s in top_structures}

        # Step 3: Re-rank based on scoring method
        t3 = time.perf_counter()
        if scoring_method == E1ScoringMethod.COSINE:
            results = self._rerank_cosine(query_embedding, harvested, top_k)
        elif scoring_method == E1ScoringMethod.RRF_CONTENT_SEED:
            results = self._rerank_rrf_content_seed(query_embedding, harvested, seed_scores, top_k)
        else:
            results = self._rerank_cosine(query_embedding, harvested, top_k)
        self.metrics.latency_rerank_ms = (time.perf_counter() - t3) * 1000
        self.metrics.content_nodes_returned = len(results)

        self.metrics.latency_total_ms = (time.perf_counter() - total_start) * 1000

        self._log.info(
            "Top-down retrieval completed",
            pool_size=self.metrics.pool_size,
            matched=self.metrics.structure_nodes_matched,
            harvested=self.metrics.content_nodes_harvested,
            returned=self.metrics.content_nodes_returned,
            latency_ms=f"{self.metrics.latency_total_ms:.0f}",
        )

        return results

    # =========================================================================
    # Step 0: Content-Rich Pool
    # =========================================================================

    async def _build_content_rich_pool(
        self,
        source_id: str | None = None,
    ) -> list[dict]:
        """Build pool of content-rich structure nodes.

        Content-rich = HAS_CONTENT count >= HAS_SECTION count.
        This filters out hub nodes (roots, chapters) keeping only
        leaf/near-leaf sections that contain actual content.

        Args:
            source_id: Optional source ID filter

        Returns:
            List of pool node dicts with _id, _key, embedding, level, summary
        """
        cache_key = source_id or "__global__"
        if cache_key in self._pool_cache:
            return self._pool_cache[cache_key]

        cursor = await asyncio.get_event_loop().run_in_executor(
            self._executor,
            lambda: self.db.aql.execute(
                AQL_CONTENT_RICH_POOL,
                bind_vars={"source_id": source_id},
            ),
        )

        pool = list(cursor)

        # Validate pool size
        if len(pool) == 0:
            self._log.warning("Content-rich pool is empty")
        elif len(pool) < POOL_MIN_WARNING:
            self._log.warning(
                "Content-rich pool is small",
                pool_size=len(pool),
                threshold=POOL_MIN_WARNING,
            )
        elif len(pool) > POOL_MAX_WARNING:
            self._log.warning(
                "Content-rich pool is large",
                pool_size=len(pool),
                threshold=POOL_MAX_WARNING,
            )

        self._pool_cache[cache_key] = pool
        return pool

    def invalidate_pool_cache(self) -> None:
        """Clear the content-rich pool cache."""
        self._pool_cache.clear()

    # =========================================================================
    # Step 1: Structure Cosine Search
    # =========================================================================

    def _structure_cosine_search(
        self,
        query_embedding: list[float],
        pool: list[dict],
        top_k: int,
    ) -> list[dict]:
        """Search pool for top-K structure nodes by cosine similarity.

        Uses in-memory numpy computation since pool is typically small
        (100-300 nodes). This avoids ArangoSearch view limitations with
        pool ID filtering.

        Args:
            query_embedding: Query embedding vector
            pool: Content-rich pool nodes
            top_k: Number of top structure nodes to return

        Returns:
            Top-K pool nodes sorted by cosine score descending
        """
        # Filter nodes with embeddings
        nodes_with_emb = [n for n in pool if n.get("embedding")]
        skipped = len(pool) - len(nodes_with_emb)

        if skipped > 0:
            self._log.debug(
                "Skipped nodes without embeddings",
                skipped=skipped,
                total=len(pool),
            )

        if not nodes_with_emb:
            return []

        scores = cosine_similarity_batch(query_embedding, [n["embedding"] for n in nodes_with_emb])

        # Attach scores and sort
        scored_nodes = []
        for node, score in zip(nodes_with_emb, scores):
            if score > 0:
                node_copy = dict(node)
                node_copy["structure_score"] = float(score)
                scored_nodes.append(node_copy)

        scored_nodes.sort(key=lambda x: x["structure_score"], reverse=True)

        return scored_nodes[:top_k]

    # =========================================================================
    # Step 2: Content Harvest
    # =========================================================================

    async def _content_harvest(
        self,
        structure_node_ids: list[str],
    ) -> list[dict]:
        """Harvest content nodes from matched structure nodes.

        Collects direct HAS_CONTENT children and descends through
        HAS_SECTION edges (max 2 levels) to gather deeper content.

        Args:
            structure_node_ids: List of structure node _ids to harvest from

        Returns:
            List of content node dicts, deduplicated, depth-prioritized
        """
        if not structure_node_ids:
            return []

        cursor = await asyncio.get_event_loop().run_in_executor(
            self._executor,
            lambda: self.db.aql.execute(
                AQL_CONTENT_HARVEST,
                bind_vars={
                    "structure_node_ids": structure_node_ids,
                    "max_descend_depth": MAX_DESCEND_DEPTH,
                },
            ),
        )

        harvested = list(cursor)

        # Deduplicate by _id, keeping minimum depth
        seen: dict[str, dict] = {}
        for item in harvested:
            node_id = item["_id"]
            if node_id not in seen or item["depth"] < seen[node_id]["depth"]:
                seen[node_id] = item
        unique = list(seen.values())

        # Cap collection size, prioritize shallow nodes
        if len(unique) > MAX_COLLECTED_NODES:
            unique.sort(key=lambda x: x["depth"])
            unique = unique[:MAX_COLLECTED_NODES]

        return unique

    # =========================================================================
    # Step 3: Re-rank & Return
    # =========================================================================

    def _rerank_cosine(
        self,
        query_embedding: list[float],
        content_nodes: list[dict],
        top_k: int,
    ) -> list[ContentWithContext]:
        """Re-rank harvested content nodes by cosine similarity with query.

        Args:
            query_embedding: Query embedding vector
            content_nodes: Harvested content node dicts
            top_k: Number of final results to return

        Returns:
            Top-K ContentWithContext results sorted by score
        """
        if not content_nodes:
            return []

        nodes_with_emb = [n for n in content_nodes if n.get("embedding")]
        nodes_without_emb = [n for n in content_nodes if not n.get("embedding")]

        if nodes_without_emb:
            self._log.debug(
                "Content nodes missing embeddings",
                count=len(nodes_without_emb),
            )

        if not nodes_with_emb:
            # Fallback: return nodes without scores
            return [to_content_with_context(n, score=0.0) for n in content_nodes[:top_k]]

        scores = cosine_similarity_batch(query_embedding, [n["embedding"] for n in nodes_with_emb])

        # Pair nodes with scores (without mutating input dicts)
        scored_pairs = sorted(
            zip(nodes_with_emb, scores),
            key=lambda pair: pair[1],
            reverse=True,
        )

        # Build results: scored nodes first, then unscored if needed
        results = [
            to_content_with_context(node, float(score)) for node, score in scored_pairs[:top_k]
        ]

        remaining = top_k - len(results)
        if remaining > 0 and nodes_without_emb:
            results.extend(
                to_content_with_context(n, score=0.0) for n in nodes_without_emb[:remaining]
            )

        return results

    def _rerank_rrf_content_seed(
        self,
        query_embedding: list[float],
        content_nodes: list[dict],
        seed_scores: dict[str, float],
        top_k: int,
    ) -> list[ContentWithContext]:
        """Re-rank using RRF of content cosine and seed cosine.

        Formula: 1/(k + rank_content_cosine) + 1/(k + rank_seed_cosine)

        Args:
            query_embedding: Query embedding vector
            content_nodes: Harvested content nodes with 'source' field
            seed_scores: Mapping seed_id -> cosine score
            top_k: Number of results to return

        Returns:
            Top-K ContentWithContext sorted by RRF score
        """
        if not content_nodes:
            return []

        # Filter nodes with embeddings
        nodes_with_emb = [n for n in content_nodes if n.get("embedding")]

        if not nodes_with_emb:
            return []

        # Signal 1: Content cosine scores
        content_cosines = cosine_similarity_batch(
            query_embedding, [n["embedding"] for n in nodes_with_emb]
        )

        # Signal 2: Seed cosine scores (lookup from seed_scores dict)
        seed_cosines = np.array(
            [seed_scores.get(n.get("source", ""), 0.0) for n in nodes_with_emb],
            dtype=np.float32,
        )

        # RRF fusion
        rrf_scores = rrf_fusion_2way(content_cosines, seed_cosines, k=RRF_K, use_tie_handling=True)

        # Sort and return top-K
        scored_tuples = sorted(
            zip(nodes_with_emb, rrf_scores, content_cosines, seed_cosines),
            key=lambda x: x[1],
            reverse=True,
        )[:top_k]

        return [
            to_content_with_context(node, float(rrf_score))
            for node, rrf_score, _, _ in scored_tuples
        ]
