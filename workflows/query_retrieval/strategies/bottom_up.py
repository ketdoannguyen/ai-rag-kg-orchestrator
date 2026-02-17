"""E3: Bottom-Up Retrieval — aggregate parent summaries from content nodes.

This strategy finds relevant content nodes and ascends to their parent
structure nodes, returning parent summaries instead of content nodes.

Algorithm:
    1. OVER-RETRIEVAL: Vector search for top-(K*M) content nodes
    2. PARENT DISCOVERY: Batch INBOUND HAS_CONTENT traversal to find parents
    3. PARENT SCORING: score = AVG(children_cosine) * ln(1 + count)
    4. SELECT & RETURN: Top-K parents as ContentWithContext with summaries

Core Insight:
    B1/E1/E2 return content nodes (detailed paragraphs).
    E3 returns parent summaries (section-level synthesis).
    This tests whether aggregated perspective can answer queries
    (especially lesson_synthesis, structural) better than details.
"""

from __future__ import annotations

import asyncio
import math
import time
from dataclasses import dataclass, field
from typing import TYPE_CHECKING

import numpy as np
from loguru import logger

from workflows.query_retrieval.models import ContentWithContext
from workflows.query_retrieval.strategies.base import (
    cosine_similarity_batch,
    get_executor,
    rrf_fusion_2way,
)
from workflows.query_retrieval.strategies.scoring import RRF_K, E3ScoringMethod

if TYPE_CHECKING:
    from arango.database import StandardDatabase

# =============================================================================
# Constants
# =============================================================================

DEFAULT_TOP_K = 10  # Final number of parents to return
OVER_RETRIEVAL_MULTIPLIER = 3  # M = 3 -> retrieve K*M content nodes
MAX_CONTENT_POOL = 100  # Cap for over-retrieval to prevent explosion

# =============================================================================
# Metrics
# =============================================================================


@dataclass
class BottomUpMetrics:
    """Metrics collected during E3 Bottom-Up Retrieval execution."""

    content_nodes_retrieved: int = 0
    unique_parents_found: int = 0
    parents_returned: int = 0
    avg_children_per_parent: float = 0.0
    max_children_per_parent: int = 0
    min_children_per_parent: int = 0
    parent_levels_distribution: dict[int, int] = field(default_factory=dict)
    parents_with_summary: int = 0
    parents_with_title_fallback: int = 0
    latency_over_retrieve_ms: float = 0.0
    latency_parent_discovery_ms: float = 0.0
    latency_scoring_ms: float = 0.0
    latency_total_ms: float = 0.0

    def to_dict(self) -> dict:
        """Convert to dict for inclusion in QueryRetrievalOutput."""
        return {
            "content_nodes_retrieved": self.content_nodes_retrieved,
            "unique_parents_found": self.unique_parents_found,
            "parents_returned": self.parents_returned,
            "avg_children_per_parent": round(self.avg_children_per_parent, 2),
            "max_children_per_parent": self.max_children_per_parent,
            "min_children_per_parent": self.min_children_per_parent,
            "parent_levels_distribution": self.parent_levels_distribution,
            "parents_with_summary": self.parents_with_summary,
            "parents_with_title_fallback": self.parents_with_title_fallback,
            "latency_over_retrieve_ms": round(self.latency_over_retrieve_ms, 1),
            "latency_parent_discovery_ms": round(self.latency_parent_discovery_ms, 1),
            "latency_scoring_ms": round(self.latency_scoring_ms, 1),
            "latency_total_ms": round(self.latency_total_ms, 1),
        }


# =============================================================================
# AQL Queries
# =============================================================================

AQL_OVER_RETRIEVE_CONTENT = """
FOR c IN knowledge_vector_view
    SEARCH ANALYZER(c.node_type IN ["paragraph", "image", "table"], "identity")
    LET score = COSINE_SIMILARITY(c.embedding, @query_embedding)
    FILTER score > 0
    SORT score DESC
    LIMIT @limit
    RETURN {
        _id: c._id,
        _key: c._key,
        node_type: c.node_type,
        text: c.text,
        score: score
    }
"""

AQL_PARENT_DISCOVERY = """
FOR content IN @content_nodes
    LET parent = FIRST(
        FOR v, e IN 1..1 INBOUND content._id GRAPH @graph_name
            FILTER e.type == "HAS_CONTENT"
            RETURN v
    )
    FILTER parent != null
    RETURN {
        content_id: content._id,
        content_score: content.score,
        parent_id: parent._id,
        parent_key: parent._key,
        parent_title: parent.attributes.title,
        parent_summary: parent.summary,
        parent_level: parent.level,
        parent_node_type: parent.node_type,
        parent_embedding: parent.embedding
    }
"""


# =============================================================================
# Bottom-Up Retrieval Strategy
# =============================================================================


class BottomUpRetrieval:
    """E3: Bottom-Up Retrieval strategy.

    Finds relevant content nodes and ascends to their parent structure nodes,
    returning parent summaries aggregated by children relevance scores.

    Attributes:
        db: ArangoDB database connection
        metrics: Collected metrics from last execution
    """

    def __init__(
        self,
        db: StandardDatabase,
        graph_name: str = "knowledge_graph",
    ) -> None:
        self.db = db
        self._graph_name = graph_name
        self.metrics = BottomUpMetrics()
        self._log = logger.bind(strategy="bottom_up")
        self._executor = get_executor()

    async def execute(
        self,
        query_embedding: list[float],
        top_k: int = DEFAULT_TOP_K,
        multiplier: int = OVER_RETRIEVAL_MULTIPLIER,
        scoring_method: E3ScoringMethod = E3ScoringMethod.AVG_LOG,
    ) -> list[ContentWithContext]:
        """Execute E3 Bottom-Up Retrieval pipeline.

        Args:
            query_embedding: Query embedding vector (3072 dims)
            top_k: Number of parent summaries to return
            multiplier: Over-retrieval multiplier M
            scoring_method: Scoring method to use (default: AVG_LOG for backward compat)

        Returns:
            List of ContentWithContext with parent summaries
        """
        total_start = time.perf_counter()

        # Reset metrics for this execution
        self.metrics = BottomUpMetrics()

        # Stage 1: Over-retrieve content nodes
        content_nodes = await self._over_retrieve_content(query_embedding, top_k, multiplier)

        if not content_nodes:
            self._log.warning("No content nodes found, returning empty results")
            self.metrics.latency_total_ms = (time.perf_counter() - total_start) * 1000
            return []

        # Stage 2: Discover parents
        parents_dict = await self._discover_parents(content_nodes)

        if not parents_dict:
            self._log.warning("No parents found, returning empty results")
            self.metrics.latency_total_ms = (time.perf_counter() - total_start) * 1000
            return []

        # Stage 3: Score parents based on scoring_method
        if scoring_method == E3ScoringMethod.PARENT_COSINE:
            scored_parents = self._score_parent_cosine(query_embedding, parents_dict)
        elif scoring_method == E3ScoringMethod.AVG_LOG:
            scored_parents = self._score_parents_avg_log(parents_dict)
        elif scoring_method == E3ScoringMethod.RRF_PARENT_AVGLOG:
            scored_parents = self._score_rrf_parent_avglog(query_embedding, parents_dict)
        else:
            scored_parents = self._score_parents_avg_log(parents_dict)

        # Stage 4: Select and return top-K
        results = self._select_and_return(scored_parents, top_k)

        self.metrics.latency_total_ms = (time.perf_counter() - total_start) * 1000

        self._log.info(
            "Bottom-up retrieval completed",
            content_retrieved=self.metrics.content_nodes_retrieved,
            unique_parents=self.metrics.unique_parents_found,
            returned=self.metrics.parents_returned,
            with_summary=self.metrics.parents_with_summary,
            latency_ms=f"{self.metrics.latency_total_ms:.0f}",
        )

        return results

    # =========================================================================
    # Stage 1: Over-Retrieval
    # =========================================================================

    async def _over_retrieve_content(
        self,
        query_embedding: list[float],
        top_k: int,
        multiplier: int,
    ) -> list[dict]:
        """Stage 1: Over-retrieve top-(K*M) content nodes.

        Args:
            query_embedding: Query embedding vector
            top_k: Target number of final results
            multiplier: Over-retrieval multiplier (M)

        Returns:
            List of content node dicts with _id, _key, score
        """
        start_time = time.perf_counter()

        limit = min(top_k * multiplier, MAX_CONTENT_POOL)

        loop = asyncio.get_event_loop()
        cursor = await loop.run_in_executor(
            self._executor,
            lambda: self.db.aql.execute(
                AQL_OVER_RETRIEVE_CONTENT,
                bind_vars={
                    "query_embedding": query_embedding,
                    "limit": limit,
                },
            ),
        )
        results = list(cursor)

        # Update metrics
        self.metrics.content_nodes_retrieved = len(results)
        self.metrics.latency_over_retrieve_ms = (time.perf_counter() - start_time) * 1000

        self._log.debug(
            "Over-retrieval done",
            requested=limit,
            returned=len(results),
        )

        return results

    # =========================================================================
    # Stage 2: Parent Discovery
    # =========================================================================

    async def _discover_parents(
        self,
        content_nodes: list[dict],
    ) -> dict[str, dict]:
        """Stage 2: Batch discover parent for each content node.

        Args:
            content_nodes: Content nodes from over-retrieval

        Returns:
            Dict mapping parent_id → {info, children_scores}
        """
        start_time = time.perf_counter()

        if not content_nodes:
            return {}

        loop = asyncio.get_event_loop()
        cursor = await loop.run_in_executor(
            self._executor,
            lambda: self.db.aql.execute(
                AQL_PARENT_DISCOVERY,
                bind_vars={
                    "content_nodes": content_nodes,
                    "graph_name": self._graph_name,
                },
            ),
        )
        raw_results = list(cursor)

        # Group by parent
        parents = self._group_by_parent(raw_results)

        # Update metrics
        self.metrics.unique_parents_found = len(parents)
        self.metrics.latency_parent_discovery_ms = (time.perf_counter() - start_time) * 1000

        self._log.debug(
            "Parent discovery done",
            content_nodes=len(content_nodes),
            unique_parents=len(parents),
        )

        return parents

    def _group_by_parent(self, raw_results: list[dict]) -> dict[str, dict]:
        """Group content nodes by their parent.

        Args:
            raw_results: Raw AQL results with content->parent mappings

        Returns:
            Dict: parent_id -> {key, title, summary, level, node_type, embedding, children_scores}
        """
        parents: dict[str, dict] = {}

        for row in raw_results:
            parent_id = row["parent_id"]

            if parent_id not in parents:
                parents[parent_id] = {
                    "key": row["parent_key"],
                    "title": row["parent_title"],
                    "summary": row["parent_summary"],
                    "level": row["parent_level"],
                    "node_type": row["parent_node_type"],
                    "embedding": row.get("parent_embedding"),
                    "children_scores": [],
                }

            parents[parent_id]["children_scores"].append(row["content_score"])

        return parents

    # =========================================================================
    # Stage 3: Parent Scoring
    # =========================================================================

    def _score_parents_avg_log(
        self,
        parents_dict: dict[str, dict],
    ) -> list[dict]:
        """Score parents using AVG * ln(1+count) formula.

        Scoring rationale:
            - AVG: Parent with high-cosine children is more relevant
            - ln(1+count): Parent with more children has broader coverage
            - Combined: Balance quality (AVG) and comprehensiveness (count)

        Args:
            parents_dict: Dict from parent discovery

        Returns:
            List of scored parents sorted by score descending
        """
        start_time = time.perf_counter()

        scored_parents: list[dict] = []
        children_counts: list[int] = []

        for parent_id, data in parents_dict.items():
            children_scores = data["children_scores"]
            count = len(children_scores)

            if count == 0:
                continue

            avg_score = sum(children_scores) / count

            # Formula: AVG * ln(1 + count)
            aggregated_score = avg_score * math.log(1 + count)

            scored_parents.append(
                {
                    "parent_id": parent_id,
                    "key": data["key"],
                    "title": data["title"],
                    "summary": data["summary"],
                    "level": data["level"],
                    "node_type": data["node_type"],
                    "aggregated_score": aggregated_score,
                    "avg_cosine": avg_score,
                    "children_count": count,
                    "scoring_method": "avg_log",
                }
            )

            children_counts.append(count)

        # Sort by score descending
        scored_parents.sort(key=lambda x: x["aggregated_score"], reverse=True)

        # Update metrics
        self._update_scoring_metrics(scored_parents, children_counts, start_time)

        return scored_parents

    def _score_parent_cosine(
        self,
        query_embedding: list[float],
        parents_dict: dict[str, dict],
    ) -> list[dict]:
        """Score parents by direct cosine similarity with query.

        Formula: score = cosine(query_embedding, parent_embedding)

        Args:
            query_embedding: Query embedding vector
            parents_dict: Dict from parent discovery (must include embedding)

        Returns:
            List of scored parents sorted by cosine score descending
        """
        start_time = time.perf_counter()

        # Collect parents with embeddings
        parent_ids = list(parents_dict.keys())
        embeddings = []
        valid_indices = []

        for i, pid in enumerate(parent_ids):
            emb = parents_dict[pid].get("embedding")
            if emb:
                embeddings.append(emb)
                valid_indices.append(i)

        if not embeddings:
            return []

        # Batch cosine computation
        scores = cosine_similarity_batch(query_embedding, embeddings)

        # Build results
        scored_parents: list[dict] = []
        children_counts: list[int] = []

        for idx, score in zip(valid_indices, scores):
            pid = parent_ids[idx]
            data = parents_dict[pid]
            count = len(data["children_scores"])
            children_counts.append(count)

            scored_parents.append(
                {
                    "parent_id": pid,
                    "key": data["key"],
                    "title": data["title"],
                    "summary": data["summary"],
                    "level": data["level"],
                    "node_type": data["node_type"],
                    "aggregated_score": float(score),
                    "children_count": count,
                    "scoring_method": "parent_cosine",
                }
            )

        scored_parents.sort(key=lambda x: x["aggregated_score"], reverse=True)

        # Update metrics
        self._update_scoring_metrics(scored_parents, children_counts, start_time)

        return scored_parents

    def _score_rrf_parent_avglog(
        self,
        query_embedding: list[float],
        parents_dict: dict[str, dict],
    ) -> list[dict]:
        """Score parents using RRF of parent_cosine and avg_log.

        Formula: 1/(k + rank_parent_cosine) + 1/(k + rank_avg_log)

        Args:
            query_embedding: Query embedding vector
            parents_dict: Dict from parent discovery

        Returns:
            List of scored parents sorted by RRF score descending
        """
        start_time = time.perf_counter()

        parent_ids = list(parents_dict.keys())
        n = len(parent_ids)

        if n == 0:
            return []

        # Signal 1: Parent cosine scores
        embeddings = [parents_dict[pid].get("embedding", []) for pid in parent_ids]
        valid_emb_mask = [bool(e) for e in embeddings]

        parent_cosine_scores = np.zeros(n, dtype=np.float32)
        if any(valid_emb_mask):
            valid_embs = [e for e, v in zip(embeddings, valid_emb_mask) if v]
            valid_indices = [i for i, v in enumerate(valid_emb_mask) if v]
            valid_scores = cosine_similarity_batch(query_embedding, valid_embs)
            for idx, score in zip(valid_indices, valid_scores):
                parent_cosine_scores[idx] = score

        # Signal 2: AVG * ln(1+count) scores
        avg_log_scores = np.zeros(n, dtype=np.float32)
        children_counts: list[int] = []

        for i, pid in enumerate(parent_ids):
            children_scores = parents_dict[pid]["children_scores"]
            count = len(children_scores)
            children_counts.append(count)
            if count > 0:
                avg_score = sum(children_scores) / count
                avg_log_scores[i] = avg_score * math.log(1 + count)

        # RRF fusion
        rrf_scores = rrf_fusion_2way(parent_cosine_scores, avg_log_scores, k=RRF_K)

        # Build results
        scored_parents: list[dict] = []
        for i, pid in enumerate(parent_ids):
            data = parents_dict[pid]
            scored_parents.append(
                {
                    "parent_id": pid,
                    "key": data["key"],
                    "title": data["title"],
                    "summary": data["summary"],
                    "level": data["level"],
                    "node_type": data["node_type"],
                    "aggregated_score": float(rrf_scores[i]),
                    "parent_cosine": float(parent_cosine_scores[i]),
                    "avg_log": float(avg_log_scores[i]),
                    "children_count": children_counts[i],
                    "scoring_method": "rrf_parent_avglog",
                }
            )

        scored_parents.sort(key=lambda x: x["aggregated_score"], reverse=True)

        # Update metrics
        self._update_scoring_metrics(scored_parents, children_counts, start_time)

        return scored_parents

    def _update_scoring_metrics(
        self,
        scored_parents: list[dict],
        children_counts: list[int],
        start_time: float,
    ) -> None:
        """Update metrics after scoring.

        Args:
            scored_parents: List of scored parent dicts
            children_counts: List of children counts per parent
            start_time: Perf counter start time
        """
        if children_counts:
            self.metrics.avg_children_per_parent = sum(children_counts) / len(children_counts)
            self.metrics.max_children_per_parent = max(children_counts)
            self.metrics.min_children_per_parent = min(children_counts)

        # Level distribution
        level_dist: dict[int, int] = {}
        for p in scored_parents:
            level = p["level"] or 0
            level_dist[level] = level_dist.get(level, 0) + 1
        self.metrics.parent_levels_distribution = level_dist

        self.metrics.latency_scoring_ms = (time.perf_counter() - start_time) * 1000

        self._log.debug(
            "Parent scoring done",
            scored_count=len(scored_parents),
            avg_children=f"{self.metrics.avg_children_per_parent:.1f}",
        )

    # =========================================================================
    # Stage 4: Select & Return
    # =========================================================================

    def _select_and_return(
        self,
        scored_parents: list[dict],
        top_k: int,
    ) -> list[ContentWithContext]:
        """Stage 4: Select top-K parents and return as ContentWithContext.

        Args:
            scored_parents: Scored and sorted parents
            top_k: Number of parents to return

        Returns:
            List of ContentWithContext with parent summaries
        """
        selected = scored_parents[:top_k]

        results: list[ContentWithContext] = []
        with_summary = 0
        with_title_fallback = 0

        for parent in selected:
            # Determine text: prefer summary, fallback to title
            summary = parent.get("summary")
            title = parent.get("title", "")

            if summary and str(summary).strip():
                text = summary
                with_summary += 1
            else:
                text = title or ""
                with_title_fallback += 1

            # Create ContentWithContext
            # Note: node_type="structure" indicates this is a parent summary
            results.append(
                ContentWithContext(
                    node_id=parent["key"],
                    text=text,
                    score=parent["aggregated_score"],
                    node_type="structure",  # NOT "paragraph" — this is parent summary
                    found_via_relation=None,  # E3 doesn't use relations
                )
            )

        # Update metrics
        self.metrics.parents_returned = len(results)
        self.metrics.parents_with_summary = with_summary
        self.metrics.parents_with_title_fallback = with_title_fallback

        return results
