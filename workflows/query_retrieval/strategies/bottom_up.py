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

from loguru import logger

from workflows.query_retrieval.models import ContentWithContext
from workflows.query_retrieval.strategies.base import get_executor

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
        parent_node_type: parent.node_type
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
    ) -> list[ContentWithContext]:
        """Execute E3 Bottom-Up Retrieval pipeline.

        Args:
            query_embedding: Query embedding vector (3072 dims)
            top_k: Number of parent summaries to return
            multiplier: Over-retrieval multiplier M

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

        # Stage 3: Score parents
        scored_parents = self._score_parents(parents_dict)

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
            raw_results: Raw AQL results with content→parent mappings

        Returns:
            Dict: parent_id → {key, title, summary, level, node_type, children_scores}
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
                    "children_scores": [],
                }

            parents[parent_id]["children_scores"].append(row["content_score"])

        return parents

    # =========================================================================
    # Stage 3: Parent Scoring
    # =========================================================================

    def _score_parents(
        self,
        parents_dict: dict[str, dict],
    ) -> list[dict]:
        """Stage 3: Score parents using AVG * ln(1+count) formula.

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
                }
            )

            children_counts.append(count)

        # Sort by score descending
        scored_parents.sort(key=lambda x: x["aggregated_score"], reverse=True)

        # Update metrics
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

        return scored_parents

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
