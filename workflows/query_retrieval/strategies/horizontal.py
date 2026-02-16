"""E2: Horizontal Retrieval — traverse Tier 2 relations from seed content nodes.

This strategy finds logically related content by:
1. SEED: Vector search for top-N content nodes (shared with B1 baseline)
2. FILTER: Score Tier 2 relations via RelationScorer, select top-K
3. TRAVERSE: 1-hop ANY traversal from seeds through selected relations
4. SCORE: Multi-signal scoring (cosine * relation_score * seed_score)
5. MERGE: Deduplicate, rank, return top-K final

Core Insight:
    B1 finds nodes SIMILAR to query (high cosine).
    E2 finds nodes RELATED to seeds via semantic relations
    (CAUSED_BY, SUPPORTS, CONTRASTS, etc.) that vector search misses.
"""

from __future__ import annotations

import asyncio
import time
from dataclasses import dataclass, field
from typing import TYPE_CHECKING

from loguru import logger

from workflows.query_retrieval.models import ContentWithContext
from workflows.query_retrieval.strategies.base import (
    cosine_similarity_batch,
    get_executor,
    to_content_with_context,
)

if TYPE_CHECKING:
    from arango.database import StandardDatabase

# =============================================================================
# Constants
# =============================================================================

DEFAULT_N_SEEDS = 5  # Number of seed nodes from vector search
DEFAULT_K_FINAL = 15  # Final content nodes to return
HOP_DEPTH = 1  # Fixed 1-hop traversal
MAX_TRAVERSED_NODES = 200  # Cap traversal results to prevent explosion

# =============================================================================
# AQL Queries
# =============================================================================

AQL_SEED_RETRIEVAL = """
FOR c IN knowledge_vector_view
    SEARCH ANALYZER(c.node_type IN ["paragraph", "image", "table"], "identity")
    LET score = COSINE_SIMILARITY(c.embedding, @query_embedding)
    FILTER score > 0
    SORT score DESC
    LIMIT @n_seeds
    RETURN {
        _id: c._id,
        _key: c._key,
        title: c.title,
        text: c.text,
        node_type: c.node_type,
        embedding: c.embedding,
        source_id: c.source_id,
        score: score
    }
"""

AQL_HORIZONTAL_TRAVERSE = """
FOR start_id IN @seed_node_ids
    FOR v, e IN 1..1 ANY start_id GRAPH @graph_name
        FILTER e.type IN @selected_relations
        FILTER v._id NOT IN @seed_node_ids
        RETURN DISTINCT {
            _id: v._id,
            _key: v._key,
            title: v.title,
            text: v.text,
            node_type: v.node_type,
            embedding: v.embedding,
            source_id: v.source_id,
            relation_type: e.type,
            from_seed_id: start_id
        }
"""


# =============================================================================
# Metrics
# =============================================================================


@dataclass
class HorizontalMetrics:
    """Metrics collected during E2 Horizontal Retrieval execution."""

    seed_count: int = 0
    relations_scored: int = 0
    relations_selected: int = 0
    selected_relation_types: list[str] = field(default_factory=list)
    traversed_nodes_raw: int = 0
    traversed_nodes_unique: int = 0
    nodes_returned: int = 0
    unique_nodes_from_traversal: int = 0
    latency_seed_ms: float = 0.0
    latency_relation_ms: float = 0.0
    latency_traverse_ms: float = 0.0
    latency_score_ms: float = 0.0
    latency_merge_ms: float = 0.0
    latency_total_ms: float = 0.0

    def to_dict(self) -> dict:
        """Convert to dict for inclusion in QueryRetrievalOutput."""
        return {
            "seed_count": self.seed_count,
            "relations_scored": self.relations_scored,
            "relations_selected": self.relations_selected,
            "selected_relation_types": self.selected_relation_types,
            "traversed_nodes_raw": self.traversed_nodes_raw,
            "traversed_nodes_unique": self.traversed_nodes_unique,
            "nodes_returned": self.nodes_returned,
            "unique_nodes_from_traversal": self.unique_nodes_from_traversal,
            "latency_seed_ms": round(self.latency_seed_ms, 1),
            "latency_relation_ms": round(self.latency_relation_ms, 1),
            "latency_traverse_ms": round(self.latency_traverse_ms, 1),
            "latency_score_ms": round(self.latency_score_ms, 1),
            "latency_merge_ms": round(self.latency_merge_ms, 1),
            "latency_total_ms": round(self.latency_total_ms, 1),
        }


# =============================================================================
# Horizontal Retrieval Strategy
# =============================================================================


class HorizontalRetrieval:
    """E2: Horizontal Retrieval strategy.

    Finds logically related content nodes by traversing Tier 2 semantic
    relations from seed content nodes found via vector search.

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
        self.metrics = HorizontalMetrics()
        self._log = logger.bind(strategy="horizontal")
        self._executor = get_executor()

    async def execute(
        self,
        query: str,
        query_embedding: list[float],
        query_type: str,
        top_k: int = DEFAULT_K_FINAL,
        n_seeds: int = DEFAULT_N_SEEDS,
    ) -> list[ContentWithContext]:
        """Execute E2 Horizontal Retrieval pipeline.

        Args:
            query: User query text (for RelationScorer keyword matching)
            query_embedding: Query embedding vector
            query_type: Query type (causal, factual, etc.)
            top_k: Number of final content nodes to return
            n_seeds: Number of seed nodes from vector search

        Returns:
            List of ContentWithContext results ranked by relevance
        """
        total_start = time.perf_counter()
        self.metrics = HorizontalMetrics()

        # Stage 1: Seed retrieval
        seed_nodes = await self._get_seed_nodes(query_embedding, n_seeds)
        if not seed_nodes:
            self._log.warning("No seed nodes found, returning empty results")
            self.metrics.latency_total_ms = (time.perf_counter() - total_start) * 1000
            return []

        # Stage 2: Relation filtering
        selected_relations = self._filter_relations(query, query_embedding, query_type)
        if not selected_relations:
            self._log.info("No relations selected, returning seeds as-is")
            self.metrics.latency_total_ms = (time.perf_counter() - total_start) * 1000
            return [to_content_with_context(s, s["score"]) for s in seed_nodes[:top_k]]

        # Build lookup dicts
        relation_scores = {r[0]: r[1] for r in selected_relations}
        selected_types = list(relation_scores.keys())
        seed_scores = {s["_id"]: s["score"] for s in seed_nodes}

        # Stage 3: Graph traversal
        traversed_nodes = await self._traverse(seed_nodes, selected_types)

        # Stage 4: Multi-signal scoring
        scored_traversed = self._score_traversed(
            query_embedding, traversed_nodes, seed_scores, relation_scores
        )

        # Stage 5: Merge, dedup, rank
        results = self._merge_dedup_rank(seed_nodes, scored_traversed, top_k)

        self.metrics.latency_total_ms = (time.perf_counter() - total_start) * 1000

        self._log.info(
            "Horizontal retrieval completed",
            seeds=self.metrics.seed_count,
            relations_selected=self.metrics.relations_selected,
            traversed=self.metrics.traversed_nodes_unique,
            returned=self.metrics.nodes_returned,
            unique_from_traversal=self.metrics.unique_nodes_from_traversal,
            latency_ms=f"{self.metrics.latency_total_ms:.0f}",
        )

        return results

    # =========================================================================
    # Stage 1: Seed Retrieval
    # =========================================================================

    async def _get_seed_nodes(self, query_embedding: list[float], n_seeds: int) -> list[dict]:
        """Vector search for top-N seed content nodes.

        Args:
            query_embedding: Query embedding vector
            n_seeds: Number of seeds to retrieve

        Returns:
            List of seed node dicts with _id, _key, text, embedding, score
        """
        t0 = time.perf_counter()

        cursor = await asyncio.get_event_loop().run_in_executor(
            self._executor,
            lambda: self.db.aql.execute(
                AQL_SEED_RETRIEVAL,
                bind_vars={"query_embedding": query_embedding, "n_seeds": n_seeds},
            ),
        )
        seeds = list(cursor)

        self.metrics.seed_count = len(seeds)
        self.metrics.latency_seed_ms = (time.perf_counter() - t0) * 1000

        self._log.debug("Seed retrieval done", count=len(seeds))
        return seeds

    # =========================================================================
    # Stage 2: Relation Filtering
    # =========================================================================

    def _filter_relations(
        self,
        query: str,
        query_embedding: list[float],
        query_type: str,
        top_k_relations: int = 10,
    ) -> list[tuple[str, float, float, float]]:
        """Score and select top-K Tier 2 relations using HybridRelationScorer.

        Uses BM25 + Embedding hybrid scoring with RRF fusion.
        K is a configurable parameter for evaluation purposes.

        Args:
            query: User query text (for BM25 lexical matching)
            query_embedding: Query embedding vector (for semantic matching)
            query_type: Query type (kept for API compatibility, not used)
            top_k_relations: Number of top relations to select (default: 10)

        Returns:
            Top-K scored relations as (id, rrf_score, bm25_score, emb_score) tuples
        """
        t0 = time.perf_counter()

        # Load relations from ontology
        from arango_client.handlers import OntologyHandler

        handler = OntologyHandler(self.db)
        all_relations = handler.get_all_with_embeddings()

        # Filter to Tier 2 + Shared only
        from arango_client.models.ontology import RelationTier

        tier2_relations = [
            r for r in all_relations if r.applies_to_tier(RelationTier.TIER_2_CONTENT)
        ]

        if not tier2_relations:
            self._log.warning("No Tier 2 relations found in ontology")
            self.metrics.latency_relation_ms = (time.perf_counter() - t0) * 1000
            return []

        # Score all relations using hybrid BM25 + Embedding + RRF
        from workflows.query_retrieval.relation_scorer import HybridRelationScorer

        scorer = HybridRelationScorer()
        scored = scorer.score_all(
            query=query,
            query_embedding=query_embedding,
            relations=tier2_relations,
        )

        if not scored:
            self.metrics.latency_relation_ms = (time.perf_counter() - t0) * 1000
            return []

        # Simple top-K selection (K is evaluation-driven, not arbitrary per-query-type)
        selected = scored[:top_k_relations]

        # Update metrics
        self.metrics.relations_scored = len(scored)
        self.metrics.relations_selected = len(selected)
        self.metrics.selected_relation_types = [r[0] for r in selected]
        self.metrics.latency_relation_ms = (time.perf_counter() - t0) * 1000

        self._log.debug(
            "Relation filtering done",
            scored=len(scored),
            selected=len(selected),
            top_relations=self.metrics.selected_relation_types[:5],
        )

        return selected

    # =========================================================================
    # Stage 3: Graph Traversal
    # =========================================================================

    async def _traverse(self, seed_nodes: list[dict], selected_relations: list[str]) -> list[dict]:
        """1-hop ANY traversal from seeds through selected relations.

        Args:
            seed_nodes: Seed content nodes
            selected_relations: List of relation type names to traverse

        Returns:
            List of traversed content node dicts with relation metadata
        """
        t0 = time.perf_counter()

        if not seed_nodes or not selected_relations:
            self.metrics.latency_traverse_ms = 0.0
            return []

        seed_ids = [s["_id"] for s in seed_nodes]

        try:
            cursor = await asyncio.get_event_loop().run_in_executor(
                self._executor,
                lambda: self.db.aql.execute(
                    AQL_HORIZONTAL_TRAVERSE,
                    bind_vars={
                        "seed_node_ids": seed_ids,
                        "graph_name": self._graph_name,
                        "selected_relations": selected_relations,
                    },
                ),
            )
            raw_results = list(cursor)
        except Exception as e:
            self._log.warning("Graph traversal failed", error=str(e))
            self.metrics.latency_traverse_ms = (time.perf_counter() - t0) * 1000
            return []

        # Filter: only content nodes (paragraph, image, table)
        content_results = [
            r for r in raw_results if r.get("node_type") in ("paragraph", "image", "table")
        ]

        # Cap to prevent explosion
        if len(content_results) > MAX_TRAVERSED_NODES:
            content_results = content_results[:MAX_TRAVERSED_NODES]

        self.metrics.traversed_nodes_raw = len(raw_results)
        self.metrics.traversed_nodes_unique = len(content_results)
        self.metrics.latency_traverse_ms = (time.perf_counter() - t0) * 1000

        self._log.debug(
            "Traversal done",
            raw=len(raw_results),
            content_only=len(content_results),
        )

        return content_results

    # =========================================================================
    # Stage 4: Multi-Signal Scoring
    # =========================================================================

    def _score_traversed(
        self,
        query_embedding: list[float],
        traversed_nodes: list[dict],
        seed_scores: dict[str, float],
        relation_scores: dict[str, float],
    ) -> list[dict]:
        """Compute final_score = cosine * relation_score * seed_score.

        Args:
            query_embedding: Query embedding vector
            traversed_nodes: Nodes found via graph traversal
            seed_scores: Mapping seed_id → cosine score
            relation_scores: Mapping relation_type → scorer score

        Returns:
            List of traversed node dicts with final_score attached
        """
        t0 = time.perf_counter()

        if not traversed_nodes:
            self.metrics.latency_score_ms = 0.0
            return []

        # Separate nodes with/without embeddings
        valid_nodes = [n for n in traversed_nodes if n.get("embedding")]

        if not valid_nodes:
            self.metrics.latency_score_ms = (time.perf_counter() - t0) * 1000
            return []

        # Batch cosine computation
        embeddings = [n["embedding"] for n in valid_nodes]
        cosine_scores = cosine_similarity_batch(query_embedding, embeddings)

        scored = []
        for i, node in enumerate(valid_nodes):
            cosine_score = float(cosine_scores[i])
            rel_score = relation_scores.get(node.get("relation_type", ""), 0.0)
            seed_score = seed_scores.get(node.get("from_seed_id", ""), 0.0)

            final_score = cosine_score * rel_score * seed_score

            scored.append(
                {
                    **node,
                    "cosine_score": cosine_score,
                    "relation_score": rel_score,
                    "seed_score": seed_score,
                    "final_score": final_score,
                    "source": "traversal",
                }
            )

        self.metrics.latency_score_ms = (time.perf_counter() - t0) * 1000
        return scored

    # =========================================================================
    # Stage 5: Merge, Dedup, Rank
    # =========================================================================

    def _merge_dedup_rank(
        self,
        seed_nodes: list[dict],
        scored_traversed: list[dict],
        top_k: int,
    ) -> list[ContentWithContext]:
        """Merge seeds + traversed, dedup by node ID, rank, return top-K.

        Args:
            seed_nodes: Seed nodes with original cosine scores
            scored_traversed: Traversed nodes with multi-signal scores
            top_k: Number of final results to return

        Returns:
            Top-K ContentWithContext results sorted by score
        """
        t0 = time.perf_counter()

        # Prepare seed entries (keep original cosine score)
        seed_entries = []
        for seed in seed_nodes:
            seed_entries.append(
                {
                    **seed,
                    "final_score": seed["score"],
                    "source": "seed",
                }
            )

        # Merge
        all_nodes = seed_entries + scored_traversed

        # Dedup by _id, keeping max score
        best_by_id: dict[str, dict] = {}
        for node in all_nodes:
            node_id = node["_id"]
            if (
                node_id not in best_by_id
                or node["final_score"] > best_by_id[node_id]["final_score"]
            ):
                best_by_id[node_id] = node

        deduplicated = list(best_by_id.values())

        # Sort by final_score descending
        deduplicated.sort(key=lambda x: x["final_score"], reverse=True)

        # Top-K
        top_k_nodes = deduplicated[:top_k]

        # Convert to ContentWithContext
        results = [
            to_content_with_context(
                node,
                node["final_score"],
                found_via_relation=node.get("relation_type")
                if node.get("source") == "traversal"
                else None,
            )
            for node in top_k_nodes
        ]

        # Update metrics
        self.metrics.nodes_returned = len(results)
        self.metrics.unique_nodes_from_traversal = len(
            [n for n in top_k_nodes if n.get("source") == "traversal"]
        )
        self.metrics.latency_merge_ms = (time.perf_counter() - t0) * 1000

        return results
