"""E2: Horizontal Retrieval — traverse Tier 2 relations from seed content nodes.

This strategy finds logically related content by:
1. SEED: Vector search for top-N content nodes (shared with B1 baseline)
2. FILTER: Score Tier 2 relations via RelationScorer, select top-K
3. TRAVERSE: 1-hop ANY traversal from seeds through selected relations
4. SCORE: RRF fusion of cosine_score and seed_score (Cormack et al., SIGIR 2009)
5. MERGE: Deduplicate, rank, return top-K final

Core Insight:
    B1 finds nodes SIMILAR to query (high cosine).
    E2 finds nodes RELATED to seeds via semantic relations
    (CAUSED_BY, SUPPORTS, CONTRASTS, etc.) that vector search misses.

Scoring:
    Uses Reciprocal Rank Fusion (RRF) instead of multiplicative scoring.
    RRF is scale-invariant and requires no weight tuning.
"""

from __future__ import annotations

import asyncio
import time
from dataclasses import dataclass, field
from typing import TYPE_CHECKING

import numpy as np
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

# RRF smoothing constant (Cormack et al., SIGIR 2009)
# k=60 is the literature standard, provides robust rank fusion
RRF_K = 60

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
    # Stage 4: Multi-Signal Scoring with RRF
    # =========================================================================

    def _score_traversed(
        self,
        query_embedding: list[float],
        traversed_nodes: list[dict],
        seed_scores: dict[str, float],
        relation_scores: dict[str, float],
    ) -> list[dict]:
        """Compute RRF fusion of cosine_score and seed_score.

        Uses Reciprocal Rank Fusion (Cormack et al., SIGIR 2009) instead of
        multiplicative scoring. RRF is scale-invariant and requires no
        arbitrary weight tuning.

        Formula: RRF(node) = 1/(k + rank_cosine) + 1/(k + rank_seed)

        Args:
            query_embedding: Query embedding vector
            traversed_nodes: Nodes found via graph traversal
            seed_scores: Mapping seed_id → cosine score
            relation_scores: Mapping relation_type → scorer score (kept for metadata)

        Returns:
            List of traversed node dicts with final_score (RRF) attached
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
        cosine_scores_arr = cosine_similarity_batch(query_embedding, embeddings)

        # Extract seed scores for each node
        seed_scores_arr = np.array(
            [seed_scores.get(n.get("from_seed_id", ""), 0.0) for n in valid_nodes],
            dtype=np.float32,
        )

        # Apply RRF fusion
        rrf_scores = self._rrf_fusion_two_signals(cosine_scores_arr, seed_scores_arr)

        # Build result list
        scored = []
        for i, node in enumerate(valid_nodes):
            cosine_score = float(cosine_scores_arr[i])
            seed_score = float(seed_scores_arr[i])
            rel_score = relation_scores.get(node.get("relation_type", ""), 0.0)
            final_score = float(rrf_scores[i])

            scored.append(
                {
                    **node,
                    "cosine_score": cosine_score,
                    "seed_score": seed_score,
                    "relation_score": rel_score,  # Kept for debugging/metrics
                    "final_score": final_score,
                    "source": "traversal",
                }
            )

        self.metrics.latency_score_ms = (time.perf_counter() - t0) * 1000
        return scored

    def _rrf_fusion_two_signals(
        self,
        cosine_scores: np.ndarray,
        seed_scores: np.ndarray,
        k: int = RRF_K,
    ) -> np.ndarray:
        """Apply Reciprocal Rank Fusion to cosine and seed scores.

        RRF combines multiple ranking signals using ranks instead of raw scores:
            RRF(d) = sum_i(1 / (k + rank_i(d)))

        Properties:
        - Scale-invariant: works regardless of score ranges
        - Robust: outlier scores don't dominate
        - Simple: no weight tuning required (only k=60, literature standard)

        Tie handling: Uses average rank for ties (e.g., nodes from same seed).

        Args:
            cosine_scores: Cosine similarity scores array
            seed_scores: Seed node scores array
            k: RRF smoothing constant (default: 60, from Cormack et al. 2009)

        Returns:
            RRF scores array (higher = better combined rank)
        """
        n = len(cosine_scores)

        if n == 0:
            return np.array([], dtype=np.float32)

        if n == 1:
            # Single item: max possible RRF score for 2 signals
            return np.array([2.0 / (k + 1)], dtype=np.float32)

        # Compute cosine ranks (1-indexed, higher score = lower rank number)
        cosine_order = np.argsort(-cosine_scores)
        cosine_ranks = np.zeros(n, dtype=np.float32)
        for rank, idx in enumerate(cosine_order, start=1):
            cosine_ranks[idx] = rank

        # Compute seed ranks with tie handling (average rank for ties)
        seed_ranks = self._compute_ranks_with_ties(seed_scores)

        # RRF formula: 1/(k + rank)
        rrf_scores = (1.0 / (k + cosine_ranks) + 1.0 / (k + seed_ranks)).astype(
            np.float32
        )

        return rrf_scores

    def _compute_ranks_with_ties(self, scores: np.ndarray) -> np.ndarray:
        """Compute ranks with average rank for ties.

        When multiple items have the same score, they all get the average
        of the ranks they would have occupied.

        Example:
            scores = [0.8, 0.6, 0.8, 0.6]
            ranks  = [1.5, 3.5, 1.5, 3.5]  # 0.8s share ranks 1-2, 0.6s share 3-4

        Args:
            scores: Array of scores (higher = better)

        Returns:
            Array of ranks (1-indexed, lower = better)
        """
        n = len(scores)
        ranks = np.zeros(n, dtype=np.float32)

        # Sort indices by descending score
        order = np.argsort(-scores)
        sorted_scores = scores[order]

        i = 0
        while i < n:
            # Find all items with same score
            j = i
            while j < n and sorted_scores[j] == sorted_scores[i]:
                j += 1

            # Average rank for this tie group (1-indexed)
            avg_rank = (i + 1 + j) / 2.0

            # Assign average rank to all items in tie group
            for idx in range(i, j):
                ranks[order[idx]] = avg_rank

            i = j

        return ranks

    # =========================================================================
    # Stage 5: Merge, Dedup, Rank with Unified RRF
    # =========================================================================

    def _merge_dedup_rank(
        self,
        seed_nodes: list[dict],
        scored_traversed: list[dict],
        top_k: int,
    ) -> list[ContentWithContext]:
        """Merge seeds + traversed with unified RRF scoring.

        Seeds and traversed nodes are scored together using RRF to ensure
        fair comparison. For seeds, seed_score equals their own cosine score
        (best possible seed provenance).

        Args:
            seed_nodes: Seed nodes with original cosine scores
            scored_traversed: Traversed nodes with cosine and seed scores
            top_k: Number of final results to return

        Returns:
            Top-K ContentWithContext results sorted by RRF score
        """
        t0 = time.perf_counter()

        # Prepare seed entries
        # For seeds: seed_score = their own cosine (they are their own source)
        seed_entries = []
        for seed in seed_nodes:
            seed_entries.append(
                {
                    **seed,
                    "cosine_score": seed["score"],
                    "seed_score": seed["score"],  # Seed is its own source
                    "source": "seed",
                }
            )

        # Prepare traversed entries (already have cosine_score and seed_score)
        traversed_entries = [
            {**node} for node in scored_traversed
        ]

        # Merge all nodes
        all_nodes = seed_entries + traversed_entries

        if not all_nodes:
            self.metrics.latency_merge_ms = (time.perf_counter() - t0) * 1000
            return []

        # Dedup by _id, keeping entry with higher cosine_score
        best_by_id: dict[str, dict] = {}
        for node in all_nodes:
            node_id = node["_id"]
            if (
                node_id not in best_by_id
                or node.get("cosine_score", 0) > best_by_id[node_id].get("cosine_score", 0)
            ):
                best_by_id[node_id] = node

        deduplicated = list(best_by_id.values())
        n = len(deduplicated)

        if n == 0:
            self.metrics.latency_merge_ms = (time.perf_counter() - t0) * 1000
            return []

        # Apply unified RRF scoring to all deduplicated nodes
        cosine_scores = np.array(
            [node.get("cosine_score", 0.0) for node in deduplicated],
            dtype=np.float32,
        )
        seed_scores = np.array(
            [node.get("seed_score", 0.0) for node in deduplicated],
            dtype=np.float32,
        )

        rrf_scores = self._rrf_fusion_two_signals(cosine_scores, seed_scores)

        # Attach final RRF scores
        for i, node in enumerate(deduplicated):
            node["final_score"] = float(rrf_scores[i])

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
