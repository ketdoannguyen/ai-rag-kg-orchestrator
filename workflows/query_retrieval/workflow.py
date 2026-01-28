"""Query retrieval workflow orchestrator.

Coordinates the full retrieval pipeline:
1. Embed query (if not pre-computed)
2. Filter relation candidates (multi-signal scoring)
3. Route query (LLM classification + relation selection)
4. Execute retrieval strategy (parallel where applicable)
5. Traverse relations (parallel per relation type)
6. Return results
"""

from __future__ import annotations

import asyncio
import time
from concurrent.futures import ThreadPoolExecutor
from typing import TYPE_CHECKING

import numpy as np
from loguru import logger

from config.ontologies.families import DEFAULT_AFFINITY, get_affinity

from .models import (
    ContentWithContext,
    QueryRetrievalInput,
    QueryRetrievalOutput,
    RelationCandidate,
    RetrievalStrategy,
)
from .router import QueryRouter

if TYPE_CHECKING:
    from arango.database import StandardDatabase
    from llm_gateway_client.client import AIGatewayClient

# ═══════════════════════════════════════════════════════════════════════════════
# CONSTANTS
# ═══════════════════════════════════════════════════════════════════════════════

# Query-type-based K defaults
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

# Retrieval constants
CONTENT_RELATIONS_MIN_K = 1  # Minimum K when dividing top_k
MAX_INITIAL_NODES_FOR_TRAVERSAL = 5  # Limit starting nodes for graph traversal

# Thread pool for blocking I/O operations
_executor = ThreadPoolExecutor(max_workers=4)


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
    """

    # Default weights (must sum to 1.0)
    ALPHA = 0.4  # Embedding weight
    BETA = 0.4  # Affinity weight
    GAMMA = 0.2  # Keyword weight

    def __init__(
        self,
        alpha: float = ALPHA,
        beta: float = BETA,
        gamma: float = GAMMA,
    ):
        """Initialize scorer with weights.

        Args:
            alpha: Embedding signal weight
            beta: Affinity signal weight
            gamma: Keyword signal weight

        Raises:
            ValueError: If weights don't sum to 1.0
        """
        total = alpha + beta + gamma
        if not (0.99 <= total <= 1.01):
            raise ValueError(f"Weights must sum to 1.0, got {total}")
        self.alpha = alpha
        self.beta = beta
        self.gamma = gamma

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

        # ═══════════════════════════════════════════════════════════════════
        # SIGNAL 1: EMBEDDING SCORES (Fully Vectorized)
        # ═══════════════════════════════════════════════════════════════════

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

                    # Assign back to original indices (vectorized)
                    for idx, sim in zip(valid_indices, similarities):
                        embedding_scores[idx] = sim

        # ═══════════════════════════════════════════════════════════════════
        # SIGNAL 2: AFFINITY SCORES (Fast dictionary lookup)
        # ═══════════════════════════════════════════════════════════════════

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

        # ═══════════════════════════════════════════════════════════════════
        # SIGNAL 3: KEYWORD SCORES (String matching)
        # ═══════════════════════════════════════════════════════════════════

        keyword_scores = np.zeros(n)
        query_text_lower = query.lower() if query else ""

        for i, rel in enumerate(relations):
            keywords = getattr(rel, "keywords", None) or []
            # Filter empty keywords
            valid_keywords = [kw for kw in keywords if kw and kw.strip()]
            if valid_keywords:
                matched = sum(1 for kw in valid_keywords if kw.lower() in query_text_lower)
                keyword_scores[i] = matched / len(valid_keywords)

        # ═══════════════════════════════════════════════════════════════════
        # FINAL SCORES (Vectorized)
        # ═══════════════════════════════════════════════════════════════════

        final_scores = (
            self.alpha * embedding_scores
            + self.beta * affinity_scores
            + self.gamma * keyword_scores
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


class QueryRetrievalWorkflow:
    """Orchestrates query retrieval pipeline."""

    def __init__(
        self,
        db: StandardDatabase,
        llm_client: AIGatewayClient,
        graph_name: str = "knowledge_graph",
    ):
        self.db = db
        self.llm = llm_client
        self.graph_name = graph_name
        self.router = QueryRouter(llm_client)
        self._log = logger.bind(workflow="query_retrieval")

    async def run(self, input_data: QueryRetrievalInput) -> QueryRetrievalOutput:
        """Run the full retrieval pipeline.

        Args:
            input_data: Query retrieval input

        Returns:
            QueryRetrievalOutput with results
        """
        total_start = time.time()

        # Step 1: Embed query if needed
        embed_start = time.time()
        query_embedding = input_data.query_embedding
        if query_embedding is None:
            query_embedding = await self._embed_query(input_data.query)
        embedding_time_ms = (time.time() - embed_start) * 1000

        # Step 2: Filter relation candidates WITH MULTI-SIGNAL + ADAPTIVE K
        filter_start = time.time()
        candidates = await self._filter_relation_candidates(
            query=input_data.query,
            query_embedding=query_embedding,
            query_type="factual",  # Default for initial filtering
            top_k=input_data.top_k_relations,  # None = use adaptive K
            tier=input_data.tier,
            use_adaptive_k=True,
        )
        relation_filter_time_ms = (time.time() - filter_start) * 1000

        # Step 3: Route query (LLM)
        route_start = time.time()
        classification = await self.router.route(
            query=input_data.query,
            candidate_relations=candidates,
        )
        llm_routing_time_ms = (time.time() - route_start) * 1000

        # Step 4: Build result
        result = QueryRetrievalOutput(
            query=input_data.query,
            classification=classification,
            embedding_time_ms=embedding_time_ms,
            relation_filter_time_ms=relation_filter_time_ms,
            llm_routing_time_ms=llm_routing_time_ms,
        )

        # Step 5: Check if retrieval needed
        if classification.strategy == RetrievalStrategy.NO_RETRIEVAL:
            result.total_time_ms = (time.time() - total_start) * 1000
            return result

        # Step 6: Execute retrieval strategy
        retrieval_start = time.time()
        await self._execute_strategy(
            result=result,
            query_embedding=query_embedding,
            top_k=input_data.top_k,
            source_id=input_data.source_id,
        )

        # Step 7: Traverse relations if needed
        if classification.use_relations and classification.selected_relations:
            await self._traverse_relations(
                result=result,
                relations=classification.selected_relations,
            )

        result.retrieval_time_ms = (time.time() - retrieval_start) * 1000
        result.total_time_ms = (time.time() - total_start) * 1000
        result.total_results = len(result.content_results) + sum(
            len(nodes) for nodes in result.related_by_relation.values()
        )

        self._log.info(
            "Query retrieval completed",
            query_type=classification.query_type.value,
            total_results=result.total_results,
            total_time_ms=f"{result.total_time_ms:.0f}",
        )

        return result

    async def _embed_query(self, query: str) -> list[float]:
        """Embed query using LLM client."""
        from llm_gateway_client.models.request import GatewayEmbedRequest

        request = GatewayEmbedRequest(texts=[query])
        response = await self.llm.embed(request)

        if response.embeddings and len(response.embeddings) > 0:
            return response.embeddings[0]
        raise ValueError("Failed to embed query")

    async def _filter_relation_candidates(
        self,
        query: str,
        query_embedding: list[float],
        query_type: str,
        top_k: int | None,
        tier: str | None,
        use_adaptive_k: bool = True,
    ) -> list[RelationCandidate]:
        """Filter relations using multi-signal scoring with adaptive K.

        Uses 3 signals:
        1. Embedding similarity (cosine)
        2. Family affinity (query_type → relation families)
        3. Keyword matching

        Args:
            query: User query text
            query_embedding: Query embedding
            query_type: Primary query type
            top_k: Fixed number of relations (overrides adaptive K if provided)
            tier: Filter by tier (optional)
            use_adaptive_k: Whether to use adaptive K (default: True)

        Returns:
            Top-K relation candidates sorted by multi-signal score
        """
        from arango_client.handlers import OntologyHandler

        handler = OntologyHandler(self.db)

        # Get relations with embeddings
        if tier:
            relations = handler.get_by_tier_with_embeddings(tier)
        else:
            relations = handler.get_all_with_embeddings()

        if not relations:
            return []

        # Multi-signal scoring
        scorer = RelationScorer()
        scored = scorer.score_all_vectorized(
            query=query,
            query_embedding=query_embedding,
            query_type=query_type,
            relations=relations,
        )

        if not scored:
            return []

        # Determine K
        if top_k is not None:
            k = top_k
        elif use_adaptive_k:
            scores = [s[1] for s in scored]  # final_score is at index 1
            k = compute_adaptive_k(scores, query_type)
        else:
            k = QUERY_TYPE_BASE_K.get(query_type.lower(), 7)

        # Log top scores for debugging
        self._log.debug(
            "Relation filtering complete",
            top_3=[(s[0], f"{s[1]:.3f}") for s in scored[:3]],
            selected_k=k,
            adaptive=use_adaptive_k,
        )

        # Convert to RelationCandidate
        return [
            RelationCandidate(
                relation_type=rel_id,
                description="",  # Will be filled from ontology if needed
                similarity=final_score,
            )
            for rel_id, final_score, _, _, _ in scored[:k]
        ]

    async def _execute_strategy(
        self,
        result: QueryRetrievalOutput,
        query_embedding: list[float],
        top_k: int,
        source_id: str | None,
    ) -> None:
        """Execute retrieval strategy."""
        from arango_client.handlers import SearchHandler

        search_handler = SearchHandler(self.db)
        strategy = result.classification.strategy

        # Ensure minimum K for split strategies
        half_k = max(top_k // 2, CONTENT_RELATIONS_MIN_K)

        if strategy == RetrievalStrategy.CONTENT_ONLY:
            await self._search_content(result, search_handler, query_embedding, top_k, source_id)

        elif strategy == RetrievalStrategy.CONTENT_WITH_RELATIONS:
            await self._search_content(result, search_handler, query_embedding, half_k, source_id)

        elif strategy == RetrievalStrategy.STRUCTURE_FIRST:
            await self._search_structure(result, search_handler, query_embedding, top_k, source_id)

        elif strategy == RetrievalStrategy.MULTI_SOURCE_AGGREGATE:
            await self._search_multi_source(
                result, search_handler, query_embedding, half_k, source_id
            )

    async def _search_content(
        self,
        result: QueryRetrievalOutput,
        search_handler,
        query_embedding: list[float],
        top_k: int,
        source_id: str | None,
    ) -> None:
        """Content search with async wrapper for blocking I/O."""
        try:
            # Run blocking DB call in thread pool for true async
            search_results = await asyncio.get_event_loop().run_in_executor(
                _executor,
                lambda: search_handler.vector_search_content(
                    query_vector=query_embedding,
                    limit=top_k,
                    source_id=source_id,
                ),
            )

            for sr in search_results:
                result.content_results.append(
                    ContentWithContext(
                        node_id=sr.node.id or "",
                        text=sr.node.attributes.get("text", "") if sr.node.attributes else "",
                        score=sr.score,
                        node_type=sr.node.node_type.value if sr.node.node_type else "paragraph",
                    )
                )
        except Exception as e:
            self._log.warning("Content search failed", error=str(e))

    async def _search_structure(
        self,
        result: QueryRetrievalOutput,
        search_handler,
        query_embedding: list[float],
        top_k: int,
        source_id: str | None,
    ) -> None:
        """Structure search with async wrapper for blocking I/O."""
        try:
            # Run blocking DB call in thread pool for true async
            search_results = await asyncio.get_event_loop().run_in_executor(
                _executor,
                lambda: search_handler.vector_search_structure(
                    query_vector=query_embedding,
                    limit=top_k,
                    source_id=source_id,
                ),
            )

            for sr in search_results:
                result.structure_results.append(
                    {
                        "node_id": sr.node.id,
                        "title": sr.node.attributes.get("title", "") if sr.node.attributes else "",
                        "score": sr.score,
                    }
                )
        except Exception as e:
            self._log.warning("Structure search failed", error=str(e))

    async def _search_multi_source(
        self,
        result: QueryRetrievalOutput,
        search_handler,
        query_embedding: list[float],
        top_k: int,
        source_id: str | None,
    ) -> None:
        """Multi-source aggregate search with TRUE PARALLEL execution."""
        # Run content and structure searches in PARALLEL (now truly async)
        await asyncio.gather(
            self._search_content(result, search_handler, query_embedding, top_k, source_id),
            self._search_structure(result, search_handler, query_embedding, top_k, source_id),
        )

    async def _traverse_relations(
        self,
        result: QueryRetrievalOutput,
        relations: list[str],
    ) -> None:
        """Traverse graph with selected relations."""
        if not result.content_results or not relations:
            return

        # Get initial node IDs (limit to MAX_INITIAL_NODES_FOR_TRAVERSAL)
        initial_ids = [
            f"content_nodes/{c.node_id}"
            for c in result.content_results[:MAX_INITIAL_NODES_FOR_TRAVERSAL]
            if c.node_id
        ]

        if not initial_ids:
            return

        # Traverse all relations at once
        query = """
        FOR start_id IN @node_ids
            FOR v, e IN 1..1 ANY start_id GRAPH @graph_name
                FILTER e.type IN @relations
                RETURN DISTINCT {
                    node: v,
                    relation: e.type,
                    from_node: start_id
                }
        """

        try:
            # Run blocking DB call in thread pool
            cursor = await asyncio.get_event_loop().run_in_executor(
                _executor,
                lambda: self.db.aql.execute(
                    query,
                    bind_vars={
                        "node_ids": initial_ids,
                        "graph_name": self.graph_name,
                        "relations": relations,
                    },
                ),
            )

            # Group by relation
            for doc in cursor:
                rel_type = doc["relation"]
                if rel_type not in result.related_by_relation:
                    result.related_by_relation[rel_type] = []

                node = doc["node"]
                result.related_by_relation[rel_type].append(
                    ContentWithContext(
                        node_id=node.get("_key", ""),
                        text=node.get("text", "") or node.get("attributes", {}).get("text", ""),
                        score=0.0,
                        node_type=node.get("node_type", "paragraph"),
                        found_via_relation=rel_type,
                    )
                )
        except Exception as e:
            self._log.warning("Relation traversal failed", error=str(e))
