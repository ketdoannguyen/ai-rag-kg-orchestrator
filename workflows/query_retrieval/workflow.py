"""Query retrieval workflow orchestrator.

Coordinates the full retrieval pipeline:
1. Embed query (if not pre-computed)
2. Filter relation candidates (multi-signal scoring)
3. Route query (LLM classification + relation selection)
4. Execute retrieval strategy
5. Traverse relations (parallel per relation type)
6. Return results
"""

from __future__ import annotations

import asyncio
import time
from concurrent.futures import ThreadPoolExecutor
from typing import TYPE_CHECKING

from loguru import logger

from .models import (
    ContentWithContext,
    QueryRetrievalInput,
    QueryRetrievalOutput,
    RelationCandidate,
    RetrievalStrategy,
)
from .relation_scorer import (
    QUERY_TYPE_BASE_K,  # noqa: F401 — re-exported for external consumers
    RelationScorer,
    compute_adaptive_k,
)
from .router import QueryRouter

if TYPE_CHECKING:
    from arango.database import StandardDatabase
    from llm_gateway_client.client import AIGatewayClient


class QueryRetrievalWorkflow:
    """Orchestrates query retrieval pipeline.

    Attributes:
        db: ArangoDB database connection
        llm: LLM gateway client
        graph_name: Name of the knowledge graph
        router: Query router for classification
        max_traversal_nodes: Maximum nodes to use as starting points for traversal
    """

    # Class-level executor for async I/O operations
    _executor = ThreadPoolExecutor(max_workers=4)

    def __init__(
        self,
        db: StandardDatabase,
        llm_client: AIGatewayClient,
        graph_name: str = "knowledge_graph",
        max_traversal_nodes: int = 5,
    ):
        """Initialize the workflow.

        Args:
            db: ArangoDB database connection
            llm_client: LLM gateway client
            graph_name: Name of the knowledge graph
            max_traversal_nodes: Maximum nodes for graph traversal starting points
        """
        self.db = db
        self.llm = llm_client
        self.graph_name = graph_name
        self.max_traversal_nodes = max_traversal_nodes
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

        # Step 2: Pre-classify query to get initial query_type for filtering
        preclassified_type = self.router._keyword_preclassify(input_data.query)
        initial_query_type = preclassified_type.value if preclassified_type else "factual"

        # Step 3: Filter relation candidates WITH MULTI-SIGNAL + ADAPTIVE K
        filter_start = time.time()
        candidates = await self._filter_relation_candidates(
            query=input_data.query,
            query_embedding=query_embedding,
            query_type=initial_query_type,
            top_k=input_data.top_k_relations,
        )
        relation_filter_time_ms = (time.time() - filter_start) * 1000

        # Step 4: Route query (LLM)
        route_start = time.time()
        classification = await self.router.route(
            query=input_data.query,
            candidate_relations=candidates,
        )
        llm_routing_time_ms = (time.time() - route_start) * 1000

        # Step 5: Build result
        result = QueryRetrievalOutput(
            query=input_data.query,
            classification=classification,
            embedding_time_ms=embedding_time_ms,
            relation_filter_time_ms=relation_filter_time_ms,
            llm_routing_time_ms=llm_routing_time_ms,
        )

        # Step 6: Execute retrieval strategy
        retrieval_start = time.time()
        await self._execute_strategy(
            result=result,
            query_embedding=query_embedding,
            top_k=input_data.top_k,
            source_id=input_data.source_id,
        )

        # Step 8: Traverse relations if needed
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
        from llm_gateway_client.models.embedding import EmbeddingRequest

        request = EmbeddingRequest(input=[query])
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
    ) -> list[RelationCandidate]:
        """Filter relations using multi-signal scoring with adaptive K.

        Uses 3 signals:
        1. Embedding similarity (cosine)
        2. Family affinity (query_type → relation families)
        3. Keyword matching

        Args:
            query: User query text
            query_embedding: Query embedding
            query_type: Primary query type (from pre-classification)
            top_k: Fixed number of relations (overrides adaptive K if provided)

        Returns:
            Top-K relation candidates sorted by multi-signal score
        """
        from arango_client.handlers import OntologyHandler

        handler = OntologyHandler(self.db)

        # Get relations with embeddings
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

        # Determine K using adaptive algorithm
        if top_k is not None:
            k = top_k
        else:
            scores = [s[1] for s in scored]  # final_score is at index 1
            k = compute_adaptive_k(scores, query_type)

        # Log top scores for debugging
        self._log.debug(
            "Relation filtering complete",
            top_3=[(s[0], f"{s[1]:.3f}") for s in scored[:3]],
            selected_k=k,
        )

        # Get descriptions from relations for RelationCandidate
        relation_map = {getattr(r, "id", ""): r for r in relations}

        # Convert to RelationCandidate with description
        return [
            RelationCandidate(
                relation_type=rel_id,
                description=getattr(relation_map.get(rel_id), "description", "") or "",
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
        strategy = result.classification.strategy

        if strategy == RetrievalStrategy.TOP_DOWN:
            # E1: Top-Down Retrieval — structure search → content descent
            from workflows.query_retrieval.strategies import TopDownRetrieval

            top_down = TopDownRetrieval(db=self.db)
            content_results = await top_down.execute(
                query_embedding=query_embedding,
                top_k=top_k,
                source_id=source_id,
            )
            result.content_results = content_results
            result.top_down_metrics = top_down.metrics.to_dict()

        else:
            # Fallback: use TOP_DOWN for all unimplemented strategies
            self._log.warning(
                "Strategy not yet implemented, falling back to TOP_DOWN",
                requested=strategy.value,
            )
            from workflows.query_retrieval.strategies import TopDownRetrieval

            top_down = TopDownRetrieval(db=self.db)
            content_results = await top_down.execute(
                query_embedding=query_embedding,
                top_k=top_k,
                source_id=source_id,
            )
            result.content_results = content_results
            result.top_down_metrics = top_down.metrics.to_dict()

    async def _traverse_relations(
        self,
        result: QueryRetrievalOutput,
        relations: list[str],
    ) -> None:
        """Traverse graph with selected relations."""
        if not result.content_results or not relations:
            return

        # Get initial node IDs (limit to max_traversal_nodes)
        initial_ids = [
            f"content_nodes/{content_result.node_id}"
            for content_result in result.content_results[: self.max_traversal_nodes]
            if content_result.node_id
        ]

        if not initial_ids:
            return

        # Traverse all relations at once
        aql_query = """
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
            cursor = await asyncio.get_event_loop().run_in_executor(
                self._executor,
                lambda: self.db.aql.execute(
                    aql_query,
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
                        text=(node.get("text", "") or node.get("attributes", {}).get("text", "")),
                        score=0.0,
                        node_type=node.get("node_type", "paragraph"),
                        found_via_relation=rel_type,
                    )
                )
        except Exception as e:
            self._log.warning("Relation traversal failed", error=str(e))
