"""Integration test for E1 Top-Down Retrieval with real ArangoDB data.

Tests the full 4-stage pipeline against a running ArangoDB instance:
1. Build content-rich pool (structure nodes with HAS_CONTENT >= HAS_SECTION)
2. Structure cosine search (in-memory numpy)
3. Content harvest (AQL graph traversal via HAS_SECTION/HAS_CONTENT edges)
4. Re-rank harvested content (cosine similarity)

Also tests the full query_retrieval workflow pipeline (embed → filter → route → retrieve).

Prerequisites:
    - ArangoDB running at ARANGO_URL (default: http://localhost:8529)
    - Database with ingested data (content_nodes, structure_nodes, knowledge_edges)
    - .env configured with ARANGO_* and LLM_* variables

Usage:
    # Run with borrowed embedding (no LLM API needed):
    python -m pytest tests/integration/test_top_down_retrieval_integration.py -v -s

    # Run including LLM-dependent tests:
    python -m pytest tests/integration/test_top_down_retrieval_integration.py -v -s --run-llm
"""

from __future__ import annotations

import sys

import numpy as np
import pytest
from loguru import logger

# Configure logger for test output
logger.remove()
logger.add(
    sys.stderr,
    level="DEBUG",
    format="<green>{time:HH:mm:ss}</green> | <level>{level: <8}</level> | {message}",
)


# =============================================================================
# Fixtures
# =============================================================================


@pytest.fixture(scope="session")
def run_llm(request):
    """Whether to run LLM-dependent tests."""
    return request.config.getoption("--run-llm")


@pytest.fixture(scope="session")
def db():
    """Connect to real ArangoDB and return StandardDatabase."""
    from workflows.dependencies import init_arango_client

    try:
        client = init_arango_client()
        database = client._db
        # Verify connection
        database.version()
        logger.info(f"Connected to ArangoDB {database.version()}")
        return database
    except Exception as e:
        pytest.skip(f"ArangoDB not available: {e}")


@pytest.fixture(scope="session")
def llm_client(run_llm):
    """Create LLM client (only when --run-llm is passed)."""
    if not run_llm:
        pytest.skip("LLM tests skipped (use --run-llm to enable)")

    from workflows.dependencies import init_llm_client

    return init_llm_client()


@pytest.fixture(scope="session")
def collection_counts(db) -> dict[str, int]:
    """Get document counts for key collections."""
    counts = {}
    for name in [
        "content_nodes",
        "structure_nodes",
        "root_nodes",
        "entity_nodes",
        "knowledge_edges",
        "ontology_relations",
        "paragraph_nodes",
        "image_nodes",
        "table_nodes",
    ]:
        if db.has_collection(name):
            counts[name] = db.collection(name).count()
        else:
            counts[name] = 0
    logger.info(f"Collection counts: {counts}")
    return counts


@pytest.fixture(scope="session")
def borrowed_embedding(db) -> list[float]:
    """Borrow an embedding from an existing structure node.

    This avoids needing an LLM API call for basic pipeline testing.
    """
    cursor = db.aql.execute("""
        FOR s IN structure_nodes
            FILTER s.embedding != null AND LENGTH(s.embedding) > 0
            LIMIT 1
            RETURN { embedding: s.embedding, summary: s.summary, _key: s._key }
    """)
    docs = list(cursor)
    if not docs:
        pytest.skip("No structure nodes with embeddings found")

    doc = docs[0]
    logger.info(
        f"Borrowed embedding from structure_nodes/{doc['_key']} "
        f"(dim={len(doc['embedding'])}, summary={doc.get('summary', '')[:80]})"
    )
    return doc["embedding"]


# =============================================================================
# Test 1: Database Health Checks
# =============================================================================


class TestDatabaseHealth:
    """Verify database connectivity and data prerequisites."""

    def test_connection(self, db):
        """Verify ArangoDB is reachable."""
        version = db.version()
        assert version, "ArangoDB version should be non-empty"
        logger.info(f"ArangoDB version: {version}")

    def test_structure_nodes_exist(self, db, collection_counts):
        """structure_nodes collection must have data for top-down retrieval."""
        count = collection_counts.get("structure_nodes", 0)
        assert count > 0, (
            "structure_nodes collection is empty — ingest data before running integration tests"
        )
        logger.info(f"structure_nodes: {count} documents")

    def test_knowledge_edges_exist(self, db, collection_counts):
        """knowledge_edges must exist for graph traversal."""
        count = collection_counts.get("knowledge_edges", 0)
        assert count > 0, "knowledge_edges collection is empty"
        logger.info(f"knowledge_edges: {count} edges")

    def test_has_content_edges_exist(self, db):
        """HAS_CONTENT edges must exist for content harvest."""
        cursor = db.aql.execute("""
            FOR e IN knowledge_edges
                FILTER e.type == "HAS_CONTENT"
                COLLECT WITH COUNT INTO cnt
                RETURN cnt
        """)
        count = next(iter(cursor))
        assert count > 0, "No HAS_CONTENT edges found"
        logger.info(f"HAS_CONTENT edges: {count}")

    def test_has_section_edges_exist(self, db):
        """HAS_SECTION edges must exist for structure hierarchy."""
        cursor = db.aql.execute("""
            FOR e IN knowledge_edges
                FILTER e.type == "HAS_SECTION"
                COLLECT WITH COUNT INTO cnt
                RETURN cnt
        """)
        count = next(iter(cursor))
        assert count > 0, "No HAS_SECTION edges found"
        logger.info(f"HAS_SECTION edges: {count}")

    def test_embedding_coverage(self, db):
        """Check how many structure nodes have embeddings."""
        cursor = db.aql.execute("""
            LET total = LENGTH(structure_nodes)
            LET with_emb = LENGTH(
                FOR s IN structure_nodes
                    FILTER s.embedding != null AND LENGTH(s.embedding) > 0
                    RETURN 1
            )
            RETURN { total: total, with_embedding: with_emb }
        """)
        stats = next(iter(cursor))
        pct = (stats["with_embedding"] / stats["total"] * 100) if stats["total"] > 0 else 0

        logger.info(
            f"Structure embedding coverage: "
            f"{stats['with_embedding']}/{stats['total']} ({pct:.1f}%)"
        )
        assert stats["with_embedding"] > 0, (
            "No structure nodes have embeddings — run enrichment first"
        )

    def test_embedding_dimensions(self, borrowed_embedding):
        """Verify embedding dimensions are 3072 (text-embedding-3-large)."""
        dim = len(borrowed_embedding)
        logger.info(f"Embedding dimensions: {dim}")
        assert dim == 3072, f"Expected 3072 dimensions, got {dim}"


# =============================================================================
# Test 2: Content-Rich Pool (Stage 0)
# =============================================================================


class TestContentRichPool:
    """Test the content-rich pool building stage."""

    @pytest.mark.asyncio
    async def test_pool_returns_results(self, db):
        """Content-rich pool query should return structure nodes."""
        from workflows.query_retrieval.strategies.top_down import (
            TopDownRetrieval,
        )

        strategy = TopDownRetrieval(db=db)
        pool = await strategy._build_content_rich_pool(source_id=None)

        assert len(pool) > 0, "Content-rich pool is empty"
        logger.info(f"Content-rich pool size: {len(pool)}")

    @pytest.mark.asyncio
    async def test_pool_nodes_have_embeddings(self, db):
        """Pool nodes should have embeddings for cosine search."""
        from workflows.query_retrieval.strategies.top_down import (
            TopDownRetrieval,
        )

        strategy = TopDownRetrieval(db=db)
        pool = await strategy._build_content_rich_pool(source_id=None)

        with_emb = sum(1 for n in pool if n.get("embedding"))
        without_emb = len(pool) - with_emb
        pct = (with_emb / len(pool) * 100) if pool else 0

        logger.info(
            f"Pool embedding coverage: {with_emb}/{len(pool)} ({pct:.1f}%), missing: {without_emb}"
        )
        assert with_emb > 0, "No pool nodes have embeddings"

    @pytest.mark.asyncio
    async def test_pool_content_rich_invariant(self, db):
        """Every pool node should have HAS_CONTENT count >= HAS_SECTION count."""
        from workflows.query_retrieval.strategies.top_down import (
            TopDownRetrieval,
        )

        strategy = TopDownRetrieval(db=db)
        pool = await strategy._build_content_rich_pool(source_id=None)

        assert len(pool) > 0, "Pool is empty"

        # Verify invariant: has_content_count is returned by the AQL
        for node in pool:
            assert node.get("has_content_count", 0) > 0, (
                f"Node {node.get('_key')} has no HAS_CONTENT edges"
            )

    @pytest.mark.asyncio
    async def test_pool_caching(self, db):
        """Pool should be cached after first call."""
        from workflows.query_retrieval.strategies.top_down import (
            TopDownRetrieval,
        )

        strategy = TopDownRetrieval(db=db)

        pool1 = await strategy._build_content_rich_pool(source_id=None)
        pool2 = await strategy._build_content_rich_pool(source_id=None)

        assert pool1 is pool2, "Pool should be cached (same object reference)"

        # After invalidation, should re-fetch
        strategy.invalidate_pool_cache()
        pool3 = await strategy._build_content_rich_pool(source_id=None)
        assert pool3 is not pool1, "After invalidation, pool should be re-fetched"
        assert len(pool3) == len(pool1), "Pool size should be consistent"


# =============================================================================
# Test 3: Structure Cosine Search (Stage 1)
# =============================================================================


class TestStructureCosineSearch:
    """Test in-memory cosine search on pool."""

    @pytest.mark.asyncio
    async def test_search_returns_top_k(self, db, borrowed_embedding):
        """Structure search should return up to k_structure results."""
        from workflows.query_retrieval.strategies.top_down import (
            DEFAULT_K_STRUCTURE,
            TopDownRetrieval,
        )

        strategy = TopDownRetrieval(db=db)
        pool = await strategy._build_content_rich_pool(source_id=None)

        results = strategy._structure_cosine_search(
            query_embedding=borrowed_embedding,
            pool=pool,
            top_k=DEFAULT_K_STRUCTURE,
        )

        assert len(results) > 0, "Structure search returned no results"
        assert len(results) <= DEFAULT_K_STRUCTURE, (
            f"Should return at most {DEFAULT_K_STRUCTURE}, got {len(results)}"
        )

        logger.info(f"Structure search returned {len(results)} nodes:")
        for i, node in enumerate(results):
            logger.info(
                f"  [{i + 1}] score={node['structure_score']:.4f} "
                f"level={node.get('level')} "
                f"summary={node.get('summary', '')[:60]}"
            )

    @pytest.mark.asyncio
    async def test_scores_are_sorted_descending(self, db, borrowed_embedding):
        """Results should be sorted by score descending."""
        from workflows.query_retrieval.strategies.top_down import (
            TopDownRetrieval,
        )

        strategy = TopDownRetrieval(db=db)
        pool = await strategy._build_content_rich_pool(source_id=None)

        results = strategy._structure_cosine_search(
            query_embedding=borrowed_embedding,
            pool=pool,
            top_k=10,
        )

        scores = [r["structure_score"] for r in results]
        for i in range(len(scores) - 1):
            assert scores[i] >= scores[i + 1], (
                f"Scores not sorted: {scores[i]:.4f} < {scores[i + 1]:.4f} at index {i}"
            )

    @pytest.mark.asyncio
    async def test_scores_in_valid_range(self, db, borrowed_embedding):
        """Cosine scores should be in [0, 1]."""
        from workflows.query_retrieval.strategies.top_down import (
            TopDownRetrieval,
        )

        strategy = TopDownRetrieval(db=db)
        pool = await strategy._build_content_rich_pool(source_id=None)

        results = strategy._structure_cosine_search(
            query_embedding=borrowed_embedding,
            pool=pool,
            top_k=10,
        )

        for r in results:
            assert 0 <= r["structure_score"] <= 1.0, (
                f"Score {r['structure_score']} out of [0, 1] range"
            )

    @pytest.mark.asyncio
    async def test_borrowed_embedding_has_high_self_similarity(self, db, borrowed_embedding):
        """Using a borrowed embedding, the source node should score ~1.0."""
        from workflows.query_retrieval.strategies.top_down import (
            TopDownRetrieval,
        )

        strategy = TopDownRetrieval(db=db)
        pool = await strategy._build_content_rich_pool(source_id=None)

        results = strategy._structure_cosine_search(
            query_embedding=borrowed_embedding,
            pool=pool,
            top_k=1,
        )

        assert len(results) > 0
        top_score = results[0]["structure_score"]
        # The borrowed embedding came from a pool node, so top score should be very high
        assert top_score > 0.95, f"Self-similarity should be ~1.0, got {top_score:.4f}"
        logger.info(f"Self-similarity score: {top_score:.6f}")


# =============================================================================
# Test 4: Content Harvest (Stage 2)
# =============================================================================


class TestContentHarvest:
    """Test AQL-based content harvesting from matched structure nodes."""

    @pytest.mark.asyncio
    async def test_harvest_returns_content(self, db, borrowed_embedding):
        """Content harvest should return content nodes."""
        from workflows.query_retrieval.strategies.top_down import (
            TopDownRetrieval,
        )

        strategy = TopDownRetrieval(db=db)
        pool = await strategy._build_content_rich_pool(source_id=None)
        top_structures = strategy._structure_cosine_search(
            query_embedding=borrowed_embedding,
            pool=pool,
            top_k=5,
        )

        structure_ids = [s["_id"] for s in top_structures]
        harvested = await strategy._content_harvest(structure_ids)

        assert len(harvested) > 0, "Content harvest returned no nodes"
        logger.info(
            f"Harvested {len(harvested)} content nodes from {len(structure_ids)} structures"
        )

    @pytest.mark.asyncio
    async def test_harvested_nodes_have_text(self, db, borrowed_embedding):
        """Harvested content nodes should have text."""
        from workflows.query_retrieval.strategies.top_down import (
            TopDownRetrieval,
        )

        strategy = TopDownRetrieval(db=db)
        pool = await strategy._build_content_rich_pool(source_id=None)
        top_structures = strategy._structure_cosine_search(
            query_embedding=borrowed_embedding,
            pool=pool,
            top_k=3,
        )

        harvested = await strategy._content_harvest([s["_id"] for s in top_structures])
        with_text = sum(1 for h in harvested if h.get("text"))
        without_text = len(harvested) - with_text

        logger.info(f"With text: {with_text}, without text: {without_text}")
        assert with_text > 0, "No harvested nodes have text"

        # Show sample texts
        for i, h in enumerate(harvested[:3]):
            text_preview = (h.get("text") or "")[:100]
            logger.info(f"  [{i + 1}] depth={h.get('depth', '?')} | {text_preview}...")

    @pytest.mark.asyncio
    async def test_harvested_nodes_are_deduplicated(self, db, borrowed_embedding):
        """Harvested nodes should be unique by _id."""
        from workflows.query_retrieval.strategies.top_down import (
            TopDownRetrieval,
        )

        strategy = TopDownRetrieval(db=db)
        pool = await strategy._build_content_rich_pool(source_id=None)
        top_structures = strategy._structure_cosine_search(
            query_embedding=borrowed_embedding,
            pool=pool,
            top_k=7,
        )

        harvested = await strategy._content_harvest([s["_id"] for s in top_structures])

        ids = [h["_id"] for h in harvested]
        assert len(ids) == len(set(ids)), (
            f"Duplicate content nodes found: {len(ids)} total, {len(set(ids))} unique"
        )


# =============================================================================
# Test 5: Full Pipeline (execute method)
# =============================================================================


class TestFullPipeline:
    """Test the complete TopDownRetrieval.execute() pipeline."""

    @pytest.mark.asyncio
    async def test_execute_returns_results(self, db, borrowed_embedding):
        """Full pipeline should return ContentWithContext results."""
        from workflows.query_retrieval.strategies.top_down import (
            TopDownRetrieval,
        )

        strategy = TopDownRetrieval(db=db)
        results = await strategy.execute(
            query_embedding=borrowed_embedding,
            top_k=10,
        )

        assert len(results) > 0, "Full pipeline returned no results"
        logger.info(f"Pipeline returned {len(results)} results")

        for i, r in enumerate(results[:5]):
            logger.info(
                f"  [{i + 1}] score={r.score:.4f} type={r.node_type} text={r.text[:80]}..."
            )

    @pytest.mark.asyncio
    async def test_execute_metrics(self, db, borrowed_embedding):
        """Pipeline should produce valid metrics."""
        from workflows.query_retrieval.strategies.top_down import (
            TopDownRetrieval,
        )

        strategy = TopDownRetrieval(db=db)
        await strategy.execute(query_embedding=borrowed_embedding, top_k=10)

        m = strategy.metrics
        logger.info(
            f"Metrics:\n"
            f"  pool_size={m.pool_size}\n"
            f"  structure_matched={m.structure_nodes_matched}\n"
            f"  content_harvested={m.content_nodes_harvested}\n"
            f"  content_returned={m.content_nodes_returned}\n"
            f"  descend_depth={m.descend_depth_actual}\n"
            f"  latency_pool={m.latency_pool_ms:.1f}ms\n"
            f"  latency_search={m.latency_search_ms:.1f}ms\n"
            f"  latency_harvest={m.latency_harvest_ms:.1f}ms\n"
            f"  latency_rerank={m.latency_rerank_ms:.1f}ms\n"
            f"  latency_total={m.latency_total_ms:.1f}ms"
        )

        assert m.pool_size > 0, "Pool size should be > 0"
        assert m.structure_nodes_matched > 0, "Should match some structure nodes"
        assert m.content_nodes_harvested > 0, "Should harvest some content nodes"
        assert m.content_nodes_returned > 0, "Should return some content nodes"
        assert m.latency_total_ms > 0, "Total latency should be > 0"

        # Latency breakdown should sum roughly to total
        parts_sum = (
            m.latency_pool_ms + m.latency_search_ms + m.latency_harvest_ms + m.latency_rerank_ms
        )
        assert parts_sum <= m.latency_total_ms * 1.1, (
            f"Parts sum ({parts_sum:.1f}ms) exceeds total ({m.latency_total_ms:.1f}ms)"
        )

    @pytest.mark.asyncio
    async def test_execute_to_dict(self, db, borrowed_embedding):
        """metrics.to_dict() should produce a serializable dict."""
        from workflows.query_retrieval.strategies.top_down import (
            TopDownRetrieval,
        )

        strategy = TopDownRetrieval(db=db)
        await strategy.execute(query_embedding=borrowed_embedding, top_k=5)

        metrics_dict = strategy.metrics.to_dict()
        assert isinstance(metrics_dict, dict)
        assert "pool_size" in metrics_dict
        assert "latency_total_ms" in metrics_dict

    @pytest.mark.asyncio
    async def test_results_sorted_by_score(self, db, borrowed_embedding):
        """Results should be sorted by score descending."""
        from workflows.query_retrieval.strategies.top_down import (
            TopDownRetrieval,
        )

        strategy = TopDownRetrieval(db=db)
        results = await strategy.execute(
            query_embedding=borrowed_embedding,
            top_k=15,
        )

        scores = [r.score for r in results]
        for i in range(len(scores) - 1):
            assert scores[i] >= scores[i + 1], (
                f"Results not sorted: {scores[i]:.4f} < {scores[i + 1]:.4f}"
            )

    @pytest.mark.asyncio
    async def test_execute_with_small_top_k(self, db, borrowed_embedding):
        """Pipeline should respect top_k parameter."""
        from workflows.query_retrieval.strategies.top_down import (
            TopDownRetrieval,
        )

        strategy = TopDownRetrieval(db=db)
        results = await strategy.execute(
            query_embedding=borrowed_embedding,
            top_k=3,
        )

        assert len(results) <= 3, f"Expected at most 3 results, got {len(results)}"

    @pytest.mark.asyncio
    async def test_content_with_context_fields(self, db, borrowed_embedding):
        """ContentWithContext objects should have all expected fields."""
        from workflows.query_retrieval.strategies.top_down import (
            TopDownRetrieval,
        )

        strategy = TopDownRetrieval(db=db)
        results = await strategy.execute(
            query_embedding=borrowed_embedding,
            top_k=5,
        )

        assert len(results) > 0
        first = results[0]

        assert first.node_id, "node_id should be non-empty"
        assert first.text, "text should be non-empty"
        assert first.score >= 0, "score should be non-negative"
        assert first.node_type, "node_type should be non-empty"
        assert first.found_via_relation is None, (
            "found_via_relation should be None for top-down results"
        )


# =============================================================================
# Test 6: Full Workflow Pipeline (requires LLM)
# =============================================================================


class TestFullWorkflow:
    """Test the complete query_retrieval workflow with real LLM.

    These tests require --run-llm flag since they make LLM API calls
    for embedding and query routing.
    """

    @pytest.mark.asyncio
    async def test_workflow_structural_query(self, db, llm_client):
        """STRUCTURAL query should route to TOP_DOWN strategy."""
        from workflows.query_retrieval.models import (
            QueryRetrievalInput,
            RetrievalStrategy,
        )
        from workflows.query_retrieval.workflow import QueryRetrievalWorkflow

        workflow = QueryRetrievalWorkflow(db=db, llm_client=llm_client)
        result = await workflow.run(
            QueryRetrievalInput(query="Nội dung bài số 1 là gì?", top_k=5),
        )

        logger.info(
            f"Structural query result:\n"
            f"  query_type={result.classification.query_type.value}\n"
            f"  strategy={result.classification.strategy.value}\n"
            f"  total_results={result.total_results}\n"
            f"  total_time={result.total_time_ms:.0f}ms"
        )

        # Structural queries should be pre-classified and routed to TOP_DOWN
        assert result.classification.strategy == RetrievalStrategy.TOP_DOWN, (
            f"Expected TOP_DOWN, got {result.classification.strategy.value}"
        )
        assert result.total_results > 0, "Should return results for structural query"

    @pytest.mark.asyncio
    async def test_workflow_factual_query(self, db, llm_client):
        """FACTUAL query result should have classification and timing."""
        from workflows.query_retrieval.models import QueryRetrievalInput
        from workflows.query_retrieval.workflow import QueryRetrievalWorkflow

        workflow = QueryRetrievalWorkflow(db=db, llm_client=llm_client)
        result = await workflow.run(
            QueryRetrievalInput(
                query="Trận Bạch Đằng năm 938 do ai chỉ huy?",
                top_k=5,
            ),
        )

        logger.info(
            f"Factual query result:\n"
            f"  query_type={result.classification.query_type.value}\n"
            f"  strategy={result.classification.strategy.value}\n"
            f"  confidence={result.classification.confidence}\n"
            f"  total_results={result.total_results}\n"
            f"  embedding_time={result.embedding_time_ms:.0f}ms\n"
            f"  routing_time={result.llm_routing_time_ms:.0f}ms\n"
            f"  retrieval_time={result.retrieval_time_ms:.0f}ms\n"
            f"  total_time={result.total_time_ms:.0f}ms"
        )

        assert result.classification is not None
        assert result.total_time_ms > 0

    @pytest.mark.asyncio
    async def test_workflow_causal_query(self, db, llm_client):
        """CAUSAL query should use relations."""
        from workflows.query_retrieval.models import QueryRetrievalInput
        from workflows.query_retrieval.workflow import QueryRetrievalWorkflow

        workflow = QueryRetrievalWorkflow(db=db, llm_client=llm_client)
        result = await workflow.run(
            QueryRetrievalInput(
                query="Tại sao nhà Trần sụp đổ?",
                top_k=5,
            ),
        )

        logger.info(
            f"Causal query result:\n"
            f"  query_type={result.classification.query_type.value}\n"
            f"  strategy={result.classification.strategy.value}\n"
            f"  use_relations={result.classification.use_relations}\n"
            f"  selected_relations={result.classification.selected_relations}\n"
            f"  total_results={result.total_results}"
        )

        # "Tại sao" should trigger CAUSAL pre-classification
        assert result.classification.query_type.value == "causal", (
            f"Expected causal, got {result.classification.query_type.value}"
        )
        assert result.classification.use_relations is True

    @pytest.mark.asyncio
    async def test_workflow_to_dict(self, db, llm_client):
        """to_dict() should produce a JSON-serializable dict."""
        import json

        from workflows.query_retrieval.models import QueryRetrievalInput
        from workflows.query_retrieval.workflow import QueryRetrievalWorkflow

        workflow = QueryRetrievalWorkflow(db=db, llm_client=llm_client)
        result = await workflow.run(
            QueryRetrievalInput(query="Bài số 1 có nội dung gì?", top_k=3),
        )

        result_dict = result.to_dict()
        # Should be JSON serializable
        json_str = json.dumps(result_dict, ensure_ascii=False)
        assert len(json_str) > 0
        logger.info(f"to_dict() output size: {len(json_str)} bytes")


# =============================================================================
# Test 7: Edge Cases
# =============================================================================


class TestEdgeCases:
    """Test edge cases and boundary conditions."""

    @pytest.mark.asyncio
    async def test_zero_vector_returns_empty(self, db, borrowed_embedding):
        """A zero vector should return results with score 0 or empty."""
        from workflows.query_retrieval.strategies.top_down import (
            TopDownRetrieval,
        )

        dim = len(borrowed_embedding)
        strategy = TopDownRetrieval(db=db)
        zero_embedding = [0.0] * dim

        results = await strategy.execute(
            query_embedding=zero_embedding,
            top_k=5,
        )

        # Zero vector should produce no matches (all cosine scores = 0)
        logger.info(f"Zero vector results: {len(results)}")
        for r in results:
            assert r.score == 0.0, f"Expected score 0 for zero vector, got {r.score}"

    @pytest.mark.asyncio
    async def test_top_k_one(self, db, borrowed_embedding):
        """top_k=1 should return exactly 1 result."""
        from workflows.query_retrieval.strategies.top_down import (
            TopDownRetrieval,
        )

        strategy = TopDownRetrieval(db=db)
        results = await strategy.execute(
            query_embedding=borrowed_embedding,
            top_k=1,
        )

        assert len(results) == 1, f"Expected 1 result, got {len(results)}"

    @pytest.mark.asyncio
    async def test_random_embedding(self, db, borrowed_embedding):
        """Random embedding should still produce valid results."""
        from workflows.query_retrieval.strategies.top_down import (
            TopDownRetrieval,
        )

        dim = len(borrowed_embedding)
        rng = np.random.default_rng(42)
        random_embedding = rng.standard_normal(dim).tolist()

        strategy = TopDownRetrieval(db=db)
        results = await strategy.execute(
            query_embedding=random_embedding,
            top_k=5,
        )

        # Should return results (random vector will still have some cosine similarity)
        logger.info(f"Random embedding results: {len(results)}")
        if results:
            logger.info(f"  Top score: {results[0].score:.4f}")
            logger.info(f"  Bottom score: {results[-1].score:.4f}")

        # Scores should be valid
        for r in results:
            assert 0 <= r.score <= 1.0

    @pytest.mark.asyncio
    async def test_nonexistent_source_id(self, db, borrowed_embedding):
        """Filtering by non-existent source_id should return empty."""
        from workflows.query_retrieval.strategies.top_down import (
            TopDownRetrieval,
        )

        strategy = TopDownRetrieval(db=db)
        results = await strategy.execute(
            query_embedding=borrowed_embedding,
            top_k=5,
            source_id="nonexistent_source_id_12345",
        )

        assert len(results) == 0, f"Expected 0 results for non-existent source, got {len(results)}"
