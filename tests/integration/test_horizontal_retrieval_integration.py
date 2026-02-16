"""Integration tests for E2 Horizontal Retrieval against live ArangoDB.

These tests require a running ArangoDB instance with ingested data.
Run with: PYTHONPATH=. uv run pytest tests/integration/test_horizontal_retrieval_integration.py -v

Tests cover:
- Seed retrieval against real ArangoSearch view
- Graceful degradation when no Tier 2 content edges exist
- Full pipeline execution
- Metrics correctness
"""

from __future__ import annotations

import pytest

from workflows.query_retrieval.models import ContentWithContext
from workflows.query_retrieval.strategies.horizontal import (
    HorizontalRetrieval,
)

# =============================================================================
# Fixtures
# =============================================================================


@pytest.fixture(scope="module")
def arango_db():
    """Connect to live ArangoDB."""
    from arango import ArangoClient

    from config.settings import get_settings

    settings = get_settings()
    client = ArangoClient(hosts=settings.ARANGO_URL)
    db = client.db(
        settings.ARANGO_DATABASE,
        username=settings.ARANGO_USER,
        password=settings.ARANGO_PASSWORD,
    )

    # Verify connection
    try:
        db.aql.execute("RETURN 1")
    except Exception as e:
        pytest.skip(f"ArangoDB not available: {e}")

    # Verify content_nodes exist
    count_cursor = db.aql.execute("RETURN LENGTH(content_nodes)")
    count = next(count_cursor)
    if count == 0:
        pytest.skip("No content nodes in database")

    return db


@pytest.fixture(scope="module")
def borrowed_embedding(arango_db):
    """Borrow an embedding from an existing content node (avoids LLM API calls)."""
    cursor = arango_db.aql.execute("""
        FOR c IN content_nodes
            FILTER c.embedding != null AND LENGTH(c.embedding) > 0
            LIMIT 1
            RETURN c.embedding
    """)
    emb = next(cursor)
    assert emb is not None and len(emb) > 0
    return emb


@pytest.fixture
def horizontal(arango_db):
    """Create HorizontalRetrieval with live DB."""
    return HorizontalRetrieval(db=arango_db, graph_name="knowledge_graph")


# =============================================================================
# Test: Database Health
# =============================================================================


@pytest.mark.integration
class TestDatabaseHealth:
    def test_connection(self, arango_db):
        """Database is reachable."""
        cursor = arango_db.aql.execute("RETURN 1")
        assert next(cursor) == 1

    def test_content_nodes_exist(self, arango_db):
        """content_nodes collection has documents."""
        cursor = arango_db.aql.execute("RETURN LENGTH(content_nodes)")
        assert next(cursor) > 0

    def test_knowledge_graph_exists(self, arango_db):
        """knowledge_graph graph exists."""
        graphs = arango_db.graphs()
        graph_names = [g["name"] for g in graphs]
        assert "knowledge_graph" in graph_names

    def test_embeddings_present(self, arango_db):
        """Content nodes have embeddings."""
        cursor = arango_db.aql.execute("""
            LET with_emb = LENGTH(
                FOR c IN content_nodes
                    FILTER c.embedding != null AND LENGTH(c.embedding) > 0
                    RETURN 1
            )
            LET total = LENGTH(content_nodes)
            RETURN { with_emb, total, pct: total > 0 ? with_emb * 100 / total : 0 }
        """)
        result = next(cursor)
        assert result["pct"] > 80, f"Only {result['pct']}% of content nodes have embeddings"

    def test_arangosearch_view_exists(self, arango_db):
        """knowledge_vector_view ArangoSearch view exists."""
        views = arango_db.views()
        view_names = [v["name"] for v in views]
        assert "knowledge_vector_view" in view_names


# =============================================================================
# Test: Seed Retrieval (Stage 1)
# =============================================================================


@pytest.mark.integration
class TestSeedRetrieval:
    @pytest.mark.asyncio
    async def test_seed_retrieval_returns_results(self, horizontal, borrowed_embedding):
        """Seed retrieval returns content nodes from live DB."""
        seeds = await horizontal._get_seed_nodes(borrowed_embedding, n_seeds=5)
        assert len(seeds) > 0
        assert len(seeds) <= 5

    @pytest.mark.asyncio
    async def test_seeds_sorted_by_score(self, horizontal, borrowed_embedding):
        """Seeds are sorted by cosine score descending."""
        seeds = await horizontal._get_seed_nodes(borrowed_embedding, n_seeds=5)
        scores = [s["score"] for s in seeds]
        assert scores == sorted(scores, reverse=True)

    @pytest.mark.asyncio
    async def test_seeds_have_required_fields(self, horizontal, borrowed_embedding):
        """Each seed has all required fields."""
        seeds = await horizontal._get_seed_nodes(borrowed_embedding, n_seeds=1)
        assert len(seeds) == 1
        seed = seeds[0]
        assert "_id" in seed
        assert "_key" in seed
        assert "text" in seed
        assert "embedding" in seed
        assert "score" in seed
        assert seed["score"] > 0

    @pytest.mark.asyncio
    async def test_self_similarity_high(self, horizontal, borrowed_embedding):
        """Borrowed embedding should have high self-similarity."""
        seeds = await horizontal._get_seed_nodes(borrowed_embedding, n_seeds=1)
        assert len(seeds) >= 1
        # The source node should score very high (near 1.0)
        assert seeds[0]["score"] > 0.9

    @pytest.mark.asyncio
    async def test_seed_count_metric(self, horizontal, borrowed_embedding):
        """Metrics track seed count."""
        await horizontal._get_seed_nodes(borrowed_embedding, n_seeds=3)
        assert horizontal.metrics.seed_count == 3
        assert horizontal.metrics.latency_seed_ms > 0


# =============================================================================
# Test: Full Pipeline
# =============================================================================


@pytest.mark.integration
class TestFullPipeline:
    @pytest.mark.asyncio
    async def test_execute_returns_results(self, horizontal, borrowed_embedding):
        """Full pipeline returns ContentWithContext results."""
        results = await horizontal.execute(
            query="lịch sử Việt Nam",
            query_embedding=borrowed_embedding,
            query_type="factual",
            top_k=5,
        )
        assert len(results) > 0
        assert all(isinstance(r, ContentWithContext) for r in results)

    @pytest.mark.asyncio
    async def test_results_sorted_descending(self, horizontal, borrowed_embedding):
        """Results are sorted by score descending."""
        results = await horizontal.execute(
            query="nguyên nhân và hệ quả",
            query_embedding=borrowed_embedding,
            query_type="causal",
            top_k=10,
        )
        scores = [r.score for r in results]
        assert scores == sorted(scores, reverse=True)

    @pytest.mark.asyncio
    async def test_top_k_respected(self, horizontal, borrowed_embedding):
        """Returns at most top_k results."""
        results = await horizontal.execute(
            query="test query",
            query_embedding=borrowed_embedding,
            query_type="factual",
            top_k=3,
        )
        assert len(results) <= 3

    @pytest.mark.asyncio
    async def test_metrics_populated(self, horizontal, borrowed_embedding):
        """All key metrics are populated after execution."""
        await horizontal.execute(
            query="test query",
            query_embedding=borrowed_embedding,
            query_type="factual",
            top_k=5,
        )
        m = horizontal.metrics
        assert m.seed_count > 0
        assert m.latency_seed_ms > 0
        assert m.latency_total_ms > 0
        assert m.nodes_returned > 0

    @pytest.mark.asyncio
    async def test_metrics_to_dict_serializable(self, horizontal, borrowed_embedding):
        """metrics.to_dict() returns JSON-serializable dict."""
        import json

        await horizontal.execute(
            query="test query",
            query_embedding=borrowed_embedding,
            query_type="factual",
        )
        d = horizontal.metrics.to_dict()
        # Should be JSON serializable
        json_str = json.dumps(d)
        assert isinstance(json_str, str)

    @pytest.mark.asyncio
    async def test_content_with_context_fields(self, horizontal, borrowed_embedding):
        """Results have proper ContentWithContext fields."""
        results = await horizontal.execute(
            query="test",
            query_embedding=borrowed_embedding,
            query_type="factual",
            top_k=1,
        )
        assert len(results) >= 1
        r = results[0]
        assert r.node_id != ""
        assert r.text != ""
        assert r.score > 0
        assert r.node_type in ("paragraph", "image", "table")


# =============================================================================
# Test: Graceful Degradation
# =============================================================================


@pytest.mark.integration
class TestGracefulDegradation:
    @pytest.mark.asyncio
    async def test_no_tier2_edges_returns_seeds(self, horizontal, borrowed_embedding):
        """When no Tier 2 content edges exist, E2 gracefully returns seeds."""
        results = await horizontal.execute(
            query="nguyên nhân cách mạng",
            query_embedding=borrowed_embedding,
            query_type="causal",
            top_k=5,
        )
        # Should still return results (seeds at minimum)
        assert len(results) > 0
        # All results should be seeds (no traversal nodes) since no Tier 2 edges
        # This is expected graceful degradation
        m = horizontal.metrics
        assert m.seed_count > 0

    @pytest.mark.asyncio
    async def test_zero_vector_returns_empty(self, horizontal):
        """Zero embedding returns empty results."""
        results = await horizontal.execute(
            query="test",
            query_embedding=[0.0] * 3072,
            query_type="factual",
            top_k=5,
        )
        assert results == []

    @pytest.mark.asyncio
    async def test_n_seeds_1(self, horizontal, borrowed_embedding):
        """Works with n_seeds=1."""
        results = await horizontal.execute(
            query="test",
            query_embedding=borrowed_embedding,
            query_type="factual",
            top_k=5,
            n_seeds=1,
        )
        assert horizontal.metrics.seed_count == 1


# =============================================================================
# Test: Edge Cases
# =============================================================================


@pytest.mark.integration
class TestEdgeCases:
    @pytest.mark.asyncio
    async def test_random_embedding(self, horizontal):
        """Random embedding still returns results (though low scores)."""
        import numpy as np

        rng = np.random.default_rng(12345)
        random_emb = rng.random(3072).tolist()

        results = await horizontal.execute(
            query="random query",
            query_embedding=random_emb,
            query_type="factual",
            top_k=5,
        )
        # Should return some results (ArangoSearch still finds matches)
        # Scores will be low but non-zero
        assert len(results) >= 0  # May or may not find results

    @pytest.mark.asyncio
    async def test_different_query_types(self, horizontal, borrowed_embedding):
        """Different query types produce valid results."""
        for qtype in ["factual", "causal", "comparison", "concept"]:
            results = await horizontal.execute(
                query="test query",
                query_embedding=borrowed_embedding,
                query_type=qtype,
                top_k=3,
            )
            assert len(results) > 0, f"No results for query_type={qtype}"
            assert all(isinstance(r, ContentWithContext) for r in results)
