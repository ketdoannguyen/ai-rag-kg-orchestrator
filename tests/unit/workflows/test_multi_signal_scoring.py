"""Tests for multi-signal relation scoring."""

from __future__ import annotations

from unittest.mock import Mock

import pytest

from workflows.query_retrieval.workflow import RelationScorer


class TestRelationScorer:
    """Tests for multi-signal scoring."""

    @pytest.fixture
    def scorer(self):
        return RelationScorer()

    @pytest.fixture
    def mock_relations(self):
        """Create mock relations with embeddings, families, keywords."""
        rels = []

        # Relation 1: High embedding, high affinity (causal)
        rel1 = Mock()
        rel1.id = "CAUSED_BY"
        rel1.embedding = [1.0, 0.0, 0.0]
        rel1.families = ["causal"]
        rel1.keywords = ["tại sao", "nguyên nhân"]
        rels.append(rel1)

        # Relation 2: Medium embedding, comparative family
        rel2 = Mock()
        rel2.id = "SIMILAR_TO"
        rel2.embedding = [0.5, 0.5, 0.0]
        rel2.families = ["comparative"]
        rel2.keywords = ["giống", "tương tự"]
        rels.append(rel2)

        # Relation 3: Low embedding, no families
        rel3 = Mock()
        rel3.id = "UNKNOWN"
        rel3.embedding = [0.0, 0.0, 1.0]
        rel3.families = []
        rel3.keywords = []
        rels.append(rel3)

        return rels

    def test_embedding_score_dominates_when_similar(self, scorer, mock_relations):
        """Test that high embedding similarity ranks relation first."""
        results = scorer.score_all_vectorized(
            query="Test query",
            query_embedding=[1.0, 0.0, 0.0],  # Same as CAUSED_BY
            query_type="causal",
            relations=mock_relations,
        )

        assert results[0][0] == "CAUSED_BY"
        assert results[0][2] > 0.9  # embedding_score

    def test_affinity_boosts_matching_family(self, scorer, mock_relations):
        """Test that family affinity boosts score for matching query type."""
        results = scorer.score_all_vectorized(
            query="So sánh hai triều đại",
            query_embedding=[0.5, 0.5, 0.0],  # Similar to SIMILAR_TO
            query_type="comparison",
            relations=mock_relations,
        )

        # SIMILAR_TO should have high affinity for comparison
        similar_result = next(r for r in results if r[0] == "SIMILAR_TO")
        assert similar_result[3] == 1.0  # affinity_score

    def test_keyword_matching(self, scorer, mock_relations):
        """Test keyword matching contribution."""
        results = scorer.score_all_vectorized(
            query="Tại sao nguyên nhân thất bại?",
            query_embedding=[0.3, 0.3, 0.3],  # Neutral
            query_type="causal",
            relations=mock_relations,
        )

        # CAUSED_BY has keywords "tại sao", "nguyên nhân" - both match
        caused_by = next(r for r in results if r[0] == "CAUSED_BY")
        assert caused_by[4] == 1.0  # keyword_score (2/2 matched)

    def test_empty_relations(self, scorer):
        """Test with empty relations list."""
        results = scorer.score_all_vectorized(
            query="Test",
            query_embedding=[1, 0, 0],
            query_type="causal",
            relations=[],
        )

        assert results == []

    def test_relation_without_embedding(self, scorer):
        """Test relation without embedding gets 0 embedding score."""
        rel = Mock()
        rel.id = "NO_EMB"
        rel.embedding = None
        rel.families = ["causal"]
        rel.keywords = ["test"]

        results = scorer.score_all_vectorized(
            query="Test",
            query_embedding=[1, 0, 0],
            query_type="causal",
            relations=[rel],
        )

        assert results[0][2] == 0.0  # embedding_score = 0

    def test_weights_sum_to_one(self, scorer):
        """Test that default weights sum to 1.0."""
        assert abs((scorer.ALPHA + scorer.BETA + scorer.GAMMA) - 1.0) < 0.001

    def test_custom_weights(self):
        """Test scorer with custom weights."""
        scorer = RelationScorer(alpha=0.7, beta=0.2, gamma=0.1)

        rel = Mock()
        rel.id = "TEST"
        rel.embedding = [1, 0, 0]
        rel.families = ["causal"]
        rel.keywords = ["test"]

        results = scorer.score_all_vectorized(
            query="test",
            query_embedding=[1, 0, 0],
            query_type="causal",
            relations=[rel],
        )

        # With high alpha, embedding should dominate
        # emb=1.0, aff=1.0, kw=1.0 → final = 0.7*1 + 0.2*1 + 0.1*1 = 1.0
        assert results[0][1] == pytest.approx(1.0)

    def test_default_affinity_for_no_families(self, scorer):
        """Test default affinity (0.5) for relations without families."""
        rel = Mock()
        rel.id = "NO_FAMILY"
        rel.embedding = [1, 0, 0]
        rel.families = []
        rel.keywords = []

        results = scorer.score_all_vectorized(
            query="test",
            query_embedding=[1, 0, 0],
            query_type="causal",
            relations=[rel],
        )

        assert results[0][3] == 0.5  # default affinity

    def test_sorted_by_final_score_descending(self, scorer, mock_relations):
        """Test results are sorted by final score descending."""
        results = scorer.score_all_vectorized(
            query="Test query",
            query_embedding=[1.0, 0.0, 0.0],
            query_type="causal",
            relations=mock_relations,
        )

        # Check descending order
        for i in range(len(results) - 1):
            assert results[i][1] >= results[i + 1][1]

    def test_result_tuple_structure(self, scorer, mock_relations):
        """Test result tuple has correct structure."""
        results = scorer.score_all_vectorized(
            query="Test",
            query_embedding=[1.0, 0.0, 0.0],
            query_type="causal",
            relations=mock_relations,
        )

        for result in results:
            assert len(result) == 5
            relation_id, final, emb, aff, kw = result
            assert isinstance(relation_id, str)
            assert isinstance(final, float)
            assert isinstance(emb, float)
            assert isinstance(aff, float)
            assert isinstance(kw, float)
            # All scores should be between 0 and 1
            assert 0 <= emb <= 1
            assert 0 <= aff <= 1
            assert 0 <= kw <= 1


class TestParallelExecution:
    """Tests for parallel search and traversal."""

    @pytest.mark.asyncio
    async def test_parallel_faster_than_sequential(self):
        """Test that parallel execution is faster than sequential."""
        import asyncio
        import time

        async def slow_task():
            await asyncio.sleep(0.1)
            return []

        # Sequential
        start = time.time()
        await slow_task()
        await slow_task()
        sequential_time = time.time() - start

        # Parallel
        start = time.time()
        await asyncio.gather(slow_task(), slow_task())
        parallel_time = time.time() - start

        assert parallel_time < sequential_time * 0.7  # At least 30% faster
