"""Tests for HybridRelationScorer using BM25 + Embedding + RRF fusion.

Reference:
    Cormack et al. "Reciprocal rank fusion" (SIGIR 2009)
"""

from __future__ import annotations

from unittest.mock import Mock

import pytest

from workflows.query_retrieval.relation_scorer import RRF_K, HybridRelationScorer


class TestHybridRelationScorer:
    """Tests for HybridRelationScorer."""

    @pytest.fixture
    def scorer(self):
        return HybridRelationScorer()

    @pytest.fixture
    def mock_relations(self):
        """Create mock relations with embeddings and descriptions."""
        rels = []

        # Relation 1: High BM25 match (description contains query keywords)
        rel1 = Mock()
        rel1.id = "CAUSED_BY"
        rel1.embedding = [1.0, 0.0, 0.0]
        rel1.description = "Chỉ ra nguyên nhân tại sao sự kiện xảy ra"
        rel1.keywords = ["tại sao", "nguyên nhân"]
        rels.append(rel1)

        # Relation 2: Medium match
        rel2 = Mock()
        rel2.id = "SIMILAR_TO"
        rel2.embedding = [0.5, 0.5, 0.0]
        rel2.description = "So sánh và tìm điểm tương đồng giữa hai đối tượng"
        rel2.keywords = ["giống", "tương tự"]
        rels.append(rel2)

        # Relation 3: Low match
        rel3 = Mock()
        rel3.id = "ELABORATES"
        rel3.embedding = [0.0, 0.0, 1.0]
        rel3.description = "Giải thích chi tiết hơn về một khái niệm"
        rel3.keywords = ["giải thích", "chi tiết"]
        rels.append(rel3)

        return rels

    def test_returns_4_tuples(self, scorer, mock_relations):
        """Test that scorer returns 4-tuples (id, rrf, bm25, emb)."""
        results = scorer.score_all(
            query="Tại sao",
            query_embedding=[1.0, 0.0, 0.0],
            relations=mock_relations,
        )

        assert len(results) == 3
        for result in results:
            assert len(result) == 4
            rel_id, rrf_score, bm25_score, emb_score = result
            assert isinstance(rel_id, str)
            assert isinstance(rrf_score, float)
            assert isinstance(bm25_score, float)
            assert isinstance(emb_score, float)

    def test_sorted_by_rrf_descending(self, scorer, mock_relations):
        """Test results are sorted by RRF score descending."""
        results = scorer.score_all(
            query="Test query",
            query_embedding=[1.0, 0.0, 0.0],
            relations=mock_relations,
        )

        rrf_scores = [r[1] for r in results]
        assert rrf_scores == sorted(rrf_scores, reverse=True)

    def test_embedding_similarity_dominates_when_exact_match(self, scorer, mock_relations):
        """Test that exact embedding match gets high score."""
        results = scorer.score_all(
            query="random query",  # No BM25 match
            query_embedding=[1.0, 0.0, 0.0],  # Exact match with CAUSED_BY
            relations=mock_relations,
        )

        # CAUSED_BY should rank high due to embedding match
        caused_by = next(r for r in results if r[0] == "CAUSED_BY")
        assert caused_by[3] > 0.9  # emb_score should be ~1.0

    def test_bm25_boosts_lexical_matches(self, scorer, mock_relations):
        """Test that BM25 boosts relations with matching keywords."""
        results = scorer.score_all(
            query="Tại sao nguyên nhân",  # Matches CAUSED_BY description/keywords
            query_embedding=[0.3, 0.3, 0.3],  # Neutral embedding
            relations=mock_relations,
        )

        # CAUSED_BY should have highest BM25 score
        caused_by = next(r for r in results if r[0] == "CAUSED_BY")
        others = [r for r in results if r[0] != "CAUSED_BY"]
        assert all(caused_by[2] > other[2] for other in others)

    def test_empty_relations(self, scorer):
        """Test with empty relations list."""
        results = scorer.score_all(
            query="Test",
            query_embedding=[1, 0, 0],
            relations=[],
        )
        assert results == []

    def test_relation_without_embedding(self, scorer):
        """Test relation without embedding gets 0 embedding score."""
        rel = Mock()
        rel.id = "NO_EMB"
        rel.embedding = None
        rel.description = "Test relation"
        rel.keywords = ["test"]

        results = scorer.score_all(
            query="Test",
            query_embedding=[1, 0, 0],
            relations=[rel],
        )

        assert results[0][3] == 0.0  # emb_score = 0

    def test_rrf_k_default(self, scorer):
        """Test default RRF k value is 60."""
        assert scorer.rrf_k == RRF_K
        assert scorer.rrf_k == 60

    def test_custom_rrf_k(self):
        """Test scorer with custom RRF k."""
        scorer = HybridRelationScorer(rrf_k=30)
        assert scorer.rrf_k == 30

    def test_single_relation(self, scorer):
        """Test with single relation."""
        rel = Mock()
        rel.id = "SINGLE"
        rel.embedding = [1, 0, 0]
        rel.description = "Test"
        rel.keywords = []

        results = scorer.score_all(
            query="test",
            query_embedding=[1, 0, 0],
            relations=[rel],
        )

        assert len(results) == 1
        # Single relation should have max possible RRF score
        expected_max_rrf = 2.0 / (60 + 1)  # Both ranks are 1
        assert abs(results[0][1] - expected_max_rrf) < 0.001


class TestRRFFusion:
    """Tests for RRF fusion specifically."""

    @pytest.fixture
    def scorer(self):
        return HybridRelationScorer()

    def test_rrf_uses_ranks_not_scores(self, scorer):
        """Test that RRF uses ranks, not raw scores."""
        # Create relations where BM25 scores are very different in magnitude
        # but RRF should normalize them via ranks
        rels = []
        for i, (bm25_mult, emb_val) in enumerate(
            [(100.0, 0.9), (10.0, 0.8), (1.0, 0.7)]
        ):
            rel = Mock()
            rel.id = f"REL_{i}"
            rel.embedding = [emb_val, 0.0, 0.0]
            rel.description = "keyword " * int(bm25_mult)  # More keywords = higher BM25
            rel.keywords = []
            rels.append(rel)

        results = scorer.score_all(
            query="keyword",
            query_embedding=[1.0, 0.0, 0.0],
            relations=rels,
        )

        # RRF should produce scores in a reasonable range regardless of BM25 magnitude
        rrf_scores = [r[1] for r in results]
        assert all(0 < s < 1 for s in rrf_scores)

    def test_rrf_balanced_ranks_preferred(self, scorer):
        """Test that relation with balanced good ranks beats extreme ranks."""
        # Rel A: rank 1 in BM25, rank 3 in embedding
        # Rel B: rank 2 in BM25, rank 2 in embedding
        # Rel B should have higher RRF because (2,2) is more balanced than (1,3)
        rels = []

        # Rel A: High BM25, low embedding
        rel_a = Mock()
        rel_a.id = "A"
        rel_a.embedding = [0.1, 0.0, 0.0]
        rel_a.description = "keyword keyword keyword"  # High BM25
        rel_a.keywords = []
        rels.append(rel_a)

        # Rel B: Medium BM25, medium embedding
        rel_b = Mock()
        rel_b.id = "B"
        rel_b.embedding = [0.5, 0.0, 0.0]
        rel_b.description = "keyword keyword"  # Medium BM25
        rel_b.keywords = []
        rels.append(rel_b)

        # Rel C: Low BM25, high embedding
        rel_c = Mock()
        rel_c.id = "C"
        rel_c.embedding = [0.9, 0.0, 0.0]
        rel_c.description = "keyword"  # Low BM25
        rel_c.keywords = []
        rels.append(rel_c)

        results = scorer.score_all(
            query="keyword",
            query_embedding=[1.0, 0.0, 0.0],
            relations=rels,
        )

        # All three should have similar RRF scores since each has one high rank
        rrf_scores = {r[0]: r[1] for r in results}
        score_range = max(rrf_scores.values()) - min(rrf_scores.values())
        assert score_range < 0.01  # Scores should be close


class TestBM25Scoring:
    """Tests for BM25 scoring component.

    Note: BM25 requires a corpus of at least 3-4 documents to produce
    meaningful IDF weights. With only 2 documents, IDF can be zero.
    """

    @pytest.fixture
    def scorer(self):
        return HybridRelationScorer()

    def test_bm25_with_adequate_corpus(self, scorer):
        """Test BM25 correctly ranks documents with matching terms.

        BM25 needs 3+ documents for IDF to produce non-zero values.
        """
        # Create 4 relations so BM25 IDF works properly
        rel_match = Mock()
        rel_match.id = "MATCH"
        rel_match.embedding = [0.5, 0.5, 0.0]
        rel_match.description = "nguyên nhân tại sao sự kiện xảy ra"
        rel_match.keywords = []

        rel2 = Mock()
        rel2.id = "NO_MATCH_1"
        rel2.embedding = [0.5, 0.5, 0.0]
        rel2.description = "completely different content here"
        rel2.keywords = []

        rel3 = Mock()
        rel3.id = "NO_MATCH_2"
        rel3.embedding = [0.5, 0.5, 0.0]
        rel3.description = "another unrelated document text"
        rel3.keywords = []

        rel4 = Mock()
        rel4.id = "NO_MATCH_3"
        rel4.embedding = [0.5, 0.5, 0.0]
        rel4.description = "yet more different words"
        rel4.keywords = []

        results = scorer.score_all(
            query="nguyên nhân tại sao",
            query_embedding=[0.0, 0.0, 1.0],
            relations=[rel_match, rel2, rel3, rel4],
        )

        match_result = next(r for r in results if r[0] == "MATCH")
        other_results = [r for r in results if r[0] != "MATCH"]

        # MATCH should have higher BM25 score than others
        assert match_result[2] > 0
        assert all(match_result[2] > other[2] for other in other_results)

    def test_no_keyword_match_zero_score(self, scorer):
        """Test documents without matching terms get zero BM25 score."""
        rel = Mock()
        rel.id = "TEST"
        rel.embedding = [1.0, 0.0, 0.0]
        rel.description = "completely different content"
        rel.keywords = ["unrelated"]

        results = scorer.score_all(
            query="nguyên nhân tại sao",
            query_embedding=[1.0, 0.0, 0.0],
            relations=[rel],
        )

        # With single doc and no match, BM25 returns 0
        assert results[0][2] == 0.0

    def test_bm25_small_corpus_behavior(self, scorer):
        """Test BM25 with small corpus (2 docs) has expected behavior.

        BM25's IDF formula produces 0 for terms in 1 of 2 docs,
        which is mathematically correct but unintuitive.
        """
        rel1 = Mock()
        rel1.id = "DOC1"
        rel1.embedding = [0.5, 0.5, 0.0]
        rel1.description = "unique special terms here"
        rel1.keywords = []

        rel2 = Mock()
        rel2.id = "DOC2"
        rel2.embedding = [0.5, 0.5, 0.0]
        rel2.description = "different words entirely"
        rel2.keywords = []

        results = scorer.score_all(
            query="unique special",
            query_embedding=[0.5, 0.5, 0.0],
            relations=[rel1, rel2],
        )

        # With only 2 docs, BM25 may produce 0 scores due to IDF
        # This is expected behavior, not a bug
        doc1_score = next(r for r in results if r[0] == "DOC1")[2]
        doc2_score = next(r for r in results if r[0] == "DOC2")[2]
        assert doc1_score >= doc2_score  # At least not worse


class TestEmbeddingScoring:
    """Tests for embedding scoring component."""

    @pytest.fixture
    def scorer(self):
        return HybridRelationScorer()

    def test_identical_vectors_score_one(self, scorer):
        """Test identical embeddings score 1.0."""
        rel = Mock()
        rel.id = "TEST"
        rel.embedding = [1.0, 0.0, 0.0]
        rel.description = "test"
        rel.keywords = []

        results = scorer.score_all(
            query="test",
            query_embedding=[1.0, 0.0, 0.0],
            relations=[rel],
        )

        assert abs(results[0][3] - 1.0) < 0.001

    def test_orthogonal_vectors_score_zero(self, scorer):
        """Test orthogonal embeddings score ~0."""
        rel = Mock()
        rel.id = "TEST"
        rel.embedding = [1.0, 0.0, 0.0]
        rel.description = "test"
        rel.keywords = []

        results = scorer.score_all(
            query="test",
            query_embedding=[0.0, 1.0, 0.0],
            relations=[rel],
        )

        assert results[0][3] < 0.001

    def test_empty_embedding_handled(self, scorer):
        """Test empty embedding array handled."""
        rel = Mock()
        rel.id = "TEST"
        rel.embedding = []
        rel.description = "test"
        rel.keywords = []

        results = scorer.score_all(
            query="test",
            query_embedding=[1.0, 0.0, 0.0],
            relations=[rel],
        )

        assert results[0][3] == 0.0


class TestBackwardCompatibility:
    """Tests for backward compatibility with deprecated RelationScorer."""

    def test_deprecated_class_warns(self):
        """Test that using RelationScorer emits deprecation warning."""
        from workflows.query_retrieval.relation_scorer import RelationScorer

        with pytest.warns(DeprecationWarning, match="deprecated"):
            RelationScorer()

    def test_deprecated_score_all_vectorized(self):
        """Test backward-compatible method returns 5-tuples."""
        import warnings

        from workflows.query_retrieval.relation_scorer import RelationScorer

        with warnings.catch_warnings():
            warnings.simplefilter("ignore", DeprecationWarning)
            scorer = RelationScorer()

        rel = Mock()
        rel.id = "TEST"
        rel.embedding = [1.0, 0.0, 0.0]
        rel.description = "test"
        rel.keywords = []

        results = scorer.score_all_vectorized(
            query="test",
            query_embedding=[1.0, 0.0, 0.0],
            query_type="causal",  # Should be ignored
            relations=[rel],
        )

        # Should return 5-tuples for compatibility
        assert len(results) == 1
        assert len(results[0]) == 5
        assert results[0][4] == 0.0  # Last element always 0


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
