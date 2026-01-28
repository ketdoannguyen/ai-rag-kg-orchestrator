"""Tests for parallel batch extraction and family inference.

Tests cover:
- verify_batch_parallel performance vs sequential
- Exception handling in parallel processing
- Semaphore rate limiting
- Batch embedding
- Family inference
"""

from __future__ import annotations

import asyncio
from unittest.mock import AsyncMock, Mock

import pytest


class TestVerifyBatchParallel:
    """Tests for parallel batch verification."""

    @pytest.fixture
    def mock_settings(self):
        """Create mock extraction settings."""
        settings = Mock()
        settings.EXTRACTION_STRONG_MODEL = "test-model"
        settings.EXTRACTION_FALLBACK_RELATION = "SIMILAR_TO"
        settings.EXTRACTION_EMBEDDING_MODEL = "test-embedding"
        settings.EXTRACTION_MERGE_THRESHOLD = 0.8
        return settings

    @pytest.fixture
    def sample_pairs(self):
        """Create sample node pairs for testing."""
        from workflows.extract_relations.llm_extractor import NodePair

        return [
            NodePair(
                node_a_id=f"a{i}",
                node_a_type="paragraph",
                node_a_text=f"Content A {i}",
                node_b_id=f"b{i}",
                node_b_type="paragraph",
                node_b_text=f"Content B {i}",
                similarity_score=0.7,
            )
            for i in range(5)
        ]

    @pytest.mark.asyncio
    async def test_parallel_faster_than_sequential(self, mock_settings, sample_pairs):
        """Test that parallel is actually faster."""
        from workflows.extract_relations.llm_extractor import (
            ExtractionResult,
            LLMRelationExtractor,
        )

        extractor = LLMRelationExtractor(settings=mock_settings)

        # Mock verify_pair to take 0.1 second
        async def slow_verify(*_args, **_kwargs):
            await asyncio.sleep(0.1)
            return ExtractionResult(
                should_merge=False,
                relation_type="TEST",
                confidence=0.8,
            )

        extractor.verify_pair = slow_verify

        import time

        start = time.time()
        results = await extractor.verify_batch_parallel(sample_pairs)
        elapsed = time.time() - start

        # 5 pairs at 0.1s each = 0.5s sequential
        # With parallel (semaphore=5), should be ~0.1s
        assert elapsed < 0.3  # Allow some overhead
        assert len(results) == 5

    @pytest.mark.asyncio
    async def test_handles_exceptions_gracefully(self, mock_settings, sample_pairs):
        """Test that exceptions don't crash the batch."""
        from workflows.extract_relations.llm_extractor import (
            ExtractionResult,
            LLMRelationExtractor,
        )

        extractor = LLMRelationExtractor(settings=mock_settings)
        call_count = 0

        async def sometimes_fail(*_args, **_kwargs):
            nonlocal call_count
            call_count += 1
            if call_count == 3:
                raise ValueError("Simulated failure")
            return ExtractionResult(
                should_merge=False,
                relation_type="TEST",
                confidence=0.8,
            )

        extractor.verify_pair = sometimes_fail

        results = await extractor.verify_batch_parallel(sample_pairs)

        assert len(results) == 5
        # One should be fallback
        fallbacks = [r for r in results if r.used_fallback]
        assert len(fallbacks) == 1

    @pytest.mark.asyncio
    async def test_respects_semaphore_limit(self, mock_settings, sample_pairs):
        """Test that semaphore limits concurrent calls."""
        from workflows.extract_relations.llm_extractor import (
            ExtractionResult,
            LLMRelationExtractor,
        )

        extractor = LLMRelationExtractor(settings=mock_settings)
        concurrent_count = 0
        max_concurrent = 0

        async def track_concurrent(*_args, **_kwargs):
            nonlocal concurrent_count, max_concurrent
            concurrent_count += 1
            max_concurrent = max(max_concurrent, concurrent_count)
            await asyncio.sleep(0.05)
            concurrent_count -= 1
            return ExtractionResult(
                should_merge=False,
                relation_type="TEST",
                confidence=0.8,
            )

        extractor.verify_pair = track_concurrent
        extractor._llm_semaphore = asyncio.Semaphore(3)  # Limit to 3

        # Use more pairs than semaphore limit
        pairs = sample_pairs * 3  # 15 pairs
        await extractor.verify_batch_parallel(pairs)

        assert max_concurrent <= 3

    @pytest.mark.asyncio
    async def test_empty_pairs_returns_empty(self, mock_settings):
        """Test that empty input returns empty output."""
        from workflows.extract_relations.llm_extractor import LLMRelationExtractor

        extractor = LLMRelationExtractor(settings=mock_settings)
        results = await extractor.verify_batch_parallel([])
        assert results == []

    @pytest.mark.asyncio
    async def test_single_pair_delegates_to_verify_pair(self, mock_settings, sample_pairs):
        """Test that single pair uses verify_pair directly."""
        from workflows.extract_relations.llm_extractor import (
            ExtractionResult,
            LLMRelationExtractor,
        )

        extractor = LLMRelationExtractor(settings=mock_settings)
        verify_pair_called = False

        async def mock_verify_pair(*_args, **_kwargs):
            nonlocal verify_pair_called
            verify_pair_called = True
            return ExtractionResult(
                should_merge=False,
                relation_type="SINGLE_TEST",
                confidence=0.9,
            )

        extractor.verify_pair = mock_verify_pair

        results = await extractor.verify_batch_parallel([sample_pairs[0]])

        assert verify_pair_called
        assert len(results) == 1
        assert results[0].relation_type == "SINGLE_TEST"


class TestBatchEmbed:
    """Tests for batch embedding."""

    @pytest.fixture
    def mock_settings(self):
        """Create mock extraction settings."""
        settings = Mock()
        settings.EXTRACTION_EMBEDDING_MODEL = "test-embedding"
        settings.EXTRACTION_MERGE_THRESHOLD = 0.8
        return settings

    @pytest.mark.asyncio
    async def test_batch_embed_multiple(self, mock_settings):
        """Test batch embedding multiple relations."""
        from workflows.extract_relations.relation_validator import NewRelationValidator

        validator = NewRelationValidator(settings=mock_settings)
        validator._llm_client = AsyncMock()
        validator._llm_client.embed.return_value = Mock(
            embeddings=[[0.1, 0.2], [0.3, 0.4], [0.5, 0.6]]
        )

        relations = [
            ("REL1", "Description 1"),
            ("REL2", "Description 2"),
            ("REL3", "Description 3"),
        ]

        embeddings = await validator.batch_embed_relations(relations)

        assert len(embeddings) == 3
        assert embeddings[0] == [0.1, 0.2]

    @pytest.mark.asyncio
    async def test_batch_embed_empty_returns_empty(self, mock_settings):
        """Test batch embed with empty input."""
        from workflows.extract_relations.relation_validator import NewRelationValidator

        validator = NewRelationValidator(settings=mock_settings)
        embeddings = await validator.batch_embed_relations([])
        assert embeddings == []

    @pytest.mark.asyncio
    async def test_batch_embed_handles_error(self, mock_settings):
        """Test batch embed handles API errors."""
        from workflows.extract_relations.relation_validator import NewRelationValidator

        validator = NewRelationValidator(settings=mock_settings)
        validator._llm_client = AsyncMock()
        validator._llm_client.embed.side_effect = Exception("API Error")

        relations = [("REL1", "Description 1")]
        embeddings = await validator.batch_embed_relations(relations)

        assert len(embeddings) == 1
        assert embeddings[0] is None


class TestFamilyInference:
    """Tests for family inference."""

    @pytest.fixture
    def mock_settings(self):
        """Create mock extraction settings."""
        settings = Mock()
        settings.EXTRACTION_EMBEDDING_MODEL = "test-embedding"
        settings.EXTRACTION_MERGE_THRESHOLD = 0.8
        return settings

    @pytest.mark.asyncio
    async def test_infer_from_similar_relation(self, mock_settings):
        """Test inferring families from similar existing relation."""
        from workflows.extract_relations.relation_validator import NewRelationValidator

        validator = NewRelationValidator(settings=mock_settings)

        # Mock ontology handler
        mock_handler = Mock()
        mock_handler.find_similar_excluding.return_value = ("CAUSED_BY", 0.85)
        mock_handler.get_by_id.return_value = Mock(families=["causal", "temporal"])
        validator._ontology_handler = mock_handler

        # Mock embedding
        validator._embed_relation = AsyncMock(return_value=[0.1, 0.2, 0.3])

        families = await validator.infer_families("NEW_REL", "Some description")

        assert "causal" in families
        assert "temporal" in families

    @pytest.mark.asyncio
    async def test_low_similarity_adds_elaborative(self, mock_settings):
        """Test that low similarity adds elaborative family."""
        from workflows.extract_relations.relation_validator import NewRelationValidator

        validator = NewRelationValidator(settings=mock_settings)

        # Mock ontology handler with low similarity
        mock_handler = Mock()
        mock_handler.find_similar_excluding.return_value = ("SOME_REL", 0.45)
        mock_handler.get_by_id.return_value = Mock(families=["causal"])
        validator._ontology_handler = mock_handler

        validator._embed_relation = AsyncMock(return_value=[0.1, 0.2, 0.3])

        families = await validator.infer_families("NEW_REL", "Description")

        assert "causal" in families
        assert "elaborative" in families

    @pytest.mark.asyncio
    async def test_default_family_when_no_handler(self, mock_settings):
        """Test default family when no ontology handler."""
        from workflows.extract_relations.relation_validator import NewRelationValidator

        validator = NewRelationValidator(settings=mock_settings)
        validator._ontology_handler = None

        families = await validator.infer_families("NEW_REL", "Description")

        assert families == ["elaborative"]

    @pytest.mark.asyncio
    async def test_default_family_when_no_match(self, mock_settings):
        """Test default family when no similar relation found."""
        from workflows.extract_relations.relation_validator import NewRelationValidator

        validator = NewRelationValidator(settings=mock_settings)

        mock_handler = Mock()
        mock_handler.find_similar_excluding.return_value = None
        validator._ontology_handler = mock_handler

        validator._embed_relation = AsyncMock(return_value=[0.1, 0.2, 0.3])

        families = await validator.infer_families("NEW_REL", "Description")

        assert families == ["elaborative"]

    @pytest.mark.asyncio
    async def test_default_family_when_embedding_fails(self, mock_settings):
        """Test default family when embedding fails."""
        from workflows.extract_relations.relation_validator import NewRelationValidator

        validator = NewRelationValidator(settings=mock_settings)

        mock_handler = Mock()
        validator._ontology_handler = mock_handler

        validator._embed_relation = AsyncMock(return_value=None)

        families = await validator.infer_families("NEW_REL", "Description")

        assert families == ["elaborative"]


class TestParallelDuplicateCheck:
    """Tests for parallel duplicate checking."""

    @pytest.fixture
    def mock_settings(self):
        """Create mock extraction settings."""
        settings = Mock()
        settings.EXTRACTION_MERGE_THRESHOLD = 0.8
        return settings

    @pytest.mark.asyncio
    async def test_parallel_check_finds_duplicates(self, mock_settings):
        """Test parallel duplicate check finds existing relations."""
        from workflows.extract_relations.relation_validator import NewRelationValidator

        validator = NewRelationValidator(settings=mock_settings)

        mock_handler = Mock()
        mock_handler.find_similar_excluding.side_effect = [
            ("EXISTING_1", 0.9),
            None,
            ("EXISTING_2", 0.85),
        ]
        validator._ontology_handler = mock_handler

        relations = [
            ("REL1", "Desc 1", [0.1, 0.2]),
            ("REL2", "Desc 2", [0.3, 0.4]),
            ("REL3", "Desc 3", [0.5, 0.6]),
        ]

        results = await validator.parallel_duplicate_check(relations)

        assert len(results) == 3
        assert results[0] == ("EXISTING_1", 0.9)
        assert results[1] == (None, 0.0)
        assert results[2] == ("EXISTING_2", 0.85)

    @pytest.mark.asyncio
    async def test_parallel_check_empty_input(self, mock_settings):
        """Test parallel check with empty input."""
        from workflows.extract_relations.relation_validator import NewRelationValidator

        validator = NewRelationValidator(settings=mock_settings)
        results = await validator.parallel_duplicate_check([])
        assert results == []

    @pytest.mark.asyncio
    async def test_parallel_check_no_handler(self, mock_settings):
        """Test parallel check without ontology handler."""
        from workflows.extract_relations.relation_validator import NewRelationValidator

        validator = NewRelationValidator(settings=mock_settings)
        validator._ontology_handler = None

        relations = [("REL1", "Desc 1", [0.1, 0.2])]
        results = await validator.parallel_duplicate_check(relations)

        assert len(results) == 1
        assert results[0] == (None, 0.0)


class TestValidationResultWithFamilies:
    """Tests for ValidationResult with families field."""

    def test_validation_result_has_families_field(self):
        """Test that ValidationResult has families field."""
        from workflows.extract_relations.relation_validator import ValidationResult

        result = ValidationResult(
            is_valid=True,
            final_relation_type="TEST_REL",
            final_description="Test description",
            families=["causal", "temporal"],
        )

        assert result.families == ["causal", "temporal"]

    def test_validation_result_families_default_none(self):
        """Test that families defaults to None."""
        from workflows.extract_relations.relation_validator import ValidationResult

        result = ValidationResult(
            is_valid=True,
            final_relation_type="TEST_REL",
            final_description="Test description",
        )

        assert result.families is None
