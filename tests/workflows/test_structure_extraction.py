"""Unit tests for Structure extraction changes in extract_relations workflow.

This test file validates:
1. _process_structure_llm_batch passes embeddings to NodePair
2. _collect_structure_cross_batch_pairs categorizes pairs correctly
3. _process_structure_cross_batch_llm_batch handles LLM results
4. _process_llm_batch (Content tier) passes embeddings to NodePair
5. STRUCTURALLY_EQUIVALENT is created when LLM says should_merge
6. LLM-determined relation type is used (not SIMILAR_TO)
7. Nodes are kept separate when LLM returns relation_type=null
"""

from __future__ import annotations

import uuid
from unittest.mock import AsyncMock, MagicMock

import pytest
from arango_client.handlers import ExtractionDecision

from workflows.extract_relations.intra_batch_dedup import RelationCandidate
from workflows.extract_relations.llm_extractor import ExtractionResult, NodePair
from workflows.extract_relations.models import (
    ExtractRelationsOutput,
    TierStats,
)
from workflows.extract_relations.workflow import ExtractRelationsWorkflow

# =========================================================================
# Test Fixtures
# =========================================================================


@pytest.fixture
def mock_settings() -> MagicMock:
    """Create mock extraction settings."""
    settings = MagicMock()
    settings.EXTRACTION_AUTO_MERGE_THRESHOLD = 0.72
    settings.EXTRACTION_LLM_THRESHOLD = 0.40
    settings.EXTRACTION_LLM_BATCH_SIZE = 5
    settings.EXTRACTION_FALLBACK_RELATION = "SIMILAR_TO"
    settings.EXTRACTION_MAX_CONCURRENT_COLLECTIONS = 5
    settings.MAX_CONCURRENT_LLM_CALLS = 3
    return settings


@pytest.fixture
def mock_ontology() -> MagicMock:
    """Create mock ontology."""
    ontology = MagicMock()
    ontology.to_llm_prompt_section.return_value = "Mock ontology section"
    return ontology


@pytest.fixture
def mock_db() -> MagicMock:
    """Create mock database."""
    db = MagicMock()
    return db


@pytest.fixture
def sample_embedding() -> list[float]:
    """Return a sample embedding vector."""
    return [0.1, 0.2, 0.3, 0.4, 0.5] * 256  # 1280 dimensions


@pytest.fixture
def sample_embedding_2() -> list[float]:
    """Return a different sample embedding vector."""
    return [0.2, 0.3, 0.4, 0.5, 0.6] * 256


def create_mock_structure_node(
    node_id: str | None = None,
    title: str = "Test Title",
    level: int = 2,
    embedding: list[float] | None = None,
    source_id: str | None = None,
) -> MagicMock:
    """Create a mock StructureNode for testing."""
    node = MagicMock()
    node.id = uuid.UUID(node_id) if node_id else uuid.uuid4()
    node.level = level
    node.embedding = embedding
    node.source_id = uuid.UUID(source_id) if source_id else uuid.uuid4()

    # Mock attributes - use actual string values, not MagicMock
    attrs = MagicMock()
    attrs.title = title
    attrs.code = ""  # Add code attribute for _get_structure_text
    node.attributes = attrs

    # Add summary attribute directly on node (not on attributes)
    node.summary = None

    return node


def create_mock_content_node(
    node_id: str | None = None,
    content: str = "Test Content",
    node_type: str = "paragraph",
    embedding: list[float] | None = None,
) -> MagicMock:
    """Create a mock ContentNode for testing."""
    node = MagicMock()
    node.id = uuid.UUID(node_id) if node_id else uuid.uuid4()
    node.embedding = embedding
    node.node_type = MagicMock()
    node.node_type.value = node_type

    # Mock attributes
    attrs = MagicMock()
    attrs.summary = None
    attrs.content = content
    attrs.original_text = None
    node.attributes = attrs

    return node


# =========================================================================
# Test: _process_structure_llm_batch passes embeddings to NodePair
# =========================================================================


class TestProcessStructureLLMBatchEmbeddings:
    """Tests for embedding passing in _process_structure_llm_batch."""

    @pytest.mark.asyncio
    async def test_embeddings_passed_to_node_pair_when_available(
        self,
        mock_settings: MagicMock,
        mock_ontology: MagicMock,
        mock_db: MagicMock,
        sample_embedding: list[float],
        sample_embedding_2: list[float],
    ) -> None:
        """Test that embeddings are passed to NodePair when nodes have embeddings."""
        # Arrange
        workflow = ExtractRelationsWorkflow(
            db=mock_db,
            settings=mock_settings,
            ontology=mock_ontology,
        )

        # Create mock nodes with embeddings
        node_a = create_mock_structure_node(
            node_id="11111111-1111-1111-1111-111111111111",
            title="Chapter 1",
            embedding=sample_embedding,
        )
        node_b = create_mock_structure_node(
            node_id="22222222-2222-2222-2222-222222222222",
            title="Chapter 2",
            embedding=sample_embedding_2,
        )

        nodes = [node_a, node_b]
        candidates = [RelationCandidate(node_a_idx=0, node_b_idx=1, similarity=0.55)]

        tier_stats = TierStats(tier_name="tier_1_structure")
        output = ExtractRelationsOutput()

        # Mock LLM extractor
        mock_result = ExtractionResult(
            should_merge=False,
            relation_type="REFERS_TO",
            confidence=0.8,
        )
        workflow.llm_extractor = MagicMock()
        workflow.llm_extractor.verify_batches_parallel = AsyncMock(return_value=[mock_result])

        # Mock structure handler
        workflow.structure_handler = MagicMock()
        workflow.structure_handler.create_structure_relation = AsyncMock()

        # Act
        await workflow._process_structure_llm_batch(
            candidates=candidates,
            nodes=nodes,
            dry_run=False,
            tier_stats=tier_stats,
            output=output,
        )

        # Assert - verify that verify_batch was called
        workflow.llm_extractor.verify_batches_parallel.assert_called_once()
        call_args = workflow.llm_extractor.verify_batches_parallel.call_args

        # Extract the node_pairs from the call (batches is list of lists)
        batches = call_args[0][0]  # First positional argument is list of batches
        node_pairs = batches[0]  # First batch

        # Verify embeddings are passed to NodePair
        assert len(node_pairs) == 1
        pair = node_pairs[0]
        assert pair.node_a_embedding == sample_embedding
        assert pair.node_b_embedding == sample_embedding_2

    @pytest.mark.asyncio
    async def test_embeddings_none_when_nodes_lack_embeddings(
        self,
        mock_settings: MagicMock,
        mock_ontology: MagicMock,
        mock_db: MagicMock,
    ) -> None:
        """Test that NodePair gets None embeddings when nodes don't have embeddings."""
        # Arrange
        workflow = ExtractRelationsWorkflow(
            db=mock_db,
            settings=mock_settings,
            ontology=mock_ontology,
        )

        # Create mock nodes without embeddings
        node_a = create_mock_structure_node(
            node_id="11111111-1111-1111-1111-111111111111",
            title="Chapter 1",
            embedding=None,
        )
        node_b = create_mock_structure_node(
            node_id="22222222-2222-2222-2222-222222222222",
            title="Chapter 2",
            embedding=None,
        )

        nodes = [node_a, node_b]
        candidates = [RelationCandidate(node_a_idx=0, node_b_idx=1, similarity=0.55)]

        tier_stats = TierStats(tier_name="tier_1_structure")
        output = ExtractRelationsOutput()

        # Mock LLM extractor
        mock_result = ExtractionResult(
            should_merge=False,
            relation_type="REFERS_TO",
            confidence=0.8,
        )
        workflow.llm_extractor = MagicMock()
        workflow.llm_extractor.verify_batches_parallel = AsyncMock(return_value=[mock_result])

        # Mock structure handler
        workflow.structure_handler = MagicMock()
        workflow.structure_handler.create_structure_relation = AsyncMock()

        # Act
        await workflow._process_structure_llm_batch(
            candidates=candidates,
            nodes=nodes,
            dry_run=False,
            tier_stats=tier_stats,
            output=output,
        )

        # Assert
        workflow.llm_extractor.verify_batches_parallel.assert_called_once()
        call_args = workflow.llm_extractor.verify_batches_parallel.call_args
        batches = call_args[0][0]  # First positional argument is list of batches
        node_pairs = batches[0]  # First batch

        # Verify embeddings are None in NodePair
        assert len(node_pairs) == 1
        pair = node_pairs[0]
        assert pair.node_a_embedding is None
        assert pair.node_b_embedding is None


# =========================================================================
# Test: _collect_structure_cross_batch_pairs categorizes pairs correctly
# =========================================================================


class TestCollectStructureCrossBatchPairs:
    """Tests for _collect_structure_cross_batch_pairs pair categorization."""

    @pytest.mark.asyncio
    async def test_medium_similarity_goes_to_llm_pairs(
        self,
        mock_settings: MagicMock,
        mock_ontology: MagicMock,
        mock_db: MagicMock,
        sample_embedding: list[float],
        sample_embedding_2: list[float],
    ) -> None:
        """Test that medium similarity (0.40-0.72) pairs go to llm_pairs."""
        workflow = ExtractRelationsWorkflow(
            db=mock_db,
            settings=mock_settings,
            ontology=mock_ontology,
        )

        node = create_mock_structure_node(
            node_id="11111111-1111-1111-1111-111111111111",
            title="Chapter 1",
            embedding=sample_embedding,
        )

        similar_node = create_mock_structure_node(
            node_id="22222222-2222-2222-2222-222222222222",
            title="Related Chapter",
            embedding=sample_embedding_2,
        )

        mock_similar = MagicMock()
        mock_similar.node_id = str(similar_node.id)
        mock_similar.node = similar_node
        mock_similar.similarity_score = 0.55  # Medium

        workflow.structure_handler = MagicMock()
        workflow.structure_handler.find_similar_structures = AsyncMock(return_value=[mock_similar])
        workflow.structure_handler.decide_action = MagicMock(
            return_value=ExtractionDecision.CREATE_RELATION
        )

        high_sim, llm, kept = await workflow._collect_structure_cross_batch_pairs(
            structures=[node],
            processed_ids=set(),
            cross_source=True,
        )

        assert len(high_sim) == 0
        assert len(llm) == 1
        assert llm[0][0] is node
        assert llm[0][1] is similar_node
        assert llm[0][2] == 0.55

    @pytest.mark.asyncio
    async def test_high_similarity_goes_to_high_sim_pairs(
        self,
        mock_settings: MagicMock,
        mock_ontology: MagicMock,
        mock_db: MagicMock,
        sample_embedding: list[float],
    ) -> None:
        """Test that high similarity (>=0.72) pairs go to high_sim_pairs."""
        workflow = ExtractRelationsWorkflow(
            db=mock_db,
            settings=mock_settings,
            ontology=mock_ontology,
        )

        node = create_mock_structure_node(
            node_id="11111111-1111-1111-1111-111111111111",
            title="Chapter 1",
            embedding=sample_embedding,
        )

        similar_node = create_mock_structure_node(
            node_id="22222222-2222-2222-2222-222222222222",
            title="Chapter 1 Copy",
            embedding=sample_embedding,
        )

        mock_similar = MagicMock()
        mock_similar.node_id = str(similar_node.id)
        mock_similar.node = similar_node
        mock_similar.similarity_score = 0.85  # High

        workflow.structure_handler = MagicMock()
        workflow.structure_handler.find_similar_structures = AsyncMock(return_value=[mock_similar])
        workflow.structure_handler.decide_action = MagicMock(return_value=ExtractionDecision.MERGE)

        high_sim, llm, kept = await workflow._collect_structure_cross_batch_pairs(
            structures=[node],
            processed_ids=set(),
            cross_source=True,
        )

        assert len(high_sim) == 1
        assert len(llm) == 0
        assert high_sim[0][2] == 0.85

    @pytest.mark.asyncio
    async def test_no_similar_nodes_kept_separate(
        self,
        mock_settings: MagicMock,
        mock_ontology: MagicMock,
        mock_db: MagicMock,
        sample_embedding: list[float],
    ) -> None:
        """Test that nodes with no similar results are counted as kept_separate."""
        workflow = ExtractRelationsWorkflow(
            db=mock_db,
            settings=mock_settings,
            ontology=mock_ontology,
        )

        node = create_mock_structure_node(
            node_id="11111111-1111-1111-1111-111111111111",
            title="Unique Chapter",
            embedding=sample_embedding,
        )

        workflow.structure_handler = MagicMock()
        workflow.structure_handler.find_similar_structures = AsyncMock(return_value=[])

        high_sim, llm, kept = await workflow._collect_structure_cross_batch_pairs(
            structures=[node],
            processed_ids=set(),
            cross_source=True,
        )

        assert len(high_sim) == 0
        assert len(llm) == 0
        assert kept == 1


# =========================================================================
# Test: _process_structure_cross_batch_llm_batch handles LLM results
# =========================================================================


class TestProcessStructureCrossBatchLLMBatch:
    """Tests for _process_structure_cross_batch_llm_batch."""

    @pytest.mark.asyncio
    async def test_structurally_equivalent_created_when_llm_says_merge(
        self,
        mock_settings: MagicMock,
        mock_ontology: MagicMock,
        mock_db: MagicMock,
        sample_embedding: list[float],
        sample_embedding_2: list[float],
    ) -> None:
        """Test that STRUCTURALLY_EQUIVALENT is created when LLM returns should_merge=True."""
        workflow = ExtractRelationsWorkflow(
            db=mock_db,
            settings=mock_settings,
            ontology=mock_ontology,
        )

        node = create_mock_structure_node(
            node_id="11111111-1111-1111-1111-111111111111",
            title="Chapter 1",
            embedding=sample_embedding,
        )

        similar_node = create_mock_structure_node(
            node_id="22222222-2222-2222-2222-222222222222",
            title="Chapter 1 Copy",
            embedding=sample_embedding_2,
        )

        workflow.structure_handler = MagicMock()
        workflow.structure_handler.create_structurally_equivalent_relation = AsyncMock()
        workflow.structure_handler.mark_as_reviewed = AsyncMock()

        mock_llm_result = ExtractionResult(
            should_merge=True,
            relation_type=None,
            confidence=0.9,
        )
        workflow.llm_extractor = MagicMock()
        workflow.llm_extractor.verify_batches_parallel = AsyncMock(return_value=[mock_llm_result])

        output = ExtractRelationsOutput()
        tier_stats = TierStats(tier_name="tier_1_structure")
        reviewed_ids: set[str] = set()

        await workflow._process_structure_cross_batch_llm_batch(
            pairs=[(node, similar_node, 0.55)],
            dry_run=False,
            reviewed_ids=reviewed_ids,
            output=output,
            tier_stats=tier_stats,
        )

        workflow.structure_handler.create_structurally_equivalent_relation.assert_called_once_with(
            source=node,
            target=similar_node,
            similarity=0.55,
        )
        assert output.structure_relations_created == 1
        assert str(node.id) in reviewed_ids

    @pytest.mark.asyncio
    async def test_llm_determined_relation_type_used(
        self,
        mock_settings: MagicMock,
        mock_ontology: MagicMock,
        mock_db: MagicMock,
        sample_embedding: list[float],
        sample_embedding_2: list[float],
    ) -> None:
        """Test that LLM-determined relation type is used (not hardcoded SIMILAR_TO)."""
        workflow = ExtractRelationsWorkflow(
            db=mock_db,
            settings=mock_settings,
            ontology=mock_ontology,
        )

        node = create_mock_structure_node(
            node_id="11111111-1111-1111-1111-111111111111",
            title="Chapter on Calculus",
            embedding=sample_embedding,
        )

        similar_node = create_mock_structure_node(
            node_id="22222222-2222-2222-2222-222222222222",
            title="Chapter on Derivatives",
            embedding=sample_embedding_2,
        )

        workflow.structure_handler = MagicMock()
        workflow.structure_handler.create_structure_relation = AsyncMock()
        workflow.structure_handler.mark_as_reviewed = AsyncMock()

        mock_llm_result = ExtractionResult(
            should_merge=False,
            relation_type="PREREQUISITE_OF",
            confidence=0.85,
            is_in_top_k=True,
        )
        workflow.llm_extractor = MagicMock()
        workflow.llm_extractor.verify_batches_parallel = AsyncMock(return_value=[mock_llm_result])

        output = ExtractRelationsOutput()
        tier_stats = TierStats(tier_name="tier_1_structure")
        reviewed_ids: set[str] = set()

        await workflow._process_structure_cross_batch_llm_batch(
            pairs=[(node, similar_node, 0.55)],
            dry_run=False,
            reviewed_ids=reviewed_ids,
            output=output,
            tier_stats=tier_stats,
        )

        workflow.structure_handler.create_structure_relation.assert_called_once_with(
            source=node,
            target=similar_node,
            relation_type="PREREQUISITE_OF",
            confidence=0.85,
            method="llm_verified",
        )
        assert output.structure_relations_created == 1
        assert tier_stats.relations_in_top_k == 1

    @pytest.mark.asyncio
    async def test_nodes_kept_separate_when_llm_returns_null_relation(
        self,
        mock_settings: MagicMock,
        mock_ontology: MagicMock,
        mock_db: MagicMock,
        sample_embedding: list[float],
        sample_embedding_2: list[float],
    ) -> None:
        """Test that nodes are kept separate when LLM returns relation_type=null."""
        workflow = ExtractRelationsWorkflow(
            db=mock_db,
            settings=mock_settings,
            ontology=mock_ontology,
        )

        node = create_mock_structure_node(
            node_id="11111111-1111-1111-1111-111111111111",
            title="Chapter on Biology",
            embedding=sample_embedding,
        )

        similar_node = create_mock_structure_node(
            node_id="22222222-2222-2222-2222-222222222222",
            title="Chapter on Physics",
            embedding=sample_embedding_2,
        )

        workflow.structure_handler = MagicMock()
        workflow.structure_handler.create_structure_relation = AsyncMock()
        workflow.structure_handler.create_structurally_equivalent_relation = AsyncMock()

        mock_llm_result = ExtractionResult(
            should_merge=False,
            relation_type=None,
            confidence=0.7,
        )
        workflow.llm_extractor = MagicMock()
        workflow.llm_extractor.verify_batches_parallel = AsyncMock(return_value=[mock_llm_result])

        output = ExtractRelationsOutput()
        tier_stats = TierStats(tier_name="tier_1_structure")
        reviewed_ids: set[str] = set()

        await workflow._process_structure_cross_batch_llm_batch(
            pairs=[(node, similar_node, 0.45)],
            dry_run=False,
            reviewed_ids=reviewed_ids,
            output=output,
            tier_stats=tier_stats,
        )

        workflow.structure_handler.create_structure_relation.assert_not_called()
        workflow.structure_handler.create_structurally_equivalent_relation.assert_not_called()
        assert output.structure_relations_created == 0


# =========================================================================
# Test: _process_llm_batch (Content tier) passes embeddings to NodePair
# =========================================================================


class TestProcessLLMBatchContentTierEmbeddings:
    """Tests for embedding passing in _process_llm_batch (Content tier)."""

    @pytest.mark.asyncio
    async def test_content_tier_embeddings_passed_to_node_pair(
        self,
        mock_settings: MagicMock,
        mock_ontology: MagicMock,
        mock_db: MagicMock,
        sample_embedding: list[float],
        sample_embedding_2: list[float],
    ) -> None:
        """Test that embeddings are passed to NodePair in content tier LLM batch."""
        # Arrange
        workflow = ExtractRelationsWorkflow(
            db=mock_db,
            settings=mock_settings,
            ontology=mock_ontology,
        )

        # Create mock content nodes with embeddings
        node_a = create_mock_content_node(
            node_id="11111111-1111-1111-1111-111111111111",
            content="Content about calculus",
            embedding=sample_embedding,
        )
        node_b = create_mock_content_node(
            node_id="22222222-2222-2222-2222-222222222222",
            content="Content about derivatives",
            embedding=sample_embedding_2,
        )

        pairs = [(node_a, node_b, 0.55)]

        processed_ids: set[str] = set()
        reviewed_ids: set[str] = set()
        output = ExtractRelationsOutput()
        tier_stats = TierStats(tier_name="tier_2_content")

        # Mock LLM extractor
        mock_result = ExtractionResult(
            should_merge=False,
            relation_type="RELATED_TO",
            confidence=0.8,
        )
        workflow.llm_extractor = MagicMock()
        workflow.llm_extractor.verify_batches_parallel = AsyncMock(return_value=[mock_result])

        # Mock content handler
        workflow.content_handler = MagicMock()
        workflow.content_handler.node_exists = AsyncMock(return_value=True)
        workflow.content_handler.create_relation_edge = AsyncMock()
        workflow.content_handler.mark_as_reviewed = AsyncMock()

        # Act
        await workflow._process_llm_batch(
            pairs=pairs,
            dry_run=False,
            processed_ids=processed_ids,
            reviewed_ids=reviewed_ids,
            output=output,
            tier_stats=tier_stats,
        )

        # Assert - verify embeddings were passed to NodePair
        workflow.llm_extractor.verify_batches_parallel.assert_called_once()
        call_args = workflow.llm_extractor.verify_batches_parallel.call_args
        batches = call_args[0][0]  # First positional argument is list of batches
        node_pairs = batches[0]  # First batch

        assert len(node_pairs) == 1
        pair = node_pairs[0]
        assert pair.node_a_embedding == sample_embedding
        assert pair.node_b_embedding == sample_embedding_2

    @pytest.mark.asyncio
    async def test_content_tier_none_embeddings_when_missing(
        self,
        mock_settings: MagicMock,
        mock_ontology: MagicMock,
        mock_db: MagicMock,
    ) -> None:
        """Test that None embeddings are passed when content nodes lack embeddings."""
        # Arrange
        workflow = ExtractRelationsWorkflow(
            db=mock_db,
            settings=mock_settings,
            ontology=mock_ontology,
        )

        # Create mock content nodes without embeddings
        node_a = create_mock_content_node(
            node_id="11111111-1111-1111-1111-111111111111",
            content="Content A",
            embedding=None,
        )
        node_b = create_mock_content_node(
            node_id="22222222-2222-2222-2222-222222222222",
            content="Content B",
            embedding=None,
        )

        pairs = [(node_a, node_b, 0.55)]

        processed_ids: set[str] = set()
        reviewed_ids: set[str] = set()
        output = ExtractRelationsOutput()
        tier_stats = TierStats(tier_name="tier_2_content")

        # Mock LLM extractor
        mock_result = ExtractionResult(
            should_merge=False,
            relation_type="RELATED_TO",
            confidence=0.8,
        )
        workflow.llm_extractor = MagicMock()
        workflow.llm_extractor.verify_batches_parallel = AsyncMock(return_value=[mock_result])

        # Mock content handler
        workflow.content_handler = MagicMock()
        workflow.content_handler.node_exists = AsyncMock(return_value=True)
        workflow.content_handler.create_relation_edge = AsyncMock()
        workflow.content_handler.mark_as_reviewed = AsyncMock()

        # Act
        await workflow._process_llm_batch(
            pairs=pairs,
            dry_run=False,
            processed_ids=processed_ids,
            reviewed_ids=reviewed_ids,
            output=output,
            tier_stats=tier_stats,
        )

        # Assert
        workflow.llm_extractor.verify_batches_parallel.assert_called_once()
        call_args = workflow.llm_extractor.verify_batches_parallel.call_args
        batches = call_args[0][0]  # First positional argument is list of batches
        node_pairs = batches[0]  # First batch

        assert len(node_pairs) == 1
        pair = node_pairs[0]
        assert pair.node_a_embedding is None
        assert pair.node_b_embedding is None


# =========================================================================
# Test: _process_structure_llm_batch behavior for LLM results
# =========================================================================


class TestProcessStructureLLMBatchResults:
    """Tests for _process_structure_llm_batch handling of LLM results."""

    @pytest.mark.asyncio
    async def test_structurally_equivalent_on_should_merge(
        self,
        mock_settings: MagicMock,
        mock_ontology: MagicMock,
        mock_db: MagicMock,
        sample_embedding: list[float],
    ) -> None:
        """Test that STRUCTURALLY_EQUIVALENT is created when LLM says should_merge in batch."""
        # Arrange
        workflow = ExtractRelationsWorkflow(
            db=mock_db,
            settings=mock_settings,
            ontology=mock_ontology,
        )

        node_a = create_mock_structure_node(
            node_id="11111111-1111-1111-1111-111111111111",
            title="Chapter 1",
            embedding=sample_embedding,
        )
        node_b = create_mock_structure_node(
            node_id="22222222-2222-2222-2222-222222222222",
            title="Chapter 1 Copy",
            embedding=sample_embedding,
        )

        nodes = [node_a, node_b]
        candidates = [RelationCandidate(node_a_idx=0, node_b_idx=1, similarity=0.60)]

        tier_stats = TierStats(tier_name="tier_1_structure")
        output = ExtractRelationsOutput()

        # Mock LLM extractor - returns should_merge=True
        mock_result = ExtractionResult(
            should_merge=True,
            relation_type=None,
            confidence=0.95,
        )
        workflow.llm_extractor = MagicMock()
        workflow.llm_extractor.verify_batches_parallel = AsyncMock(return_value=[mock_result])

        # Mock structure handler
        workflow.structure_handler = MagicMock()
        workflow.structure_handler.create_structurally_equivalent_relation = AsyncMock()

        # Act
        await workflow._process_structure_llm_batch(
            candidates=candidates,
            nodes=nodes,
            dry_run=False,
            tier_stats=tier_stats,
            output=output,
        )

        # Assert
        workflow.structure_handler.create_structurally_equivalent_relation.assert_called_once_with(
            source=node_a,
            target=node_b,
            similarity=0.60,
        )
        assert output.structure_relations_created == 1

    @pytest.mark.asyncio
    async def test_llm_relation_type_used_in_batch(
        self,
        mock_settings: MagicMock,
        mock_ontology: MagicMock,
        mock_db: MagicMock,
        sample_embedding: list[float],
    ) -> None:
        """Test that LLM-determined relation type is used in batch processing."""
        # Arrange
        workflow = ExtractRelationsWorkflow(
            db=mock_db,
            settings=mock_settings,
            ontology=mock_ontology,
        )

        node_a = create_mock_structure_node(
            node_id="11111111-1111-1111-1111-111111111111",
            title="Chapter on Limits",
            embedding=sample_embedding,
        )
        node_b = create_mock_structure_node(
            node_id="22222222-2222-2222-2222-222222222222",
            title="Chapter on Continuity",
            embedding=sample_embedding,
        )

        nodes = [node_a, node_b]
        candidates = [RelationCandidate(node_a_idx=0, node_b_idx=1, similarity=0.55)]

        tier_stats = TierStats(tier_name="tier_1_structure")
        output = ExtractRelationsOutput()

        # Mock LLM extractor - returns specific relation type
        mock_result = ExtractionResult(
            should_merge=False,
            relation_type="PREREQUISITE_OF",
            confidence=0.85,
            is_in_top_k=True,
        )
        workflow.llm_extractor = MagicMock()
        workflow.llm_extractor.verify_batches_parallel = AsyncMock(return_value=[mock_result])

        # Mock structure handler
        workflow.structure_handler = MagicMock()
        workflow.structure_handler.create_structure_relation = AsyncMock()

        # Act
        await workflow._process_structure_llm_batch(
            candidates=candidates,
            nodes=nodes,
            dry_run=False,
            tier_stats=tier_stats,
            output=output,
        )

        # Assert
        workflow.structure_handler.create_structure_relation.assert_called_once_with(
            source=node_a,
            target=node_b,
            relation_type="PREREQUISITE_OF",
            confidence=0.85,
            method="llm_verified",
        )
        assert output.structure_relations_created == 1
        assert tier_stats.relations_in_top_k == 1

    @pytest.mark.asyncio
    async def test_no_relation_created_when_llm_returns_null(
        self,
        mock_settings: MagicMock,
        mock_ontology: MagicMock,
        mock_db: MagicMock,
        sample_embedding: list[float],
    ) -> None:
        """Test that no relation is created when LLM returns relation_type=null in batch."""
        # Arrange
        workflow = ExtractRelationsWorkflow(
            db=mock_db,
            settings=mock_settings,
            ontology=mock_ontology,
        )

        node_a = create_mock_structure_node(
            node_id="11111111-1111-1111-1111-111111111111",
            title="Chapter on Biology",
            embedding=sample_embedding,
        )
        node_b = create_mock_structure_node(
            node_id="22222222-2222-2222-2222-222222222222",
            title="Chapter on History",
            embedding=sample_embedding,
        )

        nodes = [node_a, node_b]
        candidates = [RelationCandidate(node_a_idx=0, node_b_idx=1, similarity=0.45)]

        tier_stats = TierStats(tier_name="tier_1_structure")
        output = ExtractRelationsOutput()

        # Mock LLM extractor - returns no relation
        mock_result = ExtractionResult(
            should_merge=False,
            relation_type=None,  # No relation
            confidence=0.7,
        )
        workflow.llm_extractor = MagicMock()
        workflow.llm_extractor.verify_batches_parallel = AsyncMock(return_value=[mock_result])

        # Mock structure handler
        workflow.structure_handler = MagicMock()
        workflow.structure_handler.create_structure_relation = AsyncMock()
        workflow.structure_handler.create_structurally_equivalent_relation = AsyncMock()

        # Act
        await workflow._process_structure_llm_batch(
            candidates=candidates,
            nodes=nodes,
            dry_run=False,
            tier_stats=tier_stats,
            output=output,
        )

        # Assert - no relation methods were called
        workflow.structure_handler.create_structure_relation.assert_not_called()
        workflow.structure_handler.create_structurally_equivalent_relation.assert_not_called()
        assert output.structure_relations_created == 0


# =========================================================================
# Test: High similarity creates STRUCTURALLY_EQUIVALENT without LLM
# =========================================================================


class TestHighSimilarityNoLLM:
    """Tests for high similarity (>=0.72) creating STRUCTURALLY_EQUIVALENT without LLM."""

    @pytest.mark.asyncio
    async def test_high_similarity_creates_structurally_equivalent_without_llm(
        self,
        mock_settings: MagicMock,
        mock_ontology: MagicMock,
        mock_db: MagicMock,
        sample_embedding: list[float],
    ) -> None:
        """Test that high similarity (>=0.72) goes to high_sim_pairs (no LLM needed)."""
        workflow = ExtractRelationsWorkflow(
            db=mock_db,
            settings=mock_settings,
            ontology=mock_ontology,
        )

        node = create_mock_structure_node(
            node_id="11111111-1111-1111-1111-111111111111",
            title="Chapter 1",
            embedding=sample_embedding,
        )

        similar_node = create_mock_structure_node(
            node_id="22222222-2222-2222-2222-222222222222",
            title="Chapter 1 Copy",
            embedding=sample_embedding,
        )

        mock_similar = MagicMock()
        mock_similar.node_id = str(similar_node.id)
        mock_similar.node = similar_node
        mock_similar.similarity_score = 0.85  # >= 0.72

        workflow.structure_handler = MagicMock()
        workflow.structure_handler.find_similar_structures = AsyncMock(return_value=[mock_similar])
        workflow.structure_handler.decide_action = MagicMock(return_value=ExtractionDecision.MERGE)

        high_sim, llm, kept = await workflow._collect_structure_cross_batch_pairs(
            structures=[node],
            processed_ids=set(),
            cross_source=True,
        )

        # High similarity should go to high_sim_pairs, NOT llm_pairs
        assert len(high_sim) == 1
        assert len(llm) == 0
        assert high_sim[0][0] is node
        assert high_sim[0][1] is similar_node
        assert high_sim[0][2] == 0.85


# =========================================================================
# Test: Dry run mode
# =========================================================================


class TestDryRunMode:
    """Tests for dry run mode behavior."""

    @pytest.mark.asyncio
    async def test_dry_run_does_not_create_relations(
        self,
        mock_settings: MagicMock,
        mock_ontology: MagicMock,
        mock_db: MagicMock,
        sample_embedding: list[float],
    ) -> None:
        """Test that dry run mode does not create actual relations."""
        # Arrange
        workflow = ExtractRelationsWorkflow(
            db=mock_db,
            settings=mock_settings,
            ontology=mock_ontology,
        )

        node_a = create_mock_structure_node(
            node_id="11111111-1111-1111-1111-111111111111",
            title="Chapter 1",
            embedding=sample_embedding,
        )
        node_b = create_mock_structure_node(
            node_id="22222222-2222-2222-2222-222222222222",
            title="Chapter 2",
            embedding=sample_embedding,
        )

        nodes = [node_a, node_b]
        candidates = [RelationCandidate(node_a_idx=0, node_b_idx=1, similarity=0.60)]

        tier_stats = TierStats(tier_name="tier_1_structure")
        output = ExtractRelationsOutput()

        # Mock LLM extractor
        mock_result = ExtractionResult(
            should_merge=True,
            relation_type=None,
            confidence=0.9,
        )
        workflow.llm_extractor = MagicMock()
        workflow.llm_extractor.verify_batches_parallel = AsyncMock(return_value=[mock_result])

        # Mock structure handler
        workflow.structure_handler = MagicMock()
        workflow.structure_handler.create_structurally_equivalent_relation = AsyncMock()

        # Act - dry_run=True
        await workflow._process_structure_llm_batch(
            candidates=candidates,
            nodes=nodes,
            dry_run=True,  # DRY RUN
            tier_stats=tier_stats,
            output=output,
        )

        # Assert - handler methods should NOT be called
        workflow.structure_handler.create_structurally_equivalent_relation.assert_not_called()

        # But output stats should still be updated
        assert output.structure_relations_created == 1


# =========================================================================
# Test: NodePair construction
# =========================================================================


class TestNodePairConstruction:
    """Tests for NodePair construction with embeddings."""

    def test_node_pair_with_embeddings(
        self,
        sample_embedding: list[float],
        sample_embedding_2: list[float],
    ) -> None:
        """Test that NodePair correctly stores embeddings."""
        pair = NodePair(
            node_a_id="test-a",
            node_a_type="structure",
            node_a_text="Test A",
            node_b_id="test-b",
            node_b_type="structure",
            node_b_text="Test B",
            similarity_score=0.55,
            node_a_embedding=sample_embedding,
            node_b_embedding=sample_embedding_2,
        )

        assert pair.node_a_embedding == sample_embedding
        assert pair.node_b_embedding == sample_embedding_2
        assert pair.similarity_score == 0.55

    def test_node_pair_without_embeddings(self) -> None:
        """Test that NodePair works with None embeddings."""
        pair = NodePair(
            node_a_id="test-a",
            node_a_type="structure",
            node_a_text="Test A",
            node_b_id="test-b",
            node_b_type="structure",
            node_b_text="Test B",
            similarity_score=0.55,
            node_a_embedding=None,
            node_b_embedding=None,
        )

        assert pair.node_a_embedding is None
        assert pair.node_b_embedding is None

    def test_node_pair_default_embeddings(self) -> None:
        """Test that NodePair defaults embeddings to None."""
        pair = NodePair(
            node_a_id="test-a",
            node_a_type="structure",
            node_a_text="Test A",
            node_b_id="test-b",
            node_b_type="structure",
            node_b_text="Test B",
            similarity_score=0.55,
            # embeddings not provided
        )

        assert pair.node_a_embedding is None
        assert pair.node_b_embedding is None


# =========================================================================
# Test: Intra-batch same-level filtering
# =========================================================================


class TestIntraBatchSameLevelFiltering:
    """Tests for same-level filtering in find_intra_batch_components."""

    def test_same_level_nodes_are_compared(
        self, sample_embedding: list[float], sample_embedding_2: list[float]
    ) -> None:
        """Test that nodes at the same level are compared for similarity."""
        from workflows.extract_relations.intra_batch_dedup import find_intra_batch_components

        # Create nodes at same level (level=2) with high similarity
        node_a = create_mock_structure_node(
            node_id="11111111-1111-1111-1111-111111111111",
            title="Chapter 1",
            level=2,
            embedding=sample_embedding,
        )
        node_b = create_mock_structure_node(
            node_id="22222222-2222-2222-2222-222222222222",
            title="Chapter 2",
            level=2,
            embedding=sample_embedding,  # Same embedding = high similarity
        )

        result = find_intra_batch_components(
            nodes=[node_a, node_b],
            merge_threshold=0.72,
            relation_threshold=0.40,
            allow_merge=False,  # Structure tier
        )

        # Should have high_similarity_candidates since same level
        assert len(result.high_similarity_candidates) == 1
        assert result.high_similarity_candidates[0].node_a_idx == 0
        assert result.high_similarity_candidates[0].node_b_idx == 1

    def test_different_level_nodes_not_compared(self, sample_embedding: list[float]) -> None:
        """Test that nodes at different levels are NOT compared."""
        from workflows.extract_relations.intra_batch_dedup import find_intra_batch_components

        # Create nodes at DIFFERENT levels with identical embeddings
        node_chu_de = create_mock_structure_node(
            node_id="11111111-1111-1111-1111-111111111111",
            title="CHỦ ĐỀ 1",
            level=1,  # Higher level (topic)
            embedding=sample_embedding,
        )
        node_bai = create_mock_structure_node(
            node_id="22222222-2222-2222-2222-222222222222",
            title="BÀI 3",
            level=2,  # Lower level (lesson)
            embedding=sample_embedding,  # Same embedding = would be high similarity
        )

        result = find_intra_batch_components(
            nodes=[node_chu_de, node_bai],
            merge_threshold=0.72,
            relation_threshold=0.40,
            allow_merge=False,
        )

        # Should have NO candidates since different levels
        assert len(result.high_similarity_candidates) == 0
        assert len(result.relation_candidates) == 0
        # Both nodes should be leaders (standalone)
        assert len(result.leaders) == 2

    def test_mixed_levels_only_same_level_compared(
        self, sample_embedding: list[float], sample_embedding_2: list[float]
    ) -> None:
        """Test that only same-level nodes are compared in a mixed batch."""
        from workflows.extract_relations.intra_batch_dedup import find_intra_batch_components

        # Create nodes at multiple levels
        node_l1_a = create_mock_structure_node(
            node_id="11111111-1111-1111-1111-111111111111",
            title="CHỦ ĐỀ 1",
            level=1,
            embedding=sample_embedding,
        )
        node_l1_b = create_mock_structure_node(
            node_id="22222222-2222-2222-2222-222222222222",
            title="CHỦ ĐỀ 2",
            level=1,
            embedding=sample_embedding,  # Same embedding = high similarity
        )
        node_l2_a = create_mock_structure_node(
            node_id="33333333-3333-3333-3333-333333333333",
            title="BÀI 1",
            level=2,
            embedding=sample_embedding_2,
        )
        node_l2_b = create_mock_structure_node(
            node_id="44444444-4444-4444-4444-444444444444",
            title="BÀI 2",
            level=2,
            embedding=sample_embedding_2,  # Same embedding = high similarity
        )

        result = find_intra_batch_components(
            nodes=[node_l1_a, node_l1_b, node_l2_a, node_l2_b],
            merge_threshold=0.72,
            relation_threshold=0.40,
            allow_merge=False,
        )

        # Should have 2 high_similarity_candidates (one for each level pair)
        assert len(result.high_similarity_candidates) == 2

        # Verify pairs are at same levels
        for candidate in result.high_similarity_candidates:
            node_a = [node_l1_a, node_l1_b, node_l2_a, node_l2_b][candidate.node_a_idx]
            node_b = [node_l1_a, node_l1_b, node_l2_a, node_l2_b][candidate.node_b_idx]
            assert node_a.level == node_b.level, (
                f"Mismatched levels: {node_a.level} != {node_b.level}"
            )

    def test_content_nodes_always_same_level(self, sample_embedding: list[float]) -> None:
        """Test that content nodes (level=0) are always compared."""
        from workflows.extract_relations.intra_batch_dedup import find_intra_batch_components

        # Content nodes always have level=0
        node_a = create_mock_content_node(
            node_id="11111111-1111-1111-1111-111111111111",
            content="Content A",
            embedding=sample_embedding,
        )
        node_a.level = 0  # Content nodes have level=0
        node_a.created_at = "2024-01-01T00:00:00Z"  # For sorting in Union-Find

        node_b = create_mock_content_node(
            node_id="22222222-2222-2222-2222-222222222222",
            content="Content B",
            embedding=sample_embedding,
        )
        node_b.level = 0
        node_b.created_at = "2024-01-01T00:00:01Z"

        result = find_intra_batch_components(
            nodes=[node_a, node_b],
            merge_threshold=0.72,
            relation_threshold=0.40,
            allow_merge=True,  # Content tier allows merge
        )

        # Should have merge operations since same level (0) and high similarity
        assert len(result.merge_operations) == 1
