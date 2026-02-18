"""
Unit tests for extraction handlers.

Tests cover:
- ContentExtractionHandler.auto_merge() with ArangoDB Transaction
- StructureExtractionHandler.merge_structures() with ArangoDB Transaction
- Self-merge validation
- Error handling with transaction rollback

Run with: pytest tests/libs/arangodb/test_extraction_handlers.py -v
"""

from __future__ import annotations

import uuid
from unittest.mock import AsyncMock, MagicMock, patch

import pytest
from arango_client.handlers.content_extraction_handler import (
    ContentExtractionHandler,
    ExtractionDecision,
)
from arango_client.handlers.structure_extraction_handler import (
    StructureExtractionHandler,
)
from arango_client.models import ContentNode, StructureNode

# =============================================================================
# Fixtures
# =============================================================================


@pytest.fixture
def mock_txn():
    """Create a mock transaction object."""
    txn = MagicMock()
    txn.aql = MagicMock()
    txn.commit_transaction = MagicMock()
    txn.abort_transaction = MagicMock()
    return txn


@pytest.fixture
def mock_db(mock_txn):
    """Create a mock database connection with transaction support."""
    db = MagicMock()
    db.collection = MagicMock(return_value=MagicMock())
    db.aql = MagicMock()
    db.begin_transaction = MagicMock(return_value=mock_txn)
    return db


@pytest.fixture
def mock_settings():
    """Create mock extraction settings."""
    settings = MagicMock()
    settings.EXTRACTION_AUTO_MERGE_THRESHOLD = 0.72
    settings.EXTRACTION_LLM_THRESHOLD = 0.4
    settings.EXTRACTION_BATCH_SIZE = 100
    settings.EXTRACTION_FALLBACK_RELATION = "SIMILAR_TO"
    return settings


@pytest.fixture
def content_handler(mock_db, mock_settings):
    """Create ContentExtractionHandler with mocked dependencies."""
    handler = ContentExtractionHandler(mock_db, settings=mock_settings)
    return handler


@pytest.fixture
def structure_handler(mock_db, mock_settings):
    """Create StructureExtractionHandler with mocked dependencies."""
    handler = StructureExtractionHandler(mock_db, settings=mock_settings)
    return handler


@pytest.fixture
def sample_content_nodes():
    """Create sample content nodes for testing."""
    source = ContentNode(
        id=uuid.uuid4(),
        node_type="paragraph",
        attributes={"text": "Source content"},
        is_reviewed=False,
    )
    target = ContentNode(
        id=uuid.uuid4(),
        node_type="paragraph",
        attributes={"text": "Target content"},
        is_reviewed=True,
    )
    return source, target


@pytest.fixture
def sample_structure_nodes():
    """Create sample structure nodes for testing."""
    source = StructureNode(
        id=uuid.uuid4(),
        level=2,
        attributes={"title": "Source Chapter"},
        is_reviewed=False,
    )
    target = StructureNode(
        id=uuid.uuid4(),
        level=2,
        attributes={"title": "Target Chapter"},
        is_reviewed=True,
    )
    return source, target


# =============================================================================
# ContentExtractionHandler Tests
# =============================================================================


class TestContentExtractionHandlerAutoMerge:
    """Tests for ContentExtractionHandler.auto_merge() with ArangoDB Transaction."""

    @pytest.mark.asyncio
    async def test_self_merge_prevention(self, content_handler, sample_content_nodes):
        """Test that merging a node with itself is prevented."""
        source, _ = sample_content_nodes

        # Try to merge node with itself
        result = await content_handler.auto_merge(source, source)

        assert result.decision == ExtractionDecision.KEEP_SEPARATE
        assert result.target_node_id is None
        assert "Cannot merge node with itself" in result.reasoning

    @pytest.mark.asyncio
    async def test_auto_merge_success_with_transaction(
        self, content_handler, sample_content_nodes, mock_txn
    ):
        """Test successful auto merge using transaction."""
        source, target = sample_content_nodes

        # Mock successful transaction execution
        mock_cursor = MagicMock()
        mock_cursor.__iter__ = MagicMock(return_value=iter([1]))

        with patch("asyncio.to_thread", new_callable=AsyncMock) as mock_to_thread:
            # Return transaction object for begin_transaction, cursor for others
            mock_to_thread.return_value = mock_txn

            result = await content_handler.auto_merge(source, target)

            assert result.decision == ExtractionDecision.MERGE
            assert result.target_node_id == str(target.id)
            assert result.method == "embedding_auto_merge"

            # Transaction flow: begin + 5 queries + commit = 7 calls (includes merge log insert)
            assert mock_to_thread.call_count == 7

    @pytest.mark.asyncio
    async def test_auto_merge_transaction_rollback_on_error(
        self, content_handler, sample_content_nodes
    ):
        """Test that transaction is aborted on error."""
        source, target = sample_content_nodes

        # Mock transaction that fails on query execution
        mock_txn = MagicMock()
        mock_txn.aql.execute = MagicMock(side_effect=Exception("Query failed"))
        mock_txn.abort_transaction = MagicMock()

        call_count = [0]

        async def mock_to_thread_side_effect(*_args, **_kwargs):
            call_count[0] += 1
            if call_count[0] == 1:  # begin_transaction
                return mock_txn
            elif call_count[0] == 2:  # First query execution
                raise Exception("Query failed")
            else:
                return None

        with patch("asyncio.to_thread", side_effect=mock_to_thread_side_effect):
            with pytest.raises(Exception, match="Query failed"):
                await content_handler.auto_merge(source, target)


class TestContentExtractionHandlerDecision:
    """Tests for ContentExtractionHandler decision logic."""

    def test_decide_merge_high_similarity(self, content_handler):
        """Test decision for high similarity (>= auto_merge_threshold)."""
        result = content_handler.decide_action(0.85)
        assert result == ExtractionDecision.MERGE

    def test_decide_llm_medium_similarity(self, content_handler):
        """Test decision for medium similarity (>= llm_threshold)."""
        result = content_handler.decide_action(0.55)
        assert result == ExtractionDecision.CREATE_RELATION

    def test_decide_keep_separate_low_similarity(self, content_handler):
        """Test decision for low similarity (< llm_threshold)."""
        result = content_handler.decide_action(0.3)
        assert result == ExtractionDecision.KEEP_SEPARATE

    def test_decide_at_auto_merge_threshold(self, content_handler):
        """Test decision exactly at auto_merge_threshold."""
        result = content_handler.decide_action(0.72)
        assert result == ExtractionDecision.MERGE

    def test_decide_at_llm_threshold(self, content_handler):
        """Test decision exactly at llm_threshold."""
        result = content_handler.decide_action(0.4)
        assert result == ExtractionDecision.CREATE_RELATION


# =============================================================================
# StructureExtractionHandler Tests
# =============================================================================


class TestStructureExtractionHandlerMerge:
    """Tests for StructureExtractionHandler.merge_structures() with ArangoDB Transaction."""

    @pytest.mark.asyncio
    async def test_self_merge_prevention(self, structure_handler, sample_structure_nodes):
        """Test that merging a structure node with itself is prevented."""
        source, _ = sample_structure_nodes

        # Try to merge node with itself
        result = await structure_handler.merge_structures(source, source)

        assert result is False

    @pytest.mark.asyncio
    async def test_merge_structures_success_with_transaction(
        self, structure_handler, sample_structure_nodes, mock_txn
    ):
        """Test successful structure merge using transaction."""
        source, target = sample_structure_nodes

        # Mock successful transaction execution
        mock_cursor = MagicMock()
        mock_cursor.__iter__ = MagicMock(return_value=iter([1]))

        with patch("asyncio.to_thread", new_callable=AsyncMock) as mock_to_thread:
            mock_to_thread.return_value = mock_txn

            result = await structure_handler.merge_structures(source, target)

            assert result is True

            # Transaction flow: begin + 5 queries + commit = 7 calls
            assert mock_to_thread.call_count == 7

    @pytest.mark.asyncio
    async def test_merge_structures_transaction_rollback_on_error(
        self, structure_handler, sample_structure_nodes
    ):
        """Test that transaction is aborted on error and returns False."""
        source, target = sample_structure_nodes

        # Mock transaction that fails on query execution
        mock_txn = MagicMock()
        mock_txn.aql.execute = MagicMock(side_effect=Exception("Query failed"))
        mock_txn.abort_transaction = MagicMock()

        call_count = [0]

        async def mock_to_thread_side_effect(*_args, **_kwargs):
            call_count[0] += 1
            if call_count[0] == 1:  # begin_transaction
                return mock_txn
            elif call_count[0] == 2:  # First query execution
                raise Exception("Query failed")
            else:
                return None

        with patch("asyncio.to_thread", side_effect=mock_to_thread_side_effect):
            result = await structure_handler.merge_structures(source, target)

            # Should return False on error (not raise)
            assert result is False


class TestStructureExtractionHandlerTitleNormalization:
    """Tests for title normalization and matching."""

    def test_normalize_vietnamese(self, structure_handler):
        """Test Vietnamese diacritic normalization."""
        assert structure_handler._normalize_vietnamese("Chương") == "chuong"
        assert structure_handler._normalize_vietnamese("Đạo hàm") == "dao ham"
        assert structure_handler._normalize_vietnamese("VIỆT NAM") == "viet nam"

    def test_normalize_title_removes_prefix(self, structure_handler):
        """Test that common prefixes are removed."""
        result1 = structure_handler._normalize_title("Chương 1: Tuyển sinh")
        assert "tuyen sinh" in result1 or "sinh" in result1

        result2 = structure_handler._normalize_title("Section 2: Content")
        assert "content" in result2

    def test_hash_title_consistency(self, structure_handler):
        """Test that same titles produce same hash."""
        hash1 = structure_handler._hash_title("Chapter 1: Introduction")
        hash2 = structure_handler._hash_title("Chapter 1: Introduction")
        assert hash1 == hash2

    def test_hash_title_different_for_different_titles(self, structure_handler):
        """Test that different titles produce different hashes."""
        hash1 = structure_handler._hash_title("Chapter 1")
        hash2 = structure_handler._hash_title("Chapter 2")
        assert hash1 != hash2


# =============================================================================
# ArangoDB Transaction Strategy Tests
# =============================================================================


class TestTransactionStrategy:
    """Tests to verify ArangoDB Transaction implementation."""

    @pytest.mark.asyncio
    async def test_transaction_commit_on_success(self, content_handler, sample_content_nodes):
        """Test that transaction is committed on successful merge."""
        source, target = sample_content_nodes
        commit_called = [False]

        mock_txn = MagicMock()
        mock_txn.aql = MagicMock()

        def mark_commit():
            commit_called[0] = True

        mock_txn.commit_transaction = mark_commit

        call_count = [0]

        async def mock_to_thread(*_args, **_kwargs):
            call_count[0] += 1
            if call_count[0] == 1:  # begin_transaction
                return mock_txn
            elif call_count[0] == 6:  # commit_transaction
                mark_commit()
                return None
            else:
                return MagicMock(__iter__=lambda _: iter([1]))

        with patch("asyncio.to_thread", side_effect=mock_to_thread):
            result = await content_handler.auto_merge(source, target)

            assert result.decision == ExtractionDecision.MERGE
            assert commit_called[0] is True

    @pytest.mark.asyncio
    async def test_transaction_abort_on_failure(self, content_handler, sample_content_nodes):
        """Test that transaction is aborted on failure."""
        source, target = sample_content_nodes
        abort_called = [False]

        mock_txn = MagicMock()
        mock_txn.aql = MagicMock()

        def mark_abort():
            abort_called[0] = True

        mock_txn.abort_transaction = mark_abort

        call_count = [0]

        async def mock_to_thread(*_args, **_kwargs):
            call_count[0] += 1
            if call_count[0] == 1:  # begin_transaction
                return mock_txn
            elif call_count[0] == 3:  # Fail on third call (a query)
                raise Exception("Simulated failure")
            elif "abort" in str(_args):
                mark_abort()
                return None
            else:
                return MagicMock(__iter__=lambda _: iter([1]))

        with patch("asyncio.to_thread", side_effect=mock_to_thread):
            with pytest.raises(Exception, match="Simulated failure"):
                await content_handler.auto_merge(source, target)

    @pytest.mark.asyncio
    async def test_no_partial_state_on_failure(self, structure_handler, sample_structure_nodes):
        """Test that no partial state remains after transaction abort."""
        source, target = sample_structure_nodes

        # Track if abort was called
        abort_called = [False]

        mock_txn = MagicMock()

        def mark_abort():
            abort_called[0] = True

        mock_txn.abort_transaction = mark_abort

        call_count = [0]

        async def mock_to_thread(*_args, **_kwargs):
            call_count[0] += 1
            if call_count[0] == 1:  # begin_transaction
                return mock_txn
            elif call_count[0] == 4:  # Fail mid-transaction
                raise Exception("Mid-transaction failure")
            else:
                return MagicMock(__iter__=lambda _: iter([1]))

        with patch("asyncio.to_thread", side_effect=mock_to_thread):
            result = await structure_handler.merge_structures(source, target)

            # Should return False (transaction aborted)
            assert result is False
