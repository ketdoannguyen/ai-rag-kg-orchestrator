"""Tests for new relation validation flow.

Tests the refactored validation flow:
1. Check exact name in DB → Found? → MERGE
2. LLM Strong Verify + Extract FULL INFO
3. Embed (name + description)
4. Compare similarity >= 0.85? → MERGE
5. Store NEW relation to DB
"""

from __future__ import annotations

from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from workflows.extract_relations.relation_validator import (
    NewRelationValidator,
    NodeInfo,
    StrongLLMRelationOutput,
    ValidationResult,
)


@pytest.fixture
def mock_settings():
    """Mock extraction settings."""
    settings = MagicMock()
    settings.EXTRACTION_MERGE_THRESHOLD = 0.85
    settings.EXTRACTION_FALLBACK_RELATION = "RELATED_TO"
    settings.EXTRACTION_EMBEDDING_MODEL = "text-embedding-3-large"
    settings.EXTRACTION_STRONG_MODEL = "gpt-4o"
    return settings


@pytest.fixture
def mock_ontology_handler():
    """Mock ontology handler."""
    handler = MagicMock()
    handler.get_by_id.return_value = None
    handler.find_similar_excluding.return_value = None
    handler.create.return_value = None
    return handler


@pytest.fixture
def mock_llm_client():
    """Mock LLM client."""
    client = MagicMock()
    client.invoke = AsyncMock()
    client.embed = AsyncMock()
    return client


@pytest.fixture
def validator(mock_settings, mock_ontology_handler, mock_llm_client):
    """Create validator with mocks."""
    v = NewRelationValidator(
        settings=mock_settings,
        ontology_handler=mock_ontology_handler,
        llm_client=mock_llm_client,
    )
    return v


@pytest.fixture
def node_a():
    """Sample source node."""
    return NodeInfo(
        node_id="node_a",
        node_type="paragraph",
        text="Lý Thường Kiệt là con của Lý Công Uẩn.",
    )


@pytest.fixture
def node_b():
    """Sample target node."""
    return NodeInfo(
        node_id="node_b",
        node_type="paragraph",
        text="Lý Công Uẩn là vị vua sáng lập nhà Lý.",
    )


class TestStep1ExactNameMatch:
    """STEP 1: Check exact name in DB → Found? → MERGE."""

    @pytest.mark.asyncio
    async def test_exact_match_returns_merge(self, validator, node_a, node_b):
        """When relation exists in DB, return merged result."""
        # Setup: relation exists
        existing = MagicMock()
        existing.description = "Existing description"
        existing.tiers = ["tier_2_content"]
        validator._ontology_handler.get_by_id.return_value = existing

        result = await validator.validate_new_relation(
            node_a=node_a,
            node_b=node_b,
            proposed_relation="FATHER_OF",
            summary="Quan hệ cha con",
            tier="tier_2_content",
        )

        assert result.is_valid is True
        assert result.final_relation_type == "FATHER_OF"
        assert result.merged_with_existing is True
        assert result.existing_relation_id == "FATHER_OF"
        assert result.similarity_score == 1.0
        assert result.reasoning == "Exact match in DB"

    @pytest.mark.asyncio
    async def test_no_match_proceeds_to_step2(self, validator, node_a, node_b):
        """When relation not in DB, proceed to LLM Strong."""
        validator._ontology_handler.get_by_id.return_value = None

        # Mock LLM Strong response
        with patch.object(validator, "_verify_and_extract_with_strong_llm") as mock_strong:
            mock_strong.return_value = StrongLLMRelationOutput(
                accepted=True,
                reasoning="Valid relation",
                id="FATHER_OF",
                name="FATHER_OF",
                summary="Quan hệ cha con",
                description="Mô tả chi tiết về quan hệ cha con",
                source_node_types=["paragraph"],
                target_node_types=["paragraph"],
                is_symmetric=False,
                is_transitive=False,
                families=["kinship"],
                keywords=["cha", "con", "bố"],
            )

            with patch.object(validator, "_embed_relation") as mock_embed:
                mock_embed.return_value = [0.1] * 3072

                result = await validator.validate_new_relation(
                    node_a=node_a,
                    node_b=node_b,
                    proposed_relation="FATHER_OF",
                    summary="Quan hệ cha con",
                    tier="tier_2_content",
                )

                mock_strong.assert_called_once()
                assert result.is_valid is True


class TestStep2StrongLLMVerifyExtract:
    """STEP 2: LLM Strong Verify + Extract FULL INFO."""

    @pytest.mark.asyncio
    async def test_strong_llm_accepts_proposal(self, validator, node_a, node_b):
        """Strong LLM accepts proposed relation."""
        validator._ontology_handler.get_by_id.return_value = None

        with patch.object(validator, "_verify_and_extract_with_strong_llm") as mock_strong:
            mock_strong.return_value = StrongLLMRelationOutput(
                accepted=True,
                reasoning="Relation is valid",
                id="FATHER_OF",
                name="FATHER_OF",
                summary="Quan hệ cha con",
                description="Chi tiết về quan hệ cha con",
                source_node_types=["paragraph"],
                target_node_types=["paragraph"],
                is_symmetric=False,
                is_transitive=False,
                families=["kinship"],
                keywords=["cha", "con"],
            )

            with patch.object(validator, "_embed_relation") as mock_embed:
                mock_embed.return_value = [0.1] * 3072

                result = await validator.validate_new_relation(
                    node_a=node_a,
                    node_b=node_b,
                    proposed_relation="FATHER_OF",
                    summary="Quan hệ cha con",
                )

                assert result.final_relation_type == "FATHER_OF"
                assert result.families == ["kinship"]

    @pytest.mark.asyncio
    async def test_strong_llm_rejects_proposes_new(self, validator, node_a, node_b):
        """Strong LLM rejects and proposes different relation."""
        validator._ontology_handler.get_by_id.return_value = None

        with patch.object(validator, "_verify_and_extract_with_strong_llm") as mock_strong:
            # LLM rejects PARENT_OF, proposes FATHER_OF instead
            mock_strong.return_value = StrongLLMRelationOutput(
                accepted=False,
                reasoning="PARENT_OF too generic, FATHER_OF is more specific",
                id="FATHER_OF",
                name="FATHER_OF",
                summary="Quan hệ cha con",
                description="Chi tiết về quan hệ cha con",
                source_node_types=["paragraph"],
                target_node_types=["paragraph"],
                is_symmetric=False,
                is_transitive=False,
                families=["kinship"],
                keywords=["cha", "con"],
            )

            with patch.object(validator, "_embed_relation") as mock_embed:
                mock_embed.return_value = [0.1] * 3072

                result = await validator.validate_new_relation(
                    node_a=node_a,
                    node_b=node_b,
                    proposed_relation="PARENT_OF",  # Original proposal
                    summary="Quan hệ cha mẹ",
                )

                # Should use the new proposal from Strong LLM
                assert result.final_relation_type == "FATHER_OF"

    @pytest.mark.asyncio
    async def test_strong_llm_failure_returns_fallback(self, validator, node_a, node_b):
        """When Strong LLM fails, return fallback relation."""
        validator._ontology_handler.get_by_id.return_value = None

        with patch.object(validator, "_verify_and_extract_with_strong_llm") as mock_strong:
            mock_strong.return_value = None  # LLM failed

            result = await validator.validate_new_relation(
                node_a=node_a,
                node_b=node_b,
                proposed_relation="FATHER_OF",
                summary="Quan hệ cha con",
            )

            assert result.is_valid is False
            assert result.final_relation_type == "RELATED_TO"
            assert "LLM Strong failed" in result.reasoning


class TestStep4SimilarityMerge:
    """STEP 4: Compare similarity >= 0.85? → MERGE."""

    @pytest.mark.asyncio
    async def test_high_similarity_merges(self, validator, node_a, node_b):
        """Similarity >= 0.85 triggers merge."""
        validator._ontology_handler.get_by_id.return_value = None
        validator._ontology_handler.find_similar_excluding.return_value = (
            "PARENT_OF",
            0.92,
        )

        with patch.object(validator, "_verify_and_extract_with_strong_llm") as mock_strong:
            mock_strong.return_value = StrongLLMRelationOutput(
                accepted=True,
                reasoning="Valid",
                id="FATHER_OF",
                name="FATHER_OF",
                summary="Quan hệ cha con",
                description="Chi tiết",
                source_node_types=["paragraph"],
                target_node_types=["paragraph"],
                is_symmetric=False,
                is_transitive=False,
                families=["kinship"],
                keywords=["cha"],
            )

            with patch.object(validator, "_embed_relation") as mock_embed:
                mock_embed.return_value = [0.1] * 3072

                result = await validator.validate_new_relation(
                    node_a=node_a,
                    node_b=node_b,
                    proposed_relation="FATHER_OF",
                    summary="Quan hệ cha con",
                )

                assert result.merged_with_existing is True
                assert result.existing_relation_id == "PARENT_OF"
                assert result.similarity_score == 0.92

    @pytest.mark.asyncio
    async def test_low_similarity_stores_new(self, validator, node_a, node_b):
        """Similarity < 0.85 stores new relation."""
        validator._ontology_handler.get_by_id.return_value = None
        validator._ontology_handler.find_similar_excluding.return_value = None

        with patch.object(validator, "_verify_and_extract_with_strong_llm") as mock_strong:
            mock_strong.return_value = StrongLLMRelationOutput(
                accepted=True,
                reasoning="Valid kinship relation",
                id="FATHER_OF",
                name="FATHER_OF",
                summary="Quan hệ cha con",
                description="Chi tiết về quan hệ cha con",
                source_node_types=["paragraph"],
                target_node_types=["paragraph"],
                is_symmetric=False,
                is_transitive=False,
                families=["kinship"],
                keywords=["cha", "con"],
            )

            with patch.object(validator, "_embed_relation") as mock_embed:
                mock_embed.return_value = [0.1] * 3072

                with patch.object(validator, "_store_new_relation") as mock_store:
                    result = await validator.validate_new_relation(
                        node_a=node_a,
                        node_b=node_b,
                        proposed_relation="FATHER_OF",
                        summary="Quan hệ cha con",
                        tier="tier_2_content",
                    )

                    mock_store.assert_called_once()
                    assert result.is_valid is True
                    assert result.final_relation_type == "FATHER_OF"
                    assert result.merged_with_existing is False


class TestStep5StoreNewRelation:
    """STEP 5: Store NEW relation to DB."""

    @pytest.mark.asyncio
    async def test_store_creates_ontology_relation(self, validator):
        """_store_new_relation creates OntologyRelation correctly."""
        output = StrongLLMRelationOutput(
            accepted=True,
            reasoning="Valid",
            id="FATHER_OF",
            name="FATHER_OF",
            summary="Quan hệ cha con",
            description="Chi tiết về quan hệ cha con trong lịch sử Việt Nam",
            source_node_types=["paragraph"],
            target_node_types=["paragraph"],
            is_symmetric=False,
            is_transitive=False,
            families=["kinship"],
            keywords=["cha", "con", "bố"],
        )

        embedding = [0.1] * 3072

        await validator._store_new_relation(
            output=output,
            embedding=embedding,
            tier="tier_2_content",
        )

        validator._ontology_handler.create.assert_called_once()
        created_relation = validator._ontology_handler.create.call_args[0][0]

        assert created_relation.id == "FATHER_OF"
        assert created_relation.name == "FATHER_OF"
        assert created_relation.description == "Chi tiết về quan hệ cha con trong lịch sử Việt Nam"
        assert created_relation.is_builtin is False
        assert created_relation.families == ["kinship"]
        assert created_relation.keywords == ["cha", "con", "bố"]


class TestFullFlow:
    """Test complete validation flow end-to-end."""

    @pytest.mark.asyncio
    async def test_full_flow_new_relation(self, validator, node_a, node_b):
        """Full flow: LLM weak → Strong → Embed → Compare → Store."""
        # Setup: no existing relation
        validator._ontology_handler.get_by_id.return_value = None
        validator._ontology_handler.find_similar_excluding.return_value = None

        with patch.object(validator, "_verify_and_extract_with_strong_llm") as mock_strong:
            mock_strong.return_value = StrongLLMRelationOutput(
                accepted=True,
                reasoning="Valid kinship relation",
                id="FATHER_OF",
                name="FATHER_OF",
                summary="Quan hệ cha con",
                description="Chi tiết về quan hệ cha con",
                source_node_types=["paragraph"],
                target_node_types=["paragraph"],
                is_symmetric=False,
                is_transitive=False,
                families=["kinship"],
                keywords=["cha", "con"],
            )

            with patch.object(validator, "_embed_relation") as mock_embed:
                mock_embed.return_value = [0.1] * 3072

                with patch.object(validator, "_store_new_relation") as mock_store:
                    result = await validator.validate_new_relation(
                        node_a=node_a,
                        node_b=node_b,
                        proposed_relation="FATHER_OF",
                        summary="Quan hệ cha con",
                        tier="tier_2_content",
                    )

                    # Verify flow
                    validator._ontology_handler.get_by_id.assert_called_with("FATHER_OF")
                    mock_strong.assert_called_once()
                    mock_embed.assert_called_once()
                    mock_store.assert_called_once()

                    # Verify result
                    assert result.is_valid is True
                    assert result.final_relation_type == "FATHER_OF"
                    assert result.families == ["kinship"]

    @pytest.mark.asyncio
    async def test_embedding_failure_still_stores(self, validator, node_a, node_b):
        """When embedding fails, still store without similarity check."""
        validator._ontology_handler.get_by_id.return_value = None

        with patch.object(validator, "_verify_and_extract_with_strong_llm") as mock_strong:
            mock_strong.return_value = StrongLLMRelationOutput(
                accepted=True,
                reasoning="Valid",
                id="FATHER_OF",
                name="FATHER_OF",
                summary="Quan hệ cha con",
                description="Chi tiết",
                source_node_types=["paragraph"],
                target_node_types=["paragraph"],
                is_symmetric=False,
                is_transitive=False,
                families=["kinship"],
                keywords=["cha"],
            )

            with patch.object(validator, "_embed_relation") as mock_embed:
                mock_embed.return_value = None  # Embedding failed

                with patch.object(validator, "_store_new_relation") as mock_store:
                    result = await validator.validate_new_relation(
                        node_a=node_a,
                        node_b=node_b,
                        proposed_relation="FATHER_OF",
                        summary="Quan hệ cha con",
                    )

                    # Should still store, but without embedding
                    mock_store.assert_called_once()
                    call_args = mock_store.call_args
                    assert call_args[1]["embedding"] is None

                    assert result.is_valid is True
                    assert "without embedding" in result.reasoning
