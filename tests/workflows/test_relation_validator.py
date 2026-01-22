"""Unit tests for NewRelationValidator.

Tests the new relation validation flow:
1. LLM Extract with description
2. Embed (name + description) and compare with ALL tiers
3. LLM Strong Verify (validator role)
4. Re-extract with Strong LLM when invalid

Uses mock data to test all scenarios without requiring actual LLM/DB connections.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any
from unittest.mock import AsyncMock, MagicMock

import pytest

from workflows.extract_relations.relation_validator import (
    NewRelationValidator,
    NodeInfo,
    ValidationResult,
)

# =============================================================================
# Mock Data
# =============================================================================

MOCK_EMBEDDING = [0.1] * 3072  # text-embedding-3-large dimensions

MOCK_NODE_A = NodeInfo(
    node_id="node_a_123",
    node_type="paragraph",
    text="Số nguyên tố là số tự nhiên lớn hơn 1, chỉ có đúng hai ước là 1 và chính nó.",
    embedding=MOCK_EMBEDDING,
)

MOCK_NODE_B = NodeInfo(
    node_id="node_b_456",
    node_type="definition",
    text="Định nghĩa: Một số nguyên dương p > 1 được gọi là số nguyên tố nếu p chỉ chia hết cho 1 và p.",
    embedding=MOCK_EMBEDDING,
)


@dataclass
class MockEmbedResponse:
    """Mock response for embedding API."""

    embeddings: list[list[float]]


@dataclass
class MockLLMResponse:
    """Mock response for LLM invoke API."""

    message: Any


@dataclass
class MockMessage:
    """Mock message content."""

    content: str


# =============================================================================
# Test Fixtures
# =============================================================================


@pytest.fixture
def mock_settings():
    """Create mock extraction settings."""
    settings = MagicMock()
    settings.EXTRACTION_MERGE_THRESHOLD = 0.8
    settings.EXTRACTION_EMBEDDING_MODEL = "openrouter/openai/text-embedding-3-large"
    settings.EXTRACTION_STRONG_MODEL = "anthropic/claude-3.5-sonnet"
    settings.EXTRACTION_FALLBACK_RELATION = "SIMILAR_TO"
    return settings


@pytest.fixture
def mock_ontology_handler():
    """Create mock ontology handler."""
    handler = MagicMock()

    # Default: no similar relation found
    handler.find_similar_excluding.return_value = None
    handler.get_relation.return_value = None
    handler.filter_relations_by_centroid.return_value = [
        ("DEFINES", 0.85),
        ("EXPLAINS", 0.75),
        ("RELATED_TO", 0.65),
    ]

    return handler


@pytest.fixture
def mock_llm_client():
    """Create mock LLM client."""
    client = AsyncMock()

    # Default embed response
    client.embed.return_value = MockEmbedResponse(embeddings=[MOCK_EMBEDDING])

    # Default invoke response (validation passes)
    client.invoke.return_value = MockLLMResponse(
        message=MockMessage(content='{"is_valid": true, "confidence": 0.9, "reasoning": "Valid"}')
    )

    return client


@pytest.fixture
def validator(mock_settings, mock_ontology_handler, mock_llm_client):
    """Create NewRelationValidator with mocks."""
    v = NewRelationValidator(
        settings=mock_settings,
        ontology_handler=mock_ontology_handler,
        llm_client=mock_llm_client,
    )
    return v


# =============================================================================
# Test: NodeInfo and ValidationResult
# =============================================================================


class TestDataclasses:
    """Test dataclass creation."""

    def test_node_info_creation(self):
        """NodeInfo should be created with correct fields."""
        node = NodeInfo(
            node_id="test_id",
            node_type="paragraph",
            text="Test text",
            embedding=[0.1, 0.2, 0.3],
        )

        assert node.node_id == "test_id"
        assert node.node_type == "paragraph"
        assert node.text == "Test text"
        assert node.embedding == [0.1, 0.2, 0.3]

    def test_node_info_default_embedding(self):
        """NodeInfo embedding should default to None."""
        node = NodeInfo(node_id="id", node_type="type", text="text")
        assert node.embedding is None

    def test_validation_result_creation(self):
        """ValidationResult should be created with correct fields."""
        result = ValidationResult(
            is_valid=True,
            final_relation_type="DEFINES",
            final_description="Định nghĩa mối quan hệ",
            merged_with_existing=False,
        )

        assert result.is_valid is True
        assert result.final_relation_type == "DEFINES"
        assert result.final_description == "Định nghĩa mối quan hệ"
        assert result.merged_with_existing is False
        assert result.existing_relation_id is None
        assert result.similarity_score is None
        assert result.re_extracted is False
        assert result.reasoning == ""

    def test_validation_result_merged(self):
        """ValidationResult for merged relation."""
        result = ValidationResult(
            is_valid=True,
            final_relation_type="EXPLAINS",
            final_description=None,
            merged_with_existing=True,
            existing_relation_id="EXPLAINS",
            similarity_score=0.92,
            reasoning="Merged with existing",
        )

        assert result.merged_with_existing is True
        assert result.existing_relation_id == "EXPLAINS"
        assert result.similarity_score == 0.92


# =============================================================================
# Test: Embed Relation
# =============================================================================


class TestEmbedRelation:
    """Test _embed_relation method."""

    @pytest.mark.asyncio
    async def test_embed_with_description(self, validator, mock_llm_client):
        """Should embed name:description format."""
        result = await validator._embed_relation("DEFINES", "Định nghĩa khái niệm")

        assert result == MOCK_EMBEDDING
        mock_llm_client.embed.assert_called_once()

        # Check the input text format - now a list
        call_args = mock_llm_client.embed.call_args
        request = call_args[0][0]
        assert request.input == ["DEFINES: Định nghĩa khái niệm"]

    @pytest.mark.asyncio
    async def test_embed_without_description(self, validator, mock_llm_client):
        """Should embed just name when no description."""
        result = await validator._embed_relation("DEFINES", "")

        assert result == MOCK_EMBEDDING
        call_args = mock_llm_client.embed.call_args
        request = call_args[0][0]
        assert request.input == ["DEFINES"]

    @pytest.mark.asyncio
    async def test_embed_failure_returns_none(self, validator, mock_llm_client):
        """Should return None when embedding fails."""
        mock_llm_client.embed.side_effect = Exception("API Error")

        result = await validator._embed_relation("DEFINES", "Description")

        assert result is None

    @pytest.mark.asyncio
    async def test_embed_empty_response(self, validator, mock_llm_client):
        """Should return None when response is empty."""
        mock_llm_client.embed.return_value = MockEmbedResponse(embeddings=[])

        result = await validator._embed_relation("DEFINES", "Description")

        assert result is None


# =============================================================================
# Test: Find Similar in ALL Tiers
# =============================================================================


class TestFindSimilarInAllTiers:
    """Test _find_similar_in_all_tiers method."""

    @pytest.mark.asyncio
    async def test_find_similar_found(self, validator, mock_ontology_handler):
        """Should return similar relation when found."""
        mock_ontology_handler.find_similar_excluding.return_value = ("EXPLAINS", 0.92)

        result = await validator._find_similar_in_all_tiers(
            embedding=MOCK_EMBEDDING,
            exclude_ids=["DEFINES"],
        )

        assert result == ("EXPLAINS", 0.92)
        mock_ontology_handler.find_similar_excluding.assert_called_once_with(
            embedding=MOCK_EMBEDDING,
            exclude_ids=["DEFINES"],
            threshold=0.8,  # From mock_settings
        )

    @pytest.mark.asyncio
    async def test_find_similar_not_found(self, validator, mock_ontology_handler):
        """Should return None when no similar relation found."""
        mock_ontology_handler.find_similar_excluding.return_value = None

        result = await validator._find_similar_in_all_tiers(
            embedding=MOCK_EMBEDDING,
            exclude_ids=[],
        )

        assert result is None

    @pytest.mark.asyncio
    async def test_find_similar_no_handler(self, mock_settings, mock_llm_client):
        """Should return None when no ontology handler."""
        validator = NewRelationValidator(
            settings=mock_settings,
            ontology_handler=None,
            llm_client=mock_llm_client,
        )

        result = await validator._find_similar_in_all_tiers(
            embedding=MOCK_EMBEDDING,
            exclude_ids=[],
        )

        assert result is None


# =============================================================================
# Test: LLM Strong Verify (Validator Role)
# =============================================================================


class TestVerifyWithStrongLLM:
    """Test _verify_with_strong_llm method."""

    @pytest.mark.asyncio
    async def test_verify_valid(self, validator, mock_llm_client):
        """Should return True when LLM validates."""
        mock_llm_client.invoke.return_value = MockLLMResponse(
            message=MockMessage(
                content='{"is_valid": true, "confidence": 0.95, "reasoning": "Relation is accurate"}'
            )
        )

        result = await validator._verify_with_strong_llm(
            node_a=MOCK_NODE_A,
            node_b=MOCK_NODE_B,
            proposed_relation="DEFINES",
            description="Định nghĩa số nguyên tố",
            reasoning="Node A defines the concept in Node B",
        )

        assert result is True
        mock_llm_client.invoke.assert_called_once()

    @pytest.mark.asyncio
    async def test_verify_invalid(self, validator, mock_llm_client):
        """Should return False when LLM rejects."""
        mock_llm_client.invoke.return_value = MockLLMResponse(
            message=MockMessage(
                content='{"is_valid": false, "confidence": 0.3, "reasoning": "Relation is too generic"}'
            )
        )

        result = await validator._verify_with_strong_llm(
            node_a=MOCK_NODE_A,
            node_b=MOCK_NODE_B,
            proposed_relation="RELATED_TO",
            description="Có liên quan",
            reasoning="They seem related",
        )

        assert result is False

    @pytest.mark.asyncio
    async def test_verify_error_returns_true(self, validator, mock_llm_client):
        """Should return True on error (fail open)."""
        mock_llm_client.invoke.side_effect = Exception("API Error")

        result = await validator._verify_with_strong_llm(
            node_a=MOCK_NODE_A,
            node_b=MOCK_NODE_B,
            proposed_relation="DEFINES",
            description="Description",
            reasoning="Reasoning",
        )

        # Fail open - accept the relation
        assert result is True

    @pytest.mark.asyncio
    async def test_verify_prompt_contains_nodes(self, validator, mock_llm_client):
        """Verify prompt contains node information."""
        await validator._verify_with_strong_llm(
            node_a=MOCK_NODE_A,
            node_b=MOCK_NODE_B,
            proposed_relation="DEFINES",
            description="Description",
            reasoning="Reasoning",
        )

        call_args = mock_llm_client.invoke.call_args
        request = call_args[0][0]
        prompt = request.messages[1].content

        assert "paragraph" in prompt  # node_a.node_type
        assert "definition" in prompt  # node_b.node_type
        assert "DEFINES" in prompt
        assert "Description" in prompt
        assert "Reasoning" in prompt


# =============================================================================
# Test: Re-extract with Strong LLM
# =============================================================================


class TestReExtractWithStrongLLM:
    """Test _re_extract_with_strong_llm method."""

    @pytest.mark.asyncio
    async def test_reextract_success(self, validator, mock_llm_client):
        """Should return new relation when re-extraction succeeds."""
        mock_llm_client.invoke.return_value = MockLLMResponse(
            message=MockMessage(
                content="""{
                    "relation_type": "EQUIVALENT_DEFINITION",
                    "description": "Hai định nghĩa tương đương",
                    "is_new_relation": true,
                    "confidence": 0.85,
                    "reasoning": "Both nodes define the same concept"
                }"""
            )
        )

        result = await validator._re_extract_with_strong_llm(
            node_a=MOCK_NODE_A,
            node_b=MOCK_NODE_B,
            tier="tier_2_content",
            previous_attempt="RELATED_TO",
        )

        assert result is not None
        assert result["relation_type"] == "EQUIVALENT_DEFINITION"
        assert result["description"] == "Hai định nghĩa tương đương"
        assert result["is_new_relation"] is True

    @pytest.mark.asyncio
    async def test_reextract_returns_same_rejected(self, validator, mock_llm_client):
        """Should return None if re-extraction returns same as rejected."""
        mock_llm_client.invoke.return_value = MockLLMResponse(
            message=MockMessage(content='{"relation_type": "RELATED_TO", "confidence": 0.5}')
        )

        result = await validator._re_extract_with_strong_llm(
            node_a=MOCK_NODE_A,
            node_b=MOCK_NODE_B,
            tier="tier_2_content",
            previous_attempt="RELATED_TO",  # Same as returned
        )

        assert result is None

    @pytest.mark.asyncio
    async def test_reextract_error_returns_none(self, validator, mock_llm_client):
        """Should return None on error."""
        mock_llm_client.invoke.side_effect = Exception("API Error")

        result = await validator._re_extract_with_strong_llm(
            node_a=MOCK_NODE_A,
            node_b=MOCK_NODE_B,
            tier="tier_2_content",
            previous_attempt="RELATED_TO",
        )

        assert result is None

    @pytest.mark.asyncio
    async def test_reextract_no_relation_type(self, validator, mock_llm_client):
        """Should return None when no relation_type in response."""
        mock_llm_client.invoke.return_value = MockLLMResponse(
            message=MockMessage(content='{"confidence": 0.5, "reasoning": "unclear"}')
        )

        result = await validator._re_extract_with_strong_llm(
            node_a=MOCK_NODE_A,
            node_b=MOCK_NODE_B,
            tier="tier_2_content",
            previous_attempt="RELATED_TO",
        )

        assert result is None


# =============================================================================
# Test: Full Validation Flow
# =============================================================================


class TestValidateNewRelation:
    """Test validate_new_relation - full flow."""

    @pytest.mark.asyncio
    async def test_flow_merge_with_existing(
        self, validator, mock_llm_client, mock_ontology_handler
    ):
        """
        Flow: Embed → Compare ALL → MERGE (score > threshold)
        """
        # Setup: Similar relation found with high score
        mock_ontology_handler.find_similar_excluding.return_value = ("EXPLAINS", 0.92)

        result = await validator.validate_new_relation(
            node_a=MOCK_NODE_A,
            node_b=MOCK_NODE_B,
            proposed_relation="CLARIFIES",
            description="Làm rõ khái niệm",
            reasoning="Node A clarifies Node B",
            tier="tier_2_content",
            top_k_relation_ids=["DEFINES"],
        )

        assert result.is_valid is True
        assert result.merged_with_existing is True
        assert result.final_relation_type == "EXPLAINS"
        assert result.existing_relation_id == "EXPLAINS"
        assert result.similarity_score == 0.92

        # Verify LLM Strong was NOT called (merged before verification)
        # Only embed was called
        assert mock_llm_client.embed.call_count == 1
        assert mock_llm_client.invoke.call_count == 0

    @pytest.mark.asyncio
    async def test_flow_strong_verify_valid(
        self, validator, mock_llm_client, mock_ontology_handler
    ):
        """
        Flow: Embed → Compare ALL (no match) → LLM Strong Verify → VALID
        """
        # Setup: No similar relation found
        mock_ontology_handler.find_similar_excluding.return_value = None

        # Setup: LLM validates
        mock_llm_client.invoke.return_value = MockLLMResponse(
            message=MockMessage(
                content='{"is_valid": true, "confidence": 0.9, "reasoning": "Valid"}'
            )
        )

        result = await validator.validate_new_relation(
            node_a=MOCK_NODE_A,
            node_b=MOCK_NODE_B,
            proposed_relation="FORMAL_DEFINITION",
            description="Định nghĩa chính thức",
            reasoning="Node A is the formal definition of concept in Node B",
            tier="tier_2_content",
            top_k_relation_ids=["DEFINES"],
        )

        assert result.is_valid is True
        assert result.merged_with_existing is False
        assert result.final_relation_type == "FORMAL_DEFINITION"
        assert result.final_description == "Định nghĩa chính thức"

        # Verify flow: embed → compare → verify
        assert mock_llm_client.embed.call_count == 1
        assert mock_llm_client.invoke.call_count == 1  # Verify call

    @pytest.mark.asyncio
    async def test_flow_strong_verify_invalid_reextract(
        self, validator, mock_llm_client, mock_ontology_handler
    ):
        """
        Flow: Embed → Compare ALL (no match) → LLM Strong Verify → INVALID → Re-extract
        """
        # Setup: No similar relation found
        mock_ontology_handler.find_similar_excluding.return_value = None

        # Setup: First call rejects, second call is re-extraction
        mock_llm_client.invoke.side_effect = [
            # First call: Verify - REJECT
            MockLLMResponse(
                message=MockMessage(
                    content='{"is_valid": false, "confidence": 0.3, "reasoning": "Too generic"}'
                )
            ),
            # Second call: Re-extract
            MockLLMResponse(
                message=MockMessage(
                    content="""{
                        "relation_type": "EQUIVALENT_DEFINITION",
                        "description": "Hai định nghĩa tương đương",
                        "is_new_relation": true,
                        "confidence": 0.85
                    }"""
                )
            ),
        ]

        result = await validator.validate_new_relation(
            node_a=MOCK_NODE_A,
            node_b=MOCK_NODE_B,
            proposed_relation="RELATED_TO",
            description="Có liên quan",
            reasoning="They are related somehow",
            tier="tier_2_content",
            top_k_relation_ids=[],
        )

        assert result.is_valid is True
        assert result.re_extracted is True
        assert result.final_relation_type == "EQUIVALENT_DEFINITION"
        assert result.final_description == "Hai định nghĩa tương đương"

        # Verify flow: embed → compare → verify → re-extract → embed again
        assert mock_llm_client.embed.call_count == 2  # Original + re-extracted
        assert mock_llm_client.invoke.call_count == 2  # Verify + Re-extract

    @pytest.mark.asyncio
    async def test_flow_reextract_then_merge(
        self, validator, mock_llm_client, mock_ontology_handler
    ):
        """
        Flow: Embed → Compare → Verify(INVALID) → Re-extract → Embed → Compare → MERGE
        """
        # Setup: First compare - no match, second compare - match
        mock_ontology_handler.find_similar_excluding.side_effect = [
            None,  # First comparison - no match
            ("DEFINES", 0.88),  # Second comparison after re-extract - MATCH
        ]

        # Setup: Verify rejects, then re-extract succeeds
        mock_llm_client.invoke.side_effect = [
            MockLLMResponse(
                message=MockMessage(
                    content='{"is_valid": false, "confidence": 0.2, "reasoning": "Invalid"}'
                )
            ),
            MockLLMResponse(
                message=MockMessage(
                    content='{"relation_type": "DEFINITION_OF", "description": "Định nghĩa của", "is_new_relation": true}'
                )
            ),
        ]

        result = await validator.validate_new_relation(
            node_a=MOCK_NODE_A,
            node_b=MOCK_NODE_B,
            proposed_relation="SOMETHING",
            description="Something",
            reasoning="Reasoning",
            tier="tier_2_content",
            top_k_relation_ids=[],
        )

        assert result.is_valid is True
        assert result.merged_with_existing is True
        assert result.re_extracted is True
        assert result.final_relation_type == "DEFINES"

    @pytest.mark.asyncio
    async def test_flow_fallback_on_reextract_failure(
        self, validator, mock_llm_client, mock_ontology_handler, mock_settings
    ):
        """
        Flow: Embed → Compare → Verify(INVALID) → Re-extract(FAIL) → FALLBACK
        """
        mock_ontology_handler.find_similar_excluding.return_value = None

        mock_llm_client.invoke.side_effect = [
            # Verify - REJECT
            MockLLMResponse(
                message=MockMessage(
                    content='{"is_valid": false, "confidence": 0.1, "reasoning": "Bad"}'
                )
            ),
            # Re-extract - FAIL
            Exception("API Error"),
        ]

        result = await validator.validate_new_relation(
            node_a=MOCK_NODE_A,
            node_b=MOCK_NODE_B,
            proposed_relation="BAD_RELATION",
            description="Bad",
            reasoning="Bad reasoning",
            tier="tier_2_content",
            top_k_relation_ids=[],
        )

        assert result.is_valid is False
        assert result.final_relation_type == mock_settings.EXTRACTION_FALLBACK_RELATION

    @pytest.mark.asyncio
    async def test_flow_embed_failure_accepts(self, validator, mock_llm_client):
        """
        Flow: Embed(FAIL) → Accept without comparison
        """
        mock_llm_client.embed.side_effect = Exception("Embed failed")

        result = await validator.validate_new_relation(
            node_a=MOCK_NODE_A,
            node_b=MOCK_NODE_B,
            proposed_relation="NEW_RELATION",
            description="Description",
            reasoning="Reasoning",
            tier="tier_2_content",
            top_k_relation_ids=[],
        )

        assert result.is_valid is True
        assert result.final_relation_type == "NEW_RELATION"
        assert "Embedding failed" in result.reasoning


# =============================================================================
# Test: Integration with LLMRelationExtractor
# =============================================================================


class TestLLMRelationExtractorIntegration:
    """Test integration between LLMRelationExtractor and NewRelationValidator."""

    def test_extractor_has_validator(self):
        """LLMRelationExtractor should have relation_validator property."""
        from workflows.extract_relations.llm_extractor import LLMRelationExtractor

        extractor = LLMRelationExtractor()
        assert hasattr(extractor, "_relation_validator")
        assert hasattr(extractor, "relation_validator")

    def test_extraction_result_has_new_fields(self):
        """ExtractionResult should have new_relation_description and validated fields."""
        from workflows.extract_relations.llm_extractor import ExtractionResult

        result = ExtractionResult(
            should_merge=False,
            relation_type="NEW_RELATION",
            confidence=0.9,
            reasoning="Test",
            is_new_relation=True,
            new_relation_description="Description of new relation",
            is_in_top_k=False,
            used_fallback=False,
            validated=True,
        )

        assert result.new_relation_description == "Description of new relation"
        assert result.validated is True

    def test_extraction_result_default_values(self):
        """ExtractionResult new fields should have correct defaults."""
        from workflows.extract_relations.llm_extractor import ExtractionResult

        result = ExtractionResult()

        assert result.new_relation_description is None
        assert result.validated is False


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
