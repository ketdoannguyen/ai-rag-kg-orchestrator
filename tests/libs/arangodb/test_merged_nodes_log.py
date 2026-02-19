"""Unit tests for MergedNodeLog model."""

from __future__ import annotations

import pytest
from arango_client.models import MergedNodeLog


class TestMergedNodeLog:
    """Tests for MergedNodeLog model."""

    def test_create_auto_merge_log(self):
        """Test creating a log for auto_merge."""
        log = MergedNodeLog(
            source_id="content_nodes/abc123",
            source_text="Original text content",
            source_type="paragraph",
            source_parent_id="structure_nodes/section1",
            source_source_id="data_sources/doc1",
            target_id="content_nodes/def456",
            target_text="Target text content",
            similarity=0.85,
            merge_reason="auto_merge",
        )

        assert log.source_id == "content_nodes/abc123"
        assert log.merge_reason == "auto_merge"
        assert log.llm_reasoning is None
        assert log.merged_at is not None

    def test_create_llm_merge_log(self):
        """Test creating a log for llm_merge."""
        log = MergedNodeLog(
            source_id="content_nodes/abc123",
            source_text="Original text",
            source_type="paragraph",
            target_id="content_nodes/def456",
            target_text="Target text",
            similarity=0.65,
            merge_reason="llm_merge",
            llm_reasoning="Both texts describe the same concept with minor wording differences",
        )

        assert log.merge_reason == "llm_merge"
        assert log.llm_reasoning is not None
        assert "same concept" in log.llm_reasoning

    def test_similarity_validation_max(self):
        """Test similarity must be <= 1.0."""
        with pytest.raises(ValueError):
            MergedNodeLog(
                source_id="content_nodes/abc123",
                source_text="Text",
                source_type="paragraph",
                target_id="content_nodes/def456",
                target_text="Text",
                similarity=1.5,  # Invalid
                merge_reason="auto_merge",
            )

    def test_similarity_validation_min(self):
        """Test similarity must be >= 0.0."""
        with pytest.raises(ValueError):
            MergedNodeLog(
                source_id="content_nodes/abc123",
                source_text="Text",
                source_type="paragraph",
                target_id="content_nodes/def456",
                target_text="Text",
                similarity=-0.1,  # Invalid
                merge_reason="auto_merge",
            )

    def test_similarity_boundary_values(self):
        """Test similarity accepts boundary values 0.0 and 1.0."""
        log_zero = MergedNodeLog(
            source_id="content_nodes/abc123",
            source_text="Text",
            source_type="paragraph",
            target_id="content_nodes/def456",
            target_text="Text",
            similarity=0.0,
            merge_reason="auto_merge",
        )
        assert log_zero.similarity == 0.0

        log_one = MergedNodeLog(
            source_id="content_nodes/abc123",
            source_text="Text",
            source_type="paragraph",
            target_id="content_nodes/def456",
            target_text="Text",
            similarity=1.0,
            merge_reason="auto_merge",
        )
        assert log_one.similarity == 1.0

    def test_similarity_float32_epsilon(self):
        """Test float32 cosine similarity epsilon is clamped (ArangoDB precision)."""
        log = MergedNodeLog(
            source_id="content_nodes/abc123",
            source_text="Text",
            source_type="paragraph",
            target_id="content_nodes/def456",
            target_text="Text",
            similarity=1.0000001192092896,  # typical float32 epsilon
            merge_reason="auto_merge",
        )
        assert log.similarity == 1.0

    def test_merge_reason_validation(self):
        """Test merge_reason must be valid literal."""
        with pytest.raises(ValueError):
            MergedNodeLog(
                source_id="content_nodes/abc123",
                source_text="Text",
                source_type="paragraph",
                target_id="content_nodes/def456",
                target_text="Text",
                similarity=0.8,
                merge_reason="invalid_reason",  # type: ignore[arg-type]
            )

    def test_to_document(self):
        """Test conversion to ArangoDB document."""
        log = MergedNodeLog(
            source_id="content_nodes/abc123",
            source_text="Text",
            source_type="paragraph",
            target_id="content_nodes/def456",
            target_text="Text",
            similarity=0.8,
            merge_reason="auto_merge",
        )

        doc = log.to_document()

        assert isinstance(doc, dict)
        assert doc["source_id"] == "content_nodes/abc123"
        assert doc["similarity"] == 0.8
        assert doc["merge_reason"] == "auto_merge"
        assert "_key" not in doc
        assert "merged_at" in doc

    def test_from_document(self):
        """Test reconstruction from ArangoDB document."""
        doc = {
            "_key": "123456",
            "_id": "merged_nodes_log/123456",
            "_rev": "_abc123",
            "source_id": "content_nodes/abc123",
            "source_text": "Text",
            "source_type": "paragraph",
            "source_parent_id": None,
            "source_source_id": None,
            "target_id": "content_nodes/def456",
            "target_text": "Text",
            "similarity": 0.8,
            "merge_reason": "auto_merge",
            "llm_reasoning": None,
            "merged_at": "2026-02-18T10:00:00+00:00",
        }

        log = MergedNodeLog.from_document(doc)

        assert log.source_id == "content_nodes/abc123"
        assert log.similarity == 0.8
        assert log.merge_reason == "auto_merge"

    def test_optional_fields(self):
        """Test optional fields with None values."""
        log = MergedNodeLog(
            source_id="content_nodes/abc123",
            source_text="Text",
            source_type="paragraph",
            source_parent_id=None,
            source_source_id=None,
            target_id="content_nodes/def456",
            target_text="Text",
            similarity=0.8,
            merge_reason="auto_merge",
            llm_reasoning=None,
        )

        assert log.source_parent_id is None
        assert log.source_source_id is None
        assert log.llm_reasoning is None

    def test_roundtrip_serialization(self):
        """Test that model survives to_document -> from_document roundtrip."""
        original = MergedNodeLog(
            source_id="content_nodes/source1",
            source_text="Source text here",
            source_type="table",
            source_parent_id="structure_nodes/parent",
            source_source_id="data_sources/doc",
            target_id="content_nodes/target1",
            target_text="Target text here",
            similarity=0.72,
            merge_reason="llm_merge",
            llm_reasoning="LLM determined these are duplicates",
        )

        doc = original.to_document()
        # Simulate ArangoDB adding system fields
        doc["_key"] = "test_key"
        doc["_id"] = "merged_nodes_log/test_key"
        doc["_rev"] = "_rev123"

        restored = MergedNodeLog.from_document(doc)

        assert restored.source_id == original.source_id
        assert restored.source_text == original.source_text
        assert restored.source_type == original.source_type
        assert restored.target_id == original.target_id
        assert restored.similarity == original.similarity
        assert restored.merge_reason == original.merge_reason
        assert restored.llm_reasoning == original.llm_reasoning
