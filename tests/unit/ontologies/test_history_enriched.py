"""Tests for enriched history ontology."""

from __future__ import annotations

from config.ontologies.families import ALL_FAMILIES
from config.ontologies.history import HISTORY_RELATION_CONSTRAINTS


class TestHistoryEnrichedRelations:
    """Tests for enriched history relations."""

    def test_all_relations_have_summary(self):
        """Test all relations have summary field."""
        for rel in HISTORY_RELATION_CONSTRAINTS:
            assert rel.summary, f"{rel.relation_type} missing summary"
            assert len(rel.summary) <= 100, f"{rel.relation_type} summary too long"

    def test_all_relations_have_families(self):
        """Test all relations have at least one family."""
        for rel in HISTORY_RELATION_CONSTRAINTS:
            assert rel.families, f"{rel.relation_type} missing families"
            assert len(rel.families) >= 1

    def test_all_families_are_valid(self):
        """Test all families are from the defined list."""
        for rel in HISTORY_RELATION_CONSTRAINTS:
            for family in rel.families:
                assert family in ALL_FAMILIES, f"{rel.relation_type} has invalid family: {family}"

    def test_all_relations_have_keywords(self):
        """Test all relations have at least 3 keywords."""
        for rel in HISTORY_RELATION_CONSTRAINTS:
            assert rel.keywords, f"{rel.relation_type} missing keywords"
            assert len(rel.keywords) >= 3, f"{rel.relation_type} needs at least 3 keywords"

    def test_keywords_exist(self):
        """Test keywords are non-empty strings."""
        for rel in HISTORY_RELATION_CONSTRAINTS:
            for kw in rel.keywords:
                assert kw and len(kw) > 0, f"{rel.relation_type} has empty keyword"

    def test_description_has_content(self):
        """Test description has meaningful content."""
        for rel in HISTORY_RELATION_CONSTRAINTS:
            assert rel.description, f"{rel.relation_type} missing description"
            assert len(rel.description) >= 20, f"{rel.relation_type} description too short"

    def test_description_mentions_paragraphs(self):
        """Test description or summary refers to paragraphs or events."""
        for rel in HISTORY_RELATION_CONSTRAINTS:
            desc_lower = rel.description.lower()
            summary_lower = rel.summary.lower() if rel.summary else ""
            combined = desc_lower + " " + summary_lower
            # Should mention đoạn/paragraph/sự kiện/nội dung in description or summary
            mentions_content = (
                "đoạn" in combined
                or "paragraph" in combined
                or "nội dung" in combined
                or "sự kiện" in combined
                or "event" in combined
            )
            assert mentions_content, f"{rel.relation_type} should mention paragraphs or events"
