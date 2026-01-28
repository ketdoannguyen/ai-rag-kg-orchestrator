"""Tests for enriched relation descriptions.

Validates that all relations have:
- summary (short description)
- families (list of valid family names)
- keywords (list of query matching keywords)
- description with examples
"""

from __future__ import annotations

import pytest

from config.ontologies.base import BASE_RELATIONS
from config.ontologies.families import ALL_FAMILIES
from config.ontologies.history import HISTORY_RELATION_CONSTRAINTS


class TestHistoryRelationsEnrichment:
    """Test enrichment of history-specific relations."""

    def test_all_history_relations_have_summary(self) -> None:
        """All history relations should have a short summary."""
        for rel in HISTORY_RELATION_CONSTRAINTS:
            assert rel.summary, f"{rel.relation_type} missing summary"
            assert len(rel.summary) < 100, (
                f"{rel.relation_type} summary too long: {len(rel.summary)}"
            )

    def test_all_history_relations_have_families(self) -> None:
        """All history relations should have valid families."""
        for rel in HISTORY_RELATION_CONSTRAINTS:
            assert rel.families, f"{rel.relation_type} missing families"
            for family in rel.families:
                assert family in ALL_FAMILIES, f"{rel.relation_type} has invalid family: {family}"

    def test_all_history_relations_have_keywords(self) -> None:
        """All history relations should have at least 3 keywords."""
        for rel in HISTORY_RELATION_CONSTRAINTS:
            assert rel.keywords, f"{rel.relation_type} missing keywords"
            assert len(rel.keywords) >= 3, (
                f"{rel.relation_type} needs at least 3 keywords, has {len(rel.keywords)}"
            )

    def test_history_description_contains_examples(self) -> None:
        """All history relations should have examples in description."""
        for rel in HISTORY_RELATION_CONSTRAINTS:
            assert "Ví dụ:" in rel.description or "Ví dụ" in rel.description, (
                f"{rel.relation_type} description missing examples"
            )

    def test_history_description_contains_distinction(self) -> None:
        """All history relations should distinguish from similar relations."""
        for rel in HISTORY_RELATION_CONSTRAINTS:
            assert "Phân biệt với:" in rel.description or "Phân biệt" in rel.description, (
                f"{rel.relation_type} description missing distinction section"
            )

    def test_history_relations_count(self) -> None:
        """Should have exactly 14 history-specific relations."""
        assert len(HISTORY_RELATION_CONSTRAINTS) == 14


class TestBaseRelationsEnrichment:
    """Test enrichment of base relations."""

    def test_all_base_relations_have_summary(self) -> None:
        """All base relations should have a short summary."""
        for rel in BASE_RELATIONS:
            assert rel.summary, f"{rel.relation_type} missing summary"
            assert len(rel.summary) < 100, (
                f"{rel.relation_type} summary too long: {len(rel.summary)}"
            )

    def test_all_base_relations_have_families(self) -> None:
        """All base relations should have valid families."""
        for rel in BASE_RELATIONS:
            assert rel.families, f"{rel.relation_type} missing families"
            for family in rel.families:
                assert family in ALL_FAMILIES, f"{rel.relation_type} has invalid family: {family}"

    def test_all_base_relations_have_keywords(self) -> None:
        """All base relations should have at least 3 keywords."""
        for rel in BASE_RELATIONS:
            assert rel.keywords, f"{rel.relation_type} missing keywords"
            assert len(rel.keywords) >= 3, (
                f"{rel.relation_type} needs at least 3 keywords, has {len(rel.keywords)}"
            )

    def test_base_description_contains_examples(self) -> None:
        """All base relations should have examples in description."""
        for rel in BASE_RELATIONS:
            assert "Ví dụ:" in rel.description or "Ví dụ" in rel.description, (
                f"{rel.relation_type} description missing examples"
            )

    def test_base_description_contains_distinction(self) -> None:
        """All base relations should distinguish from similar relations."""
        for rel in BASE_RELATIONS:
            assert "Phân biệt với:" in rel.description or "Phân biệt" in rel.description, (
                f"{rel.relation_type} description missing distinction section"
            )

    def test_base_relations_count(self) -> None:
        """Should have exactly 24 base relations."""
        assert len(BASE_RELATIONS) == 24


class TestFamilyConsistency:
    """Test consistency between relations and families."""

    @pytest.mark.parametrize(
        "family",
        ALL_FAMILIES,
    )
    def test_each_family_has_relations(self, family: str) -> None:
        """Each family should be used by at least one relation."""
        all_relations = BASE_RELATIONS + HISTORY_RELATION_CONSTRAINTS
        relations_with_family = [r for r in all_relations if family in r.families]
        assert len(relations_with_family) > 0, f"Family '{family}' has no relations"

    def test_causal_family_has_causal_relations(self) -> None:
        """Causal family should include causal relations."""
        causal_types = ["CAUSED_BY", "LED_TO", "RESULTED_IN", "CONSEQUENCE_OF"]
        for rel in HISTORY_RELATION_CONSTRAINTS:
            rel_type = (
                rel.relation_type.value
                if hasattr(rel.relation_type, "value")
                else str(rel.relation_type)
            )
            if rel_type in causal_types:
                assert "causal" in rel.families, f"{rel_type} should be in causal family"

    def test_temporal_family_has_temporal_relations(self) -> None:
        """Temporal family should include temporal relations."""
        temporal_types = ["PRECEDED_BY", "FOLLOWED_BY", "DURING", "CONTEMPORARY_WITH"]
        for rel in HISTORY_RELATION_CONSTRAINTS:
            rel_type = (
                rel.relation_type.value
                if hasattr(rel.relation_type, "value")
                else str(rel.relation_type)
            )
            if rel_type in temporal_types:
                assert "temporal" in rel.families, f"{rel_type} should be in temporal family"


class TestKeywordsQuality:
    """Test quality of keywords for query matching."""

    def test_keywords_are_vietnamese(self) -> None:
        """Keywords should be in Vietnamese for query matching."""
        all_relations = BASE_RELATIONS + HISTORY_RELATION_CONSTRAINTS
        for rel in all_relations:
            for keyword in rel.keywords:
                # Check that keywords are not purely English
                # Vietnamese uses ASCII + diacritics, so we check for common patterns
                assert not keyword.isupper(), (
                    f"{rel.relation_type} has uppercase keyword: {keyword}"
                )

    def test_keywords_are_lowercase_phrases(self) -> None:
        """Keywords should be lowercase phrases."""
        all_relations = BASE_RELATIONS + HISTORY_RELATION_CONSTRAINTS
        for rel in all_relations:
            for keyword in rel.keywords:
                # Keywords should be lowercase
                assert keyword == keyword.lower() or any(
                    c in keyword
                    for c in "áàảãạăắằẳẵặâấầẩẫậéèẻẽẹêếềểễệíìỉĩịóòỏõọôốồổỗộơớờởỡợúùủũụưứừửữựýỳỷỹỵđ"
                ), f"{rel.relation_type} has non-lowercase keyword: {keyword}"
