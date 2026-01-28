"""Validation tests for ontology completeness."""

from __future__ import annotations

import pytest


class TestOntologyCompleteness:
    """Validate all ontology definitions are complete."""

    def test_history_relations_complete(self):
        """Validate all history relations have required fields."""
        from config.ontologies.history import HISTORY_RELATION_CONSTRAINTS

        required_fields = ["summary", "families", "keywords", "description"]

        for rel in HISTORY_RELATION_CONSTRAINTS:
            for field in required_fields:
                assert hasattr(rel, field), f"{rel.relation_type} missing {field}"

    def test_families_module_complete(self):
        """Validate families module is complete."""
        from config.ontologies import families

        assert hasattr(families, "ALL_FAMILIES")
        assert hasattr(families, "FAMILY_METADATA")
        assert hasattr(families, "FAMILY_AFFINITY_MATRIX")
        assert hasattr(families, "get_affinity")
        assert hasattr(families, "get_max_affinity")

    def test_no_duplicate_relation_ids(self):
        """Check for duplicate relation IDs."""
        from config.ontologies.history import HISTORY_RELATION_CONSTRAINTS

        all_ids = []
        for rel in HISTORY_RELATION_CONSTRAINTS:
            # relation_type can be string or enum
            rel_id = (
                rel.relation_type.value
                if hasattr(rel.relation_type, "value")
                else rel.relation_type
            )
            all_ids.append(rel_id)

        duplicates = [x for x in all_ids if all_ids.count(x) > 1]
        assert not duplicates, f"Duplicate relation IDs: {set(duplicates)}"

    def test_families_count(self):
        """Test we have expected number of families."""
        from config.ontologies.families import ALL_FAMILIES

        assert len(ALL_FAMILIES) == 6

    def test_affinity_matrix_complete(self):
        """Test affinity matrix covers all query types."""
        from config.ontologies.families import ALL_FAMILIES, FAMILY_AFFINITY_MATRIX

        expected_query_types = [
            "causal",
            "comparison",
            "figure",
            "concept",
            "factual",
            "lesson_synthesis",
            "structural",
            "other",
        ]

        for qt in expected_query_types:
            assert qt in FAMILY_AFFINITY_MATRIX, f"Missing query type: {qt}"
            for family in ALL_FAMILIES:
                assert family in FAMILY_AFFINITY_MATRIX[qt], f"Missing family {family} for {qt}"
