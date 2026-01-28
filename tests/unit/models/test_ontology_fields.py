"""Tests for OntologyRelation and RelationConstraint new fields."""

from __future__ import annotations

from arango_client.models.ontology import OntologyRelation, RelationTier

from config.ontologies.base import RelationConstraint


class TestOntologyRelationNewFields:
    """Test OntologyRelation with new fields."""

    def test_ontology_relation_new_fields(self) -> None:
        """Test OntologyRelation với fields mới."""
        rel = OntologyRelation(
            id="TEST",
            name="Test Relation",
            description="Long detailed description...",
            summary="Short desc",
            families=["causal", "temporal"],
            keywords=["tại sao", "nguyên nhân"],
        )

        assert rel.summary == "Short desc"
        assert len(rel.families) == 2
        assert "causal" in rel.families
        assert "temporal" in rel.families
        assert "tại sao" in rel.keywords
        assert "nguyên nhân" in rel.keywords

    def test_ontology_relation_default_values(self) -> None:
        """Test default values cho new fields."""
        rel = OntologyRelation(
            id="TEST",
            name="Test Relation",
            description="Test description",
        )

        assert rel.summary == ""
        assert rel.families == []
        assert rel.keywords == []

    def test_get_embedding_text_includes_name_and_description(self) -> None:
        """Test embedding text includes both name and description."""
        rel = OntologyRelation(
            id="TEST",
            name="Test Name",
            description="This is the description",
        )

        assert rel.get_embedding_text() == "Test Name: This is the description"
        assert "Test Name" in rel.get_embedding_text()
        assert "This is the description" in rel.get_embedding_text()

    def test_ontology_relation_to_document(self) -> None:
        """Test to_document includes new fields."""
        rel = OntologyRelation(
            id="TEST",
            name="Test Relation",
            description="Test description",
            summary="Short summary",
            families=["causal"],
            keywords=["keyword1"],
        )

        doc = rel.to_document()
        assert doc["_key"] == "TEST"
        assert doc["summary"] == "Short summary"
        assert doc["families"] == ["causal"]
        assert doc["keywords"] == ["keyword1"]

    def test_ontology_relation_from_document(self) -> None:
        """Test from_document parses new fields."""
        doc = {
            "_key": "TEST",
            "name": "Test Relation",
            "description": "Test description",
            "summary": "Short summary",
            "families": ["temporal"],
            "keywords": ["khi nào"],
            "tiers": [RelationTier.TIER_2_CONTENT.value],
        }

        rel = OntologyRelation.from_document(doc)
        assert rel.id == "TEST"
        assert rel.summary == "Short summary"
        assert rel.families == ["temporal"]
        assert rel.keywords == ["khi nào"]


class TestRelationConstraintNewFields:
    """Test RelationConstraint with new fields."""

    def test_relation_constraint_new_fields(self) -> None:
        """Test RelationConstraint với fields mới."""
        constraint = RelationConstraint(
            relation_type="TEST",
            source_node_types=["paragraph"],
            target_node_types=["paragraph"],
            summary="Short summary",
            families=["causal"],
            keywords=["vì sao"],
        )

        assert constraint.summary == "Short summary"
        assert "causal" in constraint.families
        assert "vì sao" in constraint.keywords

    def test_relation_constraint_default_values(self) -> None:
        """Test default values cho new fields."""
        constraint = RelationConstraint(
            relation_type="TEST",
            source_node_types=["paragraph"],
            target_node_types=["paragraph"],
        )

        assert constraint.summary == ""
        assert constraint.families == []
        assert constraint.keywords == []

    def test_relation_constraint_multi_family(self) -> None:
        """Test RelationConstraint with multiple families."""
        constraint = RelationConstraint(
            relation_type="CAUSES",
            source_node_types=["paragraph"],
            target_node_types=["paragraph"],
            families=["causal", "temporal"],
            keywords=["dẫn đến", "gây ra", "bởi vì"],
        )

        assert len(constraint.families) == 2
        assert "causal" in constraint.families
        assert "temporal" in constraint.families
        assert len(constraint.keywords) == 3
