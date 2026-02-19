"""Tests for OntologyHandler max-pooling relation filtering."""

from __future__ import annotations

from unittest.mock import Mock

import pytest
from arango_client.handlers.ontology_handler import OntologyHandler
from arango_client.models.ontology import OntologyRelation, RelationStatus, RelationTier


def _make_relation(
    rel_id: str,
    embedding: list[float],
    tiers: list[str] | None = None,
) -> OntologyRelation:
    """Create a minimal OntologyRelation with embedding."""
    return OntologyRelation(
        id=rel_id,
        name=rel_id,
        description=f"Description for {rel_id}",
        embedding=embedding,
        status=RelationStatus.ACTIVE,
        tiers=tiers or [RelationTier.TIER_2_CONTENT],
    )


class TestFilterRelationsByMaxPooling:
    """Tests for filter_relations_by_max_pooling method."""

    @pytest.fixture
    def handler(self):
        """Create OntologyHandler with mocked DB."""
        mock_db = Mock()
        return OntologyHandler(mock_db)

    @pytest.fixture
    def relations(self):
        """Create relations with known embeddings (unit vectors for easy math)."""
        return [
            _make_relation("REL_A", [1.0, 0.0, 0.0]),
            _make_relation("REL_B", [0.0, 1.0, 0.0]),
            _make_relation("REL_C", [0.0, 0.0, 1.0]),
        ]

    def test_max_pooling_basic(self, handler, relations):
        """Verify max score is selected across pairs."""
        handler.get_all_with_embeddings = Mock(return_value=relations)

        # Pair 1: centroid aligned with x-axis → high score for REL_A
        # Pair 2: centroid aligned with y-axis → high score for REL_B
        pair_embeddings = [
            ([1.0, 0.0, 0.0], [1.0, 0.0, 0.0]),  # centroid = [1,0,0]
            ([0.0, 1.0, 0.0], [0.0, 1.0, 0.0]),  # centroid = [0,1,0]
        ]

        result = handler.filter_relations_by_max_pooling(pair_embeddings, top_k=3)

        result_dict = dict(result)
        # REL_A should score ~1.0 from pair 1
        assert result_dict["REL_A"] == pytest.approx(1.0, abs=0.01)
        # REL_B should score ~1.0 from pair 2
        assert result_dict["REL_B"] == pytest.approx(1.0, abs=0.01)
        # REL_C should score ~0.0 from both pairs
        assert result_dict["REL_C"] == pytest.approx(0.0, abs=0.01)

    def test_max_pooling_heterogeneous_pairs(self, handler, relations):
        """Pairs with different top-1 relations all get represented."""
        handler.get_all_with_embeddings = Mock(return_value=relations)

        pair_embeddings = [
            ([1.0, 0.0, 0.0], [1.0, 0.0, 0.0]),  # best: REL_A
            ([0.0, 1.0, 0.0], [0.0, 1.0, 0.0]),  # best: REL_B
            ([0.0, 0.0, 1.0], [0.0, 0.0, 1.0]),  # best: REL_C
        ]

        result = handler.filter_relations_by_max_pooling(pair_embeddings, top_k=3)

        # All relations should have max score ~1.0
        result_dict = dict(result)
        assert result_dict["REL_A"] == pytest.approx(1.0, abs=0.01)
        assert result_dict["REL_B"] == pytest.approx(1.0, abs=0.01)
        assert result_dict["REL_C"] == pytest.approx(1.0, abs=0.01)

    def test_max_pooling_single_pair_same_as_centroid(self, handler, relations):
        """Single pair should produce same ranking as centroid method."""
        handler.get_all_with_embeddings = Mock(return_value=relations)

        source_emb = [0.6, 0.8, 0.0]
        target_emb = [0.6, 0.8, 0.0]

        # Max-pooling with single pair
        mp_result = handler.filter_relations_by_max_pooling([(source_emb, target_emb)], top_k=3)

        # Centroid method
        centroid_result = handler.filter_relations_by_centroid(source_emb, target_emb, top_k=3)

        # Rankings should match
        mp_ids = [r[0] for r in mp_result]
        centroid_ids = [r[0] for r in centroid_result]
        assert mp_ids == centroid_ids

        # Scores should be very close
        for (mp_id, mp_score), (c_id, c_score) in zip(mp_result, centroid_result):
            assert mp_id == c_id
            assert mp_score == pytest.approx(c_score, abs=0.01)

    def test_max_pooling_excludes_relations(self, handler, relations):
        """Excluded relations should not appear in results."""
        handler.get_all_with_embeddings = Mock(return_value=relations)

        pair_embeddings = [
            ([1.0, 0.0, 0.0], [1.0, 0.0, 0.0]),
        ]

        result = handler.filter_relations_by_max_pooling(
            pair_embeddings, top_k=3, exclude_relations={"REL_A"}
        )

        result_ids = {r[0] for r in result}
        assert "REL_A" not in result_ids
        assert "REL_B" in result_ids
        assert "REL_C" in result_ids

    def test_max_pooling_tier_filter(self, handler):
        """Tier filtering should only include matching relations."""
        tier1_rel = _make_relation("REL_T1", [1.0, 0.0, 0.0], tiers=["tier_1_structure"])
        tier2_rel = _make_relation("REL_T2", [0.0, 1.0, 0.0], tiers=["tier_2_content"])

        handler.get_by_tier_with_embeddings = Mock(return_value=[tier2_rel])

        pair_embeddings = [
            ([0.0, 1.0, 0.0], [0.0, 1.0, 0.0]),
        ]

        result = handler.filter_relations_by_max_pooling(
            pair_embeddings, top_k=3, tier="tier_2_content"
        )

        # Should only return tier_2_content relation
        result_ids = {r[0] for r in result}
        assert "REL_T2" in result_ids
        assert "REL_T1" not in result_ids
        handler.get_by_tier_with_embeddings.assert_called_once_with("tier_2_content")

    def test_max_pooling_empty_pairs(self, handler):
        """Empty pairs list should return empty result."""
        result = handler.filter_relations_by_max_pooling([], top_k=3)
        assert result == []

    def test_max_pooling_top_k_limit(self, handler):
        """Results should be limited to top_k."""
        relations = [
            _make_relation(f"REL_{i}", [float(i == j) for j in range(5)]) for i in range(5)
        ]
        handler.get_all_with_embeddings = Mock(return_value=relations)

        pair_embeddings = [
            ([1.0, 0.0, 0.0, 0.0, 0.0], [1.0, 0.0, 0.0, 0.0, 0.0]),
        ]

        result = handler.filter_relations_by_max_pooling(pair_embeddings, top_k=2)
        assert len(result) == 2

    def test_max_pooling_skips_relations_without_embedding(self, handler):
        """Relations without embeddings should be skipped."""
        rel_with = _make_relation("REL_WITH", [1.0, 0.0, 0.0])
        rel_without = OntologyRelation(
            id="REL_WITHOUT",
            name="REL_WITHOUT",
            description="No embedding",
            embedding=None,
            status=RelationStatus.ACTIVE,
            tiers=[RelationTier.TIER_2_CONTENT],
        )

        handler.get_all_with_embeddings = Mock(return_value=[rel_with, rel_without])

        pair_embeddings = [([1.0, 0.0, 0.0], [1.0, 0.0, 0.0])]
        result = handler.filter_relations_by_max_pooling(pair_embeddings, top_k=3)

        result_ids = {r[0] for r in result}
        assert "REL_WITH" in result_ids
        assert "REL_WITHOUT" not in result_ids
