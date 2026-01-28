"""Unit tests for families module."""

from __future__ import annotations

from config.ontologies.families import (
    ALL_FAMILIES,
    CAUSAL,
    COMPARATIVE,
    ELABORATIVE,
    FAMILY_AFFINITY_MATRIX,
    FAMILY_METADATA,
    PARTICIPATION,
    STRUCTURAL,
    TEMPORAL,
    get_affinity,
    get_families_for_keywords,
    get_family_metadata,
    get_max_affinity,
)


class TestFamilyDefinitions:
    """Test family definitions."""

    def test_all_families_defined(self) -> None:
        """Test that all 6 families are defined."""
        assert len(ALL_FAMILIES) == 6
        assert CAUSAL in ALL_FAMILIES
        assert TEMPORAL in ALL_FAMILIES
        assert COMPARATIVE in ALL_FAMILIES
        assert ELABORATIVE in ALL_FAMILIES
        assert PARTICIPATION in ALL_FAMILIES
        assert STRUCTURAL in ALL_FAMILIES

    def test_all_families_have_metadata(self) -> None:
        """Test that all families have metadata."""
        for family in ALL_FAMILIES:
            assert family in FAMILY_METADATA
            metadata = FAMILY_METADATA[family]
            assert "name" in metadata
            assert "description" in metadata
            assert "keywords" in metadata
            assert isinstance(metadata["keywords"], list)
            assert len(metadata["keywords"]) > 0


class TestAffinityMatrix:
    """Test affinity matrix."""

    def test_affinity_matrix_complete(self) -> None:
        """Test that affinity matrix has all query types and families."""
        query_types = [
            "causal",
            "comparison",
            "figure",
            "concept",
            "factual",
            "lesson_synthesis",
            "structural",
            "other",
        ]

        for qt in query_types:
            assert qt in FAMILY_AFFINITY_MATRIX
            for family in ALL_FAMILIES:
                assert family in FAMILY_AFFINITY_MATRIX[qt]
                # Check value is in valid range
                value = FAMILY_AFFINITY_MATRIX[qt][family]
                assert 0.0 <= value <= 1.0

    def test_diagonal_affinity_is_one(self) -> None:
        """Test that matching query type and family have affinity 1.0."""
        assert FAMILY_AFFINITY_MATRIX["causal"][CAUSAL] == 1.0
        assert FAMILY_AFFINITY_MATRIX["comparison"][COMPARATIVE] == 1.0
        assert FAMILY_AFFINITY_MATRIX["structural"][STRUCTURAL] == 1.0
        assert FAMILY_AFFINITY_MATRIX["concept"][ELABORATIVE] == 1.0
        assert FAMILY_AFFINITY_MATRIX["figure"][PARTICIPATION] == 1.0

    def test_other_query_type_has_zero_affinity(self) -> None:
        """Test that 'other' query type has zero affinity with all families."""
        for family in ALL_FAMILIES:
            assert FAMILY_AFFINITY_MATRIX["other"][family] == 0.0


class TestGetAffinity:
    """Test get_affinity function."""

    def test_get_affinity_exact_match(self) -> None:
        """Test affinity for exact query type - family match."""
        assert get_affinity("causal", "causal") == 1.0
        assert get_affinity("comparison", "comparative") == 1.0
        assert get_affinity("other", "causal") == 0.0

    def test_get_affinity_case_insensitive(self) -> None:
        """Test that get_affinity is case insensitive."""
        assert get_affinity("CAUSAL", "CAUSAL") == 1.0
        assert get_affinity("Causal", "causal") == 1.0
        assert get_affinity("causal", "TEMPORAL") == 0.6

    def test_get_affinity_unknown_query_type(self) -> None:
        """Test default value for unknown query type."""
        assert get_affinity("unknown", "causal") == 0.5

    def test_get_affinity_unknown_family(self) -> None:
        """Test default value for unknown family."""
        assert get_affinity("causal", "unknown") == 0.5


class TestGetMaxAffinity:
    """Test get_max_affinity function."""

    def test_get_max_affinity_single_family(self) -> None:
        """Test max affinity with single family."""
        assert get_max_affinity("causal", ["causal"]) == 1.0
        assert get_max_affinity("causal", ["temporal"]) == 0.6

    def test_get_max_affinity_multiple_families(self) -> None:
        """Test max affinity returns highest value."""
        # causal→causal=1.0, causal→temporal=0.6 → max=1.0
        assert get_max_affinity("causal", ["causal", "temporal"]) == 1.0

        # comparison→causal=0.2, comparison→comparative=1.0 → max=1.0
        assert get_max_affinity("comparison", ["causal", "comparative"]) == 1.0

    def test_get_max_affinity_empty_families(self) -> None:
        """Test default value for empty families list."""
        assert get_max_affinity("causal", []) == 0.5


class TestGetFamilyMetadata:
    """Test get_family_metadata function."""

    def test_get_family_metadata_exists(self) -> None:
        """Test getting metadata for existing family."""
        metadata = get_family_metadata("causal")
        assert metadata is not None
        assert metadata["name"] == "Nhân quả"
        assert "tại sao" in metadata["keywords"]

    def test_get_family_metadata_case_insensitive(self) -> None:
        """Test case insensitive lookup."""
        assert get_family_metadata("CAUSAL") is not None
        assert get_family_metadata("Causal") is not None

    def test_get_family_metadata_not_found(self) -> None:
        """Test None return for unknown family."""
        assert get_family_metadata("unknown") is None


class TestGetFamiliesForKeywords:
    """Test get_families_for_keywords function."""

    def test_get_families_for_causal_keywords(self) -> None:
        """Test detecting causal family from keywords."""
        families = get_families_for_keywords("Tại sao cuộc cách mạng xảy ra?")
        assert CAUSAL in families

    def test_get_families_for_temporal_keywords(self) -> None:
        """Test detecting temporal family from keywords."""
        families = get_families_for_keywords("Trước năm 1945 có những sự kiện gì?")
        assert TEMPORAL in families

    def test_get_families_for_comparative_keywords(self) -> None:
        """Test detecting comparative family from keywords."""
        families = get_families_for_keywords("So sánh hai cuộc khởi nghĩa")
        assert COMPARATIVE in families

    def test_get_families_for_multiple_keywords(self) -> None:
        """Test detecting multiple families."""
        families = get_families_for_keywords("Tại sao trước năm 1945 có sự khác biệt?")
        assert CAUSAL in families
        assert TEMPORAL in families

    def test_get_families_no_match(self) -> None:
        """Test empty result when no keywords match."""
        families = get_families_for_keywords("Hello world")
        assert families == []
