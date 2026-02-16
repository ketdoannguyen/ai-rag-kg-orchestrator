"""Unit tests for families module."""

from __future__ import annotations

from config.ontologies.families import (
    ALL_FAMILIES,
    CAUSAL,
    COMPARATIVE,
    ELABORATIVE,
    FAMILY_METADATA,
    PARTICIPATION,
    STRUCTURAL,
    TEMPORAL,
    get_families_for_keywords,
    get_family_metadata,
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
