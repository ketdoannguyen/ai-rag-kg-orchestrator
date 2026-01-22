"""Entity node models for Tier 3 of the knowledge graph.

This module provides:
- EntityType: Enum of entity types (Concept, Formula, Person, etc.)
- EntityNode: Pydantic model for Tier 3 entities

These entities are extracted from Tier 2 content nodes and represent
concrete knowledge entities like concepts, formulas, persons, etc.

Note: This is a scaffold - entity extraction logic will be implemented later.
"""

from __future__ import annotations

import uuid
from datetime import UTC, datetime
from enum import Enum
from typing import Any

from pydantic import BaseModel, Field


class EntityType(str, Enum):
    """Entity types for Tier 3 knowledge graph.

    Based on educational domain requirements:
    - CONCEPT: Abstract ideas, terms, definitions (KHÁI_NIỆM)
    - FORMULA: Mathematical equations and formulas (CÔNG_THỨC)
    - IMAGE_ENTITY: Structured image descriptions (HÌNH_ẢNH)
    - PERSON: Historical figures, scientists, authors (NHÂN_VẬT)
    - ORGANIZATION: Schools, institutions (TỔ_CHỨC)
    - DATE: Important dates, events (NGÀY_THÁNG)
    - LOCATION: Places, geographical entities (ĐỊA_ĐIỂM)
    """

    CONCEPT = "concept"
    FORMULA = "formula"
    IMAGE_ENTITY = "image_entity"
    PERSON = "person"
    ORGANIZATION = "organization"
    DATE = "date"
    LOCATION = "location"


class EntityNode(BaseModel):
    """Tier 3 Entity Node - extracted entities from content.

    Entities are the atomic units of knowledge extracted from
    Tier 2 content nodes. They have:
    - Canonical names and aliases
    - Type-specific attributes
    - Links back to source content
    - Embeddings for similarity search

    Example:
        >>> entity = EntityNode(
        ...     entity_type=EntityType.CONCEPT,
        ...     canonical_name="Đạo hàm",
        ...     aliases=["Đạo hàm hàm số", "Derivative"],
        ...     description="Đạo hàm là tốc độ thay đổi của hàm số...",
        ... )
    """

    id: uuid.UUID = Field(default_factory=uuid.uuid4, description="Unique identifier")
    entity_type: EntityType = Field(..., description="Type of entity")

    # Naming
    canonical_name: str = Field(..., description="Tên chuẩn (ten_chuan)")
    aliases: list[str] = Field(default_factory=list, description="Tên gọi khác (ten_goi_khac)")
    description: str = Field(default="", description="Mô tả ngắn (mo_ta_ngan)")

    # Type-specific attributes (nullable, used based on entity_type)
    # For FORMULA
    latex_string: str | None = Field(
        default=None, description="LaTeX representation (chuoi_latex)"
    )
    equation_json: dict[str, Any] | None = Field(
        default=None, description="Structured equation (json_phuong_trinh)"
    )

    # For IMAGE_ENTITY
    file_path: str | None = Field(default=None, description="Path to image file (duong_dan_file)")
    caption: str | None = Field(default=None, description="Original caption (chu_thich_goc)")
    description_json: dict[str, Any] | None = Field(
        default=None, description="Structured description (json_mo_ta)"
    )

    # For PERSON
    birth_date: str | None = Field(default=None, description="Birth date (ngay_sinh)")
    death_date: str | None = Field(default=None, description="Death date (ngay_mat)")

    # Embedding for similarity search
    embedding: list[float] | None = Field(
        default=None, description="Vector embedding for similarity search"
    )

    # Source tracking - links to Tier 2 content nodes
    source_content_ids: list[str] = Field(
        default_factory=list,
        description="IDs of content nodes this entity was extracted from",
    )

    # Domain metadata
    metadata: dict[str, Any] = Field(default_factory=dict, description="Domain-specific metadata")

    # Processing state
    is_reviewed: bool = Field(
        default=False, description="Whether entity has been reviewed for deduplication"
    )
    is_validated: bool = Field(
        default=False, description="Whether entity has been validated by strong model"
    )

    # Timestamps
    created_at: datetime = Field(
        default_factory=lambda: datetime.now(UTC), description="Creation timestamp"
    )
    updated_at: datetime | None = Field(default=None, description="Last update timestamp")

    class Config:
        """Pydantic config."""

        use_enum_values = True

    def to_document(self) -> dict[str, Any]:
        """Convert to ArangoDB document format.

        Returns:
            Dict ready for ArangoDB insertion
        """
        doc = self.model_dump(mode="json")
        doc["_key"] = str(self.id)
        return doc

    @classmethod
    def from_document(cls, doc: dict[str, Any]) -> EntityNode:
        """Create EntityNode from ArangoDB document.

        Args:
            doc: ArangoDB document dict

        Returns:
            EntityNode instance
        """
        # Handle _key -> id conversion
        if "_key" in doc and "id" not in doc:
            doc["id"] = doc["_key"]

        return cls.model_validate(doc)

    def get_searchable_text(self) -> str:
        """Get text for embedding/search.

        Returns:
            Combined text of name, aliases, and description
        """
        parts = [self.canonical_name]
        if self.aliases:
            parts.extend(self.aliases)
        if self.description:
            parts.append(self.description)
        return " ".join(parts)
