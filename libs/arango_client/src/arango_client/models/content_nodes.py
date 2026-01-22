"""Unified content node model for Tier 2 (paragraphs, images, tables).

This module provides a unified ContentNode that replaces the separate
ParagraphNode, ImageNode, and TableNode models. The node_type field
acts as a discriminator for type-specific attributes.

Benefits:
- Single collection (content_nodes) instead of three
- Simplified graph traversal and queries
- Consistent deduplication handling
- Built-in support for merge/dedup tracking
"""

from __future__ import annotations

from typing import Any, Literal

from pydantic import Field, model_validator

from .attributes_metadata import (
    ImageAttributes,
    ImageMetadata,
    ParagraphAttributes,
    ParagraphMetadata,
    TableAttributes,
    TableMetadata,
)
from .enums import NodeType
from .knowledge import KnowledgeNode

# Type alias for content types (Tier 2)
ContentNodeType = Literal["paragraph", "image", "table"]


class ContentNode(KnowledgeNode):
    """Unified content node for Tier 2 (paragraphs, images, tables).

    Replaces ParagraphNode, ImageNode, TableNode with a single model
    that uses node_type as discriminator for type-specific handling.

    Key Features:
    - node_type: Discriminator field (PARAGRAPH, IMAGE, TABLE)
    - embedding: Vector embedding for semantic search
    - text: Content text used for embedding
    - is_reviewed: Deduplication review status
    - merge_candidates: Potential merge targets from dedup process

    Type-Specific Attributes:
    - PARAGRAPH: ParagraphAttributes (text)
    - IMAGE: ImageAttributes (url, file_path, alt_text, etc.)
    - TABLE: TableAttributes (headers, rows, caption, etc.)
    """

    # Node type discriminator
    node_type: NodeType = Field(
        ...,
        description="Content type: PARAGRAPH, IMAGE, or TABLE",
    )

    # Content-specific fields (shared by all content types)
    embedding: list[float] | None = Field(
        None, description="Content embedding vector for semantic search"
    )
    text: str | None = Field(
        None, description="Content text used for embedding (extracted from content)"
    )

    # Deduplication tracking fields
    is_reviewed: bool = Field(
        default=False,
        description="Whether this node has been reviewed for deduplication",
    )
    merge_candidates: list[str] = Field(
        default_factory=list,
        description="List of potential merge target node IDs from dedup process",
    )

    # Type-specific attributes (stored as dict, validated at runtime)
    attributes: dict[str, Any] = Field(
        default_factory=dict, description="Type-specific attributes"
    )
    metadata: dict[str, Any] = Field(default_factory=dict, description="Type-specific metadata")

    def __init__(self, **data: Any) -> None:
        """Initialize content node with default level=0."""
        # Content nodes are always at leaf level
        if "level" not in data:
            data["level"] = 0
        super().__init__(**data)

    @model_validator(mode="before")
    @classmethod
    def validate_and_convert_data(cls, data: Any) -> Any:
        """Convert type-specific attributes/metadata to appropriate models."""
        if not isinstance(data, dict):
            return data

        node_type = data.get("node_type")
        if isinstance(node_type, NodeType):
            node_type = node_type.value

        # Convert attributes based on node_type
        if "attributes" in data and isinstance(data["attributes"], dict):
            attrs = data["attributes"]
            if node_type == "paragraph":
                data["attributes"] = ParagraphAttributes(**attrs).model_dump()
            elif node_type == "image":
                data["attributes"] = ImageAttributes(**attrs).model_dump()
            elif node_type == "table":
                data["attributes"] = TableAttributes(**attrs).model_dump()

        # Convert metadata based on node_type
        if "metadata" in data and isinstance(data["metadata"], dict):
            meta = data["metadata"]
            if node_type == "paragraph":
                data["metadata"] = ParagraphMetadata(**meta).model_dump()
            elif node_type == "image":
                data["metadata"] = ImageMetadata(**meta).model_dump()
            elif node_type == "table":
                data["metadata"] = TableMetadata(**meta).model_dump()

        return data

    @model_validator(mode="after")
    def validate_content_type(self) -> ContentNode:
        """Validate that node_type is a valid content type."""
        valid_types = {NodeType.PARAGRAPH, NodeType.IMAGE, NodeType.TABLE}
        if self.node_type not in valid_types:
            raise ValueError(
                f"ContentNode node_type must be one of {valid_types}, got {self.node_type}"
            )
        return self

    def get_typed_attributes(
        self,
    ) -> ParagraphAttributes | ImageAttributes | TableAttributes:
        """Get attributes as the appropriate typed model."""
        if self.node_type == NodeType.PARAGRAPH:
            return ParagraphAttributes(**self.attributes)
        elif self.node_type == NodeType.IMAGE:
            return ImageAttributes(**self.attributes)
        elif self.node_type == NodeType.TABLE:
            return TableAttributes(**self.attributes)
        else:
            raise ValueError(f"Unknown node type: {self.node_type}")

    def get_typed_metadata(
        self,
    ) -> ParagraphMetadata | ImageMetadata | TableMetadata:
        """Get metadata as the appropriate typed model."""
        if self.node_type == NodeType.PARAGRAPH:
            return ParagraphMetadata(**self.metadata)
        elif self.node_type == NodeType.IMAGE:
            return ImageMetadata(**self.metadata)
        elif self.node_type == NodeType.TABLE:
            return TableMetadata(**self.metadata)
        else:
            raise ValueError(f"Unknown node type: {self.node_type}")

    def get_content_text(self) -> str:
        """Get the primary text content for this node.

        For paragraphs: the text attribute
        For images: alt_text or extracted_text or caption
        For tables: extracted_text or caption
        """
        if self.text:
            return self.text

        if self.node_type == NodeType.PARAGRAPH:
            return self.attributes.get("text", "")
        elif self.node_type == NodeType.IMAGE:
            return (
                self.attributes.get("extracted_text")
                or self.attributes.get("alt_text")
                or self.attributes.get("caption")
                or ""
            )
        elif self.node_type == NodeType.TABLE:
            return self.attributes.get("extracted_text") or self.attributes.get("caption") or ""
        return ""

    @classmethod
    def from_paragraph(
        cls,
        text: str,
        **kwargs: Any,
    ) -> ContentNode:
        """Create a ContentNode from paragraph content.

        Args:
            text: The paragraph text content
            **kwargs: Additional fields (source_id, parent_id, metadata, etc.)
        """
        return cls(
            node_type=NodeType.PARAGRAPH,
            attributes=ParagraphAttributes(text=text).model_dump(),
            metadata=kwargs.pop("metadata", {}),
            text=text,  # Use same text for embedding
            **kwargs,
        )

    @classmethod
    def from_image(
        cls,
        url: str | None = None,
        file_path: str | None = None,
        alt_text: str | None = None,
        **kwargs: Any,
    ) -> ContentNode:
        """Create a ContentNode from image content."""
        attrs = ImageAttributes(
            url=url,
            file_path=file_path,
            alt_text=alt_text,
        )
        return cls(
            node_type=NodeType.IMAGE,
            attributes=attrs.model_dump(),
            metadata=kwargs.pop("metadata", {}),
            text=alt_text,  # Use alt_text for embedding
            **kwargs,
        )

    @classmethod
    def from_table(
        cls,
        headers: list[str] | None = None,
        rows: list[list[str]] | None = None,
        caption: str | None = None,
        **kwargs: Any,
    ) -> ContentNode:
        """Create a ContentNode from table content."""
        attrs = TableAttributes(
            headers=headers or [],
            rows=rows or [],
            caption=caption,
        )
        return cls(
            node_type=NodeType.TABLE,
            attributes=attrs.model_dump(),
            metadata=kwargs.pop("metadata", {}),
            text=caption,  # Use caption for embedding
            **kwargs,
        )
