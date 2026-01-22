"""Structure-level knowledge nodes (document/collection/topic hierarchy) for ArangoDB."""

from __future__ import annotations

from typing import Any

from pydantic import Field, model_validator

from .attributes_metadata import StructureAttributes, StructureMetadata
from .enums import NodeType
from .knowledge import KnowledgeNode


class StructureNode(KnowledgeNode):
    """
    Structure node for CORPUS, TOPIC, or SUBTOPIC levels.

    This represents hierarchical structure nodes that can contain
    other nodes (both structure and detail nodes).

    Structure-specific fields:
    - embedding: Vector embedding for semantic search on structure
    - summary: Summary text for this structure section
    - children: Child nodes (transient, not stored in DB)

    Attributes:
        - title: Section/topic title/name
        - code: Optional code/identifier (e.g., "Chapter 1", "Part A")

    Metadata:
        - start_page: Start page number in source document
        - end_page: End page number in source document
        - url: Original URL (for web sources)
        - file_path: Source file path (for file sources)
        - position: Position reference
    """

    # Structure-specific fields
    embedding: list[float] | None = Field(
        None, description="Structure embedding vector for semantic search"
    )
    summary: str | None = Field(None, description="Summary text for this structure section")
    children: list[KnowledgeNode] = Field(
        default_factory=list,
        description="Child nodes (paragraphs, images, tables, or sub-structures) - not stored in DB",
    )

    # Override base attributes/metadata with typed versions
    attributes: StructureAttributes = Field(description="Structure attributes")
    metadata: StructureMetadata = Field(
        default_factory=StructureMetadata, description="Structure metadata"
    )

    def __init__(self, **data: Any) -> None:
        """Initialize structure node and normalize attribute/metadata types."""
        # Set default node_type for structure nodes
        if "node_type" not in data:
            data["node_type"] = NodeType.STRUCTURE

        # Set default level for structure nodes (0 means unset)
        if "level" not in data:
            data["level"] = 0

        # Convert dict to models if needed
        if "attributes" in data and isinstance(data["attributes"], dict):
            data["attributes"] = StructureAttributes(**data["attributes"])
        elif "attributes" not in data:
            data["attributes"] = StructureAttributes(title="")

        if "metadata" in data and isinstance(data["metadata"], dict):
            data["metadata"] = StructureMetadata(**data["metadata"])
        elif "metadata" not in data:
            data["metadata"] = StructureMetadata()

        super().__init__(**data)

    @model_validator(mode="before")
    @classmethod
    def validate_data(cls, data: Any) -> Any:
        """Convert dict attributes/metadata to models before validation."""
        if isinstance(data, dict):
            if "attributes" in data and isinstance(data["attributes"], dict):
                data["attributes"] = StructureAttributes(**data["attributes"])
            if "metadata" in data and isinstance(data["metadata"], dict):
                data["metadata"] = StructureMetadata(**data["metadata"])
        return data


class RootNode(StructureNode):
    """
    Root node for the knowledge tree.

    Root nodes always have level=1.
    """

    def __init__(self, **data: Any) -> None:
        """Initialize root node with level=1."""
        # Root always has level=1
        data["level"] = 1
        super().__init__(**data)
