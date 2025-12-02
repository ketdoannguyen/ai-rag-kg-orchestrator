"""Structure-level knowledge nodes (document/collection/topic hierarchy)."""

from __future__ import annotations

from typing import Any

from pydantic import Field, model_validator

from .attributes_metadata import StructureAttributes, StructureMetadata
from .knowledge import KnowledgeNode
from .enums import NodeType


class StructureNode(KnowledgeNode):
    """
    Structure node for CORPUS, TOPIC, or SUBTOPIC levels.
    
    This represents hierarchical structure nodes that can contain
    other nodes (both structure and detail nodes).
    
    Attributes:
        - title: Section/topic title/name
        - code: Optional code/identifier (e.g., "Chapter 1", "Part A")
        - description: Optional description/summary
    
    Metadata:
        - start_page: Start page number in source document
        - end_page: End page number in source document
        - url: Original URL (for web sources)
        - file_path: Source file path (for file sources)
        - position: Position reference
    """

    attributes: StructureAttributes = Field(
        description="Structure attributes"
    )
    metadata: StructureMetadata = Field(
        default_factory=StructureMetadata,
        description="Structure metadata"
    )

    def __init__(self, **data: Any) -> None:
        """Initialize structure node and normalize attribute/metadata types."""
        # Set default node_type for structure nodes
        if "node_type" not in data:
            data["node_type"] = NodeType.STRUCTURE
        
        # Set default level_down and level_up for structure nodes
        if "level_down" not in data:
            data["level_down"] = 0  # Default to root level if not specified
        if "level_up" not in data:
            data["level_up"] = 0  # Default value, should be calculated based on tree depth
        
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

