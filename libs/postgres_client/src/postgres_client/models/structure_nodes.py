"""Structure-level knowledge nodes (CORPUS, TOPIC, SUBTOPIC semantic levels)."""

from __future__ import annotations

from typing import Any

from pydantic import Field, model_validator

from .attributes_metadata import StructureAttributes, StructureMetadata
from .base import KnowledgeNode
from .enums import SemanticLevel


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
        """Initialize structure node."""
        # Validate semantic level
        semantic_level = data.get("semantic_level")
        if semantic_level and semantic_level not in [
            SemanticLevel.CORPUS,
            SemanticLevel.TOPIC,
            SemanticLevel.SUBTOPIC,
        ]:
            raise ValueError(
                f"StructureNode must have semantic_level of CORPUS, TOPIC, or SUBTOPIC, "
                f"got {semantic_level}"
            )
        
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


class CorpusNode(StructureNode):
    """Corpus-level node (document/collection level)."""

    semantic_level: SemanticLevel = Field(
        default=SemanticLevel.CORPUS,
        description="Always CORPUS for corpus nodes"
    )

    def __init__(self, **data: Any) -> None:
        """Initialize with default semantic level."""
        if "semantic_level" not in data:
            data["semantic_level"] = SemanticLevel.CORPUS
        super().__init__(**data)


class TopicNode(StructureNode):
    """Topic-level node (major topic level)."""

    semantic_level: SemanticLevel = Field(
        default=SemanticLevel.TOPIC,
        description="Always TOPIC for topic nodes"
    )

    def __init__(self, **data: Any) -> None:
        """Initialize with default semantic level."""
        if "semantic_level" not in data:
            data["semantic_level"] = SemanticLevel.TOPIC
        super().__init__(**data)


class SubtopicNode(StructureNode):
    """Subtopic-level node (sub-topic level)."""

    semantic_level: SemanticLevel = Field(
        default=SemanticLevel.SUBTOPIC,
        description="Always SUBTOPIC for subtopic nodes"
    )

    def __init__(self, **data: Any) -> None:
        """Initialize with default semantic level."""
        if "semantic_level" not in data:
            data["semantic_level"] = SemanticLevel.SUBTOPIC
        super().__init__(**data)

