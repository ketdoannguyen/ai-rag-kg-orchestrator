"""Detail-level knowledge nodes (DETAIL semantic level)."""

from __future__ import annotations

from typing import Any

from pydantic import Field, model_validator

from .attributes_metadata import (
    ImageAttributes,
    ImageMetadata,
    ParagraphAttributes,
    ParagraphMetadata,
    TableAttributes,
    TableMetadata,
)
from .base import KnowledgeNode
from .enums import SemanticLevel


class ParagraphNode(KnowledgeNode):
    """
    Paragraph node for DETAIL level.
    
    Attributes:
        - text: The paragraph text content
        - title: Optional paragraph title/heading
    
    Metadata:
        - page: Page number in source document
        - bbox: Bounding box [x1, y1, x2, y2]
        - position: Position reference (e.g., 'SGK Vật Lý 12, Trang 6')
    """

    semantic_level: SemanticLevel = Field(
        default=SemanticLevel.DETAIL,
        description="Always DETAIL for paragraphs"
    )
    physical_level: int = Field(
        default=-1,
        description="Physical level for DETAIL nodes is always -1"
    )
    attributes: ParagraphAttributes = Field(
        description="Paragraph attributes"
    )
    metadata: ParagraphMetadata = Field(
        default_factory=ParagraphMetadata,
        description="Paragraph metadata"
    )

    def __init__(self, **data: Any) -> None:
        """Initialize with default semantic level and convert dict to models."""
        if "semantic_level" not in data:
            data["semantic_level"] = SemanticLevel.DETAIL
        
        # Force physical_level to -1 for DETAIL nodes
        if "physical_level" not in data:
            data["physical_level"] = -1
        else:
            # Override any provided physical_level to ensure it's -1
            data["physical_level"] = -1
        
        # Convert dict to models if needed
        if "attributes" in data and isinstance(data["attributes"], dict):
            data["attributes"] = ParagraphAttributes(**data["attributes"])
        elif "attributes" not in data:
            data["attributes"] = ParagraphAttributes(text="")
        
        if "metadata" in data and isinstance(data["metadata"], dict):
            data["metadata"] = ParagraphMetadata(**data["metadata"])
        elif "metadata" not in data:
            data["metadata"] = ParagraphMetadata()
        
        super().__init__(**data)

    @model_validator(mode="before")
    @classmethod
    def validate_data(cls, data: Any) -> Any:
        """Convert dict attributes/metadata to models before validation."""
        if isinstance(data, dict):
            if "attributes" in data and isinstance(data["attributes"], dict):
                data["attributes"] = ParagraphAttributes(**data["attributes"])
            if "metadata" in data and isinstance(data["metadata"], dict):
                data["metadata"] = ParagraphMetadata(**data["metadata"])
        return data


class ImageNode(KnowledgeNode):
    """
    Image node for DETAIL level.
    
    Attributes:
        - url: Image URL (for web sources)
        - file_path: Local file path or MinIO object path
        - alt_text: Alternative text for the image
        - caption: Image caption
        - width: Image width in pixels
        - height: Image height in pixels
        - extracted_text: Text extracted from image via OCR
    
    Metadata:
        - page: Page number where image is located
        - bbox: Bounding box [x1, y1, x2, y2]
        - position: Position reference
    """

    semantic_level: SemanticLevel = Field(
        default=SemanticLevel.DETAIL,
        description="Always DETAIL for images"
    )
    physical_level: int = Field(
        default=-1,
        description="Physical level for DETAIL nodes is always -1"
    )
    attributes: ImageAttributes = Field(
        default_factory=ImageAttributes,
        description="Image attributes"
    )
    metadata: ImageMetadata = Field(
        default_factory=ImageMetadata,
        description="Image metadata"
    )

    def __init__(self, **data: Any) -> None:
        """Initialize with default semantic level and convert dict to models."""
        if "semantic_level" not in data:
            data["semantic_level"] = SemanticLevel.DETAIL
        
        # Force physical_level to -1 for DETAIL nodes
        if "physical_level" not in data:
            data["physical_level"] = -1
        else:
            # Override any provided physical_level to ensure it's -1
            data["physical_level"] = -1
        
        # Convert dict to models if needed
        if "attributes" in data and isinstance(data["attributes"], dict):
            data["attributes"] = ImageAttributes(**data["attributes"])
        elif "attributes" not in data:
            data["attributes"] = ImageAttributes()
        
        if "metadata" in data and isinstance(data["metadata"], dict):
            data["metadata"] = ImageMetadata(**data["metadata"])
        elif "metadata" not in data:
            data["metadata"] = ImageMetadata()
        
        super().__init__(**data)

    @model_validator(mode="before")
    @classmethod
    def validate_data(cls, data: Any) -> Any:
        """Convert dict attributes/metadata to models before validation."""
        if isinstance(data, dict):
            if "attributes" in data and isinstance(data["attributes"], dict):
                data["attributes"] = ImageAttributes(**data["attributes"])
            if "metadata" in data and isinstance(data["metadata"], dict):
                data["metadata"] = ImageMetadata(**data["metadata"])
        return data


class TableNode(KnowledgeNode):
    """
    Table node for DETAIL level.
    
    Attributes:
        - headers: Table column headers
        - rows: Table rows (list of cell values)
        - caption: Table caption
        - extracted_text: Text representation of table (markdown or plain text)
    
    Metadata:
        - page: Page number where table is located
        - bbox: Bounding box [x1, y1, x2, y2]
        - position: Position reference
    """

    semantic_level: SemanticLevel = Field(
        default=SemanticLevel.DETAIL,
        description="Always DETAIL for tables"
    )
    physical_level: int = Field(
        default=-1,
        description="Physical level for DETAIL nodes is always -1"
    )
    attributes: TableAttributes = Field(
        default_factory=TableAttributes,
        description="Table attributes"
    )
    metadata: TableMetadata = Field(
        default_factory=TableMetadata,
        description="Table metadata"
    )

    def __init__(self, **data: Any) -> None:
        """Initialize with default semantic level and convert dict to models."""
        if "semantic_level" not in data:
            data["semantic_level"] = SemanticLevel.DETAIL
        
        # Force physical_level to -1 for DETAIL nodes
        if "physical_level" not in data:
            data["physical_level"] = -1
        else:
            # Override any provided physical_level to ensure it's -1
            data["physical_level"] = -1
        
        # Convert dict to models if needed
        if "attributes" in data and isinstance(data["attributes"], dict):
            data["attributes"] = TableAttributes(**data["attributes"])
        elif "attributes" not in data:
            data["attributes"] = TableAttributes()
        
        if "metadata" in data and isinstance(data["metadata"], dict):
            data["metadata"] = TableMetadata(**data["metadata"])
        elif "metadata" not in data:
            data["metadata"] = TableMetadata()
        
        super().__init__(**data)

    @model_validator(mode="before")
    @classmethod
    def validate_data(cls, data: Any) -> Any:
        """Convert dict attributes/metadata to models before validation."""
        if isinstance(data, dict):
            if "attributes" in data and isinstance(data["attributes"], dict):
                data["attributes"] = TableAttributes(**data["attributes"])
            if "metadata" in data and isinstance(data["metadata"], dict):
                data["metadata"] = TableMetadata(**data["metadata"])
        return data

