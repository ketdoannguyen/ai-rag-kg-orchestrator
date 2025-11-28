"""Common data models for all ingestion workflows (web, file, facebook)."""

from __future__ import annotations

import uuid
from enum import Enum
from typing import Any

from pydantic import BaseModel, Field


class ContentType(str, Enum):
    """Types of content blocks."""

    PARAGRAPH = "Paragraph"
    SECTION = "Section"
    IMAGE = "Image"
    TABLE = "Table"
    LIST = "List"
    CODE = "Code"


class SourceType(str, Enum):
    """Source types for ingested content."""

    WEB = "web"
    FILE = "file"
    FACEBOOK = "facebook"


class Source(BaseModel):
    """Source reference information for content blocks."""

    source_type: SourceType = Field(..., description="Type of source (web, file, facebook)")
    url: str | None = Field(None, description="Source URL (for web/facebook)")
    file_path: str | None = Field(None, description="Source file path (for file)")
    page_number: int | None = Field(None, description="Page number in source document")
    position: str | None = Field(None, description="Position reference (e.g., 'SGK Vật Lý 12, Trang 6')")
    metadata: dict[str, Any] = Field(
        default_factory=dict, description="Additional source metadata (crawled_at, job_id, etc.)"
    )


class Paragraph(BaseModel):
    """A paragraph content block containing text."""

    id: str = Field(default_factory=lambda: str(uuid.uuid4()))
    content_type: ContentType = ContentType.PARAGRAPH
    text: str = Field(..., description="Text content of the paragraph")
    page: int | None = Field(None, description="Page number where this paragraph is located")
    bbox: list[int] = Field(default_factory=list, description="Bounding box [x1, y1, x2, y2]")


class Image(BaseModel):
    """An image content block."""

    id: str = Field(default_factory=lambda: str(uuid.uuid4()))
    content_type: ContentType = ContentType.IMAGE
    url: str | None = Field(None, description="Image URL (for web sources)")
    file_path: str | None = Field(None, description="Local file path or MinIO object path")
    alt_text: str | None = Field(None, description="Alternative text for the image")
    caption: str | None = Field(None, description="Image caption")
    width: int | None = Field(None, description="Image width in pixels")
    height: int | None = Field(None, description="Image height in pixels")
    extracted_text: str | None = Field(
        None, description="Text extracted from image via OCR (if available)"
    )
    page: int | None = Field(None, description="Page number where this image is located")
    bbox: list[int] = Field(default_factory=list, description="Bounding box [x1, y1, x2, y2]")


class Table(BaseModel):
    """A table content block."""

    id: str = Field(default_factory=lambda: str(uuid.uuid4()))
    content_type: ContentType = ContentType.TABLE
    headers: list[str] = Field(default_factory=list, description="Table column headers")
    rows: list[list[str]] = Field(default_factory=list, description="Table rows (list of cell values)")
    caption: str | None = Field(None, description="Table caption")
    extracted_text: str | None = Field(
        None, description="Text representation of table (markdown or plain text)"
    )
    page: int | None = Field(None, description="Page number where this table is located")
    bbox: list[int] = Field(default_factory=list, description="Bounding box [x1, y1, x2, y2]")


class Section(BaseModel):
    """A section that can contain subsections and content blocks (paragraphs, images, tables)."""

    id: str = Field(default_factory=lambda: str(uuid.uuid4()))
    content_type: ContentType = ContentType.SECTION
    title: str = Field(..., description="Section title/name")
    level: int = Field(..., ge=1, le=10, description="Section hierarchical level/depth (1-10)")
    start_page: int | None = Field(None, description="Start page number in source document")
    end_page: int | None = Field(None, description="End page number in source document")
    children: list[Section | Paragraph | Image | Table] = Field(
        default_factory=list,
        description="Nested sections and content blocks (paragraphs, images, tables)",
    )
    source: Source | None = Field(
        None, description="Source reference (only at root level, not in nested sections)"
    )
    bbox: list[int] = Field(default_factory=list, description="Bounding box [x1, y1, x2, y2]")

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary with proper serialization."""
        return self.model_dump(mode="json", exclude_none=False)

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> Section:
        """Create from dictionary."""
        return cls(**data)
