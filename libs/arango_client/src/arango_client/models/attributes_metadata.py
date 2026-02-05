"""Attributes and Metadata models for different node types."""

from __future__ import annotations

from pydantic import BaseModel, Field


# ============================================
# Paragraph Node
# ============================================
class ParagraphAttributes(BaseModel):
    """Attributes for ParagraphNode."""

    text: str = Field(..., description="The paragraph text content")


class ParagraphMetadata(BaseModel):
    """Metadata for ParagraphNode."""

    page: int | None = Field(None, description="Page number in source document")
    bbox: list[float] = Field(default_factory=list, description="Bounding box [x1, y1, x2, y2]")
    position: str | None = Field(
        None, description="Position reference (e.g., 'SGK Vật Lý 12, Trang 6')"
    )


# ============================================
# Image Node
# ============================================
class ImageAttributes(BaseModel):
    """Attributes for ImageNode."""

    url: str | None = Field(None, description="Image URL (for web sources)")
    file_path: str | None = Field(None, description="Local file path or MinIO object path")
    alt_text: str | None = Field(None, description="Alternative text for the image")
    caption: str | None = Field(None, description="Image caption")
    width: int | None = Field(None, description="Image width in pixels")
    height: int | None = Field(None, description="Image height in pixels")
    extracted_text: str | None = Field(
        None, description="Text extracted from image via OCR (if available)"
    )


class ImageMetadata(BaseModel):
    """Metadata for ImageNode."""

    page: int | None = Field(None, description="Page number where image is located")
    bbox: list[float] = Field(default_factory=list, description="Bounding box [x1, y1, x2, y2]")
    position: str | None = Field(None, description="Position reference")


# ============================================
# Table Node
# ============================================
class TableAttributes(BaseModel):
    """Attributes for TableNode."""

    headers: list[str] = Field(default_factory=list, description="Table column headers")
    rows: list[list[str]] = Field(
        default_factory=list, description="Table rows (list of cell values)"
    )
    caption: str | None = Field(None, description="Table caption")
    extracted_text: str | None = Field(
        None, description="Text representation of table (markdown or plain text)"
    )


class TableMetadata(BaseModel):
    """Metadata for TableNode."""

    page: int | None = Field(None, description="Page number where table is located")
    bbox: list[float] = Field(default_factory=list, description="Bounding box [x1, y1, x2, y2]")
    position: str | None = Field(None, description="Position reference")


# ============================================
# Structure Node (hierarchical sections)
# ============================================
class StructureAttributes(BaseModel):
    """Attributes for StructureNode (corpus/topic/subtopic levels)."""

    title: str = Field(..., description="Section/topic title/name")
    code: str | None = Field(
        None,
        description="Optional code/identifier (e.g., 'Chapter', 'Part', 'I', 'II', 'III', '1', '2', ...)",
    )

    def model_post_init(self, __context) -> None:
        """Extract code from title after initialization if code is None."""
        if not self.code and self.title:
            extracted_code = self._extract_code_from_title(self.title)
            if extracted_code:
                object.__setattr__(self, "code", extracted_code)

    @staticmethod
    def _extract_code_from_title(title: str) -> str | None:
        """Extract code/number from title string."""
        import re

        # Pattern for Roman numerals, Arabic numbers, or letters at the start
        patterns = [
            r"^((?:Chương|Chapter|Phần|Part|Bài|Lesson)\s+[\dIVXLCDM]+)",
            r"^([\dIVXLCDM]+(?:\.\d+)*)\s*[.:\-)\]]\s*",
            r"^([A-Z])\s*[.:\-)\]]\s*",
        ]

        for pattern in patterns:
            match = re.match(pattern, title, re.IGNORECASE)
            if match:
                return match.group(1).strip()

        return None


class StructureMetadata(BaseModel):
    """Metadata for StructureNode (corpus/topic/subtopic levels)."""

    start_page: int | None = Field(None, description="Start page number in source document")
    end_page: int | None = Field(None, description="End page number in source document")
    url: str | None = Field(None, description="Original URL (for web sources)")
    file_path: str | None = Field(None, description="Source file path (for file sources)")
    position: str | None = Field(None, description="Position reference")
