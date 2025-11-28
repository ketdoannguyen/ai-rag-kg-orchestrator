from __future__ import annotations

from enum import Enum
from pathlib import Path
from typing import Any

from pydantic import BaseModel, Field, NonNegativeInt, validator


class FileIngestionStatus(str, Enum):
    COMPLETE = "complete"
    FAILED = "failed"


class SourceMetadata(BaseModel):
    """Metadata for the source PDF file."""

    local_path: str
    minio_object: str
    size_bytes: int


class MarkerFileMetadata(BaseModel):
    """Metadata for a single file output from Marker."""

    local_path: str
    minio_object: str


class MarkerImageMetadata(BaseModel):
    """Metadata for a single image extracted by Marker."""

    filename: str
    content_type: str
    local_path: str
    minio_object: str


class MarkerOutputMetadata(BaseModel):
    """Metadata for all Marker outputs."""

    request_id: str
    json: MarkerFileMetadata | None = None
    markdown: MarkerFileMetadata | None = None
    raw_payload: MarkerFileMetadata | None = None
    images: list[MarkerImageMetadata] = Field(default_factory=list)


class FileIngestionInput(BaseModel):
    """Input model for file ingestion workflow."""

    file_path: Path
    enable_ocr: bool = False
    max_retries: NonNegativeInt = Field(default=2, le=5)
    timeout_seconds: float = Field(default=300.0, gt=0.0, le=1800.0)

    @validator("file_path")
    def _validate_file_exists(cls, v: Path) -> Path:
        """Validate that the file exists and is a PDF."""
        resolved = v.expanduser().resolve()
        if not resolved.exists():
            raise ValueError(f"File not found: {resolved}")
        if not resolved.is_file():
            raise ValueError(f"Path is not a file: {resolved}")
        if resolved.suffix.lower() != ".pdf":
            raise ValueError(f"Only PDF files are supported, got: {resolved.suffix}")
        return resolved


class FileIngestionResult(BaseModel):
    """Output model for file ingestion workflow."""

    status: FileIngestionStatus
    job_id: str
    output_json_path: str | None = None
    source: SourceMetadata
    marker: MarkerOutputMetadata
    error: str | None = None

