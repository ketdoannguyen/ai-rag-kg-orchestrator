from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime
from enum import Enum
from typing import Any
from uuid import UUID

from pydantic import BaseModel, Field, validator

from arango_client import DataSource


class UploadSourceType(str, Enum):
    """Supported logical source types for uploaded files."""

    FILE = "FILE"
    FAQ = "FAQ"


class FileUploadInput(BaseModel):
    """Input payload for the upload_file workflow."""

    file: bytes = Field(..., description="Raw binary content of the file")
    filename: str = Field(..., min_length=1, description="Original filename")
    source_type: UploadSourceType = Field(..., description="Logical source type")
    created_by: UUID = Field(..., description="User ID who initiated the upload")
    additional_config: dict[str, Any] = Field(
        default_factory=dict, description="Optional metadata to persist with the data source"
    )
    content_type: str | None = Field(
        default=None,
        description="Optional MIME type reported by the client (validated against content)",
    )

    @validator("filename")
    def _ensure_filename(cls, value: str) -> str:
        cleaned = value.strip()
        if not cleaned:
            raise ValueError("filename cannot be empty or whitespace")
        return cleaned


class MinioInfo(BaseModel):
    """Information about the stored object in MinIO."""

    bucket: str
    path: str
    full_url: str
    md5_hash: str
    presigned_download_url: str | None = None
    expires_at: datetime | None = None


class FileUploadResult(BaseModel):
    """Workflow response returned to callers."""

    data_source: DataSource
    minio_info: MinioInfo
    duplicate: bool = Field(
        default=False,
        description="Indicates whether the workflow reused an existing data source",
    )


@dataclass(frozen=True)
class ValidatedFile:
    """Internal representation of a validated file."""

    original_name: str
    normalized_name: str
    extension: str
    mime_type: str
    size_bytes: int


@dataclass(frozen=True)
class StoredObject:
    """Internal representation of an object stored in MinIO."""

    object_name: str
    file_id: UUID


__all__ = [
    "UploadSourceType",
    "FileUploadInput",
    "MinioInfo",
    "FileUploadResult",
    "ValidatedFile",
    "StoredObject",
]
