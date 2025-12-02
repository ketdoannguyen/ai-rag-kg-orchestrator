from __future__ import annotations

from enum import Enum
from typing import Any
from uuid import UUID

from pydantic import BaseModel, Field, NonNegativeInt


class FileIngestionStatus(str, Enum):
    COMPLETE = "complete"
    FAILED = "failed"


class FileIngestionInput(BaseModel):
    """Input model for file ingestion workflow."""

    data_source_id: UUID = Field(..., description="ID of the DataSource to ingest")
    enable_ocr: bool = False
    max_retries: NonNegativeInt = Field(default=2, le=5)
    timeout_seconds: float = Field(default=300.0, gt=0.0, le=1800.0)


class FileIngestionResult(BaseModel):
    """Output model for file ingestion workflow."""

    status: FileIngestionStatus
    job_id: str
    data_source_id: UUID
    # Root of the knowledge tree that was persisted
    root_node: Any | None = None
    error: str | None = None

