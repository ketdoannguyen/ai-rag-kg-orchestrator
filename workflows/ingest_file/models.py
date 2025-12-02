from __future__ import annotations

from enum import Enum
from typing import Any

from pydantic import BaseModel, Field, NonNegativeInt

from postgres_client.models.system_admin import DataSource


class FileIngestionStatus(str, Enum):
    COMPLETE = "complete"
    FAILED = "failed"


class FileIngestionInput(BaseModel):
    """Input model for file ingestion workflow."""

    data_source_id: str = Field(..., description="ID of the DataSource to ingest (UUID as string)")
    enable_ocr: bool = False
    max_retries: NonNegativeInt = Field(default=2, le=5)
    timeout_seconds: float = Field(default=300.0, gt=0.0, le=1800.0)


class FileIngestionResult(BaseModel):
    """Output model for file ingestion workflow."""

    status: FileIngestionStatus
    job_id: str
    data_source: DataSource = Field(
        ...,
        description="DataSource object (status='active' after successful ingestion, 'error' on failure)",
    )
    root_node: dict[str, Any] | None = Field(
        None,
        description="CORPUS node (root) của knowledge tree (serialized as dict). None nếu status=FAILED",
    )
    statistics: dict[str, Any] = Field(
        ...,
        description="Thống kê ingestion (nodes_by_level, total_nodes, processing_duration, etc.). Empty dict nếu status=FAILED",
    )
    error: str | None = Field(
        None,
        description="Thông báo lỗi nếu status=FAILED",
    )

