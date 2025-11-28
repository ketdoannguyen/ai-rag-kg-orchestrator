from __future__ import annotations

from enum import Enum
from typing import Any

from pydantic import AnyHttpUrl, BaseModel, Field, NonNegativeInt, validator


class WebIngestionStatus(str, Enum):
    COMPLETE = "complete"
    FAILED = "failed"
    PARTIAL = "partial"


class WebIngestionUrlInput(BaseModel):
    url: AnyHttpUrl
    metadata: dict[str, Any] | None = None


class WebIngestionInput(BaseModel):
    urls: list[WebIngestionUrlInput]
    enable_ocr: bool = False
    max_url_retries: NonNegativeInt = Field(default=2, le=5)
    per_url_timeout_seconds: float = Field(default=90.0, gt=0.0, le=600.0)
    max_concurrent_urls: int = Field(default=4, ge=1, le=10)
    max_concurrent_images: int = Field(default=5, ge=1, le=10)

    @validator("urls")
    def _ensure_urls_present(cls, value: list[WebIngestionUrlInput]) -> list[WebIngestionUrlInput]:
        if not value:
            raise ValueError("At least one URL is required")
        return value


class WebIngestionSingleResult(BaseModel):
    url: AnyHttpUrl
    status: WebIngestionStatus
    job_id: str
    minio_objects: dict[str, str] = Field(default_factory=dict)
    local_paths: dict[str, str] = Field(default_factory=dict)
    images_processed: int = 0
    error: str | None = None


class WebIngestionBatchResult(BaseModel):
    status: WebIngestionStatus
    job_id: str
    total_urls: int
    succeeded: int
    failed: int
    results: list[WebIngestionSingleResult]
