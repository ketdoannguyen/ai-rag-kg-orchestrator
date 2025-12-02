from __future__ import annotations

from enum import Enum
from typing import Any
from uuid import UUID

from pydantic import BaseModel, Field, NonNegativeInt, validator

from postgres_client.models.system_admin import DataSource


class WebIngestionStatus(str, Enum):
    """Status của việc crawl một URL."""

    SUCCESS = "success"
    ERROR = "error"
    SKIPPED = "skipped"  # Invalid URL hoặc disallowed by robots.txt


class WebIngestionInput(BaseModel):
    """Input payload for the ingest_web workflow."""

    urls: list[str] = Field(
        ...,
        description="Danh sách URLs cần crawl",
        min_items=1,
    )
    created_by: UUID = Field(
        ...,
        description="User ID",
    )
    crawl_config: dict[str, Any] = Field(
        default_factory=dict,
        description="Cấu hình crawling (respect_robots_txt, max_depth, etc.)",
    )
    max_url_retries: NonNegativeInt = Field(
        default=3,
        le=5,
        description="Số lần retry tối đa cho mỗi URL",
    )
    per_url_timeout_seconds: float = Field(
        default=90.0,
        gt=0.0,
        le=600.0,
        description="Timeout cho mỗi URL (giây)",
    )
    max_concurrent_urls: int = Field(
        default=4,
        ge=1,
        le=10,
        description="Số URL crawl đồng thời tối đa",
    )

    @validator("urls")
    def _ensure_urls_present(cls, value: list[str]) -> list[str]:
        if not value:
            raise ValueError("At least one URL is required")
        return value


class WebIngestionSingleResult(BaseModel):
    """Kết quả xử lý cho một URL."""

    status: WebIngestionStatus = Field(
        ...,
        description="Status của việc crawl URL này",
    )
    data_source: DataSource | None = Field(
        None,
        description="DataSource cho web (status='processing' hoặc 'error'). None nếu status=SKIPPED",
    )
    root_node: dict[str, Any] | None = Field(
        None,
        description="CORPUS node (root) của knowledge tree (serialized as dict). None nếu status=ERROR hoặc SKIPPED",
    )
    statistics: dict[str, Any] | None = Field(
        None,
        description="Thống kê crawling (nodes_by_level, total_nodes, etc.). None nếu status=ERROR hoặc SKIPPED",
    )
    error_message: str | None = Field(
        None,
        description="Thông báo lỗi nếu status=ERROR",
    )


class WebIngestionResult(BaseModel):
    """Kết quả của workflow - list các kết quả cho từng URL."""

    results: list[WebIngestionSingleResult] = Field(
        ...,
        description="Danh sách kết quả cho từng URL",
    )
