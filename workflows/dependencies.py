"""Shared dependency helpers for workflow packages."""

from __future__ import annotations

from functools import lru_cache
from typing import Optional

from config.llm_gateway import build_llm_gateway_client
from config.settings import get_settings, settings
from llm_gateway_client import AIGatewayClient
from minio_client import StorageClient
from postgres_client import PostgresClient


@lru_cache(maxsize=1)
def get_minio_client() -> StorageClient:
    """Return a cached MinIO client configured from global settings."""
    missing = [
        name
        for name, value in [
            ("MINIO_ENDPOINT", settings.MINIO_ENDPOINT),
            ("MINIO_ACCESS_KEY", settings.MINIO_ACCESS_KEY),
            ("MINIO_SECRET_KEY", settings.MINIO_SECRET_KEY),
        ]
        if not value
    ]
    if missing:
        required = ", ".join(missing)
        raise ValueError(f"{required} must be set in the environment.")

    if not settings.MINIO_BUCKET:
        raise ValueError("MINIO_BUCKET must be set in the environment.")

    return StorageClient(
        endpoint=settings.MINIO_ENDPOINT,
        access_key=settings.MINIO_ACCESS_KEY,
        secret_key=settings.MINIO_SECRET_KEY,
    )


def init_llm_client() -> AIGatewayClient:
    """Build and return a new LLM gateway client instance."""
    cfg = get_settings()
    return build_llm_gateway_client(cfg)


def init_postgres_client(database_url: Optional[str] = None) -> PostgresClient:
    """Create a Postgres client, defaulting to the configured database URL."""
    url = database_url or get_settings().DATABASE_URL
    if not url:
        raise ValueError("DATABASE_URL must be provided via argument or settings.")
    return PostgresClient(url=url)
