"""Shared dependency helpers for workflow packages."""

from __future__ import annotations

from functools import lru_cache

from arango_client import ArangoClient
from llm_gateway_client import AIGatewayClient
from minio_client import StorageClient

from config.llm_gateway import build_llm_gateway_client
from config.settings import get_settings, settings


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


def init_arango_client(
    url: str | None = None,
    database: str | None = None,
    username: str | None = None,
    password: str | None = None,
) -> ArangoClient:
    """
    Create an ArangoDB client, defaulting to configured settings.

    Args:
        url: ArangoDB URL (defaults to settings.ARANGO_URL)
        database: Database name (defaults to settings.ARANGO_DATABASE)
        username: Username (defaults to settings.ARANGO_USER)
        password: Password (defaults to settings.ARANGO_PASSWORD)

    Returns:
        ArangoClient instance with auto-initialized schema
    """
    cfg = get_settings()
    arango_url = url or cfg.ARANGO_URL
    arango_database = database or cfg.ARANGO_DATABASE
    arango_user = username or cfg.ARANGO_USER
    arango_password = password or cfg.ARANGO_PASSWORD

    if not arango_url:
        raise ValueError("ARANGO_URL must be provided via argument or settings.")
    if not arango_database:
        raise ValueError("ARANGO_DATABASE must be provided via argument or settings.")

    return ArangoClient(
        url=arango_url,
        database=arango_database,
        username=arango_user,
        password=arango_password,
        auto_init=True,
    )
