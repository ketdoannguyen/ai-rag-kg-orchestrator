"""Shared dependency helpers for workflow packages."""

from __future__ import annotations

from functools import lru_cache
from typing import Optional

from config.llm_gateway import build_llm_gateway_client
from config.settings import get_settings, settings
from llm_gateway_client import AIGatewayClient
from minio_client import StorageClient
from neo4j_client import Neo4jClient
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


def init_neo4j_client(
    uri: Optional[str] = None,
    user: Optional[str] = None,
    password: Optional[str] = None,
) -> Neo4jClient:
    """
    Create a Neo4j client, defaulting to configured settings.
    
    Args:
        uri: Neo4j URI (defaults to settings.NEO4J_URI)
        user: Neo4j username (defaults to settings.NEO4J_USER)
        password: Neo4j password (defaults to settings.NEO4J_PASSWORD)
    
    Returns:
        Neo4jClient instance with async driver
    """
    cfg = get_settings()
    neo4j_uri = uri or cfg.NEO4J_URI
    neo4j_user = user or cfg.NEO4J_USER
    neo4j_password = password or cfg.NEO4J_PASSWORD
    
    if not neo4j_uri:
        raise ValueError("NEO4J_URI must be provided via argument or settings.")
    if not neo4j_user:
        raise ValueError("NEO4J_USER must be provided via argument or settings.")
    if not neo4j_password:
        raise ValueError("NEO4J_PASSWORD must be provided via argument or settings.")
    
    return Neo4jClient(uri=neo4j_uri, user=neo4j_user, password=neo4j_password, use_async=True)
