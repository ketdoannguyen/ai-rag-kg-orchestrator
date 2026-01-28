"""Global application settings."""

from __future__ import annotations

from functools import lru_cache
from typing import Any

from pydantic import Field
from pydantic_settings import BaseSettings


class Settings(BaseSettings):
    """Global application settings loaded from environment or .env file."""

    # ArangoDB Configuration
    ARANGO_URL: str = "http://localhost:8529"
    ARANGO_DATABASE: str = ""
    ARANGO_USER: str = "root"
    ARANGO_PASSWORD: str = ""

    REDIS_URL: str = "redis://localhost:6379/0"
    MINIO_ENDPOINT: str = "http://localhost:9000"
    MINIO_ACCESS_KEY: str = "minio"
    MINIO_SECRET_KEY: str = "minio123"
    MINIO_BUCKET: str = "ingest"
    MINIO_KNOWLEDGE_BUCKET: str = "knowledge-files"
    INGEST_FILE_WORKDIR: str = "data/ingest_file"
    # Marker (local SDK) configuration
    MARKER_USE_LLM: bool = True
    MARKER_FORCE_OCR: bool = False
    MARKER_TORCH_DEVICE: str | None = None  # e.g. "cuda", "cuda:0", "cpu"
    GOOGLE_API_KEY: str | None = None  # used when MARKER_USE_LLM is True

    # =========================================================================
    # LLM Chat Configuration
    # =========================================================================
    LLM_API_KEY: str | None = None
    LLM_API_BASE: str = "https://openrouter.ai/api/v1"
    LLM_CHAT_MODEL: str = "gpt-4o-mini"
    LLM_CHAT_API_KEY: str | None = None
    LLM_CHAT_API_BASE: str | None = None
    LLM_CHAT_OPTIONS: dict[str, Any] = Field(default_factory=dict)

    # Provider Routing (OpenRouter)
    LLM_CHAT_PROVIDERS: str | None = None
    """Comma-separated list of preferred providers (e.g., "together,fireworks")."""
    LLM_CHAT_ALLOW_FALLBACKS: bool = False
    """Whether to allow fallback to other providers if preferred ones fail."""

    # =========================================================================
    # Embedding Configuration (Separate from Chat)
    # =========================================================================
    # These take precedence over LLM_* equivalents for embeddings
    EMBEDDING_MODEL: str | None = None  # e.g., "text-embedding-3-small"
    EMBEDDING_API_KEY: str | None = None
    EMBEDDING_API_BASE: str | None = None
    EMBEDDING_OPTIONS: dict[str, Any] = Field(default_factory=dict)

    # Provider Routing for Embeddings (OpenRouter)
    LLM_EMBEDDING_PROVIDERS: str | None = None
    """Comma-separated list of preferred providers for embeddings."""
    LLM_EMBEDDING_ALLOW_FALLBACKS: bool = False
    """Whether to allow fallback to other providers for embedding requests."""

    # Legacy aliases (for backwards compatibility)
    LLM_EMBEDDING_MODEL: str = "text-embedding-3-small"
    LLM_EMBEDDING_API_KEY: str | None = None
    LLM_EMBEDDING_API_BASE: str | None = None
    LLM_EMBEDDING_OPTIONS: dict[str, Any] = Field(default_factory=dict)

    # Langfuse Cloud Observability
    LANGFUSE_PUBLIC_KEY: str | None = None
    LANGFUSE_SECRET_KEY: str | None = None
    LANGFUSE_OTEL_HOST: str = "https://us.cloud.langfuse.com"  # US region default
    LANGFUSE_ENABLED: bool = True  # Toggle tracing on/off

    class Config:
        env_file = ".env"
        env_file_encoding = "utf-8"
        extra = "ignore"  # Ignore extra fields from .env


@lru_cache
def get_settings() -> Settings:
    """Return a cached Settings instance (used by services layer)."""

    return Settings()


settings: Settings = get_settings()
