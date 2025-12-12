from functools import lru_cache
from typing import Any

from pydantic import Field
from pydantic_settings import BaseSettings


class Settings(BaseSettings):
    """Global application settings loaded from environment or .env file."""

    DATABASE_URL: str = "postgresql+psycopg://user:password@localhost:5432/db"
    NEO4J_URI: str = "bolt://localhost:7687"
    NEO4J_USER: str = "neo4j"
    NEO4J_PASSWORD: str = "password"
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
    # LLM Gateway Configuration (Unified)
    # =========================================================================
    # Single API key and base URL for all LLM/embedding calls
    # Supports OpenRouter, OpenAI, or any OpenAI-compatible API
    LLM_API_KEY: str | None = None
    LLM_API_BASE: str = "https://openrouter.ai/api/v1"  # Default to OpenRouter

    # Model names - when using api_base, use model names WITHOUT provider prefix
    # Examples: gpt-4o-mini, text-embedding-3-small
    LLM_CHAT_MODEL: str = "gpt-4o-mini"
    LLM_EMBEDDING_MODEL: str = "text-embedding-3-small"

    # Optional: separate config for chat vs embedding (if using different providers)
    LLM_CHAT_API_KEY: str | None = None
    LLM_CHAT_API_BASE: str | None = None
    LLM_EMBEDDING_API_KEY: str | None = None
    LLM_EMBEDDING_API_BASE: str | None = None

    # Extra options (temperature, max_tokens, etc.)
    LLM_CHAT_OPTIONS: dict[str, Any] = Field(default_factory=dict)
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


@lru_cache()
def get_settings() -> "Settings":
    """Return a cached Settings instance (used by services layer)."""

    return Settings()


settings: Settings = get_settings()
