"""Global application settings."""

from __future__ import annotations

from functools import lru_cache
from typing import Any, Literal

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

    # =========================================================================
    # Vision OCR Configuration (ingest_file)
    # =========================================================================
    # OpenAI-compatible chat-completions endpoint used to transcribe PDF pages.
    # Changing VISION_API_BASE also requires updating VISION_ALLOWED_HOSTS: the
    # client refuses to send VISION_API_KEY to any other host.
    VISION_API_BASE: str = "https://opencode.ai/zen/go/v1/chat/completions"
    VISION_API_KEY: str | None = None
    VISION_MODEL: str = "deepseek-v4-flash-vision-exp"
    VISION_ALLOWED_HOSTS: str = "opencode.ai"
    """Comma-separated hosts allowed to receive the API key. Empty = no restriction."""
    VISION_HEADERS: dict[str, str] = Field(default_factory=dict)
    """Extra request headers. An Authorization entry here is ignored on purpose."""
    VISION_IMAGE_DETAIL: Literal["low", "high", "auto"] = "high"
    VISION_DPI: int = 200
    VISION_TEMPERATURE: float = 0.0
    VISION_MAX_TOKENS: int = 8000
    VISION_REASONING_EFFORT: str | None = "none"
    """Reasoning budget hint sent to the vision model.

    A reasoning model spends max_tokens on hidden reasoning before it emits any
    content, so a low ceiling returns an empty answer rather than a short one.
    "none" disables reasoning outright (measured ~8x fewer output tokens with no
    loss of transcription fidelity); "low"/"medium"/"high" trade cost for it. Set
    to an empty value to omit the parameter entirely.
    """
    VISION_MAX_BLOCKS_PER_PAGE: int = 400
    VISION_PAGE_OFFSET: int = 0
    """Printed page number offset: printed = physical PDF page + VISION_PAGE_OFFSET."""

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

    # =========================================================================
    # Benchmark LLM Configuration (dedicated endpoint for query generation)
    # =========================================================================
    BENCHMARK_LLM_API_KEY: str | None = None
    BENCHMARK_LLM_API_BASE: str | None = None
    BENCHMARK_LLM_MODEL: str = "claude-sonnet-4.6"

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
