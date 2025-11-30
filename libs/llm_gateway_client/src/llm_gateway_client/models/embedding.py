from typing import Any

from pydantic import BaseModel


class EmbeddingRequest(BaseModel):
    """Request model for creating embeddings."""

    model: str | None = None
    input: list[str]
    provider_options: dict[str, Any] | None = None


class EmbeddingResponse(BaseModel):
    """Response model for created embeddings."""

    embeddings: list[list[float]]
    usage: dict[str, int] | None = None
