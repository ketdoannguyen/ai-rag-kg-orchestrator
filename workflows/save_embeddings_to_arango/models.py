"""Input/Output models for save_embeddings_to_arango workflow."""

from __future__ import annotations

import uuid

from pydantic import BaseModel, Field

from workflows.embed_nodes.models import EmbeddingData


class SaveEmbeddingsInput(BaseModel):
    """Input for save_embeddings_to_arango workflow."""

    structure_embeddings: list[EmbeddingData] = Field(
        default_factory=list,
        description="Embeddings for structure nodes",
    )
    content_embeddings: list[EmbeddingData] = Field(
        default_factory=list,
        description="Embeddings for content/detail nodes",
    )


class SaveEmbeddingsOutput(BaseModel):
    """Output from save_embeddings_to_arango workflow."""

    structures_saved: int = Field(
        default=0,
        description="Number of structure embeddings saved",
    )
    contents_saved: int = Field(
        default=0,
        description="Number of content embeddings saved",
    )
    total_saved: int = Field(
        default=0,
        description="Total embeddings saved",
    )
    failed_nodes: list[uuid.UUID] = Field(
        default_factory=list,
        description="List of node IDs that failed to save",
    )
