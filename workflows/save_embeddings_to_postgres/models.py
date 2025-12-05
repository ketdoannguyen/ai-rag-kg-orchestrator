"""Input and output models for save_embeddings_to_postgres workflow."""

from __future__ import annotations

import uuid
from typing import List

from pydantic import BaseModel, Field

from postgres_client.models.embeddings import EmbeddingStructure, EmbeddingContent


class SaveEmbeddingsInput(BaseModel):
    """Input model for save_embeddings_to_postgres workflow."""

    structure_embeddings: List[EmbeddingStructure] = Field(
        default_factory=list,
        description="Structure embeddings from embed_nodes workflow output.",
    )
    content_embeddings: List[EmbeddingContent] = Field(
        default_factory=list,
        description="Content embeddings from embed_nodes workflow output.",
    )


class SaveEmbeddingsOutput(BaseModel):
    """Output model for save_embeddings_to_postgres workflow."""

    structures_saved: int = Field(
        ...,
        description="Number of structure embeddings successfully saved.",
    )
    contents_saved: int = Field(
        ...,
        description="Number of content embeddings successfully saved.",
    )
    total_saved: int = Field(
        ...,
        description="Total number of embeddings successfully saved.",
    )
    failed_nodes: List[uuid.UUID] = Field(
        default_factory=list,
        description="Node IDs that failed to save after retry attempts (for maintenance workflow to handle later).",
    )

