"""Input and output models for embed_nodes workflow."""

from __future__ import annotations

from typing import Any, Dict, List

from pydantic import BaseModel, Field

from postgres_client.models.embeddings import EmbeddingStructure, EmbeddingContent
from postgres_client.models.knowledge import KnowledgeNode


class EmbedNodesInput(BaseModel):
    """Input model for embed_nodes workflow."""

    nodes: List[KnowledgeNode] = Field(
        ...,
        description="List of nodes from save_knowledge_to_postgres workflow output.",
    )
    batch_size: int = Field(
        default=10,
        ge=1,
        le=100,
        description="Number of texts to embed in a single batch.",
    )
    model: str | None = Field(
        default=None,
        description="Optional: override embedding model name.",
    )


class EmbedNodesOutput(BaseModel):
    """Output model for embed_nodes workflow."""

    structure_embeddings: List[EmbeddingStructure] = Field(
        default_factory=list,
        description="Embeddings for structure nodes (MUC) - from postgres_client.models.embeddings",
    )
    content_embeddings: List[EmbeddingContent] = Field(
        default_factory=list,
        description="Embeddings for detail nodes (KHOI) - from postgres_client.models.embeddings",
    )
    total_embeddings_created: int = Field(
        ...,
        description="Total number of embeddings successfully created.",
    )
    structure_embeddings_count: int = Field(
        ...,
        description="Number of structure embeddings created.",
    )
    content_embeddings_count: int = Field(
        ...,
        description="Number of content embeddings created.",
    )
    failed_nodes: List[Dict[str, Any]] = Field(
        default_factory=list,
        description="Nodes that failed to embed after retry attempts. Format: [{'node_id': UUID, 'node_type': NodeType, 'error': str, 'retry_count': int}]",
    )

