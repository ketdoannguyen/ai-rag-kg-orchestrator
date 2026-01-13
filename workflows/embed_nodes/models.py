"""Input and output models for embed_nodes workflow.

Sử dụng arango_client models - embeddings được lưu trực tiếp vào nodes.
"""

from __future__ import annotations

import uuid
from typing import Any, Dict, List

from pydantic import BaseModel, Field

from arango_client import KnowledgeNode


class EmbeddingData(BaseModel):
    """Single embedding data for a node."""

    node_id: uuid.UUID = Field(..., description="ID of the node this embedding belongs to")
    vector: List[float] = Field(..., description="Embedding vector")
    text: str | None = Field(None, description="Text that was embedded")


class EmbedNodesInput(BaseModel):
    """Input model for embed_nodes workflow."""

    nodes: List[KnowledgeNode] = Field(
        ...,
        description="List of nodes to embed.",
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

    structure_embeddings: List[EmbeddingData] = Field(
        default_factory=list,
        description="Embeddings for structure nodes",
    )
    content_embeddings: List[EmbeddingData] = Field(
        default_factory=list,
        description="Embeddings for detail nodes (paragraph, image, table)",
    )
    total_embeddings_created: int = Field(
        default=0,
        description="Total number of embeddings successfully created.",
    )
    structure_embeddings_count: int = Field(
        default=0,
        description="Number of structure embeddings created.",
    )
    content_embeddings_count: int = Field(
        default=0,
        description="Number of content embeddings created.",
    )
    failed_nodes: List[Dict[str, Any]] = Field(
        default_factory=list,
        description="Nodes that failed to embed after retry attempts.",
    )
