"""Input and output models for save_knowledge_to_postgres workflow."""

from __future__ import annotations

import uuid
from typing import Any, Dict, List

from pydantic import BaseModel, Field

from postgres_client.models.knowledge import KnowledgeNode


class SaveKnowledgeInput(BaseModel):
    """Input model for save_knowledge_to_postgres workflow."""

    enriched_tree: KnowledgeNode = Field(
        ...,
        description="Root KnowledgeNode (STRUCTURE) from enrich_knowledge_tree workflow output.",
    )
    data_source_id: uuid.UUID = Field(
        ...,
        description="ID of DataSource created from upload_file or ingest_web workflow.",
    )


class SaveKnowledgeOutput(BaseModel):
    """Output model for save_knowledge_to_postgres workflow."""

    root_node_id: uuid.UUID = Field(
        ...,
        description="ID of the root node that was saved.",
    )
    total_nodes_saved: int = Field(
        ...,
        description="Total number of nodes successfully saved.",
    )
    structure_nodes_count: int = Field(
        ...,
        description="Number of structure nodes saved.",
    )
    detail_nodes_count: int = Field(
        ...,
        description="Number of detail nodes (PARAGRAPH, IMAGE, TABLE) saved.",
    )
    saved_nodes: List[KnowledgeNode] = Field(
        ...,
        description="List of all nodes that were successfully saved.",
    )
    failed_nodes: List[Dict[str, Any]] = Field(
        default_factory=list,
        description="Nodes that failed to save after retry attempts. Format: [{'node': KnowledgeNode, 'error': str, 'retry_count': int}]",
    )
    partial_success: bool = Field(
        default=False,
        description="True if some nodes failed but workflow continued (partial success).",
    )

