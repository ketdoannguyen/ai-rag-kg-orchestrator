"""Input and output models for save_knowledge_to_neo4j workflow."""

from __future__ import annotations

import uuid
from typing import Any, Dict, List

from pydantic import BaseModel, Field

from postgres_client.models.knowledge import KnowledgeNode


class SaveKnowledgeToNeo4jInput(BaseModel):
    """Input model for save_knowledge_to_neo4j workflow."""

    enriched_tree: KnowledgeNode = Field(
        ...,
        description="Root KnowledgeNode (STRUCTURE) from enrich_knowledge_tree workflow output.",
    )


class SaveKnowledgeToNeo4jOutput(BaseModel):
    """Output model for save_knowledge_to_neo4j workflow."""

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
    relationships_created: Dict[str, int] = Field(
        ...,
        description="Statistics of relationships created: HAS_SECTION, HAS_CONTENT, NEXT_SIBLING",
    )
    saved_nodes: List[KnowledgeNode] = Field(
        default_factory=list,
        description="List of all nodes that were successfully saved.",
    )
    failed_nodes: List[Dict[str, Any]] = Field(
        default_factory=list,
        description="Nodes that failed to save after retry attempts. Format: [{'node_id': UUID, 'error': str}]",
    )
    partial_success: bool = Field(
        default=False,
        description="True if some nodes failed but workflow continued (partial success).",
    )

