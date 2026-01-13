"""Input/Output models for save_knowledge_to_arango workflow."""

from __future__ import annotations

import uuid
from typing import Any

from arango_client import KnowledgeNode, StructureNode
from pydantic import BaseModel, Field


class SaveKnowledgeInput(BaseModel):
    """Input for save_knowledge_to_arango workflow."""

    enriched_tree: StructureNode = Field(
        ...,
        description="Root node of the enriched knowledge tree to save",
    )
    data_source_id: uuid.UUID = Field(
        ...,
        description="ID of the data source this tree belongs to",
    )


class SaveKnowledgeOutput(BaseModel):
    """Output from save_knowledge_to_arango workflow."""

    root_node_id: uuid.UUID = Field(
        ...,
        description="ID of the root node that was saved",
    )
    total_nodes_saved: int = Field(
        default=0,
        description="Total number of nodes saved successfully",
    )
    structure_nodes_count: int = Field(
        default=0,
        description="Number of structure nodes saved",
    )
    detail_nodes_count: int = Field(
        default=0,
        description="Number of detail nodes (paragraph/image/table) saved",
    )
    relationships_created: dict[str, int] = Field(
        default_factory=dict,
        description="Count of each relationship type created",
    )
    saved_nodes: list[KnowledgeNode] = Field(
        default_factory=list,
        description="List of all saved nodes",
    )
    failed_nodes: list[dict[str, Any]] = Field(
        default_factory=list,
        description="List of nodes that failed to save",
    )
    partial_success: bool = Field(
        default=False,
        description="True if some nodes failed to save",
    )
