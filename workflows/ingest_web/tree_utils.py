"""Tree utilities for ingest_web workflow."""

from __future__ import annotations

import uuid
from datetime import datetime
from typing import Any

from arango_client import KnowledgeNode, StructureNode, NodeType


def compute_tree_statistics(
    root_node: KnowledgeNode,
    start_time: datetime | None = None,
    duration_key: str = "duration_seconds",
    extra: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """
    Compute statistics for a knowledge tree.
    
    Args:
        root_node: Root node of the knowledge tree
        start_time: Optional start time for calculating duration
        duration_key: Key name for duration in output dict
        extra: Extra fields to include in statistics
        
    Returns:
        Dictionary with tree statistics
    """
    statistics: dict[str, Any] = extra or {}
    
    # Count nodes by type
    total_nodes = 0
    structure_nodes = 0
    paragraph_nodes = 0
    image_nodes = 0
    table_nodes = 0
    
    def count_nodes(node: KnowledgeNode) -> None:
        nonlocal total_nodes, structure_nodes, paragraph_nodes, image_nodes, table_nodes
        
        total_nodes += 1
        
        node_type = node.node_type
        if hasattr(node_type, 'value'):
            node_type = node_type.value
        node_type = str(node_type).lower()
        
        if node_type == "structure":
            structure_nodes += 1
        elif node_type == "paragraph":
            paragraph_nodes += 1
        elif node_type == "image":
            image_nodes += 1
        elif node_type == "table":
            table_nodes += 1
        
        # Recursively count children
        if hasattr(node, 'children') and node.children:
            for child in node.children:
                count_nodes(child)
    
    count_nodes(root_node)
    
    statistics["total_nodes"] = total_nodes
    statistics["structure_nodes"] = structure_nodes
    statistics["paragraph_nodes"] = paragraph_nodes
    statistics["image_nodes"] = image_nodes
    statistics["table_nodes"] = table_nodes
    
    # Calculate duration if start_time provided
    if start_time:
        duration = (datetime.utcnow() - start_time).total_seconds()
        statistics[duration_key] = round(duration, 3)
    
    return statistics


def validate_tree_and_attach_source(
    root_node: KnowledgeNode,
    source_id: uuid.UUID,
) -> None:
    """
    Validate knowledge tree structure and attach source_id to all nodes.
    
    Args:
        root_node: Root node of the knowledge tree
        source_id: Source ID to attach to all nodes
        
    Raises:
        ValueError: If tree structure is invalid
    """
    # Validate root node is a structure node
    node_type = root_node.node_type
    if hasattr(node_type, 'value'):
        node_type = node_type.value
    
    if str(node_type).lower() != "structure":
        raise ValueError(f"Root node must be a structure node, got {node_type}")
    
    # Attach source_id to all nodes recursively
    def attach_source(node: KnowledgeNode) -> None:
        node.source_id = source_id
        if hasattr(node, 'children') and node.children:
            for child in node.children:
                attach_source(child)
    
    attach_source(root_node)


def serialize_tree_to_dict(node: KnowledgeNode) -> dict[str, Any]:
    """
    Serialize a knowledge tree to dictionary with proper polymorphic handling.
    
    Pydantic's model_dump() doesn't properly handle polymorphic children because
    children are typed as list[KnowledgeNode] (base class), which doesn't include
    the 'children' field from StructureNode subclass.
    
    This function recursively serializes the entire tree, preserving all fields
    from each node's actual type.
    
    Args:
        node: Root node of the knowledge tree
        
    Returns:
        Dictionary representation of the tree with all children serialized
    """
    # Get all fields from the actual node type (not just base class)
    result = node.model_dump(mode="json")
    
    # Recursively serialize children if present
    if hasattr(node, 'children') and node.children:
        result['children'] = [
            serialize_tree_to_dict(child) for child in node.children
        ]
    
    return result


__all__ = [
    "compute_tree_statistics",
    "validate_tree_and_attach_source",
    "serialize_tree_to_dict",
]

