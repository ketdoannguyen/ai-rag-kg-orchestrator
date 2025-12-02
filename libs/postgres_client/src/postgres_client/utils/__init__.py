"""Utility functions for postgres_client."""

from __future__ import annotations

import uuid

from ..models import (
    ImageNode,
    KnowledgeNode,
    ParagraphNode,
    StructureNode,
    TableNode,
)
from ..models.enums import NodeType
from ..models.attributes_metadata import (
    ParagraphMetadata,
    ImageMetadata,
    TableMetadata,
    StructureMetadata,
)

# Import from code_extraction module
from .code_extraction import (
    clean_title_from_code,
    extract_code_from_context,
    extract_code_from_title,
)

# Import KnowledgeNode converter utilities
from .knowledge_node_converter import (
    orm_to_pydantic,
    pydantic_to_orm,
)


def _get_metadata_class(node: KnowledgeNode):
    """Get the appropriate metadata class for a node type."""
    if isinstance(node, StructureNode):
        return StructureMetadata
    elif isinstance(node, ParagraphNode):
        return ParagraphMetadata
    elif isinstance(node, ImageNode):
        return ImageMetadata
    elif isinstance(node, TableNode):
        return TableMetadata
    return None


def parse_node_from_dict(data: dict | KnowledgeNode) -> KnowledgeNode:
    """
    Parse a dictionary or KnowledgeNode into the correct KnowledgeNode subtype.
    
    This function recursively parses a nested dictionary structure (e.g., from JSON) into the appropriate
    KnowledgeNode types (ParagraphNode, ImageNode, TableNode, StructureNode, etc.) based on node_type
    and attributes present in the data.
    
    Args:
        data: Dictionary containing node data or already parsed KnowledgeNode
        
    Returns:
        Appropriate KnowledgeNode instance (ParagraphNode, ImageNode, TableNode, StructureNode, etc.)
        
    Raises:
        ValueError: If data is not a dict or KnowledgeNode, or if node_type is missing
    """
    # If already a KnowledgeNode instance, return as-is (but ensure children are parsed)
    if isinstance(data, KnowledgeNode):
        # Recursively parse children if they're still dicts
        if data.children:
            parsed_children = []
            for child in data.children:
                if isinstance(child, dict):
                    parsed_children.append(parse_node_from_dict(child))
                elif isinstance(child, KnowledgeNode):
                    # Recursively parse children of this child
                    if child.children:
                        child.children = [
                            parse_node_from_dict(c) if isinstance(c, dict) else c
                            for c in child.children
                        ]
                    parsed_children.append(child)
                else:
                    parsed_children.append(child)
            data.children = parsed_children
        return data
    
    if not isinstance(data, dict):
        raise ValueError(f"Expected dict or KnowledgeNode, got {type(data)}")
    
    node_type_str = data.get("node_type")
    if not node_type_str:
        raise ValueError("Missing node_type in node data")
    
    attributes = data.get("attributes", {})
    children_data = data.get("children", [])
    
    # Parse children recursively first
    parsed_children = []
    for child in children_data:
        if isinstance(child, dict):
            parsed_children.append(parse_node_from_dict(child))
        elif isinstance(child, KnowledgeNode):
            # Already parsed, but ensure its children are parsed too
            parsed_children.append(parse_node_from_dict(child))
        else:
            parsed_children.append(child)
    
    # Create a copy of data with parsed children to avoid modifying original
    node_data = {**data, "children": parsed_children}
    
    # Determine node type based on node_type (if present) or attributes fallback
    try:
        node_type = NodeType(node_type_str)
    except ValueError:
        node_type = None

    if node_type == NodeType.PARAGRAPH or ("text" in attributes and "headers" not in attributes):
            return ParagraphNode.model_validate(node_data)
    if node_type == NodeType.IMAGE or "url" in attributes or "file_path" in attributes or "alt_text" in attributes:
            return ImageNode.model_validate(node_data)
    if node_type == NodeType.TABLE or "headers" in attributes or "rows" in attributes:
            return TableNode.model_validate(node_data)

    # Default to structure node family
    # We don't distinguish Corpus/Topic/Subtopic here; callers can cast if needed.
            return StructureNode.model_validate(node_data)


def set_parent_ids(node: KnowledgeNode, parent_id: uuid.UUID | None = None) -> None:
    """
    Recursively set parent_id for all nodes in the tree.
    
    This function traverses the knowledge node tree and sets the parent_id of each child
    node to the id of its parent node. The root node's parent_id remains None.
    
    Args:
        node: Root KnowledgeNode to process
        parent_id: ID of the parent node (None for root node)
        
    Example:
        >>> root = StructureNode(...)
        >>> set_parent_ids(root)
        >>> # Now all children have parent_id = root.id
    """
    # Set parent_id for current node
    node.parent_id = parent_id
    
    # Recursively set parent_id for all children
    if node.children:
        for child in node.children:
            set_parent_ids(child, parent_id=node.id)


def set_positions(node: KnowledgeNode, breadcrumb: list[str] | None = None) -> None:
    """
    Recursively set position (breadcrumb path) for all nodes in the tree.
    
    This function traverses the knowledge node tree and sets the position metadata
    of each node to a breadcrumb path from root to that node. Only structure nodes
    (node_type=STRUCTURE) are included in the breadcrumb path.
    
    Args:
        node: Root KnowledgeNode to process
        breadcrumb: List of titles from root to current node (None for root)
        
    Example:
        >>> root = StructureNode(attributes=StructureAttributes(title="Root"))
        >>> child = StructureNode(attributes=StructureAttributes(title="Child"))
        >>> root.children = [child]
        >>> set_positions(root)
        >>> # Root position = "Root"
        >>> # Child position = "Root > Child"
    """
    if breadcrumb is None:
        breadcrumb = []
    
    # Get title if this is a structure node
    current_title: str | None = None
    if isinstance(node, StructureNode):
        current_title = node.attributes.title
        # Add current title to breadcrumb
        new_breadcrumb = breadcrumb + [current_title]
    else:
        # For detail nodes, use the same breadcrumb (don't add their text to path)
        new_breadcrumb = breadcrumb
    
    # Set position for current node
    position_str = " > ".join(new_breadcrumb) if new_breadcrumb else None
    
    # Set position in metadata
    if hasattr(node, 'metadata') and node.metadata is not None:
        # Convert dict metadata to proper type if needed
        if isinstance(node.metadata, dict):
            metadata_class = _get_metadata_class(node)
            if metadata_class:
                node.metadata = metadata_class(**node.metadata)
        
        # Set position if metadata supports it
        if hasattr(node.metadata, 'position'):
            node.metadata.position = position_str
    
    # Recursively set positions for all children
    if node.children:
        for child in node.children:
            set_positions(child, breadcrumb=new_breadcrumb)


__all__ = [
    "parse_node_from_dict",
    "set_parent_ids",
    "set_positions",
    "extract_code_from_title",
    "extract_code_from_context",
    "clean_title_from_code",
    # ORM converter utilities
    "orm_to_pydantic",
    "pydantic_to_orm",
]


