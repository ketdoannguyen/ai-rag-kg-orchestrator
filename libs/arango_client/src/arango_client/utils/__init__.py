"""
Utility functions for ArangoDB client.

This module provides shared converters and helpers used across handlers.
"""

from __future__ import annotations

import uuid
from datetime import datetime
from typing import TYPE_CHECKING, Any
from uuid import UUID

import numpy as np

if TYPE_CHECKING:
    from ..models.knowledge import KnowledgeNode
    from ..models.ontology import RelationTier


# Embedding constants
MAX_EMBEDDING_DIMENSIONS = 4096
MIN_EMBEDDING_DIMENSIONS = 64


def normalize_tier(tier: RelationTier | str | None) -> str | None:
    """Convert RelationTier enum to string value.

    Args:
        tier: RelationTier enum, string, or None

    Returns:
        String value of tier or None
    """
    if tier is None:
        return None
    return tier.value if hasattr(tier, "value") else tier


def normalize_vector(vector: list[float] | np.ndarray) -> np.ndarray:
    """Normalize a vector to unit length.

    Args:
        vector: Input vector (list or numpy array)

    Returns:
        Normalized numpy array
    """
    arr = np.array(vector) if isinstance(vector, list) else vector
    return arr / (np.linalg.norm(arr) + 1e-8)


def validate_embedding(embedding: list[float] | None) -> None:
    """Validate embedding array dimensions.

    Args:
        embedding: List of floats or None

    Raises:
        ValueError: If dimensions are invalid
    """
    if embedding is None:
        return

    if not isinstance(embedding, list):
        raise ValueError(f"Embedding must be a list, got {type(embedding)}")

    if len(embedding) < MIN_EMBEDDING_DIMENSIONS:
        raise ValueError(
            f"Embedding dimensions too small: {len(embedding)} < {MIN_EMBEDDING_DIMENSIONS}"
        )

    if len(embedding) > MAX_EMBEDDING_DIMENSIONS:
        raise ValueError(
            f"Embedding dimensions exceed limit: {len(embedding)} > {MAX_EMBEDDING_DIMENSIONS}"
        )


# Import models lazily to avoid circular imports
_model_cache = {}


def _get_models():
    """Lazy import models to avoid circular imports."""
    if not _model_cache:
        from ..models.detail_nodes import ImageNode, ParagraphNode, TableNode
        from ..models.knowledge import KnowledgeNode
        from ..models.structure_nodes import RootNode, StructureNode

        _model_cache.update(
            {
                "KnowledgeNode": KnowledgeNode,
                "ParagraphNode": ParagraphNode,
                "ImageNode": ImageNode,
                "TableNode": TableNode,
                "StructureNode": StructureNode,
                "RootNode": RootNode,
            }
        )
    return _model_cache


def doc_to_knowledge_node(
    doc: dict[str, Any],
    node_class: type[KnowledgeNode] | None = None,
) -> KnowledgeNode:
    """
    Convert ArangoDB document to appropriate KnowledgeNode subclass.

    This is a shared utility to avoid code duplication between handlers.

    Args:
        doc: ArangoDB document dictionary
        node_class: Optional specific class to use for conversion.
                   If None, will be inferred from node_type field.

    Returns:
        KnowledgeNode instance of the appropriate type

    Example:
        >>> doc = {"_key": "123", "node_type": "paragraph", ...}
        >>> node = doc_to_knowledge_node(doc)
        >>> isinstance(node, ParagraphNode)
        True
    """
    models = _get_models()

    if node_class is not None:
        return node_class.from_document(doc)

    # Infer class from node_type
    node_type = doc.get("node_type", "")

    if node_type == "structure":
        # Check if it's a root node (level == 1)
        if doc.get("level") == 1:
            return models["RootNode"].from_document(doc)
        return models["StructureNode"].from_document(doc)
    elif node_type == "paragraph":
        return models["ParagraphNode"].from_document(doc)
    elif node_type == "image":
        return models["ImageNode"].from_document(doc)
    elif node_type == "table":
        return models["TableNode"].from_document(doc)
    else:
        return models["KnowledgeNode"].from_document(doc)


def parse_node_from_dict(data: dict | KnowledgeNode) -> KnowledgeNode:
    """
    Parse a dictionary or KnowledgeNode into the correct KnowledgeNode subtype.

    This function recursively parses a nested dictionary structure (e.g., from JSON)
    into the appropriate KnowledgeNode types based on node_type.

    Args:
        data: Dictionary containing node data or already parsed KnowledgeNode

    Returns:
        Appropriate KnowledgeNode instance

    Raises:
        ValueError: If data is not a dict or KnowledgeNode, or if node_type is missing
    """
    models = _get_models()
    KnowledgeNode = models["KnowledgeNode"]

    # If already a KnowledgeNode instance, recursively parse children
    if isinstance(data, KnowledgeNode):
        if hasattr(data, "children") and data.children:
            parsed_children = []
            for child in data.children:
                if isinstance(child, dict):
                    parsed_children.append(parse_node_from_dict(child))
                else:
                    parsed_children.append(parse_node_from_dict(child))
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
        parsed_children.append(parse_node_from_dict(child))

    # Create a copy of data with parsed children
    node_data = {**data, "children": parsed_children}

    # Determine node type and create appropriate instance
    if node_type_str == "paragraph" or (node_type_str is None and "text" in attributes):
        return models["ParagraphNode"].model_validate(node_data)
    elif node_type_str == "image":
        return models["ImageNode"].model_validate(node_data)
    elif node_type_str == "table":
        return models["TableNode"].model_validate(node_data)
    else:
        # Structure node family
        level = data.get("level", 0)
        if level == 1:
            return models["RootNode"].model_validate(node_data)
        return models["StructureNode"].model_validate(node_data)


def set_parent_ids(node: KnowledgeNode, parent_id: uuid.UUID | None = None) -> None:
    """
    Recursively set parent_id for all nodes in the tree.

    Args:
        node: Root KnowledgeNode to process
        parent_id: ID of the parent node (None for root node)
    """
    node.parent_id = parent_id

    if hasattr(node, "children") and node.children:
        for child in node.children:
            set_parent_ids(child, parent_id=node.id)


def set_positions(node: KnowledgeNode, breadcrumb: list[str] | None = None) -> None:
    """
    Recursively set position (breadcrumb path) for all nodes in the tree.

    Args:
        node: Root KnowledgeNode to process
        breadcrumb: List of titles from root to current node (None for root)
    """
    models = _get_models()
    StructureNode = models["StructureNode"]

    if breadcrumb is None:
        breadcrumb = []

    # Get title if this is a structure node
    current_title: str | None = None
    if isinstance(node, StructureNode) or node.node_type == "structure":
        if hasattr(node.attributes, "title"):
            current_title = node.attributes.title
        elif isinstance(node.attributes, dict):
            current_title = node.attributes.get("title")
        new_breadcrumb = breadcrumb + [current_title] if current_title else breadcrumb
    else:
        new_breadcrumb = breadcrumb

    # Set position for current node
    position_str = " > ".join(new_breadcrumb) if new_breadcrumb else None

    # Set position in metadata
    if hasattr(node, "metadata") and node.metadata is not None:
        if hasattr(node.metadata, "position"):
            node.metadata.position = position_str
        elif isinstance(node.metadata, dict):
            node.metadata["position"] = position_str

    # Recursively set positions for all children
    if hasattr(node, "children") and node.children:
        for child in node.children:
            set_positions(child, breadcrumb=new_breadcrumb)


def serialize_tree_to_dict(node: KnowledgeNode) -> dict[str, Any]:
    """
    Recursively serialize a knowledge tree to dictionary.

    This function properly handles nested children which Pydantic's model_dump()
    may not serialize correctly for subclass types.

    Args:
        node: Root KnowledgeNode to serialize

    Returns:
        Dictionary representation of the entire tree
    """
    # Get base model dump
    data = node.model_dump(mode="json")

    # If this node has children, recursively serialize them
    if hasattr(node, "children") and node.children:
        data["children"] = [serialize_tree_to_dict(child) for child in node.children]

    return data


def validate_tree_and_attach_source(
    root: KnowledgeNode,
    data_source_id: UUID,
) -> None:
    """
    Validate a knowledge tree and ensure each node points to the provided data source.

    The rules enforced:
      * Every non-root node must have parent_id matching its parent's id.
      * level_down and level_up must be non-negative.
      * level_down + level_up must remain constant along a path.
      * Detail nodes (paragraph/image/table) must contain the expected payload fields.

    Args:
        root: Root node of the knowledge tree
        data_source_id: UUID of the data source to attach

    Raises:
        ValueError: If validation fails with detailed error messages
    """
    errors: list[str] = []

    def walk(
        node: KnowledgeNode,
        parent: KnowledgeNode | None,
        path: list[UUID] | None = None,
    ) -> None:
        nonlocal errors
        if path is None:
            path = []

        node.source_id = data_source_id

        if parent is None:
            if node.parent_id is not None:
                errors.append("Root node must have parent_id=None")
        elif node.parent_id is None:
            errors.append("Non-root node must have parent_id set")
        elif node.parent_id != parent.id:
            errors.append(
                f"Node {node.id} has parent_id={node.parent_id} but parent is {parent.id}"
            )

        # Validate level (arango_client uses 'level' instead of 'level_down'/'level_up')
        node_level = getattr(node, "level", 0)
        if node_level < 0:
            errors.append(f"Node {node.id} has invalid level={node_level}, must be >= 0")

        current_path = tuple(path + [node.id])

        node_type = node.node_type
        if node_type in ["paragraph", "image", "table"]:
            attrs = getattr(node, "attributes", None)
            if attrs is not None:
                if isinstance(attrs, dict):
                    if node_type == "paragraph":
                        text = attrs.get("text", "")
                        if not text or not str(text).strip():
                            errors.append(f"Paragraph node {node.id} has empty text")
                    elif node_type == "image":
                        url_attr = attrs.get("url") or attrs.get("file_path")
                        if not url_attr:
                            errors.append(f"Image node {node.id} has no url or file_path")
                    elif node_type == "table":
                        headers = attrs.get("headers", [])
                        rows = attrs.get("rows", [])
                        if not headers and not rows:
                            errors.append(f"Table node {node.id} has no headers or rows")
                elif hasattr(attrs, "text"):
                    if node_type == "paragraph" and not attrs.text.strip():
                        errors.append(f"Paragraph node {node.id} has empty text")
                elif hasattr(attrs, "url") or hasattr(attrs, "file_path"):
                    if node_type == "image":
                        url_attr = getattr(attrs, "url", None) or getattr(attrs, "file_path", None)
                        if not url_attr:
                            errors.append(f"Image node {node.id} has no url or file_path")

        # DetailNodes (ParagraphNode, ImageNode, TableNode) don't have children
        if hasattr(node, "children") and node.children:
            for child in node.children:
                walk(child, node, list(current_path))

    walk(root, None)

    if errors:
        raise ValueError("Knowledge tree validation failed: " + "; ".join(sorted(set(errors))))


def compute_tree_statistics(
    root_node: KnowledgeNode,
    *,
    start_time: datetime | None = None,
    end_time: datetime | None = None,
    duration_key: str = "processing_duration_seconds",
    include_success_flag: bool = True,
    extra: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """
    Calculate common statistics for a knowledge tree.

    Args:
        root_node: Root of the tree.
        start_time: Optional start timestamp for duration calculation.
        end_time: Optional end timestamp (defaults to now if start_time provided).
        duration_key: Key name for duration in the returned dict.
        include_success_flag: Whether to set `successful=True` in the result.
        extra: Additional key/value pairs to merge into the result.

    Returns:
        Dictionary containing tree statistics
    """
    nodes_by_level: dict[int, int] = {}
    total_nodes = 0
    total_paragraphs = 0
    total_content_length = 0

    def walk(node: KnowledgeNode) -> None:
        nonlocal total_nodes, total_paragraphs, total_content_length
        total_nodes += 1
        # Use 'level' attribute (arango_client) instead of 'level_down' (postgres_client)
        level = getattr(node, "level", 0)
        nodes_by_level[level] = nodes_by_level.get(level, 0) + 1

        if node.node_type == "paragraph":
            total_paragraphs += 1
            attrs = getattr(node, "attributes", None)
            if attrs:
                if isinstance(attrs, dict):
                    text = attrs.get("text", "")
                elif hasattr(attrs, "text"):
                    text = attrs.text
                else:
                    text = ""
                total_content_length += len(str(text))

        # DetailNodes (ParagraphNode, ImageNode, TableNode) don't have children
        if hasattr(node, "children") and node.children:
            for child in node.children:
                walk(child)

    walk(root_node)

    duration_seconds = 0.0
    if start_time is not None:
        effective_end = end_time
        if effective_end is None:
            if start_time.tzinfo is None:
                effective_end = datetime.utcnow()
            else:
                effective_end = datetime.now(start_time.tzinfo)
        if start_time.tzinfo and effective_end.tzinfo is None:
            effective_end = effective_end.replace(tzinfo=start_time.tzinfo)
        duration_seconds = (effective_end - start_time).total_seconds()

    stats: dict[str, Any] = {
        "nodes_by_level": nodes_by_level,
        "total_nodes": total_nodes,
        "total_paragraphs": total_paragraphs,
        duration_key: duration_seconds,
        "average_content_length": (
            total_content_length / total_paragraphs if total_paragraphs > 0 else 0
        ),
    }

    if include_success_flag:
        stats["successful"] = True

    if extra:
        stats.update(extra)

    return stats


__all__ = [
    "compute_tree_statistics",
    "doc_to_knowledge_node",
    "normalize_tier",
    "normalize_vector",
    "validate_embedding",
    "parse_node_from_dict",
    "serialize_tree_to_dict",
    "set_parent_ids",
    "set_positions",
    "validate_tree_and_attach_source",
]
