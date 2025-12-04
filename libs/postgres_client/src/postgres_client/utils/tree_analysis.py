"""Utilities for validating and summarizing knowledge trees."""

from __future__ import annotations

from datetime import datetime
from typing import Any
from uuid import UUID

from ..models import KnowledgeNode
from ..models.enums import NodeType


def validate_tree_and_attach_source(
    root: KnowledgeNode,
    data_source_id: UUID,
) -> None:
    """
    Validate a knowledge tree and ensure each node points to the provided data source.

    The rules enforced mirror the legacy workflow implementations:
      * Every non-root node must have parent_id matching its parent's id.
      * level_down and level_up must be non-negative.
      * level_down + level_up must remain constant along a path.
      * Detail nodes (paragraph/image/table) must contain the expected payload fields.
    """
    errors: list[str] = []
    path_constants: dict[tuple, int] = {}

    def walk(node: KnowledgeNode, parent: KnowledgeNode | None, path: list[UUID] | None = None) -> None:
        nonlocal errors
        if path is None:
            path = []

        node.source_id = data_source_id

        if parent is None:
            if node.parent_id is not None:
                errors.append("Root node must have parent_id=None")
        else:
            if node.parent_id is None:
                errors.append("Non-root node must have parent_id set")
            elif node.parent_id != parent.id:
                errors.append(f"Node {node.id} has parent_id={node.parent_id} but parent is {parent.id}")

        if node.level_down < 0:
            errors.append(f"Node {node.id} has invalid level_down={node.level_down}, must be >= 0")
        if node.level_up < 0:
            errors.append(f"Node {node.id} has invalid level_up={node.level_up}, must be >= 0")

        current_path = tuple(path + [node.id])
        constant = node.level_down + node.level_up
        if current_path in path_constants:
            expected_constant = path_constants[current_path]
            if constant != expected_constant:
                errors.append(
                    f"Node {node.id} on path has level_down+level_up={constant}, "
                    f"but expected {expected_constant} (path constant violation)"
                )
        else:
            path_constants[current_path] = constant

        if node.node_type in [NodeType.PARAGRAPH, NodeType.IMAGE, NodeType.TABLE]:
            attrs = getattr(node, "attributes", None)
            if attrs is not None:
                if isinstance(attrs, dict):
                    if node.node_type == NodeType.PARAGRAPH:
                        text = attrs.get("text", "")
                        if not text or not str(text).strip():
                            errors.append(f"Paragraph node {node.id} has empty text")
                    elif node.node_type == NodeType.IMAGE:
                        url_attr = attrs.get("url") or attrs.get("file_path")
                        if not url_attr:
                            errors.append(f"Image node {node.id} has no url or file_path")
                    elif node.node_type == NodeType.TABLE:
                        headers = attrs.get("headers", [])
                        rows = attrs.get("rows", [])
                        if not headers and not rows:
                            errors.append(f"Table node {node.id} has no headers or rows")
                elif hasattr(attrs, "text"):
                    if node.node_type == NodeType.PARAGRAPH and not attrs.text.strip():
                        errors.append(f"Paragraph node {node.id} has empty text")
                elif hasattr(attrs, "url") or hasattr(attrs, "file_path"):
                    if node.node_type == NodeType.IMAGE:
                        url_attr = getattr(attrs, "url", None) or getattr(attrs, "file_path", None)
                        if not url_attr:
                            errors.append(f"Image node {node.id} has no url or file_path")

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
    """
    nodes_by_level: dict[int, int] = {}
    total_nodes = 0
    total_paragraphs = 0
    total_content_length = 0

    def walk(node: KnowledgeNode) -> None:
        nonlocal total_nodes, total_paragraphs, total_content_length
        total_nodes += 1
        level = node.level_down
        nodes_by_level[level] = nodes_by_level.get(level, 0) + 1

        if node.node_type == NodeType.PARAGRAPH:
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
        "average_content_length": total_content_length / total_paragraphs if total_paragraphs > 0 else 0,
    }

    if include_success_flag:
        stats["successful"] = True

    if extra:
        stats.update(extra)

    return stats


__all__ = ["validate_tree_and_attach_source", "compute_tree_statistics"]

