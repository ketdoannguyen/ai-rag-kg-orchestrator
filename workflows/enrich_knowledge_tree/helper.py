from __future__ import annotations

from typing import TYPE_CHECKING, Any

from arango_client import (
    ImageNode,
    KnowledgeNode,
    NodeType,
    ParagraphNode,
    StructureNode,
    TableNode,
)
from loguru import logger

if TYPE_CHECKING:
    pass


def count_paragraphs(node: KnowledgeNode | None) -> int:
    """Count ParagraphNode instances in the subtree rooted at node."""
    if not node:
        return 0

    count = 0
    stack: list[KnowledgeNode] = [node]
    while stack:
        current = stack.pop()
        if isinstance(current, ParagraphNode):
            count += 1
        if getattr(current, "children", None):
            stack.extend(current.children)
    return count


def get_title(node: KnowledgeNode) -> str:
    if isinstance(node, StructureNode):
        return node.attributes.title or ""
    return ""


def collect_detail_items(
    detail_nodes: list[KnowledgeNode],
) -> list[dict[str, Any]]:
    """Collect textual content from detail nodes for LLM refinement."""
    items: list[dict[str, Any]] = []
    for node in detail_nodes:
        if isinstance(node, ParagraphNode):
            text = (node.attributes.text or "").strip()
            if not text:
                continue
            items.append(
                {
                    "type": "paragraph",
                    "id": str(node.id),
                    "text": text,
                    "node": node,
                }
            )
        elif isinstance(node, TableNode):
            text = (node.attributes.extracted_text or "").strip()
            if not text:
                headers = " | ".join(node.attributes.headers or [])
                rows = [" | ".join(row) for row in (node.attributes.rows or [])]
                table_lines = []
                if headers:
                    table_lines.append(headers)
                table_lines.extend(rows[:10])
                text = "\n".join(table_lines).strip()
            if not text:
                continue
            items.append(
                {
                    "type": "table",
                    "id": str(node.id),
                    "text": text,
                    "node": node,
                }
            )
    return items


def _fallback_to_originals(
    items: list[dict[str, Any]],
    parent: StructureNode,
) -> list[ParagraphNode]:
    """Fallback: keep original paragraphs when no valid segments produced."""
    paragraphs: list[ParagraphNode] = []
    for it in items:
        source: KnowledgeNode = it["node"]
        if isinstance(source, ParagraphNode):
            paragraphs.append(source)
        else:
            paragraphs.append(
                ParagraphNode(
                    source_id=parent.source_id,
                    parent_id=parent.id,
                    node_type=NodeType.PARAGRAPH,
                    level=0,
                    attributes={"text": it["text"]},
                )
            )
    return paragraphs


def create_refined_paragraphs_from_segments(
    segments: list[dict[str, Any]],
    items: list[dict[str, Any]],
    parent: StructureNode,
) -> list[ParagraphNode]:
    """Create new ParagraphNodes from LLM segments.

    Args:
        segments: LLM output segments with text, summary, and optional metadata
        items: Original items for reference
        parent: Parent StructureNode

    Note:
        This function tracks consumed indices and recovers missing items
        as separate paragraphs to prevent data loss when LLM doesn't
        include all source_indices in output.
    """
    if not segments:
        return _fallback_to_originals(items, parent)

    refined: list[ParagraphNode] = []
    # Track which 1-indexed items have been consumed by segments
    consumed_indices: set[int] = set()

    for seg in segments:
        indices = seg.get("source_indices", [])
        content = (seg.get("text") or seg.get("content") or "").strip()
        if not content or not indices:
            continue

        valid_indices = [i - 1 for i in indices if isinstance(i, int) and 1 <= i <= len(items)]
        if not valid_indices:
            continue

        # Mark these indices as consumed (1-indexed for tracking)
        for i in indices:
            if isinstance(i, int) and 1 <= i <= len(items):
                consumed_indices.add(i)

        # Get page from source nodes
        page = None
        for idx in valid_indices:
            src_node: KnowledgeNode = items[idx]["node"]
            if isinstance(src_node, (ParagraphNode, TableNode, ImageNode)):
                page = getattr(src_node.metadata, "page", None)
                if page is not None:
                    break

        # Process summary
        summary = (seg.get("summary") or "").strip()
        paragraph_text = f"{summary} : {content}" if summary else content

        refined.append(
            ParagraphNode(
                source_id=parent.source_id,
                parent_id=parent.id,
                node_type=NodeType.PARAGRAPH,
                level=0,
                attributes={"text": paragraph_text},
                metadata={
                    "page": page,
                },
            )
        )

    # If no valid segments were produced, fallback to originals
    if not refined:
        return _fallback_to_originals(items, parent)

    # Check for missing indices and recover them as separate paragraphs
    all_indices = set(range(1, len(items) + 1))
    missing_indices = all_indices - consumed_indices

    if missing_indices:
        logger.warning(
            "LLM did not include all source items in segments, recovering missing items",
            extra={
                "missing_indices": sorted(missing_indices),
                "total_items": len(items),
                "consumed_count": len(consumed_indices),
            },
        )
        # Append missing items as separate paragraphs to prevent data loss
        for idx in sorted(missing_indices):
            item = items[idx - 1]  # Convert to 0-indexed
            source_node: KnowledgeNode = item["node"]

            # Get page from source node
            page = None
            if isinstance(source_node, (ParagraphNode, TableNode, ImageNode)):
                page = getattr(source_node.metadata, "page", None)

            refined.append(
                ParagraphNode(
                    source_id=parent.source_id,
                    parent_id=parent.id,
                    node_type=NodeType.PARAGRAPH,
                    level=0,
                    attributes={"text": item["text"]},
                    metadata={
                        "page": page,
                    },
                )
            )

    return refined


