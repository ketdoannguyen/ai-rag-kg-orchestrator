from __future__ import annotations

from typing import Any, Dict, List

from postgres_client.models import (
    KnowledgeNode,
    ParagraphNode,
    StructureNode,
    TableNode,
    ImageNode,
)
from postgres_client.models.enums import NodeType


def count_paragraphs(node: KnowledgeNode | None) -> int:
    """Count ParagraphNode instances in the subtree rooted at node."""
    if not node:
        return 0

    count = 0
    stack: List[KnowledgeNode] = [node]
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
    detail_nodes: List[KnowledgeNode],
) -> List[Dict[str, Any]]:
    """Collect textual content from detail nodes for LLM refinement."""
    items: List[Dict[str, Any]] = []
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


def create_refined_paragraphs_from_segments(
    segments: List[Dict[str, Any]],
    items: List[Dict[str, Any]],
    parent: StructureNode,
) -> List[ParagraphNode]:
    """Create new ParagraphNodes from LLM segments."""
    if not segments:
        paragraphs: List[ParagraphNode] = []
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
                        level_down=0,
                        level_up=0,
                        attributes={"text": it["text"]},
                    )
                )
        return paragraphs

    refined: List[ParagraphNode] = []
    for seg in segments:
        indices = seg.get("source_indices", [])
        content = (seg.get("text") or seg.get("content") or "").strip()
        if not content or not indices:
            continue

        valid_indices = [
            i - 1
            for i in indices
            if isinstance(i, int) and 1 <= i <= len(items)
        ]
        if not valid_indices:
            continue

        page = None
        for idx in valid_indices:
            src_node: KnowledgeNode = items[idx]["node"]
            if isinstance(src_node, (ParagraphNode, TableNode, ImageNode)):
                page = getattr(src_node.metadata, "page", None)
                if page is not None:
                    break

        summary = (seg.get("summary") or "").strip()
        paragraph_text = content
        if summary:
            paragraph_text = f"{summary} : {content}"

        refined.append(
            ParagraphNode(
                source_id=parent.source_id,
                parent_id=parent.id,
                node_type=NodeType.PARAGRAPH,
                level_down=parent.level_down + 1,
                level_up=1,
                attributes={"text": paragraph_text},
                metadata={"page": page},
            )
        )

    if not refined:
        return create_refined_paragraphs_from_segments([], items, parent)

    return refined
