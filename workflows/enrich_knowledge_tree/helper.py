from __future__ import annotations

from typing import Any

from arango_client import (
    ImageNode,
    KnowledgeNode,
    NodeType,
    ParagraphNode,
    StructureNode,
    TableNode,
)
from loguru import logger


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
    *,
    preserve_tables: bool = False,
) -> list[dict[str, Any]]:
    """Collect textual content from detail nodes for LLM refinement.

    A node with nothing to send is left out, with one exception under
    ``preserve_tables``: a table the producer could read no text out of is still a
    table between the paragraphs, so it is collected as the item the split reads
    (carrying no text) and the refiner puts the node itself back where it was.
    Leaving it out would let a merge reach across the table and drop the node from
    the rebuilt children.
    """
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
            if not text and not preserve_tables:
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


def detail_runs(
    items: list[dict[str, Any]],
    *,
    preserve_tables: bool,
) -> list[list[dict[str, Any]]]:
    """Split collected items into the groups one merge may run over.

    Without the policy the whole list is a single group, so the runtime keeps its
    one merge call over every item. With it, a table is a boundary: the
    contiguous paragraph runs between tables are the merge inputs and each table
    is its own group, so a merge can neither reach across a table nor consume
    one.
    """
    if not preserve_tables:
        return [items]

    runs: list[list[dict[str, Any]]] = []
    run: list[dict[str, Any]] = []
    for item in items:
        if item["type"] == "table":
            if run:
                runs.append(run)
                run = []
            runs.append([item])
        else:
            run.append(item)
    if run:
        runs.append(run)
    return runs


def _fallback_to_originals(
    items: list[dict[str, Any]],
    parent: StructureNode,
) -> list[KnowledgeNode]:
    """Fallback: keep original paragraphs when no valid segments produced."""
    paragraphs: list[KnowledgeNode] = []
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
    excluded_indices: list[dict[str, Any]] | None = None,
    *,
    preserve_details: bool = False,
    include_summaries: bool = True,
) -> list[KnowledgeNode]:
    """Create new ParagraphNodes from LLM segments.

    Args:
        segments: LLM output segments with text, summary, and optional metadata
        items: Original items for reference
        parent: Parent StructureNode
        excluded_indices: List of {"index": int, "reason": str} for intentionally excluded items
        preserve_details: Keep an unconsumed item as the node it already was, in
            its own place, instead of rebuilding it as a new paragraph at the end.
        include_summaries: Prefix a segment with the summary the prompt asked for.
            Off, the segment text is stored alone.

    Note:
        This function tracks consumed indices and recovers missing items
        as separate paragraphs to prevent data loss when LLM doesn't
        include all source_indices in output. With `preserve_details` the
        recovered item is the original node, put back in its own place — a
        node that is not a paragraph stays the node it was, which is why the
        result is typed as the detail nodes rather than paragraphs.
    """
    if not segments:
        return _fallback_to_originals(items, parent)

    refined: list[KnowledgeNode] = []
    # The lowest source index each emitted paragraph covers, so an unconsumed
    # item can be put back among them in its own place.
    anchors: list[int] = []
    # Track which 1-indexed items have been consumed by segments
    consumed_indices: set[int] = set()

    # Track intentionally excluded indices
    intentionally_excluded: set[int] = set()
    if excluded_indices:
        for exc in excluded_indices:
            idx = exc.get("index")
            reason = exc.get("reason", "unknown")
            if isinstance(idx, int) and 1 <= idx <= len(items):
                intentionally_excluded.add(idx)
                logger.debug(
                    f"Item {idx} excluded by LLM",
                    extra={"index": idx, "reason": reason},
                )

    for seg in segments:
        indices = seg.get("source_indices", [])
        content = (seg.get("text") or seg.get("content") or "").strip()
        if not content or not indices:
            continue

        valid_indices = [i - 1 for i in indices if isinstance(i, int) and 1 <= i <= len(items)]
        if not valid_indices:
            continue

        anchors.append(min(valid_indices))

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
        paragraph_text = f"{summary} : {content}" if summary and include_summaries else content

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

    # Check for missing indices (not in segments AND not intentionally excluded)
    all_indices = set(range(1, len(items) + 1))
    accounted_indices = consumed_indices | intentionally_excluded
    missing_indices = all_indices - accounted_indices

    if missing_indices:
        logger.warning(
            "LLM did not include all source items in segments, recovering missing items",
            extra={
                "missing_indices": sorted(missing_indices),
                "total_items": len(items),
                "consumed_count": len(consumed_indices),
                "excluded_count": len(intentionally_excluded),
            },
        )
        if preserve_details:
            # Put each unconsumed item back where it was, keeping the node it
            # already is (type, text, attributes, order, page) rather than
            # rebuilding it as a paragraph.
            recovered: list[tuple[int, KnowledgeNode]] = [
                (anchor, node) for anchor, node in zip(anchors, refined, strict=True)
            ]
            for idx in sorted(missing_indices):
                recovered.append((idx - 1, items[idx - 1]["node"]))
            recovered.sort(key=lambda pair: pair[0])
            return [node for _, node in recovered]

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
