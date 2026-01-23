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
    from config.domains import DomainSchema


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
    domain_schema: DomainSchema | None = None,
    domain: str | None = None,
) -> list[ParagraphNode]:
    """Create new ParagraphNodes from LLM segments.

    Args:
        segments: LLM output segments with text, summary, and optional metadata
        items: Original items for reference
        parent: Parent StructureNode
        domain_schema: Optional schema for metadata normalization/validation
        domain: Domain identifier to set on created nodes

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

        # Process domain filter from LLM response
        raw_filter = (
            seg.get("filter") or seg.get("metadata") or {}
        )  # fallback to metadata for backward compat
        domain_filter = {}

        if raw_filter and domain_schema:
            # Normalize using aliases
            normalized = domain_schema.normalize_metadata(raw_filter)

            # Validate against allowed_values
            errors = domain_schema.validate_metadata(normalized)
            if errors:
                logger.debug(
                    "Filtered invalid filter values",
                    errors=errors,
                    raw=raw_filter,
                )
                # Keep only valid fields
                normalized = {k: v for k, v in normalized.items() if k not in errors}

            domain_filter = normalized
        elif raw_filter:
            # No schema, keep as-is
            domain_filter = raw_filter

        refined.append(
            ParagraphNode(
                source_id=parent.source_id,
                parent_id=parent.id,
                node_type=NodeType.PARAGRAPH,
                level=0,
                domain=domain,
                attributes={"text": paragraph_text},
                metadata={
                    "page": page,
                    "domain_filter": domain_filter,
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

            # Preserve original domain_filter if exists
            original_filter = {}
            if hasattr(source_node, "metadata") and hasattr(source_node.metadata, "domain_filter"):
                original_filter = source_node.metadata.domain_filter or {}

            refined.append(
                ParagraphNode(
                    source_id=parent.source_id,
                    parent_id=parent.id,
                    node_type=NodeType.PARAGRAPH,
                    level=0,
                    domain=domain,
                    attributes={"text": item["text"]},
                    metadata={
                        "page": page,
                        "domain_filter": original_filter,
                    },
                )
            )

    return refined


def aggregate_children_filter(children: list[KnowledgeNode]) -> dict[str, Any]:
    """Aggregate domain_filter from direct children using union strategy.

    Collects all unique values for each field across direct children only.
    ALL values are returned as lists for consistency.

    Example:
        Child 1: {"year": [2024], "major_name": ["CNTT"]}
        Child 2: {"year": [2025], "major_name": ["CNTT"]}
        Result:  {"year": [2024, 2025], "major_name": ["CNTT"]}
    """
    aggregated: dict[str, set] = {}

    for child in children:
        # Get domain_filter from child's metadata (direct children only)
        metadata = getattr(child, "metadata", None)
        if not metadata:
            continue

        domain_filter = getattr(metadata, "domain_filter", None)
        if not domain_filter:
            # Try dict access
            if isinstance(metadata, dict):
                domain_filter = metadata.get("domain_filter", {})
            else:
                continue

        if not domain_filter or not isinstance(domain_filter, dict):
            continue

        for key, value in domain_filter.items():
            if value is None:
                continue
            if key not in aggregated:
                aggregated[key] = set()
            # Handle list values (expected format)
            if isinstance(value, list):
                for v in value:
                    if v is not None:
                        aggregated[key].add(v)
            else:
                # Handle legacy single values
                aggregated[key].add(value)

    # Convert sets to lists (ALL values must be lists for consistency)
    result: dict[str, Any] = {}
    for key, values in aggregated.items():
        if len(values) == 0:
            continue
        # ALWAYS return as list for consistency with schema type='list'
        try:
            result[key] = sorted(values)
        except TypeError:
            # Cannot sort mixed types
            result[key] = list(values)

    return result
