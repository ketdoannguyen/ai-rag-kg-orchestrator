from __future__ import annotations

from typing import Any, Dict, List, Optional, TYPE_CHECKING

from loguru import logger

from arango_client import (
    KnowledgeNode,
    ParagraphNode,
    StructureNode,
    TableNode,
    ImageNode,
    NodeType,
)

if TYPE_CHECKING:
    from config.domains import DomainSchema


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
            items.append({
                "type": "paragraph",
                "id": str(node.id),
                "text": text,
                "node": node,
            })
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
            items.append({
                "type": "table",
                "id": str(node.id),
                "text": text,
                "node": node,
            })
    return items


def _fallback_to_originals(
    items: List[Dict[str, Any]],
    parent: StructureNode,
) -> List[ParagraphNode]:
    """Fallback: keep original paragraphs when no valid segments produced."""
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
                    level=0,
                    attributes={"text": it["text"]},
                )
            )
    return paragraphs


def create_refined_paragraphs_from_segments(
    segments: List[Dict[str, Any]],
    items: List[Dict[str, Any]],
    parent: StructureNode,
    domain_schema: Optional["DomainSchema"] = None,
) -> List[ParagraphNode]:
    """Create new ParagraphNodes from LLM segments.
    
    Args:
        segments: LLM output segments with text, summary, and optional metadata
        items: Original items for reference
        parent: Parent StructureNode
        domain_schema: Optional schema for metadata normalization/validation
    """
    if not segments:
        return _fallback_to_originals(items, parent)

    refined: List[ParagraphNode] = []
    for seg in segments:
        indices = seg.get("source_indices", [])
        content = (seg.get("text") or seg.get("content") or "").strip()
        if not content or not indices:
            continue

        valid_indices = [
            i - 1 for i in indices
            if isinstance(i, int) and 1 <= i <= len(items)
        ]
        if not valid_indices:
            continue

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

        # Process domain metadata
        raw_metadata = seg.get("metadata") or {}
        domain_metadata = {}
        
        if raw_metadata and domain_schema:
            # Normalize using aliases
            normalized = domain_schema.normalize_metadata(raw_metadata)
            
            # Validate against allowed_values
            errors = domain_schema.validate_metadata(normalized)
            if errors:
                logger.debug(
                    "Filtered invalid metadata",
                    errors=errors,
                    raw=raw_metadata,
                )
                # Keep only valid fields
                normalized = {k: v for k, v in normalized.items() if k not in errors}
            
            domain_metadata = normalized
        elif raw_metadata:
            # No schema, keep as-is
            domain_metadata = raw_metadata

        refined.append(
            ParagraphNode(
                source_id=parent.source_id,
                parent_id=parent.id,
                node_type=NodeType.PARAGRAPH,
                level=0,
                attributes={"text": paragraph_text},
                metadata={
                    "page": page,
                    "domain_metadata": domain_metadata,
                },
            )
        )

    # If no valid segments were produced, fallback to originals
    if not refined:
        return _fallback_to_originals(items, parent)

    return refined


def aggregate_children_metadata(children: List[KnowledgeNode]) -> Dict[str, Any]:
    """Aggregate domain_metadata from children using union strategy.
    
    Collects all unique values for each field across children.
    Single values stay as-is, multiple unique values become a list.
    
    Example:
        Child 1: {"year": 2024, "major_name": "CNTT"}
        Child 2: {"year": 2025, "major_name": "CNTT"}
        Result:  {"year": [2024, 2025], "major_name": "CNTT"}
    """
    aggregated: Dict[str, set] = {}
    
    for child in children:
        # Get domain_metadata from child's metadata
        metadata = getattr(child, "metadata", None)
        if not metadata:
            continue
            
        domain_meta = getattr(metadata, "domain_metadata", None)
        if not domain_meta:
            # Try dict access
            if isinstance(metadata, dict):
                domain_meta = metadata.get("domain_metadata", {})
            else:
                continue
        
        if not domain_meta or not isinstance(domain_meta, dict):
            continue
            
        for key, value in domain_meta.items():
            if value is None:
                continue
            if key not in aggregated:
                aggregated[key] = set()
            # Handle list values
            if isinstance(value, list):
                for v in value:
                    if v is not None:
                        aggregated[key].add(v)
            else:
                aggregated[key].add(value)
    
    # Convert sets to final format
    result: Dict[str, Any] = {}
    for key, values in aggregated.items():
        if len(values) == 0:
            continue
        elif len(values) == 1:
            # Single value stays as-is (not a list)
            result[key] = next(iter(values))
        else:
            # Multiple values become sorted list
            try:
                result[key] = sorted(values)
            except TypeError:
                # Cannot sort mixed types
                result[key] = list(values)
    
    return result

