"""Helper functions for embed_nodes workflow."""

from __future__ import annotations

from arango_client import (
    ImageNode,
    KnowledgeNode,
    ParagraphNode,
    StructureNode,
    TableNode,
)


def extract_text_for_embedding(node: KnowledgeNode) -> str | None:
    """
    Extract text from a node for embedding.

    For structure nodes: returns summary (if available) or falls back to title
    For detail nodes: returns text content based on node type

    Args:
        node: KnowledgeNode to extract text from

    Returns:
        Text to embed, or None if node has no embeddable text
    """
    if isinstance(node, StructureNode):
        # Structure nodes: priority order - summary, title
        # 1. Try summary field (set by enricher)
        if node.summary and node.summary.strip():
            return node.summary.strip()

        # 2. Fallback to title
        title = node.attributes.title
        if title and title.strip():
            return title.strip()

        return None

    elif isinstance(node, ParagraphNode):
        # Paragraph nodes: use text attribute
        text = node.attributes.text
        if text and text.strip():
            return text.strip()
        return None

    elif isinstance(node, ImageNode):
        # Image nodes: use extracted_text if available
        extracted_text = node.attributes.extracted_text
        if extracted_text and extracted_text.strip():
            return extracted_text.strip()
        return None

    elif isinstance(node, TableNode):
        # Table nodes: use extracted_text if available
        extracted_text = node.attributes.extracted_text
        if extracted_text and extracted_text.strip():
            return extracted_text.strip()
        return None

    return None


def collect_embeddable_nodes(nodes: list[KnowledgeNode]) -> list[tuple[KnowledgeNode, str]]:
    """
    Collect all nodes that have embeddable text.

    Args:
        nodes: List of KnowledgeNode instances

    Returns:
        List of tuples: (node, text_to_embed)
    """
    embeddable: list[tuple[KnowledgeNode, str]] = []

    for node in nodes:
        text = extract_text_for_embedding(node)
        if text:
            embeddable.append((node, text))

    return embeddable


def create_batches(
    items: list[tuple[KnowledgeNode, str]], batch_size: int
) -> list[list[tuple[KnowledgeNode, str]]]:
    """
    Split items into batches of specified size.

    Args:
        items: List of (node, text) tuples
        batch_size: Maximum number of items per batch

    Returns:
        List of batches, each containing up to batch_size items
    """
    batches: list[list[tuple[KnowledgeNode, str]]] = []
    for i in range(0, len(items), batch_size):
        batch = items[i : i + batch_size]
        batches.append(batch)
    return batches
