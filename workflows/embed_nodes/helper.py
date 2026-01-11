"""Helper functions for embed_nodes workflow."""

from __future__ import annotations

from typing import List, Tuple

from arango_client import (
    KnowledgeNode,
    StructureNode,
    ParagraphNode,
    ImageNode,
    TableNode,
)


def extract_text_for_embedding(node: KnowledgeNode) -> str | None:
    """
    Extract text from a node for embedding.

    For structure nodes: returns attributes.description (summary)
    For detail nodes: returns text content based on node type

    Args:
        node: KnowledgeNode to extract text from

    Returns:
        Text to embed, or None if node has no embeddable text
    """
    if isinstance(node, StructureNode):
        # Structure nodes: priority order - description, summary, title
        # 1. Try description (summary field in attributes)
        description = node.attributes.description
        if description and description.strip():
            return description.strip()

        # 2. Try summary field (set by enricher on the model)
        if hasattr(node, "summary") and node.summary and node.summary.strip():
            return node.summary.strip()

        # 3. Fallback to title
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


def collect_embeddable_nodes(nodes: List[KnowledgeNode]) -> List[Tuple[KnowledgeNode, str]]:
    """
    Collect all nodes that have embeddable text.

    Args:
        nodes: List of KnowledgeNode instances

    Returns:
        List of tuples: (node, text_to_embed)
    """
    embeddable: List[Tuple[KnowledgeNode, str]] = []

    for node in nodes:
        text = extract_text_for_embedding(node)
        if text:
            embeddable.append((node, text))

    return embeddable


def create_batches(
    items: List[Tuple[KnowledgeNode, str]], batch_size: int
) -> List[List[Tuple[KnowledgeNode, str]]]:
    """
    Split items into batches of specified size.

    Args:
        items: List of (node, text) tuples
        batch_size: Maximum number of items per batch

    Returns:
        List of batches, each containing up to batch_size items
    """
    batches: List[List[Tuple[KnowledgeNode, str]]] = []
    for i in range(0, len(items), batch_size):
        batch = items[i : i + batch_size]
        batches.append(batch)
    return batches
