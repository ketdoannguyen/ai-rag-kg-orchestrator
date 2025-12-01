"""Helper functions for refine_json workflow."""

from typing import List

from postgres_client.models import ParagraphNode
from postgres_client.models.attributes_metadata import (
    ParagraphAttributes,
    ParagraphMetadata,
)


def calculate_merged_bbox(paragraphs: List[ParagraphNode]) -> List[int]:
    """
    Calculate union bounding box from multiple paragraphs.

    Args:
        paragraphs: List of paragraphs to merge bboxes

    Returns:
        Merged bbox [x1, y1, x2, y2] or empty list if invalid
    """
    valid_bboxes = [
        p.metadata.bbox for p in paragraphs
        if p.metadata.bbox and len(p.metadata.bbox) == 4
        and all(isinstance(x, (int, float)) for x in p.metadata.bbox)
    ]

    if not valid_bboxes:
        return []

    x1s = [bbox[0] for bbox in valid_bboxes]
    y1s = [bbox[1] for bbox in valid_bboxes]
    x2s = [bbox[2] for bbox in valid_bboxes]
    y2s = [bbox[3] for bbox in valid_bboxes]

    return [min(x1s), min(y1s), max(x2s), max(y2s)]


def create_refined_paragraphs(
    segments: List[dict],
    buffer: List[ParagraphNode]
) -> List[ParagraphNode]:
    """
    Create refined ParagraphNode objects from LLM response segments.

    Args:
        segments: List of segment dicts from LLM response
        buffer: Original paragraph buffer

    Returns:
        List of refined ParagraphNode objects
    """
    refined_paragraphs = []

    for seg in segments:
        indices = seg.get("source_indices", [])
        text = seg.get("content", "").strip()

        if not indices or not text:
            continue

        # Fix index out of bounds: indices are 1-based, convert to 0-based
        # Validate indices are within buffer range
        valid_indices = [
            i - 1 for i in indices
            if isinstance(i, int) and 1 <= i <= len(buffer)
        ]

        if not valid_indices:
            continue

        relevant_paragraphs = [buffer[idx] for idx in valid_indices]
        if not relevant_paragraphs:
            continue

        # Calculate merged bbox
        merged_bbox = calculate_merged_bbox(relevant_paragraphs)

        # Use page from first paragraph
        page = relevant_paragraphs[0].metadata.page

        refined_paragraphs.append(ParagraphNode(
            attributes=ParagraphAttributes(text=text),
            metadata=ParagraphMetadata(
            page=page,
            bbox=merged_bbox
            )
        ))

    return refined_paragraphs

