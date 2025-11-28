"""Helper functions for refine_json workflow."""

import json
from typing import List

from workflows.models import Paragraph


def load_json_file(file_path: str) -> dict:
    """
    Load JSON file synchronously.

    Args:
        file_path: Path to JSON file

    Returns:
        Parsed JSON data as dictionary

    Raises:
        FileNotFoundError: If file doesn't exist
        json.JSONDecodeError: If file contains invalid JSON
    """
    with open(file_path, "r", encoding="utf-8") as f:
        return json.load(f)


def write_file(file_path: str, content: str) -> None:
    """
    Write content to file synchronously.

    Args:
        file_path: Path to output file
        content: Content to write
    """
    with open(file_path, "w", encoding="utf-8") as f:
        f.write(content)


def extract_json_from_response(content: str) -> str:
    """
    Extract JSON content from LLM response (handles code blocks).

    Args:
        content: Raw LLM response content

    Returns:
        Extracted JSON string
    """
    if "```json" in content:
        return content.split("```json")[1].split("```")[0].strip()
    elif "```" in content:
        return content.split("```")[1].split("```")[0].strip()
    return content.strip()


def calculate_merged_bbox(paragraphs: List[Paragraph]) -> List[int]:
    """
    Calculate union bounding box from multiple paragraphs.

    Args:
        paragraphs: List of paragraphs to merge bboxes

    Returns:
        Merged bbox [x1, y1, x2, y2] or empty list if invalid
    """
    valid_bboxes = [
        p.bbox for p in paragraphs
        if p.bbox and len(p.bbox) == 4
        and all(isinstance(x, (int, float)) for x in p.bbox)
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
    buffer: List[Paragraph]
) -> List[Paragraph]:
    """
    Create refined Paragraph objects from LLM response segments.

    Args:
        segments: List of segment dicts from LLM response
        buffer: Original paragraph buffer

    Returns:
        List of refined Paragraph objects
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
        page = relevant_paragraphs[0].page

        refined_paragraphs.append(Paragraph(
            text=text,
            page=page,
            bbox=merged_bbox
        ))

    return refined_paragraphs

