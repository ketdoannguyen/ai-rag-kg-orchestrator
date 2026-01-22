"""Shared utilities for extract_relations workflow."""

from __future__ import annotations

import json
import re
from typing import Any


def extract_json_from_llm_response(text: str) -> dict[str, Any] | list[dict[str, Any]]:
    """Extract JSON from LLM response, handling markdown code blocks.

    Attempts parsing in order:
    1. Direct JSON parse
    2. Extract from markdown code block (```json ... ```)
    3. Find raw JSON object/array in text (greedy)
    4. Find raw JSON object/array in text (non-greedy)

    Args:
        text: LLM response text

    Returns:
        Parsed JSON as dict or list

    Raises:
        ValueError: If no valid JSON found
    """
    if not text:
        raise ValueError("Empty response text")

    errors = []

    # Direct parse
    try:
        return json.loads(text)
    except json.JSONDecodeError as e:
        errors.append(f"Direct parse failed: {e}")

    # Extract from markdown code block
    match = re.search(r"```(?:json)?\s*([\s\S]*?)\s*```", text)
    if match:
        try:
            return json.loads(match.group(1))
        except json.JSONDecodeError as e:
            errors.append(f"Markdown parse failed: {e}")

    # Find raw JSON object/array (greedy - tries to match largest block)
    match = re.search(r"(\{[\s\S]*\}|\[[\s\S]*\])", text)
    if match:
        try:
            return json.loads(match.group(1))
        except json.JSONDecodeError as e:
            errors.append(f"Greedy regex parse failed: {e}")

    # Fallback: Find first potential JSON object (non-greedy)
    # This helps when there is trailing text or multiple objects
    match = re.search(r"(\{[\s\S]*?\}|\[[\s\S]*?\])", text)
    if match:
        try:
            return json.loads(match.group(1))
        except json.JSONDecodeError as e:
            errors.append(f"Non-greedy regex parse failed: {e}")

    error_msg = "; ".join(errors)
    raise ValueError(f"Could not extract JSON: {text[:200]}... Errors: {error_msg}")
