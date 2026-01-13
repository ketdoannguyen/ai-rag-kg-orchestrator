"""File utilities for reading, writing, and parsing files."""

from __future__ import annotations

import json


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
