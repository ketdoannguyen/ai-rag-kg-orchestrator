"""Helper utilities for Router Agent.

This module contains utility functions for:
- JSON parsing and validation from LLM responses
"""

from __future__ import annotations

import json
import re
from typing import Any, Dict, FrozenSet

from loguru import logger

# Pre-compiled regex for markdown code block extraction
_CODE_BLOCK_PATTERN = re.compile(r"```(?:json)?\s*\n?(.*?)\n?```", re.DOTALL)

# Valid intents for O(1) lookup
_VALID_INTENTS: FrozenSet[str] = frozenset({
    "DISCOVERY", "SPECIFIC_FACT", "INSTRUCTION",
    "COMPARISON", "CONTACT_INFO", "CHIT_CHAT"
})


def extract_json_from_response(content: str) -> str:
    """Extract JSON from LLM response, handling markdown code blocks.
    
    Args:
        content: Raw LLM response that may contain JSON
        
    Returns:
        Extracted JSON string
        
    Raises:
        ValueError: If no valid JSON structure found
    """
    if not content:
        raise ValueError("Empty response content")
    
    # Try to extract from markdown code block first
    match = _CODE_BLOCK_PATTERN.search(content)
    if match:
        content = match.group(1).strip()
    
    # Find JSON object boundaries
    start_idx = content.find("{")
    end_idx = content.rfind("}")
    
    if start_idx == -1 or end_idx == -1 or start_idx > end_idx:
        raise ValueError(f"No valid JSON object found: {content[:200]}")
    
    return content[start_idx:end_idx + 1]


def parse_router_response(content: str, original_query: str) -> Dict[str, Any]:
    """Parse and validate router LLM response.
    
    Args:
        content: Raw LLM response
        original_query: Original user query (for fallback)
        
    Returns:
        Parsed dictionary with: intent, rewritten_query, sub_questions
    """
    try:
        json_str = extract_json_from_response(content)
        data = json.loads(json_str)
        
        if not isinstance(data, dict):
            raise ValueError("Response is not a JSON object")
        
        # Validate intent
        intent = data.get("intent", "").upper()
        if intent not in _VALID_INTENTS:
            logger.warning("Invalid intent '{}', defaulting to DISCOVERY", intent)
            data["intent"] = "DISCOVERY"
        else:
            data["intent"] = intent
        
        # Ensure rewritten_query exists
        if not data.get("rewritten_query"):
            data["rewritten_query"] = original_query
        
        # Normalize sub_questions
        sub_q = data.get("sub_questions")
        if not isinstance(sub_q, list):
            data["sub_questions"] = []
        else:
            normalized = []
            for item in sub_q:
                if isinstance(item, str):
                    normalized.append({"query": item, "keywords": [], "filters": None})
                elif isinstance(item, dict):
                    filters = item.get("filters")
                    normalized.append({
                        "query": item.get("query", ""),
                        "keywords": item.get("keywords", []),
                        "filters": filters if filters else None,
                    })
            data["sub_questions"] = normalized
        
        return data
        
    except (json.JSONDecodeError, ValueError) as e:
        logger.error("Failed to parse router response: {}", str(e))
        return {
            "intent": "CHIT_CHAT",
            "rewritten_query": original_query,
            "sub_questions": [],
        }
