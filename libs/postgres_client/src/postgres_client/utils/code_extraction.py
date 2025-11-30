"""Code extraction utilities for extracting identifiers from titles and contexts."""

from __future__ import annotations

import re
from typing import Any


def extract_code_from_title(title: str) -> str | None:
    """
    Extract code/identifier from title string.
    
    Detects patterns like:
    - "Chapter 1", "Chương 1", "Chapter I", "Chương I"
    - "Part A", "Phần A"
    - "I.", "II.", "III.", "IV.", "V.", etc. (Roman numerals)
    - "**I. ...**", "**1. ...**" (with markdown formatting)
    - "1.", "2.", "3.", etc. (Numbers)
    - "Bài 1", "Bài I"
    - "Section 1", "Mục 1"
    
    Args:
        title: Title string to extract code from
        
    Returns:
        Extracted code string or None if no code found
        
    Example:
        >>> extract_code_from_title("Chapter 1: Introduction")
        "Chapter 1"
        >>> extract_code_from_title("II. Main Content")
        "II"
        >>> extract_code_from_title("**I. Thông tin tổng quát**")
        "I"
        >>> extract_code_from_title("Bài 3: Toán học")
        "Bài 3"
    """
    if not title:
        return None
    
    # Remove markdown formatting (**, __, etc.)
    title = re.sub(r'\*\*|__|`', '', title).strip()
    
    # Pattern 1: Chapter/Chương + number or roman numeral
    # "Chapter 1", "Chương 1", "Chapter I", "Chương I"
    pattern1 = re.match(
        r'^(Chapter|Chương|CHƯƠNG)\s+([IVXLCDM]+|\d+)',
        title,
        re.IGNORECASE
    )
    if pattern1:
        prefix = pattern1.group(1)
        number = pattern1.group(2)
        return f"{prefix} {number}"
    
    # Pattern 2: Part/Phần + letter or number
    # "Part A", "Phần A", "Part 1", "Phần 1"
    pattern2 = re.match(
        r'^(Part|Phần|PHẦN)\s+([A-Z]|\d+)',
        title,
        re.IGNORECASE
    )
    if pattern2:
        prefix = pattern2.group(1)
        identifier = pattern2.group(2)
        return f"{prefix} {identifier}"
    
    # Pattern 3: Bài + number or roman numeral
    # "Bài 1", "Bài I", "BÀI 1"
    pattern3 = re.match(
        r'^(Bài|BÀI)\s+([IVXLCDM]+|\d+)',
        title,
        re.IGNORECASE
    )
    if pattern3:
        prefix = pattern3.group(1)
        number = pattern3.group(2)
        return f"{prefix} {number}"
    
    # Pattern 4: Section/Mục + number
    # "Section 1", "Mục 1", "MỤC 1"
    pattern4 = re.match(
        r'^(Section|Mục|MỤC)\s+(\d+)',
        title,
        re.IGNORECASE
    )
    if pattern4:
        prefix = pattern4.group(1)
        number = pattern4.group(2)
        return f"{prefix} {number}"
    
    # Pattern 5: Roman numeral at start (I., II., III., etc.)
    # "I.", "II.", "III.", "IV.", "V.", etc.
    # Also handles "**I. ...**" format
    pattern5 = re.match(
        r'^\s*([IVXLCDM]+)\.',
        title,
        re.IGNORECASE
    )
    if pattern5:
        return pattern5.group(1).upper()
    
    # Pattern 6: Number at start (1., 2., 3., etc.)
    # "1.", "2.", "3.", etc.
    # Also handles "**1. ...**" format
    pattern6 = re.match(
        r'^\s*(\d+)\.',
        title
    )
    if pattern6:
        return pattern6.group(1)
    
    # Pattern 7: Letter at start (a., b., c., etc. or A., B., C., etc.)
    # "a.", "b.", "A.", "B.", etc.
    pattern7 = re.match(
        r'^\s*([a-zA-Z])\.',
        title
    )
    if pattern7:
        return pattern7.group(1).upper()
    
    # Pattern 8: Letter with parenthesis (a), b), c), etc.)
    # "a)", "b)", "A)", "B)", etc.
    pattern8 = re.match(
        r'^\s*([a-zA-Z])\)',
        title
    )
    if pattern8:
        return pattern8.group(1).upper()
    
    return None


def extract_code_from_context(
    current_text: str,
    previous_items: list[Any] | None = None,
    previous_text: str | None = None
) -> str | None:
    """
    Extract code from current text or previous context items.
    
    This function tries to extract code from:
    1. Current text (using extract_code_from_title)
    2. Previous items/blocks if current text doesn't contain code
    3. Previous text if provided
    
    Important: If code is found in a previous item, it checks if that code
    is already used in the title of the next item. If so, the code is
    considered "consumed" and won't be reused for the current item.
    
    Args:
        current_text: Current title/text to check
        previous_items: List of previous blocks/items to check (e.g., from parser)
        previous_text: Previous text string to check
        
    Returns:
        Extracted code string or None if no code found
        
    Example:
        >>> # Code in current text
        >>> extract_code_from_context("I. Introduction")
        "I"
        
        >>> # Code in previous block (not consumed)
        >>> extract_code_from_context("Introduction", previous_items=[{"html": "<h1>I</h1>"}])
        "I"
        
        >>> # Code in previous block but already consumed by next item
        >>> # Block 1: "CHƯƠNG I", Block 2: "CHƯƠNG I: VẬT LÍ NHIỆT"
        >>> # Block 3: "Nội dung" -> should NOT get "CHƯƠNG I" because it's already used in Block 2
        >>> extract_code_from_context(
        ...     "Nội dung",
        ...     previous_items=[
        ...         {"html": "<h1>CHƯƠNG I</h1>"},
        ...         {"html": "<h1>CHƯƠNG I: VẬT LÍ NHIỆT</h1>"}
        ...     ]
        ... )
        None
    """
    # First, try to extract from current text
    code = extract_code_from_title(current_text)
    if code:
        return code
    
    # If not found in current text, check previous items
    if previous_items:
        # Check items in reverse order (most recent first)
        for i, item in enumerate(reversed(previous_items)):
            item_index = len(previous_items) - 1 - i
            found_code = None
            
            # Handle dict-like items (from parser blocks)
            if isinstance(item, dict):
                # Try multiple fields that might contain the code
                text_fields = ["text", "html", "content", "block_type"]
                for field in text_fields:
                    text = item.get(field, "")
                    if text and isinstance(text, str):
                        text = _clean_text_for_extraction(text)
                        # For block_type, only check if it's a simple identifier
                        if field == "block_type" and len(text) > 20:
                            continue
                        found_code = extract_code_from_title(text)
                        if found_code:
                            break
            # Handle string items
            elif isinstance(item, str):
                cleaned = _clean_text_for_extraction(item)
                found_code = extract_code_from_title(cleaned)
            
            if found_code:
                # Check if code is already consumed by next item's title
                if item_index + 1 < len(previous_items):
                    next_item = previous_items[item_index + 1]
                    next_item_text = _extract_text_from_item(next_item)
                    
                    # If code appears in next item's title, it's already consumed
                    if next_item_text and found_code.upper() in next_item_text.upper():
                        continue
                
                return found_code
    
    # Check previous text if provided
    if previous_text:
        previous_text = _clean_text_for_extraction(previous_text)
        code = extract_code_from_title(previous_text)
        if code:
            return code
    
    return None


def clean_title_from_code(title: str, code: str) -> str:
    """
    Remove code prefix from title if code is present.
    
    This function removes the code from the beginning of the title,
    handling various separators like ":", ".", " ", etc.
    
    Args:
        title: Original title string
        code: Code that was extracted from the title
        
    Returns:
        Cleaned title with code prefix removed
        
    Example:
        >>> clean_title_from_code("CHƯƠNG I: VẬT LÍ NHIỆT", "CHƯƠNG I")
        "VẬT LÍ NHIỆT"
        >>> clean_title_from_code("I. Introduction", "I")
        "Introduction"
        >>> clean_title_from_code("1. Main Content", "1")
        "Main Content"
        >>> clean_title_from_code("Bài 3: Toán học", "Bài 3")
        "Toán học"
    """
    if not title or not code:
        return title
    
    # Remove markdown formatting first
    title_clean = re.sub(r'\*\*|__|`', '', title).strip()
    code_clean = code.strip()
    
    # Try to match and remove code prefix with various separators
    # Pattern 1: Code followed by ": " or ":"
    pattern1 = re.compile(
        rf'^{re.escape(code_clean)}\s*:\s*(.+)$',
        re.IGNORECASE | re.DOTALL
    )
    match = pattern1.match(title_clean)
    if match:
        return match.group(1).strip()
    
    # Pattern 2: Code followed by ". " or "."
    pattern2 = re.compile(
        rf'^{re.escape(code_clean)}\s*\.\s*(.+)$',
        re.IGNORECASE | re.DOTALL
    )
    match = pattern2.match(title_clean)
    if match:
        return match.group(1).strip()
    
    # Pattern 3: Code followed by whitespace
    pattern3 = re.compile(
        rf'^{re.escape(code_clean)}\s+(.+)$',
        re.IGNORECASE | re.DOTALL
    )
    match = pattern3.match(title_clean)
    if match:
        return match.group(1).strip()
    
    # Pattern 4: Code at the start (exact match or with trailing punctuation)
    if title_clean.upper().startswith(code_clean.upper()):
        # Try to remove code with any trailing punctuation
        remaining = title_clean[len(code_clean):].lstrip(' :.-')
        if remaining:
            return remaining.strip()
    
    # If no pattern matches, return original title
    return title


def _extract_text_from_item(item: Any) -> str:
    """Extract text from an item (dict or string)."""
    if isinstance(item, dict):
        for field in ["text", "html", "content"]:
            text = item.get(field, "")
            if text and isinstance(text, str):
                return _clean_text_for_extraction(text)
    elif isinstance(item, str):
        return _clean_text_for_extraction(item)
    return ""


def _clean_text_for_extraction(text: str) -> str:
    """Clean text for code extraction (remove HTML tags, normalize)."""
    if not text:
        return ""
    
    # Remove HTML tags
    text = re.sub(r'<[^>]+>', '', text)
    
    # Decode HTML entities (basic ones)
    text = text.replace('&gt;', '>').replace('&lt;', '<').replace('&amp;', '&')
    text = text.replace('&nbsp;', ' ')
    
    # Normalize whitespace
    text = re.sub(r'\s+', ' ', text).strip()
    
    return text

