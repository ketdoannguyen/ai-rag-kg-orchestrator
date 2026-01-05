"""Markdown processing utilities for web ingestion.

Functions for assessing markdown quality, finding content start,
and extracting metadata from markdown.
"""

from __future__ import annotations

import re
from urllib.parse import urlparse

from loguru import logger


def clean_markdown_content(markdown: str) -> str:
    """Clean markdown content by removing TOC sections and fixing duplicates.
    
    This function:
    1. Removes Table of Contents (TOC) sections
    2. Fixes image alt text duplicates (![alt](url)alt -> ![alt](url))
    3. Removes empty links [](...) patterns
    
    Args:
        markdown: Raw markdown content
        
    Returns:
        Cleaned markdown content
    """
    if not markdown:
        return markdown
    
    lines = markdown.splitlines()
    cleaned_lines: list[str] = []
    in_toc_section = False
    toc_indent = 0
    
    for i, line in enumerate(lines):
        stripped = line.strip()
        
        # Detect TOC heading patterns
        toc_patterns = [
            r'^#{1,3}\s+(?:\*\*)?(?:Mục lục|Table of Contents|Contents|Nội dung|Danh mục)',
            r'^#{1,3}\s+Mục lục bài viết',
        ]
        
        if any(re.match(p, stripped, re.IGNORECASE) for p in toc_patterns):
            in_toc_section = True
            toc_indent = len(line) - len(line.lstrip())
            logger.debug(f"Detected TOC section at line {i}")
            continue
        
        # If in TOC section, check for end
        if in_toc_section:
            # TOC ends when we hit:
            # 1. A non-list, non-empty line (heading or paragraph)
            # 2. A line with less indentation than expected for TOC items
            is_toc_item = (
                re.match(r'^\s*\d+\.\s*\[', stripped) or  # Numbered list with link
                re.match(r'^\s*[\*\-\+]\s*\[', stripped) or  # Bullet list with link
                re.match(r'^\s*\d+\.\s*$', stripped) or  # Just a number
                stripped == ''  # Empty line
            )
            
            if is_toc_item:
                continue  # Skip TOC item
            else:
                # Check if it's still indented TOC content
                current_indent = len(line) - len(line.lstrip())
                if current_indent > toc_indent and re.search(r'\[.*?\]\(#', stripped):
                    continue  # Skip indented anchor links
                
                # TOC section ended
                in_toc_section = False
                logger.debug(f"TOC section ended at line {i}")
        
        # Remove empty links: [](url) that serve no purpose
        line = re.sub(r'\[\]\([^\)]+\)', '', line)
        
        # Skip lines that are just empty after cleaning
        if not line.strip():
            # Keep one empty line for paragraph separation, skip consecutive
            if cleaned_lines and cleaned_lines[-1].strip() == '':
                continue
        
        cleaned_lines.append(line)
    
    result = '\n'.join(cleaned_lines)
    
    # Fix image alt text duplicates: ![alt](url)alt -> ![alt](url)
    result = _fix_image_duplicates(result)
    
    # Final cleanup: remove excessive newlines
    result = re.sub(r'\n{3,}', '\n\n', result)
    
    return result


def _fix_image_duplicates(text: str) -> str:
    """Fix image alt text duplicates: ![alt](url)alt -> ![alt](url).
    
    Crawl4AI sometimes outputs images with duplicate alt text like:
    ![Truy cập...](url)Truy cập...
    
    This function removes the duplicate trailing text.
    
    Args:
        text: Markdown text with potential image duplicates
        
    Returns:
        Text with image duplicates removed
    """
    # Find all image patterns
    img_pattern = r'!\[([^\]]*)\]\(([^\)]+)\)'
    
    result = text
    offset = 0
    
    for match in re.finditer(img_pattern, text):
        alt = match.group(1)
        if not alt or len(alt) < 3:
            continue
        
        # Get position after the image markdown (accounting for previous removals)
        img_end = match.end() + offset
        remaining = result[img_end:]
        
        # Check if remaining text starts with the alt text
        if remaining.startswith(alt):
            # Remove the duplicate
            result = result[:img_end] + remaining[len(alt):]
            offset -= len(alt)
            logger.debug(f"Removed duplicate image alt text: {alt[:30]}...")
    
    return result



def assess_markdown_quality(markdown: str) -> tuple[float, str]:
    """Assess markdown quality and return (score, reason).
    
    Evaluates based on:
    - Number of headings (H1-H6)
    - Number of paragraph lines (non-link, non-list)
    - Link ratio (high link ratio = likely navigation)
    - Content length
    
    Args:
        markdown: Markdown content to assess
        
    Returns:
        tuple of (score: 0.0-1.0, reason: str describing the scoring)
    """
    if not markdown:
        return 0.0, "empty"
    
    lines = markdown.strip().splitlines()
    if not lines:
        return 0.0, "no lines"
    
    # Count different types of content
    heading_count = 0
    paragraph_lines = 0
    link_lines = 0
    list_lines = 0
    
    for line in lines:
        stripped = line.strip()
        if not stripped:
            continue
        
        # Headings
        if re.match(r'^#{1,6}\s+', stripped):
            heading_count += 1
        # List items
        elif re.match(r'^[\*\-\+]\s+', stripped) or re.match(r'^\d+[\.\)]\s+', stripped):
            list_lines += 1
        # Lines that are mostly links
        elif re.search(r'\[.*?\]\(.*?\)', stripped):
            link_only = re.sub(r'\[.*?\]\(.*?\)', '', stripped).strip()
            if len(link_only) < 20:  # Line is mostly just a link
                link_lines += 1
            else:
                paragraph_lines += 1
        # Regular paragraph content
        elif len(stripped) > 20:
            paragraph_lines += 1
    
    # Calculate score
    score = 0.0
    
    # Headings: up to 0.25 (3+ headings = max)
    score += min(heading_count * 0.08, 0.25)
    
    # Paragraphs: up to 0.35 (10+ paragraphs = max)  
    score += min(paragraph_lines * 0.035, 0.35)
    
    # Content length bonus: up to 0.20
    if len(markdown) > 3000:
        score += 0.20
    elif len(markdown) > 1500:
        score += 0.15
    elif len(markdown) > 500:
        score += 0.10
    
    # Penalty for high link ratio (navigation-like content)
    total_content_lines = heading_count + paragraph_lines + list_lines
    if total_content_lines > 0:
        link_ratio = link_lines / (total_content_lines + link_lines)
        if link_ratio > 0.5:
            score -= 0.20
        elif link_ratio > 0.3:
            score -= 0.10
    
    # Bonus for mixed content (headings + paragraphs together)
    if heading_count >= 2 and paragraph_lines >= 5:
        score += 0.10
    
    score = max(0.0, min(score, 1.0))
    
    reason = f"h={heading_count}, p={paragraph_lines}, links={link_lines}, len={len(markdown)}"
    return score, reason


def find_content_start(markdown: str, title: str = "") -> str:
    """Find the start of main content, skipping navigation/menu.
    
    Strategies:
    1. Find first H1 or H2 heading that looks like content (not menu item)
    2. Match against page title
    3. Find first paragraph with substantial content
    
    Args:
        markdown: Raw markdown from crawler
        title: Page title for matching
        
    Returns:
        Markdown with navigation/menu skipped
    """
    if not markdown:
        return markdown
    
    lines = markdown.splitlines()
    if not lines:
        return markdown
    
    start_idx = 0
    
    # Normalize title for fuzzy matching
    norm_title = ""
    if title:
        norm_title = re.sub(r'[^\w\s]', '', title).lower().strip()
    
    # Strategy 1: Find meaningful heading (H1 or H2)
    for i, line in enumerate(lines[:150]):  # Check first 150 lines
        stripped = line.strip()
        
        # Skip empty lines
        if not stripped:
            continue
        
        # Check for H1/H2 heading
        heading_match = re.match(r'^(#{1,2})\s+(.+)$', stripped)
        if heading_match:
            heading_text = heading_match.group(2).strip()
            
            # Skip short headings that look like menu items
            if len(heading_text) < 5:
                continue
            
            # Check if it matches title (good sign it's content)
            if norm_title and len(norm_title) > 10:
                norm_heading = re.sub(r'[^\w\s]', '', heading_text).lower()
                if norm_title in norm_heading or norm_heading in norm_title:
                    start_idx = i
                    break
            
            # Check if heading looks like content (not navigation)
            nav_patterns = ['menu', 'home', 'about', 'contact', 'category', 'tag', 'archive']
            if not any(p in heading_text.lower() for p in nav_patterns):
                start_idx = i
                break
        
        # Strategy 2: Find substantial paragraph (not a link/list)
        if i > 50 and start_idx == 0:  # Only if no heading found in first 50 lines
            if len(stripped) > 100 and not stripped.startswith(('[', '*', '-', '#')):
                # Check it's not mostly links
                link_stripped = re.sub(r'\[.*?\]\(.*?\)', '', stripped)
                if len(link_stripped) > 50:
                    start_idx = max(0, i - 2)  # Start a bit before
                    break
    
    if start_idx > 0:
        logger.info(f"Skipping {start_idx} lines of navigation/menu content")
        return "\n".join(lines[start_idx:])
    
    return markdown


def extract_source_name(markdown: str, url: str) -> str:
    """Extract source name from markdown (title or first heading).
    
    Args:
        markdown: Markdown content
        url: URL for fallback domain name
        
    Returns:
        Source name string (max 255 chars)
    """
    lines = markdown.split("\n")
    for line in lines:
        line = line.strip()
        if not line:
            continue
        # Check for H1
        h1_match = re.match(r"^#\s+(.+)$", line)
        if h1_match:
            title = h1_match.group(1).strip()
            if title and len(title) > 5:
                return title[:255]  # Max length for name field

    # Fallback: use domain name
    try:
        parsed = urlparse(url)
        domain = parsed.netloc.replace("www.", "")
        return f"Web: {domain}"
    except Exception:
        return "Web Source"


__all__ = [
    "clean_markdown_content",
    "assess_markdown_quality",
    "find_content_start", 
    "extract_source_name",
]
