"""Enrichment prompts for knowledge tree workflows.

Prompt builders for:
- Detail merging: Combine semantically related items into RAG chunks
- Structure enrichment: Generate summaries for structure nodes
"""

from __future__ import annotations

from typing import Any


def build_detail_merge_prompt(
    items: list[dict[str, Any]],
    breadcrumb: list[str],
    max_chunk_size: int = 2000,
) -> str:
    """Build prompt for merging detail items into RAG chunks.

    Args:
        items: List of text items to merge
        breadcrumb: Section hierarchy for context
        max_chunk_size: Maximum characters per chunk
    """
    context = " > ".join(breadcrumb)
    lines: list[str] = []
    for idx, item in enumerate(items, start=1):
        prefix = item.get("type", "paragraph").upper()
        text = (item.get("text") or "").strip()
        lines.append(f"[{idx}] ({prefix}) {text}")

    content = "\n".join(lines)

    # Calculate minimum chunk size (40% of max to ensure meaningful chunks)
    min_chunk_size = int(max_chunk_size * 0.4)

    return f"""You are an expert in text optimization for RAG retrieval.

PRIMARY TASK: Merge semantically related items into SELF-CONTAINED segments.
SECONDARY TASK: Report excluded items and provide section summary.

=== SIZE CONSTRAINTS ===
- MINIMUM: {min_chunk_size} chars. If below, MERGE with related content.
- MAXIMUM: {max_chunk_size} chars. NEVER exceed.
- IDEAL: {int(max_chunk_size * 0.6)}-{int(max_chunk_size * 0.9)} chars.

=== EXCLUSION RULES ===
EXCLUDE these items (put in excluded_indices with reason):
- "image_ref": Items starting with "Hình", "Figure", "Bảng", "Table", "Sơ đồ", "Biểu đồ"
- "question": Standalone questions starting with "Nêu...", "Trình bày...", "Phân tích...", "Giải thích...", "Tại sao...", "Vì sao..."

=== TIMELINE DATA ===
If you see standalone dates (e.g., "12-6-1941", "Từ 28-11 đến 1-12-1943"):
- These are likely from a timeline/table extraction.
- PAIR each date with its corresponding event text.
- If no matching event, EXCLUDE with reason "orphan_date".

=== MERGING RULES ===
- MERGE: Items about SAME topic/event + intro sentences with their list items.
- DO NOT MERGE: Different topics just to reach max size.
- Each segment MUST be self-contained (no "as mentioned above").

=== TEXT RULES ===
- Keep original wording VERBATIM.
- Summary: <= 8 words, same language.

=== OUTPUT FORMAT ===
Pure JSON only:
{{
  "segments": [
    {{"source_indices": [1, 2], "text": "...", "summary": "...", "estimated_char_count": 650}}
  ],
  "excluded_indices": [
    {{"index": 13, "reason": "image_ref"}}
  ],
  "section_summary": "max 500 chars"
}}

=== COVERAGE CHECK ===
Every input index (1 to N) MUST appear in EITHER segments.source_indices OR excluded_indices.

Context: {context}
Input (numbered starting from 1):
{content}
"""


def build_structure_enrich_prompt(
    title: str,
    children_content: list[dict[str, Any]],
    summary_style: str = "informative",
    short: bool = False,
    breadcrumb: list[str] | None = None,
) -> str:
    """Build prompt for enriching a structure node with a RAG-optimized summary.

    Args:
        title: Section title
        children_content: List of child content dicts
        summary_style: Style hint (e.g., 'informative', 'concise')
        short: Whether to generate a short summary
        breadcrumb: Hierarchical path to this section for context
    """
    blocks: list[str] = []
    for idx, item in enumerate(children_content, start=1):
        t = item.get("type", "paragraph")
        block_title = item.get("title") or ""
        content = (item.get("content") or "").strip()
        header = f"[{idx}] ({t.upper()})"
        if block_title:
            header += f" - {block_title}"
        blocks.append(f"{header}\n{content}")

    joined = "\n\n".join(blocks)

    context_line = ""
    if breadcrumb:
        context_line = f"CONTEXT PATH: {' > '.join(breadcrumb)}\n\n"

    char_limit = 300 if short else 500

    return f"""You are an expert summarizer optimizing content for RAG retrieval.

{context_line}SECTION: {title}

CONTENT:
{joined}

=== RAG-OPTIMIZED SUMMARY TASK ===

Write a {summary_style} summary that enables effective semantic search retrieval.

CRITICAL REQUIREMENTS:
1. BE SPECIFIC - Include actual names, dates, numbers, not just references
   - BAD: "Trình bày ba trụ cột của ASEAN..."
   - GOOD: "Ba trụ cột ASEAN gồm: (1) Chính trị-An ninh, (2) Kinh tế, (3) Văn hóa-Xã hội"

2. PRESERVE ALL ENTITIES - Names, organizations, years, statistics
   - BAD: "Tổ chức thành lập vào thập niên 60..."
   - GOOD: "ASEAN thành lập ngày 8-8-1967 bởi 5 nước: Indonesia, Malaysia, Philippines, Singapore, Thái Lan"

3. SELF-CONTAINED - Summary must be understandable without reading original content

4. ANSWER IMPLICIT QUESTIONS - What? When? Who? Where? Why? How?

5. SAME LANGUAGE as input content

OUTPUT FORMAT (JSON only):
{{ "summary": "RAG-optimized summary, max {char_limit} chars" }}
"""
