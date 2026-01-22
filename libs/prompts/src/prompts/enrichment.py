"""Enrichment prompts for knowledge tree workflows.

Prompt builders for:
- Detail merging: Combine semantically related items into RAG chunks
- Structure enrichment: Generate summaries for structure nodes
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from config.domains import DomainSchema


def build_detail_merge_prompt(
    items: list[dict[str, Any]],
    breadcrumb: list[str],
    domain_schema: DomainSchema | None = None,
    max_chunk_size: int = 2000,
) -> str:
    """Build prompt for merging detail items into RAG chunks.

    Args:
        items: List of text items to merge
        breadcrumb: Section hierarchy for context
        domain_schema: Optional domain schema for metadata extraction
        max_chunk_size: Maximum characters per chunk
    """
    context = " > ".join(breadcrumb)
    lines: list[str] = []
    for idx, item in enumerate(items, start=1):
        prefix = item.get("type", "paragraph").upper()
        text = (item.get("text") or "").strip()
        lines.append(f"[{idx}] ({prefix}) {text}")

    content = "\n".join(lines)

    # Build filter extraction section if schema provided
    filter_section = ""
    filter_output = ""
    if domain_schema:
        filter_section = f"""
FILTER EXTRACTION (per segment - for RAG retrieval):
Domain: {domain_schema.display_name}

{domain_schema.to_extraction_prompt_section()}

EXTRACTION RULES (CRITICAL):
- ONLY extract if the value is EXPLICITLY mentioned in the text
- Use EXACTLY the allowed values listed above (normalize using provided aliases)
- If uncertain or value not found, use null - DO NOT guess or infer
- Extract filter for the MERGED segment, not individual source items
- ALL filter values MUST be arrays/lists (even single values)
- These filters enable precise RAG retrieval queries

Examples:
- Text "năm 2024" → {{"year": [2024]}}
- Text "năm 2024 và 2025" → {{"year": [2024, 2025]}}
- Text "ngành CNTT" → {{"major_name": ["Công nghệ thông tin"]}}
- Text mentions "xét tuyển học bạ và THPT" → {{"admission_method": ["Học bạ", "THPT"]}}
- Text mentions nothing identifiable → null or {{}}
"""
        filter_output = ',\n      "filter": {"field": [values]} or null'

    return f"""You are an expert in text optimization for RAG retrieval.

Context: {context}
Input (numbered starting from 1):
{content}

PRIMARY TASK: MAXIMALLY MERGE semantically related items into independent segments.
SECONDARY TASK: Provide ONE overall summary max 200 characters for the entire section.
{filter_section}
MERGING PRINCIPLES (CRITICAL - READ CAREFULLY):
- HARD LIMIT: {max_chunk_size} characters per segment. This is ABSOLUTE, NO EXCEPTIONS.
- If merged text would exceed {max_chunk_size} chars, you MUST:
  a) Split into multiple segments, OR
  b) Keep items separate (do NOT merge)
- GOAL: Merge semantically related texts to APPROACH (but never exceed) {max_chunk_size} chars.
- MERGE items that belong to the same idea, explanation, or example chain.
- NEVER merge texts with DIFFERENT topics/subjects just to reach max size.
- Short segments (100-300 chars) are PERFECTLY ACCEPTABLE if standalone.
- BEFORE outputting each segment, COUNT the characters. If > {max_chunk_size}, FIX IT.

CONSTRAINTS ON TEXT:
1. DO NOT translate or paraphrase; keep original language and wording.
2. When merging, join sentences VERBATIM (preserve punctuation, casing, accents).
3. If extra context is needed, reuse headings/text from the input, copied exactly.
4. For each segment, also produce a short (<= 8 words) summary in the SAME language.
5. Include estimated_char_count for each segment (MUST be <= {max_chunk_size}).

OUTPUT FORMAT: Pure JSON, no markdown, no explanation:
{{
  "segments": [
    {{
      "source_indices": [1, 2],
      "text": "Exact original sentences merged verbatim...",
      "summary": "<= 8 words, same language",
      "estimated_char_count": 450{filter_output}
    }}
  ],
  "section_summary": "High-level summary for the entire section (same language) max 500 characters"
}}

If nothing should be merged, output one segment per original item.
"""


def build_structure_enrich_prompt(
    title: str,
    children_content: list[dict[str, Any]],
    summary_style: str = "informative",
    short: bool = False,
) -> str:
    """Build prompt for enriching a structure node with a summary.

    Args:
        title: Section title
        children_content: List of child content dicts
        summary_style: Style hint (e.g., 'informative', 'concise')
        short: Whether to generate a short summary
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

    return f"""You are an expert summarizer for knowledge trees.

SECTION TITLE: {title}

CONTENT:
{joined}

Task: Write a {"short" if short else "comprehensive"} {summary_style} summary for "{title}".

Requirements:
- Same language as input content
- Concise, focused on key ideas
- Preserve key terminology

Output JSON only:
{{ "summary": "Summary text here... max 500 characters" }}
"""
