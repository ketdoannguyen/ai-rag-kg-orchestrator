"""Prompt builders for enrich_knowledge_tree workflow."""

from __future__ import annotations

from typing import Any, Dict, List, Optional, TYPE_CHECKING

if TYPE_CHECKING:
    from config.domains import DomainSchema


def build_detail_merge_prompt(
    items: List[Dict[str, Any]],
    breadcrumb: List[str],
    domain_schema: Optional["DomainSchema"] = None,
) -> str:
    """Build prompt for merging detail items into RAG chunks.
    
    Args:
        items: List of text items to merge
        breadcrumb: Section hierarchy for context
        domain_schema: Optional domain schema for metadata extraction
    """
    context = " > ".join(breadcrumb)
    lines: List[str] = []
    for idx, item in enumerate(items, start=1):
        prefix = item.get("type", "paragraph").upper()
        text = (item.get("text") or "").strip()
        lines.append(f"[{idx}] ({prefix}) {text}")

    content = "\n".join(lines)
    
    # Build metadata extraction section if schema provided
    metadata_section = ""
    metadata_output = ""
    if domain_schema:
        metadata_section = f"""
METADATA EXTRACTION (per segment):
{domain_schema.to_extraction_prompt_section()}

For each segment, extract domain metadata if identifiable from the text.
Only include fields you can confidently extract. Use null if not found.
"""
        metadata_output = ',\n      "metadata": {"field": "value"} or null'

    return f"""You are an expert in text optimization for RAG retrieval.

Context: {context}
Input (numbered starting from 1):
{content}

PRIMARY TASK: MAXIMALLY MERGE semantically related items into independent segments.
SECONDARY TASK: Provide ONE overall summary max 200 characters for the entire section.
{metadata_section}
MERGING PRINCIPLES:
- Merge items that belong to the same idea, explanation, or example chain.
- Prefer segments around 200–300 characters; avoid extremely short fragments.
- Only keep a segment < 100 characters if it truly cannot be merged.

CONSTRAINTS ON TEXT:
1. DO NOT translate or paraphrase; keep original language and wording.
2. When merging, join sentences VERBATIM (preserve punctuation, casing, accents).
3. If extra context is needed, reuse headings/text from the input, copied exactly.
4. For each segment, also produce a short (<= 8 words) summary in the SAME language.

OUTPUT FORMAT: Pure JSON, no markdown, no explanation:
{{
  "segments": [
    {{
      "source_indices": [1, 2],
      "text": "Exact original sentences merged verbatim...",
      "summary": "<= 8 words, same language"{metadata_output}
    }}
  ],
  "section_summary": "High-level summary for the entire section (same language) max 200 characters"
}}

If nothing should be merged, output one segment per original item.
"""


def build_structure_enrich_prompt(
    title: str,
    children_content: List[Dict[str, Any]],
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
    blocks: List[str] = []
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

Task: Write a {'short' if short else 'comprehensive'} {summary_style} summary for "{title}".

Requirements:
- Same language as input content
- Concise, focused on key ideas
- Preserve key terminology
- Max 200 characters

Output JSON only:
{{ "summary": "Summary text here... max 200 characters" }}
"""
