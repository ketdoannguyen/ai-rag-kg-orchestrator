"""Prompt builders for enrich_knowledge_tree workflow."""

from __future__ import annotations

from typing import Any, Dict, List

from workflows.enrich_knowledge_tree.models import EnrichConfig


def build_detail_merge_prompt(items: List[Dict[str, Any]], breadcrumb: List[str]) -> str:
    """Single combined prompt for merging detail items into RAG chunks.

    Based on the original refinement prompt, but extended to:
    - preserve original language
    - optionally attach a short (<=10 words) summary for each merged block.
    """
    context = " > ".join(breadcrumb)
    lines: List[str] = []
    for idx, item in enumerate(items, start=1):
        prefix = item.get("type", "paragraph").upper()
        text = (item.get("text") or "").strip()
        lines.append(f"[{idx}] ({prefix}) {text}")

    content = "\n".join(lines)

    return (
        "You are an expert in text optimization for RAG retrieval.\n\n"
        f"Context: {context}\n"
        f"Input (numbered starting from 1):\n{content}\n\n"
        "PRIMARY TASK: MAXIMALLY MERGE semantically related items into independent segments.\n"
        "SECONDARY TASK: Provide ONE overall summary max 200 characters for the entire section.\n\n"
        "MERGING PRINCIPLES (VERY IMPORTANT):\n"
        "- Merge items that belong to the same idea, explanation, or example chain.\n"
        "- Prefer segments around 200–300 characters; avoid extremely short fragments.\n"
        "- Only keep a segment < 100 characters if it truly cannot be merged.\n\n"
        "CONSTRAINTS ON TEXT:\n"
        "1. DO NOT translate or paraphrase; keep original language and wording.\n"
        "2. When merging, join sentences VERBATIM (preserve punctuation, casing, accents).\n"
        "3. If extra context is needed, reuse headings/text from the input, copied exactly.\n"
        "4. For each segment, also produce a short (<= 8 words) summary in the SAME language.\n\n"
        "OUTPUT FORMAT: Pure JSON, no markdown, no explanation:\n"
        "{\n"
        '  \"segments\": [\n'
        "    {\n"
        "      \"source_indices\": [1, 2],\n"
        "      \"text\": \"Exact original sentences merged verbatim...\",\n"
        "      \"summary\": \"<= 8 words, same language\"\n"
        "    }\n"
        "  ],\n"
        "  \"section_summary\": \"High-level summary for the entire section (same language) max 200 characters\"\n"
        "}\n"
        "If you decide that nothing should be merged, output one segment per original item\n"
        "with its own index in source_indices and the same text.\n"
    )


def build_structure_enrich_prompt(
    title: str,
    children_content: List[Dict[str, Any]],
    config: EnrichConfig,
    short: bool = False,
) -> str:
    """Single combined prompt for enriching a structure node with a summary."""
    style = config.summary_style or "informative"
    max_len = config.max_summary_length

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

    return (
        "You are an expert summarizer creating structured section summaries for a knowledge tree.\n"
        "Follow the instructions and respect all constraints below.\n\n"
        f"SECTION TITLE: {title}\n\n"
        "CONTENT BELOW COMES FROM THIS SECTION AND ITS CHILDREN:\n"
        f"{joined}\n\n"
        f"Task: Write a {'short' if short else 'comprehensive'} {style} summary for the section "
        f"\"{title}\".\n"
        "Requirements:\n"
        "- The summary MUST be written in the same language as the input content above.\n"
        "- Keep the summary concise and focused on key ideas only.\n"
        "- Preserve the key terminology; do not add external facts.\n\n"
        "- The summary MUST be max 200 characters.\n"
        "Output JSON only with this structure:\n"
        '{ \"summary\": \"Summary text here... max 200 characters\" }\n'
    )

