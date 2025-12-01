from typing import List
from postgres_client.models import ParagraphNode

def build_content_merge_prompt(buffer: List[ParagraphNode], breadcrumb: List[str]) -> str:
    content = "\n".join([f"[{i+1}] {p.attributes.text}" for i, p in enumerate(buffer)])
    return (
        f"Context: {' > '.join(breadcrumb)}\n"
        f"Input:\n{content}\n\n"
        "Task: Merge and rewrite paragraphs into self-contained segments optimized for RAG.\n"
        "Rules:\n"
        "1. Merge fragmented sentences or paragraphs that form a single semantic unit.\n"
        "2. CONTEXTUALIZE: If a segment relies on a previous header/topic (e.g., list items), PREPEND that context to make it self-contained.\n"
        "   Example: If [1] is 'Goals' and [2] is '- Win', output 'Goals: - Win'.\n"
        "3. Preserve all original information. Do not summarize.\n"
        "4. Output JSON: { \"segments\": [ { \"source_indices\": [1, 2], \"content\": \"Contextualized content...\" } ] }"
    )


def get_system_prompt() -> str:
    return "You are an expert content editor optimizing text for RAG retrieval. Output valid JSON only."
