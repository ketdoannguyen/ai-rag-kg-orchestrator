"""Centralized LLM prompt templates for RAG-KG workflows.

This package provides standardized, reusable prompts for:
- Extraction verification (merge/relation decisions)
- Enrichment (chunk merging, summarization)
- Relation extraction (ontology-aware relation identification)

Organization:
- extraction: Prompts for extract_relations workflow
- enrichment: Prompts for enrich_knowledge_tree workflow
- relation: Prompts for relation extraction and proposals
"""

from .enrichment import (
    build_detail_merge_prompt,
    build_structure_enrich_prompt,
)
from .extraction import (
    VERIFICATION_BATCH_PROMPT,
    VERIFICATION_PROMPT,
)
from .relation import (
    STRONG_VERIFY_AND_EXTRACT_PROMPT,
)

__all__ = [
    # Extraction
    "VERIFICATION_PROMPT",
    "VERIFICATION_BATCH_PROMPT",
    # Enrichment
    "build_detail_merge_prompt",
    "build_structure_enrich_prompt",
    # Relation
    "STRONG_VERIFY_AND_EXTRACT_PROMPT",
]
