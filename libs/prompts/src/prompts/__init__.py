"""Centralized LLM prompt templates for RAG-KG workflows.

This package provides standardized, reusable prompts for:
- Extraction verification (merge/relation decisions)
- Enrichment (chunk merging, summarization)
- Routing (query understanding, filter extraction)
- Relation extraction (ontology-aware relation identification)

Organization:
- extraction: Prompts for extract_relations workflow
- enrichment: Prompts for enrich_knowledge_tree workflow
- routing: Prompts for Router Agent
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
    RELATION_INSTANCE_VALIDATION_PROMPT,
    RELATION_PROPOSAL_PROMPT,
    RELATION_REEXTRACTION_PROMPT,
    RELATION_VALIDATION_PROMPT,
)
from .routing import (
    build_router_system_prompt,
    build_router_user_prompt,
)

__all__ = [
    # Extraction
    "VERIFICATION_PROMPT",
    "VERIFICATION_BATCH_PROMPT",
    # Enrichment
    "build_detail_merge_prompt",
    "build_structure_enrich_prompt",
    # Routing
    "build_router_system_prompt",
    "build_router_user_prompt",
    # Relation
    "RELATION_PROPOSAL_PROMPT",
    "RELATION_VALIDATION_PROMPT",
    "RELATION_INSTANCE_VALIDATION_PROMPT",
    "RELATION_REEXTRACTION_PROMPT",
]
