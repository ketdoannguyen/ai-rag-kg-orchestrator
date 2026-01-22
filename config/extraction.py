"""Configuration for relation extraction workflow.

Settings for the extract_relations workflow:
- Embedding-based relation filtering (top-k)
- New relation validation (cosine threshold, LLM verification)
- Retry and fallback logic

All settings can be overridden via environment variables with EXTRACTION_ prefix.
"""

from __future__ import annotations

from functools import lru_cache

from pydantic import Field
from pydantic_settings import BaseSettings

# Constant for high-similarity structure pairs (used instead of merge)
STRUCTURALLY_EQUIVALENT_RELATION = "STRUCTURALLY_EQUIVALENT"


class ExtractionSettings(BaseSettings):
    """Configurable settings for extract_relations workflow.

    Relation Extraction Pipeline:
    1. Centroid Embedding: Average of source and target node embeddings
    2. Filter Top-K: Cosine similarity with relation embeddings
    3. LLM Extract: Choose from top-k or propose new relation
    4. Validate New: Cosine check + LLM strong verification

    Settings are organized by pipeline stage.
    """

    # =========================================================================
    # Embedding Filter Settings
    # =========================================================================
    EXTRACTION_TOP_K_RELATIONS: int = Field(
        default=15,
        description="Number of top relations to filter using cosine similarity.",
        ge=5,
        le=50,
    )

    # =========================================================================
    # New Relation Validation Settings
    # =========================================================================
    EXTRACTION_MERGE_THRESHOLD: float = Field(
        default=0.8,
        description="Cosine threshold to merge new relation with existing one. "
        "If cosine(new_embedding, existing_embedding) >= this, use existing relation.",
        ge=0.5,
        le=0.95,
    )

    EXTRACTION_STRONG_MODEL: str = Field(
        default="",  # Empty means use default from LLM_CHAT_MODEL
        description="Strong LLM model for validating new relations. "
        "Leave empty to use LLM_CHAT_MODEL from environment.",
    )

    EXTRACTION_MAX_RETRIES: int = Field(
        default=5,
        description="Max retries when LLM rejects proposed new relation.",
        ge=1,
        le=10,
    )

    EXTRACTION_FALLBACK_RELATION: str = Field(
        default="SIMILAR_TO",
        description="Fallback relation when all retries exhausted.",
    )

    # =========================================================================
    # LLM Settings
    # =========================================================================
    EXTRACTION_LLM_BATCH_SIZE: int = Field(
        default=5,
        description="Number of node pairs to process in single LLM call.",
        ge=1,
        le=10,
    )

    # =========================================================================
    # Threshold Settings (from deduplication, for compatibility)
    # =========================================================================
    EXTRACTION_AUTO_MERGE_THRESHOLD: float = Field(
        default=0.72,
        description="Node similarity >= this triggers auto-merge (no LLM needed).",
        ge=0.6,
        le=1.0,
    )

    EXTRACTION_LLM_THRESHOLD: float = Field(
        default=0.40,
        description="Node similarity >= this triggers LLM extraction. "
        "Below this, nodes are kept separate.",
        ge=0.2,
        le=0.95,
    )

    # =========================================================================
    # Processing Settings
    # =========================================================================
    EXTRACTION_BATCH_SIZE: int = Field(
        default=100,
        description="Number of nodes to process in each batch.",
        ge=10,
        le=500,
    )

    EXTRACTION_PARALLEL_WORKERS: int = Field(
        default=10,
        description="Number of parallel workers for async processing.",
        ge=1,
        le=50,
    )

    EXTRACTION_MAX_LLM_CALLS_PER_RUN: int = Field(
        default=999999,
        description="Maximum LLM calls per extraction run. Set to high value for accuracy-first mode.",
        ge=1,
        le=999999,
    )

    EXTRACTION_MAX_CANDIDATES_PER_NODE: int = Field(
        default=10,
        description="Maximum similar nodes to consider for each new node.",
        ge=1,
        le=50,
    )

    # =========================================================================
    # Multi-Node Processing Settings
    # =========================================================================
    EXTRACTION_MAX_RELATIONS_PER_LEADER: int = Field(
        default=10,
        description="Maximum relations to create per leader node. "
        "Limits explosion for highly connected nodes.",
        ge=1,
        le=50,
    )

    EXTRACTION_SIMILARITY_DECAY_FACTOR: float = Field(
        default=0.85,
        description="Stop processing similar nodes when similarity drops below "
        "best_match * this factor. 0.85 means stop at 85% of best.",
        ge=0.5,
        le=1.0,
    )

    EXTRACTION_MAX_CONCURRENT_COLLECTIONS: int = Field(
        default=5,
        description="Max concurrent similarity searches in cross-batch processing.",
        ge=1,
        le=20,
    )

    # =========================================================================
    # Feature Flags
    # =========================================================================
    EXTRACTION_USE_TWO_PHASE: bool = Field(
        default=True,
        description="Use Two-Phase extraction to catch intra-batch duplicates. "
        "Phase 1: Similarity matrix + Union-Find for batch dedup. "
        "Phase 2: Cross-batch check against reviewed nodes.",
    )

    EXTRACTION_ALLOW_NEW_RELATIONS: bool = Field(
        default=True,
        description="Allow LLM to propose new relation types outside ontology.",
    )

    EXTRACTION_CROSS_SOURCE: bool = Field(
        default=True,
        description="Enable cross-source relation extraction.",
    )

    EXTRACTION_DRY_RUN: bool = Field(
        default=False,
        description="If True, log decisions without making changes.",
    )

    # =========================================================================
    # Embedding Configuration
    # =========================================================================
    EXTRACTION_EMBEDDING_MODEL: str = Field(
        default="openrouter/openai/text-embedding-3-large",
        description="Embedding model for similarity search.",
    )

    EXTRACTION_EMBEDDING_DIMENSIONS: int = Field(
        default=3072,
        description="Embedding dimensions. text-embedding-3-large uses 3072.",
        ge=256,
        le=4096,
    )

    class Config:
        env_file = ".env"
        env_file_encoding = "utf-8"
        extra = "ignore"


@lru_cache
def get_extraction_settings() -> ExtractionSettings:
    """Get cached extraction settings."""
    return ExtractionSettings()
