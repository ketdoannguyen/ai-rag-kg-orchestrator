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
        default=0.6,
        description="Cosine threshold to merge new relation with existing one. "
        "If cosine(new_embedding, existing_embedding) >= this, use existing relation.",
        ge=0.5,
        le=0.95,
    )

    EXTRACTION_WEAK_MODEL: str = Field(
        default="openai/qwen/qwen3-235b-a22b-2507",
        description="Weak LLM model for initial relation extraction. Fast and cheap.",
    )

    EXTRACTION_STRONG_MODEL: str = Field(
        default="openai/qwen/qwen3.5-397b-a17b",
        description="Strong LLM model for validating new relations. Use LiteLLM format: openai/<provider>/<model> for OpenRouter.",
    )

    EXTRACTION_STRONG_MODEL_PROVIDER: str = Field(
        default="atlas-cloud/fp8",
        description="Provider for the strong LLM model.",
    )

    EXTRACTION_FALLBACK_RELATION: str = Field(
        default="SIMILAR_TO",
        description="Fallback relation when all retries exhausted.",
    )

    # =========================================================================
    # LLM Settings
    # =========================================================================
    EXTRACTION_LLM_BATCH_SIZE: int = Field(
        default=15,
        description="Number of node pairs to process in single LLM call. "
        "Increased from 8 to 15 for fewer API calls and better token efficiency.",
        ge=1,
        le=30,
    )

    # =========================================================================
    # Parallel Processing Settings
    # =========================================================================
    MAX_CONCURRENT_LLM_CALLS: int = Field(
        default=15,
        description="Maximum concurrent LLM API calls for parallel batch processing. "
        "Increase for faster extraction (uses more RAM and API rate limit).",
        ge=1,
        le=20,
    )

    # =========================================================================
    # Threshold Settings (from deduplication, for compatibility)
    # =========================================================================
    EXTRACTION_AUTO_MERGE_THRESHOLD: float = Field(
        default=0.78,
        description="Node similarity >= this triggers auto-merge (no LLM needed).",
        ge=0.6,
        le=1.0,
    )

    EXTRACTION_LLM_THRESHOLD: float = Field(
        default=0.40,
        description="Node similarity >= this triggers LLM extraction. "
        "Below this, nodes are kept separate. "
        "Raised from 0.35 based on benchmark: all samples in 0.35-0.40 are true negatives.",
        ge=0.2,
        le=0.95,
    )

    # =========================================================================
    # Processing Settings
    # =========================================================================
    EXTRACTION_BATCH_SIZE: int = Field(
        default=200,
        description="Number of nodes to process in each batch.",
        ge=10,
        le=500,
    )

    EXTRACTION_MAX_CONCURRENT_COLLECTIONS: int = Field(
        default=10,
        description="Max concurrent similarity searches in cross-batch processing.",
        ge=1,
        le=20,
    )

    # =========================================================================
    # Feature Flags
    # =========================================================================
    EXTRACTION_CROSS_SOURCE: bool = Field(
        default=True,
        description="Enable cross-source relation extraction.",
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

    # =========================================================================
    # Family Inference Settings
    # =========================================================================
    EXTRACTION_FAMILY_INFERENCE_THRESHOLD: float = Field(
        default=0.3,
        description="Low threshold for finding similar relations when inferring families. "
        "Used to find any related relation for family inheritance.",
        ge=0.1,
        le=0.6,
    )

    # =========================================================================
    # Debug Logger Settings
    # =========================================================================
    EXTRACTION_DEBUG_STORE_SIMILARITY_MATRIX: bool = Field(
        default=True,
        description="Store similarity matrix in debug logs. "
        "Disable for large batches to reduce memory usage.",
    )

    EXTRACTION_DEBUG_MAX_MATRIX_SIZE: int = Field(
        default=100,
        description="Maximum matrix size to store. Matrices larger than this are skipped.",
        ge=10,
        le=500,
    )

    class Config:
        env_file = ".env"
        env_file_encoding = "utf-8"
        extra = "ignore"


@lru_cache
def get_extraction_settings() -> ExtractionSettings:
    """Get cached extraction settings."""
    return ExtractionSettings()
