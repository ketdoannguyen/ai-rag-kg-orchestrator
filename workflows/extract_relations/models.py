"""Models for extract_relations workflow.

This module provides dataclasses and enums for the relation extraction pipeline:
- ExtractionMode: Mode enum (inline/batch/hybrid)
- ExtractionTier: Tier enum (content/structure/all)
- ExtractRelationsInput: Input dataclass
- ExtractRelationsOutput: Output dataclass
- ExtractionResult: Result from LLM extraction
"""

from __future__ import annotations

import uuid
from dataclasses import dataclass, field
from enum import Enum
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from arango_client.models import ContentNode


@dataclass
class NodeInfo:
    """Node information for relation extraction."""

    node_id: str
    node_type: str
    text: str
    embedding: list[float] | None = None


class ExtractionMode(str, Enum):
    """Relation extraction mode."""

    INLINE = "inline"  # Process during ingest workflow
    BATCH = "batch"  # Separate background job
    HYBRID = "hybrid"  # Inline for small, batch for large


class ExtractionTier(str, Enum):
    """Which tier to extract relations for."""

    CONTENT = "content"  # Tier 2: content_nodes only
    STRUCTURE = "structure"  # Tier 1: structure_nodes only
    ALL = "all"  # Both tiers


@dataclass
class ExtractRelationsInput:
    """Input for extract_relations workflow."""

    source_id: uuid.UUID | None = None  # DataSource ID to process (batch mode)
    nodes: list[ContentNode] | None = None  # Nodes to process (inline mode)
    mode: ExtractionMode = ExtractionMode.HYBRID
    tier: ExtractionTier = ExtractionTier.ALL  # Which tier(s) to process
    batch_size: int = 100
    max_llm_calls: int = 50  # Limit LLM calls per run
    dry_run: bool = False
    cross_source: bool = True  # Enable cross-source extraction


@dataclass
class ExtractRelationsOutput:
    """Output from extract_relations workflow."""

    # Content tier (Tier 2)
    content_nodes_processed: int = 0
    content_nodes_merged: int = 0
    content_relations_created: int = 0

    # Structure tier (Tier 1)
    structure_nodes_processed: int = 0
    structure_nodes_merged: int = 0
    structure_relations_created: int = 0

    # Combined stats
    nodes_processed: int = 0
    nodes_merged: int = 0
    relations_created: int = 0
    nodes_kept_separate: int = 0
    llm_calls_used: int = 0
    new_relations_proposed: int = 0  # New relations proposed by LLM
    new_relations_merged: int = 0  # New relations merged with existing
    new_relations_approved: int = 0  # New relations approved and saved
    errors: list[dict[str, Any]] = field(default_factory=list)

    # Per-tier detailed statistics
    tier_stats: dict[str, TierStats] = field(default_factory=dict)

    # Debug report path (if debug logging enabled)
    debug_report_path: str | None = None

    def update_totals(self) -> None:
        """Update combined totals from tier-specific stats."""
        self.nodes_processed = self.content_nodes_processed + self.structure_nodes_processed
        self.nodes_merged = self.content_nodes_merged + self.structure_nodes_merged
        self.relations_created = self.content_relations_created + self.structure_relations_created


@dataclass
class TierStats:
    """Detailed statistics for a single tier extraction.

    Tracks all metrics for one tier (Structure/Content):
    - Node processing counts
    - Merge statistics (auto-merge vs LLM-merge)
    - LLM extraction stats (pairs sent, top-k hits)
    - Relation creation breakdown (valid, fallback, new)
    """

    tier_name: str = ""

    # Node processing
    nodes_processed: int = 0

    # Merge stats
    auto_merged: int = 0  # embedding > 0.72 (EXTRACTION_AUTO_MERGE_THRESHOLD)
    llm_merged: int = 0  # LLM decided to merge

    # LLM extraction stats
    pairs_sent_to_llm: int = 0  # pairs sent to LLM (0.40-0.72 threshold)

    # Relation stats
    relations_in_top_k: int = 0  # relations in top-k ontology list
    relations_fallback: int = 0  # fallback to SIMILAR_TO due to errors
    relations_new: int = 0  # new relations proposed by LLM
    relations_total: int = 0  # total relations created

    # Other
    kept_separate: int = 0  # below 0.40 threshold
    errors: int = 0


@dataclass
class NodeProcessResult:
    """Internal result for single node processing."""

    merged: bool = False
    relation_created: bool = False
    kept_separate: bool = False
    skipped: bool = False  # True if node was already deleted/processed
    target_id: str | None = None  # ID of merge target (for tracking)
    relation_type: str | None = None  # Extracted relation type
    is_new_relation: bool = False  # True if LLM proposed a new relation
    is_in_top_k: bool = False  # True if relation was in top-k ontology list
    used_fallback: bool = False  # True if fallback relation was used
