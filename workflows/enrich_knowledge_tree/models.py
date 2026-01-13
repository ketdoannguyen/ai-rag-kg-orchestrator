"""Input and output models for enrich_knowledge_tree workflow."""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

from arango_client import StructureNode
from pydantic import BaseModel, Field

if TYPE_CHECKING:
    pass


class RefineConfig(BaseModel):
    """Configuration for detail-level refinement."""

    min_paragraph_length: int = Field(default=100, ge=0)
    max_paragraph_length: int = Field(default=800, ge=1)
    batch_size: int = Field(default=3, ge=1, le=100)
    llm_model: str = Field(default="openai/qwen/qwen3-next-80b-a3b-instruct")
    llm_temperature: float = Field(default=0.0, ge=0.0, le=2.0)
    preserve_structure: bool = Field(
        default=True,
        description="Whether to keep structure/ordering of children when rebuilding.",
    )
    max_concurrent_sections: int = Field(
        default=20,
        ge=1,
        le=128,
        description="Global concurrency limit for section-level processing/LLM calls.",
    )


class EnrichConfig(BaseModel):
    """Configuration for structure-level enrichment (summaries)."""

    max_summary_length: int = Field(default=300, ge=50, le=600)
    summary_style: str = Field(
        default="informative",
        description="High-level style hint for summaries (e.g., 'informative', 'concise').",
    )
    llm_model: str = Field(default="openai/qwen/qwen3-next-80b-a3b-instruct")
    llm_temperature: float = Field(default=0.0, ge=0.0, le=2.0)


class EnrichKnowledgeTreeInput(BaseModel):
    """Input model for enrich_knowledge_tree workflow."""

    model_config = {"arbitrary_types_allowed": True}

    knowledge_tree: StructureNode = Field(
        ...,
        description="Root StructureNode of the tree to enrich.",
    )
    refine_config: RefineConfig = Field(
        default_factory=RefineConfig,
        description="Configuration for detail refinement.",
    )
    enrich_config: EnrichConfig = Field(
        default_factory=EnrichConfig,
        description="Configuration for structure enrichment.",
    )
    domain_schema: Any | None = Field(
        default=None,
        description="DomainSchema for metadata extraction. If None, no metadata extracted.",
    )


class EnrichStatistics(BaseModel):
    """Aggregated statistics about refinement + enrichment."""

    paragraphs_before: int = 0
    paragraphs_after: int = 0
    structures_enriched: int = 0
    detail_llm_calls: int = 0
    structure_llm_calls: int = 0


class ChangesSummary(BaseModel):
    """High-level summary of changes applied to the tree."""

    refined_sections: dict[str, dict[str, Any]] = Field(
        default_factory=dict,
        description=(
            "Per-structure-node summary keyed by node.id: "
            "{'title', 'original_detail_count', 'refined_detail_count'}."
        ),
    )
    root_summary_preview: str | None = Field(
        default=None,
        description="Short preview of the root summary (first ~300 chars).",
    )


class EnrichKnowledgeTreeOutput(BaseModel):
    """Output model for enrich_knowledge_tree workflow."""

    enriched_tree: StructureNode = Field(
        ...,
        description="Root node of the enriched tree.",
    )
    statistics: EnrichStatistics = Field(
        ...,
        description="Detailed statistics of the operation.",
    )
    changes_summary: ChangesSummary = Field(
        ...,
        description="Compact summary of structural/content changes.",
    )
