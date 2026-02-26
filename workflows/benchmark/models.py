"""Pydantic models for Benchmark Dataset v4.0 — Subgraph-First pipeline."""

from __future__ import annotations

import uuid
from datetime import UTC, datetime
from typing import Literal

from pydantic import BaseModel, ConfigDict, field_validator, model_validator

TIER2_TYPES: list[str] = [
    "CAUSED_BY",
    "LED_TO",
    "RESULTED_IN",
    "CONTRASTS",
    "ELABORATES",
    "CONTEXTUALIZES",
    "BUILDS_UPON",
    "REFERS_TO",
    "CONTRIBUTED_TO",
    "CONSEQUENCE_OF",
    "SUPPORTS",
    "CONTRADICTS",
    "PRECEDED_BY",
    "FOLLOWED_BY",
    "DURING",
    "CONTEMPORARY_WITH",
    "LED_BY",
    "PARTICIPATED_IN",
    "OPPOSED_BY",
    "IS_A",
    "PART_OF",
    "SIMILAR_TO",
    "PREREQUISITE_OF",
    "CONTINUES",
    "INTRODUCES",
    "CONCLUDES",
    "VISUALIZES",
    "DEFINES",
    "EXEMPLIFIES",
    "SUMMARIZES",
    "DUPLICATE_OF",
    "SUPERSEDES",
]


class FactNode(BaseModel):
    """A content node with relevance grade for a benchmark query."""

    node_id: str
    grade: Literal[0, 1, 2]
    parent_section: str

    @field_validator("node_id")
    @classmethod
    def validate_node_id(cls, v: str) -> str:
        if not v or "/" not in v:
            raise ValueError(
                f"node_id must be a full ArangoDB document ID (collection/key), got: {v!r}"
            )
        return v


class ConfidenceScore(BaseModel):
    """LLM self-assessed confidence for a generated benchmark query."""

    model_config = ConfigDict(extra="forbid")

    overall: int

    @field_validator("overall")
    @classmethod
    def validate_overall(cls, v: int) -> int:
        if not 0 <= v <= 100:
            raise ValueError(f"Overall confidence must be 0-100, got {v}")
        return v


class BenchmarkQuery(BaseModel):
    """A single benchmark query with graded fact nodes and metadata."""

    model_config = ConfigDict(extra="forbid")

    query_id: str
    query_text: str
    subgraph_id: str
    fact_nodes: list[FactNode]
    relations_used: list[str]
    confidence: ConfidenceScore
    reasoning: str
    # Post-hoc fields (filled after generation)
    direction: str | None = None
    calibrated_confidence: float | None = None
    human_review_status: str | None = None

    @classmethod
    def with_generated_id(cls, **data) -> BenchmarkQuery:
        """Create a BenchmarkQuery with an auto-generated UUID."""
        if "query_id" not in data:
            data["query_id"] = str(uuid.uuid4())
        return cls(**data)

    @field_validator("query_text")
    @classmethod
    def validate_query_text(cls, v: str) -> str:
        if not v or not v.strip():
            raise ValueError("query_text cannot be empty")
        return v.strip()

    @field_validator("fact_nodes")
    @classmethod
    def validate_fact_nodes(cls, v: list[FactNode]) -> list[FactNode]:
        if not v:
            raise ValueError("fact_nodes must contain at least one node")
        return v

    @field_validator("direction")
    @classmethod
    def validate_direction(cls, v: str | None) -> str | None:
        if v is not None and v not in ("TD", "HZ", "BU", "NT", "CS", "AMBIGUOUS"):
            raise ValueError(f"Invalid direction: {v!r}")
        return v

    @field_validator("human_review_status")
    @classmethod
    def validate_review_status(cls, v: str | None) -> str | None:
        if v is not None and v not in ("approved", "rejected", "pending"):
            raise ValueError(f"Invalid human_review_status: {v!r}")
        return v

    @field_validator("calibrated_confidence")
    @classmethod
    def validate_calibrated(cls, v: float | None) -> float | None:
        if v is not None and not 0.0 <= v <= 1.0:
            raise ValueError(f"calibrated_confidence must be 0-1, got {v}")
        return v


class SubgraphNode(BaseModel):
    """A content node within a subgraph."""

    node_id: str
    text: str
    embedding: list[float] | None = None
    parent_section_key: str | None = None
    parent_section_title: str | None = None
    parent_section_level: int | None = None
    breadcrumb: str | None = None

    @field_validator("node_id")
    @classmethod
    def validate_node_id(cls, v: str) -> str:
        if not v or "/" not in v:
            raise ValueError(f"node_id must be a full ArangoDB document ID, got: {v!r}")
        return v


class SubgraphRelation(BaseModel):
    """A Tier 2 relation between two nodes in a subgraph."""

    from_id: str
    to_id: str
    relation_type: str

    @model_validator(mode="after")
    def validate_self_loop(self) -> SubgraphRelation:
        if self.from_id == self.to_id:
            raise ValueError(f"Self-loop relation not allowed: {self.from_id}")
        return self


class SubgraphStructureNode(BaseModel):
    """A structure_node (L5/L6) that organises content within a subgraph."""

    section_key: str
    title: str | None = None
    summary: str | None = None
    level: int
    parent_key: str | None = None  # For L6: the parent L5 key


class Subgraph(BaseModel):
    """A v4.0 subgraph assembled via SEED+ENRICH+BOUND."""

    subgraph_id: str
    anchor_key: str
    anchor_title: str
    anchor_summary: str | None = None
    breadcrumb: str
    nodes: list[SubgraphNode]
    core_node_ids: list[str]
    enriched_node_ids: list[str]
    relations: list[SubgraphRelation]
    structure_nodes: list[SubgraphStructureNode] = []
    content_count: int
    relation_count: int
    tier: Literal["large", "medium", "small", "tiny"]
    query_allocation: int

    @field_validator("nodes")
    @classmethod
    def validate_nodes(cls, v: list[SubgraphNode]) -> list[SubgraphNode]:
        if not v:
            raise ValueError("Subgraph must have at least one node")
        return v

    @field_validator("content_count")
    @classmethod
    def validate_content_count(cls, v: int) -> int:
        if v < 0:
            raise ValueError(f"content_count must be >= 0, got {v}")
        return v

    @field_validator("query_allocation")
    @classmethod
    def validate_query_allocation(cls, v: int) -> int:
        if v < 0:
            raise ValueError(f"query_allocation must be >= 0, got {v}")
        return v

    @model_validator(mode="after")
    def validate_node_ids_consistency(self) -> Subgraph:
        all_node_ids = {n.node_id for n in self.nodes}
        for nid in self.core_node_ids:
            if nid not in all_node_ids:
                raise ValueError(f"core_node_id {nid!r} not found in nodes list")
        for nid in self.enriched_node_ids:
            if nid not in all_node_ids:
                raise ValueError(f"enriched_node_id {nid!r} not found in nodes list")
        return self


class EvaluationResult(BaseModel):
    """Result of running one retrieval method on one query."""

    query_id: str
    method_id: str
    method_name: str
    retrieved_ids: list[str]
    ndcg_at_10: float
    recall_at_10: float
    precision_at_10: float
    mrr: float
    result_count: int

    @field_validator("ndcg_at_10", "recall_at_10", "precision_at_10", "mrr")
    @classmethod
    def validate_metric(cls, v: float) -> float:
        if not 0.0 <= v <= 1.0:
            raise ValueError(f"Metric must be 0-1, got {v}")
        return v

    @field_validator("result_count")
    @classmethod
    def validate_result_count(cls, v: int) -> int:
        if v < 0:
            raise ValueError(f"result_count must be >= 0, got {v}")
        return v


class BenchmarkDataset(BaseModel):
    """Complete benchmark dataset v4.0 with queries, subgraphs and metadata."""

    version: str = "4.0"
    created_at: str
    queries: list[BenchmarkQuery]
    subgraphs: list[Subgraph]
    metadata: dict

    @classmethod
    def create_empty(cls) -> BenchmarkDataset:
        """Create an empty dataset with current timestamp."""
        return cls(
            created_at=datetime.now(UTC).isoformat(),
            queries=[],
            subgraphs=[],
            metadata={},
        )
