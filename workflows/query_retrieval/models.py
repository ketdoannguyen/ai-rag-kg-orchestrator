"""Models for query_retrieval workflow.

This module provides dataclasses and enums for query classification and retrieval:
- QueryType: Types of history questions
- RetrievalStrategy: Retrieval strategies
- QueryRetrievalInput/Output: Workflow I/O

All models are domain-agnostic where possible to support future domains.
Note: Filter extraction removed - BM25 hybrid retrieval handles keyword matching.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum


class QueryType(str, Enum):
    """Query types for Vietnamese History domain.

    8 types covering all question patterns in history textbooks.
    """

    # Asking for specific facts - no relation traverse needed
    FACTUAL = "factual"

    # Asking for cause-effect relationships - needs relation traverse
    CAUSAL = "causal"

    # Comparing entities/events - needs relation traverse
    COMPARISON = "comparison"

    # Asking about document structure - search structure tier
    STRUCTURAL = "structural"

    # Asking for definitions/characteristics - needs DEFINES relation
    CONCEPT = "concept"

    # Asking about historical figures - needs figure relations
    FIGURE = "figure"

    # Synthesizing lessons across events - needs multiple sources
    LESSON_SYNTHESIS = "lesson_synthesis"

    # Questions not related to Vietnamese history
    OTHER = "other"


class RetrievalStrategy(str, Enum):
    """6 retrieval strategies for ontology-guided traversal comparison."""

    # Baselines (no ontology relations)
    VECTOR_ONLY = "vector_only"  # B1: cosine search on content_nodes
    BM25 = "bm25"  # B2: fulltext/keyword search
    HYBRID = "hybrid"  # B3: Vector + BM25 with RRF fusion

    # Experimental (ontology-guided)
    TOP_DOWN = "top_down"  # E1: structure → content descent
    HORIZONTAL = "horizontal"  # E2: content ↔ content via relations
    BOTTOM_UP = "bottom_up"  # E3: content → parent summaries


@dataclass
class RelationCandidate:
    """Candidate relation with similarity score.

    Used for embedding-first relation selection.
    """

    relation_type: str
    description: str
    similarity: float


@dataclass
class QueryClassification:
    """Query classification result from LLM.

    Contains all routing information for retrieval.
    Note: Filters removed - BM25 handles keyword matching implicitly.
    """

    query_type: QueryType
    confidence: float

    # Strategy
    strategy: RetrievalStrategy

    # Relation configuration
    use_relations: bool
    selected_relations: list[str] = field(default_factory=list)

    # Tier requirements
    requires_structure_tier: bool = False
    requires_content_tier: bool = True

    # Source requirements
    min_sources: int = 1
    max_sources: int = 5

    # Debug
    reasoning: str = ""
    candidate_relations_considered: int = 0


@dataclass
class ContentWithContext:
    """Content node with retrieval context."""

    node_id: str
    text: str
    score: float
    node_type: str = "paragraph"
    found_via_relation: str | None = None


@dataclass
class QueryRetrievalInput:
    """Input for query_retrieval workflow."""

    query: str
    query_embedding: list[float] | None = None  # Pre-computed, or computed in workflow
    top_k: int = 10
    top_k_relations: int = 10
    tier: str | None = None  # Filter relations by tier
    source_id: str | None = None  # Filter by data source


@dataclass
class QueryRetrievalOutput:
    """Output from query_retrieval workflow."""

    query: str
    classification: QueryClassification

    # Results
    content_results: list[ContentWithContext] = field(default_factory=list)
    structure_results: list[dict] = field(default_factory=list)

    # Related nodes from relation traverse (grouped by relation type)
    related_by_relation: dict[str, list[ContentWithContext]] = field(default_factory=dict)

    # Timing
    embedding_time_ms: float = 0.0
    relation_filter_time_ms: float = 0.0
    llm_routing_time_ms: float = 0.0
    retrieval_time_ms: float = 0.0
    total_time_ms: float = 0.0

    # Strategy-specific metrics
    top_down_metrics: dict | None = None
    horizontal_metrics: dict | None = None

    # Stats
    total_results: int = 0

    def get_all_content_texts(self) -> list[str]:
        """Get all content texts for LLM context."""
        texts = []

        # Add direct results
        for cr in self.content_results:
            if cr.text:
                texts.append(cr.text)

        # Add related results
        for nodes in self.related_by_relation.values():
            for node in nodes:
                if node.text:
                    texts.append(node.text)

        return texts

    def to_dict(self) -> dict:
        """Convert to dict for API response."""
        return {
            "query": self.query,
            "query_type": self.classification.query_type.value,
            "strategy": self.classification.strategy.value,
            "use_relations": self.classification.use_relations,
            "selected_relations": self.classification.selected_relations,
            "content_results": [
                {
                    "node_id": r.node_id,
                    "text": r.text,
                    "score": r.score,
                    "found_via_relation": r.found_via_relation,
                }
                for r in self.content_results
            ],
            "related_by_relation": {
                rel: [{"node_id": n.node_id, "text": n.text} for n in nodes]
                for rel, nodes in self.related_by_relation.items()
            },
            "timing": {
                "embedding_ms": self.embedding_time_ms,
                "relation_filter_ms": self.relation_filter_time_ms,
                "llm_routing_ms": self.llm_routing_time_ms,
                "retrieval_ms": self.retrieval_time_ms,
                "total_ms": self.total_time_ms,
            },
            "total_results": self.total_results,
        }
