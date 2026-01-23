"""Dynamic ontology extension system.

Provides:
- ProposedRelation: Model for LLM-proposed new relation types
- OntologyExtender: Validates and manages proposed extensions
- RelationProposal: Result from LLM proposing a new relation

Key features:
1. LLM proposes new relation when existing types don't fit
2. Strong model validates the proposal
3. Proposed relations are stored for admin review

This enables ontology evolution based on actual content patterns.
"""

from __future__ import annotations

import uuid
from dataclasses import dataclass
from datetime import datetime
from enum import Enum
from typing import TYPE_CHECKING, Any

from prompts import RELATION_PROPOSAL_PROMPT, RELATION_VALIDATION_PROMPT
from pydantic import BaseModel, Field

from .base import BaseOntology, RelationConstraint

if TYPE_CHECKING:
    pass


class ProposalStatus(str, Enum):
    """Status of a proposed relation."""

    PENDING = "pending"  # Awaiting admin review
    APPROVED = "approved"  # Added to ontology
    REJECTED = "rejected"  # Rejected by admin
    AUTO_APPROVED = "auto_approved"  # High-confidence auto-approval


class ProposedRelation(BaseModel):
    """A relation type proposed by LLM.

    Used when LLM determines that existing relation types
    don't adequately capture the relationship between nodes.
    """

    id: uuid.UUID = Field(default_factory=uuid.uuid4)
    relation_name: str = Field(..., description="Proposed relation name (e.g., 'ĐỊNH_NGHĨA')")
    relation_name_english: str = Field(default="", description="English name (e.g., 'DEFINES')")
    description: str = Field(..., description="What this relation means")
    source_node_type: str = Field(..., description="Source node type")
    target_node_type: str = Field(..., description="Target node type")
    is_symmetric: bool = Field(default=False)
    is_transitive: bool = Field(default=False)
    confidence: float = Field(default=0.0, description="LLM confidence in this proposal")
    reasoning: str = Field(default="", description="Why this relation is needed")
    model_used: str = Field(default="", description="Model that proposed this")
    validated_by: str | None = Field(default=None, description="Strong model that validated")
    validation_score: float = Field(default=0.0, description="Validation confidence")
    status: ProposalStatus = Field(default=ProposalStatus.PENDING)
    created_at: datetime = Field(default_factory=datetime.now)
    example_source_text: str = Field(default="", description="Example source node text")
    example_target_text: str = Field(default="", description="Example target node text")

    def to_constraint(self) -> RelationConstraint:
        """Convert to RelationConstraint for ontology."""
        return RelationConstraint(
            relation_type=self.relation_name,
            source_node_types=[self.source_node_type],
            target_node_types=[self.target_node_type],
            is_symmetric=self.is_symmetric,
            is_transitive=self.is_transitive,
            description=self.description,
        )


@dataclass
class ExtensionResult:
    """Result of ontology extension attempt."""

    success: bool = False
    used_existing: bool = False  # Used an existing relation
    proposed_new: bool = False  # Proposed a new relation
    relation_type: str = ""
    confidence: float = 0.0
    proposal: ProposedRelation | None = None
    error: str = ""


class OntologyExtender:
    """Manages dynamic ontology extension.

    Workflow:
    1. LLM tries to match existing relations
    2. If no match, LLM proposes new relation
    3. Strong model validates proposal
    4. High-confidence proposals are auto-approved
    5. Low-confidence proposals await admin review

    Configuration:
    - auto_approve_threshold: >= this score = auto-approve
    - min_validation_threshold: < this score = reject
    - max_proposals: Maximum proposals to keep in memory
    """

    # Memory limit to prevent unbounded growth
    MAX_PROPOSALS = 1000

    def __init__(
        self,
        ontology: BaseOntology,
        auto_approve_threshold: float = 0.90,
        min_validation_threshold: float = 0.70,
        max_proposals: int | None = None,
    ):
        """Initialize extender.

        Args:
            ontology: Base ontology to extend
            auto_approve_threshold: Score >= this = auto-approve
            min_validation_threshold: Score < this = reject
            max_proposals: Maximum proposals to keep (default: MAX_PROPOSALS)
        """
        self.ontology = ontology
        self.auto_approve_threshold = auto_approve_threshold
        self.min_validation_threshold = min_validation_threshold
        self.max_proposals = max_proposals or self.MAX_PROPOSALS
        # In-memory store of proposals (would be persisted to DB in production)
        self._proposals: list[ProposedRelation] = []
        # Runtime extensions (approved proposals not yet in base ontology)
        self._runtime_extensions: list[RelationConstraint] = []

    def get_extended_ontology(self) -> BaseOntology:
        """Get ontology with runtime extensions included."""
        if not self._runtime_extensions:
            return self.ontology

        # Create new ontology with extended relations
        extended_relations = list(self.ontology.relations) + self._runtime_extensions
        return BaseOntology(
            name=f"{self.ontology.name}_extended",
            display_name=f"{self.ontology.display_name} (Extended)",
            relations=extended_relations,
        )

    def add_proposal(self, proposal: ProposedRelation) -> ProposedRelation:
        """Add a new relation proposal.

        Handles validation and auto-approval logic.

        Args:
            proposal: The proposed relation

        Returns:
            Updated proposal with status
        """
        # Check if similar proposal already exists
        existing = self._find_similar_proposal(proposal)
        if existing:
            # Increment confidence of existing proposal
            existing.confidence = min(1.0, existing.confidence + 0.1)
            return existing

        # Check validation score for auto-approval
        if proposal.validation_score >= self.auto_approve_threshold:
            proposal.status = ProposalStatus.AUTO_APPROVED
            # Add to runtime extensions
            self._runtime_extensions.append(proposal.to_constraint())
        elif proposal.validation_score < self.min_validation_threshold:
            proposal.status = ProposalStatus.REJECTED
        else:
            proposal.status = ProposalStatus.PENDING

        # Evict oldest rejected proposals if at capacity
        if len(self._proposals) >= self.max_proposals:
            self._evict_rejected_proposals()

        self._proposals.append(proposal)
        return proposal

    def _evict_rejected_proposals(self) -> None:
        """Evict oldest rejected proposals to stay within memory limit."""
        # First, try to remove rejected proposals (oldest first)
        rejected = [p for p in self._proposals if p.status == ProposalStatus.REJECTED]
        if rejected:
            # Sort by created_at and remove oldest
            rejected.sort(key=lambda p: p.created_at)
            to_remove = rejected[: max(1, len(rejected) // 2)]
            for p in to_remove:
                self._proposals.remove(p)
            return

        # If no rejected proposals, cap at max by removing oldest pending
        if len(self._proposals) >= self.max_proposals:
            pending = [p for p in self._proposals if p.status == ProposalStatus.PENDING]
            if pending:
                pending.sort(key=lambda p: p.created_at)
                self._proposals.remove(pending[0])

    def _find_similar_proposal(self, proposal: ProposedRelation) -> ProposedRelation | None:
        """Find existing proposal with similar name."""
        normalized_name = proposal.relation_name.upper().replace(" ", "_")
        for p in self._proposals:
            if p.relation_name.upper().replace(" ", "_") == normalized_name:
                return p
        return None

    def approve_proposal(self, proposal_id: uuid.UUID) -> bool:
        """Manually approve a pending proposal.

        Args:
            proposal_id: UUID of the proposal to approve

        Returns:
            True if approved, False if not found
        """
        for p in self._proposals:
            if p.id == proposal_id and p.status == ProposalStatus.PENDING:
                p.status = ProposalStatus.APPROVED
                self._runtime_extensions.append(p.to_constraint())
                return True
        return False

    def reject_proposal(self, proposal_id: uuid.UUID) -> bool:
        """Manually reject a pending proposal.

        Args:
            proposal_id: UUID of the proposal to reject

        Returns:
            True if rejected, False if not found
        """
        for p in self._proposals:
            if p.id == proposal_id and p.status == ProposalStatus.PENDING:
                p.status = ProposalStatus.REJECTED
                return True
        return False

    def get_pending_proposals(self) -> list[ProposedRelation]:
        """Get all proposals awaiting review."""
        return [p for p in self._proposals if p.status == ProposalStatus.PENDING]

    def get_all_proposals(self) -> list[ProposedRelation]:
        """Get all proposals."""
        return self._proposals.copy()

    def get_relation_stats(self) -> dict[str, Any]:
        """Get statistics about relation usage and proposals."""
        return {
            "base_relations": len(self.ontology.relations),
            "runtime_extensions": len(self._runtime_extensions),
            "total_proposals": len(self._proposals),
            "pending": len([p for p in self._proposals if p.status == ProposalStatus.PENDING]),
            "approved": len(
                [
                    p
                    for p in self._proposals
                    if p.status in (ProposalStatus.APPROVED, ProposalStatus.AUTO_APPROVED)
                ]
            ),
            "rejected": len([p for p in self._proposals if p.status == ProposalStatus.REJECTED]),
        }

    def export_approved_for_ontology(self) -> list[dict[str, Any]]:
        """Export approved proposals for adding to base ontology.

        Returns a list of proposal dicts ready to be converted to
        RelationConstraint definitions in ontology files.
        """
        approved = [
            p
            for p in self._proposals
            if p.status in (ProposalStatus.APPROVED, ProposalStatus.AUTO_APPROVED)
        ]
        return [
            {
                "relation_name": p.relation_name,
                "relation_name_english": p.relation_name_english,
                "source_node_types": [p.source_node_type],
                "target_node_types": [p.target_node_type],
                "is_symmetric": p.is_symmetric,
                "is_transitive": p.is_transitive,
                "description": p.description,
                "confidence": p.confidence,
            }
            for p in approved
        ]


__all__ = [
    "ProposalStatus",
    "ProposedRelation",
    "ExtensionResult",
    "OntologyExtender",
    "RELATION_PROPOSAL_PROMPT",
    "RELATION_VALIDATION_PROMPT",
]
