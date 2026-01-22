"""New relation validation for extract_relations workflow.

Validates new relations proposed by LLM using the user's required logic:

Flow:
1. LLM Extract (weak/fast model) → output: relation_type + description + reasoning
2. If new relation → embed(name + description) → compare with ALL tiers
   - If score > threshold → MERGE (use existing relation)
   - If score < threshold → Go to Step 3
3. LLM Strong Verify (VALIDATOR role)
   - Input: Node A, Node B, proposed_relation, description, reasoning
   - If VALID → Add new relation to DB
   - If INVALID → Go to Step 4
4. Re-extract with LLM Strong
   - LLM Strong extracts relation directly
   - Embed → Compare → Add/Merge
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING

from arango_client.utils import normalize_tier
from llm_gateway_client.models.request import GatewayChatTurn, GatewayInvokeRequest
from loguru import logger
from prompts import (
    RELATION_INSTANCE_VALIDATION_PROMPT,
    RELATION_REEXTRACTION_PROMPT,
)
from tenacity import (
    retry,
    retry_if_exception_type,
    stop_after_attempt,
    wait_exponential,
)

from .utils import extract_json_from_llm_response

if TYPE_CHECKING:
    from arango.database import StandardDatabase
    from arango_client.handlers import OntologyHandler
    from llm_gateway_client.client import AIGatewayClient

    from config.extraction import ExtractionSettings


@dataclass
class NodeInfo:
    """Node information for validation."""

    node_id: str
    node_type: str
    text: str
    embedding: list[float] | None = None


@dataclass
class ValidationResult:
    """Result from new relation validation."""

    is_valid: bool
    final_relation_type: str
    final_description: str | None
    merged_with_existing: bool = False  # True if merged with existing relation
    existing_relation_id: str | None = None  # ID of relation merged with
    similarity_score: float | None = None  # Similarity score if merged
    re_extracted: bool = False  # True if had to re-extract with Strong LLM
    reasoning: str = ""


class NewRelationValidator:
    """Validates new relations proposed by LLM.

    Implements the user's required validation flow:
    1. Embed new relation (name + description)
    2. Compare with ALL existing relations (all tiers)
    3. If similar > threshold → merge with existing
    4. If similar < threshold → LLM Strong Verify
    5. If valid → add new relation
    6. If invalid → re-extract with Strong LLM
    """

    MAX_VALIDATION_RETRIES = 2  # Max retries for re-extraction

    def __init__(
        self,
        settings: ExtractionSettings | None = None,
        db: StandardDatabase | None = None,
        ontology_handler: OntologyHandler | None = None,
        llm_client: AIGatewayClient | None = None,
    ) -> None:
        """Initialize the validator.

        Args:
            settings: Extraction settings
            db: ArangoDB database connection
            ontology_handler: Handler for ontology operations
            llm_client: LLM client for API calls
        """
        self._settings = settings
        self._db = db
        self._ontology_handler = ontology_handler
        self._llm_client = llm_client
        self._log = logger.bind(component="relation_validator")

    @property
    def settings(self) -> ExtractionSettings:
        """Get extraction settings (lazy load)."""
        if self._settings is None:
            from config.extraction import get_extraction_settings

            self._settings = get_extraction_settings()
        return self._settings

    @property
    def ontology_handler(self) -> OntologyHandler | None:
        """Get ontology handler (lazy load)."""
        if self._ontology_handler is None and self._db is not None:
            from arango_client.handlers import OntologyHandler

            self._ontology_handler = OntologyHandler(self._db)
        return self._ontology_handler

    @property
    def llm_client(self) -> AIGatewayClient:
        """Get LLM client (lazy load)."""
        if self._llm_client is None:
            from workflows.dependencies import init_llm_client

            self._llm_client = init_llm_client()
        return self._llm_client

    @property
    def strong_model(self) -> str:
        """Get the strong model for validation."""
        model = self.settings.EXTRACTION_STRONG_MODEL
        if model:
            return model
        from config.settings import get_settings

        return get_settings().LLM_CHAT_MODEL

    # =========================================================================
    # Main Validation Flow
    # =========================================================================

    async def validate_new_relation(
        self,
        node_a: NodeInfo,
        node_b: NodeInfo,
        proposed_relation: str,
        description: str,
        reasoning: str,
        tier: str | None = None,
        top_k_relation_ids: list[str] | None = None,
    ) -> ValidationResult:
        """Validate a new relation proposed by LLM.

        Implements the full validation flow:
        1. Embed (name + description)
        2. Compare with ALL relations (all tiers)
        3. If > threshold → merge
        4. If < threshold → LLM Strong Verify
        5. If invalid → re-extract

        Args:
            node_a: Source node info
            node_b: Target node info
            proposed_relation: Proposed relation type name
            description: Description of the relation
            reasoning: LLM's reasoning for this relation
            tier: Current processing tier
            top_k_relation_ids: IDs of relations already considered in top-k

        Returns:
            ValidationResult with final relation and metadata
        """
        self._log.info(
            "Validating new relation",
            proposed=proposed_relation,
            description=description[:100] if description else None,
        )

        # Step 0: Check if relation already exists by exact name match
        # This handles the case where relation exists in another tier
        if self.ontology_handler:
            existing_relation = self.ontology_handler.get_by_id(proposed_relation)
            if existing_relation:
                # Relation already exists - just need to add tier if missing
                self._log.info(
                    "Relation already exists in DB, will add tier if missing",
                    relation=proposed_relation,
                    existing_tiers=[normalize_tier(t) for t in existing_relation.tiers],
                )
                return ValidationResult(
                    is_valid=True,
                    final_relation_type=proposed_relation,
                    final_description=existing_relation.description,
                    merged_with_existing=True,  # Mark as merged so tier gets updated
                    existing_relation_id=proposed_relation,
                    similarity_score=1.0,  # Exact match
                    reasoning="Exact match found in DB, adding tier",
                )

        # Step 1: Embed the new relation (name + description)
        relation_embedding = await self._embed_relation(proposed_relation, description)

        if not relation_embedding:
            self._log.warning("Failed to embed relation, using as-is")
            return ValidationResult(
                is_valid=True,
                final_relation_type=proposed_relation,
                final_description=description,
                reasoning="Embedding failed, accepted without comparison",
            )

        # Step 2: Compare with ALL existing relations (excluding top-k already considered)
        exclude_ids = top_k_relation_ids or []
        similar_result = await self._find_similar_in_all_tiers(relation_embedding, exclude_ids)

        if similar_result:
            existing_id, similarity_score = similar_result
            if similarity_score >= self.settings.EXTRACTION_MERGE_THRESHOLD:
                # MERGE: Use existing relation instead
                self._log.info(
                    "Merging new relation with existing",
                    proposed=proposed_relation,
                    existing=existing_id,
                    similarity=similarity_score,
                )
                return ValidationResult(
                    is_valid=True,
                    final_relation_type=existing_id,
                    final_description=None,  # Use existing description
                    merged_with_existing=True,
                    existing_relation_id=existing_id,
                    similarity_score=similarity_score,
                    reasoning=f"Merged with existing '{existing_id}' (similarity: {similarity_score:.2f})",
                )

        # Step 3: LLM Strong Verify (validator role)
        is_valid = await self._verify_with_strong_llm(
            node_a=node_a,
            node_b=node_b,
            proposed_relation=proposed_relation,
            description=description,
            reasoning=reasoning,
        )

        if is_valid:
            # Valid - use proposed relation
            self._log.info(
                "New relation validated by Strong LLM",
                relation=proposed_relation,
            )
            return ValidationResult(
                is_valid=True,
                final_relation_type=proposed_relation,
                final_description=description,
                reasoning="Validated by Strong LLM",
            )

        # Step 4: Invalid - re-extract with Strong LLM
        self._log.info(
            "Strong LLM rejected relation, re-extracting",
            rejected=proposed_relation,
        )

        re_extracted = await self._re_extract_with_strong_llm(
            node_a=node_a,
            node_b=node_b,
            tier=tier,
            previous_attempt=proposed_relation,
        )

        if re_extracted:
            # Validate the re-extracted relation
            new_embedding = await self._embed_relation(
                re_extracted["relation_type"],
                re_extracted.get("description", ""),
            )

            if new_embedding:
                # Check if similar to existing
                similar_result = await self._find_similar_in_all_tiers(new_embedding, exclude_ids)
                if similar_result:
                    existing_id, similarity_score = similar_result
                    if similarity_score >= self.settings.EXTRACTION_MERGE_THRESHOLD:
                        return ValidationResult(
                            is_valid=True,
                            final_relation_type=existing_id,
                            final_description=None,
                            merged_with_existing=True,
                            existing_relation_id=existing_id,
                            similarity_score=similarity_score,
                            re_extracted=True,
                            reasoning=f"Re-extracted then merged with '{existing_id}'",
                        )

            # Use re-extracted relation
            return ValidationResult(
                is_valid=True,
                final_relation_type=re_extracted["relation_type"],
                final_description=re_extracted.get("description"),
                re_extracted=True,
                reasoning="Re-extracted by Strong LLM",
            )

        # Fallback if re-extraction failed
        self._log.warning(
            "Re-extraction failed, using fallback",
            fallback=self.settings.EXTRACTION_FALLBACK_RELATION,
        )
        return ValidationResult(
            is_valid=False,
            final_relation_type=self.settings.EXTRACTION_FALLBACK_RELATION,
            final_description=None,
            reasoning="Re-extraction failed, using fallback",
        )

    # =========================================================================
    # Step 1: Embed Relation
    # =========================================================================

    async def _embed_relation(self, name: str, description: str) -> list[float] | None:
        """Embed a relation using name + description.

        Args:
            name: Relation type name
            description: Relation description

        Returns:
            Embedding vector or None if failed
        """
        text_to_embed = f"{name}: {description}" if description else name

        try:
            from llm_gateway_client.models.embedding import EmbeddingRequest

            request = EmbeddingRequest(
                model=self.settings.EXTRACTION_EMBEDDING_MODEL,
                input=[text_to_embed],  # Must be a list
            )
            response = await self.llm_client.embed(request)

            if response.embeddings and len(response.embeddings) > 0:
                return response.embeddings[0]

            self._log.warning("Empty embedding response")
            return None

        except Exception as e:
            self._log.error(f"Failed to embed relation: {e}")
            return None

    # =========================================================================
    # Step 2: Find Similar in ALL Tiers
    # =========================================================================

    async def _find_similar_in_all_tiers(
        self,
        embedding: list[float],
        exclude_ids: list[str],
    ) -> tuple[str, float] | None:
        """Find most similar relation in ALL tiers.

        Searches across all tiers (tier_1_structure, tier_2_content, tier_3_entity)
        to find relations similar to the embedded new relation.

        Args:
            embedding: Embedding of new relation
            exclude_ids: Relation IDs to exclude (already in top-k)

        Returns:
            (relation_id, similarity_score) if found above threshold, None otherwise
        """
        if not self.ontology_handler:
            self._log.warning("No ontology handler, skipping similarity check")
            return None

        # Use the handler's find_similar_excluding method
        # This searches ALL relations with embeddings
        result = self.ontology_handler.find_similar_excluding(
            embedding=embedding,
            exclude_ids=exclude_ids,
            threshold=self.settings.EXTRACTION_MERGE_THRESHOLD,
        )

        return result

    # =========================================================================
    # Step 3: LLM Strong Verify (Validator Role)
    # =========================================================================

    async def _verify_with_strong_llm(
        self,
        node_a: NodeInfo,
        node_b: NodeInfo,
        proposed_relation: str,
        description: str,
        reasoning: str,
    ) -> bool:
        """Verify proposed relation with Strong LLM.

        The Strong LLM acts as a VALIDATOR - it checks if the proposed
        relation is correct for the given node pair.

        Args:
            node_a: Source node info
            node_b: Target node info
            proposed_relation: Proposed relation type
            description: Relation description
            reasoning: Original LLM's reasoning

        Returns:
            True if valid, False if invalid
        """
        prompt = RELATION_INSTANCE_VALIDATION_PROMPT.format(
            node_a_type=node_a.node_type,
            node_a_text=node_a.text[:500],
            node_b_type=node_b.node_type,
            node_b_text=node_b.text[:500],
            proposed_relation=proposed_relation,
            description=description,
            reasoning=reasoning,
        )

        try:
            response = await self._call_strong_llm(prompt)
            data = extract_json_from_llm_response(response)

            is_valid = data.get("is_valid", False)
            confidence = data.get("confidence", 0.0)

            self._log.debug(
                "Strong LLM validation result",
                is_valid=is_valid,
                confidence=confidence,
                reasoning=data.get("reasoning", ""),
            )

            return is_valid

        except Exception as e:
            self._log.error(f"Strong LLM verification failed: {e}")
            # On error, accept the relation (fail open)
            return True

    # =========================================================================
    # Step 4: Re-extract with Strong LLM
    # =========================================================================

    async def _re_extract_with_strong_llm(
        self,
        node_a: NodeInfo,
        node_b: NodeInfo,
        tier: str | None = None,
        previous_attempt: str | None = None,
    ) -> dict | None:
        """Re-extract relation using Strong LLM.

        When the validator rejects the initial proposal, use Strong LLM
        to extract a better relation directly.

        Args:
            node_a: Source node info
            node_b: Target node info
            tier: Current processing tier
            previous_attempt: The rejected relation (to avoid)

        Returns:
            Dict with relation_type and description, or None if failed
        """
        # Get top-k relations for this tier
        ontology_section = self._build_ontology_section_for_reextract(
            node_a.embedding, node_b.embedding, tier
        )

        prompt = RELATION_REEXTRACTION_PROMPT.format(
            node_a_type=node_a.node_type,
            node_a_text=node_a.text[:500],
            node_b_type=node_b.node_type,
            node_b_text=node_b.text[:500],
            previous_attempt=previous_attempt or "None",
            ontology_section=ontology_section,
        )

        try:
            response = await self._call_strong_llm(prompt)
            data = extract_json_from_llm_response(response)

            relation_type = data.get("relation_type")
            if not relation_type:
                self._log.warning("Re-extraction returned no relation_type")
                return None

            # Check if same as rejected
            if relation_type == previous_attempt:
                self._log.warning(
                    "Re-extraction returned same as rejected",
                    relation=relation_type,
                )
                return None

            return {
                "relation_type": relation_type,
                "description": data.get("description"),
                "is_new_relation": data.get("is_new_relation", False),
                "confidence": data.get("confidence", 0.5),
            }

        except Exception as e:
            self._log.error(f"Strong LLM re-extraction failed: {e}")
            return None

    # =========================================================================
    # Helper Methods
    # =========================================================================

    def _build_ontology_section_for_reextract(
        self,
        source_embedding: list[float] | None,
        target_embedding: list[float] | None,
        tier: str | None = None,
    ) -> str:
        """Build ontology section for re-extraction prompt."""
        if not self.ontology_handler:
            return "## Available Relationships\nUse any meaningful relationship type."

        if not source_embedding or not target_embedding:
            return "## Available Relationships\nUse any meaningful relationship type."

        # Get top-k relations for context
        scored_relations = self.ontology_handler.filter_relations_by_centroid(
            source_embedding=source_embedding,
            target_embedding=target_embedding,
            top_k=15,
            tier=tier,
            exclude_relations={"SIMILAR_TO"},
        )

        if not scored_relations:
            return "## Available Relationships\nUse any meaningful relationship type."

        lines = ["## Available Relationship Types (ranked by relevance)\n"]

        for rel_id, score in scored_relations:
            rel = self.ontology_handler.get_relation(rel_id)
            if rel:
                lines.append(f"- **{rel.name}** (score: {score:.2f}): {rel.description}")

        lines.append("\n- Or propose a NEW relationship if none fit well")

        return "\n".join(lines)

    @retry(
        stop=stop_after_attempt(3),
        wait=wait_exponential(multiplier=1, min=2, max=30),
        retry=retry_if_exception_type((TimeoutError, ConnectionError, OSError)),
    )
    async def _call_strong_llm(self, prompt: str) -> str:
        """Call Strong LLM with retry logic."""
        request = GatewayInvokeRequest(
            model=self.strong_model,
            messages=[
                GatewayChatTurn(
                    role="system",
                    content="You are a knowledge graph expert. Respond only with valid JSON.",
                ),
                GatewayChatTurn(role="user", content=prompt),
            ],
            provider_options={"temperature": 0.1},
        )

        response = await self.llm_client.invoke(request)

        if response.message is None:
            return ""

        return response.message.content or ""
