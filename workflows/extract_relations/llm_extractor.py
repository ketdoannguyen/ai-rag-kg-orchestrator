"""LLM-based relation extraction.

Simplified single-model approach:
- Uses strong model (configurable) for all extractions
- Batch processing support for multiple node pairs
- Unified prompt: merge check + relation extraction in one call
- Supports new relation proposals with validation
- Uses OntologyHandler with Centroid method for relation filtering
- Validates new relations using NewRelationValidator
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from typing import TYPE_CHECKING

from arango_client.models import RelationTier
from arango_client.utils import normalize_tier
from llm_gateway_client.models.request import GatewayChatTurn, GatewayInvokeRequest
from loguru import logger
from prompts import VERIFICATION_BATCH_PROMPT, VERIFICATION_PROMPT
from tenacity import (
    retry,
    retry_if_exception_type,
    stop_after_attempt,
    wait_exponential,
)

from .relation_validator import NewRelationValidator, NodeInfo
from .utils import extract_json_from_llm_response

if TYPE_CHECKING:
    from arango.database import StandardDatabase
    from arango_client.handlers import OntologyHandler

    from config.extraction import ExtractionSettings
    from config.ontologies.base import BaseOntology


@dataclass
class ExtractionResult:
    """Result from LLM relation extraction."""

    should_merge: bool = False
    relation_type: str | None = None
    confidence: float = 0.0
    reasoning: str = ""
    is_new_relation: bool = False
    new_relation_description: str | None = None  # Description for new relations
    is_in_top_k: bool = False  # True if relation was in filtered top-k list
    used_fallback: bool = False  # True if fallback relation was used
    validated: bool = False  # True if new relation was validated by Strong LLM


@dataclass
class NodePair:
    """A pair of nodes to verify."""

    node_a_id: str
    node_a_type: str
    node_a_text: str
    node_b_id: str
    node_b_type: str
    node_b_text: str
    similarity_score: float
    node_a_embedding: list[float] | None = None
    node_b_embedding: list[float] | None = None


class LLMRelationExtractor:
    """LLM-based relation extraction from node pairs.

    Uses single strong model for quality decisions.
    Supports batch processing for efficiency.
    Uses OntologyHandler with Centroid method for relation filtering.
    """

    MAX_TEXT_LENGTH = 500
    DEFAULT_TOP_K_RELATIONS = 10  # Number of relations to filter using Centroid
    MAX_PARSE_RETRIES = 3  # Retry LLM call when JSON parse fails

    # Relations to exclude from top-k filtering
    # These are fallback relations that should only be used when LLM fails
    # Excluding them forces LLM to think deeper about the actual relationship
    EXCLUDE_FROM_TOP_K: frozenset[str] = frozenset({"SIMILAR_TO"})

    def __init__(
        self,
        settings: ExtractionSettings | None = None,
        ontology: BaseOntology | None = None,
        db: StandardDatabase | None = None,
        ontology_handler: OntologyHandler | None = None,
    ) -> None:
        """Initialize LLM relation extractor.

        Args:
            settings: Extraction settings
            ontology: Base ontology for fallback (if no handler)
            db: ArangoDB database connection
            ontology_handler: Handler for ontology operations (preferred)
        """
        self._settings = settings
        self._ontology = ontology
        self._db = db
        self._ontology_handler = ontology_handler
        self._llm_client = None
        self._relation_validator = None
        self._log = logger.bind(component="llm_extractor")

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
    def llm_client(self):
        """Get LLM client (lazy load)."""
        if self._llm_client is None:
            from workflows.dependencies import init_llm_client

            self._llm_client = init_llm_client()
        return self._llm_client

    @property
    def relation_validator(self) -> NewRelationValidator:
        """Get relation validator (lazy load)."""
        if self._relation_validator is None:
            self._relation_validator = NewRelationValidator(
                settings=self._settings,
                db=self._db,
                ontology_handler=self._ontology_handler,
                llm_client=self._llm_client,
            )
        return self._relation_validator

    @property
    def extraction_model(self) -> str:
        """Get the model to use for extraction.

        Falls back to LLM_CHAT_MODEL if EXTRACTION_STRONG_MODEL is empty.
        """
        model = self.settings.EXTRACTION_STRONG_MODEL
        if model:
            return model
        # Fallback to LLM_CHAT_MODEL from global settings
        from config.settings import get_settings

        settings = get_settings()
        return settings.LLM_CHAT_MODEL

    # =========================================================================
    # Text Sanitization
    # =========================================================================

    def _sanitize_text(self, text: str) -> str:
        """Sanitize text for use in prompt."""
        if not text:
            return ""

        text = text[: self.MAX_TEXT_LENGTH]
        text = re.sub(r"^#+\s*", "", text, flags=re.MULTILINE)
        text = re.sub(r"```[\w]*", "", text)
        text = re.sub(r"(system|user|assistant):", "", text, flags=re.IGNORECASE)
        text = text.replace("{", "(").replace("}", ")")
        return text.strip()

    def _build_ontology_section(self) -> str:
        """Build ontology relations section for prompt (fallback without embeddings)."""
        if self._ontology is None:
            return "Use SIMILAR_TO as default relation."
        return self._ontology.to_llm_prompt_section()

    def _build_filtered_ontology_section(
        self,
        source_embedding: list[float] | None,
        target_embedding: list[float] | None,
        top_k: int | None = None,
        tier: RelationTier | str | None = None,
    ) -> str:
        """Build ontology section with Centroid-filtered relations.

        Uses the Centroid method to filter top-k most relevant relations
        based on source and target embeddings.

        Note: SIMILAR_TO is excluded from results to force LLM to think deeper
        about the actual relationship. SIMILAR_TO is only used as fallback
        when LLM fails after max retries.

        Args:
            source_embedding: Embedding of source node
            target_embedding: Embedding of target node
            top_k: Number of relations to include (default: DEFAULT_TOP_K_RELATIONS)
            tier: Optional tier to filter relations (tier_1_structure, tier_2_content)

        Returns:
            Formatted prompt section with filtered relations
        """
        top_k = top_k or self.DEFAULT_TOP_K_RELATIONS

        # Fallback to static ontology if no embeddings or handler
        if not source_embedding or not target_embedding or not self.ontology_handler:
            return self._build_ontology_section()

        # Get filtered relations using Centroid method with tier filter
        # Exclude SIMILAR_TO to force LLM to think deeper about the relationship
        scored_relations = self.ontology_handler.filter_relations_by_centroid(
            source_embedding=source_embedding,
            target_embedding=target_embedding,
            top_k=top_k,
            tier=tier,
            exclude_relations=self.EXCLUDE_FROM_TOP_K,
        )

        if not scored_relations:
            return self._build_ontology_section()

        # Build prompt section with filtered relations
        lines = ["## Available Relationship Types (filtered by relevance)\n"]
        lines.append("Only use relationships from this list:\n")

        for rel_id, score in scored_relations:
            rel = self.ontology_handler.get_by_id(rel_id)
            if rel:
                line = f"- **{rel.name}** (score: {score:.2f}): {rel.description}"
                if rel.is_symmetric:
                    line += " (symmetric)"
                if rel.is_transitive:
                    line += " (transitive)"
                lines.append(line)

        lines.append("\n- Or propose a new relation type if none fit well")

        return "\n".join(lines)

    def _get_top_k_relation_ids(
        self,
        source_embedding: list[float] | None,
        target_embedding: list[float] | None,
        top_k: int | None = None,
        tier: RelationTier | str | None = None,
    ) -> set[str]:
        """Get the set of top-k relation IDs for checking results.

        Note: SIMILAR_TO is excluded from the set to match the filtered prompt.

        Args:
            source_embedding: Embedding of source node
            target_embedding: Embedding of target node
            top_k: Number of relations (default: DEFAULT_TOP_K_RELATIONS)
            tier: Optional tier filter

        Returns:
            Set of relation IDs in the top-k filtered list
        """
        top_k = top_k or self.DEFAULT_TOP_K_RELATIONS

        if not source_embedding or not target_embedding or not self.ontology_handler:
            return set()

        scored_relations = self.ontology_handler.filter_relations_by_centroid(
            source_embedding=source_embedding,
            target_embedding=target_embedding,
            top_k=top_k,
            tier=tier,
            exclude_relations=self.EXCLUDE_FROM_TOP_K,
        )

        return {rel_id for rel_id, _ in scored_relations} if scored_relations else set()

    # =========================================================================
    # Single Pair Verification
    # =========================================================================

    async def verify_pair(
        self,
        node_a_type: str,
        node_a_text: str,
        node_b_type: str,
        node_b_text: str,
        similarity_score: float,
        node_a_embedding: list[float] | None = None,
        node_b_embedding: list[float] | None = None,
        tier: RelationTier | str | None = None,
    ) -> ExtractionResult:
        """Extract relation from a single pair of nodes.

        Retries LLM call up to MAX_PARSE_RETRIES times when JSON parsing fails.
        """
        top_k_relations = self._get_top_k_relation_ids(
            node_a_embedding, node_b_embedding, tier=tier
        )

        prompt = self._build_single_prompt(
            node_a_type=node_a_type,
            node_a_text=self._sanitize_text(node_a_text),
            node_b_type=node_b_type,
            node_b_text=self._sanitize_text(node_b_text),
            similarity_score=similarity_score,
            node_a_embedding=node_a_embedding,
            node_b_embedding=node_b_embedding,
            tier=tier,
        )

        last_error: Exception | None = None

        for attempt in range(self.MAX_PARSE_RETRIES):
            try:
                response = await self._call_llm(prompt)
                result = self._parse_response(response, top_k_relations)

                # Validate new relation if LLM suggested one
                if result.is_new_relation and result.relation_type:
                    result = await self._validate_new_relation(
                        result=result,
                        node_a_id="",  # Will be passed from caller if needed
                        node_a_type=node_a_type,
                        node_a_text=node_a_text,
                        node_a_embedding=node_a_embedding,
                        node_b_id="",
                        node_b_type=node_b_type,
                        node_b_text=node_b_text,
                        node_b_embedding=node_b_embedding,
                        tier=tier,
                        top_k_relation_ids=list(top_k_relations),
                    )

                    # Store the validated relation
                    if result.relation_type and result.is_new_relation:
                        await self._store_proposed_relation(
                            relation_type=result.relation_type,
                            source_type=node_a_type,
                            target_type=node_b_type,
                            description=result.new_relation_description,
                            tier=tier,
                        )
                return result

            except Exception as e:
                last_error = e
                self._log.warning(
                    "LLM call/parse failed (attempt {attempt}/{max}): {error}",
                    attempt=attempt + 1,
                    max=self.MAX_PARSE_RETRIES,
                    error=str(e),
                )

        # All retries exhausted
        self._log.error(
            "All parse retries exhausted, using fallback",
            max_retries=self.MAX_PARSE_RETRIES,
            last_error=str(last_error) if last_error else "Unknown",
        )
        return ExtractionResult(
            should_merge=False,
            relation_type=self.settings.EXTRACTION_FALLBACK_RELATION,
            confidence=0.5,
            reasoning=f"Parse failed after {self.MAX_PARSE_RETRIES} retries",
            used_fallback=True,
        )

    def _build_single_prompt(
        self,
        node_a_type: str,
        node_a_text: str,
        node_b_type: str,
        node_b_text: str,
        similarity_score: float,
        node_a_embedding: list[float] | None = None,
        node_b_embedding: list[float] | None = None,
        tier: RelationTier | str | None = None,
    ) -> str:
        """Build prompt for single pair verification.

        Uses Centroid-filtered relations if embeddings are provided.
        Filters by tier if specified.
        """
        # Use filtered ontology if embeddings available
        if node_a_embedding and node_b_embedding:
            ontology_section = self._build_filtered_ontology_section(
                source_embedding=node_a_embedding,
                target_embedding=node_b_embedding,
                tier=tier,
            )
        else:
            ontology_section = self._build_ontology_section()

        return VERIFICATION_PROMPT.format(
            node_a_type=node_a_type,
            node_a_text=node_a_text,
            node_b_type=node_b_type,
            node_b_text=node_b_text,
            similarity_score=similarity_score,
            valid_relations=ontology_section,
        )

    # =========================================================================
    # Batch Verification
    # =========================================================================

    async def verify_batch(
        self,
        pairs: list[NodePair],
        tier: RelationTier | str | None = None,
    ) -> list[ExtractionResult]:
        """Extract relations for multiple pairs in a single LLM call.

        Retries LLM call up to MAX_PARSE_RETRIES times when JSON parsing fails.
        """
        if not pairs:
            return []

        # Single pair - delegate to verify_pair
        if len(pairs) == 1:
            result = await self.verify_pair(
                node_a_type=pairs[0].node_a_type,
                node_a_text=pairs[0].node_a_text,
                node_b_type=pairs[0].node_b_type,
                node_b_text=pairs[0].node_b_text,
                similarity_score=pairs[0].similarity_score,
                node_a_embedding=pairs[0].node_a_embedding,
                node_b_embedding=pairs[0].node_b_embedding,
                tier=tier,
            )
            return [result]

        first_pair = pairs[0]
        top_k_relations = self._get_top_k_relation_ids(
            first_pair.node_a_embedding,
            first_pair.node_b_embedding,
            top_k=15,
            tier=tier,
        )

        prompt = self._build_batch_prompt(pairs, tier=tier)
        last_error: Exception | None = None

        for attempt in range(self.MAX_PARSE_RETRIES):
            try:
                response = await self._call_llm(prompt)
                results = self._parse_batch_response(response, len(pairs), top_k_relations)

                # Validate and store new relations (same as single pair)
                validated_results = []
                for i, result in enumerate(results):
                    pair = pairs[i]
                    if result.is_new_relation and result.relation_type:
                        # Validate new relation using the same flow as single pair
                        validated_result = await self._validate_new_relation(
                            result=result,
                            node_a_id=pair.node_a_id,
                            node_a_type=pair.node_a_type,
                            node_a_text=pair.node_a_text,
                            node_a_embedding=pair.node_a_embedding,
                            node_b_id=pair.node_b_id,
                            node_b_type=pair.node_b_type,
                            node_b_text=pair.node_b_text,
                            node_b_embedding=pair.node_b_embedding,
                            tier=tier,
                            top_k_relation_ids=list(top_k_relations),
                        )

                        # Store only if still a new relation after validation
                        if validated_result.is_new_relation and validated_result.relation_type:
                            await self._store_proposed_relation(
                                relation_type=validated_result.relation_type,
                                source_type=pair.node_a_type,
                                target_type=pair.node_b_type,
                                description=validated_result.new_relation_description,
                                tier=tier,
                            )
                        validated_results.append(validated_result)
                    else:
                        validated_results.append(result)
                return validated_results

            except Exception as e:
                last_error = e
                self._log.warning(
                    "Batch LLM call/parse failed (attempt {attempt}/{max}): {error}",
                    attempt=attempt + 1,
                    max=self.MAX_PARSE_RETRIES,
                    error=str(e),
                )

        # All retries exhausted
        self._log.error(
            "All batch parse retries exhausted, using fallback",
            max_retries=self.MAX_PARSE_RETRIES,
            pair_count=len(pairs),
            last_error=str(last_error) if last_error else "Unknown",
        )
        fallback = ExtractionResult(
            should_merge=False,
            relation_type=self.settings.EXTRACTION_FALLBACK_RELATION,
            confidence=0.5,
            reasoning=f"Parse failed after {self.MAX_PARSE_RETRIES} retries",
            used_fallback=True,
        )
        return [fallback for _ in pairs]

    def _build_batch_prompt(
        self,
        pairs: list[NodePair],
        tier: RelationTier | str | None = None,
    ) -> str:
        """Build prompt for batch verification.

        Uses first pair's embeddings for Centroid-based relation filtering.
        Filters by tier if specified.
        """
        # Use first pair's embeddings for filtering (representative sample)
        first_pair = pairs[0]
        if first_pair.node_a_embedding and first_pair.node_b_embedding:
            ontology_section = self._build_filtered_ontology_section(
                source_embedding=first_pair.node_a_embedding,
                target_embedding=first_pair.node_b_embedding,
                top_k=15,  # Slightly more relations for batch mode diversity
                tier=tier,
            )
        else:
            ontology_section = self._build_ontology_section()

        pairs_section = []
        for i, pair in enumerate(pairs):
            pairs_section.append(
                f"""
### Pair {i + 1}
- Node A ({pair.node_a_type}): {self._sanitize_text(pair.node_a_text)}
- Node B ({pair.node_b_type}): {self._sanitize_text(pair.node_b_text)}
- Similarity: {pair.similarity_score:.2f}
"""
            )

        pairs_text = "\n".join(pairs_section)

        return VERIFICATION_BATCH_PROMPT.format(
            pairs_text=pairs_text,
            valid_relations=ontology_section,
            expected_count=len(pairs),
        )

    def _parse_batch_response(
        self,
        response: str,
        expected_count: int,
        top_k_relations: set[str] | None = None,
    ) -> list[ExtractionResult]:
        """Parse batch response into list of results.

        Args:
            response: LLM response string
            expected_count: Expected number of results
            top_k_relations: Set of relation IDs in top-k for tracking

        Raises:
            ValueError: If JSON parsing fails or response format is invalid
        """
        fallback_relation = self.settings.EXTRACTION_FALLBACK_RELATION
        fallback_result = ExtractionResult(
            should_merge=False,
            relation_type=fallback_relation,
            confidence=0.5,
            reasoning="Fallback",
            used_fallback=True,
        )
        top_k_relations = top_k_relations or set()

        data = extract_json_from_llm_response(response)

        if not isinstance(data, list):
            raise ValueError("Batch response is not a list")

        results = []
        for item in data:
            relation_type = item.get("relation_type")
            is_new = item.get("is_new_relation", False)
            is_in_top_k = (
                not is_new and relation_type is not None and relation_type in top_k_relations
            )

            used_fallback = relation_type == fallback_relation

            results.append(
                ExtractionResult(
                    should_merge=item.get("should_merge", False),
                    relation_type=relation_type,
                    confidence=float(item.get("confidence", 0.5)),
                    reasoning=item.get("reasoning", ""),
                    is_new_relation=is_new,
                    new_relation_description=item.get("new_relation_description"),
                    is_in_top_k=is_in_top_k,
                    used_fallback=used_fallback,
                )
            )

        # Pad with fallback results if needed
        while len(results) < expected_count:
            results.append(fallback_result)
        return results[:expected_count]

    # =========================================================================
    # LLM Call
    # =========================================================================

    @retry(
        stop=stop_after_attempt(3),
        wait=wait_exponential(multiplier=1, min=2, max=30),
        retry=retry_if_exception_type((TimeoutError, ConnectionError, OSError)),
    )
    async def _call_llm(self, prompt: str) -> str:
        """Call LLM with prompt using AIGatewayClient.invoke().

        Retries up to 3 times on transient failures (timeout, connection errors).
        Falls back to SIMILAR_TO relation if all retries are exhausted.
        """
        model = self.extraction_model
        request = GatewayInvokeRequest(
            model=model,
            messages=[
                GatewayChatTurn(
                    role="system",
                    content="You are a knowledge graph expert. Respond only with valid JSON.",
                ),
                GatewayChatTurn(role="user", content=prompt),
            ],
            # Removed max_tokens=1024 as it causes empty responses with some proxies/models
            provider_options={"temperature": 0.1},
        )

        # Debug logging - log prompt (truncated)
        self._log.debug(
            "LLM request",
            model=model,
            prompt_length=len(prompt),
            prompt_preview=prompt[:500] + "..." if len(prompt) > 500 else prompt,
        )

        try:
            response = await self.llm_client.invoke(request)

            # Debug logging - log raw response
            self._log.debug(
                "LLM raw response",
                response_type=type(response).__name__,
                has_message=response.message is not None,
                message_type=type(response.message).__name__ if response.message else None,
                raw_response=str(response)[:500],
            )

            # Extract content from GatewayChatTurn object
            if response.message is None:
                self._log.warning(
                    "LLM returned response with no message content",
                    raw_response=str(response),
                )
                return ""

            content = response.message.content

            # Debug logging - log content
            self._log.debug(
                "LLM content extracted",
                content_type=type(content).__name__,
                content_length=len(content) if content else 0,
                content_preview=content[:300] if content else "EMPTY",
            )

            if not content:
                self._log.warning(
                    "LLM returned empty content string",
                    message_object=str(response.message),
                )
                return ""

            return content
        except Exception as e:
            self._log.error(f"LLM call failed: {e}")
            raise

    # =========================================================================
    # Response Parsing
    # =========================================================================

    def _parse_response(
        self,
        response: str,
        top_k_relations: set[str] | None = None,
    ) -> ExtractionResult:
        """Parse single response into ExtractionResult.

        Args:
            response: LLM response string
            top_k_relations: Set of relation IDs in top-k for tracking

        Raises:
            ValueError: If JSON parsing fails (caller should handle retry)
        """
        fallback_relation = self.settings.EXTRACTION_FALLBACK_RELATION
        top_k_relations = top_k_relations or set()

        data = extract_json_from_llm_response(response)

        relation_type = data.get("relation_type")
        is_new = data.get("is_new_relation", False)
        is_in_top_k = not is_new and relation_type is not None and relation_type in top_k_relations

        used_fallback = relation_type == fallback_relation

        return ExtractionResult(
            should_merge=data.get("should_merge", False),
            relation_type=relation_type,
            confidence=float(data.get("confidence", 0.5)),
            reasoning=data.get("reasoning", ""),
            is_new_relation=is_new,
            new_relation_description=data.get("new_relation_description"),
            is_in_top_k=is_in_top_k,
            used_fallback=used_fallback,
        )

    # =========================================================================
    # New Relation Validation
    # =========================================================================

    async def _validate_new_relation(
        self,
        result: ExtractionResult,
        node_a_id: str,
        node_a_type: str,
        node_a_text: str,
        node_a_embedding: list[float] | None,
        node_b_id: str,
        node_b_type: str,
        node_b_text: str,
        node_b_embedding: list[float] | None,
        tier: RelationTier | str | None,
        top_k_relation_ids: list[str],
    ) -> ExtractionResult:
        """Validate a new relation using NewRelationValidator.

        Implements user's required flow:
        1. Embed (name + description)
        2. Compare with ALL relations (all tiers)
        3. If > threshold → merge with existing
        4. If < threshold → LLM Strong Verify
        5. If valid → use new relation
        6. If invalid → re-extract with Strong LLM

        Args:
            result: Original extraction result with new relation
            node_a_*: Source node info
            node_b_*: Target node info
            tier: Current processing tier
            top_k_relation_ids: IDs already considered in top-k

        Returns:
            Updated ExtractionResult with validated relation
        """
        if not result.is_new_relation or not result.relation_type:
            return result

        # Create NodeInfo objects for validator
        node_a = NodeInfo(
            node_id=node_a_id,
            node_type=node_a_type,
            text=node_a_text,
            embedding=node_a_embedding,
        )
        node_b = NodeInfo(
            node_id=node_b_id,
            node_type=node_b_type,
            text=node_b_text,
            embedding=node_b_embedding,
        )

        # Run validation
        validation_result = await self.relation_validator.validate_new_relation(
            node_a=node_a,
            node_b=node_b,
            proposed_relation=result.relation_type,
            description=result.new_relation_description or "",
            reasoning=result.reasoning,
            tier=normalize_tier(tier),
            top_k_relation_ids=top_k_relation_ids,
        )

        # Update result based on validation
        if validation_result.merged_with_existing:
            # Merged with existing relation - update tier for existing relation
            if self.ontology_handler and validation_result.existing_relation_id and tier:
                tier_added = self.ontology_handler.add_tier_if_missing(
                    validation_result.existing_relation_id, tier
                )
                if tier_added:
                    tier_value = normalize_tier(tier)
                    self._log.info(
                        "Added tier to existing relation after merge",
                        relation=validation_result.existing_relation_id,
                        tier=tier_value,
                    )

            return ExtractionResult(
                should_merge=result.should_merge,
                relation_type=validation_result.final_relation_type,
                confidence=result.confidence,
                reasoning=validation_result.reasoning,
                is_new_relation=False,  # No longer new
                new_relation_description=None,
                is_in_top_k=False,
                used_fallback=False,
                validated=True,
            )
        elif validation_result.is_valid:
            # Valid new relation
            return ExtractionResult(
                should_merge=result.should_merge,
                relation_type=validation_result.final_relation_type,
                confidence=result.confidence,
                reasoning=validation_result.reasoning,
                is_new_relation=True,
                new_relation_description=validation_result.final_description,
                is_in_top_k=False,
                used_fallback=False,
                validated=True,
            )
        else:
            # Validation failed, using fallback
            return ExtractionResult(
                should_merge=result.should_merge,
                relation_type=validation_result.final_relation_type,
                confidence=0.5,
                reasoning=validation_result.reasoning,
                is_new_relation=False,
                new_relation_description=None,
                is_in_top_k=False,
                used_fallback=True,
                validated=False,
            )

    # =========================================================================
    # Proposed Relation Storage
    # =========================================================================

    async def _store_proposed_relation(
        self,
        relation_type: str,
        source_type: str,
        target_type: str,
        description: str | None = None,
        tier: RelationTier | str | None = None,
    ) -> None:
        """Store a proposed new relation in the ontology collection.

        If the relation already exists, increments usage count and adds the
        current tier to the relation's tiers list if not already present.

        Args:
            relation_type: Proposed relation type name
            source_type: Source node type
            target_type: Target node type
            description: Optional description
            tier: Optional tier to add to the relation's tiers list
        """
        if not self.ontology_handler:
            self._log.debug("No ontology handler available, skipping proposed relation storage")
            return

        # Check if relation already exists
        existing = self.ontology_handler.get_by_id(relation_type)
        if existing:
            # Increment usage count
            self.ontology_handler.increment_usage(relation_type)

            # Add tier if provided and not already present in relation's tiers
            if tier is not None:
                tier_added = self.ontology_handler.add_tier_if_missing(relation_type, tier)
                if tier_added:
                    tier_value = normalize_tier(tier)
                    self._log.info(
                        "Extended relation to new tier",
                        relation_type=relation_type,
                        tier=tier_value,
                    )
            return

        # Create proposed relation with current tier
        try:
            tiers_list = [tier] if tier is not None else None
            self.ontology_handler.propose_relation(
                relation_id=relation_type,
                name=relation_type,
                description=description
                or f"LLM-proposed relation: {source_type} -> {target_type}",
                source_node_types=[source_type],
                target_node_types=[target_type],
                domain="proposed",
                tiers=tiers_list,
            )
            self._log.info(
                "Stored proposed relation: {relation_type}",
                relation_type=relation_type,
            )
        except Exception as e:
            self._log.warning(
                "Failed to store proposed relation: {error}",
                error=str(e),
            )
