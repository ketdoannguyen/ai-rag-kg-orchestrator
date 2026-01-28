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
    STRONG_VERIFY_AND_EXTRACT_PROMPT,
)
from tenacity import (
    retry,
    retry_if_exception_type,
    stop_after_attempt,
    wait_exponential,
)

from .models import NodeInfo
from .utils import extract_json_from_llm_response

if TYPE_CHECKING:
    from arango.database import StandardDatabase
    from arango_client.handlers import OntologyHandler
    from llm_gateway_client.client import AIGatewayClient

    from config.extraction import ExtractionSettings

# Database retry decorator for ArangoDB operations
# Retries on connection errors and server errors
db_retry = retry(
    stop=stop_after_attempt(2),
    wait=wait_exponential(multiplier=0.5, min=1, max=10),
    retry=retry_if_exception_type((ConnectionError, OSError, TimeoutError)),
)


@dataclass
class StrongLLMRelationOutput:
    """Output from LLM Strong verification and extraction.

    LLM Strong performs both validation AND full info extraction in one step.
    If it rejects the weak model's proposal, it proposes a better relation.
    """

    # Validation
    accepted: bool  # Accept relation from weak model?
    reasoning: str  # Reason for accept/reject

    # Core fields (LLM extracts)
    id: str  # SCREAMING_SNAKE_CASE
    name: str  # Human-readable name
    summary: str  # 10-20 words
    description: str  # Detailed for embedding

    # Type constraints
    source_node_types: list[str]  # ["paragraph", "structure", "entity"]
    target_node_types: list[str]  # ["paragraph", "structure", "entity"]

    # Semantic properties
    is_symmetric: bool  # A→B implies B→A?
    is_transitive: bool  # A→B, B→C implies A→C?

    # Classification
    families: list[str]  # ["kinship", "causal", "temporal"...]
    keywords: list[str]  # Vietnamese keywords


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
    families: list[str] | None = None  # Inferred families for new relations


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
        summary: str,  # Renamed from description - short summary from weak LLM
        tier: str | None = None,
    ) -> ValidationResult:
        """Validate new relation với flow mới.

        FLOW:
        1. Check exact name in DB → Found? → MERGE (add tier)
        2. LLM Strong Verify + Extract FULL INFO
        3. Embed (name + description)
        4. Compare similarity >= 0.85? → MERGE
        5. Store NEW relation to DB

        Args:
            node_a: Source node info
            node_b: Target node info
            proposed_relation: Relation type từ LLM weak
            summary: Short summary từ LLM weak
            tier: Current processing tier

        Returns:
            ValidationResult with final relation and metadata
        """
        self._log.info(
            "Validating new relation",
            proposed=proposed_relation,
            summary=summary[:100] if summary else None,
        )

        # =========================================================
        # STEP 1: Check exact name match in DB
        # =========================================================
        if self.ontology_handler:
            existing = self.ontology_handler.get_by_id(proposed_relation)
            if existing:
                self._log.info(
                    "Relation exists in DB (exact match), returning for tier update by caller",
                    relation=proposed_relation,
                    existing_tiers=[normalize_tier(t) for t in existing.tiers],
                )
                return ValidationResult(
                    is_valid=True,
                    final_relation_type=proposed_relation,
                    final_description=existing.description,
                    merged_with_existing=True,
                    existing_relation_id=proposed_relation,
                    similarity_score=1.0,
                    reasoning="Exact match in DB",
                )

        # =========================================================
        # STEP 2: LLM Strong Verify + Extract FULL INFO
        # =========================================================
        strong_output = await self._verify_and_extract_with_strong_llm(
            node_a=node_a,
            node_b=node_b,
            proposed_relation=proposed_relation,
            summary=summary,
        )

        if strong_output is None:
            # LLM call failed
            self._log.warning("Strong LLM failed, using fallback")
            return ValidationResult(
                is_valid=False,
                final_relation_type=self.settings.EXTRACTION_FALLBACK_RELATION,
                final_description=None,
                reasoning="LLM Strong failed",
            )

        # Use relation from Strong (either accepted or new proposal)
        relation_id = strong_output.id
        relation_name = strong_output.name
        relation_description = strong_output.description

        self._log.info(
            "Strong LLM output",
            accepted=strong_output.accepted,
            relation_id=relation_id,
            families=strong_output.families,
        )

        # =========================================================
        # STEP 2.5: Check if Strong LLM suggested an EXISTING relation
        # (Strong LLM may reject proposed and suggest existing one)
        # =========================================================
        if self.ontology_handler:
            existing = self.ontology_handler.get_by_id(relation_id)
            if existing:
                self._log.info(
                    "Strong LLM suggested existing relation, merging",
                    proposed=proposed_relation,
                    suggested=relation_id,
                )
                return ValidationResult(
                    is_valid=True,
                    final_relation_type=relation_id,
                    final_description=existing.description,
                    merged_with_existing=True,
                    existing_relation_id=relation_id,
                    similarity_score=1.0,
                    re_extracted=not strong_output.accepted,
                    reasoning=f"Strong LLM suggested existing relation '{relation_id}'",
                    families=existing.families if hasattr(existing, "families") else None,
                )

        # =========================================================
        # STEP 3: Embed relation (name + description)
        # =========================================================
        embedding = await self._embed_relation(relation_name, relation_description)

        if not embedding:
            self._log.warning("Embedding failed, storing without similarity check")
            # Still store but skip similarity check
            await self._store_new_relation(strong_output, embedding=None, tier=tier)
            return ValidationResult(
                is_valid=True,
                final_relation_type=relation_id,
                final_description=relation_description,
                reasoning="Stored without embedding",
                families=strong_output.families,
            )

        # =========================================================
        # STEP 4: Compare with ALL existing relations
        # =========================================================
        similar_result = await self._find_similar_in_all_tiers(embedding, exclude_ids=[])

        if similar_result:
            existing_id, similarity_score = similar_result
            if similarity_score >= self.settings.EXTRACTION_MERGE_THRESHOLD:
                self._log.info(
                    "Merging with existing relation",
                    proposed=relation_id,
                    existing=existing_id,
                    similarity=similarity_score,
                )
                return ValidationResult(
                    is_valid=True,
                    final_relation_type=existing_id,
                    final_description=None,
                    merged_with_existing=True,
                    existing_relation_id=existing_id,
                    similarity_score=similarity_score,
                    reasoning=f"Merged with '{existing_id}' (sim: {similarity_score:.2f})",
                )

        # =========================================================
        # STEP 5: Store NEW relation to DB
        # =========================================================
        await self._store_new_relation(strong_output, embedding=embedding, tier=tier)

        return ValidationResult(
            is_valid=True,
            final_relation_type=relation_id,
            final_description=relation_description,
            reasoning=strong_output.reasoning,
            families=strong_output.families,
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
    # Step 2: LLM Strong Verify + Extract Full Info
    # =========================================================================

    async def _verify_and_extract_with_strong_llm(
        self,
        node_a: NodeInfo,
        node_b: NodeInfo,
        proposed_relation: str,
        summary: str,
    ) -> StrongLLMRelationOutput | None:
        """LLM Strong verify relation và extract FULL INFO.

        Args:
            node_a: Source node info
            node_b: Target node info
            proposed_relation: Relation type từ LLM weak
            summary: Short summary từ LLM weak

        Returns:
            StrongLLMRelationOutput với full info hoặc None nếu fail
        """
        prompt = STRONG_VERIFY_AND_EXTRACT_PROMPT.format(
            node_a_text=node_a.text[:500],
            node_b_text=node_b.text[:500],
            proposed_relation=proposed_relation,
            summary=summary,
        )

        try:
            response = await self._call_strong_llm(prompt)
            data = extract_json_from_llm_response(response)

            return StrongLLMRelationOutput(
                accepted=data.get("accepted", False),
                reasoning=data.get("reasoning", ""),
                id=data.get("id", proposed_relation),
                name=data.get("name", proposed_relation),
                summary=data.get("summary", summary),
                description=data.get("description", ""),
                source_node_types=data.get("source_node_types", ["paragraph"]),
                target_node_types=data.get("target_node_types", ["paragraph"]),
                is_symmetric=data.get("is_symmetric", False),
                is_transitive=data.get("is_transitive", False),
                families=data.get("families", []),
                keywords=data.get("keywords", []),
            )
        except Exception as e:
            self._log.error(f"Strong LLM failed: {e}")
            return None

    # =========================================================================
    # Step 5: Store New Relation to DB
    # =========================================================================

    async def _store_new_relation(
        self,
        output: StrongLLMRelationOutput,
        embedding: list[float] | None,
        tier: str | None,
    ) -> None:
        """Store new relation to ontology_relations collection.

        Includes retry logic for database resilience.
        """
        if not self.ontology_handler:
            self._log.warning("No ontology handler, cannot store relation")
            return

        from arango_client.models.ontology import (
            OntologyRelation,
            RelationStatus,
            RelationTier,
        )

        # Determine tier
        tiers = []
        if tier:
            try:
                tiers.append(RelationTier(tier))
            except ValueError:
                tiers.append(RelationTier.TIER_2_CONTENT)
        else:
            tiers.append(RelationTier.TIER_2_CONTENT)

        relation = OntologyRelation(
            id=output.id,
            name=output.name,
            summary=output.summary,
            description=output.description,
            embedding=embedding,
            source_node_types=output.source_node_types,
            target_node_types=output.target_node_types,
            is_symmetric=output.is_symmetric,
            is_transitive=output.is_transitive,
            is_builtin=False,  # LLM-proposed
            status=RelationStatus.ACTIVE,
            usage_count=0,
            tiers=tiers,
            families=output.families,
            keywords=output.keywords,
        )

        # Use retry wrapper for database operation
        await self._store_relation_with_retry(relation, output.id, output.families)

    @db_retry
    async def _store_relation_with_retry(
        self,
        relation,
        relation_id: str,
        families: list[str],
    ) -> None:
        """Store relation with retry logic for database resilience."""
        self.ontology_handler.create(relation)
        self._log.info(
            "Stored new relation to DB",
            id=relation_id,
            families=families,
        )

    # =========================================================================
    # Helper Methods
    # =========================================================================

    @retry(
        stop=stop_after_attempt(2),
        wait=wait_exponential(multiplier=1, min=2, max=30),
        retry=retry_if_exception_type((TimeoutError, ConnectionError, OSError)),
    )
    async def _call_strong_llm(self, prompt: str) -> str:
        """Call Strong LLM with retry logic.

        Retries up to 2 times on transient failures.
        Uses model key "strong" configured in AIGatewayClient.
        """
        # Get the model name with proper LiteLLM format
        model_name = self.settings.EXTRACTION_STRONG_MODEL

        request = GatewayInvokeRequest(
            model=model_name,
            messages=[
                GatewayChatTurn(
                    role="system",
                    content="You are a knowledge graph expert. Respond only with valid JSON.",
                ),
                GatewayChatTurn(role="user", content=prompt),
            ],
            max_tokens=5000,
            provider_options={"temperature": 0.1},
        )

        response = await self.llm_client.invoke(request)

        if response.message is None:
            return ""

        return response.message.content or ""

    # =========================================================================
    # Batch Operations for Parallel Processing
    # =========================================================================

    async def batch_embed_relations(
        self,
        relations: list[tuple[str, str]],  # List of (name, description)
    ) -> list[list[float] | None]:
        """Batch embed multiple relations in a single API call.

        Much more efficient than calling _embed_relation for each one.

        Args:
            relations: List of (relation_name, description) tuples

        Returns:
            List of embeddings (or None for failed ones)
        """
        if not relations:
            return []

        texts = [f"{name}: {desc}" if desc else name for name, desc in relations]

        try:
            from llm_gateway_client.models.embedding import EmbeddingRequest

            request = EmbeddingRequest(
                model=self.settings.EXTRACTION_EMBEDDING_MODEL,
                input=texts,
            )
            response = await self.llm_client.embed(request)

            if response.embeddings and len(response.embeddings) == len(texts):
                return response.embeddings

            self._log.warning("Batch embed returned wrong number of embeddings")
            return [None] * len(relations)

        except Exception as e:
            self._log.error(f"Batch embed failed: {e}")
            return [None] * len(relations)

    async def parallel_duplicate_check(
        self,
        relations: list[tuple[str, str, list[float]]],  # (name, desc, embedding)
        exclude_ids: list[str] | None = None,
    ) -> list[tuple[str | None, float]]:
        """Parallel duplicate check for multiple relations.

        This is safe to parallelize because it's read-only (no writes).

        Args:
            relations: List of (name, description, embedding) tuples
            exclude_ids: Relation IDs to exclude from comparison

        Returns:
            List of (existing_id, similarity) or (None, 0.0) for each relation
        """
        if not relations or not self.ontology_handler:
            return [(None, 0.0)] * len(relations)

        exclude_ids = exclude_ids or []

        def check_one(
            _name: str,
            _desc: str,
            embedding: list[float],
        ) -> tuple[str | None, float]:
            if not embedding:
                return (None, 0.0)

            result = self.ontology_handler.find_similar_excluding(
                embedding=embedding,
                exclude_ids=exclude_ids,
                threshold=self.settings.EXTRACTION_MERGE_THRESHOLD,
            )
            return result if result else (None, 0.0)

        # Run checks (synchronous since ontology_handler methods are sync)
        results = [check_one(n, d, e) for n, d, e in relations]
        return results

    async def infer_families(
        self,
        relation_name: str,
        description: str,
        embedding: list[float] | None = None,
    ) -> list[str]:
        """Infer families for a new relation based on similarity to existing ones.

        Algorithm:
        1. Find closest existing relation by embedding
        2. Inherit families from that relation
        3. If similarity < 0.6, also add "elaborative" as default

        Args:
            relation_name: Name of the new relation
            description: Description of the relation
            embedding: Pre-computed embedding (optional)

        Returns:
            List of inferred family names
        """
        if not self.ontology_handler:
            return ["elaborative"]  # Default family

        # Get embedding if not provided
        if embedding is None:
            embedding = await self._embed_relation(relation_name, description)

        if not embedding:
            return ["elaborative"]

        # Find closest existing relation
        result = self.ontology_handler.find_similar_excluding(
            embedding=embedding,
            exclude_ids=[],
            threshold=0.3,  # Low threshold to find something
        )

        if result:
            existing_id, similarity = result
            existing_rel = self.ontology_handler.get_by_id(existing_id)

            if existing_rel and hasattr(existing_rel, "families") and existing_rel.families:
                inferred = list(existing_rel.families)

                # If low similarity, add elaborative as fallback
                if similarity < 0.6 and "elaborative" not in inferred:
                    inferred.append("elaborative")

                self._log.info(
                    "Inferred families for new relation",
                    relation=relation_name,
                    closest=existing_id,
                    similarity=f"{similarity:.2f}",
                    families=inferred,
                )
                return inferred

        return ["elaborative"]
