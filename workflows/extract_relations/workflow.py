"""Extract relations workflow implementation.

This module contains the ExtractRelationsWorkflow class that implements
the hybrid relation extraction pipeline for both Tier 1 and Tier 2:

Tier 1 (Structure) - NO MERGE:
- Never merge structure nodes (preserve tree hierarchy)
- High similarity (>= 0.72): Create STRUCTURALLY_EQUIVALENT relation
- Medium similarity (0.40-0.72): LLM determines relation type
- Extract relations with both reviewed nodes (cross-batch) and new nodes (intra-batch)

Tier 2 (Content) - WITH MERGE:
1. Embedding Filter: Fast similarity search
2. Threshold-Based Decision (unified):
   - >= 0.72: Auto-merge (no LLM needed)
   - 0.40-0.72: LLM call (merge check + relation extraction)
   - < 0.40: Keep separate
3. All LLM pairs processed (no budget limit - accuracy-first mode)
"""

from __future__ import annotations

import asyncio
from dataclasses import dataclass, field
from typing import TYPE_CHECKING

from arango_client.handlers import (
    ContentExtractionHandler,
    ExtractionDecision,
    OntologyHandler,
    StructureExtractionHandler,
)
from arango_client.models import RelationTier
from loguru import logger

from .debug_logger import ExtractionDebugLogger, PairDecision, get_debug_logger
from .intra_batch_dedup import find_intra_batch_components
from .llm_extractor import LLMRelationExtractor, NodePair
from .models import (
    ExtractionTier,
    ExtractRelationsInput,
    ExtractRelationsOutput,
    NodeProcessResult,
    TierStats,
)

if TYPE_CHECKING:
    from arango.database import StandardDatabase
    from arango_client.models import ContentNode, StructureNode

    from config.extraction import ExtractionSettings
    from config.ontologies.base import BaseOntology


@dataclass
class CrossBatchCollection:
    """Result of cross-batch pair collection for multi-node processing.

    Collects ALL similar node pairs for all leaders in parallel,
    categorizing them by decision type for efficient batch processing.
    """

    # Pairs where similarity >= auto_merge_threshold (0.72)
    # Format: (leader_node, target_node, similarity_score)
    auto_merge_pairs: list[tuple[ContentNode, ContentNode, float]] = field(default_factory=list)

    # Pairs where llm_threshold <= similarity < auto_merge_threshold (0.40-0.72)
    # Format: (leader_node, target_node, similarity_score)
    llm_pairs: list[tuple[ContentNode, ContentNode, float]] = field(default_factory=list)

    # Leaders that were merged (deleted) - skip all remaining pairs for these
    merged_leader_ids: set[str] = field(default_factory=set)

    # Count of nodes kept separate (no similar nodes found)
    kept_separate_count: int = 0

    # Count of leaders processed
    leaders_processed: int = 0


class ExtractRelationsWorkflow:
    """Implements the extract_relations workflow for Tier 1 and Tier 2.

    Key features:
    - Embedding-based relation filtering (Centroid method)
    - Top-k relation filtering for efficient LLM prompts
    - New relation validation with cosine + LLM strong verify
    - Unified thresholds for both tiers (from settings)
    - Parallel processing with asyncio.gather
    - Batch LLM extraction for efficiency
    - Cross-source relation extraction support
    """

    def __init__(
        self,
        db: StandardDatabase,
        settings: ExtractionSettings,
        ontology: BaseOntology,
        progress_callback: callable | None = None,
    ) -> None:
        self.db = db
        self.settings = settings
        self.ontology = ontology
        self._progress_callback = progress_callback

        # Initialize handlers - OntologyHandler FIRST to share across all components
        # This ensures cache consistency when new relations are added
        self.ontology_handler = OntologyHandler(db)
        self.content_handler = ContentExtractionHandler(
            db, settings, ontology_handler=self.ontology_handler
        )
        self.structure_handler = StructureExtractionHandler(
            db, settings, ontology_handler=self.ontology_handler
        )

        # Initialize LLM extractor with cache invalidation callback
        # When a new relation is stored, invalidate handler caches so they pick up the new relation
        self.llm_extractor = LLMRelationExtractor(
            settings=settings,
            ontology=ontology,
            db=db,
            ontology_handler=self.ontology_handler,
            on_new_relation_stored=self._invalidate_handler_caches,
        )
        self._log = logger.bind(workflow="extract_relations")
        self._debug_logger: ExtractionDebugLogger | None = None

    def _invalidate_handler_caches(self) -> None:
        """Invalidate caches in all extraction handlers.

        Called when a new relation is stored to ensure handlers
        pick up the newly added relation type.
        """
        self.content_handler.invalidate_cache()
        self.structure_handler.invalidate_cache()
        self._log.debug("Invalidated extraction handler caches after new relation stored")

    def enable_debug_logging(self, output_dir: str | None = None) -> None:
        """Enable debug logging for this workflow."""
        self._debug_logger = get_debug_logger(output_dir)

    def _report_progress(self, message: str, **kwargs) -> None:
        """Report progress via callback if set."""
        if self._progress_callback:
            self._progress_callback(message, **kwargs)

    async def run(self, input_data: ExtractRelationsInput) -> ExtractRelationsOutput:
        """Execute the relation extraction workflow."""
        output = ExtractRelationsOutput()

        tier = input_data.tier
        self._log.info(
            "Starting relation extraction",
            tier=tier.value,
            cross_source=input_data.cross_source,
        )

        # Start debug logging if enabled
        if self._debug_logger:
            self._debug_logger.start_run(
                {
                    "auto_merge_threshold": self.settings.EXTRACTION_AUTO_MERGE_THRESHOLD,
                    "llm_threshold": self.settings.EXTRACTION_LLM_THRESHOLD,
                    "llm_batch_size": self.settings.EXTRACTION_LLM_BATCH_SIZE,
                    "cross_source": input_data.cross_source,
                    "fallback_relation": self.settings.EXTRACTION_FALLBACK_RELATION,
                }
            )

        # Process Tier 1 (Structure) if requested
        if tier in (ExtractionTier.STRUCTURE, ExtractionTier.ALL):
            await self._process_structure_tier(input_data, output)

        # Process Tier 2 (Content) if requested
        if tier in (ExtractionTier.CONTENT, ExtractionTier.ALL):
            await self._process_content_tier(input_data, output)

        output.update_totals()

        self._log.info(
            "Relation extraction complete",
            tier=tier.value,
            processed=output.nodes_processed,
            merged=output.nodes_merged,
            relations=output.relations_created,
            llm_calls=output.llm_calls_used,
            errors=len(output.errors),
        )

        # Finish debug logging and generate report
        if self._debug_logger:
            report_path = self._debug_logger.finish_run()
            if report_path:
                self._log.info("Debug report generated", path=str(report_path))
                output.debug_report_path = str(report_path)

        return output

    # =========================================================================
    # Tier 1: Structure Extraction (NO MERGE - Relation Only)
    # =========================================================================

    async def _process_structure_tier(
        self,
        input_data: ExtractRelationsInput,
        output: ExtractRelationsOutput,
    ) -> None:
        """Process Tier 1 structure nodes - extract relations only, NO merge.

        Structure nodes preserve tree hierarchy - never merge.
        High similarity (>= 0.72) creates STRUCTURALLY_EQUIVALENT relation.
        Medium similarity (0.40-0.72) uses LLM to determine relation type.

        Phases:
        - Phase 1: Intra-batch processing (high_similarity_candidates + relation_candidates)
        - Phase 2: Cross-batch processing (compare with reviewed nodes)
        """
        self._log.info("Processing Tier 1 (Structure) - NO MERGE Mode")

        # Initialize tier stats
        tier_stats = TierStats(tier_name="tier_1_structure")

        structures = await self.structure_handler.get_unreviewed_structures(
            source_id=input_data.source_id,
            limit=input_data.batch_size,
        )

        if not structures:
            self._log.info("No structure nodes to process")
            output.tier_stats["tier_1_structure"] = tier_stats
            return

        self._log.info("Found unreviewed structures", count=len(structures))
        tier_stats.nodes_processed = len(structures)

        to_mark_reviewed: list[str] = []

        # ═══════════════════════════════════════════════════════════════════
        # PHASE 1: INTRA-BATCH PROCESSING (NO MERGE)
        # ═══════════════════════════════════════════════════════════════════
        self._log.info("Phase 1: Intra-batch structure processing (no merge)")

        intra_result = find_intra_batch_components(
            nodes=structures,
            merge_threshold=self.settings.EXTRACTION_AUTO_MERGE_THRESHOLD,
            relation_threshold=self.settings.EXTRACTION_LLM_THRESHOLD,
            allow_merge=False,  # CRITICAL: No merge for structure
        )

        # Process high_similarity_candidates -> STRUCTURALLY_EQUIVALENT (no LLM)
        for candidate in intra_result.high_similarity_candidates:
            node_a = structures[candidate.node_a_idx]
            node_b = structures[candidate.node_b_idx]

            if not input_data.dry_run:
                await self.structure_handler.create_structurally_equivalent_relation(
                    source=node_a,
                    target=node_b,
                    similarity=candidate.similarity,
                )

            output.structure_relations_created += 1
            tier_stats.relations_total += 1
            self._log.debug(
                "Created STRUCTURALLY_EQUIVALENT (intra-batch)",
                source=str(node_a.id),
                target=str(node_b.id),
                similarity=candidate.similarity,
            )

        # Process relation_candidates (0.40-0.72) with LLM
        if intra_result.relation_candidates:
            await self._process_structure_llm_batch(
                candidates=intra_result.relation_candidates,
                nodes=structures,
                dry_run=input_data.dry_run,
                tier_stats=tier_stats,
                output=output,
            )

        self._log.info(
            "Phase 1 complete",
            high_similarity_relations=len(intra_result.high_similarity_candidates),
            llm_candidates=len(intra_result.relation_candidates),
        )

        # ═══════════════════════════════════════════════════════════════════
        # PHASE 2: CROSS-BATCH PROCESSING
        # ═══════════════════════════════════════════════════════════════════
        self._log.info("Phase 2: Cross-batch structure processing")

        processed_ids: set[str] = set()

        for node in structures:
            node_id = str(node.id)

            if node_id in processed_ids:
                continue

            try:
                result = await self._process_structure_node_no_merge(
                    node,
                    input_data.dry_run,
                    processed_ids,
                    input_data.cross_source,
                )

                processed_ids.add(node_id)
                output.structure_nodes_processed += 1

                if result.relation_created:
                    output.structure_relations_created += 1
                    tier_stats.relations_total += 1
                    if result.is_in_top_k:
                        tier_stats.relations_in_top_k += 1
                    if result.used_fallback:
                        tier_stats.relations_fallback += 1
                    if result.is_new_relation:
                        tier_stats.relations_new += 1

                if result.kept_separate:
                    tier_stats.kept_separate += 1

                if not input_data.dry_run:
                    to_mark_reviewed.append(node_id)

            except Exception as e:
                self._log.exception(
                    "Error processing structure node",
                    node_id=node_id,
                    error=str(e),
                )
                output.errors.append({"node_id": node_id, "error": str(e), "tier": "structure"})
                tier_stats.errors += 1

        # Batch mark as reviewed
        if to_mark_reviewed:
            await self.structure_handler.mark_as_reviewed(to_mark_reviewed)

        # Store tier stats in output
        output.tier_stats["tier_1_structure"] = tier_stats

    async def _process_structure_node_no_merge(
        self,
        node: StructureNode,
        dry_run: bool,
        exclude_ids: set[str],
        cross_source: bool,
    ) -> NodeProcessResult:
        """Process a single structure node - extract relation only, no merge.

        High similarity (>= 0.72): Create STRUCTURALLY_EQUIVALENT relation
        Medium similarity (0.40-0.72): Use LLM to determine relation
        """
        result = NodeProcessResult()

        # Find similar structures by embedding (cross-batch with reviewed nodes)
        # No limit - return all candidates above threshold
        similar = await self.structure_handler.find_similar_structures(
            node,
            exclude_ids=exclude_ids,
            cross_source=cross_source,
        )

        if not similar:
            result.kept_separate = True
            return result

        # Process ALL similar nodes (not just best match)
        for match in similar:
            if match.similarity_score >= self.settings.EXTRACTION_AUTO_MERGE_THRESHOLD:
                # High similarity -> STRUCTURALLY_EQUIVALENT (no LLM)
                if not dry_run:
                    await self.structure_handler.create_structurally_equivalent_relation(
                        source=node,
                        target=match.node,
                        similarity=match.similarity_score,
                    )
                result.relation_created = True
                result.relation_type = "STRUCTURALLY_EQUIVALENT"
                result.target_id = match.node_id
                self._log.debug(
                    "Created STRUCTURALLY_EQUIVALENT (cross-batch)",
                    source=str(node.id),
                    target=match.node_id,
                    similarity=match.similarity_score,
                )
            elif match.similarity_score >= self.settings.EXTRACTION_LLM_THRESHOLD:
                # Medium similarity -> Use LLM to determine relation type
                llm_result = await self.llm_extractor.verify_pair(
                    node_a_type="structure",
                    node_a_text=self._get_structure_text(node),
                    node_b_type="structure",
                    node_b_text=self._get_structure_text(match.node),
                    similarity_score=match.similarity_score,
                    node_a_embedding=node.embedding,
                    node_b_embedding=match.node.embedding,
                    tier=RelationTier.TIER_1_STRUCTURE,
                )

                if llm_result.should_merge:
                    # Structure tier: NEVER merge, create STRUCTURALLY_EQUIVALENT instead
                    if not dry_run:
                        await self.structure_handler.create_structurally_equivalent_relation(
                            source=node,
                            target=match.node,
                            similarity=match.similarity_score,
                        )
                    result.relation_created = True
                    result.relation_type = "STRUCTURALLY_EQUIVALENT"
                    result.target_id = match.node_id
                    self._log.debug(
                        "LLM said merge, created STRUCTURALLY_EQUIVALENT (cross-batch)",
                        source=str(node.id),
                        target=match.node_id,
                    )
                elif llm_result.relation_type is not None:
                    # Create the relation LLM determined
                    if not dry_run:
                        await self.structure_handler.create_structure_relation(
                            source=node,
                            target=match.node,
                            relation_type=llm_result.relation_type,
                            confidence=llm_result.confidence,
                            method="llm_verified",
                        )
                    result.relation_created = True
                    result.relation_type = llm_result.relation_type
                    result.is_in_top_k = llm_result.is_in_top_k
                    result.is_new_relation = llm_result.is_new_relation
                    result.target_id = match.node_id
                    self._log.debug(
                        "Created LLM-verified relation (cross-batch)",
                        source=str(node.id),
                        target=match.node_id,
                        relation_type=llm_result.relation_type,
                    )
                else:
                    # LLM determined NO relation exists - keep separate
                    result.kept_separate = True
                    self._log.debug(
                        "LLM determined no relation (cross-batch)",
                        source=str(node.id),
                        target=match.node_id,
                    )

        return result

    async def _process_structure_llm_batch(
        self,
        candidates: list,
        nodes: list[StructureNode],
        dry_run: bool,
        tier_stats: TierStats,
        output: ExtractRelationsOutput,
    ) -> None:
        """Process structure relation candidates with LLM.

        For structure tier, if LLM says should_merge=true, create
        STRUCTURALLY_EQUIVALENT relation instead of merging.
        """
        self._log.info("Processing structure LLM batch", candidate_count=len(candidates))

        # Convert to NodePair format with embeddings for Centroid Filtering
        node_pairs = []
        for candidate in candidates:
            node_a = nodes[candidate.node_a_idx]
            node_b = nodes[candidate.node_b_idx]
            node_pairs.append(
                NodePair(
                    node_a_id=str(node_a.id),
                    node_a_type="structure",
                    node_a_text=self._get_structure_text(node_a),
                    node_b_id=str(node_b.id),
                    node_b_type="structure",
                    node_b_text=self._get_structure_text(node_b),
                    similarity_score=candidate.similarity,
                    node_a_embedding=node_a.embedding,
                    node_b_embedding=node_b.embedding,
                )
            )

        # Split into batches for parallel processing
        batch_size = self.settings.EXTRACTION_LLM_BATCH_SIZE
        batches = [node_pairs[i : i + batch_size] for i in range(0, len(node_pairs), batch_size)]
        total_batches = len(batches)

        self._log.info(
            "Processing structure LLM batches in parallel",
            total_batches=total_batches,
            max_concurrent=self.settings.MAX_CONCURRENT_LLM_CALLS,
        )
        self._report_progress(
            f"Processing {total_batches} structure LLM batches in parallel",
            total_batches=total_batches,
            max_concurrent=self.settings.MAX_CONCURRENT_LLM_CALLS,
        )

        # Use verify_batches_parallel for concurrent LLM calls
        results = await self.llm_extractor.verify_batches_parallel(
            batches, tier=RelationTier.TIER_1_STRUCTURE
        )
        output.llm_calls_used += total_batches
        tier_stats.pairs_sent_to_llm += len(node_pairs)

        # Apply results
        for idx, candidate in enumerate(candidates):
            if idx >= len(results):
                break

            node_a = nodes[candidate.node_a_idx]
            node_b = nodes[candidate.node_b_idx]
            llm_result = results[idx]

            if llm_result.should_merge:
                # Structure tier: NEVER merge, create STRUCTURALLY_EQUIVALENT instead
                if not dry_run:
                    await self.structure_handler.create_structurally_equivalent_relation(
                        source=node_a,
                        target=node_b,
                        similarity=candidate.similarity,
                    )
                output.structure_relations_created += 1
                tier_stats.relations_total += 1
                self._log.debug(
                    "LLM said merge, created STRUCTURALLY_EQUIVALENT instead",
                    source=str(node_a.id),
                    target=str(node_b.id),
                )
            elif llm_result.relation_type is not None:
                # Create the relation LLM determined
                if not dry_run:
                    await self.structure_handler.create_structure_relation(
                        source=node_a,
                        target=node_b,
                        relation_type=llm_result.relation_type,
                        confidence=llm_result.confidence,
                        method="llm_verified",
                    )
                output.structure_relations_created += 1
                tier_stats.relations_total += 1
                if llm_result.is_in_top_k:
                    tier_stats.relations_in_top_k += 1
                if llm_result.is_new_relation:
                    tier_stats.relations_new += 1
            # else: relation_type=null means no relation, keep separate

    def _get_structure_text(self, node: StructureNode) -> str:
        """Get representative text from structure node.

        Combines title (from attributes) and summary (from node) for richer context.
        """
        parts = []

        # Get title from attributes
        attrs = getattr(node, "attributes", None)
        if attrs:
            code = getattr(attrs, "code", "")
            if code:
                parts.append(code)
            title = getattr(attrs, "title", "")
            if title:
                parts.append(title)

        # Get summary directly from node (not from attributes!)
        summary = getattr(node, "summary", None)
        if summary:
            parts.append(summary)

        return " - ".join(parts) if parts else ""

    # =========================================================================
    # Tier 2: Content Extraction (Two-Phase)
    # =========================================================================

    async def _process_content_tier(
        self,
        input_data: ExtractRelationsInput,
        output: ExtractRelationsOutput,
    ) -> None:
        """Process Tier 2 content nodes with Two-Phase extraction.

        Two-Phase extraction ensures intra-batch duplicates are caught:
        - Phase 1: Compute similarity matrix within batch, merge connected components
        - Phase 2: Check remaining nodes against reviewed nodes (cross-batch)
        - Phase 3: Create intra-batch relations for medium-similarity pairs
        """
        self._log.info("Processing Tier 2 (Content) - Two-Phase Mode")

        # Initialize tier stats
        tier_stats = TierStats(tier_name="tier_2_content")

        # Get nodes to process
        if input_data.nodes:
            nodes = input_data.nodes
        elif input_data.source_id:
            nodes = await self.content_handler.get_unreviewed_nodes(
                source_id=input_data.source_id,
                limit=input_data.batch_size,
            )
        else:
            nodes = await self.content_handler.get_unreviewed_nodes(
                limit=input_data.batch_size,
            )

        if not nodes:
            self._log.info("No content nodes to process")
            output.tier_stats["tier_2_content"] = tier_stats
            return

        self._log.info("Found unreviewed content nodes", count=len(nodes))
        tier_stats.nodes_processed = len(nodes)

        # Debug logging: log tier start
        if self._debug_logger:
            node_ids = [str(n.id) for n in nodes]
            node_labels = [
                self._get_node_text(n)[:50] if hasattr(n, "attributes") else str(n.id)[:8]
                for n in nodes
            ]
            self._debug_logger.log_tier_start("tier_2_content", len(nodes), node_ids, node_labels)

        # ═══════════════════════════════════════════════════════════════════
        # PHASE 1: INTRA-BATCH DEDUPLICATION
        # Compute similarity matrix within batch, merge connected components
        # ═══════════════════════════════════════════════════════════════════
        self._log.info("Phase 1: Intra-batch deduplication")
        self._report_progress("Phase 1: Intra-batch deduplication", nodes=len(nodes))

        intra_result = find_intra_batch_components(
            nodes=nodes,
            merge_threshold=self.settings.EXTRACTION_AUTO_MERGE_THRESHOLD,
            relation_threshold=self.settings.EXTRACTION_LLM_THRESHOLD,
            allow_merge=True,  # Content tier allows merge
        )

        # Debug logging: log intra-batch result
        if self._debug_logger:
            self._debug_logger.log_intra_batch_result("tier_2_content", intra_result)

        # Execute merge operations (SEQUENTIAL - safe, no race condition)
        # Create ID->node lookup for O(1) access
        node_by_id = {str(n.id): n for n in nodes}
        merged_ids: set[str] = set()

        for merge_op in intra_result.merge_operations:
            source_node = node_by_id.get(merge_op.source_id)
            target_node = node_by_id.get(merge_op.target_id)

            if source_node and target_node and not input_data.dry_run:
                await self.content_handler.auto_merge(
                    source_node,
                    target_node,
                    similarity=merge_op.similarity,
                    merge_reason="auto_merge",
                    llm_reasoning=None,
                )

            merged_ids.add(merge_op.source_id)
            tier_stats.auto_merged += 1
            output.content_nodes_merged += 1
            self._log.debug(
                "Intra-batch merge",
                source=merge_op.source_id,
                target=merge_op.target_id,
                similarity=merge_op.similarity,
            )

        self._log.info(
            "Phase 1 complete",
            leaders=len(intra_result.leaders),
            merged=len(intra_result.merge_operations),
            relation_candidates=len(intra_result.relation_candidates),
        )
        self._report_progress(
            "Phase 1 complete",
            leaders=len(intra_result.leaders),
            merged=len(intra_result.merge_operations),
        )

        # ═══════════════════════════════════════════════════════════════════
        # PHASE 2: CROSS-BATCH PROCESSING (REFACTORED - Multi-Node)
        # Collect ALL similar pairs for ALL leaders, then process in batch
        # Key fix: Process ALL similar nodes >= 0.40, not just best match
        # ═══════════════════════════════════════════════════════════════════
        self._log.info("Phase 2: Cross-batch processing (multi-node)")
        self._report_progress("Phase 2: Cross-batch processing", leaders=len(intra_result.leaders))

        processed_ids: set[str] = set(merged_ids)
        to_mark_reviewed: list[str] = []

        # Step 1: Collect ALL pairs for ALL leaders in parallel
        # This replaces the old sequential loop that only took similar_nodes[0]
        collection = await self._collect_cross_batch_pairs(
            leaders=intra_result.leaders,
            processed_ids=processed_ids,
            cross_source=input_data.cross_source,
        )

        # Track leaders that have been processed (merged or relation created)
        processed_leader_ids: set[str] = set()

        # Step 2: Execute auto-merges FIRST
        # IMPORTANT: Once a leader is merged, skip all remaining pairs for that leader
        # Sort by similarity descending to merge with best match first
        auto_merge_sorted = sorted(
            collection.auto_merge_pairs,
            key=lambda x: x[2],
            reverse=True,
        )

        for source, target, similarity in auto_merge_sorted:
            source_id = str(source.id)
            target_id = str(target.id)

            # Skip if leader was already merged
            if source_id in collection.merged_leader_ids:
                continue

            # Execute merge
            if not input_data.dry_run:
                await self.content_handler.auto_merge(
                    source,
                    target,
                    similarity=similarity,
                    merge_reason="auto_merge",
                    llm_reasoning=None,
                )

            # Mark leader as merged - skip all remaining pairs for this leader
            collection.merged_leader_ids.add(source_id)
            processed_ids.add(source_id)
            processed_ids.add(target_id)
            processed_leader_ids.add(source_id)

            output.content_nodes_processed += 1
            output.content_nodes_merged += 1
            tier_stats.auto_merged += 1

            self._log.debug(
                "Cross-batch auto-merge (multi-node)",
                source=source_id,
                target=target_id,
                similarity=similarity,
            )

        self._log.info(
            "Phase 2 auto-merges complete",
            merged_count=len(collection.merged_leader_ids),
        )

        # Step 3: Filter LLM pairs - exclude pairs where leader was merged
        valid_llm_pairs = [
            (source, target, sim)
            for source, target, sim in collection.llm_pairs
            if str(source.id) not in collection.merged_leader_ids
        ]

        self._log.info(
            "Filtered LLM pairs after merges",
            original=len(collection.llm_pairs),
            valid=len(valid_llm_pairs),
        )

        # Step 4: Process ALL LLM pairs (no budget limit - accuracy-first mode)
        llm_pairs_to_process: list[tuple[ContentNode, ContentNode, float]] = []

        for source, target, similarity in valid_llm_pairs:
            llm_pairs_to_process.append((source, target, similarity))
            tier_stats.pairs_sent_to_llm += 1

        # Process LLM pairs from Phase 2 (cross-batch)
        if llm_pairs_to_process:
            await self._process_llm_batch(
                llm_pairs_to_process,
                input_data.dry_run,
                processed_ids,
                to_mark_reviewed,
                output,
                tier_stats,
            )

        # Mark leaders that were kept separate
        for leader in intra_result.leaders:
            leader_id = str(leader.id)
            if leader_id not in processed_leader_ids and leader_id not in merged_ids:
                # This leader had no similar nodes or all were below threshold
                output.nodes_kept_separate += 1
                tier_stats.kept_separate += 1
                output.content_nodes_processed += 1
                to_mark_reviewed.append(leader_id)
                processed_ids.add(leader_id)

        # ═══════════════════════════════════════════════════════════════════
        # PHASE 3: INTRA-BATCH RELATIONS
        # Create relations for medium-similarity pairs within batch
        # ═══════════════════════════════════════════════════════════════════
        if intra_result.relation_candidates:
            self._log.info(
                "Phase 3: Intra-batch relations",
                candidates=len(intra_result.relation_candidates),
            )

            # Filter candidates - both nodes must still exist (not merged)
            valid_candidates = [
                c
                for c in intra_result.relation_candidates
                if str(nodes[c.node_a_idx].id) not in merged_ids
                and str(nodes[c.node_b_idx].id) not in merged_ids
            ]

            # Queue ALL for LLM processing (no budget limit - accuracy-first mode)
            intra_llm_pairs: list[tuple[ContentNode, ContentNode, float]] = []
            for candidate in valid_candidates:
                node_a = nodes[candidate.node_a_idx]
                node_b = nodes[candidate.node_b_idx]
                intra_llm_pairs.append((node_a, node_b, candidate.similarity))
                tier_stats.pairs_sent_to_llm += 1

            # Process intra-batch LLM pairs
            if intra_llm_pairs:
                await self._process_llm_batch(
                    intra_llm_pairs,
                    input_data.dry_run,
                    processed_ids,
                    to_mark_reviewed,
                    output,
                    tier_stats,
                )

        # Batch mark as reviewed
        if to_mark_reviewed:
            await self.content_handler.mark_as_reviewed(to_mark_reviewed)

        # Store tier stats in output
        output.tier_stats["tier_2_content"] = tier_stats

    async def _process_llm_batch(
        self,
        pairs: list[tuple[ContentNode, ContentNode, float]],
        dry_run: bool,
        processed_ids: set[str],
        to_mark_reviewed: list[str],
        output: ExtractRelationsOutput,
        tier_stats: TierStats,
    ) -> None:
        """Process pairs with batch LLM extraction."""
        self._log.info("Processing LLM batch", pair_count=len(pairs))

        # Convert to NodePair format for batch extraction with embeddings for Centroid Filtering
        # Skip pairs where either node has empty text (prevents useless LLM calls)
        node_pairs = []
        valid_pair_indices = []  # Track which original pairs are valid
        skipped_count = 0

        for idx, (node_a, node_b, similarity) in enumerate(pairs):
            text_a = self._get_node_text(node_a)
            text_b = self._get_node_text(node_b)

            # Skip pairs with empty content - no point in calling LLM
            if not text_a or not text_b:
                skipped_count += 1
                self._log.debug(
                    "Skipping pair with empty content",
                    node_a_id=str(node_a.id),
                    node_b_id=str(node_b.id),
                    text_a_empty=not text_a,
                    text_b_empty=not text_b,
                )
                continue

            node_pairs.append(
                NodePair(
                    node_a_id=str(node_a.id),
                    node_a_type=self._get_node_type_str(node_a),
                    node_a_text=text_a,
                    node_b_id=str(node_b.id),
                    node_b_type=self._get_node_type_str(node_b),
                    node_b_text=text_b,
                    similarity_score=similarity,
                    node_a_embedding=node_a.embedding,
                    node_b_embedding=node_b.embedding,
                )
            )
            valid_pair_indices.append(idx)

        if skipped_count > 0:
            self._log.warning(
                "Skipped pairs with empty content",
                skipped=skipped_count,
                valid=len(node_pairs),
            )

        if not node_pairs:
            self._log.info("No valid pairs to process after filtering empty content")
            return

        # Split into batches for parallel processing
        batch_size = self.settings.EXTRACTION_LLM_BATCH_SIZE
        batches = [node_pairs[i : i + batch_size] for i in range(0, len(node_pairs), batch_size)]
        total_batches = len(batches)

        self._log.info(
            "Processing LLM batches in parallel",
            total_batches=total_batches,
            max_concurrent=self.settings.MAX_CONCURRENT_LLM_CALLS,
            pairs_per_batch=batch_size,
        )
        self._report_progress(
            f"Processing {total_batches} LLM batches in parallel",
            total_batches=total_batches,
            max_concurrent=self.settings.MAX_CONCURRENT_LLM_CALLS,
        )

        # Use verify_batches_parallel for concurrent LLM calls
        results = await self.llm_extractor.verify_batches_parallel(
            batches, tier=RelationTier.TIER_2_CONTENT
        )
        output.llm_calls_used += total_batches

        # Debug logging: log batch summary
        if self._debug_logger:
            batch_debug_results = [
                {
                    "should_merge": r.should_merge,
                    "relation_type": r.relation_type,
                    "confidence": r.confidence,
                    "reasoning": r.reasoning[:100] if r.reasoning else "",
                }
                for r in results
            ]
            self._debug_logger.log_llm_batch("tier_2_content", len(results), batch_debug_results)

        self._log.info("LLM parallel batch processing complete", total_results=len(results))

        # Apply results - use valid_pair_indices to map back to original pairs
        for result_idx, llm_result in enumerate(results):
            if result_idx >= len(valid_pair_indices):
                break

            original_idx = valid_pair_indices[result_idx]
            node_a, node_b, pair_similarity = pairs[original_idx]
            node_a_id = str(node_a.id)
            node_b_id = str(node_b.id)

            if llm_result.should_merge:
                if not dry_run:
                    # Check if nodes still exist before merge
                    node_a_exists = await self.content_handler.node_exists(node_a)
                    node_b_exists = await self.content_handler.node_exists(node_b)
                    if node_a_exists and node_b_exists:
                        await self.content_handler.auto_merge(
                            node_a,
                            node_b,
                            similarity=pair_similarity,
                            merge_reason="llm_merge",
                            llm_reasoning=llm_result.reasoning,
                        )
                    else:
                        self._log.debug(
                            "Skipping merge - node no longer exists",
                            source=node_a_id,
                            target=node_b_id,
                            source_exists=node_a_exists,
                            target_exists=node_b_exists,
                        )
                        continue
                processed_ids.add(node_a_id)
                processed_ids.add(node_b_id)
                output.content_nodes_processed += 1
                output.content_nodes_merged += 1
                tier_stats.llm_merged += 1

                # Debug logging: log merge decision
                if self._debug_logger:
                    self._debug_logger.log_decision(
                        "tier_2_content",
                        PairDecision(
                            node_a_id=node_a_id,
                            node_a_type=self._get_node_type_str(node_a),
                            node_a_text=self._get_node_text(node_a)[:50],
                            node_b_id=node_b_id,
                            node_b_type=self._get_node_type_str(node_b),
                            node_b_text=self._get_node_text(node_b)[:50],
                            similarity=pairs[original_idx][2],
                            decision="merge_llm",
                            llm_used=True,
                            llm_reasoning=llm_result.reasoning,
                            phase="cross_batch",
                        ),
                    )

                self._log.debug(
                    "LLM verified merge",
                    source=node_a_id,
                    target=node_b_id,
                    reasoning=llm_result.reasoning,
                )
            elif llm_result.relation_type is not None:
                # LLM determined there IS a relation
                relation_type = llm_result.relation_type
                if not dry_run:
                    # Check if nodes still exist before creating relation
                    node_a_exists = await self.content_handler.node_exists(node_a)
                    node_b_exists = await self.content_handler.node_exists(node_b)
                    if not (node_a_exists and node_b_exists):
                        self._log.debug(
                            "Skipping relation - node no longer exists",
                            source=node_a_id,
                            target=node_b_id,
                            source_exists=node_a_exists,
                            target_exists=node_b_exists,
                        )
                        continue
                    await self.content_handler.create_relation_edge(
                        source_node=node_a,
                        target_node=node_b,
                        relation_type=relation_type,
                        confidence=llm_result.confidence,
                        method="llm_verified",
                    )
                processed_ids.add(node_a_id)
                output.content_nodes_processed += 1
                output.content_relations_created += 1
                tier_stats.relations_total += 1

                # Track relation statistics
                if llm_result.is_in_top_k:
                    tier_stats.relations_in_top_k += 1
                if llm_result.used_fallback:
                    tier_stats.relations_fallback += 1
                if llm_result.is_new_relation:
                    tier_stats.relations_new += 1
                    output.new_relations_proposed += 1
                    # Track approved and merged new relations
                    if llm_result.approved_as_new:
                        output.new_relations_approved += 1
                    if llm_result.merged_with_existing:
                        output.new_relations_merged += 1

                to_mark_reviewed.append(node_a_id)

                # Debug logging: log relation decision
                if self._debug_logger:
                    self._debug_logger.log_decision(
                        "tier_2_content",
                        PairDecision(
                            node_a_id=node_a_id,
                            node_a_type=self._get_node_type_str(node_a),
                            node_a_text=self._get_node_text(node_a)[:50],
                            node_b_id=node_b_id,
                            node_b_type=self._get_node_type_str(node_b),
                            node_b_text=self._get_node_text(node_b)[:50],
                            similarity=pairs[original_idx][2],
                            decision="relation",
                            relation_type=relation_type,
                            llm_used=True,
                            llm_reasoning=llm_result.reasoning,
                            phase="cross_batch",
                        ),
                    )

                self._log.debug(
                    "LLM verified relation",
                    source=node_a_id,
                    target=node_b_id,
                    relation=relation_type,
                    reasoning=llm_result.reasoning,
                )
            else:
                # LLM determined NO relation exists - keep separate
                # DO NOT create fallback relation, just mark as processed
                processed_ids.add(node_a_id)
                output.content_nodes_processed += 1
                output.nodes_kept_separate += 1
                tier_stats.kept_separate += 1
                to_mark_reviewed.append(node_a_id)

                self._log.debug(
                    "LLM determined no relation - kept separate",
                    source=node_a_id,
                    target=node_b_id,
                    reasoning=llm_result.reasoning,
                )

        # Report LLM batch results summary
        self._report_progress(
            "LLM batch complete",
            result={
                "merged": tier_stats.llm_merged,
                "relations": tier_stats.relations_total,
                "kept_separate": tier_stats.kept_separate,
            },
        )

    def _get_node_type_str(self, node: ContentNode) -> str:
        """Get node type as string."""
        if hasattr(node.node_type, "value"):
            return node.node_type.value
        return str(node.node_type)

    def _get_node_text(self, node: ContentNode) -> str:
        """Get representative text from node.

        Uses ContentNode.get_content_text() which handles:
        - Top-level text field (used for embedding)
        - Type-specific attributes (text, alt_text, extracted_text, caption)
        """
        # ContentNode has get_content_text() method that properly handles
        # both top-level text field and type-specific attributes (dict)
        return node.get_content_text()

    async def _collect_cross_batch_pairs(
        self,
        leaders: list[ContentNode],
        processed_ids: set[str],
        cross_source: bool,
    ) -> CrossBatchCollection:
        """Collect ALL similar node pairs for all leaders in parallel.

        This method processes ALL similar nodes for each leader, not just the best match.
        Uses Semaphore + asyncio.gather for concurrent similarity searches.

        For each leader:
        1. Find ALL similar nodes >= LLM_THRESHOLD (0.40)
        2. Apply similarity decay cutoff (stop at 85% of best)
        3. Categorize pairs: auto_merge (>= 0.72) or llm (0.40-0.72)

        Args:
            leaders: Leader nodes from intra-batch deduplication
            processed_ids: Set of already processed node IDs
            cross_source: Whether to search across different sources

        Returns:
            CrossBatchCollection with categorized pairs
        """
        collection = CrossBatchCollection()
        semaphore = asyncio.Semaphore(self.settings.EXTRACTION_MAX_CONCURRENT_COLLECTIONS)

        async def collect_for_leader(
            leader: ContentNode,
        ) -> tuple[str, list[tuple[ContentNode, float]], bool]:
            """Collect similar nodes for a single leader.

            Returns:
                (leader_id, list of (target_node, similarity), node_exists)
            """
            leader_id = str(leader.id)

            async with semaphore:
                # Check if node still exists
                exists = await self.content_handler.node_exists(leader)
                if not exists:
                    return (leader_id, [], False)

                # Find ALL similar REVIEWED nodes (cross-batch)
                similar_nodes = await self.content_handler.find_similar_nodes(
                    leader, cross_source=cross_source
                )

                if not similar_nodes:
                    return (leader_id, [], True)

                # Filter out already processed nodes
                similar_nodes = [s for s in similar_nodes if s.node_id not in processed_ids]

                if not similar_nodes:
                    return (leader_id, [], True)

                # Apply similarity decay cutoff
                # Stop processing when similarity drops below best * decay_factor
                best_sim = similar_nodes[0].similarity_score
                decay_cutoff = best_sim * self.settings.EXTRACTION_SIMILARITY_DECAY_FACTOR
                min_threshold = self.settings.EXTRACTION_LLM_THRESHOLD

                filtered_nodes = []
                for s in similar_nodes:
                    if s.similarity_score < max(decay_cutoff, min_threshold):
                        break  # Sorted by similarity, so we can stop here
                    filtered_nodes.append((s.node, s.similarity_score))

                    # Apply max relations per leader limit
                    if len(filtered_nodes) >= self.settings.EXTRACTION_MAX_RELATIONS_PER_LEADER:
                        break

                return (leader_id, filtered_nodes, True)

        # Parallel collection for all leaders
        tasks = [collect_for_leader(leader) for leader in leaders]
        results = await asyncio.gather(*tasks, return_exceptions=True)

        # Aggregate results
        for i, result in enumerate(results):
            if isinstance(result, Exception):
                self._log.exception(
                    "Error collecting pairs for leader",
                    leader_id=str(leaders[i].id),
                    error=str(result),
                )
                continue

            leader_id, similar_pairs, node_exists = result

            if not node_exists:
                continue

            collection.leaders_processed += 1

            if not similar_pairs:
                collection.kept_separate_count += 1
                continue

            # Get leader node for pairing
            leader = leaders[i]

            # Categorize pairs by decision type
            for target_node, similarity in similar_pairs:
                decision = self.content_handler.decide_action(similarity)

                if decision == ExtractionDecision.MERGE:
                    collection.auto_merge_pairs.append((leader, target_node, similarity))
                elif decision == ExtractionDecision.CREATE_RELATION:
                    collection.llm_pairs.append((leader, target_node, similarity))
                # KEEP_SEPARATE pairs are not added (below threshold)

        self._log.info(
            "Cross-batch pair collection complete",
            leaders_processed=collection.leaders_processed,
            auto_merge_pairs=len(collection.auto_merge_pairs),
            llm_pairs=len(collection.llm_pairs),
            kept_separate=collection.kept_separate_count,
        )

        # Report collection result for progress tracking
        self._report_progress(
            "Cross-batch collection",
            auto_merge=len(collection.auto_merge_pairs),
            llm_pairs=len(collection.llm_pairs),
        )

        return collection
