#!/usr/bin/env python3
"""Script to run extraction on unreviewed nodes in ArangoDB.

This script:
1. Finds all nodes with is_reviewed=false
2. Runs extraction process (embedding similarity + threshold decisions)
3. Optionally uses LLM to extract semantic relations (not just SIMILAR_TO)
4. Supports continuous mode to process ALL unreviewed nodes in batches

Usage:
    python scripts/run_extraction.py
    python scripts/run_extraction.py --continuous
    python scripts/run_extraction.py --use-llm
    python scripts/run_extraction.py --continuous --use-llm --batch-size 50
"""

from __future__ import annotations

import argparse
import asyncio
import sys
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import TYPE_CHECKING, Any

from loguru import logger

# Add project root to path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from arango_client.client import ArangoClient
from arango_client.collections.schema import (
    Collections,
    EdgeCollections,
    RelationshipTypes,
)
from arango_client.handlers.content_extraction_handler import (
    ContentExtractionHandler,
    ExtractionDecision,
)
from arango_client.handlers.ontology_handler import OntologyHandler
from arango_client.handlers.structure_extraction_handler import (
    StructureExtractionHandler,
)
from arango_client.models.ontology import RelationTier

from config.extraction import get_extraction_settings
from workflows.dependencies import init_arango_client

if TYPE_CHECKING:
    from arango_client.models import ContentNode, StructureNode

    from workflows.extract_relations.debug_logger import ExtractionDebugLogger
    from workflows.extract_relations.llm_extractor import LLMRelationExtractor


# =============================================================================
# LLM Batch Helper Types
# =============================================================================
@dataclass
class LLMPendingPair:
    """A pair pending LLM verification."""

    source_node: ContentNode | StructureNode
    target_node: ContentNode | StructureNode
    similarity: float
    node_index: int  # Index in original list for progress tracking


# =============================================================================
# Statistics Tracking
# =============================================================================
@dataclass
class TierStats:
    """Detailed statistics for a single tier extraction."""

    tier_name: str = ""

    # Node processing
    nodes_processed: int = 0

    # Merge stats
    auto_merged: int = 0  # embedding > 0.72
    llm_merged: int = 0  # LLM decided to merge

    # LLM extraction stats
    pairs_sent_to_llm: int = 0  # pairs sent to LLM (0.40-0.72)

    # Relation stats
    relations_in_top_k: int = 0  # relations in top-k ontology list
    relations_fallback: int = 0  # fallback to SIMILAR_TO
    relations_new: int = 0  # new relations proposed by LLM
    relations_total: int = 0  # total relations created

    # Other
    kept_separate: int = 0
    errors: int = 0


@dataclass
class ExtractionStats:
    """Track extraction run statistics."""

    # Counts
    total_unreviewed: int = 0
    nodes_processed: int = 0
    nodes_merged: int = 0
    relations_created: int = 0
    nodes_kept_separate: int = 0
    nodes_marked_reviewed: int = 0

    # Batches
    batches_completed: int = 0

    # LLM
    llm_calls: int = 0

    # Errors
    errors: int = 0
    error_details: list[str] = field(default_factory=list)

    # Relations by type
    relations_by_type: dict[str, int] = field(default_factory=dict)
    relations_outside_ontology: list[str] = field(default_factory=list)

    # Timing
    start_time: float = 0.0
    end_time: float = 0.0

    # Merge details
    merge_pairs: list[tuple[str, str]] = field(default_factory=list)

    # Per-tier detailed statistics
    tier_stats: dict[str, TierStats] = field(default_factory=dict)

    def add_relation(self, relation_type: str, is_in_ontology: bool) -> None:
        """Track a new relation."""
        self.relations_by_type[relation_type] = self.relations_by_type.get(relation_type, 0) + 1
        if not is_in_ontology:
            if relation_type not in self.relations_outside_ontology:
                self.relations_outside_ontology.append(relation_type)

    def get_duration(self) -> float:
        """Get run duration in seconds."""
        if self.end_time > 0:
            return self.end_time - self.start_time
        return time.time() - self.start_time


# =============================================================================
# Edge Statistics Query
# =============================================================================
def get_edge_stats_before(client: ArangoClient) -> dict[str, Any]:
    """Get edge statistics before extraction."""
    db = client.db
    stats = {
        "total_edges": 0,
        "edges_by_type": {},
        "dedup_edges": 0,
    }

    # Count total edges
    query_total = """
    RETURN LENGTH(@@edge_collection)
    """
    cursor = db.aql.execute(
        query_total, bind_vars={"@edge_collection": EdgeCollections.KNOWLEDGE_EDGES}
    )
    stats["total_edges"] = next(cursor, 0)

    # Count edges by type
    query_by_type = """
    FOR e IN @@edge_collection
        COLLECT type = e.type WITH COUNT INTO count
        RETURN { type: type, count: count }
    """
    cursor = db.aql.execute(
        query_by_type, bind_vars={"@edge_collection": EdgeCollections.KNOWLEDGE_EDGES}
    )
    for item in cursor:
        if item["type"]:
            stats["edges_by_type"][item["type"]] = item["count"]

    # Count edges created by extraction (have detection_method field)
    query_dedup = """
    FOR e IN @@edge_collection
        FILTER e.detection_method != null
        COLLECT WITH COUNT INTO count
        RETURN count
    """
    cursor = db.aql.execute(
        query_dedup, bind_vars={"@edge_collection": EdgeCollections.KNOWLEDGE_EDGES}
    )
    stats["dedup_edges"] = next(cursor, 0)

    return stats


def reset_all_nodes_to_unreviewed(client: ArangoClient) -> dict[str, int]:
    """Reset all nodes to is_reviewed=false for testing purposes.

    Returns:
        Dict with counts of reset nodes per collection.
    """
    db = client.db
    reset_counts = {}

    for collection_name in [Collections.CONTENT_NODES, Collections.STRUCTURE_NODES]:
        if not db.has_collection(collection_name):
            reset_counts[collection_name] = 0
            continue

        # Update all nodes with is_reviewed=true to is_reviewed=false
        query = """
        FOR doc IN @@collection
            FILTER doc.is_reviewed == true
            UPDATE doc WITH { is_reviewed: false } IN @@collection
            RETURN 1
        """
        cursor = db.aql.execute(query, bind_vars={"@collection": collection_name})
        count = sum(1 for _ in cursor)
        reset_counts[collection_name] = count

    return reset_counts


def get_node_counts(client: ArangoClient) -> dict[str, Any]:
    """Get node counts by collection."""
    db = client.db
    counts = {}

    for collection_name in [Collections.CONTENT_NODES, Collections.STRUCTURE_NODES]:
        if not db.has_collection(collection_name):
            counts[collection_name] = {"total": 0, "unreviewed": 0, "reviewed": 0}
            continue

        # Total
        query_total = "RETURN LENGTH(@@collection)"
        cursor = db.aql.execute(query_total, bind_vars={"@collection": collection_name})
        total = next(cursor, 0)

        # Unreviewed
        query_unreviewed = """
        FOR doc IN @@collection
            FILTER doc.is_reviewed != true
            COLLECT WITH COUNT INTO count
            RETURN count
        """
        cursor = db.aql.execute(query_unreviewed, bind_vars={"@collection": collection_name})
        unreviewed = next(cursor, 0)

        counts[collection_name] = {
            "total": total,
            "unreviewed": unreviewed,
            "reviewed": total - unreviewed,
        }

    return counts


def get_total_unreviewed(client: ArangoClient, tier: str = "all") -> int:
    """Get total count of unreviewed nodes."""
    counts = get_node_counts(client)
    total = 0
    if tier in ("all", "content"):
        total += counts.get(Collections.CONTENT_NODES, {}).get("unreviewed", 0)
    if tier in ("all", "structure"):
        total += counts.get(Collections.STRUCTURE_NODES, {}).get("unreviewed", 0)
    return total


# =============================================================================
# LLM Verifier Helper
# =============================================================================
def get_llm_extractor(client: ArangoClient) -> LLMRelationExtractor:
    """Get LLM extractor instance with OntologyHandler for Centroid filtering.

    Args:
        client: ArangoDB client for OntologyHandler initialization

    Returns:
        LLMRelationExtractor with db and ontology_handler for tier-based filtering
    """
    from config.ontologies.history import HISTORY_ONTOLOGY
    from workflows.extract_relations.llm_extractor import LLMRelationExtractor

    # Initialize with OntologyHandler for Centroid-based relation filtering
    ontology_handler = OntologyHandler(client.db)

    return LLMRelationExtractor(
        ontology=HISTORY_ONTOLOGY,
        db=client.db,
        ontology_handler=ontology_handler,
    )


def get_all_ontology_relations(client: ArangoClient) -> frozenset[str]:
    """Get ALL ontology relations (all tiers) for accurate statistics.

    This includes relations from all tiers (tier_1_structure, tier_2_content, etc.)
    to correctly identify if a relation is in the ontology regardless of which tier
    is currently being processed.
    """
    ontology_handler = OntologyHandler(client.db)
    all_relations = ontology_handler.get_all_active()
    return frozenset(r.id for r in all_relations)


async def process_llm_batch(
    pending_pairs: list[LLMPendingPair],
    llm_extractor: LLMRelationExtractor,
    handler: ContentExtractionHandler | StructureExtractionHandler,
    stats: ExtractionStats,
    tier_stats: TierStats,
    all_ontology_relations: frozenset[str],
    settings: Any,
    dry_run: bool = False,
    tier: str = "content",
    debug_logger: ExtractionDebugLogger | None = None,
) -> None:
    """Process LLM pairs in batches for efficiency.

    Instead of calling LLM for each pair individually, this collects pairs
    and processes them in batches of EXTRACTION_LLM_BATCH_SIZE.

    Args:
        pending_pairs: List of pairs pending LLM verification
        llm_extractor: LLM extractor instance
        handler: Content or Structure extraction handler
        stats: Global extraction stats
        tier_stats: Tier-specific stats
        all_ontology_relations: Set of all ontology relation IDs
        settings: Extraction settings
        dry_run: If True, don't modify database
        tier: "content" or "structure" - determines which tier's relations to use
        debug_logger: Optional debug logger
    """
    if not pending_pairs:
        return

    # Determine the RelationTier for filtering
    relation_tier = (
        RelationTier.TIER_1_STRUCTURE if tier == "structure" else RelationTier.TIER_2_CONTENT
    )

    from workflows.extract_relations.llm_extractor import NodePair

    batch_size = settings.EXTRACTION_LLM_BATCH_SIZE
    logger.info(
        f"Processing {len(pending_pairs)} pairs in batches of {batch_size} "
        f"(~{(len(pending_pairs) + batch_size - 1) // batch_size} LLM calls)"
    )

    # Convert to NodePair format with embeddings for Centroid filtering
    node_pairs = []
    for pair in pending_pairs:
        source_text = (
            pair.source_node.get_searchable_text()
            if hasattr(pair.source_node, "get_searchable_text")
            else str(getattr(pair.source_node, "attributes", ""))
        )
        target_text = (
            pair.target_node.get_searchable_text()
            if hasattr(pair.target_node, "get_searchable_text")
            else str(getattr(pair.target_node, "attributes", ""))
        )
        source_type = (
            str(pair.source_node.node_type)
            if hasattr(pair.source_node, "node_type")
            else f"structure_level_{getattr(pair.source_node, 'level', 0)}"
        )
        target_type = (
            str(pair.target_node.node_type)
            if hasattr(pair.target_node, "node_type")
            else f"structure_level_{getattr(pair.target_node, 'level', 0)}"
        )
        # Get embeddings for Centroid-based relation filtering
        source_embedding = getattr(pair.source_node, "embedding", None)
        target_embedding = getattr(pair.target_node, "embedding", None)

        node_pairs.append(
            NodePair(
                node_a_id=str(pair.source_node.id),
                node_a_type=source_type,
                node_a_text=source_text,
                node_b_id=str(pair.target_node.id),
                node_b_type=target_type,
                node_b_text=target_text,
                similarity_score=pair.similarity,
                node_a_embedding=source_embedding,
                node_b_embedding=target_embedding,
            )
        )

    # Process in batches
    results = []
    for i in range(0, len(node_pairs), batch_size):
        batch = node_pairs[i : i + batch_size]
        batch_num = i // batch_size + 1
        # Pass tier parameter for proper relation filtering
        batch_results = await llm_extractor.verify_batch(batch, tier=relation_tier)
        results.extend(batch_results)
        stats.llm_calls += 1

        # Log detailed per-pair results with node text
        logger.info(f"  [Batch {batch_num}] Processing {len(batch)} pairs:")
        for j, (pair, result) in enumerate(zip(batch, batch_results, strict=False)):
            # Truncate text for readability
            a_text = (
                pair.node_a_text[:50] + "..." if len(pair.node_a_text) > 50 else pair.node_a_text
            )
            b_text = (
                pair.node_b_text[:50] + "..." if len(pair.node_b_text) > 50 else pair.node_b_text
            )
            # Clean text for logging
            a_text = a_text.replace("\n", " ").strip()
            b_text = b_text.replace("\n", " ").strip()

            status = "FALLBACK" if result.used_fallback else "OK"
            merge_status = "MERGE" if result.should_merge else result.relation_type or "NONE"
            logger.info(
                f'    [{j + 1}] "{a_text}" -- "{b_text}" '
                f"=> {merge_status} (conf: {result.confidence:.2f}, sim: {pair.similarity_score:.2f}) [{status}]"
            )

        # Log to debug logger if available
        if debug_logger:
            tier_name = f"tier_2_{tier}" if tier == "content" else f"tier_1_{tier}"
            batch_log_results = [
                {
                    "node_a_text": pair.node_a_text[:100],
                    "node_b_text": pair.node_b_text[:100],
                    "similarity": pair.similarity_score,
                    "relation_type": r.relation_type,
                    "should_merge": r.should_merge,
                    "confidence": r.confidence,
                    "is_new": r.is_new_relation,
                    "used_fallback": r.used_fallback,
                    "reasoning": r.reasoning,
                }
                for pair, r in zip(batch, batch_results, strict=False)
            ]
            debug_logger.log_llm_batch(tier_name, len(batch), batch_log_results)

    # Apply results
    for idx, pair in enumerate(pending_pairs):
        if idx >= len(results):
            break

        result = results[idx]
        source_key = str(pair.source_node.id)
        target_key = str(pair.target_node.id)

        if result.should_merge:
            # LLM says merge
            if not dry_run:
                if tier == "content":
                    merge_result = await handler.auto_merge(pair.source_node, pair.target_node)
                    if merge_result.decision == ExtractionDecision.MERGE:
                        stats.nodes_merged += 1
                        tier_stats.llm_merged += 1
                        stats.merge_pairs.append((source_key, target_key))
                        logger.debug(f"  LLM-Merged: {source_key} -> {target_key}")
                else:
                    success = await handler.merge_structures(pair.source_node, pair.target_node)
                    if success:
                        stats.nodes_merged += 1
                        tier_stats.llm_merged += 1
                        stats.merge_pairs.append((source_key, target_key))
                        logger.debug(f"  LLM-Merged: {source_key} -> {target_key}")
            else:
                stats.nodes_merged += 1
                tier_stats.llm_merged += 1
        elif result.relation_type:
            # Create relation
            relation_type = result.relation_type
            is_in_ontology = relation_type in all_ontology_relations

            if not dry_run:
                if tier == "content":
                    await handler.create_relation_edge(
                        pair.source_node,
                        pair.target_node,
                        relation_type,
                        confidence=pair.similarity,
                        method="llm_verified",
                    )
                    await handler.mark_as_reviewed([source_key])
                else:
                    await handler.create_structure_relation(
                        pair.source_node,
                        pair.target_node,
                        relation_type,
                        confidence=pair.similarity,
                        method="llm_verified",
                    )
                    await handler.mark_as_reviewed([source_key])

            stats.relations_created += 1
            tier_stats.relations_total += 1
            stats.nodes_marked_reviewed += 1

            if result.is_in_top_k:
                tier_stats.relations_in_top_k += 1
            if result.used_fallback:
                tier_stats.relations_fallback += 1
            if result.is_new_relation:
                tier_stats.relations_new += 1

            stats.add_relation(relation_type, is_in_ontology)
            logger.debug(
                f"  LLM relation: {source_key} -[{relation_type}]-> {target_key} "
                f"(confidence: {result.confidence:.2f})"
            )


# =============================================================================
# Extraction Runner
# =============================================================================
async def run_content_extraction(
    client: ArangoClient,
    stats: ExtractionStats,
    *,
    dry_run: bool = False,
    limit: int | None = None,
    use_llm: bool = False,
    llm_extractor: LLMRelationExtractor | None = None,
    all_ontology_relations: frozenset[str] | None = None,
    debug_logger: ExtractionDebugLogger | None = None,
) -> int:
    """Run extraction on content nodes. Returns count processed."""
    settings = get_extraction_settings()
    handler = ContentExtractionHandler(client.db, settings=settings)

    # Get all ontology relations for accurate is_in_ontology check
    if all_ontology_relations is None:
        all_ontology_relations = get_all_ontology_relations(client)

    # Initialize tier stats
    tier_stats = TierStats(tier_name="tier_2_content")

    logger.info("Fetching unreviewed content nodes...")

    # Get unreviewed nodes
    unreviewed = await handler.get_unreviewed_nodes(limit=limit)
    batch_count = len(unreviewed)
    stats.total_unreviewed += batch_count

    if not unreviewed:
        logger.info("No unreviewed content nodes found")
        stats.tier_stats["tier_2_content"] = tier_stats
        return 0

    logger.info(f"Found {batch_count} unreviewed content nodes")

    # Log input nodes for debug report
    if debug_logger:
        debug_logger.log_input_nodes(
            tier_name="tier_2_content",
            nodes=[
                {
                    "id": str(n.id),
                    "type": str(n.node_type) if hasattr(n, "node_type") else "content",
                    "text": n.get_searchable_text()
                    if hasattr(n, "get_searchable_text")
                    else str(n.attributes)[:200],
                    "has_embedding": n.embedding is not None if hasattr(n, "embedding") else True,
                    "source_id": str(n.source_id) if hasattr(n, "source_id") else None,
                }
                for n in unreviewed
            ],
        )
        debug_logger.start_phase("tier_2_content", "similarity_search")

    # Collect pairs for batch LLM processing
    llm_pending_pairs: list[LLMPendingPair] = []

    for i, node in enumerate(unreviewed):
        stats.nodes_processed += 1
        tier_stats.nodes_processed += 1
        node_key = str(node.id)

        try:
            # Find similar nodes
            similar_nodes = await handler.find_similar_nodes(
                node,
                limit=settings.EXTRACTION_MAX_CANDIDATES_PER_NODE,
                min_similarity=settings.EXTRACTION_LLM_THRESHOLD,
                cross_source=settings.EXTRACTION_CROSS_SOURCE,
            )

            # Determine decision for debug logging
            if not similar_nodes:
                decision_str = "NO_MATCH"
                decision_reason = "No similar nodes found above threshold"
            else:
                best_match = similar_nodes[0]
                decision = handler.decide_action(best_match.similarity_score)
                if decision == ExtractionDecision.MERGE:
                    decision_str = "MERGE"
                    decision_reason = f"Similarity {best_match.similarity_score:.3f} >= {settings.EXTRACTION_AUTO_MERGE_THRESHOLD}"
                elif decision == ExtractionDecision.CREATE_RELATION:
                    decision_str = "LLM"
                    decision_reason = f"Similarity {best_match.similarity_score:.3f} in LLM range [{settings.EXTRACTION_LLM_THRESHOLD}, {settings.EXTRACTION_AUTO_MERGE_THRESHOLD})"
                else:
                    decision_str = "SEPARATE"
                    decision_reason = f"Similarity {best_match.similarity_score:.3f} < {settings.EXTRACTION_LLM_THRESHOLD}"

            # Log similarity search for debug report
            if debug_logger:
                debug_logger.log_similarity_search(
                    tier_name="tier_2_content",
                    source_node={
                        "id": str(node.id),
                        "type": str(node.node_type) if hasattr(node, "node_type") else "content",
                        "text": node.get_searchable_text()
                        if hasattr(node, "get_searchable_text")
                        else "",
                        "has_embedding": node.embedding is not None
                        if hasattr(node, "embedding")
                        else True,
                    },
                    candidates=[
                        {
                            "id": str(s.node_id),
                            "type": str(s.node.node_type)
                            if hasattr(s.node, "node_type")
                            else "content",
                            "text": s.node.get_searchable_text()
                            if hasattr(s.node, "get_searchable_text")
                            else "",
                            "similarity": s.similarity_score,
                            "is_reviewed": True,
                        }
                        for s in similar_nodes[:5]  # Top 5 only
                    ],
                    decision=decision_str,
                    decision_reason=decision_reason,
                    threshold=settings.EXTRACTION_LLM_THRESHOLD,
                )

            if not similar_nodes:
                # No similar nodes found, mark as reviewed
                if not dry_run:
                    await handler.mark_as_reviewed([node_key])
                stats.nodes_marked_reviewed += 1
                stats.nodes_kept_separate += 1
                tier_stats.kept_separate += 1
                continue

            # Process best match
            best_match = similar_nodes[0]
            decision = handler.decide_action(best_match.similarity_score)

            if decision == ExtractionDecision.MERGE:
                if not dry_run:
                    result = await handler.auto_merge(node, best_match.node)
                    if result.decision == ExtractionDecision.MERGE:
                        stats.nodes_merged += 1
                        tier_stats.auto_merged += 1
                        stats.merge_pairs.append((node_key, str(best_match.node_id)))
                        logger.debug(
                            f"  Merged: {node_key} -> {best_match.node_id} "
                            f"(similarity: {best_match.similarity_score:.3f})"
                        )
                    else:
                        stats.nodes_kept_separate += 1
                        tier_stats.kept_separate += 1
                else:
                    stats.nodes_merged += 1
                    tier_stats.auto_merged += 1
                    logger.debug(
                        f"  [DRY-RUN] Would merge: {node_key} -> {best_match.node_id} "
                        f"(similarity: {best_match.similarity_score:.3f})"
                    )

            elif decision == ExtractionDecision.CREATE_RELATION:
                # Queue for batch LLM processing if enabled
                if use_llm and llm_extractor and not dry_run:
                    tier_stats.pairs_sent_to_llm += 1
                    llm_pending_pairs.append(
                        LLMPendingPair(
                            source_node=node,
                            target_node=best_match.node,
                            similarity=best_match.similarity_score,
                            node_index=i,
                        )
                    )
                else:
                    # No LLM or dry run - use fallback relation
                    relation_type = settings.EXTRACTION_FALLBACK_RELATION
                    is_in_ontology = relation_type in all_ontology_relations

                    if not dry_run:
                        await handler.create_relation_edge(
                            node,
                            best_match.node,
                            relation_type,
                            confidence=best_match.similarity_score,
                            method="embedding_similarity",
                        )
                        await handler.mark_as_reviewed([node_key])

                    stats.relations_created += 1
                    tier_stats.relations_total += 1
                    tier_stats.relations_fallback += 1
                    stats.nodes_marked_reviewed += 1
                    stats.add_relation(relation_type, is_in_ontology)

                    logger.debug(
                        f"  Created relation: {node_key} -[{relation_type}]-> {best_match.node_id}"
                    )

            else:
                # Keep separate
                if not dry_run:
                    await handler.mark_as_reviewed([node_key])
                stats.nodes_kept_separate += 1
                tier_stats.kept_separate += 1
                stats.nodes_marked_reviewed += 1

            # Progress log
            if (i + 1) % 10 == 0:
                logger.info(f"  Progress: {i + 1}/{batch_count} nodes processed")

        except Exception as e:
            stats.errors += 1
            tier_stats.errors += 1
            stats.error_details.append(f"Content node {node_key}: {e!s}")
            logger.error(f"Error processing content node {node_key}: {e}")

    # End similarity search phase
    if debug_logger:
        debug_logger.end_phase(
            tier_name="tier_2_content",
            phase_name="similarity_search",
            output_count=len(llm_pending_pairs),
            summary={
                "nodes_processed": tier_stats.nodes_processed,
                "auto_merged": tier_stats.auto_merged,
                "kept_separate": tier_stats.kept_separate,
                "pairs_queued_for_llm": len(llm_pending_pairs),
            },
        )

    # Process collected LLM pairs in batch
    if llm_pending_pairs and llm_extractor:
        if debug_logger:
            debug_logger.start_phase("tier_2_content", "llm_batch_processing")

        await process_llm_batch(
            pending_pairs=llm_pending_pairs,
            llm_extractor=llm_extractor,
            handler=handler,
            stats=stats,
            tier_stats=tier_stats,
            all_ontology_relations=all_ontology_relations,
            settings=settings,
            dry_run=dry_run,
            tier="content",
            debug_logger=debug_logger,
        )

        if debug_logger:
            debug_logger.end_phase(
                tier_name="tier_2_content",
                phase_name="llm_batch_processing",
                output_count=tier_stats.relations_total,
                summary={
                    "pairs_processed": len(llm_pending_pairs),
                    "relations_created": tier_stats.relations_total,
                    "llm_merged": tier_stats.llm_merged,
                },
            )

    # Store tier stats
    stats.tier_stats["tier_2_content"] = tier_stats

    return batch_count


async def run_structure_extraction(
    client: ArangoClient,
    stats: ExtractionStats,
    *,
    dry_run: bool = False,
    limit: int | None = None,
    use_llm: bool = False,
    llm_extractor: LLMRelationExtractor | None = None,
    all_ontology_relations: frozenset[str] | None = None,
    debug_logger: ExtractionDebugLogger | None = None,
) -> int:
    """Run extraction on structure nodes. Returns count processed."""
    settings = get_extraction_settings()
    handler = StructureExtractionHandler(client.db, settings=settings)

    # Get all ontology relations for accurate is_in_ontology check
    if all_ontology_relations is None:
        all_ontology_relations = get_all_ontology_relations(client)

    # Initialize tier stats
    tier_stats = TierStats(tier_name="tier_1_structure")

    logger.info("Fetching unreviewed structure nodes...")

    # Get unreviewed nodes
    unreviewed = await handler.get_unreviewed_structures(limit=limit)
    batch_count = len(unreviewed)
    stats.total_unreviewed += batch_count

    if not unreviewed:
        logger.info("No unreviewed structure nodes found")
        stats.tier_stats["tier_1_structure"] = tier_stats
        return 0

    logger.info(f"Found {batch_count} unreviewed structure nodes")

    # Log input nodes for debug report
    if debug_logger:
        debug_logger.log_input_nodes(
            tier_name="tier_1_structure",
            nodes=[
                {
                    "id": str(n.id),
                    "type": f"structure_level_{n.level}" if hasattr(n, "level") else "structure",
                    "text": n.title if hasattr(n, "title") else str(n)[:200],
                    "has_embedding": n.embedding is not None if hasattr(n, "embedding") else True,
                    "source_id": str(n.source_id) if hasattr(n, "source_id") else None,
                    "extra": {"level": n.level if hasattr(n, "level") else None},
                }
                for n in unreviewed
            ],
        )
        debug_logger.start_phase("tier_1_structure", "similarity_search")

    # Collect pairs for batch LLM processing
    llm_pending_pairs: list[LLMPendingPair] = []

    for i, node in enumerate(unreviewed):
        stats.nodes_processed += 1
        tier_stats.nodes_processed += 1
        node_key = str(node.id)

        try:
            # First try title matching (exact match = auto-merge)
            title_matches = await handler.find_title_matches(
                node, cross_source=settings.EXTRACTION_CROSS_SOURCE
            )

            if title_matches:
                # Log title match for debug
                if debug_logger:
                    debug_logger.log_similarity_search(
                        tier_name="tier_1_structure",
                        source_node={
                            "id": str(node.id),
                            "type": f"structure_level_{node.level}"
                            if hasattr(node, "level")
                            else "structure",
                            "text": node.title if hasattr(node, "title") else "",
                        },
                        candidates=[
                            {
                                "id": str(m.node_id),
                                "type": "structure",
                                "text": m.node.title if hasattr(m.node, "title") else "",
                                "similarity": 1.0,  # Exact match
                                "is_reviewed": True,
                            }
                            for m in title_matches[:3]
                        ],
                        decision="MERGE",
                        decision_reason="Exact title match",
                        threshold=1.0,
                    )

                # Exact title match = merge
                best_match = title_matches[0]
                if not dry_run:
                    success = await handler.merge_structures(node, best_match.node)
                    if success:
                        stats.nodes_merged += 1
                        tier_stats.auto_merged += 1
                        stats.merge_pairs.append((node_key, str(best_match.node_id)))
                        logger.debug(f"  Merged (title match): {node_key} -> {best_match.node_id}")
                    else:
                        stats.nodes_kept_separate += 1
                        tier_stats.kept_separate += 1
                else:
                    stats.nodes_merged += 1
                    tier_stats.auto_merged += 1
                    logger.debug(
                        f"  [DRY-RUN] Would merge (title match): {node_key} -> {best_match.node_id}"
                    )
                continue

            # Try embedding similarity
            similar_nodes = await handler.find_similar_structures(
                node,
                max_candidates=settings.EXTRACTION_MAX_CANDIDATES_PER_NODE,
                cross_source=settings.EXTRACTION_CROSS_SOURCE,
            )

            # Determine decision for debug logging
            if not similar_nodes:
                decision_str = "NO_MATCH"
                decision_reason = "No similar nodes found"
            else:
                best_match = similar_nodes[0]
                if best_match.similarity_score >= settings.EXTRACTION_AUTO_MERGE_THRESHOLD:
                    decision_str = "MERGE"
                    decision_reason = f"Similarity {best_match.similarity_score:.3f} >= {settings.EXTRACTION_AUTO_MERGE_THRESHOLD}"
                elif best_match.similarity_score >= settings.EXTRACTION_LLM_THRESHOLD:
                    decision_str = "LLM"
                    decision_reason = f"Similarity {best_match.similarity_score:.3f} in LLM range"
                else:
                    decision_str = "SEPARATE"
                    decision_reason = f"Similarity {best_match.similarity_score:.3f} < {settings.EXTRACTION_LLM_THRESHOLD}"

            # Log similarity search for debug
            if debug_logger:
                debug_logger.log_similarity_search(
                    tier_name="tier_1_structure",
                    source_node={
                        "id": str(node.id),
                        "type": f"structure_level_{node.level}"
                        if hasattr(node, "level")
                        else "structure",
                        "text": node.title if hasattr(node, "title") else "",
                    },
                    candidates=[
                        {
                            "id": str(s.node_id),
                            "type": "structure",
                            "text": s.node.title if hasattr(s.node, "title") else "",
                            "similarity": s.similarity_score,
                            "is_reviewed": True,
                        }
                        for s in similar_nodes[:5]
                    ]
                    if similar_nodes
                    else [],
                    decision=decision_str,
                    decision_reason=decision_reason,
                    threshold=settings.EXTRACTION_LLM_THRESHOLD,
                )

            if not similar_nodes:
                if not dry_run:
                    await handler.mark_as_reviewed([node_key])
                stats.nodes_marked_reviewed += 1
                stats.nodes_kept_separate += 1
                tier_stats.kept_separate += 1
                continue

            best_match = similar_nodes[0]

            if best_match.similarity_score >= settings.EXTRACTION_AUTO_MERGE_THRESHOLD:
                if not dry_run:
                    success = await handler.merge_structures(node, best_match.node)
                    if success:
                        stats.nodes_merged += 1
                        tier_stats.auto_merged += 1
                        stats.merge_pairs.append((node_key, str(best_match.node_id)))
                    else:
                        stats.nodes_kept_separate += 1
                        tier_stats.kept_separate += 1
                else:
                    stats.nodes_merged += 1
                    tier_stats.auto_merged += 1
                    logger.debug(f"  [DRY-RUN] Would merge: {node_key} -> {best_match.node_id}")

            elif best_match.similarity_score >= settings.EXTRACTION_LLM_THRESHOLD:
                # Queue for batch LLM processing if enabled
                if use_llm and llm_extractor and not dry_run:
                    tier_stats.pairs_sent_to_llm += 1
                    llm_pending_pairs.append(
                        LLMPendingPair(
                            source_node=node,
                            target_node=best_match.node,
                            similarity=best_match.similarity_score,
                            node_index=i,
                        )
                    )
                else:
                    # No LLM or dry run - use fallback relation
                    relation_type = settings.EXTRACTION_FALLBACK_RELATION
                    is_in_ontology = relation_type in all_ontology_relations

                    if not dry_run:
                        await handler.create_structure_relation(
                            node,
                            best_match.node,
                            relation_type,
                            confidence=best_match.similarity_score,
                            method="embedding_similarity",
                        )
                        await handler.mark_as_reviewed([node_key])

                    stats.relations_created += 1
                    tier_stats.relations_total += 1
                    tier_stats.relations_fallback += 1
                    stats.nodes_marked_reviewed += 1
                    stats.add_relation(relation_type, is_in_ontology)

            else:
                if not dry_run:
                    await handler.mark_as_reviewed([node_key])
                stats.nodes_kept_separate += 1
                tier_stats.kept_separate += 1
                stats.nodes_marked_reviewed += 1

            # Progress log
            if (i + 1) % 10 == 0:
                logger.info(f"  Progress: {i + 1}/{batch_count} nodes processed")

        except Exception as e:
            stats.errors += 1
            tier_stats.errors += 1
            stats.error_details.append(f"Structure node {node_key}: {e!s}")
            logger.error(f"Error processing structure node {node_key}: {e}")

    # End similarity search phase
    if debug_logger:
        debug_logger.end_phase(
            tier_name="tier_1_structure",
            phase_name="similarity_search",
            output_count=len(llm_pending_pairs),
            summary={
                "nodes_processed": tier_stats.nodes_processed,
                "auto_merged": tier_stats.auto_merged,
                "kept_separate": tier_stats.kept_separate,
                "pairs_queued_for_llm": len(llm_pending_pairs),
            },
        )

    # Process collected LLM pairs in batch
    if llm_pending_pairs and llm_extractor:
        if debug_logger:
            debug_logger.start_phase("tier_1_structure", "llm_batch_processing")

        await process_llm_batch(
            pending_pairs=llm_pending_pairs,
            llm_extractor=llm_extractor,
            handler=handler,
            stats=stats,
            tier_stats=tier_stats,
            all_ontology_relations=all_ontology_relations,
            settings=settings,
            dry_run=dry_run,
            tier="structure",
            debug_logger=debug_logger,
        )

        if debug_logger:
            debug_logger.end_phase(
                tier_name="tier_1_structure",
                phase_name="llm_batch_processing",
                output_count=tier_stats.relations_total,
                summary={
                    "llm_pairs_processed": len(llm_pending_pairs),
                    "llm_merge_decisions": tier_stats.llm_merged,
                    "relations_created": tier_stats.relations_total,
                    "kept_separate": tier_stats.kept_separate,
                },
            )

    # Store tier stats
    stats.tier_stats["tier_1_structure"] = tier_stats

    return batch_count


# =============================================================================
# Report Generation
# =============================================================================
def print_report(
    client: ArangoClient,
    stats: ExtractionStats,
    edge_stats_before: dict[str, Any],
    edge_stats_after: dict[str, Any],
    node_counts_before: dict[str, Any],
    node_counts_after: dict[str, Any],
    settings: Any,
    use_llm: bool = False,
) -> None:
    """Print comprehensive extraction report."""
    duration = stats.get_duration()

    print("\n" + "=" * 70)
    print("                    EXTRACTION REPORT")
    print("=" * 70)

    # Settings used
    print("\n[SETTINGS]")
    print(f"  Auto-merge threshold:  >= {settings.EXTRACTION_AUTO_MERGE_THRESHOLD}")
    print(f"  LLM threshold:         >= {settings.EXTRACTION_LLM_THRESHOLD}")
    print(f"  Cross-source:          {settings.EXTRACTION_CROSS_SOURCE}")
    print(f"  Fallback relation:     {settings.EXTRACTION_FALLBACK_RELATION}")
    print(f"  LLM extraction:        {'ENABLED' if use_llm else 'DISABLED (only SIMILAR_TO)'}")
    two_phase_status = (
        "ENABLED (intra-batch dedup)" if settings.EXTRACTION_USE_TWO_PHASE else "DISABLED (legacy)"
    )
    print(f"  Two-Phase mode:        {two_phase_status}")

    # Summary
    print("\n[SUMMARY]")
    print(f"  Duration:              {duration:.2f} seconds")
    print(f"  Batches completed:     {stats.batches_completed}")
    print(f"  Nodes processed:       {stats.nodes_processed}")
    print(f"  Nodes merged:          {stats.nodes_merged}")
    print(f"  Relations created:     {stats.relations_created}")
    print(f"  Nodes kept separate:   {stats.nodes_kept_separate}")
    if use_llm:
        print(f"  LLM calls:             {stats.llm_calls}")
    print(f"  Errors:                {stats.errors}")

    # Per-tier detailed statistics
    if stats.tier_stats:
        print("\n[TIER STATISTICS]")
        for tier_name, tier_stat in stats.tier_stats.items():
            tier_display = (
                "TIER 1 (Structure)" if tier_name == "tier_1_structure" else "TIER 2 (Content)"
            )
            print(f"\n  {tier_display}:")
            print(f"    Nodes processed:       {tier_stat.nodes_processed}")
            print(f"    Auto-merged (>0.72):   {tier_stat.auto_merged}")
            print(f"    Pairs sent to LLM:     {tier_stat.pairs_sent_to_llm}")
            print(f"    LLM merged:            {tier_stat.llm_merged}")
            print(f"    Relations in top-k:    {tier_stat.relations_in_top_k}")
            print(f"    Relations fallback:    {tier_stat.relations_fallback}")
            print(f"    New relations:         {tier_stat.relations_new}")
            print(f"    Total relations:       {tier_stat.relations_total}")
            print(f"    Kept separate:         {tier_stat.kept_separate}")
            print(f"    Errors:                {tier_stat.errors}")

    # Node counts comparison
    print("\n[NODE COUNTS - BEFORE vs AFTER]")
    for collection, before in node_counts_before.items():
        after = node_counts_after.get(collection, before)
        merged = before["total"] - after["total"]
        reviewed_gain = after["reviewed"] - before["reviewed"]

        print(f"\n  {collection}:")
        print(f"    Total:      {before['total']} -> {after['total']} ({-merged:+d})")
        print(
            f"    Unreviewed: {before['unreviewed']} -> {after['unreviewed']} "
            f"({after['unreviewed'] - before['unreviewed']:+d})"
        )
        print(f"    Reviewed:   {before['reviewed']} -> {after['reviewed']} ({reviewed_gain:+d})")

    # Edge statistics
    print("\n[EDGE STATISTICS]")
    edges_added = edge_stats_after["total_edges"] - edge_stats_before["total_edges"]
    dedup_edges_added = edge_stats_after["dedup_edges"] - edge_stats_before["dedup_edges"]

    print(
        f"  Total edges:           {edge_stats_before['total_edges']} -> "
        f"{edge_stats_after['total_edges']} ({edges_added:+d})"
    )
    print(
        f"  Dedup edges:           {edge_stats_before['dedup_edges']} -> "
        f"{edge_stats_after['dedup_edges']} ({dedup_edges_added:+d})"
    )

    # Get all allowed relations from ontology for stats
    ontology_handler = OntologyHandler(client.db)
    all_relations = ontology_handler.get_all_active()
    allowed_relations_set = {r.id for r in all_relations}

    # Relations by type created in this run
    if stats.relations_by_type:
        print("\n[RELATIONS CREATED BY TYPE]")
        for rel_type, count in sorted(stats.relations_by_type.items(), key=lambda x: -x[1]):
            in_ontology = rel_type in allowed_relations_set
            status = "OK" if in_ontology else "NEW (from LLM)"
            print(f"    {rel_type}: {count} [{status}]")

    # Relations outside ontology
    if stats.relations_outside_ontology:
        print("\n[NEW RELATIONS PROPOSED BY LLM]")
        print("  These relation types were proposed by LLM:")
        for rel_type in stats.relations_outside_ontology:
            print(f"    - {rel_type}")
        print(f"\n  Note: Stored in collection: {EdgeCollections.KNOWLEDGE_EDGES}")
        print("  Field: detection_method='llm_verified'")

    # Edge types in database
    print("\n[ALL EDGE TYPES IN DATABASE]")

    # Build full ontology set (including structural relationships)
    full_ontology_set = allowed_relations_set | {
        RelationshipTypes.HAS_SECTION,
        RelationshipTypes.HAS_CONTENT,
        RelationshipTypes.NEXT_SIBLING,
    }

    for edge_type, count in sorted(edge_stats_after["edges_by_type"].items(), key=lambda x: -x[1]):
        in_ontology = edge_type in full_ontology_set
        status = "OK" if in_ontology else "NEW"
        print(f"    {edge_type}: {count} [{status}]")

    # Merge pairs (sample)
    if stats.merge_pairs:
        print(f"\n[MERGE PAIRS] (showing first 10 of {len(stats.merge_pairs)})")
        for source, target in stats.merge_pairs[:10]:
            print(f"    {source} -> {target}")

    # Errors
    if stats.error_details:
        print(f"\n[ERRORS] ({stats.errors} total)")
        for error in stats.error_details[:10]:
            print(f"    {error}")
        if len(stats.error_details) > 10:
            print(f"    ... and {len(stats.error_details) - 10} more")

    print("\n" + "=" * 70)


# =============================================================================
# Main
# =============================================================================
async def async_main(
    tier: str = "all",
    dry_run: bool = False,
    batch_size: int = 100,
    continuous: bool = False,
    use_llm: bool = False,
    max_batches: int | None = None,
    use_two_phase: bool = True,
    debug_mode: bool = False,
    reset_nodes: bool = False,
) -> None:
    """Async main entry point."""
    client = init_arango_client()
    settings = get_extraction_settings()
    stats = ExtractionStats()

    # Override settings with CLI flag
    if not use_two_phase:
        settings.EXTRACTION_USE_TWO_PHASE = False

    # Initialize debug logger if debug mode is enabled
    debug_logger = None
    if debug_mode:
        from workflows.extract_relations.debug_logger import get_debug_logger

        # Use deep_debug for full prompt/response logging
        debug_logger = get_debug_logger(deep_debug=True)
        debug_logger.start_run(
            settings={
                "auto_merge_threshold": settings.EXTRACTION_AUTO_MERGE_THRESHOLD,
                "llm_threshold": settings.EXTRACTION_LLM_THRESHOLD,
                "llm_batch_size": settings.EXTRACTION_LLM_BATCH_SIZE,
                "fallback_relation": settings.EXTRACTION_FALLBACK_RELATION,
                "cross_source": settings.EXTRACTION_CROSS_SOURCE,
                "use_llm": use_llm,
                "tier": tier,
                "two_phase": use_two_phase,
                "batch_size": batch_size,
            },
            deep_debug=True,
        )
        logger.info("Debug mode enabled (deep_debug=True) - will generate detailed MD report")

    # Initialize LLM extractor if needed
    llm_extractor = None
    if use_llm:
        logger.info("Initializing LLM extractor for semantic relation extraction...")
        llm_extractor = get_llm_extractor(client)

    try:
        # Reset nodes if requested (for testing purposes)
        if reset_nodes:
            logger.info("Resetting all nodes to is_reviewed=false...")
            reset_counts = reset_all_nodes_to_unreviewed(client)
            total_reset = sum(reset_counts.values())
            logger.info(f"Reset {total_reset} nodes to unreviewed:")
            for collection, count in reset_counts.items():
                logger.info(f"  {collection}: {count} nodes reset")

        # Get stats before
        logger.info("Collecting initial statistics...")
        edge_stats_before = get_edge_stats_before(client)
        node_counts_before = get_node_counts(client)

        initial_unreviewed = get_total_unreviewed(client, tier)
        logger.info(f"Total unreviewed nodes: {initial_unreviewed}")

        # Log database state before (for debug report)
        if debug_logger:
            debug_logger.log_database_state(
                {
                    "nodes": node_counts_before,
                    "edges": edge_stats_before,
                    "total_unreviewed": initial_unreviewed,
                },
                phase="before",
            )

        # Start timing
        stats.start_time = time.time()

        if dry_run:
            logger.warning("DRY-RUN MODE: No changes will be made to database")

        if continuous:
            logger.info(f"CONTINUOUS MODE: Processing ALL nodes in batches of {batch_size}")

        if use_two_phase:
            logger.info("TWO-PHASE MODE: Intra-batch deduplication enabled")

        # Cache all ontology relations for accurate statistics (check once, use for all batches)
        all_ontology_relations = get_all_ontology_relations(client)
        logger.info(f"Loaded {len(all_ontology_relations)} ontology relations for statistics")

        # Run extraction loop
        batch_num = 0
        while True:
            batch_num += 1
            batch_processed = 0

            if max_batches and batch_num > max_batches:
                logger.info(f"Reached max batches limit ({max_batches})")
                break

            logger.info(f"\n{'=' * 50}")
            logger.info(f"BATCH {batch_num}")
            logger.info(f"{'=' * 50}")

            # Process structure nodes
            if tier in ("all", "structure"):
                logger.info("\n--- STRUCTURE NODES (Tier 1) ---")
                count = await run_structure_extraction(
                    client,
                    stats,
                    dry_run=dry_run,
                    limit=batch_size,
                    use_llm=use_llm,
                    llm_extractor=llm_extractor,
                    all_ontology_relations=all_ontology_relations,
                    debug_logger=debug_logger,
                )
                batch_processed += count

            # Process content nodes
            if tier in ("all", "content"):
                logger.info("\n--- CONTENT NODES (Tier 2) ---")
                count = await run_content_extraction(
                    client,
                    stats,
                    dry_run=dry_run,
                    limit=batch_size,
                    use_llm=use_llm,
                    llm_extractor=llm_extractor,
                    all_ontology_relations=all_ontology_relations,
                    debug_logger=debug_logger,
                )
                batch_processed += count

            stats.batches_completed += 1

            # Check if we should continue
            if batch_processed == 0:
                logger.info("No more unreviewed nodes to process")
                break

            if not continuous:
                logger.info("Single batch mode, stopping")
                break

            # Show progress
            remaining = get_total_unreviewed(client, tier)
            logger.info(f"\nBatch {batch_num} complete. Remaining unreviewed: {remaining}")

        # End timing
        stats.end_time = time.time()

        # Get stats after
        logger.info("\nCollecting final statistics...")
        edge_stats_after = get_edge_stats_before(client)
        node_counts_after = get_node_counts(client)

        # Log database state after (for debug report)
        if debug_logger:
            final_unreviewed = get_total_unreviewed(client, tier)
            debug_logger.log_database_state(
                {
                    "nodes": node_counts_after,
                    "edges": edge_stats_after,
                    "total_unreviewed": final_unreviewed,
                },
                phase="after",
            )

        # Print report
        print_report(
            client,
            stats,
            edge_stats_before,
            edge_stats_after,
            node_counts_before,
            node_counts_after,
            settings,
            use_llm=use_llm,
        )

        # Finish debug logging and generate report
        if debug_mode and debug_logger:
            report_path = debug_logger.finish_run()
            if report_path:
                print(f"\n[DEBUG REPORT]\n  Path: {report_path}")
                logger.info(f"Debug report generated: {report_path}")

    finally:
        client.close()


def main() -> None:
    """Main entry point."""
    parser = argparse.ArgumentParser(
        description="Run extraction on unreviewed nodes",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Run single batch of 100 nodes
  python scripts/run_extraction.py

  # Run until ALL unreviewed nodes are processed
  python scripts/run_extraction.py --continuous

  # Use LLM to extract semantic relations (PREREQUISITE_OF, EXPLAINS, etc.)
  python scripts/run_extraction.py --use-llm

  # Continuous + LLM with custom batch size
  python scripts/run_extraction.py --continuous --use-llm --batch-size 50

  # Dry run (preview without changes)
  python scripts/run_extraction.py --dry-run --continuous

  # Process only content nodes
  python scripts/run_extraction.py --tier content --continuous

  # Limit to 5 batches max
  python scripts/run_extraction.py --continuous --max-batches 5

  # Reset all nodes to unreviewed and run extraction (for testing)
  python scripts/run_extraction.py --reset --use-llm --debug
        """,
    )
    parser.add_argument(
        "--tier",
        choices=["all", "content", "structure"],
        default="all",
        help="Which tier to process (default: all)",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Preview changes without modifying database",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=100,
        help="Number of nodes per batch (default: 100)",
    )
    parser.add_argument(
        "--continuous",
        action="store_true",
        help="Process ALL unreviewed nodes (loop until none left)",
    )
    parser.add_argument(
        "--use-llm",
        action="store_true",
        default=True,
        help="Use LLM to extract semantic relations (default: enabled)",
    )
    parser.add_argument(
        "--no-llm",
        action="store_true",
        help="Disable LLM extraction (use only SIMILAR_TO)",
    )
    parser.add_argument(
        "--max-batches",
        type=int,
        default=None,
        help="Maximum number of batches to run (for testing)",
    )
    parser.add_argument(
        "--verbose",
        "-v",
        action="store_true",
        help="Enable verbose logging (show debug messages)",
    )
    parser.add_argument(
        "--two-phase",
        action="store_true",
        default=True,
        help="Enable Two-Phase extraction (default: enabled)",
    )
    parser.add_argument(
        "--no-two-phase",
        action="store_true",
        help="Disable Two-Phase extraction (use legacy mode)",
    )
    parser.add_argument(
        "--debug",
        action="store_true",
        help="Enable debug logging - generate MD report with full decision log",
    )
    parser.add_argument(
        "--reset",
        action="store_true",
        help="Reset all nodes to is_reviewed=false before running extraction (for testing)",
    )

    args = parser.parse_args()

    # Configure logging
    logger.remove()
    if args.verbose:
        logger.add(sys.stderr, level="DEBUG")
    else:
        logger.add(sys.stderr, level="INFO")

    # Determine LLM usage (--no-llm overrides default)
    use_llm = args.use_llm and not args.no_llm

    # Determine Two-Phase mode (--no-two-phase overrides default)
    use_two_phase = args.two_phase and not args.no_two_phase

    # Determine debug mode
    debug_mode = args.debug

    # Run
    asyncio.run(
        async_main(
            tier=args.tier,
            dry_run=args.dry_run,
            batch_size=args.batch_size,
            continuous=args.continuous,
            use_llm=use_llm,
            max_batches=args.max_batches,
            use_two_phase=use_two_phase,
            debug_mode=debug_mode,
            reset_nodes=args.reset,
        )
    )


if __name__ == "__main__":
    main()
