"""Content extraction handler for content node merge and relation extraction.

Provides ContentExtractionHandler for managing content node extraction:
1. Embedding-based similarity search (cosine similarity)
2. Threshold-based decisions (unified with structure tier)
3. Relation creation with allowlist validation
4. Cross-source extraction support

Thresholds (configurable):
- >= 0.72: Auto-merge (no LLM needed)
- 0.40-0.72: LLM unified call (merge check + relation extraction)
- < 0.40: Keep separate
"""

from __future__ import annotations

import asyncio
import uuid
from dataclasses import dataclass
from functools import partial
from typing import TYPE_CHECKING, Literal

from arango.database import StandardDatabase
from pydantic import ValidationError

from ..collections.schema import Collections, EdgeCollections
from ..models.content_nodes import ContentNode
from ..models.merged_nodes_log import MergedNodeLog
from ..utils import validate_embedding
from .base_extraction_handler import BaseExtractionHandler, ExtractionDecision
from .ontology_handler import OntologyHandler

if TYPE_CHECKING:
    from config.extraction import ExtractionSettings

from ..models.ontology import RelationTier

# Fallback relation types for content nodes (used when database is unavailable)
_FALLBACK_CONTENT_RELATIONS = frozenset(
    {
        # Base relations
        "SIMILAR_TO",
        "ELABORATES",
        "SUMMARIZES",
        "EXEMPLIFIES",
        "SUPPORTS",
        "CONTRADICTS",
        "CONTRASTS",
        "REFERS_TO",
        "DEFINES",
        "FOLLOWS",
        "CONTINUES",
        "DUPLICATE_OF",
        "VARIANT_OF",
        "SUPERSEDES",
        "DERIVED_FROM",
        "VISUALIZES",
        # Education-specific relations
        "PREREQUISITE_OF",
        "BUILDS_ON",
        "LEADS_TO",
        "EXPLAINS",
        "PROVES",
        "DERIVES",
        "APPLIES",
        "ILLUSTRATES",
        "SOLVES",
        "TESTS",
        "REQUIRES",
        "CITES",
        "SEE_ALSO",
    }
)


@dataclass
class SimilarNode:
    """A node found by similarity search with its score."""

    node_id: str
    node: ContentNode
    similarity_score: float


@dataclass
class ExtractionResult:
    """Result of extraction for a single node pair."""

    decision: ExtractionDecision
    target_node_id: str | None = None
    relation_type: str | None = None
    confidence: float = 0.0
    method: str = "unknown"
    reasoning: str = ""


class ContentExtractionHandler(BaseExtractionHandler):
    """Handle content node extraction operations.

    Provides methods for:
    - Finding similar nodes via embedding search
    - Auto-merging high-similarity pairs
    - Creating relation edges for related nodes
    - Merging nodes (redirecting edges, updating references)
    - Cross-source extraction
    """

    def __init__(
        self,
        db: StandardDatabase,
        settings: ExtractionSettings | None = None,
        ontology_handler: OntologyHandler | None = None,
    ) -> None:
        """Initialize content extraction handler.

        Args:
            db: ArangoDB database connection
            settings: Extraction settings (lazy loaded if not provided)
            ontology_handler: Handler for ontology operations (lazy loaded if not provided)
        """
        super().__init__(
            db=db,
            settings=settings,
            ontology_handler=ontology_handler,
            log_name="content_extraction",
        )
        self._content_collection = db.collection(Collections.CONTENT_NODES)
        self._edge_collection = db.collection(EdgeCollections.KNOWLEDGE_EDGES)

    def _get_allowed_relations(self) -> frozenset[str]:
        """Get allowed relations for Tier 2 from database (cached)."""
        return self._get_allowed_relations_for_tier(
            tier=RelationTier.TIER_2_CONTENT,
            fallback_relations=_FALLBACK_CONTENT_RELATIONS,
        )

    # =========================================================================
    # Similarity Search
    # =========================================================================

    async def find_similar_nodes(
        self,
        node: ContentNode,
        limit: int | None = None,
        min_similarity: float | None = None,
        cross_source: bool = True,
    ) -> list[SimilarNode]:
        """Find similar nodes using cosine similarity.

        Compares unreviewed nodes (is_reviewed=false) against reviewed/stable nodes
        (is_reviewed=true) to find potential duplicates.

        Args:
            node: The node to find similar nodes for (should be unreviewed)
            limit: Maximum number of results (None = no limit, return all above threshold)
            min_similarity: Minimum similarity threshold
            cross_source: If True, search across all sources

        Returns:
            List of SimilarNode sorted by similarity descending
        """
        if node.embedding is None:
            return []

        validate_embedding(node.embedding)

        min_similarity = min_similarity or self.settings.EXTRACTION_LLM_THRESHOLD
        collection_name = Collections.CONTENT_NODES

        # Build source filter
        source_filter = "" if cross_source else "FILTER doc.source_id == @source_id"

        # Build optional limit clause
        limit_clause = "LIMIT @limit" if limit is not None else ""

        # Compare against REVIEWED nodes only
        # This implements the logic: unreviewed nodes compare with reviewed (stable) nodes
        query = f"""
        FOR doc IN @@collection
            FILTER doc._key != @node_key
            FILTER doc.embedding != null
            FILTER doc.is_reviewed == true
            {source_filter}
            LET similarity = COSINE_SIMILARITY(doc.embedding, @embedding)
            FILTER similarity >= @min_similarity
            SORT similarity DESC
            {limit_clause}
            RETURN {{
                doc: doc,
                similarity: similarity
            }}
        """

        bind_vars = {
            "@collection": collection_name,
            "node_key": str(node.id),
            "embedding": node.embedding,
            "min_similarity": min_similarity,
        }

        if limit is not None:
            bind_vars["limit"] = limit

        if not cross_source and node.source_id:
            bind_vars["source_id"] = str(node.source_id)

        cursor = await asyncio.to_thread(
            partial(self.db.aql.execute, query, bind_vars=bind_vars),
        )

        results = []
        for item in cursor:
            doc = item["doc"]
            try:
                content_node = ContentNode.from_document(doc)
                results.append(
                    SimilarNode(
                        node_id=doc["_key"],
                        node=content_node,
                        similarity_score=item["similarity"],
                    )
                )
            except (ValidationError, KeyError, TypeError) as e:
                self._log.warning(
                    "Failed to parse document as ContentNode",
                    doc_key=doc.get("_key"),
                    error=str(e),
                )

        return results

    # =========================================================================
    # Merge Operations
    # =========================================================================

    async def auto_merge(
        self,
        source_node: ContentNode,
        target_node: ContentNode,
        similarity: float = 0.0,
        merge_reason: Literal["auto_merge", "llm_merge"] = "auto_merge",
        llm_reasoning: str | None = None,
    ) -> ExtractionResult:
        """Auto-merge two nodes (similarity >= threshold).

        The source node is merged into target using ArangoDB Stream Transaction
        for ACID guarantees. All operations either complete together or rollback.

        Operations within transaction:
        1. Mark target as reviewed
        2. Redirect edges pointing to source -> target
        3. Redirect edges from source -> target
        4. Delete source node
        5. Insert merge log record for benchmark evaluation

        Uses ArangoDB Stream Transactions (available since ArangoDB 3.5+).

        Args:
            source_node: Node to be merged (will be deleted)
            target_node: Node to merge into (will be kept)
            similarity: Cosine similarity score at merge time
            merge_reason: "auto_merge" for threshold-based, "llm_merge" for LLM-verified
            llm_reasoning: LLM reasoning if merge_reason is "llm_merge"
        """
        source_key = str(source_node.id)
        target_key = str(target_node.id)

        # Swap direction: always keep the LONGER node (delete shorter one)
        source_text_len = len(getattr(source_node, "text", "") or "")
        target_text_len = len(getattr(target_node, "text", "") or "")
        if source_text_len > target_text_len:
            source_node, target_node = target_node, source_node
            source_key, target_key = target_key, source_key
            self._log.debug(
                "Swap merge direction to keep longer node",
                kept=target_key,
                deleted=source_key,
                kept_len=source_text_len,
                deleted_len=target_text_len,
            )

        # Validation: prevent self-merge
        if source_key == target_key:
            self._log.warning(
                "Cannot merge node with itself",
                source_key=source_key,
            )
            return ExtractionResult(
                decision=ExtractionDecision.KEEP_SEPARATE,
                target_node_id=None,
                confidence=0.0,
                method="validation",
                reasoning="Cannot merge node with itself",
            )

        source_collection = Collections.CONTENT_NODES
        target_collection = Collections.CONTENT_NODES

        source_full_id = f"{source_collection}/{source_key}"
        target_full_id = f"{target_collection}/{target_key}"

        # Collections involved in transaction (read/write)
        write_collections = [
            source_collection,
            EdgeCollections.KNOWLEDGE_EDGES,
            Collections.MERGED_NODES_LOG,
        ]
        if target_collection != source_collection:
            write_collections.append(target_collection)

        try:
            # Begin Stream Transaction
            txn = await asyncio.to_thread(
                partial(
                    self.db.begin_transaction,
                    read=write_collections,
                    write=write_collections,
                ),
            )

            try:
                # Step 1: Mark target as reviewed
                query_mark = """
                UPDATE @target_key WITH { is_reviewed: true } IN @@target_collection
                RETURN 1
                """
                await asyncio.to_thread(
                    partial(
                        txn.aql.execute,
                        query_mark,
                        bind_vars={
                            "@target_collection": target_collection,
                            "target_key": target_key,
                        },
                    ),
                )

                # Step 2: Update edges where _to == source -> target
                query_edges_to = """
                FOR e IN @@edge_collection
                    FILTER e._to == @source_full_id
                    UPDATE e WITH { _to: @target_full_id } IN @@edge_collection
                    RETURN 1
                """
                await asyncio.to_thread(
                    partial(
                        txn.aql.execute,
                        query_edges_to,
                        bind_vars={
                            "@edge_collection": EdgeCollections.KNOWLEDGE_EDGES,
                            "source_full_id": source_full_id,
                            "target_full_id": target_full_id,
                        },
                    ),
                )

                # Step 3: Update edges where _from == source -> target
                query_edges_from = """
                FOR e IN @@edge_collection
                    FILTER e._from == @source_full_id
                    UPDATE e WITH { _from: @target_full_id } IN @@edge_collection
                    RETURN 1
                """
                await asyncio.to_thread(
                    partial(
                        txn.aql.execute,
                        query_edges_from,
                        bind_vars={
                            "@edge_collection": EdgeCollections.KNOWLEDGE_EDGES,
                            "source_full_id": source_full_id,
                            "target_full_id": target_full_id,
                        },
                    ),
                )

                # Step 4: Delete source node
                query_delete = """
                REMOVE @source_key IN @@source_collection
                RETURN 1
                """
                await asyncio.to_thread(
                    partial(
                        txn.aql.execute,
                        query_delete,
                        bind_vars={
                            "@source_collection": source_collection,
                            "source_key": source_key,
                        },
                    ),
                )

                # Step 5: Insert merge log record for benchmark evaluation
                merge_log = MergedNodeLog(
                    source_id=source_full_id,
                    source_text=source_node.get_content_text(),
                    source_type=(
                        source_node.node_type.value
                        if hasattr(source_node.node_type, "value")
                        else str(source_node.node_type)
                    ),
                    source_parent_id=(
                        str(source_node.parent_id) if source_node.parent_id is not None else None
                    ),
                    source_source_id=(
                        str(source_node.source_id) if source_node.source_id is not None else None
                    ),
                    target_id=target_full_id,
                    target_text=target_node.get_content_text(),
                    similarity=similarity,
                    merge_reason=merge_reason,
                    llm_reasoning=llm_reasoning,
                )
                await asyncio.to_thread(
                    partial(
                        txn.collection(Collections.MERGED_NODES_LOG).insert,
                        merge_log.to_document(),
                    ),
                )

                # Commit transaction - all operations succeed together
                await asyncio.to_thread(txn.commit_transaction)

                self._log.debug(
                    "Auto-merge completed successfully (transaction committed)",
                    source_key=source_key,
                    target_key=target_key,
                    merge_reason=merge_reason,
                    similarity=similarity,
                )

                return ExtractionResult(
                    decision=ExtractionDecision.MERGE,
                    target_node_id=target_key,
                    confidence=similarity if similarity > 0 else 1.0,
                    method=f"embedding_{merge_reason}",
                    reasoning=(
                        llm_reasoning
                        if merge_reason == "llm_merge" and llm_reasoning
                        else f"Merged via {merge_reason}: similarity={similarity:.3f}"
                    ),
                )

            except Exception as e:
                # Abort transaction - all operations rollback
                await asyncio.to_thread(txn.abort_transaction)
                self._log.error(
                    "Auto-merge failed - transaction aborted (no partial state)",
                    source_key=source_key,
                    target_key=target_key,
                    error=str(e),
                )
                raise

        except Exception as e:
            self._log.error(
                "Auto-merge transaction error",
                source_key=source_key,
                target_key=target_key,
                error=str(e),
            )
            raise

    # =========================================================================
    # Relation Operations
    # =========================================================================

    async def create_relation_edge(
        self,
        source_node: ContentNode,
        target_node: ContentNode,
        relation_type: str,
        confidence: float = 0.0,
        method: str = "unknown",
    ) -> str:
        """Create a relation edge between two nodes.

        Validates relation_type against tier_2_content relations from database.
        """
        # Validate relation type against dynamic allowlist from database
        allowed_relations = self._get_allowed_relations()
        if relation_type not in allowed_relations:
            self._log.warning(
                "Invalid relation type, using fallback",
                provided=relation_type,
                fallback=self.settings.EXTRACTION_FALLBACK_RELATION,
                allowed_count=len(allowed_relations),
                source_node=str(source_node.id)[:50],
                target_node=str(target_node.id)[:50],
            )
            relation_type = self.settings.EXTRACTION_FALLBACK_RELATION

        source_collection = Collections.CONTENT_NODES
        target_collection = Collections.CONTENT_NODES

        source_id = f"{source_collection}/{source_node.id}"
        target_id = f"{target_collection}/{target_node.id}"

        edge_doc = {
            "_key": str(uuid.uuid4()),
            "_from": source_id,
            "_to": target_id,
            "type": relation_type,
            "confidence": confidence,
            "detection_method": method,
        }

        result = await asyncio.to_thread(
            partial(self._edge_collection.insert, edge_doc, return_new=True),
        )
        return result["new"]["_key"]

    # =========================================================================
    # Node Queries
    # =========================================================================

    async def node_exists(self, node: ContentNode) -> bool:
        """Check if a node still exists in the database."""
        collection_name = Collections.CONTENT_NODES
        query = "RETURN DOCUMENT(@@collection, @key) != null"

        cursor = await asyncio.to_thread(
            partial(
                self.db.aql.execute,
                query,
                bind_vars={"@collection": collection_name, "key": str(node.id)},
            ),
        )
        return next(cursor, False)

    async def mark_as_reviewed(self, node_ids: list[str]) -> int:
        """Mark nodes as reviewed for extraction."""
        return await super().mark_as_reviewed(node_ids, Collections.CONTENT_NODES)

    async def get_unreviewed_nodes(
        self,
        source_id: str | uuid.UUID | None = None,
        limit: int | None = None,
        offset: int = 0,
    ) -> list[ContentNode]:
        """Get nodes that haven't been reviewed.

        Args:
            source_id: Optional source ID to filter by (None = all sources)
            limit: Maximum nodes to return
            offset: Number of nodes to skip
        """
        limit = limit or self.settings.EXTRACTION_BATCH_SIZE
        source_filter = "FILTER doc.source_id == @source_id" if source_id else ""

        query = f"""
        FOR doc IN @@collection
            FILTER doc.is_reviewed != true
            {source_filter}
            FILTER doc.embedding != null
            SORT doc.created_at DESC
            LIMIT @offset, @limit
            RETURN doc
        """

        bind_vars = {
            "@collection": Collections.CONTENT_NODES,
            "offset": offset,
            "limit": limit,
        }
        if source_id:
            bind_vars["source_id"] = str(source_id)

        cursor = await asyncio.to_thread(
            partial(self.db.aql.execute, query, bind_vars=bind_vars),
        )

        results = []
        for doc in cursor:
            try:
                results.append(ContentNode.from_document(doc))
            except (ValidationError, KeyError, TypeError) as e:
                self._log.warning(
                    "Failed to parse document as ContentNode",
                    doc_key=doc.get("_key"),
                    error=str(e),
                )

        return results

    async def count_unreviewed_nodes(
        self,
        source_id: str | uuid.UUID | None = None,
    ) -> int:
        """Count nodes that haven't been reviewed."""
        source_filter = "FILTER doc.source_id == @source_id" if source_id else ""

        query = f"""
        FOR doc IN @@collection
            FILTER doc.is_reviewed != true
            {source_filter}
            FILTER doc.embedding != null
            COLLECT WITH COUNT INTO count
            RETURN count
        """

        bind_vars = {"@collection": Collections.CONTENT_NODES}
        if source_id:
            bind_vars["source_id"] = str(source_id)

        cursor = await asyncio.to_thread(
            partial(self.db.aql.execute, query, bind_vars=bind_vars),
        )
        return next(cursor, 0)
