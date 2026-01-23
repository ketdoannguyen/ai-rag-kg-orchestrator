"""Structure extraction handler for Tier 1 structure nodes.

Provides StructureExtractionHandler with:
1. Title Hash Matching: Exact title match (after normalization) -> auto-merge
2. Cosine Similarity: Embedding-based similarity search
3. Unified thresholds (same as content tier)

Key Features:
- Vietnamese diacritic normalization
- Level compatibility for merge
- Hierarchy-aware merge (children re-parenting)
- Cross-source deduplication support
"""

from __future__ import annotations

import asyncio
import hashlib
import re
import unicodedata
import uuid
from dataclasses import dataclass
from functools import partial
from typing import TYPE_CHECKING

from arango.database import StandardDatabase

from ..collections.schema import Collections, EdgeCollections
from ..models.ontology import RelationTier
from ..models.structure_nodes import StructureNode
from ..utils import validate_embedding
from .base_extraction_handler import BaseExtractionHandler
from .ontology_handler import OntologyHandler

if TYPE_CHECKING:
    from config.extraction import ExtractionSettings

# Fallback relation types for structure nodes (used when database is unavailable)
_FALLBACK_STRUCTURE_RELATIONS = frozenset(
    {
        "SIMILAR_TO",
        "STRUCTURALLY_EQUIVALENT",
        "HAS_SECTION",
        "HAS_CONTENT",
        "VARIANT_OF",
        "DUPLICATE_OF",
        "NEXT_SIBLING",
        "FOLLOWS",
        "REFERS_TO",
    }
)

# Relation type for high-similarity structure pairs (instead of merge)
STRUCTURALLY_EQUIVALENT_RELATION = "STRUCTURALLY_EQUIVALENT"


@dataclass
class SimilarStructure:
    """A structure node found by similarity search with its score."""

    node_id: str
    node: StructureNode
    similarity_score: float
    title_match: bool = False


class StructureExtractionHandler(BaseExtractionHandler):
    """Handle structure node extraction with hierarchy awareness.

    Uses unified thresholds from settings (same as content tier):
    - >= AUTO_MERGE_THRESHOLD: Auto-merge
    - >= LLM_THRESHOLD: LLM verification
    - < LLM_THRESHOLD: Keep separate
    """

    def __init__(
        self,
        db: StandardDatabase,
        settings: ExtractionSettings | None = None,
        ontology_handler: OntologyHandler | None = None,
    ) -> None:
        """Initialize structure extraction handler.

        Args:
            db: ArangoDB database connection
            settings: Extraction settings (lazy loaded if not provided)
            ontology_handler: Handler for ontology operations (lazy loaded if not provided)
        """
        super().__init__(
            db=db,
            settings=settings,
            ontology_handler=ontology_handler,
            log_name="structure_extraction",
        )
        self._structure_collection = db.collection(Collections.STRUCTURE_NODES)
        self._edge_collection = db.collection(EdgeCollections.KNOWLEDGE_EDGES)

    def _get_allowed_relations(self) -> frozenset[str]:
        """Get allowed relations for Tier 1 from database (cached)."""
        return self._get_allowed_relations_for_tier(
            tier=RelationTier.TIER_1_STRUCTURE,
            fallback_relations=_FALLBACK_STRUCTURE_RELATIONS,
        )

    # =========================================================================
    # Vietnamese Diacritic Normalization
    # =========================================================================

    def _normalize_vietnamese(self, text: str) -> str:
        """Normalize Vietnamese text by removing diacritics.

        Converts: "Chương" -> "chuong", "Đạo hàm" -> "dao ham"
        """
        if not text:
            return ""

        # Unicode normalization (NFD decomposes characters)
        normalized = unicodedata.normalize("NFD", text)

        # Remove combining diacritical marks
        normalized = "".join(c for c in normalized if unicodedata.category(c) != "Mn")

        # Handle special Vietnamese characters
        replacements = {
            "đ": "d",
            "Đ": "D",
            "ø": "o",
            "Ø": "O",
        }
        for old, new in replacements.items():
            normalized = normalized.replace(old, new)

        return normalized.lower()

    def _normalize_title(self, title: str) -> str:
        """Normalize title for comparison.

        Steps:
        1. Vietnamese diacritic removal
        2. Lowercase
        3. Remove extra whitespace
        4. Remove common prefixes (Chương 1:, Phần A:, etc.)
        5. Remove numeric/roman equivalence normalization
        """
        if not title:
            return ""

        # Apply Vietnamese normalization
        normalized = self._normalize_vietnamese(title)

        # Remove extra whitespace
        normalized = re.sub(r"\s+", " ", normalized).strip()

        # Remove common prefixes (handle both with and without diacritics)
        prefix_pattern = (
            r"^(chuong|phan|muc|bai|section|chapter|part|"
            r"chu[oơ]ng|ph[aầ]n|m[uụ]c|b[aà]i)"
            r"\s*[\d\w]+[:\.\-]\s*"
        )
        normalized = re.sub(prefix_pattern, "", normalized, flags=re.IGNORECASE)

        # Normalize Roman numerals to Arabic
        roman_map = {"i": "1", "ii": "2", "iii": "3", "iv": "4", "v": "5"}
        for roman, arabic in roman_map.items():
            normalized = re.sub(
                rf"\b{roman}\b",
                arabic,
                normalized,
                flags=re.IGNORECASE,
            )

        return normalized

    def _hash_title(self, title: str) -> str:
        """Create hash of normalized title."""
        normalized = self._normalize_title(title)
        if not normalized:
            return ""
        return hashlib.sha256(normalized.encode("utf-8")).hexdigest()[:16]

    # =========================================================================
    # Similarity Search (Cosine Similarity)
    # =========================================================================

    async def find_title_matches(
        self,
        node: StructureNode,
        exclude_ids: set[str] | None = None,
        cross_source: bool = True,
    ) -> list[SimilarStructure]:
        """Find structure nodes with exact title match at same level.

        Compares unreviewed nodes against reviewed/stable nodes (is_reviewed=true).

        Args:
            node: Structure node to find matches for (should be unreviewed)
            exclude_ids: Node IDs to exclude from search
            cross_source: If True, search across all sources

        Returns:
            List of matching structure nodes
        """
        exclude_ids = exclude_ids or set()
        title = self._get_title(node)
        title_hash = self._hash_title(title)

        if not title_hash:
            return []

        # Build query - compare against REVIEWED nodes only
        if cross_source:
            query = """
            FOR doc IN @@collection
                FILTER doc._key != @current_key
                FILTER doc.level == @level
                FILTER doc.is_reviewed == true
                RETURN doc
            """
            bind_vars = {
                "@collection": Collections.STRUCTURE_NODES,
                "current_key": str(node.id),
                "level": node.level,
            }
        else:
            query = """
            FOR doc IN @@collection
                FILTER doc._key != @current_key
                FILTER doc.level == @level
                FILTER doc.source_id == @source_id
                FILTER doc.is_reviewed == true
                RETURN doc
            """
            bind_vars = {
                "@collection": Collections.STRUCTURE_NODES,
                "current_key": str(node.id),
                "level": node.level,
                "source_id": str(node.source_id) if node.source_id else None,
            }

        cursor = await asyncio.to_thread(
            partial(self.db.aql.execute, query, bind_vars=bind_vars),
        )

        results = []
        for doc in cursor:
            if doc["_key"] in exclude_ids:
                continue

            # Check title match
            doc_title = doc.get("attributes", {}).get("title", "")
            if self._hash_title(doc_title) == title_hash:
                try:
                    struct_node = StructureNode.from_document(doc)
                    results.append(
                        SimilarStructure(
                            node_id=doc["_key"],
                            node=struct_node,
                            similarity_score=1.0,
                            title_match=True,
                        )
                    )
                except Exception as e:
                    self._log.warning(
                        "Failed to parse structure node",
                        doc_key=doc.get("_key"),
                        error=str(e),
                    )

        return results

    async def find_similar_structures(
        self,
        node: StructureNode,
        max_candidates: int | None = None,
        exclude_ids: set[str] | None = None,
        cross_source: bool = True,
    ) -> list[SimilarStructure]:
        """Find similar structure nodes using cosine similarity.

        Compares unreviewed nodes (is_reviewed=false) against reviewed/stable nodes
        (is_reviewed=true) to find potential duplicates.

        Uses proper cosine similarity: dot(A,B) / (||A|| * ||B||)

        Args:
            node: Structure node to find similar nodes for (should be unreviewed)
            max_candidates: Maximum candidates (default from settings)
            exclude_ids: Node IDs to exclude from search
            cross_source: If True, search across all sources

        Returns:
            List of similar structure nodes sorted by similarity
        """
        exclude_ids = exclude_ids or set()
        max_candidates = max_candidates or self.settings.EXTRACTION_MAX_CANDIDATES_PER_NODE

        if not node.embedding:
            return []

        validate_embedding(node.embedding)

        # Cosine similarity query - compare against REVIEWED nodes only
        # This implements the logic: unreviewed nodes compare with reviewed (stable) nodes
        source_filter = "" if cross_source else "FILTER doc.source_id == @source_id"

        query = f"""
        FOR doc IN @@collection
            FILTER doc._key != @current_key
            FILTER doc.level == @level
            FILTER doc.embedding != null
            FILTER doc.is_reviewed == true
            {source_filter}
            LET similarity = COSINE_SIMILARITY(doc.embedding, @embedding)
            FILTER similarity >= @min_threshold
            SORT similarity DESC
            LIMIT @limit
            RETURN {{
                doc: doc,
                similarity: similarity
            }}
        """

        bind_vars = {
            "@collection": Collections.STRUCTURE_NODES,
            "current_key": str(node.id),
            "level": node.level,
            "embedding": node.embedding,
            "min_threshold": self.settings.EXTRACTION_LLM_THRESHOLD,
            "limit": max_candidates,
        }

        if not cross_source and node.source_id:
            bind_vars["source_id"] = str(node.source_id)

        cursor = await asyncio.to_thread(
            partial(self.db.aql.execute, query, bind_vars=bind_vars),
        )

        results = []
        for item in cursor:
            doc = item["doc"]
            if doc["_key"] in exclude_ids:
                continue

            try:
                struct_node = StructureNode.from_document(doc)
                results.append(
                    SimilarStructure(
                        node_id=doc["_key"],
                        node=struct_node,
                        similarity_score=item["similarity"],
                        title_match=False,
                    )
                )
            except Exception as e:
                self._log.warning(
                    "Failed to parse structure node",
                    doc_key=doc.get("_key"),
                    error=str(e),
                )

        return results

    # =========================================================================
    # Node Queries
    # =========================================================================

    def _get_title(self, node: StructureNode) -> str:
        """Get title from structure node."""
        attrs = getattr(node, "attributes", None)
        if not attrs:
            return ""
        return getattr(attrs, "title", "") or ""

    async def get_unreviewed_structures(
        self,
        source_id: str | uuid.UUID | None = None,
        limit: int | None = None,
        offset: int = 0,
    ) -> list[StructureNode]:
        """Get structure nodes that haven't been reviewed.

        Args:
            source_id: Optional source ID to filter by (None = all sources)
            limit: Maximum nodes to return (default from settings)
            offset: Number of nodes to skip

        Returns:
            List of StructureNode instances
        """
        limit = limit or self.settings.EXTRACTION_BATCH_SIZE
        source_filter = "FILTER doc.source_id == @source_id" if source_id else ""

        query = f"""
        FOR doc IN @@collection
            FILTER doc.is_reviewed != true
            {source_filter}
            SORT doc.level ASC, doc.created_at DESC
            LIMIT @offset, @limit
            RETURN doc
        """

        bind_vars = {
            "@collection": Collections.STRUCTURE_NODES,
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
                results.append(StructureNode.from_document(doc))
            except Exception as e:
                self._log.warning(
                    "Failed to parse structure node",
                    doc_key=doc.get("_key"),
                    error=str(e),
                )

        return results

    # =========================================================================
    # Merge Operations
    # =========================================================================

    async def merge_structures(
        self,
        source: StructureNode,
        target: StructureNode,
    ) -> bool:
        """Merge source structure into target.

        Uses ArangoDB Stream Transaction for ACID guarantees.
        All operations either complete together or rollback.

        Operations within transaction:
        1. Mark target as reviewed
        2. Re-parent children of source to target
        3. Redirect edges pointing to source -> target
        4. Redirect edges from source -> target
        5. Delete source node

        Uses ArangoDB Stream Transactions (available since ArangoDB 3.5+).

        Returns:
            True if merge successful
        """
        source_id = str(source.id)
        target_id = str(target.id)

        # Validation: prevent self-merge
        if source_id == target_id:
            self._log.warning(
                "Cannot merge structure node with itself",
                source_id=source_id,
            )
            return False

        source_full_id = f"{Collections.STRUCTURE_NODES}/{source_id}"
        target_full_id = f"{Collections.STRUCTURE_NODES}/{target_id}"

        # Collections involved in transaction (read/write)
        write_collections = [
            Collections.STRUCTURE_NODES,
            EdgeCollections.KNOWLEDGE_EDGES,
        ]

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
                UPDATE @target_id WITH { is_reviewed: true } IN @@structure_collection
                RETURN 1
                """
                await asyncio.to_thread(
                    partial(
                        txn.aql.execute,
                        query_mark,
                        bind_vars={
                            "@structure_collection": Collections.STRUCTURE_NODES,
                            "target_id": target_id,
                        },
                    ),
                )

                # Step 2: Re-parent children of source to target
                query_children = """
                FOR doc IN @@structure_collection
                    FILTER doc.parent_id == @source_id
                    UPDATE doc WITH { parent_id: @target_id } IN @@structure_collection
                    RETURN 1
                """
                await asyncio.to_thread(
                    partial(
                        txn.aql.execute,
                        query_children,
                        bind_vars={
                            "@structure_collection": Collections.STRUCTURE_NODES,
                            "source_id": source_id,
                            "target_id": target_id,
                        },
                    ),
                )

                # Step 3: Update edges where _to == source -> target
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

                # Step 4: Update edges where _from == source -> target
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

                # Step 5: Delete source node
                query_delete = """
                REMOVE @source_id IN @@structure_collection
                RETURN 1
                """
                await asyncio.to_thread(
                    partial(
                        txn.aql.execute,
                        query_delete,
                        bind_vars={
                            "@structure_collection": Collections.STRUCTURE_NODES,
                            "source_id": source_id,
                        },
                    ),
                )

                # Commit transaction - all operations succeed together
                await asyncio.to_thread(txn.commit_transaction)

                self._log.info(
                    "Merged structure nodes successfully (transaction committed)",
                    source=source_id,
                    target=target_id,
                )
                return True

            except Exception as e:
                # Abort transaction - all operations rollback
                await asyncio.to_thread(txn.abort_transaction)
                self._log.error(
                    "Failed to merge structure nodes - transaction aborted (no partial state)",
                    source=source_id,
                    target=target_id,
                    error=str(e),
                )
                return False

        except Exception as e:
            self._log.error(
                "Structure merge transaction error",
                source=source_id,
                target=target_id,
                error=str(e),
            )
            return False

    async def mark_as_reviewed(self, node_ids: list[str]) -> int:
        """Mark structure nodes as reviewed."""
        return await super().mark_as_reviewed(node_ids, Collections.STRUCTURE_NODES)

    # =========================================================================
    # Relation Operations
    # =========================================================================

    async def create_structure_relation(
        self,
        source: StructureNode,
        target: StructureNode,
        relation_type: str,
        confidence: float,
        method: str = "embedding",
    ) -> str:
        """Create a relation edge between two structure nodes.

        Validates relation_type against tier_1_structure relations from database.
        """
        # Validate relation_type against dynamic allowlist from database
        allowed_relations = self._get_allowed_relations()
        if relation_type not in allowed_relations:
            self._log.warning(
                "Invalid relation type, using fallback",
                provided=relation_type,
                fallback=self.settings.EXTRACTION_FALLBACK_RELATION,
            )
            relation_type = self.settings.EXTRACTION_FALLBACK_RELATION

        source_full_id = f"{Collections.STRUCTURE_NODES}/{source.id}"
        target_full_id = f"{Collections.STRUCTURE_NODES}/{target.id}"

        edge_doc = {
            "_from": source_full_id,
            "_to": target_full_id,
            "type": relation_type,
            "confidence": confidence,
            "detection_method": method,
        }

        result = await asyncio.to_thread(
            partial(self._edge_collection.insert, edge_doc, return_new=True),
        )
        return result["new"]["_key"]

    async def create_structurally_equivalent_relation(
        self,
        source: StructureNode,
        target: StructureNode,
        similarity: float,
    ) -> str:
        """Create STRUCTURALLY_EQUIVALENT relation for high-similarity structure pairs.

        This is used instead of merge for structure nodes to preserve tree hierarchy.
        Called when similarity >= auto_merge_threshold (0.72).

        Args:
            source: Source structure node
            target: Target structure node
            similarity: Cosine similarity score

        Returns:
            Edge key of created relation
        """
        return await self.create_structure_relation(
            source=source,
            target=target,
            relation_type=STRUCTURALLY_EQUIVALENT_RELATION,
            confidence=similarity,
            method="high_similarity_no_merge",
        )
