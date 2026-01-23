"""
Knowledge handler for ArangoDB - combines PostgreSQL and Neo4j functionality.

This is the main handler for knowledge operations, merging:
- Document CRUD (from postgres_client.handlers.knowledge_handler)
- Graph relationships (from neo4j_client.handlers.knowledge_handler)
- Vector embeddings (from postgres_client.handlers.embedding_handler)

All AQL queries are imported from the centralized aql/ module.
"""

from __future__ import annotations

import uuid
from datetime import UTC, datetime
from typing import Any, Union

from arango.database import StandardDatabase
from loguru import logger

from ..aql import (
    CRUD_GET_ALL,
    CRUD_GET_BY_FILTER,
    GET_ANCESTORS,
    GET_DESCENDANTS,
    GET_DIRECT_CHILDREN,
    GET_NEXT_SIBLING,
    GET_NODES_AT_LEVEL,
    GET_PREV_SIBLING,
    GET_SIBLINGS,
)
from ..collections.schema import (
    KNOWLEDGE_COLLECTIONS,
    KNOWLEDGE_GRAPH_NAME,
    Collections,
    EdgeCollections,
    RelationshipTypes,
)
from ..models.detail_nodes import ImageNode, ParagraphNode, TableNode
from ..models.enums import NodeType
from ..models.knowledge import KnowledgeNode
from ..models.structure_nodes import RootNode, StructureNode
from ..utils import doc_to_knowledge_node
from .base_handler import BaseHandler

# Type alias for all knowledge node types
KnowledgeNodeType = Union[
    KnowledgeNode,
    ParagraphNode,
    ImageNode,
    TableNode,
    StructureNode,
    RootNode,
]


def get_collection_for_node(node: KnowledgeNodeType) -> str:
    """
    Determine which collection a node belongs to based on type and level.

    Rules:
    - level == 1 -> root_nodes (RootNode)
    - StructureNode with level > 1 -> structure_nodes
    - ParagraphNode -> content_nodes
    - ImageNode -> content_nodes
    - TableNode -> content_nodes
    """
    # Check for RootNode or level == 1
    if isinstance(node, RootNode) or getattr(node, "level", 0) == 1:
        return Collections.ROOT_NODES
    if isinstance(node, StructureNode):
        return Collections.STRUCTURE_NODES

    # Detail nodes - check node_type
    node_type = node.node_type
    if hasattr(node_type, "value"):
        node_type = node_type.value

    return {
        "paragraph": Collections.CONTENT_NODES,
        "image": Collections.CONTENT_NODES,
        "table": Collections.CONTENT_NODES,
    }.get(node_type, Collections.STRUCTURE_NODES)


class KnowledgeNodeHandler(BaseHandler[KnowledgeNode]):
    """
    Handler for KnowledgeNode CRUD and graph operations in ArangoDB.

    Features:
    - Document CRUD operations
    - Graph relationship management (HAS_SECTION, HAS_CONTENT, NEXT_SIBLING)
    - Embedded vector storage and updates
    - Tree traversal and reconstruction

    Example:
        >>> handler = KnowledgeNodeHandler(db)
        >>> node = handler.create_node(ParagraphNode(...))
        >>> handler.create_relationship(parent_id, node.id, "HAS_CONTENT")
        >>> children = handler.get_children(parent_id)
    """

    def __init__(self, db: StandardDatabase):
        """Initialize KnowledgeNode handler."""
        # Use first collection as default for BaseHandler (not actually used)
        super().__init__(db, KnowledgeNode, Collections.CONTENT_NODES)
        self._edge_collection = db.collection(EdgeCollections.KNOWLEDGE_EDGES)
        self._graph = db.graph(KNOWLEDGE_GRAPH_NAME)

    # =========================================================================
    # Document CRUD Operations
    # =========================================================================

    def create_node(self, node: KnowledgeNodeType) -> KnowledgeNodeType:
        """Create a new knowledge node in the appropriate collection."""
        doc = node.to_document()
        collection_name = get_collection_for_node(node)
        collection = self.db.collection(collection_name)
        result = collection.insert(doc, return_new=True)
        return doc_to_knowledge_node(result["new"])

    def get_by_id(
        self,
        id: uuid.UUID | str,
        node_class: type[KnowledgeNodeType] | None = None,
    ) -> KnowledgeNodeType | None:
        """Get a knowledge node by ID, searching across all collections."""
        key = str(id)

        # Search across all knowledge collections
        for collection_name in KNOWLEDGE_COLLECTIONS:
            collection = self.db.collection(collection_name)
            doc = collection.get(key)
            if doc is not None:
                return doc_to_knowledge_node(doc, node_class)

        return None

    def _find_node_collection(self, node_id: uuid.UUID | str) -> str | None:
        """Find which collection contains the given node ID."""
        key = str(node_id)
        for collection_name in KNOWLEDGE_COLLECTIONS:
            collection = self.db.collection(collection_name)
            if collection.has(key):
                return collection_name
        return None

    def get_by_filter(
        self,
        filters: dict[str, Any],
        limit: int | None = None,
        offset: int | None = None,
        sort_field: str = "created_at",
        sort_desc: bool = True,
    ) -> list[KnowledgeNodeType]:
        """Get nodes by filter criteria with proper type conversion.

        This method queries ALL knowledge collections (content_nodes, structure_nodes,
        root_nodes, entity_nodes) and returns combined results.

        Args:
            filters: Dictionary of field-value pairs to filter by
            limit: Maximum number of results. If None, returns all matching nodes (up to 10000)
            offset: Number of results to skip
            sort_field: Field to sort by (default: created_at)
            sort_desc: Sort descending if True
        """
        sort_dir = "DESC" if sort_desc else "ASC"
        # If limit is None, use a very large number to effectively get all results
        effective_limit = limit if limit is not None else 10000
        offset = offset or 0

        filter_conditions = []
        bind_vars = {"offset": offset, "limit": effective_limit}

        for i, (field, value) in enumerate(filters.items()):
            var_name = f"val{i}"
            filter_conditions.append(f"doc.{field} == @{var_name}")
            bind_vars[var_name] = str(value) if isinstance(value, uuid.UUID) else value

        filter_clause = " AND ".join(filter_conditions) if filter_conditions else "true"

        # Query ALL knowledge collections and combine results
        all_results: list[KnowledgeNodeType] = []

        for collection_name in KNOWLEDGE_COLLECTIONS:
            # Use centralized AQL template
            query = CRUD_GET_BY_FILTER.format(
                collection=collection_name,
                filter_clause=filter_clause,
                sort_field=sort_field,
                sort_dir=sort_dir,
            )

            cursor = self.db.aql.execute(query, bind_vars=bind_vars)
            all_results.extend([doc_to_knowledge_node(doc) for doc in cursor])

        # Sort combined results if needed
        if sort_field and all_results:
            reverse = sort_desc
            all_results.sort(
                key=lambda n: getattr(n, sort_field, None) or "",
                reverse=reverse,
            )

        # Apply offset and limit to combined results
        if offset or effective_limit < len(all_results):
            all_results = all_results[offset : offset + effective_limit]

        return all_results

    def get_by_source(
        self,
        source_id: uuid.UUID | str,
        limit: int | None = None,
        offset: int | None = None,
    ) -> list[KnowledgeNodeType]:
        """Get knowledge nodes by source ID."""
        return self.get_by_filter(
            {"source_id": str(source_id)},
            limit=limit,
            offset=offset,
            sort_desc=True,
        )

    def get_by_node_type(
        self,
        node_type: NodeType | str,
        limit: int | None = None,
        offset: int | None = None,
    ) -> list[KnowledgeNodeType]:
        """Get knowledge nodes by node type."""
        type_value = node_type.value if isinstance(node_type, NodeType) else node_type
        return self.get_by_filter(
            {"node_type": type_value},
            limit=limit,
            offset=offset,
            sort_desc=True,
        )

    # =========================================================================
    # Embedding Operations
    # =========================================================================

    def update(
        self,
        id: uuid.UUID | str,
        update_data: dict[str, Any],
    ) -> KnowledgeNodeType | None:
        """Update a knowledge node by ID, searching across all collections.

        Overrides BaseHandler.update() to search across all knowledge collections
        instead of a single collection.

        Args:
            id: Primary key value
            update_data: Dictionary of fields to update

        Returns:
            Updated KnowledgeNode or None if not found
        """
        key = str(id)

        # Find which collection contains this node
        collection_name = self._find_node_collection(id)
        if collection_name is None:
            return None

        collection = self.db.collection(collection_name)

        # Add updated_at timestamp if the model supports it
        if "updated_at" not in update_data:
            update_data["updated_at"] = datetime.now(UTC).isoformat()

        # Convert UUIDs to strings
        for field, value in update_data.items():
            if isinstance(value, uuid.UUID):
                update_data[field] = str(value)

        result = collection.update(
            {"_key": key, **update_data},
            return_new=True,
        )

        return doc_to_knowledge_node(result["new"])

    def update_embedding(
        self,
        node_id: uuid.UUID | str,
        vector: list[float],
        text: str | None = None,
    ) -> KnowledgeNodeType | None:
        """
        Update embedding for any knowledge node.

        Args:
            node_id: Knowledge node ID
            vector: Vector embedding (768 or 1536 dimensions)
            text: Optional text (summary for structure, content for details)

        Returns:
            Updated node or None if not found
        """
        update_data = {
            "embedding": vector,
            "updated_at": datetime.now(UTC).isoformat(),
        }
        if text is not None:
            node = self.get_by_id(node_id)
            if node and node.node_type == NodeType.STRUCTURE:
                update_data["summary"] = text
            else:
                update_data["text"] = text

        return self.update(node_id, update_data)

    def create_with_embedding(
        self,
        node: KnowledgeNodeType,
        vector: list[float],
        text: str | None = None,
    ) -> KnowledgeNodeType:
        """Create a node with embedding in a single operation."""
        node.embedding = vector
        if text:
            if node.node_type == NodeType.STRUCTURE:
                node.summary = text
            else:
                node.text = text

        return self.create_node(node)

    def delete(self, id: uuid.UUID | str) -> bool:
        """Delete a knowledge node by ID, searching across all collections.

        Overrides BaseHandler.delete() to search across all knowledge collections.

        Args:
            id: Primary key value

        Returns:
            True if deleted, False if not found
        """
        key = str(id)

        # Find which collection contains this node
        collection_name = self._find_node_collection(id)
        if collection_name is None:
            return False

        self.db.collection(collection_name).delete(key)
        return True

    def exists(self, id: uuid.UUID | str) -> bool:
        """Check if a knowledge node exists by ID, searching across all collections.

        Overrides BaseHandler.exists() to search across all knowledge collections.

        Args:
            id: Primary key value

        Returns:
            True if exists in any collection, False otherwise
        """
        return self._find_node_collection(id) is not None

    def count(self, filters: dict[str, Any] | None = None) -> int:
        """Count knowledge nodes across all collections, optionally filtered.

        Overrides BaseHandler.count() to search across all knowledge collections.

        Args:
            filters: Optional dictionary of field: value pairs to filter by.
                     Field names are validated against ALLOWED_FILTER_FIELDS.

        Returns:
            Total number of documents matching criteria across all collections

        Raises:
            ValueError: If filter field is not in allowed list
        """
        # Allowed fields for filtering to prevent AQL injection
        ALLOWED_FILTER_FIELDS = {
            "source_id",
            "node_type",
            "level",
            "parent_id",
            "created_at",
            "updated_at",
            "is_reviewed",
            "_key",
        }

        if not filters:
            # Count all documents across all knowledge collections
            total = 0
            for collection_name in KNOWLEDGE_COLLECTIONS:
                total += self.db.collection(collection_name).count()
            return total

        # Validate filter fields against allowlist
        for field in filters:
            if field not in ALLOWED_FILTER_FIELDS:
                raise ValueError(
                    f"Invalid filter field: {field}. "
                    f"Allowed fields: {', '.join(sorted(ALLOWED_FILTER_FIELDS))}"
                )

        # Build filter conditions with validated fields
        filter_conditions = []
        bind_vars: dict[str, Any] = {}

        for i, (field, value) in enumerate(filters.items()):
            var_name = f"val{i}"
            filter_conditions.append(f"doc.{field} == @{var_name}")
            bind_vars[var_name] = str(value) if isinstance(value, uuid.UUID) else value

        filter_clause = " AND ".join(filter_conditions)

        # Count across all collections using bind variable for collection
        total = 0
        for collection_name in KNOWLEDGE_COLLECTIONS:
            query = f"""
            FOR doc IN @@collection
                FILTER {filter_clause}
                COLLECT WITH COUNT INTO count
                RETURN count
            """
            bind_vars["@collection"] = collection_name
            cursor = self.db.aql.execute(query, bind_vars=bind_vars)
            total += next(cursor, 0)

        return total

    def get_all(
        self,
        limit: int | None = None,
        offset: int | None = None,
        sort_field: str = "created_at",
        sort_desc: bool = True,
    ) -> list[KnowledgeNodeType]:
        """Get all knowledge nodes across all collections with optional pagination.

        Overrides BaseHandler.get_all() to search across all knowledge collections.

        Args:
            limit: Maximum number of records to return
            offset: Number of records to skip
            sort_field: Field to sort by (default: created_at)
            sort_desc: If True, sort descending (default: True)

        Returns:
            List of KnowledgeNode instances from all collections
        """
        sort_dir = "DESC" if sort_desc else "ASC"
        limit = limit or 100
        offset = offset or 0

        # Query all collections
        all_results: list[KnowledgeNodeType] = []

        for collection_name in KNOWLEDGE_COLLECTIONS:
            query = CRUD_GET_ALL.format(
                collection=collection_name,
                sort_field=sort_field,
                sort_dir=sort_dir,
            )
            cursor = self.db.aql.execute(query, bind_vars={"offset": 0, "limit": limit + offset})
            all_results.extend([doc_to_knowledge_node(doc) for doc in cursor])

        # Sort combined results
        if sort_field and all_results:
            all_results.sort(
                key=lambda n: getattr(n, sort_field, None) or "",
                reverse=sort_desc,
            )

        # Apply offset and limit to combined results
        return all_results[offset : offset + limit]

    def update_by_model(self, model: KnowledgeNodeType) -> KnowledgeNodeType | None:
        """Update a knowledge node using a model, searching across all collections.

        Overrides BaseHandler.update_by_model() to search across all collections.

        Args:
            model: Knowledge node model instance with updated data

        Returns:
            Updated node or None if not found
        """
        if not hasattr(model, "id"):
            raise ValueError("Model must have an 'id' field for update")

        doc = model.to_document()
        key = doc["_key"]

        # Find which collection contains this node
        collection_name = self._find_node_collection(key)
        if collection_name is None:
            return None

        result = self.db.collection(collection_name).replace(doc, return_new=True)
        return doc_to_knowledge_node(result["new"])

    def bulk_delete(self, ids: list[uuid.UUID | str]) -> int:
        """Bulk delete knowledge nodes across all collections.

        Overrides BaseHandler.bulk_delete() to search across all collections.

        Args:
            ids: List of node IDs to delete

        Returns:
            Number of documents deleted
        """
        deleted_count = 0

        for node_id in ids:
            if self.delete(node_id):
                deleted_count += 1

        return deleted_count

    def bulk_create(self, models: list[KnowledgeNodeType]) -> list[KnowledgeNodeType]:
        """Bulk create knowledge nodes in their appropriate collections.

        Overrides BaseHandler.bulk_create() to route each node to its correct collection.

        Args:
            models: List of knowledge node model instances

        Returns:
            List of created nodes
        """
        created: list[KnowledgeNodeType] = []

        # Group nodes by collection
        by_collection: dict[str, list[KnowledgeNodeType]] = {}
        for model in models:
            collection_name = get_collection_for_node(model)
            if collection_name not in by_collection:
                by_collection[collection_name] = []
            by_collection[collection_name].append(model)

        # Insert into each collection
        for collection_name, nodes in by_collection.items():
            docs = [n.to_document() for n in nodes]
            collection = self.db.collection(collection_name)
            results = collection.insert_many(docs, return_new=True)

            for result in results:
                if result.get("new"):
                    created.append(doc_to_knowledge_node(result["new"]))

        return created

    # =========================================================================
    # Graph Relationship Operations
    # =========================================================================

    def create_relationship(
        self,
        from_id: uuid.UUID | str,
        to_id: uuid.UUID | str,
        relationship_type: str,
        from_collection: str | None = None,
        to_collection: str | None = None,
    ) -> dict[str, Any]:
        """Create a relationship (edge) between two nodes."""
        # Find collections if not provided
        if from_collection is None:
            from_collection = self._find_node_collection(from_id)
            if from_collection is None:
                raise ValueError(f"Node {from_id} not found in any collection")
        if to_collection is None:
            to_collection = self._find_node_collection(to_id)
            if to_collection is None:
                raise ValueError(f"Node {to_id} not found in any collection")

        from_vertex = f"{from_collection}/{from_id!s}"
        to_vertex = f"{to_collection}/{to_id!s}"

        edge_doc = {
            "_key": str(uuid.uuid4()),
            "_from": from_vertex,
            "_to": to_vertex,
            "type": relationship_type,
            "created_at": datetime.now(UTC).isoformat(),
        }

        result = self._edge_collection.insert(edge_doc, return_new=True)
        return result["new"]

    def create_has_section_relationship(
        self,
        parent_id: uuid.UUID | str,
        child_id: uuid.UUID | str,
    ) -> dict[str, Any]:
        """Create HAS_SECTION relationship between structure nodes."""
        return self.create_relationship(parent_id, child_id, RelationshipTypes.HAS_SECTION)

    def create_has_content_relationship(
        self,
        parent_id: uuid.UUID | str,
        child_id: uuid.UUID | str,
    ) -> dict[str, Any]:
        """Create HAS_CONTENT relationship between structure and detail nodes."""
        return self.create_relationship(parent_id, child_id, RelationshipTypes.HAS_CONTENT)

    def create_next_sibling_relationship(
        self,
        sibling1_id: uuid.UUID | str,
        sibling2_id: uuid.UUID | str,
    ) -> dict[str, Any]:
        """Create NEXT_SIBLING relationship between nodes."""
        return self.create_relationship(sibling1_id, sibling2_id, RelationshipTypes.NEXT_SIBLING)

    # =========================================================================
    # Graph Traversal Operations (Using AQL Templates)
    # =========================================================================

    def get_children(
        self,
        node_id: uuid.UUID | str,
        relationship_types: list[str] | None = None,
    ) -> list[KnowledgeNodeType]:
        """Get direct children of a node via graph traversal."""
        if relationship_types is None:
            relationship_types = [
                RelationshipTypes.HAS_SECTION,
                RelationshipTypes.HAS_CONTENT,
            ]

        collection = self._find_node_collection(node_id)
        if collection is None:
            return []
        start_vertex = f"{collection}/{node_id!s}"

        cursor = self.db.aql.execute(
            GET_DIRECT_CHILDREN,
            bind_vars={
                "start_vertex": start_vertex,
                "graph_name": KNOWLEDGE_GRAPH_NAME,
                "edge_types": relationship_types,
            },
        )

        return [doc_to_knowledge_node(doc) for doc in cursor]

    def get_descendants(
        self,
        node_id: uuid.UUID | str,
        max_depth: int = 100,
        relationship_types: list[str] | None = None,
    ) -> list[KnowledgeNodeType]:
        """Get all descendants of a node (deep traversal)."""
        if relationship_types is None:
            relationship_types = [
                RelationshipTypes.HAS_SECTION,
                RelationshipTypes.HAS_CONTENT,
            ]

        collection = self._find_node_collection(node_id)
        if collection is None:
            return []
        start_vertex = f"{collection}/{node_id!s}"

        cursor = self.db.aql.execute(
            GET_DESCENDANTS,
            bind_vars={
                "start_vertex": start_vertex,
                "graph_name": KNOWLEDGE_GRAPH_NAME,
                "max_depth": max_depth,
                "edge_types": relationship_types,
            },
        )

        return [doc_to_knowledge_node(doc) for doc in cursor]

    def get_ancestors(
        self,
        node_id: uuid.UUID | str,
        max_depth: int = 100,
    ) -> list[KnowledgeNodeType]:
        """Get all ancestors of a node (upward traversal)."""
        collection = self._find_node_collection(node_id)
        if collection is None:
            return []
        start_vertex = f"{collection}/{node_id!s}"

        cursor = self.db.aql.execute(
            GET_ANCESTORS,
            bind_vars={
                "start_vertex": start_vertex,
                "graph_name": KNOWLEDGE_GRAPH_NAME,
                "max_depth": max_depth,
                "edge_types": [
                    RelationshipTypes.HAS_SECTION,
                    RelationshipTypes.HAS_CONTENT,
                ],
            },
        )

        return [doc_to_knowledge_node(doc) for doc in cursor]

    def get_parent(self, node_id: uuid.UUID | str) -> KnowledgeNodeType | None:
        """Get the parent of a node."""
        ancestors = self.get_ancestors(node_id, max_depth=1)
        return ancestors[0] if ancestors else None

    def delete_subtree(self, node_id: uuid.UUID | str) -> dict[str, int]:
        """Delete a node and all its descendants across all collections."""
        collection = self._find_node_collection(node_id)
        if collection is None:
            return {"deleted_nodes": 0, "deleted_edges": 0}

        # Get all descendants first
        descendants = self.get_descendants(node_id)
        deleted_nodes = 0
        deleted_edges = 0

        # Delete edges connected to this subtree
        all_node_ids = [str(node_id)] + [str(d.id) for d in descendants]
        for nid in all_node_ids:
            # Delete edges from/to this node
            edge_query = """
            FOR e IN @@edge_collection
                FILTER CONTAINS(e._from, @node_id) OR CONTAINS(e._to, @node_id)
                REMOVE e IN @@edge_collection
                RETURN OLD
            """
            try:
                cursor = self.db.aql.execute(
                    edge_query,
                    bind_vars={
                        "@edge_collection": EdgeCollections.KNOWLEDGE_EDGES,
                        "node_id": nid,
                    },
                )
                deleted_edges += sum(1 for _ in cursor)
            except Exception as e:
                logger.warning(f"Failed to delete edges for node {nid}: {e}")

        # Delete nodes in reverse order (leaves first)
        for desc in reversed(descendants):
            desc_collection = self._find_node_collection(desc.id)
            if desc_collection:
                try:
                    self.db.collection(desc_collection).delete(str(desc.id))
                    deleted_nodes += 1
                except Exception as e:
                    logger.warning(f"Failed to delete descendant node {desc.id}: {e}")

        # Delete the root node
        try:
            self.db.collection(collection).delete(str(node_id))
            deleted_nodes += 1
        except Exception as e:
            logger.warning(f"Failed to delete root node {node_id}: {e}")

        return {"deleted_nodes": deleted_nodes, "deleted_edges": deleted_edges}

    # =========================================================================
    # Sibling Navigation Operations
    # =========================================================================

    def get_siblings(self, node_id: uuid.UUID | str) -> list[KnowledgeNodeType]:
        """Get all siblings of a node (nodes with the same parent).

        Args:
            node_id: Node ID

        Returns:
            List of sibling nodes (excluding the input node)
        """
        collection = self._find_node_collection(node_id)
        if collection is None:
            return []
        start_vertex = f"{collection}/{node_id!s}"

        cursor = self.db.aql.execute(
            GET_SIBLINGS,
            bind_vars={
                "start_vertex": start_vertex,
                "graph_name": KNOWLEDGE_GRAPH_NAME,
            },
        )

        return [doc_to_knowledge_node(doc) for doc in cursor]

    def get_next_sibling(self, node_id: uuid.UUID | str) -> KnowledgeNodeType | None:
        """Get the next sibling via NEXT_SIBLING relationship.

        Args:
            node_id: Node ID

        Returns:
            Next sibling node or None if no next sibling
        """
        collection = self._find_node_collection(node_id)
        if collection is None:
            return None
        start_vertex = f"{collection}/{node_id!s}"

        cursor = self.db.aql.execute(
            GET_NEXT_SIBLING,
            bind_vars={
                "start_vertex": start_vertex,
                "graph_name": KNOWLEDGE_GRAPH_NAME,
            },
        )

        results = list(cursor)
        return doc_to_knowledge_node(results[0]) if results else None

    def get_prev_sibling(self, node_id: uuid.UUID | str) -> KnowledgeNodeType | None:
        """Get the previous sibling via NEXT_SIBLING relationship (reverse).

        Args:
            node_id: Node ID

        Returns:
            Previous sibling node or None if no previous sibling
        """
        collection = self._find_node_collection(node_id)
        if collection is None:
            return None
        start_vertex = f"{collection}/{node_id!s}"

        cursor = self.db.aql.execute(
            GET_PREV_SIBLING,
            bind_vars={
                "start_vertex": start_vertex,
                "graph_name": KNOWLEDGE_GRAPH_NAME,
            },
        )

        results = list(cursor)
        return doc_to_knowledge_node(results[0]) if results else None

    def get_context_window(
        self,
        node_id: uuid.UUID | str,
        before: int = 2,
        after: int = 2,
    ) -> list[KnowledgeNodeType]:
        """Get a context window of nodes around the target node.

        Useful for RAG to get surrounding paragraphs for better context.

        Args:
            node_id: Center node ID
            before: Number of preceding siblings to include
            after: Number of following siblings to include

        Returns:
            List of nodes in order [prev_siblings..., target, next_siblings...]
        """
        result = []

        # Get preceding siblings
        prev_nodes = []
        current_id = node_id
        for _ in range(before):
            prev_node = self.get_prev_sibling(current_id)
            if prev_node is None:
                break
            prev_nodes.append(prev_node)
            current_id = prev_node.id
        result.extend(reversed(prev_nodes))

        # Add target node
        target = self.get_by_id(node_id)
        if target:
            result.append(target)

        # Get following siblings
        current_id = node_id
        for _ in range(after):
            next_node = self.get_next_sibling(current_id)
            if next_node is None:
                break
            result.append(next_node)
            current_id = next_node.id

        return result

    # =========================================================================
    # Level-Based Navigation
    # =========================================================================

    def get_nodes_at_level(
        self,
        source_id: uuid.UUID | str,
        level: int,
        limit: int = 100,
    ) -> list[KnowledgeNodeType]:
        """Get all nodes at a specific level within a source.

        Useful for navigating document structure (e.g., "get all chapters").

        Args:
            source_id: Data source ID
            level: Level in the hierarchy (1=root, 2=chapters, etc.)
            limit: Maximum number of results

        Returns:
            List of nodes at the specified level
        """
        from ..collections.schema import KNOWLEDGE_VECTOR_VIEW_NAME

        cursor = self.db.aql.execute(
            GET_NODES_AT_LEVEL,
            bind_vars={
                "@view": KNOWLEDGE_VECTOR_VIEW_NAME,
                "source_id": str(source_id),
                "level": level,
                "limit": limit,
            },
        )

        return [doc_to_knowledge_node(doc) for doc in cursor]


class KnowledgeHandler:
    """Convenience class grouping all knowledge-related handlers."""

    def __init__(self, db: StandardDatabase):
        """Initialize all knowledge handlers."""
        self.db = db
        self.nodes = KnowledgeNodeHandler(db)
