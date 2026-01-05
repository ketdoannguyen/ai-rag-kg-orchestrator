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
from datetime import datetime, timezone
from typing import Any, Optional, Type, Union

from arango.database import StandardDatabase

from ..aql import (
    CRUD_GET_BY_FILTER,
    GET_DIRECT_CHILDREN,
    GET_DESCENDANTS,
    GET_ANCESTORS,
    DELETE_SUBTREE,
)
from ..collections.schema import (
    Collections,
    EdgeCollections,
    RelationshipTypes,
    KNOWLEDGE_GRAPH_NAME,
)
from ..models.knowledge import KnowledgeNode
from ..models.detail_nodes import ParagraphNode, ImageNode, TableNode
from ..models.structure_nodes import StructureNode, RootNode
from ..models.enums import NodeType
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
        super().__init__(db, KnowledgeNode, Collections.KNOWLEDGE_NODES)
        self._edge_collection = db.collection(EdgeCollections.KNOWLEDGE_EDGES)
        self._graph = db.graph(KNOWLEDGE_GRAPH_NAME)

    # =========================================================================
    # Document CRUD Operations
    # =========================================================================

    def create_node(self, node: KnowledgeNodeType) -> KnowledgeNodeType:
        """Create a new knowledge node."""
        doc = node.to_document()
        result = self._collection.insert(doc, return_new=True)
        return doc_to_knowledge_node(result["new"])

    def get_by_id(
        self,
        id: uuid.UUID | str,
        node_class: Optional[Type[KnowledgeNodeType]] = None,
    ) -> Optional[KnowledgeNodeType]:
        """Get a knowledge node by ID."""
        key = str(id)
        doc = self._collection.get(key)
        
        if doc is None:
            return None
        
        return doc_to_knowledge_node(doc, node_class)

    def get_by_filter(
        self,
        filters: dict[str, Any],
        limit: Optional[int] = None,
        offset: Optional[int] = None,
        sort_field: str = "created_at",
        sort_desc: bool = True,
    ) -> list[KnowledgeNodeType]:
        """Get nodes by filter criteria with proper type conversion.
        
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
        
        # Use centralized AQL template
        query = CRUD_GET_BY_FILTER.format(
            collection=self.collection_name,
            filter_clause=filter_clause,
            sort_field=sort_field,
            sort_dir=sort_dir,
        )
        
        cursor = self.db.aql.execute(query, bind_vars=bind_vars)
        return [doc_to_knowledge_node(doc) for doc in cursor]

    def get_by_source(
        self,
        source_id: uuid.UUID | str,
        limit: Optional[int] = None,
        offset: Optional[int] = None,
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
        limit: Optional[int] = None,
        offset: Optional[int] = None,
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

    def update_embedding(
        self,
        node_id: uuid.UUID | str,
        vector: list[float],
        text: Optional[str] = None,
    ) -> Optional[KnowledgeNodeType]:
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
            "updated_at": datetime.now(timezone.utc).isoformat(),
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
        text: Optional[str] = None,
    ) -> KnowledgeNodeType:
        """Create a node with embedding in a single operation."""
        node.embedding = vector
        if text:
            if node.node_type == NodeType.STRUCTURE:
                node.summary = text
            else:
                node.text = text
        
        return self.create_node(node)

    # =========================================================================
    # Graph Relationship Operations
    # =========================================================================

    def create_relationship(
        self,
        from_id: uuid.UUID | str,
        to_id: uuid.UUID | str,
        relationship_type: str,
    ) -> dict[str, Any]:
        """Create a relationship (edge) between two nodes."""
        from_vertex = f"{Collections.KNOWLEDGE_NODES}/{str(from_id)}"
        to_vertex = f"{Collections.KNOWLEDGE_NODES}/{str(to_id)}"
        
        edge_doc = {
            "_key": str(uuid.uuid4()),
            "_from": from_vertex,
            "_to": to_vertex,
            "type": relationship_type,
            "created_at": datetime.now(timezone.utc).isoformat(),
        }
        
        result = self._edge_collection.insert(edge_doc, return_new=True)
        return result["new"]

    def create_has_section_relationship(
        self,
        parent_id: uuid.UUID | str,
        child_id: uuid.UUID | str,
    ) -> dict[str, Any]:
        """Create HAS_SECTION relationship between structure nodes."""
        return self.create_relationship(
            parent_id, child_id, RelationshipTypes.HAS_SECTION
        )

    def create_has_content_relationship(
        self,
        parent_id: uuid.UUID | str,
        child_id: uuid.UUID | str,
    ) -> dict[str, Any]:
        """Create HAS_CONTENT relationship between structure and detail nodes."""
        return self.create_relationship(
            parent_id, child_id, RelationshipTypes.HAS_CONTENT
        )

    def create_next_sibling_relationship(
        self,
        sibling1_id: uuid.UUID | str,
        sibling2_id: uuid.UUID | str,
    ) -> dict[str, Any]:
        """Create NEXT_SIBLING relationship between nodes."""
        return self.create_relationship(
            sibling1_id, sibling2_id, RelationshipTypes.NEXT_SIBLING
        )

    # =========================================================================
    # Graph Traversal Operations (Using AQL Templates)
    # =========================================================================

    def get_children(
        self,
        node_id: uuid.UUID | str,
        relationship_types: Optional[list[str]] = None,
    ) -> list[KnowledgeNodeType]:
        """Get direct children of a node via graph traversal."""
        if relationship_types is None:
            relationship_types = [
                RelationshipTypes.HAS_SECTION,
                RelationshipTypes.HAS_CONTENT,
            ]
        
        start_vertex = f"{Collections.KNOWLEDGE_NODES}/{str(node_id)}"
        
        cursor = self.db.aql.execute(
            GET_DIRECT_CHILDREN,
            bind_vars={
                "start_vertex": start_vertex,
                "graph_name": KNOWLEDGE_GRAPH_NAME,
                "edge_types": relationship_types,
            }
        )
        
        return [doc_to_knowledge_node(doc) for doc in cursor]

    def get_descendants(
        self,
        node_id: uuid.UUID | str,
        max_depth: int = 100,
        relationship_types: Optional[list[str]] = None,
    ) -> list[KnowledgeNodeType]:
        """Get all descendants of a node (deep traversal)."""
        if relationship_types is None:
            relationship_types = [
                RelationshipTypes.HAS_SECTION,
                RelationshipTypes.HAS_CONTENT,
            ]
        
        start_vertex = f"{Collections.KNOWLEDGE_NODES}/{str(node_id)}"
        
        cursor = self.db.aql.execute(
            GET_DESCENDANTS,
            bind_vars={
                "start_vertex": start_vertex,
                "graph_name": KNOWLEDGE_GRAPH_NAME,
                "max_depth": max_depth,
                "edge_types": relationship_types,
            }
        )
        
        return [doc_to_knowledge_node(doc) for doc in cursor]

    def get_ancestors(
        self,
        node_id: uuid.UUID | str,
        max_depth: int = 100,
    ) -> list[KnowledgeNodeType]:
        """Get all ancestors of a node (upward traversal)."""
        start_vertex = f"{Collections.KNOWLEDGE_NODES}/{str(node_id)}"
        
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
            }
        )
        
        return [doc_to_knowledge_node(doc) for doc in cursor]

    def get_parent(self, node_id: uuid.UUID | str) -> Optional[KnowledgeNodeType]:
        """Get the parent of a node."""
        ancestors = self.get_ancestors(node_id, max_depth=1)
        return ancestors[0] if ancestors else None

    def delete_subtree(self, node_id: uuid.UUID | str) -> dict[str, int]:
        """Delete a node and all its descendants."""
        start_vertex = f"{Collections.KNOWLEDGE_NODES}/{str(node_id)}"
        
        cursor = self.db.aql.execute(
            DELETE_SUBTREE,
            bind_vars={
                "start_vertex": start_vertex,
                "graph_name": KNOWLEDGE_GRAPH_NAME,
                "@edge_collection": EdgeCollections.KNOWLEDGE_EDGES,
                "@node_collection": Collections.KNOWLEDGE_NODES,
                "collection_prefix": f"{Collections.KNOWLEDGE_NODES}/",
            }
        )
        
        return next(cursor, {"deleted_nodes": 0, "deleted_edges": 0})


class KnowledgeHandler:
    """Convenience class grouping all knowledge-related handlers."""

    def __init__(self, db: StandardDatabase):
        """Initialize all knowledge handlers."""
        self.db = db
        self.nodes = KnowledgeNodeHandler(db)
