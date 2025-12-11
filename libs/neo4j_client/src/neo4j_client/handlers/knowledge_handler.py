"""Handler for KnowledgeNode operations in Neo4j."""

from __future__ import annotations

import json
import uuid
from typing import Any

from neo4j import AsyncDriver

from postgres_client.models.knowledge import KnowledgeNode
from postgres_client.models.enums import NodeType

from ..queries.knowledge_queries import (
    MERGE_NODE_QUERY,
    CREATE_HAS_SECTION_QUERY,
    CREATE_HAS_CONTENT_QUERY,
    CREATE_NEXT_SIBLING_QUERY,
    GET_NODE_BY_ID_QUERY,
    NODE_LABELS,
)


class KnowledgeNodeHandler:
    """Handler for KnowledgeNode CRUD operations in Neo4j."""

    def __init__(self, driver: AsyncDriver) -> None:
        """
        Initialize KnowledgeNode handler.
        
        Args:
            driver: Neo4j async driver instance
        """
        self.driver = driver

    def _get_type_label(self, node_type: NodeType | str, level_down: int = 0) -> str:
        """Get Neo4j label based on node_type and level_down.
        
        Args:
            node_type: The node type enum or string
            level_down: The level_down value (1 = root node)
            
        Returns:
            Neo4j label string
        """
        # Root node special case: level_down=1 means it's the root
        if level_down == 1:
            return NODE_LABELS["ROOT"]
        
        node_type_str = node_type.value if isinstance(node_type, NodeType) else str(node_type)
        label_map = {
            "structure": NODE_LABELS["STRUCTURE"],
            "paragraph": NODE_LABELS["PARAGRAPH"],
            "image": NODE_LABELS["IMAGE"],
            "table": NODE_LABELS["TABLE"],
        }
        return label_map.get(node_type_str.lower(), NODE_LABELS["BASE"])

    def _serialize_to_json(self, data: Any) -> str:
        """Serialize data (attributes or metadata) to JSON string."""
        if isinstance(data, str):
            return data
        if hasattr(data, "model_dump"):
            return json.dumps(data.model_dump(), ensure_ascii=False)
        if isinstance(data, dict):
            return json.dumps(data, ensure_ascii=False)
        return json.dumps({}, ensure_ascii=False)

    async def create_or_update_node(self, node: KnowledgeNode) -> None:
        """
        Create or update a node in Neo4j using MERGE.
        
        Args:
            node: KnowledgeNode Pydantic model instance
        """
        # Serialize attributes and metadata
        attributes_json = self._serialize_to_json(node.attributes)
        metadata_json = self._serialize_to_json(node.metadata)
        
        # Get node type label (uses level_down to determine if it's a root node)
        type_label = self._get_type_label(node.node_type, level_down=node.level_down)
        
        # Prepare parameters (no name property - Neo4j will display based on existing properties)
        params = {
            "id": str(node.id),
            "node_type": node.node_type.value if isinstance(node.node_type, NodeType) else str(node.node_type),
            "attributes": attributes_json,
            "metadata": metadata_json,
            "level_down": node.level_down,
            "level_up": node.level_up,
        }
        
        # Build query with dynamic label
        # Neo4j requires labels to be specified in MERGE, so we construct the full label list
        query = f"""
        MERGE (n:KnowledgeNode:{type_label} {{id: $id}})
        SET n.node_type = $node_type,
            n.attributes = $attributes,
            n.metadata = $metadata,
            n.level_down = $level_down,
            n.level_up = $level_up
        RETURN n
        """
        
        async with self.driver.session() as session:
            await session.run(query, params)


    async def create_has_section_relationship(
        self,
        parent_id: uuid.UUID,
        child_id: uuid.UUID,
    ) -> None:
        """
        Create HAS_SECTION relationship between structure parent and structure child.
        
        Args:
            parent_id: Parent structure node ID
            child_id: Child structure node ID
        """
        params = {
            "parent_id": str(parent_id),
            "child_id": str(child_id),
        }
        
        async with self.driver.session() as session:
            result = await session.run(CREATE_HAS_SECTION_QUERY, params)
            await result.consume()

    async def create_has_content_relationship(
        self,
        parent_id: uuid.UUID,
        child_id: uuid.UUID,
    ) -> None:
        """
        Create HAS_CONTENT relationship between structure parent and detail child.
        
        Args:
            parent_id: Parent structure node ID
            child_id: Child detail node ID (paragraph/image/table)
        """
        params = {
            "parent_id": str(parent_id),
            "child_id": str(child_id),
        }
        
        async with self.driver.session() as session:
            result = await session.run(CREATE_HAS_CONTENT_QUERY, params)
            await result.consume()

    async def create_next_sibling_relationship(
        self,
        sibling1_id: uuid.UUID,
        sibling2_id: uuid.UUID,
    ) -> None:
        """
        Create NEXT_SIBLING relationship between two sibling nodes.
        
        Args:
            sibling1_id: First sibling node ID
            sibling2_id: Second sibling node ID (next sibling)
        """
        params = {
            "sibling1_id": str(sibling1_id),
            "sibling2_id": str(sibling2_id),
        }
        
        async with self.driver.session() as session:
            result = await session.run(CREATE_NEXT_SIBLING_QUERY, params)
            await result.consume()

    async def get_node_by_id(self, node_id: uuid.UUID) -> dict[str, Any] | None:
        """
        Get a node by ID.
        
        Args:
            node_id: Node ID
            
        Returns:
            Node properties as dict, or None if not found
        """
        params = {"node_id": str(node_id)}
        
        async with self.driver.session() as session:
            result = await session.run(GET_NODE_BY_ID_QUERY, params)
            record = await result.single()
            
            if record is None:
                return None
            
            node = record["n"]
            return dict(node.items())

