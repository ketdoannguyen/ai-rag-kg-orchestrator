"""Implementation of save_knowledge_to_neo4j workflow."""

from __future__ import annotations

import uuid
from typing import Any, Dict, List

from loguru import logger
from neo4j_client import Neo4jClient
from neo4j_client.handlers import KnowledgeNodeHandler
from postgres_client.models.knowledge import KnowledgeNode
from postgres_client.models.structure_nodes import StructureNode
from postgres_client.models.enums import NodeType
from tenacity import retry, retry_if_exception_type, stop_after_attempt, wait_exponential

from workflows.dependencies import init_neo4j_client

from .models import SaveKnowledgeToNeo4jInput, SaveKnowledgeToNeo4jOutput


async def start_save_knowledge_to_neo4j(
    payload: SaveKnowledgeToNeo4jInput | Dict[str, Any],
    *,
    neo4j_client: Neo4jClient | None = None,
) -> Dict[str, Any]:
    """
    Entry point for the save_knowledge_to_neo4j workflow.

    Args:
        payload: Workflow input (Pydantic model or plain dict)
        neo4j_client: Optional Neo4j client (defaults to settings)

    Returns:
        Dictionary representation of SaveKnowledgeToNeo4jOutput
    """
    input_payload = (
        payload if isinstance(payload, SaveKnowledgeToNeo4jInput) else SaveKnowledgeToNeo4jInput(**payload)
    )
    client = neo4j_client or init_neo4j_client()

    workflow = SaveKnowledgeToNeo4jWorkflow(neo4j_client=client)
    result = await workflow.run(input_payload)
    
    if hasattr(result, "model_dump"):
        return result.model_dump(mode="json")
    return result.dict()


class SaveKnowledgeToNeo4jWorkflow:
    """Implements the save_knowledge_to_neo4j workflow with recursive tree saving."""

    def __init__(
        self,
        *,
        neo4j_client: Neo4jClient,
    ) -> None:
        """
        Initialize workflow.
        
        Args:
            neo4j_client: Neo4j client instance
        """
        self._neo4j_client = neo4j_client
        self._driver = neo4j_client.async_driver
        self._log = logger.bind(workflow="save_knowledge_to_neo4j")

    async def run(self, payload: SaveKnowledgeToNeo4jInput) -> SaveKnowledgeToNeo4jOutput:
        """Execute the workflow end-to-end."""
        self._log.info("Starting save knowledge to Neo4j workflow")

        # Validate input
        self._validate_input(payload)

        # Initialize handler
        handler = KnowledgeNodeHandler(self._driver)

        # Initialize statistics
        saved_nodes: List[KnowledgeNode] = []
        failed_nodes: List[Dict[str, Any]] = []
        relationships_stats = {
            "HAS_SECTION": 0,
            "HAS_CONTENT": 0,
            "NEXT_SIBLING": 0,
        }

        # Recursively save tree
        await self._save_node_recursive(
            node=payload.enriched_tree,
            parent_id=None,
            handler=handler,
            saved_nodes=saved_nodes,
            failed_nodes=failed_nodes,
            relationships_stats=relationships_stats,
        )

        # Calculate statistics
        structure_count = sum(1 for n in saved_nodes if n.node_type == NodeType.STRUCTURE)
        detail_count = sum(
            1
            for n in saved_nodes
            if n.node_type in (NodeType.PARAGRAPH, NodeType.IMAGE, NodeType.TABLE)
        )

        partial_success = len(failed_nodes) > 0

        if partial_success:
            self._log.warning(
                "Partial success: some nodes failed to save",
                total_nodes=len(saved_nodes) + len(failed_nodes),
                saved=len(saved_nodes),
                failed=len(failed_nodes),
            )
        else:
            self._log.info(
                "All nodes saved successfully",
                total_nodes=len(saved_nodes),
                relationships=relationships_stats,
            )

        return SaveKnowledgeToNeo4jOutput(
            root_node_id=payload.enriched_tree.id,
            total_nodes_saved=len(saved_nodes),
            structure_nodes_count=structure_count,
            detail_nodes_count=detail_count,
            relationships_created=relationships_stats,
            saved_nodes=saved_nodes,
            failed_nodes=failed_nodes,
            partial_success=partial_success,
        )

    def _validate_input(self, payload: SaveKnowledgeToNeo4jInput) -> None:
        """Validate workflow input."""
        # Check node_type instead of isinstance to support both parsed KnowledgeNode and StructureNode
        node_type = payload.enriched_tree.node_type
        if hasattr(node_type, 'value'):
            node_type = node_type.value
        if str(node_type).lower() != 'structure':
            raise ValueError(f"Root node must have node_type=STRUCTURE, got {node_type}")

    def _get_node_type_str(self, node: KnowledgeNode) -> str:
        """Get node_type as string."""
        return node.node_type.value if isinstance(node.node_type, NodeType) else str(node.node_type)

    @retry(
        retry=retry_if_exception_type(Exception),
        stop=stop_after_attempt(3),
        wait=wait_exponential(multiplier=1, min=1, max=8),
        reraise=True,
    )
    async def _save_node_with_retry(
        self,
        node: KnowledgeNode,
        handler: KnowledgeNodeHandler,
    ) -> None:
        """Save a single node with retry logic."""
        await handler.create_or_update_node(node)

    async def _create_relationships_for_node(
        self,
        node: KnowledgeNode,
        parent_id: uuid.UUID | None,
        handler: KnowledgeNodeHandler,
        relationships_stats: Dict[str, int],
    ) -> None:
        """Create all relationships for a node."""
        # Create HAS_SECTION or HAS_CONTENT relationship if parent exists
        if parent_id is not None:
            try:
                # Determine relationship type based on node types
                if node.node_type == NodeType.STRUCTURE:
                    # Structure -> Structure: HAS_SECTION
                    await handler.create_has_section_relationship(parent_id, node.id)
                    relationships_stats["HAS_SECTION"] += 1
                else:
                    # Structure -> Detail (paragraph/image/table): HAS_CONTENT
                    await handler.create_has_content_relationship(parent_id, node.id)
                    relationships_stats["HAS_CONTENT"] += 1
            except Exception as exc:
                rel_type = "HAS_SECTION" if node.node_type == NodeType.STRUCTURE else "HAS_CONTENT"
                self._log.warning(
                    f"Failed to create {rel_type} relationship",
                    parent_id=str(parent_id),
                    child_id=str(node.id),
                    error=str(exc),
                )

    async def _create_sibling_relationships(
        self,
        children: List[KnowledgeNode],
        handler: KnowledgeNodeHandler,
        relationships_stats: Dict[str, int],
    ) -> None:
        """Create NEXT_SIBLING relationships for consecutive detail nodes (paragraph/image/table) only."""
        for i in range(len(children) - 1):
            sibling1 = children[i]
            sibling2 = children[i + 1]
            
            # Only create NEXT_SIBLING for detail nodes (paragraph/image/table), NOT for structure nodes
            # Both must be detail nodes and have the same type
            is_detail1 = sibling1.node_type in (NodeType.PARAGRAPH, NodeType.IMAGE, NodeType.TABLE)
            is_detail2 = sibling2.node_type in (NodeType.PARAGRAPH, NodeType.IMAGE, NodeType.TABLE)
            
            if is_detail1 and is_detail2 and sibling1.node_type == sibling2.node_type:
                try:
                    # Ensure both nodes are saved before creating relationship
                    # (they should be saved already, but just to be safe)
                    await handler.create_next_sibling_relationship(sibling1.id, sibling2.id)
                    relationships_stats["NEXT_SIBLING"] += 1
                except Exception as exc:
                    self._log.warning(
                        "Failed to create NEXT_SIBLING relationship",
                        sibling1_id=str(sibling1.id),
                        sibling2_id=str(sibling2.id),
                        sibling1_type=self._get_node_type_str(sibling1),
                        sibling2_type=self._get_node_type_str(sibling2),
                        error=str(exc),
                    )

    async def _save_node_recursive(
        self,
        node: KnowledgeNode,
        parent_id: uuid.UUID | None,
        handler: KnowledgeNodeHandler,
        saved_nodes: List[KnowledgeNode],
        failed_nodes: List[Dict[str, Any]],
        relationships_stats: Dict[str, int],
    ) -> None:
        """
        Recursively save a node and all its children.

        Uses depth-first traversal: save parent first, then children.
        """
        # Save current node
        try:
            await self._save_node_with_retry(node, handler)
            saved_nodes.append(node)
        except Exception as exc:
            import traceback

            error_msg = str(exc)
            error_traceback = traceback.format_exc()
            error_info = {
                "node_id": str(node.id),
                "node_type": self._get_node_type_str(node),
                "error": error_msg,
                "retry_count": 3,  # Max retries attempted
            }
            failed_nodes.append(error_info)
            self._log.error(
                "Failed to save node after retries",
                node_id=str(node.id),
                node_type=self._get_node_type_str(node),
                error=error_msg,
                traceback=error_traceback,
            )
            # Continue with children even if parent failed (partial success strategy)
            return

        # Create relationships for current node
        await self._create_relationships_for_node(
            node=node,
            parent_id=parent_id,
            handler=handler,
            relationships_stats=relationships_stats,
        )

        # Save children recursively first (so they exist when creating sibling relationships)
        for child in node.children:
            await self._save_node_recursive(
                node=child,
                parent_id=node.id,
                handler=handler,
                saved_nodes=saved_nodes,
                failed_nodes=failed_nodes,
                relationships_stats=relationships_stats,
            )

        # Create sibling relationships for children AFTER all children are saved
        # (only between same type)
        if node.children:
            await self._create_sibling_relationships(
                children=node.children,
                handler=handler,
                relationships_stats=relationships_stats,
            )

