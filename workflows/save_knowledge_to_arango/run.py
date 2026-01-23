"""Implementation of save_knowledge_to_arango workflow.

This workflow saves knowledge tree to ArangoDB, handling both:
- Document storage (nodes as documents)
- Graph relationships (HAS_SECTION, HAS_CONTENT, NEXT_SIBLING edges)
"""

from __future__ import annotations

import uuid
from typing import Any

from arango_client import ArangoClient, KnowledgeNodeHandler, set_parent_ids
from arango_client.handlers.system_admin_handler import SystemAdminHandler
from arango_client.models.knowledge import KnowledgeNode
from loguru import logger
from tenacity import retry, retry_if_exception_type, stop_after_attempt, wait_exponential

from workflows.dependencies import init_arango_client

from .models import SaveKnowledgeInput, SaveKnowledgeOutput


async def start_save_knowledge_to_arango(
    payload: SaveKnowledgeInput | dict[str, Any],
    *,
    arango_client: ArangoClient | None = None,
) -> dict[str, Any]:
    """
    Entry point for the save_knowledge_to_arango workflow.

    Args:
        payload: Workflow input (Pydantic model or plain dict)
        arango_client: Optional ArangoDB client (defaults to settings)

    Returns:
        Dictionary representation of SaveKnowledgeOutput
    """
    input_payload = (
        payload if isinstance(payload, SaveKnowledgeInput) else SaveKnowledgeInput(**payload)
    )
    client = arango_client or init_arango_client()

    workflow = SaveKnowledgeToArangoWorkflow(arango_client=client)
    result = await workflow.run(input_payload)

    if hasattr(result, "model_dump"):
        return result.model_dump(mode="json")
    return result.dict()


class SaveKnowledgeToArangoWorkflow:
    """Implements the save_knowledge_to_arango workflow with recursive tree saving."""

    def __init__(
        self,
        *,
        arango_client: ArangoClient,
    ) -> None:
        """
        Initialize workflow.

        Args:
            arango_client: ArangoDB client instance
        """
        self._arango_client = arango_client
        self._db = arango_client.db
        self._log = logger.bind(workflow="save_knowledge_to_arango")

    async def run(self, payload: SaveKnowledgeInput) -> SaveKnowledgeOutput:
        """Execute the workflow end-to-end."""
        self._log.info("Starting save knowledge to ArangoDB workflow")

        # Validate input
        self._validate_input(payload)

        # Verify data_source exists
        admin_handler = SystemAdminHandler(self._db)
        data_source = admin_handler.sources.get_by_id(payload.data_source_id)
        if not data_source:
            raise ValueError(f"DataSource with id {payload.data_source_id} not found")

        # Set source_id for all nodes in tree
        self._set_source_id(payload.enriched_tree, payload.data_source_id)

        # Set parent_ids for all nodes
        set_parent_ids(payload.enriched_tree, parent_id=None)

        # Initialize handler
        handler = KnowledgeNodeHandler(self._db)

        # Initialize statistics
        saved_nodes: list[KnowledgeNode] = []
        failed_nodes: list[dict[str, Any]] = []
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
        structure_count = sum(1 for n in saved_nodes if self._get_node_type_str(n) == "structure")
        detail_count = sum(
            1 for n in saved_nodes if self._get_node_type_str(n) in ("paragraph", "image", "table")
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

        return SaveKnowledgeOutput(
            root_node_id=payload.enriched_tree.id,
            total_nodes_saved=len(saved_nodes),
            structure_nodes_count=structure_count,
            detail_nodes_count=detail_count,
            relationships_created=relationships_stats,
            saved_nodes=saved_nodes,
            failed_nodes=failed_nodes,
            partial_success=partial_success,
        )

    def _validate_input(self, payload: SaveKnowledgeInput) -> None:
        """Validate workflow input."""
        node_type = self._get_node_type_str(payload.enriched_tree)
        if node_type != "structure":
            raise ValueError(f"Root node must have node_type=STRUCTURE, got {node_type}")

    def _set_source_id(self, node: KnowledgeNode, source_id: uuid.UUID) -> None:
        """Recursively set source_id for all nodes in the tree."""
        node.source_id = source_id
        if hasattr(node, "children") and node.children:
            for child in node.children:
                self._set_source_id(child, source_id)

    def _get_node_type_str(self, node: KnowledgeNode) -> str:
        """Get node_type as string."""
        node_type = node.node_type
        if hasattr(node_type, "value"):
            return node_type.value
        return str(node_type).lower()

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
    ) -> KnowledgeNode:
        """Save a single node with retry logic."""
        # Check if node already exists
        existing = handler.get_by_id(node.id)
        if existing is not None:
            # Node exists, update it
            return handler.update_by_model(node)
        # Node doesn't exist, create it
        return handler.create_node(node)

    async def _create_relationships_for_node(
        self,
        node: KnowledgeNode,
        parent_id: uuid.UUID | None,
        handler: KnowledgeNodeHandler,
        relationships_stats: dict[str, int],
    ) -> None:
        """Create all relationships for a node."""
        if parent_id is not None:
            try:
                node_type = self._get_node_type_str(node)
                if node_type == "structure":
                    handler.create_has_section_relationship(parent_id, node.id)
                    relationships_stats["HAS_SECTION"] += 1
                else:
                    handler.create_has_content_relationship(parent_id, node.id)
                    relationships_stats["HAS_CONTENT"] += 1
            except Exception as exc:
                rel_type = (
                    "HAS_SECTION"
                    if self._get_node_type_str(node) == "structure"
                    else "HAS_CONTENT"
                )
                self._log.warning(
                    f"Failed to create {rel_type} relationship",
                    parent_id=str(parent_id),
                    child_id=str(node.id),
                    error=str(exc),
                )

    async def _create_sibling_relationships(
        self,
        children: list[KnowledgeNode],
        handler: KnowledgeNodeHandler,
        relationships_stats: dict[str, int],
    ) -> None:
        """Create NEXT_SIBLING relationships for consecutive detail nodes only."""
        for i in range(len(children) - 1):
            sibling1 = children[i]
            sibling2 = children[i + 1]

            type1 = self._get_node_type_str(sibling1)
            type2 = self._get_node_type_str(sibling2)

            is_detail1 = type1 in ("paragraph", "image", "table")
            is_detail2 = type2 in ("paragraph", "image", "table")

            if is_detail1 and is_detail2 and type1 == type2:
                try:
                    handler.create_next_sibling_relationship(sibling1.id, sibling2.id)
                    relationships_stats["NEXT_SIBLING"] += 1
                except Exception as exc:
                    self._log.warning(
                        "Failed to create NEXT_SIBLING relationship",
                        sibling1_id=str(sibling1.id),
                        sibling2_id=str(sibling2.id),
                        error=str(exc),
                    )

    async def _save_node_recursive(
        self,
        node: KnowledgeNode,
        parent_id: uuid.UUID | None,
        handler: KnowledgeNodeHandler,
        saved_nodes: list[KnowledgeNode],
        failed_nodes: list[dict[str, Any]],
        relationships_stats: dict[str, int],
    ) -> None:
        """
        Recursively save a node and all its children.

        Uses depth-first traversal: save parent first, then children.
        """
        # Save current node
        try:
            saved_node = await self._save_node_with_retry(node, handler)
            saved_nodes.append(saved_node)
            self._log.debug(
                "Node saved", node_id=str(node.id), node_type=self._get_node_type_str(node)
            )
        except Exception as exc:
            error_msg = str(exc)
            error_info = {
                "node_id": str(node.id),
                "node_type": self._get_node_type_str(node),
                "error": error_msg,
                "retry_count": 3,
            }
            failed_nodes.append(error_info)
            self._log.error(
                "Failed to save node after retries",
                node_id=str(node.id),
                node_type=self._get_node_type_str(node),
                error=error_msg,
            )
            return

        # Create relationships for current node
        await self._create_relationships_for_node(
            node=node,
            parent_id=parent_id,
            handler=handler,
            relationships_stats=relationships_stats,
        )

        # Save children recursively
        if hasattr(node, "children") and node.children:
            for child in node.children:
                await self._save_node_recursive(
                    node=child,
                    parent_id=node.id,
                    handler=handler,
                    saved_nodes=saved_nodes,
                    failed_nodes=failed_nodes,
                    relationships_stats=relationships_stats,
                )

            # Create sibling relationships after all children are saved
            await self._create_sibling_relationships(
                children=node.children,
                handler=handler,
                relationships_stats=relationships_stats,
            )
