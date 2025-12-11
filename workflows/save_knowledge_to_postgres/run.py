"""Implementation of save_knowledge_to_postgres workflow."""

from __future__ import annotations

import uuid
from typing import Any, Dict, List

from loguru import logger
from postgres_client import PostgresClient
from postgres_client.handlers.knowledge_handler import KnowledgeNodeHandler
from postgres_client.handlers.system_admin_handler import SystemAdminHandler
from postgres_client.models.knowledge import KnowledgeNode
from postgres_client.models.structure_nodes import StructureNode
from postgres_client.models.enums import NodeType
from postgres_client.utils import set_parent_ids
from tenacity import retry, retry_if_exception_type, stop_after_attempt, wait_exponential

from workflows.dependencies import init_postgres_client

from .models import SaveKnowledgeInput, SaveKnowledgeOutput


async def start_save_knowledge_to_postgres(
    payload: SaveKnowledgeInput | Dict[str, Any],
    *,
    pg_client: PostgresClient | None = None,
) -> Dict[str, Any]:
    """
    Entry point for the save_knowledge_to_postgres workflow.

    Args:
        payload: Workflow input (Pydantic model or plain dict)
        pg_client: Optional Postgres client (defaults to settings.DATABASE_URL)

    Returns:
        Dictionary representation of SaveKnowledgeOutput
    """
    input_payload = payload if isinstance(payload, SaveKnowledgeInput) else SaveKnowledgeInput(**payload)
    postgres_client = pg_client or init_postgres_client()

    workflow = SaveKnowledgeWorkflow(pg_client=postgres_client)
    result = await workflow.run(input_payload)
    if hasattr(result, "model_dump"):
        return result.model_dump(mode="json")
    return result.dict()


class SaveKnowledgeWorkflow:
    """Implements the save_knowledge_to_postgres workflow with recursive tree saving."""

    def __init__(
        self,
        *,
        pg_client: PostgresClient,
    ) -> None:
        self._sessionmaker = pg_client.sessionmaker
        self._log = logger.bind(workflow="save_knowledge_to_postgres")

    async def run(self, payload: SaveKnowledgeInput) -> SaveKnowledgeOutput:
        """Execute the workflow end-to-end."""
        self._log.info("Starting save knowledge workflow", data_source_id=str(payload.data_source_id))

        # Validate input
        self._validate_input(payload)

        async with self._sessionmaker() as session:
            # Verify data_source exists
            handler = SystemAdminHandler(session)
            data_source = await handler.sources.get_by_id(payload.data_source_id)
            if not data_source:
                raise ValueError(f"DataSource with id {payload.data_source_id} not found")

            # Set source_id for all nodes in tree
            self._set_source_id(payload.enriched_tree, payload.data_source_id)

            # Set parent_ids for all nodes
            set_parent_ids(payload.enriched_tree, parent_id=None)

            # Recursively save tree
            saved_nodes: List[KnowledgeNode] = []
            failed_nodes: List[Dict[str, Any]] = []

            await self._save_node_recursive(
                node=payload.enriched_tree,
                handler=KnowledgeNodeHandler(session),
                saved_nodes=saved_nodes,
                failed_nodes=failed_nodes,
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
                )

            return SaveKnowledgeOutput(
                root_node_id=payload.enriched_tree.id,
                total_nodes_saved=len(saved_nodes),
                structure_nodes_count=structure_count,
                detail_nodes_count=detail_count,
                saved_nodes=saved_nodes,
                failed_nodes=failed_nodes,
                partial_success=partial_success,
            )

    def _validate_input(self, payload: SaveKnowledgeInput) -> None:
        """Validate workflow input."""
        # Check node_type instead of isinstance to support both parsed KnowledgeNode and StructureNode
        node_type = payload.enriched_tree.node_type
        if hasattr(node_type, 'value'):
            node_type = node_type.value
        if str(node_type).lower() != 'structure':
            raise ValueError(f"Root node must have node_type=STRUCTURE, got {node_type}")

    def _set_source_id(self, node: KnowledgeNode, source_id: uuid.UUID) -> None:
        """Recursively set source_id for all nodes in the tree."""
        node.source_id = source_id
        for child in node.children:
            self._set_source_id(child, source_id)

    def _get_node_type_str(self, node: KnowledgeNode) -> str:
        """Get node_type as string, handling both enum and string types."""
        node_type = node.node_type
        if hasattr(node_type, "value"):
            return node_type.value
        return str(node_type)

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
        """Save a single node with retry logic. Update if exists."""
        # Check if node already exists
        existing = await handler.get_by_id(node.id)
        if existing is not None:
            # Node exists, update it
            return await handler.update_by_model(node)
        # Node doesn't exist, create it
        try:
            return await handler.create(node)
        except Exception as exc:
            # If create fails due to duplicate key (race condition), try update
            error_str = str(exc)
            if "duplicate key" in error_str.lower() or "unique constraint" in error_str.lower():
                # Rollback and try update
                await handler.session.rollback()
                updated = await handler.update_by_model(node)
                if updated is None:
                    # If update returns None, node was deleted, try create again
                    return await handler.create(node)
                return updated
            raise

    async def _save_node_recursive(
        self,
        node: KnowledgeNode,
        handler: KnowledgeNodeHandler,
        saved_nodes: List[KnowledgeNode],
        failed_nodes: List[Dict[str, Any]],
    ) -> None:
        """
        Recursively save a node and all its children.

        Uses depth-first traversal: save parent first, then children.
        """
        # Save current node
        try:
            saved_node = await self._save_node_with_retry(node, handler)
            saved_nodes.append(saved_node)
            self._log.debug("Node saved", node_id=str(node.id), node_type=self._get_node_type_str(node))
        except Exception as exc:
            import traceback
            error_msg = str(exc)
            error_traceback = traceback.format_exc()
            error_info = {
                "node": node,
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

        # Save children recursively
        for child in node.children:
            await self._save_node_recursive(
                node=child,
                handler=handler,
                saved_nodes=saved_nodes,
                failed_nodes=failed_nodes,
            )

