"""Implementation of embed_nodes workflow.

Sử dụng arango_client models mới.
"""

from __future__ import annotations

from typing import Any, Dict, List

from loguru import logger
from llm_gateway_client import AIGatewayClient
from llm_gateway_client.models import EmbeddingRequest
from arango_client import (
    KnowledgeNode,
    StructureNode,
    NodeType,
)
from tenacity import retry, retry_if_exception_type, stop_after_attempt, wait_exponential

from workflows.dependencies import init_llm_client

from .models import EmbedNodesInput, EmbedNodesOutput, EmbeddingData
from .helper import collect_embeddable_nodes, create_batches, extract_text_for_embedding


async def start_embed_nodes(
    payload: EmbedNodesInput | Dict[str, Any],
    *,
    llm_client: AIGatewayClient | None = None,
) -> Dict[str, Any]:
    """
    Entry point for the embed_nodes workflow.

    Args:
        payload: Workflow input (Pydantic model or plain dict)
        llm_client: Optional LLM gateway client (defaults to init_llm_client())

    Returns:
        Dictionary representation of EmbedNodesOutput
    """
    input_payload = payload if isinstance(payload, EmbedNodesInput) else EmbedNodesInput(**payload)
    gateway_client = llm_client or init_llm_client()

    workflow = EmbedNodesWorkflow(llm_client=gateway_client)
    result = await workflow.run(input_payload)
    if hasattr(result, "model_dump"):
        return result.model_dump(mode="json")
    return result.dict()


class EmbedNodesWorkflow:
    """Implements the embed_nodes workflow with batch embedding and retry logic."""

    def __init__(
        self,
        *,
        llm_client: AIGatewayClient,
    ) -> None:
        self._llm_client = llm_client
        self._log = logger.bind(workflow="embed_nodes")

    async def run(self, payload: EmbedNodesInput) -> EmbedNodesOutput:
        """Execute the workflow end-to-end."""
        self._log.info("Starting embed nodes workflow", nodes_count=len(payload.nodes))

        # Validate input
        if not payload.nodes:
            raise ValueError("nodes list cannot be empty")

        # Collect nodes with embeddable text
        embeddable_items = collect_embeddable_nodes(payload.nodes)
        self._log.info(
            "Collected embeddable nodes",
            total_nodes=len(payload.nodes),
            embeddable_count=len(embeddable_items),
        )

        if not embeddable_items:
            self._log.warning("No nodes with embeddable text found")
            return EmbedNodesOutput(
                structure_embeddings=[],
                content_embeddings=[],
                total_embeddings_created=0,
                structure_embeddings_count=0,
                content_embeddings_count=0,
                failed_nodes=[],
            )

        # Create batches
        batches = create_batches(embeddable_items, payload.batch_size)
        self._log.info("Created batches", batch_count=len(batches), batch_size=payload.batch_size)

        # Process batches
        structure_embeddings: List[EmbeddingData] = []
        content_embeddings: List[EmbeddingData] = []
        failed_nodes: List[Dict[str, Any]] = []

        for batch_idx, batch in enumerate(batches):
            self._log.debug("Processing batch", batch_index=batch_idx + 1, batch_size=len(batch))
            await self._process_batch(
                batch=batch,
                model=payload.model,
                structure_embeddings=structure_embeddings,
                content_embeddings=content_embeddings,
                failed_nodes=failed_nodes,
            )

        # Calculate statistics
        total_created = len(structure_embeddings) + len(content_embeddings)

        self._log.info(
            "Embedding workflow completed",
            total_embeddings=total_created,
            structure_count=len(structure_embeddings),
            content_count=len(content_embeddings),
            failed_count=len(failed_nodes),
        )

        return EmbedNodesOutput(
            structure_embeddings=structure_embeddings,
            content_embeddings=content_embeddings,
            total_embeddings_created=total_created,
            structure_embeddings_count=len(structure_embeddings),
            content_embeddings_count=len(content_embeddings),
            failed_nodes=failed_nodes,
        )

    @retry(
        retry=retry_if_exception_type(Exception),
        stop=stop_after_attempt(3),
        wait=wait_exponential(multiplier=1, min=1, max=8),
        reraise=False,
    )
    async def _embed_batch_with_retry(
        self,
        texts: List[str],
        model: str | None,
    ) -> List[List[float]] | None:
        """Embed a batch of texts with retry logic."""
        try:
            request = EmbeddingRequest(input=texts, model=model)
            response = await self._llm_client.embed(request)
            return response.embeddings
        except Exception as exc:
            self._log.warning("Embedding batch failed", error=str(exc), text_count=len(texts))
            raise

    def _get_node_type_str(self, node: KnowledgeNode) -> str:
        """Get node_type as string."""
        node_type = node.node_type
        if hasattr(node_type, 'value'):
            return node_type.value
        return str(node_type).lower()

    async def _process_batch(
        self,
        batch: List[tuple[KnowledgeNode, str]],
        model: str | None,
        structure_embeddings: List[EmbeddingData],
        content_embeddings: List[EmbeddingData],
        failed_nodes: List[Dict[str, Any]],
    ) -> None:
        """Process a single batch of nodes."""
        # Extract texts from batch
        texts = [text for _, text in batch]
        nodes = [node for node, _ in batch]

        # Embed batch with retry
        try:
            vectors = await self._embed_batch_with_retry(texts, model)
            if vectors is None:
                # Retry exhausted
                for node, text in batch:
                    failed_nodes.append({
                        "node_id": str(node.id),
                        "node_type": self._get_node_type_str(node),
                        "error": "Embedding failed after retries",
                        "retry_count": 3,
                    })
                return

            # Map vectors back to nodes
            if len(vectors) != len(nodes):
                self._log.error(
                    "Vector count mismatch",
                    expected=len(nodes),
                    got=len(vectors),
                )
                for node, text in batch:
                    failed_nodes.append({
                        "node_id": str(node.id),
                        "node_type": self._get_node_type_str(node),
                        "error": "Vector count mismatch",
                        "retry_count": 0,
                    })
                return

            # Create embedding data
            for (node, text), vector in zip(batch, vectors):
                node_type = self._get_node_type_str(node)
                
                embedding_data = EmbeddingData(
                    node_id=node.id,
                    vector=vector,
                    text=text,
                )
                
                if node_type == "structure":
                    structure_embeddings.append(embedding_data)
                else:
                    content_embeddings.append(embedding_data)

        except Exception as exc:
            self._log.error(
                "Batch embedding failed after retries",
                error=str(exc),
                batch_size=len(batch),
            )
            for node, text in batch:
                failed_nodes.append({
                    "node_id": str(node.id),
                    "node_type": self._get_node_type_str(node),
                    "error": str(exc),
                    "retry_count": 3,
                })
