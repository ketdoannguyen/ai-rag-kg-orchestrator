from __future__ import annotations

import asyncio
import json
from dataclasses import dataclass, field
from typing import Any

from arango_client import (
    KnowledgeNode,
    ParagraphNode,
    StructureNode,
)
from llm_gateway_client import AIGatewayClient
from llm_gateway_client.models import GatewayChatTurn, GatewayInvokeRequest
from loguru import logger
from prompts import build_structure_enrich_prompt
from tenacity import retry, retry_if_exception_type, stop_after_attempt, wait_exponential

from workflows.shared.file_utils import extract_json_from_response

from .detail_refiner import DetailRefiner
from .helper import (
    count_paragraphs,
    get_title,
)
from .models import (
    ChangesSummary,
    EnrichKnowledgeTreeInput,
    EnrichKnowledgeTreeOutput,
    EnrichStatistics,
)


@dataclass
class _EnrichState:
    """Mutable state shared across the traversal."""

    node_summaries: dict[str, str] = field(default_factory=dict)
    stats: EnrichStatistics = field(default_factory=EnrichStatistics)
    changes: ChangesSummary = field(default_factory=ChangesSummary)


class KnowledgeTreeEnricher:
    """Enriches a KnowledgeNode tree by refining detail nodes and summarising structure nodes."""

    def __init__(self, llm_client: AIGatewayClient) -> None:
        self.llm_client = llm_client
        self.detail_refiner = DetailRefiner(llm_client)

    # ---------------------------------------------------------------------
    # Low-level LLM invocation with retry
    # ---------------------------------------------------------------------
    @retry(
        retry=retry_if_exception_type(Exception),
        stop=stop_after_attempt(3),
        wait=wait_exponential(multiplier=1, min=1, max=8),
        reraise=True,
    )
    async def _invoke_with_retry(self, request: GatewayInvokeRequest):
        """Invoke LLM with simple exponential backoff retry."""
        return await self.llm_client.invoke(request)

    async def enrich(self, workflow_input: EnrichKnowledgeTreeInput) -> EnrichKnowledgeTreeOutput:
        """Run the enrich_knowledge_tree workflow."""
        # Expect a Pydantic KnowledgeNode/StructureNode as input
        root = workflow_input.knowledge_tree

        # Basic validation
        self._validate_root(root)

        state = _EnrichState()

        # Pre-compute simple statistics: paragraphs_before
        state.stats.paragraphs_before = count_paragraphs(root)

        # Global semaphore to limit concurrent section processing / LLM calls
        semaphore = asyncio.Semaphore(workflow_input.refine_config.max_concurrent_sections)

        # Bottom-up traversal: process all Structure nodes; children first, parent after.
        await self._process_structure_node(
            node=root,
            breadcrumb=[get_title(root)],
            state=state,
            workflow_input=workflow_input,
            semaphore=semaphore,
        )

        # Final statistics
        state.stats.paragraphs_after = count_paragraphs(root)

        # Root summary preview
        root_desc = None
        if isinstance(root, StructureNode):
            root_desc = root.summary
        if root_desc:
            state.changes.root_summary_preview = root_desc[:300]

        return EnrichKnowledgeTreeOutput(
            enriched_tree=root,
            statistics=state.stats,
            changes_summary=state.changes,
        )

    # ---------------------------------------------------------------------
    # Validation & helpers
    # ---------------------------------------------------------------------
    def _validate_root(self, root: KnowledgeNode) -> None:
        if root is None:
            raise ValueError("knowledge_tree cannot be None")
        if not isinstance(root, StructureNode):
            raise ValueError("Root knowledge_tree must be a StructureNode")
        if root.parent_id is not None:
            logger.warning(
                "Root StructureNode has non-null parent_id; continuing but this is unusual",
                extra={"root_id": str(root.id), "parent_id": str(root.parent_id)},
            )
        if not root.children:
            logger.info("Root StructureNode has no children; nothing to enrich")

    # ---------------------------------------------------------------------
    # Core traversal
    # ---------------------------------------------------------------------
    async def _process_structure_node(
        self,
        node: StructureNode,
        breadcrumb: list[str],
        state: _EnrichState,
        workflow_input: EnrichKnowledgeTreeInput,
        semaphore: asyncio.Semaphore,
    ) -> None:
        """Post-order processing for a single StructureNode.

        The semaphore is NOT acquired here — only at LLM call sites.
        Wrapping the entire recursive traversal with the semaphore caused
        deadlocks when the tree had more structure nodes than semaphore
        slots: parents held slots while waiting for children that could
        never acquire their own slots.
        """
        if not isinstance(node, StructureNode):
            return

        # 1) Process all STRUCTURE children concurrently (bottom-up)
        structure_children: list[StructureNode] = [
            c for c in node.children if isinstance(c, StructureNode)
        ]

        tasks = []
        for child in structure_children:
            child_breadcrumb = breadcrumb + [get_title(child)]
            tasks.append(
                self._process_structure_node(
                    node=child,
                    breadcrumb=child_breadcrumb,
                    state=state,
                    workflow_input=workflow_input,
                    semaphore=semaphore,
                )
            )

        if tasks:
            await asyncio.gather(*tasks, return_exceptions=False)

        # 2) Refine DETAIL children of current node (paragraphs/tables)
        (
            refined_children,
            section_stats,
            detail_section_summary,
        ) = await self._refine_detail_children(
            node=node,
            breadcrumb=breadcrumb,
            state=state,
            workflow_input=workflow_input,
            semaphore=semaphore,
        )

        # Replace children with refined version (images + refined paragraphs/tables + structure children)
        node.children = refined_children

        # 3) Enrich current StructureNode with summary that takes into account:
        #    - summaries from STRUCTURE children (stored in state.node_summaries)
        # If this node has no STRUCTURE children and detail-level LLM already
        # produced a section_summary, reuse it to avoid an extra LLM call.
        if not structure_children and detail_section_summary:
            section_summary = detail_section_summary
        else:
            section_summary = await self._enrich_structure_node(
                node=node,
                breadcrumb=breadcrumb,
                state=state,
                workflow_input=workflow_input,
                semaphore=semaphore,
            )

        # Attach summary to node
        if section_summary:
            node.summary = section_summary
            state.node_summaries[str(node.id)] = section_summary
            state.stats.structures_enriched += 1

        # Track per-section changes for changes_summary
        state.changes.refined_sections[str(node.id)] = {
            "title": get_title(node),
            "original_detail_count": section_stats.get("original_detail_count", 0),
            "refined_detail_count": section_stats.get("refined_detail_count", 0),
        }

    # ---------------------------------------------------------------------
    # Detail refinement
    # ---------------------------------------------------------------------
    async def _refine_detail_children(
        self,
        node: StructureNode,
        breadcrumb: list[str],
        state: _EnrichState,
        workflow_input: EnrichKnowledgeTreeInput,
        semaphore: asyncio.Semaphore,
    ) -> tuple[list[KnowledgeNode], dict[str, int], str | None]:
        """Refine DETAIL children (Paragraph/Table) of a StructureNode.

        The merge itself lives in `DetailRefiner`; this adapter feeds the
        traversal-level statistics and keeps the runtime call site unchanged.

        Returns:
            (new_children, section_stats, detail_section_summary)
        """
        refinement = await self.detail_refiner.refine_details(
            node=node,
            breadcrumb=breadcrumb,
            refine_config=workflow_input.refine_config,
            semaphore=semaphore,
        )
        state.stats.detail_llm_calls += refinement.llm_calls

        return (
            refinement.children,
            {
                "original_detail_count": refinement.original_detail_count,
                "refined_detail_count": refinement.refined_detail_count,
            },
            refinement.section_summary,
        )

    # ---------------------------------------------------------------------
    # Structure enrichment
    # ---------------------------------------------------------------------
    async def _enrich_structure_node(
        self,
        node: StructureNode,
        breadcrumb: list[str],
        state: _EnrichState,
        workflow_input: EnrichKnowledgeTreeInput,
        semaphore: asyncio.Semaphore | None = None,
    ) -> str | None:
        """Generate a full summary for a StructureNode, leveraging children summaries."""
        children_content: list[dict[str, Any]] = []

        for child in node.children:
            if isinstance(child, ParagraphNode):
                text = (child.attributes.text or "").strip()
                if not text:
                    continue
                children_content.append(
                    {
                        "type": "paragraph",
                        "title": None,
                        "content": text,
                    }
                )
            elif isinstance(child, StructureNode):
                child_summary = state.node_summaries.get(str(child.id))
                if not child_summary:
                    continue
                children_content.append(
                    {
                        "type": "section",
                        "title": get_title(child),
                        "content": child_summary,
                    }
                )

        if not children_content:
            # No textual content to summarize
            return None

        prompt = build_structure_enrich_prompt(
            title=get_title(node),
            children_content=children_content,
            summary_style=workflow_input.enrich_config.summary_style,
            short=False,
            breadcrumb=breadcrumb,
        )
        return await self._invoke_structure_llm_for_summary(
            prompt=prompt,
            workflow_input=workflow_input,
            state=state,
            semaphore=semaphore,
        )

    async def _invoke_structure_llm_for_summary(
        self,
        prompt: str,
        workflow_input: EnrichKnowledgeTreeInput,
        state: _EnrichState,
        semaphore: asyncio.Semaphore | None = None,
    ) -> str | None:
        """Call LLM for structure-level enrichment, expecting a JSON with 'summary'."""
        request = GatewayInvokeRequest(
            model=workflow_input.enrich_config.llm_model,
            messages=[
                GatewayChatTurn(
                    role="user",
                    content=prompt,
                )
            ],
            provider_options={"temperature": workflow_input.enrich_config.llm_temperature},
        )

        try:
            if semaphore:
                async with semaphore:
                    response = await self._invoke_with_retry(request)
            else:
                response = await self._invoke_with_retry(request)
            state.stats.structure_llm_calls += 1
        except Exception as e:
            logger.error(
                "Structure-level LLM call failed; falling back to None summary",
                extra={"error_type": type(e).__name__},
            )
            return None

        try:
            content = response.message.content
            parsed = extract_json_from_response(content)
            data = json.loads(parsed)
            summary = (data.get("summary") or "").strip()
            if not summary:
                return None
            return summary
        except Exception as e:
            logger.error(
                "Failed to parse structure LLM response; returning None",
                extra={"error_type": type(e).__name__},
            )
            return None
