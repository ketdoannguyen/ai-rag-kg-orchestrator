"""Offline tests for the detail-merge switch in enrich_knowledge_tree.

The LLM client is faked, so these tests only describe which calls the enricher
decides to make and what it does with the children in each mode.
"""

from __future__ import annotations

import json
from typing import Any

from arango_client import (
    ParagraphAttributes,
    ParagraphNode,
    RootNode,
    StructureAttributes,
    StructureNode,
)
from llm_gateway_client.models import GatewayChatTurn, GatewayInvokeRequest, InvokeResponse

from workflows.enrich_knowledge_tree.enricher import KnowledgeTreeEnricher
from workflows.enrich_knowledge_tree.models import (
    EnrichKnowledgeTreeInput,
    RefineConfig,
)

BODY: dict[str, Any] = {
    "segments": [{"source_indices": [1, 2], "text": "Đoạn một\nĐoạn hai"}],
    "summary": "Tóm tắt bài",
}


class _FakeLLM:
    """Replays one canned JSON body for every call and records the prompts."""

    def __init__(self, body: dict[str, Any] | None = None) -> None:
        self.body = body if body is not None else BODY
        self.prompts: list[str] = []

    async def invoke(self, request: GatewayInvokeRequest) -> InvokeResponse:
        self.prompts.append(request.messages[0].content)
        return InvokeResponse(
            message=GatewayChatTurn(role="assistant", content=json.dumps(self.body)),
            finish_reason="stop",
            usage={},
        )


def _tree() -> StructureNode:
    section = StructureNode(level=2, attributes=StructureAttributes(title="Bài 1"))
    section.children = [
        ParagraphNode(attributes=ParagraphAttributes(text="Đoạn một")),
        ParagraphNode(attributes=ParagraphAttributes(text="Đoạn hai")),
    ]
    root = RootNode(attributes=StructureAttributes(title="Vật lí 12"))
    root.children = [section]
    return root


async def _enrich(refine_details: bool) -> tuple[Any, StructureNode, _FakeLLM]:
    root = _tree()
    llm = _FakeLLM()
    output = await KnowledgeTreeEnricher(llm).enrich(
        EnrichKnowledgeTreeInput(
            knowledge_tree=root,
            refine_config=RefineConfig(refine_details=refine_details),
        )
    )
    return output, root, llm


class TestDetailMergeSwitch:
    """The producer may already have chunked the details; then merging is skipped."""

    async def test_disabled_keeps_the_producer_chunks(self) -> None:
        output, root, llm = await _enrich(refine_details=False)

        section = root.children[0]
        assert [isinstance(child, ParagraphNode) for child in section.children] == [True, True]
        assert [child.attributes.text for child in section.children] == ["Đoạn một", "Đoạn hai"]
        assert output.statistics.detail_llm_calls == 0
        # Nothing that resembles a detail-merge prompt reaches the model.
        assert all("segments" not in prompt for prompt in llm.prompts)
        assert output.statistics.structure_llm_calls == 2

    async def test_enabled_merges_the_details(self) -> None:
        output, root, _ = await _enrich(refine_details=True)

        section = root.children[0]
        assert [child.attributes.text for child in section.children] == ["Đoạn một\nĐoạn hai"]
        assert output.statistics.detail_llm_calls == 1

    async def test_disabled_still_summarises_structure_nodes(self) -> None:
        output, root, _ = await _enrich(refine_details=False)

        assert root.children[0].summary == "Tóm tắt bài"
        assert output.statistics.paragraphs_before == output.statistics.paragraphs_after == 2


class TestRuntimeSummary:
    """The merge summary still reaches the tree at runtime (research must not)."""

    async def test_detail_summary_fills_sections_without_structure_children(self) -> None:
        body: dict[str, Any] = {
            "segments": [{"source_indices": [1, 2], "text": "Đoạn một\nĐoạn hai"}],
            "section_summary": "Tóm tắt mục",
            "summary": "Tóm tắt bài",
        }
        root = _tree()
        llm = _FakeLLM(body)

        output = await KnowledgeTreeEnricher(llm).enrich(
            EnrichKnowledgeTreeInput(knowledge_tree=root)
        )

        # The section has no structure children, so it reuses the merge summary
        # and the root is the only node that needs the structure summariser.
        assert root.children[0].summary == "Tóm tắt mục"
        assert root.summary == "Tóm tắt bài"
        assert output.statistics.detail_llm_calls == 1
        assert output.statistics.structure_llm_calls == 1
