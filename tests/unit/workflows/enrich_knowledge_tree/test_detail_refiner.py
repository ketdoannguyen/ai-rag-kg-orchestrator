"""Offline tests for the detail-only refinement seam.

The LLM client is faked, so these tests only describe what the seam decides:
when it skips the merge, when it calls the model, what it does with the answer,
and that the research call path never reaches structure summaries.
"""

from __future__ import annotations

import asyncio
import json
from collections.abc import Callable, Mapping, Sequence
from typing import Any

import pytest
from arango_client import (
    ImageNode,
    ParagraphAttributes,
    ParagraphNode,
    StructureAttributes,
    StructureNode,
    TableAttributes,
    TableNode,
)
from llm_gateway_client.models import GatewayChatTurn, GatewayInvokeRequest, InvokeResponse
from tenacity import wait_none

from workflows.enrich_knowledge_tree.detail_refiner import DetailRefinement, DetailRefiner
from workflows.enrich_knowledge_tree.models import RefineConfig

BODY: dict[str, Any] = {
    "segments": [{"source_indices": [1, 2], "text": "Đoạn một\nĐoạn hai"}],
    "section_summary": "Tóm tắt mục",
}

BODY_WITH_SEGMENT_SUMMARY: dict[str, Any] = {
    "segments": [
        {
            "source_indices": [1, 2],
            "text": "Đoạn một\nĐoạn hai",
            "summary": "Tóm tắt đoạn",
        }
    ],
    "section_summary": "Tóm tắt mục",
}


class _FakeLLM:
    """Replays one canned body for every call and records the prompts.

    ``usage`` is the normalized view every response carries, its default the
    all-zero envelope the gateway builds for a response without usage.
    ``stated_usage`` and ``model`` are what the provider itself reported, which a
    real response always carries and a double may leave out. Both replay one
    entry per answered call and repeat the last, so two groups answering in one
    section can be told apart. ``attempts`` is how many calls the gateway itself
    made for the answer it returned: a real one reports that.
    """

    def __init__(
        self,
        body: Any = None,
        fail_times: int = 0,
        usage: dict[str, int] | None = None,
        stated_usage: Sequence[Mapping[str, Any] | None] | None = None,
        model: Sequence[str | None] | None = None,
        attempts: int = 1,
    ) -> None:
        self.body = BODY if body is None else body
        self.fail_times = fail_times
        self.usage = {} if usage is None else usage
        self.stated_usage = [None] if stated_usage is None else list(stated_usage)
        self.models = [None] if model is None else list(model)
        self.attempts = attempts
        self.calls = 0
        self.prompts: list[str] = []

    def _answered(self, answers: list[Any]) -> Any:
        """The entry for the call this is, counting only the ones that answered."""
        return answers[min(self.calls - self.fail_times - 1, len(answers) - 1)]

    async def invoke(self, request: GatewayInvokeRequest) -> InvokeResponse:
        self.calls += 1
        self.prompts.append(request.messages[0].content)
        if self.calls <= self.fail_times:
            raise RuntimeError("gateway unavailable")
        content = self.body if isinstance(self.body, str) else json.dumps(self.body)
        return InvokeResponse(
            message=GatewayChatTurn(role="assistant", content=content),
            finish_reason="stop",
            usage=self.usage,
            model=self._answered(self.models),
            stated_usage=self._answered(self.stated_usage),
            attempts=self.attempts,
        )


@pytest.fixture(autouse=True)
def _no_retry_backoff(monkeypatch: pytest.MonkeyPatch) -> None:
    """Keep the real 3-attempt retry but drop the exponential sleep."""
    monkeypatch.setattr(DetailRefiner._invoke_with_retry.retry, "wait", wait_none())


def _section(*texts: str) -> StructureNode:
    section = StructureNode(level=2, attributes=StructureAttributes(title="Bài 1"))
    section.children = [ParagraphNode(attributes=ParagraphAttributes(text=text)) for text in texts]
    return section


def _table(text: str = "Bảng 1.1: Nhiệt độ theo thời gian", page: int = 7) -> TableNode:
    return TableNode(
        attributes=TableAttributes(
            extracted_text=text,
            headers=["t (s)", "T (K)"],
            rows=[["0", "273"]],
        ),
        metadata={"page": page},
    )


def _empty_table() -> TableNode:
    """A table the producer could read no text out of."""
    return TableNode(attributes=TableAttributes(extracted_text="", headers=[], rows=[]))


async def _refine(
    section: StructureNode,
    llm: _FakeLLM,
    *,
    refine_details: bool = True,
    preserve_tables: bool = False,
    segment_summaries: bool = True,
    semaphore: asyncio.Semaphore | None = None,
    on_spend: Callable[[dict[str, Any]], None] | None = None,
) -> DetailRefinement:
    return await DetailRefiner(llm).refine_details(
        node=section,
        breadcrumb=["Vật lí 12", "Bài 1"],
        refine_config=RefineConfig(
            refine_details=refine_details,
            preserve_tables=preserve_tables,
            segment_summaries=segment_summaries,
        ),
        semaphore=semaphore,
        on_spend=on_spend,
    )


class TestMergeSkip:
    """One or zero details cannot be merged, so no LLM call is made."""

    async def test_single_detail_never_calls_the_model(self) -> None:
        section = _section("Đoạn một")
        llm = _FakeLLM()

        result = await _refine(section, llm)

        assert result.merge_skipped is True
        assert result.merge_fell_back is False
        assert result.llm_calls == 0
        assert llm.calls == 0
        assert result.section_summary is None
        assert result.original_detail_count == result.refined_detail_count == 1
        assert result.children == list(section.children)

    async def test_disabled_merge_keeps_the_producer_chunks(self) -> None:
        section = _section("Đoạn một", "Đoạn hai")
        llm = _FakeLLM()

        result = await _refine(section, llm, refine_details=False)

        assert result.merge_skipped is True
        assert llm.calls == 0
        assert result.children == list(section.children)

    async def test_details_without_text_are_kept_as_is(self) -> None:
        section = _section("", "")
        llm = _FakeLLM()

        result = await _refine(section, llm)

        assert result.merge_skipped is True
        assert llm.calls == 0
        assert result.children == list(section.children)


class TestMerge:
    """Two or more details are merged by one LLM call."""

    async def test_two_details_merge_into_one_segment(self) -> None:
        section = _section("Đoạn một", "Đoạn hai")
        llm = _FakeLLM()

        result = await _refine(section, llm)

        assert result.merge_skipped is False
        assert result.merge_fell_back is False
        assert result.llm_calls == 1
        assert [child.attributes.text for child in result.children] == ["Đoạn một\nĐoạn hai"]
        assert result.original_detail_count == 2
        assert result.refined_detail_count == 1

    async def test_images_and_structure_children_follow_the_merged_details(self) -> None:
        section = _section("Đoạn một", "Đoạn hai")
        image = ImageNode()
        subsection = StructureNode(level=3, attributes=StructureAttributes(title="Mục con"))
        section.children = [section.children[0], image, section.children[1], subsection]

        result = await _refine(section, _FakeLLM())

        assert [type(child) for child in result.children] == [
            ParagraphNode,
            ImageNode,
            StructureNode,
        ]

    async def test_semaphore_does_not_block_the_merge(self) -> None:
        section = _section("Đoạn một", "Đoạn hai")

        result = await _refine(section, _FakeLLM(), semaphore=asyncio.Semaphore(1))

        assert result.llm_calls == 1
        assert [child.attributes.text for child in result.children] == ["Đoạn một\nĐoạn hai"]


class TestResearchContract:
    """The research call refines details only; it never writes to the tree."""

    async def test_merge_summary_is_reported_but_not_applied(self) -> None:
        section = _section("Đoạn một", "Đoạn hai")
        original_children = list(section.children)

        result = await _refine(section, _FakeLLM())

        assert result.section_summary == "Tóm tắt mục"
        assert section.summary is None
        assert section.children == original_children

    async def test_only_the_detail_prompt_reaches_the_model(self) -> None:
        section = _section("Đoạn một", "Đoạn hai")
        llm = _FakeLLM()

        await _refine(section, llm)

        assert len(llm.prompts) == 1
        assert "PRIMARY TASK: Merge semantically related items" in llm.prompts[0]
        assert "RAG-OPTIMIZED SUMMARY TASK" not in llm.prompts[0]


class TestFallback:
    """A failed or unusable merge keeps the original details."""

    async def test_transient_failure_is_retried(self) -> None:
        section = _section("Đoạn một", "Đoạn hai")
        llm = _FakeLLM(fail_times=1)

        result = await _refine(section, llm)

        assert llm.calls == 2  # one failed attempt + one success
        assert result.llm_calls == 1
        assert [child.attributes.text for child in result.children] == ["Đoạn một\nĐoạn hai"]

    async def test_exhausted_retries_keep_the_originals(self) -> None:
        section = _section("Đoạn một", "Đoạn hai")
        original_children = list(section.children)
        llm = _FakeLLM(fail_times=3)

        result = await _refine(section, llm)

        assert llm.calls == 3  # the 3 attempts of the runtime retry policy
        assert result.merge_skipped is False  # the merge was attempted
        assert result.merge_fell_back is True  # and the originals survived it
        assert result.llm_calls == 0  # no invocation completed
        assert result.children == original_children
        assert result.section_summary is None

    async def test_unparseable_response_keeps_the_originals(self) -> None:
        section = _section("Đoạn một", "Đoạn hai")
        original_children = list(section.children)

        result = await _refine(section, _FakeLLM(body="not json"))

        assert result.llm_calls == 1  # the call completed, the parse failed
        assert result.merge_fell_back is True  # an answer nothing could be read from
        assert result.children == original_children
        assert result.section_summary is None


class TestMergeAccounting:
    """The merge's spend is observed at the gateway seam, never invented."""

    async def test_a_lone_call_reports_what_the_gateway_stated(self) -> None:
        section = _section("Đoạn một", "Đoạn hai")
        llm = _FakeLLM(usage={"prompt_tokens": 1200, "completion_tokens": 800})

        result = await _refine(section, llm)

        assert result.attempts == 1
        assert (result.prompt_tokens, result.completion_tokens) == (1200, 800)
        assert result.usage_unknown_reason is None
        assert result.latency_seconds >= 0.0

    async def test_usage_the_gateway_never_reported_is_null_with_a_reason(self) -> None:
        section = _section("Đoạn một", "Đoạn hai")
        llm = _FakeLLM(usage={})  # the all-zero envelope of an absent usage

        result = await _refine(section, llm)

        assert result.attempts == 1  # measured even when the usage is not
        assert (result.prompt_tokens, result.completion_tokens) == (None, None)
        assert result.usage_unknown_reason is not None
        assert "no token usage" in result.usage_unknown_reason

    async def test_retries_count_as_attempts_not_as_logical_calls(self) -> None:
        section = _section("Đoạn một", "Đoạn hai")
        llm = _FakeLLM(fail_times=2)

        result = await _refine(section, llm)

        assert llm.calls == 3
        assert result.llm_calls == 1  # one invocation completed
        assert result.attempts == 3  # three calls at the seam

    async def test_a_retried_call_reports_an_unknown_bill_not_a_subtotal(self) -> None:
        section = _section("Đoạn một", "Đoạn hai")
        llm = _FakeLLM(fail_times=1, usage={"prompt_tokens": 1200, "completion_tokens": 800})

        result = await _refine(section, llm)

        assert result.attempts == 2
        assert (result.prompt_tokens, result.completion_tokens) == (None, None)
        assert result.usage_unknown_reason is not None
        assert "no response" in result.usage_unknown_reason

    async def test_a_failed_merge_still_shows_its_spend(self) -> None:
        section = _section("Đoạn một", "Đoạn hai")
        llm = _FakeLLM(fail_times=3)

        result = await _refine(section, llm)

        assert result.merge_skipped is False  # the merge ran and failed
        assert result.llm_calls == 0
        assert result.attempts == 3
        assert (result.prompt_tokens, result.completion_tokens) == (None, None)
        assert result.usage_unknown_reason is not None
        assert result.latency_seconds >= 0.0

    async def test_a_parse_failure_keeps_the_spend_of_the_call_that_answered(self) -> None:
        section = _section("Đoạn một", "Đoạn hai")
        llm = _FakeLLM(body="not json", usage={"prompt_tokens": 1200, "completion_tokens": 800})

        result = await _refine(section, llm)

        assert result.llm_calls == 1
        assert result.attempts == 1
        assert (result.prompt_tokens, result.completion_tokens) == (1200, 800)

    async def test_a_skipped_merge_spends_nothing(self) -> None:
        section = _section("Đoạn một")
        llm = _FakeLLM()

        result = await _refine(section, llm)

        assert result.merge_skipped is True
        assert result.attempts == 0
        assert (result.prompt_tokens, result.completion_tokens) == (0, 0)
        assert result.usage_unknown_reason is None
        assert result.latency_seconds == 0.0


class TestResearchTablePolicy:
    """`preserve_tables` makes a table a merge boundary, never a merge input.

    The research pipeline scores the tree a merge returns, so a table the merge
    consumed or demoted would corrupt the thing the pipeline measures.
    """

    async def test_a_table_is_never_demoted_and_the_paragraphs_merge_around_it(self) -> None:
        table = _table()
        section = _section("Đoạn một", "Đoạn hai", "Đoạn ba")
        p1, p2, p3 = section.children
        section.children = [p1, table, p2, p3]
        llm = _FakeLLM()

        result = await _refine(section, llm, preserve_tables=True)

        assert llm.calls == 1
        assert "Đoạn hai" in llm.prompts[0]
        assert "Đoạn ba" in llm.prompts[0]
        assert "Đoạn một" not in llm.prompts[0]  # a lone paragraph is not merged
        assert table.attributes.extracted_text not in llm.prompts[0]
        assert result.children[0] is p1
        assert result.children[1] is table
        assert result.children[1].attributes is table.attributes
        assert result.children[1].metadata is table.metadata
        assert [child.attributes.text for child in result.children[2:]] == ["Đoạn một\nĐoạn hai"]

    async def test_the_refined_count_includes_the_preserved_table(self) -> None:
        table = _table()
        section = _section("Đoạn một", "Đoạn hai", "Đoạn ba")
        p1, p2, p3 = section.children
        section.children = [p1, table, p2, p3]

        result = await _refine(section, _FakeLLM(), preserve_tables=True)

        assert result.original_detail_count == 4
        assert result.refined_detail_count == 3  # merged p2+p3, plus p1 and the table

    async def test_a_malformed_merge_keeps_every_original_node(self) -> None:
        table = _table()
        section = _section("Đoạn một", "Đoạn hai", "Đoạn ba")
        p1, p2, p3 = section.children
        section.children = [p1, table, p2, p3]
        original_children = list(section.children)

        result = await _refine(section, _FakeLLM(body="not json"), preserve_tables=True)

        assert result.llm_calls == 1  # the call completed, the parse failed
        assert result.merge_fell_back is True
        assert result.children[1] is table
        assert not isinstance(result.children[1], ParagraphNode)  # never demoted
        assert all(a is b for a, b in zip(result.children, original_children, strict=True))

    async def test_a_table_with_no_text_is_still_a_boundary_and_survives(self) -> None:
        # A table the producer read no text out of has nothing to merge, but it is
        # still a table between the paragraphs: a merge reaching across it would
        # both rewrite text around it and drop the node from the rebuilt children.
        table = _empty_table()
        section = _section("Đoạn một", "Đoạn hai")
        p1, p2 = section.children
        section.children = [p1, table, p2]
        llm = _FakeLLM()

        result = await _refine(section, llm, preserve_tables=True)

        assert llm.calls == 0  # a lone paragraph on either side has nothing to merge
        assert result.children == [p1, table, p2]
        assert result.children[1] is table
        assert result.original_detail_count == result.refined_detail_count == 3

    async def test_paragraphs_merge_around_an_empty_table_without_consuming_it(self) -> None:
        table = _empty_table()
        section = _section("Đoạn một", "Đoạn hai", "Đoạn ba", "Đoạn bốn")
        p1, p2, p3, p4 = section.children
        section.children = [p1, p2, table, p3, p4]
        llm = _FakeLLM()

        result = await _refine(section, llm, preserve_tables=True)

        assert llm.calls == 2
        assert [type(child) for child in result.children] == [
            ParagraphNode,
            TableNode,
            ParagraphNode,
        ]
        assert result.children[1] is table
        assert result.original_detail_count == 5
        assert result.refined_detail_count == 3

    async def test_the_runtime_default_still_ignores_a_table_with_no_text(self) -> None:
        # The policy is the research runner's; the app's own path keeps the one
        # merge over the details it has always made.
        table = _empty_table()
        section = _section("Đoạn một", "Đoạn hai")
        p1, p2 = section.children
        section.children = [p1, table, p2]
        llm = _FakeLLM(body={"segments": [{"source_indices": [1, 2, 3], "text": "Gộp"}]})

        result = await _refine(section, llm)

        assert llm.calls == 1
        assert [child.attributes.text for child in result.children] == ["Gộp"]

    async def test_an_unconsumed_detail_keeps_the_node_it_already_was(self) -> None:
        section = _section("Đoạn một", "Đoạn hai", "Đoạn ba")
        p1, p2, p3 = section.children
        body = {"segments": [{"source_indices": [1, 2], "text": "Gộp"}]}

        result = await _refine(section, _FakeLLM(body=body), preserve_tables=True)

        assert [child.attributes.text for child in result.children] == ["Gộp", "Đoạn ba"]
        assert result.children[1] is p3  # the node itself, not a rebuilt paragraph

    async def test_a_section_of_lone_details_never_calls_the_model(self) -> None:
        table = _table()
        section = _section("Đoạn một")
        p1 = section.children[0]
        section.children = [p1, table]
        llm = _FakeLLM()

        result = await _refine(section, llm, preserve_tables=True)

        assert llm.calls == 0
        assert result.merge_skipped is True
        assert result.llm_calls == 0
        assert result.children == [p1, table]
        assert result.original_detail_count == result.refined_detail_count == 2

    async def test_two_runs_around_a_table_cost_two_calls(self) -> None:
        table = _table()
        section = _section("Đoạn một", "Đoạn hai", "Đoạn ba", "Đoạn bốn")
        p1, p2, p3, p4 = section.children
        section.children = [p1, p2, table, p3, p4]
        llm = _FakeLLM(usage={"prompt_tokens": 1200, "completion_tokens": 800})

        result = await _refine(section, llm, preserve_tables=True)

        assert llm.calls == 2
        assert result.llm_calls == 2
        assert result.attempts == 2
        assert (result.prompt_tokens, result.completion_tokens) == (2400, 1600)
        assert result.usage_unknown_reason is None
        assert result.children[1] is table

    def test_the_research_policy_is_opt_in(self) -> None:
        # The app's own paths keep their defaults: the research policy is a
        # choice the research runner makes, not a new default for everybody.
        config = RefineConfig(refine_details=True)

        assert config.preserve_tables is False
        assert config.segment_summaries is True

    async def test_the_runtime_default_still_merges_the_whole_list_in_one_call(self) -> None:
        table = _table()
        section = _section("Đoạn một", "Đoạn hai")
        p1, p2 = section.children
        section.children = [p1, table, p2]
        llm = _FakeLLM(body={"segments": [{"source_indices": [1, 2, 3], "text": "Gộp"}]})

        result = await _refine(section, llm)  # preserve_tables defaults to off

        assert llm.calls == 1
        assert table.attributes.extracted_text in llm.prompts[0]
        assert [child.attributes.text for child in result.children] == ["Gộp"]

    async def test_the_runtime_prefixes_a_segment_with_its_summary(self) -> None:
        section = _section("Đoạn một", "Đoạn hai")

        result = await _refine(section, _FakeLLM(body=BODY_WITH_SEGMENT_SUMMARY))

        assert [child.attributes.text for child in result.children] == [
            "Tóm tắt đoạn : Đoạn một\nĐoạn hai"
        ]

    async def test_research_stores_the_segment_without_its_summary(self) -> None:
        section = _section("Đoạn một", "Đoạn hai")

        result = await _refine(
            section,
            _FakeLLM(body=BODY_WITH_SEGMENT_SUMMARY),
            segment_summaries=False,
        )

        assert [child.attributes.text for child in result.children] == ["Đoạn một\nĐoạn hai"]
        assert result.section_summary == "Tóm tắt mục"  # reported as data, never applied


class TestProviderProvenance:
    """What the provider itself reported: who answered, and what it cost.

    The merge requests a model by name; a gateway may answer with another one, so
    only the model in the response is evidence of who served the call, and only
    that name can be priced. The records are the shape pricing reads, one per
    served model, so nothing has to be inferred between the two.
    """

    async def test_the_model_that_answered_carries_the_sections_tokens(self) -> None:
        section = _section("Đoạn một", "Đoạn hai")
        llm = _FakeLLM(
            stated_usage=[{"prompt_tokens": 1200, "completion_tokens": 800}],
            model=["deepseek-v4.1-flash"],
        )

        result = await _refine(section, llm)

        assert result.serving_models == [
            {
                "model": "deepseek-v4.1-flash",
                "usage": {
                    "prompt_tokens": 1200,
                    "completion_tokens": 800,
                    "cached_tokens": 0,
                },
            }
        ]
        assert (result.prompt_tokens, result.completion_tokens) == (1200, 800)

    async def test_two_models_split_the_bill_of_one_section(self) -> None:
        table = _table()
        section = _section("Đoạn một", "Đoạn hai", "Đoạn ba", "Đoạn bốn")
        p1, p2, p3, p4 = section.children
        section.children = [p1, p2, table, p3, p4]
        llm = _FakeLLM(
            stated_usage=[
                {"prompt_tokens": 1200, "completion_tokens": 800},
                {"prompt_tokens": 100, "completion_tokens": 50},
            ],
            model=["served-a", "served-b"],
        )

        result = await _refine(section, llm, preserve_tables=True)

        assert result.serving_models == [
            {
                "model": "served-a",
                "usage": {"prompt_tokens": 1200, "completion_tokens": 800, "cached_tokens": 0},
            },
            {
                "model": "served-b",
                "usage": {"prompt_tokens": 100, "completion_tokens": 50, "cached_tokens": 0},
            },
        ]
        assert (result.prompt_tokens, result.completion_tokens) == (1300, 850)

    async def test_an_answer_that_names_no_model_is_recorded_as_unattributed(self) -> None:
        section = _section("Đoạn một", "Đoạn hai")
        llm = _FakeLLM(stated_usage=[{"prompt_tokens": 1200, "completion_tokens": 800}])

        result = await _refine(section, llm)

        assert (result.prompt_tokens, result.completion_tokens) == (1200, 800)
        # The tokens are known but their model is not, so the record is kept
        # without a name: pricing reads that as the merge it cannot price.
        assert result.serving_models == [
            {
                "model": None,
                "usage": {"prompt_tokens": 1200, "completion_tokens": 800, "cached_tokens": 0},
            }
        ]

    async def test_a_response_with_no_envelope_of_its_own_is_still_read(self) -> None:
        section = _section("Đoạn một", "Đoạn hai")
        llm = _FakeLLM(usage={"prompt_tokens": 1200, "completion_tokens": 800})

        result = await _refine(section, llm)

        assert (result.prompt_tokens, result.completion_tokens) == (1200, 800)
        assert result.serving_models == [
            {
                "model": None,
                "usage": {"prompt_tokens": 1200, "completion_tokens": 800, "cached_tokens": 0},
            }
        ]

    async def test_an_unobservable_bill_is_not_split(self) -> None:
        # The call that answered stated 1200/800, but the dead attempt before it
        # makes those a subtotal: pricing them would report a part as the whole.
        section = _section("Đoạn một", "Đoạn hai")
        llm = _FakeLLM(
            fail_times=1,
            stated_usage=[{"prompt_tokens": 1200, "completion_tokens": 800}],
            model=["served-a"],
        )

        result = await _refine(section, llm)

        assert result.prompt_tokens is None
        assert result.serving_models == []


class TestCacheAccounting:
    """The cache share is what the provider stated, never a zero it did not claim."""

    async def test_a_stated_cache_share_is_reported_where_pricing_reads_it(self) -> None:
        section = _section("Đoạn một", "Đoạn hai")
        llm = _FakeLLM(
            stated_usage=[
                {
                    "prompt_tokens": 1200,
                    "completion_tokens": 800,
                    "prompt_tokens_details": {"cached_tokens": 900},
                }
            ],
            model=["served-a"],
        )

        result = await _refine(section, llm)

        assert result.cached_tokens == 900
        assert result.serving_models[0]["usage"]["cached_tokens"] == 900

    async def test_an_endpoint_that_keeps_no_cache_reports_a_measured_zero(self) -> None:
        section = _section("Đoạn một", "Đoạn hai")
        llm = _FakeLLM(stated_usage=[{"prompt_tokens": 1200, "completion_tokens": 800}])

        result = await _refine(section, llm)

        assert result.cached_tokens == 0

    async def test_a_cache_share_mentioned_without_a_count_is_unknown(self) -> None:
        section = _section("Đoạn một", "Đoạn hai")
        llm = _FakeLLM(
            stated_usage=[
                {
                    "prompt_tokens": 1200,
                    "completion_tokens": 800,
                    "prompt_tokens_details": {"cached_tokens": None},
                }
            ]
        )

        result = await _refine(section, llm)

        assert result.cached_tokens is None

    async def test_answers_that_disagree_about_the_cache_leave_it_unknown(self) -> None:
        table = _table()
        section = _section("Đoạn một", "Đoạn hai", "Đoạn ba", "Đoạn bốn")
        p1, p2, p3, p4 = section.children
        section.children = [p1, p2, table, p3, p4]
        llm = _FakeLLM(
            stated_usage=[
                {"prompt_tokens": 1200, "completion_tokens": 800, "cached_tokens": 900},
                {"prompt_tokens": 100, "completion_tokens": 50},
            ]
        )

        result = await _refine(section, llm, preserve_tables=True)

        # One answer spoke and one stayed silent, so the sum of the one that spoke
        # would be a subtotal presented as the share.
        assert result.cached_tokens is None
        assert result.serving_models[0]["usage"]["cached_tokens"] is None

    async def test_a_retried_call_leaves_the_cache_unknown(self) -> None:
        section = _section("Đoạn một", "Đoạn hai")
        llm = _FakeLLM(
            fail_times=1,
            stated_usage=[{"prompt_tokens": 1200, "completion_tokens": 800, "cached_tokens": 900}],
        )

        result = await _refine(section, llm)

        assert result.prompt_tokens is None  # the dead attempt's bill is unobservable
        assert result.cached_tokens is None

    async def test_a_skipped_merge_reports_a_measured_zero_cache(self) -> None:
        section = _section("Đoạn một")

        result = await _refine(section, _FakeLLM())

        assert result.cached_tokens == 0


class TestStatedCounts:
    """A count the provider left out is unknown, not a zero it never claimed."""

    async def test_a_count_the_provider_left_out_is_unknown(self) -> None:
        section = _section("Đoạn một", "Đoạn hai")
        llm = _FakeLLM(stated_usage=[{"completion_tokens": 800}])

        result = await _refine(section, llm)

        assert result.prompt_tokens is None
        assert result.completion_tokens is None  # the bill is all or nothing
        assert result.usage_unknown_reason is not None

    async def test_a_half_stated_bill_is_unknown_with_a_reason(self) -> None:
        # The prompt count came back and the completion count did not: a bill
        # with no reason beside ``None`` would read as one nobody explained.
        section = _section("Đoạn một", "Đoạn hai")
        llm = _FakeLLM(stated_usage=[{"prompt_tokens": 1200}])

        result = await _refine(section, llm)

        assert result.attempts == 1
        assert (result.prompt_tokens, result.completion_tokens) == (None, None)
        assert result.usage_unknown_reason is not None
        assert result.serving_models == []

    async def test_a_zero_the_provider_stated_is_a_measured_zero(self) -> None:
        # The all-zero envelope used to mean "not reported" because a normalized
        # response could not say otherwise; the provider's own envelope can.
        section = _section("Đoạn một", "Đoạn hai")
        llm = _FakeLLM(stated_usage=[{"prompt_tokens": 0, "completion_tokens": 0}])

        result = await _refine(section, llm)

        assert result.attempts == 1
        assert (result.prompt_tokens, result.completion_tokens) == (0, 0)
        assert result.usage_unknown_reason is None


class TestGatewayRetrySpend:
    """The gateway retries on its own behalf, and its answer is then a subtotal.

    One call at this seam can be several calls to the provider. The answer that
    came back states the counts of its last attempt alone, and whether the
    attempts before it were billed is not something it can say, so pricing those
    counts would report a part of the bill as the whole of it.
    """

    async def test_an_answer_after_a_gateway_retry_is_not_priced_as_the_whole_bill(self) -> None:
        section = _section("Đoạn một", "Đoạn hai")
        llm = _FakeLLM(
            attempts=2,
            stated_usage=[{"prompt_tokens": 1200, "completion_tokens": 800}],
            model=["served-a"],
        )

        result = await _refine(section, llm)

        assert llm.calls == 1  # one call here, retried behind the gateway
        assert result.llm_calls == 1  # and it completed
        assert result.attempts == 1
        assert (result.prompt_tokens, result.completion_tokens) == (None, None)
        assert result.usage_unknown_reason is not None
        assert "retried" in result.usage_unknown_reason
        assert result.serving_models == []

    async def test_an_answer_with_no_gateway_retry_behind_it_still_reports_its_bill(self) -> None:
        section = _section("Đoạn một", "Đoạn hai")
        llm = _FakeLLM(
            attempts=1,
            stated_usage=[{"prompt_tokens": 1200, "completion_tokens": 800}],
            model=["served-a"],
        )

        result = await _refine(section, llm)

        assert (result.prompt_tokens, result.completion_tokens) == (1200, 800)
        assert result.usage_unknown_reason is None


class TestPartialSpend:
    """Each call is handed over as it closes, so a cancelled walk keeps its spend."""

    async def test_every_call_is_handed_over_with_its_own_spend(self) -> None:
        table = _table()
        section = _section("Đoạn một", "Đoạn hai", "Đoạn ba", "Đoạn bốn")
        p1, p2, p3, p4 = section.children
        section.children = [p1, p2, table, p3, p4]
        llm = _FakeLLM(
            stated_usage=[
                {"prompt_tokens": 1200, "completion_tokens": 800},
                {"prompt_tokens": 100, "completion_tokens": 50},
            ],
            model=["served-a", "served-b"],
        )
        seen: list[dict[str, Any]] = []

        result = await _refine(section, llm, preserve_tables=True, on_spend=seen.append)

        assert [spend["attempts"] for spend in seen] == [1, 1]
        assert [(spend["prompt_tokens"], spend["completion_tokens"]) for spend in seen] == [
            (1200, 800),
            (100, 50),
        ]
        assert seen[-1]["usage_unknown_reason"] is None
        assert seen[-1]["latency_seconds"] >= 0.0
        # What was handed over is what the refinement reports.
        assert result.attempts == sum(spend["attempts"] for spend in seen)
        assert (result.prompt_tokens, result.completion_tokens) == (1300, 850)

    async def test_a_failed_call_is_handed_over_with_the_attempts_it_made(self) -> None:
        section = _section("Đoạn một", "Đoạn hai")
        llm = _FakeLLM(fail_times=3)
        seen: list[dict[str, Any]] = []

        result = await _refine(section, llm, on_spend=seen.append)

        assert llm.calls == 3
        assert [spend["attempts"] for spend in seen] == [3]
        assert seen[0]["prompt_tokens"] is None
        assert seen[0]["usage_unknown_reason"] == result.usage_unknown_reason

    async def test_a_cancelled_call_is_handed_over_before_the_walk_unwinds(self) -> None:
        section = _section("Đoạn một", "Đoạn hai")

        class _BlockingLLM:
            """Holds every call open until it is released, so the walk can be cancelled."""

            def __init__(self) -> None:
                self.calls = 0
                self.entered = asyncio.Event()
                self.release = asyncio.Event()

            async def invoke(self, request: GatewayInvokeRequest) -> InvokeResponse:
                self.calls += 1
                self.entered.set()
                await self.release.wait()
                return InvokeResponse(
                    message=GatewayChatTurn(role="assistant", content=json.dumps(BODY)),
                    finish_reason="stop",
                    usage={"prompt_tokens": 1200, "completion_tokens": 800},
                )

        llm = _BlockingLLM()
        seen: list[dict[str, Any]] = []
        task = asyncio.create_task(_refine(section, llm, on_spend=seen.append))
        await asyncio.wait_for(llm.entered.wait(), timeout=5)
        task.cancel()
        with pytest.raises(asyncio.CancelledError):
            await task

        assert llm.calls == 1  # the call that was in flight when the walk was taken down
        assert [spend["attempts"] for spend in seen] == [1]
        assert seen[0]["prompt_tokens"] is None  # an attempt that never answered
        assert seen[0]["usage_unknown_reason"] is not None
