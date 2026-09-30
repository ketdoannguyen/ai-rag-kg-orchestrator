"""Focused tests for the concurrent merge.

`merge_tree` walks a tree bottom-up and calls the model once per section. The
walk has an order -- a section merges after the sections below it -- and a bound:
the semaphore at the gateway seam lets at most N calls be in flight at once.
Neither may depend on which section happens to answer first, and a walk that
stops must take its siblings down with it instead of leaving them calling.

The doubles here answer with a fixed segment list and state a fixed bill, so
every number below is the wiring's and not a model's. A tree a fake client built
says nothing about quality, which is why a fake merge leaves its run unrankable
(see `FakeMergeClient`); nothing here assesses a tree's quality.
"""

from __future__ import annotations

import asyncio
import json
import re
from collections.abc import Mapping, Sequence
from typing import Any

import pytest
from arango_client import ParagraphNode, StructureNode
from llm_gateway_client.models import GatewayChatTurn, InvokeResponse

from research.ocr_pipeline.runner import MergeAbortedError, merge_tree
from workflows.enrich_knowledge_tree.models import RefineConfig

MERGE = RefineConfig(refine_details=True, llm_model="a-double-not-a-model", preserve_tables=True)
BILL = {"prompt_tokens": 10, "completion_tokens": 5}
CONTEXT = re.compile(r"^Context: (.*)$", re.MULTILINE)
TIMINGS = ("latency_seconds", "wall_seconds")


# --------------------------------------------------------------------------- #
# trees and doubles
# --------------------------------------------------------------------------- #


def detail_section(
    title: str, *, paragraphs: int = 2, children: Sequence[StructureNode] = ()
) -> StructureNode:
    """A section with paragraphs of its own, so its merge has something to do."""
    return StructureNode(
        attributes={"title": title},
        children=[
            *(
                ParagraphNode(attributes={"text": f"{title} sentence {index}"})
                for index in range(paragraphs)
            ),
            *children,
        ],
    )


def flat_tree(*, sections: int = 4) -> StructureNode:
    """A root whose children are all leaves: every leaf merges, the root does not."""
    return StructureNode(
        attributes={"title": "Root"},
        children=[detail_section(f"S{index}") for index in range(sections)],
    )


def nested_tree() -> StructureNode:
    """Two levels, each parent holding paragraphs and child sections of its own."""
    return StructureNode(
        attributes={"title": "Root"},
        children=[
            detail_section(
                f"P{parent}",
                children=[detail_section(f"C{parent}-{child}") for child in range(2)],
            )
            for parent in range(2)
        ],
    )


def answer() -> InvokeResponse:
    """One merge answer: both items in one segment, in a stated bill."""
    body = {"segments": [{"source_indices": [1, 2], "text": "merged"}], "section_summary": None}
    return InvokeResponse(
        message=GatewayChatTurn(role="assistant", content=json.dumps(body)),
        finish_reason="stop",
        usage=dict(BILL),
    )


class GatedMergeClient:
    """A merge double that reports the overlap it was actually asked to serve.

    ``gate_at`` calls are held open at once: the gate opens when that many are in
    flight and they all proceed together, so a test observes a real overlap
    rather than one that only looked overlapped. A call whose context is in
    ``hold`` stays parked until the test releases it, and every wait has a timeout,
    so a walk that never gets there fails instead of hanging the suite.
    """

    def __init__(
        self,
        *,
        gate_at: int = 1,
        timeout: float = 5.0,
        hold: Sequence[str] = (),
        delays: Mapping[str, float] | None = None,
    ) -> None:
        self.gate_at = gate_at
        self.timeout = timeout
        self.hold = tuple(hold)
        self.delays = dict(delays or {})
        self.in_flight = 0
        self.max_in_flight = 0
        self.calls = 0
        self.completed: list[str] = []
        self.contexts: list[str] = []
        self.opened = asyncio.Event()
        self._gate = asyncio.Event()
        self._hold = asyncio.Event()

    def release(self) -> None:
        self._hold.set()

    async def invoke(self, request: Any) -> Any:
        self.calls += 1
        self.in_flight += 1
        self.max_in_flight = max(self.max_in_flight, self.in_flight)
        try:
            match = CONTEXT.search(request.messages[0].content)
            context = match.group(1).strip() if match else ""
            self.contexts.append(context)
            if self.in_flight >= self.gate_at:
                self.opened.set()
                self._gate.set()
            if any(held in context for held in self.hold):
                await asyncio.wait_for(self._hold.wait(), timeout=self.timeout)
            else:
                await asyncio.wait_for(self._gate.wait(), timeout=self.timeout)
            delay = self.delays.get(context, 0.0)
            if delay:
                await asyncio.sleep(delay)
            self.completed.append(context)
        finally:
            self.in_flight -= 1
        return answer()


def rendered(node: StructureNode) -> Any:
    """A tree as plain data, so two walks can be compared without their timings."""
    return [
        (child.attributes.title, rendered(child))
        if isinstance(child, StructureNode)
        else (child.attributes.text, [])
        for child in node.children
    ]


def counted(counters: Mapping[str, Any]) -> dict[str, Any]:
    """The counters that are not wall time: what a walk has to reproduce exactly."""
    return {name: value for name, value in counters.items() if name not in TIMINGS}


# --------------------------------------------------------------------------- #
# the bound
# --------------------------------------------------------------------------- #


async def test_sibling_sections_merge_at_the_same_time_up_to_the_limit() -> None:
    client = GatedMergeClient(gate_at=3)

    task = asyncio.create_task(
        merge_tree(
            flat_tree(sections=6),
            llm_client=client,
            refine_config=MERGE,
            max_concurrent_sections=3,
        )
    )
    try:
        await asyncio.wait_for(client.opened.wait(), timeout=2.0)
    finally:
        client.release()
    counters = await task

    # Three calls served at once, and never the six this tree could offer: the
    # limit is what bounds the walk, not the shape of the tree.
    assert client.max_in_flight == 3
    assert counters["sections"] == 7
    assert counters["sections_merged"] == 6
    assert counters["llm_calls"] == client.calls == 6


async def test_one_slot_merges_one_section_at_a_time_to_the_end() -> None:
    client = GatedMergeClient(gate_at=1)

    counters = await merge_tree(
        flat_tree(sections=4), llm_client=client, refine_config=MERGE, max_concurrent_sections=1
    )

    assert client.max_in_flight == 1
    assert counters["llm_calls"] == client.calls == 4
    # A single slot is a queue, not a stop: the walk finished.
    assert counters["sections"] == 5
    assert counters["usage_unknown_reason"] is None


async def test_the_limit_is_never_exceeded_when_the_tree_offers_more_sections() -> None:
    client = GatedMergeClient(gate_at=2)

    counters = await merge_tree(
        flat_tree(sections=5), llm_client=client, refine_config=MERGE, max_concurrent_sections=2
    )

    assert client.max_in_flight == 2
    assert counters["sections_merged"] == 5


# --------------------------------------------------------------------------- #
# the order
# --------------------------------------------------------------------------- #


async def test_a_parent_merges_only_after_the_sections_below_it_have() -> None:
    client = GatedMergeClient()

    await merge_tree(
        nested_tree(), llm_client=client, refine_config=MERGE, max_concurrent_sections=4
    )

    # Each parent's prompt names the parent; its child's prompt has to come first,
    # because the merge it runs reads the children its subtrees already rebuilt.
    for parent in range(2):
        for child in range(2):
            parent_call = client.contexts.index(f"Root > P{parent}")
            child_call = client.contexts.index(f"Root > P{parent} > C{parent}-{child}")
            assert child_call < parent_call


async def test_the_counters_and_the_tree_do_not_depend_on_who_answers_first() -> None:
    # Every call takes time, so the summed logical latency is real; one section
    # takes longer, so a walk that folded in completion order would fold it last
    # while a walk that folds in document order folds it fourth either way.
    delays = {f"Root > S{index}": 0.03 for index in range(4)} | {"Root > S0": 0.06}

    serial_client = GatedMergeClient(gate_at=1)
    serial_tree = flat_tree(sections=4)
    serial = await merge_tree(
        serial_tree, llm_client=serial_client, refine_config=MERGE, max_concurrent_sections=1
    )

    overlap_client = GatedMergeClient(gate_at=4, delays=delays)
    overlap_tree = flat_tree(sections=4)
    overlap = await merge_tree(
        overlap_tree, llm_client=overlap_client, refine_config=MERGE, max_concurrent_sections=4
    )

    assert overlap_client.max_in_flight == 4
    assert counted(overlap) == counted(serial)
    assert rendered(overlap_tree) == rendered(serial_tree)
    # The same four calls cost the same four latencies; only the wall time moved.
    assert overlap["latency_seconds"] > 3 * 0.03
    assert overlap["wall_seconds"] < overlap["latency_seconds"]


# --------------------------------------------------------------------------- #
# the cost of stopping
# --------------------------------------------------------------------------- #


async def test_a_cancelled_merge_leaves_no_call_running() -> None:
    client = GatedMergeClient(gate_at=4, hold=("Root > S",))
    before = set(asyncio.all_tasks())

    task = asyncio.create_task(
        merge_tree(
            flat_tree(sections=4),
            llm_client=client,
            refine_config=MERGE,
            max_concurrent_sections=4,
        )
    )
    await asyncio.wait_for(client.opened.wait(), timeout=2.0)
    assert client.in_flight == 4

    task.cancel()
    with pytest.raises(asyncio.CancelledError):
        await task

    # The four held calls were cancelled and awaited, not left calling behind the
    # walk that made them.
    assert client.in_flight == 0
    assert client.completed == []
    assert {task for task in asyncio.all_tasks() if task is not asyncio.current_task()} <= before


async def test_a_section_that_raises_takes_the_walk_down_and_keeps_what_it_spent(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from workflows.enrich_knowledge_tree.detail_refiner import DetailRefiner

    # The failing section is the parent, so the two sections below it have merged
    # -- and been billed -- before it is even called: the spend the error carries
    # is what the document order guarantees, not a scheduling accident.
    tree = StructureNode(
        attributes={"title": "Root"},
        children=[detail_section("P", children=[detail_section("C0"), detail_section("C1")])],
    )
    client = GatedMergeClient()
    real = DetailRefiner.refine_details

    async def flaky(self: Any, **kwargs: Any) -> Any:
        refinement = await real(self, **kwargs)
        if kwargs["breadcrumb"] == ["Root", "P"]:
            raise RuntimeError("boom")
        return refinement

    monkeypatch.setattr(DetailRefiner, "refine_details", flaky)

    with pytest.raises(MergeAbortedError) as failure:
        await merge_tree(tree, llm_client=client, refine_config=MERGE, max_concurrent_sections=1)

    # The walk stopped and said so: no tree came back pretending to be merged.
    assert isinstance(failure.value.cause, RuntimeError)
    assert client.calls == 3  # the failing section called before it failed
    # The two sections that finished were billed, and their bill rides the error
    # rather than dying with the walk. The failing section's own call is not in
    # there: its refinement never returned, so its bill is unobservable, and a
    # subtotal presented as the whole is the one thing the counters never do.
    assert failure.value.counters["llm_calls"] == 2
    assert failure.value.counters["prompt_tokens"] == 2 * BILL["prompt_tokens"]
    assert failure.value.counters["usage_unknown_reason"] is None
    assert failure.value.counters["sections"] == 2


async def test_the_walk_reports_no_bill_when_nothing_was_called() -> None:
    client = GatedMergeClient()

    # Every section is a single paragraph: there is nothing to merge and no call.
    tree = StructureNode(
        attributes={"title": "Root"},
        children=[detail_section(f"S{index}", paragraphs=1) for index in range(3)],
    )
    counters = await merge_tree(
        tree, llm_client=client, refine_config=MERGE, max_concurrent_sections=4
    )

    assert client.calls == 0
    assert counters["sections_skipped"] == counters["sections"] == 4
    assert counters["llm_calls"] == counters["attempts"] == 0
    assert counters["usage_unknown_reason"] is None
    assert counters["wall_seconds"] >= 0.0


async def test_the_merge_reports_its_own_wall_time_apart_from_the_call_latency() -> None:
    delays = {f"Root > S{index}": 0.03 for index in range(4)}
    client = GatedMergeClient(gate_at=4, delays=delays)

    counters = await merge_tree(
        flat_tree(sections=4), llm_client=client, refine_config=MERGE, max_concurrent_sections=4
    )

    # Four calls, each of which took time: the summed logical latency is what
    # grows with the work, and the wall time is what says they overlapped.
    assert counters["latency_seconds"] > 3 * 0.03
    assert counters["wall_seconds"] < counters["latency_seconds"]

    serial_client = GatedMergeClient(gate_at=1, delays=delays)
    serial = await merge_tree(
        flat_tree(sections=4),
        llm_client=serial_client,
        refine_config=MERGE,
        max_concurrent_sections=1,
    )

    # One slot: nothing overlaps, so the wall time is the sum, not less than it.
    assert serial["wall_seconds"] >= serial["latency_seconds"]
