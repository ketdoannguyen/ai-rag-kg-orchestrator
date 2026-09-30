"""Detail-only refinement seam for the enrich_knowledge_tree workflow.

`KnowledgeTreeEnricher` delegates its detail-merge step to `DetailRefiner`, and
offline research replays call `DetailRefiner.refine_details` directly. Both run
the same logic; they differ only in what the caller does with the result:

- Runtime applies it: the returned children replace the section's children, and
  `section_summary` may become `StructureNode.summary` when the section has no
  structure children (see `enricher._process_structure_node`).
- Research must not: `refine_details` never writes to the node, never invokes
  the structure-level summariser, and reports `section_summary` as data so a
  merge-produced summary cannot leak into a scored tree.

`RefineConfig.preserve_tables` is the research policy: a TableNode is a merge
boundary, so the paragraph runs between tables are merged separately and a
table is never rewritten or demoted into a paragraph. Off, the runtime default,
the whole detail list is one merge input, exactly as before.

Accounting: `DetailRefinement.llm_calls` counts completed invocations (0, or
one per merge the section ran), mirroring `EnrichStatistics.detail_llm_calls`.
Alongside it the refinement reports what the merge cost. `attempts` counts the
calls made at the gateway seam, retries included. The gateway retries on its own
behalf as well, and an answer that came back after one of those retries states
the counts of its last attempt alone: they become an unknown bill with a reason
rather than a subtotal priced as the section's. `prompt_tokens`,
`completion_tokens` and `cached_tokens` carry the counts the provider itself
stated, or `None` with `usage_unknown_reason` once one of its answers stated
nothing usable — or only half of its bill — because the counts of the answers
that did state them would then be a subtotal presented as the section's bill.
`serving_models` splits that bill over the models the answers named — one
`{"model", "usage"}` record per model, the shape pricing reads — so a merge is
priced against the model that served it rather than the one it requested.

A caller that has to keep the spend of a section it never got a result from —
one cancelled mid-walk — passes `on_spend` to `refine_details` and is handed
each merge call's counters as the call closes.
"""

from __future__ import annotations

import asyncio
import json
import time
from collections.abc import Callable, Mapping
from dataclasses import dataclass, field
from typing import Any

from arango_client import (
    ImageNode,
    KnowledgeNode,
    ParagraphNode,
    StructureNode,
    TableNode,
)
from llm_gateway_client import AIGatewayClient
from llm_gateway_client.models import GatewayChatTurn, GatewayInvokeRequest
from loguru import logger
from prompts import build_detail_merge_prompt
from tenacity import retry, retry_if_exception_type, stop_after_attempt, wait_exponential

from config.chunking import MAX_CHUNK_SIZE
from workflows.shared.file_utils import extract_json_from_response

from .helper import collect_detail_items, create_refined_paragraphs_from_segments, detail_runs
from .models import RefineConfig

_UNREPORTED_USAGE = "the gateway reported no token usage"
_PARTIAL_BILL = "an answer stated only part of its bill"
_UNANSWERED_ATTEMPT = "an attempt returned no response, so its bill is unobservable"
_GATEWAY_RETRY = (
    "the gateway retried before answering, so the bill of the attempt before it is unobservable"
)

_CACHE_KEYS = ("cached_tokens", "prompt_cache_hit_tokens")
"""Usage field names that carry a cached-prompt count, generic first."""


def _count(value: Any) -> int | None:
    """A non-negative integer count, or ``None`` when the value is unusable."""
    if isinstance(value, bool) or not isinstance(value, int) or value < 0:
        return None
    return value


def _gateway_attempts(response: Any) -> int:
    """How many calls the gateway itself made for one answered request.

    The gateway retries on its own behalf, so one call here can be several calls
    to the provider, and an answer that came back after a retry states the counts
    of its last attempt alone. A response that states no readable count is read
    as one call: that is the shape a gateway which answered first try returns.
    """
    attempts = getattr(response, "attempts", None)
    if isinstance(attempts, bool) or not isinstance(attempts, int) or attempts < 1:
        return 1
    return attempts


def _add(total: int | None, value: int | None) -> int | None:
    """Add one count to a running total, or ``None`` once a count is unknown.

    A total that skipped a count nobody stated would present a subtotal as the
    whole bill, so one unknown count makes the total unknown.
    """
    if total is None or value is None:
        return None
    return total + value


def _reported_counts(usage: Mapping[str, Any] | None) -> tuple[int | None, int | None]:
    """The prompt/completion counts a gateway's normalized usage view states, if it
    states any.

    The gateway normalizes a response that carried no usage into all-zero counts,
    and a merge prompt is never zero tokens, so zeroes across the board are how
    it says "not reported". Reading them as measured zeroes would price a call
    that may have been billed as free.
    """
    stated = usage or {}
    prompt = stated.get("prompt_tokens")
    completion = stated.get("completion_tokens")
    if not isinstance(prompt, int) or not isinstance(completion, int):
        return None, None
    if prompt == 0 and completion == 0:
        return None, None
    return prompt, completion


def _stated_counts(stated: Mapping[str, Any] | None) -> tuple[int | None, int | None]:
    """The prompt/completion counts the provider's own envelope stated.

    Nothing is defaulted: a count the envelope left out, or wrote as anything but
    a non-negative integer, is unknown rather than a zero the provider never
    claimed.
    """
    if not isinstance(stated, Mapping):
        return None, None
    return _count(stated.get("prompt_tokens")), _count(stated.get("completion_tokens"))


def _stated_cache(stated: Mapping[str, Any]) -> tuple[int | None, bool]:
    """The cached-prompt share one answer stated, and whether it mentioned one.

    An OpenAI-compatible endpoint states it under ``prompt_tokens_details``; some
    providers state it flat. The flag separates an answer that says nothing about
    the cache — an endpoint that does no cache accounting — from one that
    mentions a share it does not state, which is an unknown share rather than a
    zero.
    """
    details = stated.get("prompt_tokens_details")
    if isinstance(details, Mapping) and "cached_tokens" in details:
        return _count(details.get("cached_tokens")), True
    for key in _CACHE_KEYS:
        if key in stated:
            return _count(stated.get(key)), True
    return None, False


@dataclass
class _Usage:
    """What one or more gateway answers stated, and how completely they said it.

    The counts are all or nothing: an answer that stated no usable count leaves
    their total unknown, because the counts of the answers that did state one
    would be a subtotal presented as the whole bill. ``answers`` and ``mentions``
    are what makes silence readable — answers that never mention a cache share
    belong to an endpoint that does no cache accounting, which is a measured zero,
    while answers that disagree leave the share unknown.
    """

    prompt_tokens: int | None = 0
    completion_tokens: int | None = 0
    cached_tokens: int | None = 0
    answers: int = 0
    mentions: int = 0

    def add(
        self,
        prompt: int | None,
        completion: int | None,
        cached: int | None,
        mentioned: bool,
    ) -> None:
        """Fold one answered response's counts in.

        All or nothing per answer too: one that stated only half its bill leaves
        the bill unknown, because the half it did state would be a subtotal
        presented as the whole.
        """
        self.answers += 1
        self.mentions += 1 if mentioned else 0
        if prompt is None or completion is None:
            self.unobservable()
            return
        self.prompt_tokens = _add(self.prompt_tokens, prompt)
        self.completion_tokens = _add(self.completion_tokens, completion)
        if mentioned:
            self.cached_tokens = _add(self.cached_tokens, cached)

    def fold(self, other: _Usage) -> None:
        """Fold another set of answers in, under the same all-or-nothing rule.

        An answer that left the bill unobservable keeps it unobservable here: the
        counts of the answers that did state one would otherwise be a subtotal.
        """
        self.prompt_tokens = _add(self.prompt_tokens, other.prompt_tokens)
        self.completion_tokens = _add(self.completion_tokens, other.completion_tokens)
        self.cached_tokens = _add(self.cached_tokens, other.cached_tokens)
        self.answers += other.answers
        self.mentions += other.mentions

    def unobservable(self) -> None:
        """Mark the bill unknown: no answer stated it, so there is no total to read."""
        self.prompt_tokens = None
        self.completion_tokens = None
        self.cached_tokens = None

    def cache_share(self) -> int | None:
        """The cache share these answers add up to, or ``None`` when it is unknown.

        An unknown bill has no share to read off it; answers that never mention a
        cache share are an endpoint that does no cache accounting, which is a
        measured zero; and answers that disagree — some mention a share, some do
        not — leave it unknown, because the sum of the ones that spoke would be a
        subtotal presented as the share.
        """
        if self.prompt_tokens is None or self.completion_tokens is None:
            return None
        if self.answers == 0 or self.mentions == 0:
            return 0
        if self.mentions < self.answers:
            return None
        return self.cached_tokens

    def record(self) -> dict[str, int | None]:
        """The usage record pricing reads, cache share included."""
        return {
            "prompt_tokens": self.prompt_tokens,
            "completion_tokens": self.completion_tokens,
            "cached_tokens": self.cache_share(),
        }


@dataclass
class _MergeSpend:
    """What one section's merge cost, observed at the gateway seam.

    ``attempts`` counts every call the section made here, so the retries behind one
    logical invocation stay visible; it is not the number of requests the provider
    answered, because the gateway retries on its own behalf too. The token counts
    stay at a measured zero while nothing has been called and go unknown as soon as
    a call goes unanswered: it leaves what the provider billed unobservable, and
    the counts of the calls that did answer would then be a subtotal presented as
    the section's bill.

    ``usage`` is the section's whole bill; ``by_model`` splits it over the models
    the answers named, so a merge can be priced per model instead of written off
    as unattributed. An answer that named no model is credited to ``None`` rather
    than dropped, so the part no model can be priced for stays visible in the
    split as the record nothing could be charged to.
    """

    attempts: int = 0
    usage: _Usage = field(default_factory=_Usage)
    by_model: dict[str | None, _Usage] = field(default_factory=dict)
    usage_unknown_reason: str | None = None
    latency_seconds: float = 0.0

    def enter(self) -> None:
        """Count one call at the seam, retries included."""
        self.attempts += 1

    def seal(self) -> None:
        """Close the meter, going unknown for a call that never stated its bill.

        An attempt that raised, or an answer that only came after an earlier dead
        attempt, leaves what the provider billed unobservable; the counts of the
        calls that did answer would be a subtotal presented as the section's bill.
        """
        if self.attempts > 0 and self.usage.answers == 0:
            self.usage.unobservable()
            if self.usage_unknown_reason is None:
                self.usage_unknown_reason = _UNANSWERED_ATTEMPT

    def record(self, response: Any) -> None:
        """Take the counts one answered response stated, when they are the whole bill."""
        if self.attempts > 1:
            # An earlier attempt did not answer, so whether it was billed cannot
            # be observed and these counts cover only part of the section's bill.
            return
        if _gateway_attempts(response) > 1:
            # The gateway retried before this answer: the attempts before it may
            # have been billed too, and nothing in the answer states them, so
            # these counts are a subtotal of the section's bill.
            self.usage.unobservable()
            self.usage_unknown_reason = _GATEWAY_RETRY
            return
        stated = getattr(response, "stated_usage", None)
        if isinstance(stated, Mapping):
            prompt, completion = _stated_counts(stated)
            cached, mentioned = _stated_cache(stated)
        else:
            # No envelope of the provider's own: the normalized view is all this
            # answer carries, and it cannot tell an absent count from a zero.
            prompt, completion = _reported_counts(getattr(response, "usage", None))
            cached, mentioned = None, False
        self.usage.add(prompt, completion, cached, mentioned)
        if prompt is None or completion is None:
            # Half a bill is no bill: the count that was stated would be a
            # subtotal presented as the section's own, and the reason has to say
            # so rather than leave ``None`` counts unexplained.
            self.usage_unknown_reason = (
                _UNREPORTED_USAGE if prompt is None and completion is None else _PARTIAL_BILL
            )
        self._credit(getattr(response, "model", None), prompt, completion, cached, mentioned)

    def _credit(
        self,
        model: Any,
        prompt: int | None,
        completion: int | None,
        cached: int | None,
        mentioned: bool,
    ) -> None:
        """Credit one answer's counts to the model it named, if it named one.

        An answer that named no model is kept under ``None``: a name nobody stated
        cannot be priced, and inventing one from the model the request asked for
        would price the bill against a name that is not evidence of who served it.
        Keeping the record makes the tokens nothing could be charged for a visible
        part of the split instead of a sum that quietly leaves them out.
        """
        if not isinstance(model, str) or not model:
            model = None
        self.by_model.setdefault(model, _Usage()).add(prompt, completion, cached, mentioned)

    def serving_models(self) -> list[dict[str, Any]]:
        """The section's bill split over the models the answers named.

        One record per model, the shape pricing reads, so the caller prices the
        merge against the models that served it instead of the one it requested.
        Reported only when the whole bill is observable: the records of the calls
        that answered would otherwise price a subtotal as the section's cost.
        """
        if self.usage.prompt_tokens is None or self.usage.completion_tokens is None:
            return []
        return [
            {"model": model, "usage": usage.record()} for model, usage in self.by_model.items()
        ]

    def merge(self, other: _MergeSpend) -> None:
        """Fold another group's spend into this section's total.

        A group whose bill is unobservable makes the section's total
        unobservable too: the counts of the groups that did report would
        otherwise be a subtotal presented as the whole.
        """
        self.attempts += other.attempts
        self.latency_seconds += other.latency_seconds
        if other.attempts == 0:
            return
        self.usage.fold(other.usage)
        for model, usage in other.by_model.items():
            self.by_model.setdefault(model, _Usage()).fold(usage)
        if self.usage_unknown_reason is None:
            self.usage_unknown_reason = other.usage_unknown_reason

    def counters(self) -> dict[str, Any]:
        """This spend as the counters a caller folds, one call's bill at a time.

        The keys are the ones a refinement reports — ``attempts``, the token
        counts, ``usage_unknown_reason``, ``latency_seconds`` — so a caller that
        is handed a call as it closes folds it by the same rule it folds a
        section's counters.
        """
        return {
            "attempts": self.attempts,
            "prompt_tokens": self.usage.prompt_tokens,
            "completion_tokens": self.usage.completion_tokens,
            "usage_unknown_reason": self.usage_unknown_reason,
            "latency_seconds": self.latency_seconds,
        }


@dataclass
class DetailRefinement:
    """Outcome of refining one section's detail children.

    `children` keeps the runtime child order: refined details + images +
    structure children. `section_summary` is reported as data only — the caller
    decides whether it belongs on the node. `merge_skipped` is True when the
    merge never ran (detail merging disabled, fewer than two detail children, or
    no textual detail content); a merge whose LLM call failed ran, so it is not
    reported as skipped. `merge_fell_back` is True when a merge that ran kept
    the original details — no call completed, or the answer named nothing the
    refiner could refine — so a fallen-back merge is not mistaken for one that
    refined a section.

    The accounting fields describe what the merge cost. A skipped merge made no
    call, so it reports zero attempts and a measured zero of tokens;
    `usage_unknown_reason` says why the counts are `None` when they are, and is
    `None` whenever they are a real total. `latency_seconds` is the wall time of
    the logical invocation, excluding time spent waiting for the semaphore.
    `cached_tokens` is the cached-prompt share the provider stated, a measured
    zero when no answer mentioned one, and `None` when the answers disagreed.
    `serving_models` splits the same bill over the models the answers named, one
    `{"model": str | None, "usage": {...}}` record each, so
    `pricing.cost_for_attributed_usage` prices it directly; a record whose model
    is `None` is what no answer named, and it is reported — like every record —
    only when the split is the whole bill.
    """

    children: list[KnowledgeNode]
    section_summary: str | None
    original_detail_count: int
    refined_detail_count: int
    merge_skipped: bool
    merge_fell_back: bool
    llm_calls: int
    attempts: int = 0
    prompt_tokens: int | None = 0
    completion_tokens: int | None = 0
    cached_tokens: int | None = 0
    serving_models: list[dict[str, Any]] = field(default_factory=list)
    latency_seconds: float = 0.0
    usage_unknown_reason: str | None = None


class DetailRefiner:
    """Merges a section's Paragraph/Table children into retrieval-sized chunks."""

    def __init__(self, llm_client: AIGatewayClient) -> None:
        self.llm_client = llm_client

    # ---------------------------------------------------------------------
    # Low-level LLM invocation with retry
    # ---------------------------------------------------------------------
    @retry(
        retry=retry_if_exception_type(Exception),
        stop=stop_after_attempt(3),
        wait=wait_exponential(multiplier=1, min=1, max=8),
        reraise=True,
    )
    async def _invoke_with_retry(self, request: GatewayInvokeRequest, spend: _MergeSpend):
        """Invoke LLM with simple exponential backoff retry, billing every call.

        tenacity re-enters this body for each retry with the same arguments, so
        the meter sees a retried invocation as the calls it cost while the caller
        still reports one logical call.
        """
        spend.enter()
        response = await self.llm_client.invoke(request)
        spend.record(response)
        return response

    async def _invoke_timed(
        self,
        request: GatewayInvokeRequest,
        spend: _MergeSpend,
        on_spend: Callable[[dict[str, Any]], None] | None = None,
    ):
        """Run the retried invocation, time it, and hand its counters over.

        Timed here rather than around the semaphore so the latency is the call's
        own, not the queue it waited in. ``on_spend`` is handed the call's
        counters as it closes — answered, failed or cancelled — so the spend of
        the calls a section already made outlives the walk that made them.
        """
        started = time.perf_counter()
        try:
            return await self._invoke_with_retry(request, spend)
        finally:
            spend.latency_seconds += time.perf_counter() - started
            spend.seal()
            if on_spend is not None:
                on_spend(spend.counters())

    # ---------------------------------------------------------------------
    # Detail refinement
    # ---------------------------------------------------------------------
    async def refine_details(
        self,
        node: StructureNode,
        breadcrumb: list[str],
        refine_config: RefineConfig,
        semaphore: asyncio.Semaphore | None = None,
        on_spend: Callable[[dict[str, Any]], None] | None = None,
    ) -> DetailRefinement:
        """Refine the DETAIL children of a StructureNode with the merges its groups call for.

        One merge runs over every detail, unless `preserve_tables` makes the
        tables boundaries and the paragraph runs between them the inputs.

        This is the research entry point: it only refines details. It never calls
        the structure-level summariser and never writes to `node`, so the merge
        summary it returns cannot leak into a scored tree.

        ``on_spend`` is handed each merge call's counters as the call closes, in
        the order the section made them. It is how a caller keeps the spend of a
        section it never got a result from: a section that is cancelled or taken
        down mid-walk returns nothing, so the calls that did answer would
        otherwise be dropped with it.

        Returns:
            DetailRefinement with the rebuilt children, the merge summary as
            data, and the counters the caller reports on.
        """
        detail_nodes: list[KnowledgeNode] = []
        structure_children: list[KnowledgeNode] = []
        image_children: list[KnowledgeNode] = []

        for child in node.children:
            if isinstance(child, (ParagraphNode, TableNode)):
                detail_nodes.append(child)
            elif isinstance(child, StructureNode):
                structure_children.append(child)
            elif isinstance(child, ImageNode):
                image_children.append(child)
            else:
                # Preserve unknown nodes as-is (defensive)
                structure_children.append(child)

        original_detail_count = len(detail_nodes)

        if not refine_config.refine_details:
            # The producer already grouped its blocks into retrieval-sized
            # chunks, so re-chunking here would only let the LLM rewrite text
            # that should stay verbatim.
            return DetailRefinement(
                children=list(node.children),
                section_summary=None,
                original_detail_count=original_detail_count,
                refined_detail_count=original_detail_count,
                merge_skipped=True,
                merge_fell_back=False,
                llm_calls=0,
            )

        def skipped() -> DetailRefinement:
            return DetailRefinement(
                children=[*detail_nodes, *image_children, *structure_children],
                section_summary=None,
                original_detail_count=original_detail_count,
                refined_detail_count=original_detail_count,
                merge_skipped=True,
                merge_fell_back=False,
                llm_calls=0,
            )

        if original_detail_count <= 1:
            return skipped()

        # Case: >=2 detail nodes → call LLM to merge/refine
        logger.info(
            "Refining detail children of section",
            extra={
                "section_depth": len(breadcrumb),
                "detail_count": len(detail_nodes),
                "structure_children": len(structure_children),
            },
        )

        # Build text items for LLM
        collected_items = collect_detail_items(
            detail_nodes,
            preserve_tables=refine_config.preserve_tables,
        )

        if not collected_items:
            # Nothing textual to refine
            return skipped()

        runs = detail_runs(
            collected_items,
            preserve_tables=refine_config.preserve_tables,
        )

        if refine_config.preserve_tables and all(len(run) < 2 for run in runs):
            # Every detail is alone in its group, so no merge has anything to
            # combine and no call is made.
            return skipped()

        refined_children: list[KnowledgeNode] = []
        section_summary: str | None = None
        llm_calls = 0
        merge_fell_back = False
        spend = _MergeSpend()

        for run in runs:
            if refine_config.preserve_tables and len(run) < 2:
                # A table, or a paragraph run of one, is already at retrieval
                # size: it is kept as the node it already is.
                refined_children.extend(item["node"] for item in run)
                continue

            detail_result = await self._invoke_detail_llm(
                items=run,
                breadcrumb=breadcrumb,
                refine_config=refine_config,
                semaphore=semaphore,
                on_spend=on_spend,
            )
            llm_calls += detail_result.get("llm_calls", 0)
            section_summary = section_summary or detail_result.get("section_summary")
            spend.merge(detail_result["spend"])

            originals = {id(item["node"]) for item in run}
            merged = create_refined_paragraphs_from_segments(
                segments=detail_result.get("segments", []),
                items=run,
                parent=node,
                excluded_indices=detail_result.get("excluded_indices", []),
                preserve_details=refine_config.preserve_tables,
                include_summaries=refine_config.segment_summaries,
            )
            if not any(id(child) not in originals for child in merged):
                # Every node came back unchanged: the refiner kept the originals
                # instead of refining them.
                merge_fell_back = True
            refined_children.extend(merged)

        # Rebuild children: refined paragraphs + images + structure children
        return DetailRefinement(
            children=[
                *refined_children,
                *image_children,
                *structure_children,
            ],
            section_summary=section_summary,
            original_detail_count=original_detail_count,
            refined_detail_count=sum(
                1 for child in refined_children if isinstance(child, (ParagraphNode, TableNode))
            ),
            merge_skipped=False,
            merge_fell_back=merge_fell_back,
            llm_calls=llm_calls,
            attempts=spend.attempts,
            prompt_tokens=spend.usage.prompt_tokens,
            completion_tokens=spend.usage.completion_tokens,
            cached_tokens=spend.usage.cache_share(),
            serving_models=spend.serving_models(),
            latency_seconds=spend.latency_seconds,
            usage_unknown_reason=spend.usage_unknown_reason,
        )

    async def _invoke_detail_llm(
        self,
        items: list[dict[str, Any]],
        breadcrumb: list[str],
        refine_config: RefineConfig,
        semaphore: asyncio.Semaphore | None = None,
        on_spend: Callable[[dict[str, Any]], None] | None = None,
    ) -> dict[str, Any]:
        """Call LLM to merge/refine detail items.

        Returns dict with:
            - segments: list of segment dicts
            - section_summary: optional summary string for the whole section
            - excluded_indices: list of {"index", "reason"} dicts
            - llm_calls: 1 when the invocation completed (retries included), else 0
            - spend: the `_MergeSpend` every exit path carries, so a merge that
              failed or answered unusably still reports what it cost

        ``on_spend`` is handed this call's counters as it closes, whether it
        answered, failed or was cancelled.
        """
        prompt = build_detail_merge_prompt(
            items,
            breadcrumb,
            max_chunk_size=MAX_CHUNK_SIZE,
        )
        # Single combined prompt used as system message
        request = GatewayInvokeRequest(
            model=refine_config.llm_model,
            messages=[
                GatewayChatTurn(
                    role="user",
                    content=prompt,
                )
            ],
            provider_options={"temperature": refine_config.llm_temperature},
        )

        spend = _MergeSpend()

        try:
            if semaphore:
                async with semaphore:
                    response = await self._invoke_timed(request, spend, on_spend)
            else:
                response = await self._invoke_timed(request, spend, on_spend)
        except Exception as e:
            logger.error(
                "Detail-level LLM call failed; keeping original detail nodes",
                extra={"error_type": type(e).__name__},
            )
            return {"segments": [], "section_summary": None, "llm_calls": 0, "spend": spend}

        try:
            content = response.message.content
            parsed = extract_json_from_response(content)
            data = json.loads(parsed)
            segments = data.get("segments", [])
            if not isinstance(segments, list):
                logger.error("Invalid segments format in detail LLM response")
                return {"segments": [], "section_summary": None, "llm_calls": 1, "spend": spend}
            section_summary = (data.get("section_summary") or "").strip() or None
            excluded_indices = data.get("excluded_indices", [])
            if not isinstance(excluded_indices, list):
                excluded_indices = []
            return {
                "segments": segments,
                "section_summary": section_summary,
                "excluded_indices": excluded_indices,
                "llm_calls": 1,
                "spend": spend,
            }
        except Exception as e:
            logger.error(
                "Failed to parse detail LLM response; keeping original detail nodes",
                extra={"error_type": type(e).__name__},
            )
            return {"segments": [], "section_summary": None, "llm_calls": 1, "spend": spend}
