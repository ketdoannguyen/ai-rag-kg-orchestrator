"""An offline double for the gateway client that answers the merge prompt.

A tree is merged by calling a gateway client with the detail-merge prompt; this
double answers that call deterministically, by reading the numbered items out of
the prompt and grouping them into segments. It is not a model. A caller who
injects it is testing the merge *wiring* -- prompt, parse, apply, counters -- and
the tree it merges says nothing about quality, which is why a run that merged
through it is not a measurement of the merge.

It lives beside the tests because production has no use for it: the study's own
``_gateway_client`` builds the real client, and only a test needs a stand-in.
"""

from __future__ import annotations

import json
import re
from collections.abc import Sequence
from typing import Any

__all__ = ["FAKE_SEGMENT_CHARS", "FakeMergeClient"]

FAKE_SEGMENT_CHARS = 1000
"""The segment budget the double groups items under, small enough to split a section."""

_ITEM_LINE = re.compile(r"^\[(\d+)\] \(([^)]*)\) (.*)$")


def _parse_prompt_items(prompt: str) -> list[tuple[int, str, str]]:
    """Read the numbered items ``build_detail_merge_prompt`` ends with.

    Returns ``(index, type, text)`` in the order the prompt lists them. A text
    that spans lines keeps them: everything up to the next numbered item is part
    of that item.
    """
    items: list[tuple[int, str, str]] = []
    current: tuple[int, str, list[str]] | None = None
    for line in prompt.splitlines():
        match = _ITEM_LINE.match(line)
        if match is not None:
            if current is not None:
                items.append((current[0], current[1], "\n".join(current[2]).strip()))
            current = (int(match.group(1)), match.group(2).strip(), [match.group(3)])
        elif current is not None:
            current[2].append(line)
    if current is not None:
        items.append((current[0], current[1], "\n".join(current[2]).strip()))
    return items


def _greedy_segments(
    items: Sequence[tuple[int, str, str]], max_chars: int
) -> list[dict[str, Any]]:
    """Group consecutive items into segments no longer than ``max_chars``."""
    segments: list[dict[str, Any]] = []
    current: list[tuple[int, str, str]] = []
    length = 0
    for item in items:
        if current and length + len(item[2]) > max_chars:
            segments.append(_segment(current))
            current, length = [], 0
        current.append(item)
        length += len(item[2])
    if current:
        segments.append(_segment(current))
    return segments


def _segment(items: Sequence[tuple[int, str, str]]) -> dict[str, Any]:
    return {
        "source_indices": [item[0] for item in items],
        "text": "\n".join(item[2] for item in items),
        "summary": None,
    }


class FakeMergeClient:
    """A deterministic stand-in for the gateway that answers the merge prompt."""

    def __init__(self, *, max_segment_chars: int = FAKE_SEGMENT_CHARS) -> None:
        self.max_segment_chars = max_segment_chars
        self.calls = 0
        self.prompts: list[str] = []

    async def invoke(self, request: Any) -> Any:
        from llm_gateway_client.models import GatewayChatTurn, InvokeResponse

        self.calls += 1
        prompt = request.messages[0].content
        self.prompts.append(prompt)
        items = _parse_prompt_items(prompt)
        body = {
            "segments": _greedy_segments(items, self.max_segment_chars),
            "section_summary": None,
        }
        return InvokeResponse(
            message=GatewayChatTurn(role="assistant", content=json.dumps(body)),
            finish_reason="stop",
            usage={},
        )
