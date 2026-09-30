"""Render one OCR research run as JSON, CSV and Markdown.

``build_report`` assembles the computed metric dicts (from
:mod:`research.ocr_pipeline.metrics` / :mod:`research.ocr_pipeline.usage` or any
other producer) plus the study sections (metadata, per-stage timings,
resource observations, fidelity profiles, limitations) into one canonical
report; ``to_json``, ``to_csv`` and ``to_markdown`` render that same dict, so
every format carries identical ``counts``. All three renderers walk the report
with the same code path, which is what keeps the numbers in sync.

Raw transcription never enters a report — not even the document *title*, which
is source text under an innocent name. ``build_report`` and the renderers
reject payloads whose keys look like page text and whose keys or strings are
long enough to be prose (see :func:`find_raw_text_keys`); only numbers,
statuses, ids and short metadata strings survive.

Report shape::

    {"run": {"run_id": ..., "engine": ..., ...},   # whitelisted scalars only
     "counts": {...},                              # the shared count block
     "metadata": {...}, "stages": {...},
     "resources": {...}, "fidelity": {...},
     "text_metrics": {...}, "chunk_metrics": {...},
     "usage": {...}, "paired_usage": {...},        # input/output tokens only
     "limitations": [...]}                         # what the numbers cannot say
"""

from __future__ import annotations

import csv
import io
import json
from collections.abc import Sequence
from typing import Any

from .metrics import MEASURED, NOT_MEASURED

RUN_FIELDS = ("run_id", "engine", "model", "created_at", "label")

RAW_TEXT_KEYS = (
    "text",
    "reference",
    "hypothesis",
    "content",
    "raw",
    "transcript",
    "transcription",
    "ocr_text",
    "gold",
    "page_text",
    "full_text",
    "title",
)

MAX_STRING_LENGTH = 512

CSV_COLUMNS = (
    "section",
    "metric",
    "status",
    "value",
    "numerator",
    "denominator",
    "unit",
    "denominator_unit",
    "reason",
)

METRIC_FIELDS = frozenset(CSV_COLUMNS)

__all__ = [
    "MAX_STRING_LENGTH",
    "RAW_TEXT_KEYS",
    "build_report",
    "find_raw_text_keys",
    "to_csv",
    "to_json",
    "to_markdown",
]


def find_raw_text_keys(payload: Any, *, max_string_length: int = MAX_STRING_LENGTH) -> list[str]:
    """Paths of payload entries that must not reach a report.

    Flags three things: keys named like transcription (``reference``,
    ``hypothesis``, ``text``, ``title``, ...) at any depth *when they hold a
    string*; string values longer than ``max_string_length`` characters; and
    *keys* longer than it — ids, models and statuses are short, so a long string
    or key is page text under an innocent name. A count block that happens to be
    named ``hypothesis`` holds only numbers and is not an offender.

    This is a fail-closed heuristic, not a proof: a short unlabelled snippet
    under an innocent key could still slip through.
    """
    found: list[str] = []
    _scan(payload, (), found, max_string_length)
    return found


def _scan(node: Any, path: tuple[str, ...], found: list[str], max_string_length: int) -> None:
    if isinstance(node, dict):
        for key, value in node.items():
            name = str(key)
            child = (*path, name)
            flagged = len(name) > max_string_length or (
                key in RAW_TEXT_KEYS and _holds_text(value)
            )
            if flagged:
                found.append(".".join(child))
            else:
                _scan(value, child, found, max_string_length)
    elif isinstance(node, (list, tuple)):
        for index, item in enumerate(node):
            _scan(item, (*path, str(index)), found, max_string_length)
    elif isinstance(node, str) and len(node) > max_string_length:
        found.append(".".join(path))


def _holds_text(node: Any) -> bool:
    """Whether a subtree holds a string: a count block holds only numbers."""
    if isinstance(node, str):
        return True
    if isinstance(node, dict):
        return any(_holds_text(value) for value in node.values())
    if isinstance(node, (list, tuple)):
        return any(_holds_text(item) for item in node)
    return False


def build_report(
    run: dict[str, Any] | None = None,
    *,
    metadata: dict[str, Any] | None = None,
    stages: dict[str, Any] | None = None,
    resources: dict[str, Any] | None = None,
    fidelity: dict[str, Any] | None = None,
    limitations: Sequence[str] | None = None,
    text_metrics: dict[str, Any] | None = None,
    chunk_metrics: dict[str, Any] | None = None,
    usage: dict[str, Any] | None = None,
    paired_usage: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Assemble the canonical report dict.

    ``metadata`` (which corpus, repeat, cold/warm), ``stages`` (per-stage
    timings), ``resources`` (per-stage resource observations) and ``fidelity``
    (chunking profiles and their paired comparisons) describe the study around
    the metrics; they are passed through as the producers computed them.
    ``limitations`` is the study's own list of what its numbers cannot support.

    Raises ``ValueError`` if any input contains raw transcription — a document
    title included. Missing sections become typed misses so a reader can tell
    "not run" from "ran and produced nothing".
    """
    sections: dict[str, Any] = {
        "metadata": metadata,
        "stages": stages,
        "resources": resources,
        "fidelity": fidelity,
        "text_metrics": text_metrics,
        "chunk_metrics": chunk_metrics,
        "usage": usage,
        "paired_usage": paired_usage,
    }
    stated_limits = list(limitations) if limitations is not None else []
    offenders = find_raw_text_keys({"run": run, "limitations": limitations, **sections})
    if offenders:
        raise ValueError(f"report inputs contain raw transcription: {offenders}")
    return {
        "run": {
            field: run[field] for field in RUN_FIELDS if isinstance(run, dict) and field in run
        },
        "counts": _counts(sections, stated_limits),
        "limitations": stated_limits,
        **{key: value if value is not None else _missing() for key, value in sections.items()},
    }


def _missing() -> dict[str, Any]:
    return {"status": NOT_MEASURED, "reason": "not_provided"}


def _dig(source: Any, *path: str) -> Any:
    node = source
    for key in path:
        if not isinstance(node, dict) or key not in node:
            return None
        node = node[key]
    return node


def _counts(sections: dict[str, Any], limitations: Sequence[str]) -> dict[str, Any]:
    """The count block every renderer must agree on.

    A count taken from a section that was not provided is ``None`` — "not run"
    must not read as "ran and found nothing". ``limitations`` is a plain list of
    strings rather than a metric, so its count is always a real integer.
    """
    text_metrics = sections["text_metrics"]
    chunk_metrics = sections["chunk_metrics"]
    usage = sections["usage"]
    paired_usage = sections["paired_usage"]
    return {
        "documents": _dig(text_metrics, "documents_total"),
        "documents_with_gold": _dig(text_metrics, "documents_with_gold"),
        "pages_total": _dig(text_metrics, "pages", "total"),
        "pages_with_gold": _dig(text_metrics, "pages", "with_gold"),
        "pages_without_gold": _dig(text_metrics, "pages", "without_gold"),
        "pages_missing_hypothesis": _dig(text_metrics, "hypothesis", "missing_pages"),
        "pages_empty_hypothesis": _dig(text_metrics, "pages", "empty_hypothesis"),
        "chunks": _dig(chunk_metrics, "chunks"),
        "chunks_with_page_metadata": _dig(chunk_metrics, "page_fidelity", "chunks_with_pages"),
        "documents_with_usage": _dig(usage, "reported_documents"),
        "attempts": _dig(usage, "attempts"),
        "prompt_tokens": _dig(usage, "prompt_tokens"),
        "completion_tokens": _dig(usage, "completion_tokens"),
        "paired_documents": _dig(paired_usage, "paired_documents"),
        "union_documents": _dig(paired_usage, "union_documents"),
        "stages_total": _section_size(sections["stages"]),
        "stages_measured": _measured_stages(sections["stages"]),
        "resources_stages": _section_size(sections["resources"]),
        "fidelity_profiles": _section_size(sections["fidelity"]),
        "limitations": len(limitations),
    }


def _section_size(section: dict[str, Any] | None) -> int | None:
    return len(section) if isinstance(section, dict) else None


def _measured_stages(stages: dict[str, Any] | None) -> int | None:
    """Stages that produced a number: a stage may exist and still have failed."""
    if not isinstance(stages, dict):
        return None
    return sum(
        1
        for stage in stages.values()
        if isinstance(stage, dict) and stage.get("status") == MEASURED
    )


# --------------------------------------------------------------------------- #
# rendering
# --------------------------------------------------------------------------- #


def to_json(report: dict[str, Any]) -> str:
    """Canonical JSON, stable key order, UTF-8 text, trailing newline."""
    _reject_raw_text(report)
    return json.dumps(report, ensure_ascii=False, indent=2, sort_keys=True) + "\n"


def to_csv(report: dict[str, Any]) -> str:
    """One row per leaf value: counts, metrics and metadata."""
    _reject_raw_text(report)
    buffer = io.StringIO()
    writer = csv.writer(buffer, lineterminator="\n")
    writer.writerow(CSV_COLUMNS)
    for row in _rows(report):
        writer.writerow([row[column] for column in CSV_COLUMNS])
    return buffer.getvalue()


def to_markdown(report: dict[str, Any]) -> str:
    """One table per report section, counts first."""
    _reject_raw_text(report)
    rows = _rows(report)
    sections: dict[str, list[dict[str, str]]] = {}
    for row in rows:
        sections.setdefault(row["section"], []).append(row)
    lines = ["# OCR run report", ""]
    for section in ("run", "counts"):
        if section in sections:
            lines.extend(_table(section, sections.pop(section)))
    for section in sorted(sections):
        lines.extend(_table(section, sections[section]))
    return "\n".join(lines) + "\n"


def _table(section: str, rows: Sequence[dict[str, str]]) -> list[str]:
    """One table per section. ``reason`` travels with the value it explains."""
    lines = [
        f"## {section}",
        "",
        "| metric | status | value | numerator | denominator | unit | reason |",
        "| --- | --- | --- | --- | --- | --- | --- |",
    ]
    for row in rows:
        cells = (
            row["metric"],
            row["status"],
            row["value"],
            row["numerator"],
            row["denominator"],
            row["unit"],
            row["reason"],
        )
        lines.append("| " + " | ".join(_escape(cell) for cell in cells) + " |")
    lines.append("")
    return lines


def _escape(value: str) -> str:
    return value.replace("|", "\\|").replace("\n", " ")


def _rows(report: dict[str, Any]) -> list[dict[str, str]]:
    """Flatten the report into renderable rows.

    A dict with ``status`` and ``value`` is a metric leaf: it becomes one row,
    and its extra keys (token counts, nested tables) still recurse so nothing
    silently disappears.
    """
    rows: list[dict[str, str]] = []
    _walk(report, (), rows)
    return rows


def _walk(node: Any, path: tuple[str, ...], rows: list[dict[str, str]]) -> None:
    if isinstance(node, dict):
        leaf = "status" in node and "value" in node
        if leaf:
            rows.append(_row(path, node))
        for key, value in node.items():
            if leaf and key in METRIC_FIELDS:
                continue
            _walk(value, (*path, str(key)), rows)
    elif isinstance(node, list):
        for index, item in enumerate(node):
            _walk(item, (*path, str(index)), rows)
    else:
        rows.append(_scalar_row(path, node))


def _row(path: tuple[str, ...], metric: dict[str, Any]) -> dict[str, str]:
    row = dict.fromkeys(CSV_COLUMNS, "")
    row["section"] = path[0] if path else ""
    # A metric that is the section itself (a whole-run total, say) names itself.
    row["metric"] = ".".join(path[1:]) or row["section"]
    for field in METRIC_FIELDS:
        if field not in ("section", "metric"):
            row[field] = _format(metric.get(field))
    return row


def _scalar_row(path: tuple[str, ...], value: Any) -> dict[str, str]:
    row = dict.fromkeys(CSV_COLUMNS, "")
    row["section"] = path[0] if path else ""
    row["metric"] = ".".join(path[1:]) or row["section"]
    row["status"] = "count" if path and path[0] == "counts" else "value"
    row["value"] = _format(value)
    return row


def _format(value: Any) -> str:
    if value is None:
        return ""
    if isinstance(value, bool):
        return "true" if value else "false"
    if isinstance(value, float):
        return repr(round(value, 6))
    return str(value)


def _reject_raw_text(report: dict[str, Any]) -> None:
    offenders = find_raw_text_keys(report)
    if offenders:
        raise ValueError(f"report contains raw transcription: {offenders}")
