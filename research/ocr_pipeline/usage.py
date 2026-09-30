"""Token accounting: what a run asked of an engine, in attempts and tokens.

An engine records, per page, the attempts it made and the tokens the endpoint
reported; this module adds those up into the block a report publishes. Money is
not here and never was a property of a run: the study compares input tokens with
input tokens and output tokens with output tokens, and a token count is the same
number under every plan.

A subtotal is not a total
-------------------------
A page that reported nothing is not a page that spent nothing, so a run whose
pages did not all report publishes ``None`` for every counter and keeps what it
does know in ``subtotal``: an unknown count is never written as a zero somebody
measured, and a subtotal is never read as a total. The same rule holds one level
up in :func:`run_usage`, where the documents of a comparison are pooled.

Nothing in this module opens a PDF, calls a model, or touches the network.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from typing import Any

from .metrics import MEASURED, NOT_MEASURED
from .models import ExtractionArtifact

__all__ = [
    "COUNTERS",
    "PARTIAL",
    "USAGE_SCHEMA_VERSION",
    "artifact_usage",
    "failure_usage",
    "paired_usage",
    "run_usage",
    "usage_block",
]

USAGE_SCHEMA_VERSION = 1
"""Layout version of the usage block a report publishes."""

PARTIAL = "partial"
"""A subtotal that is not a total: some pages reported, some did not."""

COUNTERS = ("attempts", "prompt_tokens", "completion_tokens")
"""The counters a usage block aggregates, in the order it reports them."""


def _counter(value: Any) -> int | None:
    """One reported counter as a non-negative integer, or ``None`` when unknown."""
    if isinstance(value, int) and not isinstance(value, bool) and value >= 0:
        return value
    return None


def usage_values(usage: Any) -> dict[str, Any] | None:
    """One page's reported counters, or ``None`` when the page reported none."""
    if usage is None:
        return None
    if isinstance(usage, Mapping):
        return dict(usage)
    dump = getattr(usage, "model_dump", None)
    return dict(dump()) if callable(dump) else None


def usage_block(
    pages: Sequence[Any],
    *,
    source: str = "artifact",
    reason: str | None = None,
) -> dict[str, Any]:
    """Known subtotal plus completeness over one run's pages.

    A subtotal is not a total. A page that reported nothing is not a page that
    spent nothing, so whenever any page went unreported every top-level counter
    stays ``None`` (unknown) and ``subtotal`` carries what was actually known —
    the run's tokens are never quietly reduced to the pages that answered.
    """
    totals = dict.fromkeys(COUNTERS, 0)
    pages_with_usage = 0
    for page in pages:
        values = usage_values(getattr(page, "usage", None))
        if values is None:
            continue
        pages_with_usage += 1
        for name in COUNTERS:
            value = _counter(values.get(name))
            if value is not None:
                totals[name] += value
    complete = bool(pages) and pages_with_usage == len(pages)
    block: dict[str, Any] = {
        "usage_schema_version": USAGE_SCHEMA_VERSION,
        "source": source,
        "pages": len(pages),
        "pages_with_usage": pages_with_usage,
        "complete": complete,
        "subtotal": {**totals, "pages_with_usage": pages_with_usage},
    }
    if complete:
        block.update({name: totals[name] for name in COUNTERS})
    else:
        block.update(dict.fromkeys(COUNTERS, None))
    if not pages:
        block.update(status=NOT_MEASURED, reason=reason or "no_pages")
    elif not pages_with_usage:
        block.update(status=NOT_MEASURED, reason=reason or "usage_missing")
    elif not complete:
        block.update(status=PARTIAL, reason=reason or "usage_missing")
    else:
        block.update(status=MEASURED, reason=reason)
    return block


def artifact_usage(artifact: ExtractionArtifact) -> dict[str, Any]:
    """The accounting over an artifact's pages."""
    return usage_block(artifact.pages)


def failure_usage(error: BaseException) -> dict[str, Any]:
    """The accounting a failed run can still name.

    An engine that gave up after three tries and two answered pages spent those
    tokens, and the exception is the only place left that knows it: the failed
    page's attempts and tokens are read off the error the adapter raised, and a
    failure raised before the first request reports zero attempts with no tokens
    rather than a token count of zero.
    """
    attempts = getattr(error, "attempts", 0)
    attempts = attempts if isinstance(attempts, int) and attempts >= 0 else 0
    counts = {name: _counter(getattr(error, name, None)) for name in COUNTERS[1:]}
    stated = sum(1 for value in counts.values() if value is not None)
    block = usage_block([], source="failure", reason="usage_missing")
    block.update(
        {
            "pages": 1 if attempts else 0,
            "pages_with_usage": 1 if attempts else 0,
            "complete": bool(attempts),
            "attempts": attempts,
            "subtotal": {"attempts": attempts, "pages_with_usage": 1 if attempts else 0, **counts},
        }
    )
    # The failed page did report — it reported that it gave up. What it could not
    # say stays unknown, and the attempts it made stay counted either way.
    if not attempts:
        block.update(status=NOT_MEASURED, reason="usage_missing")
    elif stated == len(counts):
        block.update(status=MEASURED, reason=None, **counts)
    else:
        block.update(status=PARTIAL, reason="usage_missing", **counts)
    return block


# --------------------------------------------------------------------------- #
# a comparison's runs
# --------------------------------------------------------------------------- #


def _part_counters(block: Any) -> dict[str, int | None] | None:
    """One usage block's counters, falling back to the subtotal it still knows.

    ``None`` when there is no block at all, which is not the same as a block that
    reported nothing: a run that made no call has no counters to be missing.
    """
    if not isinstance(block, Mapping):
        return None
    stated = block.get("subtotal")
    subtotal: Mapping[Any, Any] = stated if isinstance(stated, Mapping) else {}
    counters: dict[str, int | None] = {}
    for name in COUNTERS:
        value = _counter(block.get(name))
        counters[name] = value if value is not None else _counter(subtotal.get(name))
    return counters


def _parts(document: Mapping[str, Any]) -> list[Mapping[str, Any]]:
    """The usage blocks one document spent: its extraction, then its merge.

    The parts are disjoint token sets -- the extraction counts the pages a
    document was read from, the merge counts the sections it was rebuilt from --
    so a total sums them without counting a token twice.
    """
    parts: list[Mapping[str, Any]] = []
    if "usage" in document:
        parts.append(document)
    merge = document.get("merge")
    if isinstance(merge, Mapping):
        parts.append(merge)
    return parts


def _totals(parts: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    """Pool the given parts: what every counter came to, and whether all reported.

    ``status`` is ``measured`` only when every part stated every counter;
    ``partial`` when some part stated something, and ``not_measured`` when none
    did. The counters stay ``None`` unless the status is ``measured``, so a
    partial total can never be read as a complete one.

    A part that answered nothing is not a part that spent nothing: it holds no
    counters, so pooling it leaves the pool incomplete rather than making the
    numbers of the parts that did answer come out as a total.
    """
    reported = False
    complete = True
    totals = dict.fromkeys(COUNTERS, 0)
    for part in parts:
        block = part.get("usage") if part is not None else None
        counters = _part_counters(block)
        if counters is None:
            continue
        status = block.get("status") if isinstance(block, Mapping) else None
        if status == NOT_MEASURED:
            complete = False
            continue
        if status is not None and status != MEASURED:
            complete = False
        for name in COUNTERS:
            value = counters[name]
            if value is None:
                complete = False
            else:
                reported = True
                totals[name] += value
    if not reported:
        status, reason = NOT_MEASURED, "usage_missing"
    elif not complete:
        status, reason = PARTIAL, "usage_missing"
    else:
        status, reason = MEASURED, None
    return {
        "status": status,
        "reason": reason,
        **{name: totals[name] if status == MEASURED else None for name in COUNTERS},
        "subtotal": dict(totals),
    }


def _document_usage(document: Mapping[str, Any]) -> dict[str, Any]:
    """One document's tokens, extraction and merge pooled."""
    pooled = _totals(_parts(document))
    return {
        "doc_id": document.get("doc_id"),
        "model": document.get("model"),
        **pooled,
    }


def run_usage(documents: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    """Input and output tokens over a run's documents, extraction and merge alike.

    ``documents`` is ``[{"doc_id": str, "model": str | None, "usage": block}, ...]``,
    where ``usage`` is a :func:`usage_block`; a document may also carry a
    ``"merge"`` part holding the usage of the LLM pass that rebuilt its tree.

    ``prompt_tokens`` is what the run read and ``completion_tokens`` what it
    wrote. Both are ``None`` unless every part reported, with ``subtotal``
    carrying what was known -- a run whose tokens are half-reported has no total
    to compare, and a document that reported nothing is counted in
    ``reported_documents`` rather than in the tokens.
    """
    rows = [_document_usage(document) for document in documents]
    pooled = _totals([{"usage": row} for row in rows])
    reported = [row for row in rows if row["status"] != NOT_MEASURED]
    merges = [
        document["merge"] for document in documents if isinstance(document.get("merge"), Mapping)
    ]
    if not rows:
        # No document to pool at all is not the same as documents that reported
        # nothing: one says there was nothing to count, the other that what was
        # counted went unstated.
        pooled.update(status=NOT_MEASURED, reason="no_documents")
    elif not reported:
        pooled.update(status=NOT_MEASURED, reason="usage_missing")
    block: dict[str, Any] = {
        "documents": len(rows),
        "reported_documents": len(reported),
        "per_document": rows,
        **pooled,
    }
    if merges:
        block["merge"] = _totals([{"usage": merge.get("usage")} for merge in merges])
    return block


def paired_usage(run_a: Mapping[str, Any], run_b: Mapping[str, Any]) -> dict[str, Any]:
    """Compare the tokens two runs spent on the documents they both cover.

    Runs are ``{"run_id": str, "documents": [{"doc_id", "model", "usage"}, ...]}``.
    Only documents present in *both* runs enter the comparison -- the paired
    denominator -- and the ids that break pairing are listed per side. A side
    whose documents did not all report has no total, so the delta is a typed miss
    instead of a number computed from the documents that answered.
    """
    documents_a = _documents_by_id(run_a)
    documents_b = _documents_by_id(run_b)
    shared = sorted(set(documents_a) & set(documents_b))
    usage_a = run_usage([documents_a[doc_id] for doc_id in shared])
    usage_b = run_usage([documents_b[doc_id] for doc_id in shared])
    return {
        "run_a": run_a.get("run_id"),
        "run_b": run_b.get("run_id"),
        "documents_in_a": len(documents_a),
        "documents_in_b": len(documents_b),
        "union_documents": len(set(documents_a) | set(documents_b)),
        "paired_documents": len(shared),
        "missing_in_a": sorted(set(documents_b) - set(documents_a)),
        "missing_in_b": sorted(set(documents_a) - set(documents_b)),
        "usage_a": usage_a,
        "usage_b": usage_b,
        "delta": _delta(usage_a, usage_b),
    }


def _delta(usage_a: Mapping[str, Any], usage_b: Mapping[str, Any]) -> dict[str, Any]:
    """B minus A per counter, or a typed miss when either side has no total."""
    if usage_a["status"] != MEASURED or usage_b["status"] != MEASURED:
        reason = (
            "incomplete_pairing"
            if usage_a["documents"] or usage_b["documents"]
            else "no_documents"
        )
        return {"status": NOT_MEASURED, "reason": reason, **dict.fromkeys(COUNTERS, None)}
    return {
        "status": MEASURED,
        "reason": None,
        **{name: usage_b[name] - usage_a[name] for name in COUNTERS},
    }


def _documents_by_id(run: Mapping[str, Any]) -> dict[str, Mapping[str, Any]]:
    """One run's documents keyed by doc_id, refusing duplicates and bad records."""
    documents = run.get("documents")
    if not isinstance(documents, list):
        raise ValueError("run must carry a 'documents' list")
    by_id: dict[str, Mapping[str, Any]] = {}
    for document in documents:
        if not isinstance(document, Mapping):
            raise ValueError(f"run documents must be mappings, got {type(document).__name__}")
        doc_id = document.get("doc_id")
        if not isinstance(doc_id, str):
            raise ValueError(f"run documents must carry a string doc_id, got {doc_id!r}")
        if doc_id in by_id:
            raise ValueError(f"duplicate doc_id in run: {doc_id!r}")
        by_id[doc_id] = document
    return by_id
