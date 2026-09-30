"""Pure OCR metrics: page CER/WER, chunking and page-structure statistics.

Every function takes plain ``dict``/``list`` values and returns plain ``dict``
values, so the module has no dependency on the corpus/store/model layers and can
be tested on synthetic fixtures alone. No I/O, no network, no logging.

Normalisation contract (used by CER/WER)
----------------------------------------
:func:`normalize_text` applies exactly two transformations:

1. Unicode **NFC** (canonical composition), so ``e`` + U+0302 + U+0301 and the
   precomposed ``ế`` compare equal — Vietnamese text is routinely decomposed in
   PDF text layers. Compatibility forms (NFKC) are deliberately *not* applied:
   they would rewrite characters such as ``ﬁ`` or full-width digits and hide
   real OCR errors.
2. Whitespace collapse: every run of Unicode whitespace (space, tab, newline,
   NBSP, ...) becomes one space, then the result is stripped. Line wrapping and
   indentation therefore never count as errors.

Case and punctuation are preserved — both are part of what page CER/WER
measures.

Typed missingness
-----------------
Every measured quantity is a metric dict::

    {"status": "measured" | "not_measured" | "empty_reference",
     "reason": str | None,          # set when status != "measured"
     "value": float | None,         # always None unless status == "measured"
     "numerator": int | None,       # e.g. edits, covered pages
     "denominator": int | None,     # e.g. reference characters, pages
     "unit": str,                   # what "value" is measured in ("ratio", ...)
     "denominator_unit": str | None}  # what "denominator" counts

An empty gold reference yields ``status="empty_reference"``, never ``0.0``:
"no reference" and "perfect transcript" must not collapse into the same number.

Missing transcripts and failed pages
------------------------------------
A page the engine never produced a transcript for (``hypothesis`` absent or
``None``) is ``status="not_measured"`` with ``reason="hypothesis_missing"``: a
failed page is not a 100% error rate, and folding it in as one would drag every
average down. An empty *string* is a transcript the engine did produce — the
page scores as a full deletion (``value=1.0``). The two events stay countable:
:func:`document_text_metrics` reports them separately, and the summary blocks
``pages`` / ``hypothesis`` carry the counts.

Page numbering
--------------
Physical page indices are **0-based** everywhere the corpus, gold and artifact
layers address pages (``ArtifactPage.page_index``, ``Span.page_index``,
``DocumentSpec.page_indices``). Page arithmetic in this module — coverage and
uncovered-page lists — uses that same 0-based index space.
"""

from __future__ import annotations

import math
import statistics
import unicodedata
from collections.abc import Sequence
from typing import Any

MEASURED = "measured"
NOT_MEASURED = "not_measured"
EMPTY_REFERENCE = "empty_reference"

__all__ = [
    "EMPTY_REFERENCE",
    "MEASURED",
    "NOT_MEASURED",
    "cer",
    "chunk_metrics",
    "corpus_text_metrics",
    "document_text_metrics",
    "edit_distance",
    "normalize_text",
    "page_structure_metrics",
    "wer",
]


# --------------------------------------------------------------------------- #
# typed metric helpers
# --------------------------------------------------------------------------- #


def _metric(
    value: float | None,
    status: str,
    *,
    reason: str | None = None,
    numerator: int | None = None,
    denominator: int | None = None,
    unit: str = "ratio",
    denominator_unit: str | None = None,
) -> dict[str, Any]:
    return {
        "status": status,
        "reason": reason,
        "value": value,
        "numerator": numerator,
        "denominator": denominator,
        "unit": unit,
        "denominator_unit": denominator_unit,
    }


def _ratio(numerator: int, denominator: int, *, denominator_unit: str | None) -> dict[str, Any]:
    """A measured rate, or a typed miss when the denominator is empty.

    The unit is ``None`` only for a caller pooling no measured row at all: there
    is no page whose unit could be named, so the miss carries none.
    """
    if denominator <= 0:
        return _metric(
            None,
            NOT_MEASURED,
            reason="empty_denominator",
            numerator=numerator,
            denominator=0,
            denominator_unit=denominator_unit,
        )
    return _metric(
        numerator / denominator,
        MEASURED,
        numerator=numerator,
        denominator=denominator,
        denominator_unit=denominator_unit,
    )


def _mean(values: Sequence[float], *, denominator_unit: str, reason: str) -> dict[str, Any]:
    """Unweighted mean over ``values``; each element counts once."""
    if not values:
        return _metric(
            None,
            NOT_MEASURED,
            reason=reason,
            denominator=0,
            denominator_unit=denominator_unit,
        )
    return _metric(
        statistics.mean(values),
        MEASURED,
        denominator=len(values),
        denominator_unit=denominator_unit,
    )


def _length_stats(lengths: Sequence[int]) -> dict[str, Any]:
    if not lengths:
        return {"status": NOT_MEASURED, "reason": "no_items", "count": 0}
    ordered = sorted(lengths)
    return {
        "status": MEASURED,
        "reason": None,
        "count": len(ordered),
        "min": ordered[0],
        "max": ordered[-1],
        "mean": statistics.mean(ordered),
        "median": statistics.median(ordered),
        "p90": _percentile(ordered, 0.90),
    }


def _percentile(ordered: Sequence[int | float], q: float) -> float:
    """Nearest-rank percentile over an already sorted sequence."""
    index = max(0, min(len(ordered) - 1, math.ceil(q * len(ordered)) - 1))
    return float(ordered[index])


def _require_dict(value: Any, what: str) -> dict[str, Any]:
    if not isinstance(value, dict):
        raise TypeError(f"{what} must be a dict, got {type(value).__name__}")
    return value


# --------------------------------------------------------------------------- #
# text normalisation and edit distance
# --------------------------------------------------------------------------- #


def normalize_text(text: str | None) -> str:
    """NFC-normalise ``text`` and collapse whitespace runs to single spaces.

    ``None`` normalises to the empty string. :func:`cer` and :func:`wer` test
    for ``None`` before calling this, because a missing transcript is a typed
    miss while an empty one is a measured full deletion; this helper only
    normalises text it was given.
    """
    if text is None:
        return ""
    if not isinstance(text, str):
        raise TypeError(f"text must be a str or None, got {type(text).__name__}")
    return " ".join(unicodedata.normalize("NFC", text).split())


def edit_distance(a: Sequence[Any], b: Sequence[Any]) -> int:
    """Levenshtein distance between two sequences, using two DP rows.

    Memory is ``O(min(len(a), len(b)))`` instead of a full matrix, which matters
    for page-sized character sequences. Insertions, deletions and substitutions
    each cost 1.
    """
    if not a:
        return len(b)
    if not b:
        return len(a)
    if len(a) < len(b):  # keep the shorter sequence on the row axis
        a, b = b, a
    previous = list(range(len(b) + 1))
    for i, item_a in enumerate(a, start=1):
        current = [i]
        for j, item_b in enumerate(b, start=1):
            substitution = previous[j - 1] + (0 if item_a == item_b else 1)
            current.append(min(previous[j] + 1, current[j - 1] + 1, substitution))
        previous = current
    return previous[-1]


def _missing_transcript(denominator: int, denominator_unit: str) -> dict[str, Any]:
    """A gold page with no transcript at all: unscored, never a 100% error."""
    return _metric(
        None,
        NOT_MEASURED,
        reason="hypothesis_missing",
        numerator=None,
        denominator=denominator,
        denominator_unit=denominator_unit,
    )


def cer(reference: str | None, hypothesis: str | None) -> dict[str, Any]:
    """Character error rate against one page's gold text.

    ``value = edits / reference characters`` after :func:`normalize_text`. A
    ``None`` hypothesis is a typed miss: the engine produced no transcript, so
    the page is unscored rather than scored as a full deletion.
    """
    ref = normalize_text(reference)
    if not ref:
        return _metric(
            None,
            EMPTY_REFERENCE,
            reason="empty_reference",
            numerator=len(normalize_text(hypothesis)),
            denominator=0,
            denominator_unit="characters",
        )
    if hypothesis is None:
        return _missing_transcript(len(ref), "characters")
    return _ratio(
        edit_distance(ref, normalize_text(hypothesis)), len(ref), denominator_unit="characters"
    )


def wer(reference: str | None, hypothesis: str | None) -> dict[str, Any]:
    """Word error rate against one page's gold text.

    Tokens are whitespace-separated after :func:`normalize_text`, so a reference
    of only whitespace is an empty reference, not a one-token reference. As in
    :func:`cer`, a ``None`` hypothesis is a typed miss, while ``""`` scores as
    every word deleted.
    """
    ref = normalize_text(reference).split()
    if not ref:
        return _metric(
            None,
            EMPTY_REFERENCE,
            reason="empty_reference",
            numerator=len(normalize_text(hypothesis).split()),
            denominator=0,
            denominator_unit="words",
        )
    if hypothesis is None:
        return _missing_transcript(len(ref), "words")
    return _ratio(
        edit_distance(ref, normalize_text(hypothesis).split()), len(ref), denominator_unit="words"
    )


# --------------------------------------------------------------------------- #
# per-page / per-document / corpus aggregation
# --------------------------------------------------------------------------- #


def _page_metrics(page: dict[str, Any]) -> dict[str, Any]:
    reference = page.get("reference")
    hypothesis = page.get("hypothesis")
    return {
        "page": page.get("page"),
        "has_gold": bool(normalize_text(reference)),
        "has_hypothesis": hypothesis is not None,
        "cer": cer(reference, hypothesis),
        "wer": wer(reference, hypothesis),
    }


def _pool(rows: Sequence[dict[str, Any]], metric: str) -> dict[str, Any]:
    """Micro rate: pool numerators and denominators over measured pages."""
    measured = [row[metric] for row in rows if row[metric]["status"] == MEASURED]
    return _ratio(
        sum(item["numerator"] for item in measured),
        sum(item["denominator"] for item in measured),
        denominator_unit=measured[0]["denominator_unit"] if measured else None,
    )


def _page_mean(rows: Sequence[dict[str, Any]], metric: str) -> dict[str, Any]:
    """Macro rate: unweighted mean of measured page values."""
    values = [row[metric]["value"] for row in rows if row[metric]["status"] == MEASURED]
    return _mean(values, denominator_unit="pages", reason="no_measured_pages")


def _text_summary(rows: Sequence[dict[str, Any]]) -> dict[str, Any]:
    """Pool one set of pages: counts in, micro and macro rates out."""
    measured = [row for row in rows if row["cer"]["status"] == MEASURED]
    return {
        "pages": len(rows),
        "measured_pages": len(measured),
        "missing_transcript_pages": sum(
            1 for row in rows if row["has_gold"] and not row["has_hypothesis"]
        ),
        "cer": {"micro": _pool(rows, "cer"), "macro": _page_mean(rows, "cer")},
        "wer": {"micro": _pool(rows, "wer"), "macro": _page_mean(rows, "wer")},
    }


def document_text_metrics(document: dict[str, Any]) -> dict[str, Any]:
    """Page CER/WER for one document.

    ``document`` is ``{"doc_id": str, "pages": [page, ...]}`` where a page is::

        {"page": 0,                    # int, optional, 0-based physical index
         "reference": "<gold text>",   # absent/"" => not measured
         "hypothesis": "<ocr text>"}   # absent/None => not_measured (failed page)

    A page the engine never produced a transcript for stays unscored: it is
    counted in ``hypothesis["missing_pages"]`` and in ``rates``'s
    ``missing_transcript_pages``, never as a 100% error rate. An empty string is
    a produced transcript and scores as a full deletion.

    ``macro`` averages the per-page values with every page weighted once,
    ``micro`` pools edits over pooled reference characters/words.
    """
    doc = _require_dict(document, "document")
    pages = [_require_dict(page, "page") for page in (doc.get("pages") or [])]
    rows = [_page_metrics(page) for page in pages]
    with_gold = sum(1 for row in rows if row["has_gold"])
    present = sum(1 for row in rows if row["has_hypothesis"])
    empty_hypothesis = sum(
        1
        for page, row in zip(pages, rows)
        if row["has_gold"] and row["has_hypothesis"] and not normalize_text(page.get("hypothesis"))
    )
    return {
        "doc_id": doc.get("doc_id"),
        "pages": {
            "total": len(rows),
            "with_gold": with_gold,
            "without_gold": len(rows) - with_gold,
            "empty_hypothesis": empty_hypothesis,
        },
        "hypothesis": {
            "present_pages": present,
            "missing_pages": len(rows) - present,
        },
        "rates": _text_summary(rows),
        "pages_detail": rows,
    }


def _document_summary(result: dict[str, Any]) -> dict[str, Any]:
    return {
        "doc_id": result["doc_id"],
        "pages": result["pages"],
        "hypothesis": result["hypothesis"],
        "rates": result["rates"],
    }


def _corpus_summary(per_document: Sequence[dict[str, Any]]) -> dict[str, Any]:
    """Micro pools every page; macro averages the per-document macro values.

    Each document counts once in ``macro`` regardless of how many pages it has,
    so one long document cannot carry the corpus figure.
    """
    rows = [row for result in per_document for row in result["pages_detail"]]
    pooled = _text_summary(rows)
    macro: dict[str, Any] = {}
    for metric in ("cer", "wer"):
        values = [
            result["rates"][metric]["macro"]["value"]
            for result in per_document
            if result["rates"][metric]["macro"]["status"] == MEASURED
        ]
        macro[metric] = _mean(values, denominator_unit="documents", reason="no_measured_documents")
    return {
        "pages": pooled["pages"],
        "measured_pages": pooled["measured_pages"],
        "missing_transcript_pages": pooled["missing_transcript_pages"],
        "cer": {"micro": pooled["cer"]["micro"], "macro": macro["cer"]},
        "wer": {"micro": pooled["wer"]["micro"], "macro": macro["wer"]},
    }


def corpus_text_metrics(documents: Sequence[dict[str, Any]]) -> dict[str, Any]:
    """Aggregate :func:`document_text_metrics` over documents.

    ``micro`` pools every measured page across documents. ``macro`` is the
    unweighted mean of the per-document macro values, so each document counts
    once regardless of length. ``per_document`` carries a compact projection
    (doc_id, page counts, rates) without raw text.
    """
    per_document = [document_text_metrics(document) for document in documents]
    return {
        "documents_total": len(per_document),
        "documents_with_gold": sum(1 for r in per_document if r["pages"]["with_gold"]),
        "pages": {
            "total": sum(r["pages"]["total"] for r in per_document),
            "with_gold": sum(r["pages"]["with_gold"] for r in per_document),
            "without_gold": sum(r["pages"]["without_gold"] for r in per_document),
            "empty_hypothesis": sum(r["pages"]["empty_hypothesis"] for r in per_document),
        },
        "hypothesis": {
            "present_pages": sum(r["hypothesis"]["present_pages"] for r in per_document),
            "missing_pages": sum(r["hypothesis"]["missing_pages"] for r in per_document),
        },
        "rates": _corpus_summary(per_document),
        "per_document": [_document_summary(result) for result in per_document],
    }


# --------------------------------------------------------------------------- #
# chunking and page-structure metrics
# --------------------------------------------------------------------------- #


def chunk_metrics(
    chunks: Sequence[dict[str, Any]],
    *,
    max_chunk_size: int | None = None,
    page_totals: dict[str, int] | None = None,
) -> dict[str, Any]:
    """Chunk length statistics plus page-fidelity checks.

    ``chunks`` is a list of ``{"chunk_id": str, "doc_id": str, "text": str,
    "pages": [int]}``; only ``text`` is required. Lengths are raw character
    counts (not whitespace-normalised) because they describe what is stored.
    ``pages`` holds **0-based** physical page indices, the index space of
    ``ArtifactPage.page_index`` and ``Span.page_index``.

    ``page_totals`` maps ``doc_id`` to the number of physical pages in scope,
    whose indices are ``0 .. count - 1``, and enables the coverage ratio and the
    uncovered-page lists; without it coverage is a typed miss rather than a
    guess. A chunk page outside that index space never counts as covered.
    """
    rows = [_require_dict(chunk, "chunk") for chunk in chunks]
    lengths = [len(chunk.get("text") or "") for chunk in rows]
    empty = sum(1 for length in lengths if length == 0)
    with_pages = [chunk for chunk in rows if chunk.get("pages")]
    spans = [_page_span(chunk["pages"]) for chunk in with_pages]
    return {
        "status": MEASURED if rows else NOT_MEASURED,
        "reason": None if rows else "no_chunks",
        "chunks": len(rows),
        "char_length": _length_stats(lengths),
        "empty_chunks": {"count": empty, "ratio": (empty / len(rows)) if rows else None},
        "oversized_chunks": _oversized(rows, lengths, max_chunk_size),
        "page_fidelity": {
            "chunks_with_pages": len(with_pages),
            "chunks_without_pages": len(rows) - len(with_pages),
            "multi_page_chunks": sum(1 for span in spans if span > 1),
            "page_span": _length_stats(spans)
            if spans
            else {"status": NOT_MEASURED, "reason": "no_page_metadata", "count": 0},
            "coverage": _coverage(rows, page_totals),
            "uncovered_pages": _uncovered(rows, page_totals),
        },
    }


def _oversized(
    chunks: Sequence[dict[str, Any]], lengths: Sequence[int], max_chunk_size: int | None
) -> dict[str, Any]:
    if max_chunk_size is None:
        return {"status": NOT_MEASURED, "reason": "no_limit", "limit": None, "count": None}
    offenders = [
        chunk.get("chunk_id") for chunk, length in zip(chunks, lengths) if length > max_chunk_size
    ]
    return {
        "status": MEASURED,
        "reason": None,
        "limit": max_chunk_size,
        "count": len(offenders),
        "chunk_ids": offenders,
    }


def _page_span(pages: Sequence[Any]) -> int:
    values = [int(page) for page in pages]
    return max(values) - min(values) + 1


def _covered_pages(
    chunks: Sequence[dict[str, Any]], page_totals: dict[str, int]
) -> dict[str, set[int]]:
    covered: dict[str, set[int]] = {doc_id: set() for doc_id in page_totals}
    for chunk in chunks:
        doc_id = chunk.get("doc_id")
        if doc_id in covered:
            covered[doc_id].update(int(page) for page in (chunk.get("pages") or []))
    return covered


def _expected_pages(total: int) -> set[int]:
    """The page indices a document of ``total`` pages declares, 0-based."""
    return set(range(total))


def _coverage(
    chunks: Sequence[dict[str, Any]], page_totals: dict[str, int] | None
) -> dict[str, Any]:
    if not page_totals:
        return _metric(None, NOT_MEASURED, reason="no_page_totals", denominator_unit="pages")
    covered = _covered_pages(chunks, page_totals)
    expected = {doc_id: _expected_pages(total) for doc_id, total in page_totals.items()}
    return _ratio(
        sum(len(pages & expected[doc_id]) for doc_id, pages in covered.items()),
        sum(len(pages) for pages in expected.values()),
        denominator_unit="pages",
    )


def _uncovered(
    chunks: Sequence[dict[str, Any]], page_totals: dict[str, int] | None
) -> dict[str, Any]:
    """Pages a document declares but no chunk references — dropped content."""
    if not page_totals:
        return {"status": NOT_MEASURED, "reason": "no_page_totals", "pages": None}
    covered = _covered_pages(chunks, page_totals)
    pages = {
        doc_id: sorted(_expected_pages(total) - covered[doc_id])
        for doc_id, total in page_totals.items()
    }
    return {"status": MEASURED, "reason": None, "pages": pages}


def page_structure_metrics(document: dict[str, Any]) -> dict[str, Any]:
    """Page-level structure statistics for one document.

    ``document`` is ``{"doc_id": str, "pages": [page, ...]}`` where a page is
    ``{"page": int, "text": str, "blocks": [{"kind": str}]}``; all three page
    keys are optional and every unavailable statistic is a typed miss.
    ``blocks`` is summarised by a raw ``by_kind`` histogram — no kind names are
    assumed, so a caller with a different block vocabulary still gets numbers.
    """
    doc = _require_dict(document, "document")
    pages = [_require_dict(page, "page") for page in (doc.get("pages") or [])]
    numbers = [int(page["page"]) for page in pages if isinstance(page.get("page"), int)]
    texts = [page.get("text") for page in pages]
    blocks = [block for page in pages for block in (page.get("blocks") or [])]
    by_kind: dict[str, int] = {}
    for block in blocks:
        kind = _require_dict(block, "block").get("kind")
        by_kind[str(kind)] = by_kind.get(str(kind), 0) + 1
    return {
        "doc_id": doc.get("doc_id"),
        "pages": len(pages),
        "page_numbers": _page_numbers(numbers, len(pages)),
        "text_chars": _length_stats([len(text or "") for text in texts])
        if texts
        else {"status": NOT_MEASURED, "reason": "no_pages", "count": 0},
        "empty_text_pages": sum(1 for text in texts if not (text or "").strip()),
        "blocks": {
            "status": MEASURED if blocks else NOT_MEASURED,
            "reason": None if blocks else "no_blocks",
            "total": len(blocks),
            "by_kind": by_kind,
        },
    }


def _page_numbers(numbers: Sequence[int], page_count: int) -> dict[str, Any]:
    if not numbers or len(numbers) != page_count:
        return {
            "status": NOT_MEASURED,
            "reason": "pages_without_number" if numbers else "no_page_numbers",
            "first": None,
            "last": None,
            "missing": [],
            "duplicates": [],
        }
    expected = set(range(min(numbers), max(numbers) + 1))
    seen: set[int] = set()
    duplicates: list[int] = []
    for number in numbers:
        if number in seen and number not in duplicates:
            duplicates.append(number)
        seen.add(number)
    return {
        "status": MEASURED,
        "reason": None,
        "first": min(numbers),
        "last": max(numbers),
        "missing": sorted(expected - seen),
        "duplicates": sorted(duplicates),
    }
