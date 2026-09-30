"""Heading and section-boundary scoring: gold headings against a run's own marks.

The metric answers one question — did the run put a heading, and a section start,
where the gold says one is? Two gates stand between a page and a score:

* **The heading gate.** Only a gold page that lists headings is scored, because
  the marks are the gold's own heading strings located in the gold's own
  transcript; a page whose gold marks nothing would measure the absence of a
  claim rather than the placement of one.
* **The alignment gate.** The run's text is lined up with the gold text token by
  token and the page is scored only when the two agree at or above an explicit
  confidence (default :data:`DEFAULT_CONFIDENCE`). Below it the page is left
  unscored with its reason counted: a run whose words cannot be located in the
  gold cannot have its boundaries located there either.

Marks are compared as single character offsets into the *raw* page text, never as
spans. The metric is about where a boundary sits, not how far a heading extends —
character error rate already measures the extent. A run's marks enter in the
run's own coordinates and are mapped onto gold offsets through the alignment, so
a heading found at a different offset than in the gold still counts when it
anchors to the same gold token; a mark whose text the alignment cannot place is
not credited, and is counted as a false positive instead of quietly disappearing.

Nothing here compares the gold with itself: a page whose candidate text is the
gold text is ``self_scoring``, because a copy would score perfectly and say
nothing. No returned value carries page text — only offsets are internal while
counts, statuses and the confidence are returned, which is what
:mod:`research.ocr_pipeline.report` accepts.

Physical page indices are 0-based and are the only page identity here; a run's
``request_page`` and the folio printed on the page never enter a comparison.
Nothing in this module opens a PDF, calls a model, or touches the network.
"""

from __future__ import annotations

import difflib
import re
import unicodedata
from collections import Counter
from collections.abc import Iterator, Mapping, Sequence
from dataclasses import dataclass
from typing import Any

from .metrics import MEASURED, NOT_MEASURED, _metric, _ratio, normalize_text
from .models import FURNITURE_KINDS, ArtifactPage, ExtractionArtifact, page_text
from .reference import GoldReference, align_pages

DEFAULT_CONFIDENCE = 0.90
"""Token agreement a page needs before its boundaries are scored.

Deliberately conservative. A page whose run text agrees with the reference on
nine words in ten is a page the alignment can be trusted on; below that the
matching blocks stop describing the same sentences and a "found" heading says
more about the matcher than about the run.
"""

MARK_KINDS: tuple[str, ...] = ("heading", "boundary")
"""The kinds a mark can carry, the vocabulary gold and a run both speak."""

OK_STATUS = "ok"
"""The page status a run's own text is scored from; a page it could not read has none."""

_MEASURE_NAMES = ("precision", "recall", "f1")
_TOKEN_RE = re.compile(r"\S+")

__all__ = [
    "DEFAULT_CONFIDENCE",
    "MARK_KINDS",
    "OK_STATUS",
    "CandidateMark",
    "artifact_marks",
    "heading_boundary_metrics",
]


@dataclass(frozen=True)
class CandidateMark:
    """One boundary a run claims, in the run's own page-text coordinates.

    ``offset`` indexes the same string the gold's own marks index on their side:
    ``page_text`` of the page the mark names. A mark carries the shared
    vocabulary's kind, so it is only ever compared with gold marks of its own
    kind, and a ``boundary`` mark is always attributed to the page its
    ``page_index`` names — never to a page number the engine asked for or one
    printed on the page.
    """

    page_index: int
    offset: int
    kind: str

    def __post_init__(self) -> None:
        if type(self.page_index) is not int or self.page_index < 0:
            raise ValueError(f"page_index must be a non-negative integer, got {self.page_index!r}")
        if type(self.offset) is not int or self.offset < 0:
            raise ValueError(f"offset must be a non-negative integer, got {self.offset!r}")
        if self.kind not in MARK_KINDS:
            raise ValueError(f"unknown mark kind {self.kind!r}; expected one of {MARK_KINDS}")


@dataclass(frozen=True)
class _Alignment:
    """Where a run's text sat in the reference text, token by token."""

    confidence: float
    tokens: tuple[tuple[str, int], ...]
    """The run's ``(NFC token, raw offset)`` pairs, in reading order."""
    reference_offsets: Mapping[int, int]
    """Run token index -> offset of the reference token it aligned with."""

    def map_offset(self, offset: int) -> int | None:
        """The reference offset the run's ``offset`` anchors to, or ``None``.

        An offset belongs to the first token starting at or after it: that is the
        token a boundary written at that offset introduces. A token the alignment
        could not place leaves the mark unmappable — the run claimed a boundary
        the reference text cannot confirm.
        """
        for index, (_, start) in enumerate(self.tokens):
            if start >= offset:
                return self.reference_offsets.get(index)
        return None


def artifact_marks(run: ExtractionArtifact) -> list[CandidateMark]:
    """The marks a run itself claims: every heading block, offered as both kinds.

    A heading block is where the run says a heading is *and* where it says a
    section starts, so the one block answers both comparisons. A page the engine
    could not read claims nothing: it has no text to place a mark in.
    """
    marks: list[CandidateMark] = []
    for page in run.pages:
        if page.status != OK_STATUS:
            continue
        for offset in _heading_offsets(page):
            marks.append(CandidateMark(page.page_index, offset, "heading"))
            marks.append(CandidateMark(page.page_index, offset, "boundary"))
    return marks


def heading_boundary_metrics(
    reference: GoldReference,
    run: ExtractionArtifact,
    *,
    candidates: Sequence[CandidateMark] | None = None,
    confidence: float = DEFAULT_CONFIDENCE,
) -> dict[str, Any]:
    """Score the run's heading and section-start placement against gold headings.

    Args:
        reference: The gold whose transcripts the marks are located in. Each
            heading it lists marks a heading's own text *and* where a section
            starts; only the mark's ``start`` is scored (see the module doc).
        run: The extraction whose marks are being scored.
        candidates: The run's marks, in ``page_text`` coordinates. ``None`` uses
            :func:`artifact_marks`, the run's own heading blocks; passing marks
            replaces that set rather than adding to it, which is how a chunker's
            own boundaries are scored.
        confidence: Token agreement a page needs before it is scored, in ``(0, 1]``.

    Returns:
        A metric dict: ``status``/``reason`` for the document, a ``pages`` census
        and the ``skipped`` reasons counted by name, ``kinds["heading"]`` and
        ``kinds["boundary"]`` with pooled counts plus precision, recall and F1,
        and ``pages_detail`` with one text-free row per considered page.

    Raises:
        GoldError: if a gold heading does not occur in its page's own transcript.
        ValueError: if ``confidence`` is outside ``(0, 1]``, or a candidate mark
            is malformed, names a page the run does not cover, or points past the
            end of that page's text.
    """
    if not 0.0 < confidence <= 1.0:
        raise ValueError(f"confidence must be in (0, 1], got {confidence!r}")
    marks = _validated_marks(artifact_marks(run) if candidates is None else candidates, run)
    reference_marks = _reference_marks(reference)
    aligned = {page.page_index: page for page in align_pages(reference, run)}
    run_pages = {page.page_index: page for page in run.pages}
    considered = sorted(reference_marks)
    rows: list[dict[str, Any]] = []
    skipped: Counter[str] = Counter()
    for index in considered:
        run_page = run_pages.get(index)
        candidate_text = page_text(run_page) if run_page is not None else ""
        reference_text = aligned[index].reference or ""
        reason = _skip_reason(index, run_pages=run_pages)
        if reason is None:
            alignment = _align(candidate_text, reference_text)
            if normalize_text(candidate_text) == normalize_text(reference_text):
                reason = "self_scoring"
            elif alignment.confidence < confidence:
                reason = "alignment_below_threshold"
        if reason is not None:
            skipped[reason] += 1
            rows.append(_skipped_row(index, reason))
            continue
        rows.append(_scored_row(index, alignment, marks.get(index, {}), reference_marks[index]))
    scored = [row for row in rows if row["status"] == MEASURED]
    if scored:
        status, reason = MEASURED, None
    else:
        status, reason = NOT_MEASURED, _dominant_reason(skipped)
    return {
        "doc_id": reference.doc_id,
        "status": status,
        "reason": reason,
        "confidence": confidence,
        "pages": {
            "marked": len(reference_marks),
            "considered": len(considered),
            "scored": len(scored),
            "skipped": len(rows) - len(scored),
        },
        "skipped": dict(sorted(skipped.items())),
        "kinds": {
            kind: _kind_block(kind, scored, status=status, reason=reason) for kind in MARK_KINDS
        },
        "pages_detail": rows,
    }


def _skip_reason(index: int, *, run_pages: Mapping[int, ArtifactPage]) -> str | None:
    """Why this page is not scored, or ``None`` when it is.

    Whether the run read the page, and whether it read it at all: a page the run
    never read, a page it could not read and a page it read to nothing stay
    countable as the different events they are. Only pages the gold marks are
    ever asked, which is the gate that keeps a page whose gold makes no claim
    from measuring the absence of one.
    """
    run_page = run_pages.get(index)
    if run_page is None:
        return "hypothesis_missing"
    if run_page.status != OK_STATUS:
        # The engine's own word for it: a page it could not read is not a page
        # that failed, and the report says which of the two happened.
        return run_page.status
    if not page_text(run_page).strip():
        return "empty_candidate_text"
    return None


def _scored_row(
    index: int,
    alignment: _Alignment,
    page_marks: Mapping[str, Sequence[CandidateMark]],
    reference_marks: Mapping[str, set[int]],
) -> dict[str, Any]:
    """One scored page: what each kind found, in counts, with no text in it."""
    kinds: dict[str, Any] = {}
    for kind in MARK_KINDS:
        gold = reference_marks[kind]
        marks = page_marks.get(kind, ())
        predicted: set[int] = set()
        unmapped = 0
        for mark in marks:
            offset = alignment.map_offset(mark.offset)
            if offset is None:
                unmapped += 1
            else:
                predicted.add(offset)
        kinds[kind] = {
            "gold": len(gold),
            "candidate": len(marks),
            "unmapped": unmapped,
            "true_positives": len(predicted & gold),
            "false_positives": len(predicted - gold) + unmapped,
            "false_negatives": len(gold - predicted),
        }
    return {
        "page_index": index,
        "status": MEASURED,
        "reason": None,
        "alignment": alignment.confidence,
        "kinds": kinds,
    }


def _skipped_row(index: int, reason: str) -> dict[str, Any]:
    return {
        "page_index": index,
        "status": NOT_MEASURED,
        "reason": reason,
        "alignment": None,
        "kinds": {},
    }


def _kind_block(
    kind: str,
    scored: Sequence[dict[str, Any]],
    *,
    status: str,
    reason: str | None,
) -> dict[str, Any]:
    """Pool one mark kind over the scored pages: counts in, rates out.

    A kind with no reference marks is not measurable — there is no gold boundary
    to be right about — and says so instead of reporting a perfect zero. A run
    that placed nothing leaves precision undefined while recall is a measured
    zero, which is the same split :mod:`~research.ocr_pipeline.metrics` draws
    between a missing denominator and a real miss.
    """
    counts = [row["kinds"][kind] for row in scored]
    true_positives = sum(item["true_positives"] for item in counts)
    false_positives = sum(item["false_positives"] for item in counts)
    false_negatives = sum(item["false_negatives"] for item in counts)
    gold = sum(item["gold"] for item in counts)
    if status != MEASURED:
        block_status, block_reason = NOT_MEASURED, reason
    elif not gold:
        block_status, block_reason = NOT_MEASURED, "no_gold_marks"
    else:
        block_status, block_reason = MEASURED, None
    measures = (
        {
            "precision": _ratio(
                true_positives, true_positives + false_positives, denominator_unit="marks"
            ),
            "recall": _ratio(
                true_positives, true_positives + false_negatives, denominator_unit="marks"
            ),
            "f1": _ratio(
                2 * true_positives,
                2 * true_positives + false_positives + false_negatives,
                denominator_unit="marks",
            ),
        }
        if block_status == MEASURED
        else {
            name: _metric(None, NOT_MEASURED, reason=block_reason, denominator_unit="marks")
            for name in _MEASURE_NAMES
        }
    )
    return {
        "status": block_status,
        "reason": block_reason,
        "pages": len(counts),
        "gold_marks": gold,
        "candidate_marks": sum(item["candidate"] for item in counts),
        "unmapped_marks": sum(item["unmapped"] for item in counts),
        "true_positives": true_positives,
        "false_positives": false_positives,
        "false_negatives": false_negatives,
        **measures,
    }


def _dominant_reason(skipped: Counter[str]) -> str:
    """The reason most pages were left unscored, named for the document.

    Every reason is still counted in ``skipped``; this only says which one a
    reader should look at first. A tie falls back to the reason name so the same
    study always reports the same string.
    """
    if not skipped:
        return "no_eligible_pages"
    return min(skipped.items(), key=lambda item: (-item[1], item[0]))[0]


# --------------------------------------------------------------------------- #
# the gold side of the comparison
# --------------------------------------------------------------------------- #


def _reference_marks(reference: GoldReference) -> dict[int, dict[str, set[int]]]:
    """Gold marks per page and kind: the start of every heading the page lists.

    A heading is where the gold says a heading is *and* where it says a section
    starts, so -- like a run's heading block -- it answers both comparisons. Two
    headings with the same words on one page are two marks at two offsets. A page
    whose gold lists no heading contributes no marks and is not scored at all.

    Raises:
        GoldError: if a heading does not occur in its page's own transcript.
    """
    marks: dict[int, dict[str, set[int]]] = {}
    for page in reference.pages:
        starts = {span[0] for span in page.heading_spans()}
        if starts:
            marks[page.page_index] = {kind: set(starts) for kind in MARK_KINDS}
    return marks


# --------------------------------------------------------------------------- #
# the run's side of the comparison
# --------------------------------------------------------------------------- #


def _validated_marks(
    candidates: Sequence[CandidateMark], run: ExtractionArtifact
) -> dict[int, dict[str, list[CandidateMark]]]:
    """Group the run's marks by page and kind, after checking them against the run.

    A mark is only meaningful where the run has text: a page the run never read,
    an offset past the end of that page's text, or the same mark twice is a
    caller's bug rather than a zero, so it raises.
    """
    pages = {page.page_index: page for page in run.pages}
    grouped: dict[int, dict[str, list[CandidateMark]]] = {}
    seen: set[tuple[int, int, str]] = set()
    for mark in candidates:
        if not isinstance(mark, CandidateMark):
            raise ValueError(f"candidate marks must be CandidateMark, got {type(mark).__name__}")
        page = pages.get(mark.page_index)
        if page is None:
            raise ValueError(
                f"candidate mark names page {mark.page_index}, which the run does not cover"
            )
        length = len(page_text(page))
        if mark.offset > length:
            raise ValueError(
                f"candidate mark at {mark.offset} is beyond page {mark.page_index} "
                f"({length} character(s))"
            )
        if (mark.page_index, mark.offset, mark.kind) in seen:
            raise ValueError(
                f"duplicate {mark.kind} mark on page {mark.page_index} at {mark.offset}"
            )
        seen.add((mark.page_index, mark.offset, mark.kind))
        grouped.setdefault(mark.page_index, {kind: [] for kind in MARK_KINDS})[mark.kind].append(
            mark
        )
    return grouped


def _heading_offsets(page: ArtifactPage) -> Iterator[int]:
    """Offsets of the page's heading blocks inside ``page_text(page)``.

    The walk repeats ``page_text``'s own filter and joiner, so an offset is a
    position in exactly the string the gold's headings are located in; furniture
    and wordless blocks take no part in it.
    """
    offset = 0
    for block in sorted(page.blocks, key=lambda item: item.order):
        if block.normalized_kind in FURNITURE_KINDS or not block.text.strip():
            continue
        if block.normalized_kind == "heading":
            yield offset
        offset += len(block.text) + 1


# --------------------------------------------------------------------------- #
# alignment
# --------------------------------------------------------------------------- #


def _align(candidate: str, reference: str) -> _Alignment:
    """Line the run's tokens up with the reference's and say how well they agree.

    ``difflib`` rather than a hand-written matcher, with its junk heuristic off:
    on a page of repeated words that heuristic calls any common token junk and the
    matching blocks stop describing the same sentences. Confidence is the Dice
    coefficient over tokens, so a dropped line costs its own length and no more,
    and line wrapping — which no OCR comparison should count — costs nothing.
    """
    candidate_tokens = _tokens(candidate)
    reference_tokens = _tokens(reference)
    matcher = difflib.SequenceMatcher(
        None,
        [token for token, _ in candidate_tokens],
        [token for token, _ in reference_tokens],
        autojunk=False,
    )
    reference_offsets: dict[int, int] = {}
    matched = 0
    for candidate_start, reference_start, size in matcher.get_matching_blocks():
        matched += size
        for step in range(size):
            reference_offsets[candidate_start + step] = reference_tokens[reference_start + step][1]
    total = len(candidate_tokens) + len(reference_tokens)
    return _Alignment(
        confidence=(2 * matched / total) if total else 0.0,
        tokens=candidate_tokens,
        reference_offsets=reference_offsets,
    )


def _tokens(text: str) -> tuple[tuple[str, int], ...]:
    """``(NFC token, raw offset)`` pairs: the comparison unit and where it sits.

    Tokens are what :func:`~research.ocr_pipeline.metrics.normalize_text` yields
    when split, so alignment and the text metrics compare the same units and
    Vietnamese diacritics composed either way are the same token; the offsets
    stay in the raw text the gold's headings are located in.
    """
    return tuple(
        (unicodedata.normalize("NFC", match.group()), match.start())
        for match in _TOKEN_RE.finditer(text)
    )
