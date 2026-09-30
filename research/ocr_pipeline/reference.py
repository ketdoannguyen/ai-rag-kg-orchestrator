"""Gold transcripts: what a run should have produced, page by page.

A gold file is one JSON document, written by a human and taken as checked:

.. code-block:: json

    {
      "doc_id": "vl12-10pages",
      "pages": [
        {"page_index": 0, "text": "…", "headings": ["BÀI 1", "I. Điện trường"]}
      ]
    }

``doc_id`` names the document the transcripts were read from, so a gold file
cannot be attached to another document by accident. ``text`` is the page's
transcript. ``headings`` is optional: the heading strings on that page, in
reading order, each of which must occur in that page's own ``text`` -- a heading
that cannot be found is an error at load time rather than a scored miss,
because gold that disagrees with itself would measure the search instead of the
run. A page that has no transcript at all is simply left out of ``pages``, and
:func:`coverage` names it as missing.

Alignment pairs gold pages with the pages a run actually produced, over the
union of both page sets. A gold page the run never read has no hypothesis and is
not scored; a page the gold does not cover has no reference text.

Nothing in this module opens a PDF, calls a model, or touches the network.
"""

from __future__ import annotations

import hashlib
import json
from collections.abc import Sequence
from dataclasses import dataclass
from itertools import pairwise
from pathlib import Path
from typing import Any

from pydantic import (
    BaseModel,
    ConfigDict,
    Field,
    ValidationError,
    field_validator,
    model_validator,
)

from .models import DOC_ID_PATTERN, ExtractionArtifact, page_text

__all__ = [
    "AlignedPage",
    "CoverageReport",
    "GoldError",
    "GoldPage",
    "GoldReference",
    "align_pages",
    "coverage",
    "load_gold",
    "metrics_document",
]


class GoldError(RuntimeError):
    """Raised when a gold file cannot be loaded or fails validation."""


# --------------------------------------------------------------------------- #
# the gold file
# --------------------------------------------------------------------------- #


class GoldPage(BaseModel):
    """One gold page: its transcript, and the headings it lists.

    ``text`` is the whole page as a human read it, so every heading is a literal
    substring of it; ``headings`` is what the boundary metrics are scored
    against. A page with nothing worth transcribing is left out of the file
    rather than written blank, because a blank transcript would score every
    character the run read as an insertion.
    """

    model_config = ConfigDict(extra="forbid", frozen=True)

    page_index: int = Field(ge=0)
    """0-based physical page index, matching the artifact's page index."""
    text: str
    """The expected transcript of the page."""
    headings: tuple[str, ...] = ()
    """Heading strings on this page, in reading order; each occurs in ``text``."""

    @field_validator("text")
    @classmethod
    def _require_transcript(cls, value: str) -> str:
        if not value.strip():
            raise ValueError("a transcript must not be blank")
        return value

    @field_validator("headings")
    @classmethod
    def _require_heading(cls, value: tuple[str, ...]) -> tuple[str, ...]:
        for heading in value:
            if not heading.strip():
                raise ValueError("a heading must not be blank")
        return value

    def heading_spans(self) -> tuple[tuple[int, int], ...]:
        """Half-open character spans of this page's headings, in page order.

        Each heading is searched for from the end of the one before it, so a
        heading that occurs twice on a page is located twice and the spans stay
        in reading order. A heading that cannot be found raises rather than
        being dropped: it would silently stop measuring a boundary the gold
        claims exists.
        """
        spans: list[tuple[int, int]] = []
        cursor = 0
        for heading in self.headings:
            start = self.text.find(heading, cursor)
            if start < 0:
                raise GoldError(
                    f"heading {heading!r} does not occur in page {self.page_index} "
                    "at or after the heading before it"
                )
            spans.append((start, start + len(heading)))
            cursor = start + len(heading)
        return tuple(spans)


class GoldReference(BaseModel):
    """The gold file for one document."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    doc_id: str
    pages: tuple[GoldPage, ...] = Field(min_length=1)
    """Gold pages in ascending physical order."""

    @field_validator("doc_id")
    @classmethod
    def _clean_doc_id(cls, value: str) -> str:
        doc_id = value.strip()
        if not DOC_ID_PATTERN.fullmatch(doc_id):
            raise ValueError(f"doc_id must be a slug matching {DOC_ID_PATTERN.pattern!r}")
        return doc_id

    @model_validator(mode="after")
    def _check_pages(self) -> GoldReference:
        indices = [page.page_index for page in self.pages]
        if any(later <= earlier for earlier, later in pairwise(indices)):
            raise ValueError(f"gold pages must be in strictly increasing order, got {indices!r}")
        return self

    def page(self, index: int) -> GoldPage | None:
        """The gold page at ``index``, or ``None`` when the file does not hold it."""
        for page in self.pages:
            if page.page_index == index:
                return page
        return None

    @property
    def content_digest(self) -> str:
        """A name for the gold, independent of file layout.

        Only what a comparison depends on is hashed -- the document, and each
        page's transcript and headings. Reformatting the JSON keeps the same
        digest; editing a transcript does not.
        """
        payload = {
            "doc_id": self.doc_id,
            "pages": [
                {
                    "page_index": page.page_index,
                    "text": page.text,
                    "headings": list(page.headings),
                }
                for page in self.pages
            ],
        }
        text = json.dumps(payload, sort_keys=True, ensure_ascii=False, separators=(",", ":"))
        return hashlib.sha256(text.encode("utf-8")).hexdigest()


def load_gold(path: str | Path, *, expected_doc_id: str | None = None) -> GoldReference:
    """Load a gold file from disk and validate it.

    Args:
        path: Gold file to read.
        expected_doc_id: When given, the file's ``doc_id`` must match it, so the
            wrong gold cannot be attached to a document silently.

    Raises:
        GoldError: if the file cannot be read, is not valid JSON, does not match
            :class:`GoldReference`, names another document, or lists a heading
            that does not occur in its page's transcript.
    """
    payload = _read_json(path, what="gold")
    try:
        gold = GoldReference.model_validate(payload)
    except ValidationError as exc:
        locations = [
            ".".join(str(part) for part in error["loc"])
            for error in exc.errors(include_input=False)
        ]
        raise GoldError(f"gold {path} does not match the schema at {locations}") from None
    if expected_doc_id is not None and gold.doc_id != expected_doc_id:
        raise GoldError(
            f"gold {path} is for document {gold.doc_id!r}, expected {expected_doc_id!r}"
        )
    for page in gold.pages:
        try:
            page.heading_spans()
        except GoldError as exc:
            raise GoldError(f"gold {path}: page {page.page_index}: {exc}") from None
    return gold


# --------------------------------------------------------------------------- #
# coverage of the page selection
# --------------------------------------------------------------------------- #


@dataclass(frozen=True)
class CoverageReport:
    """How much of a page selection the gold actually covers."""

    requested_pages: tuple[int, ...]
    """The selection under study, as given."""
    covered_pages: tuple[int, ...]
    """Selected pages the gold holds a transcript for, ascending."""
    missing_pages: tuple[int, ...]
    """Selected pages the gold does not hold, ascending."""
    out_of_scope_pages: tuple[int, ...]
    """Gold pages the selection does not include, ascending."""

    @property
    def complete(self) -> bool:
        """True when the gold covers every selected page."""
        return not self.missing_pages


def coverage(reference: GoldReference, page_indices: Sequence[int]) -> CoverageReport:
    """Count how the gold covers one page selection."""
    requested = tuple(page_indices)
    selected = set(requested)
    held = {page.page_index for page in reference.pages}
    return CoverageReport(
        requested_pages=requested,
        covered_pages=tuple(index for index in requested if index in held),
        missing_pages=tuple(index for index in requested if index not in held),
        out_of_scope_pages=tuple(sorted(held - selected)),
    )


# --------------------------------------------------------------------------- #
# alignment: the gold text next to what a run produced
# --------------------------------------------------------------------------- #


@dataclass(frozen=True)
class AlignedPage:
    """One page of a gold/run comparison."""

    page_index: int
    """0-based physical page index."""
    request_page: int | None
    """The 1-based number the engine was asked for, or ``None`` when it never read the page."""
    reference: str | None
    """The expected text, or ``None`` when the gold does not cover this page."""
    hypothesis: str | None
    """What the run produced; ``None`` when the run never read the page."""

    def to_metrics_page(self) -> dict[str, Any]:
        """This page as :func:`~research.ocr_pipeline.metrics.document_text_metrics` reads it."""
        return {
            "page": self.request_page if self.request_page is not None else self.page_index + 1,
            "reference": self.reference,
            "hypothesis": self.hypothesis,
        }


def align_pages(reference: GoldReference, run: ExtractionArtifact) -> list[AlignedPage]:
    """Pair gold pages with the pages a run produced, over the union of both.

    Args:
        reference: The gold the run is compared against.
        run: One extraction artifact for the same document.

    Raises:
        GoldError: if the run belongs to another document.
    """
    if run.doc_id != reference.doc_id:
        raise GoldError(
            f"run is for document {run.doc_id!r}, but the gold is {reference.doc_id!r}"
        )
    run_pages = {page.page_index: page for page in run.pages}
    aligned: list[AlignedPage] = []
    for index in sorted(set(run_pages) | {page.page_index for page in reference.pages}):
        reference_page = reference.page(index)
        run_page = run_pages.get(index)
        aligned.append(
            AlignedPage(
                page_index=index,
                request_page=run_page.request_page if run_page is not None else None,
                reference=reference_page.text if reference_page is not None else None,
                hypothesis=(
                    page_text(run_page)
                    if run_page is not None and run_page.status == "ok"
                    else None
                ),
            )
        )
    return aligned


def metrics_document(reference: GoldReference, run: ExtractionArtifact) -> dict[str, Any]:
    """The ``document_text_metrics`` input for one gold/run pair."""
    return {
        "doc_id": reference.doc_id,
        "pages": [page.to_metrics_page() for page in align_pages(reference, run)],
    }


def _read_json(path: str | Path, *, what: str) -> Any:
    """Read and parse a JSON file, with every failure named as a :class:`GoldError`."""
    try:
        raw = Path(path).read_text(encoding="utf-8")
    except OSError as exc:
        raise GoldError(f"cannot read {what} {path}: {exc}") from exc
    try:
        return json.loads(raw)
    except json.JSONDecodeError as exc:
        raise GoldError(f"{what} {path} is not valid JSON: {exc}") from exc
