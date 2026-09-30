"""Strict schemas for the OCR research corpus and the artifacts extracted from it.

Two families of persisted data live here:

* The versioned corpus manifest, which is public config tracked in git. Its
  documents declare an explicit, allowlisted set of fields (modality, page
  subset, permission note, reference paths), so a stray credential key such as
  ``api_key`` is a validation error instead of a silent write.
* The engine-neutral :class:`ExtractionArtifact`, whose content address is the
  sha256 of its canonical JSON serialization. The artifact keeps pages in
  physical PDF order and blocks in reading order, independently of whether the
  engine was a vision model, Marker, or the PDF text layer.

The artifact's vocabulary — engine ids, versions, and the neutral block kinds —
is the engine registry's (:mod:`.engines`), imported here rather than restated,
so a schema change that would reject an engine's own output fails at import
time. Nothing in this module reads a file, renders a page, or touches the
network.
"""

from __future__ import annotations

import hashlib
import json
import re
from collections.abc import Mapping, Sequence
from enum import Enum
from itertools import pairwise
from typing import Any, Literal
from urllib.parse import urlsplit

from pydantic import (
    BaseModel,
    ConfigDict,
    Field,
    field_validator,
    model_validator,
)

from .engines import (
    ARTIFACT_VERSION,
    ENGINE_IDS,
    MAPPING_VERSION,
    NORMALIZED_KINDS,
    RETURNED_STATUSES,
    EngineArtifact,
)

FURNITURE_KINDS: tuple[str, ...] = ("page_header", "page_footer")
"""Non-content block kinds: a running head or folio is the page's, not the document's."""

TEXTLESS_KINDS: tuple[str, ...] = ("figure", "unmapped")
"""Kinds allowed to carry no words: a picture, and a kind the adapter could not map."""

SHA256_PATTERN = re.compile(r"[0-9a-f]{64}")
"""What a digest looks like everywhere a file is named after its content."""

DOC_ID_PATTERN = re.compile(r"[A-Za-z0-9][A-Za-z0-9._-]*")
"""What a doc id looks like everywhere it becomes a path segment or a report name."""

_BARE_HOST_RE = re.compile(r"[A-Za-z0-9.-]+(?::[0-9]+)?")
_SECRET_WORDS = frozenset(
    {
        "apikey",
        "auth",
        "authorization",
        "bearer",
        "cookie",
        "credential",
        "credentials",
        "header",
        "headers",
        "key",
        "password",
        "passwd",
        "secret",
        "session",
        "token",
    }
)

__all__ = [
    "ARTIFACT_VERSION",
    "ENGINE_IDS",
    "MAPPING_VERSION",
    "NORMALIZED_KINDS",
    "DOC_ID_PATTERN",
    "SHA256_PATTERN",
    "ArtifactBlock",
    "ArtifactPage",
    "ArtifactStatistics",
    "DocumentSpec",
    "ExtractionArtifact",
    "Manifest",
    "Modality",
    "PageUsage",
    "canonical_payload",
    "page_text",
    "semantic_fingerprint",
]


# --------------------------------------------------------------------------- #
# shared field checks
# --------------------------------------------------------------------------- #


def _slug(value: str) -> str:
    """Return ``value`` as a doc id slug, or raise."""
    doc_id = value.strip()
    if not DOC_ID_PATTERN.fullmatch(doc_id):
        raise ValueError(
            f"doc_id must be a slug matching {DOC_ID_PATTERN.pattern!r}, got {value!r}"
        )
    return doc_id


def _sha256(value: str | None) -> str | None:
    """Return ``value`` as a lowercase sha256, ``None`` staying ``None``."""
    if value is None:
        return None
    digest = value.strip()
    if not SHA256_PATTERN.fullmatch(digest):
        raise ValueError(f"expected 64 lowercase hex characters, got {value!r}")
    return digest


def _non_blank(value: str | None, what: str) -> str | None:
    """Return ``value`` stripped, ``None`` staying ``None`` and blanks rejected."""
    if value is None:
        return None
    text = value.strip()
    if not text:
        raise ValueError(f"{what} must be null or non-blank")
    return text


def _digest(payload: Mapping[str, Any]) -> str:
    """Hex sha256 of a payload's canonical JSON, the one address convention here."""
    text = json.dumps(dict(payload), sort_keys=True, ensure_ascii=False, separators=(",", ":"))
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


# --------------------------------------------------------------------------- #
# corpus manifest
# --------------------------------------------------------------------------- #


class Modality(str, Enum):
    """What the PDF itself is, never which engine reads it."""

    SCAN = "scan"
    """Pages are images: no text layer, so a text-layer engine will refuse them."""
    DIGITAL = "digital"
    """Every page in scope carries its own text layer."""
    MIXED = "mixed"
    """Some pages in scope have a text layer and some do not."""


class DocumentSpec(BaseModel):
    """One PDF under study, the page subset in scope, and its gold.

    ``gold_path`` must be declared explicitly, as a path or as ``null``. A
    manifest that forgets to mention gold can then never be confused with one
    that states the document has no gold yet.
    """

    model_config = ConfigDict(extra="forbid", frozen=True)

    doc_id: str
    """Stable slug used in reports and artifact metadata."""
    pdf_path: str
    """Path to the PDF; relative paths resolve against the manifest directory."""
    pdf_sha256: str
    """Lowercase sha256 of the PDF, verified before any page is read."""
    page_indices: tuple[int, ...] = Field(min_length=1)
    """0-based physical page indices in scope, strictly increasing."""
    modality: Modality
    """What the PDF is (scan, digital, mixed), which validation re-checks per page."""
    permission_note: str
    """Why this document may be used, recorded at the manifest level."""
    gold_path: str | None
    """The JSON gold transcript for this document; ``null`` when none exists."""

    @field_validator("doc_id")
    @classmethod
    def _clean_doc_id(cls, value: str) -> str:
        return _slug(value)

    @field_validator("pdf_path", "gold_path")
    @classmethod
    def _require_non_blank(cls, value: str | None) -> str | None:
        return _non_blank(value, "paths")

    @field_validator("pdf_sha256")
    @classmethod
    def _require_pdf_sha256(cls, value: str) -> str | None:
        return _sha256(value)

    @field_validator("permission_note")
    @classmethod
    def _require_permission(cls, value: str) -> str:
        note = value.strip()
        if not note:
            raise ValueError("permission_note must state why the document may be used")
        return note

    @field_validator("page_indices")
    @classmethod
    def _require_ordered_pages(cls, value: tuple[int, ...]) -> tuple[int, ...]:
        if any(type(index) is not int or index < 0 for index in value):
            raise ValueError(f"page_indices must be non-negative integers, got {value!r}")
        if any(later <= earlier for earlier, later in pairwise(value)):
            raise ValueError(f"page_indices must be strictly increasing, got {value!r}")
        return value


class Manifest(BaseModel):
    """Versioned multi-document manifest: the tracked description of a corpus."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    corpus_version: Literal[1]
    """Bumped when a field changes meaning; an unknown version fails validation."""
    documents: list[DocumentSpec] = Field(min_length=1)

    @field_validator("corpus_version", mode="before")
    @classmethod
    def _require_supported_version(cls, value: Any) -> Any:
        if type(value) is not int or value != 1:
            raise ValueError(f"unsupported corpus_version {value!r}: this loader reads version 1")
        return value

    @model_validator(mode="after")
    def _check_unique_doc_ids(self) -> Manifest:
        seen: set[str] = set()
        for document in self.documents:
            if document.doc_id in seen:
                raise ValueError(f"duplicate doc_id {document.doc_id!r} in manifest")
            seen.add(document.doc_id)
        return self


# --------------------------------------------------------------------------- #
# artifact pages and blocks
# --------------------------------------------------------------------------- #


class PageUsage(BaseModel):
    """What one page asked of the engine: attempts, input tokens, output tokens.

    The counters are meaningless apart — an attempt count without tokens, or a
    token count without knowing how many tries bought them — so a page either
    records the whole set or none of it.
    """

    model_config = ConfigDict(extra="forbid", frozen=True)

    attempts: int = Field(ge=0)
    prompt_tokens: int = Field(ge=0)
    completion_tokens: int = Field(ge=0)


class ArtifactBlock(BaseModel):
    """One block of a page, in reading order, in the engine's own words.

    ``raw_kind`` is what the engine called it and ``normalized_kind`` is the
    neutral vocabulary, so a Marker replay can still see ``SectionHeader`` while
    every builder works in neutral terms.
    """

    model_config = ConfigDict(extra="forbid", frozen=True)

    order: int = Field(ge=0)
    """0-based position within the page, strictly increasing across the page's blocks."""
    raw_kind: str
    """The kind the engine itself reported (``Text``, ``SectionHeader``, ...)."""
    normalized_kind: str
    """Neutral kind, one of :data:`NORMALIZED_KINDS`."""
    text: str
    """The block's words; allowed to be empty only for a picture or an unmapped kind."""
    marker: str | None = None
    depth: int | None = Field(default=None, ge=1)
    """Structural depth of a heading, when the engine reported one."""
    group: int | None = Field(default=None, ge=1)
    """Consecutive blocks sharing a group belong to one idea; ``null`` when ungrouped."""
    bbox: tuple[float, float, float, float] | None = None
    """Coordinates in the engine's own space, never normalised across engines."""
    source_id: str | None = None
    """Engine provenance id (a Marker block id, a native line range, ...)."""

    @field_validator("raw_kind")
    @classmethod
    def _require_raw_kind(cls, value: str) -> str:
        kind = value.strip()
        if not kind:
            raise ValueError("raw_kind must name the engine's own kind")
        return kind

    @field_validator("normalized_kind")
    @classmethod
    def _require_neutral_kind(cls, value: str) -> str:
        if value not in NORMALIZED_KINDS:
            raise ValueError(
                f"unknown normalized_kind {value!r}; expected one of {NORMALIZED_KINDS}"
            )
        return value

    @field_validator("marker", "source_id")
    @classmethod
    def _require_non_blank(cls, value: str | None) -> str | None:
        return _non_blank(value, "engine provenance fields")

    @model_validator(mode="after")
    def _check_kind_and_text(self) -> ArtifactBlock:
        if self.normalized_kind == "heading" and self.depth is None:
            raise ValueError("heading blocks require depth")
        if self.normalized_kind not in TEXTLESS_KINDS and not self.text.strip():
            raise ValueError("text must not be blank")
        return self


class ArtifactPage(BaseModel):
    """One page of an artifact, addressed by its physical index in the PDF."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    page_index: int = Field(ge=0)
    """0-based physical page index in the source PDF, never the printed folio number."""
    request_page: int = Field(ge=1)
    """The 1-based number the engine was asked for: ``page_index + 1 + page_offset``."""
    printed_folio: str | None = None
    """The folio printed on the page, when the engine reported one."""
    running: str = ""
    """Running head or footer the engine reported for this page."""
    continues_previous: bool = False
    """True when the first block continues the group left open on the previous page."""
    status: str = "ok"
    """``ok``, or ``unsupported_modality`` for a page the engine could not read."""
    blocks: tuple[ArtifactBlock, ...] = ()
    usage: PageUsage | None = None
    """Attempts and tokens the engine actually spent on this page, or ``None``."""

    @field_validator("status")
    @classmethod
    def _require_status(cls, value: str) -> str:
        if value not in RETURNED_STATUSES:
            raise ValueError(f"unknown page status {value!r}; expected one of {RETURNED_STATUSES}")
        return value

    @field_validator("printed_folio")
    @classmethod
    def _require_folio(cls, value: str | None) -> str | None:
        return _non_blank(value, "printed_folio")

    @model_validator(mode="after")
    def _check_block_order(self) -> ArtifactPage:
        orders = [block.order for block in self.blocks]
        if any(later <= earlier for earlier, later in pairwise(orders)):
            raise ValueError(f"blocks must be in strictly increasing order, got {orders!r}")
        return self


def page_text(page: ArtifactPage) -> str:
    """The page's content text: what an OCR comparison is allowed to see.

    Running heads and folios are page furniture, and a block with no words says
    nothing about the text layer, so both are left out; the rest are joined in
    reading order. This is the definition every gold reference span indexes into.
    """
    return "\n".join(
        block.text
        for block in sorted(page.blocks, key=lambda item: item.order)
        if block.normalized_kind not in FURNITURE_KINDS and block.text.strip()
    )


class ArtifactStatistics(BaseModel):
    """The summary an engine derives from its own pages, checked against them."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    blocks_by_kind: dict[str, int]
    """Blocks per neutral kind, sorted so two runs compare byte for byte."""
    normalization: dict[str, int]
    """How the adapter mapped the engine output, counted per rule."""


# --------------------------------------------------------------------------- #
# the artifact
# --------------------------------------------------------------------------- #


class ExtractionArtifact(BaseModel):
    """One extraction run over a page subset, independent of the engine that ran it.

    The artifact carries no run id and no timestamp: two runs that read the same
    pages through the same configuration are the same artifact, so a stored file
    is named after the sha256 of its own canonical JSON and a repeat run costs no
    new file. The provenance a replay needs — engine, versions, prompt digest,
    render settings, selected pages — is part of the payload.
    """

    model_config = ConfigDict(extra="forbid", frozen=True)

    artifact_version: Literal[1]
    mapping_version: int
    """Version of the adapter's mapping rules; a newer mapping is a different extraction."""
    engine_id: str
    engine_version: str | None = None
    """Version the engine reported, or ``None`` when its probe could not name one."""
    status: str = "ok"
    doc_id: str
    pdf_sha256: str | None
    """sha256 of the source PDF; ``None`` only when the engine could not hash it."""
    source_sha256: str | None = None
    """Digest of the bytes the engine read, when that is not the PDF itself."""
    physical_pages: tuple[int, ...]
    """0-based physical pages the run covered, exactly the pages listed below."""
    render_dpi: int | None = Field(default=None, ge=1)
    render_page_offset: int | None = None
    prompt_sha256: str | None = None
    public_config: dict[str, Any]
    """Allowlisted, resolved configuration; never a key, header, or full URL."""
    pages: tuple[ArtifactPage, ...] = Field(min_length=1)
    """Extracted pages in ascending physical order; a run that extracted nothing is no artifact."""
    statistics: ArtifactStatistics
    notes: tuple[str, ...] = ()

    @field_validator("artifact_version", mode="before")
    @classmethod
    def _require_version_one(cls, value: Any) -> Any:
        if type(value) is not int or value != 1:
            raise ValueError(
                f"unsupported artifact_version {value!r}: this schema persists version 1 only"
            )
        return value

    @field_validator("mapping_version", mode="before")
    @classmethod
    def _require_mapping_version(cls, value: Any) -> Any:
        # Unlike the schema version, a mapping version is the adapter's own and
        # moves forward; only a real, positive integer names one.
        if type(value) is not int or value < 1:
            raise ValueError(f"mapping_version must be a positive integer, got {value!r}")
        return value

    @field_validator("engine_id")
    @classmethod
    def _require_engine_id(cls, value: str) -> str:
        engine_id = value.strip()
        if engine_id not in ENGINE_IDS:
            raise ValueError(f"unknown engine_id {value!r}; expected one of {ENGINE_IDS}")
        return engine_id

    @field_validator("status")
    @classmethod
    def _require_status(cls, value: str) -> str:
        if value not in RETURNED_STATUSES:
            raise ValueError(
                f"unsupported status {value!r}; a returned artifact is one of {RETURNED_STATUSES}"
            )
        return value

    @field_validator("doc_id")
    @classmethod
    def _clean_doc_id(cls, value: str) -> str:
        return _slug(value)

    @field_validator("engine_version")
    @classmethod
    def _require_engine_version(cls, value: str | None) -> str | None:
        return _non_blank(value, "engine_version")

    @field_validator("pdf_sha256", "source_sha256", "prompt_sha256")
    @classmethod
    def _require_sha256(cls, value: str | None) -> str | None:
        return _sha256(value)

    @field_validator("notes")
    @classmethod
    def _require_notes(cls, value: tuple[str, ...]) -> tuple[str, ...]:
        for note in value:
            if not note.strip():
                raise ValueError("notes must not be blank")
        return value

    @field_validator("public_config")
    @classmethod
    def _require_public_config(cls, value: dict[str, Any]) -> dict[str, Any]:
        _scan_public_config(value)
        return value

    @model_validator(mode="after")
    def _check_pages_and_summary(self) -> ExtractionArtifact:
        indices = [page.page_index for page in self.pages]
        if any(later <= earlier for earlier, later in pairwise(indices)):
            raise ValueError(f"pages must be in strictly increasing order, got {indices!r}")
        if tuple(self.physical_pages) != tuple(indices):
            raise ValueError(
                f"physical_pages {tuple(self.physical_pages)!r} must list the pages it covers "
                f"({tuple(indices)!r})"
            )
        counted: dict[str, int] = {}
        for page in self.pages:
            for block in page.blocks:
                counted[block.normalized_kind] = counted.get(block.normalized_kind, 0) + 1
        counted = dict(sorted(counted.items()))
        if dict(self.statistics.blocks_by_kind) != counted:
            raise ValueError(
                f"statistics.blocks_by_kind {self.statistics.blocks_by_kind!r} disagrees with "
                f"the pages ({counted!r})"
            )
        return self

    @classmethod
    def from_engine(cls, engine: EngineArtifact) -> ExtractionArtifact:
        """Persist an adapter's product, field for field.

        The adapter and this schema describe the same run; the translation is
        mechanical (``kind`` becomes ``raw_kind``, blocks are numbered by their
        position, the adapter's counters move under ``statistics``) and any
        disagreement raises instead of being repaired.
        """
        return cls.model_validate(
            {
                "artifact_version": engine.artifact_version,
                "mapping_version": engine.mapping_version,
                "engine_id": engine.engine_id,
                "engine_version": engine.engine_version,
                "status": engine.status,
                "doc_id": engine.doc_id,
                "pdf_sha256": engine.pdf_sha256,
                "source_sha256": engine.source_sha256,
                "physical_pages": list(engine.physical_pages),
                "render_dpi": engine.render_dpi,
                "render_page_offset": engine.render_page_offset,
                "prompt_sha256": engine.prompt_sha256,
                "public_config": dict(engine.public_config),
                "pages": [
                    {
                        "page_index": page.page_index,
                        "request_page": page.request_page,
                        "printed_folio": page.printed_folio,
                        "running": page.running,
                        "continues_previous": page.continues_previous,
                        "status": page.status,
                        "blocks": [
                            {
                                "order": order,
                                "raw_kind": block.kind,
                                "normalized_kind": block.normalized_kind,
                                "text": block.text,
                                "marker": block.marker,
                                "depth": block.depth,
                                "group": block.group,
                                "bbox": list(block.bbox) if block.bbox is not None else None,
                                "source_id": block.source_id,
                            }
                            for order, block in enumerate(page.blocks)
                        ],
                        "usage": dict(page.usage) if page.usage is not None else None,
                    }
                    for page in engine.pages
                ],
                "statistics": {
                    "blocks_by_kind": dict(engine.blocks_by_kind),
                    "normalization": dict(engine.normalization),
                },
                "notes": list(engine.notes),
            }
        )

    @property
    def content_digest(self) -> str:
        """Content address of this artifact: sha256 of its canonical JSON."""
        return hashlib.sha256(canonical_payload(self).encode("utf-8")).hexdigest()

    @property
    def extraction_fingerprint(self) -> str:
        """Name the *extraction*, not its output.

        Two runs that read the same pages with the same engine and the same
        configuration share this fingerprint even when they transcribed the
        pages differently, which is what a replay check needs; the words are
        addressed by :attr:`content_digest` instead.
        """
        return _digest(
            {
                "artifact_version": self.artifact_version,
                "mapping_version": self.mapping_version,
                "engine_id": self.engine_id,
                "engine_version": self.engine_version,
                "status": self.status,
                "doc_id": self.doc_id,
                "pdf_sha256": self.pdf_sha256,
                "source_sha256": self.source_sha256,
                "physical_pages": list(self.physical_pages),
                "render_dpi": self.render_dpi,
                "render_page_offset": self.render_page_offset,
                "prompt_sha256": self.prompt_sha256,
                "public_config": self.public_config,
            }
        )


def _scan_public_config(config: Mapping[str, Any]) -> None:
    """Reject any credential-shaped key or URL anywhere in a public config."""
    for key, value in config.items():
        name = str(key).strip().lower().replace("-", "_")
        if set(name.split("_")) & _SECRET_WORDS:
            raise ValueError(f"public_config must not carry a credential field like {key!r}")
        if isinstance(value, Mapping):
            _scan_public_config(value)
        elif isinstance(value, Sequence) and not isinstance(value, (str, bytes)):
            for item in value:
                if isinstance(item, Mapping):
                    _scan_public_config(item)
                elif isinstance(item, str):
                    _check_public_value(name, item)
        elif isinstance(value, str):
            _check_public_value(name, value)


def _check_public_value(name: str, value: str) -> None:
    """A value is public only if it is a bare host or a credential-free URL."""
    if value.startswith(("http://", "https://")):
        parts = urlsplit(value)
        if parts.username or parts.password or parts.query or parts.fragment:
            raise ValueError(
                "public_config must not carry a URL with credentials, a query or a fragment"
            )
    if (name == "endpoint_host" or name.endswith("_host")) and not _BARE_HOST_RE.fullmatch(value):
        raise ValueError(f"{name} must be a bare hostname, got {value!r}")


def canonical_payload(artifact: ExtractionArtifact) -> str:
    """Return the deterministic JSON text that defines the artifact's address.

    Keys are sorted and separators are compact, so the same artifact always
    serializes to the same bytes regardless of field order or dict insertion.
    """
    data = artifact.model_dump(mode="json")
    return json.dumps(data, sort_keys=True, ensure_ascii=False, separators=(",", ":"))


def semantic_fingerprint(
    artifact: ExtractionArtifact,
    *,
    builder_id: str,
    builder_version: str | int,
    prompt_sha256: str | None = None,
    thresholds: Mapping[str, int] | None = None,
) -> str:
    """Name a tree-building profile: the extraction plus the builder that read it.

    The builder decides the tree, so its identity (id, version, its own prompt
    digest, its thresholds) belongs in the name alongside
    :attr:`ExtractionArtifact.extraction_fingerprint`. The artifact's *text* is
    deliberately absent: editing a page for readability does not create a new
    profile, while switching engines or render settings does.
    """
    profile = builder_id.strip()
    if not profile:
        raise ValueError("builder_id must be non-blank")
    if isinstance(builder_version, bool) or not isinstance(builder_version, (str, int)):
        raise ValueError(
            f"builder_version must be an int or a non-blank string, got {builder_version!r}"
        )
    version = builder_version.strip() if isinstance(builder_version, str) else builder_version
    if version == "":
        raise ValueError("builder_version must be non-blank")
    if prompt_sha256 is not None and not SHA256_PATTERN.fullmatch(prompt_sha256):
        raise ValueError(
            f"prompt_sha256 must be 64 lowercase hex characters, got {prompt_sha256!r}"
        )
    cleaned: dict[str, int] = {}
    for key, value in (thresholds or {}).items():
        name = str(key).strip()
        if not name:
            raise ValueError("threshold names must be non-blank")
        if type(value) is not int:
            raise ValueError(f"threshold {name!r} must be an int, got {value!r}")
        cleaned[name] = value
    return _digest(
        {
            "extraction_fingerprint": artifact.extraction_fingerprint,
            "builder_id": profile,
            "builder_version": version,
            "prompt_sha256": prompt_sha256,
            "thresholds": cleaned,
        }
    )
