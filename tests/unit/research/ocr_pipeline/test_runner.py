"""Offline tests for the S4/S5 study runner.

The runner is the only place a study actually runs, so these tests drive it
through its real seams -- injected adapter modules, an injected provenance and an
injected merge client -- and then assert on what it recorded. Nothing here opens
a network connection, imports an engine package or reads a real book: the PDFs are
synthetic, the engines are doubles, and the only credentials are never built.

The three properties that matter most are each pinned below:

* a replay of a stored artifact calls no engine at all;
* a run that failed, was not available, or was merged with a fake keeps its own
  denominator instead of disappearing;
* nothing opens a paid or live path until the caller has said so explicitly.
"""

from __future__ import annotations

import csv
import hashlib
import io
import json
import os
import sys
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, replace
from pathlib import Path
from typing import Any

import pytest
from merge_double import FakeMergeClient

from research.ocr_pipeline.corpus import CorpusError, load_manifest, sha256_file
from research.ocr_pipeline.engines import (
    STATUS_OK,
    STATUS_UNSUPPORTED_MODALITY,
    EngineArtifact,
    EngineBlock,
    EngineNotAvailableError,
    EnginePage,
    content_pages,
)
from research.ocr_pipeline.engines.worker import ENGINE_PYTHON_ENV
from research.ocr_pipeline.models import PageUsage
from research.ocr_pipeline.report import find_raw_text_keys
from research.ocr_pipeline.runner import (
    KIND_EXTRACT,
    KIND_TREE,
    RUN_FAILED,
    RUN_NOT_AVAILABLE,
    RUN_OK,
    MergeNotAllowedError,
    RunnerError,
    RunRef,
    StudyRunner,
    UploadNotAllowedError,
    engine_pages,
    parse_page_selection,
)
from research.ocr_pipeline.usage import artifact_usage
from workflows.enrich_knowledge_tree.detail_refiner import DetailRefiner
from workflows.enrich_knowledge_tree.models import RefineConfig
from workflows.ingest_file.vision_ocr import PROMPT, VisionOCRError

PROVENANCE: dict[str, Any] = {
    "git_commit": "0" * 40,
    "git_dirty": False,
    "python": "3.11.0",
    "platform": "test",
    "label": "runner tests",
}
"""A provenance stand-in: the git probe must never run inside a test."""

REFINE = RefineConfig(
    refine_details=True, llm_model="a-double-not-a-model", max_concurrent_sections=1
)
"""A merge configuration that is fully offline: the client is the runner's fake.

One section at a time, so a fixture that fails the first calls and meters the
later ones lands on the same sections whatever the machine is doing. The walk's
own overlap has its own tests (``test_merge_concurrency.py``); these are about
what a merge costs and what a tree keeps, not about when it called.
"""


# --------------------------------------------------------------------------- #
# synthetic corpus
# --------------------------------------------------------------------------- #


def make_pdf(page_count: int, *, text_pages: tuple[int, ...] = ()) -> bytes:
    """A valid PDF whose ``text_pages`` carry a text object.

    A page with no text object is what a scanned page looks like to pypdfium2:
    the text layer exists and is empty. Kept local so this module owns its
    fixtures and no two suites share mutable state.
    """
    objects: list[bytes] = []
    font_number = 3 + page_count
    content_numbers = {index: font_number + 1 + index for index in range(page_count)}
    kids = " ".join(f"{3 + index} 0 R" for index in range(page_count))
    objects.append(b"<< /Type /Catalog /Pages 2 0 R >>")
    objects.append(f"<< /Type /Pages /Kids [{kids}] /Count {page_count} >>".encode("ascii"))
    for index in range(page_count):
        contents = f"/Contents {content_numbers[index]} 0 R " if index in text_pages else ""
        objects.append(
            f"<< /Type /Page /Parent 2 0 R /MediaBox [0 0 612 792] "
            f"/Resources << /Font << /F1 {font_number} 0 R >> >> {contents}>>".encode("ascii")
        )
    objects.append(b"<< /Type /Font /Subtype /Type1 /BaseFont /Helvetica >>")
    for index in range(page_count):
        stream = b"BT /F1 24 Tf 72 700 Td (Trang mot) Tj ET" if index in text_pages else b""
        objects.append(b"<< /Length %d >>\nstream\n" % len(stream) + stream + b"\nendstream")

    document = b"%PDF-1.4\n"
    offsets: list[int] = []
    for number, body in enumerate(objects, start=1):
        offsets.append(len(document))
        document += f"{number} 0 obj\n".encode("ascii") + body + b"\nendobj\n"
    xref_offset = len(document)
    document += f"xref\n0 {len(objects) + 1}\n".encode("ascii")
    document += b"0000000000 65535 f \n"
    for offset in offsets:
        document += f"{offset:010d} 00000 n \n".encode("ascii")
    document += (
        f"trailer\n<< /Size {len(objects) + 1} /Root 1 0 R >>\nstartxref\n{xref_offset}\n%%EOF\n"
    ).encode("ascii")
    return document


def write_pdf(
    directory: Path, name: str, page_count: int, *, text_pages: tuple[int, ...] = ()
) -> Path:
    """Write a synthetic PDF and return its path."""
    directory.mkdir(parents=True, exist_ok=True)
    path = directory / name
    path.write_bytes(make_pdf(page_count, text_pages=text_pages))
    return path


def document(**overrides: Any) -> dict[str, Any]:
    """One document entry for a corpus manifest."""
    data: dict[str, Any] = {
        "doc_id": "digital-1",
        "pdf_path": "digital.pdf",
        "pdf_sha256": "ab" * 32,
        "page_indices": [0, 1, 2],
        "modality": "digital",
        "permission_note": "synthetic pages, so there is nothing to license",
        "gold_path": None,
    }
    data.update(overrides)
    return data


def write_manifest(
    directory: Path, documents: list[dict[str, Any]], *, name: str = "manifest.json"
) -> Path:
    """Write a corpus manifest and return its path."""
    directory.mkdir(parents=True, exist_ok=True)
    path = directory / name
    path.write_text(json.dumps({"corpus_version": 1, "documents": documents}), encoding="utf-8")
    return path


def heading_text(index: int) -> str:
    return f"Section {index + 1}"


def body_text(doc_id: str, index: int) -> str:
    return f"body of {doc_id} page {index}"


def extra_text(doc_id: str, index: int) -> str:
    return f"more of {doc_id} page {index}"


def gold_payload(doc_id: str, page_indices: Sequence[int]) -> dict[str, Any]:
    """A checked transcript whose text is close to, but not equal to, the engine's.

    The engine's hypothesis is the content blocks joined; the reference has one
    extra word, so every page scores a real (small) error rate instead of a perfect
    zero. The heading each page lists is where the gold says a section starts, so
    the same file scores the run's heading placement.
    """
    return {
        "doc_id": doc_id,
        "pages": [
            {
                "page_index": index,
                "text": (
                    f"{heading_text(index)}\n{body_text(doc_id, index)}\n"
                    f"{extra_text(doc_id, index)} checked"
                ),
                "headings": [heading_text(index)],
            }
            for index in page_indices
        ],
    }


@dataclass(frozen=True)
class Corpus:
    """The synthetic corpus on disk, plus the digests its manifest declares."""

    manifest_path: Path
    base_dir: Path
    pdf_sha256: dict[str, str]


def study_root(corpus: Corpus) -> Path:
    """Where this corpus's runs live: inside the fixture, never in the repo."""
    return corpus.base_dir / "study"


@pytest.fixture
def corpus(tmp_path: Path) -> Corpus:
    """Three heterogeneous documents: digital, scan and mixed.

    ``digital-1`` is the only one with a checked reference, and the only one with
    a page count of three.
    """
    digital = write_pdf(tmp_path, "digital.pdf", 3, text_pages=(0, 1, 2))
    scan = write_pdf(tmp_path, "scan.pdf", 2)
    mixed = write_pdf(tmp_path, "mixed.pdf", 4, text_pages=(0, 2))
    gold = tmp_path / "gold-digital.json"
    gold.write_text(json.dumps(gold_payload("digital-1", [0, 1, 2])), encoding="utf-8")
    documents = [
        document(
            doc_id="digital-1",
            pdf_path="digital.pdf",
            pdf_sha256=sha256_file(digital),
            page_indices=[0, 1, 2],
            modality="digital",
            gold_path="gold-digital.json",
        ),
        document(
            doc_id="scan-1",
            pdf_path="scan.pdf",
            pdf_sha256=sha256_file(scan),
            page_indices=[0, 1],
            modality="scan",
        ),
        document(
            doc_id="mixed-1",
            pdf_path="mixed.pdf",
            pdf_sha256=sha256_file(mixed),
            page_indices=[0, 1, 2, 3],
            modality="mixed",
        ),
    ]
    manifest = load_manifest(write_manifest(tmp_path, documents))
    return Corpus(
        manifest_path=tmp_path / "manifest.json",
        base_dir=tmp_path,
        pdf_sha256={spec.doc_id: spec.pdf_sha256 for spec in manifest.documents},
    )


def make_runner(
    corpus: Corpus,
    engines: Mapping[str, Any] | None = None,
    *,
    root: Path | None = None,
    settings: Any | None = None,
    merge_concurrency: int | None = None,
) -> StudyRunner:
    """A runner over the synthetic corpus with its engines injected."""
    extra = {} if merge_concurrency is None else {"merge_concurrency": merge_concurrency}
    return StudyRunner(
        load_manifest(corpus.manifest_path),
        base_dir=corpus.base_dir,
        root=root if root is not None else study_root(corpus),
        engines=engines,
        provenance=PROVENANCE,
        settings=settings,
        **extra,
    )


@pytest.fixture(autouse=True)
def _no_merge_retry_backoff(monkeypatch: pytest.MonkeyPatch) -> None:
    """Keep the merge double's dead calls from sleeping the suite away.

    The refiner retries a failed call with exponential backoff -- a second or
    two per retry -- so a test modelling an unreachable gateway would spend most
    of its time asleep. Only the wait is replaced: every attempt is still made
    and still metered, which is what those tests assert.
    """

    async def no_wait(_seconds: float) -> None:
        return None

    monkeypatch.setattr(DetailRefiner._invoke_with_retry.retry, "sleep", no_wait)


# --------------------------------------------------------------------------- #
# engine doubles
# --------------------------------------------------------------------------- #


class FakeEngine:
    """A local adapter double, in the call shape ``_call_engine`` uses.

    It never opens the PDF: the subject of these tests is what the runner did
    with an engine's answer, so the double records every call and hands back one
    page per requested index, with a heading, two paragraphs and a folio. The
    usage a page reports carries the counters the engine protocol defines --
    attempts, prompt and completion tokens -- and nothing else.
    """

    def __init__(
        self,
        engine_id: str = "native",
        *,
        kinds: Mapping[str, str] | None = None,
        usage: bool = False,
        groups: bool = True,
        version: str = "0.1-fake",
        model: str | None = None,
        unsupported: bool = False,
        unsupported_pages: Sequence[int] = (),
    ) -> None:
        self.engine_id = engine_id
        self.kinds = dict(
            kinds or {"heading": "SectionHeader", "p": "Text", "footer": "PageFooter"}
        )
        self.usage = usage
        self.groups = groups
        self.version = version
        self.model = model
        self.unsupported = unsupported
        self.unsupported_pages = set(unsupported_pages)
        self.available = True
        self.available_detail = "a local double, always ready"
        self.varying = False
        self.error: Exception | None = None
        self.not_available = False
        self.probe_error: Exception | None = None
        self.calls: list[dict[str, Any]] = []
        self.clients: list[Any] = []
        self.configs: list[Any] = []
        self.last_pages: tuple[EnginePage, ...] = ()

    # -- the adapter surface --------------------------------------------- #

    def probe(self) -> Any:
        if self.probe_error is not None:
            raise self.probe_error
        from research.ocr_pipeline.engines import EngineProbe

        return EngineProbe(
            engine_id=self.engine_id,
            available=self.available,
            version=self.version,
            detail=self.available_detail,
        )

    def extract(
        self,
        source: Any,
        *,
        doc_id: str,
        page_offset: int = 0,
        page_indices: Sequence[int] = (),
        converter: Any | None = None,
    ) -> EngineArtifact:
        """The plain adapter shape: a PDF in, an artifact out."""
        self._record(source, doc_id=doc_id, page_offset=page_offset, page_indices=page_indices)
        return self._artifact(doc_id, list(page_indices), page_offset=page_offset)

    # -- recording and failure ------------------------------------------- #

    def _record(
        self, source: Any, *, doc_id: str, page_offset: int, page_indices: Sequence[int]
    ) -> None:
        """Record the call, then fail it if this double was told to."""
        self.calls.append(
            {
                "source": str(source),
                "doc_id": doc_id,
                "offset": page_offset,
                "pages": list(page_indices),
                "ordinal": len(self.calls) + 1,
            }
        )
        if self.error is not None:
            raise self.error
        if self.not_available:
            raise EngineNotAvailableError(self.engine_id, "not installed in this test")

    def _artifact(
        self,
        doc_id: str,
        page_indices: Sequence[int],
        *,
        page_offset: int,
        dpi: int | None = None,
        public_config: Mapping[str, Any] | None = None,
        engine_version: str | None = None,
    ) -> EngineArtifact:
        pages = tuple(
            self._page(doc_id, index, page_offset=page_offset, ordinal=len(self.calls))
            for index in page_indices
        )
        self.last_pages = pages
        declared = public_config
        if declared is None:
            declared = {"model": self.model} if self.model else {}
        return EngineArtifact(
            engine_id=self.engine_id,
            engine_version=engine_version or self.version,
            doc_id=doc_id,
            pages=pages,
            render_dpi=dpi,
            render_page_offset=page_offset,
            public_config=dict(declared),
            normalization={"fake_pages": len(pages)},
        )

    def _page(self, doc_id: str, index: int, *, page_offset: int, ordinal: int) -> EnginePage:
        if self.unsupported or index in self.unsupported_pages:
            # What a scanned page looks like to a text-layer engine: no blocks at
            # all, and an explicit status instead of an empty page.
            return EnginePage(
                page_index=index,
                request_page=index + 1 + page_offset,
                status=STATUS_UNSUPPORTED_MODALITY,
            )
        group = index + 1 if self.groups else None
        suffix = f" (run {ordinal})" if self.varying else ""
        blocks = (
            EngineBlock(
                kind=self.kinds["heading"],
                normalized_kind="heading",
                text=heading_text(index),
                depth=1,
                group=group,
                source_id=f"{index}:heading",
            ),
            EngineBlock(
                kind=self.kinds["p"],
                normalized_kind="p",
                text=f"{body_text(doc_id, index)}{suffix}",
                group=group,
                source_id=f"{index}:body",
            ),
            EngineBlock(
                kind=self.kinds["p"],
                normalized_kind="p",
                text=extra_text(doc_id, index),
                group=group,
                source_id=f"{index}:extra",
            ),
            EngineBlock(
                kind=self.kinds["footer"],
                normalized_kind="page_footer",
                text=str(index + 1),
                group=group,
                source_id=f"{index}:footer",
            ),
        )
        return EnginePage(
            page_index=index,
            request_page=index + 1 + page_offset,
            blocks=blocks,
            running=f"{doc_id} running head",
            printed_folio=str(index + 1),
            usage=(
                {"attempts": 1, "prompt_tokens": 100, "completion_tokens": 20}
                if self.usage
                else None
            ),
        )


class FakeVisionEngine(FakeEngine):
    """The vision adapter shape: async, with a config, a client and a DPI."""

    def __init__(self, **overrides: Any) -> None:
        overrides.setdefault("kinds", {"heading": "heading", "p": "p", "footer": "page_footer"})
        overrides.setdefault("usage", True)
        super().__init__("vision", **overrides)

    async def extract(  # type: ignore[override]
        self,
        pdf_path: Path,
        *,
        doc_id: str,
        config: Any,
        client: Any,
        dpi: int,
        page_offset: int = 0,
        page_indices: Sequence[int] = (),
        transport: Any = None,
        sleep: Any = None,
    ) -> EngineArtifact:
        self._record(pdf_path, doc_id=doc_id, page_offset=page_offset, page_indices=page_indices)
        self.configs.append(config)
        self.clients.append(client)
        return self._artifact(
            doc_id,
            list(page_indices),
            page_offset=page_offset,
            dpi=dpi,
            public_config={"model": config.model},
            engine_version=config.model,
        )


class StubVisionConfig:
    """The allowlisted fields the runner may copy out of a configuration.

    Mirrors the real :class:`~workflows.ingest_file.vision_ocr.VisionOCRConfig`
    field for field, minus everything that must never reach a manifest: the key,
    the endpoint URL and the headers.
    """

    model = "fake-vision-1"
    host = "vision.invalid"
    image_detail = "high"
    temperature = 0.0
    max_tokens = 8000
    reasoning_effort = None
    max_blocks_per_page = 400


class StubSettings:
    """The project settings a vision configuration is derived from.

    Only the field names ``VisionOCRConfig.from_settings`` reads are here, and the
    endpoint is a reserved-TLD host, so deriving a configuration needs no key and
    reaches nothing.
    """

    VISION_API_BASE = "https://vision.invalid/v1"
    VISION_MODEL = "fake-vision-1"
    VISION_API_KEY = None
    VISION_ALLOWED_HOSTS = "vision.invalid"
    VISION_HEADERS: dict[str, Any] = {}
    VISION_IMAGE_DETAIL = "high"
    VISION_TEMPERATURE = 0.0
    VISION_MAX_TOKENS = 8000
    VISION_REASONING_EFFORT = None
    VISION_MAX_BLOCKS_PER_PAGE = 400


# --------------------------------------------------------------------------- #
# corpus validation
# --------------------------------------------------------------------------- #


def test_validate_reports_every_declared_document(corpus: Corpus) -> None:
    reports = make_runner(corpus).validate()

    assert [report["doc_id"] for report in reports] == ["digital-1", "scan-1", "mixed-1"]
    assert [report["page_indices"] for report in reports] == [[0, 1, 2], [0, 1], [0, 1, 2, 3]]
    assert [report["page_count"] for report in reports] == [3, 2, 4]
    assert reports[0]["pages_with_text"] == [0, 1, 2]
    assert reports[1]["pages_with_text"] == []
    assert reports[2]["pages_with_text"] == [0, 2]
    assert [report["modality"] for report in reports] == ["digital", "scan", "mixed"]
    assert all(report["pdf_sha256"] == corpus.pdf_sha256[report["doc_id"]] for report in reports)
    assert reports[1]["pdf_path"] == str(corpus.base_dir / "scan.pdf")


def test_validate_can_be_narrowed_to_one_document(corpus: Corpus) -> None:
    assert [report["doc_id"] for report in make_runner(corpus).validate(doc_ids=["scan-1"])] == [
        "scan-1"
    ]
    with pytest.raises(RunnerError, match="unknown document"):
        make_runner(corpus).validate(doc_ids=["nope"])


def test_validate_refuses_a_modality_that_contradicts_the_pdf(corpus: Corpus) -> None:
    path = write_manifest(
        corpus.base_dir,
        [
            document(
                doc_id="scan-1",
                pdf_path="scan.pdf",
                pdf_sha256=corpus.pdf_sha256["scan-1"],
                page_indices=[0, 1],
                modality="digital",
            )
        ],
        name="wrong-modality.json",
    )
    runner = StudyRunner(
        load_manifest(path),
        base_dir=corpus.base_dir,
        root=study_root(corpus),
        provenance=PROVENANCE,
    )

    with pytest.raises(CorpusError, match="declares modality 'digital'"):
        runner.validate()


# --------------------------------------------------------------------------- #
# extraction
# --------------------------------------------------------------------------- #


def test_extract_persists_one_artifact_and_one_manifest_per_run(corpus: Corpus) -> None:
    engine = FakeEngine("native")
    runner = make_runner(corpus, {"native": engine})

    run = runner.extract("digital-1", "native", label="first pass")

    assert run["status"] == RUN_OK
    assert run["kind"] == KIND_EXTRACT
    assert run["run_id"] == "digital-1.native.r1"
    assert run["repeat"] == 1
    assert run["label"] == "first pass"
    assert run["reason"] is None
    assert run["error"] is None
    assert run["physical_pages"] == [0, 1, 2]
    assert run["request"]["page_indices"] == [0, 1, 2]
    assert run["request"]["pdf_sha256"] == corpus.pdf_sha256["digital-1"]
    assert run["request"]["config"] == {}
    assert run["public_config"] == {}
    assert run["provenance"] == PROVENANCE
    assert run["usage"] == {
        "usage_schema_version": 1,
        "source": "artifact",
        "pages": 3,
        "pages_with_usage": 0,
        "complete": False,
        "subtotal": {
            "attempts": 0,
            "prompt_tokens": 0,
            "completion_tokens": 0,
            "pages_with_usage": 0,
        },
        "attempts": None,
        "prompt_tokens": None,
        "completion_tokens": None,
        "status": "not_measured",
        "reason": "usage_missing",
    }
    assert run["stages"]["extract"]["status"] == "measured"
    assert isinstance(run["stages"]["extract"]["seconds"], float)
    assert run["engine_version"] == "0.1-fake"
    assert engine.calls == [
        {
            "source": str(corpus.base_dir / "digital.pdf"),
            "doc_id": "digital-1",
            "offset": 0,
            "pages": [0, 1, 2],
            "ordinal": 1,
        }
    ]

    digest = run["artifact_digest"]
    assert isinstance(digest, str) and len(digest) == 64
    directory = study_root(corpus) / "digital-1" / "runs" / run["run_id"]
    assert (
        json.loads((directory / "manifest.json").read_text(encoding="utf-8"))["status"] == RUN_OK
    )
    assert (study_root(corpus) / "digital-1" / "artifacts" / f"{digest}.json").is_file()
    assert not (directory / "tree.json").exists()

    artifact = runner.artifact("digital-1", digest)
    assert artifact.engine_id == "native"
    assert artifact.pdf_sha256 == corpus.pdf_sha256["digital-1"]  # filled from the manifest
    assert artifact.physical_pages == (0, 1, 2)


def test_a_stored_artifact_replays_the_pages_it_was_built_from(corpus: Corpus) -> None:
    engine = FakeEngine("native")
    runner = make_runner(corpus, {"native": engine})
    run = runner.extract("scan-1", "native")

    artifact = runner.artifact("scan-1", run["artifact_digest"])

    # The round trip the whole replay design rests on: what the runner stores can
    # be turned back into exactly the pages the engine returned.
    assert content_pages(engine_pages(artifact)) == content_pages(engine.last_pages)

    stored = json.loads(
        (study_root(corpus) / "scan-1" / "artifacts" / f"{run['artifact_digest']}.json").read_text(
            encoding="utf-8"
        )
    )
    footer = stored["pages"][0]["blocks"][-1]
    assert footer["raw_kind"] == "PageFooter"  # the artifact keeps the engine's own kind
    assert footer["normalized_kind"] == "page_footer"


def test_replay_builds_the_same_tree_with_no_engine_call(corpus: Corpus) -> None:
    engine = FakeEngine("native")
    runner = make_runner(corpus, {"native": engine})
    run = runner.extract("digital-1", "native")

    first = runner.run_tree("digital-1", "native_atomic", extract_run_id=run["run_id"])
    second = runner.run_tree("digital-1", "native_atomic", extract_run_id=run["run_id"], repeat=2)

    assert len(engine.calls) == 1  # both trees replayed the stored artifact
    assert first["status"] == RUN_OK
    assert first["kind"] == KIND_TREE
    assert first["run_id"] == "digital-1.native_atomic.r1.none"
    assert second["run_id"] == "digital-1.native_atomic.r2.none"
    assert first["semantic_signature"] == second["semantic_signature"]
    assert first["semantic_fingerprint"] == second["semantic_fingerprint"]
    assert first["page_signature"] == second["page_signature"]
    assert first["pages"] == [0, 1, 2]
    assert first["artifact_digest"] == run["artifact_digest"]
    assert first["engine_id"] == "native"
    assert first["builder"] == "atomic"
    assert first["rankable"] is True
    assert first["quality"] == "measured"
    assert first["quality_reason"] is None
    assert first["merge_mode"] == "off"
    assert first["stages"]["build"]["status"] == "measured"
    assert first["stages"]["merge"] == {
        "status": "not_measured",
        "seconds": None,
        "reason": "profile_merge_off",
    }
    assert first["tree_nodes"] >= 1

    directory = study_root(corpus) / "digital-1" / "runs" / first["run_id"]
    metrics = json.loads((directory / "metrics.json").read_text(encoding="utf-8"))
    fidelity = metrics["fidelity"]
    # The folio never reaches this comparison: it is furniture, and content_pages
    # drops it before the builder sees the page. Of the content that is left, the
    # atomic builder keeps every character.
    assert fidelity["status"] == "measured"
    assert fidelity["exact"] is True
    assert fidelity["numerator"] == 0
    assert fidelity["artifact_chars"] == fidelity["tree_chars"] > 0
    signature = json.loads((directory / "signature.json").read_text(encoding="utf-8"))
    assert signature["semantic_signature"] == first["semantic_signature"]
    assert signature["builder"] == "atomic"

    resumed = runner.run_tree(
        "digital-1", "native_atomic", extract_run_id=run["run_id"], resume=True
    )
    assert resumed["path"] == first["path"]
    assert len(engine.calls) == 1


def test_repeats_of_a_stochastic_engine_are_separate_runs_and_artifacts(corpus: Corpus) -> None:
    engine = FakeEngine("native", usage=True)
    engine.varying = True
    runner = make_runner(corpus, {"native": engine})

    first = runner.extract("scan-1", "native", repeat=1)
    second = runner.extract("scan-1", "native", repeat=2)

    assert first["run_id"] == "scan-1.native.r1"
    assert second["run_id"] == "scan-1.native.r2"
    assert first["path"] != second["path"]
    assert first["artifact_digest"] != second["artifact_digest"]  # the text really moved
    assert len(engine.calls) == 2
    stored = list((study_root(corpus) / "scan-1" / "artifacts").iterdir())
    assert len(stored) == 2
    assert sorted(entry["repeat"] for entry in runner.list_runs("scan-1", kind=KIND_EXTRACT)) == [
        1,
        2,
    ]


def test_repeats_of_a_deterministic_engine_share_one_artifact_file(corpus: Corpus) -> None:
    engine = FakeEngine("native")
    runner = make_runner(corpus, {"native": engine})

    first = runner.extract("scan-1", "native", repeat=1)
    second = runner.extract("scan-1", "native", repeat=2)

    assert first["artifact_digest"] == second["artifact_digest"]
    assert first["path"] != second["path"]  # two runs were recorded ...
    assert len(engine.calls) == 2  # ... because the engine really ran twice ...
    stored = list((study_root(corpus) / "scan-1" / "artifacts").iterdir())
    assert len(stored) == 1  # ... and the store still holds one artifact


def test_resume_reuses_the_identical_request_and_a_new_one_allocates_an_attempt(
    corpus: Corpus,
) -> None:
    engine = FakeEngine("native")
    runner = make_runner(corpus, {"native": engine})

    first = runner.extract("mixed-1", "native", page_indices=[0, 1])
    again = runner.extract("mixed-1", "native", page_indices=[0, 1], resume=True)

    assert again["path"] == first["path"]
    assert len(engine.calls) == 1  # resume read the record instead of re-running

    different = runner.extract("mixed-1", "native", page_indices=[2, 3])
    assert different["run_id"] == "mixed-1.native.r1-a2"
    assert Path(first["path"]).name == "mixed-1.native.r1"
    assert len(engine.calls) == 2


def test_resume_reuses_a_complete_attempt_under_a_suffix(corpus: Corpus) -> None:
    """A run that had to take ``-a2`` is still the record a resume reuses.

    The id itself was taken by an older request, so the run in hand was written
    beside it. Resuming must read that attempt, not pay the engine to rebuild
    what is already on disk.
    """
    engine = FakeEngine("native")
    runner = make_runner(corpus, {"native": engine})

    runner.extract("mixed-1", "native", page_indices=[2, 3])  # the id, another request
    attempt = runner.extract("mixed-1", "native", page_indices=[0, 1])
    assert attempt["run_id"] == "mixed-1.native.r1-a2"

    resumed = runner.extract("mixed-1", "native", page_indices=[0, 1], resume=True)

    assert resumed["run_id"] == "mixed-1.native.r1-a2"
    assert resumed["path"] == attempt["path"]
    assert len(engine.calls) == 2  # the record was read, not rebuilt


def test_a_resume_takes_the_lowest_attempt_that_answers_the_request(corpus: Corpus) -> None:
    """Two attempts answer one request; the resume takes the earlier one.

    The engine is stochastic, so ``-a2`` and ``-a3`` are different records of the
    same request. Handing back the later one would move a run id a report may
    already name, so the earliest complete answer is the one reused.
    """
    engine = FakeEngine("native")
    engine.varying = True
    runner = make_runner(corpus, {"native": engine})

    runner.extract("mixed-1", "native", page_indices=[2, 3])  # the id, another request
    earliest = runner.extract("mixed-1", "native", page_indices=[0, 1])
    later = runner.extract("mixed-1", "native", page_indices=[0, 1])
    assert (earliest["run_id"], later["run_id"]) == (
        "mixed-1.native.r1-a2",
        "mixed-1.native.r1-a3",
    )
    assert earliest["artifact_digest"] != later["artifact_digest"]

    resumed = runner.extract("mixed-1", "native", page_indices=[0, 1], resume=True)

    assert resumed["run_id"] == "mixed-1.native.r1-a2"
    assert resumed["artifact_digest"] == earliest["artifact_digest"]
    assert len(engine.calls) == 3


def test_a_resume_walks_past_a_failed_attempt_but_never_into_one(corpus: Corpus) -> None:
    """The search for a reusable attempt passes a failure instead of handing it back.

    ``-a2`` holds this request and failed; ``-a3`` holds it and finished. The
    resume reuses the finished one -- and would rather allocate a new attempt
    than return a run that did not succeed.
    """
    failing = FakeEngine("native")
    failing.error = RuntimeError("the engine fell over")
    runner = make_runner(corpus, {"native": failing})

    runner.extract("mixed-1", "native", page_indices=[2, 3])  # the id, another request
    failed = runner.extract("mixed-1", "native", page_indices=[0, 1])
    assert (failed["run_id"], failed["status"]) == ("mixed-1.native.r1-a2", RUN_FAILED)

    good = FakeEngine("native")
    runner = make_runner(corpus, {"native": good})
    complete = runner.extract("mixed-1", "native", page_indices=[0, 1])
    assert complete["run_id"] == "mixed-1.native.r1-a3"

    resumed = runner.extract("mixed-1", "native", page_indices=[0, 1], resume=True)

    assert resumed["run_id"] == "mixed-1.native.r1-a3"
    assert resumed["path"] == complete["path"]
    assert len(good.calls) == 1


def test_a_dry_run_names_the_attempt_the_run_will_reuse(corpus: Corpus) -> None:
    """The plan and the run that follows it name one record, or the plan lies.

    The line a dry run prints is the promise the study keeps: a resume that would
    read ``-a2`` must say ``-a2``, not the id it would have allocated.
    """
    engine = FakeEngine("native")
    runner = make_runner(corpus, {"native": engine})
    runner.extract("mixed-1", "native", page_indices=[2, 3])
    attempt = runner.extract("mixed-1", "native", page_indices=[0, 1])

    planned = runner.plan(
        doc_ids=["mixed-1"], engine_ids=("native",), page_indices=[0, 1], resume=True
    )
    line = next(
        entry
        for entry in planned["extractions"]
        if entry["doc_id"] == "mixed-1" and entry["repeat"] == 1
    )
    assert (line["run_id"], line["cache"], line["cache_reason"]) == (
        "mixed-1.native.r1-a2",
        "hit",
        None,
    )

    landed = runner.extract("mixed-1", "native", page_indices=[0, 1], resume=True)

    assert landed["run_id"] == line["run_id"]
    assert landed["path"] == attempt["path"]
    assert len(engine.calls) == 2  # the plan read the record and the run reused it


def test_extract_refuses_an_artifact_missing_requested_pages(corpus: Corpus) -> None:
    class PartialEngine(FakeEngine):
        def _artifact(
            self, doc_id: str, page_indices: list[int], *, page_offset: int
        ) -> EngineArtifact:
            return super()._artifact(doc_id, page_indices[:1], page_offset=page_offset)

    engine = PartialEngine("native")
    runner = make_runner(corpus, {"native": engine})

    with pytest.raises(CorpusError, match="requested pages"):
        runner.extract("digital-1", "native", page_indices=[0, 2])

    assert not runner.list_runs("digital-1")
    assert not (study_root(corpus) / "digital-1" / "artifacts").exists()


def test_the_request_key_covers_the_prompt_the_config_and_the_engine_version(
    corpus: Corpus,
) -> None:
    engine = FakeVisionEngine()
    runner = make_runner(corpus, {"vision": engine})

    run = runner.extract("scan-1", "vision", allow_upload=True, vision_config=StubVisionConfig())

    request = run["request"]
    assert request["pdf_sha256"] == corpus.pdf_sha256["scan-1"]
    # The prompt is part of the request: a reworded prompt is a different run.
    assert request["prompt_sha256"] == hashlib.sha256(PROMPT.encode("utf-8")).hexdigest()
    assert request["engine_version"] == "fake-vision-1"
    # The whole resolved request configuration is in the key, not just the model:
    # a changed temperature or token ceiling is a different request.
    assert request["config"] == {
        "model": "fake-vision-1",
        "endpoint_host": "vision.invalid",
        "image_detail": "high",
        "temperature": 0.0,
        "max_tokens": 8000,
        "max_blocks_per_page": 400,
        "dpi": 200,
    }
    assert run["public_config"] == request["config"]


def test_a_local_engine_records_no_prompt_and_no_source(corpus: Corpus) -> None:
    engine = FakeEngine("native")
    runner = make_runner(corpus, {"native": engine})

    request = runner.extract("digital-1", "native")["request"]

    assert request["prompt_sha256"] is None  # no prompt is sent anywhere
    assert request["source_sha256"] is None  # every engine reads the PDF itself
    assert request["engine_version"] == "0.1-fake"
    assert request["config"] == {}


def test_resume_refuses_a_run_from_a_different_vision_config(corpus: Corpus) -> None:
    engine = FakeVisionEngine()
    runner = make_runner(corpus, {"vision": engine})
    first = runner.extract("scan-1", "vision", allow_upload=True, vision_config=StubVisionConfig())
    same = runner.extract(
        "scan-1", "vision", allow_upload=True, vision_config=StubVisionConfig(), resume=True
    )
    assert same["path"] == first["path"]
    assert len(engine.calls) == 1

    class OtherConfig(StubVisionConfig):
        """The same endpoint, asked for a different number of tokens."""

        max_tokens = 4000

    changed = runner.extract(
        "scan-1", "vision", allow_upload=True, vision_config=OtherConfig(), resume=True
    )

    assert changed["path"] != first["path"]
    assert changed["run_id"] == "scan-1.vision.r1-a2"
    assert changed["request"]["config"]["max_tokens"] == 4000
    assert len(engine.calls) == 2  # a different request is never silently reused


def test_resume_refuses_a_run_from_a_different_engine_version(corpus: Corpus) -> None:
    engine = FakeEngine("native")
    runner = make_runner(corpus, {"native": engine})
    first = runner.extract("digital-1", "native")

    engine.version = "0.2-fake"
    second = runner.extract("digital-1", "native", resume=True)

    assert second["path"] != first["path"]
    assert second["run_id"] == "digital-1.native.r1-a2"
    assert second["request"]["engine_version"] == "0.2-fake"
    assert len(engine.calls) == 2


# --------------------------------------------------------------------------- #
# a local engine runs in a worker, and its measurement is the worker's
# --------------------------------------------------------------------------- #


def _worker_available() -> bool:
    """Whether this environment can run the text-layer engine the worker tests use."""
    if not Path("/proc/self/stat").exists():
        return False
    from research.ocr_pipeline.engines import load_engine

    return bool(load_engine("native").probe().available)


requires_worker = pytest.mark.skipif(
    not _worker_available(),
    reason="needs a readable Linux /proc and the text-layer engine (pypdfium2)",
)


@requires_worker
def test_a_local_engine_is_measured_where_it_runs_and_not_where_the_runner_does(
    corpus: Corpus,
) -> None:
    runner = make_runner(corpus)

    run = runner.extract("digital-1", "native")

    assert run["status"] == RUN_OK
    assert run["worker"]["pid"] != os.getpid(), "the run must happen in its own process"
    assert run["worker"]["requests_served"] == 1
    assert run["resources"]["sampling"]["tree_peak"] == 1, "the measured tree is the worker alone"
    assert run["resources"]["rss_peak"]["status"] == "measured"
    assert run["warmth"] == "cold"  # a one-shot worker has no evidence of warmth


@requires_worker
def test_a_repeat_is_not_warm_until_a_worker_reports_that_it_was(corpus: Corpus) -> None:
    runner = make_runner(corpus)

    result = runner.study(
        families=["pipeline"], engines=["native"], doc_ids=["digital-1"], repeats=2
    )
    runs = [runner.read_run("digital-1", entry["run_id"]) for entry in result["extractions"]]

    assert [run["repeat"] for run in runs] == [1, 2]
    assert [run["warmth"] for run in runs] == ["cold", "cold"]  # a repeat is not evidence
    assert [run["worker"]["model_process_reused"] for run in runs] == [False, False]
    assert len({run["worker"]["pid"] for run in runs}) == 2  # each repeat is its own process


@requires_worker
def test_a_failed_local_worker_keeps_the_resources_it_used(corpus: Corpus) -> None:
    (corpus.base_dir / "digital.pdf").unlink()
    runner = make_runner(corpus)

    run = runner.extract("digital-1", "native")

    assert run["status"] == RUN_FAILED
    assert run["error"] == {"class": "IsolatedWorkerError", "code": "engine_failed"}
    assert run["artifact_digest"] is None
    assert run["worker"]["pid"] != os.getpid()
    assert run["resources"]["rss_peak"]["status"] == "measured"
    assert run["resources"]["sampling"]["samples"] > 0


def test_an_injected_adapter_runs_where_the_caller_put_it(corpus: Corpus) -> None:
    engine = FakeEngine("native")
    runner = make_runner(corpus, {"native": engine})

    run = runner.extract("digital-1", "native")

    # The caller's adapter is a double, and a worker process could not reach it:
    # it runs here, and the manifest says nothing about a worker.
    assert len(engine.calls) == 1
    assert run["worker"] is None


# --------------------------------------------------------------------------- #
# a page the engine could not read
# --------------------------------------------------------------------------- #


def test_a_run_that_read_no_page_is_unsupported_modality_with_counts(corpus: Corpus) -> None:
    engine = FakeEngine("native", unsupported=True)
    runner = make_runner(corpus, {"native": engine})

    run = runner.extract("scan-1", "native")

    # The engine ran and answered. What it answered is that it cannot read this
    # modality: a fact about the document, not a failure of the run.
    assert run["status"] == RUN_OK
    assert run["reason"] is None
    assert run["artifact_status"] == STATUS_UNSUPPORTED_MODALITY
    assert run["unsupported_pages"] == 2
    assert run["physical_pages"] == [0, 1]  # the pages keep their place
    assert run["usage"]["pages"] == 2
    assert run["artifact_digest"] is not None  # and the artifact is still stored


def test_a_run_that_read_some_pages_stays_ok_and_counts_the_rest(corpus: Corpus) -> None:
    engine = FakeEngine("native", unsupported_pages=(1, 3))
    runner = make_runner(corpus, {"native": engine})

    run = runner.extract("mixed-1", "native")

    assert run["artifact_status"] == STATUS_OK
    assert run["unsupported_pages"] == 2
    assert run["physical_pages"] == [0, 1, 2, 3]


def test_a_tree_over_an_all_unsupported_artifact_is_not_rankable(corpus: Corpus) -> None:
    engine = FakeEngine("native", unsupported=True)
    runner = make_runner(corpus, {"native": engine})
    run = runner.extract("scan-1", "native")

    tree = runner.run_tree("scan-1", "native_atomic", extract_run_id=run["run_id"])

    assert tree["status"] == RUN_OK  # the build succeeded; there was nothing to build from
    assert tree["rankable"] is False
    assert tree["quality"] == "not_measured"
    assert tree["quality_reason"] == STATUS_UNSUPPORTED_MODALITY


def test_a_tree_over_a_partly_supported_artifact_stays_rankable(corpus: Corpus) -> None:
    engine = FakeEngine("native", unsupported_pages=(1,))
    runner = make_runner(corpus, {"native": engine})
    run = runner.extract("mixed-1", "native")

    tree = runner.run_tree("mixed-1", "native_atomic", extract_run_id=run["run_id"])

    assert tree["rankable"] is True
    assert tree["quality_reason"] is None


def test_an_all_unsupported_run_is_kept_out_of_a_pair(corpus: Corpus) -> None:
    unread = FakeEngine("native", unsupported=True)
    read = FakeEngine("docling")
    runner = make_runner(corpus, {"native": unread, "docling": read})
    first = runner.extract("scan-1", "native")
    second = runner.extract("scan-1", "docling")

    result = runner.compare(
        "ocr",
        [RunRef("scan-1", first["run_id"]), RunRef("scan-1", second["run_id"])],
        write=False,
    )

    row = result["runs"][0]
    assert row["rankable"] is False
    assert row["quality"] == "not_measured"
    assert row["quality_reason"] == STATUS_UNSUPPORTED_MODALITY
    assert row["artifact_status"] == STATUS_UNSUPPORTED_MODALITY
    assert row["pages_without_text"] == 2  # the unread pages keep their place
    pair = result["report"]["paired_usage"]["pairs"][0]
    assert pair["compatible"] is False
    assert pair["incompatible_reasons"] == ["run_a_unsupported_modality"]
    assert pair["paired_documents"] == 0


def test_an_engine_failure_is_recorded_not_raised(corpus: Corpus) -> None:
    engine = FakeEngine("marker")
    engine.error = RuntimeError("the engine gave up")
    runner = make_runner(corpus, {"marker": engine})

    run = runner.extract("scan-1", "marker")

    assert run["status"] == RUN_FAILED
    assert run["reason"] == "engine_failed"
    assert run["error"] == {"class": "RuntimeError", "code": "engine_failed"}
    assert run["artifact_digest"] is None
    assert run["stages"]["extract"]["status"] == "failed"
    assert run["stages"]["extract"]["reason"] == "engine_failed"
    assert run["path"] == str(study_root(corpus) / "scan-1" / "runs" / "scan-1.marker.r1")
    assert Path(run["path"], "manifest.json").is_file()
    assert not (study_root(corpus) / "scan-1" / "artifacts").exists()
    assert [entry["run_id"] for entry in runner.list_runs("scan-1", kind=KIND_EXTRACT)] == [
        "scan-1.marker.r1"
    ]


def test_an_unavailable_engine_keeps_its_place_as_not_measured(corpus: Corpus) -> None:
    engine = FakeEngine("docling")
    engine.not_available = True
    runner = make_runner(corpus, {"docling": engine})

    run = runner.extract("digital-1", "docling")

    assert run["status"] == RUN_NOT_AVAILABLE
    assert run["reason"] == "engine_not_available"
    assert run["error"] == {"class": "EngineNotAvailableError", "code": "engine_not_available"}
    assert run["artifact_digest"] is None
    assert run["stages"]["extract"]["status"] == "not_measured"
    assert run["physical_pages"] == [0, 1, 2]  # the pages asked for remain the denominator


def test_a_probe_reports_availability_and_never_lets_one_failure_escape(
    corpus: Corpus,
) -> None:
    engine = FakeEngine("marker")
    runner = make_runner(corpus, {"marker": engine})

    assert runner.probe("marker") == {
        "engine_id": "marker",
        "available": True,
        "version": "0.1-fake",
        "detail": "a local double, always ready",
    }

    # A probe that raises is a probe that says no: a study must be able to plan
    # around an engine whose import itself fails.
    engine.probe_error = RuntimeError("import marker failed")
    assert runner.probe("marker") == {
        "engine_id": "marker",
        "available": False,
        "version": None,
        "detail": "probe raised RuntimeError",
    }
    with pytest.raises(RunnerError, match="unknown engine"):
        runner.probe("tesseract")


@requires_worker
def test_a_probe_of_an_engine_kept_elsewhere_is_asked_there_and_only_once(
    corpus: Corpus, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """An engine in an interpreter of its own is asked there -- once per study.

    The engine is not installed *here*, so an answer from here would be about the
    wrong interpreter: the question has to reach the one the engine runs in, and
    the answer has to say which one that was. Asking costs a process, so the
    answer is kept -- however many documents, repeats and profiles then read the
    version it reported, that interpreter is started once.
    """
    starts = tmp_path / "starts.log"
    interpreter = tmp_path / "engine-python"
    interpreter.write_text(
        f'#!/bin/sh\necho start >> "{starts}"\nexec "{sys.executable}" "$@"\n', encoding="utf-8"
    )
    interpreter.chmod(0o755)
    monkeypatch.setenv(ENGINE_PYTHON_ENV, f"native={interpreter}")
    runner = make_runner(corpus)

    first = runner.probe("native")
    second = runner.probe("native")

    assert first == second
    assert first["available"] is True  # the text-layer engine, seen from that interpreter
    assert first["detail"].endswith(f"(asked {interpreter})")
    assert starts.read_text(encoding="utf-8").splitlines() == ["start"]


# --------------------------------------------------------------------------- #
# the two gates: paid uploads and live merges
# --------------------------------------------------------------------------- #


def test_a_paid_engine_is_refused_before_any_call(corpus: Corpus) -> None:
    engine = FakeVisionEngine()
    runner = make_runner(corpus, {"vision": engine})

    with pytest.raises(UploadNotAllowedError, match="--allow-upload"):
        runner.extract("scan-1", "vision")

    assert engine.calls == []
    assert not study_root(corpus).exists()


def test_a_paid_engine_runs_once_the_upload_is_allowed(corpus: Corpus) -> None:
    engine = FakeVisionEngine()
    config = StubVisionConfig()
    client = object()
    runner = make_runner(corpus, {"vision": engine})

    run = runner.extract(
        "scan-1",
        "vision",
        allow_upload=True,
        vision_config=config,
        client=client,
        dpi=144,
    )

    assert run["status"] == RUN_OK
    assert run["public_config"] == {
        "dpi": 144,
        "model": "fake-vision-1",
        "endpoint_host": "vision.invalid",
        "image_detail": "high",
        "temperature": 0.0,
        "max_tokens": 8000,
        "max_blocks_per_page": 400,
    }
    assert run["usage"]["prompt_tokens"] == 200  # two pages at 100 prompt tokens each
    assert run["usage"]["completion_tokens"] == 40
    assert run["usage"]["pages_with_usage"] == 2
    assert engine.clients == [client]
    assert engine.configs[0] is config

    artifact = runner.artifact("scan-1", run["artifact_digest"])
    assert artifact.engine_id == "vision"
    assert artifact.render_dpi == 144


def test_a_live_merge_is_refused_before_the_gateway_client_is_built(
    corpus: Corpus, monkeypatch: pytest.MonkeyPatch
) -> None:
    engine = FakeEngine("marker")
    runner = make_runner(corpus, {"marker": engine})
    run = runner.extract("mixed-1", "marker")

    def refuse(_runner: StudyRunner) -> Any:
        raise AssertionError("the gateway client must not be built")

    monkeypatch.setattr(StudyRunner, "_gateway_client", refuse)
    run_id = "mixed-1.marker_llm.r1.live"

    with pytest.raises(MergeNotAllowedError, match="--allow-merge-upload"):
        runner.run_tree("mixed-1", "marker_llm", extract_run_id=run["run_id"], llm_mode="live")

    assert not (study_root(corpus) / "mixed-1" / "runs" / run_id).exists()


def test_a_live_merge_is_handed_the_client_the_app_factory_builds(
    corpus: Corpus, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A live merge must run on the app's gateway, not a client with no models.

    The defect this guards: the runner built a bare ``AIGatewayClient()``, whose
    registry is empty, so every call raised inside the process and the refiner
    kept the originals -- a study that reached no model looked successful. The
    settings the runner was given are the ones the factory has to be asked with: a
    client built from other settings would answer with another endpoint's models.
    """
    engine = FakeEngine("marker")
    settings = StubSettings()
    runner = make_runner(corpus, {"marker": engine}, settings=settings)
    run = runner.extract("mixed-1", "marker")
    asked: list[Any] = []

    def build(asked_settings: Any) -> Any:
        asked.append(asked_settings)
        return FakeMergeClient()

    monkeypatch.setattr("config.llm_gateway.build_llm_gateway_client", build)

    tree = runner.run_tree(
        "mixed-1",
        "marker_llm",
        extract_run_id=run["run_id"],
        llm_mode="live",
        allow_merge_upload=True,
        refine_config=REFINE,
    )

    assert asked == [settings]
    # The factory's client is the one that merged. A bare client completes no
    # call, so this would stay at zero and every section would keep its originals.
    assert tree["merge"]["llm_calls"] > 0
    assert tree["merge"]["sections_fallback"] == 0


def test_a_tree_refuses_an_artifact_from_another_engine(corpus: Corpus) -> None:
    engine = FakeEngine("native")
    runner = make_runner(corpus, {"native": engine})
    run = runner.extract("scan-1", "native")

    # The run id is checked against the engine the tree reads, before anything is
    # read from it: the id names a record, and the record is what refuses.
    with pytest.raises(RunnerError, match="was extracted by 'native', not 'marker'"):
        runner.run_tree("scan-1", "marker_atomic", extract_run_id=run["run_id"])
    # A bare digest carries no run, so the artifact itself is what refuses.
    with pytest.raises(RunnerError, match="reads engine 'marker'"):
        runner.run_tree("scan-1", "marker_atomic", artifact_digest=run["artifact_digest"])
    with pytest.raises(RunnerError, match="needs an artifact digest or an extract run"):
        runner.run_tree("scan-1", "native_atomic")
    with pytest.raises(RunnerError, match="unknown llm mode"):
        runner.run_tree("scan-1", "native_atomic", extract_run_id=run["run_id"], llm_mode="maybe")


# --------------------------------------------------------------------------- #
# comparison
# --------------------------------------------------------------------------- #


def test_compare_keeps_a_failed_run_and_a_missing_gold_in_the_denominator(corpus: Corpus) -> None:
    engine = FakeEngine("native")
    runner = make_runner(corpus, {"native": engine})
    good = runner.extract("digital-1", "native")
    engine.error = RuntimeError("the second document was too much")
    failed = runner.extract("scan-1", "native")

    result = runner.compare(
        "ocr",
        [RunRef("digital-1", good["run_id"]), RunRef("scan-1", failed["run_id"])],
        write=False,
    )

    assert [row["status"] for row in result["runs"]] == [RUN_OK, RUN_FAILED]
    assert result["runs"][1]["artifact"] == {
        "status": "not_measured",
        "reason": "engine_failed",
        "value": None,
    }
    assert result["report"]["metadata"]["runs"] == [good["run_id"], failed["run_id"]]
    assert result["report"]["metadata"]["documents"] == ["digital-1"]
    # The row records the reference it was scored against, which is the digest a
    # pair reads before it will call two runs a comparison.
    assert result["runs"][0]["gold_digest"] is not None
    counts = result["report"]["counts"]
    assert counts["documents"] == 1  # only the document with a reference was scored
    assert counts["documents_with_gold"] == 1
    assert counts["pages_total"] == 3
    assert counts["pages_with_gold"] == 3
    assert counts["limitations"] == len(result["report"]["limitations"])
    assert any("no artifact" in line for line in result["report"]["limitations"])
    assert find_raw_text_keys(result["report"]) == []


def test_compare_without_gold_reports_typed_misses(corpus: Corpus) -> None:
    engine = FakeEngine("native")
    other = FakeEngine("docling")
    runner = make_runner(corpus, {"native": engine, "docling": other})
    first = runner.extract("scan-1", "native", repeat=1)
    second = runner.extract("scan-1", "docling", repeat=1)

    result = runner.compare(
        "ocr",
        [RunRef("scan-1", first["run_id"]), RunRef("scan-1", second["run_id"])],
        write=False,
    )
    report = result["report"]

    assert report["text_metrics"] == {"status": "not_measured", "reason": "no_scored_documents"}
    assert report["counts"]["documents"] is None  # not run, rather than ran and found nothing
    assert report["counts"]["pages_total"] is None
    # Two engines over one document can still be paired, even with nothing to score.
    assert report["paired_usage"]["status"] == "measured"
    assert report["paired_usage"]["paired_documents"] == 1
    assert report["counts"]["paired_documents"] == 1
    assert any("no gold" in line for line in report["limitations"])
    assert any(
        "no run produced an artifact with a usable reference" in line
        for line in report["limitations"]
    )


def test_two_engines_of_one_document_in_a_repeat_pair_into_a_zero_delta(corpus: Corpus) -> None:
    vision = FakeVisionEngine()
    other = FakeEngine("marker", usage=True)
    runner = make_runner(corpus, {"vision": vision, "marker": other})
    first = runner.extract(
        "mixed-1", "vision", allow_upload=True, vision_config=StubVisionConfig()
    )
    second = runner.extract("mixed-1", "marker")

    paired = runner.compare(
        "ocr",
        [RunRef("mixed-1", first["run_id"]), RunRef("mixed-1", second["run_id"])],
        write=False,
    )["report"]["paired_usage"]

    assert paired["status"] == "measured"
    assert paired["paired_documents"] == 1
    assert paired["union_documents"] == 1
    pair = paired["pairs"][0]
    assert pair["compatible"] is True
    assert pair["incompatible_reasons"] == []
    # Tokens and quality are separate deltas: the token delta is its own key so a
    # run that read more can never be mistaken for a worse one.
    delta = pair["usage"]["delta"]
    assert delta["status"] == "measured"
    assert delta["prompt_tokens"] == 0  # both doubles report 100 prompt tokens per page
    assert delta["completion_tokens"] == 0
    assert pair["metrics"]  # the quality deltas travel beside it


def test_pairs_are_built_within_a_repeat_and_never_between_repeats(corpus: Corpus) -> None:
    vision = FakeVisionEngine()
    other = FakeEngine("marker")
    runner = make_runner(corpus, {"vision": vision, "marker": other})
    runs = [
        runner.extract(
            "mixed-1", "vision", repeat=1, allow_upload=True, vision_config=StubVisionConfig()
        ),
        runner.extract(
            "mixed-1", "vision", repeat=2, allow_upload=True, vision_config=StubVisionConfig()
        ),
        runner.extract("mixed-1", "marker", repeat=1),
        runner.extract("mixed-1", "marker", repeat=2),
    ]
    a1, a2, b1, b2 = runs

    # The runs arrive A1, A2, B1, B2. The repeat is the pairing key, so an A is
    # only ever paired with the B measured beside it in the same repeat.
    pairs = runner.compare("ocr", [RunRef("mixed-1", run["run_id"]) for run in runs], write=False)[
        "report"
    ]["paired_usage"]["pairs"]

    assert [(pair["run_a"], pair["run_b"]) for pair in pairs] == [
        (a1["run_id"], b1["run_id"]),
        (a2["run_id"], b2["run_id"]),
    ]


def test_two_runs_of_one_participant_in_a_repeat_are_not_a_pair(corpus: Corpus) -> None:
    engine = FakeEngine("native")
    runner = make_runner(corpus, {"native": engine})
    first = runner.extract("scan-1", "native", repeat=1)
    again = runner.extract("scan-1", "native", repeat=1)  # a second attempt, same request

    assert again["run_id"] == "scan-1.native.r1-a2"
    paired = runner.compare(
        "ocr",
        [RunRef("scan-1", first["run_id"]), RunRef("scan-1", again["run_id"])],
        write=False,
    )["report"]["paired_usage"]

    # Two repeats of one engine measure repeatability, not an A/B difference.
    assert paired["pairs"] == []
    assert paired["status"] == "not_measured"
    assert paired["reason"] == "no_pair"
    assert paired["paired_documents"] == 0


def test_a_pair_is_refused_when_the_page_selection_differs(corpus: Corpus) -> None:
    native = FakeEngine("native")
    dump = FakeEngine("marker")
    runner = make_runner(corpus, {"native": native, "marker": dump})
    left = runner.extract("mixed-1", "native", page_indices=[0, 1])
    right = runner.extract("mixed-1", "marker", page_indices=[2, 3])

    result = runner.compare(
        "ocr",
        [RunRef("mixed-1", left["run_id"]), RunRef("mixed-1", right["run_id"])],
        write=False,
    )
    rows = result["runs"]
    # The rows carry what the pair refused on, so the refusal is about the runs
    # themselves and not about a field the row builder never filled in.
    assert rows[0]["pdf_sha256"] == rows[1]["pdf_sha256"] == corpus.pdf_sha256["mixed-1"]
    assert rows[0]["page_indices"] == [0, 1]
    assert rows[1]["page_indices"] == [2, 3]

    pair = result["report"]["paired_usage"]["pairs"][0]
    assert pair["compatible"] is False
    assert pair["incompatible_reasons"] == ["different_page_selection"]
    assert pair["paired_documents"] == 0


def test_a_pair_is_refused_when_the_pdf_differs(corpus: Corpus) -> None:
    """A guard against a manifest revision that moved under a stored run.

    The row builder already refuses an artifact whose PDF is not the manifest's,
    so this is the second line of defence: two rows that disagree about the bytes
    they read are never a pair. The reference they were scored against is *not*
    part of this: that is evaluation provenance, and it abstains metrics instead
    of refusing the pair (see the gold-provenance tests below).
    """
    runner = make_runner(corpus, {"native": FakeEngine("native")})
    row: dict[str, Any] = {
        "run_id": "digital-1.native.r1",
        "kind": KIND_EXTRACT,
        "doc_id": "digital-1",
        "engine": "native",
        "status": RUN_OK,
        "rankable": True,
        "artifact_digest": "a" * 64,
        "pdf_sha256": "b" * 64,
        "page_indices": [0, 1, 2],
        "gold_digest": "c" * 64,
    }

    assert runner._pairing_reasons(row, row, "ocr") == []
    assert runner._pairing_reasons(row, {**row, "pdf_sha256": "d" * 64}, "ocr") == [
        "different_pdf"
    ]
    assert runner._pairing_reasons(row, {**row, "page_indices": [0]}, "ocr") == [
        "different_page_selection"
    ]


def metric_row(
    run_id: str,
    *,
    doc_id: str = "digital-1",
    cer: float = 0.10,
    wer: float = 0.20,
    boundary_f1: float = 0.50,
    fidelity_omitted: float = 0.05,
    seconds: float = 4.0,
    gold_digest: str = "c" * 64,
) -> dict[str, Any]:
    """A comparison row as the runner builds one, carrying its metric blocks.

    The numbers are chosen so every delta is a clean multiple: b minus a is the
    difference of the arguments, and the denominators travel with it.
    """
    ratio = {
        "status": "measured",
        "reason": None,
        "value": cer,
        "numerator": round(cer * 100),
        "denominator": 100,
        "unit": "ratio",
        "denominator_unit": "characters",
    }
    return {
        "run_id": run_id,
        "kind": KIND_EXTRACT,
        "doc_id": doc_id,
        "engine": run_id,
        "status": RUN_OK,
        "rankable": True,
        "artifact_digest": "a" * 64,
        "pdf_sha256": "b" * 64,
        "page_indices": [0, 1, 2],
        "gold_digest": gold_digest,
        "text_metrics": {
            "rates": {
                "cer": {"micro": ratio, "macro": {**ratio, "denominator_unit": "documents"}},
                "wer": {
                    "micro": {**ratio, "value": wer},
                    "macro": {**ratio, "value": wer, "denominator_unit": "documents"},
                },
            }
        },
        "structure_metrics": {"kinds": {"boundary": {"f1": {**ratio, "value": boundary_f1}}}},
        "fidelity": {**ratio, "value": fidelity_omitted},
        "seconds": {
            "status": "measured",
            "reason": None,
            "value": seconds,
            "numerator": seconds,
            "denominator": 1,
            "unit": "seconds",
            "denominator_unit": "runs",
        },
    }


def test_a_different_gold_is_evaluation_provenance_not_an_incompatibility(
    corpus: Corpus,
) -> None:
    """Two engines that read the same pages stay comparable under two references.

    Gold, annotations and review status describe how a run was *judged*, not
    what it read, so they belong to evaluation provenance. Refusing the pair
    would drop the extraction comparison the study exists to make, and would
    silently shrink the denominator every time a reference was revised.
    """
    runner = make_runner(corpus, {"native": FakeEngine("native")})
    left = metric_row("digital-1.native.r1")
    right = metric_row("digital-1.marker.r1", gold_digest="e" * 64)

    assert runner._pairing_reasons(left, right, "ocr") == []
    assert runner._pairing_reasons(left, {**right, "pdf_sha256": "d" * 64}, "ocr") == [
        "different_pdf"
    ]


def test_a_different_gold_abstains_only_the_metrics_that_read_it(corpus: Corpus) -> None:
    """The gold-dependent deltas abstain; the run's own figures still measure.

    CER and a boundary F1 are statements about a transcript *against a
    reference*: two different references make the difference between them a
    statement about the references. Time, fidelity and the rest describe the run
    itself and are unaffected.
    """
    runner = make_runner(corpus, {"native": FakeEngine("native")})
    left = metric_row("digital-1.native.r1", cer=0.10, seconds=4.0)
    right = metric_row("digital-1.marker.r1", cer=0.20, seconds=6.0)

    same = runner._quality_delta(left, right, [])
    assert same["cer_micro"]["status"] == "measured"
    assert same["cer_micro"]["value"] == pytest.approx(0.10)
    assert same["cer_micro"]["unit"] == "ratio"
    assert same["cer_micro"]["denominator_unit"] == "characters"
    assert same["cer_micro"]["a_numerator"] == 10
    assert same["cer_micro"]["b_denominator"] == 100
    assert same["seconds"]["value"] == pytest.approx(2.0)
    assert same["seconds"]["unit"] == "seconds"

    across = runner._quality_delta(left, {**right, "gold_digest": "e" * 64}, [])
    for name in ("cer_micro", "wer_micro", "cer_macro", "wer_macro", "boundary_f1"):
        assert across[name]["status"] == "not_measured", name
        assert across[name]["reason"] == "different_gold", name
    assert across["fidelity_omitted"]["status"] == "measured"
    assert across["seconds"]["status"] == "measured"
    assert across["seconds"]["value"] == pytest.approx(2.0)


def test_a_pair_records_the_gold_and_annotations_it_was_scored_against(
    corpus: Corpus,
) -> None:
    """The provenance a pair was scored under travels with the pair."""
    runner = make_runner(corpus, {"native": FakeEngine("native")})
    left = metric_row("digital-1.native.r1")
    right = metric_row("digital-1.marker.r1", gold_digest="e" * 64)

    evaluation = runner._pair_evaluation(left, right)

    assert evaluation["same_gold"] is False
    assert evaluation["gold_digests"] == ["c" * 64, "e" * 64]
    assert evaluation["a"]["gold_digest"] == "c" * 64
    assert evaluation["b"]["gold_digest"] == "e" * 64


def test_the_three_report_formats_agree_on_every_count(corpus: Corpus) -> None:
    engine = FakeEngine("native")
    runner = make_runner(corpus, {"native": engine})
    run = runner.extract("digital-1", "native")

    result = runner.compare(
        "ocr",
        [RunRef("digital-1", run["run_id"])],
        out_dir=study_root(corpus) / "comparison",
    )
    counts = result["report"]["counts"]
    assert counts["documents"] == 1
    assert counts["pages_total"] == 3
    assert counts["pages_with_gold"] == 3

    files = result["files"]
    payload = json.loads(Path(files["json"]).read_text(encoding="utf-8"))
    assert payload["counts"] == counts

    rows = {
        row["metric"]: row["value"]
        for row in csv.DictReader(io.StringIO(Path(files["csv"]).read_text(encoding="utf-8")))
        if row["section"] == "counts"
    }
    assert rows["documents"] == "1"
    assert rows["pages_total"] == "3"
    assert rows["pages_with_gold"] == "3"

    markdown = Path(files["markdown"]).read_text(encoding="utf-8")
    assert "| documents | count | 1 |" in markdown
    assert "| pages_total | count | 3 |" in markdown
    assert "| pages_with_gold | count | 3 |" in markdown

    manifest = json.loads(Path(result["path"], "manifest.json").read_text(encoding="utf-8"))
    assert manifest["counts"] == counts
    assert manifest["family"] == "ocr"
    assert manifest["runs"] == [run["run_id"]]


def test_a_comparison_carries_no_transcript_text_in_any_format(corpus: Corpus) -> None:
    engine = FakeEngine("native")
    runner = make_runner(corpus, {"native": engine})
    run = runner.extract("digital-1", "native")

    result = runner.compare(
        "ocr",
        [RunRef("digital-1", run["run_id"])],
        out_dir=study_root(corpus) / "no-text",
    )

    # The transcript lives in the artifact and in the gold reference. A report is
    # numbers, ids and statuses: it may not quote a word of either.
    assert find_raw_text_keys(result["report"]) == []
    written = json.dumps(result["runs"], ensure_ascii=False)
    for name in ("json", "csv", "markdown"):
        written += Path(result["files"][name]).read_text(encoding="utf-8")
    for page in (0, 1, 2):
        assert body_text("digital-1", page) not in written
        assert extra_text("digital-1", page) not in written


def test_a_run_reference_names_a_document_and_its_run(corpus: Corpus) -> None:
    engine = FakeEngine("native")
    runner = make_runner(corpus, {"native": engine})
    run = runner.extract("scan-1", "native")

    ref = RunRef.parse(f"scan-1:{run['run_id']}")
    assert ref == RunRef("scan-1", run["run_id"])
    assert str(ref) == f"scan-1:{run['run_id']}"
    assert runner.compare("ocr", [ref], write=False)["runs"][0]["run_id"] == run["run_id"]

    with pytest.raises(RunnerError, match="must be DOC_ID:RUN_ID"):
        RunRef.parse("scan-1")
    with pytest.raises(RunnerError, match="unknown family"):
        runner.compare("modalities", [ref], write=False)


# --------------------------------------------------------------------------- #
# corpus growth, references and planning
# --------------------------------------------------------------------------- #


def test_a_document_is_added_by_editing_the_manifest_json_only(corpus: Corpus) -> None:
    engine = FakeEngine("native")
    runner = make_runner(corpus, {"native": engine})
    before = runner.compare(
        "ocr",
        [RunRef("digital-1", runner.extract("digital-1", "native")["run_id"])],
        write=False,
    )
    assert before["report"]["counts"]["documents"] == 1

    extra = write_pdf(corpus.base_dir, "extra.pdf", 2, text_pages=(0, 1))
    extra_gold = corpus.base_dir / "gold-extra.json"
    extra_gold.write_text(json.dumps(gold_payload("extra-1", [0, 1])), encoding="utf-8")
    payload = json.loads(corpus.manifest_path.read_text(encoding="utf-8"))
    payload["documents"].append(
        document(
            doc_id="extra-1",
            pdf_path="extra.pdf",
            pdf_sha256=sha256_file(extra),
            page_indices=[0, 1],
            modality="digital",
            gold_path="gold-extra.json",
        )
    )
    corpus.manifest_path.write_text(json.dumps(payload), encoding="utf-8")

    grown = make_runner(corpus, {"native": engine})
    assert [report["doc_id"] for report in grown.validate()] == [
        "digital-1",
        "scan-1",
        "mixed-1",
        "extra-1",
    ]
    after = grown.compare(
        "ocr", [RunRef("extra-1", grown.extract("extra-1", "native")["run_id"])], write=False
    )
    assert after["report"]["counts"]["documents"] == 1
    assert after["report"]["counts"]["pages_total"] == 2


def test_study_dry_run_plans_before_it_runs_anything(corpus: Corpus) -> None:
    engine = FakeEngine("native")
    runner = make_runner(corpus, {"native": engine})

    result = runner.study(
        families=["ocr", "pipeline"], engines=["native"], repeats=2, dry_run=True
    )

    assert result["executed"] is False
    plan = result["plan"]
    assert plan["corpus"]["documents"] == ["digital-1", "scan-1", "mixed-1"]
    assert plan["corpus"]["pages"] == {"digital-1": 3, "scan-1": 2, "mixed-1": 4}
    assert len(plan["extractions"]) == 6  # three documents times two repeats
    assert len(plan["trees"]) == 6
    assert {entry["profile"] for entry in plan["trees"]} == {"native_atomic"}
    assert plan["limitations"] == []
    assert plan["allow_upload"] is False
    assert engine.calls == []
    assert not study_root(corpus).exists()


def test_study_extracts_builds_and_compares_offline(corpus: Corpus) -> None:
    engine = FakeEngine("native")
    runner = make_runner(corpus, {"native": engine})

    result = runner.study(families=["ocr", "pipeline", "chunk"], engines=["native"])

    assert result["executed"] is True
    assert [entry["status"] for entry in result["extractions"]] == [RUN_OK] * 3
    assert len(engine.calls) == 3  # one extraction per document; the trees replayed them
    assert [entry["profile"] for entry in result["trees"]] == ["native_atomic"] * 3
    assert all(entry["rankable"] is True for entry in result["trees"])
    assert [report["doc_id"] for report in result["corpus"]] == ["digital-1", "scan-1", "mixed-1"]
    # The chunk family needs a vision run, so it plans nothing when only native ran.
    assert {entry["family"] for entry in result["comparisons"]} == {"ocr", "pipeline"}
    assert len(result["comparisons"]) == 6
    # Only the digital document has a reference, and it is scored in each of its families.
    assert sum(entry["counts"]["documents"] or 0 for entry in result["comparisons"]) == 2
    assert all(entry["path"] for entry in result["comparisons"])


def test_a_study_resumes_instead_of_paying_for_the_same_extraction_twice(corpus: Corpus) -> None:
    engine = FakeVisionEngine()
    runner = make_runner(corpus, {"vision": engine}, settings=StubSettings())

    first = runner.study(families=["ocr"], engines=["vision"], allow_upload=True)
    paid = len(engine.calls)
    assert first["executed"] is True
    assert paid == 3  # three documents, one repeat

    second = runner.study(families=["ocr"], engines=["vision"], allow_upload=True)

    assert len(engine.calls) == paid  # the identical request was replayed, not repaid
    assert [entry["run_id"] for entry in second["extractions"]] == [
        entry["run_id"] for entry in first["extractions"]
    ]


def test_a_study_refuses_a_live_merge_before_it_pays_for_an_extraction(corpus: Corpus) -> None:
    engine = FakeVisionEngine()
    runner = make_runner(corpus, {"vision": engine}, settings=StubSettings())

    with pytest.raises(MergeNotAllowedError, match="--allow-merge-upload"):
        runner.study(families=["pipeline"], engines=["vision"], allow_upload=True, llm_mode="live")

    assert engine.calls == []  # nothing was uploaded before the refusal
    assert not study_root(corpus).exists()


def test_a_study_rejects_a_profile_its_llm_mode_would_drop(corpus: Corpus) -> None:
    engine = FakeVisionEngine()
    runner = make_runner(corpus, {"vision": engine}, settings=StubSettings())

    with pytest.raises(RunnerError, match="vision_atomic_merge"):
        runner.study(
            families=["pipeline"],
            engines=["vision"],
            profiles=["vision_atomic_merge"],
            allow_upload=True,
        )

    assert engine.calls == []  # an explicitly named profile is never silently dropped


def test_a_study_records_why_a_family_was_skipped(corpus: Corpus) -> None:
    engine = FakeEngine("native")
    engine.error = RuntimeError("the engine gave up")
    runner = make_runner(corpus, {"native": engine})

    result = runner.study(families=["ocr"], engines=["native"])

    assert result["comparisons"] == []
    assert result["skipped"] == [
        {"family": "ocr", "doc_id": doc_id, "reason": "no_successful_extraction"}
        for doc_id in ("digital-1", "scan-1", "mixed-1")
    ]


def test_a_study_names_the_family_it_had_no_tree_for(corpus: Corpus) -> None:
    engine = FakeEngine("native")
    runner = make_runner(corpus, {"native": engine})

    result = runner.study(families=["ocr", "chunk"], engines=["native"])

    assert {entry["family"] for entry in result["comparisons"]} == {"ocr"}
    assert [entry["family"] for entry in result["skipped"]] == ["chunk"] * 3
    assert {entry["reason"] for entry in result["skipped"]} == {"no_chunk_profile_tree"}


# --------------------------------------------------------------------------- #
# a family list that is OCR and nothing else
# --------------------------------------------------------------------------- #


def test_an_ocr_only_plan_builds_no_trees(corpus: Corpus) -> None:
    """Trees exist to be compared under a reference; with OCR alone there is none.

    A plan that names only the extraction family lists its three extractions and
    no tree at all -- not a tree per document that the study would then refuse
    to compare. The absence is stated, not implied: the trees list is empty, the
    profile list is empty with it, and no limitation stands in their place.
    """
    engine = FakeEngine("native")
    runner = make_runner(corpus, {"native": engine})

    plan = runner.plan(families=["ocr"], engine_ids=("native",))

    assert plan["families"] == ["ocr"]
    assert len(plan["extractions"]) == 3  # one per document
    assert plan["trees"] == []
    assert plan["profiles"] == []
    assert plan["limitations"] == []
    assert engine.calls == []


def test_an_ocr_only_plan_rejects_an_explicit_profile(corpus: Corpus) -> None:
    """A profile names a tree to build; there is no tree to build here.

    Silently dropping it would answer a request the caller did not make, so the
    plan refuses in the same words a study uses.
    """
    runner = make_runner(corpus, {"native": FakeEngine("native")})

    with pytest.raises(RunnerError, match="only applies when a study builds trees"):
        runner.plan(families=["ocr"], engine_ids=("native",), profile_names=("native_atomic",))


def test_an_ocr_only_study_rejects_an_explicit_profile_before_it_extracts(
    corpus: Corpus,
) -> None:
    """The refusal comes before anything is uploaded or billed."""
    engine = FakeEngine("native")
    runner = make_runner(corpus, {"native": engine})

    with pytest.raises(RunnerError, match="only applies when a study builds trees"):
        runner.study(families=["ocr"], engines=["native"], profiles=["native_atomic"])

    assert engine.calls == []
    assert not study_root(corpus).exists()


def test_an_ocr_only_study_counts_tokens_and_builds_no_trees(corpus: Corpus) -> None:
    """The record is the extraction and its counters: tokens, time, resources.

    There is no tree, but the one run that happened is measured as fully as any
    other: the manifest states the page counters, the stage's seconds and the
    sampler's resources. The counters are input and output tokens, and a run
    that reported them all publishes a total rather than a subtotal.
    """
    engine = FakeEngine("native", usage=True, model="fake-vision-1")
    runner = make_runner(corpus, {"native": engine})

    result = runner.study(families=["ocr"], engines=["native"])

    assert result["executed"] is True
    assert result["trees"] == []
    assert [entry["status"] for entry in result["extractions"]] == [RUN_OK] * 3
    assert len(engine.calls) == 3
    assert {entry["family"] for entry in result["comparisons"]} == {"ocr"}
    assert len(result["comparisons"]) == 3
    assert all(
        set(entry) == {"artifact_digest", "path", "run_id", "status"}
        for entry in result["extractions"]
    )

    manifest = runner.read_run("digital-1", result["extractions"][0]["run_id"])
    usage = manifest["usage"]
    assert usage["status"] == "measured"
    assert usage["source"] == "artifact"
    assert usage["attempts"] == 3
    assert usage["prompt_tokens"] == 300  # three pages, a hundred each
    assert usage["completion_tokens"] == 60
    assert usage["pages_with_usage"] == 3

    assert manifest["stages"]["extract"]["status"] == "measured"
    assert manifest["stages"]["extract"]["seconds"] >= 0
    assert manifest["resources"]["cpu"]["status"] == "measured"
    assert manifest["resources"]["wall"]["status"] == "measured"
    assert manifest["resources"]["wall"]["value"] >= 0
    assert manifest["resources"]["rss_peak"]["status"] == "measured"

    # The report counts the tokens the run stated, and no money anywhere.
    counts = comparison_report(result, 0)["counts"]
    assert counts["prompt_tokens"] == 300
    assert counts["completion_tokens"] == 60
    assert counts["documents_with_usage"] == 1


def test_a_mixed_family_study_still_builds_its_trees(corpus: Corpus) -> None:
    """Add one family that needs a tree and the trees are planned and built.

    The OCR-only shape is a property of that family list, not of the runner: the
    same engines and documents under ``ocr`` and ``pipeline`` produce a tree per
    document from the one extraction.
    """
    engine = FakeEngine("native")
    runner = make_runner(corpus, {"native": engine})

    result = runner.study(families=["ocr", "pipeline"], engines=["native"])

    assert [entry["profile"] for entry in result["trees"]] == ["native_atomic"] * 3
    assert all(entry["status"] == RUN_OK for entry in result["trees"])
    assert {entry["family"] for entry in result["comparisons"]} == {"ocr", "pipeline"}
    assert len(engine.calls) == 3  # the trees replay the extraction, not repay it


def test_study_refuses_a_paid_engine_until_the_upload_is_allowed(corpus: Corpus) -> None:
    engine = FakeVisionEngine()
    runner = make_runner(corpus, {"vision": engine}, settings=StubSettings())

    with pytest.raises(UploadNotAllowedError, match="--allow-upload"):
        runner.study(families=["ocr"], engines=["vision"])
    assert engine.calls == []

    planned = runner.study(families=["ocr"], engines=["vision"], dry_run=True)["plan"]
    assert planned["limitations"] == [
        "vision is planned but not permitted: pass --allow-upload to run it"
    ]

    result = runner.study(families=["ocr"], engines=["vision"], allow_upload=True)

    assert result["executed"] is True
    assert len(engine.calls) == 3
    # The study built the configuration from the injected settings: the endpoint
    # is a reserved-TLD host and the configuration holds no key at all.
    config = engine.configs[0]
    assert config.model == "fake-vision-1"
    assert config.host == "vision.invalid"
    assert config.reasoning_effort is None
    assert config.api_key is None
    assert all(entry["status"] == RUN_OK for entry in result["extractions"])


def test_parse_page_selection_reads_ranges_and_refuses_nonsense() -> None:
    assert parse_page_selection("0-2,5") == (0, 1, 2, 5)
    assert parse_page_selection(" 3 ") == (3,)
    assert parse_page_selection("7") == (7,)

    with pytest.raises(RunnerError, match="expected N or N-M"):
        parse_page_selection("two")
    with pytest.raises(RunnerError, match="end before start"):
        parse_page_selection("3-1")
    with pytest.raises(RunnerError, match="repeats a page"):
        parse_page_selection("1,1,2")
    with pytest.raises(RunnerError, match="is empty"):
        parse_page_selection(",")


# --------------------------------------------------------------------------- #
# the boundary between a tree run and its artifact
# --------------------------------------------------------------------------- #


def test_engine_pages_is_the_exact_inverse_of_the_stored_artifact(corpus: Corpus) -> None:
    engine = FakeEngine("native")
    runner = make_runner(corpus, {"native": engine})

    artifact = runner.artifact(
        "digital-1", runner.extract("digital-1", "native")["artifact_digest"]
    )

    rebuilt = engine_pages(artifact)
    assert len(rebuilt) == len(engine.last_pages)
    for stored, original in zip(rebuilt, engine.last_pages, strict=True):
        assert stored.page_index == original.page_index
        assert stored.request_page == original.request_page
        assert stored.running == original.running
        assert stored.printed_folio == original.printed_folio
        assert stored.blocks == original.blocks


def test_a_vision_artifact_without_groups_cannot_drive_the_grouping_builder(
    corpus: Corpus,
) -> None:
    engine = FakeVisionEngine(groups=False)
    runner = make_runner(corpus, {"vision": engine})
    run = runner.extract("scan-1", "vision", allow_upload=True, vision_config=StubVisionConfig())
    assert run["status"] == RUN_OK

    failed = runner.run_tree("scan-1", "vision_direct", extract_run_id=run["run_id"])
    assert failed["status"] == RUN_FAILED
    assert failed["rankable"] is False
    assert failed["quality"] == "not_measured"
    assert failed["quality_reason"] == "build_failed"
    assert failed["semantic_signature"] is None

    # The same artifact still builds an atomic tree, which does not need groups.
    atomic = runner.run_tree("scan-1", "vision_atomic", extract_run_id=run["run_id"])
    assert atomic["status"] == RUN_OK
    assert atomic["rankable"] is True


# --------------------------------------------------------------------------- #
# continuity: a claim across a page nobody read is refused on both sides
# --------------------------------------------------------------------------- #


def tree_nodes(node: Any) -> list[dict[str, Any]]:
    """Every node of one serialized tree, the node itself first."""
    found = [node]
    for child in node.get("children") or ():
        found.extend(tree_nodes(child))
    return found


def texts_of_kind(tree: Any, kind: str) -> list[str]:
    """The text of every node of one kind, in document order."""
    return [
        node.get("attributes", {}).get("text", "")
        for node in tree_nodes(tree)
        if node.get("node_type") == kind
    ]


CARRIED_PARAGRAPH = "the paragraph a page carries on from its predecessor when nothing resets it. "
"""Longer than the parser's short-fragment threshold, three times over.

A paragraph under that threshold is absorbed into the one before it anyway, so a
refused claim would be invisible in the tree.
"""


def carried_text(index: int) -> str:
    """The one paragraph a page of :class:`ContinuingVisionEngine` carries."""
    return f"{CARRIED_PARAGRAPH * 3}page {index}."


class ContinuingVisionEngine(FakeVisionEngine):
    """A vision double whose pages are one paragraph group and claim to continue.

    A claim that survives really welds two pages into one paragraph, which is
    what makes a refused claim observable. The extract path drops a claim that
    crosses a gap, so a stored artifact from *this* runner cannot carry one: the
    double stands in for an artifact from somewhere else, or for an adapter that
    does not apply the policy.
    """

    def __init__(self, *, claims: bool = True, **overrides: Any) -> None:
        super().__init__(**overrides)
        self.claims = claims

    def _page(self, doc_id: str, index: int, *, page_offset: int, ordinal: int) -> EnginePage:
        return EnginePage(
            page_index=index,
            request_page=index + 1 + page_offset,
            blocks=(
                EngineBlock(
                    kind="p",
                    normalized_kind="p",
                    text=carried_text(index),
                    group=1,
                    source_id=f"{index}:body",
                ),
            ),
            running=f"{doc_id} running head",
            printed_folio=str(index + 1),
            continues_previous=self.claims and index != 0,
        )


def continuing_vision_run(
    corpus: Corpus, pages: Sequence[int], *, claims: bool
) -> tuple[StudyRunner, dict[str, Any]]:
    """A stored vision extraction over ``pages``, from a double that claims to continue."""
    runner = make_runner(corpus, {"vision": ContinuingVisionEngine(claims=claims)})
    run = runner.extract(
        "digital-1",
        "vision",
        page_indices=pages,
        allow_upload=True,
        vision_config=StubVisionConfig(),
    )
    assert run["status"] == RUN_OK
    return runner, run


def test_a_claim_across_a_gap_is_refused_when_the_artifact_is_replayed(corpus: Corpus) -> None:
    runner, run = continuing_vision_run(corpus, (0, 2), claims=True)

    tree = runner.run_tree("digital-1", "vision_direct", extract_run_id=run["run_id"])
    counters = runner.read_run_file("digital-1", tree["run_id"], "signature.json")["counters"]
    paragraphs = texts_of_kind(
        runner.read_run_file("digital-1", tree["run_id"], "tree.json"), "paragraph"
    )

    assert tree["status"] == RUN_OK
    assert tree["continuity_resets"] == [2]
    assert counters["cross_page_continues"] == 0  # the claim was refused, not carried
    assert paragraphs == [carried_text(0), carried_text(2)]  # one paragraph per page


def test_replay_refuses_the_claims_the_extract_side_refuses(corpus: Corpus) -> None:
    claimed_runner, claimed = continuing_vision_run(corpus, (0, 2), claims=True)
    refused_runner, refused = continuing_vision_run(corpus, (0, 2), claims=False)

    first = claimed_runner.run_tree("digital-1", "vision_direct", extract_run_id=claimed["run_id"])
    second = refused_runner.run_tree(
        "digital-1", "vision_direct", extract_run_id=refused["run_id"]
    )

    # The artifact that kept the claim and the artifact that never made one replay
    # to the same tree: the policy, not the artifact, decides.
    assert first["semantic_signature"] == second["semantic_signature"]
    assert first["continuity_resets"] == second["continuity_resets"] == [2]


def test_a_claim_between_two_pages_that_were_both_read_is_carried(corpus: Corpus) -> None:
    runner, run = continuing_vision_run(corpus, (0, 1), claims=True)

    tree = runner.run_tree("digital-1", "vision_direct", extract_run_id=run["run_id"])
    counters = runner.read_run_file("digital-1", tree["run_id"], "signature.json")["counters"]
    paragraphs = texts_of_kind(
        runner.read_run_file("digital-1", tree["run_id"], "tree.json"), "paragraph"
    )

    # The policy refuses unread regions, not the continuation the model made.
    assert tree["continuity_resets"] == []
    assert counters["cross_page_continues"] == 1
    assert paragraphs == [f"{carried_text(0)}\n{carried_text(1)}"]


# --------------------------------------------------------------------------- #
# accounting: a subtotal is not a total
# --------------------------------------------------------------------------- #


def with_usage(artifact: Any, usages: Sequence[PageUsage | None]) -> Any:
    """The same artifact, with one usage record per page, in page order."""
    pages = tuple(
        page.model_copy(update={"usage": usage})
        for page, usage in zip(artifact.pages, usages, strict=True)
    )
    return artifact.model_copy(update={"pages": pages})


def two_page_artifact(corpus: Corpus) -> Any:
    """A stored artifact over the two scanned pages, every page reporting usage."""
    engine = FakeEngine("native", usage=True)
    runner = make_runner(corpus, {"native": engine})
    run = runner.extract("scan-1", "native")
    return runner.artifact("scan-1", run["artifact_digest"])


def billed(
    *,
    attempts: int = 1,
    prompt: int = 100,
    completion: int = 20,
) -> PageUsage:
    return PageUsage(
        attempts=attempts,
        prompt_tokens=prompt,
        completion_tokens=completion,
    )


class MeteredMergeClient:
    """The runner's merge double, stating what a gateway states -- or nothing.

    It wraps ``FakeMergeClient``, so a tree is still built by the same offline
    wiring, and adds the one thing that double cannot say: what the gateway
    reported. ``usages`` is read one entry per answered call and its last entry
    repeats, so one section can answer with counts while another states none;
    ``failures`` is how many of the first calls die unanswered, which is how a
    retry and an unknown bill become observable together. ``models`` is read the
    same way, one entry per answered call: it is the model each answer says
    served it, and an entry of ``None`` is an answer that names none -- which is
    what a gateway that attributes nothing looks like.
    """

    def __init__(
        self,
        usages: Sequence[Mapping[str, Any]],
        *,
        failures: int = 0,
        models: Sequence[str | None] | None = None,
    ) -> None:
        self._merge = FakeMergeClient()
        self.usages = [dict(usage) for usage in usages]
        self.models = list(models) if models is not None else []
        self.failures = failures
        self.calls = 0

    async def invoke(self, request: Any) -> Any:
        self.calls += 1
        if self.calls <= self.failures:
            raise RuntimeError("the gateway answered with an error")
        response = await self._merge.invoke(request)
        answered = self.calls - 1 - self.failures
        update: dict[str, Any] = {
            "usage": self.usages[min(answered, len(self.usages) - 1)],
        }
        if self.models:
            update["model"] = self.models[min(answered, len(self.models) - 1)]
        return response.model_copy(update=update)


# The model the merge answers say served them: a name of its own, so a record
# that credited the tokens to the requested name instead cannot come out the same.
SERVED_MERGE_MODEL = "fake-merge-served-1"


def merge_run(
    corpus: Corpus,
    client: Any,
    *,
    usage: bool = True,
    refine: Any = REFINE,
) -> tuple[StudyRunner, dict[str, Any]]:
    """A runner over ``mixed-1``'s marker artifact and one live merge of it.

    ``llm_mode="live"`` is the mode that asks for a real merge; ``client`` is a
    double, so the merge is wired as the live one and no request leaves the
    machine. ``usage=True`` has the artifact state a page usage, which is what
    gives the tree's extraction side counters to show beside its merge's. Both
    the runner and the tree come back, because the counters are read through
    ``compare`` as well as on the run itself.
    """
    engine = FakeEngine("marker", usage=usage, model="fake-vision-1" if usage else None)
    runner = make_runner(corpus, {"marker": engine})
    extraction = runner.extract("mixed-1", "marker")
    tree = runner.run_tree(
        "mixed-1",
        "marker_llm",
        extract_run_id=extraction["run_id"],
        llm_mode="live",
        allow_merge_upload=True,
        llm_client=client,
        refine_config=refine,
    )
    return runner, tree


def live_merged_tree(corpus: Corpus, client: Any) -> dict[str, Any]:
    """One merged tree over ``mixed-1``'s marker artifact, built with ``client``.

    ``llm_mode="live"`` is the mode that asks for a real merge; the client is a
    double, so the merge is wired as the live one and no request leaves the
    machine.
    """
    return merge_run(corpus, client, usage=False)[1]


def test_a_page_that_reported_nothing_leaves_the_total_unknown(corpus: Corpus) -> None:
    artifact = two_page_artifact(corpus)
    # The second page never got an answer, so its spend is not zero: it is unknown.
    partial = with_usage(artifact, [billed(), None])

    usage = artifact_usage(partial)

    assert usage["status"] == "partial"
    assert usage["reason"] == "usage_missing"
    assert usage["complete"] is False
    assert usage["pages"] == 2
    assert usage["pages_with_usage"] == 1
    # The sum of the pages that reported is a subtotal, never presented as a total.
    assert usage["prompt_tokens"] is None
    assert usage["completion_tokens"] is None
    assert usage["attempts"] is None
    assert usage["subtotal"]["prompt_tokens"] == 100
    assert usage["subtotal"]["completion_tokens"] == 20
    assert usage["subtotal"]["attempts"] == 1
    assert usage["subtotal"]["pages_with_usage"] == 1


def test_a_whole_run_of_reported_pages_is_a_measured_total(corpus: Corpus) -> None:
    artifact = two_page_artifact(corpus)

    usage = artifact_usage(with_usage(artifact, [billed(), billed(attempts=2)]))

    assert usage["status"] == "measured"
    assert usage["reason"] is None
    assert usage["complete"] is True
    assert usage["pages_with_usage"] == 2
    assert usage["attempts"] == 3
    assert usage["prompt_tokens"] == 200
    assert usage["completion_tokens"] == 40
    assert usage["subtotal"]["prompt_tokens"] == 200


def test_a_run_with_no_usage_at_all_is_unknown_and_never_zero(corpus: Corpus) -> None:
    artifact = two_page_artifact(corpus)

    usage = artifact_usage(with_usage(artifact, [None, None]))

    assert usage["status"] == "not_measured"
    assert usage["reason"] == "usage_missing"
    assert usage["complete"] is False
    assert usage["attempts"] is None
    assert usage["prompt_tokens"] is None
    assert usage["subtotal"] == {
        "attempts": 0,
        "prompt_tokens": 0,
        "completion_tokens": 0,
        "pages_with_usage": 0,
    }


def test_a_failed_run_keeps_the_spend_the_engine_reported(corpus: Corpus) -> None:
    engine = FakeVisionEngine()
    engine.error = VisionOCRError(
        "the endpoint gave up on page 2",
        attempts=3,
        prompt_tokens=500,
        completion_tokens=40,
    )
    runner = make_runner(corpus, {"vision": engine})

    run = runner.extract("scan-1", "vision", allow_upload=True, vision_config=StubVisionConfig())

    assert run["status"] == RUN_FAILED
    assert run["artifact_digest"] is None
    # The counters are kept even though there is no artifact to read them off.
    assert run["usage"]["attempts"] == 3
    assert run["usage"]["prompt_tokens"] == 500
    assert run["usage"]["completion_tokens"] == 40
    assert run["usage"]["source"] == "failure"
    # A failure is not published as a partial success.
    assert not (study_root(corpus) / "scan-1" / "artifacts").exists()
    result = runner.compare("ocr", [RunRef("scan-1", run["run_id"])], write=False)
    assert result["runs"][0]["usage"]["prompt_tokens"] == 500


def test_a_failure_before_any_request_spends_nothing_it_cannot_name(corpus: Corpus) -> None:
    engine = FakeVisionEngine()
    engine.error = VisionOCRError("refused before the first call")
    runner = make_runner(corpus, {"vision": engine})

    run = runner.extract("scan-1", "vision", allow_upload=True, vision_config=StubVisionConfig())

    assert run["status"] == RUN_FAILED
    assert run["usage"]["attempts"] == 0
    assert run["usage"]["prompt_tokens"] is None
    assert run["usage"]["status"] == "not_measured"


def test_a_failed_run_stays_in_the_usage_denominator(corpus: Corpus) -> None:
    engine = FakeVisionEngine()
    engine.error = VisionOCRError("gave up", attempts=1, prompt_tokens=100, completion_tokens=20)
    runner = make_runner(corpus, {"vision": engine})
    run = runner.extract("scan-1", "vision", allow_upload=True, vision_config=StubVisionConfig())

    report = runner.compare("ocr", [RunRef("scan-1", run["run_id"])], write=False)["report"]

    # A run that was requested and reported tokens is not dropped from the
    # denominator just because it has no artifact to score.
    assert report["counts"]["documents_with_usage"] == 1
    assert report["usage"]["prompt_tokens"] == 100
    assert report["usage"]["completion_tokens"] == 20


def test_a_live_merge_reports_the_spend_the_gateway_stated(corpus: Corpus) -> None:
    client = MeteredMergeClient([{"prompt_tokens": 1200, "completion_tokens": 800}])

    merge = live_merged_tree(corpus, client)["merge"]

    assert merge["attempts"] == merge["llm_calls"] == client.calls > 0  # no retries
    assert merge["prompt_tokens"] == 1200 * merge["attempts"]
    assert merge["completion_tokens"] == 800 * merge["attempts"]
    assert merge["usage_unknown_reason"] is None
    assert merge["latency_seconds"] >= 0.0


def test_one_unobservable_section_makes_the_trees_merge_bill_unknown(corpus: Corpus) -> None:
    # The first section's merge answers in the all-zero envelope an absent usage
    # arrives in; every later call states its counts.
    client = MeteredMergeClient([{}, {"prompt_tokens": 1200, "completion_tokens": 800}])

    merge = live_merged_tree(corpus, client)["merge"]

    assert merge["attempts"] == merge["llm_calls"] == client.calls > 1
    # The calls that did answer are a subtotal, and a subtotal is not the bill.
    assert merge["prompt_tokens"] is None
    assert merge["completion_tokens"] is None
    assert "no token usage" in merge["usage_unknown_reason"]
    assert merge["latency_seconds"] >= 0.0  # measured even when the bill is not


def test_a_retried_merge_counts_attempts_and_keeps_its_bill_unknown(corpus: Corpus) -> None:
    client = MeteredMergeClient([{"prompt_tokens": 1200, "completion_tokens": 800}], failures=1)

    merge = live_merged_tree(corpus, client)["merge"]

    # One call died and was retried, so the tree made one more attempt than it
    # completed invocations -- and the section it belonged to was billed twice,
    # which is why its counts are unknown rather than the one answer it got.
    assert merge["attempts"] == client.calls == merge["llm_calls"] + 1
    assert merge["prompt_tokens"] is None
    assert merge["completion_tokens"] is None
    assert "no response" in merge["usage_unknown_reason"]
    assert merge["latency_seconds"] >= 0.0


class UnusableMergeClient:
    """A merge client that answers, with an answer no merge can read as segments."""

    def __init__(self) -> None:
        self._merge = FakeMergeClient()
        self.calls = 0

    async def invoke(self, request: Any) -> Any:
        self.calls += 1
        response = await self._merge.invoke(request)
        return response.model_copy(
            update={"message": response.message.model_copy(update={"content": "not json"})}
        )


def test_a_merge_that_answered_nothing_usable_is_counted_as_a_fallback(corpus: Corpus) -> None:
    client = UnusableMergeClient()

    tree = live_merged_tree(corpus, client)

    # The call happened and was billed, so the section was merged -- and the
    # refiner kept the originals, so it fell back. Both halves are counted.
    assert tree["merge"]["llm_calls"] == client.calls > 0
    assert tree["merge"]["sections_merged"] == tree["merge"]["sections_fallback"] > 0
    assert tree["merge"]["changed_sections"] == 0


class UnreachableMergeClient:
    """A merge client that reaches no model: every call dies unanswered.

    This is what a gateway with no configured chat model does from inside a run --
    the exception the refiner catches per section -- so a tree built with it has
    the shape a study reaches when its endpoint was never wired.
    """

    def __init__(self) -> None:
        self.calls = 0

    async def invoke(self, request: Any) -> Any:
        self.calls += 1
        raise RuntimeError("No chat models have been configured.")


def test_a_live_merge_that_reached_no_model_is_not_rankable(corpus: Corpus) -> None:
    """A merge whose every call died is not a result, however the run is stamped.

    The defect this guards: the verdict only looked at the merge mode, so a merge
    that reached no model -- every section kept its originals -- was published as
    measured and rankable, and the study reported a merge it never made. The run
    itself is fine, so the run stays ok and only the merge's numbers are refused.
    """
    client = UnreachableMergeClient()

    tree = live_merged_tree(corpus, client)

    assert tree["status"] == RUN_OK
    assert client.calls > 0  # the calls were made; none of them was answered
    assert tree["merge"]["llm_calls"] == 0
    assert tree["merge"]["sections_merged"] == tree["merge"]["sections_fallback"] > 0
    assert tree["quality"] == "not_measured"
    assert tree["quality_reason"] == "merge_rebuilt_nothing"
    assert tree["rankable"] is False


def test_a_live_merge_that_kept_every_original_is_not_rankable(corpus: Corpus) -> None:
    """A merge that answered unusably is the same tree as one that never ran.

    The defect this guards: completed calls are not the test. A call can answer
    and name nothing the refiner can refine, so every section falls back and the
    merged tree is the atomic tree under another name; the verdict has to refuse
    it rather than count the calls that were billed.
    """
    client = UnusableMergeClient()

    tree = live_merged_tree(corpus, client)

    assert tree["merge"]["llm_calls"] == client.calls > 0
    assert tree["merge"]["sections_merged"] == tree["merge"]["sections_fallback"] > 0
    assert tree["quality"] == "not_measured"
    assert tree["quality_reason"] == "merge_rebuilt_nothing"
    assert tree["rankable"] is False


def test_a_partial_merge_stays_measured_and_rankable(corpus: Corpus) -> None:
    """A merge that rebuilt some sections is a result, not a failure.

    The defect this guards: a fallback anywhere must not read as a failed merge.
    The section that kept its originals is that section's own numbers, and the
    sections that did merge are the merge, so the tree stays measured and its
    numbers stay rankable.
    """
    # Three attempts die unanswered -- the first section's retries -- and every
    # later call answers: one section keeps its originals, the rest merge.
    client = MeteredMergeClient([{"prompt_tokens": 1200, "completion_tokens": 800}], failures=3)

    tree = live_merged_tree(corpus, client)

    assert tree["merge"]["sections_fallback"] > 0
    assert tree["merge"]["sections_merged"] > tree["merge"]["sections_fallback"]
    assert tree["quality"] == "measured"
    assert tree["quality_reason"] is None
    assert tree["rankable"] is True


def test_a_merge_that_rebuilt_nothing_is_named_in_the_comparison(corpus: Corpus) -> None:
    """A report-only reader must be able to tell a failed merge from the numbers.

    The defect this guards: the limitations named a faked merge and a missing
    merge, but never a merge that ran and kept every original, so a reader with
    the report alone could take its fidelity and chunk figures for a merge's. The
    document is named once, not once per repeat: every repeat of it has the same
    gap, and a tree whose profile merges nothing at all is not named.
    """
    runner = make_runner(corpus, {"native": FakeEngine("native"), "marker": FakeEngine("marker")})
    extraction = runner.extract("mixed-1", "marker")
    unmerged = runner.run_tree(
        "mixed-1",
        "native_atomic",
        extract_run_id=runner.extract("mixed-1", "native")["run_id"],
    )

    def merged_repeat(repeat: int) -> dict[str, Any]:
        return runner.run_tree(
            "mixed-1",
            "marker_llm",
            extract_run_id=extraction["run_id"],
            repeat=repeat,
            llm_mode="live",
            allow_merge_upload=True,
            llm_client=UnusableMergeClient(),
            refine_config=REFINE,
        )

    first = merged_repeat(1)
    second = merged_repeat(2)

    result = runner.compare(
        "pipeline",
        [
            RunRef("mixed-1", unmerged["run_id"]),
            RunRef("mixed-1", first["run_id"]),
            RunRef("mixed-1", second["run_id"]),
        ],
        write=False,
    )

    assert first["run_id"] != second["run_id"]  # two repeats, two rows
    assert len(result["runs"]) == 3
    assert [
        entry for entry in result["report"]["limitations"] if "rebuilt no section" in entry
    ] == [
        "mixed-1: the merge rebuilt no section, so its merged tree is the tree it "
        "was built from under another name"
    ]


# --------------------------------------------------------------------------- #
# what a merge spent: its own counters, beside the extraction's
# --------------------------------------------------------------------------- #


def merge_report(
    corpus: Corpus,
    client: Any,
    *,
    usage: bool = True,
    refine: Any = REFINE,
) -> tuple[dict[str, Any], dict[str, Any]]:
    """One merged tree and the report its comparison wrote."""
    runner, tree = merge_run(corpus, client, usage=usage, refine=refine)
    report = runner.compare("pipeline", [RunRef("mixed-1", tree["run_id"])], write=False)["report"]
    return tree, report


def merged_document(report: Mapping[str, Any]) -> Mapping[str, Any]:
    """The one document of a one-run report's usage block."""
    documents = report["usage"]["per_document"]
    assert len(documents) == 1
    return documents[0]


def test_a_merged_tree_reports_what_its_merge_spent(corpus: Corpus) -> None:
    """The defect this guards: a pipeline's usage that omitted the merge.

    A real merged run's manifest holds the merge's token counts, and the report
    read only the extraction artifact's usage, so the merge's spend never reached
    the run's usage block. Both are visible now and they stay apart: the
    extraction's counters are the pages it read, the merge's are its own calls,
    and neither is folded into the other.
    """
    client = MeteredMergeClient([{"prompt_tokens": 1200, "completion_tokens": 800}])

    tree, report = merge_report(corpus, client)

    calls = tree["merge"]["attempts"]
    assert calls == tree["merge"]["llm_calls"] > 0  # no retries
    assert tree["merge"]["prompt_tokens"] == 1200 * calls
    document = merged_document(report)
    # The extraction read four pages at 100 prompt and 20 completion tokens each.
    assert document["prompt_tokens"] == 400 + 1200 * calls
    assert document["completion_tokens"] == 80 + 800 * calls
    assert document["status"] == "measured"
    assert report["usage"]["prompt_tokens"] == document["prompt_tokens"]
    assert report["counts"]["prompt_tokens"] == document["prompt_tokens"]


def test_a_merge_that_reached_no_model_leaves_the_total_unknown(corpus: Corpus) -> None:
    """A merge whose every call died has unobservable tokens, not zero of them.

    The counters hold the attempts the merge made and no token counts, because
    the calls it could not observe are unknown. The run's total follows: it is a
    typed miss rather than the extraction's tokens offered as the whole spend.
    """
    client = UnreachableMergeClient()

    tree, report = merge_report(corpus, client)

    assert tree["merge"]["attempts"] == client.calls > 0
    assert tree["merge"]["prompt_tokens"] is None
    assert tree["merge"]["usage_unknown_reason"]
    document = merged_document(report)
    assert document["status"] == "partial"
    assert document["reason"] == "usage_missing"
    assert document["prompt_tokens"] is None
    assert document["completion_tokens"] is None
    # The pages that did answer are a subtotal, and a subtotal is not a total.
    assert document["subtotal"]["prompt_tokens"] == 400
    assert report["usage"]["status"] == "partial"
    assert report["usage"]["prompt_tokens"] is None
    assert report["counts"]["prompt_tokens"] is None
    assert "an unknown number of tokens" in " ".join(report["limitations"])


def test_a_run_whose_extraction_reported_nothing_still_shows_its_merge_spend(
    corpus: Corpus,
) -> None:
    """The other direction: a known merge spend must not hide behind a missing one.

    This artifact states no counters at all, so the extraction side has nothing
    to report; the merge's tokens are known. The run has no total -- an
    extraction that reported nothing is not an extraction that spent nothing --
    and the merge's own counters stay visible instead of being swallowed.
    """
    client = MeteredMergeClient([{"prompt_tokens": 1200, "completion_tokens": 800}])

    tree, report = merge_report(corpus, client, usage=False)
    document = merged_document(report)

    assert document["status"] == "partial"
    assert document["subtotal"]["prompt_tokens"] == tree["merge"]["prompt_tokens"] > 0
    assert document["subtotal"]["completion_tokens"] == tree["merge"]["completion_tokens"] > 0
    assert document["prompt_tokens"] is None
    # The document did report something -- the merge -- so it is a document the
    # usage block accounts for, with no total for the run it belongs to.
    assert report["usage"]["status"] == "partial"
    assert report["counts"]["documents_with_usage"] == 1
    assert report["counts"]["prompt_tokens"] is None


def test_a_merge_that_made_no_call_has_no_part_to_report(corpus: Corpus) -> None:
    """A merge can walk a tree and call nothing; that is a measured zero.

    ``refine_details=False`` skips every section before any call, so the counters
    say ``attempts: 0`` over a zero token count. A merge that spent no token is
    no part of what the run spent, so the usage block carries the extraction
    alone and no limitation is written about a spend that never happened.
    """
    client = UnreachableMergeClient()

    tree, report = merge_report(
        corpus, client, refine=RefineConfig(refine_details=False, llm_model=REFINE.llm_model)
    )

    assert tree["merge"]["sections"] > 0  # the merge walked the tree
    assert tree["merge"]["attempts"] == tree["merge"]["llm_calls"] == 0
    assert client.calls == 0  # and reached no model
    document = merged_document(report)
    assert document["status"] == "measured"
    assert document["prompt_tokens"] == 400  # the extraction's four pages alone
    assert report["counts"]["prompt_tokens"] == 400
    # The merge ran and rebuilt nothing, which its own limitation already names;
    # what must not appear is a spend the counters cannot stand behind.
    assert not any("the merge spent" in line for line in report["limitations"])


def test_answers_that_named_no_model_are_named_in_the_report(corpus: Corpus) -> None:
    """A gateway that attributes nothing leaves the merge's tokens unattributed.

    The tokens are real and counted; what no answer said is which model served
    them, and the requested name is not offered as the served one. The report
    says so in words rather than inventing an attribution.
    """
    client = MeteredMergeClient([{"prompt_tokens": 1200, "completion_tokens": 800}], models=[None])

    tree, report = merge_report(corpus, client)

    row = report["metadata"]["run_rows"][0]
    assert "models" not in row["merge_bill"]
    assert tree["merge"]["prompt_tokens"] > 0  # counted all the same
    assert any(
        "no answer named the model that served it" in line for line in report["limitations"]
    )


def test_answers_that_named_a_model_publish_the_split(corpus: Corpus) -> None:
    """One answer naming its model is enough for the split to be worth publishing.

    The second call is answered by a different model, so the record carries two
    entries, each with the tokens attributed to it -- the whole spend, never the
    part that was named.
    """
    second_model = "fake-merge-served-2"
    client = MeteredMergeClient(
        [{"prompt_tokens": 1200, "completion_tokens": 800}],
        models=[SERVED_MERGE_MODEL, second_model],
    )

    tree, report = merge_report(corpus, client)

    models = report["metadata"]["run_rows"][0]["merge_bill"]["models"]
    assert [entry["model"] for entry in models] == [SERVED_MERGE_MODEL, second_model]
    assert (
        sum(entry["usage"]["prompt_tokens"] for entry in models) == tree["merge"]["prompt_tokens"]
    )
    assert not any("no answer named the model" in line for line in report["limitations"])


def test_a_pairs_token_delta_counts_the_extraction_and_its_merge_together(
    corpus: Corpus, monkeypatch: pytest.MonkeyPatch
) -> None:
    """The chunk family pairs a grouped tree with the merged one built from it.

    Both arms read the same pages and the merged arm adds its merge, so the pair
    is the one place the merge's tokens are visible as a difference. A merge
    whose tokens were not reported leaves its run without a total, so the delta
    is a typed miss rather than a number computed from the side that answered.
    """
    runner = make_runner(corpus, {"vision": FakeVisionEngine()}, settings=StubSettings())
    monkeypatch.setattr(
        StudyRunner,
        "_gateway_client",
        lambda _self: MeteredMergeClient([{"prompt_tokens": 1200, "completion_tokens": 800}]),
    )

    result = runner.study(
        families=["chunk"],
        engines=["vision"],
        doc_ids=["mixed-1"],
        profiles=["vision_direct", "vision_atomic_merge"],
        allow_upload=True,
        allow_merge_upload=True,
        llm_mode="live",
    )

    assert len(result["comparisons"]) == 1
    report = comparison_report(result)
    pairs = report["paired_usage"]["pairs"]
    assert len(pairs) == 1
    pair = pairs[0]
    # The merged arm is the pair's B side, so the delta is what the merge added.
    assert pair["run_a"].split(".")[1] == "vision_direct"
    assert pair["run_b"].split(".")[1] == "vision_atomic_merge"
    delta = pair["usage"]["delta"]
    assert delta["status"] == "measured"
    # The grouped arm bought nothing of its own, so the whole difference is the
    # merge's own counters -- never the pages, which both arms read alike.
    assert delta["prompt_tokens"] == pair["usage"]["usage_b"]["merge"]["prompt_tokens"] > 0
    assert delta["completion_tokens"] == pair["usage"]["usage_b"]["merge"]["completion_tokens"] > 0
    assert pair["usage"]["usage_a"]["prompt_tokens"] == 400
    assert pair["usage"]["usage_b"]["prompt_tokens"] == 400 + delta["prompt_tokens"]
    # A reader of the deltas is told the delta counts both parts together.
    assert any(
        "count the extraction and its merge together" in line for line in report["limitations"]
    )


def test_a_run_with_no_merge_reports_the_extraction_alone(corpus: Corpus) -> None:
    """A merge-free run must not grow a merge part or a merge limitation.

    An extraction and a tree whose profile merges nothing have no merge counters,
    so the usage block is the block a reader saw before merges were accounted
    for: one document per run, the counters the pages reported, no merge part.
    """
    runner = make_runner(corpus, {"vision": FakeVisionEngine()})
    extraction = runner.extract(
        "scan-1", "vision", allow_upload=True, vision_config=StubVisionConfig()
    )
    tree = runner.run_tree("scan-1", "vision_atomic", extract_run_id=extraction["run_id"])

    report = runner.compare(
        "pipeline",
        [RunRef("scan-1", extraction["run_id"]), RunRef("scan-1", tree["run_id"])],
        write=False,
    )["report"]

    assert all("merge" not in document for document in report["usage"]["per_document"])
    assert report["usage"]["status"] == "measured"
    # Both runs read scan-1's two pages at a hundred prompt tokens each, and the
    # tree replayed the extraction rather than buying a third reading.
    assert report["usage"]["prompt_tokens"] == 400
    assert report["usage"]["documents"] == 2
    assert not any("the merge spent" in line for line in report["limitations"])


# --------------------------------------------------------------------------- #
# the merge policy a scored tree is built with, and what names it
# --------------------------------------------------------------------------- #


class TableEngine(FakeEngine):
    """A marker double whose pages carry one table block among their paragraphs."""

    def __init__(self, **overrides: Any) -> None:
        overrides.setdefault(
            "kinds", {"heading": "SectionHeader", "p": "Text", "footer": "PageFooter"}
        )
        super().__init__("marker", **overrides)

    def _page(self, doc_id: str, index: int, *, page_offset: int, ordinal: int) -> EnginePage:
        page = super()._page(doc_id, index, page_offset=page_offset, ordinal=ordinal)
        return replace(
            page,
            blocks=(
                *page.blocks,
                EngineBlock(
                    kind="Table",
                    normalized_kind="table",
                    text=f"Bảng {index + 1}: nội dung bảng",
                    group=page.blocks[0].group,
                    source_id=f"{index}:table",
                ),
            ),
        )


def merged_marker_tree(
    corpus: Corpus, *, refine: Any | None = None
) -> tuple[StudyRunner, dict[str, Any]]:
    """A merged tree over a marker artifact whose sections each carry a table."""
    runner = make_runner(corpus, {"marker": TableEngine()})
    run = runner.extract("digital-1", "marker")
    tree = runner.run_tree(
        "digital-1",
        "marker_llm",
        extract_run_id=run["run_id"],
        llm_mode="live",
        allow_merge_upload=True,
        llm_client=FakeMergeClient(),
        refine_config=refine,
    )
    return runner, tree


def merged_kinds(runner: StudyRunner, tree: dict[str, Any]) -> list[str]:
    """The node type of every node of one merged tree, in document order."""
    return [
        node.get("node_type")
        for node in tree_nodes(runner.read_run_file("digital-1", tree["run_id"], "tree.json"))
    ]


def test_the_merge_a_tree_is_scored_with_never_demotes_a_table(corpus: Corpus) -> None:
    runner, tree = merged_marker_tree(corpus)

    kinds = merged_kinds(runner, tree)

    assert tree["status"] == RUN_OK
    assert kinds.count("table") == 3  # one per page, kept as the node it already is
    assert tree["merge"]["sections_merged"] > 0  # the paragraphs around them still merged


def test_the_runtime_policy_is_still_a_callers_choice(corpus: Corpus) -> None:
    runner, tree = merged_marker_tree(corpus, refine=REFINE)

    # A caller who asks for the runtime policy gets it, tables demoted and all:
    # the research policy is not a new default for everybody else.
    assert merged_kinds(runner, tree).count("table") == 0


def test_a_changed_merge_policy_rebuilds_the_tree_and_reuses_the_extraction(
    corpus: Corpus,
) -> None:
    engine = FakeEngine("marker")
    runner = make_runner(corpus, {"marker": engine}, settings=OtherModelSettings())
    run = runner.extract("digital-1", "marker")
    # The same merge model, so the policy is the only thing that differs.
    runtime_policy = RefineConfig(refine_details=True, llm_model="merge-model-b")

    def build(refine: Any | None = None) -> dict[str, Any]:
        return runner.run_tree(
            "digital-1",
            "marker_llm",
            extract_run_id=run["run_id"],
            llm_mode="live",
            allow_merge_upload=True,
            llm_client=FakeMergeClient(),
            refine_config=refine,
        )

    scored = build()
    runtime = build(runtime_policy)

    assert runtime["run_id"] != scored["run_id"]
    assert runtime["semantic_fingerprint"] != scored["semantic_fingerprint"]
    assert runtime["artifact_digest"] == scored["artifact_digest"]
    # The policy each tree was built with is readable beside its fingerprint.
    assert scored["refine"]["preserve_tables"] is True
    assert scored["refine"]["segment_summaries"] is False
    assert runtime["refine"]["preserve_tables"] is False
    assert runtime["refine"]["segment_summaries"] is True
    assert len(engine.calls) == 1  # the extraction was replayed, not repaid


# --------------------------------------------------------------------------- #
# the execution plan: arms, repeats, and cells that must not disappear
# --------------------------------------------------------------------------- #


def comparison_report(result: dict[str, Any], index: int = 0) -> dict[str, Any]:
    """The report a study comparison wrote, read back from disk."""
    return json.loads((Path(result["comparisons"][index]["path"]) / "report.json").read_text())


def test_the_chunk_family_pairs_the_grouped_tree_with_the_merged_one(
    corpus: Corpus, monkeypatch: pytest.MonkeyPatch
) -> None:
    engine = FakeVisionEngine()
    runner = make_runner(corpus, {"vision": engine}, settings=StubSettings())
    monkeypatch.setattr(StudyRunner, "_gateway_client", lambda _self: FakeMergeClient())

    result = runner.study(
        families=["chunk"],
        engines=["vision"],
        profiles=["vision_direct", "vision_atomic", "vision_atomic_merge"],
        allow_upload=True,
        allow_merge_upload=True,
        llm_mode="live",
    )

    assert result["skipped"] == []
    assert [entry["family"] for entry in result["comparisons"]] == ["chunk"] * 3
    pairs = comparison_report(result)["paired_usage"]["pairs"]
    # One fixed pair per document: the grouped tree against the merged one. The
    # atomic tree is the control the merge was built from, not an arm of it.
    assert len(pairs) == 1
    assert {pairs[0]["run_a"].split(".")[1], pairs[0]["run_b"].split(".")[1]} == {
        "vision_direct",
        "vision_atomic_merge",
    }


def test_the_chunk_family_is_skipped_when_only_one_arm_ran(corpus: Corpus) -> None:
    engine = FakeVisionEngine()
    runner = make_runner(corpus, {"vision": engine}, settings=StubSettings())

    # No merge mode, so the merged arm cannot exist: a single arm is not a
    # comparison, and it is skipped by name rather than compared with itself.
    result = runner.study(
        families=["chunk"],
        engines=["vision"],
        profiles=["vision_direct", "vision_atomic"],
        allow_upload=True,
    )

    assert result["comparisons"] == []
    assert [entry["family"] for entry in result["skipped"]] == ["chunk"] * 3
    assert {entry["reason"] for entry in result["skipped"]} == {"no_chunk_profile_tree"}


def test_the_chunk_family_is_skipped_when_the_merge_merged_nothing(
    corpus: Corpus, monkeypatch: pytest.MonkeyPatch
) -> None:
    engine = FakeVisionEngine()
    runner = make_runner(corpus, {"vision": engine}, settings=StubSettings())
    monkeypatch.setattr(StudyRunner, "_gateway_client", lambda _self: FakeMergeClient())

    def no_sections(*_args: Any, **_kwargs: Any) -> Any:
        return {"sections": 0, "sections_merged": 0, "sections_skipped": 0, "llm_calls": 0}

    monkeypatch.setattr("research.ocr_pipeline.runner.merge_tree", no_sections)

    result = runner.study(
        families=["chunk"],
        engines=["vision"],
        profiles=["vision_direct", "vision_atomic_merge"],
        allow_upload=True,
        allow_merge_upload=True,
        llm_mode="live",
    )

    # Two trees exist, but the merged one merged nothing: it is not a merge, and
    # comparing against it would report a difference the merge never made.
    assert result["comparisons"] == []
    assert {entry["reason"] for entry in result["skipped"]} == {"no_merged_section"}


def test_the_chunk_family_is_skipped_when_the_merge_kept_every_original(
    corpus: Corpus, monkeypatch: pytest.MonkeyPatch
) -> None:
    """An arm whose merge kept every original is the atomic tree, not a merge.

    The defect this guards: the pair tested ``sections_merged > 0``, a counter
    that counts fallbacks too, so a merge that reached no model still entered the
    pair and the comparison reported a difference between a tree and itself. The
    counters here are a merge that ran over real sections and rebuilt none of
    them, which only the fallback counter tells apart from one that did.
    """
    engine = FakeVisionEngine()
    runner = make_runner(corpus, {"vision": engine}, settings=StubSettings())
    monkeypatch.setattr(StudyRunner, "_gateway_client", lambda _self: UnusableMergeClient())

    result = runner.study(
        families=["chunk"],
        engines=["vision"],
        profiles=["vision_direct", "vision_atomic_merge"],
        allow_upload=True,
        allow_merge_upload=True,
        llm_mode="live",
    )

    merged = [
        manifest
        for manifest in runner.list_runs(kind=KIND_TREE)
        if manifest.get("profile") == "vision_atomic_merge"
    ]
    assert len(merged) == 3  # one per document, and each one ran its merge
    assert all(entry["merge"]["sections_merged"] > 0 for entry in merged)
    assert all(
        entry["merge"]["sections_merged"] == entry["merge"]["sections_fallback"]
        for entry in merged
    )
    assert result["comparisons"] == []
    assert {entry["reason"] for entry in result["skipped"]} == {"no_merged_section"}


def test_a_failed_engine_still_has_a_row_of_its_own(corpus: Corpus) -> None:
    good = FakeEngine("native")
    bad = FakeEngine("docling")
    bad.error = RuntimeError("the engine gave up")
    runner = make_runner(corpus, {"native": good, "docling": bad})

    result = runner.study(families=["ocr"], engines=["native", "docling"])

    assert len(result["extractions"]) == 6  # three documents, two engines
    assert {entry["status"] for entry in result["extractions"]} == {RUN_OK, RUN_FAILED}
    rows = comparison_report(result)["metadata"]["run_rows"]
    assert len(rows) == 2  # both participants, failure included
    assert {row["run_id"] for row in rows} == {"digital-1.native.r1", "digital-1.docling.r1"}
    failed = next(row for row in rows if row["status"] == RUN_FAILED)
    assert failed["artifact"]["status"] == "not_measured"
    assert failed["artifact"]["reason"] == "engine_failed"


def test_a_study_runs_every_repeat_and_keeps_them_apart(corpus: Corpus) -> None:
    engine = FakeVisionEngine()
    engine.varying = True
    runner = make_runner(corpus, {"vision": engine}, settings=StubSettings())

    result = runner.study(families=["ocr"], engines=["vision"], repeats=2, allow_upload=True)

    assert len(engine.calls) == 6  # three documents, two repeats, really extracted
    run_ids = [entry["run_id"] for entry in result["extractions"]]
    assert len(set(run_ids)) == 6
    assert {entry["run_id"].rsplit(".r", 1)[1] for entry in result["extractions"]} == {"1", "2"}


def test_a_dry_run_reports_the_profiles_execute_would_build(corpus: Corpus) -> None:
    engine = FakeVisionEngine()
    runner = make_runner(corpus, {"vision": engine}, settings=StubSettings())

    planned = runner.study(
        families=["pipeline"],
        engines=["vision"],
        profiles=["vision_direct"],
        dry_run=True,
    )["plan"]

    assert {entry["profile"] for entry in planned["trees"]} == {"vision_direct"}
    assert planned["profiles"] == ["vision_direct"]
    assert engine.calls == []


# --------------------------------------------------------------------------- #
# R2: the resume policy, the identity of a tree request, fresh repeats
# --------------------------------------------------------------------------- #


def tree_runs(runner: StudyRunner, doc_id: str) -> list[dict[str, Any]]:
    """Every tree run recorded for one document."""
    return runner.list_runs(doc_id, kind=KIND_TREE)


class OtherModelSettings(StubSettings):
    """The same settings, merging with a different model."""

    BENCHMARK_LLM_MODEL = "merge-model-b"


def test_a_study_resumes_the_tree_it_already_built(corpus: Corpus) -> None:
    engine = FakeEngine("native")
    runner = make_runner(corpus, {"native": engine})

    first = runner.study(families=["pipeline"], engines=["native"], resume=True)
    before = {
        doc_id: len(tree_runs(runner, doc_id)) for doc_id in ("digital-1", "scan-1", "mixed-1")
    }

    second = runner.study(families=["pipeline"], engines=["native"], resume=True)

    assert len(engine.calls) == 3  # the extractions were reused, not repaid
    assert {doc_id: len(tree_runs(runner, doc_id)) for doc_id in before} == before
    assert [entry["run_id"] for entry in second["trees"]] == [
        entry["run_id"] for entry in first["trees"]
    ]


def test_a_fresh_study_extracts_every_repeat_again(corpus: Corpus) -> None:
    engine = FakeEngine("native")
    engine.varying = True
    runner = make_runner(corpus, {"native": engine})

    first = runner.study(families=["ocr"], engines=["native"], repeats=2, resume=False)
    calls = len(engine.calls)

    assert calls == 6  # three documents, two repeats, every one really run

    second = runner.study(families=["ocr"], engines=["native"], repeats=2, resume=False)

    # A fresh study never hands back an old artifact as a new repeat: both
    # repeats run again, and both land in run ids of their own.
    assert len(engine.calls) == calls + 6
    assert {entry["run_id"] for entry in second["extractions"]} != {
        entry["run_id"] for entry in first["extractions"]
    }
    assert all(entry["status"] == RUN_OK for entry in second["extractions"])


def test_a_zero_call_resume_asks_for_no_upload_permission(corpus: Corpus) -> None:
    engine = FakeVisionEngine()
    runner = make_runner(corpus, {"vision": engine}, settings=StubSettings())

    paid = runner.study(families=["ocr"], engines=["vision"], allow_upload=True, resume=True)

    # The pages are already on disk. A resume that calls nothing has nothing to
    # permit, so the identical study runs with the upload flag withheld.
    again = runner.study(families=["ocr"], engines=["vision"], resume=True)

    assert again["executed"] is True
    assert len(engine.calls) == 3
    assert [entry["run_id"] for entry in again["extractions"]] == [
        entry["run_id"] for entry in paid["extractions"]
    ]


def test_a_zero_call_resume_builds_no_live_gateway_client(
    corpus: Corpus, monkeypatch: pytest.MonkeyPatch
) -> None:
    engine = FakeVisionEngine()
    runner = make_runner(corpus, {"vision": engine}, settings=StubSettings())
    built: list[FakeMergeClient] = []

    def build() -> Any:
        built.append(FakeMergeClient())
        return built[-1]

    monkeypatch.setattr(StudyRunner, "_gateway_client", staticmethod(build))
    kwargs: dict[str, Any] = {
        "families": ["pipeline"],
        "engines": ["vision"],
        "profiles": ["vision_atomic_merge"],
        "doc_ids": ["digital-1"],
        "allow_upload": True,
        "allow_merge_upload": True,
        "llm_mode": "live",
        "resume": True,
    }
    runner.study(**kwargs)

    assert built and built[0].calls > 0  # the first study really did merge

    def forbidden() -> Any:
        raise AssertionError("a zero-call resume built a live gateway client")

    monkeypatch.setattr(StudyRunner, "_gateway_client", staticmethod(forbidden))
    again = runner.study(**kwargs)

    # Every run was on disk, so the second study made no call at all -- and a
    # run that calls nothing must not reach for a credential either.
    assert again["executed"] is True
    assert len(engine.calls) == 1
    assert len(tree_runs(runner, "digital-1")) == 1


def test_a_changed_merge_model_rebuilds_the_tree_and_reuses_the_extraction(
    corpus: Corpus, monkeypatch: pytest.MonkeyPatch
) -> None:
    engine = FakeVisionEngine()
    monkeypatch.setattr(StudyRunner, "_gateway_client", staticmethod(FakeMergeClient))
    kwargs: dict[str, Any] = {
        "families": ["pipeline"],
        "engines": ["vision"],
        "profiles": ["vision_atomic_merge"],
        "doc_ids": ["digital-1"],
        "allow_upload": True,
        "allow_merge_upload": True,
        "llm_mode": "live",
        "resume": True,
    }
    first = make_runner(corpus, {"vision": engine}, settings=StubSettings()).study(**kwargs)

    # The same corpus, the same artifact, a different merge model: the
    # extraction is still the answer to its question, the tree is not.
    other = make_runner(corpus, {"vision": engine}, settings=OtherModelSettings())
    second = other.study(**kwargs)

    assert len(engine.calls) == 1
    assert [entry["run_id"] for entry in second["extractions"]] == [
        entry["run_id"] for entry in first["extractions"]
    ]
    assert len(tree_runs(other, "digital-1")) == 2
    assert [entry["run_id"] for entry in second["trees"]] != [
        entry["run_id"] for entry in first["trees"]
    ]


def test_a_legacy_tree_without_a_fingerprint_is_a_cache_miss(corpus: Corpus) -> None:
    engine = FakeEngine("native")
    runner = make_runner(corpus, {"native": engine})
    run = runner.extract("digital-1", "native")
    first = runner.run_tree("digital-1", "native_atomic", extract_run_id=run["run_id"])

    # A tree written before the semantic fingerprint existed cannot say which
    # question it answers, so it is not reused as the answer to this one.
    path = Path(first["path"]) / "manifest.json"
    manifest = json.loads(path.read_text(encoding="utf-8"))
    manifest.pop("semantic_fingerprint")
    path.write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8")

    again = runner.run_tree(
        "digital-1", "native_atomic", extract_run_id=run["run_id"], resume=True
    )

    assert again["run_id"] == f"{first['run_id']}-a2"
    assert again["semantic_fingerprint"] is not None
    assert len(tree_runs(runner, "digital-1")) == 2
    assert len(engine.calls) == 1  # only the tree was rebuilt


def test_an_interrupted_tree_is_never_resumed_as_a_success(corpus: Corpus) -> None:
    engine = FakeEngine("native")
    runner = make_runner(corpus, {"native": engine})
    run = runner.extract("digital-1", "native")
    first = runner.run_tree("digital-1", "native_atomic", extract_run_id=run["run_id"])

    # The manifest is written last, so a run whose tree file is missing was
    # interrupted: the record of a half-written tree must never be handed back.
    (Path(first["path"]) / "tree.json").unlink()

    again = runner.run_tree(
        "digital-1", "native_atomic", extract_run_id=run["run_id"], resume=True
    )

    assert again["run_id"] != first["run_id"]
    assert again["status"] == RUN_OK
    assert (Path(again["path"]) / "tree.json").is_file()
    assert len(tree_runs(runner, "digital-1")) == 2


def test_a_dry_run_reports_a_cache_miss_before_anything_is_spent(
    corpus: Corpus, monkeypatch: pytest.MonkeyPatch
) -> None:
    engine = FakeEngine("marker")
    monkeypatch.setattr(StudyRunner, "_gateway_client", staticmethod(FakeMergeClient))
    runner = make_runner(corpus, {"marker": engine})
    kwargs: dict[str, Any] = {
        "families": ["pipeline"],
        "engines": ["marker"],
        "profiles": ["marker_llm"],
        "doc_ids": ["mixed-1"],
        "llm_mode": "live",
        "allow_merge_upload": True,
        "resume": True,
    }

    planned = runner.study(dry_run=True, **kwargs)["plan"]

    assert planned["cache"] == {"hits": 0, "misses": 2}
    assert {entry["cache"] for entry in planned["extractions"]} == {"miss"}
    assert {entry["cache"] for entry in planned["trees"]} == {"miss"}
    assert engine.calls == []

    runner.study(**kwargs)
    reused = runner.study(dry_run=True, **kwargs)["plan"]

    assert reused["cache"] == {"hits": 2, "misses": 0}
    assert {entry["cache_reason"] for entry in reused["trees"]} == {None}

    # A different merge model is a different tree, and the dry run says so
    # before the calls are made rather than after.
    other = make_runner(corpus, {"marker": engine}, settings=OtherModelSettings())
    rebuilt = other.study(dry_run=True, **kwargs)["plan"]

    assert rebuilt["cache"] == {"hits": 1, "misses": 1}
    assert rebuilt["extractions"][0]["cache"] == "hit"
    assert rebuilt["trees"][0]["cache"] == "miss"
    assert rebuilt["trees"][0]["cache_reason"] == "different_request"


def test_a_changed_merge_prompt_invalidates_the_tree_and_reuses_the_extraction(
    corpus: Corpus, monkeypatch: pytest.MonkeyPatch
) -> None:
    from research.ocr_pipeline import runner as runner_module

    engine = FakeVisionEngine()
    monkeypatch.setattr(StudyRunner, "_gateway_client", staticmethod(FakeMergeClient))
    runner = make_runner(corpus, {"vision": engine}, settings=StubSettings())
    kwargs: dict[str, Any] = {
        "families": ["pipeline"],
        "engines": ["vision"],
        "profiles": ["vision_atomic_merge"],
        "doc_ids": ["digital-1"],
        "allow_upload": True,
        "allow_merge_upload": True,
        "llm_mode": "live",
        "resume": True,
    }
    runner.study(**kwargs)

    # The prompt's own source is what a tree's identity carries, so a reworded
    # template is a different tree even though the model has not moved.
    digest = runner_module._merge_prompt_sha256()
    assert isinstance(digest, str) and len(digest) == 64
    monkeypatch.setattr(runner_module, "_merge_prompt_sha256", lambda: "0" * 64)
    runner.study(**kwargs)

    assert len(engine.calls) == 1
    assert len(tree_runs(runner, "digital-1")) == 2


def test_the_plan_names_the_extraction_run_the_tree_will_replay(corpus: Corpus) -> None:
    """The source run is part of a tree's identity, and the plan names the one used.

    Two extractions can leave two artifacts with one digest -- a repeat, a second
    read of the same pages -- and the tree built over one of them is not the tree
    built over the other. The plan resolves the extraction run first and computes
    the tree's verdict against that record, so a resume meets the tree only when
    the source it names is the source it replays; without it the second run's tree
    would answer with the first's, and a study would report a tree it never built.
    """
    engine = FakeEngine("native")
    runner = make_runner(corpus, {"native": engine})
    first = runner.extract("digital-1", "native")
    second = runner.extract("digital-1", "native")

    assert second["run_id"] != first["run_id"]
    assert second["artifact_digest"] == first["artifact_digest"]  # one artifact, two runs

    built = runner.run_tree("digital-1", "native_atomic", extract_run_id=first["run_id"])

    assert built["extract_run_id"] == first["run_id"]

    # A resume over the other source is not this tree, whatever the digest says.
    other = runner.run_tree(
        "digital-1", "native_atomic", extract_run_id=second["run_id"], resume=True
    )

    assert other["run_id"] != built["run_id"]
    assert other["extract_run_id"] == second["run_id"]

    # The plan reuses the earliest attempt that answers and names the run it will
    # replay, so a study that follows it lands on the record it just promised.
    planned = runner.plan(
        doc_ids=["digital-1"], engine_ids=("native",), profile_names=["native_atomic"], resume=True
    )
    line = next(entry for entry in planned["extractions"] if entry["repeat"] == 1)
    tree = next(entry for entry in planned["trees"] if entry["repeat"] == 1)

    assert (line["run_id"], line["cache"]) == (first["run_id"], "hit")
    assert (tree["run_id"], tree["cache"]) == (built["run_id"], "hit")

    studied = runner.study(
        families=["pipeline"], engines=["native"], doc_ids=["digital-1"], dry_run=True
    )["plan"]

    assert [entry["run_id"] for entry in studied["extractions"]] == [first["run_id"]]
    assert [entry["run_id"] for entry in studied["trees"]] == [built["run_id"]]


def test_a_tree_from_a_bare_digest_names_no_source_run(corpus: Corpus) -> None:
    """A digest is not an id: a tree built from one records no source run.

    The ``run`` command can replay a bare artifact digest, and nothing on disk
    then says which extraction run left it, so the manifest records that as
    unknown rather than reading the digest to guess an id it does not carry. The
    unknown source answers its own request and is not confused with a tree that
    can name the run it replays -- the two records differ in what they claim.
    """
    engine = FakeEngine("native")
    runner = make_runner(corpus, {"native": engine})
    run = runner.extract("digital-1", "native")
    digest = str(run["artifact_digest"])

    unnamed = runner.run_tree("digital-1", "native_atomic", artifact_digest=digest)
    resumed = runner.run_tree("digital-1", "native_atomic", artifact_digest=digest, resume=True)
    named = runner.run_tree(
        "digital-1", "native_atomic", extract_run_id=run["run_id"], resume=True
    )

    assert unnamed["extract_run_id"] is None
    assert resumed["run_id"] == unnamed["run_id"]  # the same unknown source answers it
    assert named["run_id"] != unnamed["run_id"]
    assert named["extract_run_id"] == run["run_id"]


def test_the_merge_limit_is_part_of_the_tree_the_plan_prints(corpus: Corpus) -> None:
    """One artifact and one model merged under two limits are two trees.

    The limit is a field of the merge configuration, and the request key names
    that configuration, so a study that widens its merge pays for the trees the
    narrower one built -- and the dry run prints the limit it would merge with,
    beside the cache verdict the limit decides.
    """
    narrow = make_runner(
        corpus,
        {"marker": FakeEngine("marker")},
        settings=StubSettings(),
        merge_concurrency=1,
    )
    run = narrow.extract("mixed-1", "marker")
    first = merged_attempt(
        narrow, run, MeteredMergeClient([{"prompt_tokens": 10, "completion_tokens": 5}])
    )

    assert first["refine"]["max_concurrent_sections"] == 1
    wide_engine = FakeEngine("marker")
    wide = make_runner(
        corpus,
        {"marker": wide_engine},
        settings=StubSettings(),
        merge_concurrency=4,
    )
    planned = wide.plan(
        doc_ids=["mixed-1"],
        engine_ids=("marker",),
        profile_names=["marker_llm"],
        llm_mode="live",
        resume=True,
    )
    line = next(entry for entry in planned["trees"] if entry["doc_id"] == "mixed-1")

    assert planned["merge_concurrency"] == 4
    # The artifact answers the extraction; the tree does not answer this merge.
    assert planned["extractions"][0]["cache"] == "hit"
    assert (line["cache"], line["cache_reason"]) == ("miss", "different_request")

    again = merged_attempt(
        wide, run, MeteredMergeClient([{"prompt_tokens": 10, "completion_tokens": 5}])
    )

    assert again["run_id"] != first["run_id"]
    assert again["refine"]["max_concurrent_sections"] == 4
    assert wide_engine.calls == []  # the extraction was reused, not repaid


def test_a_changed_chunk_threshold_invalidates_the_tree_and_reuses_the_extraction(
    corpus: Corpus, monkeypatch: pytest.MonkeyPatch
) -> None:
    engine = FakeEngine("native")
    runner = make_runner(corpus, {"native": engine})
    kwargs: dict[str, Any] = {
        "families": ["pipeline"],
        "engines": ["native"],
        "doc_ids": ["digital-1"],
        "resume": True,
    }
    runner.study(**kwargs)

    # The chunk threshold decides what a chunk is, so it decides what the tree
    # is; the artifact it was read from is untouched.
    monkeypatch.setattr("config.chunking.MAX_CHUNK_SIZE", 4096)
    runner.study(**kwargs)

    assert len(engine.calls) == 1
    assert len(tree_runs(runner, "digital-1")) == 2


def test_a_gold_change_re_scores_without_reading_anything_again(corpus: Corpus) -> None:
    """Gold is evaluation provenance, not extraction identity.

    A study handed a different reference re-scores the runs it already has: no
    page is read again and no tree is rebuilt, and the comparison that lands on
    disk is a new one, scored against the gold it was given.
    """
    revised = gold_payload("digital-1", [0, 1, 2])
    revised["pages"][0]["headings"] = []
    (corpus.base_dir / "gold-revised.json").write_text(json.dumps(revised), encoding="utf-8")
    engine = FakeEngine("native")
    runner = make_runner(corpus, {"native": engine})
    first = runner.study(families=["pipeline"], engines=["native"], doc_ids=["digital-1"])

    second = runner.study(
        families=["pipeline"],
        engines=["native"],
        doc_ids=["digital-1"],
        gold_paths={"digital-1": "gold-revised.json"},
    )

    assert len(engine.calls) == 1
    assert len(tree_runs(runner, "digital-1")) == 1
    assert [entry["path"] for entry in second["comparisons"]] != [
        entry["path"] for entry in first["comparisons"]
    ]
    assert (
        comparison_report(first)["metadata"]["run_rows"][0]["gold_digest"]
        != (comparison_report(second)["metadata"]["run_rows"][0]["gold_digest"])
    )


# --------------------------------------------------------------------------- #
# R2: a run whose merge rebuilt nothing does not answer a merge request
# --------------------------------------------------------------------------- #


def merged_attempt(runner: StudyRunner, run: Mapping[str, Any], client: Any) -> dict[str, Any]:
    """One live merge over ``mixed-1``'s marker artifact, with ``client``.

    Written without ``resume``, so every call leaves the next attempt under the
    one run id and each of them answers the identical request. The merge policy
    is the runner's own, the same one ``plan`` resolves, so a dry run computes
    the key these attempts carry.
    """
    return runner.run_tree(
        "mixed-1",
        "marker_llm",
        extract_run_id=str(run["run_id"]),
        llm_mode="live",
        allow_merge_upload=True,
        llm_client=client,
    )


def restamp_measured(attempt: Mapping[str, Any]) -> None:
    """Rewrite a stored verdict to the one the pre-fix code wrote: measured.

    Evidence on disk carries the verdict of the code that ran, and that code
    read only the merge mode, so a live merge that reached no model was stamped
    ``measured`` and ``rankable``. The counters beside it are the fact, and this
    makes a test's record carry exactly that stale stamp.
    """
    path = Path(str(attempt["path"])) / "manifest.json"
    manifest = json.loads(path.read_text(encoding="utf-8"))
    manifest["quality"] = "measured"
    manifest["quality_reason"] = None
    manifest["rankable"] = True
    path.write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def test_a_resume_walks_past_a_merge_that_rebuilt_nothing(corpus: Corpus) -> None:
    """The measured shape: one request, three attempts, the merge only in the last.

    ``.live`` and ``.live-a2`` reached no model, so every section kept its
    originals and each record holds the tree the merge ran over under the
    merge's name -- stamped measured by the code of the day, which the restamp
    above reproduces, so this proves the choice is made on the counters and not
    on the stamp. Only ``-a3`` really merged, and it is the attempt a resume
    must hand back: handing back the first is what kept a study reporting a
    merge it never made, study after study, with no way to recover.
    """
    runner = make_runner(corpus, {"marker": FakeEngine("marker")}, settings=StubSettings())
    run = runner.extract("mixed-1", "marker")
    first = merged_attempt(runner, run, UnreachableMergeClient())
    second = merged_attempt(runner, run, UnreachableMergeClient())
    merged = merged_attempt(
        runner,
        run,
        MeteredMergeClient([{"prompt_tokens": 1200, "completion_tokens": 800}]),
    )

    assert [entry["run_id"] for entry in (first, second, merged)] == [
        "mixed-1.marker_llm.r1.live",
        "mixed-1.marker_llm.r1.live-a2",
        "mixed-1.marker_llm.r1.live-a3",
    ]
    assert first["request_key"] == merged["request_key"]  # one request, three answers
    for stale in (first, second):
        assert stale["merge"]["sections_merged"] == stale["merge"]["sections_fallback"] > 0
        restamp_measured(stale)

    client = MeteredMergeClient([{"prompt_tokens": 1200, "completion_tokens": 800}])
    resumed = runner.run_tree(
        "mixed-1",
        "marker_llm",
        extract_run_id=run["run_id"],
        llm_mode="live",
        allow_merge_upload=True,
        llm_client=client,
        resume=True,
    )

    assert resumed["run_id"] == "mixed-1.marker_llm.r1.live-a3"
    assert resumed["path"] == merged["path"]
    assert client.calls == 0  # the merging record was read, not merged again
    assert len(tree_runs(runner, "mixed-1")) == 3  # and nothing new was written


def test_the_plan_and_the_run_pick_the_merging_attempt_together(corpus: Corpus) -> None:
    """A dry run and the run it plans must name one attempt, merges included.

    The plan and the run share ``_resume_candidate``, so a plan that named the
    rebuilt-nothing attempt while the run walked past it would be a promise the
    run does not keep.
    """
    runner = make_runner(corpus, {"marker": FakeEngine("marker")}, settings=StubSettings())
    run = runner.extract("mixed-1", "marker")
    merged_attempt(runner, run, UnreachableMergeClient())
    merged = merged_attempt(
        runner,
        run,
        MeteredMergeClient([{"prompt_tokens": 1200, "completion_tokens": 800}]),
    )

    planned = runner.plan(
        doc_ids=["mixed-1"],
        engine_ids=("marker",),
        profile_names=["marker_llm"],
        llm_mode="live",
        resume=True,
    )
    line = next(entry for entry in planned["trees"] if entry["doc_id"] == "mixed-1")
    assert (line["run_id"], line["cache"], line["cache_reason"]) == (
        "mixed-1.marker_llm.r1.live-a2",
        "hit",
        None,
    )

    client = MeteredMergeClient([{"prompt_tokens": 1200, "completion_tokens": 800}])
    landed = runner.run_tree(
        "mixed-1",
        "marker_llm",
        extract_run_id=run["run_id"],
        llm_mode="live",
        allow_merge_upload=True,
        llm_client=client,
        resume=True,
    )

    assert landed["run_id"] == line["run_id"]
    assert landed["path"] == merged["path"]
    assert client.calls == 0  # the plan named a record and the run read it


def test_a_resume_with_no_merged_attempt_anywhere_allocates_and_merges(corpus: Corpus) -> None:
    """When no attempt on disk merged, the resume pays for one instead of reusing a lie.

    Every attempt so far kept its originals, so none answers the request. The
    resume must not hand back the first and must not give up: it allocates the
    next attempt and really merges into it -- and the plan says why the record
    it found was not an answer, before the money is spent.
    """
    runner = make_runner(corpus, {"marker": FakeEngine("marker")}, settings=StubSettings())
    run = runner.extract("mixed-1", "marker")
    first = merged_attempt(runner, run, UnreachableMergeClient())

    planned = runner.plan(
        doc_ids=["mixed-1"],
        engine_ids=("marker",),
        profile_names=["marker_llm"],
        llm_mode="live",
        resume=True,
    )
    line = next(entry for entry in planned["trees"] if entry["doc_id"] == "mixed-1")
    # The record on disk merged nothing, so the run pays for a real merge, in the
    # directory it will allocate: the plan names that directory, not the id it
    # would have asked for and the run walks past.
    assert (line["run_id"], line["cache"], line["cache_reason"]) == (
        "mixed-1.marker_llm.r1.live-a2",
        "miss",
        "merge_rebuilt_nothing",
    )

    client = MeteredMergeClient([{"prompt_tokens": 1200, "completion_tokens": 800}])
    landed = runner.run_tree(
        "mixed-1",
        "marker_llm",
        extract_run_id=run["run_id"],
        llm_mode="live",
        allow_merge_upload=True,
        llm_client=client,
        resume=True,
    )

    assert landed["run_id"] == line["run_id"]  # the plan named the directory the run took
    assert landed["path"] != first["path"]
    assert client.calls > 0  # the merge really ran this time
    assert landed["merge"]["sections_merged"] > landed["merge"]["sections_fallback"]
    assert landed["quality"] == "measured"
    assert landed["rankable"] is True
    assert len(tree_runs(runner, "mixed-1")) == 2


def test_a_merge_off_tree_still_resumes_its_run(corpus: Corpus) -> None:
    """The profiles with no merge are untouched: their zero counters answer them.

    A merge-off run records the zero placeholder, and a merge that rebuilt no
    section is false for it by construction. A request that asks for no merge is
    answered by exactly that record, so a fix for the merge case must not make
    the atomic profiles unresumable and rebuild a study's trees every time.
    """
    engine = FakeEngine("native")
    runner = make_runner(corpus, {"native": engine})
    run = runner.extract("digital-1", "native")
    first = runner.run_tree("digital-1", "native_atomic", extract_run_id=run["run_id"])

    assert first["merge"] == {
        "sections": 0,
        "sections_merged": 0,
        "sections_skipped": 0,
        "llm_calls": 0,
    }
    assert first["rankable"] is True  # the run is a result, counters and all

    planned = runner.plan(
        doc_ids=["digital-1"],
        engine_ids=("native",),
        profile_names=["native_atomic"],
        resume=True,
    )
    line = next(entry for entry in planned["trees"] if entry["doc_id"] == "digital-1")
    assert (line["run_id"], line["cache"], line["cache_reason"]) == (
        "digital-1.native_atomic.r1.none",
        "hit",
        None,
    )

    resumed = runner.run_tree(
        "digital-1", "native_atomic", extract_run_id=run["run_id"], resume=True
    )

    assert resumed["run_id"] == first["run_id"]
    assert resumed["path"] == first["path"]
    assert len(tree_runs(runner, "digital-1")) == 1  # the record was read, not rebuilt


def test_a_merge_profile_asked_for_no_merge_is_still_resumable(corpus: Corpus) -> None:
    """What decides is the request's merge, not the profile's declared one.

    The same profile can be asked for a tree with no merge, and that request is
    answered by a run whose merge counters are the zero placeholder. Reading the
    profile's ``merge`` instead of the request's mode would refuse the record
    and rebuild a tree that answers exactly what was asked.
    """
    engine = FakeVisionEngine()
    runner = make_runner(corpus, {"vision": engine}, settings=StubSettings())
    run = runner.extract(
        "digital-1", "vision", allow_upload=True, vision_config=StubVisionConfig()
    )
    first = runner.run_tree("digital-1", "vision_atomic_merge", extract_run_id=run["run_id"])

    assert first["stages"]["merge"]["reason"] == "merge_not_run"
    assert first["merge"]["sections"] == 0

    resumed = runner.run_tree(
        "digital-1", "vision_atomic_merge", extract_run_id=run["run_id"], resume=True
    )

    assert resumed["run_id"] == first["run_id"]
    assert resumed["path"] == first["path"]
    assert len(tree_runs(runner, "digital-1")) == 1


def test_a_study_recovers_from_a_merge_that_rebuilt_nothing(
    corpus: Corpus, monkeypatch: pytest.MonkeyPatch
) -> None:
    """One bad merge must not poison every later study, as a live run measured.

    The first study's merged arm ran its merge and kept every original, so the
    chunk family had nothing to compare and said so. The second study finds that
    record on disk and must walk past it, merge for real and let the chunk family
    run -- the defect was that a resume kept handing the bad record back, so the
    study could never recover on its own.
    """
    engine = FakeVisionEngine()
    runner = make_runner(corpus, {"vision": engine}, settings=StubSettings())
    kwargs: dict[str, Any] = {
        "families": ["chunk"],
        "engines": ["vision"],
        "profiles": ["vision_direct", "vision_atomic_merge"],
        "doc_ids": ["digital-1"],
        "allow_upload": True,
        "allow_merge_upload": True,
        "llm_mode": "live",
        "resume": True,
    }
    monkeypatch.setattr(StudyRunner, "_gateway_client", lambda _self: UnreachableMergeClient())

    first = runner.study(**kwargs)

    assert [entry["run_id"] for entry in first["trees"]] == [
        "digital-1.vision_direct.r1.live",
        "digital-1.vision_atomic_merge.r1.live",
    ]
    assert first["comparisons"] == []
    assert [entry["reason"] for entry in first["skipped"]] == ["no_merged_section"]

    good = MeteredMergeClient([{"prompt_tokens": 1200, "completion_tokens": 800}])
    monkeypatch.setattr(StudyRunner, "_gateway_client", lambda _self: good)
    second = runner.study(**kwargs)

    merged = next(entry for entry in second["trees"] if entry["profile"] == "vision_atomic_merge")
    assert merged["run_id"] == "digital-1.vision_atomic_merge.r1.live-a2"
    assert merged["rankable"] is True
    assert good.calls > 0
    assert second["skipped"] == []
    assert [entry["family"] for entry in second["comparisons"]] == ["chunk"]
    assert len(engine.calls) == 1  # the extraction was reused; only the tree was rebuilt


def test_a_page_override_is_the_selection_the_coverage_counts(corpus: Corpus) -> None:
    engine = FakeEngine("native")
    runner = make_runner(corpus, {"native": engine})

    planned = runner.study(
        families=["pipeline"],
        engines=["native"],
        doc_ids=["digital-1"],
        page_indices=[0, 1],
        dry_run=True,
    )["plan"]

    assert planned["corpus"]["pages"] == {"digital-1": 2}
    assert planned["extractions"][0]["pages"] == [0, 1]

    runner.study(
        families=["pipeline"], engines=["native"], doc_ids=["digital-1"], page_indices=[0, 1]
    )
    tree = tree_runs(runner, "digital-1")[0]
    coverage = runner.read_run_file("digital-1", tree["run_id"], "metrics.json")["chunk_metrics"][
        "page_fidelity"
    ]["coverage"]

    # Two pages were read, so two pages are in scope: the manifest's third page
    # was never asked for, and must not be reported as uncovered.
    assert coverage["status"] == "measured"
    assert coverage["denominator"] == 2


def test_paired_documents_counts_documents_not_pairs(corpus: Corpus) -> None:
    """Three participants make three pairs and still one document.

    ``paired_documents`` is the statistical denominator: a document measured by
    three engines is one document. Counting it once per pair would let a study
    grow its sample by adding engines instead of documents.
    """
    runner = make_runner(
        corpus,
        {
            "native": FakeEngine("native"),
            "docling": FakeEngine("docling"),
            "marker": FakeEngine("marker"),
        },
    )
    runs = [
        RunRef("digital-1", runner.extract("digital-1", engine)["run_id"])
        for engine in ("native", "docling", "marker")
    ]

    paired = runner.compare("ocr", runs, write=False)["report"]["paired_usage"]

    assert paired["n_pairs"] == 3
    assert paired["n_repeats"] == 1
    assert paired["paired_documents"] == 1
    assert paired["union_documents"] == 1
    assert all(pair["compatible"] for pair in paired["pairs"])
    # The document count is the same whether one pair or three measured it.
    assert {pair["documents_in_a"] for pair in paired["pairs"]} == {1}


def test_the_study_summary_is_per_family_and_pair_over_the_corpus(corpus: Corpus) -> None:
    """The roll-up a study is read for: whole corpus, per family and per pair."""
    runner = make_runner(
        corpus, {"native": FakeEngine("native"), "docling": FakeEngine("docling")}
    )
    gold = corpus.base_dir / "gold-scan.json"
    gold.write_text(json.dumps(gold_payload("scan-1", [0, 1])), encoding="utf-8")

    result = runner.study(
        families=["ocr"],
        engines=["native", "docling"],
        doc_ids=["digital-1", "scan-1"],
        repeats=2,
        gold_paths={"scan-1": str(gold)},
    )

    summary = result["summary"]
    assert summary["documents"] == 2
    family = summary["families"]["ocr"]
    assert family["documents"] == 2
    assert set(family["participants"]) == {"native", "docling"}
    assert family["participants"]["native"]["documents"] == 2

    ((name, pair),) = family["pairs"].items()
    assert set(name.split("|")) == {"native", "docling"}
    assert pair["documents"] == 2
    assert pair["n_pairs"] == 4  # two documents, two repeats each
    assert pair["n_repeats"] == 2
    cer = pair["metrics"]["cer_micro"]
    assert cer["status"] == "measured"
    assert cer["documents"] == 2
    assert cer["unit"] == "ratio"
    assert cer["denominator_unit"] == "characters"
    assert cer["median"] is not None
    assert cer["variation"] is not None
    assert cer["significance"]["status"] == "not_measured"


def test_a_pair_summary_averages_repeats_inside_their_document(corpus: Corpus) -> None:
    """The document is the unit: two repeats are one observation, not two.

    A study must not buy its sample size by running the same document twice, so
    the repeats are averaged inside their document before the corpus figure is
    taken. The repeats are still two rows and two pairs -- the aggregation is
    what folds them, not the record.
    """
    runner = make_runner(
        corpus, {"native": FakeEngine("native"), "docling": FakeEngine("docling")}
    )

    result = runner.study(
        families=["ocr"], engines=["native", "docling"], doc_ids=["digital-1"], repeats=2
    )

    (pair,) = result["summary"]["families"]["ocr"]["pairs"].values()
    assert pair["documents"] == 1
    assert pair["n_pairs"] == 2
    assert pair["n_repeats"] == 2
    cer = pair["metrics"]["cer_micro"]
    # One document is one observation: the two repeats were averaged, so the
    # mean is the delta itself and there is no spread to report.
    assert cer["documents"] == 1
    assert cer["variation"] is None
    assert cer["variation_reason"] == "single_document"
    assert cer["significance"]["status"] == "not_measured"
    assert cer["significance"]["reason"] == "no_test_run"


def test_a_summary_reads_a_flipped_pair_backwards(corpus: Corpus) -> None:
    """A pair that arrives in the other order is not a zero difference.

    The study reverses its engine order on even repeats, so the same pair shows
    up as (a, b) once and (b, a) once. Averaging those raw would report no
    difference at all for a pair that differs, so the flipped one is read
    backwards -- value and both sides together. The direction the summary
    reports is the canonical one: the first participant in sorted order is the
    pair's ``a`` side.
    """
    runner = make_runner(corpus, {"native": FakeEngine("native")})
    forward = {
        "participants": ["alpha", "beta"],
        "repeat": 1,
        "compatible": True,
        "metrics": {
            "cer_micro": {
                "status": "measured",
                "value": 0.20,
                "unit": "ratio",
                "a": 0.10,
                "b": 0.30,
            }
        },
    }
    backward = {
        "participants": ["beta", "alpha"],
        "repeat": 2,
        "compatible": True,
        "metrics": {
            "cer_micro": {
                "status": "measured",
                "value": -0.20,
                "unit": "ratio",
                "a": 0.30,
                "b": 0.10,
            }
        },
    }

    summary = runner._pair_summary([("digital-1", forward), ("digital-1", backward)])

    assert summary["documents"] == 1
    assert summary["n_pairs"] == 2
    assert summary["n_repeats"] == 2
    # Both repeats say the same thing, so the document's mean is that thing.
    assert summary["metrics"]["cer_micro"]["value"] == pytest.approx(0.20)


# --------------------------------------------------------------------------- #
# what a study spent, counted the one way its reports cannot
# --------------------------------------------------------------------------- #


def metered_native(corpus: Corpus) -> StudyRunner:
    """A runner whose text-layer double reports tokens for every page it reads."""
    return make_runner(corpus, {"native": FakeEngine("native", usage=True, model="fake-vision-1")})


def test_a_study_counts_each_run_once_however_many_families_name_it(corpus: Corpus) -> None:
    """One extraction, named by two families, is one run and one set of counters.

    The extraction is made before the study, so the study resumes it instead of
    reading the pages again. It appears in each family's report, and summing
    those reports counts it once per family; the study's own usage block counts
    it once, and the run counts beside it say how many runs there were.
    """
    runner = metered_native(corpus)
    runner.extract("digital-1", "native")

    result = runner.study(families=["ocr", "pipeline"], engines=["native"], doc_ids=["digital-1"])

    usage = result["usage"]
    assert usage["runs"] == {
        "extractions": 1,
        "trees": 1,
        "unique": 2,
        "replayed": 1,
        "new": 1,
    }
    # Three pages at a hundred prompt and twenty completion tokens each, counted
    # once for the study. The tree built from those pages reads them again and
    # buys nothing.
    assert usage["unique_work"]["status"] == "measured"
    assert usage["unique_work"]["prompt_tokens"] == 300
    assert usage["unique_work"]["completion_tokens"] == 60
    assert usage["unique_work"]["documents"] == 1
    # The tree was built in this invocation and made no call of its own, so it
    # contributes no document: this invocation read nothing, and says so rather
    # than reporting a zero somebody measured.
    assert usage["new_work"]["status"] == "not_measured"
    assert usage["new_work"]["reason"] == "no_documents"
    assert usage["new_work"]["prompt_tokens"] is None
    # Each report counts the extraction too, which is why the study's own block
    # is the one a total is read from.
    assert len(result["comparisons"]) == 2
    assert all(
        comparison_report(result, index)["counts"]["prompt_tokens"] == 300 for index in range(2)
    )


def test_a_study_counts_a_tree_for_its_merge_and_not_for_the_pages_again(
    corpus: Corpus, monkeypatch: pytest.MonkeyPatch
) -> None:
    """The pages are the extraction's tokens; a tree's are the merge it ran on them.

    The merged tree reads the same pages the grouped tree does, so counting the
    pages again under it would count one reading twice -- and a study total that
    did would grow every time a family was added. The two sets of counters are
    the extraction and the merge; the grouped tree buys nothing of its own.
    """
    runner = make_runner(corpus, {"vision": FakeVisionEngine()}, settings=StubSettings())
    client = MeteredMergeClient(
        [{"prompt_tokens": 1200, "completion_tokens": 800}], models=[SERVED_MERGE_MODEL]
    )
    monkeypatch.setattr(StudyRunner, "_gateway_client", lambda _self: client)

    result = runner.study(
        families=["chunk"],
        engines=["vision"],
        doc_ids=["mixed-1"],
        profiles=["vision_direct", "vision_atomic_merge"],
        allow_upload=True,
        allow_merge_upload=True,
        llm_mode="live",
    )

    usage = result["usage"]
    assert usage["runs"]["unique"] == 3  # one extraction and its two trees
    assert usage["runs"]["new"] == 3
    # Two sets of counters for three runs: the grouped tree merged nothing and
    # reads the pages the extraction already counted; the merged tree adds the
    # merge, and the pages are not counted a second time.
    assert usage["unique_work"]["documents"] == 2
    # The extraction's four pages at a hundred prompt and twenty completion
    # tokens each, then the merge: the pages are not bought a second time.
    assert usage["unique_work"]["prompt_tokens"] == 400 + 1200 * client.calls
    assert usage["unique_work"]["completion_tokens"] == 80 + 800 * client.calls
    assert usage["unique_work"]["status"] == "measured"


def test_a_replayed_study_reports_no_new_work(corpus: Corpus) -> None:
    """A resume hands back runs already read; the work is counted, the reading is not.

    The invocation made no call at all, so its new work has no document to pool
    -- a typed miss naming that fact, never a zero token count standing in for
    pages nobody read.
    """
    runner = metered_native(corpus)

    first = runner.study(families=["ocr"], engines=["native"], doc_ids=["digital-1"])
    second = runner.study(families=["ocr"], engines=["native"], doc_ids=["digital-1"])

    usage = second["usage"]
    assert usage["runs"] == {
        "extractions": 1,
        "trees": 0,
        "unique": 1,
        "replayed": 1,
        "new": 0,
    }
    assert usage["new_work"]["status"] == "not_measured"
    assert usage["new_work"]["reason"] == "no_documents"
    assert usage["new_work"]["documents"] == 0
    assert usage["new_work"]["prompt_tokens"] is None
    # The study's work is still the whole study's work.
    assert usage["unique_work"]["prompt_tokens"] == first["usage"]["unique_work"]["prompt_tokens"]
    assert usage["unique_work"]["prompt_tokens"] == 300


def test_two_fresh_repeats_of_one_request_are_two_runs_and_two_readings(
    corpus: Corpus,
) -> None:
    """Two runs may share an artifact digest, and sharing it does not merge them.

    A deterministic engine writes one artifact for both repeats, so the digests
    are identical and the counters are not: each repeat made its own calls. The
    later study replays both runs, and both keep their own tokens.
    """
    runner = metered_native(corpus)
    first = runner.study(
        families=["ocr"],
        engines=["native"],
        doc_ids=["digital-1"],
        repeats=2,
        resume=False,
    )

    digests = [entry["artifact_digest"] for entry in first["extractions"]]
    assert len(digests) == 2 and len(set(digests)) == 1
    assert first["usage"]["runs"] == {
        "extractions": 2,
        "trees": 0,
        "unique": 2,
        "replayed": 0,
        "new": 2,
    }
    assert first["usage"]["new_work"]["documents"] == 2  # two readings, two sets of counters
    assert first["usage"]["new_work"]["prompt_tokens"] == 600

    second = runner.study(families=["ocr"], engines=["native"], doc_ids=["digital-1"], repeats=2)

    usage = second["usage"]
    assert usage["runs"]["replayed"] == 2
    assert usage["runs"]["new"] == 0
    # One digest, still two runs: the repeats were read apart and are counted
    # apart, not folded into one document because the bytes are the same.
    assert usage["unique_work"]["documents"] == 2
    assert usage["unique_work"]["prompt_tokens"] == 600


def test_a_study_keeps_a_failed_run_in_its_usage_denominator(corpus: Corpus) -> None:
    """A run that was requested is a run, however it ended.

    Dropping failures from the accounting would let a study look cheaper by
    failing. The failure here named no counters, so it counts as a run and as a
    document whose tokens are unknown rather than as a zero.
    """
    engine = FakeEngine("native", usage=True, model="fake-vision-1")
    engine.error = RuntimeError("the engine gave up")
    runner = make_runner(corpus, {"native": engine})

    result = runner.study(families=["ocr"], engines=["native"], doc_ids=["digital-1"])

    assert [entry["status"] for entry in result["extractions"]] == [RUN_FAILED]
    assert result["comparisons"] == []
    usage = result["usage"]
    assert usage["runs"]["extractions"] == 1
    assert usage["runs"]["unique"] == 1  # no tree was built from a failed extraction
    assert usage["new_work"]["documents"] == 1
    # Counters the failure could not name are a miss, never a zero.
    assert usage["new_work"]["status"] == "not_measured"
    assert usage["new_work"]["reason"] == "usage_missing"
    assert usage["new_work"]["prompt_tokens"] is None


def test_a_summary_orients_a_flipped_pair_for_every_reading(corpus: Corpus) -> None:
    """A flipped pair is read backwards for every delta, resources included.

    The direction a summary reports is the canonical one, so a pair that arrives
    as (b, a) contributes the same figure as (a, b) rather than cancelling it out
    -- and a resource reading is sampled, so its delta says so.
    """
    runner = make_runner(corpus, {"native": FakeEngine("native")})
    forward = {
        "participants": ["alpha", "beta"],
        "repeat": 1,
        "compatible": True,
        "metrics": {
            "seconds": {"status": "measured", "value": 0.5, "unit": "seconds", "a": 2.0, "b": 2.5},
            "cpu_seconds": {
                "status": "measured",
                "value": 0.25,
                "unit": "seconds",
                "a": 1.25,
                "b": 1.50,
                "estimate": True,
            },
            "rss_peak_bytes": {
                "status": "measured",
                "value": 4_000_000,
                "unit": "bytes",
                "a": 100_000_000,
                "b": 104_000_000,
                "estimate": True,
            },
        },
    }
    backward = {
        "participants": ["beta", "alpha"],
        "repeat": 2,
        "compatible": True,
        "metrics": {
            "seconds": {
                "status": "measured",
                "value": -0.5,
                "unit": "seconds",
                "a": 2.5,
                "b": 2.0,
            },
            "cpu_seconds": {
                "status": "measured",
                "value": -0.25,
                "unit": "seconds",
                "a": 1.50,
                "b": 1.25,
                "estimate": True,
            },
            "rss_peak_bytes": {
                "status": "measured",
                "value": -4_000_000,
                "unit": "bytes",
                "a": 104_000_000,
                "b": 100_000_000,
                "estimate": True,
            },
        },
    }

    metrics = runner._pair_summary([("digital-1", forward), ("digital-1", backward)])["metrics"]

    # One document, so the mean of the two repeats is the delta itself, and a
    # paired sum would have cancelled it to zero. A resource delta is a
    # difference of sampled readings, so the corpus figure says so too.
    assert metrics["cpu_seconds"]["value"] == pytest.approx(0.25)
    assert metrics["cpu_seconds"]["estimate"] is True
    assert metrics["rss_peak_bytes"]["value"] == pytest.approx(4_000_000)
    assert metrics["rss_peak_bytes"]["estimate"] is True
    # Wall time is a clock reading, not a sample, so it grows no such flag.
    assert metrics["seconds"]["value"] == pytest.approx(0.5)
    assert "estimate" not in metrics["seconds"]


def test_a_hosted_run_keeps_its_unknown_resources_unknown(corpus: Corpus) -> None:
    """A hosted engine's memory is unobservable, and that is not a zero.

    The same pair measures wall time on both sides and cannot measure CPU or
    memory on the hosted one: the delta it cannot take stays the miss it is, and
    names the side that could not answer, rather than reading as no difference.
    """
    runner = make_runner(
        corpus,
        {"native": FakeEngine("native"), "vision": FakeVisionEngine()},
        settings=StubSettings(),
    )

    result = runner.study(
        families=["ocr"], engines=["native", "vision"], doc_ids=["digital-1"], allow_upload=True
    )
    assert result["executed"] is True

    metrics = comparison_report(result)["paired_usage"]["pairs"][0]["metrics"]
    assert metrics["rss_peak_bytes"]["status"] == "not_measured"
    assert metrics["rss_peak_bytes"]["value"] is None
    assert metrics["rss_peak_bytes"]["b_reason"] == "hosted_backend"
    assert metrics["cpu_seconds"]["b_reason"] == "hosted_backend"
    # Wall time is measured wherever the stage ran, hosted or not.
    assert metrics["seconds"]["status"] == "measured"

    # The reading never reaches the corpus as a number: no figure is published
    # for the resource the backend could not observe.
    summary = result["summary"]["families"]["ocr"]
    pair = next(iter(summary["pairs"].values()))
    assert pair["metrics"]["rss_peak_bytes"]["status"] == "not_measured"
    assert pair["metrics"]["seconds"]["status"] == "measured"


def test_a_tree_arm_carries_the_resources_of_the_extraction_it_replays(
    corpus: Corpus, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A tree replay ran no engine: its arm's readings are the run it names.

    The pipeline family compares two pipelines, and the engine work of each is in
    the extraction its tree replays. The join is that run's id -- never a digest,
    which two runs of one engine share -- so an arm can be attributed the CPU and
    memory of the engine it used, and its row names the run they came from. A
    hosted engine's backend stays unobservable: that arm joins a miss, not a zero.
    """
    vision = FakeVisionEngine()
    marker = FakeEngine("marker")
    runner = make_runner(corpus, {"vision": vision, "marker": marker}, settings=StubSettings())
    monkeypatch.setattr(StudyRunner, "_gateway_client", lambda _self: FakeMergeClient())

    result = runner.study(
        families=["pipeline"],
        engines=["vision", "marker"],
        doc_ids=["mixed-1"],
        repeats=2,
        allow_upload=True,
        allow_merge_upload=True,
        llm_mode="live",
    )
    assert result["executed"] is True

    marker_runs = [
        runner.read_run("mixed-1", entry["run_id"])
        for entry in result["extractions"]
        if ".marker." in entry["run_id"]
    ]
    # Two runs of one deterministic engine: one digest, two ids. A join that read
    # the digest could not tell them apart, which is why the id is the linkage.
    assert len(marker_runs) == 2
    assert len({run["artifact_digest"] for run in marker_runs}) == 1

    report = comparison_report(result)
    rows = {row["run_id"]: row for row in report["metadata"]["run_rows"]}
    second = next(run for run in marker_runs if run["repeat"] == 2)
    marker_row = rows["mixed-1.marker_llm.r2.live"]
    assert marker_row["resource_run_id"] == second["run_id"]
    assert marker_row["resources"]["rss_peak"]["status"] == "measured"
    assert marker_row["resources"]["rss_peak"]["estimate"] is True

    vision_row = rows["mixed-1.vision_direct.r2.live"]
    assert vision_row["resource_run_id"] == "mixed-1.vision.r2"
    assert vision_row["resources"]["rss_peak"]["status"] == "not_available"
    assert vision_row["resources"]["rss_peak"]["reason"] == "hosted_backend"

    # The joined stages and resources keep their own run's key, so a replay's own
    # wall time is never read as the engine's, and a reader can see the scope of
    # every reading in the report.
    assert report["stages"][f"{second['run_id']}.extract"]["status"] == "measured"
    assert report["resources"][second["run_id"]]["rss_peak"]["status"] == "measured"

    # The pair reads the joined readings: the memory of the engine behind each
    # arm, and the miss of the engine that ran where nothing can be observed.
    pair = next(
        pair for pair in report["paired_usage"]["pairs"] if pair["run_b"].endswith(".r2.live")
    )
    metrics = pair["metrics"]
    assert metrics["rss_peak_bytes"]["status"] == "not_measured"
    assert metrics["rss_peak_bytes"]["a_reason"] == "hosted_backend"


def test_the_pipeline_family_pairs_the_vision_tree_with_the_marker_tree(
    corpus: Corpus, monkeypatch: pytest.MonkeyPatch
) -> None:
    """The pipeline family is one fixed pair, and the other trees are diagnostics.

    A study builds more trees than the family compares: the atomic arms are the
    merge family's control, and pairing two trees of one engine would measure an
    engine against itself. Each keeps a row of its own beside the pair it is not
    part of, so a reader sees the tree that ran without reading it as an arm.
    """
    runner = make_runner(
        corpus,
        {"vision": FakeVisionEngine(), "marker": FakeEngine("marker")},
        settings=StubSettings(),
    )
    monkeypatch.setattr(StudyRunner, "_gateway_client", lambda _self: FakeMergeClient())

    result = runner.study(
        families=["pipeline"],
        engines=["vision", "marker"],
        doc_ids=["mixed-1"],
        allow_upload=True,
        allow_merge_upload=True,
        llm_mode="live",
    )
    assert result["executed"] is True

    report = comparison_report(result)
    assert [pair["participants"] for pair in report["paired_usage"]["pairs"]] == [
        ["vision_direct", "marker_llm"]
    ]
    assert {entry["profile"] for entry in result["trees"]} == {
        "vision_direct",
        "vision_atomic_merge",
        "vision_atomic",
        "marker_llm",
        "marker_atomic",
    }
    # The family's rows are its two arms: the trees it does not compare are not
    # read as participants, and a reader sees them only as runs of the study.
    assert {row["run_id"] for row in report["metadata"]["run_rows"]} == {
        "mixed-1.vision_direct.r1.live",
        "mixed-1.marker_llm.r1.live",
    }


def test_a_pipeline_document_with_one_arm_names_the_arm_it_is_missing(corpus: Corpus) -> None:
    """One arm is not a comparison, and the report says which one is absent.

    A reader of an empty pair list cannot tell a family that was never asked for
    from one whose other arm did not run, so the missing arm is named.
    """
    runner = make_runner(corpus, {"vision": FakeVisionEngine()}, settings=StubSettings())
    extraction = runner.extract(
        "mixed-1", "vision", allow_upload=True, vision_config=StubVisionConfig()
    )
    direct = runner.run_tree("mixed-1", "vision_direct", extract_run_id=extraction["run_id"])

    report = runner.compare("pipeline", [RunRef("mixed-1", direct["run_id"])], write=False)[
        "report"
    ]

    assert report["paired_usage"]["pairs"] == []
    assert any("marker_llm has no run here" in limitation for limitation in report["limitations"])
