"""The drop-in gate: a whole OCR study driven through the production CLI, offline.

This is R7 of the research repair plan. It exists to be run *before* anyone
spends tokens on a live study, and it is deliberately not satisfied by "the
modules and their unit tests exist": it builds a small but complete corpus and
then drives ``study`` and ``compare`` exactly as a shell would, so family
selection, pairing, accounting, resume and every written file are production
code paths, not test scaffolding.

What the gate puts on disk
--------------------------
Three synthetic PDFs of the three modalities the corpus schema knows
(``digital-1`` readable, ``scan-1`` a scan, ``mixed-1`` half and half) and a
hand-written gold for two of them, the third having none. That is the whole
authoring side: a gold file is the only input a scored study needs beside the
PDFs, and the tokens a run spent are compared as tokens. N=2 repeats. Engines
are injected at the one seam the production runner already offers
(``StudyRunner(..., engines=...)``) and ``cli.make_runner`` is patched to hand
back a *real* runner with them, so nothing below the adapter is bypassed.

The declared policy of the doubles (``POLICY``) makes every cell of R7
reachable: a page that reports no usage (the paid engine on ``mixed-1`` page
0), an engine that fails outright (``docling`` on ``mixed-1``), an engine that
cannot read a whole document (``native`` on ``scan-1``: every page
``unsupported_modality``) and an engine with one unreadable page (``marker``
on ``scan-1`` page 1).

What it asserts
---------------
The expected grid, cell by cell and counted by denominator -- not the runs that
happened to work; the null semantics (a failed, unsupported or unreported cell
keeps its own status, and a missing denominator is ``None``, never a fabricated
zero); a page scored against a hand-written gold, a page whose gold is a copy
of the candidate abstaining as ``self_scoring``, and a document with no gold
at all measuring nothing rather than zero; resume replaying from disk without
reading a page again -- the run whose engine failed is retried, and only it --
with zero merge calls and zero gateway clients built; a gold change re-scoring
from disk without re-reading anything; a document added by editing the manifest
JSON alone; a ``--fresh`` repeat landing in new run ids; ``--dry-run``,
``compare`` and both gold discovery paths (declared in the manifest, and named
on the command line); and a scan of every written file for the API key and for
the merge prompt.

What it does not claim
----------------------
The merge in study A is the tests' own ``FakeMergeClient``, reached by patching
``StudyRunner._gateway_client`` while the CLI is told ``--llm-mode live``.
Study A proves the wiring, the accounting and the files, and says nothing about
quality: a merge answered by a wiring double is not a measurement, whoever
counted it.

Nothing here reads a PDF page, imports an engine package or opens a socket.
"""

from __future__ import annotations

import contextlib
import io
import json
import re
from collections import Counter
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import pytest
from merge_double import FakeMergeClient

import scripts.run_ocr_study as cli
from research.ocr_pipeline.corpus import load_manifest, sha256_file
from research.ocr_pipeline.engines import (
    STATUS_UNSUPPORTED_MODALITY,
    EngineArtifact,
    EngineBlock,
    EnginePage,
    EngineProbe,
)
from research.ocr_pipeline.report import find_raw_text_keys
from research.ocr_pipeline.runner import (
    CHUNK_ARMS,
    KIND_TREE,
    StudyRunner,
    primary_profile,
)

PROVENANCE: dict[str, Any] = {
    "git_commit": "0" * 40,
    "git_dirty": False,
    "python": "3.11.0",
    "platform": "test",
    "label": "drop-in gate",
}
"""A provenance stand-in: the git probe must never run inside a test."""

ENGINES = ("vision", "marker", "docling", "native")
DOCS = ("digital-1", "scan-1", "mixed-1")
FAMILIES = ("pipeline", "ocr", "chunk")
REPEATS = 2
PAGES = (0, 1)

LIVE_PROFILES = (
    "vision_direct",
    "vision_atomic_merge",
    "vision_atomic",
    "marker_llm",
    "marker_atomic",
    "docling_atomic",
    "native_atomic",
)
"""Every profile a live study of ``ENGINES`` builds a tree with.

Written out rather than derived: if the registry grows, gains or loses a profile
for one of these engines, the plan's profile list must change too, and the gate
is the place that says so out loud.
"""

FAKE_API_KEY = "sk-drop-in-gate-not-a-real-key"
"""The key the paid engine is configured with. Scanned for in every output file."""

FAKE_HEADER = "drop-in-gate"
"""A trace header value; scanned for beside the key."""

MISSING_USAGE_DOC = "mixed-1"
FAILING_DOC = "mixed-1"
UNSUPPORTED_DOC = "scan-1"


@dataclass(frozen=True)
class Policy:
    """What one engine double is told to do, document by document."""

    fails: frozenset[str] = frozenset()
    """Documents whose extraction raises: the run is a failed run, not an absent one."""

    unsupported: Mapping[str, frozenset[int]] = field(default_factory=dict)
    """Document -> pages the engine answers with ``unsupported_modality`` and no blocks."""

    quiet: Mapping[str, frozenset[int]] = field(default_factory=dict)
    """Document -> pages that report no usage: a paid answer nobody can bill."""


POLICY: dict[str, Policy] = {
    "vision": Policy(quiet={MISSING_USAGE_DOC: frozenset({0})}),
    "marker": Policy(unsupported={UNSUPPORTED_DOC: frozenset({1})}),
    "docling": Policy(fails=frozenset({FAILING_DOC})),
    "native": Policy(unsupported={UNSUPPORTED_DOC: frozenset(PAGES)}),
}
"""Every R7 case, declared once: a failure, a page failure, a whole-doc miss, a quiet page."""


# --------------------------------------------------------------------------- #
# synthetic corpus
# --------------------------------------------------------------------------- #


def make_pdf(page_count: int, *, text_pages: tuple[int, ...] = ()) -> bytes:
    """A valid PDF whose ``text_pages`` carry a text object; see the corpus tests."""
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


def write_pdf(directory: Path, name: str, page_count: int, *, text_pages: tuple[int, ...]) -> Path:
    path = directory / name
    path.write_bytes(make_pdf(page_count, text_pages=text_pages))
    return path


def heading_text(index: int) -> str:
    return f"Chuong {index + 1}"


def body_text(doc_id: str, index: int) -> str:
    return f"Than bai cua {doc_id} trang {index} noi ve mot chu de dai va ro rang"


def extra_text(doc_id: str, index: int) -> str:
    return f"Doan phu cua {doc_id} trang {index} bo sung them mot vai chi tiet"


def page_blocks_text(doc_id: str, index: int) -> str:
    """The content text the doubles report for one page: three blocks, joined."""
    return "\n".join((heading_text(index), body_text(doc_id, index), extra_text(doc_id, index)))


def checked_text(doc_id: str, index: int) -> str:
    """A human's transcript of one page: the doubles' reading, one word apart.

    A word the candidate does not have is a real, small, measurable difference
    with the heading still at its own offset. A transcript identical to the
    candidate would abstain as ``self_scoring`` instead of measuring a page
    that agrees with itself -- which is the other half of this gate, below.
    """
    return f"{page_blocks_text(doc_id, index)} nua"


def copied_text(doc_id: str, index: int) -> str:
    """A gold that is a copy of what the run reports: a page nobody checked.

    Authored here, as a literal, before any run -- it is the *situation* the
    ``self_scoring`` abstention is for, not a run's output pasted in.
    """
    return page_blocks_text(doc_id, index)


def gold_page(index: int, text: str, headings: Sequence[str] = ()) -> dict[str, Any]:
    """One gold page as a human writes it: the transcript, and its headings."""
    return {"page_index": index, "text": text, "headings": list(headings)}


def checked_page(doc_id: str, index: int) -> dict[str, Any]:
    """One page of the gate's hand-written gold, heading included."""
    return gold_page(index, checked_text(doc_id, index), (heading_text(index),))


def changed_page(index: int) -> dict[str, Any]:
    """A page the human rewrote: the heading stays, the body is another sentence.

    The heading is still where it was, so the page is still considered; the body
    is far enough from the candidate that the alignment refuses to score it.
    """
    text = "\n".join(
        (heading_text(index), "Phan ket luan da duoc nguoi doc sua lai thanh mot cau khac")
    )
    return gold_page(index, text, (heading_text(index),))


def write_gold(path: Path, doc_id: str, pages: Sequence[Mapping[str, Any]]) -> Path:
    payload = {"doc_id": doc_id, "pages": list(pages)}
    path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    return path


def document_entry(
    doc_id: str,
    pdf_path: str,
    pdf_sha256: str,
    *,
    modality: str,
    gold_path: str | None = None,
) -> dict[str, Any]:
    return {
        "doc_id": doc_id,
        "pdf_path": pdf_path,
        "pdf_sha256": pdf_sha256,
        "page_indices": list(PAGES),
        "modality": modality,
        "permission_note": "synthetic pages written by the gate; nothing to license",
        "gold_path": gold_path,
    }


def write_manifest(path: Path, documents: Sequence[Mapping[str, Any]]) -> Path:
    path.write_text(
        json.dumps({"corpus_version": 1, "documents": list(documents)}, indent=2),
        encoding="utf-8",
    )
    return path


@dataclass(frozen=True)
class Corpus:
    """The corpus on disk, plus the digests its manifest declares."""

    base_dir: Path
    manifest_path: Path
    documents: list[dict[str, Any]]
    pdf_sha256: dict[str, str]


def build_corpus(base_dir: Path) -> Corpus:
    """Three modalities, a checked gold for two of them, no gold for the third."""
    digital = write_pdf(base_dir, "digital.pdf", len(PAGES), text_pages=PAGES)
    scan = write_pdf(base_dir, "scan.pdf", len(PAGES), text_pages=())
    mixed = write_pdf(base_dir, "mixed.pdf", len(PAGES), text_pages=(0,))

    gold_digital = write_gold(
        base_dir / "gold-digital.json",
        "digital-1",
        [
            # Page 0 is a copy of the candidate: the study must refuse to score
            # it rather than report a perfect page.
            gold_page(0, copied_text("digital-1", 0), (heading_text(0),)),
            checked_page("digital-1", 1),
        ],
    )
    gold_scan = write_gold(
        base_dir / "gold-scan.json",
        "scan-1",
        [checked_page("scan-1", index) for index in PAGES],
    )

    documents = [
        document_entry(
            "digital-1",
            "digital.pdf",
            sha256_file(digital),
            modality="digital",
            gold_path=gold_digital.name,
        ),
        document_entry(
            "scan-1",
            "scan.pdf",
            sha256_file(scan),
            modality="scan",
            gold_path=gold_scan.name,
        ),
        document_entry("mixed-1", "mixed.pdf", sha256_file(mixed), modality="mixed"),
    ]
    manifest = write_manifest(base_dir / "manifest.json", documents)
    return Corpus(
        base_dir=base_dir,
        manifest_path=manifest,
        documents=documents,
        pdf_sha256={doc["doc_id"]: doc["pdf_sha256"] for doc in documents},
    )


# --------------------------------------------------------------------------- #
# engine doubles
# --------------------------------------------------------------------------- #


class Engine:
    """A local adapter double, in the call shape ``_call_engine`` uses.

    It never opens the PDF: the subject is what the study did with an engine's
    answer, so it records every call and hands back one page per requested index
    -- heading, two paragraphs, folio -- with a page's fault applied when
    ``POLICY`` declares one.
    """

    paid = False

    def __init__(self, engine_id: str, policy: Policy) -> None:
        self.engine_id = engine_id
        self.policy = policy
        self.calls = 0
        self.sources: list[str] = []
        self.documents: list[str] = []

    def probe(self) -> EngineProbe:
        return EngineProbe(self.engine_id, True, "0.1-fake", "an offline double")

    def extract(
        self,
        source: Any,
        *,
        doc_id: str,
        page_offset: int = 0,
        page_indices: Sequence[int] = (),
        converter: Any | None = None,
    ) -> EngineArtifact:
        self._record(source, doc_id=doc_id)
        return self._artifact(doc_id, list(page_indices), page_offset=page_offset)

    def _record(self, source: Any, *, doc_id: str) -> None:
        self.calls += 1
        self.sources.append(str(source))
        self.documents.append(doc_id)
        if doc_id in self.policy.fails:
            raise RuntimeError(f"the {self.engine_id} double was told to fail on {doc_id}")

    def _artifact(
        self,
        doc_id: str,
        page_indices: Sequence[int],
        *,
        page_offset: int,
        public_config: Mapping[str, Any] | None = None,
        engine_version: str | None = None,
    ) -> EngineArtifact:
        pages = tuple(self._page(doc_id, index, page_offset=page_offset) for index in page_indices)
        return EngineArtifact(
            engine_id=self.engine_id,
            engine_version=engine_version or "0.1-fake",
            doc_id=doc_id,
            pages=pages,
            render_page_offset=page_offset,
            public_config=dict(public_config or {}),
            normalization={"fake_pages": len(pages)},
        )

    def _page(self, doc_id: str, index: int, *, page_offset: int) -> EnginePage:
        request_page = index + 1 + page_offset
        if index in self.policy.unsupported.get(doc_id, frozenset()):
            # What a scanned page looks like to an engine that cannot read it: no
            # blocks at all, and an explicit status instead of an empty page.
            return EnginePage(
                page_index=index,
                request_page=request_page,
                status=STATUS_UNSUPPORTED_MODALITY,
            )
        return EnginePage(
            page_index=index,
            request_page=request_page,
            blocks=(
                EngineBlock(
                    kind="SectionHeader",
                    normalized_kind="heading",
                    text=heading_text(index),
                    depth=1,
                    group=index + 1,
                    source_id=f"{index}:heading",
                ),
                EngineBlock(
                    kind="Text",
                    normalized_kind="p",
                    text=body_text(doc_id, index),
                    group=index + 1,
                    source_id=f"{index}:body",
                ),
                EngineBlock(
                    kind="Text",
                    normalized_kind="p",
                    text=extra_text(doc_id, index),
                    group=index + 1,
                    source_id=f"{index}:extra",
                ),
                EngineBlock(
                    kind="PageFooter",
                    normalized_kind="page_footer",
                    text=str(index + 1),
                    group=index + 1,
                    source_id=f"{index}:footer",
                ),
            ),
            running=f"{doc_id} running head",
            printed_folio=str(index + 1),
            usage=self._usage(doc_id, index),
        )

    def _usage(self, doc_id: str, index: int) -> dict[str, Any] | None:
        """Usage per page: local engines have no provider to report one."""
        if not self.paid:
            return None
        if index in self.policy.quiet.get(doc_id, frozenset()):
            return None
        return {"attempts": 1, "prompt_tokens": 100, "completion_tokens": 20}


class PaidEngine(Engine):
    """The paid adapter shape: async, with a config, a client and a DPI.

    It reads the key off the configuration it is handed, so the gate can prove
    the secret really was in play before scanning the files for it.
    """

    paid = True

    def __init__(self, engine_id: str, policy: Policy) -> None:
        super().__init__(engine_id, policy)
        self.keys: list[str] = []

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
        self._record(pdf_path, doc_id=doc_id)
        self.keys.append(config.api_key.get_secret_value())
        return self._artifact(
            doc_id,
            list(page_indices),
            page_offset=page_offset,
            public_config={"model": config.model},
            engine_version=config.model,
        )


class StubSettings:
    """The project settings a vision configuration is derived from.

    Only the field names ``VisionOCRConfig.from_settings`` reads are here, the
    host is a reserved TLD, and the key is the gate's own placeholder: deriving a
    configuration needs no network and reaches nothing.
    """

    VISION_API_BASE = "https://vision.invalid/v1"
    VISION_MODEL = "fake-vision-1"
    VISION_API_KEY = FAKE_API_KEY
    VISION_ALLOWED_HOSTS = "vision.invalid"
    VISION_HEADERS = {"X-Study-Probe": FAKE_HEADER}
    VISION_IMAGE_DETAIL = "high"
    VISION_TEMPERATURE = 0.0
    VISION_MAX_TOKENS = 8000
    VISION_REASONING_EFFORT = None
    VISION_MAX_BLOCKS_PER_PAGE = 400
    BENCHMARK_LLM_MODEL = "a-double-not-a-model"


class Gateway:
    """One shared fake merge client, plus a count of how often one was built.

    ``build`` is what the production runner calls when a merge is about to run,
    so ``builds`` is the number of live merges that were reached at all: a
    resume replays from disk and must build none.
    """

    def __init__(self) -> None:
        self.client = FakeMergeClient()
        self.builds = 0

    def build(self) -> FakeMergeClient:
        self.builds += 1
        return self.client


# --------------------------------------------------------------------------- #
# the session: a corpus, its doubles, and the studies run against them
# --------------------------------------------------------------------------- #


@dataclass
class Result:
    """What a shell would see: the exit code, stdout and stderr."""

    code: int
    out: str
    err: str

    @property
    def payload(self) -> dict[str, Any]:
        if not self.out.strip():
            raise AssertionError(
                f"the CLI reported no study at all (exit {self.code}): {self.err.strip()}"
            )
        return json.loads(self.out)


@dataclass
class Session:
    """The gate's whole world: one corpus, one study root, one set of doubles."""

    corpus: Corpus
    root: Path
    engines: dict[str, Engine]
    vision: PaidEngine
    gateway: Gateway
    settings: Any = StubSettings
    _steps: dict[str, Any] = field(default_factory=dict)

    # -- driving the CLI --------------------------------------------------- #

    def run_cli(self, argv: Sequence[str], *, root: Path | None = None) -> Result:
        """Drive ``cli.main`` with the engines injected at its one seam.

        Only ``make_runner`` and the merge client are replaced. Family
        selection, pairing, accounting, resume and every written file are the
        production code paths.
        """
        target = root if root is not None else self.root
        stream, errors = io.StringIO(), io.StringIO()
        with pytest.MonkeyPatch.context() as patch:
            patch.setattr(cli, "make_runner", lambda args: self.make_runner(args))
            patch.setattr(StudyRunner, "_gateway_client", staticmethod(self.gateway.build))
            with contextlib.redirect_stdout(stream), contextlib.redirect_stderr(errors):
                code = cli.main(
                    [
                        *argv,
                        "--corpus",
                        str(self.corpus.manifest_path),
                        "--root",
                        str(target),
                    ]
                )
        return Result(code=code, out=stream.getvalue(), err=errors.getvalue())

    def make_runner(self, args: Any) -> StudyRunner:
        return StudyRunner(
            load_manifest(args.corpus),
            base_dir=self.corpus.base_dir,
            root=Path(args.root),
            engines=self.engines,
            provenance=PROVENANCE,
            settings=self.settings,
        )

    def calls(self) -> dict[str, int]:
        return {engine_id: engine.calls for engine_id, engine in self.engines.items()}

    def once(self, name: str, run: Callable[[], Any]) -> Any:
        """Memoise a step, so tests that need it share one run of the CLI."""
        if name not in self._steps:
            self._steps[name] = run()
        return self._steps[name]

    # -- the studies ------------------------------------------------------- #

    def study_args(self, *extra: str) -> list[str]:
        return [
            "study",
            "--families",
            ",".join(FAMILIES),
            "--engines",
            ",".join(ENGINES),
            "--repeats",
            str(REPEATS),
            "--llm-mode",
            "live",
            "--allow-upload",
            "--allow-merge-upload",
            *extra,
        ]

    def study(self) -> Result:
        return self.once("study", lambda: self.run_cli(self.study_args()))

    def replay(self) -> Result:
        self.study()
        return self.once("replay", lambda: self.run_cli(self.study_args()))

    def regolded(self) -> Result:
        """The same study after the reference has been rewritten."""
        self.replay()
        return self.once("regolded", lambda: self.run_cli(self.study_args()))

    def extended(self) -> Result:
        """The same study after a document was added by editing the manifest JSON.

        The edit itself is memoised with the study: adding the document twice
        would append a second entry for it, and a manifest that lists the same
        document twice is not the manifest any of these runs were made against.
        """
        self.regolded()
        self.once("extra-document", lambda: add_document(self))
        return self.once("extended", lambda: self.run_cli(self.study_args()))

    def fresh(self) -> Result:
        """The same study with ``--fresh``: every repeat again, in new run ids."""
        self.extended()
        return self.once("fresh", lambda: self.run_cli(self.study_args("--fresh")))

    def dry_run(self, *extra: str) -> Result:
        return self.once(
            f"dry-run{extra}", lambda: self.run_cli(self.study_args("--dry-run", *extra))
        )


def add_document(session: Session) -> Corpus:
    """Add ``extra-1`` by editing the manifest JSON and writing its PDF.

    No production code changes: the corpus schema is the interface, so a
    document is data.
    """
    base_dir = session.corpus.base_dir
    pdf = write_pdf(base_dir, "extra.pdf", len(PAGES), text_pages=PAGES)
    documents = list(session.corpus.documents) + [
        document_entry("extra-1", "extra.pdf", sha256_file(pdf), modality="digital")
    ]
    write_manifest(session.corpus.manifest_path, documents)
    corpus = Corpus(
        base_dir=base_dir,
        manifest_path=session.corpus.manifest_path,
        documents=documents,
        pdf_sha256={**session.corpus.pdf_sha256, "extra-1": sha256_file(pdf)},
    )
    session.corpus = corpus
    return corpus


def manifest_docs(session: Session) -> list[str]:
    """The documents the manifest holds *now*: a later test may have added one."""
    payload = json.loads(session.corpus.manifest_path.read_text(encoding="utf-8"))
    return [str(document["doc_id"]) for document in payload["documents"]]


@pytest.fixture(scope="module")
def session(tmp_path_factory: pytest.TempPathFactory) -> Session:
    base_dir = tmp_path_factory.mktemp("drop-in")
    corpus = build_corpus(base_dir)
    engines: dict[str, Engine] = {
        engine_id: (
            PaidEngine(engine_id, POLICY[engine_id])
            if engine_id == "vision"
            else Engine(engine_id, POLICY[engine_id])
        )
        for engine_id in ENGINES
    }
    return Session(
        corpus=corpus,
        root=base_dir / "study",
        engines=engines,
        vision=engines["vision"],  # type: ignore[assignment]
        gateway=Gateway(),
    )


# --------------------------------------------------------------------------- #
# reading what the CLI wrote
# --------------------------------------------------------------------------- #


def comparison_reports(payload: Mapping[str, Any]) -> dict[tuple[str, str], dict[str, Any]]:
    """Every comparison the study made, keyed by ``(family, document)``.

    A report names the documents its rows read, so each per-document comparison
    is found from the file the study wrote, not from the order it happened to
    run in. The directory holds ``report.json`` (the report, all three formats
    rendered from it) and ``manifest.json`` (the comparison's own manifest, which
    carries the counts and no rows), so the report is read by name.
    """
    found: dict[tuple[str, str], dict[str, Any]] = {}
    for entry in payload["comparisons"]:
        directory = Path(str(entry["path"]))
        if not directory.is_dir():
            directory = directory.parent
        path = directory / "report.json"
        assert path.is_file(), f"{directory}: the study wrote no report.json"
        report = json.loads(path.read_text(encoding="utf-8"))
        documents = report["metadata"]["documents"]
        assert len(documents) == 1, f"{path}: expected one document, got {documents}"
        key = (str(report["metadata"]["family"]), str(documents[0]))
        assert key not in found, f"{path}: {key} reported twice"
        found[key] = report
    return found


def rows_by_run(report: Mapping[str, Any]) -> dict[str, dict[str, Any]]:
    return {str(row["run_id"]): row for row in report["metadata"]["run_rows"]}


def merge_counters(
    payload: Mapping[str, Any], reports: Mapping[tuple[str, str], Mapping[str, Any]]
) -> dict[str, dict[str, Any]]:
    """Each tree's own merge counters, read from the comparison that carries it.

    The counters live on the tree run's manifest, so a replayed tree reports the
    calls it made when it was first built, not the calls this study made. A tree
    is a cell of the family whose participant asked for it: a chunk arm belongs
    to ``chunk``, every other profile to ``pipeline``.
    """
    counters: dict[str, dict[str, Any]] = {}
    for entry in payload["trees"]:
        run_id = str(entry["run_id"])
        family = "chunk" if str(entry["profile"]) in CHUNK_ARMS else "pipeline"
        row = rows_by_run(reports[(family, run_id.split(".")[0])]).get(run_id)
        if row is None or row["kind"] != KIND_TREE:
            continue
        counters[run_id] = row["merge"]
    return counters


def extraction_run(doc_id: str, engine_id: str, repeat: int = 1) -> str:
    return f"{doc_id}.{engine_id}.r{repeat}"


def tree_run(doc_id: str, profile: str, repeat: int = 1, *, llm_mode: str = "live") -> str:
    return f"{doc_id}.{profile}.r{repeat}.{llm_mode}"


def files_containing(root: Path, needle: str) -> list[Path]:
    """Every file under ``root`` whose text contains ``needle``."""
    hits: list[Path] = []
    for path in sorted(root.rglob("*")):
        if not path.is_file():
            continue
        try:
            text = path.read_text(encoding="utf-8")
        except (OSError, UnicodeDecodeError):
            continue
        if needle in text:
            hits.append(path)
    return hits


# --------------------------------------------------------------------------- #
# the expected grid
# --------------------------------------------------------------------------- #


def expected_participants(
    family: str,
    doc_id: str,
    *,
    engines: Sequence[str] = ENGINES,
    arms: set[str] | None = None,
) -> set[str]:
    """Who the study asked for, per family: never who answered."""
    if family == "ocr":
        return set(engines)
    if family == "chunk":
        return set(CHUNK_ARMS) if arms is None else set(arms)
    return {
        primary_profile(engine_id).name
        for engine_id in engines
        if doc_id not in POLICY[engine_id].fails
    }


def expected_cells(
    docs: Sequence[str],
    *,
    engines: Sequence[str] = ENGINES,
    families: Sequence[str] = FAMILIES,
    arms: set[str] | None = None,
) -> dict[str, int]:
    """The expected grid, counted from the declared policy.

    One cell per engine per document per repeat for ``ocr``; one per engine per
    document per repeat whose extraction returned at all for ``pipeline`` (a
    tree needs an artifact, so only the engine that raised leaves a hole); one
    per chunk arm per document per repeat.
    """
    counts: dict[str, int] = {}
    for family in families:
        if family == "ocr":
            counts[family] = len(engines) * len(docs) * REPEATS
        elif family == "pipeline":
            counts[family] = sum(
                REPEATS
                for engine_id in engines
                for doc_id in docs
                if doc_id not in POLICY[engine_id].fails
            )
        else:
            counts[family] = len(CHUNK_ARMS if arms is None else arms) * len(docs) * REPEATS
    return counts


def assert_expected_grid(
    grid: Sequence[Mapping[str, Any]],
    docs: Sequence[str],
    *,
    engines: Sequence[str] = ENGINES,
    families: Sequence[str] = FAMILIES,
    arms: set[str] | None = None,
) -> None:
    """Every cell the study asked for, exactly once -- holes included.

    A cell that is missing here is the bug this gate exists to catch: the
    denominators below it would be computed from what survived.
    """
    assert Counter(cell["family"] for cell in grid) == Counter(
        expected_cells(docs, engines=engines, families=families, arms=arms)
    )
    for family in families:
        for doc_id in docs:
            cells = [
                cell for cell in grid if cell["family"] == family and cell["doc_id"] == doc_id
            ]
            filled = [(cell["repeat"], cell["participant"]) for cell in cells]
            expected = {
                (repeat, participant)
                for repeat in range(1, REPEATS + 1)
                for participant in expected_participants(
                    family, doc_id, engines=engines, arms=arms
                )
            }
            assert len(filled) == len(set(filled)), f"{family}/{doc_id}: a cell twice: {filled}"
            assert set(filled) == expected, (
                f"{family}/{doc_id}: expected {sorted(expected)}, got {sorted(set(filled))}"
            )


# --------------------------------------------------------------------------- #
# 1. the grid the study asked for, holes and all
# --------------------------------------------------------------------------- #


def test_the_study_keeps_the_whole_expected_grid(session: Session) -> None:
    result = session.study()
    payload = result.payload

    assert result.code == 1, "a failed run is an exit code, and the study still writes everything"
    assert payload["executed"] is True
    assert expected_cells(DOCS) == {"ocr": 24, "pipeline": 22, "chunk": 12}
    assert len(payload["grid"]) == 58
    assert_expected_grid(payload["grid"], DOCS)

    # The failure is a filled cell with its own status, in the family that is
    # compared at extraction level -- not a hole where a participant used to be.
    failed = [
        cell
        for cell in payload["grid"]
        if cell["family"] == "ocr"
        and cell["participant"] == "docling"
        and cell["doc_id"] == FAILING_DOC
    ]
    assert len(failed) == REPEATS
    assert {cell["status"] for cell in failed} == {"failed"}
    assert {cell["reason"] for cell in failed} == {"engine_failed"}

    # Every extraction the study asked for is in the payload, the failed one too.
    extractions = {str(entry["run_id"]): entry for entry in payload["extractions"]}
    assert len(extractions) == expected_cells(DOCS)["ocr"]
    assert extractions[extraction_run(FAILING_DOC, "docling")]["status"] == "failed"
    assert extractions[extraction_run(FAILING_DOC, "docling")]["artifact_digest"] is None
    assert extractions[extraction_run(FAILING_DOC, "docling", 2)]["status"] == "failed"

    # No family was skipped: every family ran for every document.
    assert payload["skipped"] == []

    reports = comparison_reports(payload)
    # The report's denominator is the grid, not the successes: one row per cell.
    for family in FAMILIES:
        for doc_id in DOCS:
            report = reports[(family, doc_id)]
            assert len(report["metadata"]["run_rows"]) == len(
                [
                    cell
                    for cell in payload["grid"]
                    if cell["family"] == family and cell["doc_id"] == doc_id
                ]
            ), f"{family}/{doc_id}: run_rows do not match the grid"
    # And the failed cell is in the report with its own status, not dropped.
    ocr_mixed = rows_by_run(reports[("ocr", FAILING_DOC)])
    failed_row = ocr_mixed[extraction_run(FAILING_DOC, "docling")]
    assert failed_row["status"] == "failed"
    assert failed_row["artifact"] == {
        "status": "not_measured",
        "reason": "engine_failed",
        "value": None,
    }
    assert failed_row["rankable"] is None, "nothing was ranked, so nothing is rankable"
    assert failed_row["quality"] is None and failed_row["quality_reason"] is None
    assert any(
        "docling" in limitation and "no artifact" in limitation
        for limitation in reports[("ocr", FAILING_DOC)]["limitations"]
    )

    # The engine that read no page at all is a named, unrankable artifact -- and
    # the engine with one unreadable page still has a rankable one.
    ocr_scan = rows_by_run(reports[("ocr", UNSUPPORTED_DOC)])
    native_row = ocr_scan[extraction_run(UNSUPPORTED_DOC, "native")]
    assert native_row["artifact_status"] == STATUS_UNSUPPORTED_MODALITY
    assert native_row["pages_without_text"] == len(PAGES)
    assert native_row["rankable"] is False
    marker_row = ocr_scan[extraction_run(UNSUPPORTED_DOC, "marker")]
    assert marker_row["artifact_status"] == "ok"
    assert marker_row["pages_without_text"] == 1
    assert marker_row["rankable"] is True
    # The census belongs to the run that was scored, so it is read per run: the
    # engine that read no page has both pages without a hypothesis, and the
    # engine that read one has one.
    runs = reports[("ocr", UNSUPPORTED_DOC)]["text_metrics"]["runs"]
    assert runs[extraction_run(UNSUPPORTED_DOC, "native")]["hypothesis"]["missing_pages"] == len(
        PAGES
    )
    assert runs[extraction_run(UNSUPPORTED_DOC, "marker")]["hypothesis"]["missing_pages"] == 1


def test_the_pairs_name_why_an_unsupported_run_cannot_be_compared(session: Session) -> None:
    reports = comparison_reports(session.study().payload)
    report = reports[("ocr", UNSUPPORTED_DOC)]
    pairs = report["paired_usage"]["pairs"]
    assert pairs, "the ocr family pairs engines; an empty list would be a silent absence"

    with_native = [pair for pair in pairs if "native" in pair["participants"]]
    assert with_native, "every engine pair is formed, including the unrankable ones"
    for pair in with_native:
        assert pair["compatible"] is False
        assert any(
            reason.endswith("unsupported_modality") for reason in pair["incompatible_reasons"]
        ), pair["incompatible_reasons"]

    comparable = [pair for pair in pairs if pair["compatible"]]
    assert comparable, "the readable engines still pair"


# --------------------------------------------------------------------------- #
# 2. null semantics: a checked page, a copy, and no gold at all
# --------------------------------------------------------------------------- #


def test_a_checked_page_scores_and_a_copy_of_the_candidate_abstains(session: Session) -> None:
    reports = comparison_reports(session.study().payload)
    report = reports[("ocr", "digital-1")]
    counts = report["counts"]
    assert counts["documents"] == 1
    assert counts["documents_with_gold"] == 1
    assert counts["pages_total"] == len(PAGES)
    assert counts["pages_with_gold"] == len(PAGES)
    assert counts["pages_without_gold"] == 0

    # The text stage measures every page the gold covers: a transcript that
    # agrees with the candidate is a perfect page, and it is the structure stage
    # that refuses to read one as a placement the run got right.
    rates = report["text_metrics"]["runs"][extraction_run("digital-1", "vision")]["rates"]
    assert rates["cer"]["micro"]["status"] == "measured"
    assert rates["cer"]["micro"]["denominator"] > 0

    row = rows_by_run(report)[extraction_run("digital-1", "vision")]
    structure = row["structure_metrics"]
    assert structure["status"] == "measured"
    assert structure["confidence"] == 0.90
    assert structure["pages"] == {
        "marked": len(PAGES),
        "considered": len(PAGES),
        "scored": 1,
        "skipped": 1,
    }
    assert structure["skipped"] == {"self_scoring": 1}
    detail = {page["page_index"]: page for page in structure["pages_detail"]}
    assert detail[0]["status"] == "not_measured"
    assert detail[0]["reason"] == "self_scoring"
    assert detail[1]["status"] == "measured"
    # The heading and the section start the human wrote down are the ones the
    # engine placed, and the word the gold adds is the only difference: a clean 1.0.
    for kind in ("heading", "boundary"):
        block = structure["kinds"][kind]
        assert block["status"] == "measured"
        assert block["gold_marks"] == 1
        assert block["true_positives"] == 1
        assert block["false_positives"] == 0
        assert block["false_negatives"] == 0
        assert block["f1"]["value"] == 1.0


def test_a_page_the_engine_could_not_read_abstains_by_its_own_name(session: Session) -> None:
    reports = comparison_reports(session.study().payload)
    report = reports[("ocr", UNSUPPORTED_DOC)]
    row = rows_by_run(report)[extraction_run(UNSUPPORTED_DOC, "marker")]
    structure = row["structure_metrics"]
    # One page was read and scored; the other carries the engine's own word for
    # why it could not be, rather than a blanket failure name.
    assert structure["status"] == "measured"
    assert structure["pages"] == {
        "marked": len(PAGES),
        "considered": len(PAGES),
        "scored": 1,
        "skipped": 1,
    }
    assert structure["skipped"] == {STATUS_UNSUPPORTED_MODALITY: 1}
    detail = {page["page_index"]: page for page in structure["pages_detail"]}
    assert detail[1]["status"] == "not_measured"
    assert detail[1]["reason"] == STATUS_UNSUPPORTED_MODALITY

    # An engine that read no page at all measures nothing, and says the same.
    nothing = rows_by_run(report)[extraction_run(UNSUPPORTED_DOC, "native")]["structure_metrics"]
    assert nothing["status"] == "not_measured"
    assert nothing["reason"] == STATUS_UNSUPPORTED_MODALITY
    assert nothing["skipped"] == {STATUS_UNSUPPORTED_MODALITY: len(PAGES)}
    assert nothing["pages"]["scored"] == 0
    for kind in ("heading", "boundary"):
        assert nothing["kinds"][kind]["status"] == "not_measured"
        assert nothing["kinds"][kind]["gold_marks"] == 0
        assert nothing["kinds"][kind]["f1"]["value"] is None


def test_a_document_with_no_gold_is_not_measured_rather_than_zero(session: Session) -> None:
    reports = comparison_reports(session.study().payload)
    report = reports[("ocr", FAILING_DOC)]
    counts = report["counts"]
    # No reference at all: not "ran and found nothing" but "nothing to run
    # against". Every census over the pages is unknown -- ``None``, not zero --
    # so the unknowns read as "not measured" rather than "not reported".
    for name in (
        "documents",
        "documents_with_gold",
        "pages_total",
        "pages_with_gold",
        "pages_without_gold",
        "pages_empty_hypothesis",
        "pages_missing_hypothesis",
    ):
        assert counts[name] is None, name
    assert counts["paired_documents"] == 1
    assert report["text_metrics"]["status"] == "not_measured"
    assert report["text_metrics"]["reason"] == "no_scored_documents"
    assert any(
        limitation == f"{FAILING_DOC}: no gold, so its pages cannot be scored"
        for limitation in report["limitations"]
    )

    row = rows_by_run(report)[extraction_run(FAILING_DOC, "vision")]
    assert "structure_metrics" not in row
    assert "gold_reference" not in row
    # There is no reference, so there is no digest field and no zero pretending
    # to be one: absence is absence, and it is not the engine's fault.
    assert "gold_digest" not in row
    assert row["rankable"] is True, "no reference is not the engine's fault"
    # The document still has an artifact, it is simply unscored.
    assert row["artifact"]["status"] == "measured"
    assert row["artifact"]["value"] == len(PAGES)


# --------------------------------------------------------------------------- #
# 3. a paid answer with no usage is a miss, never a free page
# --------------------------------------------------------------------------- #


def test_a_page_with_no_usage_is_a_partial_block(session: Session) -> None:
    reports = comparison_reports(session.study().payload)
    report = reports[("ocr", MISSING_USAGE_DOC)]
    row = rows_by_run(report)[extraction_run(MISSING_USAGE_DOC, "vision")]
    usage = row["usage"]
    assert usage["source"] == "artifact"
    assert usage["pages"] == len(PAGES)
    assert usage["pages_with_usage"] == 1
    assert usage["complete"] is False
    assert usage["status"] == "partial"
    assert usage["reason"] == "usage_missing"
    # Every top-level counter is unknown, not zero -- the known page is a subtotal.
    for name in ("attempts", "prompt_tokens", "completion_tokens"):
        assert usage[name] is None, name
    assert usage["subtotal"]["prompt_tokens"] == 100
    assert usage["subtotal"]["completion_tokens"] == 20
    assert usage["subtotal"]["pages_with_usage"] == 1

    # A local engine has no provider to report, so its usage is not measured --
    # and never a zero bill.
    local = rows_by_run(report)[extraction_run(MISSING_USAGE_DOC, "native")]
    assert local["model"] is None
    assert local["usage"]["status"] == "not_measured"
    assert local["usage"]["reason"] == "usage_missing"
    assert local["usage"]["prompt_tokens"] is None
    assert local["usage"]["subtotal"]["prompt_tokens"] == 0


def test_a_fully_reported_run_keeps_its_tokens(session: Session) -> None:
    """A run that stated every counter publishes them, input and output apart."""
    reports = comparison_reports(session.study().payload)
    report = reports[("ocr", "digital-1")]
    row = rows_by_run(report)[extraction_run("digital-1", "vision")]
    usage = row["usage"]
    assert usage["complete"] is True
    assert usage["status"] == "measured"
    assert usage["attempts"] == len(PAGES)
    assert usage["prompt_tokens"] == 100 * len(PAGES)
    assert usage["completion_tokens"] == 20 * len(PAGES)
    assert usage["subtotal"] == {
        "attempts": len(PAGES),
        "prompt_tokens": 100 * len(PAGES),
        "completion_tokens": 20 * len(PAGES),
        "pages_with_usage": len(PAGES),
    }
    # The pair reads the two runs' tokens against each other, and nothing else.
    # One side has no provider to report to, so there is no difference to state
    # -- a typed miss, never the paid run's tokens read as if the side that said
    # nothing had spent zero.
    pair = next(
        pair
        for pair in report["paired_usage"]["pairs"]
        if pair["participants"] == ["native", "vision"]
    )
    tokens = pair["usage"]
    assert tokens["usage_a"]["status"] == "not_measured", "the local engine stated no tokens"
    assert tokens["usage_b"]["status"] == "measured"
    assert tokens["usage_b"]["prompt_tokens"] == 100 * len(PAGES)
    assert tokens["usage_b"]["completion_tokens"] == 20 * len(PAGES)
    assert tokens["delta"]["status"] == "not_measured"
    assert tokens["delta"]["reason"] == "incomplete_pairing"
    assert tokens["delta"]["prompt_tokens"] is None
    assert tokens["delta"]["completion_tokens"] is None


# --------------------------------------------------------------------------- #
# 4. resume: replay from disk, call nothing, spend nothing
# --------------------------------------------------------------------------- #


def test_a_resume_replays_every_run_and_calls_nothing(session: Session) -> None:
    first = session.study()
    before = session.calls()
    builds = session.gateway.builds
    merges = session.gateway.client.calls
    first_reports = comparison_reports(first.payload)
    usage_before = first_reports[("ocr", "digital-1")]["usage"]
    digests_before = {
        str(entry["run_id"]): entry["artifact_digest"] for entry in first.payload["extractions"]
    }
    signatures_before = {
        str(entry["run_id"]): entry["semantic_signature"] for entry in first.payload["trees"]
    }
    counters_before = merge_counters(first.payload, first_reports)

    second = session.replay()
    assert second.code == first.code == 1

    # A resume reads nothing it already holds. The one exception is the run
    # whose engine failed: a failed run is not a completed run, so it is
    # attempted again -- once per repeat -- and its tree still finds no artifact
    # to build from. Nothing else moves, and nothing is merged again.
    assert session.calls() == {**before, "docling": before["docling"] + REPEATS}
    assert session.gateway.builds == builds, "a replay must not build a gateway client"
    assert session.gateway.client.calls == merges, "a replay must not merge a section again"
    assert merges > 0, "the first study did merge; otherwise this proves nothing"

    payload = second.payload
    plan = payload["plan"]
    # The requested id names the work; a retry lands beside the attempt that
    # failed rather than on top of it, so its run id carries a suffix and the
    # base is what says which run is being attempted again.
    misses = {
        (re.sub(r"(-a\d+)+$", "", str(entry["run_id"])), str(entry["cache_reason"]))
        for entry in [*plan["extractions"], *plan["trees"]]
        if entry["cache"] == "miss"
    }
    assert misses == {
        (extraction_run(FAILING_DOC, "docling"), "not_successful"),
        (extraction_run(FAILING_DOC, "docling", REPEATS), "not_successful"),
        (tree_run(FAILING_DOC, "docling_atomic"), "no_artifact"),
        (tree_run(FAILING_DOC, "docling_atomic", REPEATS), "no_artifact"),
    }, "a replay retries the failed run, and only the failed run"
    assert plan["cache"]["hits"] == len(plan["extractions"]) + len(plan["trees"]) - len(misses)

    # Every completed run comes back under its own id, with its own digest and
    # its own tree signature: replayed from disk, not recomputed, and the run
    # that had to be made again is a new run beside the one that failed.
    extractions = {
        str(entry["run_id"]): entry["artifact_digest"] for entry in payload["extractions"]
    }
    assert set(digests_before) - set(extractions) == {
        extraction_run(FAILING_DOC, "docling"),
        extraction_run(FAILING_DOC, "docling", REPEATS),
    }, "only the failed run changed identity"
    replayed = {run_id: extractions[run_id] for run_id in extractions if run_id in digests_before}
    assert replayed == {run_id: digests_before[run_id] for run_id in replayed}
    assert len(replayed) == len(digests_before) - REPEATS
    retried = {
        run_id: digest for run_id, digest in extractions.items() if run_id not in digests_before
    }
    assert len(retried) == REPEATS
    assert set(retried.values()) == {None}, "the retry failed again, and the report says so"
    assert {
        str(entry["run_id"])
        for entry in plan["extractions"]
        if entry["run_id"] not in digests_before and entry["cache"] == "miss"
    } == set(retried), "the plan named the directory each retry took"
    assert {
        str(entry["run_id"]): entry["semantic_signature"] for entry in payload["trees"]
    } == signatures_before

    # The merge counters are stable, and each run's counters are its own calls:
    # the report does not attribute the first study's merges to the second.
    after_reports = comparison_reports(payload)
    counters_after = merge_counters(payload, after_reports)
    assert counters_after == counters_before
    merged = [counters for counters in counters_after.values() if counters["llm_calls"]]
    assert merged, "some tree profile merges; otherwise the counter check is vacuous"
    for counters in merged:
        assert counters["attempts"] == counters["llm_calls"]
        assert counters["sections_merged"] <= counters["llm_calls"]
        assert counters["usage_unknown_reason"] == "the gateway reported no token usage"
    # Every merge call the gateway served belongs to a tree's own counter. Not
    # every tree is a cell of a family -- the profiles a family does not compare
    # are built and carried by no report -- so the total is what gives the
    # accounting its teeth: a merge that went uncounted would not add up here.
    assert sum(counters["llm_calls"] for counters in merged) == merges
    assert after_reports[("ocr", "digital-1")]["usage"] == usage_before


def test_a_dry_run_plans_the_same_study_and_runs_none_of_it(session: Session) -> None:
    session.study()
    before = session.calls()
    builds = session.gateway.builds

    planned = session.dry_run()
    assert planned.code == 0
    payload = planned.payload
    assert payload["executed"] is False
    plan = payload["plan"]
    assert plan["llm_mode"] == "live"
    assert plan["resume"] is True
    assert plan["families"] == list(FAMILIES)
    assert plan["corpus"]["documents"] == list(DOCS)
    # The plan names the whole grid before it runs any of it: every engine's
    # extraction, and every profile the families ask for.
    assert set(plan["profiles"]) == set(LIVE_PROFILES)
    assert profiles_of(plan) == set(LIVE_PROFILES)
    assert len(plan["extractions"]) == len(ENGINES) * len(DOCS) * REPEATS
    assert len(plan["trees"]) == len(LIVE_PROFILES) * len(DOCS) * REPEATS

    # Everything is on disk except the run whose engine failed: a dry run plans
    # that retry, names why, and plans nothing else as a miss.
    misses = {
        (re.sub(r"(-a\d+)+$", "", str(entry["run_id"])), str(entry["cache_reason"]))
        for entry in [*plan["extractions"], *plan["trees"]]
        if entry["cache"] == "miss"
    }
    assert misses == {
        (extraction_run(FAILING_DOC, "docling"), "not_successful"),
        (extraction_run(FAILING_DOC, "docling", REPEATS), "not_successful"),
        (tree_run(FAILING_DOC, "docling_atomic"), "no_artifact"),
        (tree_run(FAILING_DOC, "docling_atomic", REPEATS), "no_artifact"),
    }
    assert plan["cache"]["hits"] == len(plan["extractions"]) + len(plan["trees"]) - len(misses)
    hits = [entry for entry in [*plan["extractions"], *plan["trees"]] if entry["cache"] == "hit"]
    assert all(entry["cache_reason"] is None for entry in hits), "a hit carries no excuse"

    fresh = session.dry_run("--fresh")
    fresh_plan = fresh.payload["plan"]
    assert fresh_plan["resume"] is False
    assert fresh_plan["cache"] == {
        "hits": 0,
        "misses": len(fresh_plan["extractions"]) + len(fresh_plan["trees"]),
    }
    assert all(entry["cache_reason"] == "fresh" for entry in fresh_plan["extractions"])
    assert all(entry["cache_reason"] == "fresh" for entry in fresh_plan["trees"])

    assert session.calls() == before, "a dry run that runs something is not a dry run"
    assert session.gateway.builds == builds


def profiles_of(plan: Mapping[str, Any]) -> set[str]:
    return {str(entry["profile"]) for entry in plan["trees"]}


# --------------------------------------------------------------------------- #
# 5. changing the gold re-scores from disk
# --------------------------------------------------------------------------- #


def test_a_changed_gold_re_scores_without_reading_anything_again(session: Session) -> None:
    first = session.study()
    # ``regolded`` replays first: pin that replay here so the delta asserted
    # below is the re-score's own reads, whatever order the tests run in.
    session.replay()
    before = session.calls()
    merges = session.gateway.client.calls
    builds = session.gateway.builds

    # Change the reference: page 1 is rewritten by the human. An authoring-side
    # change; the corpus PDFs are untouched.
    write_gold(
        session.corpus.base_dir / "gold-digital.json",
        "digital-1",
        [
            gold_page(0, copied_text("digital-1", 0), (heading_text(0),)),
            changed_page(1),
        ],
    )

    second = session.regolded()
    # A reference change is not a reason to read a page again: every engine is
    # left alone. The failed docling run is the one exception a resume always
    # makes -- a failure is not a completed run -- and it fails again.
    assert session.calls() == {**before, "docling": before["docling"] + REPEATS}
    assert session.gateway.client.calls == merges, "re-scoring is not a reason to merge again"
    assert session.gateway.builds == builds

    reports = comparison_reports(second.payload)
    first_report = comparison_reports(first.payload)[("ocr", "digital-1")]
    report = reports[("ocr", "digital-1")]
    row = rows_by_run(report)[extraction_run("digital-1", "vision")]
    first_row = rows_by_run(first_report)[extraction_run("digital-1", "vision")]
    assert row["gold_digest"] != first_row["gold_digest"], "the reference really changed"
    assert row["artifact_digest"] == first_row["artifact_digest"]
    # A worse reference is a worse score: the number moved with the gold, which
    # is the only thing that could have moved it. The tokens did not move.
    assert (
        report["text_metrics"]["runs"][extraction_run("digital-1", "vision")]["rates"]["cer"][
            "micro"
        ]["value"]
        > first_report["text_metrics"]["runs"][extraction_run("digital-1", "vision")]["rates"][
            "cer"
        ]["micro"]["value"]
    )
    assert row["usage"] == first_row["usage"]
    # The changed page is now too different to align rather than scored wrong.
    structure = row["structure_metrics"]
    assert structure["pages"]["scored"] == 0
    assert "alignment_below_threshold" in structure["skipped"]
    assert structure["status"] == "not_measured"


# --------------------------------------------------------------------------- #
# 6. a document added by editing the manifest JSON alone
# --------------------------------------------------------------------------- #


def test_a_document_added_by_editing_the_manifest_is_run(session: Session) -> None:
    session.regolded()
    before = session.calls()
    manifest_before = json.loads(session.corpus.manifest_path.read_text(encoding="utf-8"))

    # ``extended`` adds ``extra-1`` by appending its entry to the manifest JSON
    # and writing its PDF. No production code changes: a document is data, and
    # the corpus schema is the interface.
    result = session.extended()
    after = session.calls()
    docs = (*DOCS, "extra-1")
    manifest_after = json.loads(session.corpus.manifest_path.read_text(encoding="utf-8"))
    assert [doc["doc_id"] for doc in manifest_after["documents"]] == list(docs)
    assert manifest_after["documents"][: len(DOCS)] == manifest_before["documents"], (
        "the existing entries are unchanged: the manifest edit is additive"
    )
    assert manifest_after["corpus_version"] == manifest_before["corpus_version"]

    payload = result.payload
    assert payload["plan"]["corpus"]["documents"] == list(docs)
    assert_expected_grid(payload["grid"], docs)
    assert expected_cells(docs) == {"ocr": 32, "pipeline": 30, "chunk": 16}
    assert len(payload["grid"]) == 78
    assert len(payload["trees"]) == len(LIVE_PROFILES) * len(docs) * REPEATS - REPEATS

    # Only the new document is read -- plus the run a resume is obliged to
    # retry, which is the failed engine on the failing document, once per
    # repeat. Nothing that was already on disk is read again.
    read_back = {
        engine_id: Counter(session.engines[engine_id].documents[before[engine_id] :])
        for engine_id in ENGINES
    }
    assert read_back == {
        "vision": Counter({"extra-1": REPEATS}),
        "marker": Counter({"extra-1": REPEATS}),
        "docling": Counter({"extra-1": REPEATS, FAILING_DOC: REPEATS}),
        "native": Counter({"extra-1": REPEATS}),
    }
    # Every engine reads the new document twice; docling additionally retries
    # the failing document's repeats, which no completed run leaves behind.
    expected_calls = {engine_id: before[engine_id] + REPEATS for engine_id in ENGINES}
    expected_calls["docling"] += REPEATS
    assert after == expected_calls

    # The new document is planned as reads: its extractions for the first time,
    # its trees for want of the artifact those extractions are about to leave.
    # Every old run is a hit, except the failure a resume must retry, and a hit
    # carries no reason at all -- there is nothing to explain.
    plan = payload["plan"]
    entries = [*plan["extractions"], *plan["trees"]]
    assert Counter(
        str(entry["cache_reason"]) for entry in entries if str(entry["doc_id"]) == "extra-1"
    ) == Counter({"no_run": len(ENGINES) * REPEATS, "no_artifact": len(LIVE_PROFILES) * REPEATS})
    old = [entry for entry in entries if str(entry["doc_id"]) != "extra-1"]
    hits = [entry for entry in old if entry["cache"] == "hit"]
    assert len(hits) == len(old) - 2 * REPEATS, "only the failed run is planned again"
    assert all(entry["cache_reason"] is None for entry in hits)
    attempted_again = {
        re.sub(r"(-a\d+)+$", "", str(entry["run_id"])) for entry in old if entry["cache"] == "miss"
    }
    assert attempted_again == {
        extraction_run(FAILING_DOC, "docling"),
        extraction_run(FAILING_DOC, "docling", REPEATS),
        tree_run(FAILING_DOC, "docling_atomic"),
        tree_run(FAILING_DOC, "docling_atomic", REPEATS),
    }

    # The new document has no gold: it is unscored, not scored as zero.
    reports = comparison_reports(payload)
    report = reports[("ocr", "extra-1")]
    assert report["counts"]["documents"] is None
    assert report["counts"]["documents_with_gold"] is None
    assert report["text_metrics"]["status"] == "not_measured"
    assert report["text_metrics"]["reason"] == "no_scored_documents"
    assert any(limitation.startswith("extra-1: no gold") for limitation in report["limitations"])


# --------------------------------------------------------------------------- #
# 7. --fresh: everything again, in run ids of its own
# --------------------------------------------------------------------------- #


def test_a_fresh_repeat_lands_in_new_run_ids_and_rebills_everything(session: Session) -> None:
    extended = session.extended()
    before = session.calls()
    merges = session.gateway.client.calls
    docs = (*DOCS, "extra-1")
    previous = {
        str(entry["run_id"])
        for entry in [*extended.payload["extractions"], *extended.payload["trees"]]
    }

    result = session.fresh()
    payload = result.payload
    plan = payload["plan"]
    # A fresh repeat is not a resume: nothing is on disk as far as the plan is
    # concerned, and every run is planned for the same reason.
    assert plan["resume"] is False
    assert plan["cache"] == {"hits": 0, "misses": len(plan["extractions"]) + len(plan["trees"])}
    assert_expected_grid(payload["grid"], docs)
    assert len(payload["extractions"]) == len(ENGINES) * len(docs) * REPEATS
    assert len(payload["trees"]) == len(LIVE_PROFILES) * len(docs) * REPEATS - REPEATS

    grew = {engine_id: session.calls()[engine_id] - before[engine_id] for engine_id in ENGINES}
    assert grew == {engine_id: len(docs) * REPEATS for engine_id in ENGINES}
    assert session.gateway.client.calls > merges, "a fresh repeat merges again"

    # Every run lands in a directory of its own, beside the run it repeats --
    # never on top of it, and the plan names the directory each run will take,
    # which is what makes a repeat auditable rather than a silent overwrite.
    fresh_ids = {str(entry["run_id"]) for entry in [*payload["extractions"], *payload["trees"]]}
    assert not (fresh_ids & previous), "a fresh run reused a run id"
    assert all(re.search(r"-a\d+$", run_id) for run_id in fresh_ids)
    planned = {str(entry["run_id"]) for entry in [*plan["extractions"], *plan["trees"]]}
    assert fresh_ids <= planned, "every run is a run the plan named"
    assert planned - fresh_ids == {
        tree_run(FAILING_DOC, "docling_atomic", repeat) for repeat in range(1, REPEATS + 1)
    }, "the only planned runs with no run of their own are the trees with no artifact"

    # The first study's directories are still there, and the fresh ones beside
    # them: a repeat is a new run, not a rewrite.
    base_run = extraction_run("digital-1", "vision")
    runs_root = session.root / "digital-1" / "runs"
    assert (runs_root / base_run).is_dir()
    assert any(path.name.startswith(f"{base_run}-a") for path in runs_root.iterdir())


# --------------------------------------------------------------------------- #
# 8. compare: the same numbers off the same runs, and the override path
# --------------------------------------------------------------------------- #


def test_compare_reports_two_runs_offline(session: Session) -> None:
    session.study()
    out = session.corpus.base_dir / "compare-1"
    result = session.run_cli(
        [
            "compare",
            "--family",
            "ocr",
            "--run",
            f"digital-1:{extraction_run('digital-1', 'vision')}",
            "--run",
            f"digital-1:{extraction_run('digital-1', 'native')}",
            "--out",
            str(out),
        ]
    )
    assert result.code == 0, result.err
    written = sorted(path.suffix for path in out.iterdir())
    assert written.count(".json") == 2, "the report and the comparison manifest"
    assert written.count(".csv") == 1 and written.count(".md") == 1

    report = json.loads((out / "report.json").read_text(encoding="utf-8"))
    assert report["counts"]["documents"] == 1
    assert report["counts"]["documents_with_gold"] == 1
    assert report["counts"]["pages_with_gold"] == len(PAGES)
    assert len(report["metadata"]["run_rows"]) == 2
    # The paid run reported its tokens; the local one has no provider to report
    # to, so the comparison keeps the tokens it has and leaves the total unknown
    # rather than reading an unreported side as a free one.
    usage = report["usage"]
    assert usage["status"] == "partial"
    assert usage["reason"] == "usage_missing"
    assert usage["documents"] == 2
    assert usage["reported_documents"] == 1
    assert usage["subtotal"]["prompt_tokens"] == 100 * len(PAGES)
    pair = report["paired_usage"]["pairs"][0]
    assert pair["compatible"] is True, "both runs read the document; only their bills differ"
    sides = dict(zip(pair["participants"], (pair["usage"]["usage_a"], pair["usage"]["usage_b"])))
    assert sides["native"]["status"] == "not_measured"
    assert sides["vision"]["prompt_tokens"] == 100 * len(PAGES)
    delta = pair["usage"]["delta"]
    assert delta["status"] == "not_measured"
    assert delta["reason"] == "incomplete_pairing"
    assert delta["prompt_tokens"] is None


def test_gold_is_found_in_the_manifest_and_overridden_on_the_command_line(
    session: Session,
) -> None:
    added = session.extended()
    assert added.payload["plan"]["corpus"]["documents"] == [*DOCS, "extra-1"]

    # ``extra-1`` declares no reference in the manifest, so the study could not
    # score it -- that is the discovery half of the rule. A document with no
    # reference is not a document with zero references: the counters are absent,
    # never fabricated as zeros, and a limitation names the missing reference.
    without = comparison_reports(added.payload)[("ocr", "extra-1")]
    assert without["text_metrics"]["status"] == "not_measured"
    assert without["text_metrics"]["reason"] == "no_scored_documents"
    assert without["counts"]["documents_with_gold"] is None
    assert any(limitation.startswith("extra-1: no gold") for limitation in without["limitations"])

    # Naming the file on the command line is the override half: the same run,
    # now scored, with its headings placed.
    base_dir = session.corpus.base_dir
    write_gold(
        base_dir / "gold-extra.json",
        "extra-1",
        [checked_page("extra-1", index) for index in PAGES],
    )
    out = base_dir / "compare-override"
    result = session.run_cli(
        [
            "compare",
            "--family",
            "ocr",
            "--run",
            f"extra-1:{extraction_run('extra-1', 'vision')}",
            "--run",
            f"extra-1:{extraction_run('extra-1', 'native')}",
            "--gold",
            f"extra-1={base_dir / 'gold-extra.json'}",
            "--out",
            str(out),
        ]
    )
    assert result.code == 0, result.err
    report = json.loads((out / "report.json").read_text(encoding="utf-8"))
    assert report["counts"]["documents_with_gold"] == 1
    assert report["counts"]["pages_with_gold"] == len(PAGES)
    row = rows_by_run(report)[extraction_run("extra-1", "vision")]
    assert row["structure_metrics"]["pages"] == {
        "marked": len(PAGES),
        "considered": len(PAGES),
        "scored": len(PAGES),
        "skipped": 0,
    }
    assert row["structure_metrics"]["kinds"]["heading"]["f1"]["value"] == 1.0


# --------------------------------------------------------------------------- #
# 9. a family whose arms were never planned is a named gap
# --------------------------------------------------------------------------- #


def test_a_family_whose_arms_were_never_planned_is_a_named_gap(session: Session) -> None:
    """A chunk family needs two profiles of one engine; a mode that drops them says so.

    The family was asked for and could not run, and no cell is invented to fill
    the denominator: one named gap per document, and no comparison at all.
    """
    docs = manifest_docs(session)
    before = session.calls()
    gap = session.run_cli(
        [
            "study",
            "--families",
            "chunk",
            "--engines",
            "marker",
            "--repeats",
            str(REPEATS),
            "--llm-mode",
            "none",
        ],
        root=session.corpus.base_dir / "study-gap",
    )
    assert gap.code == 0, gap.err
    payload = gap.payload
    assert payload["plan"]["profiles"] == ["marker_atomic"], "the arms' profiles are not planned"
    assert payload["comparisons"] == []
    assert_expected_grid(
        payload["grid"], docs, engines=("marker",), families=("chunk",), arms=set()
    )
    assert payload["grid"] == []
    assert sorted(
        (entry["family"], entry["doc_id"], entry["reason"]) for entry in payload["skipped"]
    ) == [("chunk", doc_id, "no_chunk_profile_tree") for doc_id in sorted(docs)]
    assert session.calls() != before, "the marker engine did run; only the merge was skipped"


# --------------------------------------------------------------------------- #
# 10. no file carries the key or the merge prompt
# --------------------------------------------------------------------------- #


def test_no_written_file_carries_the_key_or_a_merge_prompt(session: Session) -> None:
    session.fresh()
    roots = [session.root, session.corpus.base_dir / "study-gap"]
    assert session.vision.keys, "the paid engine was called, so the key was in play"
    assert set(session.vision.keys) == {FAKE_API_KEY}
    assert session.gateway.client.prompts, "the fake merge was called, so its prompt exists"

    # The scanner is not vacuous: a file written with the needle is found, and
    # the artifacts are reached -- they hold the pages the engines reported.
    probe = session.root / "scan-probe.tmp"
    probe.write_text(f"a planted {FAKE_API_KEY} and {FAKE_HEADER}", encoding="utf-8")
    assert files_containing(session.root, FAKE_API_KEY) == [probe]
    probe.unlink()
    assert files_containing(session.root, "Chuong 1"), "the scan reaches the stored artifacts"

    for root in roots:
        assert files_containing(root, FAKE_API_KEY) == [], f"{root}: the key reached a file"
        assert files_containing(root, FAKE_HEADER) == [], (
            f"{root}: the header value reached a file"
        )
        for prompt in session.gateway.client.prompts:
            instruction = prompt.splitlines()[0]
            assert files_containing(root, instruction) == [], f"{root}: a merge prompt was written"

    # Every report passes the fail-closed raw-text scan, and the scanner bites.
    assert find_raw_text_keys({"page": {"text": "a transcript"}}) == ["page.text"]
    assert find_raw_text_keys({"blob": "x" * 600}) == ["blob"]
    reports = list(session.root.rglob("*.json"))
    assert reports, "the study wrote reports to scan"
    scanned = 0
    for path in reports:
        payload = json.loads(path.read_text(encoding="utf-8"))
        if not (isinstance(payload, dict) and "counts" in payload):
            continue
        assert find_raw_text_keys(payload) == [], f"{path} carries raw transcription"
        scanned += 1
    assert scanned >= len(FAMILIES) * len(DOCS), "every comparison report was scanned"
