"""Offline tests for the engine adapters: hand-written inputs, synthetic engines, no network.

Every test here runs without a model, a model download, a package install or a
paid request. The three engines that need a real one are pinned differently:

* :mod:`marker` is exercised on a hand-written dump through an injected
  converter, so its mapping, its scoping and its lazy import are covered
  without ``create_model_dict`` ever running;
* :mod:`docling` is not installed, and its tests say so: they build a synthetic
  document from the documented API and skip the run-time path rather than
  pretending it was verified;
* :mod:`native` reads a real text layer, from a PDF written byte by byte here;
* :mod:`vision` is driven by an injected client and by an httpx mock transport,
  so the real client's payload, retry and token accounting are covered without
  contacting anything.
"""

from __future__ import annotations

import dataclasses
import hashlib
import json
import re
import subprocess
import sys
from collections.abc import Sequence
from dataclasses import dataclass
from enum import Enum
from pathlib import Path
from typing import Any

import httpx
import pypdfium2
import pytest

from research.ocr_pipeline.engines import (
    ARTIFACT_ONLY_KINDS,
    ARTIFACT_VERSION,
    ENGINE_IDS,
    FALLBACK_DEPTH,
    NORMALIZED_KINDS,
    RUNTIME_BLOCK_KINDS,
    EngineArtifact,
    EngineBlock,
    EngineNotAvailableError,
    EnginePage,
    content_pages,
    continuity_breaks,
    count_block_kinds,
    follows_previous,
    lift_running,
    load_engine,
    reset_continuity,
    sha256_bytes,
)
from research.ocr_pipeline.engines.worker import (
    IsolatedWorkerError,
    artifact_from_payload,
    artifact_payload,
    worker_main,
)
from research.ocr_pipeline.tree_builders import (
    MARKER_BLOCK_KINDS,
    MarkerStructureError,
    ReplayError,
    build_atomic,
    build_marker_structure,
    marker_compatibility,
)
from workflows.ingest_file.vision_ocr import (
    PROMPT,
    PageExtraction,
    VisionOCRConfig,
    VisionOCRError,
    VisionPage,
)
from workflows.ingest_file.vision_parser import VisionDocumentParser

REPO_ROOT = Path(__file__).resolve().parents[4]

DETECTABLE_KINDS = ("marker", "docling")
"""Packages importing the adapters must never pull in: both own weights."""


# --- synthetic inputs -----------------------------------------------------------


def make_pdf(page_count: int, *, text_pages: tuple[int, ...] = ()) -> bytes:
    """Return a valid PDF whose ``text_pages`` carry a text object.

    A page without a text object is what a scanned page looks like to pypdfium2:
    the text layer exists but is empty. Written byte by byte so nothing is read
    from disk and no book leaves the repository.
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
        stream = b"BT /F1 24 Tf 72 700 Td (Alpha beta gamma) Tj ET" if index in text_pages else b""
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


@pytest.fixture
def scanned_pdf(tmp_path: Path) -> Path:
    """A two-page PDF: the first page has a text layer, the second does not."""
    path = tmp_path / "synthetic.pdf"
    path.write_bytes(make_pdf(2, text_pages=(0,)))
    return path


def marker_block(
    kind: str,
    markup: str | None = None,
    *,
    page: int = 0,
    number: int = 1,
    hierarchy: dict[str, str] | None = None,
    bbox: list[float] | None = None,
) -> dict[str, Any]:
    """One block of a Marker JSON dump, shaped as the JSONRenderer writes it."""
    block: dict[str, Any] = {
        "id": f"/page/{page}/{kind}/{number}",
        "block_type": kind,
        "children": None,
        "bbox": bbox if bbox is not None else [1.0, 2.0, 3.0, 4.0],
    }
    if markup is not None:
        block["html"] = markup
    if hierarchy is not None:
        block["section_hierarchy"] = hierarchy
    return block


def marker_dump(
    pages: list[list[dict[str, Any]]], page_ids: Sequence[int] | None = None
) -> dict[str, Any]:
    """A Marker document dump whose children are the given pages.

    ``page_ids`` overrides the physical index each page id carries, which is how a
    dump from a scoped live run looks: only the scoped pages are present, and their
    ids still say where each one sits in the PDF.
    """
    ids = page_ids if page_ids is not None else range(len(pages))
    return {
        "id": "/0",
        "block_type": "Document",
        "children": [
            {
                "id": f"/page/{index}/Page/1",
                "block_type": "Page",
                "children": blocks,
            }
            for index, blocks in zip(ids, pages, strict=True)
        ],
    }


@pytest.fixture
def marker_document() -> dict[str, Any]:
    """A dump holding every kind the mapping knows, plus one it does not."""
    return marker_dump(
        [
            [
                marker_block(
                    "SectionHeader",
                    "<h2>1.1 Chuyển động cơ</h2>",
                    page=0,
                    number=1,
                    hierarchy={"2": "/page/0/SectionHeader/1"},
                ),
                marker_block("Text", "<p>Một đoạn văn bản.</p>", page=0, number=2),
                marker_block("ListGroup", "<ul><li>a</li><li>b</li></ul>", page=0, number=3),
                marker_block(
                    "Table",
                    "<table><tbody><tr><td>x</td><td>y</td></tr>"
                    "<tr><td>1</td><td>2</td></tr></tbody></table>",
                    page=0,
                    number=4,
                ),
                marker_block(
                    "Picture",
                    '<img src="data:image/png;base64,QUJD"/>',
                    page=0,
                    number=5,
                ),
                marker_block("PageHeader", "7", page=0, number=6),
                marker_block("PageFooter", "Chương 1", page=0, number=7),
            ],
            [
                marker_block("SectionHeader", "<h3>Bài tập</h3>", page=1, number=1),
                marker_block("SectionHeader", "Tóm tắt", page=1, number=2),
                marker_block("Caption", "<p>Hình 1.1: minh hoạ</p>", page=1, number=3),
                marker_block("Equation", "<p><math>x^2</math></p>", page=1, number=4),
                marker_block("Text", "<p></p>", page=1, number=5),
                marker_block("Weird", "<p>giữ nguyên</p>", page=1, number=6),
            ],
        ]
    )


def convert_dump(dump: dict[str, Any], **overrides: Any) -> EngineArtifact:
    """Convert a dump with Marker's own mapping, defaulting to one document id."""
    kwargs: dict[str, Any] = {"doc_id": "vat-li-12"}
    kwargs.update(overrides)
    return load_engine("marker").convert(dump, **kwargs)


# --- registry and vocabulary ----------------------------------------------------


def test_every_registered_engine_loads_and_answers_to_its_own_id() -> None:
    for engine_id in ENGINE_IDS:
        module = load_engine(engine_id)
        assert engine_id == module.ENGINE_ID
        assert callable(module.probe)
        assert callable(module.extract)


def test_load_engine_accepts_the_cli_spelling_and_rejects_unknown_ids() -> None:
    assert load_engine("MARKER").ENGINE_ID == "marker"
    with pytest.raises(ValueError, match="unknown engine_id"):
        load_engine("tesseract")


def test_normalized_kinds_cover_the_runtime_vocabulary_and_every_mapping() -> None:
    assert set(NORMALIZED_KINDS) >= RUNTIME_BLOCK_KINDS
    assert set(ARTIFACT_ONLY_KINDS) <= set(NORMALIZED_KINDS)
    marker_map = load_engine("marker").MARKER_KIND_MAP
    docling_map = load_engine("docling").DOCLING_KIND_MAP
    assert set(marker_map.values()) <= set(NORMALIZED_KINDS)
    assert set(docling_map.values()) <= set(NORMALIZED_KINDS)


def test_marker_mapping_recognizes_exactly_what_the_builder_recognizes() -> None:
    """A block this adapter converts is a block the text-first builder can classify."""
    assert set(load_engine("marker").MARKER_KIND_MAP) == set(MARKER_BLOCK_KINDS)


def test_importing_the_adapters_imports_no_engine_that_owns_weights() -> None:
    """Import purity, checked in a fresh interpreter: probes must not load engines."""
    code = (
        "import sys\n"
        "from research.ocr_pipeline.engines import load_engine\n"
        "for engine_id in ('vision', 'marker', 'docling', 'native'):\n"
        "    load_engine(engine_id)\n"
        f"print(','.join(sorted(m for m in {DETECTABLE_KINDS!r} if m in sys.modules)))\n"
    )
    finished = subprocess.run(
        [sys.executable, "-c", code],
        cwd=REPO_ROOT,
        capture_output=True,
        text=True,
        timeout=120,
        check=True,
    )
    assert finished.stdout.strip() == ""


def test_importing_the_application_pulls_no_optional_engine() -> None:
    """The runtime, the research runner and the worker import without marker or docling."""
    code = (
        "import sys\n"
        "import research.ocr_pipeline.runner\n"
        "import research.ocr_pipeline.engines.worker\n"
        "import workflows.ingest_file.run\n"
        f"print(','.join(sorted(m for m in {DETECTABLE_KINDS!r} if m in sys.modules)))\n"
    )
    finished = subprocess.run(
        [sys.executable, "-c", code],
        cwd=REPO_ROOT,
        capture_output=True,
        text=True,
        timeout=120,
        check=True,
    )
    assert finished.stdout.strip() == ""


# --- the neutral artifact's own rules -------------------------------------------


def test_a_block_rejects_a_kind_outside_the_neutral_vocabulary() -> None:
    with pytest.raises(ValueError, match="unknown normalized_kind"):
        EngineBlock(kind="Text", normalized_kind="paragraph", text="x")


@pytest.mark.parametrize(
    ("block", "message"),
    [
        ({"kind": "heading", "normalized_kind": "heading", "text": "x"}, "heading requires depth"),
        ({"kind": "Text", "normalized_kind": "p", "text": "   "}, "text must not be blank"),
        ({"kind": "Text", "normalized_kind": "p", "text": "x", "group": 0}, "group must be >= 1"),
        (
            {"kind": "Text", "normalized_kind": "p", "text": "x", "bbox": (1.0, 2.0)},
            "bbox must have four",
        ),
    ],
)
def test_a_block_validates_its_own_fields(block: dict[str, Any], message: str) -> None:
    with pytest.raises(ValueError, match=message):
        EngineBlock(**block)


def test_a_figure_is_the_one_kind_allowed_to_carry_no_text() -> None:
    assert EngineBlock(kind="Picture", normalized_kind="figure", text="").text == ""


def test_a_page_accepts_only_a_status_a_returned_artifact_may_carry() -> None:
    EnginePage(page_index=0, request_page=1, status="unsupported_modality")
    with pytest.raises(ValueError, match="unknown page status"):
        EnginePage(page_index=0, request_page=1, status="not_available")
    with pytest.raises(ValueError, match="usage must carry exactly"):
        EnginePage(page_index=0, request_page=1, usage={"attempts": 1})


def test_an_artifact_requires_ascending_pages_and_reports_its_span() -> None:
    pages = (
        EnginePage(page_index=1, request_page=2),
        EnginePage(page_index=3, request_page=4),
    )
    artifact = EngineArtifact(engine_id="native", engine_version=None, doc_id="d", pages=pages)
    assert artifact.physical_pages == (1, 3)
    assert artifact.artifact_version == ARTIFACT_VERSION
    with pytest.raises(ValueError, match="strictly increasing"):
        EngineArtifact(
            engine_id="native",
            engine_version=None,
            doc_id="d",
            pages=(EnginePage(page_index=2, request_page=3), pages[0]),
        )
    with pytest.raises(ValueError, match="at least one page"):
        EngineArtifact(engine_id="native", engine_version=None, doc_id="d", pages=())


def test_content_pages_translates_what_the_runtime_vocabulary_cannot_replay() -> None:
    page = EnginePage(
        page_index=0,
        request_page=1,
        running="Chương 1",
        printed_folio="7",
        blocks=(
            EngineBlock(kind="Heading", normalized_kind="heading", text="Bài 1", depth=1),
            EngineBlock(kind="Caption", normalized_kind="caption", text="Hình 1.1"),
            EngineBlock(kind="Weird", normalized_kind="unmapped", text="giữ nguyên"),
            EngineBlock(kind="PageHeader", normalized_kind="page_header", text="7"),
        ),
    )
    (translated,) = content_pages((page,))
    assert [block.normalized_kind for block in translated.blocks] == ["heading", "p", "p"]
    assert [block.text for block in translated.blocks] == ["Bài 1", "Hình 1.1", "giữ nguyên"]
    # The furniture is gone from the page but still lifted onto it.
    assert (translated.running, translated.printed_folio) == ("Chương 1", "7")


def test_count_block_kinds_is_sorted_and_lift_running_splits_folio_from_head() -> None:
    page = EnginePage(
        page_index=0,
        request_page=1,
        blocks=(
            EngineBlock(kind="Text", normalized_kind="p", text="x"),
            EngineBlock(kind="Text", normalized_kind="p", text="y"),
            EngineBlock(kind="Table", normalized_kind="table", text="a"),
        ),
    )
    assert count_block_kinds((page,)) == {"p": 2, "table": 1}
    header = EngineBlock(kind="PageHeader", normalized_kind="page_header", text=" 12 ")
    footer = EngineBlock(kind="PageFooter", normalized_kind="page_footer", text="Vật lí 12")
    assert lift_running((footer, header)) == ("Vật lí 12", "12")
    assert lift_running((EngineBlock(kind="Text", normalized_kind="p", text="x"),)) == ("", None)


# --- the Marker mapping: one hand-written dump ----------------------------------


@pytest.mark.parametrize(
    ("markup", "expected"),
    [
        ('<h2 id="x">1.1 Chuyển động</h2>', "1.1 Chuyển động"),
        ("<p>Một đoạn &amp; hai</p>", "Một đoạn & hai"),
        ("<p>dòng một<br/>dòng hai</p>", "dòng một\ndòng hai"),
        ("<ul><li>a</li><li>b</li></ul>", "a\nb"),
        ("<table><tbody><tr><td>x</td><td>y</td></tr></tbody></table>", "x | y"),
        ('<img src="data:image/png;base64,QUJD" alt="hình"/>', ""),
        (None, ""),
    ],
)
def test_html_text_keeps_the_visible_text_and_never_an_image_payload(
    markup: str | None, expected: str
) -> None:
    assert load_engine("marker").html_text(markup) == expected


def test_the_mapping_converts_every_kind_with_depth_running_head_and_folio(
    marker_document: dict[str, Any],
) -> None:
    artifact = convert_dump(marker_document, source_sha256="ab" * 32)

    assert artifact.engine_id == "marker"
    assert artifact.engine_version is None
    assert artifact.source_sha256 == "ab" * 32
    assert artifact.physical_pages == (0, 1)

    first, second = artifact.pages
    assert (first.request_page, second.request_page) == (1, 2)
    assert (first.running, first.printed_folio) == ("Chương 1", "7")
    assert [block.normalized_kind for block in first.blocks] == [
        "heading",
        "p",
        "list",
        "table",
        "figure",
        "page_header",
        "page_footer",
    ]
    assert first.blocks[0].kind == "SectionHeader"
    # The hierarchy's deepest key is the level Marker itself assigned.
    assert first.blocks[0].depth == 2
    assert first.blocks[0].marker is None
    assert first.blocks[2].text == "a\nb"
    assert first.blocks[3].text == "x | y\n1 | 2"
    assert first.blocks[4].source_id == "/page/0/Picture/5"
    assert first.blocks[4].bbox == (1.0, 2.0, 3.0, 4.0)

    # The h-tag decides the level when the hierarchy is empty, and v1's fallback
    # decides it when there is no tag either.
    assert [block.depth for block in second.blocks[:2]] == [3, FALLBACK_DEPTH]
    assert [block.normalized_kind for block in second.blocks[2:]] == [
        "caption",
        "equation",
        "unmapped",
    ]
    assert artifact.blocks_by_kind["caption"] == 1
    assert artifact.normalization["marker_depth_from_hierarchy"] == 1
    assert artifact.normalization["marker_depth_from_htag"] == 1
    assert artifact.normalization["marker_depth_fallback"] == 1
    assert artifact.normalization["marker_empty_blocks"] == 1
    assert artifact.normalization["marker_unmapped_kind:Weird"] == 1
    assert (artifact.normalization["marker_pages"], artifact.normalization["marker_blocks"]) == (
        2,
        12,
    )


def test_a_marker_dump_never_carries_an_image_payload_into_its_text(
    marker_document: dict[str, Any],
) -> None:
    artifact = convert_dump(marker_document)
    dumped = json.dumps([dataclasses.asdict(page) for page in artifact.pages])
    assert "base64" not in dumped
    assert "data:image" not in dumped
    assert "QUJD" not in dumped


def test_page_scope_follows_the_physical_index_in_the_block_id(
    marker_document: dict[str, Any],
) -> None:
    artifact = convert_dump(marker_document, page_offset=3, page_indices=[1])
    assert artifact.physical_pages == (1,)
    assert artifact.pages[0].request_page == 5
    assert artifact.render_page_offset == 3


def test_a_scoped_dump_maps_by_the_physical_index_its_ids_carry() -> None:
    """A live run scoped to a page range dumps only those pages, ids intact."""
    dump = marker_dump(
        [[marker_block("Text", f"<p>trang {index}</p>", page=index)] for index in (5, 6, 7)],
        page_ids=[5, 6, 7],
    )
    artifact = convert_dump(dump, page_indices=[5, 6, 7])
    assert artifact.physical_pages == (5, 6, 7)
    assert [block.text for page in artifact.pages for block in page.blocks] == [
        "trang 5",
        "trang 6",
        "trang 7",
    ]
    # A narrower scope names the same physical pages, not positions 0..2 of the dump.
    assert convert_dump(dump, page_indices=[6]).physical_pages == (6,)


@pytest.mark.parametrize(
    ("page_indices", "message"),
    [
        ([0, 2], "page index 2 is not a page in the dump"),
        ([-1], "page index must be an int >= 0"),
    ],
)
def test_a_page_scope_the_dump_cannot_answer_is_refused(
    marker_document: dict[str, Any], page_indices: list[int], message: str
) -> None:
    with pytest.raises(ValueError, match=re.escape(message)):
        convert_dump(marker_document, page_indices=page_indices)


def test_a_page_without_a_marker_id_falls_back_to_its_position() -> None:
    dump = marker_dump([[marker_block("Text", "<p>a</p>")]])
    dump["children"][0]["id"] = "rewritten"
    assert convert_dump(dump).physical_pages == (0,)


@pytest.mark.parametrize(
    "dump",
    [
        {"block_type": "Document", "children": [{"block_type": "Table", "children": []}]},
        {"block_type": "Document", "children": []},
        {"block_type": "Document"},
    ],
    ids=["non-page-root-child", "no-pages", "no-children-list"],
)
def test_a_dump_that_is_not_page_structured_is_refused(dump: dict[str, Any]) -> None:
    with pytest.raises(ValueError):
        convert_dump(dump)


def test_a_block_without_a_kind_is_refused_instead_of_guessed() -> None:
    dump = marker_dump([[{"id": "/page/0/Text/1", "html": "<p>x</p>"}]])
    with pytest.raises(ValueError, match="no block_type"):
        convert_dump(dump)


def test_a_marker_dump_builds_through_the_text_first_marker_builder(
    marker_document: dict[str, Any],
) -> None:
    artifact = convert_dump(marker_document)
    compatibility = marker_compatibility(artifact.pages)
    assert compatibility.compatible is True
    # The one kind the builder does not know is named rather than hidden, and the
    # dump is still recognized as Marker output because the other eleven are.
    assert compatibility.unknown_kinds == ("Weird",)
    assert compatibility.recognized == 11

    result = build_marker_structure(artifact.pages, "Vật lí 12")
    assert result.pages_processed == 2
    # Every heading kept the converter's decision instead of guessing from words.
    assert result.counters["marker_depth_from_artifact"] == 3
    assert "marker_depth_fallback" not in result.counters
    # The one ListGroup became one detail per item.
    assert result.counters["marker_list_items"] == 2
    assert result.counters["marker_unknown_kind:Weird"] == 1
    assert result.counters["marker_dropped_blocks"] == 2


def test_the_marker_builder_refuses_an_artifact_that_is_not_marker_output() -> None:
    pages = (
        EnginePage(
            page_index=0,
            request_page=1,
            blocks=(EngineBlock(kind="text", normalized_kind="p", text="một đoạn"),),
        ),
    )
    with pytest.raises(MarkerStructureError, match="no raw Marker block kind"):
        build_marker_structure(pages, "Vật lí 12")


def test_running_heads_and_captions_are_why_content_pages_exists(
    marker_document: dict[str, Any],
) -> None:
    artifact = convert_dump(marker_document)
    # The runtime vocabulary has no caption and no running head, so the raw
    # artifact cannot be replayed as a vision tree.
    with pytest.raises(ReplayError, match="not a runtime vision kind"):
        build_atomic(artifact.pages, "Vật lí 12")

    result = build_atomic(content_pages(artifact.pages), "Vật lí 12")
    assert result.pages_processed == 2
    assert result.blocks_by_kind["p"] >= 1


def test_the_live_marker_adapter_reads_an_injected_converters_dump(
    scanned_pdf: Path, marker_document: dict[str, Any]
) -> None:
    marker = load_engine("marker")

    class Converter:
        """A converter that answers with a dump instead of running any model."""

        def __call__(self, path: str) -> dict[str, Any]:
            assert Path(path) == scanned_pdf
            return marker_document

    artifact = marker.extract(scanned_pdf, doc_id="vat-li-12", converter=Converter())
    assert artifact.engine_id == "marker"
    assert artifact.engine_version == marker.marker_version()
    assert artifact.pdf_sha256 == marker.sha256_file(scanned_pdf)
    assert artifact.physical_pages == (0, 1)


def test_the_live_marker_adapter_scopes_and_verifies_an_injected_converters_dump(
    scanned_pdf: Path, marker_document: dict[str, Any]
) -> None:
    """A caller's converter is not trusted to have honored the scope it was given."""
    marker = load_engine("marker")

    class Converter:
        def __call__(self, path: str) -> dict[str, Any]:
            return marker_document

    artifact = marker.extract(scanned_pdf, doc_id="d", converter=Converter(), page_indices=[1])
    assert artifact.physical_pages == (1,)
    with pytest.raises(ValueError, match="page index 2 is not a page in the dump"):
        marker.extract(scanned_pdf, doc_id="d", converter=Converter(), page_indices=[2])


def test_the_live_marker_adapter_reads_a_pydantic_result_and_refuses_anything_else(
    scanned_pdf: Path, marker_document: dict[str, Any]
) -> None:
    marker = load_engine("marker")

    class Result:
        def model_dump(self, *, mode: str) -> dict[str, Any]:
            assert mode == "json"
            return marker_document

    class Converter:
        def __call__(self, path: str) -> Result:
            return Result()

    assert marker.extract(scanned_pdf, doc_id="d", converter=Converter()).physical_pages == (0, 1)
    with pytest.raises(TypeError, match="must be a mapping or a pydantic model"):
        marker.render_document(lambda _: object(), scanned_pdf)


def test_marker_probe_reads_metadata_and_build_converter_refuses_a_missing_import(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    marker = load_engine("marker")
    probe = marker.probe()
    assert probe.engine_id == "marker"
    assert probe.available is (marker.marker_version() is not None)
    if probe.available:
        assert "weights" in probe.detail
    else:
        assert "not installed" in probe.detail

    # The failure path of the lazy import, without ever downloading a weight.
    monkeypatch.setitem(sys.modules, "marker.converters.pdf", None)
    with pytest.raises(EngineNotAvailableError, match="marker:"):
        marker.build_converter()


# --- Docling: documented API, synthetic document, honest about the missing install


@dataclass
class FakeProv:
    """One provenance entry, as Docling documents it."""

    page_no: int
    bbox: Any = None


class FakeBBox:
    def __init__(self, *values: float) -> None:
        self._values = values

    def as_tuple(self) -> tuple[float, ...]:
        return self._values


@dataclass
class FakeItem:
    """One Docling item: a label, some text and where it came from."""

    label: Any = "text"
    text: str = ""
    prov: tuple[FakeProv, ...] = ()
    self_ref: str | None = None
    markdown: str | None = None

    def export_to_markdown(self) -> str:
        return self.markdown or ""


class DocumentTable:
    """A table whose exporter takes the document, as Docling's own does."""

    def __init__(self, seen: list[Any]) -> None:
        self.label = "table"
        self.text = ""
        self.prov = (FakeProv(1),)
        self.self_ref = "#/tables/0"
        self._seen = seen

    def export_to_markdown(self, document: Any) -> str:
        self._seen.append(document)
        return "| a | b |"


class OptionalDocumentTable:
    """A table whose exporter takes the document optionally, as Docling's own does."""

    def __init__(self, seen: list[Any]) -> None:
        self.label = "table"
        self.text = ""
        self.prov = (FakeProv(1),)
        self.self_ref = "#/tables/1"
        self._seen = seen

    def export_to_markdown(self, document: Any = None) -> str:
        self._seen.append(document)
        return "| c | d |"


class BareTable:
    """A table whose exporter takes nothing at all."""

    label = "table"
    text = ""
    prov = (FakeProv(1),)
    self_ref = "#/tables/2"

    def export_to_markdown(self) -> str:
        return "| e | f |"


class BrokenTable:
    """A table whose exporter raises, as a renderer with nothing to say would."""

    label = "table"
    text = ""
    prov = (FakeProv(1),)
    self_ref = "#/tables/0"

    def export_to_markdown(self) -> str:
        raise RuntimeError("docling could not render this table")


class FakeDocument:
    """A Docling document: a page table keyed by page number, and an item stream."""

    def __init__(self, pages: Sequence[int], items: list[tuple[Any, int]]) -> None:
        self.pages: dict[int, Any] = {number: object() for number in pages}
        self._items = items

    def iterate_items(self) -> list[tuple[Any, int]]:
        return list(self._items)


def fake_document() -> FakeDocument:
    """A three-page document holding every label the mapping knows, plus one it does not."""
    return FakeDocument(
        (1, 2, 3),
        [
            (
                FakeItem(
                    "title",
                    "Chuyển động cơ",
                    (FakeProv(1, FakeBBox(1.0, 2.0, 3.0, 4.0)),),
                    "#/texts/0",
                ),
                1,
            ),
            (FakeItem("text", "Một đoạn.", (FakeProv(1),), "#/texts/1"), 1),
            (FakeItem("list_item", "- ý một", (FakeProv(1),), "#/texts/2"), 1),
            (
                FakeItem("table", "", (FakeProv(1),), "#/tables/0", markdown="| a | b |"),
                1,
            ),
            (FakeItem("picture", "", (FakeProv(1),), "#/pictures/0"), 1),
            (FakeItem("page_header", "Giáo dục", (FakeProv(1),), "#/texts/3"), 1),
            (FakeItem("page_footer", "7", (FakeProv(1),), "#/texts/4"), 1),
            (FakeItem("caption", "Hình 1.1", (FakeProv(3),), "#/texts/5"), 1),
            (FakeItem("formula", "E = mc^2", (FakeProv(3),), "#/texts/6"), 1),
            (FakeItem("mystery_label", "nhãn mới", (FakeProv(3),), "#/texts/7"), 1),
            (FakeItem("text", "   ", (FakeProv(3),), "#/texts/8"), 1),
        ],
    )


def fake_scoped_document(page_numbers: Sequence[int]) -> FakeDocument:
    """The document a conversion of ``page_numbers`` answers with, in Docling's numbering."""
    return FakeDocument(
        page_numbers,
        [
            (FakeItem("text", f"trang {number}", (FakeProv(number),), f"#/texts/{number}"), 1)
            for number in page_numbers
        ],
    )


def docling_result(document: FakeDocument, *, status: Any = None) -> Any:
    """A ``ConversionResult`` as Docling documents it: a document and a status."""
    values: dict[str, Any] = {"document": document}
    if status is not None:
        values["status"] = status
    return type("Result", (), values)()


class ConversionStatus(Enum):
    """Docling's ``ConversionStatus``, which the adapter must read by value."""

    SUCCESS = "success"
    PARTIAL_SUCCESS = "partial_success"
    FAILURE = "failure"


def test_docling_probe_reports_exactly_what_the_environment_has() -> None:
    docling = load_engine("docling")
    probe = docling.probe()
    assert probe.engine_id == "docling"
    assert probe.available is (docling.docling_version() is not None)
    if not probe.available:
        assert probe.version is None
        assert "not installed" in probe.detail


def test_a_docling_document_maps_labels_depths_and_every_edge_case() -> None:
    artifact = load_engine("docling").convert_document(
        fake_document(), doc_id="vat-li-12", engine_version="test"
    )
    assert artifact.engine_id == "docling"
    # The note is the only place a reader learns the mapping's limits, so it has to
    # keep naming both halves: what real runs reached, and the gap that remains --
    # reaching a label is not the same claim as having checked it against gold.
    assert artifact.notes
    assert "nothing unmapped" in artifact.notes[0]
    assert "unmeasured" in artifact.notes[0]
    assert artifact.physical_pages == (0, 1, 2)

    first = artifact.pages[0]
    assert [block.normalized_kind for block in first.blocks] == [
        "heading",
        "p",
        "list",
        "table",
        "figure",
        "page_header",
        "page_footer",
    ]
    assert first.blocks[0].depth == 1
    assert first.blocks[0].kind == "title"
    assert first.blocks[0].bbox == (1.0, 2.0, 3.0, 4.0)
    assert first.blocks[0].source_id == "#/texts/0"
    # A table has no plain text, so its markdown rendering stands in for one.
    assert first.blocks[3].text == "| a | b |"
    assert (first.running, first.printed_folio) == ("Giáo dục", "7")

    assert [block.normalized_kind for block in artifact.pages[2].blocks] == [
        "caption",
        "equation",
        "unmapped",
    ]
    assert artifact.normalization["docling_unmapped_blocks"] == 1
    assert artifact.normalization["docling_unmapped_kind:mystery_label"] == 1
    assert artifact.normalization["docling_empty_blocks"] == 1


def test_docling_refuses_an_item_it_cannot_place_instead_of_calling_it_page_0() -> None:
    """An item whose provenance names no page is not evidence that it came from page 1."""
    docling = load_engine("docling")
    nameless = FakeDocument((1,), [(FakeItem("text", "không có nguồn", (), "#/texts/9"), 1)])
    with pytest.raises(ValueError, match=re.escape("refusing to report it as page 0")):
        docling.convert_document(nameless, doc_id="d")

    # A provenance entry that names no page is the same fact, not page -1 or page 0.
    unnumbered = FakeDocument((1,), [(FakeItem("text", "x", (FakeProv(0),), "#/texts/1"), 1)])
    with pytest.raises(ValueError, match="no provenance page"):
        docling.convert_document(unnumbered, doc_id="d")


def test_docling_calls_the_table_exporter_with_the_arguments_its_signature_declares() -> None:
    """Docling's exporter takes the document; one that takes nothing must not be handed it."""
    seen: list[Any] = []
    document = FakeDocument(
        (1,), [(DocumentTable(seen), 1), (OptionalDocumentTable(seen), 1), (BareTable(), 1)]
    )

    artifact = load_engine("docling").convert_document(document, doc_id="d")

    assert [block.text for block in artifact.pages[0].blocks] == [
        "| a | b |",
        "| c | d |",
        "| e | f |",
    ]
    assert seen == [document, document], (
        "an exporter that takes the document is handed that document"
    )


def test_docling_does_not_swallow_a_table_the_exporter_could_not_render() -> None:
    document = FakeDocument((1,), [(BrokenTable(), 1)])
    with pytest.raises(RuntimeError, match="could not render this table"):
        load_engine("docling").convert_document(document, doc_id="d")


def test_docling_maps_a_scoped_documents_pages_back_to_physical_indices() -> None:
    """A selected-page conversion is joined back by physical index, not by position."""
    artifact = load_engine("docling").convert_document(
        fake_scoped_document((1, 2)), doc_id="d", page_offset=4, page_map=[2, 5]
    )
    assert artifact.physical_pages == (2, 5)
    assert [page.request_page for page in artifact.pages] == [7, 10]
    assert [block.text for page in artifact.pages for block in page.blocks] == [
        "trang 1",
        "trang 2",
    ]


def test_docling_asks_a_converter_that_declares_a_page_range_for_one_run_at_a_time(
    tmp_path: Path,
) -> None:
    """A declared page_range is a real scope: contiguous runs, joined back by index."""
    docling = load_engine("docling")
    path = tmp_path / "three.pdf"
    path.write_bytes(make_pdf(3))
    calls: list[tuple[Path, Any]] = []

    class Converter:
        """A converter shaped like Docling's own, whose ``convert`` declares page_range."""

        def convert(self, source: str, page_range: Any = None) -> Any:
            calls.append((Path(source), page_range))
            return docling_result(fake_scoped_document(range(page_range[0], page_range[1] + 1)))

    artifact = docling.extract(path, doc_id="d", page_indices=[0, 1, 2], converter=Converter())
    assert calls == [(path, (1, 3))]
    assert artifact.physical_pages == (0, 1, 2)
    assert artifact.normalization["docling_page_runs"] == 1
    assert artifact.normalization["docling_context_resets"] == 0

    calls.clear()
    artifact = docling.extract(path, doc_id="d", page_indices=[0, 2], converter=Converter())
    assert calls == [(path, (1, 1)), (path, (3, 3))], "a gap becomes two runs"
    assert artifact.physical_pages == (0, 2)
    assert [block.text for page in artifact.pages for block in page.blocks] == [
        "trang 1",
        "trang 3",
    ]
    assert artifact.normalization["docling_page_runs"] == 2
    assert artifact.normalization["docling_context_resets"] == 1
    assert any("contiguous" in note for note in artifact.notes)


def test_docling_builds_a_selected_page_pdf_for_a_converter_that_cannot_scope(
    tmp_path: Path,
) -> None:
    """A converter with no page-range argument never sees the whole PDF."""
    docling = load_engine("docling")
    path = tmp_path / "three.pdf"
    path.write_bytes(make_pdf(3, text_pages=(0, 1, 2)))
    seen: list[tuple[Path, int]] = []

    class Converter:
        """A converter shaped like a Docling version whose ``convert`` takes no page_range."""

        def convert(self, source: str) -> Any:
            selected = pypdfium2.PdfDocument(source)
            try:
                seen.append((Path(source), len(selected)))
            finally:
                selected.close()
            return docling_result(fake_scoped_document((1, 2)))

    artifact = docling.extract(path, doc_id="d", page_indices=[0, 2], converter=Converter())

    assert seen[0][0] != path, "the engine is handed a PDF built for the selection"
    assert seen[0][1] == 2, "and that PDF holds only the selected pages"
    assert not seen[0][0].exists(), "the selected-page copy is temporary"
    # The copy is numbered 1..2; the artifact keeps the physical indices it was built from.
    assert artifact.physical_pages == (0, 2)
    assert [block.text for page in artifact.pages for block in page.blocks] == [
        "trang 1",
        "trang 2",
    ]
    assert artifact.normalization["docling_page_runs"] == 1
    assert artifact.normalization["docling_context_resets"] == 0
    assert any("selected-page PDF" in note for note in artifact.notes)
    assert artifact.pdf_sha256 == hashlib.sha256(path.read_bytes()).hexdigest()


def test_docling_refuses_a_conversion_that_did_not_honour_the_scope(tmp_path: Path) -> None:
    """A scoped run is never reported as the pages it was not asked for."""
    path = tmp_path / "three.pdf"
    path.write_bytes(make_pdf(3))

    class RangeConverter:
        """A converter that declares page_range and ignores it."""

        def convert(self, source: str, page_range: Any = None) -> Any:
            return docling_result(fake_scoped_document((1, 2, 3)))

    with pytest.raises(ValueError, match=re.escape("the page scope was not honoured")):
        load_engine("docling").extract(
            path, doc_id="d", page_indices=[1], converter=RangeConverter()
        )

    class WideConverter:
        """A converter without page_range that answers with more pages than it was given."""

        def convert(self, source: str) -> Any:
            return docling_result(fake_scoped_document((1, 2, 3)))

    with pytest.raises(ValueError, match=re.escape("only 1 pages were converted")):
        load_engine("docling").extract(
            path, doc_id="d", page_indices=[1], converter=WideConverter()
        )


def test_docling_reports_a_partial_conversion_page_by_page(tmp_path: Path) -> None:
    """A page Docling did not produce is unsupported, not a page with no content."""
    path = tmp_path / "two.pdf"
    path.write_bytes(make_pdf(2))

    class Converter:
        def convert(self, source: str) -> Any:
            return docling_result(
                fake_scoped_document((1,)), status=ConversionStatus.PARTIAL_SUCCESS
            )

    artifact = load_engine("docling").extract(
        path, doc_id="d", page_indices=[0, 1], converter=Converter()
    )

    assert artifact.physical_pages == (0, 1)
    assert artifact.pages[0].status == "ok"
    assert artifact.pages[1].status == "unsupported_modality"
    assert artifact.pages[1].blocks == ()
    assert artifact.normalization["docling_pages_not_converted"] == 1
    assert artifact.normalization["docling_partial_conversion"] == 1
    assert any("partial" in note for note in artifact.notes)


def test_docling_fails_the_run_when_the_conversion_failed(tmp_path: Path) -> None:
    path = tmp_path / "one.pdf"
    path.write_bytes(make_pdf(1))
    docling = load_engine("docling")

    class Converter:
        def convert(self, source: str) -> Any:
            return docling_result(fake_scoped_document((1,)), status=ConversionStatus.FAILURE)

    with pytest.raises(docling.DoclingConversionError, match="failure"):
        docling.extract(path, doc_id="d", page_indices=[0], converter=Converter())


def test_docling_refuses_a_selection_the_pdf_cannot_answer(tmp_path: Path) -> None:
    path = tmp_path / "two.pdf"
    path.write_bytes(make_pdf(2))
    docling = load_engine("docling")

    class Converter:
        """A converter that must never be reached by a refused selection."""

        def convert(self, source: str, page_range: Any = None) -> Any:
            raise AssertionError("a refused selection must not reach the engine")

    with pytest.raises(ValueError, match="outside the PDF"):
        docling.extract(path, doc_id="d", page_indices=[2], converter=Converter())
    with pytest.raises(ValueError, match="selected twice"):
        docling.extract(path, doc_id="d", page_indices=[0, 0], converter=Converter())
    with pytest.raises(ValueError, match="selection is empty"):
        docling.extract(path, doc_id="d", page_indices=[], converter=Converter())


def test_docling_extract_reports_a_missing_install_instead_of_an_empty_run(
    scanned_pdf: Path,
) -> None:
    docling = load_engine("docling")

    class Converter:
        def convert(self, source: str) -> Any:
            assert Path(source) == scanned_pdf
            return docling_result(fake_document())

    assert docling.extract(scanned_pdf, doc_id="d", converter=Converter()).physical_pages == (
        0,
        1,
        2,
    )
    assert (
        docling.extract(scanned_pdf, doc_id="d", converter=Converter()).pdf_sha256
        == hashlib.sha256(scanned_pdf.read_bytes()).hexdigest()
    )
    if docling.docling_version() is None:
        with pytest.raises(EngineNotAvailableError, match="docling:"):
            docling.extract(scanned_pdf, doc_id="d")


# --- native PDF text layer ------------------------------------------------------


def test_paragraphs_join_the_lines_of_one_paragraph_and_split_at_blank_lines() -> None:
    native = load_engine("native")
    assert native.paragraphs("dòng một\ndòng hai\n\ndòng ba\n") == [
        "dòng một dòng hai",
        "dòng ba",
    ]
    assert native.paragraphs("\n\n") == []


def test_native_reads_a_real_text_layer_and_refuses_a_page_that_has_none(
    scanned_pdf: Path,
) -> None:
    artifact = load_engine("native").extract(scanned_pdf, doc_id="vat-li-12")
    assert artifact.engine_id == "native"
    assert artifact.pdf_sha256 == hashlib.sha256(scanned_pdf.read_bytes()).hexdigest()
    assert artifact.notes and "no OCR fallback" in artifact.notes[0]

    text_page, scan_page = artifact.pages
    assert text_page.status == "ok"
    assert any("Alpha beta gamma" in block.text for block in text_page.blocks)
    assert all(block.normalized_kind == "p" for block in text_page.blocks)
    # The scan is a fact about the page, not a hole: no blocks, explicit status.
    assert scan_page.status == "unsupported_modality"
    assert scan_page.blocks == ()
    assert artifact.normalization["native_pages_without_text"] == 1
    assert artifact.blocks_by_kind == {"p": len(text_page.blocks)}


def test_native_reports_mixed_coverage_page_by_page(tmp_path: Path) -> None:
    """A mixed document is neither all-digital nor all-scan: each page says which it is."""
    path = tmp_path / "mixed.pdf"
    path.write_bytes(make_pdf(3, text_pages=(0, 2)))

    artifact = load_engine("native").extract(path, doc_id="d")

    assert artifact.physical_pages == (0, 1, 2)
    assert [page.status for page in artifact.pages] == ["ok", "unsupported_modality", "ok"]
    # The scan is a fact about that one page, and it is never filled in by OCR: the
    # digital pages keep their text and the scan keeps no blocks at all.
    assert [len(page.blocks) for page in artifact.pages] == [1, 0, 1]
    assert artifact.blocks_by_kind == {"p": 2}
    digital = artifact.pages[0].blocks + artifact.pages[2].blocks
    assert all("Alpha beta gamma" in block.text for block in digital)
    assert artifact.normalization["native_pages"] == 3
    assert artifact.normalization["native_pages_without_text"] == 1


def test_native_page_scope_keeps_the_physical_index_and_the_requested_number(
    scanned_pdf: Path,
) -> None:
    artifact = load_engine("native").extract(
        scanned_pdf, doc_id="d", page_indices=[1], page_offset=5
    )
    assert artifact.physical_pages == (1,)
    assert artifact.pages[0].request_page == 7
    assert artifact.pages[0].status == "unsupported_modality"
    with pytest.raises(ValueError, match="outside the PDF"):
        load_engine("native").extract(scanned_pdf, doc_id="d", page_indices=[9])


def test_native_reports_a_missing_pdfium_as_engine_not_available(
    scanned_pdf: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setitem(sys.modules, "pypdfium2", None)
    with pytest.raises(EngineNotAvailableError, match="native:"):
        load_engine("native").extract(scanned_pdf, doc_id="d")


# --- vision: the runtime client, mocked, never contacted ------------------------


def vision_config(*, api_key: str | None = "test-key") -> VisionOCRConfig:
    return VisionOCRConfig(
        api_base="https://vision.example.test/v1/chat/completions",
        model="test-vision",
        api_key=api_key,
        allowed_hosts=("vision.example.test",),
    )


class FakeVisionClient:
    """A client that answers with blocks and records what it was asked.

    ``pages`` is keyed by the number the client is asked for, not by the physical
    index, because that is what a request carries. ``continues`` names the pages
    whose answer claims to continue the group left open before them -- a claim the
    model can only make about a region it was actually shown.
    """

    def __init__(
        self,
        pages: dict[int, list[dict[str, Any]]] | None = None,
        *,
        continues: dict[int, bool] | None = None,
    ) -> None:
        self.pages = pages or {}
        self.continues = continues or {}
        self.page_numbers: list[int] = []
        self.states: list[dict[str, Any]] = []
        self.closed = False

    async def extract_page(
        self, png_bytes: bytes, *, page_number: int, state: dict[str, Any]
    ) -> PageExtraction:
        assert png_bytes[:8] == b"\x89PNG\r\n\x1a\n", "a rendered page must be a PNG"
        self.page_numbers.append(page_number)
        self.states.append(state)
        blocks = self.pages.get(page_number, [{"kind": "p", "text": "một đoạn", "group": 1}])
        return PageExtraction(
            page=VisionPage(
                page=page_number,
                blocks=blocks,
                continues_previous=self.continues.get(page_number, False),
            ),
            attempts=1,
            prompt_tokens=11,
            completion_tokens=7,
        )

    async def aclose(self) -> None:
        self.closed = True


def fake_settings(**overrides: Any) -> Any:
    """The settings fields ``VisionOCRConfig.from_settings`` reads, and nothing else."""

    class Settings:
        VISION_API_BASE = "https://vision.example.test/v1/chat/completions"
        VISION_API_KEY: str | None = "test-key"
        VISION_MODEL = "test-vision"
        VISION_ALLOWED_HOSTS = "vision.example.test"
        VISION_HEADERS: dict[str, str] = {}
        VISION_IMAGE_DETAIL = "high"
        VISION_TEMPERATURE = 0.0
        VISION_MAX_TOKENS = 8000
        VISION_REASONING_EFFORT: str | None = "none"
        VISION_MAX_BLOCKS_PER_PAGE = 400

    settings = Settings()
    for name, value in overrides.items():
        setattr(settings, name, value)
    return settings


def test_vision_probe_reports_a_missing_key_without_posting_anything() -> None:
    vision = load_engine("vision")
    without_key = vision.probe(vision_config(api_key=None))
    assert without_key.available is False
    assert "VISION_API_KEY" in without_key.detail
    assert "test-key" not in without_key.detail

    ready = vision.probe(vision_config())
    assert ready.available is True
    assert ready.version == "test-vision"
    # The host is not a secret; the URL and the key are.
    assert "vision.example.test" in ready.detail
    assert "/v1/chat/completions" not in ready.detail
    assert "test-key" not in ready.detail


def test_vision_probe_validates_the_settings_endpoint_before_a_run(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    vision = load_engine("vision")
    monkeypatch.setattr("config.settings.get_settings", lambda: fake_settings(VISION_API_KEY=None))
    assert vision.probe().available is False
    assert "VISION_API_KEY" in vision.probe().detail

    monkeypatch.setattr(
        "config.settings.get_settings",
        lambda: fake_settings(VISION_ALLOWED_HOSTS="other.example.test"),
    )
    unusable = vision.probe()
    assert unusable.available is False
    assert "VISION_ALLOWED_HOSTS" in unusable.detail


async def test_vision_extract_maps_pages_usage_and_the_prompt_digest(
    scanned_pdf: Path,
) -> None:
    vision = load_engine("vision")
    config = vision_config()
    # The offset is 3, so the first page is requested as page 4.
    client = FakeVisionClient(
        {
            4: [
                {
                    "kind": "heading",
                    "marker": None,
                    "depth": 1,
                    "group": 1,
                    "text": "Chuyển động cơ",
                }
            ]
        }
    )

    artifact = await vision.extract(
        scanned_pdf, doc_id="vat-li-12", config=config, client=client, page_offset=3
    )

    assert artifact.engine_id == "vision"
    assert artifact.engine_version == "test-vision"
    assert artifact.pdf_sha256 == hashlib.sha256(scanned_pdf.read_bytes()).hexdigest()
    assert (artifact.render_dpi, artifact.render_page_offset) == (200, 3)
    assert artifact.prompt_sha256 == sha256_bytes(PROMPT.encode("utf-8"))
    assert artifact.physical_pages == (0, 1)
    assert [page.request_page for page in artifact.pages] == [4, 5]

    first, second = artifact.pages
    assert first.blocks[0].kind == first.blocks[0].normalized_kind == "heading"
    assert first.blocks[0].depth == 1
    assert second.blocks[0].normalized_kind == "p"
    assert first.usage == {"attempts": 1, "prompt_tokens": 11, "completion_tokens": 7}
    assert artifact.normalization["vision_attempts"] == 2
    assert artifact.normalization["vision_prompt_tokens"] == 22
    assert artifact.normalization["vision_completion_tokens"] == 14
    assert artifact.normalization["vision_pages"] == 2
    assert artifact.notes and "aborts the run" in artifact.notes[0]

    # The prompt saw a real parser's state: empty before page 1, then the heading
    # page 1 opened, so the model can continue the same chain on page 2.
    assert client.states[0]["open_headings"] == []
    assert client.states[1]["open_headings"] == [
        {"marker": None, "depth": 1, "text": "Chuyển động cơ"}
    ]
    assert client.closed is False, "a client the caller owns is left open"


def _usage_handler(reported: dict[int, dict[str, Any]]) -> Any:
    """A transport handler that reports usage only for the pages in ``reported``."""

    def handler(request: httpx.Request) -> httpx.Response:
        payload = json.loads(request.content)
        text = payload["messages"][0]["content"][0]["text"]
        page_number = int(re.search(r"PAGE=(\d+)", text).group(1))
        body: dict[str, Any] = {
            "choices": [
                {
                    "finish_reason": "stop",
                    "message": {
                        "content": json.dumps(
                            {
                                "page": page_number,
                                "blocks": [
                                    {"kind": "p", "text": f"trang {page_number}", "group": 1}
                                ],
                            }
                        )
                    },
                }
            ]
        }
        if page_number in reported:
            body["usage"] = reported[page_number]
        return httpx.Response(200, json=body)

    return handler


async def test_vision_extract_keeps_an_unreported_usage_unknown(scanned_pdf: Path) -> None:
    """An endpoint that answers without usage leaves tokens unknown, not zero."""
    artifact = await load_engine("vision").extract(
        scanned_pdf,
        doc_id="d",
        config=vision_config(),
        transport=httpx.MockTransport(_usage_handler({})),
    )

    # The attempts were really spent, so they stay measured; the tokens were
    # never reported, so no page claims a usage it cannot price.
    assert [page.usage for page in artifact.pages] == [None, None]
    assert artifact.normalization["vision_attempts"] == 2
    assert artifact.normalization["vision_pages"] == 2
    # A total that skipped an unmeasured page would price that page at zero, so
    # the token counters are absent rather than zero.
    assert "vision_prompt_tokens" not in artifact.normalization
    assert "vision_completion_tokens" not in artifact.normalization


async def test_vision_extract_drops_token_totals_when_a_page_is_unmeasured(
    scanned_pdf: Path,
) -> None:
    """One unmeasured page makes the run's token totals unknown, not partial."""
    artifact = await load_engine("vision").extract(
        scanned_pdf,
        doc_id="d",
        config=vision_config(),
        transport=httpx.MockTransport(
            _usage_handler({1: {"prompt_tokens": 5, "completion_tokens": 3}})
        ),
    )

    assert [page.usage for page in artifact.pages] == [
        {"attempts": 1, "prompt_tokens": 5, "completion_tokens": 3},
        None,
    ]
    assert artifact.normalization["vision_attempts"] == 2
    assert "vision_prompt_tokens" not in artifact.normalization
    assert "vision_completion_tokens" not in artifact.normalization


async def test_vision_extract_bills_a_page_for_the_retry_it_bought(scanned_pdf: Path) -> None:
    """A 200 whose page was rejected is still a bill, and the retry is on top of it."""
    calls: list[int] = []

    def handler(request: httpx.Request) -> httpx.Response:
        payload = json.loads(request.content)
        text = payload["messages"][0]["content"][0]["text"]
        page_number = int(re.search(r"PAGE=(\d+)", text).group(1))
        calls.append(page_number)
        page = {
            "page": page_number,
            "blocks": [{"kind": "p", "text": f"trang {page_number}", "group": 1}],
        }
        if page_number == 1 and calls.count(1) == 1:
            # The first answer for page 1 was billed and rejected: the page
            # number inside it does not match the number that was asked for.
            page = {**page, "page": 99}
        return httpx.Response(
            200,
            json={
                "choices": [
                    {
                        "finish_reason": "stop",
                        "message": {"content": json.dumps(page)},
                    }
                ],
                "usage": {"prompt_tokens": 5, "completion_tokens": 3},
            },
        )

    delays: list[float] = []

    async def record_sleep(seconds: float) -> None:
        delays.append(seconds)

    artifact = await load_engine("vision").extract(
        scanned_pdf,
        doc_id="d",
        config=vision_config(),
        transport=httpx.MockTransport(handler),
        sleep=record_sleep,
    )

    assert delays == [1.0]
    assert calls == [1, 1, 2]
    assert artifact.pages[0].usage == {"attempts": 2, "prompt_tokens": 10, "completion_tokens": 6}
    assert artifact.normalization["vision_attempts"] == 3
    assert artifact.normalization["vision_prompt_tokens"] == 15
    assert artifact.normalization["vision_completion_tokens"] == 9


async def test_vision_extract_does_not_write_the_cached_share_the_endpoint_stated(
    scanned_pdf: Path,
) -> None:
    """A stated cache share is not carried into the run: the adapter drops it.

    The provider's cached prompt share is not tracked, so the artifact never
    claims a cache the report could mistake for a measured zero. The prompt and
    completion tokens the endpoint did state still travel, page by page and in
    the run's totals.
    """
    artifact = await load_engine("vision").extract(
        scanned_pdf,
        doc_id="d",
        config=vision_config(),
        transport=httpx.MockTransport(
            _usage_handler(
                {
                    1: {
                        "prompt_tokens": 5,
                        "completion_tokens": 3,
                        "prompt_tokens_details": {"cached_tokens": 4},
                    },
                    2: {
                        "prompt_tokens": 7,
                        "completion_tokens": 2,
                        "prompt_tokens_details": {"cached_tokens": 1},
                    },
                }
            )
        ),
    )

    # A page's usage schema is attempts and tokens only, and the run counters
    # follow it: the cached share the endpoint stated is not written anywhere.
    assert "vision_cached_tokens" not in artifact.normalization
    assert artifact.normalization["vision_prompt_tokens"] == 12
    assert artifact.normalization["vision_completion_tokens"] == 5
    assert [page.usage for page in artifact.pages] == [
        {"attempts": 1, "prompt_tokens": 5, "completion_tokens": 3},
        {"attempts": 1, "prompt_tokens": 7, "completion_tokens": 2},
    ]


async def test_vision_extract_drops_the_cached_total_when_a_page_states_none(
    scanned_pdf: Path,
) -> None:
    """A page that says nothing about the cache changes nothing: it is never recorded."""
    artifact = await load_engine("vision").extract(
        scanned_pdf,
        doc_id="d",
        config=vision_config(),
        transport=httpx.MockTransport(
            _usage_handler(
                {
                    1: {
                        "prompt_tokens": 5,
                        "completion_tokens": 3,
                        "prompt_tokens_details": {"cached_tokens": 4},
                    },
                    2: {"prompt_tokens": 7, "completion_tokens": 2},
                }
            )
        ),
    )

    # The run is never credited with a cache share -- a stated page and a silent
    # one are alike -- so the counter is absent rather than partial, and the
    # counters the endpoint did state still travel whole.
    assert "vision_cached_tokens" not in artifact.normalization
    assert artifact.normalization["vision_prompt_tokens"] == 12
    assert artifact.normalization["vision_completion_tokens"] == 5


async def test_vision_extract_keeps_the_key_and_the_url_out_of_the_artifact(
    scanned_pdf: Path,
) -> None:
    artifact = await load_engine("vision").extract(
        scanned_pdf, doc_id="d", config=vision_config(), client=FakeVisionClient()
    )
    assert artifact.public_config == {
        "model": "test-vision",
        "endpoint_host": "vision.example.test",
        "image_detail": "high",
        "temperature": 0.0,
        "max_tokens": 8000,
        "reasoning_effort": "none",
        "max_blocks_per_page": 400,
        "dpi": 200,
    }
    dumped = json.dumps(dataclasses.asdict(artifact))
    assert "test-key" not in dumped
    assert "/v1/chat/completions" not in dumped


async def test_vision_extract_drives_the_real_client_through_a_mock_transport(
    scanned_pdf: Path,
) -> None:
    vision = load_engine("vision")
    calls: list[int] = []
    seen: list[tuple[str, str]] = []

    def handler(request: httpx.Request) -> httpx.Response:
        payload = json.loads(request.content)
        content = payload["messages"][0]["content"]
        page_number = int(re.search(r"PAGE=(\d+)", content[0]["text"]).group(1))
        calls.append(page_number)
        seen.append((request.headers["authorization"], content[1]["image_url"]["url"][:22]))
        if calls.count(1) == 1:
            return httpx.Response(429, json={"error": "slow down"})
        body = {
            "choices": [
                {
                    "finish_reason": "stop",
                    "message": {
                        "content": json.dumps(
                            {
                                "page": page_number,
                                "blocks": [
                                    {"kind": "p", "text": f"trang {page_number}", "group": 1}
                                ],
                            }
                        )
                    },
                }
            ],
            "usage": {"prompt_tokens": 5, "completion_tokens": 3},
        }
        return httpx.Response(200, json=body)

    delays: list[float] = []

    async def record_sleep(seconds: float) -> None:
        delays.append(seconds)

    artifact = await vision.extract(
        scanned_pdf,
        doc_id="d",
        config=vision_config(),
        transport=httpx.MockTransport(handler),
        sleep=record_sleep,
    )

    # The retry waited the client's own backoff, on the injected seam, so the test
    # observes the wait instead of serving one.
    assert delays == [1.0]
    # Page 1 was retried after the 429, and the retry count reached the artifact.
    assert calls == [1, 1, 2]
    assert [page.usage for page in artifact.pages] == [
        {"attempts": 2, "prompt_tokens": 5, "completion_tokens": 3},
        {"attempts": 1, "prompt_tokens": 5, "completion_tokens": 3},
    ]
    assert artifact.normalization["vision_attempts"] == 3
    assert seen == [("Bearer test-key", "data:image/png;base64,")] * 3
    assert [page.blocks[0].text for page in artifact.pages] == ["trang 1", "trang 2"]


async def test_vision_a_failed_page_aborts_the_run_and_ownership_decides_the_close(
    scanned_pdf: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    vision = load_engine("vision")
    closed: list[str] = []

    async def record_close(self: Any) -> None:
        closed.append(type(self).__name__)

    monkeypatch.setattr(vision.VisionOCRClient, "aclose", record_close)

    def refuse(_request: httpx.Request) -> httpx.Response:
        return httpx.Response(400, json={"error": "bad request"})

    with pytest.raises(VisionOCRError, match="HTTP 400"):
        await vision.extract(
            scanned_pdf, doc_id="d", config=vision_config(), transport=httpx.MockTransport(refuse)
        )
    assert closed == ["VisionOCRClient"], "a client the adapter built is closed on failure"

    class FailingClient(FakeVisionClient):
        async def extract_page(
            self, png_bytes: bytes, *, page_number: int, state: dict[str, Any]
        ) -> PageExtraction:
            if page_number == 2:
                raise VisionOCRError(f"page {page_number}: HTTP 500")
            return await super().extract_page(png_bytes, page_number=page_number, state=state)

    failing = FailingClient()
    with pytest.raises(VisionOCRError, match="page 2"):
        await vision.extract(scanned_pdf, doc_id="d", config=vision_config(), client=failing)
    assert failing.closed is False
    assert closed == ["VisionOCRClient"], "an injected client is never closed by the adapter"


async def test_vision_extract_refuses_a_scope_that_renders_no_page(
    scanned_pdf: Path,
) -> None:
    with pytest.raises(ValueError, match="no renderable pages"):
        await load_engine("vision").extract(
            scanned_pdf,
            doc_id="d",
            config=vision_config(),
            client=FakeVisionClient(),
            page_indices=[],
        )


def test_the_vision_prompt_digest_is_the_one_the_runtime_sends() -> None:
    """The artifact records a digest, so a changed prompt cannot look like the same run."""
    assert len(sha256_bytes(PROMPT.encode("utf-8"))) == 64
    parser = VisionDocumentParser("Vật lí 12")
    assert parser.state == {"open_headings": [], "open_group": None, "open_group_tail": ""}


async def test_vision_resets_continuity_where_the_document_was_not_read(tmp_path: Path) -> None:
    """F08: a page after a region nobody read starts a new one, and the run says where.

    ``[0, 2]`` skips physical page 1. The answer for page 3 claims to continue the
    group left open on page 1, but page 3 was never shown that group as something
    to continue: the request after the gap is asked as a fresh start, the claim is
    refused, both pages are kept, and the reset is recorded. Replaying the artifact
    applies the same policy, so the replayed grouping matches the live one.
    """
    page_one = (
        "Chuyển động cơ học là sự thay đổi vị trí của một vật theo thời gian so với vật "
        "làm mốc, và sự thay đổi đó chỉ có ý nghĩa khi gắn với một vật mốc cụ thể nào đó."
    )
    page_three = (
        "Tốc độ cho biết mức độ nhanh hay chậm của chuyển động, đo bằng quãng đường "
        "đi được trong một đơn vị thời gian, nên nó là một đại lượng có hướng và có độ lớn."
    )
    path = tmp_path / "three.pdf"
    path.write_bytes(make_pdf(3, text_pages=(0, 1, 2)))
    client = FakeVisionClient(
        {
            1: [
                {
                    "kind": "heading",
                    "marker": None,
                    "depth": 1,
                    "group": 1,
                    "text": "Chuyển động cơ",
                },
                {"kind": "p", "text": page_one, "group": 1},
            ],
            3: [{"kind": "p", "text": page_three, "group": 1}],
        },
        continues={3: True},
    )

    artifact = await load_engine("vision").extract(
        path, doc_id="d", config=vision_config(), client=client, page_indices=[0, 2]
    )

    assert artifact.physical_pages == (0, 2)
    assert client.page_numbers == [1, 3]
    # Page 3 is asked as a fresh start: the heading page 1 opened is not offered as
    # something page 3 could continue.
    assert client.states[1] == {"open_headings": [], "open_group": None, "open_group_tail": ""}
    # The claim to continue across a region the model never saw is refused...
    assert [page.continues_previous for page in artifact.pages] == [False, False]
    # ...but the page read before the gap is kept: a reset is not a restart.
    assert [block.text for page in artifact.pages for block in page.blocks] == [
        "Chuyển động cơ",
        page_one,
        page_three,
    ]
    assert artifact.normalization["vision_continuity_resets"] == 1
    assert any("reset at physical pages [2]" in note for note in artifact.notes)

    # The same policy reads the reset back out of the artifact -- and only there: a
    # page that really follows the previous one keeps its continuation.
    assert continuity_breaks(artifact.pages) == (2,)
    pairs = ((0, None), (2, 0), (2, 1), (1, 0))
    assert [follows_previous(index, previous) for index, previous in pairs] == [
        False,
        False,
        True,
        True,
    ]
    claimed = (
        EnginePage(page_index=0, request_page=1, continues_previous=True),
        EnginePage(page_index=2, request_page=3, continues_previous=True),
    )
    assert [page.continues_previous for page in reset_continuity(claimed)] == [False, False]
    adjacent = (
        EnginePage(page_index=0, request_page=1),
        EnginePage(page_index=1, request_page=2, continues_previous=True),
    )
    assert reset_continuity(adjacent)[1].continues_previous is True

    # A replay of the artifact alone builds the grouping the run built: page 3 is a
    # chunk of its own instead of continuing the one left open before the gap.
    replayed = VisionDocumentParser("d")
    for page in reset_continuity(content_pages(artifact.pages)):
        replayed.add_page(page, page_number=page.request_page)
    replayed.finish()

    assert replayed.normalization["cross_page_continues"] == 0
    assert replayed.normalization["cross_page_joins"] == 0
    section = replayed.root.children[0]
    assert section.attributes.title == "Chuyển động cơ"
    assert [child.attributes.text for child in section.children] == [page_one, page_three]
    assert (replayed.root.metadata.start_page, replayed.root.metadata.end_page) == (1, 3)


# --- the isolated worker's wire form --------------------------------------------


def test_an_artifact_survives_the_workers_wire_form_exactly() -> None:
    """The worker's payload *is* the artifact: every field round-trips, none is invented."""
    artifact = EngineArtifact(
        engine_id="marker",
        engine_version="1.2.3",
        doc_id="vat-li-12",
        pdf_sha256="a" * 64,
        source_sha256="b" * 64,
        render_dpi=200,
        render_page_offset=3,
        prompt_sha256="c" * 64,
        public_config={"model": "test-vision", "dpi": 200},
        normalization={"marker_blocks": 1, "marker_pages": 2},
        notes=("a note",),
        pages=(
            EnginePage(
                page_index=0,
                request_page=4,
                blocks=(
                    EngineBlock(
                        kind="SectionHeader",
                        normalized_kind="heading",
                        text="Chuyển động cơ",
                        marker="1.1",
                        depth=2,
                        group=1,
                        bbox=(1.0, 2.0, 3.0, 4.0),
                        source_id="/page/0/SectionHeader/1",
                    ),
                ),
                running="Vật lí 12",
                printed_folio="7",
                continues_previous=True,
                usage={"attempts": 2, "prompt_tokens": 11, "completion_tokens": 7},
            ),
            EnginePage(page_index=2, request_page=6, status="unsupported_modality"),
        ),
    )

    # Through real JSON, because that is the only way the child's file is read.
    rebuilt = artifact_from_payload(json.loads(json.dumps(artifact_payload(artifact))))

    assert rebuilt == artifact
    assert rebuilt.pages[0].blocks[0].bbox == (1.0, 2.0, 3.0, 4.0)

    # A payload that lost a page is refused, never rebuilt as a smaller run.
    payload = artifact_payload(artifact)
    del payload["pages"]
    with pytest.raises(IsolatedWorkerError, match="could not be read"):
        artifact_from_payload(payload)


def test_the_worker_entrypoint_refuses_a_request_it_cannot_read(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    """An unreadable request is an exit code and one line of stderr, never a traceback."""
    absent = tmp_path / "absent.json"
    assert worker_main([str(absent)]) == 2
    assert "could not be read" in capsys.readouterr().err

    malformed = tmp_path / "malformed.json"
    malformed.write_text("[1, 2]", encoding="utf-8")
    assert worker_main([str(malformed)]) == 2
    assert "must be a JSON object" in capsys.readouterr().err

    assert worker_main([]) == 2
    assert "usage" in capsys.readouterr().err

    # No result file was written for a request that never ran.
    assert [path.name for path in tmp_path.iterdir()] == ["malformed.json"]
