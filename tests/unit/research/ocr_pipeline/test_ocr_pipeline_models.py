"""Offline tests for the research schemas: corpus manifest and neutral artifact.

Every fixture is a plain dictionary, and the artifact fixtures mirror what
``research.ocr_pipeline.engines`` writes, so a schema change that would reject an
engine's own output fails here. No PDF is opened, no network is touched and no
book content is read.
"""

from __future__ import annotations

import hashlib
import json
from typing import Any

import pytest
from pydantic import ValidationError

from research.ocr_pipeline.models import (
    ARTIFACT_VERSION,
    ENGINE_IDS,
    MAPPING_VERSION,
    NORMALIZED_KINDS,
    ArtifactBlock,
    ArtifactPage,
    DocumentSpec,
    ExtractionArtifact,
    Manifest,
    Modality,
    PageUsage,
    canonical_payload,
    page_text,
    semantic_fingerprint,
)

SHA = "ab" * 32
OTHER_SHA = "cd" * 32
PROMPT_SHA = "ef" * 32


# --------------------------------------------------------------------------- #
# fixtures shaped like the engine artifact contract
# --------------------------------------------------------------------------- #


def block(**overrides: Any) -> dict[str, Any]:
    """One block as an engine mapping writes it."""
    data: dict[str, Any] = {
        "order": 0,
        "raw_kind": "Text",
        "normalized_kind": "p",
        "text": "Nội dung trang",
        "marker": None,
        "depth": None,
        "group": None,
        "bbox": None,
        "source_id": None,
    }
    data.update(overrides)
    return data


def page(index: int = 0, **overrides: Any) -> dict[str, Any]:
    """One artifact page with a single paragraph block, unless overridden."""
    data: dict[str, Any] = {
        "page_index": index,
        "request_page": index + 5,
        "printed_folio": str(index + 4),
        "running": "VẬT LÍ 12",
        "continues_previous": False,
        "status": "ok",
        "blocks": [block()],
        "usage": None,
    }
    data.update(overrides)
    return data


def blocks_by_kind(pages: list[dict[str, Any]]) -> dict[str, int]:
    """The summary an engine derives from its own pages."""
    counts: dict[str, int] = {}
    for entry in pages:
        for item in entry["blocks"]:
            kind = str(item["normalized_kind"])
            counts[kind] = counts.get(kind, 0) + 1
    return dict(sorted(counts.items()))


def artifact_dict(**overrides: Any) -> dict[str, Any]:
    """One artifact as ``engines.build_artifact`` writes it."""
    pages = overrides.pop("pages", [page(0), page(1)])
    data: dict[str, Any] = {
        "artifact_version": 1,
        "mapping_version": 1,
        "engine_id": "vision",
        "engine_version": "vision-2026-09",
        "status": "ok",
        "doc_id": "vat-li-12",
        "pdf_sha256": SHA,
        "source_sha256": None,
        "physical_pages": [entry["page_index"] for entry in pages],
        "render_dpi": 200,
        "render_page_offset": 4,
        "prompt_sha256": PROMPT_SHA,
        "public_config": {"model": "vision-model", "endpoint_host": "opencode.ai"},
        "pages": pages,
        "statistics": {"blocks_by_kind": blocks_by_kind(pages), "normalization": {}},
        "notes": [],
    }
    data.update(overrides)
    return data


def artifact(**overrides: Any) -> ExtractionArtifact:
    """A validated artifact; overrides are applied to the raw dictionary."""
    return ExtractionArtifact.model_validate(artifact_dict(**overrides))


def document(**overrides: Any) -> dict[str, Any]:
    """One corpus document entry."""
    data: dict[str, Any] = {
        "doc_id": "vat-li-12",
        "pdf_path": "docs/vat-li-12.pdf",
        "pdf_sha256": SHA,
        "page_indices": [0, 1],
        "modality": "scan",
        "permission_note": "local research use only",
        "gold_path": None,
    }
    data.update(overrides)
    return data


# --------------------------------------------------------------------------- #
# the artifact contract
# --------------------------------------------------------------------------- #


def test_artifact_keeps_the_provenance_a_replay_needs() -> None:
    loaded = artifact(
        pages=[
            page(
                0,
                usage={"attempts": 2, "prompt_tokens": 11, "completion_tokens": 7},
            ),
            page(1),
        ]
    )

    assert loaded.pdf_sha256 == SHA
    assert loaded.doc_id == "vat-li-12"
    assert loaded.engine_id == "vision"
    assert loaded.engine_version == "vision-2026-09"
    assert loaded.mapping_version == 1
    assert loaded.physical_pages == (0, 1)
    assert loaded.render_dpi == 200
    assert loaded.render_page_offset == 4
    assert loaded.prompt_sha256 == PROMPT_SHA
    assert loaded.pages[0].request_page == 5
    assert loaded.pages[0].printed_folio == "4"
    assert loaded.pages[0].usage == PageUsage(attempts=2, prompt_tokens=11, completion_tokens=7)
    assert loaded.pages[1].usage is None
    assert loaded.statistics.blocks_by_kind == {"p": 2}


def test_artifact_validates_what_every_engine_writes() -> None:
    from research.ocr_pipeline import engines

    engine = engines.EngineArtifact(
        engine_id="vision",
        engine_version="vision-2026-09",
        doc_id="vat-li-12",
        pdf_sha256=SHA,
        render_dpi=200,
        render_page_offset=4,
        prompt_sha256=PROMPT_SHA,
        public_config={"model": "vision-model", "endpoint_host": "opencode.ai"},
        pages=(
            engines.EnginePage(
                page_index=0,
                request_page=5,
                running="VẬT LÍ 12",
                blocks=(
                    engines.EngineBlock(kind="Text", normalized_kind="p", text="Nội dung trang"),
                ),
            ),
        ),
    )

    loaded = ExtractionArtifact.from_engine(engine)

    assert loaded.engine_id == "vision"
    assert loaded.pdf_sha256 == SHA
    assert loaded.pages[0].request_page == 5
    assert loaded.pages[0].blocks[0].raw_kind == "Text"
    assert loaded.pages[0].blocks[0].normalized_kind == "p"
    assert loaded.physical_pages == (0,)
    assert loaded.statistics.blocks_by_kind == {"p": 1}
    # The adapter product becomes the persisted artifact, field for field.
    dumped = json.loads(canonical_payload(loaded))
    assert set(dumped) == set(ExtractionArtifact.model_fields)
    assert set(dumped["pages"][0]) == set(ArtifactPage.model_fields)
    assert set(dumped["pages"][0]["blocks"][0]) == set(ArtifactBlock.model_fields)


def test_schema_vocabularies_agree_with_the_engine_registry() -> None:
    from research.ocr_pipeline import engines

    assert ENGINE_IDS == engines.ENGINE_IDS
    assert NORMALIZED_KINDS == engines.NORMALIZED_KINDS
    assert ARTIFACT_VERSION == engines.ARTIFACT_VERSION
    assert MAPPING_VERSION == engines.MAPPING_VERSION


def test_artifact_carries_its_defaults_in_the_dump() -> None:
    dumped = json.loads(canonical_payload(artifact()))

    assert dumped["artifact_version"] == 1
    assert dumped["notes"] == []
    assert dumped["statistics"]["normalization"] == {}
    assert dumped["pages"][0]["continues_previous"] is False
    assert dumped["pages"][0]["blocks"][0]["group"] is None


def test_artifact_has_no_run_identity_of_its_own() -> None:
    assert "run_id" not in ExtractionArtifact.model_fields
    assert "created_at" not in ExtractionArtifact.model_fields

    with pytest.raises(ValidationError):
        artifact(run_id="0123456789abcdef0123456789abcdef")

    with pytest.raises(ValidationError):
        artifact(created_at="2026-09-27T12:00:00+00:00")


def test_artifact_allowlists_its_fields() -> None:
    assert set(ExtractionArtifact.model_fields) == {
        "artifact_version",
        "mapping_version",
        "engine_id",
        "engine_version",
        "status",
        "doc_id",
        "pdf_sha256",
        "source_sha256",
        "physical_pages",
        "render_dpi",
        "render_page_offset",
        "prompt_sha256",
        "public_config",
        "pages",
        "statistics",
        "notes",
    }


@pytest.mark.parametrize(
    ("field", "value"),
    [
        ("artifact_version", 2),
        ("artifact_version", "1"),
        ("mapping_version", 0),
        ("mapping_version", True),
        ("engine_id", "paddleocr"),
        ("status", "failed"),
        ("status", "not_available"),
        ("doc_id", "vat li"),
        ("pdf_sha256", OTHER_SHA.upper()),
        ("pdf_sha256", "ab"),
        ("source_sha256", "nope"),
        ("prompt_sha256", "nope"),
        ("render_dpi", 0),
        ("render_page_offset", 1.5),
        ("notes", ["ok", ""]),
    ],
)
def test_artifact_rejects_values_the_schema_cannot_mean(field: str, value: Any) -> None:
    with pytest.raises(ValidationError):
        artifact(**{field: value})


def test_artifact_allows_a_missing_pdf_sha_only_as_null() -> None:
    assert artifact(pdf_sha256=None).pdf_sha256 is None

    with pytest.raises(ValidationError):
        ExtractionArtifact.model_validate(
            {key: item for key, item in artifact_dict().items() if key != "pdf_sha256"}
        )


def test_artifact_requires_at_least_one_page() -> None:
    with pytest.raises(ValidationError):
        artifact(pages=[])


@pytest.mark.parametrize("pages", [[page(1), page(0)], [page(0), page(0)]])
def test_artifact_rejects_pages_out_of_physical_order(pages: list[dict[str, Any]]) -> None:
    with pytest.raises(ValidationError, match="strictly increasing"):
        artifact(pages=pages)


def test_artifact_rejects_a_summary_that_disagrees_with_its_pages() -> None:
    with pytest.raises(ValidationError, match="blocks_by_kind"):
        artifact(statistics={"blocks_by_kind": {"p": 99}, "normalization": {}})


def test_artifact_rejects_a_page_selection_that_disagrees_with_its_pages() -> None:
    with pytest.raises(ValidationError, match="physical_pages"):
        artifact(physical_pages=[0])


def test_artifact_rejects_an_unknown_statistics_key() -> None:
    with pytest.raises(ValidationError):
        artifact(statistics={"blocks_by_kind": {"p": 2}, "normalization": {}, "other": 1})


def test_artifact_rejects_a_public_config_that_could_carry_a_secret() -> None:
    for config in (
        {"api_key": "sk-live"},
        {"authorization": "Bearer sk-live"},
        {"headers": {"x-api-key": "sk-live"}},
        {"nested": {"access_token": "sk-live"}},
        {"endpoint": "https://user:sk-live@opencode.ai/v1?key=1"},
        {"endpoint_host": "opencode.ai:443/v1"},
    ):
        with pytest.raises(ValidationError):
            artifact(public_config=config)


def test_artifact_accepts_a_plain_public_config() -> None:
    loaded = artifact(
        public_config={"model": "vision-model", "endpoint_host": "opencode.ai", "max_tokens": 4096}
    )

    assert loaded.public_config["endpoint_host"] == "opencode.ai"


# --------------------------------------------------------------------------- #
# pages, blocks and the neutral page text
# --------------------------------------------------------------------------- #


def test_page_requires_a_request_number_and_a_known_status() -> None:
    assert ArtifactPage.model_validate(page(3)).page_index == 3

    for bad in ({"request_page": 0}, {"status": "failed"}, {"page_index": -1}):
        with pytest.raises(ValidationError):
            ArtifactPage.model_validate(page(0, **bad))


def test_page_usage_must_be_all_or_nothing() -> None:
    usage = PageUsage(attempts=0, prompt_tokens=0, completion_tokens=0)

    assert usage.attempts == 0

    for bad in (
        {"attempts": -1, "prompt_tokens": 0, "completion_tokens": 0},
        {"attempts": 1},
        {"attempts": 1, "prompt_tokens": 0, "completion_tokens": 0, "latency_ms": 5},
    ):
        with pytest.raises(ValidationError):
            PageUsage.model_validate(bad)


def test_block_requires_an_ordered_raw_kind_and_a_known_neutral_kind() -> None:
    with pytest.raises(ValidationError):
        ArtifactBlock.model_validate(block(normalized_kind="paragraph"))

    with pytest.raises(ValidationError):
        ArtifactBlock.model_validate(block(raw_kind="  "))

    with pytest.raises(ValidationError):
        ArtifactBlock.model_validate(block(order=-1))


def test_block_accepts_engine_provenance_for_a_marker_dump() -> None:
    loaded = ArtifactBlock.model_validate(
        block(
            raw_kind="SectionHeader",
            normalized_kind="heading",
            text="Bài 1",
            marker="1",
            depth=1,
            bbox=[1.0, 2.0, 3.0, 4.0],
            source_id="page/0/section-header/1",
        )
    )

    assert loaded.bbox == (1.0, 2.0, 3.0, 4.0)
    assert loaded.source_id == "page/0/section-header/1"


def test_block_heading_requires_depth_and_group_is_never_zero() -> None:
    with pytest.raises(ValidationError, match="heading blocks require depth"):
        ArtifactBlock.model_validate(block(normalized_kind="heading", depth=None))

    with pytest.raises(ValidationError):
        ArtifactBlock.model_validate(block(group=0))


def test_block_allows_blank_text_only_where_the_engine_can_own_it() -> None:
    assert ArtifactBlock.model_validate(block(normalized_kind="figure", text="")).text == ""
    assert ArtifactBlock.model_validate(block(normalized_kind="unmapped", text="")).text == ""

    with pytest.raises(ValidationError, match="text must not be blank"):
        ArtifactBlock.model_validate(block(normalized_kind="p", text="   "))


def test_block_bbox_needs_four_coordinates() -> None:
    with pytest.raises(ValidationError, match="bbox"):
        ArtifactBlock.model_validate(block(bbox=[1.0, 2.0]))


def test_page_text_joins_content_blocks_in_reading_order() -> None:
    loaded = ArtifactPage.model_validate(
        page(
            0,
            running="VẬT LÍ 12",
            blocks=[
                block(
                    order=0, raw_kind="PageHeader", normalized_kind="page_header", text="VẬT LÍ 12"
                ),
                block(order=1, raw_kind="Text", normalized_kind="p", text="Đoạn một."),
                block(order=2, raw_kind="Figure", normalized_kind="figure", text=""),
                block(order=3, raw_kind="Text", normalized_kind="p", text="Đoạn hai."),
                block(order=4, raw_kind="PageFooter", normalized_kind="page_footer", text="12"),
            ],
        )
    )

    assert page_text(loaded) == "Đoạn một.\nĐoạn hai."


def test_page_text_is_empty_for_a_page_with_no_content_blocks() -> None:
    assert page_text(ArtifactPage.model_validate(page(0, blocks=[]))) == ""


# --------------------------------------------------------------------------- #
# content digest and fingerprints
# --------------------------------------------------------------------------- #


def test_content_digest_is_the_address_of_the_bytes() -> None:
    loaded = artifact()
    payload = canonical_payload(loaded).encode("utf-8")

    assert loaded.content_digest == hashlib.sha256(payload).hexdigest()
    assert len(loaded.content_digest) == 64


def test_content_digest_is_stable_across_equal_artifacts_and_key_order() -> None:
    first = artifact()
    second = artifact()

    assert first == second
    assert first.content_digest == second.content_digest

    reordered = dict(reversed(list(artifact_dict().items())))
    assert ExtractionArtifact.model_validate(reordered).content_digest == first.content_digest


def test_content_digest_changes_when_content_or_provenance_changes() -> None:
    baseline = artifact()

    assert (
        baseline.content_digest
        != artifact(pages=[page(0, blocks=[block(text="Nội dung khác")]), page(1)]).content_digest
    )
    assert baseline.content_digest != artifact(engine_version="vision-2026-10").content_digest
    assert baseline.content_digest != artifact(notes=["one page retried"]).content_digest


def test_extraction_fingerprint_names_the_extraction_not_its_output() -> None:
    baseline = artifact()
    other_output = artifact(
        pages=[
            page(0, status="unsupported_modality", blocks=[]),
            page(1, blocks=[block(text="Trang này khác")]),
        ],
        notes=["page 0 unsupported"],
    )

    assert baseline.extraction_fingerprint == other_output.extraction_fingerprint

    for change in (
        {"engine_id": "marker"},
        {"engine_version": "vision-2026-10"},
        {"mapping_version": 2},
        {"pdf_sha256": OTHER_SHA},
        {"source_sha256": PROMPT_SHA},
        {"prompt_sha256": OTHER_SHA},
        {"render_dpi": 300},
        {"render_page_offset": None},
        {"public_config": {"model": "other-model"}},
        {"status": "unsupported_modality"},
    ):
        assert artifact(**change).extraction_fingerprint != baseline.extraction_fingerprint, change


def test_extraction_fingerprint_follows_the_selected_pages() -> None:
    assert (
        artifact(pages=[page(1)]).extraction_fingerprint
        != artifact(pages=[page(2)]).extraction_fingerprint
    )


def test_semantic_fingerprint_separates_the_builder_from_the_extraction() -> None:
    baseline = artifact()
    semantic = semantic_fingerprint(
        baseline, builder_id="atomic", builder_version=1, thresholds={"fragment_chars": 150}
    )

    assert semantic == semantic_fingerprint(
        baseline, builder_id="atomic", builder_version=1, thresholds={"fragment_chars": 150}
    )
    assert semantic != semantic_fingerprint(
        baseline, builder_id="vision_groups", builder_version=1
    )
    assert semantic != semantic_fingerprint(baseline, builder_id="atomic", builder_version=2)
    assert semantic != semantic_fingerprint(
        baseline, builder_id="atomic", builder_version=1, prompt_sha256=SHA
    )
    assert semantic != semantic_fingerprint(
        baseline, builder_id="atomic", builder_version=1, thresholds={"fragment_chars": 120}
    )
    # The extraction config is part of the profile, the extracted text is not.
    profile = {"builder_id": "atomic", "builder_version": 1, "thresholds": {"fragment_chars": 150}}
    assert semantic != semantic_fingerprint(artifact(engine_version="vision-2026-10"), **profile)
    assert semantic == semantic_fingerprint(artifact(notes=["repeat"]), **profile)


@pytest.mark.parametrize(
    "call",
    [
        {"builder_id": "  ", "builder_version": 1},
        {"builder_id": "atomic", "builder_version": ""},
        {"builder_id": "atomic", "builder_version": 1, "prompt_sha256": "nope"},
        {"builder_id": "atomic", "builder_version": 1, "thresholds": {"": 1}},
    ],
)
def test_semantic_fingerprint_rejects_incomplete_profile_identity(call: dict[str, Any]) -> None:
    with pytest.raises(ValueError):
        semantic_fingerprint(artifact(), **call)


def test_canonical_payload_round_trips() -> None:
    loaded = artifact(pages=[page(0, continues_previous=True)])

    assert ExtractionArtifact.model_validate_json(canonical_payload(loaded)) == loaded


# --------------------------------------------------------------------------- #
# corpus manifest
# --------------------------------------------------------------------------- #


def test_manifest_parses_a_multi_document_corpus() -> None:
    manifest = Manifest.model_validate(
        {
            "corpus_version": 1,
            "documents": [
                document(doc_id="vat-li-12", page_indices=[0, 1]),
                document(
                    doc_id="lich-su-12",
                    page_indices=[5],
                    modality="mixed",
                    gold_path="docs/lich-su.gold.json",
                ),
            ],
        }
    )

    assert [entry.doc_id for entry in manifest.documents] == ["vat-li-12", "lich-su-12"]
    assert manifest.documents[0].page_indices == (0, 1)
    assert manifest.documents[0].modality is Modality.SCAN
    assert manifest.documents[0].gold_path is None
    assert manifest.documents[1].modality is Modality.MIXED
    assert manifest.documents[1].gold_path == "docs/lich-su.gold.json"


@pytest.mark.parametrize("modality", ["scan", "digital", "mixed"])
def test_modality_describes_the_pdf_not_the_engine(modality: str) -> None:
    assert DocumentSpec.model_validate(document(modality=modality)).modality.value == modality


@pytest.mark.parametrize("modality", ["vision", "marker", "native_text", "native", "VISION"])
def test_modality_rejects_engine_names(modality: str) -> None:
    with pytest.raises(ValidationError):
        DocumentSpec.model_validate(document(modality=modality))


@pytest.mark.parametrize(
    "payload", [{"documents": [document()]}, {"corpus_version": 2, "documents": [document()]}]
)
def test_manifest_rejects_an_unknown_corpus_version(payload: dict[str, Any]) -> None:
    with pytest.raises(ValidationError):
        Manifest.model_validate(payload)


def test_manifest_rejects_undeclared_and_duplicate_documents() -> None:
    with pytest.raises(ValidationError):
        Manifest.model_validate(
            {"corpus_version": 1, "documents": [document()], "api_key": "sk-live-secret"}
        )

    with pytest.raises(ValidationError, match="duplicate doc_id"):
        Manifest.model_validate(
            {"corpus_version": 1, "documents": [document(), document(page_indices=[9])]}
        )

    with pytest.raises(ValidationError):
        Manifest.model_validate({"corpus_version": 1, "documents": []})


def test_manifest_is_frozen() -> None:
    manifest = Manifest.model_validate({"corpus_version": 1, "documents": [document()]})

    with pytest.raises(ValidationError):
        manifest.corpus_version = 2


@pytest.mark.parametrize("digest", ["AB" * 32, "ab" * 31, "z" * 64, "", "sha256:abc"])
def test_document_rejects_a_pdf_sha256_that_is_not_lowercase_hex(digest: str) -> None:
    with pytest.raises(ValidationError):
        DocumentSpec.model_validate(document(pdf_sha256=digest))


@pytest.mark.parametrize("page_indices", [[1, 0], [0, 0], [2, 1, 0], [0, 2, 1], [-1, 0], []])
def test_document_rejects_unordered_repeated_or_empty_page_indices(
    page_indices: list[int],
) -> None:
    with pytest.raises(ValidationError):
        DocumentSpec.model_validate(document(page_indices=page_indices))


def test_document_requires_a_permission_note_and_declared_references() -> None:
    with pytest.raises(ValidationError, match="permission_note"):
        DocumentSpec.model_validate(document(permission_note="   "))

    payload = document()
    del payload["gold_path"]
    with pytest.raises(ValidationError):
        DocumentSpec.model_validate(payload)


@pytest.mark.parametrize("field", ["gold_path", "pdf_path"])
def test_document_rejects_blank_paths(field: str) -> None:
    with pytest.raises(ValidationError):
        DocumentSpec.model_validate(document(**{field: "  "}))


@pytest.mark.parametrize("doc_id", ["", "  ", "vat li", "-vat-li", "vat/li"])
def test_document_rejects_a_doc_id_that_is_not_a_slug(doc_id: str) -> None:
    with pytest.raises(ValidationError, match="doc_id"):
        DocumentSpec.model_validate(document(doc_id=doc_id))
