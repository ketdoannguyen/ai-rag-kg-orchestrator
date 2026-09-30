"""Offline tests for the study CLI.

The CLI is a thin skin over the runner: it parses arguments, resolves the few
names a user would otherwise look up by hand, prints JSON and returns an exit
status. These tests drive ``main`` exactly as a shell would, with the runner's
dependencies replaced at the one seam the module offers (``make_runner``), and
assert on what a caller actually sees: the JSON on stdout, the message on stderr
and the exit code.

Nothing here reads a book, imports an engine package or opens a socket.
"""

from __future__ import annotations

import argparse
import json
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

import pytest

import scripts.run_ocr_study as cli
from research.ocr_pipeline.corpus import load_manifest, sha256_file
from research.ocr_pipeline.engines import (
    ENGINE_IDS,
    EngineArtifact,
    EngineBlock,
    EnginePage,
    EngineProbe,
)
from research.ocr_pipeline.runner import (
    DEFAULT_MERGE_CONCURRENCY,
    LLM_MODES,
    StudyRunner,
)

PROVENANCE: dict[str, Any] = {
    "git_commit": "0" * 40,
    "git_dirty": False,
    "python": "3.11.0",
    "platform": "test",
    "label": "cli tests",
}


# --------------------------------------------------------------------------- #
# synthetic corpus and a synthetic engine
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


class CliEngine:
    """A local adapter double, in the call shape ``_call_engine`` uses."""

    def __init__(
        self, engine_id: str = "native", *, kinds: Mapping[str, str] | None = None
    ) -> None:
        self.engine_id = engine_id
        self.kinds = dict(kinds or {"heading": "SectionHeader", "p": "Text"})
        self.error: Exception | None = None
        self.calls: list[str] = []

    def probe(self) -> EngineProbe:
        return EngineProbe(self.engine_id, True, "0.1-fake", "a local double")

    def extract(
        self,
        source: Any,
        *,
        doc_id: str,
        page_offset: int = 0,
        page_indices: Sequence[int] = (),
        converter: Any | None = None,
    ) -> EngineArtifact:
        self.calls.append(doc_id)
        if self.error is not None:
            raise self.error
        return EngineArtifact(
            engine_id=self.engine_id,
            engine_version="0.1-fake",
            doc_id=doc_id,
            pages=tuple(
                EnginePage(
                    page_index=index,
                    request_page=index + 1 + page_offset,
                    blocks=(
                        EngineBlock(
                            kind=self.kinds["heading"],
                            normalized_kind="heading",
                            text=f"Section {index + 1}",
                            depth=1,
                            group=index + 1,
                        ),
                        EngineBlock(
                            kind=self.kinds["p"],
                            normalized_kind="p",
                            text=f"body of {doc_id} page {index}",
                            group=index + 1,
                        ),
                        EngineBlock(
                            kind=self.kinds["p"],
                            normalized_kind="p",
                            text=f"more of {doc_id} page {index}",
                            group=index + 1,
                        ),
                    ),
                    printed_folio=str(index + 1),
                )
                for index in page_indices
            ),
            render_page_offset=page_offset,
        )


def fake_engines(**overrides: CliEngine) -> dict[str, CliEngine]:
    """One double per engine id, so no real adapter module is ever imported."""
    engines = {engine_id: CliEngine(engine_id) for engine_id in ENGINE_IDS}
    engines.update(overrides)
    return engines


class Fixture:
    """The corpus on disk, the study root, and the runner the CLI will use."""

    def __init__(self, base_dir: Path, engines: Mapping[str, Any] | None = None) -> None:
        self.base_dir = base_dir
        self.root = base_dir / "study"
        self.manifest_path = base_dir / "manifest.json"
        self.gold_path = base_dir / "gold.json"
        self.runner = StudyRunner(
            load_manifest(self.manifest_path),
            base_dir=base_dir,
            root=self.root,
            engines=dict(engines or fake_engines()),
            provenance=PROVENANCE,
        )


def build_corpus(base_dir: Path, *, declare_gold: bool = True) -> Fixture:
    """Write one digital document, two pages, under a checked reference.

    ``declare_gold=False`` leaves the manifest's ``gold_path`` null, which is how
    a document whose reference lives elsewhere is described.
    """
    base_dir.mkdir(parents=True, exist_ok=True)
    pdf = base_dir / "study.pdf"
    pdf.write_bytes(make_pdf(2, text_pages=(0, 1)))
    digest = sha256_file(pdf)
    (base_dir / "gold.json").write_text(
        json.dumps(
            {
                "doc_id": "study-1",
                "pages": [
                    {
                        "page_index": index,
                        "text": f"Section {index + 1}\nbody of study-1 page {index}\n"
                        f"more of study-1 page {index} checked",
                        "headings": [f"Section {index + 1}"],
                    }
                    for index in (0, 1)
                ],
            }
        ),
        encoding="utf-8",
    )
    (base_dir / "manifest.json").write_text(
        json.dumps(
            {
                "corpus_version": 1,
                "documents": [
                    {
                        "doc_id": "study-1",
                        "pdf_path": "study.pdf",
                        "pdf_sha256": digest,
                        "page_indices": [0, 1],
                        "modality": "digital",
                        "permission_note": "synthetic pages, so there is nothing to license",
                        "gold_path": "gold.json" if declare_gold else None,
                    }
                ],
            }
        ),
        encoding="utf-8",
    )
    return Fixture(base_dir)


@pytest.fixture
def fixture(tmp_path: Path) -> Fixture:
    return build_corpus(tmp_path)


def run_cli(fixture: Fixture, monkeypatch: pytest.MonkeyPatch, *argv: str) -> int:
    """Run one command the way a shell would, with the injected runner."""
    monkeypatch.setattr(cli, "make_runner", lambda _args: fixture.runner)
    return cli.main([*argv, "--corpus", str(fixture.manifest_path), "--root", str(fixture.root)])


def payload(capsys: pytest.CaptureFixture[str]) -> dict[str, Any]:
    """The JSON a command printed, asserting it really was JSON."""
    out = capsys.readouterr().out
    assert out.strip().startswith("{")
    return json.loads(out)


def ref_args(*pairs: str) -> list[str]:
    """``--run doc:run`` arguments from ``"doc:run"`` strings."""
    args: list[str] = []
    for pair in pairs:
        args += ["--run", pair]
    return args


# --------------------------------------------------------------------------- #
# the parser
# --------------------------------------------------------------------------- #


def test_help_lists_the_engines_profiles_families_and_the_two_permissions(
    capsys: pytest.CaptureFixture[str],
) -> None:
    with pytest.raises(SystemExit) as exit_info:
        cli.main(["--help"])

    assert exit_info.value.code == 0
    text = capsys.readouterr().out
    for engine_id in ENGINE_IDS:
        assert engine_id in text
    for profile in ("vision_direct", "marker_llm", "native_atomic"):
        assert profile in text
    for family in ("pipeline", "ocr", "chunk"):
        assert family in text
    # The two permissions are documented where a user will look for them.
    assert "--allow-upload" in text
    assert "--allow-merge-upload" in text
    # The modes are not in the epilog: they are the ``run`` and ``study`` flags'
    # own choices, and a mode outside the set is a usage error -- see below.


def test_a_command_without_a_corpus_is_a_usage_error() -> None:
    with pytest.raises(SystemExit) as exit_info:
        cli.main(["extract", "--engine", "native"])
    assert exit_info.value.code == 2


def test_an_engine_outside_the_registry_is_a_usage_error() -> None:
    with pytest.raises(SystemExit) as exit_info:
        cli.main(["extract", "--corpus", "corpus.json", "--engine", "tesseract"])
    assert exit_info.value.code == 2


def test_an_llm_mode_outside_the_registry_is_a_usage_error() -> None:
    """The modes are a closed set, and the one that faked a merge is not in it."""
    assert "fake" not in LLM_MODES, "a merge is run live or not at all"
    with pytest.raises(SystemExit) as exit_info:
        cli.main(
            [
                "run",
                "--profile",
                "native_atomic",
                "--llm-mode",
                "fake",
                "--corpus",
                "corpus.json",
            ]
        )

    assert exit_info.value.code == 2


def test_an_unknown_profile_is_refused_by_name(
    fixture: Fixture, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    code = run_cli(
        fixture,
        monkeypatch,
        "run",
        "--profile",
        "nope",
        "--extract-run",
        "study-1:study-1.native.r1",
    )
    assert code == 1
    assert "unknown profile 'nope'" in capsys.readouterr().err
    assert fixture.runner._engines["native"].calls == []


def test_an_unknown_document_is_refused_by_name(
    fixture: Fixture, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    code = run_cli(fixture, monkeypatch, "extract", "--engine", "native", "--doc", "nope")
    assert code == 1
    assert "unknown document 'nope'" in capsys.readouterr().err


# --------------------------------------------------------------------------- #
# corpus, extract, runs
# --------------------------------------------------------------------------- #


def test_validate_prints_the_documents_and_every_engine_probe(
    fixture: Fixture, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    code = run_cli(fixture, monkeypatch, "corpus", "validate")

    assert code == 0
    body = payload(capsys)
    assert [report["doc_id"] for report in body["documents"]] == ["study-1"]
    assert body["documents"][0]["pages_with_text"] == [0, 1]
    assert [probe["engine_id"] for probe in body["engines"]] == list(ENGINE_IDS)
    assert all(probe["available"] for probe in body["engines"])


def test_extract_stores_one_run_per_repeat_and_exits_zero(
    fixture: Fixture, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    code = run_cli(fixture, monkeypatch, "extract", "--engine", "native", "--repeats", "2")

    assert code == 0
    body = payload(capsys)
    assert [run["run_id"] for run in body["runs"]] == ["study-1.native.r1", "study-1.native.r2"]
    assert all(run["status"] == "ok" for run in body["runs"])
    assert fixture.runner._engines["native"].calls == ["study-1", "study-1"]


def test_extract_prints_no_transcript(
    fixture: Fixture, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    run_cli(fixture, monkeypatch, "extract", "--engine", "native")

    # The run manifest records digests and counters; the words stay in the store.
    assert "body of study-1" not in capsys.readouterr().out


def test_a_failed_extraction_exits_nonzero_and_is_still_printed(
    fixture: Fixture, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    fixture.runner._engines["native"].error = RuntimeError("the engine gave up")

    code = run_cli(fixture, monkeypatch, "extract", "--engine", "native")

    assert code == 1
    body = payload(capsys)
    assert body["runs"][0]["status"] == "failed"
    assert body["runs"][0]["reason"] == "engine_failed"


def test_a_paid_engine_is_refused_without_the_upload_flag(
    fixture: Fixture, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    code = run_cli(fixture, monkeypatch, "extract", "--engine", "vision")

    assert code == 1
    captured = capsys.readouterr()
    assert captured.out == ""  # nothing ran, so there is nothing to print
    assert "--allow-upload" in captured.err


def test_runs_lists_the_manifests_on_disk(
    fixture: Fixture, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    run_cli(fixture, monkeypatch, "extract", "--engine", "native")
    capsys.readouterr()

    code = run_cli(fixture, monkeypatch, "runs", "--kind", "extract")

    assert code == 0
    body = payload(capsys)
    assert [entry["run_id"] for entry in body["runs"]] == ["study-1.native.r1"]
    assert body["runs"][0]["artifact_digest"]
    assert set(body["runs"][0]) == {
        "run_id",
        "kind",
        "doc_id",
        "engine_id",
        "profile",
        "status",
        "rankable",
        "artifact_digest",
        "created_at",
        "path",
    }


# --------------------------------------------------------------------------- #
# run: replay only
# --------------------------------------------------------------------------- #


def test_run_replays_a_stored_artifact_without_calling_the_engine(
    fixture: Fixture, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    run_cli(fixture, monkeypatch, "extract", "--engine", "native")
    capsys.readouterr()

    code = run_cli(
        fixture,
        monkeypatch,
        "run",
        "--profile",
        "native_atomic",
        "--extract-run",
        "study-1:study-1.native.r1",
    )

    assert code == 0
    body = payload(capsys)
    assert len(body["trees"]) == 1
    tree = body["trees"][0]
    assert tree["run_id"] == "study-1.native_atomic.r1.none"
    assert tree["status"] == "ok"
    assert tree["rankable"] is True
    assert tree["pages"] == [0, 1]
    assert fixture.runner._engines["native"].calls == ["study-1"]  # the replay added none


def test_run_with_all_compatible_builds_the_engine_table(
    fixture: Fixture, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    run_cli(fixture, monkeypatch, "extract", "--engine", "native")
    capsys.readouterr()

    code = run_cli(
        fixture,
        monkeypatch,
        "run",
        "--profile",
        "all-compatible",
        "--extract-run",
        "study-1:study-1.native.r1",
    )

    assert code == 0
    body = payload(capsys)
    assert [tree["profile"] for tree in body["trees"]] == ["native_atomic"]


def test_run_refuses_two_artifact_sources_at_once(
    fixture: Fixture, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    code = run_cli(
        fixture,
        monkeypatch,
        "run",
        "--profile",
        "native_atomic",
        "--extract-run",
        "study-1:study-1.native.r1",
        "--artifact",
        "ab" * 32,
    )

    assert code == 1
    assert "not both" in capsys.readouterr().err


def test_run_refuses_an_artifact_for_more_than_one_document(
    fixture: Fixture, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    code = run_cli(
        fixture,
        monkeypatch,
        "run",
        "--profile",
        "native_atomic",
        "--doc",
        "study-1",
        "--doc",
        "study-1",
        "--artifact",
        "ab" * 32,
    )

    assert code == 1
    assert "exactly one --doc" in capsys.readouterr().err


def test_run_without_anything_to_replay_says_what_to_do(
    fixture: Fixture, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    code = run_cli(fixture, monkeypatch, "run", "--profile", "native_atomic")

    assert code == 1
    assert "nothing to replay for study-1" in capsys.readouterr().err


# --------------------------------------------------------------------------- #
# the merge gates
# --------------------------------------------------------------------------- #


def test_a_live_merge_needs_the_merge_upload_flag(
    fixture: Fixture, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    run_cli(fixture, monkeypatch, "extract", "--engine", "marker")
    capsys.readouterr()

    code = run_cli(
        fixture,
        monkeypatch,
        "run",
        "--profile",
        "marker_llm",
        "--llm-mode",
        "live",
        "--extract-run",
        "study-1:study-1.marker.r1",
    )

    assert code == 1
    assert "--allow-merge-upload" in capsys.readouterr().err


# compare
# --------------------------------------------------------------------------- #


def test_compare_writes_the_three_formats_and_prints_the_report(
    fixture: Fixture, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    run_cli(fixture, monkeypatch, "extract", "--engine", "native")
    capsys.readouterr()
    out = fixture.base_dir / "comparison"

    code = run_cli(
        fixture,
        monkeypatch,
        "compare",
        "--family",
        "ocr",
        *ref_args("study-1:study-1.native.r1"),
        "--out",
        str(out),
    )

    assert code == 0
    body = payload(capsys)
    assert body["family"] == "ocr"
    assert body["report"]["counts"]["documents"] == 1
    assert body["report"]["counts"]["pages_with_gold"] == 2
    for name in ("json", "csv", "markdown"):
        assert Path(body["files"][name]).is_file()
    assert (out / "manifest.json").is_file()


def test_compare_dry_run_prints_without_writing(
    fixture: Fixture, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    run_cli(fixture, monkeypatch, "extract", "--engine", "native")
    capsys.readouterr()

    code = run_cli(
        fixture,
        monkeypatch,
        "compare",
        "--family",
        "ocr",
        *ref_args("study-1:study-1.native.r1"),
        "--dry-run",
    )

    assert code == 0
    body = payload(capsys)
    assert "path" not in body
    assert not (fixture.root / "comparisons").exists()


def test_compare_refuses_a_gold_flag_that_is_not_doc_equals_path(
    fixture: Fixture, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    code = run_cli(
        fixture,
        monkeypatch,
        "compare",
        "--family",
        "ocr",
        *ref_args("study-1:study-1.native.r1"),
        "--gold",
        "nonsense",
    )

    assert code == 1
    assert "--gold 'nonsense' must be DOC_ID=PATH" in capsys.readouterr().err


# gold and study
# --------------------------------------------------------------------------- #


def test_gold_check_reports_the_page_coverage(
    fixture: Fixture, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    code = run_cli(
        fixture,
        monkeypatch,
        "gold",
        "check",
        "--doc",
        "study-1",
        "--file",
        str(fixture.gold_path),
    )

    assert code == 0
    body = payload(capsys)
    assert body["doc_id"] == "study-1"
    assert body["requested_pages"] == [0, 1]
    assert body["covered_pages"] == [0, 1]
    assert body["missing_pages"] == []
    assert body["out_of_scope_pages"] == []
    assert body["complete"] is True


def test_study_dry_run_plans_without_running(
    fixture: Fixture, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    code = run_cli(
        fixture,
        monkeypatch,
        "study",
        "--families",
        "ocr,pipeline",
        "--engines",
        "native",
        "--dry-run",
    )

    assert code == 0
    body = payload(capsys)
    assert body["executed"] is False
    assert body["plan"]["corpus"]["documents"] == ["study-1"]
    assert len(body["plan"]["extractions"]) == 1
    assert fixture.runner._engines["native"].calls == []


def test_the_merge_limit_reaches_the_runner_and_the_dry_run_says_so(
    fixture: Fixture, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    """One flag settles the limit the run is fingerprinted with and the plan prints.

    The limit is part of a tree's identity, so a study that merges with three
    slots is not the study that merged with one: the dry run names it before
    anything is spent, and the runner the CLI builds carries the same number.
    """
    built: list[StudyRunner] = []

    def make(args: argparse.Namespace) -> StudyRunner:
        runner = StudyRunner(
            load_manifest(fixture.manifest_path),
            base_dir=fixture.base_dir,
            root=fixture.root,
            engines=fake_engines(),
            provenance=PROVENANCE,
            merge_concurrency=args.merge_concurrency,
        )
        built.append(runner)
        return runner

    monkeypatch.setattr(cli, "make_runner", make)
    code = cli.main(
        [
            "study",
            "--families",
            "ocr",
            "--engines",
            "native",
            "--dry-run",
            "--merge-concurrency",
            "3",
            "--corpus",
            str(fixture.manifest_path),
            "--root",
            str(fixture.root),
        ]
    )

    assert code == 0
    assert [runner.merge_concurrency for runner in built] == [3]
    assert payload(capsys)["plan"]["merge_concurrency"] == 3


def test_the_merge_limit_has_a_research_default_and_refuses_zero(
    fixture: Fixture, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    code = run_cli(
        fixture, monkeypatch, "study", "--families", "ocr", "--engines", "native", "--dry-run"
    )

    assert code == 0
    assert payload(capsys)["plan"]["merge_concurrency"] == DEFAULT_MERGE_CONCURRENCY

    # Both commands that merge take the flag, and neither takes a zero: a merge
    # with no slot would never call anything.
    parsed = cli.build_parser().parse_args(
        [
            "run",
            "--profile",
            "native_atomic",
            "--merge-concurrency",
            "2",
            "--corpus",
            str(fixture.manifest_path),
        ]
    )
    assert parsed.merge_concurrency == 2

    with pytest.raises(SystemExit) as exit_info:
        cli.main(
            [
                "study",
                "--families",
                "ocr",
                "--engines",
                "native",
                "--merge-concurrency",
                "0",
                "--corpus",
                str(fixture.manifest_path),
            ]
        )

    assert exit_info.value.code == 2
    assert "below one" in capsys.readouterr().err


def test_study_refuses_a_family_outside_the_registry(
    fixture: Fixture, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    code = run_cli(
        fixture,
        monkeypatch,
        "study",
        "--families",
        "pipeline,modalities",
        "--engines",
        "native",
    )

    assert code == 1
    assert "--families modalities" in capsys.readouterr().err


def test_study_runs_extracts_builds_and_compares(
    fixture: Fixture, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    code = run_cli(
        fixture, monkeypatch, "study", "--families", "ocr,pipeline", "--engines", "native"
    )

    assert code == 0
    body = payload(capsys)
    assert body["executed"] is True
    assert [entry["status"] for entry in body["extractions"]] == ["ok"]
    assert [entry["profile"] for entry in body["trees"]] == ["native_atomic"]
    assert {entry["family"] for entry in body["comparisons"]} == {"ocr", "pipeline"}
    # The one document has a checked reference, so every family scores it.
    assert all(entry["counts"]["documents"] == 1 for entry in body["comparisons"])
    assert fixture.runner._engines["native"].calls == ["study-1"]


def test_study_exits_nonzero_when_a_run_it_stored_failed(
    fixture: Fixture, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    fixture.runner._engines["native"].error = RuntimeError("the engine gave up")

    code = run_cli(fixture, monkeypatch, "study", "--families", "ocr", "--engines", "native")

    assert code == 1  # a failed run is printed and recorded, and the shell hears about it
    body = payload(capsys)
    assert body["executed"] is True
    assert [entry["status"] for entry in body["extractions"]] == ["failed"]
    assert [entry["reason"] for entry in body["skipped"]] == ["no_successful_extraction"]


def test_study_reports_the_family_it_could_not_run(
    fixture: Fixture, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    code = run_cli(fixture, monkeypatch, "study", "--families", "ocr,chunk", "--engines", "native")

    assert code == 0
    body = payload(capsys)
    assert [entry["status"] for entry in body["trees"]] == ["ok"]
    assert {entry["family"] for entry in body["comparisons"]} == {"ocr"}
    # The chunk family wanted a vision tree that this study never built, and says so.
    assert body["skipped"] == [
        {"family": "chunk", "doc_id": "study-1", "reason": "no_chunk_profile_tree"}
    ]


def test_study_refuses_a_profile_its_llm_mode_would_drop(
    fixture: Fixture, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    code = run_cli(
        fixture,
        monkeypatch,
        "study",
        "--families",
        "pipeline",
        "--engines",
        "vision",
        "--profile",
        "vision_atomic_merge",
        "--allow-upload",
    )

    assert code == 1
    error = capsys.readouterr().err
    assert "vision_atomic_merge" in error
    assert "--llm-mode" in error
    assert fixture.runner._engines["vision"].calls == []  # refused before any upload


def test_study_takes_gold_for_a_document_that_declares_none(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    undeclared = build_corpus(tmp_path / "undeclared", declare_gold=False)

    code = run_cli(undeclared, monkeypatch, "study", "--families", "ocr", "--engines", "native")

    assert code == 0
    body = payload(capsys)
    assert body["comparisons"][0]["counts"]["documents_with_gold"] in (None, 0)
    # The report says, in words, why nothing was scored.
    written = Path(body["comparisons"][0]["path"]) / "report.json"
    assert any("no gold" in line for line in json.loads(written.read_text())["limitations"])

    # The same study with an explicit --gold scores the two checked pages.
    code = run_cli(
        undeclared,
        monkeypatch,
        "study",
        "--families",
        "ocr",
        "--engines",
        "native",
        "--gold",
        f"study-1={undeclared.gold_path}",
    )

    assert code == 0
    counts = payload(capsys)["comparisons"][0]["counts"]
    assert counts["documents_with_gold"] == 1
    assert counts["pages_with_gold"] == 2


def test_a_study_resumes_by_default_and_fresh_runs_every_repeat_again(
    fixture: Fixture, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    engine = fixture.runner._engines["native"]
    args = ["study", "--families", "ocr", "--engines", "native", "--repeats", "2"]

    assert run_cli(fixture, monkeypatch, *args) == 0
    first = payload(capsys)
    calls = len(engine.calls)
    assert calls == 2  # two repeats of one document, both really run

    assert run_cli(fixture, monkeypatch, *args) == 0
    second = payload(capsys)
    assert len(engine.calls) == calls  # the default policy reused both repeats
    assert [entry["run_id"] for entry in second["extractions"]] == [
        entry["run_id"] for entry in first["extractions"]
    ]

    assert run_cli(fixture, monkeypatch, *args, "--fresh") == 0
    fresh = payload(capsys)
    assert len(engine.calls) == calls + 2
    assert {entry["run_id"] for entry in fresh["extractions"]} == {
        "study-1.native.r1-a2",
        "study-1.native.r2-a2",
    }


def test_a_study_dry_run_reports_the_cache_it_would_reuse(
    fixture: Fixture, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    args = ["study", "--families", "ocr", "--engines", "native", "--dry-run"]
    mixed_args = ["study", "--families", "ocr,pipeline", "--engines", "native", "--dry-run"]

    # An OCR-only study asks for extractions and no trees at all: its plan is
    # one miss, not one per family that was never asked for.
    assert run_cli(fixture, monkeypatch, *args) == 0
    empty = payload(capsys)["plan"]
    assert empty["trees"] == []
    assert empty["cache"] == {"hits": 0, "misses": 1}
    assert empty["extractions"][0]["cache_reason"] == "no_run"

    # A mixed study adds the pipeline family, and with it a tree per run.
    assert run_cli(fixture, monkeypatch, *mixed_args) == 0
    mixed = payload(capsys)["plan"]
    assert [entry["profile"] for entry in mixed["trees"]] == ["native_atomic"]
    assert mixed["cache"] == {"hits": 0, "misses": 2}
    assert mixed["trees"][0]["cache_reason"] == "no_artifact"

    assert run_cli(fixture, monkeypatch, "study", "--families", "ocr", "--engines", "native") == 0
    payload(capsys)

    # The extraction is on disk now, so an OCR-only study reuses it; the mixed
    # study reuses the same extraction and still has to build its tree.
    assert run_cli(fixture, monkeypatch, *args) == 0
    reused = payload(capsys)["plan"]
    assert reused["cache"] == {"hits": 1, "misses": 0}

    assert run_cli(fixture, monkeypatch, *mixed_args) == 0
    mixed_reused = payload(capsys)["plan"]
    assert mixed_reused["cache"] == {"hits": 1, "misses": 1}
    assert mixed_reused["trees"][0]["cache_reason"] == "no_run"

    assert run_cli(fixture, monkeypatch, *args, "--fresh") == 0
    planned = payload(capsys)["plan"]
    assert planned["cache"] == {"hits": 0, "misses": 1}
    assert planned["extractions"][0]["cache_reason"] == "fresh"
