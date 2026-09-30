#!/usr/bin/env python
"""Run the OCR knowledge-tree study from the command line.

Every command speaks the corpus manifest's vocabulary: 0-based physical page
indices, the doc ids it declares, engines and profiles by their registry names.
The runner does the work; this module only parses arguments, resolves the few
things a user would otherwise have to look up by hand, and prints JSON.

    python scripts/run_ocr_study.py corpus validate --corpus corpus.json
    python scripts/run_ocr_study.py render --corpus corpus.json --doc vat-li-12 --pages 0-9 --out data/research/ocr_pipeline/renders
    python scripts/run_ocr_study.py gold check --corpus corpus.json --doc vat-li-12 --file gold.json
    python scripts/run_ocr_study.py extract --corpus corpus.json --engine marker --repeats 2
    python scripts/run_ocr_study.py runs --corpus corpus.json
    python scripts/run_ocr_study.py run --corpus corpus.json --profile marker_atomic --extract-run vat-li-12:vat-li-12.marker.r1
    python scripts/run_ocr_study.py compare --corpus corpus.json --family ocr --run vat-li-12:vat-li-12.marker.r1 --run vat-li-12:vat-li-12.native.r1
    python scripts/run_ocr_study.py study --corpus corpus.json --families pipeline,ocr,chunk --engines marker,native --repeats 2 --gold vat-li-12=gold.json

Nothing is written to stdout but JSON, and nothing leaves the machine unless a
flag says so: a paid engine needs ``--allow-upload`` and a live detail merge
needs ``--allow-merge-upload``. Both are refused before any client is built.
The exit status is 0 when every run the command touched succeeded and 1 when one
did not — a failed run is printed and recorded, never hidden.
"""

from __future__ import annotations

import argparse
import json
import sys
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

REPO_ROOT = Path(__file__).resolve().parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from research.ocr_pipeline.corpus import CorpusError
from research.ocr_pipeline.engines import ENGINE_IDS
from research.ocr_pipeline.profiles import PROFILES, get_profile
from research.ocr_pipeline.reference import GoldError
from research.ocr_pipeline.runner import (
    DEFAULT_MERGE_CONCURRENCY,
    DEFAULT_ROOT,
    FAMILIES,
    KIND_EXTRACT,
    KIND_TREE,
    LLM_MODES,
    RUN_OK,
    RunnerError,
    RunRef,
    StudyRunner,
    parse_page_selection,
    profiles_for_engine,
)
from research.ocr_pipeline.store import ArtifactStoreError

ALL_COMPATIBLE = "all-compatible"
"""The one ``--profile`` value that is not a profile: the artifact engine's table."""

_ENGINE_NOTES = {
    "vision": "paid: renders pages and uploads them (needs --allow-upload)",
    "marker": "local: needs the marker package and its weights",
    "docling": "local: needs the docling package",
    "native": "local: reads the PDF text layer; a scanned page is unsupported_modality",
}

_FAMILY_NOTES = {
    "pipeline": "full path, extraction to tree, one primary profile per engine",
    "ocr": "transcripts only; it says nothing about the trees built from them",
    "chunk": "the same pages grouped (vision_direct) against merged (vision_atomic_merge)",
}


# --------------------------------------------------------------------------- #
# plumbing
# --------------------------------------------------------------------------- #


def make_runner(args: argparse.Namespace) -> StudyRunner:
    """The one place a study's real dependencies are assembled."""
    return StudyRunner.from_corpus(
        args.corpus,
        root=args.root,
        merge_concurrency=getattr(args, "merge_concurrency", DEFAULT_MERGE_CONCURRENCY),
    )


def _emit(payload: Any) -> None:
    print(json.dumps(payload, indent=2, ensure_ascii=False, sort_keys=True))


def _fail(exc: Exception) -> int:
    print(f"error: {exc}", file=sys.stderr)
    return 1


def _documents(runner: StudyRunner, args: argparse.Namespace) -> list[str]:
    """The documents a command works on: the named ones, or the whole corpus."""
    if args.doc:
        for doc_id in args.doc:
            runner.document(doc_id)  # raises, naming the declared ids, when unknown
        return list(args.doc)
    return [spec.doc_id for spec in runner.documents()]


def _pages(args: argparse.Namespace) -> tuple[int, ...] | None:
    return None if args.pages is None else parse_page_selection(args.pages)


def _doc_paths(entries: Sequence[str] | None, *, flag: str) -> dict[str, str]:
    paths: dict[str, str] = {}
    for entry in entries or []:
        doc_id, separator, path = entry.partition("=")
        if not separator or not doc_id or not path:
            raise RunnerError(f"{flag} {entry!r} must be DOC_ID=PATH")
        paths[doc_id] = path
    return paths


def _names(value: str, allowed: Sequence[str], *, flag: str) -> list[str]:
    names = [part.strip() for part in value.split(",") if part.strip()]
    if not names:
        raise RunnerError(f"{flag} is empty")
    unknown = [name for name in names if name not in allowed]
    if unknown:
        raise RunnerError(f"{flag} {', '.join(unknown)}: expected one of {', '.join(allowed)}")
    return names


def _known_profile(name: str) -> str:
    try:
        return get_profile(name).name
    except ValueError as exc:
        raise RunnerError(str(exc)) from exc


def _exit_for(runs: Sequence[Mapping[str, Any]]) -> int:
    return 0 if all(entry.get("status") == RUN_OK for entry in runs) else 1


# --------------------------------------------------------------------------- #
# commands
# --------------------------------------------------------------------------- #


def cmd_corpus_validate(args: argparse.Namespace) -> int:
    """Check every document, then list which engines could actually read it."""
    runner = make_runner(args)
    reports = runner.validate(doc_ids=_documents(runner, args))
    engines = [runner.probe(engine_id) for engine_id in ENGINE_IDS]
    _emit({"corpus": str(args.corpus), "documents": reports, "engines": engines})
    return 0


def cmd_render(args: argparse.Namespace) -> int:
    """Render pages to PNGs, the images a gold transcript is read from."""
    runner = make_runner(args)
    pages = _pages(args)
    parent = Path(args.out)
    rendered = [
        runner.render_pages(
            doc_id,
            page_indices=pages if pages is not None else runner.document(doc_id).page_indices,
            out_dir=parent / doc_id,
            dpi=args.dpi,
            page_offset=args.offset,
        )
        for doc_id in _documents(runner, args)
    ]
    _emit({"renders": rendered})
    return 0


def cmd_gold_check(args: argparse.Namespace) -> int:
    """How much of a document's page selection its gold file covers."""
    runner = make_runner(args)
    _emit(runner.check_gold(args.doc, args.file))
    return 0


def cmd_extract(args: argparse.Namespace) -> int:
    """Run one engine over the selected documents, once per repeat."""
    runner = make_runner(args)
    pages = _pages(args)
    runs = [
        runner.extract(
            doc_id,
            args.engine,
            repeat=repeat,
            page_indices=pages,
            allow_upload=args.allow_upload,
            dpi=args.dpi,
            page_offset=args.offset,
            warmth=args.warmth,
            resume=args.resume,
            label=args.label,
        )
        for doc_id in _documents(runner, args)
        for repeat in range(1, args.repeats + 1)
    ]
    _emit({"runs": runs})
    return _exit_for(runs)


def cmd_runs(args: argparse.Namespace) -> int:
    """List the runs already on disk, so a comparison can name them."""
    runner = make_runner(args)
    found = [
        {
            key: manifest.get(key)
            for key in (
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
            )
        }
        for doc_id in _documents(runner, args)
        for manifest in runner.list_runs(doc_id, kind=args.kind)
    ]
    _emit({"runs": found})
    return 0


def cmd_run(args: argparse.Namespace) -> int:
    """Build a tree from an artifact already on disk: no engine runs here."""
    runner = make_runner(args)
    doc_ids = _documents(runner, args)
    plan = _replay_plan(args, doc_ids)
    profiles = _run_profiles(runner, args, plan)
    trees = []
    for doc_id in doc_ids:
        choice = plan.get(doc_id)
        if choice is None:
            raise RunnerError(
                f"nothing to replay for {doc_id}: pass --extract-run {doc_id}:RUN_ID "
                f"or --artifact DIGEST (see the runs command)"
            )
        for profile_name in profiles[doc_id]:
            trees.append(
                runner.run_tree(
                    doc_id,
                    profile_name,
                    artifact_digest=choice.get("artifact_digest"),
                    extract_run_id=choice.get("extract_run_id"),
                    llm_mode=args.llm_mode,
                    allow_merge_upload=args.allow_merge_upload,
                    resume=args.resume,
                    label=args.label,
                )
            )
    _emit({"trees": trees})
    return _exit_for(trees)


def _replay_plan(args: argparse.Namespace, doc_ids: Sequence[str]) -> dict[str, dict[str, str]]:
    """Which artifact each document replays: a digest, or an extraction run id."""
    if args.artifact is not None:
        if args.extract_run:
            raise RunnerError("pass either --artifact or --extract-run, not both")
        if len(doc_ids) != 1:
            raise RunnerError("--artifact names one artifact, so it needs exactly one --doc")
        return {doc_ids[0]: {"artifact_digest": args.artifact}}
    plan: dict[str, dict[str, str]] = {}
    for value in args.extract_run or []:
        ref = RunRef.parse(value)
        if ref.doc_id not in doc_ids:
            raise RunnerError(f"--extract-run {value} is outside the selected documents")
        plan[ref.doc_id] = {"extract_run_id": ref.run_id}
    return plan


def _run_profiles(
    runner: StudyRunner, args: argparse.Namespace, plan: Mapping[str, Mapping[str, str]]
) -> dict[str, list[str]]:
    """Which profiles each document builds: the named one, or its engine's table."""
    if args.profile != ALL_COMPATIBLE:
        name = _known_profile(args.profile)
        return {doc_id: [name] for doc_id in plan}
    profiles = {}
    for doc_id, choice in plan.items():
        engine = _engine_of(runner, doc_id, choice)
        names = [
            profile.name
            for profile in profiles_for_engine(engine)
            if args.llm_mode != "none" or profile.merge == "off"
        ]
        if not names:
            raise RunnerError(
                f"engine {engine!r} declares no profile with merge off; "
                "pass --llm-mode live to build one"
            )
        profiles[doc_id] = names
    return profiles


def _engine_of(runner: StudyRunner, doc_id: str, choice: Mapping[str, str]) -> str:
    """The engine an artifact belongs to: from its run, or from the artifact itself."""
    if "artifact_digest" in choice:
        return runner.artifact(doc_id, choice["artifact_digest"]).engine_id
    manifest = runner.read_run(doc_id, choice["extract_run_id"])
    assert manifest is not None  # read_run raises when required and missing
    engine = manifest.get("engine_id")
    if not isinstance(engine, str):
        raise RunnerError(f"{doc_id}:{choice['extract_run_id']} names no engine")
    return engine


def cmd_compare(args: argparse.Namespace) -> int:
    runner = make_runner(args)
    result = runner.compare(
        args.family,
        [RunRef.parse(value) for value in args.run],
        gold_paths=_doc_paths(args.gold, flag="--gold"),
        label=args.label,
        out_dir=args.out,
        write=not args.dry_run,
    )
    _emit(result)
    return 0


def cmd_study(args: argparse.Namespace) -> int:
    runner = make_runner(args)
    result = runner.study(
        families=_names(args.families, FAMILIES, flag="--families"),
        engines=_names(args.engines, ENGINE_IDS, flag="--engines"),
        doc_ids=args.doc or None,
        profiles=[_known_profile(name) for name in args.profile or []],
        repeats=args.repeats,
        allow_upload=args.allow_upload,
        llm_mode=args.llm_mode,
        allow_merge_upload=args.allow_merge_upload,
        gold_paths=_doc_paths(args.gold, flag="--gold"),
        page_indices=_pages(args),
        label=args.label,
        dry_run=args.dry_run,
        resume=args.resume,
    )
    _emit(result)
    if not result.get("executed"):
        return 0  # a dry run planned; it ran nothing, so there is nothing to fail
    return _exit_for([*result["extractions"], *result["trees"]])


# --------------------------------------------------------------------------- #
# the parser
# --------------------------------------------------------------------------- #


def _add_corpus(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--corpus", required=True, help="path to the corpus manifest JSON")
    parser.add_argument(
        "--root",
        default=str(DEFAULT_ROOT),
        help="study root that holds artifacts, runs and comparisons (default: %(default)s)",
    )


def _add_doc(parser: argparse.ArgumentParser) -> None:
    parser.add_argument(
        "--doc",
        action="append",
        metavar="DOC_ID",
        help="limit to this document (repeatable; default: every document in the corpus)",
    )


def _add_pages(parser: argparse.ArgumentParser) -> None:
    parser.add_argument(
        "--pages",
        metavar="N-M,...",
        help="0-based physical pages, e.g. 0-9,20 (default: the document's own selection)",
    )


def _add_llm_mode(parser: argparse.ArgumentParser) -> None:
    parser.add_argument(
        "--llm-mode",
        choices=list(LLM_MODES),
        default="none",
        help="none skips the detail merge; live runs it and needs "
        "--allow-merge-upload (default: %(default)s)",
    )


def _concurrency(value: str) -> int:
    """One whole number of merge calls, at least one."""
    try:
        number = int(value)
    except ValueError:
        raise argparse.ArgumentTypeError(f"{value!r} is not a whole number") from None
    if number < 1:
        raise argparse.ArgumentTypeError(f"{value!r} is below one; a merge needs a slot")
    return number


def _add_merge_concurrency(parser: argparse.ArgumentParser) -> None:
    parser.add_argument(
        "--merge-concurrency",
        type=_concurrency,
        default=DEFAULT_MERGE_CONCURRENCY,
        metavar="N",
        help="how many detail-merge calls one tree may have in flight; one is "
        "strictly sequential (default: %(default)s)",
    )


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="run_ocr_study.py",
        description="Run the OCR knowledge-tree study over a corpus manifest.",
        epilog=_epilog(),
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    commands = parser.add_subparsers(dest="command", required=True)

    corpus = commands.add_parser("corpus", help="corpus-level checks")
    corpus_commands = corpus.add_subparsers(dest="corpus_command", required=True)
    validate = corpus_commands.add_parser(
        "validate", help="hash, page range and modality of every document, plus engine probes"
    )
    _add_corpus(validate)
    _add_doc(validate)
    validate.set_defaults(handler=cmd_corpus_validate)

    render = commands.add_parser("render", help="render pages to PNGs for reading a gold off")
    _add_corpus(render)
    _add_doc(render)
    _add_pages(render)
    render.add_argument(
        "--out", required=True, help="parent directory of <out>/<doc_id>/page-NNNN.png"
    )
    render.add_argument("--dpi", type=int, default=200)
    render.add_argument(
        "--offset", type=int, default=0, help="the page offset the engine would use"
    )
    render.set_defaults(handler=cmd_render)

    gold = commands.add_parser("gold", help="check a gold file against the corpus")
    gold_commands = gold.add_subparsers(dest="gold_command", required=True)
    gold_check = gold_commands.add_parser(
        "check", help="how much of the document's page selection the gold covers"
    )
    _add_corpus(gold_check)
    gold_check.add_argument("--doc", required=True, help="the document the gold belongs to")
    gold_check.add_argument("--file", required=True, help="path to the gold JSON")
    gold_check.set_defaults(handler=cmd_gold_check)

    extract = commands.add_parser("extract", help="run one engine over the corpus pages")
    _add_corpus(extract)
    _add_doc(extract)
    _add_pages(extract)
    extract.add_argument("--engine", required=True, choices=list(ENGINE_IDS))
    extract.add_argument("--repeats", type=int, default=1, help="how many separate runs to store")
    extract.add_argument("--dpi", type=int, default=200, help="vision render DPI")
    extract.add_argument(
        "--offset", type=int, default=0, help="the page offset the engine is given"
    )
    extract.add_argument(
        "--allow-upload",
        action="store_true",
        help="permit a paid engine (vision) to upload pages to the configured endpoint",
    )
    extract.add_argument(
        "--resume", action="store_true", help="reuse a run of the identical request"
    )
    extract.add_argument("--label", help="free-form label recorded in every run manifest")
    warmth = extract.add_mutually_exclusive_group()
    warmth.add_argument("--cold", dest="warmth", action="store_const", const="cold")
    warmth.add_argument("--warm", dest="warmth", action="store_const", const="warm")
    extract.set_defaults(handler=cmd_extract, warmth=None)

    runs = commands.add_parser("runs", help="list the run directories already on disk")
    _add_corpus(runs)
    _add_doc(runs)
    runs.add_argument("--kind", choices=[KIND_EXTRACT, KIND_TREE])
    runs.set_defaults(handler=cmd_runs)

    run = commands.add_parser("run", help="build a tree from a stored artifact (no OCR)")
    _add_corpus(run)
    _add_doc(run)
    run.add_argument(
        "--profile",
        required=True,
        metavar="NAME",
        help=f"a profile name, or {ALL_COMPATIBLE!r} for the artifact engine's own table",
    )
    run.add_argument(
        "--extract-run",
        action="append",
        metavar="DOC_ID:RUN_ID",
        help="the extraction run to replay (repeatable, one per document)",
    )
    run.add_argument("--artifact", metavar="DIGEST", help="the artifact digest to replay")
    _add_llm_mode(run)
    _add_merge_concurrency(run)
    run.add_argument(
        "--allow-merge-upload", action="store_true", help="permit a live merge's gateway calls"
    )
    run.add_argument("--resume", action="store_true", help="reuse a run of the identical request")
    run.add_argument("--label", help="free-form label recorded in the run manifest")
    run.set_defaults(handler=cmd_run)

    compare = commands.add_parser("compare", help="compare runs of one family, in three formats")
    _add_corpus(compare)
    compare.add_argument("--family", required=True, choices=list(FAMILIES))
    compare.add_argument(
        "--run", action="append", required=True, metavar="DOC_ID:RUN_ID", help="repeatable"
    )
    compare.add_argument(
        "--gold",
        action="append",
        metavar="DOC_ID=PATH",
        help="gold for a document that does not declare one (repeatable)",
    )
    compare.add_argument(
        "--out", help="target directory (default: <root>/comparisons/<family>.<stamp>)"
    )
    compare.add_argument("--label", help="free-form label recorded in the report")
    compare.add_argument(
        "--dry-run", action="store_true", help="print the report without writing it"
    )
    compare.set_defaults(handler=cmd_compare)

    study = commands.add_parser("study", help="run extract, run and compare over the corpus")
    _add_corpus(study)
    _add_doc(study)
    _add_pages(study)
    study.add_argument("--families", required=True, help="comma-separated: " + ", ".join(FAMILIES))
    study.add_argument(
        "--engines", required=True, help="comma-separated: " + ", ".join(ENGINE_IDS)
    )
    study.add_argument(
        "--profile", action="append", metavar="NAME", help="profile to build (repeatable)"
    )
    study.add_argument("--repeats", type=int, default=1, help="how many runs per engine")
    study.add_argument(
        "--gold",
        action="append",
        metavar="DOC_ID=PATH",
        help="gold for a document that does not declare one (repeatable)",
    )
    _add_llm_mode(study)
    _add_merge_concurrency(study)
    study.add_argument(
        "--allow-upload", action="store_true", help="permit a paid engine's uploads (vision)"
    )
    study.add_argument(
        "--allow-merge-upload", action="store_true", help="permit a live merge's gateway calls"
    )
    study.add_argument("--label", help="free-form label recorded in every run manifest")
    study.add_argument("--dry-run", action="store_true", help="print the plan and run nothing")
    policy = study.add_mutually_exclusive_group()
    policy.add_argument(
        "--resume",
        dest="resume",
        action="store_true",
        help="reuse a run of the identical request already on disk (the default)",
    )
    policy.add_argument(
        "--fresh",
        dest="resume",
        action="store_false",
        help="run everything again in new run ids, whatever is already on disk",
    )
    study.set_defaults(handler=cmd_study, resume=True)

    return parser


def _epilog() -> str:
    engines = "\n".join(
        f"  {engine_id:<12} {_ENGINE_NOTES[engine_id]}" for engine_id in ENGINE_IDS
    )
    profiles = "\n".join(
        f"  {profile.name:<20} {profile.engine:<12} builder {profile.builder:<16} merge {profile.merge}"
        for profile in PROFILES.values()
    )
    families = "\n".join(f"  {name:<10} {_FAMILY_NOTES[name]}" for name in FAMILIES)
    return (
        "engines\n"
        f"{engines}\n\n"
        "profiles\n"
        f"{profiles}\n\n"
        "families\n"
        f"{families}\n\n"
        "Nothing leaves the machine unless a flag says so:\n"
        "  --allow-upload        a paid engine (vision) uploads pages to the configured endpoint\n"
        "  --allow-merge-upload  a live detail merge sends prompts to the LLM gateway\n"
        "  --merge-concurrency   only the merge's own calls: the limit is part of the\n"
        "                        run's identity, so a resume meets it only when it matches\n"
        "A study reuses a run only when it answers the identical request, succeeded and\n"
        "left every file it promises on disk; --fresh runs everything again instead.\n"
        "A run without gold is reported as not_measured, never as a number; a run whose\n"
        "tokens were not reported is a typed miss rather than a zero, and a corpus is\n"
        "never modified by a run.\n"
    )


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    try:
        return args.handler(args)
    except (RunnerError, CorpusError, GoldError, ArtifactStoreError) as exc:
        return _fail(exc)
    except ValueError as exc:
        # A validation failure -- pydantic's ``ValidationError`` is a ValueError --
        # is a configuration mistake with a message worth printing.
        return _fail(exc)


if __name__ == "__main__":
    raise SystemExit(main())
