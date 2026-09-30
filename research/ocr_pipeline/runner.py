"""The S4/S5 study runner: extract, build, replay, compare, and report.

This module is the only place that *runs* a study. The slices below it are
deliberately pure: adapters turn an engine's output into a neutral artifact,
builders turn an artifact into a tree, metrics score trees and transcripts,
usage adds the tokens a run spent. The runner decides what to run, records what
happened, and refuses to let a number outlive the evidence behind it.

What it refuses to do
---------------------
* **No network by accident.** A paid engine (vision) needs ``allow_upload``
  before the first byte leaves the machine, and a live detail merge needs
  ``allow_merge_upload``. Both checks run before any engine or client is built.
* **No overwriting.** A run directory is append-only: the manifest is written
  last, and a second attempt at the same request lands in ``-a2``, ``-a3``, …
  so a stochastic rerun can never be confused with the run it replaces.
* **No unranked results in a ranking.** A tree built from an artifact the engine
  could not read, or by a merge that rebuilt nothing, is recorded as
  ``quality: not_measured`` and a comparison keeps it out of the paired deltas.
* **No raw text in a report.** Transcripts stay in the artifacts and the gold
  reference; ``report.build_report`` raises if any slips in.

Storage layout::

    <root>/<doc_id>/artifacts/<digest>.json      # content-addressed extractions
    <root>/<doc_id>/runs/<run_id>/manifest.json  # what ran, and what happened
    <root>/<doc_id>/runs/<run_id>/tree.json      # a built tree, when there is one
    <root>/<doc_id>/runs/<run_id>/metrics.json   # fidelity, chunks, signatures
    <root>/comparisons/<family>.<stamp>/report.{json,csv,md}

There is no database and no index: a run directory plus the corpus manifest is
everything a reader needs to re-derive a number.
"""

from __future__ import annotations

import asyncio
import hashlib
import inspect
import json
import os
import platform
import re
import statistics
import subprocess
import time
from collections.abc import Callable, Iterator, Mapping, Sequence
from contextlib import contextmanager
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any, TypeGuard

from arango_client import (
    KnowledgeNode,
    ParagraphNode,
    StructureNode,
    TableNode,
    serialize_tree_to_dict,
)

from .alignment import heading_boundary_metrics
from .corpus import (
    CorpusError,
    load_manifest,
    resolve_path,
    validate_document,
    verify_artifact,
)
from .engines import (
    ENGINE_IDS,
    ISOLATED_ENGINES,
    STATUS_OK,
    STATUS_UNSUPPORTED_MODALITY,
    USAGE_FIELDS,
    EngineArtifact,
    EngineBlock,
    EngineNotAvailableError,
    EnginePage,
    content_pages,
    continuity_breaks,
    load_engine,
    reset_continuity,
)
from .engines.worker import (
    IsolatedWorkerError,
    engine_interpreter,
    engine_probe,
    isolated_extract,
)
from .metrics import MEASURED, NOT_MEASURED, chunk_metrics, corpus_text_metrics
from .models import (
    DocumentSpec,
    ExtractionArtifact,
    Manifest,
    semantic_fingerprint,
)
from .profiles import PROFILES, Profile, build, get_profile
from .reference import GoldError, GoldReference, coverage, load_gold, metrics_document
from .report import build_report, to_csv, to_json, to_markdown
from .resources import measure_hosted_stage, measure_process_tree
from .store import ArtifactStore, ArtifactStoreError, write_once
from .tree_builders import page_signature, semantic_signature, text_fidelity, validate_tree
from .usage import artifact_usage, failure_usage, paired_usage, run_usage

__all__ = [
    "COMPARISON_VERSION",
    "DEFAULT_MERGE_CONCURRENCY",
    "DEFAULT_ROOT",
    "FAMILIES",
    "KIND_EXTRACT",
    "KIND_TREE",
    "LLM_MODES",
    "PAID_ENGINES",
    "RUN_MANIFEST_VERSION",
    "RUN_FAILED",
    "RUN_NOT_AVAILABLE",
    "RUN_OK",
    "MergeAbortedError",
    "MergeNotAllowedError",
    "RunExistsError",
    "RunnerError",
    "StudyRunner",
    "UploadNotAllowedError",
    "engine_pages",
    "load_corpus",
    "merge_tree",
    "parse_page_selection",
    "primary_profile",
    "profiles_for_engine",
    "repo_provenance",
    "tree_chunks",
]

RUN_MANIFEST_VERSION = 1
"""Layout version of ``runs/<run_id>/manifest.json``."""

COMPARISON_VERSION = 1
"""Layout version of a comparison manifest and its report."""

KIND_EXTRACT = "extract"
KIND_TREE = "tree"

RUN_OK = "ok"
RUN_FAILED = "failed"
RUN_NOT_AVAILABLE = "not_available"
"""Run-level statuses. ``not_available`` means the engine could not run at all."""

LLM_MODES = ("none", "live")
"""How a profile's detail merge is allowed to run."""

FAMILIES = ("pipeline", "ocr", "chunk")
"""Comparison families: full path, extraction text, and chunking profile."""

PAID_ENGINES = ("vision",)
"""Engines that send pages to a hosted endpoint of somebody else's machine."""

CHUNK_ARMS = ("vision_direct", "vision_atomic_merge")
"""The chunk family's two arms: one artifact, grouped, then merged.

``vision_atomic`` is the control the merged tree was built from, not an arm: a
merged tree *is* an atomic tree whose details a model rewrote, so pairing those
two would measure the merge against its own input instead of against a reading
that grouped the pages the way a chunker would.
"""

PIPELINE_ARMS = ("vision_direct", "marker_llm")
"""The pipeline family's two arms: the vision tree and a real Marker tree.

The family exists to compare what the two engines read: every other tree of the
same document is a reading of one of them, so pairing those would measure an
engine against itself. Only the two engines' own trees are an arm of it.
"""

_CONTINUITY_COUNTER = "cross_page_continues"
"""What the runtime parser calls a page that carried the group before it.

The parser keeps its normalization counters in a ``Counter``, which drops a key
it never incremented -- so an absent continuity counter reads exactly like one
nobody looked for. A tree's signature seeds it at zero so a replay that refused a
claim says so, and the builder's own count stands whenever it made one.
"""

_PUBLIC_CONFIG_FIELDS = (
    ("model", "model"),
    ("endpoint_host", "host"),
    ("image_detail", "image_detail"),
    ("temperature", "temperature"),
    ("max_tokens", "max_tokens"),
    ("reasoning_effort", "reasoning_effort"),
    ("max_blocks_per_page", "max_blocks_per_page"),
)
"""The only engine configuration a manifest may carry, and where to read it.

A resolved request configuration is part of the request: a changed token ceiling
or temperature is a different request, so the key that decides whether a run can
be resumed has to cover these fields. The key, the endpoint URL and the headers
are deliberately absent — they are credentials, not request parameters.
"""

DEFAULT_ROOT = Path("data/research/ocr_pipeline")

DEFAULT_MERGE_CONCURRENCY = 4
"""How many merge calls one research tree may have in flight at once.

The runtime default stays with ``RefineConfig`` (20); this is the research
default, low enough that a study's own merge does not crowd out the machine
that is measuring it, and a study may override it per run.
"""

STAGE_FAILED = "failed"
STAGE_STATUSES = (MEASURED, NOT_MEASURED, STAGE_FAILED)

FURNITURE = ("page_header", "page_footer")

_DETAIL_ATTRS = ("text", "extracted_text")


# --------------------------------------------------------------------------- #
# errors
# --------------------------------------------------------------------------- #


class RunnerError(RuntimeError):
    """A study was asked to do something it must not do."""


class RunExistsError(RunnerError):
    """A run directory already holds a different run under the same id."""


class UploadNotAllowedError(RunnerError):
    """A paid engine was requested without an explicit upload flag."""


class MergeNotAllowedError(RunnerError):
    """A live detail merge was requested without an explicit upload flag."""


class MergeAbortedError(RunnerError):
    """A tree merge stopped before it finished, carrying what it had spent.

    The sections that finished before the failure were called and answered, so
    the counters they folded travel with the error instead of dying with the
    walk: a run that failed still reports the calls it made. What the failing
    section spent itself is not in there -- its refinement never returned, so its
    tokens are unobservable, and the counters never present a subtotal as the
    whole.
    """

    def __init__(self, cause: BaseException, counters: Mapping[str, Any]) -> None:
        super().__init__(f"the merge stopped: {type(cause).__name__}")
        self.cause = cause
        self.counters = dict(counters)


# --------------------------------------------------------------------------- #
# small helpers
# --------------------------------------------------------------------------- #


def _now() -> str:
    return datetime.now(UTC).isoformat(timespec="seconds")


def _stamp() -> str:
    return datetime.now(UTC).strftime("%Y%m%dT%H%M%SZ")


def _canonical(value: Any) -> str:
    return json.dumps(value, sort_keys=True, ensure_ascii=False, separators=(",", ":"))


def _sha256_text(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def _sha256_json(value: Any) -> str:
    return _sha256_text(_canonical(value))


def _read_json(path: Path, *, what: str) -> Any:
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except OSError as exc:
        raise RunnerError(f"cannot read {what} {path}: {exc}") from exc
    except json.JSONDecodeError as exc:
        raise RunnerError(f"{what} {path} is not valid JSON: {exc}") from exc


def _read_optional_json(path: Path) -> Any | None:
    return None if not path.is_file() else _read_json(path, what="run file")


def _write_json_once(path: Path, payload: Any) -> None:
    """Write a JSON file that must never be rewritten with different bytes."""
    text = json.dumps(payload, indent=2, ensure_ascii=False, sort_keys=True) + "\n"
    try:
        write_once(path, text.encode("utf-8"))
    except ArtifactStoreError as exc:
        raise RunExistsError(f"{path} already exists with different bytes: {exc}") from exc


# --------------------------------------------------------------------------- #
# provenance
# --------------------------------------------------------------------------- #


def _git_state(cwd: Path) -> tuple[str | None, bool | None]:
    """Commit and dirty flag, or ``(None, None)`` outside a repository."""
    try:
        commit = subprocess.run(
            ["git", "rev-parse", "HEAD"], cwd=cwd, capture_output=True, text=True, check=False
        )
        if commit.returncode != 0:
            return None, None
        status = subprocess.run(
            ["git", "status", "--porcelain"],
            cwd=cwd,
            capture_output=True,
            text=True,
            check=False,
        )
    except OSError:
        return None, None
    dirty = status.stdout.strip() if status.returncode == 0 else None
    return commit.stdout.strip() or None, (bool(dirty) if dirty is not None else None)


def repo_provenance(cwd: str | Path | None = None, *, label: str | None = None) -> dict[str, Any]:
    """The environment a run must be reproducible in: commit, dirty flag, interpreter.

    Recorded in every manifest so a number can always be traced to the code that
    produced it. ``label`` names the caller's own environment when a study runs
    somewhere the repository itself cannot describe.
    """
    root = Path.cwd() if cwd is None else Path(cwd)
    commit, dirty = _git_state(root)
    return {
        "git_commit": commit,
        "git_dirty": dirty,
        "python": platform.python_version(),
        "platform": platform.platform(),
        "label": label,
    }


# --------------------------------------------------------------------------- #
# page selection
# --------------------------------------------------------------------------- #


def parse_page_selection(text: str, *, max_pages: int = 10_000) -> tuple[int, ...]:
    """Parse ``"0-9,20"`` into 0-based page indices, strictly increasing.

    The manifest speaks in 0-based physical indices; the CLI accepts the same
    vocabulary so a page number in a report can always be checked by hand.
    """
    selection: list[int] = []
    for part in text.replace(" ", "").split(","):
        if not part:
            continue
        match = re.fullmatch(r"(\d+)(?:-(\d+))?", part)
        if match is None:
            raise RunnerError(f"invalid page selection {part!r}; expected N or N-M")
        first = int(match.group(1))
        last = int(match.group(2)) if match.group(2) is not None else first
        if last < first:
            raise RunnerError(f"invalid page range {part!r}: end before start")
        selection.extend(range(first, last + 1))
    if not selection:
        raise RunnerError(f"page selection {text!r} is empty")
    if len(set(selection)) != len(selection):
        raise RunnerError(f"page selection {text!r} repeats a page")
    if len(selection) > max_pages:
        raise RunnerError(f"page selection {text!r} asks for {len(selection)} pages")
    return tuple(sorted(selection))


def load_corpus(path: str | Path) -> tuple[Manifest, Path]:
    """Load a corpus manifest and the directory its relative paths resolve against."""
    manifest_path = Path(path)
    return load_manifest(manifest_path), manifest_path.parent


def profiles_for_engine(engine_id: str) -> tuple[Profile, ...]:
    """Every profile that reads the given engine, in table order."""
    return tuple(profile for profile in PROFILES.values() if profile.engine == engine_id)


def primary_profile(engine_id: str) -> Profile:
    """The profile a study compares engines with when it needs exactly one.

    The first profile the table lists for an engine: the one that consumes its
    most faithful output (``vision_direct``, ``marker_llm``, ``native_atomic``…).
    """
    candidates = profiles_for_engine(engine_id)
    if not candidates:
        raise RunnerError(
            f"engine {engine_id!r} has no profile; expected one of {sorted(PROFILES)}"
        )
    return candidates[0]


# --------------------------------------------------------------------------- #
# stages
# --------------------------------------------------------------------------- #


def _local_stage() -> Any:
    """Measure a stage that runs locally, by sampling this process tree."""
    return measure_process_tree(os.getpid())


def _failed_worker_observation(
    error: BaseException,
    observed: dict[str, Any],
    worker: dict[str, Any] | None,
) -> tuple[dict[str, Any], dict[str, Any] | None]:
    """What a failed worker still accounts for: its resources and its report.

    A worker that could not answer -- it crashed, it overran its timeout, its
    engine was unavailable -- carries what was observed before it died, so the
    failed run shows the memory it spent rather than nothing at all. An error
    raised without those fields says nothing, and what is already known stands.
    """
    resources = getattr(error, "resources", None)
    report = getattr(error, "worker", None)
    return (
        dict(resources) if resources else observed,
        dict(report) if report else worker,
    )


@contextmanager
def _stage(stages: dict[str, Any], name: str) -> Iterator[dict[str, Any]]:
    """Time one stage and record it, including when the body raises."""
    entry: dict[str, Any] = {"status": MEASURED, "seconds": None, "reason": None}
    stages[name] = entry
    started = time.perf_counter()
    try:
        yield entry
    except BaseException as exc:
        entry["status"] = STAGE_FAILED
        entry["reason"] = type(exc).__name__
        raise
    finally:
        entry["seconds"] = round(time.perf_counter() - started, 6)


# --------------------------------------------------------------------------- #
# artifact -> engine pages, for replay
# --------------------------------------------------------------------------- #


def _engine_usage(usage: Any) -> dict[str, int]:
    """A stored usage projected onto the counters the neutral page accepts.

    An artifact may record more than a neutral engine page has a slot for, and
    ``EnginePage`` refuses any key outside ``USAGE_FIELDS``. The replay carries
    what the protocol has a field for, so the inverse stays exact for every
    counter that protocol defines.
    """
    values = dict(usage.model_dump())
    return {
        name: values[name]
        for name in USAGE_FIELDS
        if isinstance(values.get(name), int) and not isinstance(values.get(name), bool)
    }


def engine_pages(artifact: ExtractionArtifact) -> tuple[EnginePage, ...]:
    """Rebuild the neutral pages a stored artifact replays.

    ``ExtractionArtifact.from_engine`` is the forward translation (``kind``
    becomes ``raw_kind``, blocks are numbered by position); this is its inverse,
    and it is exact: a tree built from these pages is the tree the live run
    built, which is what makes a replay make no OCR call.
    """
    return tuple(
        EnginePage(
            page_index=page.page_index,
            request_page=page.request_page,
            blocks=tuple(
                EngineBlock(
                    kind=block.raw_kind,
                    normalized_kind=block.normalized_kind,
                    text=block.text,
                    marker=block.marker,
                    depth=block.depth,
                    group=block.group,
                    bbox=block.bbox,
                    source_id=block.source_id,
                )
                for block in page.blocks
            ),
            running=page.running,
            printed_folio=page.printed_folio,
            continues_previous=page.continues_previous,
            status=page.status,
            usage=_engine_usage(page.usage) if page.usage is not None else None,
        )
        for page in artifact.pages
    )


# --------------------------------------------------------------------------- #
# tree helpers
# --------------------------------------------------------------------------- #


def _structure_nodes(root: KnowledgeNode) -> Iterator[StructureNode]:
    """Every structure node under ``root``, parents before children."""
    stack: list[KnowledgeNode] = [root]
    while stack:
        node = stack.pop()
        if not isinstance(node, StructureNode):
            continue
        yield node
        stack.extend(reversed(node.children))


def _node_title(node: KnowledgeNode) -> str:
    attributes = getattr(node, "attributes", None)
    title = getattr(attributes, "title", None)
    return title.strip() if isinstance(title, str) else ""


def _detail_text(node: KnowledgeNode) -> str:
    attributes = getattr(node, "attributes", None)
    for name in _DETAIL_ATTRS:
        value = getattr(attributes, name, None)
        if isinstance(value, str) and value.strip():
            return value
    return ""


def tree_chunks(root: KnowledgeNode, *, doc_id: str, page_offset: int = 0) -> list[dict[str, Any]]:
    """The retrieval-sized pieces a tree would hand to a reader.

    A chunk is a detail node (paragraph or table): that is what the merge step
    produces and what a chunking metric is about. Pages are converted from the
    request-page space the builder recorded back to 0-based physical indices,
    which is the index space ``chunk_metrics`` documents.
    """
    chunks: list[dict[str, Any]] = []
    for node in _structure_nodes(root):
        for child in node.children:
            if not isinstance(child, (ParagraphNode, TableNode)):
                continue
            metadata = getattr(child, "metadata", None)
            page = getattr(metadata, "page", None)
            pages = [page - 1 - page_offset] if isinstance(page, int) else []
            chunks.append(
                {
                    "chunk_id": str(getattr(child, "id", "")),
                    "doc_id": doc_id,
                    "text": _detail_text(child),
                    "pages": pages,
                }
            )
    return chunks


def _fidelity_block(fidelity: Any) -> dict[str, Any]:
    """A text-fidelity observation as a metric dict: how much text the tree lost."""
    measured = fidelity.artifact_chars > 0
    return {
        "status": MEASURED if measured else NOT_MEASURED,
        "reason": None if measured else "empty_artifact",
        "value": fidelity.omitted_ratio,
        "numerator": fidelity.omitted if measured else None,
        "denominator": fidelity.artifact_chars if measured else None,
        "unit": "ratio",
        "denominator_unit": "characters",
        "artifact_chars": fidelity.artifact_chars,
        "tree_chars": fidelity.tree_chars,
        "added_chars": fidelity.added,
        "reordered_chars": fidelity.reordered,
        "exact": fidelity.exact,
    }


GOLD_DEPENDENT_METRICS = frozenset(
    {"cer_micro", "wer_micro", "cer_macro", "wer_macro", "heading_f1", "boundary_f1"}
)
"""Pair deltas that read the reference, so they need one reference on both sides.

Everything else a pair measures — time, fidelity, chunking — describes the run
itself, and two runs scored against different golds are still comparable on it.
"""


def _seconds_metric(stages: Any) -> dict[str, Any] | None:
    """A run's measured time as a metric: the sum of its stages, or a typed miss.

    A stage that failed is still time that was spent, so a stage that reported a
    number counts whatever its status. A stage with no number leaves the run's
    total unknown rather than quietly short, and the known subtotal travels with
    the miss so a reader can see how much of the run was timed.
    """
    if not isinstance(stages, dict) or not stages:
        return None
    reported = [entry.get("seconds") for entry in stages.values() if isinstance(entry, dict)]
    known = [value for value in reported if isinstance(value, (int, float))]
    if not known:
        return {"status": NOT_MEASURED, "reason": "no_stage_timings", "value": None}
    if len(known) != len(reported):
        return {
            "status": NOT_MEASURED,
            "reason": "incomplete_stage_timings",
            "value": None,
            "known_seconds": sum(known),
            "timed_stages": len(known),
            "stages": len(reported),
        }
    total = sum(known)
    return {
        "status": MEASURED,
        "reason": None,
        "value": total,
        "numerator": total,
        "denominator": 1,
        "unit": "seconds",
        "denominator_unit": "runs",
    }


def _resource_metric(resources: Any, key: str) -> dict[str, Any] | None:
    """One resource reading of a run as a pair metric, or ``None`` when absent.

    A reading that is merely unreadable -- a hosted backend's memory cannot be
    observed from here -- travels as the miss it is, so the delta abstains with
    that reason instead of reading the silence as a zero.
    """
    if not isinstance(resources, dict):
        return None
    reading = resources.get(key)
    if not isinstance(reading, dict):
        return None
    return {
        "status": reading.get("status"),
        "reason": reading.get("reason"),
        "value": reading.get("value"),
        "unit": reading.get("unit"),
        "numerator": reading.get("value"),
        "denominator": 1,
        "denominator_unit": "runs",
        "estimate": bool(reading.get("estimate")),
    }


def _run_resources(resources: Any) -> dict[str, Any] | None:
    """A run's CPU and peak memory as pair metrics, or ``None`` when it has none.

    The readings travel as its stage reported them -- one observation per run
    today -- and a peak stays the peak it is: stages that each held a gigabyte
    never held two, so this is what a pair may compare and never a sum.
    """
    parts = {key: _resource_metric(resources, key) for key in ("cpu", "rss_peak")}
    return parts if any(parts.values()) else None


def _delta_miss(reason: str, **extra: Any) -> dict[str, Any]:
    """A delta that was not measured, with the reason it was not."""
    return {"status": NOT_MEASURED, "reason": reason, "value": None, **extra}


DELTA_METRICS = (
    "cer_micro",
    "wer_micro",
    "cer_macro",
    "wer_macro",
    "heading_f1",
    "boundary_f1",
    "fidelity_omitted",
    "seconds",
    "cpu_seconds",
    "rss_peak_bytes",
)
"""The metrics a pair measures, in the order a report reads them.

The names are the units' own: ``cer_micro`` pools a run's pages,
``cer_macro`` is its per-document mean, ``fidelity_omitted`` is the share of
the artifact's characters the tree lost, ``seconds`` is wall time, and the last
two are what a locally measured stage used: its CPU seconds, and the largest
peak memory of its stages rather than their sum. A chunk count is deliberately
not among them: it is structure, not accuracy, and it stays the chunk family's
own headline delta. Tokens are not among them either -- they are the pair's own
``usage`` -- so the arm that read more can never be read as the worse one.
"""


def _row_metrics(row: Mapping[str, Any]) -> dict[str, Any]:
    """Every metric dict one comparison row can offer a pair delta, by name."""
    text = (row.get("text_metrics") or {}).get("rates") or {}
    kinds = (row.get("structure_metrics") or {}).get("kinds") or {}
    resources = row.get("resources") or {}
    return {
        "cer_micro": (text.get("cer") or {}).get("micro"),
        "wer_micro": (text.get("wer") or {}).get("micro"),
        "cer_macro": (text.get("cer") or {}).get("macro"),
        "wer_macro": (text.get("wer") or {}).get("macro"),
        "heading_f1": (kinds.get("heading") or {}).get("f1"),
        "boundary_f1": (kinds.get("boundary") or {}).get("f1"),
        "fidelity_omitted": row.get("fidelity"),
        "seconds": row.get("seconds"),
        "cpu_seconds": resources.get("cpu"),
        "rss_peak_bytes": resources.get("rss_peak"),
    }


def _documents_of(items: Sequence[tuple[str, Mapping[str, Any]]]) -> dict[str, list[Any]]:
    """Group items by document, so repeats can be folded inside their own."""
    grouped: dict[str, list[Any]] = {}
    for doc_id, item in items:
        grouped.setdefault(doc_id, []).append(item)
    return grouped


def _mean_of(metrics: Sequence[Any]) -> dict[str, Any] | None:
    """The mean of measured metric dicts, or ``None`` when none was measured.

    Counts are averaged along with the value, so a document measured twice
    contributes one observation of its own size rather than two.
    """
    measured = [
        metric
        for metric in metrics
        if isinstance(metric, dict) and metric.get("status") == MEASURED
    ]
    if not measured:
        return None
    count = len(measured)
    mean = {
        "value": sum(metric["value"] for metric in measured) / count,
        "unit": measured[0].get("unit"),
        "denominator_unit": measured[0].get("denominator_unit"),
        "numerator": sum(metric.get("numerator") or 0 for metric in measured) / count,
        "denominator": sum(metric.get("denominator") or 0 for metric in measured) / count,
    }
    if any("estimate" in metric for metric in measured):
        # A mean of sampled readings is a sampled reading: the flag travels with
        # the quantity so a reader never meets a sampled figure as an exact one.
        mean["estimate"] = any(bool(metric.get("estimate")) for metric in measured)
    return mean


_MIRRORED_FIELDS = {"a": "b", "a_numerator": "b_numerator", "a_denominator": "b_denominator"}
"""The keys of a delta that name a side, and the key naming the other one."""


def _oriented(pair: Mapping[str, Any], metric: Any) -> Any:
    """One pair delta in the pair's canonical direction.

    A study alternates its engine order between repeats, so the same pair
    arrives as (a, b) in one repeat and (b, a) in the next. Averaging the raw
    deltas would cancel a real difference to zero, so a pair whose rows are in
    the other order has its delta read backwards, both sides swapped with it. A
    delta that names no sides has only its value to turn around.
    """
    participants = [str(part) for part in pair.get("participants") or []]
    if not isinstance(metric, dict) or participants == sorted(participants):
        return metric
    if metric.get("status") != MEASURED:
        return metric
    flipped = {**metric, "value": -metric["value"]}
    for name, other in _MIRRORED_FIELDS.items():
        if name in metric and other in metric:
            flipped[name], flipped[other] = metric[other], metric[name]
    return flipped


def _document_statistics(
    per_document: Mapping[str, Mapping[str, Any]], *, reason: str
) -> dict[str, Any]:
    """One metric over documents: mean, median, variation and significance.

    ``per_document`` holds one already-averaged reading per document, so the
    document is the statistical unit. A single document has no spread and says
    so. Significance is never claimed: this study reports descriptive statistics
    and runs no test, so the field is always a typed miss naming that.
    """
    if not per_document:
        return {"status": NOT_MEASURED, "reason": reason, "value": None, "documents": 0}
    values = [reading["value"] for reading in per_document.values()]
    sample = next(iter(per_document.values()))
    variation = statistics.pstdev(values) if len(values) > 1 else None
    block = {
        "status": MEASURED,
        "reason": None,
        "value": statistics.fmean(values),
        "unit": sample.get("unit"),
        "denominator_unit": sample.get("denominator_unit"),
        "documents": len(values),
        "median": statistics.median(values),
        "variation": variation,
        "variation_reason": None if variation is not None else "single_document",
        "significance": {
            "status": NOT_MEASURED,
            "reason": "no_test_run",
            "value": None,
            "documents": len(values),
        },
    }
    if "estimate" in sample:
        block["estimate"] = bool(sample["estimate"])
    return block


def _pooled_over_documents(
    per_document: Mapping[str, Mapping[str, Any]], *, reason: str
) -> dict[str, Any]:
    """Micro average over documents: the pooled edits over the pooled bases.

    The counts are summed rather than averaged, which is what makes this the
    micro figure: a long document weighs more than a short one, exactly as its
    share of the reference characters says it should.
    """
    if not per_document:
        return {"status": NOT_MEASURED, "reason": reason, "value": None, "documents": 0}
    sample = next(iter(per_document.values()))
    numerator = sum(reading["numerator"] for reading in per_document.values())
    denominator = sum(reading["denominator"] for reading in per_document.values())
    if not denominator:
        return {
            "status": NOT_MEASURED,
            "reason": "no_reference_characters",
            "value": None,
            "documents": len(per_document),
        }
    return {
        "status": MEASURED,
        "reason": None,
        "value": numerator / denominator,
        "numerator": numerator,
        "denominator": denominator,
        "unit": sample.get("unit"),
        "denominator_unit": sample.get("denominator_unit"),
        "documents": len(per_document),
    }


async def merge_tree(
    root: StructureNode,
    *,
    llm_client: Any,
    refine_config: Any,
    max_concurrent_sections: int = 1,
) -> dict[str, Any]:
    """Merge every section's detail children, bottom-up, and apply the result.

    This mirrors the runtime traversal: post-order, one merge call per section,
    ``children`` replaced with the refined list. Three things differ on purpose.
    ``section_summary`` is reported as data and never attached to a node, so a
    merge-produced summary cannot leak into a scored tree; the spend counters are
    summed from what the refiner measured at the gateway seam, never invented for
    it; and sibling subtrees are walked at the same time, up to the limit the
    caller allows.

    ``max_concurrent_sections`` bounds the calls *in flight*, not the walk: every
    subtree of a section is started at once, and the shared semaphore inside the
    refiner -- held around the call alone, never across a recursion -- is what
    keeps at most that many calls open. One slot therefore still merges one
    section at a time, in input order, while the sections that are not calling
    wait their turn. A parent is visited after its children have returned, so its
    merge reads the children its subtrees already rebuilt.

    ``attempts`` and ``latency_seconds`` are always counted, so a merge whose
    gateway exposed nothing still reports the calls it made and the time it took.
    ``wall_seconds`` is the walk's own elapsed time: with calls in flight it is
    what the merge actually took, while ``latency_seconds`` sums the logical calls
    and grows with them, so the two apart say whether the limit bought overlap.
    The token counts are all or nothing: a section that made a call but could not
    state its tokens turns the tree's counts into ``None`` and hands its
    ``usage_unknown_reason`` along, because the sections that did answer would
    otherwise be a subtotal presented as the whole. A skipped merge made no call,
    so it is a measured zero. The research merge records prompt and completion
    counts, and nothing else the provider might have counted.

    Sections are folded into the tree in document order, not in the order they
    answer, so two walks of one tree agree on every counter. A walk that fails
    takes its siblings down with it and raises :class:`MergeAbortedError` rather
    than returning counters that would read as a merge that finished; a walk that
    is cancelled stays cancelled.

    Returns the counters a run manifest records.
    """
    from workflows.enrich_knowledge_tree.detail_refiner import DetailRefiner

    refiner = DetailRefiner(llm_client)
    semaphore = asyncio.Semaphore(max(1, max_concurrent_sections))
    counters = _merge_counters()

    async def visit(node: StructureNode, breadcrumb: list[str], sink: dict[str, Any]) -> None:
        """Merge one section into ``sink``: its subtrees first, then itself."""
        children = [child for child in node.children if isinstance(child, StructureNode)]
        subtotals = [_merge_counters() for _ in children]

        async def walk(child: StructureNode, subtotal: dict[str, Any]) -> None:
            title = _node_title(child)
            await visit(child, breadcrumb + ([title] if title else []), subtotal)

        tasks = [
            asyncio.create_task(walk(child, subtotal))
            for child, subtotal in zip(children, subtotals, strict=True)
        ]
        if tasks:
            try:
                await asyncio.gather(*tasks)
            except BaseException as exc:
                for task in tasks:
                    task.cancel()
                # Await them before handing the failure on: a cancelled sibling
                # must be finished, or its call outlives the merge that made it.
                await asyncio.gather(*tasks, return_exceptions=True)
                for subtotal in subtotals:
                    _fold_subtree(sink, subtotal)
                if isinstance(exc, asyncio.CancelledError):
                    raise
                raise MergeAbortedError(exc, sink) from exc
            for subtotal in subtotals:
                _fold_subtree(sink, subtotal)

        before = _children_text(node)
        refinement = await refiner.refine_details(
            node=node, breadcrumb=breadcrumb, refine_config=refine_config, semaphore=semaphore
        )
        sink["sections"] += 1
        sink["original_details"] += refinement.original_detail_count
        sink["refined_details"] += refinement.refined_detail_count
        sink["llm_calls"] += refinement.llm_calls
        _fold_merge_spend(sink, refinement)
        if refinement.section_summary:
            sink["section_summaries"] += 1
        if refinement.merge_skipped:
            sink["sections_skipped"] += 1
            return
        sink["sections_merged"] += 1
        if refinement.merge_fell_back:
            # The merge ran and kept the original details — no call completed, or
            # the answer named nothing the refiner could refine: a real attempt,
            # not a skip.
            sink["sections_fallback"] += 1
        node.children = refinement.children
        if _children_text(node) != before:
            sink["changed_sections"] += 1

    started = time.perf_counter()
    await visit(root, [_node_title(root)] if _node_title(root) else [], counters)
    counters["wall_seconds"] = time.perf_counter() - started
    return counters


_COUNTED_SECTIONS = (
    "sections",
    "sections_merged",
    "sections_skipped",
    "sections_fallback",
    "changed_sections",
    "llm_calls",
    "original_details",
    "refined_details",
    "section_summaries",
)

# The counts one serving model's record carries into the research report.
_MODEL_COUNTS = ("prompt_tokens", "completion_tokens")


def _merge_counters() -> dict[str, Any]:
    """The merge counters a tree accumulates, at their zero state.

    ``wall_seconds`` is not a sum: it belongs to the walk that returned these
    counters, and the trees folded into it keep their own. ``serving_models``
    starts empty -- a merge that made no call was served by nobody -- and turns
    ``None`` the moment a call whose tokens could not be stated makes the split
    unknown along with the counts it splits.
    """
    return {
        **dict.fromkeys(_COUNTED_SECTIONS, 0),
        "attempts": 0,
        "prompt_tokens": 0,
        "completion_tokens": 0,
        "serving_models": [],
        "latency_seconds": 0.0,
        "wall_seconds": 0.0,
        "usage_unknown_reason": None,
    }


def _stated_model(value: Any) -> str | None:
    """A model name an answer stated, or ``None`` when it stated none.

    A blank name is no name: nothing can be looked up under it, so it must not be
    published as the model that served a call.
    """
    return value if isinstance(value, str) and value else None


def _stated_count(value: Any) -> int | None:
    """One stated token count, or ``None`` when nothing stated one."""
    return value if isinstance(value, int) and not isinstance(value, bool) else None


def _add_count(total: Any, value: Any) -> int | None:
    """Two stated counts added, or unknown when either side stated none.

    Sticky on purpose: a count that is already unknown cannot be recovered by a
    later record that did answer.
    """
    if _stated_count(total) is None or _stated_count(value) is None:
        return None
    return total + value


def _add_serving_models(counters: dict[str, Any], records: Any) -> None:
    """Add one spend's serving-model records to the tree's running split.

    One record per model, in the order the models were first credited, so a merge
    served by two models keeps each one's own tokens rather than the first one's
    alone. A record that named no model is kept under ``None``: naming one from
    the model the request asked for would credit the tokens to a name that is not
    evidence of who served them.

    The counts add up the way the tree's own do: a count any record leaves
    unknown stays unknown rather than summing the records that did answer.
    """
    known = counters.get("serving_models")
    if not isinstance(known, list) or not isinstance(records, list):
        return
    by_model = {record["model"]: record for record in known}
    for record in records:
        if not isinstance(record, Mapping):
            continue
        usage = record.get("usage")
        if not isinstance(usage, Mapping):
            continue
        name = _stated_model(record.get("model"))
        target = by_model.get(name)
        if target is None:
            target = {"model": name, "usage": {}}
            by_model[name] = target
            known.append(target)
        counted = target["usage"]
        for key in _MODEL_COUNTS:
            # A fresh record takes the first count it is given; adding to a zero
            # would turn a count that was never stated into a measured zero.
            counted[key] = (
                _add_count(counted[key], usage.get(key))
                if key in counted
                else _stated_count(usage.get(key))
            )


def _spend_of(spend: Any, name: str) -> Any:
    """One field of a section's spend, from a refiner result or a folded subtree."""
    if isinstance(spend, Mapping):
        return spend.get(name)
    return getattr(spend, name)


def _fold_subtree(counters: dict[str, Any], subtree: Mapping[str, Any]) -> None:
    """Add one subtree's counters to its parent's, in document order.

    The sections are added as they are and the spend by :func:`_fold_merge_spend`,
    so a subtree's tokens are folded by the same rule a section's are. The
    subtree's own ``wall_seconds`` stays with the subtree: a walk's elapsed time
    is not a quantity that adds up.
    """
    for name in _COUNTED_SECTIONS:
        counters[name] += subtree.get(name) or 0
    _fold_merge_spend(counters, subtree)


def _fold_merge_spend(counters: dict[str, Any], spend: Any) -> None:
    """Add one section's or subtree's spend to the tree's running merge total.

    Attempts and latency are measured facts and are always counted, even when the
    gateway stated no tokens. The token counts are all or nothing: once a section
    that called cannot state its tokens, the tree's counts become ``None`` and
    keep the first reason why, because the sections that did answer would
    otherwise be a subtotal presented as the whole. A skipped merge made no call,
    so it is a measured zero and neither adds nor poisons anything.
    """
    counters["attempts"] += _spend_of(spend, "attempts") or 0
    counters["latency_seconds"] += _spend_of(spend, "latency_seconds") or 0.0
    if not _spend_of(spend, "attempts"):
        return
    prompt = _spend_of(spend, "prompt_tokens")
    if counters["prompt_tokens"] is None or prompt is None:
        counters["prompt_tokens"] = None
        counters["completion_tokens"] = None
        # The split is only as observable as the counts it splits.
        counters["serving_models"] = None
        counters["usage_unknown_reason"] = counters["usage_unknown_reason"] or _spend_of(
            spend, "usage_unknown_reason"
        )
    else:
        counters["prompt_tokens"] += prompt
        counters["completion_tokens"] += _spend_of(spend, "completion_tokens") or 0
        _add_serving_models(counters, _spend_of(spend, "serving_models"))


def _merge_limit(refine: Any) -> int:
    """How many merge calls one tree may have in flight, from its own config.

    The limit is a config field rather than an argument of the walk, so the run
    that fingerprints the configuration and the run that walks the tree read the
    same number. A configuration that states none gets one section at a time:
    an unbounded default is not a safer one.
    """
    limit = getattr(refine, "max_concurrent_sections", None)
    if isinstance(limit, bool) or not isinstance(limit, int) or limit < 1:
        return 1
    return limit


def _children_text(node: StructureNode) -> str:
    return "\n".join(_detail_text(child) for child in node.children)


# --------------------------------------------------------------------------- #
# extraction calls
# --------------------------------------------------------------------------- #


def _prompt_sha256(engine_id: str) -> str | None:
    """The digest of the prompt a paid engine sends, or ``None`` for a local one.

    The prompt is part of the request: reworded instructions are a different run
    and a resume has to see that. Imported lazily, the same way the adapter that
    sends it imports it.
    """
    if engine_id != "vision":
        return None
    from workflows.ingest_file.vision_ocr import PROMPT

    return _sha256_text(PROMPT)


def _merge_prompt_sha256() -> str | None:
    """The digest of the merge prompt's own source, or ``None`` when unreadable.

    Reworded merge instructions are a different tree, so the request key has to
    carry the prompt and not just the model. The merge prompt is built per
    section, so what is hashed is the builder that writes it: every word of the
    template is in there, and nothing else is.
    """
    try:
        from prompts.enrichment import build_detail_merge_prompt
    except Exception:
        return None
    try:
        return _sha256_text(inspect.getsource(build_detail_merge_prompt))
    except OSError:
        return None


def _refine_payload(config: Any) -> Any:
    """A refine configuration as plain JSON, so a request key can name it exactly."""
    if config is None:
        return None
    dump = getattr(config, "model_dump", None)
    if callable(dump):
        return dump()
    if isinstance(config, Mapping):
        return dict(config)
    raise RunnerError(
        f"a refine configuration must be a pydantic model or a mapping, "
        f"got {type(config).__name__}"
    )


def _tree_profile_version(artifact: ExtractionArtifact, refine: Any) -> str | int:
    """The version of the tree-building profile: the mapping plus its merge policy.

    Two trees over one artifact built with different merge options are different
    trees, so those options belong in the profile's name. ``thresholds`` is the
    one field that would otherwise carry them, but it takes ints and a policy is
    a set of flags and a model name, so the options ride on the version instead:
    the mapping version names the builder, and the digest of the resolved policy
    names what the merge was asked to do. Without a merge the bare mapping
    version stands, exactly as before.
    """
    if refine is None:
        return artifact.mapping_version
    policy = _sha256_json(_refine_payload(refine))
    return f"{artifact.mapping_version}+{policy}"


def _call_engine(
    engine_id: str,
    module: Any,
    spec: DocumentSpec,
    *,
    base_dir: Path,
    page_indices: Sequence[int],
    page_offset: int,
    dpi: int,
    vision_config: Any | None,
    client: Any | None,
    converter: Any | None,
    transport: Any | None = None,
    sleep: Callable[[float], Any] | None = None,
) -> EngineArtifact:
    """Run one adapter, in the one call shape that adapter declares.

    Every shape is explicit: there is no ``**options`` bag, so a study can never
    pass an adapter an argument that silently does nothing.
    """
    pages = list(page_indices)
    pdf_path = resolve_path(spec.pdf_path, base_dir)
    if engine_id == "vision":
        if vision_config is None:
            raise RunnerError("engine vision needs a VisionOCRConfig")
        return asyncio.run(
            module.extract(
                pdf_path,
                doc_id=spec.doc_id,
                config=vision_config,
                client=client,
                dpi=dpi,
                page_offset=page_offset,
                page_indices=pages,
                transport=transport,
                sleep=sleep,
            )
        )
    if engine_id == "docling":
        return module.extract(
            pdf_path,
            doc_id=spec.doc_id,
            page_offset=page_offset,
            page_indices=pages,
            converter=converter,
        )
    return module.extract(
        pdf_path, doc_id=spec.doc_id, page_offset=page_offset, page_indices=pages
    )


# --------------------------------------------------------------------------- #
# the runner
# --------------------------------------------------------------------------- #


def _repeat_of(row: Mapping[str, Any]) -> int:
    """The repeat a row belongs to; a row recorded without one is the first."""
    repeat = row.get("repeat")
    return repeat if isinstance(repeat, int) else 0


def _participant_of(row: Mapping[str, Any]) -> str | None:
    """What a pair varies: an engine for an extraction, a profile for a tree."""
    value = row.get("profile") if row.get("kind") == KIND_TREE else row.get("engine")
    return str(value) if value is not None else None


def _run_label(manifest: Mapping[str, Any]) -> str:
    """What names one run in a roll-up: its document and its run id.

    Two runs of one document spent their tokens apart, so a roll-up that labelled
    both by document alone would show two rows a reader cannot tell apart. The
    artifact digest is not part of the label: two fresh extractions of the same
    pages with the same engine have the same digest and are still two runs apart.
    """
    return f"{manifest.get('doc_id')}:{manifest.get('run_id')}"


def _merge_rebuilt_section(merge: Mapping[str, Any] | None) -> bool:
    """Whether a merge rebuilt the details of at least one section.

    The one definition of "this merge really merged", shared by the tree verdict
    and the chunk arm check so the two cannot drift. ``sections_merged`` counts
    every section the merge did not skip, fallbacks included: a section whose
    call reached no model keeps its originals and is still counted there, so it
    alone cannot tell a merged tree from the atomic tree under another name.
    What can is the difference: ``sections_fallback`` is the subset that kept the
    originals, so a positive difference is at least one section whose details the
    merge rebuilt. A merge that rebuilt none of them left the tree it was built
    from untouched, so its numbers are that tree's numbers.

    ``changed_sections`` is not the test: it compares a section's children text,
    and a merge that regroups details without rewording them -- four details into
    two, say -- leaves that text identical while rebuilding the section.
    """
    if not isinstance(merge, Mapping):
        return False
    merged = merge.get("sections_merged")
    fallback = merge.get("sections_fallback")
    if not isinstance(merged, int) or not isinstance(fallback, int):
        return False
    return merged > fallback


def _merge_ran(merge: Mapping[str, Any] | None) -> TypeGuard[Mapping[str, Any]]:
    """Whether a merge walked the tree at all, as its own counters record it.

    A merge the profile never ran -- off, or refused for this study -- leaves the
    counters at their zero placeholder, so no report may name it for a merge it
    never made. ``sections`` counts the sections the merge visited, so a positive
    count is the merge having run over this tree.

    The answer is also the narrowing: a caller that read the counters only after
    this said yes holds the merge's counters, not a ``None`` it has to re-check.
    """
    if not isinstance(merge, Mapping):
        return False
    sections = merge.get("sections")
    return isinstance(sections, int) and sections > 0


def _merge_bill(manifest: Mapping[str, Any]) -> dict[str, Any] | None:
    """What a tree's merge spent, as a usage part, or ``None`` with none to report.

    The counters are read as the merge wrote them, not derived from anything
    else. The token counts are all or nothing: a merge that could not state its
    tokens carries ``None`` and its ``usage_unknown_reason`` rather than a
    subtotal. A merge the profile never ran left its counters at the zero
    placeholder (``sections`` is zero) and has nothing to report; a merge that
    ran without calling anything states a measured zero, and that has nothing to
    report either, because a merge that spent no token is no part of what the run
    spent.

    ``refine.llm_model`` travels as the model the merge *requested*, and ``models``
    carries what the answers said served it -- one record per model, the split the
    refiner folded -- when the counts are whole and an answer named one. A gateway
    that resolves an unmatched name to a model of its own choosing is exactly why
    the two are not the same thing, so the requested name is never offered as the
    served one and the served one is never invented from it.
    """
    merge = manifest.get("merge")
    if not _merge_ran(merge):
        return None
    usage = {
        "prompt_tokens": merge.get("prompt_tokens"),
        "completion_tokens": merge.get("completion_tokens"),
        "complete": _merge_usage_complete(merge),
        "attempts": merge.get("attempts"),
        "llm_calls": merge.get("llm_calls"),
        "usage_unknown_reason": merge.get("usage_unknown_reason"),
    }
    if usage["complete"] and not usage["prompt_tokens"] and not usage["completion_tokens"]:
        return None
    refine = manifest.get("refine")
    requested = refine.get("llm_model") if isinstance(refine, Mapping) else None
    part: dict[str, Any] = {"requested_model": requested, "usage": usage}
    models = _serving_models(merge) if usage["complete"] else None
    if models is not None:
        part["models"] = models
    return part


def _serving_models(merge: Mapping[str, Any]) -> list[dict[str, Any]] | None:
    """The serving-model split a merge's tokens are attributed by, or ``None``.

    A split is worth publishing only when at least one answer named the model
    that served the call; without a name no record can say who served those
    tokens, and the run says so in words instead. A record that named none is
    kept in the list even then: its tokens are a real part of what the merge
    spent, so dropping them would silently shrink the record the counters wrote.
    """
    records = merge.get("serving_models")
    if not isinstance(records, list):
        return None
    models = []
    for record in records:
        usage = record.get("usage") if isinstance(record, Mapping) else None
        if isinstance(usage, Mapping):
            models.append({"model": _stated_model(record.get("model")), "usage": dict(usage)})
    if not any(record["model"] for record in models):
        return None
    return models


def _merge_usage_complete(merge: Mapping[str, Any]) -> bool:
    """Whether a merge's counters are whole rather than a subtotal.

    ``usage_unknown_reason`` is the merge's own word on it: when it is set, a call
    the merge made did not report its tokens, so the counts are ``None``.
    Counts that are absent -- a manifest written before the merge counted tokens
    -- are unknown too, because silence is not a zero.
    """
    if merge.get("usage_unknown_reason") is not None:
        return False
    return all(
        isinstance(merge.get(name), int) and not isinstance(merge.get(name), bool)
        for name in ("prompt_tokens", "completion_tokens")
    )


def _cell(entry: Mapping[str, Any], *, participant: Any, kind: str) -> dict[str, Any]:
    """One cell of the expected grid: a participant measured in one repeat.

    The cell carries the run that filled it and that run's own status, so a
    failure is a filled cell rather than a hole, and a hole is only ever a
    participant the study never ran.
    """
    return {
        "doc_id": entry.get("doc_id"),
        "repeat": _repeat_of(entry),
        "participant": str(participant) if participant is not None else None,
        "kind": kind,
        "run_id": entry.get("run_id"),
        "status": entry.get("status"),
        "reason": entry.get("reason"),
    }


@dataclass(frozen=True)
class RunRef:
    """A run directory: one document, one run id."""

    doc_id: str
    run_id: str

    def __str__(self) -> str:
        return f"{self.doc_id}:{self.run_id}"

    @classmethod
    def parse(cls, value: str) -> RunRef:
        doc_id, separator, run_id = value.partition(":")
        if not separator or not doc_id or not run_id:
            raise RunnerError(f"run reference {value!r} must be DOC_ID:RUN_ID")
        return cls(doc_id, run_id)


class StudyRunner:
    """Runs a study over one corpus manifest.

    Everything the runner needs from the outside world is injectable: ``engines``
    replaces adapter modules (so tests never touch a network), ``provenance``
    replaces the git probe, ``vision_config`` replaces the settings-derived
    configuration, and ``llm_client`` replaces the gateway. The live paths stay
    closed until a caller opens them explicitly.
    """

    def __init__(
        self,
        manifest: Manifest,
        *,
        base_dir: str | Path,
        root: str | Path = DEFAULT_ROOT,
        engines: Mapping[str, Any] | None = None,
        provenance: Mapping[str, Any] | None = None,
        settings: Any | None = None,
        merge_concurrency: int = DEFAULT_MERGE_CONCURRENCY,
    ) -> None:
        if isinstance(merge_concurrency, bool) or not isinstance(merge_concurrency, int):
            raise RunnerError(
                f"merge concurrency must be a whole number of sections, got {merge_concurrency!r}"
            )
        if merge_concurrency < 1:
            raise RunnerError(f"merge concurrency must be at least 1, got {merge_concurrency}")
        self.manifest = manifest
        self.base_dir = Path(base_dir)
        self.root = Path(root)
        self.store = ArtifactStore(self.root)
        self._engines = dict(engines or {})
        self._provenance = dict(provenance) if provenance is not None else None
        self._settings = settings
        self.merge_concurrency = merge_concurrency
        self._probes_elsewhere: dict[tuple[str, str], dict[str, Any]] = {}

    # -- construction ----------------------------------------------------- #

    @classmethod
    def from_corpus(
        cls,
        corpus_path: str | Path,
        *,
        root: str | Path = DEFAULT_ROOT,
        **kwargs: Any,
    ) -> StudyRunner:
        manifest, base_dir = load_corpus(corpus_path)
        return cls(manifest, base_dir=base_dir, root=root, **kwargs)

    # -- corpus ----------------------------------------------------------- #

    def documents(self) -> tuple[DocumentSpec, ...]:
        return tuple(self.manifest.documents)

    def document(self, doc_id: str) -> DocumentSpec:
        for spec in self.manifest.documents:
            if spec.doc_id == doc_id:
                return spec
        known = ", ".join(spec.doc_id for spec in self.manifest.documents)
        raise RunnerError(f"unknown document {doc_id!r}; the corpus declares {known}")

    def provenance(self) -> dict[str, Any]:
        if self._provenance is None:
            self._provenance = repo_provenance(self.base_dir)
        return dict(self._provenance)

    def validate(self, *, doc_ids: Sequence[str] | None = None) -> list[dict[str, Any]]:
        """Hash and check every document before anything else runs.

        This is the gate the CLI calls first: a corpus whose PDF moved, whose
        page range is out of bounds, or whose declared modality contradicts the
        real text layer fails here, with no engine and no upload involved.
        """
        wanted = (
            list(doc_ids) if doc_ids is not None else [spec.doc_id for spec in self.documents()]
        )
        reports = []
        for doc_id in wanted:
            spec = self.document(doc_id)
            report = validate_document(spec, base_dir=self.base_dir)
            reports.append(
                {
                    "doc_id": report.doc_id,
                    "pdf_path": str(report.pdf_path),
                    "pdf_sha256": spec.pdf_sha256,
                    "page_count": report.page_count,
                    "page_indices": list(report.page_indices),
                    "pages_with_text": list(report.pages_with_text),
                    "modality": spec.modality.value,
                }
            )
        return reports

    # -- engines ---------------------------------------------------------- #

    def engine_module(self, engine_id: str) -> Any:
        if engine_id in self._engines:
            return self._engines[engine_id]
        if engine_id not in ENGINE_IDS:
            known = ", ".join(ENGINE_IDS)
            raise RunnerError(f"unknown engine {engine_id!r}; expected one of {known}")
        return load_engine(engine_id)

    def probe(self, engine_id: str) -> dict[str, Any]:
        """Whether an engine can run, asked where that engine itself would run.

        A local engine kept in an interpreter of its own — the override
        ``OCR_RESEARCH_ENGINE_PYTHON`` names one per engine — is not installed
        *here* and cannot be seen from here, so the question goes to that
        interpreter, and the answer it gives says which one was asked. Asking
        starts a process in that interpreter, so its answer is remembered: the
        distributions one interpreter holds do not change while this study runs,
        and the plan and the runs it plans then read the one answer. An engine
        this runner was handed directly is a module this process runs; it is
        asked directly, and asked again every time, because a module's state can
        change under it.
        """
        module = self.engine_module(engine_id)
        injected = engine_id in self._engines
        try:
            interpreter = None if injected else engine_interpreter(engine_id)
            if interpreter is not None:
                remembered = self._probes_elsewhere.get((engine_id, interpreter))
                if remembered is not None:
                    return dict(remembered)
            probe = module.probe() if interpreter is None else engine_probe(engine_id, module)
        except Exception as exc:  # a probe that raises is a probe that says no
            return {
                "engine_id": engine_id,
                "available": False,
                "version": None,
                "detail": f"probe raised {type(exc).__name__}",
            }
        answer = {
            "engine_id": probe.engine_id,
            "available": bool(probe.available),
            "version": probe.version,
            "detail": probe.detail,
        }
        if interpreter is not None:
            self._probes_elsewhere[(engine_id, interpreter)] = answer
        return dict(answer)

    def vision_config(self) -> Any:
        """Build the vision configuration from project settings.

        Imports and settings reads happen here, not at module import, so a study
        that never touches the paid engine never needs a key. The render DPI is
        not part of the configuration: it is a per-call argument to the adapter.
        """
        if self._settings is None:
            from config.settings import get_settings

            self._settings = get_settings()
        from workflows.ingest_file.vision_ocr import VisionOCRConfig

        return VisionOCRConfig.from_settings(
            self._settings,
            # Retry policy is a per-run option in the runtime (FileIngestionInput),
            # so a study uses the same defaults it declares.
            max_retries=2,
            timeout_seconds=300.0,
        )

    # -- run directories -------------------------------------------------- #

    def runs_root(self, doc_id: str) -> Path:
        return self.root / doc_id / "runs"

    def run_dir(self, doc_id: str, run_id: str) -> Path:
        return self.runs_root(doc_id) / run_id

    def _allocate(
        self,
        doc_id: str,
        run_id: str,
        *,
        resume: bool,
        request_key: str,
        kind: str,
        needs_merge: bool = False,
    ) -> tuple[Path, dict[str, Any] | None]:
        """The directory a run may be written to, and the run it resumes.

        The directory written to is the id itself, or ``-a2``, ``-a3``… — the
        first of those free, so an attempt never overwrites the record of
        another. Reuse happens only when the caller asked to resume *and* an
        attempt on disk answers the identical request, succeeded, and left every
        file the run promises — and, when the request needs a merge, records a
        merge that rebuilt a section: a candidate whose merge rebuilt nothing is
        the tree it was built from under another name, so a resume walks past it
        rather than hand it back. The id itself is reused when it answers, and
        otherwise the lowest attempt under a suffix. Reusing an attempt reads the
        record it already has; it does not write over one.

        Anything else allocates a fresh directory, so a half-written run is
        never handed back as a success.
        """
        if resume:
            reusable, _ = self._resume_candidate(
                doc_id, run_id, request_key, kind=kind, needs_merge=needs_merge
            )
            if reusable is not None:
                return self.run_dir(doc_id, reusable), self.read_run(
                    doc_id, reusable, required=False
                )
        return self.run_dir(doc_id, self._fresh_attempt(doc_id, run_id)), None

    def _fresh_attempt(self, doc_id: str, run_id: str) -> str:
        """The run id a fresh run of this request is written under.

        The id itself when it is free, otherwise the lowest free ``-a2``, ``-a3``…
        beside it. The naming rule lives here so the allocator and a plan reach it
        together: a dry run names the directory the run will create, rather than
        the id it would have asked for and the run does not use.
        """
        base = self.runs_root(doc_id)
        if not (base / run_id).exists():
            return run_id
        for suffix in range(2, 1000):
            candidate = f"{run_id}-a{suffix}"
            if not (base / candidate).exists():
                return candidate
        raise RunExistsError(f"{base / run_id}: too many attempts under the same run id")

    def _resume_candidate(
        self, doc_id: str, run_id: str, request_key: str, *, kind: str, needs_merge: bool = False
    ) -> tuple[str | None, str | None]:
        """The attempt a resume would hand back, and why there is none to hand back.

        The attempts of one run id are asked lowest first — the id itself before
        the ``-a2``, ``-a3``… beside it — so a resume reuses the earliest record
        that answers the request, and a run id in a report cannot drift when
        another attempt is appended. An attempt is handed back only when
        ``_cache_state`` calls it a ``hit``: the identical request, a success,
        every file the run promises, and — when the request needs a merge — a
        merge that rebuilt a section. Nothing else is ever reused, so an
        incomplete, failed, different-request or rebuilt-nothing run is not
        resumable here, and the search walks past it to the next attempt rather
        than stopping at the id a later study would then keep reporting.

        A miss carries the reason of the first attempt, which is the id a plan
        reports, so the reason explains the run the line names.
        """
        reason = None
        for attempt in self._attempt_ids(doc_id, run_id):
            state, why = self._cache_state(
                doc_id, attempt, request_key, kind=kind, needs_merge=needs_merge
            )
            if state == "hit":
                return attempt, None
            if reason is None:
                reason = why
        return None, reason

    def _attempt_ids(self, doc_id: str, run_id: str) -> list[str]:
        """The run ids one run's attempts were written under, lowest first.

        The first attempt is the id itself; the ones after it are the suffixes
        the allocator makes when the id is taken, ``-a2`` first. A name the
        allocator would not make — ``-a1``, ``-a02``, anything else in the runs
        root that merely starts the same way, or a file rather than a directory
        — is not an attempt and is not read.
        """
        root = self.runs_root(doc_id)
        attempts = {1: run_id}
        if root.is_dir():
            prefix = f"{run_id}-a"
            for entry in root.iterdir():
                digits = entry.name[len(prefix) :] if entry.name.startswith(prefix) else ""
                if (
                    digits.isdigit()
                    and entry.is_dir()
                    and digits == str(int(digits))
                    and int(digits) >= 2
                ):
                    attempts[int(digits)] = entry.name
        return [attempts[suffix] for suffix in sorted(attempts)]

    def _cache_state(
        self, doc_id: str, run_id: str, request_key: str, *, kind: str, needs_merge: bool = False
    ) -> tuple[str, str | None]:
        """Whether a run on disk answers this exact request, and if not, why not.

        A ``hit`` is a run a resume may hand back instead of running again: the
        same request, a success, and every file it promises. A request that
        needs a merge adds one demand, because a merge is not a file: the run's
        counters must show a section whose details it rebuilt, or the record
        holds the tree the merge ran over under the merge's name. The stored
        ``quality`` is not read for that — it was computed by the code of the
        day, while the counters are the fact — so a run stamped measured by a
        different version is refused on what it did rather than what it said.
        Everything else is a miss with the reason a dry run can print before any
        call is made.
        """
        directory = self.run_dir(doc_id, run_id)
        if not directory.is_dir():
            return "miss", "no_run"
        manifest = _read_optional_json(directory / "manifest.json")
        if not isinstance(manifest, dict):
            return "miss", "no_manifest"
        if manifest.get("request_key") != request_key:
            return "miss", "different_request"
        if manifest.get("status") != RUN_OK:
            return "miss", "not_successful"
        missing = self._incomplete_run(manifest, directory, kind=kind)
        if missing is not None:
            return "miss", missing
        if needs_merge and not _merge_rebuilt_section(manifest.get("merge")):
            return "miss", "merge_rebuilt_nothing"
        return "hit", None

    def _incomplete_run(
        self, manifest: Mapping[str, Any], directory: Path, *, kind: str
    ) -> str | None:
        """Which promise of a finished run is missing, or ``None`` when none is.

        The manifest is written last, so its presence is not the whole story: a
        tree is a manifest, a tree, its metrics and a signature that agrees with
        the manifest, and an extraction is a manifest whose artifact is in the
        store. A tree that predates the semantic fingerprint is not resumable
        either — it cannot say which question it answers.
        """
        if kind == KIND_TREE:
            for name in ("tree.json", "metrics.json", "signature.json"):
                if not (directory / name).is_file():
                    return f"incomplete_{name.split('.')[0]}"
            if not manifest.get("semantic_fingerprint"):
                return "no_fingerprint"
            signature = _read_optional_json(directory / "signature.json")
            if not isinstance(signature, dict):
                return "incomplete_signature"
            for field in ("semantic_signature", "page_signature", "semantic_fingerprint"):
                if signature.get(field) != manifest.get(field):
                    return "signature_mismatch"
            return None
        digest = manifest.get("artifact_digest")
        if (
            not isinstance(digest, str)
            or not self.store.path_for(manifest["doc_id"], digest).is_file()
        ):
            return "missing_artifact"
        return None

    def list_runs(
        self, doc_id: str | None = None, *, kind: str | None = None
    ) -> list[dict[str, Any]]:
        """Every run manifest on disk, optionally for one document or one kind."""
        doc_ids = [doc_id] if doc_id is not None else [spec.doc_id for spec in self.documents()]
        found = []
        for current in doc_ids:
            runs_root = self.runs_root(current)
            if not runs_root.is_dir():
                continue
            for directory in sorted(runs_root.iterdir()):
                manifest = _read_optional_json(directory / "manifest.json")
                if not isinstance(manifest, dict):
                    continue
                if kind is not None and manifest.get("kind") != kind:
                    continue
                found.append({**manifest, "path": str(directory)})
        return found

    def read_run(
        self, doc_id: str, run_id: str, *, required: bool = True
    ) -> dict[str, Any] | None:
        path = self.run_dir(doc_id, run_id) / "manifest.json"
        manifest = _read_optional_json(path)
        if manifest is None:
            if required:
                raise RunnerError(f"no run manifest at {path}")
            return None
        if not isinstance(manifest, dict):
            raise RunnerError(f"run manifest {path} is not a JSON object")
        return {**manifest, "path": str(path.parent)}

    def read_run_file(self, doc_id: str, run_id: str, name: str) -> Any:
        return _read_json(self.run_dir(doc_id, run_id) / name, what=f"run {name}")

    # -- extraction ------------------------------------------------------- #

    def extract(
        self,
        doc_id: str,
        engine_id: str,
        *,
        repeat: int = 1,
        page_indices: Sequence[int] | None = None,
        allow_upload: bool = False,
        dpi: int = 200,
        page_offset: int = 0,
        warmth: str | None = None,
        resume: bool = False,
        vision_config: Any | None = None,
        client: Any | None = None,
        converter: Any | None = None,
        transport: Any | None = None,
        sleep: Callable[[float], Any] | None = None,
        label: str | None = None,
    ) -> dict[str, Any]:
        """Run one engine over one document's page selection.

        Returns the run manifest. A ``failed`` or ``not_available`` manifest is
        returned, not raised: a study keeps going, and the denominator keeps the
        document that did not make it.
        """
        spec = self.document(doc_id)
        module = self.engine_module(engine_id)
        selection = tuple(page_indices) if page_indices is not None else spec.page_indices
        config = self._engine_config(engine_id, vision_config)
        public_config = self._public_config(engine_id, config, dpi=dpi)
        request = self._extraction_request(
            spec,
            engine_id,
            config=config,
            selection=selection,
            page_offset=page_offset,
            dpi=dpi,
        )
        request_key = _sha256_json(request)
        run_id = f"{spec.doc_id}.{engine_id}.r{repeat}"
        directory, existing = self._allocate(
            doc_id, run_id, resume=resume, request_key=request_key, kind=KIND_EXTRACT
        )
        if existing is not None:
            # Handed back rather than run: the tokens it carries were spent by the
            # invocation that wrote it, so the caller is told which it got.
            return {**existing, "replayed": True}
        # Nothing on disk answers this request, so the engine has to run -- and
        # only now does it need the upload permission it would use.
        if engine_id in PAID_ENGINES and not allow_upload:
            raise UploadNotAllowedError(
                f"engine {engine_id!r} uploads pages to a hosted endpoint; "
                "pass --allow-upload to permit it"
            )

        def run_engine() -> EngineArtifact:
            return _call_engine(
                engine_id,
                module,
                spec,
                base_dir=self.base_dir,
                page_indices=selection,
                page_offset=page_offset,
                dpi=dpi,
                vision_config=config,
                client=client,
                converter=converter,
                transport=transport,
                sleep=sleep,
            )

        stages: dict[str, Any] = {}
        started = _now()
        status = RUN_OK
        reason: str | None = None
        error: dict[str, Any] | None = None
        artifact: ExtractionArtifact | None = None
        failure: BaseException | None = None
        observed: dict[str, Any] = {}
        observed_resources: dict[str, Any] = {}
        worker_report: dict[str, Any] | None = None
        # A local engine owns the process that runs it, so it runs in a worker of
        # its own and only that worker is measured. A caller-supplied adapter is
        # a double living in *this* process -- a worker could not reach it -- and
        # a hosted engine answers elsewhere, so both run here.
        isolated = engine_id in ISOLATED_ENGINES and engine_id not in self._engines
        # Warmth is never read off the repeat number: a repeat is a second run,
        # not evidence that a loaded model process served it.
        resolved_warmth = warmth
        try:
            with _stage(stages, KIND_EXTRACT):
                if isolated:
                    isolate_run = isolated_extract(
                        engine_id,
                        resolve_path(spec.pdf_path, self.base_dir),
                        doc_id=spec.doc_id,
                        page_indices=selection,
                        page_offset=page_offset,
                    )
                    engine_artifact = isolate_run.artifact
                    observed_resources = dict(isolate_run.resources)
                    worker_report = dict(isolate_run.worker)
                    resolved_warmth = warmth or isolate_run.warmth
                else:
                    measure = (
                        measure_hosted_stage() if engine_id in PAID_ENGINES else _local_stage()
                    )
                    with measure as observed_resources:
                        engine_artifact = run_engine()
                observed = dict(observed_resources)
                artifact = self._persist(engine_artifact, spec, expected_pages=selection)
        except EngineNotAvailableError as exc:
            status, reason = RUN_NOT_AVAILABLE, "engine_not_available"
            error = {"class": type(exc).__name__, "code": reason}
            observed, worker_report = _failed_worker_observation(exc, observed, worker_report)
        except IsolatedWorkerError as exc:
            # A worker that died keeps what it spent before it did.
            status, reason = RUN_FAILED, "engine_failed"
            error = {"class": type(exc).__name__, "code": reason}
            failure = exc
            observed, worker_report = _failed_worker_observation(exc, observed, worker_report)
        except (CorpusError, RunnerError):
            raise
        except Exception as exc:  # the engine's own failure, recorded without its body
            status, reason = RUN_FAILED, "engine_failed"
            error = {"class": type(exc).__name__, "code": reason}
            failure = exc
        if status != RUN_OK:
            observed = observed or dict(observed_resources)
            stage = stages.setdefault(KIND_EXTRACT, {"seconds": None})
            stage["status"] = NOT_MEASURED if status == RUN_NOT_AVAILABLE else STAGE_FAILED
            stage["reason"] = reason

        manifest = {
            "run_manifest_version": RUN_MANIFEST_VERSION,
            "kind": KIND_EXTRACT,
            "run_id": directory.name,
            "doc_id": spec.doc_id,
            "engine_id": engine_id,
            "engine_version": getattr(artifact, "engine_version", None),
            "repeat": repeat,
            "status": status,
            "reason": reason,
            "warmth": resolved_warmth,
            "worker": worker_report,
            "label": label,
            "request_key": request_key,
            "request": request,
            "physical_pages": list(artifact.physical_pages)
            if artifact is not None
            else list(selection),
            "artifact_digest": artifact.content_digest if artifact is not None else None,
            "extraction_fingerprint": (
                artifact.extraction_fingerprint if artifact is not None else None
            ),
            "artifact_status": artifact.status if artifact is not None else None,
            "unsupported_pages": (
                sum(1 for page in artifact.pages if page.status != STATUS_OK)
                if artifact is not None
                else None
            ),
            "public_config": public_config,
            "stages": stages,
            "resources": observed or None,
            "usage": (
                artifact_usage(artifact)
                if artifact is not None
                else (failure_usage(failure) if failure is not None else None)
            ),
            "error": error,
            "provenance": self.provenance(),
            "created_at": _now(),
            "started_at": started,
        }
        _write_json_once(directory / "manifest.json", manifest)
        return {**manifest, "path": str(directory), "replayed": False}

    def _engine_config(self, engine_id: str, config: Any | None) -> Any | None:
        """The configuration an engine call needs: the caller's, or the settings'.

        Built here and not at module import, so a study that never touches the
        paid engine never reads a key. A dry run resolves it the same way the
        run does, which is the point: the plan and the run must not disagree.
        """
        if config is not None or engine_id != "vision":
            return config
        return self.vision_config()

    def _extraction_request(
        self,
        spec: DocumentSpec,
        engine_id: str,
        *,
        config: Any | None,
        selection: Sequence[int],
        page_offset: int,
        dpi: int,
    ) -> dict[str, Any]:
        """The request an extraction makes: everything its answer depends on.

        The plan builds one of these per planned run and the run builds the same
        one, so a dry run's cache verdict is the verdict the run reaches.
        """
        return {
            "doc_id": spec.doc_id,
            "pdf_sha256": spec.pdf_sha256,
            "engine_id": engine_id,
            "engine_version": self._engine_version(engine_id, config),
            "prompt_sha256": _prompt_sha256(engine_id),
            # ``None`` for every engine: each adapter reads the PDF the request
            # already pins by digest. It stays named -- and null -- because it is
            # part of the request a stored run answered, so dropping the field
            # would re-key every run on disk and pay an engine to redo work its
            # artifact still holds.
            "source_sha256": None,
            "page_indices": list(selection),
            "page_offset": page_offset,
            "dpi": dpi if engine_id == "vision" else None,
            "config": self._public_config(engine_id, config, dpi=dpi),
        }

    def _public_config(self, engine_id: str, config: Any | None, *, dpi: int) -> dict[str, Any]:
        """The allowlisted configuration a manifest may carry: no key, no URL.

        Every field the endpoint is asked with is copied out — a request is the
        configuration as much as it is the page selection — and every field is
        read defensively, so an adapter whose configuration is a smaller object
        than the vision one contributes what it has.
        """
        entry: dict[str, Any] = {"dpi": dpi if engine_id == "vision" else None}
        if config is not None:
            for name, attribute in _PUBLIC_CONFIG_FIELDS:
                value = getattr(config, attribute, None)
                if isinstance(value, (str, int, float, bool)):
                    entry[name] = value
        return {key: value for key, value in entry.items() if value is not None}

    def _engine_version(self, engine_id: str, config: Any | None) -> str | None:
        """The version of the engine that will answer this request.

        A vision run records the model it called, so that model *is* its version;
        every other adapter names its own through ``probe``, which answers with
        ``None`` rather than raising when the engine cannot even be imported.
        """
        if engine_id == "vision" and config is not None:
            model = getattr(config, "model", None)
            if isinstance(model, str):
                return model
        return self.probe(engine_id).get("version")

    def _persist(
        self, engine_artifact: EngineArtifact, spec: DocumentSpec, *, expected_pages: Sequence[int]
    ) -> ExtractionArtifact:
        """Turn an adapter's output into the stored artifact for this document.

        The adapter's own ``pdf_sha256`` wins when it has one; when it does not
        (a dump reader has no PDF to hash) the manifest's verified digest is
        filled in, because an artifact that cannot be tied to a PDF revision
        cannot be scored against a reference. A *disagreeing* digest is a failure,
        not something to overwrite.
        """
        artifact = ExtractionArtifact.from_engine(engine_artifact)
        if artifact.pdf_sha256 is None:
            artifact = ExtractionArtifact.model_validate(
                {**artifact.model_dump(), "pdf_sha256": spec.pdf_sha256}
            )
        elif artifact.pdf_sha256 != spec.pdf_sha256:
            raise RunnerError(
                f"{spec.doc_id}: {artifact.engine_id} recorded pdf_sha256 "
                f"{artifact.pdf_sha256}, the manifest declares {spec.pdf_sha256}"
            )
        artifact = self._with_page_status(artifact)
        verify_artifact(artifact, spec, expected_pages=expected_pages)
        self.store.put(artifact)
        return artifact

    def _with_page_status(self, artifact: ExtractionArtifact) -> ExtractionArtifact:
        """An artifact whose every page is unreadable is itself unsupported.

        An adapter marks each page it could not read. When that is *every* page
        the document was never read at all, and storing it as an ordinary ``ok``
        artifact would put a transcript that does not exist into the ranking.
        """
        if any(page.status == STATUS_OK for page in artifact.pages):
            return artifact
        return ExtractionArtifact.model_validate(
            {**artifact.model_dump(), "status": STATUS_UNSUPPORTED_MODALITY}
        )

    def artifact(self, doc_id: str, digest: str) -> ExtractionArtifact:
        try:
            artifact = self.store.get(doc_id, digest)
        except ArtifactStoreError as exc:
            raise RunnerError(f"{doc_id}: no artifact {digest}: {exc}") from exc
        verify_artifact(artifact, self.document(doc_id))
        return artifact

    # -- tree runs -------------------------------------------------------- #

    def run_tree(
        self,
        doc_id: str,
        profile_name: str,
        *,
        artifact_digest: str | None = None,
        extract_run_id: str | None = None,
        repeat: int = 1,
        llm_mode: str = "none",
        allow_merge_upload: bool = False,
        resume: bool = False,
        llm_client: Any | None = None,
        refine_config: Any | None = None,
        label: str | None = None,
    ) -> dict[str, Any]:
        """Build (and optionally merge) one profile's tree from a stored artifact.

        No engine is called here at all: the artifact already holds the
        transcript, so a second tree over the same extraction costs no OCR call.
        """
        if llm_mode not in LLM_MODES:
            raise RunnerError(f"unknown llm mode {llm_mode!r}; expected one of {list(LLM_MODES)}")
        profile = get_profile(profile_name)
        spec = self.document(doc_id)
        digest = artifact_digest
        if extract_run_id is not None:
            source = self._source_run(doc_id, extract_run_id, profile.engine)
            if digest is not None and digest != source["artifact_digest"]:
                raise RunnerError(
                    f"artifact {digest} is not the one {extract_run_id} produced "
                    f"({source['artifact_digest']})"
                )
            digest = source["artifact_digest"]
        if digest is None:
            raise RunnerError("run_tree needs an artifact digest or an extract run")
        artifact = self.artifact(doc_id, digest)
        if artifact.engine_id != profile.engine:
            raise RunnerError(
                f"profile {profile.name!r} reads engine {profile.engine!r}, "
                f"but the artifact is {artifact.engine_id!r}"
            )
        merge_mode = self._merge_mode_for(profile, llm_mode)
        merging = merge_mode != "off"
        # The refine configuration is resolved before the resume lookup, not
        # after: it is part of what a tree *is*, so the lookup has to see the
        # configuration the merge would actually be given.
        resolved_refine = refine_config
        if resolved_refine is None and merging:
            resolved_refine = self._refine_config(profile_name)
        request_key = _sha256_json(
            self._tree_request(
                profile,
                artifact,
                doc_id=doc_id,
                repeat=repeat,
                llm_mode=llm_mode,
                merge_mode=merge_mode,
                refine=resolved_refine,
                extract_run_id=extract_run_id,
            )
        )
        run_id = f"{doc_id}.{profile.name}.r{repeat}.{llm_mode}"
        directory, existing = self._allocate(
            doc_id,
            run_id,
            resume=resume,
            request_key=request_key,
            kind=KIND_TREE,
            needs_merge=merging,
        )
        if existing is not None:
            # Handed back rather than run: what its merge spent was spent by the
            # invocation that wrote it, so the caller is told which it got.
            return {**existing, "replayed": True}
        # Nothing on disk answers this request, so the run has to happen -- and
        # only now does it need the permission, the credentials and the client
        # it would use. A resume that calls nothing asks for none of them.
        if merge_mode == "live" and not allow_merge_upload:
            raise MergeNotAllowedError(
                f"profile {profile.name!r} merges details with a live model; "
                "pass --allow-merge-upload to permit the calls"
            )
        client = llm_client
        if client is None and merge_mode == "live":
            client = self._gateway_client()

        # A replay follows the same continuity policy the extract side follows:
        # a page whose predecessor in the document was never read cannot carry a
        # group across that region, whatever the stored artifact claims. Without
        # this the replayed tree would group pages the live run kept apart.
        pages = reset_continuity(content_pages(engine_pages(artifact)))
        continuity_resets = list(continuity_breaks(pages))
        page_offset = artifact.render_page_offset or 0
        stages: dict[str, Any] = {}
        result = None
        root: StructureNode | None = None
        merge_counters: dict[str, Any] | None = None
        failure: dict[str, Any] | None = None
        try:
            with _stage(stages, "build"):
                result = build(profile.name, pages, doc_id)
            root = result.root
            if merge_mode == "off":
                stages["merge"] = {
                    "status": NOT_MEASURED,
                    "seconds": None,
                    "reason": "profile_merge_off" if profile.merge == "off" else "merge_not_run",
                }
            else:
                with _stage(stages, "merge"):
                    merge_counters = asyncio.run(
                        merge_tree(
                            root,
                            llm_client=client,
                            refine_config=resolved_refine,
                            max_concurrent_sections=_merge_limit(resolved_refine),
                        )
                    )
        except MergeAbortedError as exc:
            # A merge that stopped keeps the spend it made before it did: the run
            # failed, but the calls that answered spent their tokens all the same.
            merge_counters = exc.counters
            failure = {"class": type(exc.cause).__name__, "code": "merge_aborted"}
        except (CorpusError, RunnerError):
            raise
        except Exception as exc:  # a builder or merge failure is a run outcome
            failure = {"class": type(exc).__name__, "code": "build_failed"}
        common = {
            "run_manifest_version": RUN_MANIFEST_VERSION,
            "kind": KIND_TREE,
            "run_id": directory.name,
            "doc_id": doc_id,
            "profile": profile.name,
            "engine_id": artifact.engine_id,
            "merge_mode": profile.merge,
            "llm_mode": llm_mode,
            "merge": merge_counters
            or {"sections": 0, "sections_merged": 0, "sections_skipped": 0, "llm_calls": 0},
            "repeat": repeat,
            "label": label,
            "request_key": request_key,
            "artifact_digest": artifact.content_digest,
            "extract_run_id": extract_run_id,
            "extraction_fingerprint": artifact.extraction_fingerprint,
            "continuity_resets": continuity_resets,
            "refine": _refine_payload(resolved_refine),
            "stages": stages,
            "resources": None,
            "provenance": self.provenance(),
            "created_at": _now(),
        }
        if failure is not None or root is None or result is None:
            manifest = {
                **common,
                "builder": profile.builder,
                "status": RUN_FAILED,
                "reason": (failure or {}).get("code", "build_failed"),
                "rankable": False,
                "quality": NOT_MEASURED,
                "quality_reason": (failure or {}).get("code", "build_failed"),
                "semantic_fingerprint": None,
                "semantic_signature": None,
                "page_signature": None,
                "error": failure,
            }
            _write_json_once(directory / "manifest.json", manifest)
            return {**manifest, "path": str(directory), "replayed": False}
        statistics = validate_tree(root)
        chunks = tree_chunks(root, doc_id=doc_id, page_offset=page_offset)
        fidelity = _fidelity_block(text_fidelity(root, pages))
        chunk_block = chunk_metrics(
            chunks,
            max_chunk_size=self._max_chunk_size(),
            page_totals=self._page_totals(spec, artifact),
        )
        quality, quality_reason = self._quality(profile, merge_mode, artifact, merge_counters)
        manifest = {
            **common,
            "builder": result.builder,
            "status": RUN_OK,
            "reason": None,
            "rankable": quality == MEASURED,
            "quality": quality,
            "quality_reason": quality_reason,
            "semantic_fingerprint": semantic_fingerprint(
                artifact,
                builder_id=result.builder,
                builder_version=_tree_profile_version(artifact, resolved_refine),
                prompt_sha256=artifact.prompt_sha256,
            ),
            "semantic_signature": semantic_signature(root),
            "page_signature": page_signature(root),
            "pages": [page.page_index for page in pages],
            "tree_nodes": statistics.get("total_nodes"),
            "tree_statistics": statistics,
            "error": None,
        }
        # Files first, the manifest last: a half-written run is invisible.
        _write_json_once(directory / "tree.json", serialize_tree_to_dict(root))
        _write_json_once(
            directory / "metrics.json",
            {
                "fidelity": fidelity,
                "chunk_metrics": chunk_block,
                "pages_processed": result.pages_processed,
                "blocks_by_kind": result.blocks_by_kind,
            },
        )
        _write_json_once(
            directory / "signature.json",
            {
                "builder": result.builder,
                "semantic_signature": manifest["semantic_signature"],
                "page_signature": manifest["page_signature"],
                "semantic_fingerprint": manifest["semantic_fingerprint"],
                "blocks_by_kind": result.blocks_by_kind,
                # The builder's counters, with the continuity count stated even
                # when it is zero: a refused claim and an unrecorded one must
                # not read the same in a signature a comparison rests on.
                "counters": {_CONTINUITY_COUNTER: 0, **result.counters},
            },
        )
        _write_json_once(directory / "manifest.json", manifest)
        return {**manifest, "path": str(directory), "replayed": False}

    def _source_run(self, doc_id: str, extract_run_id: str, engine_id: str) -> dict[str, Any]:
        """The extraction run a tree replays, checked before anything is written.

        A run id is a caller's claim about a record, so it is read back rather
        than trusted: the record must be on disk under this document, be an
        extraction, name this engine, and carry an artifact digest. A claim that
        does not hold is an error, never a tree silently attributed to whatever
        record happens to carry the digest -- nothing here reads a digest to guess
        an id from it, or an id to guess a digest.
        """
        manifest = self.read_run(doc_id, extract_run_id, required=False)
        if manifest is None:
            raise RunnerError(f"{doc_id}:{extract_run_id} is not a run in this study")
        if manifest.get("kind") != KIND_EXTRACT:
            raise RunnerError(f"{doc_id}:{extract_run_id} is not an extraction run")
        if manifest.get("engine_id") != engine_id:
            raise RunnerError(
                f"{doc_id}:{extract_run_id} was extracted by {manifest.get('engine_id')!r}, "
                f"not {engine_id!r}"
            )
        digest = manifest.get("artifact_digest")
        if not isinstance(digest, str):
            raise RunnerError(f"{doc_id}:{extract_run_id} produced no artifact")
        return manifest

    def _merge_mode(self, profile: Profile, llm_mode: str, *, allow_merge_upload: bool) -> str:
        """Which merge this profile is actually allowed to run.

        A profile that declares a merge runs it live, and a live merge needs the
        caller's explicit permission; a profile that declares none has no merge to
        run, whatever mode the caller asked for.
        """
        mode = self._merge_mode_for(profile, llm_mode)
        if mode == "live" and not allow_merge_upload:
            raise MergeNotAllowedError(
                f"profile {profile.name!r} merges details with a live model; "
                "pass --allow-merge-upload to permit the calls"
            )
        return mode

    def _merge_mode_for(self, profile: Profile, llm_mode: str) -> str:
        """Which merge this profile would run, permission aside.

        Split from :meth:`_merge_mode` because a request key has to name the
        merge a run would make before anyone has permitted it: a resume that
        answers the request never makes the call, so it never needs the
        permission.
        """
        if profile.merge == "off" or llm_mode == "none":
            return "off"
        return "live"

    def _tree_request(
        self,
        profile: Profile,
        artifact: ExtractionArtifact,
        *,
        doc_id: str,
        repeat: int,
        llm_mode: str,
        merge_mode: str,
        refine: Any,
        extract_run_id: str | None = None,
    ) -> dict[str, Any]:
        """Everything one tree run's identity depends on.

        A tree is a function of the artifact it replays, the extraction run the
        artifact was read from, the builder that replays it, the thresholds that
        turn it into chunks, and — when a merge runs — the mode, the model, the
        prompt and the options the merge is given. Two runs that differ in any of
        these are different runs, and a resume must never hand back one as the
        other. The merge fields are settled here and not after the lookup, so the
        key names the merge the run would really make.

        The extraction run is named, not derived: two runs of one engine can leave
        two artifacts with the same digest, and a tree built over one of them is
        not the tree built over the other. A caller who gave a digest instead of a
        run leaves ``None`` here, which is what the manifest then records -- the
        digest is not read to guess an id it does not carry.
        """
        merging = merge_mode != "off"
        return {
            "doc_id": doc_id,
            "profile": profile.name,
            "artifact_digest": artifact.content_digest,
            "extract_run_id": extract_run_id,
            "builder": profile.builder,
            "builder_version": artifact.mapping_version,
            "max_chunk_size": self._max_chunk_size(),
            "llm_mode": llm_mode,
            "merge_mode": merge_mode,
            "merge_model": self._model_name(profile.name) if merging else None,
            "merge_prompt_sha256": _merge_prompt_sha256() if merging else None,
            "refine": _refine_payload(refine) if merging else None,
            "repeat": repeat,
        }

    def _gateway_client(self) -> Any:
        """The gateway client a live merge calls, built from project settings.

        A bare ``AIGatewayClient()`` is not a gateway: it carries an empty model
        registry, so every call raises ``ConfigurationError`` inside the process.
        The refiner catches that per section and keeps the originals, so the run
        would look successful while no model was ever reached -- which is why the
        merge goes through the app's factory, the one place the chat model and
        its credentials are wired. The settings load is lazy, exactly as
        ``vision_config`` does it, so a study that never merges never reads one.
        """
        if self._settings is None:
            from config.settings import get_settings

            self._settings = get_settings()
        from config.llm_gateway import build_llm_gateway_client

        return build_llm_gateway_client(self._settings)

    def _refine_config(self, profile_name: str) -> Any:
        """The merge policy a scored tree is built with.

        The research policy, not the runtime default: a scored tree treats a
        table as a merge boundary so a ``TableNode`` is never rebuilt as a
        paragraph, and it takes the merge's text without its summary so an
        LLM-written sentence cannot enter a tree that is then measured. The
        runtime defaults are untouched -- the app's own paths still demote
        tables and still prefix summaries; a caller who wants that policy asks
        for it explicitly, as any other caller-supplied configuration.

        The walk's own limit is settled here too, so the configuration a run is
        fingerprinted with and the configuration the walk is given are one
        object: a run cannot record a limit it then merges without.
        """
        from workflows.enrich_knowledge_tree.models import RefineConfig

        return RefineConfig(
            refine_details=True,
            llm_model=self._model_name(profile_name),
            preserve_tables=True,
            segment_summaries=False,
            max_concurrent_sections=self.merge_concurrency,
        )

    def _model_name(self, profile_name: str) -> str:
        """The model a merge would call, from settings when there is one to read."""
        if self._settings is None:
            try:
                from config.settings import get_settings

                self._settings = get_settings()
            except Exception:
                return profile_name
        model = getattr(self._settings, "BENCHMARK_LLM_MODEL", None)
        return model if isinstance(model, str) and model else profile_name

    def _max_chunk_size(self) -> int:
        from config.chunking import MAX_CHUNK_SIZE

        return int(MAX_CHUNK_SIZE)

    def _page_totals(
        self, spec: DocumentSpec, artifact: ExtractionArtifact
    ) -> dict[str, int] | None:
        """``{doc_id: n}`` only when the run covered the 0-based prefix ``0..n-1``.

        ``chunk_metrics`` reads ``page_totals`` as an index space starting at
        zero, so a non-contiguous selection reports coverage as a typed miss
        instead of a number computed in the wrong space. The pages that count
        are the pages this run actually read: a study that overrode the
        manifest's selection with a page subset has that subset in scope, and
        the pages it never asked for are not uncovered.
        """
        covered = artifact.physical_pages
        if covered == tuple(range(len(covered))):
            return {spec.doc_id: len(covered)}
        return None

    def _quality(
        self,
        profile: Profile,
        merge_mode: str,
        artifact: ExtractionArtifact,
        merge_counters: Mapping[str, Any] | None,
    ) -> tuple[str, str | None]:
        """Whether a tree is a research result, and why not when it is not.

        An artifact the engine could not read comes first: a tree built from no
        pages is not a lesser result of the same question, it is no answer to it.
        A merge that rebuilt no section is the other way to answer nothing: every
        section kept its originals -- no call completed, or none of the answers
        rewrote anything -- so the tree is the one it was built from under
        another name and its numbers are that tree's. A partial merge is a
        result: the sections that did merge are the merge.
        """
        if artifact.status != STATUS_OK:
            return NOT_MEASURED, artifact.status
        if profile.merge == "off":
            return MEASURED, None
        if merge_mode == "live":
            if _merge_rebuilt_section(merge_counters):
                return MEASURED, None
            return NOT_MEASURED, "merge_rebuilt_nothing"
        return NOT_MEASURED, "merge_not_run"

    # -- rendering -------------------------------------------------------- #

    def render_pages(
        self,
        doc_id: str,
        *,
        page_indices: Sequence[int],
        out_dir: str | Path,
        dpi: int = 200,
        page_offset: int = 0,
    ) -> dict[str, Any]:
        """Render a page selection to PNGs, so a vision run can be replayed offline.

        Rendering is not a run: the images are an input cache, written beside the
        runs and left alone when they are already there.
        """
        from workflows.ingest_file.vision_ocr import render_pdf_pages

        spec = self.document(doc_id)
        pdf_path = resolve_path(spec.pdf_path, self.base_dir)
        target = Path(out_dir)
        target.mkdir(parents=True, exist_ok=True)
        written = []
        for request_page, png in render_pdf_pages(
            pdf_path, dpi=dpi, page_offset=page_offset, page_indices=list(page_indices)
        ):
            # Named by the 0-based physical index the corpus speaks in, so a file
            # name and a page index in a report are the same number.
            path = target / f"page-{request_page - 1 - page_offset:04d}.png"
            if not path.exists():
                path.write_bytes(png)
                written.append(str(path))
        return {
            "doc_id": doc_id,
            "pages": list(page_indices),
            "dpi": dpi,
            "out_dir": str(target),
            "written": written,
            "existing": len(list(page_indices)) - len(written),
        }

    # -- gold ------------------------------------------------------------- #

    def check_gold(self, doc_id: str, gold_path: str | Path) -> dict[str, Any]:
        """Check a gold file against the corpus: how much of the selection it covers."""
        spec = self.document(doc_id)
        gold = load_gold(gold_path, expected_doc_id=doc_id)
        report = coverage(gold, spec.page_indices)
        return {
            "doc_id": doc_id,
            "gold_path": str(gold_path),
            "gold_digest": gold.content_digest,
            "requested_pages": list(report.requested_pages),
            "covered_pages": list(report.covered_pages),
            "missing_pages": list(report.missing_pages),
            "out_of_scope_pages": list(report.out_of_scope_pages),
            "complete": report.complete,
        }

    # -- planning --------------------------------------------------------- #

    def plan(
        self,
        *,
        doc_ids: Sequence[str] | None = None,
        engine_ids: Sequence[str] = (),
        profile_names: Sequence[str] = (),
        families: Sequence[str] = (),
        repeats: int = 1,
        allow_upload: bool = False,
        llm_mode: str = "none",
        page_indices: Sequence[int] | None = None,
        dpi: int = 200,
        page_offset: int = 0,
        resume: bool = True,
    ) -> dict[str, Any]:
        """What a study would run, before it runs anything.

        The plan is a document, not an action: it names every document, page
        count, engine, profile and repeat, and states what the caller has not
        permitted. ``--dry-run`` prints it; ``study`` runs it. Each planned run
        also carries the verdict a resume would reach — ``hit``, or ``miss``
        with the reason — so a caller can see what a rerun would rebuild and
        what it would merge again before any of it runs.
        """
        wanted_docs = (
            [spec.doc_id for spec in self.documents()] if doc_ids is None else list(doc_ids)
        )
        for doc_id in wanted_docs:
            self.document(doc_id)
        engines = list(engine_ids) or list(ENGINE_IDS)
        for engine_id in engines:
            self.engine_module(engine_id)
        # One resolver for the plan and the run: the plan is the promise the run
        # keeps, so it must not derive a profile set the run would refuse.
        if set(families) == {"ocr"}:
            if profile_names:
                raise RunnerError("--profile only applies when a study builds trees")
            resolved = []
        else:
            resolved = self._study_profiles(engines, profile_names, llm_mode=llm_mode)
        profiles = [get_profile(name) for name in resolved]
        probes = [self.probe(engine_id) for engine_id in engines]
        # The selection a page override names is the selection the run makes, so
        # it is the one the plan counts and the one its cache verdicts are about.
        selections = {
            doc_id: (
                tuple(page_indices)
                if page_indices is not None
                else self.document(doc_id).page_indices
            )
            for doc_id in wanted_docs
        }
        requests = {
            (doc_id, probe["engine_id"], repeat): _sha256_json(
                self._extraction_request(
                    self.document(doc_id),
                    probe["engine_id"],
                    config=self._engine_config(probe["engine_id"], None),
                    selection=selections[doc_id],
                    page_offset=page_offset,
                    dpi=dpi,
                )
            )
            for doc_id in wanted_docs
            for probe in probes
            for repeat in range(1, repeats + 1)
        }

        def verdict(
            doc_id: str, run_id: str, request_key: str, kind: str
        ) -> tuple[str, str, str | None]:
            """What a resume would do with this run: the id and the cache verdict.

            The id is the attempt a resume would hand back when one answers the
            request, and the free directory a run would allocate when none does:
            the plan names the run the run would use rather than the id the
            allocator walks past.
            """
            if not resume:
                return self._fresh_attempt(doc_id, run_id), "miss", "fresh"
            reusable, reason = self._resume_candidate(doc_id, run_id, request_key, kind=kind)
            if reusable is None:
                return self._fresh_attempt(doc_id, run_id), "miss", reason
            return reusable, "hit", None

        extractions = []
        trees = []
        # The extraction run each tree will replay, per document, engine and
        # repeat: the plan and the study resolve it the same way, so the tree
        # verdict is computed against the record the run is really handed.
        sources: dict[tuple[str, str, int], str] = {}
        for doc_id in wanted_docs:
            for probe in probes:
                engine_id = probe["engine_id"]
                for repeat in range(1, repeats + 1):
                    run_id = f"{doc_id}.{engine_id}.r{repeat}"
                    run_id, cache, reason = verdict(
                        doc_id, run_id, requests[(doc_id, engine_id, repeat)], KIND_EXTRACT
                    )
                    sources[(doc_id, engine_id, repeat)] = run_id
                    extractions.append(
                        {
                            "doc_id": doc_id,
                            "engine_id": engine_id,
                            "repeat": repeat,
                            "run_id": run_id,
                            "pages": list(selections[doc_id]),
                            "cache": cache,
                            "cache_reason": reason,
                        }
                    )
            for profile in profiles:
                for repeat in range(1, repeats + 1):
                    run_id = f"{doc_id}.{profile.name}.r{repeat}.{llm_mode}"
                    if not resume:
                        # Nothing on disk is read, so the run allocates: the plan
                        # names the directory it will take.
                        cache, reason = "miss", "fresh"
                        run_id = self._fresh_attempt(doc_id, run_id)
                    else:
                        run_id, cache, reason = self._tree_verdict(
                            profile,
                            doc_id,
                            repeat,
                            llm_mode=llm_mode,
                            run_id=run_id,
                            extract_run_id=sources.get((doc_id, profile.engine, repeat)),
                        )
                    trees.append(
                        {
                            "doc_id": doc_id,
                            "profile": profile.name,
                            "engine_id": profile.engine,
                            "merge": profile.merge,
                            "repeat": repeat,
                            "run_id": run_id,
                            "cache": cache,
                            "cache_reason": reason,
                        }
                    )
        limitations = []
        if not allow_upload and any(engine in PAID_ENGINES for engine in engines):
            limitations.append(
                "vision is planned but not permitted: pass --allow-upload to run it"
            )
        if llm_mode == "live":
            limitations.append(
                "live merges upload prompts to the gateway; each run is marked plainly"
            )
        for probe in probes:
            if not probe["available"]:
                limitations.append(f"{probe['engine_id']} is unavailable: {probe['detail']}")
        planned = extractions + trees
        return {
            "study_plan_version": RUN_MANIFEST_VERSION,
            "corpus": {
                "documents": wanted_docs,
                "pages": {doc_id: len(selections[doc_id]) for doc_id in wanted_docs},
            },
            "engines": probes,
            "extractions": extractions,
            "trees": trees,
            "families": list(families),
            "profiles": resolved,
            "llm_mode": llm_mode,
            "merge_concurrency": self.merge_concurrency,
            "allow_upload": allow_upload,
            "resume": resume,
            "cache": {
                "hits": sum(1 for entry in planned if entry["cache"] == "hit"),
                "misses": sum(1 for entry in planned if entry["cache"] == "miss"),
            },
            "limitations": limitations,
            "created_at": _now(),
        }

    def _tree_verdict(
        self,
        profile: Profile,
        doc_id: str,
        repeat: int,
        *,
        llm_mode: str,
        run_id: str,
        extract_run_id: str | None,
    ) -> tuple[str, str, str | None]:
        """The run id a resume would use, its verdict, and why there is no hit.

        The id is the free directory a run would allocate when no attempt can be
        reused and the attempt a resume would hand back when one can, so a plan
        names the tree run a run would actually leave in place.

        The source is the extraction run the study will hand this tree, named by
        the plan's own extraction line rather than found by looking for a digest:
        two runs of one engine can leave two artifacts with one digest, and a tree
        built over one of them is not the tree built over the other. Until that
        record is on disk there is nothing to predict, so the verdict is a miss
        and says so, never a guess about an extraction that has not run yet.
        """
        fresh = self._fresh_attempt(doc_id, run_id)
        source = (
            self.read_run(doc_id, extract_run_id, required=False)
            if extract_run_id is not None
            else None
        )
        digest = source.get("artifact_digest") if source is not None else None
        if not isinstance(digest, str):
            return fresh, "miss", "no_artifact"
        artifact = self.artifact(doc_id, digest)
        merge_mode = self._merge_mode_for(profile, llm_mode)
        refine = self._refine_config(profile.name) if merge_mode != "off" else None
        request_key = _sha256_json(
            self._tree_request(
                profile,
                artifact,
                doc_id=doc_id,
                repeat=repeat,
                llm_mode=llm_mode,
                merge_mode=merge_mode,
                refine=refine,
                extract_run_id=extract_run_id,
            )
        )
        reusable, reason = self._resume_candidate(
            doc_id, run_id, request_key, kind=KIND_TREE, needs_merge=merge_mode != "off"
        )
        if reusable is None:
            return fresh, "miss", reason
        return reusable, "hit", None

    # -- comparison ------------------------------------------------------- #

    def _name_the_missing_arm(
        self, rows: Sequence[Mapping[str, Any]], limitations: list[str]
    ) -> None:
        """Say which document has one pipeline arm and no pair to put it in.

        The family pairs the vision tree with the Marker tree and nothing else, so
        a document whose other arm did not run has no pair at all -- which a reader
        of the pairs alone would take for a family that was never asked for. Each
        such document is named with the arm it is missing.
        """
        arms = set(PIPELINE_ARMS)
        for doc_id in sorted({str(row.get("doc_id")) for row in rows}):
            present = {_participant_of(row) for row in rows if str(row.get("doc_id")) == doc_id}
            missing = sorted(arms - present)
            if missing and present & arms:
                limitations.append(
                    f"{doc_id}: the pipeline pair is {PIPELINE_ARMS[0]} against "
                    f"{PIPELINE_ARMS[1]}, and {' and '.join(missing)} has no run here, so "
                    "this document has no pipeline pair"
                )

    def compare(
        self,
        family: str,
        runs: Sequence[RunRef],
        *,
        gold_paths: Mapping[str, str] | None = None,
        label: str | None = None,
        out_dir: str | Path | None = None,
        write: bool = True,
    ) -> dict[str, Any]:
        """Compare runs of one family and write the report in all three formats.

        Denominators are built from what was asked for, not from what worked: a
        document with no gold still appears, a page with no hypothesis still
        counts, and a run that failed keeps its own row instead of vanishing.
        """
        if family not in FAMILIES:
            raise RunnerError(f"unknown family {family!r}; expected one of {list(FAMILIES)}")
        resolved = [self._resolve_run(ref) for ref in runs]
        if not resolved:
            raise RunnerError("compare needs at least one run")
        limitations: list[str] = []
        rows: list[dict[str, Any]] = []
        per_run_documents: dict[str, list[dict[str, Any]]] = {}
        usage_documents: list[dict[str, Any]] = []
        for manifest in resolved:
            row, documents, usage_rows = self._run_row(
                manifest,
                gold_paths=gold_paths or {},
                limitations=limitations,
            )
            rows.append(row)
            per_run_documents[str(row.get("run_id"))] = documents
            usage_documents.extend(usage_rows)
        # A merge that ran and rebuilt no section is a gap the numbers cannot
        # show: its tree is the one it was built from under another name, so the
        # fidelity and chunk figures its row carries measure that tree, not a
        # merge. Named once per document -- every repeat of it has the same gap --
        # and never for a merge that never ran, which its own verdict already
        # names.
        for doc_id in sorted(
            {
                str(row.get("doc_id"))
                for row in rows
                if row.get("kind") == KIND_TREE
                and _merge_ran(row.get("merge"))
                and not _merge_rebuilt_section(row.get("merge"))
            }
        ):
            limitations.append(
                f"{doc_id}: the merge rebuilt no section, so its merged tree is the "
                "tree it was built from under another name"
            )
        # Tokens are reported part by part, and a pair's delta counts both parts of
        # the document it compares: the extraction and the merge. A merge whose
        # tokens the endpoint never reported leaves its run without a total, so
        # such a pair's delta is a typed miss instead of a number. A reader of the
        # deltas has to be told both.
        if any(isinstance(document.get("merge"), Mapping) for document in usage_documents):
            limitations.append(
                "paired token deltas count the extraction and its merge together: a "
                "merge whose tokens were not reported leaves its run without a total, "
                "so a pair that counts one reports a typed miss rather than a number"
            )
        text_metrics = self._text_metrics(per_run_documents, limitations)
        # Each row carries its own run's text metrics, computed once here, so a
        # pair delta reads the same numbers the report published.
        for row in rows:
            row["text_metrics"] = (text_metrics.get("runs") or {}).get(str(row.get("run_id")))
        if family == "ocr":
            limitations.append(
                "ocr compares transcripts only: it says nothing about the trees built from them"
            )
        chunk_section = self._chunk_section(rows)
        # This comparison's token accounting: each run's usage counted once, as
        # the family that names it must be read. Whether those calls were made
        # now or replayed from disk is a fact about an invocation, not about the
        # runs -- and a comparison is handed runs alone -- so it is the study
        # that publishes it.
        usage_section = run_usage(usage_documents)
        if family == "pipeline":
            self._name_the_missing_arm(rows, limitations)
        # The join comes before the pairing: a pair reads each arm's readings off
        # its row, so an arm whose engine ran in the extraction it replays must
        # carry that extraction's resources by the time the delta is taken.
        stages, resources = self._stage_rollup(resolved, rows)
        paired = self._paired(family, rows)
        metadata = {
            "source": "compare",
            "family": family,
            "label": label,
            "documents": sorted(
                {str(row.get("doc_id")) for row in rows if row.get("artifact_digest")}
            ),
            "runs": [str(row.get("run_id")) for row in rows],
            "run_rows": rows,
            "created_at": _now(),
        }
        report = build_report(
            {"run_id": label or f"{family}.comparison", "label": label, "created_at": _now()},
            metadata=metadata,
            stages=stages,
            resources=resources,
            fidelity=self._fidelity_section(rows),
            limitations=limitations,
            text_metrics=text_metrics,
            chunk_metrics=chunk_section,
            usage=usage_section,
            paired_usage=paired,
        )
        result = {
            "comparison_version": COMPARISON_VERSION,
            "family": family,
            "runs": rows,
            "report": report,
        }
        if write:
            target = Path(out_dir) if out_dir is not None else self._allocate_comparison(family)
            result["path"] = str(target)
            result["files"] = self._write_comparison(target, report)
        return result

    def _allocate_comparison(self, family: str) -> Path:
        base = self.root / "comparisons"
        stamp = _stamp()
        candidate = base / f"{family}.{stamp}"
        if not candidate.exists():
            return candidate
        for suffix in range(2, 1000):
            alt = base / f"{family}.{stamp}-a{suffix}"
            if not alt.exists():
                return alt
        raise RunExistsError(f"{base}: too many comparisons in the same second")

    def _write_comparison(self, target: Path, report: dict[str, Any]) -> dict[str, str]:
        target.mkdir(parents=True, exist_ok=True)
        files = {
            "json": target / "report.json",
            "csv": target / "report.csv",
            "markdown": target / "report.md",
        }
        # Every format is rendered from the same dict, so their counts agree.
        write_once(files["json"], to_json(report).encode("utf-8"))
        write_once(files["csv"], to_csv(report).encode("utf-8"))
        write_once(files["markdown"], to_markdown(report).encode("utf-8"))
        _write_json_once(
            target / "manifest.json",
            {
                "comparison_version": COMPARISON_VERSION,
                "family": report["metadata"].get("family"),
                "runs": report["metadata"].get("runs"),
                "counts": report["counts"],
                "created_at": _now(),
                "files": [str(path) for path in files.values()],
            },
        )
        return {name: str(path) for name, path in files.items()}

    def _resolve_run(self, ref: RunRef) -> dict[str, Any]:
        manifest = self.read_run(ref.doc_id, ref.run_id)
        assert manifest is not None  # read_run raises when required and missing
        return manifest

    def _run_row(
        self,
        manifest: Mapping[str, Any],
        *,
        gold_paths: Mapping[str, str],
        limitations: list[str],
    ) -> tuple[dict[str, Any], list[dict[str, Any]], list[dict[str, Any]]]:
        """One row per run, plus the scored documents and usage records behind it."""
        doc_id = str(manifest.get("doc_id"))
        kind = manifest.get("kind")
        digest = manifest.get("artifact_digest")
        row: dict[str, Any] = {
            "run_id": manifest.get("run_id"),
            "kind": kind,
            "doc_id": doc_id,
            "engine": manifest.get("engine_id"),
            "profile": manifest.get("profile"),
            "repeat": manifest.get("repeat"),
            "status": manifest.get("status"),
            "rankable": manifest.get("rankable"),
            "quality": manifest.get("quality"),
            "quality_reason": manifest.get("quality_reason"),
            "artifact_digest": digest,
            "semantic_signature": manifest.get("semantic_signature"),
            "merge_mode": manifest.get("llm_mode"),
            "path": manifest.get("path"),
            # Wall time as a metric, so a pair can read time beside quality.
            "seconds": _seconds_metric(manifest.get("stages")),
            # What the run's own stage used, in the same shape, so a pair can
            # read CPU and memory beside it. A hosted run measured none of the
            # backend's, and carries the miss rather than a zero.
            "resources": _run_resources(manifest.get("resources")),
        }
        documents: list[dict[str, Any]] = []
        usage_rows: list[dict[str, Any]] = []
        artifact: ExtractionArtifact | None = None

        def account() -> None:
            """The run's usage record, artifact or failure alike.

            A run that was requested and made calls belongs in the usage
            denominator even when it produced nothing to score: dropping it would
            let a study look cheaper by failing. Its usage block says what is
            unknown.

            A tree run also carries what its merge spent, which the extraction
            usage cannot show: a second set of counters, apart from the first so
            a reader can see both. The merge is attributed by the models its
            answers named -- never against the model it requested, so the
            requested name cannot pass for the served one -- and where no answer
            named a model the report says in words what the split left
            unattributed.
            """
            model, usage = self._usage_of(manifest, artifact)
            row["model"] = model
            row["usage"] = usage
            merge = _merge_bill(manifest)
            # The merge's counters travel with the row so the pair that counts
            # this run counts the same document the run-level usage block did.
            row["merge_bill"] = merge
            if merge is not None:
                tokens = merge["usage"]
                spent = (
                    f"{tokens['prompt_tokens']} prompt and "
                    f"{tokens['completion_tokens']} completion tokens"
                    if isinstance(tokens["prompt_tokens"], int)
                    else "an unknown number of tokens"
                )
                if merge.get("models") is None:
                    limitations.append(
                        f"{manifest.get('run_id')}: the merge spent {spent}, and no answer "
                        "named the model that served it: the run records the model the "
                        "merge requested, not the one the gateway served its calls with"
                    )
                elif any(record["model"] is None for record in merge["models"]):
                    limitations.append(
                        f"{manifest.get('run_id')}: the merge spent {spent}, and the part "
                        "of it no answer credited to a model cannot be attributed"
                    )
            usage_rows.append(self._usage_document(row))

        if not isinstance(digest, str):
            row["artifact"] = {
                "status": NOT_MEASURED,
                "reason": manifest.get("reason") or "no_artifact",
                "value": None,
            }
            limitations.append(
                f"{doc_id}/{manifest.get('run_id')}: no artifact ({manifest.get('status')}), "
                "so it contributes no text metric"
            )
            account()
            return row, documents, usage_rows
        try:
            artifact = self.artifact(doc_id, digest)
        except (RunnerError, CorpusError) as exc:
            row["artifact"] = {
                "status": NOT_MEASURED,
                "reason": "artifact_unreadable",
                "value": None,
            }
            limitations.append(f"{doc_id}/{manifest.get('run_id')}: {exc}")
            account()
            return row, documents, usage_rows
        row["artifact"] = {
            "status": MEASURED,
            "reason": None,
            "value": len(artifact.pages),
            "unit": "pages",
        }
        row["artifact_status"] = artifact.status
        row["pages_without_text"] = sum(1 for page in artifact.pages if page.status != STATUS_OK)
        # What this row read, so a pair can refuse two rows that disagree about it.
        row["pdf_sha256"] = artifact.pdf_sha256
        row["page_indices"] = list(artifact.physical_pages)
        if kind == KIND_EXTRACT:
            # An extraction has no tree quality of its own: its status is the
            # artifact's, and an artifact nothing could be read from is not a
            # result a pair may score.
            readable = artifact.status == STATUS_OK
            row["rankable"] = readable
            row["quality"] = MEASURED if readable else NOT_MEASURED
            row["quality_reason"] = None if readable else artifact.status
        gold = self._gold_for(doc_id, gold_paths, limitations)
        if gold is not None:
            documents.append(metrics_document(gold, artifact))
            row["gold_digest"] = gold.content_digest
            row["gold_reference"] = {
                "status": MEASURED,
                "reason": None,
                "value": gold.content_digest[:12],
                "pages": len(gold.pages),
            }
            row["structure_metrics"] = heading_boundary_metrics(gold, artifact)
        account()
        if kind == KIND_TREE:
            metrics = self._run_metrics(doc_id, manifest)
            row["fidelity"] = metrics.get("fidelity")
            row["chunk_metrics"] = metrics.get("chunk_metrics")
            row["merge"] = manifest.get("merge")
        return row, documents, usage_rows

    def _model_for(
        self, manifest: Mapping[str, Any], artifact: ExtractionArtifact | None
    ) -> str | None:
        """The model a run's calls are attributed to, from the manifest or the artifact."""
        config = manifest.get("public_config")
        if isinstance(config, dict):
            model = config.get("model")
            if isinstance(model, str):
                return model
        if artifact is None or not artifact.public_config:
            return None
        model = artifact.public_config.get("model")
        return model if isinstance(model, str) else None

    def _gold_for(
        self,
        doc_id: str,
        gold_paths: Mapping[str, str],
        limitations: list[str],
    ) -> GoldReference | None:
        """The gold for this document: an explicit file, the manifest's, or none."""
        spec = self.document(doc_id)
        path = gold_paths.get(doc_id) or spec.gold_path
        if not path:
            limitations.append(f"{doc_id}: no gold, so its pages cannot be scored")
            return None
        resolved = Path(path) if Path(path).is_absolute() else self.base_dir / path
        try:
            return load_gold(resolved, expected_doc_id=doc_id)
        except GoldError as exc:
            limitations.append(f"{doc_id}: gold not usable: {exc}")
            return None

    def _run_metrics(self, doc_id: str, manifest: Mapping[str, Any]) -> dict[str, Any]:
        run_id = str(manifest.get("run_id"))
        path = self.run_dir(doc_id, run_id) / "metrics.json"
        if not path.is_file():
            return {}
        payload = _read_json(path, what="run metrics")
        return payload if isinstance(payload, dict) else {}

    def _text_metrics(
        self, per_run: Mapping[str, Sequence[dict[str, Any]]], limitations: list[str]
    ) -> dict[str, Any]:
        """Corpus text metrics: one number per run, and the corpus's own counts.

        Runs are never pooled into a single CER: two engines mixed into one
        average would answer a question nobody asked. The top-level block takes
        each document once, from the first run that scored it, so it can carry
        the report's corpus counts; each run's own numbers sit under ``runs``.
        """
        scored = {run_id: list(documents) for run_id, documents in per_run.items() if documents}
        if not scored:
            limitations.append("no run produced an artifact with a usable reference")
            return {"status": NOT_MEASURED, "reason": "no_scored_documents"}
        corpus_documents: dict[str, dict[str, Any]] = {}
        for documents in scored.values():
            for document in documents:
                corpus_documents.setdefault(str(document.get("doc_id")), document)
        if len(scored) > 1:
            limitations.append(
                "corpus counts take each document once, from the first run that scored it; "
                "per-run text metrics are under text_metrics.runs"
            )
        corpus = corpus_text_metrics(list(corpus_documents.values()))
        corpus["runs"] = {
            run_id: corpus_text_metrics(documents) for run_id, documents in scored.items()
        }
        return corpus

    def _chunk_section(self, rows: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
        """Chunk metrics over every tree run, pooled, with the profile named."""
        chunks: list[dict[str, Any]] = []
        for row in rows:
            metrics = row.get("chunk_metrics")
            if isinstance(metrics, dict):
                chunks.append(metrics)
        if not chunks:
            return {"status": NOT_MEASURED, "reason": "no_tree_runs"}
        measured = [entry for entry in chunks if entry.get("status") == MEASURED]
        if not measured:
            return {"status": NOT_MEASURED, "reason": "no_chunks", "runs": len(chunks)}
        return {
            "status": MEASURED,
            "reason": None,
            "runs": len(chunks),
            "chunks": sum(entry.get("chunks") or 0 for entry in measured),
            "per_run": chunks,
        }

    def _fidelity_section(self, rows: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
        """Text fidelity per tree run: what the tree kept of its own artifact."""
        section: dict[str, Any] = {}
        for row in rows:
            fidelity = row.get("fidelity")
            if isinstance(fidelity, dict):
                section[str(row.get("run_id"))] = {"profile": row.get("profile"), **fidelity}
        return section

    def _stage_rollup(
        self, manifests: Sequence[Mapping[str, Any]], rows: Sequence[dict[str, Any]]
    ) -> tuple[dict[str, Any], dict[str, Any]]:
        """Stage timings and resource observations across the compared runs.

        A tree replay ran no engine, so the engine work of a tree arm is not in
        its own observation: the extraction the run names is joined by that run's
        id -- never by a digest two runs may share -- and its stages and its
        resources keep their own run's key, where their scope is readable. The
        row reads the joined readings as its own arm's, and names the run they
        came from; a run that measured its own resources is never overwritten.
        """
        stages: dict[str, Any] = {}
        resources: dict[str, Any] = {}
        for manifest, row in zip(manifests, rows):
            run_id = str(row.get("run_id"))
            for name, stage in (manifest.get("stages") or {}).items():
                stages[f"{run_id}.{name}"] = stage
            if manifest.get("resources"):
                resources[run_id] = manifest["resources"]
            source = self._named_source(manifest)
            if source is None:
                continue
            source_id = str(source.get("run_id"))
            for name, stage in (source.get("stages") or {}).items():
                stages[f"{source_id}.{name}"] = stage
            if not source.get("resources"):
                continue
            resources[source_id] = source["resources"]
            if row.get("resources") is None:
                row["resources"] = _run_resources(source["resources"])
                row["resource_run_id"] = source_id
        return stages, resources

    def _named_source(self, manifest: Mapping[str, Any]) -> Mapping[str, Any] | None:
        """The extraction run a tree names, read back by that run's id.

        The linkage is the id the manifest carries: nothing here derives an id
        from a digest, and a run that names none -- a bare-digest replay, or a
        record written before trees carried the id -- has no source rather than a
        guessed one.
        """
        run_id = manifest.get("extract_run_id")
        if not isinstance(run_id, str):
            return None
        return self.read_run(str(manifest.get("doc_id")), run_id, required=False)

    def _pair_indices(
        self, rows: Sequence[Mapping[str, Any]], family: str
    ) -> list[tuple[int, int]]:
        """Which rows form an A/B pair: the same repeat, two different participants.

        The pairing is explicit: every unordered pair of distinct participants
        measured in the same repeat, not just neighbours in the caller's list, so
        a third engine cannot hide the pairs it is not adjacent to. The repeat is
        the pairing key — two repeats of one participant measure repeatability,
        not an engine difference, and two participants measured in different
        repeats are not a comparison either.

        Two families are one fixed pair — ``chunk`` and ``pipeline`` — and pair
        nothing else.
        """
        groups: dict[int, list[int]] = {}
        for index, row in enumerate(rows):
            groups.setdefault(_repeat_of(row), []).append(index)
        pairs: list[tuple[int, int]] = []
        for repeat in sorted(groups):
            indices = groups[repeat]
            for position, left in enumerate(indices):
                for right in indices[position + 1 :]:
                    if self._is_pair(family, rows[left], rows[right]):
                        pairs.append((left, right))
        return pairs

    def _is_pair(
        self,
        family: str,
        row_left: Mapping[str, Any],
        row_right: Mapping[str, Any],
    ) -> bool:
        """Whether two rows of one repeat are the two sides of a family's delta."""
        left, right = _participant_of(row_left), _participant_of(row_right)
        if left is None or left == right:
            return False
        if family == "chunk":
            # The chunk family is one fixed pair. The atomic tree is the control
            # the merged arm was built from, so pairing it would measure the
            # merge against its own input.
            return {left, right} == set(CHUNK_ARMS)
        if family == "pipeline":
            # So is the pipeline family: the vision tree against the Marker tree.
            # Pairing the two Marker arms with each other, or a vision tree with
            # another tree of its own engine, would measure an engine against
            # itself -- two readings of one engine are not the pipeline.
            return {left, right} == set(PIPELINE_ARMS)
        return True

    def _paired(
        self,
        family: str,
        rows: Sequence[Mapping[str, Any]],
    ) -> dict[str, Any]:
        """Pair the runs that are an A/B comparison, keeping the rest visible.

        A pair contributes a delta only when both runs succeeded, are rankable,
        and measured the same thing. A faked merge, a failed engine or a missing
        document keeps its own bucket: dropping it would silently inflate the
        denominator. Two runs of one participant in one repeat are not paired at
        all, because their difference is repeatability, not an engine delta.
        """
        pairs = []
        for left_index, right_index in self._pair_indices(rows, family):
            row_left, row_right = rows[left_index], rows[right_index]
            reasons = self._pairing_reasons(row_left, row_right, family)
            pair: dict[str, Any] = {
                "run_a": row_left.get("run_id"),
                "run_b": row_right.get("run_id"),
                # What the pair varies and which repeat it belongs to, so a
                # corpus summary can group by pair without re-deriving either.
                "participants": [_participant_of(row_left), _participant_of(row_right)],
                "repeat": _repeat_of(row_left),
                "compatible": not reasons,
                "incompatible_reasons": reasons,
                "documents_in_a": 1 if row_left.get("artifact_digest") else 0,
                "documents_in_b": 1 if row_right.get("artifact_digest") else 0,
                "paired_documents": 1 if not reasons else 0,
                "union_documents": 1 if not reasons else 0,
                "delta": {
                    "status": NOT_MEASURED,
                    "reason": "not_paired",
                    "value": None,
                    "unit": None,
                },
            }
            pair["evaluation"] = self._pair_evaluation(row_left, row_right)
            # The headline delta this family exists to measure, and the whole
            # set of metric deltas behind it. They are different things: the
            # headline is what the family is named for, while ``metrics`` is
            # uniformly shaped so a corpus summary can aggregate it per pair.
            if family == "chunk":
                pair["delta"] = self._chunk_delta(row_left, row_right, reasons)
                pair["metrics"] = {
                    "chunks": pair["delta"],
                    **self._quality_delta(row_left, row_right, reasons),
                }
            else:
                pair["metrics"] = self._quality_delta(row_left, row_right, reasons)
            if not reasons:
                # Tokens and quality are separate deltas: one must never
                # overwrite the other, or a run that read more would look like a
                # worse one. A run whose usage is unknown still pairs, as a
                # typed miss.
                pair["usage"] = paired_usage(
                    {
                        "run_id": row_left.get("run_id"),
                        "documents": [self._usage_document(row_left)],
                    },
                    {
                        "run_id": row_right.get("run_id"),
                        "documents": [self._usage_document(row_right)],
                    },
                )
            pairs.append(pair)
        # A document is the statistical unit, so a document measured by several
        # pairs is one document, not one per pair. The pair and repeat counts are
        # separate denominators and are never folded into the document count.
        paired_docs = {
            str(row.get("doc_id"))
            for pair in pairs
            if pair["compatible"]
            for row in (self._row_of(pair, "a", rows), self._row_of(pair, "b", rows))
            if row is not None and row.get("artifact_digest")
        }
        union_docs = {
            str(row.get("doc_id"))
            for pair in pairs
            if pair["compatible"]
            for row in (self._row_of(pair, "a", rows), self._row_of(pair, "b", rows))
            if row is not None
        }
        return {
            "status": MEASURED if pairs else NOT_MEASURED,
            "reason": None if pairs else "no_pair",
            "pairs": pairs,
            "n_pairs": len(pairs),
            "n_repeats": len({_repeat_of(row) for row in rows}),
            "paired_documents": len(paired_docs),
            "union_documents": len(union_docs),
        }

    @staticmethod
    def _row_of(
        pair: Mapping[str, Any], side: str, rows: Sequence[Mapping[str, Any]]
    ) -> Mapping[str, Any] | None:
        """The row one side of a pair was built from."""
        run_id = pair.get(f"run_{side}")
        for row in rows:
            if row.get("run_id") == run_id:
                return row
        return None

    def _usage_document(self, row: Mapping[str, Any]) -> dict[str, Any]:
        """One run as a usage document: its extraction, and its merge when it has one.

        The one definition of what a run spent, so the run-level usage block and
        the pair that counts the same run count the same document rather than two
        shapes of it that can drift.
        """
        return {
            "doc_id": row.get("doc_id"),
            "model": row.get("model"),
            "usage": row.get("usage"),
            "merge": row.get("merge_bill"),
        }

    def _usage_of(
        self, manifest: Mapping[str, Any], artifact: ExtractionArtifact | None
    ) -> tuple[str | None, Any]:
        """The model a run is attributed to and the usage record its tokens are read from.

        The artifact speaks first: an engine that answered wrote its model and
        its page usage there, and that is what the run spent. The manifest is the
        fallback for a run that left no artifact to read -- one that failed is
        the usual case -- and its own ``usage`` mapping is taken as it stands.
        """
        if artifact is not None:
            return self._model_for(manifest, artifact), artifact_usage(artifact)
        return self._model_for(manifest, artifact), manifest.get("usage")

    def _artifact_of(self, manifest: Mapping[str, Any]) -> ExtractionArtifact | None:
        """The artifact a run wrote, or ``None`` when there is none to read."""
        digest = manifest.get("artifact_digest")
        if not isinstance(digest, str):
            return None
        try:
            return self.artifact(str(manifest.get("doc_id")), digest)
        except (RunnerError, CorpusError):
            return None

    def _pairing_reasons(
        self,
        row_left: Mapping[str, Any],
        row_right: Mapping[str, Any],
        family: str,
    ) -> list[str]:
        """Why two runs are not an A/B comparison; empty when they are.

        The first reason that applies is the one reported, so a pair that is
        wrong in several ways still names the most basic thing that is wrong.
        """
        reasons = []
        for side, row in (("a", row_left), ("b", row_right)):
            if row.get("status") != RUN_OK:
                reasons.append(f"run_{side}_not_ok")
            elif row.get("rankable") is False:
                # An engine that read no page did not fail; it answered that this
                # document carries no text it can read. That is a fact about the
                # document, and it is named apart from a run whose quality was
                # never measured.
                if row.get("artifact_status") == STATUS_UNSUPPORTED_MODALITY:
                    reasons.append(f"run_{side}_unsupported_modality")
                else:
                    reasons.append(f"run_{side}_unrankable")
        if row_left.get("doc_id") != row_right.get("doc_id"):
            reasons.append("different_documents")
        if not reasons and row_left.get("artifact_digest") is None:
            reasons.append("run_a_without_artifact")
        if not reasons and row_right.get("artifact_digest") is None:
            reasons.append("run_b_without_artifact")
        # What the runs read, and what they were scored against, must agree: a
        # delta between two different page selections, two different PDFs or two
        # different reference trees measures the difference between those inputs
        # as much as the difference between the engines.
        if not reasons and row_left.get("pdf_sha256") != row_right.get("pdf_sha256"):
            reasons.append("different_pdf")
        if not reasons and (
            (row_left.get("page_indices") or []) != (row_right.get("page_indices") or [])
        ):
            reasons.append("different_page_selection")
        # What the runs were *scored against* is not here. Gold is evaluation
        # provenance: it says how a run was judged, not what it read, and a pair
        # that disagrees about it is still the extraction comparison the study
        # asked for. The metrics that read the gold abstain instead (see
        # ``_quality_delta``), and the pair carries both sides' provenance.
        if family == "chunk" and not reasons:
            left_artifact = row_left.get("artifact_digest")
            right_artifact = row_right.get("artifact_digest")
            if left_artifact != right_artifact:
                reasons.append("different_artifacts")
        return reasons

    @staticmethod
    def _pair_evaluation(
        row_left: Mapping[str, Any], row_right: Mapping[str, Any]
    ) -> dict[str, Any]:
        """What the two runs were scored against: evaluation provenance.

        Gold says how a run was judged, so it travels with the pair instead of
        gating it. ``same_gold`` is what the gold-reading deltas key off; the
        per-side detail says which gold each number came from, so a pair scored
        under two golds is never read as if the difference between them were an
        engine difference.
        """

        def side(row: Mapping[str, Any]) -> dict[str, Any]:
            structure = row.get("structure_metrics")
            reference = row.get("gold_reference")
            pages = structure.get("pages") if isinstance(structure, dict) else None
            return {
                "gold_digest": row.get("gold_digest"),
                "gold_pages": reference.get("pages") if isinstance(reference, dict) else None,
                "structure_status": (
                    structure.get("status") if isinstance(structure, dict) else None
                ),
                "structure_reason": (
                    structure.get("reason") if isinstance(structure, dict) else None
                ),
                "scored_pages": pages.get("scored") if isinstance(pages, dict) else None,
            }

        left, right = side(row_left), side(row_right)
        return {
            "same_gold": left["gold_digest"] == right["gold_digest"],
            "gold_digests": [left["gold_digest"], right["gold_digest"]],
            "a": left,
            "b": right,
        }

    def _quality_delta(
        self,
        row_left: Mapping[str, Any],
        row_right: Mapping[str, Any],
        reasons: Sequence[str],
    ) -> dict[str, dict[str, Any]]:
        """Every quality delta of a pair: b minus a, each in its own unit.

        Accuracy, structure and fidelity travel together but never merge: CER is
        a ratio of edits to gold characters, a boundary F1 is a ratio of marks,
        and fidelity is the share of the artifact's characters the tree kept.
        Tokens are not here at all -- they are the pair's ``usage`` -- so the arm
        that read more can never be read as the worse one.

        A pair that is not compatible measures nothing. A pair whose two sides
        were scored against different golds measures the reference-free figures
        only: the difference between two references is not an engine difference.
        """
        gold_differs = row_left.get("gold_digest") != row_right.get("gold_digest")
        left, right = _row_metrics(row_left), _row_metrics(row_right)
        return {
            name: self._metric_delta(name, left.get(name), right.get(name), reasons, gold_differs)
            for name in DELTA_METRICS
        }

    @staticmethod
    def _metric_delta(
        name: str,
        left: Any,
        right: Any,
        reasons: Sequence[str],
        gold_differs: bool,
    ) -> dict[str, Any]:
        """One metric's b-minus-a, keeping the unit and both sides' counts."""
        if reasons:
            return _delta_miss("runs_incompatible")
        if gold_differs and name in GOLD_DEPENDENT_METRICS:
            return _delta_miss("different_gold")
        if not isinstance(left, dict) or not isinstance(right, dict):
            return _delta_miss("not_measured")
        if left.get("status") != MEASURED or right.get("status") != MEASURED:
            return _delta_miss(
                "not_measured", a_reason=left.get("reason"), b_reason=right.get("reason")
            )
        delta = {
            "status": MEASURED,
            "reason": None,
            "value": right["value"] - left["value"],
            "unit": left.get("unit"),
            "denominator_unit": left.get("denominator_unit"),
            "a": left["value"],
            "b": right["value"],
            "a_numerator": left.get("numerator"),
            "a_denominator": left.get("denominator"),
            "b_numerator": right.get("numerator"),
            "b_denominator": right.get("denominator"),
        }
        if "estimate" in left or "estimate" in right:
            # A resource reading is sampled, and a delta between two samples is
            # one too. Accuracy metrics carry no such flag and get none.
            delta["estimate"] = bool(left.get("estimate") or right.get("estimate"))
        return delta

    def _chunk_delta(
        self, row_left: Mapping[str, Any], row_right: Mapping[str, Any], reasons: Sequence[str]
    ) -> dict[str, Any]:
        """Chunk-profile delta: chunks, and the share of them that span pages."""
        if reasons:
            return {"status": NOT_MEASURED, "reason": "runs_incompatible", "value": None}
        left = row_left.get("chunk_metrics") or {}
        right = row_right.get("chunk_metrics") or {}
        left_chunks, right_chunks = left.get("chunks"), right.get("chunks")
        if left_chunks is None or right_chunks is None:
            return {"status": NOT_MEASURED, "reason": "chunks_not_measured", "value": None}
        return {
            "status": MEASURED,
            "reason": None,
            "value": right_chunks - left_chunks,
            "unit": "chunks",
            "denominator_unit": "documents",
            "documents": 1,
            "chunks_a": left_chunks,
            "chunks_b": right_chunks,
        }

    # -- the study -------------------------------------------------------- #

    def study(
        self,
        *,
        families: Sequence[str],
        engines: Sequence[str],
        doc_ids: Sequence[str] | None = None,
        profiles: Sequence[str] = (),
        repeats: int = 1,
        allow_upload: bool = False,
        llm_mode: str = "none",
        allow_merge_upload: bool = False,
        gold_paths: Mapping[str, str] | None = None,
        page_indices: Sequence[int] | None = None,
        label: str | None = None,
        dry_run: bool = False,
        resume: bool = True,
    ) -> dict[str, Any]:
        """Run a whole study: every document, engine, repeat, then each family.

        Repeats alternate their order so a replayed request or a throttled one
        cannot be mistaken for an engine difference. Nothing runs before the
        corpus checks out and every permission the plan needs has been given.
        ``gold_paths`` names a gold file for a document the manifest declares
        without one.

        ``resume`` is the study's run policy: the default reuses a run only when
        it answers the identical request and is complete, and ``resume=False``
        runs everything again in fresh run ids. Either way a repeat is a run of
        its own: nothing is ever handed back as a repeat it did not make.
        """
        for family in families:
            if family not in FAMILIES:
                raise RunnerError(f"unknown family {family!r}; expected one of {list(FAMILIES)}")
        # The profiles are resolved before the plan, not after the dry run: a
        # dry run that reported a plan the run would refuse would be worse than
        # no plan at all, and a refusal after the extractions have run is
        # worse still.
        ocr_only = set(families) == {"ocr"}
        tree_profiles = (
            [] if ocr_only else self._study_profiles(engines, profiles, llm_mode=llm_mode)
        )
        plan = self.plan(
            doc_ids=doc_ids,
            engine_ids=engines,
            profile_names=profiles if ocr_only else tree_profiles,
            families=families,
            repeats=repeats,
            allow_upload=allow_upload,
            llm_mode=llm_mode,
            page_indices=page_indices,
            resume=resume,
        )
        if dry_run:
            return {"plan": plan, "executed": False}
        # Only the runs that are actually going to happen need a permission. A
        # resume that answers every request from disk makes no call, so it has
        # nothing to permit and must not be made to ask.
        pending = [entry for entry in plan["extractions"] if entry["cache"] == "miss"]
        if pending and any(engine in PAID_ENGINES for engine in engines) and not allow_upload:
            raise UploadNotAllowedError(
                "study includes a paid engine; pass --allow-upload to permit the uploads"
            )
        # Every merge the trees would run is settled now. A tree is built after
        # the extractions it reads have been made, so a refusal that waited
        # until then would have spent the tokens first.
        for profile_name in tree_profiles:
            if not any(
                entry["cache"] == "miss" and entry["profile"] == profile_name
                for entry in plan["trees"]
            ):
                continue
            self._merge_mode(
                get_profile(profile_name), llm_mode, allow_merge_upload=allow_merge_upload
            )
        wanted_docs = list(plan["corpus"]["documents"])
        corpus = self.validate(doc_ids=wanted_docs)
        extractions: list[dict[str, Any]] = []
        trees: list[dict[str, Any]] = []
        for repeat in range(1, repeats + 1):
            order = list(engines)
            if repeat % 2 == 0:
                order.reverse()  # alternate, so ordering cannot be the story
            for doc_id in wanted_docs:
                for engine_id in order:
                    manifest = self.extract(
                        doc_id,
                        engine_id,
                        repeat=repeat,
                        page_indices=page_indices,
                        allow_upload=allow_upload,
                        label=label,
                        # Under ``resume`` the same request already on disk is
                        # replayed, not sent again; under ``fresh`` every
                        # repeat runs again and lands in a run id of its own.
                        resume=resume,
                    )
                    extractions.append(manifest)
                for profile_name in tree_profiles:
                    profile = get_profile(profile_name)
                    matching = [
                        entry
                        for entry in extractions
                        if entry.get("doc_id") == doc_id
                        and entry.get("engine_id") == profile.engine
                        and entry.get("repeat") == repeat
                        and entry.get("status") == RUN_OK
                    ]
                    if not matching:
                        continue
                    trees.append(
                        self.run_tree(
                            doc_id,
                            profile_name,
                            extract_run_id=matching[-1]["run_id"],
                            repeat=repeat,
                            llm_mode=llm_mode,
                            allow_merge_upload=allow_merge_upload,
                            label=label,
                            resume=resume,
                        )
                    )
        comparisons = []
        skipped: list[dict[str, Any]] = []
        for family in families:
            for doc_id in wanted_docs:
                refs = self._family_refs(family, doc_id, extractions, trees)
                if not refs:
                    # A family that could not run is named with the thing it was
                    # missing, so an empty comparison list is never silent.
                    skipped.append(
                        {
                            "family": family,
                            "doc_id": doc_id,
                            "reason": self._skip_reason(family, doc_id, extractions, trees),
                        }
                    )
                    continue
                comparisons.append(
                    self.compare(family, refs, gold_paths=gold_paths, label=label or doc_id)
                )
        return {
            "plan": plan,
            "executed": True,
            "corpus": corpus,
            "extractions": [
                {
                    "run_id": entry.get("run_id"),
                    "status": entry.get("status"),
                    "artifact_digest": entry.get("artifact_digest"),
                    "path": entry.get("path"),
                }
                for entry in extractions
            ],
            "trees": [
                {
                    "run_id": entry.get("run_id"),
                    "profile": entry.get("profile"),
                    "status": entry.get("status"),
                    "rankable": entry.get("rankable"),
                    "semantic_signature": entry.get("semantic_signature"),
                    "path": entry.get("path"),
                }
                for entry in trees
            ],
            "comparisons": [
                {
                    "family": entry["family"],
                    "path": entry.get("path"),
                    "counts": entry["report"]["counts"],
                }
                for entry in comparisons
            ],
            # The grid the study expected, cell by cell: a participant that
            # failed is a filled cell, and a family that could not run is named
            # in ``skipped`` rather than left as an absence.
            "grid": [
                {"family": family, **cell}
                for family in families
                for doc_id in wanted_docs
                for cell in self._family_cells(family, doc_id, extractions, trees)
            ],
            # The whole corpus, per family, participant and pair: the roll-up
            # the per-document comparisons above cannot give on their own.
            "summary": self._study_summary(comparisons),
            # What the study's work spent, counted the one way its reports cannot:
            # each run once, rather than once per family that named it.
            "usage": self._study_usage([*extractions, *trees]),
            "skipped": skipped,
        }

    def _study_usage(self, runs: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
        """What a study's runs spent: every run once, and what this invocation spent.

        ``unique_work`` counts each run once. A study extracts once and builds
        every tree from that extraction, so one run is named by several families
        and counted in each of their reports; summing those reports counts it
        once per family, and this block is what a total should be read from
        instead. No run is dropped here for sharing an artifact digest with
        another: two repeats of one request are two runs, two sets of counters
        apart.

        ``new_work`` is the same accounting over the runs this invocation
        actually executed. A run a resume handed back is still the study's work,
        but the invocation that wrote it spent its tokens, so it is not new
        spend.

        ``runs`` counts the runs themselves, failed and unpaired ones included --
        a requested run belongs in the denominator whatever it spent -- while the
        tokens count what was reported. A run with nothing to report (a tree that
        merged nothing) is a run in the counts and no document in the usage.

        A tree is counted for its merge alone. The pages it was built from were
        extracted once and that run carries their tokens, so a tree contributes
        no extraction part -- not a missing one, which would say the reading went
        unreported -- and a tree that merged nothing has no part at all. The run
        counts still name it: a run with nothing to report was still a run.
        """
        fresh = [run for run in runs if not run.get("replayed")]
        documents: list[dict[str, Any]] = []
        incurred: list[dict[str, Any]] = []
        for run in runs:
            merge = _merge_bill(run)
            document: dict[str, Any]
            if run.get("kind") == KIND_TREE:
                if merge is None:
                    continue
                document = {"doc_id": _run_label(run), "merge": merge}
            else:
                model, usage = self._usage_of(run, self._artifact_of(run))
                document = {"doc_id": _run_label(run), "model": model, "usage": usage}
            documents.append(document)
            if not run.get("replayed"):
                incurred.append(document)
        return {
            "unique_work": run_usage(documents),
            "new_work": run_usage(incurred),
            "runs": {
                "extractions": sum(1 for run in runs if run.get("kind") == KIND_EXTRACT),
                "trees": sum(1 for run in runs if run.get("kind") == KIND_TREE),
                "unique": len(runs),
                "replayed": len(runs) - len(fresh),
                "new": len(fresh),
            },
        }

    def _skip_reason(
        self,
        family: str,
        doc_id: str,
        extractions: Sequence[Mapping[str, Any]],
        trees: Sequence[Mapping[str, Any]],
    ) -> str:
        """What a family was missing for one document.

        The reason names the thing that was absent, so an empty comparison list
        is never silent. A family whose participants all failed is named apart
        from one whose arms were never built at all.
        """
        if family == "chunk":
            arms = {str(entry.get("profile")) for entry in trees if entry.get("doc_id") == doc_id}
            if not set(CHUNK_ARMS) <= arms:
                return "no_chunk_profile_tree"
            # Both arms exist, so what was missing is a merge: a merged tree that
            # merged no section is the atomic tree under another name.
            return "no_merged_section"
        if any(
            entry.get("doc_id") == doc_id and entry.get("status") == RUN_OK
            for entry in extractions
        ):
            return "no_primary_profile_tree"
        return "no_successful_extraction"

    def _study_profiles(
        self, engines: Sequence[str], profiles: Sequence[str], *, llm_mode: str
    ) -> list[str]:
        """The tree profiles a study builds, never dropping one it was handed.

        Without a list the study derives the profiles from the engines and leaves
        out the ones this ``llm_mode`` cannot run: that is a default, not a
        promise. A profile the caller named is a request, so a mode that would
        drop it, or an engine the study is not running, is refused instead.
        """
        if not profiles:
            names = [
                profile.name for engine_id in engines for profile in profiles_for_engine(engine_id)
            ]
            if llm_mode == "none":
                names = [name for name in names if get_profile(name).merge == "off"]
            return names
        for name in profiles:
            profile = get_profile(name)
            if profile.engine not in engines:
                raise RunnerError(
                    f"profile {name!r} reads the {profile.engine!r} engine, which this study "
                    f"does not run; add it to --engines or drop the profile"
                )
            if llm_mode == "none" and profile.merge != "off":
                raise RunnerError(
                    f"profile {name!r} merges details with a model, and --llm-mode none "
                    "would drop it; pass --llm-mode live to run it"
                )
        return list(profiles)

    def _family_cells(
        self,
        family: str,
        doc_id: str,
        extractions: Sequence[Mapping[str, Any]],
        trees: Sequence[Mapping[str, Any]],
    ) -> list[dict[str, Any]]:
        """The expected grid for one family and document: one cell per run.

        A cell is a participant in a repeat, filled by the run that measured it.
        The grid is what the report counts its denominators over, so it is built
        from the runs the study asked for — never from the ones that worked — and
        a failure stays a filled cell instead of leaving the grid.
        """
        if family == "ocr":
            return [
                _cell(entry, participant=entry.get("engine_id"), kind=KIND_EXTRACT)
                for entry in extractions
                if entry.get("doc_id") == doc_id
            ]
        candidates = [entry for entry in trees if entry.get("doc_id") == doc_id]
        if family == "chunk":
            return self._chunk_cells(candidates)
        cells = []
        for entry in candidates:
            engine = str(entry.get("engine_id"))
            if str(entry.get("profile")) != primary_profile(engine).name:
                continue
            cells.append(_cell(entry, participant=entry.get("profile"), kind=KIND_TREE))
        return cells

    def _chunk_cells(self, trees: Sequence[Mapping[str, Any]]) -> list[dict[str, Any]]:
        """The chunk arms per repeat: the grouped tree and the merged one.

        Both arms must read the same extraction artifact, or the comparison would
        count a merge against a different reading of the document, and the merged
        arm must have merged something, or it is the atomic tree under another
        name. A repeat that fails either test contributes no cells: one arm is
        not a comparison, and the study names the gap instead.
        """
        cells: list[dict[str, Any]] = []
        for repeat in sorted({_repeat_of(entry) for entry in trees}):
            by_profile = {
                str(entry.get("profile")): entry for entry in trees if _repeat_of(entry) == repeat
            }
            grouped = by_profile.get(CHUNK_ARMS[0])
            merged = by_profile.get(CHUNK_ARMS[1])
            if grouped is None or merged is None:
                continue
            if grouped.get("artifact_digest") != merged.get("artifact_digest"):
                continue
            if not _merge_rebuilt_section(merged.get("merge")):
                continue
            cells.append(_cell(grouped, participant=CHUNK_ARMS[0], kind=KIND_TREE))
            cells.append(_cell(merged, participant=CHUNK_ARMS[1], kind=KIND_TREE))
        return cells

    def _study_summary(self, comparisons: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
        """The whole corpus: one block per family, participant and pair.

        The per-document comparisons are above; this is the roll-up a study is
        read for. Repeats are averaged inside their document first and the
        document is the statistical unit, so a study cannot grow its sample by
        running one document more often.
        """
        families: dict[str, dict[str, Any]] = {}
        documents: set[str] = set()
        for comparison in comparisons:
            rows = list(comparison.get("runs") or [])
            if not rows:
                continue
            doc_id = str(rows[0].get("doc_id"))
            documents.add(doc_id)
            entry = families.setdefault(str(comparison["family"]), {"rows": [], "pairs": []})
            entry["rows"].append((doc_id, rows))
            entry["pairs"].append(
                (doc_id, comparison["report"]["paired_usage"].get("pairs") or [])
            )
        return {
            "documents": len(documents),
            "families": {
                family: self._family_summary(entry) for family, entry in sorted(families.items())
            },
        }

    def _family_summary(self, entry: Mapping[str, Any]) -> dict[str, Any]:
        """One family's corpus roll-up: per participant and per pair."""
        participants: dict[str, list[tuple[str, Mapping[str, Any]]]] = {}
        for doc_id, rows in entry["rows"]:
            for row in rows:
                name = _participant_of(row)
                if name is not None:
                    participants.setdefault(name, []).append((doc_id, row))
        pairs: dict[str, list[tuple[str, Mapping[str, Any]]]] = {}
        for doc_id, pair_list in entry["pairs"]:
            for pair in pair_list:
                # An incompatible pair measured nothing: it is already named in
                # the per-document report, and folding it in would add a zero
                # the study never measured.
                if not pair.get("compatible"):
                    continue
                # The name is order-free: the study alternates its engine order
                # between repeats, so the same pair arrives as (a, b) in one
                # repeat and (b, a) in the next and is still one pair.
                name = "|".join(sorted(str(part) for part in pair.get("participants") or []))
                pairs.setdefault(name, []).append((doc_id, pair))
        return {
            "documents": len({doc_id for doc_id, _ in entry["rows"]}),
            "participants": {
                name: self._participant_summary(items)
                for name, items in sorted(participants.items())
            },
            "pairs": {name: self._pair_summary(items) for name, items in sorted(pairs.items())},
        }

    def _participant_summary(
        self, items: Sequence[tuple[str, Mapping[str, Any]]]
    ) -> dict[str, Any]:
        """One participant over the corpus: pooled accuracy and its spread.

        ``micro`` pools every document's edits over its reference characters, so
        a long document weighs what its characters say; ``macro`` gives every
        document one vote whatever its length.
        """
        by_document = _documents_of(items)
        readings = {
            doc_id: [_row_metrics(row) for row in rows] for doc_id, rows in by_document.items()
        }
        block: dict[str, Any] = {"documents": len(by_document)}
        for metric in ("cer", "wer"):
            per_document = {
                suffix: {
                    doc_id: _mean_of([reading[f"{metric}_{suffix}"] for reading in rows])
                    for doc_id, rows in readings.items()
                }
                for suffix in ("micro", "macro")
            }
            block[metric] = {
                "micro": _pooled_over_documents(
                    {doc: value for doc, value in per_document["micro"].items() if value},
                    reason="no_measured_documents",
                ),
                "macro": _document_statistics(
                    {doc: value for doc, value in per_document["macro"].items() if value},
                    reason="no_measured_documents",
                ),
            }
        return block

    def _pair_summary(self, items: Sequence[tuple[str, Mapping[str, Any]]]) -> dict[str, Any]:
        """One pair over the corpus: every delta, averaged inside its document.

        The pair's repeats are averaged within their document first, so a pair
        measured three times on one document is one observation. A delta that
        abstained -- two different golds, say -- is absent rather than a zero.
        """
        by_document = _documents_of(items)
        metrics: dict[str, Any] = {}
        for name in DELTA_METRICS:
            per_document = {
                doc_id: _mean_of(
                    [_oriented(pair, (pair.get("metrics") or {}).get(name)) for pair in pairs]
                )
                for doc_id, pairs in by_document.items()
            }
            metrics[name] = _document_statistics(
                {doc: value for doc, value in per_document.items() if value},
                reason="no_measured_pairs",
            )
        return {
            "documents": len(by_document),
            "n_pairs": len(items),
            "n_repeats": len({pair.get("repeat") for _, pair in items} - {None}),
            "metrics": metrics,
        }

    def _family_refs(
        self,
        family: str,
        doc_id: str,
        extractions: Sequence[Mapping[str, Any]],
        trees: Sequence[Mapping[str, Any]],
    ) -> list[RunRef]:
        """The runs one family compares for one document, in a stable order.

        Every cell of the expected grid becomes a ref, so a failed run keeps its
        row beside the successes and two repeats of one participant stay two
        rows. A grid with nothing successful in it is not a comparison at all:
        it is a gap, and the caller names it instead of comparing failures.
        """
        cells = self._family_cells(family, doc_id, extractions, trees)
        if not any(cell["status"] == RUN_OK for cell in cells):
            return []
        return [RunRef(doc_id, str(cell["run_id"])) for cell in cells]
