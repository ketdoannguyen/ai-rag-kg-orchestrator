"""Isolated local engine workers: one run in its own process, measured where it runs.

Why a process of its own
------------------------
A local engine loads its weights into the process that runs it, so a caller that
samples its own tree samples itself: that heap already holds every model an
earlier turn left behind, and the "engine peak" it reports depends on what the
caller did before rather than on the engine being measured. Two engines measured
that way cannot be compared, and neither number is a cost.

:func:`isolated_extract` therefore starts a fresh interpreter for one local run
and measures *that* process tree, from before the engine builds its converter
until the process exits. The child hands the artifact back through a file: never
through its stdout, where a document's worth of text can fill the pipe and
deadlock both ends, and never through a pipe at all for the same reason its
stderr goes to a file.

What the report says, and what it refuses to say
------------------------------------------------
``resources``
    The :func:`~research.ocr_pipeline.resources.measure_process_tree` observation
    of the worker's own tree: wall seconds, CPU seconds and peak RSS, each an
    interval estimate with the interval recorded beside it.
``worker``
    What only the child can know: its pid and interpreter, the engine version, the
    device its runtime reports, how long the converter took to build, and how many
    requests this process served. ``model_process_reused`` is ``True`` only when
    this process is known to have served an earlier request with the model already
    loaded — a one-shot worker never does, so repeat 2 of a fresh worker is
    ``cold``. Warmth the OS page cache could supply is a different scope and is
    recorded as an unknown one, never as warm.
``not_available``
    A missing engine travels back as :class:`EngineNotAvailableError`, and a failed
    run as an :class:`IsolatedWorkerError` that still carries the resources it used.

Protocol
--------
The parent writes ``request.json`` into a private temporary directory and starts
``python -m research.ocr_pipeline.engines.worker <request.json>``. The child
writes ``result.json`` beside it — ``ok`` with the artifact, ``not_available``
with the reason, or ``error`` with the class name alone — and exits 0 once it has.
A child that exits without a result crashed, and the parent says so.

An engine routed to another interpreter (:data:`ENGINE_PYTHON_ENV`) is started with
a bootstrap instead of ``-m``: the caller's import path is appended to the child's
``sys.path`` *after* its own site-packages are in place. That path has to reach the
child — the code that started the run lives on it — but through ``PYTHONPATH`` it
would be read *first*, and the caller's packages would shadow the engine's: the
version conflict the override exists to avoid.

A request whose action is ``probe`` asks the child for its interpreter's answer to
the adapter's availability probe instead of a run. It travels the same way out and
back — the same launch path, the same result file — and the adapter's probe inside
it still only reads distribution metadata, so the answer says whether a routed
engine is installed *in the interpreter it would run in*, which is a question this
interpreter cannot answer about another one.
"""

from __future__ import annotations

import json
import os
import signal
import subprocess
import sys
import tempfile
import time
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from research.ocr_pipeline.resources import (
    DEFAULT_INTERVAL_SECONDS,
    accelerator,
    measure_process_tree,
)

from . import (
    ENGINE_IDS,
    ISOLATED_ENGINES,
    EngineArtifact,
    EngineBlock,
    EngineNotAvailableError,
    EnginePage,
    EngineProbe,
    load_engine,
)

PROTOCOL_VERSION = 1
"""Layout version of the request and result files."""

REQUEST_NAME = "request.json"
RESULT_NAME = "result.json"
"""The two files a run owns inside its temporary directory."""

RESULT_OK = "ok"
RESULT_NOT_AVAILABLE = "not_available"
RESULT_ERROR = "error"
"""The three results a worker may write; anything else is a broken run."""

ACTION_EXTRACT = "extract"
PROBE_ACTION = "probe"
"""What a request asks the child for: an engine run, or an availability probe.

An extraction request may leave the action out — every request written before the
key existed is one — and a request naming any other action is refused rather than
run as something else.
"""

WARMTH_COLD = "cold"
WARMTH_WARM = "warm"
"""Warmth of a run: warm only with evidence that a loaded model process was reused."""

ACCELERATOR_ENGINES = ("marker", "docling")
"""Engines whose own stack carries a compute runtime worth asking about."""

UNSUPPORTED_IN_WORKER = ("vision",)
"""Engines that cannot run here: vision is a paid client whose backend is unobservable."""

ENGINE_PYTHON_ENV = "OCR_RESEARCH_ENGINE_PYTHON"
"""Where a per-engine interpreter is named, as ``docling=/path/to/python``.

Engines whose pins cannot coexist are kept apart without forcing them into one
environment: the override runs that engine in its own interpreter, and the
study records which interpreter it was. Entries are comma-separated; an engine
with no entry runs in the caller's own interpreter.
"""

WORKER_MODULE = "research.ocr_pipeline.engines.worker"

_BOOTSTRAP = (
    "import runpy, sys\n"
    "module, request, *caller_path = sys.argv[1:]\n"
    "sys.path.extend(caller_path)\n"
    "sys.argv = [module, request]\n"
    "runpy.run_module(module, run_name='__main__', alter_sys=True)\n"
)
"""What a worker in another interpreter starts as: the caller's path, appended, then the worker.

``PYTHONPATH`` cannot carry that path for this child, because it would be read before the
child's own site-packages instead of after them. The entries arrive as arguments instead
and are appended here, once ``site`` has already given the child's packages their place;
the worker module is then run exactly as ``-m`` would have run it.
"""

DEFAULT_TIMEOUT_SECONDS = 3600.0
"""How long a worker may run before it is killed; a hung engine must not hang the study."""

PROBE_TIMEOUT_SECONDS = 30.0
"""How long an interpreter is given to answer an availability probe.

The probe is a distribution-metadata read in a fresh interpreter, which takes a
fraction of a second; an interpreter that has not answered in half a minute is not
going to, and a dry run must not be made to wait on one that cannot start.
"""

_REAP_SECONDS = 30.0
"""How long to wait for a killed worker to be reaped."""

_STDERR_TAIL_CHARS = 500
"""How much of a crashed worker's stderr a parent may quote back."""

__all__ = [
    "ACCELERATOR_ENGINES",
    "ACTION_EXTRACT",
    "DEFAULT_TIMEOUT_SECONDS",
    "ENGINE_PYTHON_ENV",
    "PROBE_ACTION",
    "PROBE_TIMEOUT_SECONDS",
    "PROTOCOL_VERSION",
    "REQUEST_NAME",
    "RESULT_ERROR",
    "RESULT_NAME",
    "RESULT_NOT_AVAILABLE",
    "RESULT_OK",
    "UNSUPPORTED_IN_WORKER",
    "WARMTH_COLD",
    "WARMTH_WARM",
    "IsolatedRun",
    "IsolatedWorkerError",
    "artifact_from_payload",
    "artifact_payload",
    "engine_interpreter",
    "engine_probe",
    "engine_python",
    "isolated_extract",
    "warmth",
    "worker_main",
]


class IsolatedWorkerError(RuntimeError):
    """An isolated run that produced no usable artifact.

    ``resources`` and ``worker`` keep what was observed before the failure —
    including on a timeout, when the child had no chance to report anything — so a
    failed run still shows the memory and CPU it spent instead of nothing at all.
    The engine's own error body is never carried: only its class name is.
    """

    def __init__(
        self,
        message: str,
        *,
        resources: Mapping[str, Any] | None = None,
        worker: Mapping[str, Any] | None = None,
    ) -> None:
        super().__init__(message)
        self.resources: dict[str, Any] = dict(resources or {})
        self.worker: dict[str, Any] = dict(worker or {})


@dataclass(frozen=True)
class IsolatedRun:
    """One local engine run: its artifact, the resources it used, and the worker's report."""

    artifact: EngineArtifact
    resources: dict[str, Any]
    worker: dict[str, Any]

    @property
    def warmth(self) -> str:
        """``warm`` only with evidence that a loaded model process served this run again."""
        return warmth(self.worker)


def warmth(worker_report: Mapping[str, Any]) -> str:
    """The warmth a worker's own report supports, and no more than it supports.

    Only the reuse of a model process is evidence of warmth: a fresh worker
    re-reads whatever the OS page cache holds, which this side cannot see and
    therefore cannot claim.
    """
    return WARMTH_WARM if worker_report.get("model_process_reused") is True else WARMTH_COLD


# --------------------------------------------------------------------------- #
# the artifact's wire form
# --------------------------------------------------------------------------- #


def artifact_payload(artifact: EngineArtifact) -> dict[str, Any]:
    """One artifact as plain JSON data, field for field."""
    return {
        "engine_id": artifact.engine_id,
        "engine_version": artifact.engine_version,
        "doc_id": artifact.doc_id,
        "status": artifact.status,
        "pdf_sha256": artifact.pdf_sha256,
        "source_sha256": artifact.source_sha256,
        "render_dpi": artifact.render_dpi,
        "render_page_offset": artifact.render_page_offset,
        "prompt_sha256": artifact.prompt_sha256,
        "public_config": dict(artifact.public_config),
        "normalization": dict(artifact.normalization),
        "notes": list(artifact.notes),
        "mapping_version": artifact.mapping_version,
        "artifact_version": artifact.artifact_version,
        "pages": [_page_payload(page) for page in artifact.pages],
    }


def artifact_from_payload(payload: Mapping[str, Any]) -> EngineArtifact:
    """Rebuild the artifact the worker wrote, refusing anything malformed.

    Every field is validated by the artifact's own constructors, so a result file
    that lost a page or a block fails here instead of becoming a smaller artifact
    nobody notices.
    """
    try:
        pages = tuple(_page_from_payload(entry) for entry in payload["pages"])
        artifact = EngineArtifact(
            engine_id=payload["engine_id"],
            engine_version=payload["engine_version"],
            doc_id=payload["doc_id"],
            status=payload["status"],
            pdf_sha256=payload["pdf_sha256"],
            source_sha256=payload["source_sha256"],
            render_dpi=payload["render_dpi"],
            render_page_offset=payload["render_page_offset"],
            prompt_sha256=payload["prompt_sha256"],
            public_config=payload["public_config"],
            normalization=payload["normalization"],
            notes=tuple(payload["notes"]),
            mapping_version=payload["mapping_version"],
            artifact_version=payload["artifact_version"],
            pages=pages,
        )
    except (KeyError, TypeError, ValueError) as exc:
        raise IsolatedWorkerError(f"the worker's artifact could not be read: {exc!r}") from exc
    return artifact


def _page_payload(page: EnginePage) -> dict[str, Any]:
    return {
        "page_index": page.page_index,
        "request_page": page.request_page,
        "running": page.running,
        "printed_folio": page.printed_folio,
        "continues_previous": page.continues_previous,
        "status": page.status,
        "usage": dict(page.usage) if page.usage is not None else None,
        "blocks": [_block_payload(block) for block in page.blocks],
    }


def _page_from_payload(payload: Mapping[str, Any]) -> EnginePage:
    usage = payload["usage"]
    return EnginePage(
        page_index=payload["page_index"],
        request_page=payload["request_page"],
        running=payload["running"],
        printed_folio=payload["printed_folio"],
        continues_previous=payload["continues_previous"],
        status=payload["status"],
        usage=None if usage is None else {str(key): int(value) for key, value in usage.items()},
        blocks=tuple(_block_from_payload(entry) for entry in payload["blocks"]),
    )


def _block_payload(block: EngineBlock) -> dict[str, Any]:
    return {
        "kind": block.kind,
        "normalized_kind": block.normalized_kind,
        "text": block.text,
        "marker": block.marker,
        "depth": block.depth,
        "group": block.group,
        "bbox": list(block.bbox) if block.bbox is not None else None,
        "source_id": block.source_id,
    }


def _block_from_payload(payload: Mapping[str, Any]) -> EngineBlock:
    bbox = payload["bbox"]
    return EngineBlock(
        kind=payload["kind"],
        normalized_kind=payload["normalized_kind"],
        text=payload["text"],
        marker=payload["marker"],
        depth=payload["depth"],
        group=payload["group"],
        bbox=None if bbox is None else (bbox[0], bbox[1], bbox[2], bbox[3]),
        source_id=payload["source_id"],
    )


# --------------------------------------------------------------------------- #
# the child: one request, one result file
# --------------------------------------------------------------------------- #


def worker_main(argv: Sequence[str] | None = None) -> int:
    """The worker's entry point: run ``argv[0]`` as a request and write the result.

    Exit codes say whether a result exists, not whether the engine succeeded: the
    result file carries the engine's own outcome, so a parent can always tell a
    reported failure from a dead process.
    """
    arguments = list(sys.argv[1:] if argv is None else argv)
    if len(arguments) != 1:
        sys.stderr.write(f"usage: python -m {WORKER_MODULE} {REQUEST_NAME}\n")
        return 2
    request_path = Path(arguments[0])
    try:
        request = json.loads(request_path.read_text(encoding="utf-8"))
    except (OSError, ValueError) as exc:
        sys.stderr.write(f"{REQUEST_NAME} could not be read: {type(exc).__name__}\n")
        return 2
    if not isinstance(request, Mapping):
        sys.stderr.write(f"{REQUEST_NAME} must be a JSON object\n")
        return 2
    result = _answer_request(request)
    (request_path.parent / RESULT_NAME).write_text(json.dumps(result), encoding="utf-8")
    return 0


def _answer_request(request: Mapping[str, Any]) -> dict[str, Any]:
    """One request's result: the action it names, or a refusal to run as another.

    An extraction request need not name its action — every request written before
    the key existed is one — so the default is the engine run. An action this
    worker does not know is reported as such, never run as something else.
    """
    action = request.get("action", ACTION_EXTRACT)
    if action == ACTION_EXTRACT:
        return _run_request(request)
    if action == PROBE_ACTION:
        return _probe_request(request)
    return {
        "status": RESULT_ERROR,
        "error_class": "UnknownAction",
        "worker": _probe_report(request.get("engine_id")),
    }


def _probe_request(request: Mapping[str, Any]) -> dict[str, Any]:
    """The adapter's own availability probe, run in the interpreter that answers it.

    Every adapter's probe reads distribution metadata and nothing else — no engine
    import, no network, no weight check — so it costs a metadata read wherever it
    runs. What this child adds is *where* it is asked: the verdict is about this
    interpreter, which is the one the engine would run in.
    """
    engine_id = str(request.get("engine_id"))
    report = _probe_report(engine_id)
    if request.get("protocol") != PROTOCOL_VERSION:
        return {"status": RESULT_ERROR, "error_class": "ProtocolMismatch", "worker": report}
    try:
        probe = load_engine(engine_id).probe()
    except Exception as exc:  # a probe that raises is a probe that says no
        return {"status": RESULT_ERROR, "error_class": type(exc).__name__, "worker": report}
    return {
        "status": RESULT_OK,
        "probe": {
            "engine_id": probe.engine_id,
            "available": bool(probe.available),
            "version": probe.version,
            "detail": probe.detail,
        },
        "worker": report,
    }


def _probe_report(engine_id: Any) -> dict[str, Any]:
    """What only the child can say about the probe it answered: the interpreter it ran in.

    The executable is not decoration: it is the interpreter the verdict is about,
    and the parent carries it into the detail a plan prints.
    """
    return {
        "protocol": PROTOCOL_VERSION,
        "pid": os.getpid(),
        "executable": sys.executable,
        "python": sys.version.split()[0],
        "engine_id": engine_id,
    }


def _run_request(request: Mapping[str, Any]) -> dict[str, Any]:
    """One engine run, reported as data: never a traceback and never an engine body."""
    report: dict[str, Any] = {
        "protocol": PROTOCOL_VERSION,
        "pid": os.getpid(),
        # Which interpreter, not just which version of Python: an engine kept in its own
        # environment runs the same version as the app while holding different pins, so
        # the version alone cannot say which pins an artifact was produced against.
        "executable": sys.executable,
        "python": sys.version.split()[0],
        "engine_id": request.get("engine_id"),
        "device": _device(request.get("engine_id")),
        "load_scope": "none",
        "load_seconds": None,
        "requests_served": 1,
        "model_process_reused": False,
        "cache": {
            "scope": "os_page_cache",
            "status": "unknown",
            "reason": "not_observed",
        },
    }
    if request.get("protocol") != PROTOCOL_VERSION:
        return {"status": RESULT_ERROR, "error_class": "ProtocolMismatch", "worker": report}
    engine_id = str(request["engine_id"])
    source = Path(request["source"])
    pages = request.get("page_indices")
    page_indices = None if pages is None else [int(index) for index in pages]
    try:
        module = load_engine(engine_id)
        converter, load_seconds = _build_converter(
            engine_id,
            module,
            page_indices=page_indices,
            engine_config=request.get("engine_config"),
        )
        report["load_scope"] = "converter_build" if converter is not None else "none"
        report["load_seconds"] = load_seconds
        artifact = _run_extract(
            engine_id,
            module,
            source,
            doc_id=str(request["doc_id"]),
            page_indices=page_indices,
            page_offset=int(request["page_offset"]),
            engine_config=request.get("engine_config"),
            converter=converter,
        )
    except EngineNotAvailableError as exc:
        return {"status": RESULT_NOT_AVAILABLE, "detail": exc.detail, "worker": report}
    except Exception as exc:  # the engine's failure, reported without its body
        return {"status": RESULT_ERROR, "error_class": type(exc).__name__, "worker": report}
    report["engine_version"] = artifact.engine_version
    return {"status": RESULT_OK, "artifact": artifact_payload(artifact), "worker": report}


def _device(engine_id: Any) -> dict[str, Any]:
    """The device the engine's own runtime reports, or why nothing was asked.

    Only an engine whose stack carries an accelerator runtime is asked: importing
    one for an engine that has none would charge that engine for an import it
    never makes.
    """
    if engine_id not in ACCELERATOR_ENGINES:
        return {
            "status": "not_available",
            "reason": "engine_has_no_accelerator",
            "value": None,
            "unit": "device",
            "estimate": False,
        }
    return accelerator()


def _build_converter(
    engine_id: str,
    module: Any,
    *,
    page_indices: Sequence[int] | None,
    engine_config: Mapping[str, Any] | None,
) -> tuple[Any, float | None]:
    """Build the engine's converter and time it, when the adapter has one.

    Marker's ``build_converter`` creates the predictors and downloads whatever
    weights are missing; Docling's imports its document stack. Both are the load a
    reused model process would skip, so both are timed here and reported as the
    run's ``load_seconds``. The text layer has nothing to build.
    """
    builder = getattr(module, "build_converter", None)
    if not callable(builder):
        return None, None
    started = time.monotonic()
    converter = (
        builder(page_indices=page_indices, config=engine_config)
        if engine_id == "marker"
        else builder()
    )
    return converter, round(time.monotonic() - started, 6)


def _run_extract(
    engine_id: str,
    module: Any,
    source: Path,
    *,
    doc_id: str,
    page_indices: Sequence[int] | None,
    page_offset: int,
    engine_config: Mapping[str, Any] | None,
    converter: Any,
) -> EngineArtifact:
    """Run one adapter in exactly the call shape that adapter declares."""
    pages = None if page_indices is None else list(page_indices)
    if engine_id == "marker":
        return module.extract(
            source,
            doc_id=doc_id,
            page_indices=pages,
            page_offset=page_offset,
            converter=converter,
            config=engine_config,
        )
    if engine_id == "docling":
        return module.extract(
            source,
            doc_id=doc_id,
            page_indices=pages,
            page_offset=page_offset,
            converter=converter,
        )
    return module.extract(source, doc_id=doc_id, page_indices=pages, page_offset=page_offset)


# --------------------------------------------------------------------------- #
# the parent: one worker, measured from before the load until it exits
# --------------------------------------------------------------------------- #


def engine_python(engine_id: str, *, environ: Mapping[str, str] | None = None) -> str:
    """The interpreter ``engine_id`` runs in: its override, else this interpreter.

    Raises:
        ValueError: if the override names an interpreter that is not a file. A
            typo must fail before a study spends time, not as a worker that
            cannot start.
    """
    source = os.environ if environ is None else environ
    for entry in source.get(ENGINE_PYTHON_ENV, "").split(","):
        name, separator, value = entry.partition("=")
        if not separator or name.strip() != engine_id:
            continue
        interpreter = value.strip()
        if not interpreter:
            break
        if not Path(interpreter).is_file():
            raise ValueError(
                f"{ENGINE_PYTHON_ENV} names {interpreter!r} for engine "
                f"{engine_id!r}, which is not a file"
            )
        return interpreter
    return sys.executable


def engine_interpreter(engine_id: str) -> str | None:
    """The interpreter ``engine_id`` is routed to, or ``None`` when it runs here.

    The routing rule ``engine_probe`` follows, for callers that must know whether
    a question costs a process before asking it: only an engine this module can
    isolate is routed at all, and an engine whose interpreter is this one is not
    routed either — it is here, and answering costs nothing.
    """
    if engine_id not in ISOLATED_ENGINES:
        return None
    interpreter = engine_python(engine_id)
    return None if _same_interpreter(interpreter) else interpreter


def engine_probe(
    engine_id: str, module: Any, *, timeout_seconds: float = PROBE_TIMEOUT_SECONDS
) -> EngineProbe:
    """Whether ``engine_id`` can run, asked in the interpreter it would run in.

    ``module`` is the adapter this process already holds, and it answers directly
    when the engine runs here. An engine routed to an interpreter of its own is a
    different question: what matters is the distribution *that* interpreter can
    see, so the adapter's own metadata-only probe is run there, over the launch
    path a run uses. An answer from another interpreter always names it, so
    "not installed" is never read as "not installed here" when the engine was
    never going to run here.

    This is not a run and is not measured as one: no adapter's probe imports the
    engine, so nothing is loaded and nothing is spent.
    """
    interpreter = engine_interpreter(engine_id)
    if interpreter is None:
        return module.probe()
    probe = _probe_in_interpreter(engine_id, interpreter, timeout_seconds=timeout_seconds)
    return EngineProbe(
        engine_id, probe.available, probe.version, f"{probe.detail} (asked {interpreter})"
    )


def _probe_in_interpreter(
    engine_id: str, interpreter: str, *, timeout_seconds: float
) -> EngineProbe:
    """The given interpreter's answer to the adapter's probe, or why it gave none.

    Every way of not hearing an answer — a child that cannot start, writes no
    result, overruns the timeout, or reports a failure — comes back as an
    unavailable probe carrying the reason. A probe must never claim an engine that
    a run in that interpreter could not load, and it must never raise into a plan.
    """
    with tempfile.TemporaryDirectory(prefix="ocr-probe-") as work:
        directory = Path(work)
        request_path = directory / REQUEST_NAME
        stderr_path = directory / "worker.stderr"
        request_path.write_text(
            json.dumps(
                {"protocol": PROTOCOL_VERSION, "action": PROBE_ACTION, "engine_id": engine_id}
            ),
            encoding="utf-8",
        )
        command, env = _worker_launch(interpreter, request_path)
        with stderr_path.open("wb") as log:
            try:
                process = subprocess.Popen(
                    command,
                    stdin=subprocess.DEVNULL,
                    stdout=subprocess.DEVNULL,
                    stderr=log,
                    env=env,
                    start_new_session=True,
                )
            except OSError as exc:
                return EngineProbe(engine_id, False, None, f"cannot be started: {exc}")
            try:
                process.wait(timeout=timeout_seconds)
                payload = _read_result(process, directory / RESULT_NAME, stderr_path, resources={})
            except subprocess.TimeoutExpired:
                _terminate(process)
                return EngineProbe(
                    engine_id, False, None, f"did not answer within {timeout_seconds:g}s"
                )
            except IsolatedWorkerError as exc:
                return EngineProbe(engine_id, False, None, str(exc))
    if payload.get("status") != RESULT_OK:
        return EngineProbe(engine_id, False, None, f"probe raised {payload.get('error_class')}")
    probe = payload.get("probe")
    if not isinstance(probe, Mapping):
        return EngineProbe(engine_id, False, None, "the worker wrote no probe in its result")
    return EngineProbe(
        engine_id,
        bool(probe.get("available")),
        probe.get("version"),
        str(probe.get("detail") or "the probe gave no detail"),
    )


def isolated_extract(
    engine_id: str,
    source: str | Path,
    *,
    doc_id: str,
    page_indices: Sequence[int] | None = None,
    page_offset: int = 0,
    engine_config: Mapping[str, Any] | None = None,
    python: str | None = None,
    timeout_seconds: float = DEFAULT_TIMEOUT_SECONDS,
    interval_seconds: float = DEFAULT_INTERVAL_SECONDS,
    temp_root: str | Path | None = None,
) -> IsolatedRun:
    """Run one local engine in a worker process and measure that process's tree.

    The sampler starts before the child could have loaded anything and stops when
    the child is gone, so a load that happens inside the run is inside the
    measurement. The temporary directory — request, result and the worker's stderr
    — is removed on every path, and a worker that overruns ``timeout_seconds`` is
    killed with its process group and reaped before this returns.

    ``temp_root`` is where the run's private directory is created; it must exist.
    A caller that wants to prove nothing leaked passes a directory it owns.
    """
    if engine_id in UNSUPPORTED_IN_WORKER:
        raise ValueError(f"engine {engine_id!r} cannot run in a worker")
    if engine_id not in ENGINE_IDS:
        raise ValueError(f"unknown engine_id: {engine_id!r}; expected one of {ENGINE_IDS}")
    if not timeout_seconds > 0:
        raise ValueError(f"timeout_seconds must be > 0, got {timeout_seconds!r}")
    interpreter = engine_python(engine_id) if python is None else python

    path = Path(source).resolve()
    with tempfile.TemporaryDirectory(prefix="ocr-worker-", dir=temp_root) as work:
        directory = Path(work)
        request_path = directory / REQUEST_NAME
        stderr_path = directory / "worker.stderr"
        request_path.write_text(
            json.dumps(
                {
                    "protocol": PROTOCOL_VERSION,
                    "engine_id": engine_id,
                    "source": str(path),
                    "doc_id": doc_id,
                    "page_indices": None if page_indices is None else list(page_indices),
                    "page_offset": page_offset,
                    "engine_config": None if engine_config is None else dict(engine_config),
                }
            ),
            encoding="utf-8",
        )
        command, env = _worker_launch(interpreter, request_path)
        with stderr_path.open("wb") as log:
            process = subprocess.Popen(
                command,
                stdin=subprocess.DEVNULL,
                stdout=subprocess.DEVNULL,
                stderr=log,
                env=env,
                start_new_session=True,
            )
            resources = _measure_until_exit(
                process, timeout_seconds=timeout_seconds, interval_seconds=interval_seconds
            )
        payload = _read_result(process, directory / RESULT_NAME, stderr_path, resources=resources)
    return _unpack(engine_id, payload, resources, process.pid)


def _worker_launch(interpreter: str, request_path: Path) -> tuple[list[str], dict[str, str]]:
    """How to start the worker: its command line and the environment it starts in.

    This interpreter's import path has to reach the child — that is where the code that
    started the run lives — but where it may sit depends on the child. A child in *this*
    interpreter starts with that path in ``PYTHONPATH``, which is how every worker has
    always started. A child in another interpreter has site-packages of its own, and
    ``PYTHONPATH`` would put the caller's *ahead* of them: the caller's ``torch`` would
    shadow the engine's, and an engine kept apart precisely because its pins cannot
    coexist with the app's would be loaded against the wrong ones. So that child is
    handed :data:`_BOOTSTRAP` instead, which appends the path after its own
    site-packages: reachable for the code that started the run, unable to shadow what
    the engine's interpreter was built with.
    """
    caller_path = _caller_path()
    if _same_interpreter(interpreter):
        return [interpreter, "-m", WORKER_MODULE, str(request_path)], _worker_env()
    env = dict(os.environ)
    # The path travels in the command line here; PYTHONPATH would place it first.
    env.pop("PYTHONPATH", None)
    return (
        [interpreter, "-c", _BOOTSTRAP, WORKER_MODULE, str(request_path), *caller_path],
        env,
    )


def _same_interpreter(interpreter: str) -> bool:
    """Whether ``interpreter`` is this one: the interpreter whose site-packages these are.

    Compared as a path, never as a resolved file: a venv's ``bin/python`` is a symlink to
    its base interpreter, so two venvs of one version resolve to the same file while
    starting children with different site-packages. This interpreter named by another path
    is read as another interpreter, which costs nothing — the child's site-packages are
    then this interpreter's, and the appended path answers the same imports.
    """
    return Path(interpreter).absolute() == Path(sys.executable).absolute()


def _worker_env() -> dict[str, str]:
    """The child's environment: this interpreter's import path, so it finds the same package.

    A worker is a fresh interpreter, and ``-m`` resolves ``research.ocr_pipeline``
    from its own ``sys.path``. Passing the parent's path through means the child
    imports the code that started it rather than whatever else is installed.
    """
    env = dict(os.environ)
    env["PYTHONPATH"] = os.pathsep.join(_caller_path())
    return env


def _caller_path() -> list[str]:
    """This interpreter's import path as entries: its own ``sys.path``, then its PYTHONPATH.

    An empty PYTHONPATH component means the working directory, which a child already has
    at the front of its path, so empty entries are left out.
    """
    entries = [entry for entry in sys.path if entry]
    entries.extend(part for part in os.environ.get("PYTHONPATH", "").split(os.pathsep) if part)
    return entries


def _measure_until_exit(
    process: subprocess.Popen[bytes], *, timeout_seconds: float, interval_seconds: float
) -> dict[str, Any]:
    """Wait for the worker inside a process-tree measurement, killing it on timeout.

    The observation is finalised before the timeout is raised, so the error a
    caller sees carries the resources the worker used up to the kill.
    """
    timed_out = False
    with measure_process_tree(process.pid, interval_seconds=interval_seconds) as resources:
        try:
            process.wait(timeout=timeout_seconds)
        except subprocess.TimeoutExpired:
            timed_out = True
        finally:
            # A worker the timeout or a crash left behind must not outlive this call.
            if process.poll() is None:
                _terminate(process)
    if timed_out:
        raise IsolatedWorkerError(
            f"the worker timed out after {timeout_seconds}s",
            resources=resources,
            worker={"pid": process.pid},
        )
    return resources


def _terminate(process: subprocess.Popen[bytes]) -> None:
    """Kill the worker and anything it started, then reap it: no orphan, no zombie."""
    if process.poll() is None:
        try:
            os.killpg(os.getpgid(process.pid), signal.SIGKILL)
        except OSError:  # the group is already gone; the process may still be there
            process.kill()
    try:
        process.wait(timeout=_REAP_SECONDS)
    except subprocess.TimeoutExpired:
        return  # the kernel still owns the pid; nothing on this side can do more


def _read_result(
    process: subprocess.Popen[bytes],
    result_path: Path,
    stderr_path: Path,
    *,
    resources: Mapping[str, Any],
) -> dict[str, Any]:
    """The worker's result file, or the reason there is none.

    A worker that died — a segfault, the OOM killer, a killed timeout — leaves no
    result, and its exit code is the only account of that. Whatever it managed to
    write to stderr is quoted back, bounded, because that is where a crash says
    what happened. What the run spent before it died travels with the error, so a
    worker killed mid-run is told apart from one that answered: the measurement
    taken up to its exit, and the pid the parent observed itself, which is the
    whole of the report when the child never got to write one.
    """
    worker: dict[str, Any] = {"pid": process.pid}
    if not result_path.is_file():
        raise IsolatedWorkerError(
            f"the worker exited with {process.returncode} without a result: {_stderr_tail(stderr_path)}",
            resources=resources,
            worker=worker,
        )
    try:
        payload = json.loads(result_path.read_text(encoding="utf-8"))
    except (OSError, ValueError) as exc:
        raise IsolatedWorkerError(
            f"the worker's result could not be read: {type(exc).__name__}",
            resources=resources,
            worker=worker,
        ) from exc
    if not isinstance(payload, dict):
        raise IsolatedWorkerError(
            f"the worker's result must be an object, got {type(payload)}",
            resources=resources,
            worker=worker,
        )
    return payload


def _stderr_tail(path: Path) -> str:
    """The end of the worker's stderr, bounded, or a note that there was none."""
    try:
        text = path.read_text(encoding="utf-8", errors="replace").strip()
    except OSError:
        return "no stderr was captured"
    if not text:
        return "the worker wrote nothing to stderr"
    return text[-_STDERR_TAIL_CHARS:]


def _unpack(
    engine_id: str,
    payload: Mapping[str, Any],
    resources: dict[str, Any],
    pid: int,
) -> IsolatedRun:
    """One worker's result, or the error its status stands for."""
    worker = dict(payload.get("worker") or {})
    # The pid is the parent's own observation of the process it started, so it is
    # there even when the child never got to report anything.
    worker.setdefault("pid", pid)
    status = payload.get("status")
    if status == RESULT_NOT_AVAILABLE:
        raise EngineNotAvailableError(
            engine_id, str(payload.get("detail", "")), resources=resources, worker=worker
        )
    if status != RESULT_OK:
        raise IsolatedWorkerError(
            f"the {engine_id} worker failed with {payload.get('error_class')}",
            resources=resources,
            worker=worker,
        )
    return IsolatedRun(
        artifact=artifact_from_payload(payload["artifact"]),
        resources=dict(resources),
        worker=worker,
    )


if __name__ == "__main__":  # pragma: no cover - the parent always starts this as a module
    raise SystemExit(worker_main())
