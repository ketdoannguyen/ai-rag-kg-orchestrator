"""Resource observations for one OCR research stage.

A stage is one engine invocation, wrapped by the caller in one of the two
context managers here: :func:`measure_process_tree` for a local engine, whose
work happens in a subprocess, and :func:`measure_hosted_stage` for an engine
behind a network call. Each yields a plain ``dict`` that is filled in when the
block ends, *including* when the body raises, so a failed stage keeps the
resources it used up to the failure. Only the stage itself belongs in the
observation: replay and report work must be measured separately, never wrapped
into an extraction peak.

What is measured, and what is refused
-------------------------------------
``wall``
    Monotonic elapsed seconds of the stage. Measured by both context managers:
    the clock needs no sampler and no platform support.
``cpu`` / ``rss_peak``
    Process-tree CPU seconds and peak resident set size of a local stage, read
    from ``/proc`` by a background thread that polls every ``interval_seconds``
    (50 ms by default). The tree is the root pid plus every descendant it can
    see, so a child worker's memory counts. Both quantities are interval
    observations, so they carry ``"estimate": True`` and the ``sampling`` block
    reports the interval, the number of usable samples and the largest tree size
    observed. RSS is a sum over the tree, so pages shared by a parent and its
    child are counted once per process. Only ``/proc/<pid>/stat`` and
    ``/proc/<pid>/statm`` are parsed — never ``cmdline`` or ``environ``, which
    can carry a credential — and this module writes no logs.
``gpu_vram``
    Device-scope VRAM in use, read from the DRM sysfs counters the kernel
    exposes when a driver provides them (``/sys/class/drm/card*/device/
    mem_info_vram_used``). No external tool is started and no package is
    installed to obtain it; where those counters do not exist the quantity is
    ``not_available``.
``accelerator``
    The compute device a local engine's own runtime reports, named only when the
    runtime is importable and answers: ``cuda`` or ``cpu``, with ``source``
    naming the runtime that said so. It is the engine's *stack* that is asked, so
    a run whose engine ships no accelerator runtime reports
    ``not_available``/``engine_has_no_accelerator`` rather than a guess.

A hosted engine runs on the provider's machine, so its backend memory, CPU and
GPU are not observable from here. :func:`measure_hosted_stage` reports wall time
and marks all three ``not_available``: the client's own RSS is never reported in
place of the backend's memory.

Statuses
--------
``measured``
    The quantity was observed; a measured zero is a real zero.
``not_measured``
    The source exists but produced nothing to report, e.g. the root process was
    already gone before the first sample.
``not_available``
    The platform or the backend cannot provide the quantity at all: no readable
    ``/proc``, no DRM VRAM counters, a hosted backend.

Every reading carries ``status``, ``reason``, ``value``, ``unit`` and
``estimate``, and contains no raw text.
"""

from __future__ import annotations

import importlib
import os
import threading
import time
from collections.abc import Iterator
from contextlib import contextmanager
from pathlib import Path
from typing import Any

from research.ocr_pipeline.metrics import MEASURED, NOT_MEASURED

NOT_AVAILABLE = "not_available"
"""The platform or backend cannot provide the quantity at all."""

DEFAULT_INTERVAL_SECONDS = 0.05
"""Default ``/proc`` poll interval, reported with every sampled observation."""

PROC_ROOT = "/proc"
"""Linux process filesystem, overridable so a caller can name another mount."""

DRM_ROOT = "/sys/class/drm"
"""DRM sysfs root, which may expose per-card VRAM counters."""

_JOIN_TIMEOUT_SECONDS = 5.0
"""Upper bound on waiting for the sampler thread to see the stop event."""

ACCELERATOR_RUNTIME = "torch"
"""The runtime asked for a compute device; only one a run's engine already ships."""

__all__ = [
    "ACCELERATOR_RUNTIME",
    "DEFAULT_INTERVAL_SECONDS",
    "DRM_ROOT",
    "NOT_AVAILABLE",
    "PROC_ROOT",
    "accelerator",
    "gpu_vram",
    "hosted_backend_memory",
    "measure_hosted_stage",
    "measure_process_tree",
]


# --------------------------------------------------------------------------- #
# reading helpers
# --------------------------------------------------------------------------- #


def _metric(
    value: float | int | str | None,
    status: str,
    *,
    unit: str,
    reason: str | None = None,
    estimate: bool = False,
) -> dict[str, Any]:
    """One resource reading. ``value`` is numeric, or a label such as a device name."""
    return {
        "status": status,
        "reason": reason,
        "value": value,
        "unit": unit,
        "estimate": estimate,
    }


def _measured(value: float | int, *, unit: str, estimate: bool = False) -> dict[str, Any]:
    return _metric(value, MEASURED, unit=unit, estimate=estimate)


def _hosted_miss(unit: str) -> dict[str, Any]:
    return _metric(None, NOT_AVAILABLE, unit=unit, reason="hosted_backend")


def _read_int(path: Path) -> int | None:
    """Read one integer from a kernel file, or ``None`` when it is not there."""
    try:
        return int(path.read_text(encoding="ascii").strip())
    except (OSError, ValueError):
        return None


def _sysconf(name: str) -> int | None:
    """One ``os.sysconf`` value, or ``None`` where the platform lacks it."""
    try:
        return int(os.sysconf(name))
    except (OSError, ValueError):
        return None


def _process_stat(pid: int, proc_root: Path) -> tuple[int, int, float] | None:
    """``(ppid, starttime_ticks, cpu_seconds)`` for one pid, or ``None``.

    ``/proc/<pid>/stat`` puts the executable name in parentheses, where it may
    contain spaces and parentheses of its own, so the fields are split after the
    last ``)``. ``starttime`` is kept so a recycled pid can be told apart from
    the process it replaced.
    """
    try:
        raw = (proc_root / str(pid) / "stat").read_bytes()
    except OSError:
        return None
    closing = raw.rfind(b")")
    if closing < 0:
        return None
    fields = raw[closing + 1 :].split()
    # fields[0] is the state, field 3 of proc(5); utime/stime are 14/15 and
    # starttime is 22, i.e. the same offset minus the three leading fields.
    ticks = _sysconf("SC_CLK_TCK")
    if len(fields) < 20 or not ticks:
        return None
    try:
        ppid = int(fields[1])
        cpu_ticks = int(fields[11]) + int(fields[12])
        starttime = int(fields[19])
    except ValueError:
        return None
    return ppid, starttime, cpu_ticks / ticks


def _rss_bytes(pid: int, proc_root: Path) -> int | None:
    """Resident set size of one pid in bytes, from ``/proc/<pid>/statm``."""
    try:
        fields = (proc_root / str(pid) / "statm").read_text(encoding="ascii").split()
    except OSError:
        return None
    page_size = _sysconf("SC_PAGE_SIZE")
    if page_size is None or len(fields) < 2:
        return None
    try:
        return int(fields[1]) * page_size
    except ValueError:
        return None


def _process_table(proc_root: Path) -> dict[int, tuple[int, int, float]] | None:
    """``{pid: (ppid, starttime, cpu_seconds)}``, or ``None`` without ``/proc``.

    ``None`` means the directory could not be listed at all — no ``/proc`` on
    this platform, or no permission to read it — which is what separates an
    unavailable source from an empty one.
    """
    try:
        entries = list(proc_root.iterdir())
    except OSError:
        return None
    table: dict[int, tuple[int, int, float]] = {}
    for entry in entries:
        if not entry.name.isdigit():
            continue
        stat = _process_stat(int(entry.name), proc_root)
        if stat is not None:
            table[int(entry.name)] = stat
    return table


def _tree_pids(root_pid: int, table: dict[int, tuple[int, int, float]]) -> set[int]:
    """The root pid plus every descendant reachable through ``ppid`` links."""
    children: dict[int, list[int]] = {}
    for pid, (ppid, _starttime, _cpu) in table.items():
        children.setdefault(ppid, []).append(pid)
    tree = {root_pid}
    pending = [root_pid]
    while pending:
        for child in children.get(pending.pop(), ()):
            if child not in tree:
                tree.add(child)
                pending.append(child)
    return tree


def gpu_vram(*, sysfs_root: str | Path = DRM_ROOT) -> dict[str, Any]:
    """Device-scope VRAM in use, from DRM sysfs counters when they exist.

    ``value`` is the sum over every card that publishes
    ``mem_info_vram_used``, in bytes, and ``cards`` counts them. The scope is the
    device rather than one process, so it is only meaningful as a stage
    observation on a machine where nothing else is using the GPU. Without any
    such counter the reading is ``not_available``; no GPU library is imported
    and no external tool is run to obtain it.
    """
    used_bytes = 0
    cards = 0
    for path in sorted(Path(sysfs_root).glob("card*/device/mem_info_vram_used")):
        value = _read_int(path)
        if value is None:
            continue
        used_bytes += value
        cards += 1
    if not cards:
        return _metric(None, NOT_AVAILABLE, unit="bytes", reason="no_drm_vram_counter")
    reading = _measured(used_bytes, unit="bytes", estimate=True)
    reading["scope"] = "device"
    reading["cards"] = cards
    return reading


def hosted_backend_memory() -> dict[str, Any]:
    """Backend memory of a hosted engine: always ``not_available``.

    The model runs on the provider's machine, so its peak memory cannot be
    observed from here. This is one explicit value rather than an omission,
    because the tempting substitute — the client's own RSS — measures the
    wrong machine.
    """
    return _hosted_miss("bytes")


def accelerator(runtime: str = ACCELERATOR_RUNTIME) -> dict[str, Any]:
    """The device a local engine's runtime would compute on, or an explicit miss.

    ``cuda`` when the runtime reports one, ``cpu`` when it reports none: both are
    answers from the runtime itself and carry ``source``. A runtime that is not
    installed, or that fails while being asked, yields ``not_available`` with the
    reason, because "no GPU library answered" is not the same fact as "no GPU".

    The runtime is imported here rather than looked up, so a broken install is
    reported instead of hidden; the caller decides whether to pay that import by
    asking only for engines whose stack already uses it.
    """
    try:
        module = importlib.import_module(runtime)
    except Exception:  # a missing or unloadable runtime is a miss, not a crash
        return _metric(None, NOT_AVAILABLE, unit="device", reason="no_torch_runtime")
    try:
        available = bool(module.cuda.is_available())
    except Exception:  # the driver answered with an error instead of a device
        return _metric(None, NOT_AVAILABLE, unit="device", reason="device_probe_failed")
    reading = _metric("cuda" if available else "cpu", MEASURED, unit="device")
    reading["source"] = runtime
    return reading


# --------------------------------------------------------------------------- #
# sampler
# --------------------------------------------------------------------------- #


class _ProcessTreeSampler:
    """Polls ``/proc`` for one root pid and keeps the tree's peak RSS and CPU."""

    def __init__(
        self,
        root_pid: int,
        *,
        interval_seconds: float,
        proc_root: str | Path,
        sysfs_root: str | Path,
    ) -> None:
        if isinstance(root_pid, bool) or not isinstance(root_pid, int) or root_pid <= 0:
            raise ValueError(f"pid must be a positive int, got {root_pid!r}")
        if not interval_seconds > 0:  # written this way so NaN is rejected too
            raise ValueError(f"interval_seconds must be > 0, got {interval_seconds!r}")
        self._root_pid = root_pid
        self._interval_seconds = float(interval_seconds)
        self._proc_root = Path(proc_root)
        self._sysfs_root = sysfs_root
        self._stop = threading.Event()
        self._lock = threading.Lock()
        self._thread: threading.Thread | None = None
        self._started_at = 0.0
        self._baseline: dict[int, tuple[int, float]] = {}
        self._cpu_seconds = 0.0
        self._rss_peak: int | None = None
        self._gpu_peak: int | None = None
        self._tree_peak = 0
        self._samples = 0
        self._last_sample_at: float | None = None
        self._proc_readable = False
        self._process_seen = False

    # -- lifecycle ---------------------------------------------------------- #

    def start(self) -> None:
        self._started_at = time.monotonic()
        self._sample()
        self._thread = threading.Thread(target=self._run, name="ocr-stage-resources", daemon=True)
        self._thread.start()

    def stop(self) -> dict[str, Any]:
        self._stop.set()
        if self._thread is not None:
            self._thread.join(timeout=_JOIN_TIMEOUT_SECONDS)
        self._sample()
        return self._observation()

    def _run(self) -> None:
        while not self._stop.wait(self._interval_seconds):
            self._sample()

    # -- sampling ----------------------------------------------------------- #

    def _sample(self) -> None:
        gpu = gpu_vram(sysfs_root=self._sysfs_root)
        table = _process_table(self._proc_root)
        with self._lock:
            if gpu["status"] == MEASURED:
                peak = self._gpu_peak
                self._gpu_peak = gpu["value"] if peak is None else max(peak, gpu["value"])
            if table is None:
                return
            self._proc_readable = True
            self._samples += 1
            tree = _tree_pids(self._root_pid, table)
            self._tree_peak = max(self._tree_peak, len(tree))
            rss_bytes: int | None = None
            baseline: dict[int, tuple[int, float]] = {}
            for pid in tree:
                entry = table.get(pid)
                if entry is None:
                    continue  # exited between the stat read and this lookup
                starttime, cpu_seconds = entry[1], entry[2]
                previous = self._baseline.get(pid)
                if previous is not None and previous[0] == starttime:
                    delta = cpu_seconds - previous[1]
                    if delta > 0:
                        self._cpu_seconds += delta
                baseline[pid] = (starttime, cpu_seconds)
                residency = _rss_bytes(pid, self._proc_root)
                if residency is not None:
                    rss_bytes = residency if rss_bytes is None else rss_bytes + residency
            # Keep the baseline bounded by the live tree; a pid that comes back
            # with the same starttime is rebased instead of double counted.
            self._baseline = baseline
            if self._root_pid in table:
                self._process_seen = True
            if rss_bytes is not None:
                peak = self._rss_peak
                self._rss_peak = rss_bytes if peak is None else max(peak, rss_bytes)
            self._last_sample_at = time.monotonic()

    # -- result ------------------------------------------------------------- #

    def _observation(self) -> dict[str, Any]:
        with self._lock:
            if not self._proc_readable:
                reason = "proc_unavailable"
                status = NOT_AVAILABLE
            elif not self._process_seen:
                reason = "no_process_observed"
                status = NOT_MEASURED
            else:
                reason = None
                status = MEASURED
            if status != MEASURED:
                cpu = _metric(None, status, unit="seconds", reason=reason)
                rss = _metric(None, status, unit="bytes", reason=reason)
            else:
                cpu = _measured(round(self._cpu_seconds, 6), unit="seconds", estimate=True)
                rss = (
                    _measured(self._rss_peak, unit="bytes", estimate=True)
                    if self._rss_peak is not None
                    else _metric(None, NOT_MEASURED, unit="bytes", reason="rss_unreadable")
                )
            if self._gpu_peak is None:
                gpu = _metric(None, NOT_AVAILABLE, unit="bytes", reason="no_drm_vram_counter")
            else:
                gpu = _measured(self._gpu_peak, unit="bytes", estimate=True)
                gpu["scope"] = "device"
            last_sample_at = self._last_sample_at
            sampling: dict[str, Any] = {
                "interval_seconds": self._interval_seconds,
                "samples": self._samples,
                "span_seconds": None
                if last_sample_at is None
                else round(last_sample_at - self._started_at, 6),
                "tree_peak": self._tree_peak,
            }
            wall_seconds = time.monotonic() - self._started_at
        return {
            "wall": _measured(round(wall_seconds, 6), unit="seconds"),
            "cpu": cpu,
            "rss_peak": rss,
            "gpu_vram": gpu,
            "sampling": sampling,
        }


# --------------------------------------------------------------------------- #
# context managers
# --------------------------------------------------------------------------- #


@contextmanager
def measure_process_tree(
    pid: int,
    *,
    interval_seconds: float = DEFAULT_INTERVAL_SECONDS,
    proc_root: str | Path = PROC_ROOT,
    sysfs_root: str | Path = DRM_ROOT,
) -> Iterator[dict[str, Any]]:
    """Measure one local stage by sampling the ``/proc`` tree rooted at ``pid``.

    ``pid`` is the stage's own process, typically a just-started subprocess::

        process = subprocess.Popen(argv)
        with measure_process_tree(process.pid) as resources:
            process.wait()
        report["extraction_resources"] = resources

    ``wall`` is the duration of the block, measured with a monotonic clock, so
    the caller decides where the stage starts by where the block starts. The
    yielded dict is finalised when the block ends, including when the body
    raises, so a stage that fails still reports what it used up to the failure.
    ``proc_root`` and ``sysfs_root`` are injectable so a caller can point at
    another mount and so the unavailable cases are testable.
    """
    sampler = _ProcessTreeSampler(
        pid,
        interval_seconds=interval_seconds,
        proc_root=proc_root,
        sysfs_root=sysfs_root,
    )
    observation: dict[str, Any] = {}
    sampler.start()
    try:
        yield observation
    finally:
        observation.update(sampler.stop())


@contextmanager
def measure_hosted_stage() -> Iterator[dict[str, Any]]:
    """Measure a hosted stage: wall time measured, backend resources unknown.

    The engine runs on the provider's machine. Wall time here is the stage's
    latency and is measured; backend RAM, backend CPU and the backend's GPU are
    not observable from the client and stay ``not_available``. The yielded dict
    is finalised when the block ends, including when the body raises.
    """
    started_at = time.monotonic()
    observation: dict[str, Any] = {}
    try:
        yield observation
    finally:
        observation.update(
            {
                "wall": _measured(round(time.monotonic() - started_at, 6), unit="seconds"),
                "cpu": _hosted_miss("seconds"),
                "rss_peak": hosted_backend_memory(),
                "gpu_vram": _hosted_miss("bytes"),
                "sampling": None,
            }
        )
