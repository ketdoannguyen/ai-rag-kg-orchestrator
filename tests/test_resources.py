"""Tests for the OCR research resource sampler.

These start real, short-lived subprocesses — no engine, no PDF, no network — so
a synthetic parent/child tree is observed through the real ``/proc``. The
unavailable cases (missing ``/proc``, no DRM counters, hosted backend) are
driven through the injectable roots, and are asserted to be typed misses rather
than fabricated numbers.
"""

from __future__ import annotations

import importlib.metadata
import os
import subprocess
import sys
import time
import types
from pathlib import Path
from typing import Any

import pytest

from research.ocr_pipeline.engines import EngineNotAvailableError
from research.ocr_pipeline.engines.worker import (
    PROTOCOL_VERSION,
    WARMTH_COLD,
    WARMTH_WARM,
    IsolatedWorkerError,
    isolated_extract,
    warmth,
)
from research.ocr_pipeline.metrics import MEASURED, NOT_MEASURED
from research.ocr_pipeline.resources import (
    NOT_AVAILABLE,
    accelerator,
    gpu_vram,
    hosted_backend_memory,
    measure_hosted_stage,
    measure_process_tree,
)


def _docling_installed() -> bool:
    """Whether this environment has the one engine the missing-engine case needs absent."""
    try:
        importlib.metadata.version("docling")
    except importlib.metadata.PackageNotFoundError:
        return False
    return True


MIB = 1024 * 1024

PROC_AVAILABLE = Path("/proc/self/stat").exists()

requires_proc = pytest.mark.skipif(not PROC_AVAILABLE, reason="needs a readable Linux /proc")

CHILD_SCRIPT = """
import sys
import time

payload = bytearray(160 * 1024 * 1024)  # touched pages, so the RSS is real
deadline = time.monotonic() + 0.5
while time.monotonic() < deadline:
    pass
sys.exit(0 if payload[0] == 0 else 1)
"""

PARENT_SCRIPT = """
import subprocess
import sys

child = subprocess.Popen([sys.executable, sys.argv[1]])
sys.exit(child.wait())
"""


def _start_synthetic_tree(tmp_path: Path) -> subprocess.Popen[bytes]:
    """A parent that waits on a child which allocates memory and burns CPU."""
    child = tmp_path / "child.py"
    child.write_text(CHILD_SCRIPT, encoding="utf-8")
    parent = tmp_path / "parent.py"
    parent.write_text(PARENT_SCRIPT, encoding="utf-8")
    return subprocess.Popen([sys.executable, str(parent), str(child)])


def _write_vram(sysfs_root: Path, card: str, used_bytes: int) -> None:
    device = sysfs_root / card / "device"
    device.mkdir(parents=True)
    (device / "mem_info_vram_used").write_text(f"{used_bytes}\n", encoding="ascii")


@requires_proc
def test_process_tree_peak_includes_the_child(tmp_path: Path) -> None:
    process = _start_synthetic_tree(tmp_path)
    with measure_process_tree(
        process.pid,
        interval_seconds=0.02,
        sysfs_root=tmp_path / "no-drm",
    ) as resources:
        assert process.wait(timeout=60) == 0

    assert resources["rss_peak"]["status"] == MEASURED
    assert resources["rss_peak"]["estimate"] is True
    # The child touches 160 MiB; a bare interpreter is an order below that.
    assert resources["rss_peak"]["value"] > 100 * MIB
    # The parent only waits, so CPU this high is the child's burn.
    assert resources["cpu"]["status"] == MEASURED
    assert resources["cpu"]["value"] >= 0.2
    assert resources["wall"]["status"] == MEASURED
    assert resources["wall"]["value"] > 0
    assert resources["sampling"]["tree_peak"] >= 2
    assert resources["sampling"]["interval_seconds"] == 0.02
    assert resources["sampling"]["samples"] > 0


@requires_proc
def test_failed_stage_keeps_the_resources_it_used(tmp_path: Path) -> None:
    process = subprocess.Popen([sys.executable, "-c", "import time; time.sleep(0.4)"])
    resources: dict[str, Any] = {}
    with pytest.raises(RuntimeError, match="engine failed"):
        with measure_process_tree(
            process.pid,
            interval_seconds=0.02,
            sysfs_root=tmp_path / "no-drm",
        ) as resources:
            process.wait(timeout=60)
            raise RuntimeError("engine failed")

    assert resources["wall"]["status"] == MEASURED
    assert resources["rss_peak"]["status"] == MEASURED
    assert resources["rss_peak"]["value"] > 0
    assert resources["sampling"]["samples"] > 0


@requires_proc
def test_process_gone_before_the_first_sample_is_not_measured(tmp_path: Path) -> None:
    process = subprocess.Popen([sys.executable, "-c", "pass"])
    process.wait(timeout=60)
    if Path(f"/proc/{process.pid}").exists():
        pytest.skip("the reaped pid was reused by another process")

    with measure_process_tree(
        process.pid,
        interval_seconds=0.01,
        sysfs_root=tmp_path / "no-drm",
    ) as resources:
        time.sleep(0.05)

    assert resources["rss_peak"]["status"] == NOT_MEASURED
    assert resources["rss_peak"]["reason"] == "no_process_observed"
    assert resources["rss_peak"]["value"] is None
    assert resources["cpu"]["status"] == NOT_MEASURED
    assert resources["wall"]["status"] == MEASURED


def test_missing_proc_reports_not_available(tmp_path: Path) -> None:
    with measure_process_tree(
        os.getpid(),
        interval_seconds=0.01,
        proc_root=tmp_path / "no-proc",
        sysfs_root=tmp_path / "no-drm",
    ) as resources:
        time.sleep(0.05)

    assert resources["rss_peak"]["status"] == NOT_AVAILABLE
    assert resources["rss_peak"]["reason"] == "proc_unavailable"
    assert resources["rss_peak"]["value"] is None
    assert resources["cpu"]["status"] == NOT_AVAILABLE
    assert resources["gpu_vram"]["status"] == NOT_AVAILABLE
    assert resources["sampling"]["samples"] == 0
    # Wall time never depends on a sampler.
    assert resources["wall"]["status"] == MEASURED
    assert resources["wall"]["value"] >= 0.05


def test_hosted_stage_never_reports_client_memory() -> None:
    with measure_hosted_stage() as resources:
        time.sleep(0.02)

    assert resources["wall"]["status"] == MEASURED
    assert resources["wall"]["value"] >= 0.02
    assert resources["rss_peak"] == hosted_backend_memory()
    assert resources["rss_peak"]["status"] == NOT_AVAILABLE
    assert resources["rss_peak"]["reason"] == "hosted_backend"
    assert resources["rss_peak"]["value"] is None
    assert resources["cpu"]["status"] == NOT_AVAILABLE
    assert resources["gpu_vram"]["status"] == NOT_AVAILABLE
    assert resources["sampling"] is None


def test_gpu_vram_is_measured_when_a_driver_exposes_it(tmp_path: Path) -> None:
    _write_vram(tmp_path, "card0", 5 * MIB)
    _write_vram(tmp_path, "card1", 3 * MIB)
    (tmp_path / "card2" / "device").mkdir(parents=True)  # a card without counters

    reading = gpu_vram(sysfs_root=tmp_path)

    assert reading["status"] == MEASURED
    assert reading["value"] == 8 * MIB
    assert reading["unit"] == "bytes"
    assert reading["scope"] == "device"
    assert reading["cards"] == 2
    assert reading["estimate"] is True


def test_gpu_vram_is_not_available_without_counters(tmp_path: Path) -> None:
    (tmp_path / "card0").mkdir()

    reading = gpu_vram(sysfs_root=tmp_path)

    assert reading["status"] == NOT_AVAILABLE
    assert reading["reason"] == "no_drm_vram_counter"
    assert reading["value"] is None


def test_measurement_writes_nothing(capsys: pytest.CaptureFixture[str], tmp_path: Path) -> None:
    with measure_process_tree(
        os.getpid(),
        interval_seconds=0.01,
        sysfs_root=tmp_path / "no-drm",
    ):
        time.sleep(0.03)

    captured = capsys.readouterr()
    assert captured.out == ""
    assert captured.err == ""


def test_invalid_pid_and_interval_are_rejected() -> None:
    with pytest.raises(ValueError, match="pid"):
        with measure_process_tree(0):
            pass
    with pytest.raises(ValueError, match="interval_seconds"):
        with measure_process_tree(os.getpid(), interval_seconds=0):
            pass


def _fake_torch(is_available: Any) -> Any:
    """Stand-in for the torch module, with the one call the probe makes."""
    module = types.ModuleType("torch")
    module.cuda = type("Cuda", (), {"is_available": staticmethod(is_available)})()
    return module


def test_accelerator_names_the_device_a_torch_runtime_offers(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setitem(sys.modules, "torch", _fake_torch(lambda: True))
    reading = accelerator()
    assert reading["status"] == MEASURED
    assert reading["value"] == "cuda"
    assert reading["unit"] == "device"
    assert reading["source"] == "torch"

    monkeypatch.setitem(sys.modules, "torch", _fake_torch(lambda: False))
    assert accelerator()["value"] == "cpu"


def test_accelerator_is_not_available_without_a_torch_runtime(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setitem(sys.modules, "torch", None)  # makes ``import torch`` fail
    reading = accelerator()
    assert reading["status"] == NOT_AVAILABLE
    assert reading["reason"] == "no_torch_runtime"
    assert reading["value"] is None


def test_accelerator_is_not_available_when_the_device_probe_fails(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    def boom() -> bool:
        raise RuntimeError("cuda driver exploded")

    monkeypatch.setitem(sys.modules, "torch", _fake_torch(boom))
    reading = accelerator()
    assert reading["status"] == NOT_AVAILABLE
    assert reading["reason"] == "device_probe_failed"
    assert reading["value"] is None


# --- isolated local engine workers ----------------------------------------------
#
# These run a real engine in a real worker process: the text-layer engine needs no
# model and no network, so the whole path — request, subprocess, result file,
# sampler — is exercised offline. The failure paths use that engine on a file that
# is not there, and a timeout.


def one_page_pdf(text: str) -> bytes:
    """A one-page PDF whose text layer holds ``text``; see the corpus tests."""
    stream = f"BT /F1 24 Tf 72 700 Td ({text}) Tj ET".encode("ascii")
    objects = [
        b"<< /Type /Catalog /Pages 2 0 R >>",
        b"<< /Type /Pages /Kids [3 0 R] /Count 1 >>",
        b"<< /Type /Page /Parent 2 0 R /MediaBox [0 0 612 792] "
        b"/Resources << /Font << /F1 5 0 R >> >> /Contents 4 0 R >>",
        b"<< /Length %d >>\nstream\n" % len(stream) + stream + b"\nendstream",
        b"<< /Type /Font /Subtype /Type1 /BaseFont /Helvetica >>",
    ]
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
    return document + (
        f"trailer\n<< /Size {len(objects) + 1} /Root 1 0 R >>\nstartxref\n{xref_offset}\n%%EOF\n"
    ).encode("ascii")


def _own_rss() -> int:
    """This process's resident set, for the assertion that the worker's is its own."""
    fields = Path("/proc/self/statm").read_text(encoding="ascii").split()
    return int(fields[1]) * os.sysconf("SC_PAGE_SIZE")


def _worker_temp_dirs(root: Path) -> list[Path]:
    """Temp directories an isolated run left behind, if any."""
    return list(root.glob("ocr-worker-*"))


@pytest.fixture
def page_path(tmp_path: Path) -> Path:
    """A one-page PDF to run through a worker: an engine with no weights."""
    directory = tmp_path / "inputs"
    directory.mkdir()
    path = directory / "page.pdf"
    path.write_bytes(one_page_pdf("Trang mot"))
    return path


@pytest.mark.skipif(not PROC_AVAILABLE, reason="needs a readable Linux /proc")
def test_an_isolated_run_measures_the_worker_and_never_the_caller(
    page_path: Path, tmp_path: Path
) -> None:
    """The finding: the runner's own heap (with earlier turns' models) is not an engine peak."""
    buffer = bytearray(192 * MIB)
    for offset in range(0, len(buffer), 4096):
        buffer[offset] = 1  # touched, so the pages are resident
    caller_rss = _own_rss()

    run = isolated_extract("native", page_path, doc_id="vat-li-12", temp_root=tmp_path)

    assert run.worker["pid"] != os.getpid(), "the run must happen in its own process"
    assert run.resources["sampling"]["tree_peak"] == 1, "the measured tree is the worker alone"
    assert run.resources["rss_peak"]["status"] == MEASURED
    assert run.resources["rss_peak"]["estimate"] is True
    assert caller_rss > run.resources["rss_peak"]["value"] + 128 * MIB
    assert run.resources["sampling"]["interval_seconds"] > 0
    assert run.resources["wall"]["status"] == MEASURED
    assert buffer[0] == 1  # the caller's heap is still the caller's


def test_a_worker_reports_its_own_device_version_and_load_time(
    page_path: Path, tmp_path: Path
) -> None:
    run = isolated_extract("native", page_path, doc_id="d", temp_root=tmp_path)

    assert run.artifact.engine_id == "native"
    assert run.artifact.physical_pages == (0,)
    assert run.artifact.blocks_by_kind == {"p": 1}
    worker = run.worker
    assert worker["protocol"] == PROTOCOL_VERSION
    assert worker["engine_id"] == "native"
    # The version comes from the worker's own environment, which is the point of the worker.
    assert worker["engine_version"] == importlib.metadata.version("pypdfium2")
    assert worker["python"]
    # The version alone would not tell two same-version environments apart, and an engine
    # kept in its own environment is exactly that case.
    assert worker["executable"] == sys.executable
    assert worker["load_scope"] == "none", "the text layer has no converter to build"
    assert worker["load_seconds"] is None
    assert worker["device"]["status"] == NOT_AVAILABLE
    assert worker["device"]["reason"] == "engine_has_no_accelerator"
    assert worker["requests_served"] == 1


def test_warmth_is_cold_without_evidence_that_the_model_process_was_reused(
    page_path: Path, tmp_path: Path
) -> None:
    """Repeat 2 of a fresh worker process is not warm: only its page cache could be."""
    first = isolated_extract("native", page_path, doc_id="d", temp_root=tmp_path)
    second = isolated_extract("native", page_path, doc_id="d", temp_root=tmp_path)

    assert first.warmth == second.warmth == WARMTH_COLD
    assert second.worker["pid"] != first.worker["pid"]
    assert second.worker["model_process_reused"] is False
    # Warmth the OS could provide is a different scope, and it is not observable here.
    assert second.worker["cache"] == {
        "scope": "os_page_cache",
        "status": "unknown",
        "reason": "not_observed",
    }
    # Evidence of reuse is the only thing that makes a run warm.
    assert warmth({"model_process_reused": True}) == WARMTH_WARM
    assert warmth({"model_process_reused": False}) == WARMTH_COLD
    assert warmth({}) == WARMTH_COLD


@pytest.mark.skipif(not PROC_AVAILABLE, reason="needs a readable Linux /proc")
def test_a_failed_worker_keeps_its_resources_and_leaves_nothing_behind(tmp_path: Path) -> None:
    with pytest.raises(IsolatedWorkerError) as failure:
        isolated_extract(
            "native",
            tmp_path / "absent.pdf",
            doc_id="d",
            temp_root=tmp_path,
        )

    error = failure.value
    # The resources the worker used before it failed are kept, not discarded.
    assert error.resources["rss_peak"]["status"] == MEASURED
    assert error.resources["rss_peak"]["value"] > 0
    assert error.resources["wall"]["status"] == MEASURED
    assert error.resources["sampling"]["samples"] > 0
    assert error.worker["pid"] > 0
    assert "pypdfium2" not in str(error), "an engine error body is not echoed"
    # Nothing is left running and nothing is left on disk.
    assert not Path(f"/proc/{error.worker['pid']}").exists()
    assert _worker_temp_dirs(tmp_path) == []


@pytest.mark.skipif(not PROC_AVAILABLE, reason="needs a readable Linux /proc")
def test_a_timed_out_worker_is_killed_reaped_and_its_resources_kept(
    page_path: Path, tmp_path: Path
) -> None:
    with pytest.raises(IsolatedWorkerError, match="timed out") as failure:
        isolated_extract(
            "native",
            page_path,
            doc_id="d",
            timeout_seconds=0.05,
            temp_root=tmp_path,
        )

    error = failure.value
    assert error.resources["rss_peak"]["status"] == MEASURED
    assert error.resources["sampling"]["samples"] >= 1
    pid = error.worker["pid"]
    assert pid > 0 and pid != os.getpid()
    for _ in range(50):  # SIGKILL is immediate, but reaping is the parent's
        if not Path(f"/proc/{pid}").exists():
            break
        time.sleep(0.02)
    assert not Path(f"/proc/{pid}").exists(), "a timed-out worker must not be left running"
    assert _worker_temp_dirs(tmp_path) == []


def test_a_worker_reports_a_missing_engine_as_not_available(tmp_path: Path) -> None:
    if _docling_installed():
        pytest.skip("docling is installed here, so the missing-engine path cannot be driven")
    with pytest.raises(EngineNotAvailableError, match="docling") as failure:
        isolated_extract("docling", tmp_path / "absent.pdf", doc_id="d", temp_root=tmp_path)

    assert failure.value.status == "not_available"
    assert failure.value.resources["wall"]["status"] == MEASURED


def test_an_engine_that_cannot_be_isolated_is_refused(page_path: Path, tmp_path: Path) -> None:
    with pytest.raises(ValueError, match="vision"):
        isolated_extract("vision", page_path, doc_id="d", temp_root=tmp_path)
    with pytest.raises(ValueError, match="unknown engine_id"):
        isolated_extract("tesseract", page_path, doc_id="d", temp_root=tmp_path)
