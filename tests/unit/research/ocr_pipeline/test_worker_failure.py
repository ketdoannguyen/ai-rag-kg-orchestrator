"""What a killed worker still reports: the resources it spent, and the pid it ran as.

The parent measures a worker's tree from before the engine loads until the process
is gone, and for a run that never wrote a result that measurement is the only
account of it. Losing it at the moment of failure makes an OOM kill
indistinguishable from a crash that cost nothing, which is the comparison a study
needs: a run that died is weighed against a run that answered by what it spent.

The worker here is a real one, started the way the parent starts every worker --
``python -m research.ocr_pipeline.engines.worker`` -- and killed by the signal the
OOM killer sends. Only the engine's dependency is a stand-in: a ``pypdfium2`` that
dies the moment the adapter imports it, holding a block of memory, so the peak the
error carries can only have come from the child.
"""

from __future__ import annotations

import os
from pathlib import Path

import pytest

from research.ocr_pipeline.engines.worker import IsolatedWorkerError, isolated_extract

_LIBRARY_DIR = "library"
"""Where the stand-in goes: a directory of the caller's import path, so the worker's
``import pypdfium2`` resolves to it before the engine's own dependency and before the
PDF is ever opened."""

_BLOCK_BYTES = 32 * 1024 * 1024
"""How much the stand-in holds while it waits. The peak the parent reports has to be at
least this, or the resources came from somewhere other than the child."""

_LINGER_SECONDS = 1.0
"""How long the stand-in holds that block before killing itself: long enough that the
sampler observes it, since a worker the parent never saw is a different failure."""

_DYING_PACKAGE = '''\
"""A pypdfium2 stand-in that dies the way the OOM killer kills: SIGKILL, no result.

The native adapter imports this before it reads a byte of the PDF, so the worker
stops exactly where an engine's own load would have been stopped -- the shape of
the failure this guards. It writes nothing to stderr either, so the exit code and
the parent's measurement are the whole account of the run.
"""

import os
import signal
import time

block = bytearray(%(block_bytes)d)
for page in range(0, len(block), 4096):  # touch every page: RSS counts what is resident
    block[page] = 1
time.sleep(%(linger_seconds)r)
os.kill(os.getpid(), signal.SIGKILL)
'''


def _dying_library(root: Path) -> Path:
    """A package root holding the stand-in, for the worker's import to resolve to."""
    library = root / _LIBRARY_DIR
    library.mkdir(parents=True)
    (library / "pypdfium2.py").write_text(
        _DYING_PACKAGE % {"block_bytes": _BLOCK_BYTES, "linger_seconds": _LINGER_SECONDS},
        encoding="utf-8",
    )
    return library


def test_a_worker_killed_without_a_result_keeps_what_it_spent(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """The measurement outlives the failure: the error carries it, not nothing.

    A worker killed by the kernel reaches its parent as an exit code and no result,
    and this is the one path where the parent has nothing of the child's own to
    report. What it measured while the child was alive must therefore travel with
    the error -- and the private directory must still be gone afterwards.
    """
    monkeypatch.syspath_prepend(str(_dying_library(tmp_path)))
    document = tmp_path / "page.pdf"
    document.write_bytes(b"%PDF-1.4\n% the stand-in dies before this is read\n")

    with pytest.raises(IsolatedWorkerError) as error:
        isolated_extract(
            "native",
            document,
            doc_id="d",
            timeout_seconds=30.0,
            interval_seconds=0.02,
            temp_root=tmp_path,
        )

    assert "exited with -9" in str(error.value), "the worker was killed, not reported"
    resources = error.value.resources
    assert resources, "the measurement is not thrown away by the failure"
    assert resources["rss_peak"]["status"] == "measured"
    assert resources["rss_peak"]["value"] >= _BLOCK_BYTES, "the peak is the block the worker held"
    assert resources["sampling"]["samples"] > 0
    assert error.value.worker["pid"] != os.getpid(), "the pid is the worker's, seen by the parent"
    assert list(tmp_path.glob("ocr-worker-*")) == [], "a failed run leaves no directory behind"
