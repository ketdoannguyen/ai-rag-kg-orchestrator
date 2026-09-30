"""Whose packages the engine interpreter's child resolves: its own, with the caller's behind it.

An engine named in ``OCR_RESEARCH_ENGINE_PYTHON`` runs in an interpreter that is not the
caller's, which is the whole point of the override: docling's pins and the app's cannot
coexist, so the engine is given an interpreter of its own. That only works if the child's
own site-packages win. The caller's path still has to reach the child -- the code that
started the run lives on it, and so does every package the worker imports -- but as a
fallback for what the engine's interpreter does not have.

The availability probe travels the same way: whether an engine can run is a question
about the interpreter it would run in, so the probe is asked there -- still reading only
distribution metadata, importing nothing -- and an answer from another interpreter always
says which one was asked.

These tests build that arrangement for real. The engine interpreter is a ``bin/python``
beside a ``pyvenv.cfg``, which is what a venv is: a real interpreter that resolves its own
site-packages, exactly as the docling venv does. Both sides hold the one package the
``native`` engine imports (``pypdfium2``), each with a different answer in it, so the
artifact the worker wrote says which copy the child imported -- child or caller -- and no
assertion here has to read a path list.
"""

from __future__ import annotations

import os
import subprocess
import sys
from pathlib import Path

import pytest

from research.ocr_pipeline.engines import load_engine
from research.ocr_pipeline.engines.docling import DISTRIBUTION
from research.ocr_pipeline.engines.worker import (
    ENGINE_PYTHON_ENV,
    IsolatedRun,
    engine_probe,
    engine_python,
    isolated_extract,
)

PROBE = "pypdfium2"
"""The package the probe engine imports, so the artifact shows which copy was imported."""

_PROBE_SOURCE = '''\
"""A stand-in for pypdfium2: only the surface the native engine reads."""

TEXT = "%(mark)s text layer"


class _TextPage:
    def get_text_range(self):
        return TEXT

    def close(self):
        pass


class _Page:
    def get_textpage(self):
        return _TextPage()

    def close(self):
        pass


class PdfDocument:
    def __init__(self, path):
        self.path = path

    def __len__(self):
        return 1

    def __getitem__(self, index):
        return _Page()

    def close(self):
        pass
'''


def _write_probe_package(site_packages: Path, *, mark: str) -> None:
    """Put one copy of the probe package into ``site_packages``, marked with ``mark``."""
    package = site_packages / PROBE
    package.mkdir(parents=True)
    (package / "__init__.py").write_text(_PROBE_SOURCE % {"mark": mark}, encoding="utf-8")


def _write_distribution(interpreter: Path, *, name: str, version: str) -> None:
    """Put one distribution's metadata where that interpreter resolves it.

    Metadata is the whole of what an availability probe reads -- it imports
    nothing -- so a ``*.dist-info`` with no package under it is enough to make
    that interpreter the one an engine is installed in.
    """
    home = interpreter.parent.parent
    version_info = sys.version_info
    site_packages = (
        home / "lib" / f"python{version_info.major}.{version_info.minor}" / "site-packages"
    )
    info = site_packages / f"{name}-{version}.dist-info"
    info.mkdir(parents=True)
    (info / "METADATA").write_text(
        f"Metadata-Version: 2.1\nName: {name}\nVersion: {version}\n", encoding="utf-8"
    )


def _engine_interpreter(root: Path, *, mark: str | None) -> Path:
    """An engine interpreter of its own, with ``mark``'s probe package in its site-packages.

    A venv is a ``bin/python`` and a ``pyvenv.cfg`` beside it, and CPython resolves both:
    an interpreter started that way reports this directory as its prefix and reads the
    site-packages under it. Built here rather than with ``python -m venv`` so this test
    needs nothing the interpreter does not already do by itself; with ``mark`` as
    ``None`` its site-packages stays empty, and the probe package exists only on the
    caller's path.
    """
    (root / "bin").mkdir(parents=True)
    interpreter = root / "bin" / "python"
    interpreter.symlink_to(sys.executable)
    version = sys.version_info
    (root / "pyvenv.cfg").write_text(
        "\n".join(
            [
                f"home = {Path(os.path.realpath(sys.executable)).parent}",
                "include-system-site-packages = false",
                f"version = {version.major}.{version.minor}.{version.micro}",
                "",
            ]
        ),
        encoding="utf-8",
    )
    site_packages = root / "lib" / f"python{version.major}.{version.minor}" / "site-packages"
    site_packages.mkdir(parents=True)
    if mark is not None:
        _write_probe_package(site_packages, mark=mark)
    started = subprocess.run(
        [str(interpreter), "-c", "import sys; print(sys.prefix)"],
        capture_output=True,
        text=True,
        check=True,
    )
    assert Path(started.stdout.strip()).resolve() == root.resolve(), (
        "the premise: an interpreter started this way has a prefix of its own"
    )
    return interpreter


def _document(root: Path) -> Path:
    """A file for the probe engine to read: the stand-in parses no bytes, the hash does."""
    path = root / "page.pdf"
    path.write_bytes(b"%PDF-1.4\n% the probe engine reads no bytes of this\n")
    return path


def _paragraphs(run: IsolatedRun) -> list[str]:
    """Every paragraph the worker's artifact holds, in page order."""
    return [block.text for page in run.artifact.pages for block in page.blocks]


def test_a_child_interpreter_answers_for_its_own_package_before_the_callers(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """The engine's own copy wins: the caller's is behind it, not in front of it."""
    interpreter = _engine_interpreter(tmp_path / "engine", mark="child")
    caller = tmp_path / "caller"
    _write_probe_package(caller, mark="caller")
    monkeypatch.syspath_prepend(str(caller))
    monkeypatch.setenv(ENGINE_PYTHON_ENV, f"native={interpreter}")
    assert engine_python("native") == str(interpreter)

    run = isolated_extract("native", _document(tmp_path), doc_id="d", temp_root=tmp_path)

    assert run.worker["pid"] != os.getpid(), "the engine ran in the interpreter it was given"
    assert _paragraphs(run) == ["child text layer"]


def test_a_probe_reads_the_interpreter_the_engine_runs_in(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """The engine's own interpreter answers, and the answer says which one.

    The engine is installed there and not here, and a dist-info is all a
    metadata-only probe reads. Asked in this interpreter the answer would be
    about the wrong environment; asked in that one it is the version that
    interpreter holds.
    """
    interpreter = _engine_interpreter(tmp_path / "engine", mark=None)
    _write_distribution(interpreter, name=DISTRIBUTION, version="9.9.9")
    monkeypatch.setenv(ENGINE_PYTHON_ENV, f"docling={interpreter}")
    here = load_engine("docling")
    assert here.probe().available is False, "the premise: docling is not installed here"

    probe = engine_probe("docling", here)

    assert probe.engine_id == "docling"
    assert probe.available is True
    assert probe.version == "9.9.9"
    assert probe.detail.endswith(f"(asked {interpreter})")


def test_a_probe_that_finds_no_engine_says_which_interpreter_lacks_it(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A missing engine is reported as missing there, and the answer names where.

    The engine interpreter holds nothing, so the answer is that docling is
    missing -- and because the answer came from another interpreter it carries
    which one, which is the difference a reader needs to tell the two apart.
    """
    interpreter = _engine_interpreter(tmp_path / "engine", mark=None)
    monkeypatch.setenv(ENGINE_PYTHON_ENV, f"docling={interpreter}")
    here = load_engine("docling")
    assert here.probe().available is False, "the premise: docling is not installed here"

    probe = engine_probe("docling", here)

    assert probe.available is False
    assert probe.version is None
    assert "is not installed" in probe.detail
    assert probe.detail.endswith(f"(asked {interpreter})")


def test_a_probe_of_an_interpreter_that_cannot_start_fails_closed(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A probe that cannot be asked is a probe that says no, with the reason.

    The override names a file that is not a program. A probe must never raise
    into a plan, and must never claim an engine it could not ask about.
    """
    interpreter = tmp_path / "not-an-interpreter"
    interpreter.write_text("not a program\n", encoding="utf-8")
    monkeypatch.setenv(ENGINE_PYTHON_ENV, f"docling={interpreter}")

    probe = engine_probe("docling", load_engine("docling"))

    assert probe.available is False
    assert probe.version is None
    assert "cannot be started" in probe.detail


def test_a_probe_of_an_interpreter_that_never_answers_gives_up(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """An interpreter that hangs must not hang a plan: the probe gives up and says so."""
    interpreter = tmp_path / "engine-python"
    interpreter.write_text("#!/bin/sh\nsleep 30\n", encoding="utf-8")
    interpreter.chmod(0o755)
    monkeypatch.setenv(ENGINE_PYTHON_ENV, f"docling={interpreter}")

    probe = engine_probe("docling", load_engine("docling"), timeout_seconds=0.5)

    assert probe.available is False
    assert probe.version is None
    assert "did not answer within 0.5s" in probe.detail


def test_the_callers_path_answers_what_the_engine_interpreter_does_not_have(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """The other half of the rule: the path that started the run is still read.

    The engine interpreter's site-packages is empty here, so the probe package can only
    come from the caller's path -- the same path the worker module and everything it
    imports comes from, which is why this run succeeds at all. A fix that dropped the
    caller's path instead of demoting it would fail this test.
    """
    interpreter = _engine_interpreter(tmp_path / "engine", mark=None)
    caller = tmp_path / "caller"
    _write_probe_package(caller, mark="caller")
    monkeypatch.syspath_prepend(str(caller))
    monkeypatch.setenv(ENGINE_PYTHON_ENV, f"native={interpreter}")
    assert engine_python("native") == str(interpreter)

    run = isolated_extract("native", _document(tmp_path), doc_id="d", temp_root=tmp_path)

    assert run.worker["pid"] != os.getpid()
    assert _paragraphs(run) == ["caller text layer"]
