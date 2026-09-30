"""Immutable, content-addressed storage for extraction artifacts.

An artifact is written once, under a name equal to the content digest of the
bytes being written. The store never overwrites or mutates an existing artifact:
writing the same bytes again is a no-op, and finding different bytes at an
address is an integrity error rather than a silent replacement. That is what
lets a replay trust that the artifact it reads is the artifact the earlier run
produced.

Writes go to a temporary file in the store directory and are published with
``os.link``, which fails instead of clobbering, so a reader never observes a
half-written artifact and two writers of the same payload cannot corrupt each
other.

The layout is ``<root>/<doc_id>/artifacts/<address>.json``. Both the doc id and
the address are checked against a slug / hex whitelist before they become path
segments, so no address or document name can climb out of the root.
"""

from __future__ import annotations

import hashlib
import os
import re
import tempfile
from pathlib import Path

from pydantic import ValidationError

from .models import ExtractionArtifact, canonical_payload

_ADDRESS_RE = re.compile(r"[0-9a-f]{64}")
_DOC_ID_RE = re.compile(r"[A-Za-z0-9][A-Za-z0-9._-]*")


class ArtifactStoreError(RuntimeError):
    """Base class for every store failure."""


class ArtifactNotFound(ArtifactStoreError):  # noqa: N818 -- callers catch this name
    """Raised when the requested address is not in the store."""


class ArtifactIntegrityError(ArtifactStoreError):
    """Raised when stored bytes and their content address disagree."""


class ArtifactStore:
    """A document-scoped directory of artifacts, addressed by content digest."""

    def __init__(self, root: str | Path) -> None:
        self.root = Path(root)

    def path_for(self, doc_id: str, address: str) -> Path:
        """Return the file an artifact at ``address`` lives in, reading nothing.

        Raises:
            ArtifactStoreError: if ``doc_id`` or ``address`` is not a name this
                store is willing to turn into a path.
        """
        return self.root / _doc_directory(doc_id) / "artifacts" / f"{_address(address)}.json"

    def put(self, artifact: ExtractionArtifact) -> str:
        """Store ``artifact`` once and return its content address.

        Re-storing identical bytes is a no-op, so a repeat run costs no new file.

        Raises:
            ArtifactIntegrityError: if different bytes already sit at the address.
        """
        payload = canonical_payload(artifact).encode("utf-8")
        address = hashlib.sha256(payload).hexdigest()
        _publish(self.path_for(artifact.doc_id, address), payload)
        return address

    def get(self, doc_id: str, address: str) -> ExtractionArtifact:
        """Read back the artifact stored at ``address``.

        Raises:
            ArtifactNotFound: if nothing is stored there.
            ArtifactIntegrityError: if the file is not a valid artifact, or if
                its bytes do not hash back to the address it was fetched by.
        """
        target = self.path_for(doc_id, address)
        if not target.is_file():
            raise ArtifactNotFound(f"{doc_id}: no artifact stored at {address}")
        raw = target.read_bytes()
        try:
            artifact = ExtractionArtifact.model_validate_json(raw)
        except ValidationError as exc:
            raise ArtifactIntegrityError(f"{target} is not a valid artifact: {exc}") from exc
        if hashlib.sha256(raw).hexdigest() != address:
            raise ArtifactIntegrityError(f"{target} hashes not to its address {address}")
        return artifact


def write_once(path: str | Path, payload: bytes) -> None:
    """Write ``payload`` at ``path``, creating parents, exactly once.

    Identical bytes already present are accepted as already written.

    Raises:
        ArtifactStoreError: if different bytes are already at ``path``.
    """
    _publish(Path(path), payload)


def _publish(target: Path, payload: bytes) -> None:
    """Publish ``payload`` at ``target`` without ever replacing different bytes."""
    target.parent.mkdir(parents=True, exist_ok=True)
    if target.exists():
        _refuse_to_replace(target, payload)
        return

    descriptor, temporary_name = tempfile.mkstemp(dir=target.parent, prefix=f".{target.name}.")
    temporary = Path(temporary_name)
    try:
        with os.fdopen(descriptor, "wb") as stream:
            stream.write(payload)
        try:
            os.link(temporary, target)
        except FileExistsError:
            # Another writer won the race; the winner's bytes must still be ours.
            _refuse_to_replace(target, payload)
    finally:
        temporary.unlink(missing_ok=True)


def _refuse_to_replace(target: Path, payload: bytes) -> None:
    """Accept identical bytes already written; refuse to replace anything else."""
    if target.read_bytes() != payload:
        raise ArtifactIntegrityError(f"{target}: refusing to overwrite different bytes")


def _address(address: str) -> str:
    """Return ``address`` as a sha256 hex digest, refusing anything else as a name."""
    if not _ADDRESS_RE.fullmatch(address):
        raise ArtifactStoreError(f"{address!r} is not a sha256 address")
    return address


def _doc_directory(doc_id: str) -> str:
    """Return ``doc_id`` as a directory name, refusing anything that could escape."""
    if not _DOC_ID_RE.fullmatch(doc_id):
        raise ArtifactStoreError(f"{doc_id!r} is not a usable doc_id")
    return doc_id
