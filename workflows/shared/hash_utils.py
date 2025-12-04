"""Hash utilities shared across workflows."""

from __future__ import annotations

import hashlib


def calculate_md5(data: bytes) -> str:
    """Return the hexadecimal MD5 digest for the provided bytes."""
    digest = hashlib.md5()
    digest.update(data)
    return digest.hexdigest()


__all__ = ["calculate_md5"]

