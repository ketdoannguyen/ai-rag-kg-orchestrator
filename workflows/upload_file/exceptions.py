"""Custom exceptions for the upload_file workflow."""

from __future__ import annotations


class FileValidationError(ValueError):
    """Raised when the incoming file fails validation checks."""


class DuplicateFileError(RuntimeError):
    """Raised when an uploaded file already exists and duplicates are not allowed."""


class StorageUploadError(RuntimeError):
    """Raised when uploading to MinIO fails."""


class DatabaseError(RuntimeError):
    """Raised when database operations fail."""


__all__ = [
    "DatabaseError",
    "DuplicateFileError",
    "FileValidationError",
    "StorageUploadError",
]
