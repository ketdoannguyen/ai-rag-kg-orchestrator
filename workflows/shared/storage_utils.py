"""Storage utilities for MinIO operations."""

from __future__ import annotations

import asyncio
from pathlib import Path
from typing import TYPE_CHECKING

from loguru import logger

if TYPE_CHECKING:
    from minio_client import StorageClient  # type: ignore[import-not-found]

RETRY_DELAY_SECONDS = 2.0


async def ensure_bucket_with_retry(
    storage: StorageClient,
    bucket: str,
    max_attempts: int = 3,
    retry_delay: float = RETRY_DELAY_SECONDS,
) -> None:
    """
    Ensure bucket exists with retry logic.

    Args:
        storage: MinIO storage client
        bucket: Bucket name to ensure
        max_attempts: Maximum number of retry attempts (default: 3)
        retry_delay: Base delay in seconds between retries (default: 2.0)
    """
    for attempt in range(max_attempts):
        try:
            await storage.ensure_bucket(bucket)
            return
        except Exception as exc:
            logger.warning(f"Bucket ensure failed (attempt {attempt + 1}): {exc}")
            if attempt < max_attempts - 1:
                await asyncio.sleep(retry_delay * (attempt + 1))
    logger.warning("Proceeding without confirming bucket availability")


async def upload_file(
    storage: StorageClient,
    bucket: str,
    local_path: Path,
    object_name: str,
    content_type: str,
    metadata: dict[str, str] | None = None,
) -> str:
    """
    Upload a local file to MinIO.

    Args:
        storage: MinIO storage client
        bucket: Bucket name
        local_path: Path to local file
        object_name: MinIO object name/path
        content_type: MIME content type

    Returns:
        The object_name that was uploaded
    """
    local_path = local_path.expanduser().resolve()
    await storage.upload_file(
        bucket_name=bucket,
        object_name=object_name,
        file_path=local_path,
        content_type=content_type,
        metadata=metadata,
    )
    return object_name


async def upload_bytes(
    storage: StorageClient,
    bucket: str,
    data: bytes | str,
    object_name: str,
    content_type: str,
    metadata: dict[str, str] | None = None,
) -> str:
    """
    Upload bytes or string content to MinIO.

    Args:
        storage: MinIO storage client
        bucket: Bucket name
        data: Bytes or string content to upload
        object_name: MinIO object name/path
        content_type: MIME content type

    Returns:
        The object_name that was uploaded
    """
    if isinstance(data, str):
        data = data.encode("utf-8")
    await storage.upload_bytes(
        bucket_name=bucket,
        object_name=object_name,
        data=data,
        content_type=content_type,
        metadata=metadata,
    )
    return object_name

