from __future__ import annotations

import asyncio
import io
from datetime import timedelta
from pathlib import Path
from typing import Callable, Sequence, TypeVar

from loguru import logger
from minio import Minio
from minio.deleteobjects import DeleteObject

T = TypeVar("T")


class StorageClient:
    """Async-friendly helper around MinIO for uploads, downloads, and housekeeping."""

    def __init__(
        self,
        endpoint: str,
        access_key: str,
        secret_key: str,
        *,
        region: str | None = None,
        secure: bool | None = None,
    ) -> None:
        sanitized_endpoint = endpoint.replace("http://", "").replace("https://", "")
        secure = secure if secure is not None else endpoint.startswith("https")
        self._client = Minio(
            sanitized_endpoint,
            access_key=access_key,
            secret_key=secret_key,
            secure=secure,
            region=region,
        )
        self._log = logger.bind(component="minio_client", endpoint=sanitized_endpoint)

    async def _run(self, func: Callable[..., T], *args, **kwargs) -> T:
        return await asyncio.to_thread(func, *args, **kwargs)

    async def ensure_bucket(self, bucket_name: str) -> None:
        if await self.bucket_exists(bucket_name):
            self._log.debug("Bucket '{}' already exists", bucket_name)
            return

        await self._run(self._client.make_bucket, bucket_name)
        self._log.info("Created bucket '{}'", bucket_name)

    async def upload_file(
        self,
        bucket_name: str,
        object_name: str,
        file_path: Path | str,
        *,
        content_type: str | None = None,
        metadata: dict[str, str] | None = None,
    ) -> str:
        path = Path(file_path).expanduser().resolve()
        await self._run(
            self._client.fput_object,
            bucket_name=bucket_name,
            object_name=object_name,
            file_path=str(path),
            content_type=content_type,
            metadata=metadata,
        )
        self._log.info(
            "Uploaded file '{}' from '{}' to bucket '{}'",
            object_name,
            path,
            bucket_name,
        )
        return object_name

    async def upload_bytes(
        self,
        bucket_name: str,
        object_name: str,
        data: bytes | str,
        *,
        content_type: str | None = None,
        metadata: dict[str, str] | None = None,
    ) -> str:
        payload = data.encode("utf-8") if isinstance(data, str) else data
        buffer = io.BytesIO(payload)
        await self._run(
            self._client.put_object,
            bucket_name=bucket_name,
            object_name=object_name,
            data=buffer,
            length=len(payload),
            content_type=content_type,
            metadata=metadata,
        )
        self._log.info(
            "Uploaded bytes object '{}' ({} bytes) to bucket '{}'",
            object_name,
            len(payload),
            bucket_name,
        )
        return object_name

    async def download_file(
        self,
        bucket_name: str,
        object_name: str,
        destination: Path | str,
    ) -> Path:
        destination_path = Path(destination).expanduser().resolve()
        destination_path.parent.mkdir(parents=True, exist_ok=True)
        await self._run(
            self._client.fget_object,
            bucket_name=bucket_name,
            object_name=object_name,
            file_path=str(destination_path),
        )
        self._log.info(
            "Downloaded object '{}' from bucket '{}' to '{}'",
            object_name,
            bucket_name,
            destination_path,
        )
        return destination_path

    async def stat_object(self, bucket_name: str, object_name: str):
        """Fetch metadata for an object; raises if not found."""
        return await self._run(
            self._client.stat_object,
            bucket_name=bucket_name,
            object_name=object_name,
        )

    async def generate_presigned_url(
        self,
        bucket_name: str,
        object_name: str,
        *,
        expires_in: int = 3600,
    ) -> str:
        """
        Generate a presigned download URL for an object.

        Args:
            bucket_name: MinIO bucket
            object_name: Object key
            expires_in: Expiration window in seconds (default 1 hour)
        """
        expires_in = max(1, min(expires_in, 7 * 24 * 3600))
        url = await self._run(
            self._client.presigned_get_object,
            bucket_name=bucket_name,
            object_name=object_name,
            expires=timedelta(seconds=expires_in),
        )
        self._log.debug(
            "Generated presigned URL for '{}' (expires in {} seconds)",
            object_name,
            expires_in,
        )
        return url

    async def delete_object(self, bucket_name: str, object_name: str) -> None:
        await self._run(self._client.remove_object, bucket_name=bucket_name, object_name=object_name)
        self._log.info("Deleted object '{}' from bucket '{}'", object_name, bucket_name)

    async def delete_objects(self, bucket_name: str, object_names: Sequence[str]) -> list[str]:
        """Delete many objects, returning any keys that failed to delete."""

        keys = list(object_names)
        if not keys:
            return []

        def _remove_many() -> list[str]:
            delete_list = (DeleteObject(name) for name in keys)
            return [
                error.object_name
                for error in self._client.remove_objects(
                    bucket_name=bucket_name,
                    delete_object_list=delete_list,
                )
            ]

        failures = await self._run(_remove_many)
        if failures:
            self._log.warning(
                "Failed to delete {} object(s) from bucket '{}'", len(failures), bucket_name
            )
        else:
            self._log.info(
                "Deleted {} object(s) from bucket '{}'",
                len(keys),
                bucket_name,
            )
        return failures

    async def list_objects(
        self,
        bucket_name: str,
        *,
        prefix: str | None = None,
        recursive: bool = True,
    ) -> list[str]:
        def _list() -> list[str]:
            return [
                obj.object_name
                for obj in self._client.list_objects(
                    bucket_name, prefix=prefix, recursive=recursive
                )
            ]

        objects = await self._run(_list)
        self._log.debug(
            "Listed {} object(s) from bucket '{}' with prefix '{}'",
            len(objects),
            bucket_name,
            prefix or "",
        )
        return objects

    async def bucket_exists(self, bucket_name: str) -> bool:
        exists = await self._run(self._client.bucket_exists, bucket_name)
        self._log.debug("Bucket '{}' exists: {}", bucket_name, exists)
        return exists

