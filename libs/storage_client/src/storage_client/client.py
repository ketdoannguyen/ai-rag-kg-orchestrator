from __future__ import annotations

import io
from pathlib import Path
from typing import Sequence

from minio import Minio
from minio.deleteobjects import DeleteObject


class StorageClient:
    """Lightweight helper around MinIO for uploads, downloads, and housekeeping."""

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

    def ensure_bucket(self, bucket_name: str) -> None:
        if self._client.bucket_exists(bucket_name):
            return
        self._client.make_bucket(bucket_name)

    def upload_file(
        self,
        bucket_name: str,
        object_name: str,
        file_path: Path | str,
        *,
        content_type: str | None = None,
        metadata: dict[str, str] | None = None,
    ) -> str:
        path = Path(file_path).expanduser().resolve()
        self._client.fput_object(
            bucket_name=bucket_name,
            object_name=object_name,
            file_path=str(path),
            content_type=content_type,
            metadata=metadata,
        )
        return object_name

    def upload_bytes(
        self,
        bucket_name: str,
        object_name: str,
        data: bytes | str,
        *,
        content_type: str | None = None,
        metadata: dict[str, str] | None = None,
    ) -> str:
        if isinstance(data, str):
            payload = data.encode("utf-8")
        else:
            payload = data

        buffer = io.BytesIO(payload)
        self._client.put_object(
            bucket_name=bucket_name,
            object_name=object_name,
            data=buffer,
            length=len(payload),
            content_type=content_type,
            metadata=metadata,
        )
        return object_name

    def download_file(
        self,
        bucket_name: str,
        object_name: str,
        destination: Path | str,
    ) -> Path:
        destination_path = Path(destination).expanduser().resolve()
        destination_path.parent.mkdir(parents=True, exist_ok=True)
        self._client.fget_object(
            bucket_name=bucket_name,
            object_name=object_name,
            file_path=str(destination_path),
        )
        return destination_path

    def delete_object(self, bucket_name: str, object_name: str) -> None:
        self._client.remove_object(bucket_name=bucket_name, object_name=object_name)

    def delete_objects(self, bucket_name: str, object_names: Sequence[str]) -> list[str]:
        """Delete many objects, returning any keys that failed to delete."""

        delete_list = (DeleteObject(name) for name in object_names)
        return [
            error.object_name
            for error in self._client.remove_objects(
                bucket_name=bucket_name,
                delete_object_list=delete_list,
            )
        ]

    def list_objects(
        self,
        bucket_name: str,
        *,
        prefix: str | None = None,
        recursive: bool = True,
    ) -> list[str]:
        return [
            obj.object_name
            for obj in self._client.list_objects(bucket_name, prefix=prefix, recursive=recursive)
        ]

    def bucket_exists(self, bucket_name: str) -> bool:
        return self._client.bucket_exists(bucket_name)

