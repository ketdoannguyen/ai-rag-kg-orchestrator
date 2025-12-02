from __future__ import annotations

import asyncio
import hashlib
import io
import mimetypes
import shutil
import tempfile
import uuid
import zipfile
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any, Awaitable, Callable

from config.settings import settings
from loguru import logger
from minio_client import StorageClient
from postgres_client import PostgresClient
from postgres_client.handlers.system_admin_handler import SystemAdminHandler
from postgres_client.models.enums import SourceType
from postgres_client.models.system_admin import DataSource

from tenacity import retry, retry_if_exception_type, stop_after_attempt, wait_exponential

from workflows.dependencies import get_minio_client, init_postgres_client
from workflows.shared.storage_utils import ensure_bucket_with_retry, upload_file
from .helper import log_upload_retry
from .exceptions import (
    DatabaseError,
    DuplicateFileError,
    FileValidationError,
    StorageUploadError,
)
from .models import (
    FileUploadInput,
    FileUploadResult,
    MinioInfo,
    StoredObject,
    UploadSourceType,
    ValidatedFile,
)

ALLOWED_EXTENSIONS = {".pdf", ".docx", ".txt", ".xlsx", ".csv", ".pptx"}
TEXTUAL_EXTENSIONS = {".txt", ".csv"}
MAX_FILE_SIZE_BYTES = 100 * 1024 * 1024  # 100MB
DEFAULT_PRESIGNED_TTL_SECONDS = 3600

SourceTypeMapper = dict[UploadSourceType, SourceType]


AntivirusScanner = Callable[[bytes, str], Awaitable[None]]


async def start_upload_file(
    payload: FileUploadInput | dict[str, Any],
    *,
    storage: StorageClient | None = None,
    pg_client: PostgresClient | None = None,
    antivirus_scanner: AntivirusScanner | None = None,
    fail_on_duplicate: bool = False,
) -> dict[str, Any]:
    """
    Entry point for the upload_file workflow.

    Args:
        payload: Workflow input (Pydantic model or plain dict)
        storage: Optional MinIO client (defaults to shared singleton)
        pg_client: Optional Postgres client (defaults to settings.DATABASE_URL)
        antivirus_scanner: Optional coroutine that raises on infection
        fail_on_duplicate: If True, raise DuplicateFileError instead of reusing record
    """

    input_payload = payload if isinstance(payload, FileUploadInput) else FileUploadInput(**payload)
    storage_client = storage or get_minio_client()
    postgres_client = pg_client or init_postgres_client()

    workflow = FileUploadWorkflow(
        storage=storage_client,
        bucket=settings.MINIO_KNOWLEDGE_BUCKET or settings.MINIO_BUCKET,
        pg_client=postgres_client,
        antivirus_scanner=antivirus_scanner,
        fail_on_duplicate=fail_on_duplicate,
    )
    result = await workflow.run(input_payload)
    if hasattr(result, "model_dump"):
        return result.model_dump(mode="json")
    return result.dict()


def start_upload_file_sync(
    payload: FileUploadInput | dict[str, Any],
    *,
    storage: StorageClient | None = None,
    pg_client: PostgresClient | None = None,
    antivirus_scanner: AntivirusScanner | None = None,
    fail_on_duplicate: bool = False,
) -> dict[str, Any]:
    """
    Synchronous helper for environments without event loop control.
    """

    return asyncio.run(
        start_upload_file(
            payload,
            storage=storage,
            pg_client=pg_client,
            antivirus_scanner=antivirus_scanner,
            fail_on_duplicate=fail_on_duplicate,
        )
    )


class FileUploadWorkflow:
    """Implements the upload_file workflow with validation, storage, and persistence."""

    SOURCE_TYPE_MAPPING: SourceTypeMapper = {
        UploadSourceType.FILE: SourceType.PDF,
        UploadSourceType.FAQ: SourceType.FAQ_FILE,
    }

    def __init__(
        self,
        *,
        storage: StorageClient,
        bucket: str,
        pg_client: PostgresClient,
        antivirus_scanner: AntivirusScanner | None = None,
        fail_on_duplicate: bool = False,
        presigned_ttl_seconds: int = DEFAULT_PRESIGNED_TTL_SECONDS,
    ) -> None:
        self._storage = storage
        self._bucket = bucket
        self._sessionmaker = pg_client.sessionmaker
        self._antivirus_scanner = antivirus_scanner
        self._fail_on_duplicate = fail_on_duplicate
        self._presigned_ttl = presigned_ttl_seconds
        self._log = logger.bind(workflow="upload_file")

    async def run(self, payload: FileUploadInput) -> FileUploadResult:
        """Execute the workflow end-to-end."""
        self._log.info("Validating incoming file", filename=payload.filename)
        validated = self._validate_file(payload)
        await self._maybe_scan_for_malware(payload)

        file_hash = self._calculate_md5(payload.file)
        self._log.info("Calculated file hash", hash=file_hash)

        await ensure_bucket_with_retry(self._storage, self._bucket)

        async with self._sessionmaker() as session:
            handler = SystemAdminHandler(session)
            existing = await handler.sources.get_by_hash(file_hash)
            if existing:
                self._log.warning(
                    "Duplicate file detected",
                    data_source_id=str(existing.id),
                    filename=payload.filename,
                )
                if self._fail_on_duplicate:
                    raise DuplicateFileError(
                        f"File '{payload.filename}' already uploaded as data_source={existing.id}"
                    )
                minio_info = await self._build_minio_info_from_config(existing.config, file_hash)
                return FileUploadResult(data_source=existing, minio_info=minio_info, duplicate=True)

            stored_object = await self._upload_to_storage(
                payload=payload,
                validated=validated,
                hash_content=file_hash,
            )

            data_source_model = self._build_data_source(
                payload=payload,
                validated=validated,
                hash_content=file_hash,
                minio_path=stored_object.object_name,
                file_id=stored_object.file_id,
            )

            try:
                saved = await handler.sources.create(data_source_model)
            except Exception as exc:  # pragma: no cover - defensive cleanup
                self._log.error(
                    "Failed to persist DataSource, cleaning up uploaded object",
                    error=str(exc),
                )
                await self._cleanup_object(stored_object.object_name)
                raise DatabaseError("Could not save data_source record") from exc

        minio_info = await self._build_minio_info(
            bucket=self._bucket,
            object_name=stored_object.object_name,
            hash_content=file_hash,
        )
        self._log.info(
            "Upload workflow completed",
            data_source_id=str(saved.id),
            bucket=self._bucket,
            object_name=stored_object.object_name,
        )
        return FileUploadResult(data_source=saved, minio_info=minio_info, duplicate=False)

    def _validate_file(self, payload: FileUploadInput) -> ValidatedFile:
        normalized = self._normalize_filename(payload.filename)
        extension = Path(normalized).suffix.lower()
        if extension not in ALLOWED_EXTENSIONS:
            raise FileValidationError(
                f"Extension '{extension}' is not allowed. "
                f"Supported extensions: {sorted(ALLOWED_EXTENSIONS)}"
            )

        size_bytes = len(payload.file)
        if size_bytes == 0:
            raise FileValidationError("File is empty")
        if size_bytes > MAX_FILE_SIZE_BYTES:
            raise FileValidationError("File exceeds 100MB limit")

        mime_type = payload.content_type or mimetypes.guess_type(normalized)[0] or "application/octet-stream"
        self._enforce_signature(extension, payload.file)

        self._log.info(
            "Validation completed",
            filename=normalized,
            size_bytes=size_bytes,
            mime_type=mime_type,
        )

        return ValidatedFile(
            original_name=payload.filename,
            normalized_name=normalized,
            extension=extension,
            mime_type=mime_type,
            size_bytes=size_bytes,
        )

    async def _maybe_scan_for_malware(self, payload: FileUploadInput) -> None:
        if not self._antivirus_scanner:
            self._log.debug("Skipping malware scan (no scanner configured)")
            return
        self._log.info("Scanning file for malware", filename=payload.filename)
        await self._antivirus_scanner(payload.file, payload.filename)
        self._log.info("Malware scan completed", filename=payload.filename)

    def _calculate_md5(self, data: bytes) -> str:
        digest = hashlib.md5()
        digest.update(data)
        return digest.hexdigest()

    @retry(
        reraise=True,
        stop=stop_after_attempt(3),
        wait=wait_exponential(multiplier=1, min=1, max=8),
        retry=retry_if_exception_type(StorageUploadError),
        before_sleep=log_upload_retry,
    )
    async def _upload_to_storage(
        self,
        *,
        payload: FileUploadInput,
        validated: ValidatedFile,
        hash_content: str,
    ) -> StoredObject:
        file_id = uuid.uuid4()
        object_name = self._build_object_name(
            user_id=str(payload.created_by),
            file_id=file_id,
            filename=validated.normalized_name,
        )

        tmp_dir = Path(tempfile.mkdtemp(prefix="upload-file-"))
        tmp_path = tmp_dir / validated.normalized_name
        await asyncio.to_thread(tmp_path.write_bytes, payload.file)

        metadata = {
            "content-md5": hash_content,
            "x-upload-user": str(payload.created_by),
            "original-filename": validated.original_name,
        }

        try:
            await upload_file(
                storage=self._storage,
                bucket=self._bucket,
                local_path=tmp_path,
                object_name=object_name,
                content_type=validated.mime_type,
                metadata=metadata,
            )
            await self._storage.stat_object(self._bucket, object_name)
            self._log.info("Uploaded object to MinIO", bucket=self._bucket, object_name=object_name)
            return StoredObject(object_name=object_name, file_id=file_id)
        except Exception as exc:  # pragma: no cover - relies on MinIO failures
            self._log.error("Failed to upload file to MinIO", error=str(exc))
            raise StorageUploadError("Uploading file to MinIO failed") from exc
        finally:
            shutil.rmtree(tmp_dir, ignore_errors=True)

    def _build_data_source(
        self,
        *,
        payload: FileUploadInput,
        validated: ValidatedFile,
        hash_content: str,
        minio_path: str,
        file_id: uuid.UUID,
    ) -> DataSource:
        source_type = self.SOURCE_TYPE_MAPPING.get(payload.source_type)
        if not source_type:
            raise FileValidationError(f"Unsupported source_type '{payload.source_type}'")

        config = {
            "minio_bucket": self._bucket,
            "minio_path": minio_path,
            "file_size": validated.size_bytes,
            "mime_type": validated.mime_type,
            "original_filename": validated.original_name,
            "file_id": str(file_id),
        }
        if payload.additional_config:
            config.update(payload.additional_config)

        return DataSource(
            name=validated.original_name,
            type=source_type,
            config=config,
            hash_content=hash_content,
            status="processing",
            created_by=payload.created_by,
        )

    async def _build_minio_info(self, *, bucket: str, object_name: str, hash_content: str) -> MinioInfo:
        presigned_url, expires_at = await self._try_presign(bucket, object_name)
        return MinioInfo(
            bucket=bucket,
            path=object_name,
            full_url=f"minio://{bucket}/{object_name}",
            md5_hash=hash_content,
            presigned_download_url=presigned_url,
            expires_at=expires_at,
        )

    async def _build_minio_info_from_config(self, config: dict[str, Any], hash_content: str) -> MinioInfo:
        bucket = config.get("minio_bucket") or self._bucket
        path = config.get("minio_path")
        if not path:
            raise DatabaseError("Existing data source is missing MinIO path information")
        return await self._build_minio_info(bucket=bucket, object_name=path, hash_content=hash_content)

    async def _try_presign(self, bucket: str, object_name: str) -> tuple[str | None, datetime | None]:
        try:
            url = await self._storage.generate_presigned_url(
                bucket_name=bucket,
                object_name=object_name,
                expires_in=self._presigned_ttl,
            )
            # Use timezone-aware UTC timestamp
            expires_at = datetime.now(timezone.utc) + timedelta(seconds=self._presigned_ttl)
            return url, expires_at
        except Exception as exc:  # pragma: no cover - depends on MinIO features
            self._log.warning(
                "Could not generate presigned URL",
                bucket=bucket,
                object_name=object_name,
                error=str(exc),
            )
            return None, None

    async def _cleanup_object(self, object_name: str) -> None:
        try:
            await self._storage.delete_object(self._bucket, object_name)
            self._log.info("Deleted orphaned MinIO object", object_name=object_name)
        except Exception as exc:  # pragma: no cover - best-effort cleanup
            self._log.warning(
                "Failed to delete orphaned object",
                object_name=object_name,
                error=str(exc),
            )

    def _enforce_signature(self, extension: str, data: bytes) -> None:
        if extension == ".pdf":
            valid = self._looks_like_pdf(data)
        elif extension in {".docx", ".pptx", ".xlsx"}:
            folder = {".docx": "word/", ".pptx": "ppt/", ".xlsx": "xl/"}[extension]
            valid = self._office_archive_contains(data, folder)
        elif extension in TEXTUAL_EXTENSIONS:
            valid = self._looks_textual(data)
        else:
            valid = True

        if not valid:
            raise FileValidationError(
                f"File content does not match the declared extension '{extension}'"
            )

    @staticmethod
    def _normalize_filename(filename: str) -> str:
        normalized = Path(filename).name or "upload.bin"
        normalized = normalized.replace("\x00", "")
        return normalized

    @staticmethod
    def _looks_like_pdf(data: bytes) -> bool:
        return data.startswith(b"%PDF-")

    @staticmethod
    def _office_archive_contains(data: bytes, prefix: str) -> bool:
        try:
            with zipfile.ZipFile(io.BytesIO(data)) as archive:
                return any(name.startswith(prefix) for name in archive.namelist())
        except zipfile.BadZipFile:
            return False

    @staticmethod
    def _looks_textual(data: bytes) -> bool:
        sample = data[:2048]
        if not sample:
            return True
        if b"\x00" in sample:
            return False
        try:
            sample.decode("utf-8")
            return True
        except UnicodeDecodeError:
            return False

    @staticmethod
    def _build_object_name(*, user_id: str, file_id: uuid.UUID, filename: str) -> str:
        return f"{user_id}/{file_id}/{filename}"


__all__ = ["start_upload_file", "start_upload_file_sync", "FileUploadWorkflow"]
