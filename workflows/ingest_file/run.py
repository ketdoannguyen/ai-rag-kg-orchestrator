from __future__ import annotations

import asyncio
import base64
import json
import mimetypes
import shutil
from datetime import datetime
from pathlib import Path
from typing import Any

from config.settings import settings
from loguru import logger
from typing import TYPE_CHECKING
from workflows.dependencies import get_minio_client
from .marker_client import MarkerClient
from .models import (
    FileIngestionInput,
    FileIngestionResult,
    FileIngestionStatus,
    MarkerFileMetadata,
    MarkerImageMetadata,
    MarkerOutputMetadata,
    SourceMetadata,
)

if TYPE_CHECKING:
    from minio_client import StorageClient  # type: ignore[import-not-found]


def start_ingest(
    payload: FileIngestionInput | str | Path,
) -> dict[str, Any]:
    """
    Entry point for the ingest_file workflow.

    Accepts either a prepared FileIngestionInput or a simple file path.
    """
    if isinstance(payload, (str, Path)):
        input_payload = FileIngestionInput(file_path=Path(payload))
    else:
        input_payload = payload

    marker_client = MarkerClient(
        use_llm=settings.MARKER_USE_LLM,
        force_ocr=input_payload.enable_ocr or settings.MARKER_FORCE_OCR,
        torch_device=settings.MARKER_TORCH_DEVICE,
        google_api_key=settings.GOOGLE_API_KEY,
    )

    storage = get_minio_client()
    workflow = FileIngestionWorkflow(
        storage=storage,
        bucket=settings.MINIO_BUCKET,
        workspace_root=Path(settings.INGEST_FILE_WORKDIR),
        marker_client=marker_client,
    )

    result = asyncio.run(workflow.run(input_payload))
    if hasattr(result, "model_dump"):
        return result.model_dump(mode="json")
    return result.dict()


class FileIngestionWorkflow:
    """Coordinates file ingestion, storage, and Marker parsing."""

    def __init__(
        self,
        *,
        storage: StorageClient,
        bucket: str,
        workspace_root: Path,
        marker_client: MarkerClient,
    ) -> None:
        self._storage = storage
        self._bucket = bucket
        self._workspace_root = workspace_root
        self._marker_client = marker_client
        self._date_prefix = datetime.utcnow().strftime("%d-%m-%Y")
        self._object_root = f"file/{self._date_prefix}"
        self._logger = logger.bind(job_id=None)

    async def run(self, payload: FileIngestionInput) -> FileIngestionResult:
        """Run the file ingestion workflow."""
        file_path = payload.file_path
        self._logger = logger.bind(job_id=None)

        self._workspace_root.mkdir(parents=True, exist_ok=True)
        await self._ensure_bucket_with_retry()

        job_id = datetime.utcnow().strftime("%H-%M-%S")
        self._logger = logger.bind(job_id=job_id)
        self._logger.info(f"Starting file ingestion for {file_path.name}")

        job_dir = self._workspace_root / job_id
        job_dir.mkdir(parents=True, exist_ok=True)

        try:
            source_meta = await self._persist_source(
                file_path=file_path, job_dir=job_dir, job_id=job_id
            )

            # Convert PDF in thread pool to avoid blocking event loop
            marker_result = await asyncio.to_thread(
                self._marker_client.convert_pdf, file_path
            )

            marker_meta = await self._persist_marker_outputs(
                marker_result=marker_result,
                job_dir=job_dir,
                job_id=job_id,
            )

            output_json_path = (
                marker_meta.json.local_path if marker_meta.json else None
            )

            self._logger.info("File ingestion completed successfully")

            return FileIngestionResult(
                status=FileIngestionStatus.COMPLETE,
                job_id=job_id,
                output_json_path=output_json_path,
                source=source_meta,
                marker=marker_meta,
            )
        except Exception as exc:
            self._logger.exception("File ingestion failed")
            return FileIngestionResult(
                status=FileIngestionStatus.FAILED,
                job_id=job_id,
                output_json_path=None,
                source=SourceMetadata(
                    local_path="",
                    minio_object="",
                    size_bytes=0,
                ),
                marker=MarkerOutputMetadata(request_id="", images=[]),
                error=str(exc),
            )

    async def _ensure_bucket_with_retry(self) -> None:
        """Ensure bucket exists with retry logic."""
        for attempt in range(3):
            try:
                await self._storage.ensure_bucket(self._bucket)
                return
            except Exception as exc:
                self._logger.warning(f"Bucket ensure failed (attempt {attempt + 1}): {exc}")
                if attempt < 2:
                    await asyncio.sleep(2.0 * (attempt + 1))
        self._logger.warning("Proceeding without confirming bucket availability")

    async def _persist_source(
        self, file_path: Path, job_dir: Path, job_id: str
    ) -> SourceMetadata:
        """Persist source PDF file to workspace and MinIO."""
        copied_path = job_dir / file_path.name
        await asyncio.to_thread(shutil.copy2, file_path, copied_path)

        minio_object = await self._upload_file(
            local_path=copied_path,
            object_name=f"{self._object_root}/source/{copied_path.name}",
            content_type="application/pdf",
        )

        def get_size() -> int:
            return copied_path.stat().st_size

        size_bytes = await asyncio.to_thread(get_size)
        return SourceMetadata(
            local_path=str(copied_path),
            minio_object=minio_object,
            size_bytes=size_bytes,
        )

    async def _persist_marker_outputs(
        self,
        *,
        marker_result: dict[str, Any],
        job_dir: Path,
        job_id: str,
    ) -> MarkerOutputMetadata:
        """Persist all Marker outputs to workspace and MinIO."""
        payload = marker_result["payload"]
        request_id = marker_result["request_id"]

        # Save raw payload
        raw_payload_path = job_dir / "marker_raw.json"
        await asyncio.to_thread(
            raw_payload_path.write_text,
            json.dumps(payload, indent=2, ensure_ascii=False),
            encoding="utf-8",
        )
        raw_payload_meta = MarkerFileMetadata(
            local_path=str(raw_payload_path),
            minio_object=await self._upload_file(
                local_path=raw_payload_path,
                object_name=f"{self._object_root}/output/{job_id}/marker_raw.json",
                content_type="application/json",
            ),
        )

        # Process JSON payload
        json_meta: MarkerFileMetadata | None = None
        json_payload = payload.get("json")
        json_data: Any | None = None
        if isinstance(json_payload, str):
            try:
                json_data = json.loads(json_payload)
            except json.JSONDecodeError:
                json_data = {"raw": json_payload}
        elif json_payload is not None:
            json_data = json_payload

        if json_data is not None:
            json_path = job_dir / "marker.json"
            await asyncio.to_thread(
                json_path.write_text,
                json.dumps(json_data, indent=2, ensure_ascii=False),
                encoding="utf-8",
            )
            json_meta = MarkerFileMetadata(
                local_path=str(json_path),
                minio_object=await self._upload_file(
                    local_path=json_path,
                    object_name=f"{self._object_root}/output/{job_id}/marker.json",
                    content_type="application/json",
                ),
            )

        # Process markdown payload
        markdown_meta: MarkerFileMetadata | None = None
        markdown_payload = payload.get("markdown")
        if isinstance(markdown_payload, str) and markdown_payload.strip():
            markdown_path = job_dir / "marker.md"
            await asyncio.to_thread(
                markdown_path.write_text, markdown_payload, encoding="utf-8"
            )
            markdown_meta = MarkerFileMetadata(
                local_path=str(markdown_path),
                minio_object=await self._upload_file(
                    local_path=markdown_path,
                    object_name=f"{self._object_root}/output/{job_id}/marker.md",
                    content_type="text/markdown",
                ),
            )

        # Process images
        images: list[MarkerImageMetadata] = []
        images_payload = payload.get("images")
        if images_payload:
            images_dir = job_dir / "images"
            images_dir.mkdir(parents=True, exist_ok=True)
            images = await self._save_images(
                images_payload=images_payload,
                images_dir=images_dir,
                job_id=job_id,
            )

        return MarkerOutputMetadata(
            request_id=request_id,
            json=json_meta,
            markdown=markdown_meta,
            raw_payload=raw_payload_meta,
            images=images,
        )

    async def _save_images(
        self,
        *,
        images_payload: Any,
        images_dir: Path,
        job_id: str,
    ) -> list[MarkerImageMetadata]:
        """Save extracted images to workspace and MinIO."""
        saved_images: list[MarkerImageMetadata] = []

        if isinstance(images_payload, dict):
            iterator = images_payload.items()
        elif isinstance(images_payload, list):
            iterator = enumerate(images_payload)
        else:
            return saved_images

        for key, value in iterator:
            image_info = self._extract_image_info(key, value)
            if image_info is None:
                continue

            filename = image_info["filename"]
            image_path = images_dir / filename
            await asyncio.to_thread(image_path.write_bytes, image_info["data"])

            minio_object = await self._upload_file(
                local_path=image_path,
                object_name=f"{self._object_root}/output/{job_id}/images/{filename}",
                content_type=image_info["content_type"],
            )

            saved_images.append(
                MarkerImageMetadata(
                    filename=filename,
                    content_type=image_info["content_type"],
                    local_path=str(image_path),
                    minio_object=minio_object,
                ),
            )

        return saved_images

    def _extract_image_info(
        self,
        key: Any,
        value: Any,
    ) -> dict[str, Any] | None:
        data_b64: str | None = None
        content_type: str | None = None
        filename: str | None = None

        if isinstance(value, dict):
            data_b64 = value.get("data") or value.get("content")
            content_type = value.get("content_type") or value.get("mime_type")
            filename = value.get("filename")
        elif isinstance(value, str):
            data_b64 = value

        if not filename:
            filename = f"{key}.png"
        if not content_type:
            guessed_type, _ = mimetypes.guess_type(filename)
            content_type = guessed_type or "image/png"

        if not data_b64:
            return None

        try:
            binary = base64.b64decode(data_b64)
        except (ValueError, TypeError):
            return None

        return {"filename": filename, "content_type": content_type, "data": binary}

    async def _upload_file(self, *, local_path: Path, object_name: str, content_type: str) -> str:
        local_path = local_path.expanduser().resolve()
        await self._storage.upload_file(
            bucket_name=self._bucket,
            object_name=object_name,
            file_path=local_path,
            content_type=content_type,
        )
        return object_name
