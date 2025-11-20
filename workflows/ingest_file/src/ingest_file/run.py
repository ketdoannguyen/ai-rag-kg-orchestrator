from __future__ import annotations

import base64
import json
import mimetypes
import shutil
from datetime import datetime
from pathlib import Path
from typing import Any

from config.settings import settings
from typing import TYPE_CHECKING
from ._dependencies import get_storage_client
from .marker_client import MarkerClient, MarkerResult

if TYPE_CHECKING:
    from storage_client import StorageClient  # type: ignore[import-not-found]

storage = get_storage_client()

def start_ingest(file_path: str) -> dict[str, Any]:
    """Entry point for the ingest_file workflow."""
 
    marker_client = MarkerClient(
        use_llm=settings.MARKER_USE_LLM,
        force_ocr=settings.MARKER_FORCE_OCR,
        torch_device=settings.MARKER_TORCH_DEVICE,
        google_api_key=settings.GOOGLE_API_KEY,
    )
    
    workflow = FileIngestionWorkflow(
        storage=storage,
        bucket=settings.MINIO_BUCKET,
        workspace_root=Path(settings.INGEST_FILE_WORKDIR),
        marker_client=marker_client,
    )

    return workflow.run(Path(file_path))


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

    def run(self, file_path: Path) -> dict[str, Any]:
        file_path = file_path.expanduser().resolve()
        if not file_path.exists():
            raise FileNotFoundError(f"Input file not found: {file_path}")

        self._workspace_root.mkdir(parents=True, exist_ok=True)
        self._storage.ensure_bucket(self._bucket)

        job_id = datetime.utcnow().strftime("%H-%M-%S")
        job_dir = self._workspace_root / job_id
        job_dir.mkdir(parents=True, exist_ok=True)

        source_meta = self._persist_source(file_path=file_path, job_dir=job_dir, job_id=job_id)

        marker_result = self._marker_client.convert_pdf(file_path)
        marker_meta = self._persist_marker_outputs(
            marker_result=marker_result,
            job_dir=job_dir,
            job_id=job_id,
        )

        return {
            "status": "complete",
            "job_id": job_id,
            "output_json_path": marker_meta.get("json", {}).get("local_path"),
            "source": source_meta,
            "marker": marker_meta,
        }

    def _persist_source(self, file_path: Path, job_dir: Path, job_id: str) -> dict[str, Any]:
        copied_path = job_dir / file_path.name
        shutil.copy2(file_path, copied_path)

        minio_object = self._upload_file(
            local_path=copied_path,
            object_name=f"{self._object_root}/source/{copied_path.name}",
            content_type="application/pdf",
        )

        return {
            "local_path": str(copied_path),
            "minio_object": minio_object,
            "size_bytes": copied_path.stat().st_size,
        }

    def _persist_marker_outputs(
        self,
        *,
        marker_result: MarkerResult,
        job_dir: Path,
        job_id: str,
    ) -> dict[str, Any]:
        payload = marker_result["payload"]
        outputs: dict[str, Any] = {
            "request_id": marker_result["request_id"],
            "json": None,
            "markdown": None,
            "images": [],
            "raw_payload": None,
        }

        raw_payload_path = job_dir / "marker_raw.json"
        raw_payload_path.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")
        outputs["raw_payload"] = {
            "local_path": str(raw_payload_path),
            "minio_object": self._upload_file(
                local_path=raw_payload_path,
                object_name=f"{self._object_root}/output/{job_id}/marker_raw.json",
                content_type="application/json",
            ),
        }

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
            json_path.write_text(json.dumps(json_data, indent=2, ensure_ascii=False), encoding="utf-8")
            outputs["json"] = {
                "local_path": str(json_path),
                "minio_object": self._upload_file(
                    local_path=json_path,
                    object_name=f"{self._object_root}/output/{job_id}/marker.json",
                    content_type="application/json",
                ),
            }

        markdown_payload = payload.get("markdown")
        if isinstance(markdown_payload, str) and markdown_payload.strip():
            markdown_path = job_dir / "marker.md"
            markdown_path.write_text(markdown_payload, encoding="utf-8")
            outputs["markdown"] = {
                "local_path": str(markdown_path),
                "minio_object": self._upload_file(
                    local_path=markdown_path,
                    object_name=f"{self._object_root}/output/{job_id}/marker.md",
                    content_type="text/markdown",
                ),
            }

        images_payload = payload.get("images")
        if images_payload:
            images_dir = job_dir / "images"
            images_dir.mkdir(parents=True, exist_ok=True)
            outputs["images"] = self._save_images(
                images_payload=images_payload,
                images_dir=images_dir,
                job_id=job_id,
            )

        return outputs

    def _save_images(
        self,
        *,
        images_payload: Any,
        images_dir: Path,
        job_id: str,
    ) -> list[dict[str, Any]]:
        saved_images: list[dict[str, Any]] = []

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
            image_path.write_bytes(image_info["data"])

            minio_object = self._upload_file(
                local_path=image_path,
                object_name=f"{self._object_root}/output/{job_id}/images/{filename}",
                content_type=image_info["content_type"],
            )

            saved_images.append(
                {
                    "filename": filename,
                    "content_type": image_info["content_type"],
                    "local_path": str(image_path),
                    "minio_object": minio_object,
                },
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

    def _upload_file(self, *, local_path: Path, object_name: str, content_type: str) -> str:
        local_path = local_path.expanduser().resolve()
        self._storage.upload_file(
            bucket_name=self._bucket,
            object_name=object_name,
            file_path=local_path,
            content_type=content_type,
        )
        return object_name
