from __future__ import annotations

import asyncio
import json
import shutil
import warnings
from datetime import datetime, timezone
from pathlib import Path
from typing import Any
from uuid import UUID

from config.settings import settings
from loguru import logger
from arango_client import (
    ArangoClient,
    set_parent_ids,
    set_positions,
    compute_tree_statistics,
    validate_tree_and_attach_source,
    serialize_tree_to_dict,
    SystemAdminHandler,
    DataSource as DataSourceModel,
    SourceType,
)
from typing import TYPE_CHECKING

from workflows.dependencies import get_minio_client, init_arango_client
from workflows.shared import calculate_md5
from .marker_client import MarkerClient
from .parser import TextbookParser
from .models import FileIngestionInput, FileIngestionResult, FileIngestionStatus

if TYPE_CHECKING:
    from minio_client import StorageClient  # type: ignore[import-not-found]


def start_ingest(
    payload: FileIngestionInput | dict[str, Any] | UUID | str,
    *,
    storage: StorageClient | None = None,
    arango_client: ArangoClient | None = None,
) -> dict[str, Any]:
    """
    Entry point for the ingest_file workflow.

    Accepts either:
    - a prepared FileIngestionInput
    - a plain dict (will be unpacked into FileIngestionInput)
    - a UUID / string UUID (interpreted as data_source_id)
    """
    if isinstance(payload, FileIngestionInput):
        input_payload = payload
    elif isinstance(payload, (UUID, str)):
        input_payload = FileIngestionInput(data_source_id=str(payload))
    else:
        input_payload = FileIngestionInput(**payload)

    marker_client = MarkerClient(
        use_llm=settings.MARKER_USE_LLM,
        force_ocr=input_payload.enable_ocr or settings.MARKER_FORCE_OCR,
        torch_device=settings.MARKER_TORCH_DEVICE,
        google_api_key=settings.GOOGLE_API_KEY,
    )

    storage_client = storage or get_minio_client()
    db_client = arango_client or init_arango_client()

    workflow = FileIngestionWorkflow(
        storage=storage_client,
        workspace_root=Path(settings.INGEST_FILE_WORKDIR),
        marker_client=marker_client,
        arango_client=db_client,
    )

    result = asyncio.run(workflow.run(input_payload))
    if hasattr(result, "model_dump"):
        # Suppress Pydantic serialization warnings for nested models
        # These warnings occur because attributes/metadata are Pydantic models in derived classes
        # but are correctly serialized to dict by model_dump(mode="json")
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", UserWarning)
            return result.model_dump(mode="json")
    return result.dict()


class FileIngestionWorkflow:
    """Coordinates DataSource-based ingestion, Marker parsing, and in-memory knowledge tree build."""

    def __init__(
        self,
        *,
        storage: StorageClient,
        workspace_root: Path,
        marker_client: MarkerClient,
        arango_client: ArangoClient,
    ) -> None:
        self._storage = storage
        self._workspace_root = workspace_root
        self._marker_client = marker_client
        self._arango_client = arango_client
        self._logger = logger.bind(job_id=None, workflow="ingest_file")

    async def run(self, payload: FileIngestionInput) -> FileIngestionResult:
        """
        Run the ingestion workflow:

        1. Load DataSource and validate status/config
        2. Download file from MinIO
        3. Verify MD5 hash
        4. Parse file using Marker
        5. Build knowledge tree via TextbookParser
        6. Validate nodes
        7. Update DataSource metadata
        """
        # Use timezone-aware UTC for job IDs and timing
        job_id = datetime.now(timezone.utc).strftime("%H-%M-%S")
        self._logger = logger.bind(job_id=job_id, workflow="ingest_file")
        ingestion_start_time = datetime.now(timezone.utc)

        self._workspace_root.mkdir(parents=True, exist_ok=True)
        job_dir = self._workspace_root / job_id
        job_dir.mkdir(parents=True, exist_ok=True)

        data_source_id = UUID(payload.data_source_id)

        try:
            # ArangoClient uses sync operations - no need for async context
            sys_handler = SystemAdminHandler(self._arango_client.db)

            # Bước 1: Load DataSource - Lấy thông tin nguồn
            data_source = sys_handler.sources.get_by_id(data_source_id)
            if not data_source:
                raise ValueError(f"DataSource not found: {data_source_id}")

            if data_source.status == "active":
                raise ValueError(
                    f"DataSource {data_source_id} is already ingested (status=active)"
                )

            config = data_source.config or {}
            minio_bucket = config.get("minio_bucket")
            minio_path = config.get("minio_path")
            if not minio_bucket or not minio_path:
                raise ValueError(
                    f"DataSource {data_source_id} is missing minio_bucket/minio_path in config"
                )

            self._logger.info(
                "Loaded DataSource for ingestion",
                data_source_id=str(data_source_id),
                bucket=minio_bucket,
                path=minio_path,
            )

            # Bước 2: Download File from MinIO - Tải file về
            local_filename = config.get("original_filename") or "source.pdf"
            local_path = job_dir / local_filename
            await self._storage.download_file(
                bucket_name=minio_bucket,
                object_name=minio_path,
                destination=local_path,
            )

            file_bytes = await asyncio.to_thread(local_path.read_bytes)

            # Bước 3: Verify Hash - Kiểm tra tính toàn vẹn
            calculated_hash = calculate_md5(file_bytes)
            if data_source.hash_content and calculated_hash != data_source.hash_content:
                raise ValueError(
                    "Integrity check failed: downloaded file hash does not match DataSource.hash_content"
                )

            self._logger.info(
                "Integrity check passed",
                expected_hash=data_source.hash_content,
                actual_hash=calculated_hash,
            )

            # Bước 4: Parse File - Phân tích nội dung (Marker)
            marker_result = await asyncio.to_thread(self._marker_client.convert_pdf, local_path)
            payload_dict: dict[str, Any] = marker_result.get("payload", {})
            raw_json = payload_dict.get("json")
            if raw_json is None:
                raise ValueError("Marker output does not contain 'json' payload")

            # Persist raw JSON locally for parser only (không upload MinIO)
            raw_json_path = job_dir / "marker_raw.json"
            await asyncio.to_thread(
                raw_json_path.write_text,
                json.dumps(raw_json, ensure_ascii=False, indent=2),
                encoding="utf-8",
            )

            # Bước 5: Build Knowledge Tree - Tạo cây phân cấp (in-memory)
            processed_json_path = job_dir / "processed.json"
            parser = TextbookParser(
                input_path=str(raw_json_path),
                output_path=str(processed_json_path),
                page_offset=5,
                extract_images=True,
                bucket=minio_bucket,
            )

            await parser.parse()

            # Set parent IDs & positions cho toàn bộ cây
            set_parent_ids(parser.root)
            set_positions(parser.root)

            # Gán source_id + validate tree theo rule
            validate_tree_and_attach_source(parser.root, data_source_id)

            # Bước 7: Calculate Statistics - Thống kê
            statistics = compute_tree_statistics(
                parser.root,
                start_time=ingestion_start_time,
            )

            # Bước 8: Update DataSource metadata (không persist tree vào DB)
            # ArangoClient sync operations - convert datetime to ISO string
            sys_handler.sources.update(
                data_source_id,
                {
                    "updated_at": datetime.now(timezone.utc).isoformat(),
                    "status": "active",
                },
            )

            # Reload DataSource to get updated version
            final_data_source = sys_handler.sources.get_by_id(data_source_id)
            if not final_data_source:
                raise ValueError(f"DataSource {data_source_id} not found after update")

            self._logger.info(
                "File ingestion completed successfully (in-memory tree only)",
                data_source_id=str(data_source_id),
                total_nodes=statistics.get("total_nodes", 0),
            )

            # Convert root_node to dict using serialize_tree_to_dict for proper children serialization
            # model_dump() doesn't handle nested subclass children correctly
            with warnings.catch_warnings():
                warnings.simplefilter("ignore", UserWarning)
                root_node_dict = serialize_tree_to_dict(parser.root) if parser.root else None

            return FileIngestionResult(
                status=FileIngestionStatus.COMPLETE,
                job_id=job_id,
                data_source=final_data_source,
                root_node=root_node_dict,
                statistics=statistics,
            )
        except Exception as exc:
            self._logger.exception("File ingestion failed")
            # Try to get DataSource even on failure
            data_source_on_error = None
            try:
                sys_handler = SystemAdminHandler(self._arango_client.db)
                data_source_on_error = sys_handler.sources.get_by_id(data_source_id)
            except Exception:
                pass

            # If we can't get DataSource, create a minimal one with just the ID
            if not data_source_on_error:
                data_source_on_error = DataSourceModel(
                    id=data_source_id,
                    name="Unknown",
                    type=SourceType.PDF,
                    status="error",
                )

            return FileIngestionResult(
                status=FileIngestionStatus.FAILED,
                job_id=job_id,
                data_source=data_source_on_error,
                root_node=None,
                statistics={
                    "successful": False,
                    "error": str(exc),
                },
                error=str(exc),
            )
        finally:
            # Best-effort cleanup of temp workspace
            try:
                shutil.rmtree(job_dir, ignore_errors=True)
            except Exception:
                pass
