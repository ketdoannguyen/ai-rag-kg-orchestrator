from __future__ import annotations

import asyncio
import shutil
import warnings
from datetime import UTC, datetime
from pathlib import Path
from typing import TYPE_CHECKING, Any
from uuid import UUID

from arango_client import (
    ArangoClient,
    SourceType,
    SystemAdminHandler,
    compute_tree_statistics,
    serialize_tree_to_dict,
    set_parent_ids,
    set_positions,
    validate_tree_and_attach_source,
)
from arango_client import (
    DataSource as DataSourceModel,
)
from loguru import logger

from config.settings import settings
from workflows.dependencies import get_minio_client, init_arango_client
from workflows.shared import calculate_md5

from .models import FileIngestionInput, FileIngestionResult, FileIngestionStatus
from .vision_ocr import VisionOCRClient, VisionOCRConfig, render_pdf_pages
from .vision_parser import VisionDocumentParser

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

    vision_client = VisionOCRClient(
        VisionOCRConfig.from_settings(
            settings,
            max_retries=input_payload.max_retries,
            timeout_seconds=input_payload.timeout_seconds,
        )
    )

    storage_client = storage or get_minio_client()
    db_client = arango_client or init_arango_client()

    workflow = FileIngestionWorkflow(
        storage=storage_client,
        workspace_root=Path(settings.INGEST_FILE_WORKDIR),
        vision_client=vision_client,
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
    """Coordinates DataSource-based ingestion, vision OCR, and in-memory knowledge tree build."""

    def __init__(
        self,
        *,
        storage: StorageClient,
        workspace_root: Path,
        vision_client: VisionOCRClient,
        arango_client: ArangoClient,
    ) -> None:
        self._storage = storage
        self._workspace_root = workspace_root
        self._vision_client = vision_client
        self._arango_client = arango_client
        self._logger = logger.bind(job_id=None, workflow="ingest_file")

    async def run(self, payload: FileIngestionInput) -> FileIngestionResult:
        """
        Run the ingestion workflow:

        1. Load DataSource and validate status/config
        2. Download file from MinIO and verify MD5
        3. Render and transcribe every page with the vision model
        4. Normalize accepted pages into the knowledge tree
        5. Validate nodes and update DataSource metadata
        """
        # Use timezone-aware UTC for job IDs and timing
        job_id = datetime.now(UTC).strftime("%H-%M-%S")
        self._logger = logger.bind(job_id=job_id, workflow="ingest_file")
        ingestion_start_time = datetime.now(UTC)

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

            # Bước 4: OCR từng trang bằng vision model - Trang lỗi làm hỏng cả tài liệu
            parser = VisionDocumentParser(title=data_source.name)
            pages_processed = 0
            attempts_used = 0
            prompt_tokens = 0
            completion_tokens = 0

            for page_number, png_bytes in render_pdf_pages(
                local_path,
                dpi=settings.VISION_DPI,
                page_offset=settings.VISION_PAGE_OFFSET,
            ):
                extraction = await self._vision_client.extract_page(
                    png_bytes,
                    page_number=page_number,
                    state=parser.state,
                )
                parser.add_page(extraction.page, page_number=page_number)
                pages_processed += 1
                attempts_used += extraction.attempts
                prompt_tokens += extraction.prompt_tokens
                completion_tokens += extraction.completion_tokens

            if pages_processed == 0:
                raise ValueError(f"Document has no renderable pages: {local_filename}")

            # Bước 5: Build Knowledge Tree - Tạo cây phân cấp (in-memory)
            root = parser.finish()

            # Set parent IDs & positions cho toàn bộ cây
            set_parent_ids(root)
            set_positions(root)

            # Gán source_id + validate tree theo rule
            validate_tree_and_attach_source(root, data_source_id)

            # Bước 6: Calculate Statistics - Thống kê
            statistics = compute_tree_statistics(
                root,
                start_time=ingestion_start_time,
                extra={
                    "pages_processed": pages_processed,
                    "ocr_attempts": attempts_used,
                    "ocr_prompt_tokens": prompt_tokens,
                    "ocr_completion_tokens": completion_tokens,
                    "blocks_by_kind": dict(parser.blocks_by_kind),
                    "normalization": dict(parser.normalization),
                    "vision_model": self._vision_client.config.model,
                    "vision_endpoint_host": self._vision_client.config.host,
                },
            )

            # Bước 7: Update DataSource metadata (không persist tree vào DB)
            # ArangoClient sync operations - convert datetime to ISO string
            sys_handler.sources.update(
                data_source_id,
                {
                    "updated_at": datetime.now(UTC).isoformat(),
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
                root_node_dict = serialize_tree_to_dict(root) if root else None

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
                # Never downgrade an already ingested source
                if data_source_on_error and data_source_on_error.status != "active":
                    sys_handler.sources.update(
                        data_source_id,
                        {
                            "updated_at": datetime.now(UTC).isoformat(),
                            "status": "error",
                        },
                    )
                    data_source_on_error = (
                        sys_handler.sources.get_by_id(data_source_id) or data_source_on_error
                    )
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
            # Best-effort cleanup of temp workspace and HTTP connections
            try:
                await self._vision_client.aclose()
            except Exception:
                pass
            try:
                shutil.rmtree(job_dir, ignore_errors=True)
            except Exception:
                pass
