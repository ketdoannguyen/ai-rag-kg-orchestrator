from __future__ import annotations

import asyncio
import hashlib
import json
import shutil
from datetime import datetime
from pathlib import Path
from typing import Any
from uuid import UUID

from config.settings import settings
from loguru import logger
from postgres_client import PostgresClient, set_parent_ids, set_positions
from postgres_client.handlers.system_admin_handler import SystemAdminHandler
from postgres_client.models.knowledge import KnowledgeNode
from postgres_client.models.structure_nodes import StructureNode
from postgres_client.models.enums import NodeType
from typing import TYPE_CHECKING

from workflows.dependencies import get_minio_client, init_postgres_client
from workflows.process_raw_json.parser import TextbookParser
from .marker_client import MarkerClient
from .models import FileIngestionInput, FileIngestionResult, FileIngestionStatus

if TYPE_CHECKING:
    from minio_client import StorageClient  # type: ignore[import-not-found]


def _calculate_md5(data: bytes) -> str:
    digest = hashlib.md5()
    digest.update(data)
    return digest.hexdigest()


def start_ingest(
    payload: FileIngestionInput | dict[str, Any] | UUID | str,
    *,
    storage: StorageClient | None = None,
    pg_client: PostgresClient | None = None,
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
        input_payload = FileIngestionInput(data_source_id=UUID(str(payload)))
    else:
        input_payload = FileIngestionInput(**payload)

    marker_client = MarkerClient(
        use_llm=settings.MARKER_USE_LLM,
        force_ocr=input_payload.enable_ocr or settings.MARKER_FORCE_OCR,
        torch_device=settings.MARKER_TORCH_DEVICE,
        google_api_key=settings.GOOGLE_API_KEY,
    )

    storage_client = storage or get_minio_client()
    postgres_client = pg_client or init_postgres_client()

    workflow = FileIngestionWorkflow(
        storage=storage_client,
        workspace_root=Path(settings.INGEST_FILE_WORKDIR),
        marker_client=marker_client,
        pg_client=postgres_client,
    )

    result = asyncio.run(workflow.run(input_payload))
    if hasattr(result, "model_dump"):
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
        pg_client: PostgresClient,
    ) -> None:
        self._storage = storage
        self._workspace_root = workspace_root
        self._marker_client = marker_client
        self._pg_client = pg_client
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
        job_id = datetime.utcnow().strftime("%H-%M-%S")
        self._logger = logger.bind(job_id=job_id, workflow="ingest_file")

        self._workspace_root.mkdir(parents=True, exist_ok=True)
        job_dir = self._workspace_root / job_id
        job_dir.mkdir(parents=True, exist_ok=True)

        data_source_id = payload.data_source_id

        try:
            async with self._pg_client.sessionmaker() as session:
                sys_handler = SystemAdminHandler(session)

                # Bước 1: Load DataSource - Lấy thông tin nguồn
                data_source = await sys_handler.sources.get_by_id(data_source_id)
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
                calculated_hash = _calculate_md5(file_bytes)
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
                marker_result = await asyncio.to_thread(
                    self._marker_client.convert_pdf, local_path
                )
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
                self._validate_and_attach_source(parser.root, data_source_id)

                # Bước 7: Update DataSource metadata (không persist tree vào DB)
                await sys_handler.sources.update(
                    data_source_id,
                    {
                        "updated_at": datetime.utcnow(),
                        "status": "active",
                    },
                )

                self._logger.info(
                    "File ingestion completed successfully (in-memory tree only)",
                    data_source_id=str(data_source_id),
                )

                return FileIngestionResult(
                    status=FileIngestionStatus.COMPLETE,
                    job_id=job_id,
                    data_source_id=data_source_id,
                    root_node=parser.root,
                )
        except Exception as exc:
            self._logger.exception("File ingestion failed")
            return FileIngestionResult(
                status=FileIngestionStatus.FAILED,
                job_id=job_id,
                data_source_id=data_source_id,
                root_node=None,
                error=str(exc),
            )
        finally:
            # Best-effort cleanup of temp workspace
            try:
                shutil.rmtree(job_dir, ignore_errors=True)
            except Exception:
                pass

    def _validate_and_attach_source(
        self,
        root: StructureNode,
        data_source_id: UUID,
    ) -> None:
        """
        Bước 6: Validate Nodes - Kiểm tra tính hợp lệ và gán source_id.

        Rules:
        - Mỗi node (trừ root) PHẢI có parent_id
        - source_id PHẢI match với DataSource
        - level_down và level_up PHẢI hợp lệ (>= 1)
        - Trên mỗi path từ root đến detail: level_down + level_up phải là constant
        - Detail nodes (PARAGRAPH, IMAGE, TABLE) PHẢI có attributes.text/url/headers không empty (nếu tồn tại)
        """
        errors: list[str] = []
        # Track path constants for validation
        path_constants: dict[tuple, int] = {}  # (path_signature) -> constant value

        def walk(node: KnowledgeNode, parent: KnowledgeNode | None, path: list[UUID] = None) -> None:
            if path is None:
                path = []
            
            # Attach source_id
            node.source_id = data_source_id

            # Parent rule
            if parent is None:
                if node.parent_id is not None:
                    errors.append("Root node must have parent_id=None")
            else:
                if node.parent_id is None:
                    errors.append("Non-root node must have parent_id set")
                elif node.parent_id != parent.id:
                    errors.append(f"Node {node.id} has parent_id={node.parent_id} but parent is {parent.id}")

            # Level validation
            if node.level_down < 0:
                errors.append(f"Node {node.id} has invalid level_down={node.level_down}, must be >= 0")
            if node.level_up < 0:
                errors.append(f"Node {node.id} has invalid level_up={node.level_up}, must be >= 0")

            # Validate level_down + level_up constant along path
            current_path = tuple(path + [node.id])
            constant = node.level_down + node.level_up
            
            if current_path in path_constants:
                expected_constant = path_constants[current_path]
                if constant != expected_constant:
                    errors.append(
                        f"Node {node.id} on path has level_down+level_up={constant}, "
                        f"but expected {expected_constant} (path constant violation)"
                    )
            else:
                # First node on this path - record the constant
                path_constants[current_path] = constant

            # Validate detail node content
            if node.node_type in [NodeType.PARAGRAPH, NodeType.IMAGE, NodeType.TABLE]:
                attrs = getattr(node, "attributes", None)
                if attrs is not None:
                    if isinstance(attrs, dict):
                        # Check for empty text/url/headers based on node type
                        if node.node_type == NodeType.PARAGRAPH:
                            text = attrs.get("text", "")
                            if not text or not str(text).strip():
                                errors.append(f"Paragraph node {node.id} has empty text")
                        elif node.node_type == NodeType.IMAGE:
                            url = attrs.get("url") or attrs.get("file_path")
                            if not url:
                                errors.append(f"Image node {node.id} has no url or file_path")
                        elif node.node_type == NodeType.TABLE:
                            headers = attrs.get("headers", [])
                            rows = attrs.get("rows", [])
                            if not headers and not rows:
                                errors.append(f"Table node {node.id} has no headers or rows")
                    elif hasattr(attrs, "text"):
                        # Pydantic model
                        if node.node_type == NodeType.PARAGRAPH and not attrs.text.strip():
                            errors.append(f"Paragraph node {node.id} has empty text")
                    elif hasattr(attrs, "url") or hasattr(attrs, "file_path"):
                        if node.node_type == NodeType.IMAGE:
                            url = getattr(attrs, "url", None) or getattr(attrs, "file_path", None)
                            if not url:
                                errors.append(f"Image node {node.id} has no url or file_path")

            # Recursively validate children
            for child in node.children:
                walk(child, node, list(current_path))

        walk(root, None)

        if errors:
            raise ValueError(
                "Knowledge tree validation failed: " + "; ".join(sorted(set(errors)))
            )
