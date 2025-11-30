from __future__ import annotations

import asyncio
import json
import re
from datetime import datetime
from pathlib import Path
from typing import Any, Sequence

from config.settings import settings
from loguru import logger
from minio_client import StorageClient

from workflows.dependencies import get_minio_client
from postgres_client.models import SourceType
from postgres_client import set_parent_ids, set_positions
from .content_extractor import ContentExtractor
from .crawl4ai_client import Crawl4AIClient
from .image_handler import ImageHandler
from .markdown_to_json import MarkdownToJSONConverter
from .models import (
    WebIngestionBatchResult,
    WebIngestionInput,
    WebIngestionSingleResult,
    WebIngestionStatus,
    WebIngestionUrlInput,
)

# Local output directory for testing
LOCAL_OUTPUT_DIR = Path("data/ingest_web")
RETRY_DELAY_SECONDS = 2.0
MAX_FILENAME_LENGTH = 80
FORBIDDEN_FILENAME_CHARS = re.compile(r'[<>:"|?*\x00-\x1f]')


def start_ingest(
    payload: WebIngestionInput | Sequence[str],
    enable_ocr: bool | None = None,
) -> dict[str, Any]:
    """
    Entry point for the ingest_web workflow.

    Accepts either a prepared WebIngestionInput or a simple sequence of URLs.
    """

    if isinstance(payload, WebIngestionInput):
        if enable_ocr is not None:
            payload = payload.copy(update={"enable_ocr": enable_ocr})
        input_payload = payload
    else:
        if isinstance(payload, (str, bytes)):
            raise TypeError("payload must be WebIngestionInput or a sequence of URLs")
        url_models = [WebIngestionUrlInput(url=url) for url in payload]
        input_payload = WebIngestionInput(
            urls=url_models,
            enable_ocr=enable_ocr or False,
        )

    storage = get_minio_client()
    workflow = WebIngestionWorkflow(
        storage=storage,
        bucket=settings.MINIO_BUCKET,
        enable_ocr=input_payload.enable_ocr,
    )
    result = asyncio.run(workflow.run(input_payload))
    if hasattr(result, "model_dump"):
        return result.model_dump(mode="json")
    return result.dict()


class WebIngestionWorkflow:
    """Coordinates web ingestion, content extraction, and storage."""

    def __init__(
        self,
        *,
        storage: StorageClient,
        bucket: str,
        enable_ocr: bool = False,
    ) -> None:
        self._storage = storage
        self._bucket = bucket
        self._enable_ocr = enable_ocr
        self._date_prefix = datetime.utcnow().strftime("%d-%m-%Y")
        self._job_id = datetime.utcnow().strftime("%H:%M:%S-%f")[:-3]
        self._logger = logger.bind(job_id=self._job_id)

    async def run(self, payload: WebIngestionInput) -> WebIngestionBatchResult:
        """Run the ingestion workflow for multiple URLs."""
        await self._ensure_bucket_with_retry()

        crawler = Crawl4AIClient()
        content_extractor = ContentExtractor()
        image_handler = ImageHandler(self._storage, self._bucket)
        json_converter = MarkdownToJSONConverter()

        url_semaphore = asyncio.Semaphore(payload.max_concurrent_urls)
        image_semaphore = asyncio.Semaphore(payload.max_concurrent_images)

        async def process_entry(entry: WebIngestionUrlInput) -> WebIngestionSingleResult:
            async with url_semaphore:
                return await self._process_url_with_retry(
                    entry=entry,
                    payload=payload,
                    crawler=crawler,
                    content_extractor=content_extractor,
                    image_handler=image_handler,
                    json_converter=json_converter,
                    image_semaphore=image_semaphore,
                )

        self._logger.info(
            f"Starting ingestion for {len(payload.urls)} URL(s)",
            urls=len(payload.urls),
        )

        try:
            results = await asyncio.gather(*(process_entry(entry) for entry in payload.urls))
        finally:
            await crawler.close()

        succeeded = [r for r in results if r.status == WebIngestionStatus.COMPLETE]
        failed = [r for r in results if r.status == WebIngestionStatus.FAILED]

        if failed and succeeded:
            batch_status = WebIngestionStatus.PARTIAL
        elif failed:
            batch_status = WebIngestionStatus.FAILED
        else:
            batch_status = WebIngestionStatus.COMPLETE

        self._logger.info(
            "Ingestion finished",
            status=batch_status.value,
            succeeded=len(succeeded),
            failed=len(failed),
        )

        return WebIngestionBatchResult(
            status=batch_status,
            job_id=self._job_id,
            total_urls=len(results),
            succeeded=len(succeeded),
            failed=len(failed),
            results=results,
        )

    async def _process_url_with_retry(
        self,
        *,
        entry: WebIngestionUrlInput,
        payload: WebIngestionInput,
        crawler: Crawl4AIClient,
        content_extractor: ContentExtractor,
        image_handler: ImageHandler,
        json_converter: MarkdownToJSONConverter,
        image_semaphore: asyncio.Semaphore,
    ) -> WebIngestionSingleResult:
        url_str = str(entry.url)
        last_error: Exception | None = None

        for attempt in range(payload.max_url_retries + 1):
            try:
                return await asyncio.wait_for(
                    self._process_single_url(
                        url=url_str,
                        entry_metadata=entry.metadata,
                        crawler=crawler,
                        content_extractor=content_extractor,
                        image_handler=image_handler,
                        json_converter=json_converter,
                        image_semaphore=image_semaphore,
                    ),
                    timeout=payload.per_url_timeout_seconds,
                )
            except Exception as exc:
                last_error = exc
                self._logger.warning(
                    f"Attempt {attempt + 1} failed for {url_str}: {exc}",
                )
                if attempt < payload.max_url_retries:
                    await asyncio.sleep(RETRY_DELAY_SECONDS * (attempt + 1))

        return WebIngestionSingleResult(
            url=url_str,
            status=WebIngestionStatus.FAILED,
            job_id=self._job_id,
            error=str(last_error) if last_error else "Unknown error",
        )

    async def _process_single_url(
        self,
        *,
        url: str,
        entry_metadata: dict[str, Any] | None,
        crawler: Crawl4AIClient,
        content_extractor: ContentExtractor,
        image_handler: ImageHandler,
        json_converter: MarkdownToJSONConverter,
        image_semaphore: asyncio.Semaphore,
    ) -> WebIngestionSingleResult:
        """Process a single URL."""
        crawl_result = await crawler.crawl_url(url)
        html = crawl_result["html"]

        # Extract content in thread pool to avoid blocking event loop
        markdown = await asyncio.to_thread(content_extractor.extract_main_content, html, url)
        if not markdown:
            self._logger.warning(f"No content extracted from {url}, falling back to crawler markdown")
            markdown = crawl_result.get("markdown", "")

        image_urls = image_handler.extract_image_urls_from_markdown(markdown)
        image_mappings = await self._process_images(
            url=url,
            image_urls=image_urls,
            image_handler=image_handler,
            image_semaphore=image_semaphore,
        )

        if image_mappings:
            markdown = json_converter.replace_image_urls(markdown, image_mappings)

        metadata = {
            "crawled_at": datetime.utcnow().isoformat(),
            "job_id": self._job_id,
            "images_count": len(image_mappings),
        }
        if entry_metadata:
            metadata["source_metadata"] = entry_metadata

        root_section = json_converter.convert_to_section(
            markdown=markdown,
            source_type=SourceType.WEB,
            url=url,
            metadata=metadata,
        )
        # Set parent_id and position for all nodes in the tree before serializing
        set_parent_ids(root_section)
        set_positions(root_section)
        structured_json = root_section.model_dump(mode="json", exclude_none=False)

        minio_objects = await self._persist_outputs(
            url=url,
            markdown=markdown,
            structured_json=structured_json,
            html=html,
        )
        local_paths = await asyncio.to_thread(
            self._save_locally,
            url=url,
            markdown=markdown,
            structured_json=structured_json,
            html=html,
        )

        return WebIngestionSingleResult(
            url=url,
            status=WebIngestionStatus.COMPLETE,
            job_id=self._job_id,
            minio_objects=minio_objects,
            local_paths=local_paths,
            images_processed=len(image_mappings),
        )

    async def _process_images(
        self,
        *,
        url: str,
        image_urls: list[str],
        image_handler: ImageHandler,
        image_semaphore: asyncio.Semaphore,
    ) -> dict[str, str]:
        if not image_urls:
            return {}

        async def process(image_url: str) -> tuple[str, str]:
            async with image_semaphore:
                try:
                    image_info = await image_handler.download_and_upload_image(
                        image_url,
                        url,
                        self._job_id,
                    )
                    if image_info:
                        minio_path = image_info["minio_object"]
                        return image_url, f"[Image stored at: {minio_path}]"
                except Exception as exc:
                    self._logger.warning(f"Could not process image {image_url} from {url}: {exc}")
                return image_url, f"[Image: {image_url}]"

        pairs = await asyncio.gather(*(process(image_url) for image_url in image_urls))
        return {original: replacement for original, replacement in pairs}

    async def _ensure_bucket_with_retry(self) -> None:
        for attempt in range(3):
            try:
                await self._storage.ensure_bucket(self._bucket)
                return
            except Exception as exc:
                self._logger.warning(f"Bucket ensure failed (attempt {attempt + 1}): {exc}")
                await asyncio.sleep(RETRY_DELAY_SECONDS * (attempt + 1))
        self._logger.warning("Proceeding without confirming bucket availability")

    def _build_object_name(self, safe_name: str, extension: str) -> str:
        return f"web/{self._date_prefix}/{self._job_id}/{safe_name}.{extension}"

    async def _upload_content(
        self,
        *,
        safe_name: str,
        content: str,
        content_type: str,
        extension: str,
    ) -> str:
        object_name = self._build_object_name(safe_name, extension)
        await self._storage.upload_bytes(
            bucket_name=self._bucket,
            object_name=object_name,
            data=content.encode("utf-8"),
            content_type=content_type,
        )
        return object_name

    async def _persist_outputs(
        self,
        *,
        url: str,
        markdown: str,
        structured_json: dict[str, Any],
        html: str,
    ) -> dict[str, str]:
        """Persist outputs to MinIO. Allows partial failures."""
        safe_name = self._safe_filename(url)
        outputs: dict[str, str] = {}
        json_payload = json.dumps(structured_json, indent=2, ensure_ascii=False)
        upload_plan = (
            ("html", html, "text/html", "html"),
            ("markdown", markdown, "text/markdown", "md"),
            ("json", json_payload, "application/json", "json"),
        )

        for key, content, content_type, extension in upload_plan:
            try:
                outputs[key] = await self._upload_content(
                    safe_name=safe_name,
                    content=content,
                    content_type=content_type,
                    extension=extension,
                )
            except Exception as exc:
                self._logger.warning(f"Failed to upload {key} for {url}: {exc}")
                outputs[key] = "not_saved"

        return outputs

    def _save_locally(
        self,
        *,
        url: str,
        markdown: str,
        structured_json: dict[str, Any],
        html: str,
    ) -> dict[str, str]:
        """Save outputs locally for testing."""
        job_dir = LOCAL_OUTPUT_DIR / self._date_prefix / self._job_id
        job_dir.mkdir(parents=True, exist_ok=True)

        safe_name = self._safe_filename(url)
        paths: dict[str, str] = {}

        try:
            html_path = job_dir / f"{safe_name}.html"
            html_path.write_text(html, encoding="utf-8")
            paths["html"] = str(html_path)

            markdown_path = job_dir / f"{safe_name}.md"
            markdown_path.write_text(markdown, encoding="utf-8")
            paths["markdown"] = str(markdown_path)

            json_path = job_dir / f"{safe_name}.json"
            json_path.write_text(
                json.dumps(structured_json, indent=2, ensure_ascii=False),
                encoding="utf-8",
            )
            paths["json"] = str(json_path)
        except Exception as exc:
            self._logger.warning(f"Could not save outputs locally for {url}: {exc}")

        return paths

    @staticmethod
    def _safe_filename(url: str) -> str:
        filename = re.sub(r"^https?://", "", url)
        filename = filename.split("?")[0]
        filename = filename.replace("/", "_").replace("\\", "_")
        filename = FORBIDDEN_FILENAME_CHARS.sub("", filename)
        filename = filename[:MAX_FILENAME_LENGTH].rstrip(". ")
        return filename or "document"
