from __future__ import annotations

import asyncio
import json
from datetime import datetime
from pathlib import Path
from typing import Any

from config.settings import settings
from loguru import logger
from storage_client import StorageClient

from workflows.models import SourceType

from ._dependencies import get_storage_client
from .content_extractor import ContentExtractor
from .crawl4ai_client import Crawl4AIClient
from .image_handler import ImageHandler
from .markdown_to_json import MarkdownToJSONConverter

# Local output directory for testing
LOCAL_OUTPUT_DIR = Path("data/ingest_web")


def start_ingest(urls: list[str], enable_ocr: bool = False) -> dict[str, Any]:
    """Entry point for the ingest_web workflow."""
    storage = get_storage_client()
    workflow = WebIngestionWorkflow(
        storage=storage,
        bucket=settings.MINIO_BUCKET,
        enable_ocr=enable_ocr,
    )
    return asyncio.run(workflow.run(urls))


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

    async def run(self, urls: list[str]) -> dict[str, Any]:
        """Run the ingestion workflow for multiple URLs."""
        if not urls:
            raise ValueError("At least one URL is required")

        try:
            self._storage.ensure_bucket(self._bucket)
        except Exception as e:
            logger.warning(f"Could not connect to MinIO: {e}. Continuing without storage...")

        crawler = Crawl4AIClient()
        content_extractor = ContentExtractor()
        image_handler = ImageHandler(self._storage, self._bucket)
        json_converter = MarkdownToJSONConverter()

        results = []

        try:
            for url in urls:
                logger.info(f"Processing URL: {url}")
                result = await self._process_single_url(
                    url=url,
                    crawler=crawler,
                    content_extractor=content_extractor,
                    image_handler=image_handler,
                    json_converter=json_converter,
                )
                results.append(result)
        finally:
            await crawler.close()

        # Combine results if multiple URLs
        if len(results) == 1:
            return results[0]

        return {
            "status": "complete",
            "job_id": self._job_id,
            "urls_processed": len(results),
            "results": results,
        }

    async def _process_single_url(
        self,
        url: str,
        crawler: Crawl4AIClient,
        content_extractor: ContentExtractor,
        image_handler: ImageHandler,
        json_converter: MarkdownToJSONConverter,
    ) -> dict[str, Any]:
        """Process a single URL."""
        # Step 1: Crawl the URL
        crawl_result = await crawler.crawl_url(url)
        html = crawl_result["html"]

        # Step 2: Extract main content (remove header/footer/ads)
        markdown = content_extractor.extract_main_content(html, url)
        if not markdown:
            logger.warning(f"No content extracted from {url}")
            markdown = crawl_result.get("markdown", "")

        # Step 3: Extract and handle images
        image_urls = image_handler.extract_image_urls_from_markdown(markdown)
        image_mappings: dict[str, str] = {}

        for image_url in image_urls:
            try:
                image_info = image_handler.download_and_upload_image(
                    image_url, url, self._job_id
                )
                if image_info:
                    # For now, replace with MinIO path (OCR will be added later)
                    minio_path = image_info["minio_object"]
                    image_mappings[image_url] = f"[Image stored at: {minio_path}]"
            except Exception as e:
                logger.warning(f"Could not process image {image_url}: {e}")
                # Keep original image URL if processing fails
                image_mappings[image_url] = f"[Image: {image_url}]"

        # Step 4: Replace image URLs in markdown
        if image_mappings:
            markdown = json_converter.replace_image_urls(markdown, image_mappings)

        # Step 5: Convert markdown to structured Section
        root_section = json_converter.convert_to_section(
            markdown=markdown,
            source_type=SourceType.WEB,
            url=url,
            metadata={
                "crawled_at": datetime.utcnow().isoformat(),
                "job_id": self._job_id,
                "images_count": len(image_mappings),
            },
        )

        # Convert to dict for storage
        structured_json = root_section.to_dict()

        # Step 6: Save to MinIO and locally
        minio_objects = self._persist_outputs(
            url=url,
            markdown=markdown,
            structured_json=structured_json,
            html=html,
        )

        # Also save locally for testing
        local_paths = self._save_locally(
            url=url,
            markdown=markdown,
            structured_json=structured_json,
            html=html,
        )

        return {
            "status": "complete",
            "url": url,
            "job_id": self._job_id,
            "output_json_path": local_paths.get("json") or minio_objects.get("json", "not_saved"),
            "minio_objects": minio_objects,
            "local_paths": local_paths,
            "images_processed": len(image_mappings),
        }

    def _persist_outputs(
        self,
        url: str,
        markdown: str,
        structured_json: dict[str, Any],
        html: str,
    ) -> dict[str, str]:
        """Persist outputs to MinIO."""
        # Create safe filename from URL
        url_hash = url.replace("https://", "").replace("http://", "").replace("/", "_")[:50]

        outputs = {}

        try:
            # Save HTML
            html_object = f"web/{self._date_prefix}/{self._job_id}/{url_hash}.html"
            self._storage.upload_bytes(
                bucket_name=self._bucket,
                object_name=html_object,
                data=html.encode("utf-8"),
                content_type="text/html",
            )
            outputs["html"] = html_object

            # Save markdown
            markdown_object = f"web/{self._date_prefix}/{self._job_id}/{url_hash}.md"
            self._storage.upload_bytes(
                bucket_name=self._bucket,
                object_name=markdown_object,
                data=markdown.encode("utf-8"),
                content_type="text/markdown",
            )
            outputs["markdown"] = markdown_object

            # Save structured JSON
            json_object = f"web/{self._date_prefix}/{self._job_id}/{url_hash}.json"
            json_str = json.dumps(structured_json, indent=2, ensure_ascii=False)
            self._storage.upload_bytes(
                bucket_name=self._bucket,
                object_name=json_object,
                data=json_str.encode("utf-8"),
                content_type="application/json",
            )
            outputs["json"] = json_object

            logger.info(f"Saved outputs for {url} to MinIO")
        except Exception as e:
            logger.warning(f"Could not save to MinIO: {e}. Outputs available in result only.")
            # Return local paths or indicate not saved
            outputs["html"] = "not_saved"
            outputs["markdown"] = "not_saved"
            outputs["json"] = "not_saved"

        return outputs

    def _save_locally(
        self,
        url: str,
        markdown: str,
        structured_json: dict[str, Any],
        html: str,
    ) -> dict[str, str]:
        """Save outputs locally for testing."""
        LOCAL_OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
        job_dir = LOCAL_OUTPUT_DIR / self._date_prefix / self._job_id
        job_dir.mkdir(parents=True, exist_ok=True)

        # Create safe filename from URL
        url_hash = url.replace("https://", "").replace("http://", "").replace("/", "_")[:50]

        paths = {}

        # Save HTML
        html_path = job_dir / f"{url_hash}.html"
        html_path.write_text(html, encoding="utf-8")
        paths["html"] = str(html_path)

        # Save markdown
        markdown_path = job_dir / f"{url_hash}.md"
        markdown_path.write_text(markdown, encoding="utf-8")
        paths["markdown"] = str(markdown_path)

        # Save structured JSON
        json_path = job_dir / f"{url_hash}.json"
        json_path.write_text(
            json.dumps(structured_json, indent=2, ensure_ascii=False), encoding="utf-8"
        )
        paths["json"] = str(json_path)

        logger.info(f"Saved outputs locally to: {job_dir}")
        return paths


