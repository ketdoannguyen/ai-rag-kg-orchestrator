"""Web ingestion workflow - Entry point for crawling and processing web content."""

from __future__ import annotations

import asyncio
import warnings
from datetime import datetime
from typing import Any

from loguru import logger
from arango_client import (
    ArangoClient,
    set_parent_ids,
    set_positions,
    SystemAdminHandler,
    SourceType,
)
from tenacity import retry, retry_if_exception_type, stop_after_attempt, wait_exponential

from workflows.dependencies import init_arango_client
from .tree_utils import (
    compute_tree_statistics,
    validate_tree_and_attach_source,
    serialize_tree_to_dict,
)
from .content_extractor import ContentExtractor
from .crawl4ai_client import Crawl4AIClient
from .markdown_to_json import MarkdownToJSONConverter
from .url_validation import validate_urls
from .markdown_processing import (
    assess_markdown_quality,
    find_content_start,
    extract_source_name,
    clean_markdown_content,
)
from .models import (
    WebIngestionInput,
    WebIngestionResult,
    WebIngestionSingleResult,
    WebIngestionStatus,
)


async def start_ingest_web(
    payload: WebIngestionInput | dict[str, Any],
    *,
    arango_client: ArangoClient | None = None,
) -> dict[str, Any]:
    """
    Entry point for the ingest_web workflow.

    Args:
        payload: Workflow input (Pydantic model or plain dict)
        arango_client: Optional ArangoDB client (defaults to settings)
    """
    input_payload = (
        payload if isinstance(payload, WebIngestionInput) else WebIngestionInput(**payload)
    )
    client = arango_client or init_arango_client()

    workflow = WebIngestionWorkflow(arango_client=client)
    result = await workflow.run(input_payload)
    if hasattr(result, "model_dump"):
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", UserWarning)
            return result.model_dump(mode="json")
    return result.dict()


class WebIngestionWorkflow:
    """Coordinates web ingestion, content extraction, and knowledge tree building."""

    def __init__(
        self,
        *,
        arango_client: ArangoClient,
    ) -> None:
        self._arango_client = arango_client
        self._db = arango_client.db
        self._logger = logger.bind(workflow="ingest_web")

    async def run(self, payload: WebIngestionInput) -> WebIngestionResult:
        """
        Run the ingestion workflow:

        1. Validate URLs (format, DNS, robots.txt)
        2. Process each URL (crawl, extract, build tree)
        3. For each URL:
           - Create DataSource
           - Crawl URL
           - Build Knowledge Tree
           - Validate Nodes
           - Calculate Statistics
           - Save DataSource to Postgres
        """
        self._logger.info("Starting web ingestion", total_urls=len(payload.urls))

        # Validate URLs and create results for skipped URLs
        validation_results = await validate_urls(payload.urls, payload.crawl_config)
        valid_urls = validation_results["valid"]
        skipped_info = validation_results["skipped"]

        # Convert skipped info to WebIngestionSingleResult
        skipped_results = [
            WebIngestionSingleResult(
                status=WebIngestionStatus.SKIPPED,
                data_source=None,
                root_node=None,
                statistics={"url": info["url"], "skip_reason": info["skip_reason"]},
                error_message=info["skip_reason"],
            )
            for info in skipped_info
        ]

        if not valid_urls and not skipped_results:
            raise ValueError("All URLs are invalid")

        crawler = Crawl4AIClient()
        content_extractor = ContentExtractor()
        json_converter = MarkdownToJSONConverter()

        url_semaphore = asyncio.Semaphore(payload.max_concurrent_urls)

        async def process_url(url: str) -> WebIngestionSingleResult:
            async with url_semaphore:
                try:
                    return await self._process_single_url_with_retry(
                        url=url,
                        payload=payload,
                        crawler=crawler,
                        content_extractor=content_extractor,
                        json_converter=json_converter,
                    )
                except Exception as exc:
                    self._logger.error(f"Failed to process URL {url} after retries: {exc}")
                    return WebIngestionSingleResult(
                        status=WebIngestionStatus.ERROR,
                        data_source=None,
                        root_node=None,
                        statistics=None,
                        error_message=str(exc),
                    )

        results = skipped_results.copy()

        if valid_urls:
            try:
                processed_results = await asyncio.gather(*(process_url(url) for url in valid_urls))
                results.extend(processed_results)
            finally:
                await crawler.close()

        succeeded = len([r for r in results if r.status == WebIngestionStatus.SUCCESS])
        failed = len([r for r in results if r.status == WebIngestionStatus.ERROR])
        skipped = len([r for r in results if r.status == WebIngestionStatus.SKIPPED])

        self._logger.info(
            "Web ingestion finished",
            total_urls=len(payload.urls),
            succeeded=succeeded,
            failed=failed,
            skipped=skipped,
        )

        return WebIngestionResult(results=results)

    async def _process_single_url_with_retry(
        self,
        *,
        url: str,
        payload: WebIngestionInput,
        crawler: Crawl4AIClient,
        content_extractor: ContentExtractor,
        json_converter: MarkdownToJSONConverter,
    ) -> WebIngestionSingleResult:
        """Process a single URL with retry logic."""
        data_source = await self._create_data_source(url, payload)

        @retry(
            reraise=True,
            stop=stop_after_attempt(payload.max_url_retries + 1),
            wait=wait_exponential(multiplier=1, min=1, max=8),
            retry=retry_if_exception_type((Exception,)),
        )
        async def _crawl_and_build_tree() -> WebIngestionSingleResult:
            """Retry only the crawl and tree building part."""
            try:
                return await asyncio.wait_for(
                    self._process_single_url(
                        url=url,
                        payload=payload,
                        data_source=data_source,
                        crawler=crawler,
                        content_extractor=content_extractor,
                        json_converter=json_converter,
                    ),
                    timeout=payload.per_url_timeout_seconds,
                )
            except asyncio.TimeoutError:
                self._logger.error(f"Timeout processing URL: {url}")
                raise
            except Exception as exc:
                self._logger.warning(f"Error processing URL {url}: {exc}")
                raise

        try:
            return await _crawl_and_build_tree()
        except Exception as exc:
            error_msg = str(exc)
            self._logger.error(f"All retries failed for URL {url}: {error_msg}")
            handler = SystemAdminHandler(self._db)
            handler.sources.update_status(data_source.id, "error")

            return WebIngestionSingleResult(
                status=WebIngestionStatus.ERROR,
                data_source=data_source,
                root_node=None,
                statistics=None,
                error_message=error_msg,
            )

    async def _process_single_url(
        self,
        *,
        url: str,
        payload: WebIngestionInput,
        data_source: Any,
        crawler: Crawl4AIClient,
        content_extractor: ContentExtractor,
        json_converter: MarkdownToJSONConverter,
    ) -> WebIngestionSingleResult:
        """
        Process a single URL:
        1. Crawl URL
        2. Build Knowledge Tree
        3. Validate Nodes
        4. Calculate Statistics
        5. Update DataSource to Postgres
        """
        crawl_start_time = datetime.utcnow()

        try:
            # Crawl URL
            crawl_result = await crawler.crawl_url(url)
            html = crawl_result["html"]
            crawler_markdown = crawl_result.get("markdown", "")

            # Extract content using ContentExtractor
            extracted_markdown = await asyncio.to_thread(
                content_extractor.extract_main_content, html, url
            )

            # Smart content selection: Compare crawler markdown vs extracted markdown
            crawler_markdown_clean = crawler_markdown.strip() if crawler_markdown else ""
            extracted_markdown_clean = extracted_markdown.strip() if extracted_markdown else ""

            # Assess quality of both sources
            crawler_quality, crawler_reason = assess_markdown_quality(crawler_markdown_clean)
            extracted_quality, extracted_reason = assess_markdown_quality(extracted_markdown_clean)

            self._logger.debug(
                f"Content quality comparison for {url}: "
                f"crawler={crawler_quality:.2f} ({crawler_reason}), "
                f"extracted={extracted_quality:.2f} ({extracted_reason})"
            )

            # Decision: Use the higher quality source
            if crawler_quality > extracted_quality and crawler_quality >= 0.3:
                self._logger.info(
                    f"Using crawler markdown (quality={crawler_quality:.2f}) over extracted "
                    f"(quality={extracted_quality:.2f}) for {url}"
                )
                markdown = crawler_markdown_clean
                markdown = find_content_start(markdown, crawl_result.get("title", ""))
            elif extracted_quality >= 0.3:
                markdown = extracted_markdown_clean
            elif len(crawler_markdown_clean) > len(extracted_markdown_clean):
                self._logger.warning(
                    f"Both sources have low quality, using longer one (crawler) for {url}"
                )
                markdown = crawler_markdown_clean
                markdown = find_content_start(markdown, crawl_result.get("title", ""))
            else:
                markdown = extracted_markdown_clean

            # Clean markdown: remove TOC, fix image duplicates, etc.
            markdown = clean_markdown_content(markdown)

            if not markdown.strip():
                raise ValueError(f"No content extracted from {url}")

            # Extract source_name from content
            source_name = extract_source_name(markdown, url)
            data_source.name = source_name

            # Build Knowledge Tree
            metadata = {
                "crawled_at": datetime.utcnow().isoformat(),
            }
            if payload.crawl_config:
                metadata["crawl_config"] = payload.crawl_config

            root_node = json_converter.convert_to_section(
                markdown=markdown,
                source_type=SourceType.WEB,
                url=url,
                metadata=metadata,
            )

            # Set parent_id and position for all nodes
            set_parent_ids(root_node)
            set_positions(root_node)

            # Validate Nodes
            validate_tree_and_attach_source(root_node, data_source.id)

            # Calculate Statistics
            statistics = compute_tree_statistics(
                root_node,
                start_time=crawl_start_time,
                duration_key="crawl_duration_seconds",
                extra={"url": url},
            )

            # Update DataSource
            updated_config = {
                "url": url,
                "crawl_config": payload.crawl_config,
                "crawl_initiated_at": crawl_start_time.isoformat(),
                "statistics": statistics,
            }

            handler = SystemAdminHandler(self._db)
            saved_data_source = handler.sources.update(
                data_source.id,
                {
                    "name": source_name,
                    "config": updated_config,
                },
            )

            self._logger.info(
                "Successfully processed URL",
                url=url,
                data_source_id=str(data_source.id),
                nodes_count=statistics.get("total_nodes", 0),
            )

            # Serialize tree using custom function for polymorphic children
            root_node_dict = serialize_tree_to_dict(root_node) if root_node else None

            return WebIngestionSingleResult(
                status=WebIngestionStatus.SUCCESS,
                data_source=saved_data_source or data_source,
                root_node=root_node_dict,
                statistics=statistics,
                error_message=None,
            )

        except Exception as exc:
            self._logger.error(f"Failed to process URL {url}: {exc}")
            raise

    async def _create_data_source(
        self,
        url: str,
        payload: WebIngestionInput,
    ) -> Any:
        """Create DataSource for web ingestion."""
        from arango_client import DataSource as DataSourceModel

        crawl_initiated_at = datetime.utcnow()

        config = {
            "urls": [url],
            "crawl_config": payload.crawl_config,
            "crawl_initiated_at": crawl_initiated_at.isoformat(),
        }

        data_source_model = DataSourceModel(
            name=url,  # Temporary, will be updated after crawling
            type=SourceType.WEB,
            config=config,
            hash_content=None,
            status="processing",
            created_by=payload.created_by,
        )

        handler = SystemAdminHandler(self._db)
        saved = handler.sources.create(data_source_model)

        self._logger.info("Created DataSource", data_source_id=str(saved.id), url=url)
        return saved


__all__ = ["start_ingest_web", "WebIngestionWorkflow"]
