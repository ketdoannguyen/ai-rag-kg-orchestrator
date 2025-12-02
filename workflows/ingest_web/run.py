from __future__ import annotations

import asyncio
import re
import socket
import warnings
from datetime import datetime
from typing import Any
from urllib.parse import urlparse
from uuid import UUID

import aiohttp
from loguru import logger
from postgres_client import PostgresClient, set_parent_ids, set_positions
from postgres_client.handlers.system_admin_handler import SystemAdminHandler
from postgres_client.models.enums import NodeType, SourceType
from postgres_client.models.knowledge import KnowledgeNode
from postgres_client.models.structure_nodes import StructureNode
from tenacity import retry, retry_if_exception_type, stop_after_attempt, wait_exponential

from workflows.dependencies import init_postgres_client
from .content_extractor import ContentExtractor
from .crawl4ai_client import Crawl4AIClient
from .markdown_to_json import MarkdownToJSONConverter

# Note: image_handler.py is kept in the codebase for future use but is not currently imported
# or used in this workflow. It will be integrated when image processing functionality is needed.
from .models import (
    WebIngestionInput,
    WebIngestionResult,
    WebIngestionSingleResult,
    WebIngestionStatus,
)


async def start_ingest_web(
    payload: WebIngestionInput | dict[str, Any],
    *,
    pg_client: PostgresClient | None = None,
) -> dict[str, Any]:
    """
    Entry point for the ingest_web workflow.

    Args:
        payload: Workflow input (Pydantic model or plain dict)
        pg_client: Optional Postgres client (defaults to settings.DATABASE_URL)
    """
    input_payload = payload if isinstance(payload, WebIngestionInput) else WebIngestionInput(**payload)
    postgres_client = pg_client or init_postgres_client()

    workflow = WebIngestionWorkflow(pg_client=postgres_client)
    result = await workflow.run(input_payload)
    if hasattr(result, "model_dump"):
        # Suppress Pydantic serialization warnings for nested models
        # These warnings occur because attributes/metadata are Pydantic models in derived classes
        # but are correctly serialized to dict by model_dump(mode="json")
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", UserWarning)
            return result.model_dump(mode="json")
    return result.dict()


class WebIngestionWorkflow:
    """Coordinates web ingestion, content extraction, and knowledge tree building."""

    def __init__(
        self,
        *,
        pg_client: PostgresClient,
    ) -> None:
        self._pg_client = pg_client
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

        # Bước 2: Validate URLs và tạo results cho skipped URLs
        validation_results = await self._validate_urls(payload.urls, payload.crawl_config)
        valid_urls = validation_results["valid"]
        skipped_results = validation_results["skipped"]

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
                    # Return error result instead of raising
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

    async def _validate_urls(
        self,
        urls: list[str],
        crawl_config: dict[str, Any],
    ) -> dict[str, Any]:
        """
        Bước 2: Validate URLs - Kiểm tra URLs.

        - Check format hợp lệ (regex: ^https?://)
        - Check domain có accessible không (DNS lookup)
        - Check robots.txt nếu respect_robots_txt=true

        Returns:
            dict với keys:
            - "valid": list[str] - URLs hợp lệ
            - "skipped": list[WebIngestionSingleResult] - URLs bị skip với reason
        """
        valid_urls: list[str] = []
        skipped_results: list[WebIngestionSingleResult] = []
        respect_robots = crawl_config.get("respect_robots_txt", True)

        for url in urls:
            skip_reason: str | None = None

            # Check format
            if not re.match(r"^https?://", url):
                skip_reason = f"Invalid URL format: {url}"
                self._logger.warning(skip_reason)
            else:
                try:
                    parsed = urlparse(url)
                    domain = parsed.netloc

                    # DNS lookup
                    try:
                        socket.gethostbyname(domain)
                    except socket.gaierror:
                        skip_reason = f"DNS lookup failed for {url}"
                        self._logger.warning(skip_reason)

                    # Check robots.txt
                    if not skip_reason and respect_robots:
                        if not await self._check_robots_txt(url, domain):
                            skip_reason = f"URL disallowed by robots.txt: {url}"
                            self._logger.warning(skip_reason)

                    if not skip_reason:
                        valid_urls.append(url)
                except Exception as exc:
                    skip_reason = f"Error validating URL {url}: {exc}"
                    self._logger.warning(skip_reason)

            if skip_reason:
                skipped_results.append(
                    WebIngestionSingleResult(
                        status=WebIngestionStatus.SKIPPED,
                        data_source=None,
                        root_node=None,
                        statistics={"url": url, "skip_reason": skip_reason},
                        error_message=skip_reason,
                    )
                )

        return {"valid": valid_urls, "skipped": skipped_results}

    async def _check_robots_txt(self, url: str, domain: str) -> bool:
        """Check if URL is allowed by robots.txt."""
        try:
            robots_url = f"https://{domain}/robots.txt"
            async with aiohttp.ClientSession() as session:
                async with session.get(robots_url, timeout=aiohttp.ClientTimeout(total=5)) as resp:
                    if resp.status != 200:
                        # If robots.txt doesn't exist, allow crawling
                        return True

                    robots_content = await resp.text()
                    # Simple robots.txt parser
                    user_agent = "*"
                    disallowed_paths: list[str] = []

                    for line in robots_content.split("\n"):
                        line = line.strip()
                        if not line or line.startswith("#"):
                            continue

                        if line.lower().startswith("user-agent:"):
                            user_agent = line.split(":", 1)[1].strip()
                        elif line.lower().startswith("disallow:") and user_agent == "*":
                            path = line.split(":", 1)[1].strip()
                            if path:
                                disallowed_paths.append(path)

                    # Check if URL path matches any disallowed path
                    parsed = urlparse(url)
                    url_path = parsed.path

                    for disallowed in disallowed_paths:
                        if url_path.startswith(disallowed):
                            return False

                    return True
        except Exception as exc:
            self._logger.warning(f"Error checking robots.txt for {url}: {exc}")
            # If we can't check robots.txt, allow crawling
            return True

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
        # Create DataSource once before retry
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
            # Update DataSource status to error if all retries failed
            error_msg = str(exc)
            self._logger.error(f"All retries failed for URL {url}: {error_msg}")
            async with self._pg_client.sessionmaker() as session:
                handler = SystemAdminHandler(session)
                await handler.sources.update_status(data_source.id, "error")

            # Return error result instead of raising
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
            # Bước 3: Crawl URL
            crawl_result = await crawler.crawl_url(url)
            html = crawl_result["html"]

            # Extract content
            markdown = await asyncio.to_thread(content_extractor.extract_main_content, html, url)
            if not markdown:
                self._logger.warning(f"No content extracted from {url}, falling back to crawler markdown")
                markdown = crawl_result.get("markdown", "")

            if not markdown.strip():
                raise ValueError(f"No content extracted from {url}")

            # Extract source_name from content (title or first heading)
            source_name = self._extract_source_name(markdown, url)

            # Update DataSource name
            data_source.name = source_name

            # Bước 4: Build Knowledge Tree
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

            # Set parent_id and position for all nodes in the tree
            set_parent_ids(root_node)
            set_positions(root_node)

            # Bước 5: Validate Nodes
            self._validate_and_attach_source(root_node, data_source.id)

            # Bước 9: Calculate Statistics
            statistics = self._calculate_statistics(root_node, url, crawl_start_time)

            # Bước 8: Update DataSource with config and save
            updated_config = {
                "url": url,
                "crawl_config": payload.crawl_config,
                "crawl_initiated_at": crawl_start_time.isoformat(),
                "statistics": statistics,
            }

            async with self._pg_client.sessionmaker() as session:
                handler = SystemAdminHandler(session)
                saved_data_source = await handler.sources.update(data_source.id, {
                    "name": source_name,
                    "config": updated_config,
                })

            self._logger.info(
                "Successfully processed URL",
                url=url,
                data_source_id=str(data_source.id),
                nodes_count=statistics.get("total_nodes", 0),
            )

            # Convert root_node to dict to avoid Pydantic serialization warnings
            # This ensures attributes/metadata Pydantic models are properly serialized
            # Suppress Pydantic serialization warnings for nested models (they are correctly converted to dict)
            with warnings.catch_warnings():
                warnings.simplefilter("ignore", UserWarning)
                root_node_dict = root_node.model_dump(mode="json") if root_node else None

            return WebIngestionSingleResult(
                status=WebIngestionStatus.SUCCESS,
                data_source=saved_data_source or data_source,
                root_node=root_node_dict,
                statistics=statistics,
                error_message=None,
            )

        except Exception as exc:
            self._logger.error(f"Failed to process URL {url}: {exc}")
            # Re-raise to trigger retry (status will be updated in _process_single_url_with_retry)
            raise

    async def _create_data_source(
        self,
        url: str,
        payload: WebIngestionInput,
    ) -> Any:
        """Bước 1: Create DataSource - Tạo nguồn web."""
        from postgres_client.models.system_admin import DataSource as DataSourceModel

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

        async with self._pg_client.sessionmaker() as session:
            handler = SystemAdminHandler(session)
            saved = await handler.sources.create(data_source_model)

        self._logger.info("Created DataSource", data_source_id=str(saved.id), url=url)
        return saved

    def _extract_source_name(self, markdown: str, url: str) -> str:
        """Extract source name from markdown (title or first heading)."""
        lines = markdown.split("\n")
        for line in lines:
            line = line.strip()
            if not line:
                continue
            # Check for H1
            h1_match = re.match(r"^#\s+(.+)$", line)
            if h1_match:
                title = h1_match.group(1).strip()
                if title and len(title) > 5:
                    return title[:255]  # Max length for name field

        # Fallback: use domain name
        try:
            parsed = urlparse(url)
            domain = parsed.netloc.replace("www.", "")
            return f"Web: {domain}"
        except Exception:
            return "Web Source"

    def _validate_and_attach_source(
        self,
        root: StructureNode,
        data_source_id: UUID,
    ) -> None:
        """
        Bước 5: Validate Nodes - Kiểm tra tính hợp lệ và gán source_id.

        Rules:
        - Mỗi node (trừ root) PHẢI có parent_id
        - source_id PHẢI match với DataSource
        - level_down và level_up PHẢI hợp lệ (>= 1)
        - Trên mỗi path từ root đến detail: level_down + level_up phải là constant
        - Detail nodes (PARAGRAPH, IMAGE, TABLE) PHẢI có attributes.text/url/headers không empty
        """
        errors: list[str] = []
        path_constants: dict[tuple, int] = {}

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
                path_constants[current_path] = constant

            # Validate detail node content
            if node.node_type in [NodeType.PARAGRAPH, NodeType.IMAGE, NodeType.TABLE]:
                attrs = getattr(node, "attributes", None)
                if attrs is not None:
                    if isinstance(attrs, dict):
                        if node.node_type == NodeType.PARAGRAPH:
                            text = attrs.get("text", "")
                            if not text or not str(text).strip():
                                errors.append(f"Paragraph node {node.id} has empty text")
                        elif node.node_type == NodeType.IMAGE:
                            url_attr = attrs.get("url") or attrs.get("file_path")
                            if not url_attr:
                                errors.append(f"Image node {node.id} has no url or file_path")
                        elif node.node_type == NodeType.TABLE:
                            headers = attrs.get("headers", [])
                            rows = attrs.get("rows", [])
                            if not headers and not rows:
                                errors.append(f"Table node {node.id} has no headers or rows")
                    elif hasattr(attrs, "text"):
                        if node.node_type == NodeType.PARAGRAPH and not attrs.text.strip():
                            errors.append(f"Paragraph node {node.id} has empty text")
                    elif hasattr(attrs, "url") or hasattr(attrs, "file_path"):
                        if node.node_type == NodeType.IMAGE:
                            url_attr = getattr(attrs, "url", None) or getattr(attrs, "file_path", None)
                            if not url_attr:
                                errors.append(f"Image node {node.id} has no url or file_path")

            # Recursively validate children
            for child in node.children:
                walk(child, node, list(current_path))

        walk(root, None)

        if errors:
            raise ValueError("Knowledge tree validation failed: " + "; ".join(sorted(set(errors))))

    def _calculate_statistics(
        self,
        root_node: KnowledgeNode,
        url: str,
        crawl_start_time: datetime,
    ) -> dict[str, Any]:
        """
        Bước 9: Calculate Statistics - Thống kê.

        - Count nodes by level
        - Count successful vs failed URLs
        - Calculate crawl duration
        - Calculate average content length
        """
        nodes_by_level: dict[int, int] = {}
        total_nodes = 0
        total_paragraphs = 0
        total_content_length = 0

        def walk(node: KnowledgeNode) -> None:
            nonlocal total_nodes, total_paragraphs, total_content_length

            total_nodes += 1
            level = node.level_down
            nodes_by_level[level] = nodes_by_level.get(level, 0) + 1

            if node.node_type == NodeType.PARAGRAPH:
                total_paragraphs += 1
                attrs = getattr(node, "attributes", None)
                if attrs:
                    if isinstance(attrs, dict):
                        text = attrs.get("text", "")
                    elif hasattr(attrs, "text"):
                        text = attrs.text
                    else:
                        text = ""
                    total_content_length += len(str(text))

            for child in node.children:
                walk(child)

        walk(root_node)

        crawl_duration = (datetime.utcnow() - crawl_start_time).total_seconds()
        avg_content_length = total_content_length / total_paragraphs if total_paragraphs > 0 else 0

        return {
            "nodes_by_level": nodes_by_level,
            "total_nodes": total_nodes,
            "total_paragraphs": total_paragraphs,
            "crawl_duration_seconds": crawl_duration,
            "average_content_length": avg_content_length,
            "url": url,
            "successful": True,
        }


__all__ = ["start_ingest_web", "WebIngestionWorkflow"]
