from __future__ import annotations

from typing import Any

from crawl4ai import AsyncWebCrawler
from loguru import logger


class Crawl4AIClient:
    """Wrapper around crawl4ai for web scraping."""

    def __init__(self) -> None:
        self._crawler: AsyncWebCrawler | None = None

    async def _get_crawler(self) -> AsyncWebCrawler:
        if self._crawler is None:
            self._crawler = AsyncWebCrawler()
        return self._crawler

    async def crawl_url(self, url: str) -> dict[str, Any]:
        """Crawl a URL and return the result."""
        try:
            crawler = await self._get_crawler()
            result = await crawler.arun(url=url)

            if not result.success:
                error_msg = getattr(result, "error_message", "Unknown error")
                raise ValueError(f"Failed to crawl {url}: {error_msg}")

            html = getattr(result, "html", "") or getattr(result, "cleaned_html", "")
            markdown = getattr(result, "markdown", "") or ""
            metadata = getattr(result, "metadata", {}) or {}

            return {
                "url": url,
                "html": html,
                "markdown": markdown,
                "metadata": metadata,
            }
        except Exception as e:
            logger.error(f"Error crawling {url}: {e}")
            raise

    async def close(self) -> None:
        """Close the crawler."""
        if self._crawler:
            try:
                await self._crawler.aclose()
            except Exception:
                pass
            self._crawler = None
