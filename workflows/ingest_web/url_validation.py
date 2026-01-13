"""URL validation utilities for web ingestion."""

from __future__ import annotations

import re
import socket
from typing import Any
from urllib.parse import urlparse

import aiohttp
from loguru import logger


async def validate_urls(
    urls: list[str],
    crawl_config: dict[str, Any],
) -> dict[str, Any]:
    """
    Validate URLs for web ingestion.

    - Check format hợp lệ (regex: ^https?://)
    - Check domain có accessible không (DNS lookup)  
    - Check robots.txt nếu respect_robots_txt=true

    Args:
        urls: List of URLs to validate
        crawl_config: Configuration dict with respect_robots_txt flag
        
    Returns:
        dict with keys:
        - "valid": list[str] - Valid URLs
        - "skipped": list[dict] - URLs skipped with reason (url, skip_reason)
    """
    valid_urls: list[str] = []
    skipped_results: list[dict[str, Any]] = []
    respect_robots = crawl_config.get("respect_robots_txt", True)

    for url in urls:
        skip_reason: str | None = None

        # Check format
        if not re.match(r"^https?://", url):
            skip_reason = f"Invalid URL format: {url}"
            logger.warning(skip_reason)
        else:
            try:
                parsed = urlparse(url)
                domain = parsed.netloc

                # DNS lookup
                try:
                    socket.gethostbyname(domain)
                except socket.gaierror:
                    skip_reason = f"DNS lookup failed for {url}"
                    logger.warning(skip_reason)

                # Check robots.txt
                if not skip_reason and respect_robots:
                    if not await check_robots_txt(url, domain):
                        skip_reason = f"URL disallowed by robots.txt: {url}"
                        logger.warning(skip_reason)

                if not skip_reason:
                    valid_urls.append(url)
            except Exception as exc:
                skip_reason = f"Error validating URL {url}: {exc}"
                logger.warning(skip_reason)

        if skip_reason:
            skipped_results.append({
                "url": url,
                "skip_reason": skip_reason,
            })

    return {"valid": valid_urls, "skipped": skipped_results}


async def check_robots_txt(url: str, domain: str) -> bool:
    """Check if URL is allowed by robots.txt.
    
    Args:
        url: URL to check
        domain: Domain name for robots.txt lookup
        
    Returns:
        True if URL is allowed, False if disallowed
    """
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
        logger.warning(f"Error checking robots.txt for {url}: {exc}")
        # If we can't check robots.txt, allow crawling
        return True


__all__ = ["validate_urls", "check_robots_txt"]
