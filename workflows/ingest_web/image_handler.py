from __future__ import annotations

import asyncio
import hashlib
import mimetypes
from pathlib import Path
from typing import Any
from urllib.parse import urljoin, urlparse

import requests
from loguru import logger
from minio_client import StorageClient


class ImageHandler:
    """Handle image downloading and uploading to MinIO."""

    def __init__(self, storage: StorageClient, bucket: str) -> None:
        self._storage = storage
        self._bucket = bucket

    async def download_and_upload_image(
        self, image_url: str, base_url: str, job_id: str
    ) -> dict[str, Any] | None:
        """Download image from URL and upload to MinIO."""
        try:
            # Resolve relative URLs
            if not image_url.startswith(("http://", "https://")):
                image_url = urljoin(base_url, image_url)

            # Download image in a worker thread because requests is blocking.
            response = await asyncio.to_thread(
                requests.get, image_url, timeout=30, stream=True
            )
            response.raise_for_status()

            # Determine content type
            content_type = response.headers.get("content-type")
            if not content_type:
                content_type, _ = mimetypes.guess_type(image_url)
            if not content_type:
                content_type = "image/jpeg"

            # Generate filename
            parsed_url = urlparse(image_url)
            filename = Path(parsed_url.path).name or "image.jpg"
            if not filename or "." not in filename:
                ext = mimetypes.guess_extension(content_type) or ".jpg"
                filename = f"image{ext}"

            # Create unique filename with hash
            url_hash = hashlib.md5(image_url.encode()).hexdigest()[:8]
            name_part = Path(filename).stem
            ext_part = Path(filename).suffix
            unique_filename = f"{name_part}_{url_hash}{ext_part}"

            # Upload to MinIO
            image_data = response.content
            object_name = f"web/{job_id}/images/{unique_filename}"

            await self._storage.upload_bytes(
                bucket_name=self._bucket,
                object_name=object_name,
                data=image_data,
                content_type=content_type,
            )

            logger.info(f"Uploaded image: {object_name}")

            return {
                "original_url": image_url,
                "minio_object": object_name,
                "filename": unique_filename,
                "content_type": content_type,
                "size_bytes": len(image_data),
            }
        except Exception as e:
            logger.error(f"Error handling image {image_url}: {e}")
            return None

    def extract_image_urls_from_markdown(self, markdown: str) -> list[str]:
        """Extract image URLs from markdown text."""
        import re

        # Pattern for markdown images: ![alt](url)
        pattern = r"!\[.*?\]\((.*?)\)"
        matches = re.findall(pattern, markdown)
        return [url.strip() for url in matches if url.strip()]
