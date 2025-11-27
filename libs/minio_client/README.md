# Storage Client

Lightweight wrapper around the MinIO Python SDK used by workflows/services to push and fetch artifacts.

## Features

- Automatic MinIO client initialization (secure/http aware).
- Bucket utilities: ensure, list, delete single or multiple objects.
- High-level helpers for uploading local files or in-memory bytes with optional metadata.
- Download helper that guarantees destination directories exist.

## Usage

```python
import asyncio
from minio_client import StorageClient


async def main() -> None:
    storage = StorageClient(
        endpoint="http://localhost:9000",
        access_key="minio",
        secret_key="minio123",
    )

    await storage.ensure_bucket("ingest")
    await storage.upload_file("ingest", "file/source/sample.pdf", "/tmp/sample.pdf")


if __name__ == "__main__":
    asyncio.run(main())
```

All I/O helpers are `async def`, so either await them from existing async workflows or wrap them with `asyncio.run` inside synchronous entry points.

Credentials and endpoints are typically provided via `config.settings`.
