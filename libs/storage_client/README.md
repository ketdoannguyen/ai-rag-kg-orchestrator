# Storage Client

Lightweight wrapper around the MinIO Python SDK used by workflows/services to push and fetch artifacts.

## Features

- Automatic MinIO client initialization (secure/http aware).
- Bucket utilities: ensure, list, delete single or multiple objects.
- High-level helpers for uploading local files or in-memory bytes with optional metadata.
- Download helper that guarantees destination directories exist.

## Usage

```python
from storage_client import StorageClient

storage = StorageClient(
    endpoint="http://localhost:9000",
    access_key="minio",
    secret_key="minio123",
)

storage.ensure_bucket("ingest")
storage.upload_file("ingest", "file/source/sample.pdf", "/tmp/sample.pdf")
```

Credentials and endpoints are typically provided via `config.settings`.






