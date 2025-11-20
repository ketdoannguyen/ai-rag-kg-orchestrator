from functools import lru_cache
from config.settings import settings
from storage_client import StorageClient

@lru_cache(maxsize=1)
def get_storage_client() -> StorageClient:
    if (
        not settings.MINIO_ENDPOINT
        and not settings.MINIO_ACCESS_KEY
        and not settings.MINIO_SECRET_KEY
    ):
        raise ValueError(
            "MINIO_ENDPOINT, MINIO_ACCESS_KEY, and MINIO_SECRET_KEY must be set in the environment."
        )
        
    if not settings.MINIO_BUCKET:
        raise ValueError(
            "MINIO_BUCKET must be set in the environment."
        )
        
    return StorageClient(
        endpoint=settings.MINIO_ENDPOINT,
        access_key=settings.MINIO_ACCESS_KEY,
        secret_key=settings.MINIO_SECRET_KEY,
    )




