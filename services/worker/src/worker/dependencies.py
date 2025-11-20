from functools import lru_cache

from config.settings import get_settings
from ingest_file import start_ingest
from postgres_client import PostgresClient


@lru_cache()
def get_postgres_client() -> PostgresClient:
    settings = get_settings()
    return PostgresClient(url=settings.DATABASE_URL)


def get_ingest_workflow():
    settings = get_settings()

    def runner(file_path: str) -> dict:
        return start_ingest(file_path=file_path, database_url=settings.DATABASE_URL)

    return runner


