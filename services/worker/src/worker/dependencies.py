from functools import lru_cache

from config.settings import get_settings
from arango_client import ArangoClient
from workflows.ingest_file import start_ingest


@lru_cache()
def get_arango_client() -> ArangoClient:
    settings = get_settings()
    return ArangoClient(
        url=settings.ARANGO_URL,
        database=settings.ARANGO_DATABASE,
        username=settings.ARANGO_USER,
        password=settings.ARANGO_PASSWORD,
    )


def get_ingest_workflow():
    def runner(file_path: str) -> dict:
        return start_ingest(file_path=file_path)

    return runner

