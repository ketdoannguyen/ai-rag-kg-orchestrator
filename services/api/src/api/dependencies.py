from functools import lru_cache
from typing import Callable

from fastapi import Depends

from config.llm_gateway import build_llm_gateway_client
from config.settings import Settings, get_settings
from llm_gateway_client import AIGatewayClient
from arango_client import ArangoClient
from workflows.admin_cms import update_content
from workflows.ingest_file import start_ingest
from workflows.query_chat import execute_query


@lru_cache()
def get_app_settings() -> Settings:
    return get_settings()


@lru_cache()
def get_arango_client(
    settings: Settings = Depends(get_app_settings),
) -> ArangoClient:
    return ArangoClient(
        url=settings.ARANGO_URL,
        database=settings.ARANGO_DATABASE,
        username=settings.ARANGO_USER,
        password=settings.ARANGO_PASSWORD,
    )


@lru_cache()
def get_llm_client(
    settings: Settings = Depends(get_app_settings),
) -> AIGatewayClient:
    return build_llm_gateway_client(settings)


def get_ingest_workflow(
    settings: Settings = Depends(get_app_settings),
) -> Callable[[str], dict]:
    def workflow_instance(file_path: str) -> dict:
        return start_ingest(file_path=file_path)

    return workflow_instance


def get_chat_workflow(
    arango_client: ArangoClient = Depends(get_arango_client),
    llm_client: AIGatewayClient = Depends(get_llm_client),
) -> Callable[[str], dict]:
    def workflow_instance(query: str) -> dict:
        return execute_query(query, arango_client, llm_client)

    return workflow_instance


def get_cms_workflow(
    arango_client: ArangoClient = Depends(get_arango_client),
) -> Callable[[str, str], dict]:
    def workflow_instance(chunk_id: str, text: str) -> dict:
        return update_content(arango_client, chunk_id, text)

    return workflow_instance
