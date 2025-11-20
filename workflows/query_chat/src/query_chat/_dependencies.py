from config.llm_gateway import build_llm_gateway_client
from config.settings import get_settings
from llm_gateway_client import AIGatewayClient
from postgres_client import PostgresClient


def init_llm_client() -> AIGatewayClient:
    settings = get_settings()
    return build_llm_gateway_client(settings)


def init_postgres_client(database_url: str) -> PostgresClient:
    return PostgresClient(url=database_url)


