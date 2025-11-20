from postgres_client import cms, PostgresClient


def update_content(client: PostgresClient, chunk_id: str, text: str) -> dict:
    cms.update_chunk_text(client.engine, chunk_id, text)
    return {"chunk_id": chunk_id, "status": "updated"}


