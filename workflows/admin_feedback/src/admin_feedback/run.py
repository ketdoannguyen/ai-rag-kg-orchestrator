from postgres_client import PostgresClient


def log_feedback(client: PostgresClient, payload: dict) -> dict:
    with client.engine.begin() as conn:
        conn.execute(
            "INSERT INTO feedback (content) VALUES (%s)",
            (payload.get("content", ""),),
        )
    return {"status": "logged"}


