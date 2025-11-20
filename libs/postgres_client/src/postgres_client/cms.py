from sqlalchemy.engine import Engine


def update_chunk_text(engine: Engine, chunk_id: str, text: str) -> None:
    with engine.begin() as conn:
        conn.execute(
            "UPDATE chunks SET text = %s WHERE id = %s",
            (text, chunk_id),
        )


def delete_chunk(engine: Engine, chunk_id: str) -> None:
    with engine.begin() as conn:
        conn.execute("DELETE FROM chunks WHERE id = %s", (chunk_id,))


