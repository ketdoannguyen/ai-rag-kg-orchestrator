from sqlalchemy.engine import Engine


def search_lop_1(engine: Engine, query: str) -> list[dict]:
    with engine.begin() as conn:
        rows = conn.execute(
            "SELECT id, title FROM lop1 WHERE title ILIKE %s LIMIT 5",
            (f"%{query}%",),
        )
    return [dict(row) for row in rows]


def search_lop_2(engine: Engine, query: str) -> list[dict]:
    with engine.begin() as conn:
        rows = conn.execute(
            "SELECT id, title FROM lop2 WHERE title ILIKE %s LIMIT 5",
            (f"%{query}%",),
        )
    return [dict(row) for row in rows]


