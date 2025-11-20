from sqlalchemy.engine import Engine


def create_muc(engine: Engine, muc_id: str, name: str) -> None:
    with engine.begin() as conn:
        conn.execute(
            "INSERT INTO muc (id, name) VALUES (%s, %s) ON CONFLICT (id) DO NOTHING",
            (muc_id, name),
        )


def create_khoi(engine: Engine, khoi_id: str, name: str) -> None:
    with engine.begin() as conn:
        conn.execute(
            "INSERT INTO khoi (id, name) VALUES (%s, %s) ON CONFLICT (id) DO NOTHING",
            (khoi_id, name),
        )


