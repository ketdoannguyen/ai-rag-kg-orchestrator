from neo4j import Driver


def create_muc_node(driver: Driver, muc_id: str, name: str) -> None:
    with driver.session() as session:
        session.run(
            "MERGE (m:Muc {id: $id}) SET m.name = $name",
            id=muc_id,
            name=name,
        )


def create_khoi_edge(driver: Driver, muc_id: str, khoi_id: str) -> None:
    with driver.session() as session:
        session.run(
            """
            MATCH (m:Muc {id: $muc_id})
            MERGE (k:Khoi {id: $khoi_id})
            MERGE (m)-[:HAS_KHOI]->(k)
            """,
            muc_id=muc_id,
            khoi_id=khoi_id,
        )

