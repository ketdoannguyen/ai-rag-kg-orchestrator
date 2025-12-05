from neo4j import GraphDatabase, AsyncGraphDatabase, AsyncDriver, Driver


class Neo4jClient:
    """Neo4j driver wrapper with async support."""

    def __init__(self, uri: str, user: str, password: str, *, use_async: bool = True) -> None:
        """
        Initialize Neo4j client.
        
        Args:
            uri: Neo4j connection URI (e.g., "bolt://localhost:7687")
            user: Neo4j username
            password: Neo4j password
            use_async: If True, use async driver (default). If False, use sync driver.
        """
        if use_async:
            self._driver: AsyncDriver = AsyncGraphDatabase.driver(uri, auth=(user, password))
            self._sync_driver: Driver | None = None
        else:
            self._sync_driver: Driver = GraphDatabase.driver(uri, auth=(user, password))
            self._driver: AsyncDriver | None = None
        self._use_async = use_async

    @property
    def driver(self) -> AsyncDriver | Driver:
        """Get the Neo4j driver (async or sync depending on initialization)."""
        return self._driver if self._use_async else self._sync_driver

    @property
    def async_driver(self) -> AsyncDriver:
        """Get the async driver. Raises if async driver is not available."""
        if not self._use_async or self._driver is None:
            raise ValueError("Async driver not available. Initialize with use_async=True")
        return self._driver

    async def close(self) -> None:
        """Close the driver connection."""
        if self._use_async and self._driver:
            await self._driver.close()
        elif self._sync_driver:
            self._sync_driver.close()

    async def verify_connectivity(self) -> bool:
        """Verify connectivity to Neo4j database."""
        try:
            if self._use_async and self._driver:
                await self._driver.verify_connectivity()
                return True
            elif self._sync_driver:
                self._sync_driver.verify_connectivity()
                return True
            return False
        except Exception:
            return False


