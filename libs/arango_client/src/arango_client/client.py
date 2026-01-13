"""
ArangoDB Client for multi-model database operations.

This client manages connections to ArangoDB and provides auto-initialization
of database, collections, indexes, graphs, and views.

Usage:
    >>> from arango_client import ArangoClient
    >>> client = ArangoClient(url="http://localhost:8529", database="mydb")
    >>> # Use handlers for operations
    >>> knowledge = KnowledgeHandler(client.db)
"""

from __future__ import annotations

import logging
from typing import Any

from arango import ArangoClient as ArangoDriver
from arango.database import StandardDatabase
from arango.exceptions import (
    CollectionCreateError,
    DatabaseCreateError,
    GraphCreateError,
    IndexCreateError,
    ViewCreateError,
)

from .collections.schema import (
    ALL_DOCUMENT_COLLECTIONS,
    ALL_EDGE_COLLECTIONS,
    INDEXES,
    KNOWLEDGE_COLLECTIONS,
    KNOWLEDGE_GRAPH_DEFINITION,
    KNOWLEDGE_GRAPH_NAME,
    KNOWLEDGE_VECTOR_INDEX_DEFINITION,
    KNOWLEDGE_VECTOR_VIEW_DEFINITION,
    KNOWLEDGE_VECTOR_VIEW_NAME,
)

logger = logging.getLogger(__name__)


class ArangoClient:
    """
    ArangoDB client wrapper with auto-initialization.

    This client handles:
    - Connection management
    - Database creation (if not exists)
    - Collection creation (document and edge)
    - Index creation
    - Graph creation
    - ArangoSearch view creation for vector search

    Attributes:
        url: ArangoDB server URL
        database_name: Name of the database to use
        db: StandardDatabase instance for operations

    Example:
        >>> client = ArangoClient(
        ...     url="http://localhost:8529",
        ...     database="knowledge_db",
        ...     username="root",
        ...     password="password",
        ... )
        >>> # Get a collection
        >>> collection = client.db.collection("knowledge_nodes")
        >>> # Insert a document
        >>> collection.insert({"_key": "123", "name": "test"})
    """

    def __init__(
        self,
        url: str = "http://localhost:8529",
        database: str = "knowledge_db",
        username: str = "root",
        password: str = "",
        *,
        auto_init: bool = True,
    ) -> None:
        """
        Initialize ArangoDB client.

        Args:
            url: ArangoDB server URL (default: http://localhost:8529)
            database: Database name to use (default: knowledge_db)
            username: Username for authentication (default: root)
            password: Password for authentication (default: empty)
            auto_init: If True, automatically create database, collections,
                      indexes, graphs, and views if they don't exist.
        """
        self._url = url
        self._database_name = database
        self._username = username
        self._password = password

        # Create driver
        self._driver = ArangoDriver(hosts=url)

        # Connect to system database first (for database creation)
        self._sys_db = self._driver.db(
            "_system",
            username=username,
            password=password,
        )

        # Ensure database exists
        if auto_init:
            self._ensure_database_exists()

        # Connect to target database
        self._db: StandardDatabase = self._driver.db(
            database,
            username=username,
            password=password,
        )

        # Initialize schema
        if auto_init:
            self._initialize_schema()

    @property
    def db(self) -> StandardDatabase:
        """Get the database instance for operations."""
        return self._db

    @property
    def database_name(self) -> str:
        """Get the database name."""
        return self._database_name

    def _ensure_database_exists(self) -> None:
        """Create database if it doesn't exist."""
        if not self._sys_db.has_database(self._database_name):
            try:
                self._sys_db.create_database(self._database_name)
                logger.info(f"Created database: {self._database_name}")
            except DatabaseCreateError as e:
                # Database might have been created by another process
                if "duplicate" not in str(e).lower():
                    raise
                logger.debug(f"Database already exists: {self._database_name}")

    def _initialize_schema(self) -> None:
        """Initialize all collections, indexes, graphs, and views."""
        self._create_collections()
        self._create_indexes()
        self._create_vector_indexes()
        self._create_graph()
        self._create_vector_view()

    def _create_collections(self) -> None:
        """Create all document and edge collections."""
        # Document collections
        for collection_name in ALL_DOCUMENT_COLLECTIONS:
            if not self._db.has_collection(collection_name):
                try:
                    self._db.create_collection(collection_name)
                    logger.info(f"Created document collection: {collection_name}")
                except CollectionCreateError as e:
                    if "duplicate" not in str(e).lower():
                        raise
                    logger.debug(f"Collection already exists: {collection_name}")

        # Edge collections
        for collection_name in ALL_EDGE_COLLECTIONS:
            if not self._db.has_collection(collection_name):
                try:
                    self._db.create_collection(collection_name, edge=True)
                    logger.info(f"Created edge collection: {collection_name}")
                except CollectionCreateError as e:
                    if "duplicate" not in str(e).lower():
                        raise
                    logger.debug(f"Edge collection already exists: {collection_name}")

    def _create_indexes(self) -> None:
        """Create all defined indexes."""
        for index_def in INDEXES:
            collection = self._db.collection(index_def.collection)

            # Get existing indexes
            existing_indexes = {idx.get("name") for idx in collection.indexes()}

            # Skip if index already exists
            if index_def.name and index_def.name in existing_indexes:
                logger.debug(f"Index already exists: {index_def.name}")
                continue

            try:
                if index_def.type == "persistent":
                    collection.add_persistent_index(
                        fields=index_def.fields,
                        unique=index_def.unique,
                        sparse=index_def.sparse,
                        name=index_def.name,
                    )
                elif index_def.type == "hash":
                    collection.add_hash_index(
                        fields=index_def.fields,
                        unique=index_def.unique,
                        sparse=index_def.sparse,
                        name=index_def.name,
                    )
                elif index_def.type == "fulltext":
                    # Fulltext is deprecated, use ArangoSearch instead
                    logger.warning(f"Fulltext index type is deprecated: {index_def.name}")

                logger.info(f"Created index: {index_def.name} on {index_def.collection}")

            except IndexCreateError as e:
                if "duplicate" not in str(e).lower():
                    raise
                logger.debug(f"Index already exists: {index_def.name}")

    def _create_vector_indexes(self) -> None:
        """Create vector indexes for ANN search (ArangoDB 3.12+).

        NOTE: Inverted indexes do NOT support embedding arrays directly.
        For vector search, we rely on:
        1. ArangoSearch view with COSINE_SIMILARITY() (works on raw embedding field)
        2. Native vector index (ArangoDB 3.12+) if available

        This method attempts to create a native vector index. If that fails
        (older ArangoDB version), we fall back to ArangoSearch view-based search.
        """
        # Native vector index only available in ArangoDB 3.12+
        # Check ArangoDB version first
        try:
            version_info = self._db.version()
            version = version_info.get("version", "0.0.0")
            major, minor = map(int, version.split(".")[:2])

            if major < 3 or (major == 3 and minor < 12):
                logger.info(
                    f"ArangoDB {version} does not support native vector indexes. "
                    "Using ArangoSearch view with COSINE_SIMILARITY() for vector search."
                )
                return

            # Create native vector indexes for ArangoDB 3.12+
            index_def = KNOWLEDGE_VECTOR_INDEX_DEFINITION
            for collection_name in KNOWLEDGE_COLLECTIONS:
                collection = self._db.collection(collection_name)
                existing_indexes = {idx.get("name") for idx in collection.indexes()}
                index_name = f"{index_def['name']}_{collection_name}"

                if index_name in existing_indexes:
                    logger.debug(f"Vector index already exists: {index_name}")
                    continue

                try:
                    collection.add_index(
                        {
                            "type": "mdi-prefixed",  # MDI index for vector search in 3.12+
                            "name": index_name,
                            "fields": ["embedding"],
                            "fieldValueTypes": "double",
                            "prefixFields": ["source_id"],  # Optional prefix for filtering
                            "unique": False,
                            "sparse": True,  # Skip documents without embedding
                        }
                    )
                    logger.info(f"Created vector index: {index_name}")
                except Exception as e:
                    logger.debug(f"Could not create MDI vector index {index_name}: {e}")

        except Exception as e:
            logger.debug(f"Could not check ArangoDB version or create vector indexes: {e}")

    def _create_graph(self) -> None:
        """Create or update the knowledge graph.

        If graph exists but edge definitions are outdated (e.g., vertex collections
        changed), the graph will be replaced with new definitions.
        """
        graph_name = KNOWLEDGE_GRAPH_NAME
        expected_edge_defs = KNOWLEDGE_GRAPH_DEFINITION["edge_definitions"]

        if self._db.has_graph(graph_name):
            # Check if edge definitions need update
            graph = self._db.graph(graph_name)
            current_edge_defs = graph.edge_definitions()

            # Compare vertex collections
            needs_update = False
            for expected_def in expected_edge_defs:
                edge_col = expected_def["edge_collection"]
                expected_from = set(expected_def["from_vertex_collections"])
                expected_to = set(expected_def["to_vertex_collections"])

                # Find matching current definition
                current_def = next(
                    (d for d in current_edge_defs if d["edge_collection"] == edge_col), None
                )

                if current_def is None:
                    needs_update = True
                    break

                current_from = set(current_def.get("from_vertex_collections", []))
                current_to = set(current_def.get("to_vertex_collections", []))

                if current_from != expected_from or current_to != expected_to:
                    logger.info(
                        f"Graph edge definition outdated. "
                        f"Expected from={expected_from}, to={expected_to}. "
                        f"Current from={current_from}, to={current_to}"
                    )
                    needs_update = True
                    break

            if needs_update:
                # Delete and recreate graph with new definitions
                logger.info(f"Recreating graph {graph_name} with updated edge definitions")
                self._db.delete_graph(graph_name, drop_collections=False)
            else:
                logger.debug(f"Graph already exists with correct definitions: {graph_name}")
                return

        # Create graph (either new or after deletion)
        try:
            graph = self._db.create_graph(graph_name)

            for edge_def in expected_edge_defs:
                if not graph.has_edge_definition(edge_def["edge_collection"]):
                    graph.create_edge_definition(
                        edge_collection=edge_def["edge_collection"],
                        from_vertex_collections=edge_def["from_vertex_collections"],
                        to_vertex_collections=edge_def["to_vertex_collections"],
                    )

            logger.info(f"Created graph: {graph_name}")

        except GraphCreateError as e:
            if "duplicate" not in str(e).lower():
                raise
            logger.debug(f"Graph already exists: {graph_name}")

    def _create_vector_view(self) -> None:
        """Create ArangoSearch view for vector similarity search."""
        # Check if view exists
        existing_views = [v["name"] for v in self._db.views()]

        if KNOWLEDGE_VECTOR_VIEW_NAME not in existing_views:
            try:
                self._db.create_arangosearch_view(
                    name=KNOWLEDGE_VECTOR_VIEW_NAME,
                    properties=KNOWLEDGE_VECTOR_VIEW_DEFINITION,
                )
                logger.info(f"Created ArangoSearch view: {KNOWLEDGE_VECTOR_VIEW_NAME}")

            except ViewCreateError as e:
                if "duplicate" not in str(e).lower():
                    raise
                logger.debug(f"View already exists: {KNOWLEDGE_VECTOR_VIEW_NAME}")
        else:
            logger.debug(f"View already exists: {KNOWLEDGE_VECTOR_VIEW_NAME}")

    def close(self) -> None:
        """Close the client connection."""
        # python-arango doesn't require explicit close
        # but we keep this method for API consistency
        pass

    def __enter__(self) -> ArangoClient:
        """Context manager entry."""
        return self

    def __exit__(
        self,
        exc_type: type[BaseException] | None,
        exc: BaseException | None,
        tb: Any,
    ) -> None:
        """Context manager exit."""
        self.close()

    # =========================================================================
    # Utility methods
    # =========================================================================

    def drop_database(self, *, confirm: bool = False) -> bool:
        """
        Drop the current database.

        Args:
            confirm: Must be True to confirm deletion (safety check)

        Returns:
            True if database was dropped, False otherwise

        Warning:
            This is a destructive operation and cannot be undone.
        """
        if not confirm:
            logger.warning("Database drop requires confirm=True")
            return False

        if self._sys_db.has_database(self._database_name):
            self._sys_db.delete_database(self._database_name)
            logger.info(f"Dropped database: {self._database_name}")
            return True

        return False

    def get_collection(self, name: str) -> Any:
        """
        Get a collection by name.

        Args:
            name: Collection name

        Returns:
            Collection instance
        """
        return self._db.collection(name)

    def execute_aql(
        self,
        query: str,
        bind_vars: dict[str, Any] | None = None,
        **kwargs: Any,
    ) -> list[Any]:
        """
        Execute an AQL query and return results as a list.

        Args:
            query: AQL query string
            bind_vars: Query bind variables
            **kwargs: Additional cursor options

        Returns:
            List of query results
        """
        cursor = self._db.aql.execute(query, bind_vars=bind_vars or {}, **kwargs)
        return list(cursor)

    def begin_transaction(
        self,
        read: list[str] | None = None,
        write: list[str] | None = None,
        exclusive: list[str] | None = None,
    ) -> Any:
        """
        Begin a transaction.

        Args:
            read: Collections for read access
            write: Collections for write access
            exclusive: Collections for exclusive access

        Returns:
            Transaction database instance
        """
        return self._db.begin_transaction(
            read=read or [],
            write=write or [],
            exclusive=exclusive or [],
        )
