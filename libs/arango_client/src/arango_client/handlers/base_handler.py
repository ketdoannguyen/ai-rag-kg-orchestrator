"""
Base handler class with common CRUD operations for ArangoDB.

This provides a foundation for all model-specific handlers.
All AQL queries are imported from the centralized aql/ module.
"""

from __future__ import annotations

import uuid
from datetime import datetime, timezone
from typing import Any, Generic, Optional, Type, TypeVar

from arango.database import StandardDatabase
from arango.collection import StandardCollection

from ..aql import CRUD_GET_ALL, CRUD_GET_BY_FILTER, CRUD_COUNT_WITH_FILTER

# Type variables for generic handler
ModelT = TypeVar("ModelT")


class BaseHandler(Generic[ModelT]):
    """
    Base handler class providing common CRUD operations for ArangoDB.

    This class uses generics to work with any Pydantic model.
    Subclasses should specify the model type and collection name.

    Pattern:
        1. Input: Pydantic model
        2. Convert to document using to_document() method
        3. Use AQL for database operations
        4. Convert back to Pydantic using from_document() method

    Example:
        >>> class ChatSessionHandler(BaseHandler[ChatSession]):
        ...     def __init__(self, db: StandardDatabase):
        ...         super().__init__(db, ChatSession, "chat_sessions")
    """

    def __init__(
        self,
        db: StandardDatabase,
        model_class: Type[ModelT],
        collection_name: str,
    ):
        """
        Initialize handler with database and model info.

        Args:
            db: ArangoDB database instance
            model_class: Pydantic model class
            collection_name: Name of the ArangoDB collection
        """
        self.db = db
        self.model_class = model_class
        self.collection_name = collection_name
        self._collection: StandardCollection = db.collection(collection_name)

    @property
    def collection(self) -> StandardCollection:
        """Get the collection instance."""
        return self._collection

    def create(self, model: ModelT) -> ModelT:
        """
        Create a new document from Pydantic model.

        Args:
            model: Validated Pydantic model instance

        Returns:
            Created Pydantic model instance (with database-generated fields)
        """
        # Convert Pydantic to document
        if hasattr(model, "to_document"):
            doc = model.to_document()
        else:
            doc = model.model_dump()
            doc["_key"] = str(doc.pop("id", uuid.uuid4()))

        # Insert document
        result = self._collection.insert(doc, return_new=True)

        # Convert back to Pydantic
        if hasattr(self.model_class, "from_document"):
            return self.model_class.from_document(result["new"])
        return self.model_class(**result["new"])

    def get_by_id(self, id: uuid.UUID | str) -> Optional[ModelT]:
        """
        Get a document by ID.

        Args:
            id: Primary key value (UUID or string)

        Returns:
            Pydantic model instance or None if not found
        """
        key = str(id)
        doc = self._collection.get(key)

        if doc is None:
            return None

        if hasattr(self.model_class, "from_document"):
            return self.model_class.from_document(doc)
        return self.model_class(**doc)

    def get_all(
        self,
        limit: Optional[int] = None,
        offset: Optional[int] = None,
        sort_field: str = "created_at",
        sort_desc: bool = True,
    ) -> list[ModelT]:
        """
        Get all documents with optional pagination and ordering.

        Args:
            limit: Maximum number of records to return
            offset: Number of records to skip
            sort_field: Field to sort by (default: created_at)
            sort_desc: If True, sort descending (default: True)

        Returns:
            List of Pydantic model instances
        """
        sort_dir = "DESC" if sort_desc else "ASC"
        limit = limit or 100
        offset = offset or 0

        # Use centralized AQL template
        query = CRUD_GET_ALL.format(
            collection=self.collection_name,
            sort_field=sort_field,
            sort_dir=sort_dir,
        )

        cursor = self.db.aql.execute(query, bind_vars={"offset": offset, "limit": limit})

        results = []
        for doc in cursor:
            if hasattr(self.model_class, "from_document"):
                results.append(self.model_class.from_document(doc))
            else:
                results.append(self.model_class(**doc))

        return results

    def get_by_filter(
        self,
        filters: dict[str, Any],
        limit: Optional[int] = None,
        offset: Optional[int] = None,
        sort_field: str = "created_at",
        sort_desc: bool = True,
    ) -> list[ModelT]:
        """
        Get documents by filter criteria.

        Args:
            filters: Dictionary of field: value pairs to filter by
            limit: Maximum number of records to return
            offset: Number of records to skip
            sort_field: Field to sort by
            sort_desc: If True, sort descending

        Returns:
            List of Pydantic model instances matching filters
        """
        sort_dir = "DESC" if sort_desc else "ASC"
        limit = limit or 100
        offset = offset or 0

        # Build filter conditions
        filter_conditions = []
        bind_vars = {"offset": offset, "limit": limit}

        for i, (field, value) in enumerate(filters.items()):
            var_name = f"val{i}"
            filter_conditions.append(f"doc.{field} == @{var_name}")
            bind_vars[var_name] = str(value) if isinstance(value, uuid.UUID) else value

        filter_clause = " AND ".join(filter_conditions) if filter_conditions else "true"

        # Use centralized AQL template
        query = CRUD_GET_BY_FILTER.format(
            collection=self.collection_name,
            filter_clause=filter_clause,
            sort_field=sort_field,
            sort_dir=sort_dir,
        )

        cursor = self.db.aql.execute(query, bind_vars=bind_vars)

        results = []
        for doc in cursor:
            if hasattr(self.model_class, "from_document"):
                results.append(self.model_class.from_document(doc))
            else:
                results.append(self.model_class(**doc))

        return results

    def update(
        self,
        id: uuid.UUID | str,
        update_data: dict[str, Any],
    ) -> Optional[ModelT]:
        """
        Update a document by ID.

        Args:
            id: Primary key value
            update_data: Dictionary of fields to update

        Returns:
            Updated Pydantic model instance or None if not found
        """
        key = str(id)

        # Check if document exists
        if not self._collection.has(key):
            return None

        # Add updated_at timestamp if the model supports it
        if "updated_at" not in update_data:
            update_data["updated_at"] = datetime.now(timezone.utc).isoformat()

        # Convert UUIDs to strings
        for field, value in update_data.items():
            if isinstance(value, uuid.UUID):
                update_data[field] = str(value)

        result = self._collection.update(
            {"_key": key, **update_data},
            return_new=True,
        )

        if hasattr(self.model_class, "from_document"):
            return self.model_class.from_document(result["new"])
        return self.model_class(**result["new"])

    def update_by_model(self, model: ModelT) -> Optional[ModelT]:
        """
        Update a document using a Pydantic model.

        Args:
            model: Pydantic model instance with updated data

        Returns:
            Updated Pydantic model instance or None if not found
        """
        if not hasattr(model, "id"):
            raise ValueError("Pydantic model must have an 'id' field for update")

        if hasattr(model, "to_document"):
            doc = model.to_document()
        else:
            doc = model.model_dump()
            doc["_key"] = str(doc.pop("id"))

        key = doc["_key"]

        if not self._collection.has(key):
            return None

        result = self._collection.replace(doc, return_new=True)

        if hasattr(self.model_class, "from_document"):
            return self.model_class.from_document(result["new"])
        return self.model_class(**result["new"])

    def delete(self, id: uuid.UUID | str) -> bool:
        """
        Delete a document by ID.

        Args:
            id: Primary key value

        Returns:
            True if deleted, False if not found
        """
        key = str(id)

        if not self._collection.has(key):
            return False

        self._collection.delete(key)
        return True

    def count(self, filters: Optional[dict[str, Any]] = None) -> int:
        """
        Count documents, optionally filtered.

        Args:
            filters: Optional dictionary of field: value pairs to filter by

        Returns:
            Number of documents matching criteria
        """
        if not filters:
            return self._collection.count()

        # Build filter conditions
        filter_conditions = []
        bind_vars = {}

        for i, (field, value) in enumerate(filters.items()):
            var_name = f"val{i}"
            filter_conditions.append(f"doc.{field} == @{var_name}")
            bind_vars[var_name] = str(value) if isinstance(value, uuid.UUID) else value

        filter_clause = " AND ".join(filter_conditions)

        # Use centralized AQL template
        query = CRUD_COUNT_WITH_FILTER.format(
            collection=self.collection_name,
            filter_clause=filter_clause,
        )

        cursor = self.db.aql.execute(query, bind_vars=bind_vars)
        return next(cursor, 0)

    def exists(self, id: uuid.UUID | str) -> bool:
        """Check if a document exists by ID."""
        return self._collection.has(str(id))

    def bulk_create(self, models: list[ModelT]) -> list[ModelT]:
        """
        Bulk create multiple documents.

        Args:
            models: List of Pydantic model instances

        Returns:
            List of created Pydantic model instances
        """
        docs = []
        for model in models:
            if hasattr(model, "to_document"):
                docs.append(model.to_document())
            else:
                doc = model.model_dump()
                doc["_key"] = str(doc.pop("id", uuid.uuid4()))
                docs.append(doc)

        results = self._collection.insert_many(docs, return_new=True)

        created = []
        for result in results:
            if result.get("new"):
                if hasattr(self.model_class, "from_document"):
                    created.append(self.model_class.from_document(result["new"]))
                else:
                    created.append(self.model_class(**result["new"]))

        return created

    def bulk_delete(self, ids: list[uuid.UUID | str]) -> int:
        """
        Bulk delete multiple documents.

        Args:
            ids: List of document IDs to delete

        Returns:
            Number of documents deleted
        """
        keys = [str(id) for id in ids]
        results = self._collection.delete_many([{"_key": k} for k in keys])
        return len([r for r in results if not r.get("error")])
