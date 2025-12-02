# CRUD Handlers

Comprehensive CRUD (Create, Read, Update, Delete) handlers for all PostgreSQL models.

## Architecture

1. **Input**: Pydantic models (validated data)
2. **Conversion**: Convert to ORM using `to_orm()` method (or `pydantic_to_orm()` for KnowledgeNode)
3. **Database Operations**: Use SQLAlchemy Session for database operations
4. **Output**: Convert back to Pydantic using `from_orm()` method (or `orm_to_pydantic()` for KnowledgeNode)

## Quick Start

```python
from sqlalchemy.orm import Session, sessionmaker
from postgres_client.client import PostgresClient
from postgres_client.handlers import ChatHandler, KnowledgeHandler

client = PostgresClient("postgresql://...")
SessionLocal = sessionmaker(bind=client.engine)

with SessionLocal() as session:
    chat_handler = ChatHandler(session)
    knowledge_handler = KnowledgeHandler(session)
    
    # Use handlers...
    session.commit()
```

## Handlers

### ChatHandler
```python
from postgres_client.models import ChatSession, ChatMessage, Feedback

# Create session
session_model = ChatSession(user_identifier="user123")
created = chat_handler.sessions.create(session_model)

# Create message
message = ChatMessage(
    session_id=created.id,
    role="user",
    content="Hello",
)
chat_handler.messages.create(message)

# Get messages
messages = chat_handler.messages.get_by_session(created.id)
```

### KnowledgeHandler
```python
from postgres_client.models import ParagraphNode, StructureNode

# Create paragraph
paragraph = ParagraphNode(
    source_id=source_id,
    attributes={"text": "Content"},
    metadata={"page": 1},
)
created = knowledge_handler.nodes.create(paragraph)

# Get by source
nodes = knowledge_handler.nodes.get_by_source(source_id)

# Get children
children = knowledge_handler.nodes.get_children(node_id)
```

### SystemAdminHandler
```python
from postgres_client.models import User, DataSource

# Create user
user = User(username="admin", hashed_password="...")
created_user = system_admin_handler.users.create(user)

# Get by username
user = system_admin_handler.users.get_by_username("admin")

# Create source
source = DataSource(name="doc.pdf", type=SourceType.PDF)
system_admin_handler.sources.create(source)
```

### EmbeddingHandler
```python
from postgres_client.models import EmbeddingStructure, EmbeddingContent

# Create structure embedding
structure = EmbeddingStructure(
    node_id=node_id,
    summary_text="Summary",
    vector=[0.1] * 1024,
)
embedding_handler.structures.create(structure)

# Update vector
embedding_handler.structures.update_vector(node_id, new_vector)
```

### MiningHandler
```python
from postgres_client.models import SuggestedFAQ

# Create FAQ
faq = SuggestedFAQ(
    representative_question="How to...?",
    frequency=5,
)
mining_handler.faqs.create(faq)

# Approve
mining_handler.faqs.approve(faq_id)
```

## Base CRUD Operations

All handlers provide these methods:

- `create(pydantic_model)` - Create a new record
- `get_by_id(id)` - Get a record by ID
- `get_all(limit, offset, order_by)` - Get all records with pagination
- `get_by_filter(filters, limit, offset, order_by)` - Get records by filter criteria
- `update(id, update_data)` - Update a record by ID
- `update_by_model(pydantic_model)` - Update using a Pydantic model
- `delete(id)` - Delete a record by ID
- `count(filters)` - Count records matching criteria

## Error Handling

Handlers return `None` for not found (get operations) or `False` (delete operations):

```python
user = handler.users.get_by_id(user_id)
if user is None:
    print("User not found")
    return
```

## Transactions

Handlers use the provided session. Manage transactions explicitly:

```python
with SessionLocal() as session:
    try:
        handler = ChatHandler(session)
        # Operations...
        session.commit()
    except Exception:
        session.rollback()
        raise
```

## Notes

- **KnowledgeNode** uses special converters (`pydantic_to_orm`, `orm_to_pydantic`) due to complex subtypes
- Other models use standard `to_orm()` and `from_orm()` methods
- All timestamps are handled automatically by the database
