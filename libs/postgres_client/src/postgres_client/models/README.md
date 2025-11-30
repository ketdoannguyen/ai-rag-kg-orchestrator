# Pydantic Models

Pydantic models for data validation and serialization, with conversion methods to/from ORM classes.

## Overview

All Pydantic models provide:
- **Data validation** through Pydantic
- **`to_orm()`** method: Convert Pydantic → ORM
- **`from_orm()`** classmethod: Convert ORM → Pydantic

## Conversion Methods

### KnowledgeNode (and subtypes)

KnowledgeNode has multiple subtypes (ParagraphNode, ImageNode, TableNode, CorpusNode, TopicNode, SubtopicNode) but uses the same conversion pattern.

```python
from postgres_client.models import ParagraphNode

# Pydantic → ORM
paragraph = ParagraphNode(
    source_id=uuid.uuid4(),
    attributes={"text": "Hello"},
    metadata={"page": 1}
)
orm_node = paragraph.to_orm()

# ORM → Pydantic
pydantic_node = ParagraphNode.from_orm(orm_node)
```

**Note**: For automatic subtype inference, use converter utilities:
```python
from postgres_client.utils.knowledge_node_converter import orm_to_pydantic
pydantic_node = orm_to_pydantic(orm_node)  # Auto-detects subtype
```

### Other Models

All other models follow the same pattern:

```python
from postgres_client.models import User, DataSource, ChatSession

# Pydantic → ORM
user = User(username="john", hashed_password="...")
orm_user = user.to_orm()

# ORM → Pydantic
pydantic_user = User.from_orm(orm_user)
```

## Model Categories

### System Admin
- **User**: User accounts and authentication
- **DataSource**: Source documents/files

### Knowledge
- **KnowledgeNode**: Base class for all knowledge nodes
- **ParagraphNode**: Detail-level paragraph nodes
- **ImageNode**: Detail-level image nodes
- **TableNode**: Detail-level table nodes
- **CorpusNode**: Corpus-level structure nodes
- **TopicNode**: Topic-level structure nodes
- **SubtopicNode**: Subtopic-level structure nodes

### Embeddings
- **EmbeddingStructure**: Structure embeddings (CORPUS, TOPIC, SUBTOPIC)
- **EmbeddingContent**: Content embeddings (DETAIL chunks)

### Chat
- **ChatSession**: Chat conversation sessions
- **ChatMessage**: Individual messages in a chat session
- **Feedback**: User feedback on chat messages

### Mining
- **SuggestedFAQ**: Suggested FAQs from chat message clustering

## Conversion Summary

| Model | Pydantic → ORM | ORM → Pydantic |
|-------|----------------|----------------|
| KnowledgeNode subtypes | `node.to_orm()` | `ParagraphNode.from_orm(orm_node)` |
| User | `user.to_orm()` | `User.from_orm(orm_user)` |
| DataSource | `source.to_orm()` | `DataSource.from_orm(orm_source)` |
| EmbeddingStructure | `embedding.to_orm()` | `EmbeddingStructure.from_orm(orm_embedding)` |
| EmbeddingContent | `embedding.to_orm()` | `EmbeddingContent.from_orm(orm_embedding)` |
| ChatSession | `session.to_orm()` | `ChatSession.from_orm(orm_session)` |
| ChatMessage | `message.to_orm()` | `ChatMessage.from_orm(orm_message)` |
| Feedback | `feedback.to_orm()` | `Feedback.from_orm(orm_feedback)` |
| SuggestedFAQ | `faq.to_orm()` | `SuggestedFAQ.from_orm(orm_faq)` |

## Notes

- All Pydantic models have `to_orm()` and `from_orm()` methods
- ORM classes are plain SQLAlchemy classes without conversion methods
- KnowledgeNode internally uses converter utilities for subtype handling
- All conversions validate data through Pydantic
- Use handlers for CRUD operations instead of direct conversion
