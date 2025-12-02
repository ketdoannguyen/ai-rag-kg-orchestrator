# ORM Models

SQLAlchemy ORM classes that map directly to the database schema.

## Structure

```
orm_models/
├── __init__.py          # Exports all ORM classes
├── base.py              # Base class for all ORM models
├── system_admin.py      # Users, DataSources
├── knowledge.py         # KnowledgeNodes, Embeddings
├── chat.py              # ChatSessions, ChatMessages, Feedbacks
└── mining.py            # SuggestedFAQs
```

## ORM Classes

### System Admin (`system_admin.py`)
- **UserORM**: `users` table
  - User accounts and authentication
  - Relationships: `created_sources`, `reviewed_feedbacks`
  
- **DataSourceORM**: `data_sources` table
  - Source documents/files
  - Relationships: `creator`, `nodes`

### Knowledge (`knowledge.py`)
- **KnowledgeNodeORM**: `knowledge_nodes` table
  - Core knowledge graph nodes
  - Classification fields: `node_type`, `level_down`, `level_up`  
  - Relationships: `children`, `parent`, `source`, `structure_embedding`, `content_embedding`

- **EmbeddingStructureORM**: `embeddings_structure` table
  - Embeddings cho các node cấu trúc (StructureNode)  
  - Vector dimension: 1024

- **EmbeddingContentORM**: `embeddings_content` table
  - Embeddings cho các node nội dung chi tiết (Paragraph/Image/Table)  
  - Vector dimension: 1024

### Chat (`chat.py`)
- **ChatSessionORM**: `chat_sessions` table
  - Chat conversation sessions
  - Relationships: `messages`

- **ChatMessageORM**: `chat_messages` table
  - Individual messages in a chat session
  - Relationships: `session`, `feedbacks`
  - Constraint: `role IN ('user', 'assistant')`

- **FeedbackORM**: `feedbacks` table
  - User feedback on chat messages
  - Relationships: `message`, `reviewer`

### Mining (`mining.py`)
- **SuggestedFAQORM**: `suggested_faqs` table
  - Suggested FAQs from chat message clustering
  - Vector dimension: 1024 (cluster_vector)

## Usage

### Import ORM Classes

```python
from postgres_client.orm_models import (
    Base,
    UserORM,
    DataSourceORM,
    KnowledgeNodeORM,
    EmbeddingStructureORM,
    EmbeddingContentORM,
    ChatSessionORM,
    ChatMessageORM,
    FeedbackORM,
    SuggestedFAQORM,
)
```

### Create Tables

```python
from postgres_client.orm_models import Base
from sqlalchemy import create_engine

engine = create_engine("postgresql://...")
Base.metadata.create_all(engine)
```

## Notes

- Use `node_metadata` attribute instead of `metadata` (reserved in SQLAlchemy)
- Database column name remains `metadata`, but ORM attribute is `node_metadata`
- All relationships are properly configured with cascade options
