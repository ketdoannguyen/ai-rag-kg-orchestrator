"""update_vector_dimension_1536

Revision ID: update_vector_1536
Revises: 5c4ea5c650cf
Create Date: 2025-12-05 19:12:41.000000

"""
from typing import Sequence, Union

from alembic import op
import sqlalchemy as sa
import pgvector.sqlalchemy

# revision identifiers, used by Alembic.
revision: str = 'update_vector_1536'
down_revision: Union[str, Sequence[str], None] = '5c4ea5c650cf'
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    """Upgrade schema: Change vector dimensions from 1024 to 1536.
    
    Note: This migration will clear existing vector data as dimensions change.
    Embeddings need to be regenerated with the new embedding model (1536 dimensions).
    """
    # Update embeddings_structure.vector
    # pgvector doesn't support direct dimension change, so we drop and recreate
    op.execute("ALTER TABLE embeddings_structure DROP COLUMN IF EXISTS vector")
    op.execute("ALTER TABLE embeddings_structure ADD COLUMN vector vector(1536)")
    
    # Update embeddings_content.vector
    op.execute("ALTER TABLE embeddings_content DROP COLUMN IF EXISTS vector")
    op.execute("ALTER TABLE embeddings_content ADD COLUMN vector vector(1536)")
    
    # Update suggested_faqs.cluster_vector
    op.execute("ALTER TABLE suggested_faqs DROP COLUMN IF EXISTS cluster_vector")
    op.execute("ALTER TABLE suggested_faqs ADD COLUMN cluster_vector vector(1536)")


def downgrade() -> None:
    """Downgrade schema: Change vector dimensions back from 1536 to 1024.
    
    Note: This migration will clear existing vector data as dimensions change.
    """
    # Update embeddings_structure.vector
    op.execute("ALTER TABLE embeddings_structure DROP COLUMN IF EXISTS vector")
    op.execute("ALTER TABLE embeddings_structure ADD COLUMN vector vector(1024)")
    
    # Update embeddings_content.vector
    op.execute("ALTER TABLE embeddings_content DROP COLUMN IF EXISTS vector")
    op.execute("ALTER TABLE embeddings_content ADD COLUMN vector vector(1024)")
    
    # Update suggested_faqs.cluster_vector
    op.execute("ALTER TABLE suggested_faqs DROP COLUMN IF EXISTS cluster_vector")
    op.execute("ALTER TABLE suggested_faqs ADD COLUMN cluster_vector vector(1024)")

