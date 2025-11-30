# Alembic Migrations

Guide for managing database migrations using Alembic with async support.

## Setup

### Environment Variables

```bash
# In .env file
DATABASE_URL=postgresql+psycopg://user:password@localhost:5432/db

# Or set DB_URL for Alembic (async)
DB_URL=postgresql+asyncpg://user:password@localhost:5432/db
```

**Note**: 
- `DATABASE_URL` uses `postgresql+psycopg://` (for application)
- `DB_URL` uses `postgresql+asyncpg://` (for Alembic async)
- Makefile automatically converts `DATABASE_URL` to `DB_URL` if needed

## Daily Workflow

### 1. Modify ORM Models

Edit ORM models in `src/postgres_client/orm_models/`:

```python
class UserORM(Base):
    # ... existing columns ...
    phone: Mapped[str | None] = mapped_column(String(20), nullable=True)
```

### 2. Generate Migration

```bash
make db-gen msg="add phone column to user"
```

Alembic will:
- Compare ORM models with current database
- Create new migration file in `migrations/versions/`
- Format: `xxxx_add_phone_column_to_user.py`

### 3. Review Migration

**IMPORTANT**: Always review migration files before running!

```bash
cat libs/postgres_client/migrations/versions/xxxx_add_phone_column_to_user.py
```

Check:
- ✅ Changes are correct
- ✅ No unwanted changes
- ✅ Constraints and indexes are correct

### 4. Apply Migration

```bash
make db-up
```

This will:
- Connect to database
- Run all pending migrations
- Update database schema

### 5. Rollback (if needed)

```bash
make db-down  # Rollback one step
```

## Makefile Commands

### `make db-gen msg="message"`
Generate new migration file from ORM model changes.

**Example:**
```bash
make db-gen msg="add phone column to user"
make db-gen msg="create feedbacks table"
```

**Note**: `msg` parameter is required.

### `make db-up`
Apply all pending migrations, update database to latest version.

### `make db-down`
Rollback one migration step.

### `make db-history`
View migration history.

### `make db-current`
View current database revision.

## Troubleshooting

### Error: "DB_URL environment variable is missing"

**Solution**:
```bash
export DB_URL=postgresql+asyncpg://user:password@localhost:5432/db
# Or
export DATABASE_URL=postgresql+psycopg://user:password@localhost:5432/db
```

### Error: "Cannot import postgres_client.orm_models"

**Solution**: Makefile automatically sets `PYTHONPATH=src`. If running alembic directly:

```bash
cd libs/postgres_client
PYTHONPATH=src DB_URL=postgresql+asyncpg://... alembic ...
```

### Error: "Target database is not up to date"

**Solution**: Run migrations:
```bash
make db-up
```

### Migration doesn't detect changes

**Solution**: 
1. Check `migrations/env.py` imports all models
2. Ensure all ORM models are imported in `env.py`

## Best Practices

1. **Always review migration files** before running `db-up`
2. **Commit migration files** to git with code changes
3. **Test migrations** on development database before production
4. **Backup database** before running migrations on production
5. **One migration = one logical change**: Avoid large migrations, split them
6. **Clear migration names**: Use descriptive names in English or Vietnamese

## Important Notes

1. **Database column name vs ORM attribute name**: 
   - ORM uses `node_metadata` (because `metadata` is reserved in SQLAlchemy)
   - Database column name is still `metadata`
   - Use `orm_instance.node_metadata` instead of `orm_instance.metadata`

2. **Async support**: 
   - Alembic is configured to run async with `asyncpg`
   - All migrations run in async context

3. **Environment variables**:
   - Makefile automatically loads from `.env` if available
   - Can override by setting directly in shell
