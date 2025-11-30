import asyncio
import os
from logging.config import fileConfig

from sqlalchemy import pool
from sqlalchemy.engine import Connection
from sqlalchemy.ext.asyncio import async_engine_from_config

from alembic import context

# 1. Import Base (chứa metadata các bảng) từ code của bạn
# Lưu ý: Phải set PYTHONPATH đúng hoặc cài đặt package ở chế độ editable
# Import trực tiếp từ package - __init__.py đã được sửa để tránh circular import
from postgres_client.orm_models.base import Base
# Import tất cả models để đảm bảo metadata được đăng ký
from postgres_client.orm_models import system_admin  # noqa: F401
from postgres_client.orm_models import knowledge  # noqa: F401
from postgres_client.orm_models import chat  # noqa: F401
from postgres_client.orm_models import mining  # noqa: F401

# this is the Alembic Config object, which provides
# access to the values within the .ini file in use.
config = context.config

# Interpret the config file for Python logging.
# This line sets up loggers basically.
if config.config_file_name is not None:
    fileConfig(config.config_file_name)

# 2. Gán Metadata của ORM cho Alembic
target_metadata = Base.metadata

# 3. Lấy Database URL từ biến môi trường (Bảo mật hơn để cứng trong ini)
# Ví dụ: postgresql+asyncpg://user:pass@localhost:5432/db
db_url = os.getenv("DB_URL")
if not db_url:
    raise ValueError("DB_URL environment variable is missing")

config.set_main_option("sqlalchemy.url", db_url)

# other values from the config, defined by the needs of env.py,
# can be acquired:
# my_important_option = config.get_main_option("my_important_option")
# ... etc.


def run_migrations_offline() -> None:
    """Run migrations in 'offline' mode.

    This configures the context with just a URL
    and not an Engine, though an Engine is acceptable
    here as well.  By skipping the Engine creation
    we don't even need a DBAPI to be available.

    Calls to context.execute() here emit the given string to the
    script output.

    """
    url = config.get_main_option("sqlalchemy.url")
    context.configure(
        url=url,
        target_metadata=target_metadata,
        literal_binds=True,
        dialect_opts={"paramstyle": "named"},
    )

    with context.begin_transaction():
        context.run_migrations()


def do_run_migrations(connection: Connection) -> None:
    context.configure(
        connection=connection,
        target_metadata=target_metadata,
        compare_type=True,  # Tự động phát hiện thay đổi kiểu cột (VARCHAR -> TEXT)
    )

    with context.begin_transaction():
        context.run_migrations()


async def run_async_migrations() -> None:
    """In this scenario we need to create an Engine
    and associate a connection with the context.

    """

    connectable = async_engine_from_config(
        config.get_section(config.config_ini_section, {}),
        prefix="sqlalchemy.",
        poolclass=pool.NullPool,
    )

    async with connectable.connect() as connection:
        await connection.run_sync(do_run_migrations)

    await connectable.dispose()


def run_migrations_online() -> None:
    """Run migrations in 'online' mode."""

    asyncio.run(run_async_migrations())


if context.is_offline_mode():
    run_migrations_offline()
else:
    run_migrations_online()
