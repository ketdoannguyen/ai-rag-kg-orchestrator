"""Create the local ArangoDB schema and MinIO buckets.

Reuses the application clients instead of re-declaring the schema, so the local
environment always matches what the code expects. Safe to run repeatedly: every
step checks for existing state before creating anything.

Usage:
    uv run python scripts/init_local_storage.py
"""

from __future__ import annotations

import argparse
import asyncio
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))

from arango_client import ArangoClient
from arango_client.collections import KNOWLEDGE_GRAPH_NAME
from minio_client import StorageClient

from config.settings import Settings

PLACEHOLDER_PREFIX = "<"
# ArangoDB adds these to every collection itself; the application declares neither.
IMPLICIT_INDEX_TYPES = {"primary", "edge"}


def require_value(value: str | None, name: str) -> str:
    """Fail fast when a credential is missing or still a template placeholder."""
    if not value or value.strip().startswith(PLACEHOLDER_PREFIX):
        raise SystemExit(
            f"{name} is not set. Copy .env.example to .env and replace every <placeholder>."
        )
    return value.strip()


def require(settings: Settings, field: str) -> str:
    """Read one settings field, failing fast when it is missing or a placeholder."""
    return require_value(getattr(settings, field), field)


def load_settings(env_file: Path) -> Settings:
    """Load settings from the repository-root env file (env vars still win)."""
    if not env_file.exists():
        raise SystemExit(f"{env_file} not found. Copy .env.example to .env first.")
    return Settings(_env_file=env_file)


def arango_client(settings: Settings, *, auto_init: bool) -> ArangoClient:
    """Build the application ArangoDB client from the environment contract."""
    return ArangoClient(
        url=require(settings, "ARANGO_URL"),
        database=require(settings, "ARANGO_DATABASE"),
        username=require(settings, "ARANGO_USER"),
        password=require(settings, "ARANGO_PASSWORD"),
        auto_init=auto_init,
    )


def storage_client(settings: Settings) -> StorageClient:
    """Build the application MinIO client from the environment contract."""
    return StorageClient(
        endpoint=require(settings, "MINIO_ENDPOINT"),
        access_key=require(settings, "MINIO_ACCESS_KEY"),
        secret_key=require(settings, "MINIO_SECRET_KEY"),
    )


def bucket_names(settings: Settings) -> list[str]:
    """The two buckets the application expects."""
    return [require(settings, "MINIO_BUCKET"), require(settings, "MINIO_KNOWLEDGE_BUCKET")]


def init_arango(settings: Settings) -> None:
    """Create the database, collections, indexes, graph and view if missing."""
    client = arango_client(settings, auto_init=True)
    try:
        db = client.db
        # python-arango renames ArangoDB's "isSystem" flag to "system".
        collections = [c["name"] for c in db.collections() if not c["system"]]
        indexes = sum(
            1
            for name in collections
            for index in db.collection(name).indexes()
            if index.get("type") not in IMPLICIT_INDEX_TYPES
        )
        views = [v["name"] for v in db.views()]
        graph = KNOWLEDGE_GRAPH_NAME if db.has_graph(KNOWLEDGE_GRAPH_NAME) else "MISSING"
        print(f"ArangoDB database   : {db.name}")
        print(f"ArangoDB collections: {len(collections)}")
        print(f"ArangoDB indexes    : {indexes}")
        print(f"ArangoDB graph      : {graph}")
        print(f"ArangoDB views      : {', '.join(views) if views else 'MISSING'}")
    finally:
        client.close()


async def init_minio(settings: Settings) -> None:
    """Create the two buckets the application expects, if they are missing."""
    client = storage_client(settings)
    buckets = bucket_names(settings)
    for name in buckets:
        await client.ensure_bucket(name)
    print(f"MinIO buckets       : {', '.join(buckets)}")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--env-file",
        type=Path,
        default=REPO_ROOT / ".env",
        help="env file to read (default: <repo>/.env)",
    )
    args = parser.parse_args()

    settings = load_settings(args.env_file)
    print(f"Initializing local storage from {args.env_file}")
    init_arango(settings)
    asyncio.run(init_minio(settings))
    print("Local storage is ready.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
