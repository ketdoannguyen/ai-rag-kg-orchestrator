"""Verify the local ArangoDB schema and MinIO buckets end to end.

Read-only against ArangoDB. MinIO gets one temporary object that is always
removed, so a successful run leaves both stores exactly as it found them.

Usage:
    uv run python scripts/verify_local_storage.py
"""

from __future__ import annotations

import argparse
import asyncio
import sys
import tempfile
import urllib.request
import uuid
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))

from arango_client import ArangoClient
from arango_client.collections import (
    ALL_DOCUMENT_COLLECTIONS,
    ALL_EDGE_COLLECTIONS,
    INDEXES,
    KNOWLEDGE_COLLECTIONS,
    KNOWLEDGE_GRAPH_DEFINITION,
    KNOWLEDGE_GRAPH_NAME,
    KNOWLEDGE_VECTOR_VIEW_NAME,
)

from config.settings import Settings
from scripts.init_local_storage import (
    arango_client,
    bucket_names,
    load_settings,
    require,
    storage_client,
)

EXPECTED_ARANGO_VERSION = "3.12.12"
# ArangoDB publishes no Community container: the official `arangodb` image is the
# Enterprise build. Unlicensed it stays usable below a disk limit, so the gate
# checks that state instead of asserting an edition that does not exist.
EXPECTED_ARANGO_LICENSE = "enterprise"
MINIO_HEALTH_PATHS = ("/minio/health/live", "/minio/health/ready")
VERIFY_PREFIX = "verify/"
EXPECTED_KINDS = dict.fromkeys(ALL_DOCUMENT_COLLECTIONS, "document") | dict.fromkeys(
    ALL_EDGE_COLLECTIONS, "edge"
)


class Checks:
    """Print each result and keep the failures for the final summary."""

    def __init__(self) -> None:
        self.failures: list[str] = []

    def check(self, ok: bool, message: str) -> bool:
        print(f"[{'PASS' if ok else 'FAIL'}] {message}")
        if not ok:
            self.failures.append(message)
        return ok

    def note(self, message: str) -> None:
        print(f"[INFO] {message}")


def check_license(db, checks: Checks) -> None:
    """Unlicensed Enterprise stays usable below its disk limit; assert that state."""
    try:
        usage = db.license()["diskUsage"]
        seconds_left = usage.get("secondsUntilReadOnly", 0)
        checks.check(
            usage.get("status") == "good" and not usage.get("limitReached") and seconds_left > 0,
            f"ArangoDB license usable: status={usage.get('status')} "
            f"limit={usage.get('bytesLimit', 0) / 1024**3:.0f}GiB readOnlyIn={seconds_left / 86400:.0f}d",
        )
    except Exception as exc:  # reported as a failed check instead of aborting the run
        checks.check(False, f"ArangoDB license usable: {type(exc).__name__}: {exc}")


def index_matches(index: dict | None, spec) -> bool:
    """Compare a live index against its declared fields, uniqueness and sparsity."""
    return index is not None and (
        sorted(index.get("fields", [])) == sorted(spec.fields)
        and bool(index.get("unique", False)) == spec.unique
        and bool(index.get("sparse", False)) == spec.sparse
    )


def edge_signature(definition: dict) -> tuple:
    """Normalise an edge definition so two definitions compare directly."""
    return (
        definition["edge_collection"],
        sorted(definition.get("from_vertex_collections", [])),
        sorted(definition.get("to_vertex_collections", [])),
    )


def check_server(db, checks: Checks, expected_version: str, expected_license: str) -> None:
    """Assert the server version and edition, and that the license is usable."""
    # db.version() returns a bare string unless details are requested.
    version = db.version(details=True)
    server = version.get("version", "unknown")
    license_name = str(version.get("license", "unknown")).lower()
    checks.check(
        server == expected_version, f"ArangoDB version: {server} (expected {expected_version})"
    )
    checks.check(
        license_name == expected_license,
        f"ArangoDB edition: {license_name} (expected {expected_license})",
    )
    if license_name == "enterprise":
        check_license(db, checks)


def check_collections(db, checks: Checks) -> None:
    """Assert exactly the declared collections exist, are empty, and have the right kind."""
    # python-arango renames ArangoDB's "isSystem" flag to "system" and maps the
    # numeric collection type to the strings "document" and "edge". Comparing the
    # whole mapping catches a missing, unexpected or wrongly-typed collection.
    actual = {c["name"]: c["type"] for c in db.collections() if not c["system"]}
    wrong: dict[str, str] = {}
    for name, kind in EXPECTED_KINDS.items():
        found = actual.get(name)
        if found != kind:
            wrong[name] = "missing" if found is None else f"{found}, expected {kind}"
    wrong |= dict.fromkeys(actual.keys() - EXPECTED_KINDS.keys(), "unexpected")
    checks.check(
        not wrong,
        f"Collections ({len(EXPECTED_KINDS)} expected, {len(actual)} found): {wrong or 'none'}",
    )

    counts = {name: db.collection(name).count() for name in actual}
    non_empty = {name: count for name, count in counts.items() if count}
    checks.check(not non_empty, f"Collections are empty: non-empty={non_empty or 'none'}")


def check_indexes(db, checks: Checks) -> None:
    """Assert every declared index exists with matching fields, uniqueness and sparsity."""
    live = {
        (spec.collection, index["name"]): index
        for spec in INDEXES
        if db.has_collection(spec.collection)
        for index in db.collection(spec.collection).indexes()
    }
    bad = [
        spec.name
        for spec in INDEXES
        if not index_matches(live.get((spec.collection, spec.name)), spec)
    ]
    checks.check(not bad, f"Indexes ({len(INDEXES)}): missing or mismatched={bad or 'none'}")


def check_graph(db, checks: Checks) -> None:
    """Assert the graph carries exactly the declared edge definitions."""
    # db.graph().edge_definitions() is a method.
    edge_defs = (
        db.graph(KNOWLEDGE_GRAPH_NAME).edge_definitions()
        if db.has_graph(KNOWLEDGE_GRAPH_NAME)
        else []
    )
    checks.check(
        sorted(map(edge_signature, edge_defs))
        == sorted(map(edge_signature, KNOWLEDGE_GRAPH_DEFINITION["edge_definitions"])),
        f"Graph ({KNOWLEDGE_GRAPH_NAME}): edges={[d['edge_collection'] for d in edge_defs] or 'none'}",
    )


def check_view(db, checks: Checks) -> None:
    """Assert the vector view exists and links the knowledge collections."""
    # db.view() returns the formatted properties dict, not a View object.
    linked = (
        sorted(db.view(KNOWLEDGE_VECTOR_VIEW_NAME).get("links", {}))
        if KNOWLEDGE_VECTOR_VIEW_NAME in [v["name"] for v in db.views()]
        else []
    )
    checks.check(
        linked == sorted(KNOWLEDGE_COLLECTIONS),
        f"View ({KNOWLEDGE_VECTOR_VIEW_NAME}): links={linked or 'none'}",
    )


def check_similarity(db, checks: Checks) -> None:
    """Assert COSINE_SIMILARITY() evaluates, without inserting any data."""
    try:
        similarity = list(
            db.aql.execute(
                "RETURN COSINE_SIMILARITY(@a, @b)", bind_vars={"a": [1, 0], "b": [1, 0]}
            )
        )[0]
        checks.check(
            abs(float(similarity) - 1.0) < 1e-9,
            f"COSINE_SIMILARITY() available: result={similarity}",
        )
    except Exception as exc:  # reported as a failed check instead of aborting the run
        checks.check(False, f"COSINE_SIMILARITY() available: {type(exc).__name__}: {exc}")


def note_index_types(db, checks: Checks) -> None:
    """Report which index types exist, so no ANN capability is implied."""
    index_types = sorted(
        {
            index.get("type", "unknown")
            for collection in KNOWLEDGE_COLLECTIONS
            if db.has_collection(collection)
            for index in db.collection(collection).indexes()
        }
    )
    checks.note(f"ArangoDB index types: {', '.join(index_types)}")
    if "vector" not in index_types:
        checks.note(
            "Vector capability: no native vector index, so retrieval is a brute-force "
            "COSINE_SIMILARITY() scan and no ANN speedup is claimed"
        )


def verify_arango(
    client: ArangoClient, checks: Checks, expected_version: str, expected_license: str
) -> None:
    """Run every ArangoDB check against the configured database."""
    db = client.db
    check_server(db, checks, expected_version, expected_license)
    checks.check(db.name == client.database_name, f"ArangoDB database: {db.name}")
    check_collections(db, checks)
    check_indexes(db, checks)
    check_graph(db, checks)
    check_view(db, checks)
    check_similarity(db, checks)
    note_index_types(db, checks)


def check_health(checks: Checks, endpoint: str) -> bool:
    """Probe both MinIO health endpoints, report each, and return overall status."""
    base = endpoint.rstrip("/")
    healthy = True
    for path in MINIO_HEALTH_PATHS:
        try:
            with urllib.request.urlopen(f"{base}{path}", timeout=10) as response:
                status = response.status
        except OSError as exc:  # includes urllib.error.URLError
            status = f"error: {exc}"
        healthy = checks.check(status == 200, f"MinIO {path}: status={status}") and healthy
    return healthy


async def verify_minio(settings: Settings, checks: Checks) -> None:
    """Check both buckets exist and are empty, then round-trip one temporary object."""
    client = storage_client(settings)
    buckets = bucket_names(settings)

    if not check_health(checks, require(settings, "MINIO_ENDPOINT")):
        checks.note("MinIO integration skipped: the endpoint did not answer its health probes")
        return

    exists = {name: await client.bucket_exists(name) for name in buckets}
    for name, present in exists.items():
        checks.check(present, f"MinIO bucket '{name}' exists")
        if present:
            objects = await client.list_objects(name)
            checks.check(not objects, f"MinIO bucket '{name}' is empty: objects={len(objects)}")

    probe_bucket = buckets[0]
    if not exists[probe_bucket]:
        checks.note("MinIO round-trip skipped: the probe bucket does not exist")
        return

    object_name = f"{VERIFY_PREFIX}{uuid.uuid4()}.txt"
    payload = f"verify-local-storage {uuid.uuid4()}".encode()
    try:
        await client.upload_bytes(probe_bucket, object_name, payload, content_type="text/plain")
        size = (await client.stat_object(probe_bucket, object_name)).size
        checks.check(size == len(payload), f"MinIO upload size: {size} bytes")
        with tempfile.TemporaryDirectory() as tmp_dir:
            probe_path = Path(tmp_dir) / "probe.txt"
            await client.download_file(probe_bucket, object_name, probe_path)
            checks.check(
                probe_path.read_bytes() == payload,
                f"MinIO download round-trip: bucket={probe_bucket}",
            )
    except Exception as exc:  # reported as a failed check instead of aborting the run
        checks.check(False, f"MinIO upload/download round-trip: {type(exc).__name__}: {exc}")
    finally:
        # The probe bucket exists by now, so the whole prefix can be cleared outright.
        try:
            for leftover in await client.list_objects(probe_bucket, prefix=VERIFY_PREFIX):
                await client.delete_object(probe_bucket, leftover)
            remaining = await client.list_objects(probe_bucket, prefix=VERIFY_PREFIX)
            checks.check(not remaining, f"MinIO probe cleanup: leftovers={remaining or 'none'}")
        except Exception as exc:  # reported as a failed check instead of aborting the run
            checks.check(False, f"MinIO probe cleanup: {type(exc).__name__}: {exc}")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--env-file", type=Path, default=REPO_ROOT / ".env", help="env file to read"
    )
    parser.add_argument(
        "--expect-arango-version",
        default=EXPECTED_ARANGO_VERSION,
        help=f"expected ArangoDB server version (default: {EXPECTED_ARANGO_VERSION})",
    )
    parser.add_argument(
        "--expect-license",
        default=EXPECTED_ARANGO_LICENSE,
        help=f"expected ArangoDB edition (default: {EXPECTED_ARANGO_LICENSE})",
    )
    args = parser.parse_args()

    settings = load_settings(args.env_file)
    checks = Checks()
    client = arango_client(settings, auto_init=False)
    try:
        verify_arango(client, checks, args.expect_arango_version, args.expect_license)
    except Exception as exc:  # reported as a failed check instead of aborting the run
        checks.check(False, f"ArangoDB reachable: {type(exc).__name__}: {exc}")
    finally:
        client.close()

    asyncio.run(verify_minio(settings, checks))

    if not checks.failures:
        print("\nAll local storage checks passed.")
        return 0
    print(f"\n{len(checks.failures)} check(s) failed:")
    for failure in checks.failures:
        print(f"  - {failure}")
    return 1


if __name__ == "__main__":
    raise SystemExit(main())
