#!/usr/bin/env python3
"""Seed ontology relations into ArangoDB with embeddings.

This script:
1. Reads all relation types from config/ontologies/ dynamically
2. Computes embeddings for each relation (name + description)
3. Stores them in the ontology_relations collection

Usage:
    python scripts/seed_ontology.py
    python scripts/seed_ontology.py --overwrite  # Overwrite existing relations
    python scripts/seed_ontology.py --ontology education  # Specific ontology only
"""

from __future__ import annotations

import argparse
import asyncio
import re
import sys
import unicodedata
from pathlib import Path

# Add project root to path FIRST
# This allows running the script directly from the project root
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from openai import AsyncOpenAI

from config.ontologies import (
    get_ontology,
    list_ontologies,
)
from config.settings import settings
from libs.arango_client.src.arango_client.client import ArangoClient
from libs.arango_client.src.arango_client.handlers.ontology_handler import OntologyHandler


def sanitize_relation_id(rel_id: str) -> str:
    """Sanitize relation ID to be a valid ArangoDB document key.

    ArangoDB keys can only contain: a-z, A-Z, 0-9, _ (underscore), - (dash), : (colon)
    """
    # Normalize unicode (NFKD decomposition)
    normalized = unicodedata.normalize("NFKD", rel_id)
    # Remove diacritics (combining characters)
    ascii_text = normalized.encode("ascii", "ignore").decode("ascii")
    # Replace spaces with underscores
    ascii_text = ascii_text.replace(" ", "_")
    # Remove any remaining invalid characters (keep only alphanumeric, _, -, :)
    sanitized = re.sub(r"[^a-zA-Z0-9_\-:]", "", ascii_text)
    # Ensure it's not empty
    return sanitized or "UNKNOWN_RELATION"


def build_relation_data_from_constraint(constraint) -> dict:
    """Build relation data dict from RelationConstraint.

    Uses the description field directly from the constraint.
    """
    rel_type = constraint.relation_type
    rel_id_raw = rel_type.value if hasattr(rel_type, "value") else str(rel_type)
    rel_id = sanitize_relation_id(rel_id_raw)

    return {
        "id": rel_id,
        "name": rel_id_raw,  # Keep original name with Vietnamese characters
        "description": constraint.description,
        "source_node_types": constraint.source_node_types,
        "target_node_types": constraint.target_node_types,
        "is_symmetric": constraint.is_symmetric,
        "is_transitive": constraint.is_transitive,
        "tiers": constraint.tiers,  # Include tiers from constraint
    }


def collect_relations_from_ontology(ontology_name: str) -> list[dict]:
    """Collect all relations from a specific ontology.

    Args:
        ontology_name: Name of the ontology (e.g., "base", "education", "entity")

    Returns:
        List of relation dicts with domain set
    """
    ontology = get_ontology(ontology_name)
    relations = []

    for constraint in ontology.relations:
        rel_data = build_relation_data_from_constraint(constraint)
        rel_data["domain"] = ontology_name
        relations.append(rel_data)

    return relations


def collect_all_relations() -> dict[str, list[dict]]:
    """Collect relations from all registered ontologies.

    Returns:
        Dict mapping ontology name to list of relations
    """
    all_relations: dict[str, list[dict]] = {}
    seen_ids: set[str] = set()

    for ontology_name in list_ontologies():
        ontology = get_ontology(ontology_name)
        relations = []

        for constraint in ontology.relations:
            rel_data = build_relation_data_from_constraint(constraint)
            rel_id = rel_data["id"]

            # Skip duplicates (e.g., base relations included in education)
            if rel_id in seen_ids:
                continue

            seen_ids.add(rel_id)
            rel_data["domain"] = ontology_name
            relations.append(rel_data)

        if relations:
            all_relations[ontology_name] = relations

    return all_relations


async def compute_embeddings(
    relations: list[dict],
    model: str = "text-embedding-3-large",
) -> list[dict]:
    """Compute embeddings for all relations."""
    # Use OpenRouter for embeddings directly (bypass local proxy)
    api_key = settings.LLM_API_KEY

    # Strip "test#" prefix if present (used in test environments)
    if api_key and api_key.startswith("test#"):
        api_key = api_key[5:]

    # OpenRouter requires /api/v1 for embeddings
    client = AsyncOpenAI(api_key=api_key, base_url="https://openrouter.ai/api/v1")

    # Prepare texts
    texts = [f"{r['name']}: {r['description']}" for r in relations]

    print(f"Computing embeddings for {len(texts)} relations...")

    # Call API
    response = await client.embeddings.create(
        model=f"openai/{model}",
        input=texts,
    )

    # Add embeddings to relations
    for i, item in enumerate(response.data):
        relations[i]["embedding"] = item.embedding

    print(f"✓ Computed {len(relations)} embeddings (dimension: {len(response.data[0].embedding)})")

    return relations


async def main():
    """Main entry point."""
    parser = argparse.ArgumentParser(description="Seed ontology relations to ArangoDB")
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Overwrite existing relations",
    )
    parser.add_argument(
        "--clear",
        action="store_true",
        help="Clear all existing relations before seeding (delete and recreate collection)",
    )
    parser.add_argument(
        "--skip-embeddings",
        action="store_true",
        help="Skip computing embeddings (for testing)",
    )
    parser.add_argument(
        "--ontology",
        type=str,
        default=None,
        help=f"Specific ontology to seed (available: {', '.join(list_ontologies())})",
    )
    parser.add_argument(
        "--list",
        action="store_true",
        help="List available ontologies and exit",
    )
    args = parser.parse_args()

    # List ontologies and exit
    if args.list:
        print("Available ontologies:")
        for name in list_ontologies():
            ontology = get_ontology(name)
            print(f"  - {name}: {ontology.display_name} ({len(ontology.relations)} relations)")
        return

    print("=" * 60)
    print("SEED ONTOLOGY RELATIONS")
    print("=" * 60)

    # Collect relations from config
    if args.ontology:
        # Specific ontology only
        if args.ontology not in list_ontologies():
            print(f"Error: Unknown ontology '{args.ontology}'")
            print(f"Available: {', '.join(list_ontologies())}")
            return
        ontology_relations = {args.ontology: collect_relations_from_ontology(args.ontology)}
    else:
        # All ontologies (deduplicated)
        ontology_relations = collect_all_relations()

    # Print summary
    total_count = 0
    for ontology_name, relations in ontology_relations.items():
        print(f"\n{ontology_name}: {len(relations)} relations")
        total_count += len(relations)

    print(f"\nTotal unique relations: {total_count}")

    # Flatten all relations for embedding computation
    all_relations: list[dict] = []
    for relations in ontology_relations.values():
        all_relations.extend(relations)

    # Compute embeddings
    if not args.skip_embeddings:
        all_relations = await compute_embeddings(all_relations)
    else:
        print("Skipping embedding computation...")

    # Connect to ArangoDB
    print("\nConnecting to ArangoDB...")
    print(f"  URL: {settings.ARANGO_URL}")
    print(f"  Database: {settings.ARANGO_DATABASE}")
    print(f"  User: {settings.ARANGO_USER}")

    arango_client = ArangoClient(
        url=settings.ARANGO_URL,
        database=settings.ARANGO_DATABASE,
        username=settings.ARANGO_USER,
        password=settings.ARANGO_PASSWORD,
    )
    db = arango_client.db

    # Create handler and seed
    handler = OntologyHandler(db)

    # Clear collection if requested
    if args.clear:
        print("\nClearing existing ontology_relations collection...")
        collection_name = handler.collection_name
        if db.has_collection(collection_name):
            db.delete_collection(collection_name)
            print(f"  ✓ Deleted collection: {collection_name}")
        db.create_collection(collection_name)
        print(f"  ✓ Created fresh collection: {collection_name}")

    # Seed relations by domain
    total_seeded = 0
    for ontology_name in ontology_relations:
        print(f"\nSeeding {ontology_name} relations...")
        domain_relations = [r for r in all_relations if r["domain"] == ontology_name]
        count = handler.seed_from_ontology(
            domain_relations, domain=ontology_name, overwrite=args.overwrite
        )
        print(f"  ✓ Seeded {count} {ontology_name} relations")
        total_seeded += count

    # Summary
    print("\n" + "=" * 60)
    print("SUMMARY")
    print("=" * 60)
    print(f"Total seeded: {total_seeded} relations")

    # Verify
    all_active = handler.get_all_active()
    with_embeddings = handler.get_all_with_embeddings()
    print(f"Active relations in DB: {len(all_active)}")
    print(f"Relations with embeddings: {len(with_embeddings)}")

    if with_embeddings:
        print("\nSample relation:")
        sample = with_embeddings[0]
        print(f"  ID: {sample.id}")
        print(f"  Name: {sample.name}")
        print(f"  Description: {sample.description[:80]}...")
        print(f"  Embedding dim: {len(sample.embedding) if sample.embedding else 0}")
        print(f"  Domain: {sample.domain}")

    print("\n✓ Done!")


if __name__ == "__main__":
    asyncio.run(main())
