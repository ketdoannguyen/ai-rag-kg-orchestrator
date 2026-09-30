"""Foundation of the OCR research pipeline: corpus, gold, and artifact store.

The names re-exported here are the supported interface; the submodules hold the
detail. Slice S1 defines the persisted schemas, the manifest/checksum/page
validation, gold references, and the immutable content-addressed artifact store.
Extraction adapters and report metrics are built on top of it in later slices.
"""

from .corpus import (
    CorpusError,
    DocumentReport,
    load_manifest,
    resolve_path,
    sha256_file,
    validate_document,
    validate_manifest,
    verify_artifact,
)
from .models import (
    ARTIFACT_VERSION,
    ENGINE_IDS,
    MAPPING_VERSION,
    NORMALIZED_KINDS,
    ArtifactBlock,
    ArtifactPage,
    ArtifactStatistics,
    DocumentSpec,
    ExtractionArtifact,
    Manifest,
    Modality,
    PageUsage,
    canonical_payload,
    page_text,
    semantic_fingerprint,
)
from .reference import (
    AlignedPage,
    CoverageReport,
    GoldError,
    GoldPage,
    GoldReference,
    align_pages,
    coverage,
    load_gold,
    metrics_document,
)
from .store import (
    ArtifactIntegrityError,
    ArtifactNotFound,
    ArtifactStore,
    ArtifactStoreError,
    write_once,
)

__all__ = [
    # models
    "ARTIFACT_VERSION",
    "ENGINE_IDS",
    "MAPPING_VERSION",
    "NORMALIZED_KINDS",
    "ArtifactBlock",
    "ArtifactPage",
    "ArtifactStatistics",
    "DocumentSpec",
    "ExtractionArtifact",
    "Manifest",
    "Modality",
    "PageUsage",
    "canonical_payload",
    "page_text",
    "semantic_fingerprint",
    # corpus
    "CorpusError",
    "DocumentReport",
    "load_manifest",
    "resolve_path",
    "sha256_file",
    "validate_document",
    "validate_manifest",
    "verify_artifact",
    # reference
    "AlignedPage",
    "CoverageReport",
    "GoldError",
    "GoldPage",
    "GoldReference",
    "align_pages",
    "coverage",
    "load_gold",
    "metrics_document",
    # store
    "ArtifactIntegrityError",
    "ArtifactNotFound",
    "ArtifactStore",
    "ArtifactStoreError",
    "write_once",
]
