"""Benchmark script for evaluating EXTRACTION_AUTO_MERGE_THRESHOLD and EXTRACTION_LLM_THRESHOLD.

This script:
1. Loads 100 sample node pairs from samples.py
2. Computes embedding similarity for each pair using real embeddings
3. Evaluates different threshold combinations
4. Generates a detailed analysis report with optimal thresholds

Usage:
    python benchmarks/node_similarity_threshold/test_threshold.py
    python benchmarks/node_similarity_threshold/test_threshold.py --verbose
    python benchmarks/node_similarity_threshold/test_threshold.py --no-cache
"""

from __future__ import annotations

import argparse
import asyncio
import json
import sys
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path

import numpy as np

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from samples import (
    ALL_SAMPLES,
    ExpectedZone,
    NodeCategory,
    NodePairSample,
    get_boundary_cases,
    get_negative_samples,
    get_positive_samples,
    get_samples_by_category,
    get_samples_by_zone,
)


@dataclass
class ThresholdResult:
    """Result of evaluating a threshold pair."""

    auto_merge_threshold: float
    llm_threshold: float

    # Auto-merge metrics
    auto_merge_precision: float
    auto_merge_recall: float
    auto_merge_f1: float
    auto_merge_tp: int
    auto_merge_fp: int  # CRITICAL: merged when shouldn't
    auto_merge_fn: int

    # LLM zone metrics
    llm_zone_coverage: float  # % of pairs sent to LLM
    llm_zone_efficiency: float  # % of LLM pairs that are boundary cases

    # Separate zone metrics
    separate_precision: float
    separate_recall: float
    separate_tn: int
    separate_fp: int  # kept separate when should merge

    # Overall
    total_accuracy: float


@dataclass
class SampleResult:
    """Result for a single sample."""

    sample_id: int
    similarity: float
    ground_truth: bool  # should_merge
    category: str
    expected_zone: str


def compute_cosine_similarity(vec_a: list[float], vec_b: list[float]) -> float:
    """Compute cosine similarity between two vectors."""
    a = np.array(vec_a)
    b = np.array(vec_b)
    dot_product = np.dot(a, b)
    norm_a = np.linalg.norm(a)
    norm_b = np.linalg.norm(b)
    if norm_a == 0 or norm_b == 0:
        return 0.0
    return float(dot_product / (norm_a * norm_b))


async def embed_batch(
    client,
    model: str,
    texts: list[str],
    batch_idx: int,
    verbose: bool = False,
) -> tuple[int, list[list[float]]]:
    """Embed a single batch of texts. Returns (batch_idx, embeddings)."""
    if verbose:
        print(f"  Starting batch {batch_idx + 1}...")

    response = await client.embeddings.create(
        model=model,
        input=texts,
    )
    embeddings = [item.embedding for item in response.data]

    if verbose:
        print(f"  Completed batch {batch_idx + 1} ({len(embeddings)} embeddings)")

    return batch_idx, embeddings


async def compute_similarities(
    samples: list[NodePairSample],
    verbose: bool = False,
    max_concurrent: int = 5,
) -> list[tuple[int, float]]:
    """Compute similarity scores for all samples using PARALLEL async embeddings."""
    import os
    import time

    from dotenv import load_dotenv
    from openai import AsyncOpenAI

    # Load .env file
    load_dotenv()

    # Use OpenAI API directly
    api_key = os.getenv("EMBEDDING_API_KEY") or os.getenv("LLM_EMBEDDING_API_KEY") or os.getenv("LLM_API_KEY")
    api_base = os.getenv("EMBEDDING_API_BASE") or os.getenv("LLM_EMBEDDING_API_BASE") or "https://api.openai.com/v1"
    model = os.getenv("EMBEDDING_MODEL") or os.getenv("LLM_EMBEDDING_MODEL") or "text-embedding-3-small"

    client = AsyncOpenAI(api_key=api_key, base_url=api_base)

    total = len(samples)

    # Collect all texts
    print(f"\nEmbedding {total * 2} texts in PARALLEL...")
    print(f"Using model: {model}")
    print(f"API base: {api_base}")
    print(f"Max concurrent requests: {max_concurrent}")

    all_texts = []
    for sample in samples:
        all_texts.extend([sample.node_a_text, sample.node_b_text])

    # Split into batches
    batch_size = 20
    batches = []
    for i in range(0, len(all_texts), batch_size):
        batches.append(all_texts[i : i + batch_size])

    total_batches = len(batches)
    print(f"Total batches: {total_batches}")

    start_time = time.time()

    # Process batches in parallel with semaphore to limit concurrency
    semaphore = asyncio.Semaphore(max_concurrent)

    async def embed_with_semaphore(batch_idx: int, batch: list[str]) -> tuple[int, list[list[float]]]:
        async with semaphore:
            return await embed_batch(client, model, batch, batch_idx, verbose)

    # Launch all tasks in parallel
    tasks = [
        embed_with_semaphore(idx, batch)
        for idx, batch in enumerate(batches)
    ]

    # Wait for all to complete
    batch_results = await asyncio.gather(*tasks)

    # Sort by batch index to maintain order
    batch_results.sort(key=lambda x: x[0])

    # Flatten embeddings
    all_embeddings = []
    for _, embeddings in batch_results:
        all_embeddings.extend(embeddings)

    elapsed = time.time() - start_time
    print(f"Embedding completed in {elapsed:.2f}s ({len(all_embeddings)} embeddings)")

    # Compute similarities
    results = []
    for i, sample in enumerate(samples):
        emb_a = all_embeddings[i * 2]
        emb_b = all_embeddings[i * 2 + 1]
        similarity = compute_cosine_similarity(emb_a, emb_b)
        results.append((sample.id, similarity))

        if verbose:
            status = "✓" if sample.should_merge else "✗"
            print(
                f"  [{status}] ID {sample.id} ({sample.category.value[:12]}): "
                f"{similarity:.4f} (should_merge={sample.should_merge})"
            )

    return results


def evaluate_thresholds(
    samples: list[NodePairSample],
    similarities: dict[int, float],
    auto_merge_threshold: float,
    llm_threshold: float,
) -> ThresholdResult:
    """Evaluate a pair of thresholds."""
    # Auto-merge zone: similarity >= auto_merge_threshold
    # LLM zone: llm_threshold <= similarity < auto_merge_threshold
    # Separate zone: similarity < llm_threshold

    auto_merge_tp = 0  # Correctly auto-merged
    auto_merge_fp = 0  # Auto-merged but shouldn't (CRITICAL ERROR)
    auto_merge_fn = 0  # Should merge but not auto-merged

    llm_zone_count = 0
    llm_zone_boundary = 0

    separate_tn = 0  # Correctly kept separate
    separate_fp = 0  # Kept separate but should merge

    for sample in samples:
        sim = similarities[sample.id]
        should_merge = sample.should_merge

        if sim >= auto_merge_threshold:
            # Auto-merge zone
            if should_merge:
                auto_merge_tp += 1
            else:
                auto_merge_fp += 1  # CRITICAL: merged when shouldn't
        elif sim >= llm_threshold:
            # LLM zone
            llm_zone_count += 1
            if sample.is_boundary_case:
                llm_zone_boundary += 1
            # For merge metrics, count as "not auto-merged"
            if should_merge:
                auto_merge_fn += 1
        else:
            # Separate zone
            if not should_merge:
                separate_tn += 1
            else:
                separate_fp += 1  # Kept separate but should merge
                auto_merge_fn += 1

    # Calculate metrics
    total = len(samples)

    # Auto-merge precision: of all auto-merged, how many correct?
    auto_merge_total = auto_merge_tp + auto_merge_fp
    auto_merge_precision = auto_merge_tp / auto_merge_total if auto_merge_total > 0 else 1.0

    # Auto-merge recall: of all should-merge, how many auto-merged?
    should_merge_total = len(get_positive_samples())
    auto_merge_recall = auto_merge_tp / should_merge_total if should_merge_total > 0 else 0.0

    auto_merge_f1 = (
        2 * auto_merge_precision * auto_merge_recall / (auto_merge_precision + auto_merge_recall)
        if (auto_merge_precision + auto_merge_recall) > 0
        else 0.0
    )

    # LLM zone metrics
    llm_zone_coverage = llm_zone_count / total if total > 0 else 0.0
    llm_zone_efficiency = llm_zone_boundary / llm_zone_count if llm_zone_count > 0 else 0.0

    # Separate zone precision: of all kept separate, how many correct?
    separate_total = separate_tn + separate_fp
    separate_precision = separate_tn / separate_total if separate_total > 0 else 1.0

    # Separate recall: of all should-not-merge, how many kept separate?
    should_not_merge_total = len(get_negative_samples())
    separate_recall = separate_tn / should_not_merge_total if should_not_merge_total > 0 else 0.0

    # Overall accuracy
    correct = auto_merge_tp + separate_tn + llm_zone_count  # LLM zone = correct routing
    total_accuracy = correct / total if total > 0 else 0.0

    return ThresholdResult(
        auto_merge_threshold=auto_merge_threshold,
        llm_threshold=llm_threshold,
        auto_merge_precision=auto_merge_precision,
        auto_merge_recall=auto_merge_recall,
        auto_merge_f1=auto_merge_f1,
        auto_merge_tp=auto_merge_tp,
        auto_merge_fp=auto_merge_fp,
        auto_merge_fn=auto_merge_fn,
        llm_zone_coverage=llm_zone_coverage,
        llm_zone_efficiency=llm_zone_efficiency,
        separate_precision=separate_precision,
        separate_recall=separate_recall,
        separate_tn=separate_tn,
        separate_fp=separate_fp,
        total_accuracy=total_accuracy,
    )


def find_optimal_thresholds(
    samples: list[NodePairSample],
    similarities: dict[int, float],
    auto_merge_thresholds: list[float],
    llm_thresholds: list[float],
) -> tuple[ThresholdResult, list[ThresholdResult]]:
    """Find optimal threshold combination."""
    all_results = []

    for auto_thresh in auto_merge_thresholds:
        for llm_thresh in llm_thresholds:
            if llm_thresh >= auto_thresh:
                continue  # Invalid: LLM threshold must be < auto-merge threshold
            result = evaluate_thresholds(samples, similarities, auto_thresh, llm_thresh)
            all_results.append(result)

    # Find optimal: prioritize zero auto-merge FP, then maximize F1
    # Filter for zero or minimal FP first
    zero_fp_results = [r for r in all_results if r.auto_merge_fp == 0]

    if zero_fp_results:
        # Among zero FP, maximize auto-merge F1
        optimal = max(zero_fp_results, key=lambda r: r.auto_merge_f1)
    else:
        # If no zero FP, minimize FP then maximize precision
        min_fp = min(r.auto_merge_fp for r in all_results)
        min_fp_results = [r for r in all_results if r.auto_merge_fp == min_fp]
        optimal = max(min_fp_results, key=lambda r: r.auto_merge_precision)

    return optimal, all_results


def analyze_distribution(
    samples: list[NodePairSample],
    similarities: dict[int, float],
) -> dict:
    """Analyze similarity distribution by category and zone."""
    analysis = {
        "by_category": {},
        "by_expected_zone": {},
        "by_should_merge": {"True": [], "False": []},
        "overall": {},
    }

    # By category
    for cat in NodeCategory:
        cat_samples = get_samples_by_category(cat)
        sims = [similarities[s.id] for s in cat_samples]
        if sims:
            analysis["by_category"][cat.value] = {
                "count": len(sims),
                "min": min(sims),
                "max": max(sims),
                "mean": np.mean(sims),
                "median": np.median(sims),
                "std": np.std(sims),
            }

    # By expected zone
    for zone in ExpectedZone:
        zone_samples = get_samples_by_zone(zone)
        sims = [similarities[s.id] for s in zone_samples]
        if sims:
            analysis["by_expected_zone"][zone.value] = {
                "count": len(sims),
                "min": min(sims),
                "max": max(sims),
                "mean": np.mean(sims),
                "median": np.median(sims),
            }

    # By should_merge
    for sample in samples:
        key = str(sample.should_merge)
        analysis["by_should_merge"][key].append(similarities[sample.id])

    for key in ["True", "False"]:
        sims = analysis["by_should_merge"][key]
        if sims:
            analysis["by_should_merge"][key] = {
                "count": len(sims),
                "min": min(sims),
                "max": max(sims),
                "mean": np.mean(sims),
                "median": np.median(sims),
            }

    # Overall
    all_sims = list(similarities.values())
    analysis["overall"] = {
        "count": len(all_sims),
        "min": min(all_sims),
        "max": max(all_sims),
        "mean": np.mean(all_sims),
        "median": np.median(all_sims),
    }

    return analysis


def generate_report(
    optimal: ThresholdResult,
    all_results: list[ThresholdResult],
    similarities: dict[int, float],
    samples: list[NodePairSample],
    distribution: dict,
) -> str:
    """Generate markdown report."""
    lines = [
        "# Node Similarity Threshold Benchmark Report",
        "",
        f"**Generated:** {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}",
        f"**Total Samples:** {len(samples)}",
        f"**Optimal AUTO_MERGE_THRESHOLD:** {optimal.auto_merge_threshold}",
        f"**Optimal LLM_THRESHOLD:** {optimal.llm_threshold}",
        "",
        "---",
        "",
        "## 1. Executive Summary",
        "",
        "### Recommended Configuration",
        "",
        "```python",
        f"EXTRACTION_AUTO_MERGE_THRESHOLD = {optimal.auto_merge_threshold}",
        f"EXTRACTION_LLM_THRESHOLD = {optimal.llm_threshold}",
        "```",
        "",
        "### Key Metrics at Optimal Thresholds",
        "",
        "| Metric | Value |",
        "|--------|-------|",
        f"| Auto-merge Precision | {optimal.auto_merge_precision:.1%} |",
        f"| Auto-merge Recall | {optimal.auto_merge_recall:.1%} |",
        f"| Auto-merge F1 | {optimal.auto_merge_f1:.1%} |",
        f"| Auto-merge False Positives | {optimal.auto_merge_fp} ({'⚠️ CRITICAL' if optimal.auto_merge_fp > 0 else '✅'}) |",
        f"| LLM Zone Coverage | {optimal.llm_zone_coverage:.1%} |",
        f"| Separate Zone Precision | {optimal.separate_precision:.1%} |",
        "",
        "---",
        "",
        "## 2. Dataset Summary",
        "",
        "| Category | Count | Should Merge |",
        "|----------|-------|--------------|",
    ]

    for cat in NodeCategory:
        cat_samples = get_samples_by_category(cat)
        merge_count = len([s for s in cat_samples if s.should_merge])
        no_merge_count = len([s for s in cat_samples if not s.should_merge])
        lines.append(f"| {cat.value} | {len(cat_samples)} | Yes: {merge_count}, No: {no_merge_count} |")

    lines.extend([
        "",
        f"**Positive samples (should_merge=True):** {len(get_positive_samples())}",
        f"**Negative samples (should_merge=False):** {len(get_negative_samples())}",
        f"**Boundary cases:** {len(get_boundary_cases())}",
        "",
        "---",
        "",
        "## 3. Similarity Distribution Analysis",
        "",
        "### By Ground Truth (should_merge)",
        "",
    ])

    pos_stats = distribution["by_should_merge"]["True"]
    neg_stats = distribution["by_should_merge"]["False"]

    lines.extend([
        "| Metric | Should Merge (True) | Should NOT Merge (False) |",
        "|--------|---------------------|--------------------------|",
        f"| Count | {pos_stats['count']} | {neg_stats['count']} |",
        f"| Min | {pos_stats['min']:.3f} | {neg_stats['min']:.3f} |",
        f"| Max | {pos_stats['max']:.3f} | {neg_stats['max']:.3f} |",
        f"| Mean | {pos_stats['mean']:.3f} | {neg_stats['mean']:.3f} |",
        f"| Median | {pos_stats['median']:.3f} | {neg_stats['median']:.3f} |",
        "",
    ])

    # Overlap analysis
    pos_min = pos_stats['min']
    pos_max = pos_stats['max']
    neg_min = neg_stats['min']
    neg_max = neg_stats['max']

    overlap_low = max(pos_min, neg_min)
    overlap_high = min(pos_max, neg_max)

    if overlap_low < overlap_high:
        lines.extend([
            "### Overlap Zone",
            "",
            f"**Overlap range:** {overlap_low:.3f} - {overlap_high:.3f}",
            f"**Overlap width:** {overlap_high - overlap_low:.3f}",
            "",
            "⚠️ There is overlap between positive and negative samples. This means no threshold can perfectly separate them.",
            "",
        ])
    else:
        lines.extend([
            "### Separation Analysis",
            "",
            "✅ **No overlap** - positive and negative samples are well-separated!",
            f"**Gap:** {neg_max:.3f} (max negative) to {pos_min:.3f} (min positive)",
            "",
        ])

    # Threshold comparison table
    lines.extend([
        "---",
        "",
        "## 4. Threshold Comparison",
        "",
        "### Top 10 Configurations (sorted by F1, then by FP)",
        "",
        "| Auto-Merge | LLM | Precision | Recall | F1 | FP | LLM% | Separate% |",
        "|------------|-----|-----------|--------|----|----|------|-----------|",
    ])

    # Sort by F1 descending, then FP ascending
    sorted_results = sorted(all_results, key=lambda r: (-r.auto_merge_f1, r.auto_merge_fp))[:10]

    for r in sorted_results:
        marker = " ⭐" if r == optimal else ""
        separate_pct = (r.separate_tn + r.separate_fp) / len(samples) if samples else 0
        lines.append(
            f"| {r.auto_merge_threshold:.2f}{marker} | {r.llm_threshold:.2f} | "
            f"{r.auto_merge_precision:.1%} | {r.auto_merge_recall:.1%} | "
            f"{r.auto_merge_f1:.1%} | {r.auto_merge_fp} | "
            f"{r.llm_zone_coverage:.1%} | {separate_pct:.1%} |"
        )

    # False positives analysis
    lines.extend([
        "",
        "---",
        "",
        "## 5. Error Analysis at Optimal Threshold",
        "",
        f"### Auto-Merge False Positives ({optimal.auto_merge_fp} cases)",
        "",
        "These pairs were auto-merged but SHOULD NOT have been merged:",
        "",
    ])

    fps = []
    for sample in samples:
        sim = similarities[sample.id]
        if sim >= optimal.auto_merge_threshold and not sample.should_merge:
            fps.append((sample, sim))

    if fps:
        for sample, sim in sorted(fps, key=lambda x: -x[1]):
            lines.append(f"- **ID {sample.id}** (sim={sim:.3f}, cat={sample.category.value})")
            lines.append(f"  - Node A: {sample.node_a_text[:100]}...")
            lines.append(f"  - Node B: {sample.node_b_text[:100]}...")
            lines.append(f"  - Reasoning: {sample.reasoning}")
            lines.append("")
    else:
        lines.append("✅ *No false positives at optimal threshold*")
        lines.append("")

    lines.extend([
        f"### Auto-Merge False Negatives ({optimal.auto_merge_fn} cases)",
        "",
        "These pairs should have been merged but were NOT auto-merged:",
        "",
    ])

    fns = []
    for sample in samples:
        sim = similarities[sample.id]
        if sim < optimal.auto_merge_threshold and sample.should_merge:
            zone = "LLM" if sim >= optimal.llm_threshold else "SEPARATE"
            fns.append((sample, sim, zone))

    if fns:
        for sample, sim, zone in sorted(fns, key=lambda x: -x[1])[:10]:  # Top 10
            lines.append(f"- **ID {sample.id}** (sim={sim:.3f}, zone={zone}, cat={sample.category.value})")
            lines.append(f"  - This will be sent to {zone} zone instead of auto-merge")
            lines.append("")
    else:
        lines.append("*All should-merge pairs are auto-merged*")
        lines.append("")

    # All samples table
    lines.extend([
        "---",
        "",
        "## 6. All Sample Similarities",
        "",
        "| ID | Category | Similarity | Should Merge | Predicted Zone |",
        "|----|----------|------------|--------------|----------------|",
    ])

    sorted_samples = sorted(samples, key=lambda s: -similarities[s.id])
    for s in sorted_samples:
        sim = similarities[s.id]
        cat = s.category.value[:15]
        merge = "Yes" if s.should_merge else "No"
        if sim >= optimal.auto_merge_threshold:
            zone = "AUTO_MERGE"
        elif sim >= optimal.llm_threshold:
            zone = "LLM_DECIDE"
        else:
            zone = "SEPARATE"
        lines.append(f"| {s.id} | {cat} | {sim:.3f} | {merge} | {zone} |")

    # Recommendations
    lines.extend([
        "",
        "---",
        "",
        "## 7. Recommendations",
        "",
        f"### Primary Recommendation",
        "",
        "```python",
        "# config/extraction.py",
        f"EXTRACTION_AUTO_MERGE_THRESHOLD = {optimal.auto_merge_threshold}",
        f"EXTRACTION_LLM_THRESHOLD = {optimal.llm_threshold}",
        "```",
        "",
    ])

    if optimal.auto_merge_fp == 0:
        lines.append("✅ **Zero false positives** - no risk of incorrectly merging distinct nodes!")
    else:
        lines.append(f"⚠️ **{optimal.auto_merge_fp} false positives** - consider raising AUTO_MERGE_THRESHOLD")

    lines.extend([
        "",
        "### Decision Flow at Optimal Thresholds",
        "",
        "```",
        f"Similarity >= {optimal.auto_merge_threshold:.2f}  →  AUTO-MERGE (no LLM needed)",
        f"Similarity >= {optimal.llm_threshold:.2f}  →  Send to LLM for decision",
        f"Similarity < {optimal.llm_threshold:.2f}   →  Keep SEPARATE",
        "```",
        "",
    ])

    return "\n".join(lines)


async def main():
    """Main benchmark function."""
    parser = argparse.ArgumentParser(description="Benchmark Node Similarity Thresholds")
    parser.add_argument("--verbose", "-v", action="store_true", help="Verbose output")
    parser.add_argument(
        "--auto-merge-thresholds",
        type=str,
        default="0.65,0.68,0.70,0.72,0.75,0.78,0.80,0.82,0.85",
        help="Comma-separated list of auto-merge thresholds to test",
    )
    parser.add_argument(
        "--llm-thresholds",
        type=str,
        default="0.35,0.40,0.45,0.50,0.55,0.60",
        help="Comma-separated list of LLM thresholds to test",
    )
    parser.add_argument("--output", "-o", type=str, default=None, help="Output file path")
    parser.add_argument("--cache", type=str, default=None, help="Cache file for similarities")
    parser.add_argument("--no-cache", action="store_true", help="Ignore cache and recompute")
    args = parser.parse_args()

    auto_merge_thresholds = [float(t) for t in args.auto_merge_thresholds.split(",")]
    llm_thresholds = [float(t) for t in args.llm_thresholds.split(",")]
    samples = ALL_SAMPLES

    print("=" * 70)
    print("NODE SIMILARITY THRESHOLD BENCHMARK")
    print("=" * 70)
    print(f"Total samples: {len(samples)}")
    print(f"  - Positive (should_merge=True): {len(get_positive_samples())}")
    print(f"  - Negative (should_merge=False): {len(get_negative_samples())}")
    print(f"  - Boundary cases: {len(get_boundary_cases())}")
    print(f"Auto-merge thresholds: {auto_merge_thresholds}")
    print(f"LLM thresholds: {llm_thresholds}")
    print("=" * 70)

    # Check for cached similarities
    cache_path = Path(args.cache) if args.cache else Path(__file__).parent / "results" / "similarities_cache.json"
    similarities = {}

    if cache_path.exists() and not args.no_cache:
        print(f"\nLoading cached similarities from {cache_path}")
        with open(cache_path) as f:
            cached = json.load(f)
            similarities = {int(k): v for k, v in cached.items()}
        print(f"Loaded {len(similarities)} cached similarities")

        # Validate cache has all samples
        missing = [s.id for s in samples if s.id not in similarities]
        if missing:
            print(f"Cache missing {len(missing)} samples, recomputing...")
            similarities = {}

    if not similarities:
        print("\nComputing embeddings and similarities...")
        print("(This may take a few minutes)")

        try:
            sim_results = await compute_similarities(samples, verbose=args.verbose)
            similarities = {sample_id: sim for sample_id, sim in sim_results}

            # Save cache
            cache_path.parent.mkdir(parents=True, exist_ok=True)
            with open(cache_path, "w") as f:
                json.dump(similarities, f, indent=2)
            print(f"Cached similarities to {cache_path}")
        except Exception as e:
            print(f"\nError computing similarities: {e}")
            import traceback
            traceback.print_exc()
            return

    # Analyze distribution
    distribution = analyze_distribution(samples, similarities)

    # Find optimal thresholds
    print("\nEvaluating threshold combinations...")
    optimal, all_results = find_optimal_thresholds(
        samples, similarities, auto_merge_thresholds, llm_thresholds
    )

    # Print summary
    print("\n" + "=" * 70)
    print("RESULTS SUMMARY")
    print("=" * 70)

    print(f"\n🏆 Optimal Thresholds:")
    print(f"   AUTO_MERGE_THRESHOLD = {optimal.auto_merge_threshold}")
    print(f"   LLM_THRESHOLD = {optimal.llm_threshold}")

    print(f"\nAuto-Merge Zone Metrics:")
    print(f"   Precision: {optimal.auto_merge_precision:.1%}")
    print(f"   Recall: {optimal.auto_merge_recall:.1%}")
    print(f"   F1 Score: {optimal.auto_merge_f1:.1%}")
    print(f"   False Positives: {optimal.auto_merge_fp} {'⚠️ CRITICAL' if optimal.auto_merge_fp > 0 else '✅'}")

    print(f"\nLLM Zone:")
    print(f"   Coverage: {optimal.llm_zone_coverage:.1%} of pairs sent to LLM")

    print(f"\nSeparate Zone:")
    print(f"   Precision: {optimal.separate_precision:.1%}")

    # Distribution insight
    pos_stats = distribution["by_should_merge"]["True"]
    neg_stats = distribution["by_should_merge"]["False"]
    print(f"\n📊 Similarity Distribution:")
    print(f"   Positive (merge): {pos_stats['min']:.3f} - {pos_stats['max']:.3f} (mean: {pos_stats['mean']:.3f})")
    print(f"   Negative (no merge): {neg_stats['min']:.3f} - {neg_stats['max']:.3f} (mean: {neg_stats['mean']:.3f})")

    # Generate and save report
    report = generate_report(optimal, all_results, similarities, samples, distribution)

    output_path = Path(args.output) if args.output else Path(__file__).parent / "results" / "threshold_analysis.md"
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w") as f:
        f.write(report)

    print(f"\n📄 Full report saved to: {output_path}")

    # Save optimal config
    config_path = output_path.parent / "optimal_config.json"
    with open(config_path, "w") as f:
        json.dump({
            "EXTRACTION_AUTO_MERGE_THRESHOLD": optimal.auto_merge_threshold,
            "EXTRACTION_LLM_THRESHOLD": optimal.llm_threshold,
            "metrics": {
                "auto_merge_precision": optimal.auto_merge_precision,
                "auto_merge_recall": optimal.auto_merge_recall,
                "auto_merge_f1": optimal.auto_merge_f1,
                "auto_merge_fp": optimal.auto_merge_fp,
                "llm_zone_coverage": optimal.llm_zone_coverage,
            },
            "generated_at": datetime.now().isoformat(),
        }, f, indent=2)
    print(f"📄 Optimal config saved to: {config_path}")


if __name__ == "__main__":
    asyncio.run(main())
