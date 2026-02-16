"""Benchmark script for evaluating EXTRACTION_MERGE_THRESHOLD.

This script:
1. Loads 70 sample pairs from samples.py
2. Computes embedding similarity for each pair using real embeddings
3. Evaluates different threshold values
4. Generates a detailed analysis report

Usage:
    python benchmarks/merge_threshold/test_threshold.py
    python benchmarks/merge_threshold/test_threshold.py --verbose
    python benchmarks/merge_threshold/test_threshold.py --thresholds 0.70,0.75,0.80,0.85,0.90
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
    MergeCategory,
    RelationSample,
    get_boundary_cases,
    get_negative_samples,
    get_positive_samples,
    get_samples_by_category,
)


@dataclass
class EvaluationResult:
    """Result of evaluating a single threshold."""

    threshold: float
    accuracy: float
    precision: float
    recall: float
    f1_score: float
    f2_score: float  # Precision-weighted (penalize FP more)
    true_positives: int
    true_negatives: int
    false_positives: int
    false_negatives: int
    total_samples: int


@dataclass
class SampleResult:
    """Result for a single sample."""

    sample_id: int
    similarity: float
    ground_truth: bool  # should_merge
    predicted: dict[float, bool]  # threshold -> prediction
    category: str


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


def build_relation_text(name: str, summary: str, description: str) -> str:
    """Build text representation for embedding.

    Matches the format used in relation_validator.py:
    text_to_embed = f"{name}: {description}"
    """
    # Combine summary and description for richer embedding
    full_desc = f"{summary}\n\n{description}" if summary else description
    return f"{name}: {full_desc}"


async def compute_similarities(
    samples: list[RelationSample],
    verbose: bool = False,
) -> list[tuple[int, float]]:
    """Compute similarity scores for all samples using real embeddings."""
    import os

    from dotenv import load_dotenv
    from openai import AsyncOpenAI

    # Load .env file
    load_dotenv()

    # Use OpenAI API directly
    api_key = (
        os.getenv("EMBEDDING_API_KEY")
        or os.getenv("LLM_EMBEDDING_API_KEY")
        or os.getenv("LLM_API_KEY")
    )
    api_base = (
        os.getenv("EMBEDDING_API_BASE")
        or os.getenv("LLM_EMBEDDING_API_BASE")
        or "https://api.openai.com/v1"
    )
    model = (
        os.getenv("EMBEDDING_MODEL")
        or os.getenv("LLM_EMBEDDING_MODEL")
        or "text-embedding-3-small"
    )

    client = AsyncOpenAI(api_key=api_key, base_url=api_base)

    results = []
    total = len(samples)

    # Batch embed for efficiency
    print(f"\nEmbedding {total * 2} texts...")
    print(f"Using model: {model}")
    print(f"API base: {api_base}")

    all_texts = []
    for sample in samples:
        text_a = build_relation_text(
            sample.relation_a_name,
            sample.relation_a_summary,
            sample.relation_a_description,
        )
        text_b = build_relation_text(
            sample.relation_b_name,
            sample.relation_b_summary,
            sample.relation_b_description,
        )
        all_texts.extend([text_a, text_b])

    # Batch embed in chunks of 20
    batch_size = 20
    all_embeddings = []

    for i in range(0, len(all_texts), batch_size):
        batch = all_texts[i : i + batch_size]
        if verbose:
            print(
                f"  Embedding batch {i // batch_size + 1}/{(len(all_texts) + batch_size - 1) // batch_size}"
            )

        response = await client.embeddings.create(
            model=model,
            input=batch,
        )
        for item in response.data:
            all_embeddings.append(item.embedding)

    # Compute similarities
    for i, sample in enumerate(samples):
        emb_a = all_embeddings[i * 2]
        emb_b = all_embeddings[i * 2 + 1]
        similarity = compute_cosine_similarity(emb_a, emb_b)
        results.append((sample.id, similarity))

        if verbose:
            status = "✓" if sample.should_merge else "✗"
            print(
                f"  [{status}] {sample.relation_a_name} vs {sample.relation_b_name}: "
                f"{similarity:.4f} (should_merge={sample.should_merge})"
            )

    return results


def evaluate_threshold(
    samples: list[RelationSample],
    similarities: dict[int, float],
    threshold: float,
) -> EvaluationResult:
    """Evaluate a single threshold value."""
    tp = tn = fp = fn = 0

    for sample in samples:
        similarity = similarities[sample.id]
        predicted_merge = similarity >= threshold
        actual_merge = sample.should_merge

        if predicted_merge and actual_merge:
            tp += 1
        elif not predicted_merge and not actual_merge:
            tn += 1
        elif predicted_merge and not actual_merge:
            fp += 1
        else:  # not predicted_merge and actual_merge
            fn += 1

    total = tp + tn + fp + fn
    accuracy = (tp + tn) / total if total > 0 else 0
    precision = tp / (tp + fp) if (tp + fp) > 0 else 1.0  # 1.0 if no predictions
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0
    # F2 score: weights precision higher (penalizes FP more)
    f2 = 5 * precision * recall / (4 * precision + recall) if (4 * precision + recall) > 0 else 0

    return EvaluationResult(
        threshold=threshold,
        accuracy=accuracy,
        precision=precision,
        recall=recall,
        f1_score=f1,
        f2_score=f2,
        true_positives=tp,
        true_negatives=tn,
        false_positives=fp,
        false_negatives=fn,
        total_samples=total,
    )


def find_optimal_threshold(
    samples: list[RelationSample],
    similarities: dict[int, float],
    thresholds: list[float],
    optimize_for: str = "f2",  # f1, f2, or precision
) -> tuple[float, list[EvaluationResult]]:
    """Find the optimal threshold based on chosen metric."""
    results = []
    for threshold in thresholds:
        result = evaluate_threshold(samples, similarities, threshold)
        results.append(result)

    # Find best by chosen metric
    if optimize_for == "precision":
        # Filter to only thresholds with reasonable recall (> 50%)
        valid_results = [r for r in results if r.recall > 0.5]
        if valid_results:
            best = max(valid_results, key=lambda r: r.precision)
        else:
            best = max(results, key=lambda r: r.precision)
    elif optimize_for == "f2":
        best = max(results, key=lambda r: r.f2_score)
    else:  # f1
        best = max(results, key=lambda r: r.f1_score)

    return best.threshold, results


def analyze_errors(
    samples: list[RelationSample],
    similarities: dict[int, float],
    threshold: float,
) -> dict:
    """Analyze false positives and false negatives for a threshold."""
    false_positives = []
    false_negatives = []

    for sample in samples:
        similarity = similarities[sample.id]
        predicted_merge = similarity >= threshold
        actual_merge = sample.should_merge

        if predicted_merge and not actual_merge:
            false_positives.append(
                {
                    "id": sample.id,
                    "similarity": similarity,
                    "category": sample.category.value
                    if hasattr(sample.category, "value")
                    else str(sample.category),
                    "relation_a": sample.relation_a_name,
                    "relation_b": sample.relation_b_name,
                    "reasoning": sample.reasoning,
                    "is_boundary": sample.is_boundary_case,
                }
            )
        elif not predicted_merge and actual_merge:
            false_negatives.append(
                {
                    "id": sample.id,
                    "similarity": similarity,
                    "category": sample.category.value
                    if hasattr(sample.category, "value")
                    else str(sample.category),
                    "relation_a": sample.relation_a_name,
                    "relation_b": sample.relation_b_name,
                    "reasoning": sample.reasoning,
                    "is_boundary": sample.is_boundary_case,
                }
            )

    # Sort by similarity (FP: highest first, FN: lowest first)
    false_positives.sort(key=lambda x: -x["similarity"])
    false_negatives.sort(key=lambda x: x["similarity"])

    return {
        "false_positives": false_positives,
        "false_negatives": false_negatives,
    }


def generate_report(
    results: list[EvaluationResult],
    similarities: dict[int, float],
    samples: list[RelationSample],
    optimal_threshold: float,
) -> str:
    """Generate markdown report."""
    lines = [
        "# EXTRACTION_MERGE_THRESHOLD Benchmark Report",
        "",
        f"**Generated:** {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}",
        f"**Total Samples:** {len(samples)}",
        f"**Optimal Threshold:** {optimal_threshold}",
        "",
        "---",
        "",
        "## 1. Dataset Summary",
        "",
        "| Category | Count | Should Merge | Description |",
        "|----------|-------|--------------|-------------|",
    ]

    for cat in MergeCategory:
        cat_samples = get_samples_by_category(cat)
        merge_count = len([s for s in cat_samples if s.should_merge])
        no_merge_count = len([s for s in cat_samples if not s.should_merge])
        merge_label = f"Yes ({merge_count})" if merge_count > 0 else f"No ({no_merge_count})"
        lines.append(
            f"| {cat.value} | {len(cat_samples)} | {merge_label} | {cat.name.replace('_', ' ').title()} |"
        )

    lines.extend(
        [
            "",
            f"**Positive samples (should_merge=True):** {len(get_positive_samples())}",
            f"**Negative samples (should_merge=False):** {len(get_negative_samples())}",
            f"**Boundary cases:** {len(get_boundary_cases())}",
            "",
            "---",
            "",
            "## 2. Threshold Comparison",
            "",
            "| Threshold | Accuracy | Precision | Recall | F1 | F2 | TP | TN | FP | FN |",
            "|-----------|----------|-----------|--------|----|----|----|----|----|----|",
        ]
    )

    for r in sorted(results, key=lambda x: x.threshold):
        marker = " ⭐" if r.threshold == optimal_threshold else ""
        lines.append(
            f"| {r.threshold:.2f}{marker} | {r.accuracy:.1%} | {r.precision:.1%} | "
            f"{r.recall:.1%} | {r.f1_score:.1%} | {r.f2_score:.1%} | "
            f"{r.true_positives} | {r.true_negatives} | "
            f"{r.false_positives} | {r.false_negatives} |"
        )

    lines.extend(
        [
            "",
            "**Legend:**",
            "- **Precision**: Of all merges predicted, how many were correct? (High = few false merges)",
            "- **Recall**: Of all should-merge pairs, how many did we catch? (High = few missed merges)",
            "- **F2**: Precision-weighted score (penalizes false positives more)",
            "",
            "---",
            "",
            "## 3. Optimal Threshold Analysis",
            "",
            f"### Recommended Threshold: **{optimal_threshold}**",
            "",
        ]
    )

    # Get errors for optimal threshold
    errors = analyze_errors(samples, similarities, optimal_threshold)

    lines.extend(
        [
            f"### False Positives ({len(errors['false_positives'])} cases) - CRITICAL",
            "",
            "These pairs were incorrectly predicted to MERGE (similarity >= threshold but should NOT merge):",
            "**These are the most dangerous errors - they corrupt the knowledge graph!**",
            "",
        ]
    )

    if errors["false_positives"]:
        for fp in errors["false_positives"]:
            boundary_marker = " 🔥" if fp["is_boundary"] else ""
            lines.append(
                f"- **ID {fp['id']}**: `{fp['relation_a']}` vs `{fp['relation_b']}` "
                f"(sim={fp['similarity']:.3f}, cat={fp['category']}){boundary_marker}"
            )
            lines.append(f"  - Reasoning: {fp['reasoning'][:200]}...")
    else:
        lines.append("*No false positives - excellent!*")

    lines.extend(
        [
            "",
            f"### False Negatives ({len(errors['false_negatives'])} cases)",
            "",
            "These pairs were incorrectly predicted to NOT MERGE (similarity < threshold but SHOULD merge):",
            "(These are less critical - they just create duplicates that can be fixed later)",
            "",
        ]
    )

    if errors["false_negatives"]:
        for fn in errors["false_negatives"]:
            boundary_marker = " 🔥" if fn["is_boundary"] else ""
            lines.append(
                f"- **ID {fn['id']}**: `{fn['relation_a']}` vs `{fn['relation_b']}` "
                f"(sim={fn['similarity']:.3f}, cat={fn['category']}){boundary_marker}"
            )
            lines.append(f"  - Reasoning: {fn['reasoning'][:200]}...")
    else:
        lines.append("*No false negatives*")

    # Similarity distribution
    lines.extend(
        [
            "",
            "---",
            "",
            "## 4. Similarity Distribution",
            "",
            "### Positive Samples (should_merge=True)",
            "",
        ]
    )

    positive_sims = [similarities[s.id] for s in samples if s.should_merge]
    if positive_sims:
        lines.extend(
            [
                f"- Min: {min(positive_sims):.3f}",
                f"- Max: {max(positive_sims):.3f}",
                f"- Mean: {np.mean(positive_sims):.3f}",
                f"- Median: {np.median(positive_sims):.3f}",
                f"- Std: {np.std(positive_sims):.3f}",
            ]
        )

    lines.extend(
        [
            "",
            "### Negative Samples (should_merge=False)",
            "",
        ]
    )

    negative_sims = [similarities[s.id] for s in samples if not s.should_merge]
    if negative_sims:
        lines.extend(
            [
                f"- Min: {min(negative_sims):.3f}",
                f"- Max: {max(negative_sims):.3f}",
                f"- Mean: {np.mean(negative_sims):.3f}",
                f"- Median: {np.median(negative_sims):.3f}",
                f"- Std: {np.std(negative_sims):.3f}",
            ]
        )

    # Find overlap zone
    if positive_sims and negative_sims:
        overlap_low = max(min(positive_sims), min(negative_sims))
        overlap_high = min(max(positive_sims), max(negative_sims))

        lines.extend(
            [
                "",
                "### Overlap Analysis",
                "",
            ]
        )

        if overlap_low < overlap_high:
            lines.append(f"- **Overlap zone:** {overlap_low:.3f} - {overlap_high:.3f}")
            lines.append(f"- **Overlap width:** {overlap_high - overlap_low:.3f}")

            # Count samples in overlap
            overlap_positive = len([s for s in positive_sims if overlap_low <= s <= overlap_high])
            overlap_negative = len([s for s in negative_sims if overlap_low <= s <= overlap_high])
            lines.append(f"- **Positive samples in overlap:** {overlap_positive}")
            lines.append(f"- **Negative samples in overlap:** {overlap_negative}")
        else:
            lines.append(
                "- **No overlap** - clear separation between positive and negative samples!"
            )

    # Recommendations
    lines.extend(
        [
            "",
            "---",
            "",
            "## 5. Recommendations",
            "",
        ]
    )

    optimal_result = next(r for r in results if r.threshold == optimal_threshold)

    lines.extend(
        [
            f"### Primary Recommendation: `EXTRACTION_MERGE_THRESHOLD = {optimal_threshold}`",
            "",
            "At this threshold:",
            f"- **Accuracy:** {optimal_result.accuracy:.1%}",
            f"- **Precision:** {optimal_result.precision:.1%} (of predicted merges, {optimal_result.precision:.0%} are correct)",
            f"- **Recall:** {optimal_result.recall:.1%} (of should-merge pairs, we catch {optimal_result.recall:.0%})",
            f"- **False Positives:** {optimal_result.false_positives} (merged when shouldn't - DANGEROUS)",
            f"- **False Negatives:** {optimal_result.false_negatives} (didn't merge when should - recoverable)",
            "",
        ]
    )

    # Risk analysis
    if optimal_result.false_positives == 0:
        lines.append("✅ **Zero false positives** - no risk of corrupting knowledge graph!")
    elif optimal_result.false_positives <= 2:
        lines.append(
            f"⚠️ **Low false positive risk** ({optimal_result.false_positives} cases) - monitor these in production"
        )
    else:
        lines.append(
            f"🔴 **High false positive risk** ({optimal_result.false_positives} cases) - consider raising threshold"
        )

    lines.extend(
        [
            "",
            "### Action Items",
            "",
            f"1. Update `config/extraction.py`: `EXTRACTION_MERGE_THRESHOLD = {optimal_threshold}`",
            "2. Monitor false positive rate in production logs",
            "3. Review any false positive cases manually before deployment",
            "",
        ]
    )

    # All similarities for reference
    lines.extend(
        [
            "---",
            "",
            "## 6. All Sample Similarities",
            "",
            "| ID | Category | Relation A | Relation B | Similarity | Should Merge | Boundary |",
            "|----|----------|------------|------------|------------|--------------|----------|",
        ]
    )

    sorted_samples = sorted(samples, key=lambda s: -similarities[s.id])
    for s in sorted_samples:
        sim = similarities[s.id]
        cat = s.category.value if hasattr(s.category, "value") else str(s.category)
        merge = "Yes" if s.should_merge else "No"
        boundary = "🔥" if s.is_boundary_case else ""
        lines.append(
            f"| {s.id} | {cat[:15]} | {s.relation_a_name[:15]} | {s.relation_b_name[:15]} | {sim:.3f} | {merge} | {boundary} |"
        )

    return "\n".join(lines)


async def main():
    """Main benchmark function."""
    parser = argparse.ArgumentParser(description="Benchmark EXTRACTION_MERGE_THRESHOLD")
    parser.add_argument("--verbose", "-v", action="store_true", help="Verbose output")
    parser.add_argument(
        "--thresholds",
        type=str,
        default="0.60,0.65,0.70,0.72,0.75,0.78,0.80,0.82,0.85,0.88,0.90",
        help="Comma-separated list of thresholds to test",
    )
    parser.add_argument("--output", "-o", type=str, default=None, help="Output file path")
    parser.add_argument("--cache", type=str, default=None, help="Cache file for similarities")
    parser.add_argument("--no-cache", action="store_true", help="Ignore cache and recompute")
    parser.add_argument(
        "--optimize",
        type=str,
        default="f2",
        choices=["f1", "f2", "precision"],
        help="Metric to optimize (f2 penalizes false positives more)",
    )
    args = parser.parse_args()

    thresholds = [float(t) for t in args.thresholds.split(",")]
    samples = ALL_SAMPLES

    print("=" * 70)
    print("EXTRACTION_MERGE_THRESHOLD Benchmark")
    print("=" * 70)
    print(f"Total samples: {len(samples)}")
    print(f"  - Positive (should_merge=True): {len(get_positive_samples())}")
    print(f"  - Negative (should_merge=False): {len(get_negative_samples())}")
    print(f"  - Boundary cases: {len(get_boundary_cases())}")
    print(f"Thresholds to test: {thresholds}")
    print(f"Optimizing for: {args.optimize}")
    print("=" * 70)

    # Check for cached similarities
    cache_path = (
        Path(args.cache)
        if args.cache
        else Path(__file__).parent / "results" / "similarities_cache.json"
    )
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

    # Evaluate thresholds
    print("\nEvaluating thresholds...")
    optimal_threshold, results = find_optimal_threshold(
        samples, similarities, thresholds, optimize_for=args.optimize
    )

    # Print summary
    print("\n" + "=" * 70)
    print("RESULTS SUMMARY")
    print("=" * 70)
    print(
        f"\n{'Threshold':<12} {'Accuracy':<10} {'Precision':<10} {'Recall':<10} {'F1':<10} {'F2':<10} {'FP':<5} {'FN':<5}"
    )
    print("-" * 70)

    for r in sorted(results, key=lambda x: x.threshold):
        marker = " ⭐" if r.threshold == optimal_threshold else ""
        print(
            f"{r.threshold:<12.2f} {r.accuracy:<10.1%} {r.precision:<10.1%} "
            f"{r.recall:<10.1%} {r.f1_score:<10.1%} {r.f2_score:<10.1%} "
            f"{r.false_positives:<5} {r.false_negatives:<5}{marker}"
        )

    print("-" * 70)
    print(f"\n🏆 Optimal Threshold: {optimal_threshold}")
    optimal_result = next(r for r in results if r.threshold == optimal_threshold)
    print(f"   Accuracy: {optimal_result.accuracy:.1%}")
    print(f"   Precision: {optimal_result.precision:.1%}")
    print(f"   Recall: {optimal_result.recall:.1%}")
    print(f"   F1 Score: {optimal_result.f1_score:.1%}")
    print(f"   F2 Score: {optimal_result.f2_score:.1%}")
    print(f"   False Positives: {optimal_result.false_positives}")
    print(f"   False Negatives: {optimal_result.false_negatives}")

    # Analyze errors
    errors = analyze_errors(samples, similarities, optimal_threshold)

    if errors["false_positives"]:
        print("\n⚠️  FALSE POSITIVES (merged when shouldn't):")
        for fp in errors["false_positives"][:5]:
            print(f"   - {fp['relation_a']} vs {fp['relation_b']} (sim={fp['similarity']:.3f})")

    # Generate report
    report = generate_report(results, similarities, samples, optimal_threshold)

    output_path = (
        Path(args.output)
        if args.output
        else Path(__file__).parent / "results" / "threshold_analysis.md"
    )
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w") as f:
        f.write(report)

    print(f"\n📄 Full report saved to: {output_path}")

    # Print similarity distribution insight
    positive_sims = [similarities[s.id] for s in samples if s.should_merge]
    negative_sims = [similarities[s.id] for s in samples if not s.should_merge]

    print("\n📊 Similarity Distribution:")
    print(
        f"   Positive (merge): {min(positive_sims):.3f} - {max(positive_sims):.3f} (mean: {np.mean(positive_sims):.3f})"
    )
    print(
        f"   Negative (no merge): {min(negative_sims):.3f} - {max(negative_sims):.3f} (mean: {np.mean(negative_sims):.3f})"
    )

    # Suggest threshold based on distribution
    suggested = (min(positive_sims) + max(negative_sims)) / 2
    print(f"\n💡 Suggested threshold based on distribution midpoint: {suggested:.3f}")


if __name__ == "__main__":
    asyncio.run(main())
