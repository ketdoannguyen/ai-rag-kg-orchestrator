"""Synthetic tests for research.ocr_pipeline.metrics.

No corpus, no data files, no network: every input is a dict built in the test.
"""

from __future__ import annotations

import pytest

from research.ocr_pipeline.metrics import (
    EMPTY_REFERENCE,
    MEASURED,
    NOT_MEASURED,
    cer,
    chunk_metrics,
    corpus_text_metrics,
    document_text_metrics,
    edit_distance,
    normalize_text,
    page_structure_metrics,
    wer,
)

_ABSENT = object()


def _page(
    reference: object = _ABSENT,
    hypothesis: object = _ABSENT,
    page: int | None = None,
) -> dict:
    entry: dict = {}
    if reference is not _ABSENT:
        entry["reference"] = reference
    if hypothesis is not _ABSENT:
        entry["hypothesis"] = hypothesis
    if page is not None:
        entry["page"] = page
    return entry


class TestNormalizeText:
    """NFC composition plus whitespace collapse, nothing else."""

    def test_nfc_composes_vietnamese_glyphs(self) -> None:
        decomposed = "ế"  # e + circumflex + acute
        assert normalize_text(decomposed) == "ế"
        assert normalize_text(decomposed) == normalize_text("ế")

    def test_compatibility_forms_are_preserved(self) -> None:
        # NFKC would rewrite U+FB01 to "fi" and hide a real OCR difference.
        assert normalize_text("ﬁ") == "ﬁ"

    def test_whitespace_runs_collapse_and_strip(self) -> None:
        assert normalize_text("  a\tb\n c d  ") == "a b c d"

    def test_case_and_punctuation_are_preserved(self) -> None:
        assert normalize_text("AbC, x.") == "AbC, x."

    def test_none_is_empty(self) -> None:
        assert normalize_text(None) == ""

    def test_non_string_is_rejected(self) -> None:
        with pytest.raises(TypeError):
            normalize_text(42)  # type: ignore[arg-type]


class TestEditDistance:
    """Two-row DP behaves like the textbook matrix."""

    @pytest.mark.parametrize(
        ("left", "right", "expected"),
        [
            ("", "", 0),
            ("abc", "", 3),
            ("", "abc", 3),
            ("kitten", "sitting", 3),
            ("flaw", "lawn", 2),
            ("abc", "abc", 0),
            ("a", "b", 1),
        ],
    )
    def test_known_distances(self, left: str, right: str, expected: int) -> None:
        assert edit_distance(left, right) == expected
        assert edit_distance(right, left) == expected

    def test_works_on_token_sequences(self) -> None:
        assert edit_distance(["a", "b"], ["a", "c", "d"]) == 2


class TestCerWer:
    """Page-level rate contract."""

    def test_identical_text_scores_zero(self) -> None:
        result = cer("abc", "abc")
        assert result["status"] == MEASURED
        assert result["value"] == 0.0
        assert result["numerator"] == 0
        assert result["denominator"] == 3
        assert result["denominator_unit"] == "characters"

    def test_substitution_deletion_insertion(self) -> None:
        assert cer("abc", "abd")["value"] == pytest.approx(1 / 3)
        assert cer("abc", "ab")["value"] == pytest.approx(1 / 3)
        assert cer("ab", "abc")["value"] == pytest.approx(1 / 2)

    def test_whitespace_differences_are_free(self) -> None:
        assert cer("a  b\nc", "a b c")["value"] == 0.0

    def test_nfc_differences_are_free(self) -> None:
        assert cer("ế", "ế")["value"] == 0.0

    def test_empty_gold_is_a_typed_miss_not_zero(self) -> None:
        for reference in ("", "   ", None):
            result = cer(reference, "abc")
            assert result["status"] == EMPTY_REFERENCE
            assert result["value"] is None
            assert result["denominator"] == 0

    def test_missing_transcript_is_a_typed_miss(self) -> None:
        # A page the engine never produced text for is not a 100% error; it is
        # an unscored page, so it must not enter any error-rate average.
        chars = cer("abc", None)
        assert chars["status"] == NOT_MEASURED
        assert chars["reason"] == "hypothesis_missing"
        assert chars["value"] is None
        assert chars["denominator"] == 3
        assert chars["denominator_unit"] == "characters"
        words = wer("con mèo", None)
        assert words["status"] == NOT_MEASURED
        assert words["reason"] == "hypothesis_missing"
        assert words["value"] is None
        assert words["denominator"] == 2
        assert words["denominator_unit"] == "words"

    def test_empty_transcript_is_a_measured_full_deletion(self) -> None:
        # "" is a transcript the engine did produce: it scores as every gold
        # character deleted, which is not the same event as a missing page.
        for empty in ("", "   ", "\n"):
            assert cer("abc", empty)["status"] == MEASURED
            assert cer("abc", empty)["value"] == 1.0
        assert wer("con mèo", "")["value"] == 1.0

    def test_empty_gold_wins_over_a_missing_transcript(self) -> None:
        # Both sides absent is a page with no gold to score, not a failed page.
        assert cer(None, None)["status"] == EMPTY_REFERENCE
        assert wer(None, None)["status"] == EMPTY_REFERENCE

    def test_wer_is_token_level(self) -> None:
        assert wer("con mèo ngồi", "con chó ngồi")["value"] == pytest.approx(1 / 3)
        assert wer("con mèo ngồi", "con  mèo\nngồi")["value"] == 0.0

    def test_wer_empty_gold(self) -> None:
        result = wer("   ", "con")
        assert result["status"] == EMPTY_REFERENCE
        assert result["denominator_unit"] == "words"


class TestDocumentTextMetrics:
    """Per-document micro/macro over the pages the gold covers."""

    def test_single_page_document(self) -> None:
        result = document_text_metrics({"doc_id": "d1", "pages": [_page("abcd", "abcd", page=1)]})
        assert result["doc_id"] == "d1"
        assert result["pages"] == {
            "total": 1,
            "with_gold": 1,
            "without_gold": 0,
            "empty_hypothesis": 0,
        }
        assert result["rates"]["cer"]["micro"]["value"] == 0.0
        assert result["rates"]["cer"]["macro"]["value"] == 0.0
        assert result["hypothesis"] == {"present_pages": 1, "missing_pages": 0}
        assert result["pages_detail"][0]["page"] == 1

    def test_micro_pools_edits_macro_averages_pages(self) -> None:
        result = document_text_metrics(
            {
                "doc_id": "d1",
                "pages": [
                    _page("ab", "ab", page=1),  # 0 edits / 2 chars
                    _page("abcd", "abxy", page=2),  # 2 edits / 4 chars
                ],
            }
        )
        micro = result["rates"]["cer"]["micro"]
        macro = result["rates"]["cer"]["macro"]
        assert micro["value"] == pytest.approx(2 / 6)
        assert micro["denominator"] == 6
        assert macro["value"] == pytest.approx((0.0 + 0.5) / 2)
        assert macro["denominator"] == 2
        assert macro["denominator_unit"] == "pages"

    def test_a_page_without_a_transcript_is_a_typed_miss_not_an_error(self) -> None:
        result = document_text_metrics({"doc_id": "d1", "pages": [_page("abcd", page=1)]})
        assert result["pages"]["with_gold"] == 1
        assert result["pages"]["empty_hypothesis"] == 0
        assert result["hypothesis"] == {"present_pages": 0, "missing_pages": 1}
        assert result["rates"]["cer"]["micro"]["status"] == NOT_MEASURED
        assert result["rates"]["cer"]["micro"]["reason"] == "empty_denominator"
        assert result["rates"]["measured_pages"] == 0
        assert result["rates"]["missing_transcript_pages"] == 1
        assert result["pages_detail"][0]["cer"]["reason"] == "hypothesis_missing"

    def test_empty_transcript_is_measured_as_a_full_deletion(self) -> None:
        result = document_text_metrics({"doc_id": "d1", "pages": [_page("abcd", "", page=1)]})
        assert result["pages"]["empty_hypothesis"] == 1
        assert result["hypothesis"] == {"present_pages": 1, "missing_pages": 0}
        assert result["rates"]["cer"]["micro"]["value"] == 1.0

    def test_a_page_the_gold_does_not_cover_is_not_a_zero(self) -> None:
        # The run read a page the gold does not hold: there is no reference to
        # score against, which is an empty reference rather than a perfect run.
        result = document_text_metrics({"doc_id": "d1", "pages": [_page(None, "abcd", page=1)]})
        assert result["pages"]["with_gold"] == 0
        assert result["rates"]["measured_pages"] == 0
        assert result["pages_detail"][0]["cer"]["status"] == EMPTY_REFERENCE

    def test_no_pages_at_all(self) -> None:
        result = document_text_metrics({"doc_id": "d1"})
        assert result["pages"]["total"] == 0
        assert result["rates"]["cer"]["micro"]["status"] == NOT_MEASURED
        assert result["rates"]["cer"]["micro"]["reason"] == "empty_denominator"

    def test_non_dict_page_is_rejected(self) -> None:
        with pytest.raises(TypeError):
            document_text_metrics({"doc_id": "d1", "pages": ["page one"]})


class TestCorpusTextMetrics:
    """Corpus aggregation over synthetic documents."""

    def test_single_document_matches_its_own_metrics(self) -> None:
        document = {"doc_id": "d1", "pages": [_page("ab", "ab", page=1)]}
        corpus = corpus_text_metrics([document])
        document_result = document_text_metrics(document)
        assert corpus["documents_total"] == 1
        assert corpus["rates"]["cer"]["micro"] == document_result["rates"]["cer"]["micro"]
        assert (
            corpus["rates"]["cer"]["macro"]["value"]
            == document_result["rates"]["cer"]["macro"]["value"]
        )
        # Corpus macro averages documents, so its denominator unit differs.
        assert corpus["rates"]["cer"]["macro"]["denominator_unit"] == "documents"

    def test_two_documents_pool_micro_and_average_macro(self) -> None:
        corpus = corpus_text_metrics(
            [
                {"doc_id": "d1", "pages": [_page("ab", "ab", page=1)]},
                {"doc_id": "d2", "pages": [_page("abcd", "abxy", page=1)]},
            ]
        )
        micro = corpus["rates"]["cer"]["micro"]
        macro = corpus["rates"]["cer"]["macro"]
        assert corpus["documents_total"] == 2
        assert corpus["pages"]["with_gold"] == 2
        assert micro["value"] == pytest.approx(2 / 6)
        assert macro["value"] == pytest.approx((0.0 + 0.5) / 2)
        assert macro["denominator"] == 2
        assert macro["denominator_unit"] == "documents"
        assert [doc["doc_id"] for doc in corpus["per_document"]] == ["d1", "d2"]

    def test_document_without_gold_is_excluded_from_macro(self) -> None:
        corpus = corpus_text_metrics(
            [
                {"doc_id": "d1", "pages": [_page("abcd", "abcd", page=1)]},
                {"doc_id": "d2", "pages": [_page("", "", page=1)]},
            ]
        )
        assert corpus["documents_with_gold"] == 1
        assert corpus["rates"]["cer"]["macro"]["denominator"] == 1

    def test_corpus_aggregates_missing_transcripts(self) -> None:
        corpus = corpus_text_metrics(
            [
                {"doc_id": "d1", "pages": [_page("ab", "ab", page=1)]},
                {"doc_id": "d2", "pages": [_page("abcd", page=1)]},
            ]
        )
        assert corpus["pages"]["with_gold"] == 2
        assert corpus["hypothesis"] == {"present_pages": 1, "missing_pages": 1}
        assert corpus["rates"]["missing_transcript_pages"] == 1
        # The unscored page is not in the pooled rate: micro stays the one page
        # that was measured, and macro keeps its own denominator.
        assert corpus["rates"]["cer"]["micro"]["value"] == 0.0
        assert corpus["rates"]["cer"]["macro"]["denominator"] == 1
        assert corpus["per_document"][1]["hypothesis"]["missing_pages"] == 1

    def test_empty_corpus(self) -> None:
        corpus = corpus_text_metrics([])
        assert corpus["documents_total"] == 0
        assert corpus["rates"]["cer"]["micro"]["status"] == NOT_MEASURED
        assert corpus["rates"]["cer"]["macro"]["status"] == NOT_MEASURED


class TestChunkMetrics:
    """Chunk lengths, oversize detection and page fidelity.

    Chunk page lists carry 0-based physical page indices, the same index space
    as ``ArtifactPage.page_index`` and ``Span.page_index``.
    """

    CHUNKS = [
        {"chunk_id": "c1", "doc_id": "d1", "text": "abc", "pages": [0]},
        {"chunk_id": "c2", "doc_id": "d1", "text": "", "pages": [1, 2]},
    ]

    def test_lengths_and_empties(self) -> None:
        result = chunk_metrics(self.CHUNKS)
        assert result["status"] == MEASURED
        assert result["chunks"] == 2
        assert result["char_length"]["min"] == 0
        assert result["char_length"]["max"] == 3
        assert result["char_length"]["mean"] == pytest.approx(1.5)
        assert result["char_length"]["p90"] == 3.0
        assert result["empty_chunks"]["count"] == 1
        assert result["empty_chunks"]["ratio"] == pytest.approx(0.5)

    def test_oversized_requires_a_limit(self) -> None:
        without_limit = chunk_metrics(self.CHUNKS)["oversized_chunks"]
        assert without_limit["status"] == NOT_MEASURED
        assert without_limit["reason"] == "no_limit"
        with_limit = chunk_metrics(self.CHUNKS, max_chunk_size=2)["oversized_chunks"]
        assert with_limit["count"] == 1
        assert with_limit["chunk_ids"] == ["c1"]

    def test_page_fidelity_and_coverage(self) -> None:
        result = chunk_metrics(self.CHUNKS, page_totals={"d1": 4})
        fidelity = result["page_fidelity"]
        assert fidelity["chunks_with_pages"] == 2
        assert fidelity["chunks_without_pages"] == 0
        assert fidelity["multi_page_chunks"] == 1
        assert fidelity["page_span"]["max"] == 2
        assert fidelity["coverage"]["value"] == pytest.approx(0.75)
        assert fidelity["coverage"]["denominator"] == 4
        assert fidelity["uncovered_pages"]["pages"] == {"d1": [3]}

    def test_page_zero_is_a_real_page(self) -> None:
        result = chunk_metrics(
            [{"chunk_id": "c1", "doc_id": "d1", "text": "abc", "pages": [0]}],
            page_totals={"d1": 1},
        )
        fidelity = result["page_fidelity"]
        assert fidelity["coverage"]["value"] == 1.0
        assert fidelity["uncovered_pages"]["pages"] == {"d1": []}

    def test_pages_outside_the_index_space_do_not_cover(self) -> None:
        result = chunk_metrics(
            [{"chunk_id": "c1", "doc_id": "d1", "text": "abc", "pages": [2]}],
            page_totals={"d1": 2},
        )
        fidelity = result["page_fidelity"]
        assert fidelity["coverage"]["value"] == 0.0
        assert fidelity["uncovered_pages"]["pages"] == {"d1": [0, 1]}

    def test_page_metadata_and_totals_are_typed_misses(self) -> None:
        result = chunk_metrics([{"chunk_id": "c1", "text": "abc"}])
        fidelity = result["page_fidelity"]
        assert fidelity["chunks_with_pages"] == 0
        assert fidelity["page_span"]["status"] == NOT_MEASURED
        assert fidelity["page_span"]["reason"] == "no_page_metadata"
        assert fidelity["coverage"]["status"] == NOT_MEASURED
        assert fidelity["coverage"]["reason"] == "no_page_totals"
        assert fidelity["uncovered_pages"]["status"] == NOT_MEASURED

    def test_empty_chunk_list(self) -> None:
        result = chunk_metrics([])
        assert result["status"] == NOT_MEASURED
        assert result["reason"] == "no_chunks"
        assert result["char_length"]["count"] == 0


class TestPageStructureMetrics:
    """Page numbering, emptiness and block histogram."""

    def test_numbers_gaps_and_duplicates(self) -> None:
        result = page_structure_metrics(
            {
                "doc_id": "d1",
                "pages": [
                    {"page": 1, "text": "a", "blocks": [{"kind": "heading"}, {"kind": "p"}]},
                    {"page": 3, "text": "", "blocks": [{"kind": "p"}]},
                ],
            }
        )
        assert result["pages"] == 2
        assert result["page_numbers"]["missing"] == [2]
        assert result["page_numbers"]["duplicates"] == []
        assert result["empty_text_pages"] == 1
        assert result["text_chars"]["count"] == 2
        assert result["blocks"]["total"] == 3
        assert result["blocks"]["by_kind"] == {"heading": 1, "p": 2}

    def test_duplicate_page_numbers(self) -> None:
        result = page_structure_metrics(
            {"doc_id": "d1", "pages": [{"page": 1}, {"page": 1}, {"page": 2}]}
        )
        assert result["page_numbers"]["duplicates"] == [1]
        assert result["page_numbers"]["missing"] == []

    def test_missing_page_numbers_are_typed_misses(self) -> None:
        none_at_all = page_structure_metrics({"doc_id": "d1", "pages": [{"text": "a"}]})
        assert none_at_all["page_numbers"]["status"] == NOT_MEASURED
        assert none_at_all["page_numbers"]["reason"] == "no_page_numbers"
        partial = page_structure_metrics({"doc_id": "d1", "pages": [{"page": 1}, {"text": "a"}]})
        assert partial["page_numbers"]["reason"] == "pages_without_number"

    def test_empty_document(self) -> None:
        result = page_structure_metrics({"doc_id": "d1"})
        assert result["pages"] == 0
        assert result["text_chars"]["status"] == NOT_MEASURED
        assert result["blocks"]["status"] == NOT_MEASURED
        assert result["blocks"]["reason"] == "no_blocks"
