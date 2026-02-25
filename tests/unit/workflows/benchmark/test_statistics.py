"""Tests for statistics.py — Cliff's delta, effect size classification."""

from __future__ import annotations

import pytest

from workflows.benchmark.statistics import (
    _safe_mean,
    classify_effect_size,
    cliffs_delta,
)


class TestCliffsDelta:
    def test_identical_lists(self):
        x = [1.0, 2.0, 3.0]
        y = [1.0, 2.0, 3.0]
        assert cliffs_delta(x, y) == 0.0

    def test_x_dominates(self):
        x = [4.0, 5.0, 6.0]
        y = [1.0, 2.0, 3.0]
        assert cliffs_delta(x, y) == 1.0

    def test_y_dominates(self):
        x = [1.0, 2.0, 3.0]
        y = [4.0, 5.0, 6.0]
        assert cliffs_delta(x, y) == -1.0

    def test_partial_overlap(self):
        x = [1.0, 2.0, 5.0]
        y = [3.0, 4.0, 6.0]
        delta = cliffs_delta(x, y)
        assert -1.0 < delta < 0.0

    def test_empty_lists(self):
        assert cliffs_delta([], [1.0]) == 0.0
        assert cliffs_delta([1.0], []) == 0.0


class TestClassifyEffectSize:
    def test_negligible(self):
        assert classify_effect_size(0.1) == "negligible"

    def test_small(self):
        assert classify_effect_size(0.2) == "small"

    def test_medium(self):
        assert classify_effect_size(0.4) == "medium"

    def test_large(self):
        assert classify_effect_size(0.5) == "large"

    def test_negative_delta(self):
        assert classify_effect_size(-0.5) == "large"


class TestSafeMean:
    def test_normal(self):
        assert _safe_mean([1.0, 2.0, 3.0]) == pytest.approx(2.0)

    def test_empty(self):
        assert _safe_mean([]) == 0.0

    def test_single(self):
        assert _safe_mean([5.0]) == 5.0
