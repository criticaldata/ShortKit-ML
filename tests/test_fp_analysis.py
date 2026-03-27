"""Tests for shortcut_detect.benchmark.fp_analysis module."""

from __future__ import annotations

import pytest

from shortcut_detect.benchmark.fp_analysis import FalsePositiveAnalyzer, FPResult


class TestFalsePositiveAnalyzer:
    @pytest.fixture()
    def small_analyzer(self):
        """Analyzer with few seeds for fast testing."""
        return FalsePositiveAnalyzer(
            methods=["probe", "geometric"],
            n_seeds=3,
            base_seed=9000,
        )

    def test_run_returns_fp_result(self, small_analyzer):
        result = small_analyzer.run(n_samples=200, embedding_dim=32)
        assert isinstance(result, FPResult)
        assert result.n_seeds == 3
        assert "probe" in result.method_fp_rates
        assert "geometric" in result.method_fp_rates

    def test_fp_rates_are_bounded(self, small_analyzer):
        result = small_analyzer.run(n_samples=200, embedding_dim=32)
        for method, rate in result.method_fp_rates.items():
            assert 0.0 <= rate <= 1.0, f"{method} FP rate out of bounds: {rate}"
        assert 0.0 <= result.convergence_fp_rate <= 1.0

    def test_fp_rates_not_all_one(self):
        """On clean data, at least some methods should not always fire."""
        analyzer = FalsePositiveAnalyzer(
            methods=["probe", "geometric"],
            n_seeds=5,
            base_seed=8000,
        )
        result = analyzer.run(n_samples=300, embedding_dim=64)
        all_one = all(r == 1.0 for r in result.method_fp_rates.values())
        assert not all_one, "All methods have FP rate 1.0 on clean data — suspicious"

    def test_convergence_fp_rate_leq_individual(self):
        """Convergence FP rate should be <= worst individual FP rate."""
        analyzer = FalsePositiveAnalyzer(
            methods=["probe", "geometric"],
            n_seeds=5,
            base_seed=8500,
        )
        result = analyzer.run(n_samples=300, embedding_dim=64)
        max_individual = max(result.method_fp_rates.values())
        assert result.convergence_fp_rate <= max_individual + 1e-9

    def test_per_seed_results_dataframe(self, small_analyzer):
        result = small_analyzer.run(n_samples=200, embedding_dim=32)
        df = result.per_seed_results
        assert "seed" in df.columns
        assert "method" in df.columns
        assert "flagged" in df.columns
        # 3 seeds * (2 methods + 1 convergence) = 9 rows
        assert len(df) == 9

    def test_summary_string(self, small_analyzer):
        result = small_analyzer.run(n_samples=200, embedding_dim=32)
        s = result.summary()
        assert isinstance(s, str)
        assert "False Positive Analysis" in s
        assert "probe" in s
        assert "geometric" in s
        assert "Convergence FP rate" in s
