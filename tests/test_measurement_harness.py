"""Tests for shortcut_detect.benchmark.measurement module."""

from __future__ import annotations

import numpy as np
import pytest

from shortcut_detect.benchmark.measurement import (
    HarnessResult,
    MeasurementHarness,
    MethodResult,
    bootstrap_ci,
    method_detected,
    precision_recall_f1,
    probe_permutation_pvalue,
)
from shortcut_detect.benchmark.synthetic_generator import SyntheticGenerator

# ---------------------------------------------------------------------------
# precision_recall_f1
# ---------------------------------------------------------------------------


class TestPrecisionRecallF1:
    def test_perfect_match(self):
        p, r, f1, j = precision_recall_f1([0, 1, 2], [0, 1, 2])
        assert p == 1.0
        assert r == 1.0
        assert f1 == 1.0
        assert j == 1.0

    def test_no_overlap(self):
        p, r, f1, j = precision_recall_f1([0, 1], [2, 3])
        assert p == 0.0
        assert r == 0.0
        assert f1 == 0.0
        assert j == 0.0

    def test_partial_overlap(self):
        p, r, f1, j = precision_recall_f1([0, 1, 2], [0, 1, 3])
        assert p == pytest.approx(2 / 3)
        assert r == pytest.approx(2 / 3)
        assert f1 == pytest.approx(2 / 3)
        assert j == pytest.approx(2 / 4)

    def test_empty_predicted(self):
        p, r, f1, j = precision_recall_f1([], [0, 1])
        assert p == 0.0
        assert r == 0.0
        assert f1 == 0.0

    def test_both_empty(self):
        p, r, f1, j = precision_recall_f1([], [])
        # Convention: both empty => jaccard 1.0
        assert j == 1.0

    def test_numpy_arrays(self):
        p, r, f1, j = precision_recall_f1(np.array([0, 1]), np.array([0, 1]))
        assert p == 1.0
        assert r == 1.0


# ---------------------------------------------------------------------------
# method_detected
# ---------------------------------------------------------------------------


class TestMethodDetected:
    def test_probe_detected(self):
        result = {"success": True, "results": {"shortcut_detected": True}}
        assert method_detected("probe", result) is True

    def test_probe_not_detected(self):
        result = {"success": True, "results": {"shortcut_detected": False}}
        assert method_detected("probe", result) is False

    def test_hbac_detected(self):
        result = {"success": True, "report": {"has_shortcut": {"exists": True}}}
        assert method_detected("hbac", result) is True

    def test_hbac_not_detected(self):
        result = {"success": True, "report": {"has_shortcut": {"exists": False}}}
        assert method_detected("hbac", result) is False

    def test_statistical_detected(self):
        result = {"success": True, "significant_features": {"bonferroni": [0, 1]}}
        assert method_detected("statistical", result) is True

    def test_statistical_not_detected(self):
        result = {"success": True, "significant_features": {"bonferroni": []}}
        assert method_detected("statistical", result) is False

    def test_geometric_detected_high(self):
        result = {"success": True, "summary": {"risk_level": "high"}}
        assert method_detected("geometric", result) is True

    def test_geometric_detected_moderate(self):
        result = {"success": True, "summary": {"risk_level": "moderate"}}
        assert method_detected("geometric", result) is True

    def test_geometric_not_detected(self):
        result = {"success": True, "summary": {"risk_level": "low"}}
        assert method_detected("geometric", result) is False

    def test_failed_result(self):
        result = {"success": False}
        assert method_detected("probe", result) is False
        assert method_detected("hbac", result) is False
        assert method_detected("statistical", result) is False
        assert method_detected("geometric", result) is False

    def test_unknown_method(self):
        result = {"success": True}
        assert method_detected("unknown_method", result) is False


# ---------------------------------------------------------------------------
# MeasurementHarness.evaluate
# ---------------------------------------------------------------------------


class TestMeasurementHarness:
    @pytest.fixture()
    def strong_shortcut_data(self):
        """Generate data with a strong shortcut (effect_size=2.0)."""
        gen = SyntheticGenerator(
            n_samples=500,
            embedding_dim=64,
            shortcut_dims=5,
            group_ratio=0.5,
            seed=42,
        )
        return gen.generate(effect_size=2.0)

    def test_evaluate_returns_harness_result(self, strong_shortcut_data):
        data = strong_shortcut_data
        harness = MeasurementHarness(methods=["probe", "geometric"], seed=42)
        result = harness.evaluate(
            data.embeddings,
            data.labels,
            data.group_labels,
            data.shortcut_dims,
        )
        assert isinstance(result, HarnessResult)
        assert len(result.method_results) == 2
        for mr in result.method_results:
            assert isinstance(mr, MethodResult)
            assert mr.method in ("probe", "geometric")
            assert 0.0 <= mr.precision <= 1.0
            assert 0.0 <= mr.recall <= 1.0

    def test_evaluate_strong_shortcut_has_high_scores(self, strong_shortcut_data):
        data = strong_shortcut_data
        harness = MeasurementHarness(methods=["probe", "geometric"], seed=42)
        result = harness.evaluate(
            data.embeddings,
            data.labels,
            data.group_labels,
            data.shortcut_dims,
        )
        # With effect_size=2.0, probe and geometric should have decent recall
        for mr in result.method_results:
            assert mr.recall >= 0.4, f"{mr.method} recall too low: {mr.recall}"

    def test_convergence_level_format(self, strong_shortcut_data):
        data = strong_shortcut_data
        harness = MeasurementHarness(methods=["probe", "geometric"], seed=42)
        result = harness.evaluate(
            data.embeddings,
            data.labels,
            data.group_labels,
            data.shortcut_dims,
        )
        # Should be in "X/Y" format
        parts = result.convergence_level.split("/")
        assert len(parts) == 2
        assert int(parts[1]) == 2

    def test_convergence_bucket_values(self):
        harness = MeasurementHarness(methods=["probe"], seed=42)
        assert harness._convergence_bucket(0, 4) == "no_detection"
        assert harness._convergence_bucket(1, 4) == "likely_false_alarm"
        assert harness._convergence_bucket(2, 4) == "intermediate"
        assert harness._convergence_bucket(3, 4) == "moderate_confidence"
        assert harness._convergence_bucket(4, 4) == "high_confidence"

    def test_evaluate_batch(self, strong_shortcut_data):
        data = strong_shortcut_data
        harness = MeasurementHarness(methods=["probe"], seed=42)
        datasets = [
            {
                "embeddings": data.embeddings,
                "labels": data.labels,
                "group_labels": data.group_labels,
                "true_shortcut_dims": data.shortcut_dims,
            }
        ]
        df = harness.evaluate_batch(datasets, seeds=[42])
        assert len(df) == 1
        assert "method" in df.columns
        assert "precision" in df.columns
        assert "convergence_level" in df.columns

    def test_unsupported_method_raises(self):
        with pytest.raises(ValueError, match="Unsupported method"):
            MeasurementHarness(methods=["not_a_method"])


# ---------------------------------------------------------------------------
# probe_permutation_pvalue (S01)
# ---------------------------------------------------------------------------


class TestProbePermutationPvalue:
    def test_returns_expected_keys(self):
        rng = np.random.RandomState(0)
        X = rng.randn(100, 10)
        y = (X[:, 0] > 0).astype(int)
        result = probe_permutation_pvalue(X, y, n_permutations=20, seed=42)
        assert set(result.keys()) == {
            "observed_accuracy",
            "null_mean",
            "null_std",
            "p_value",
            "n_permutations",
        }
        assert result["n_permutations"] == 20

    def test_strong_signal_low_pvalue(self):
        rng = np.random.RandomState(1)
        n = 200
        X = rng.randn(n, 5)
        y = (X[:, 0] > 0).astype(int)
        # Add strong signal
        X[:, 0] += 3.0 * y
        result = probe_permutation_pvalue(X, y, n_permutations=50, seed=42)
        assert result["observed_accuracy"] > result["null_mean"]
        assert result["p_value"] <= 0.1

    def test_no_signal_high_pvalue(self):
        rng = np.random.RandomState(2)
        X = rng.randn(100, 5)
        y = rng.randint(0, 2, size=100)
        result = probe_permutation_pvalue(X, y, n_permutations=50, seed=42)
        # With random labels, p-value should not be very small
        assert result["p_value"] >= 0.0


# ---------------------------------------------------------------------------
# bootstrap_ci (S02)
# ---------------------------------------------------------------------------


class TestBootstrapCI:
    def test_returns_expected_keys(self):
        result = bootstrap_ci(np.array([1.0, 2.0, 3.0, 4.0, 5.0]))
        assert set(result.keys()) == {
            "mean",
            "ci_lower",
            "ci_upper",
            "std",
            "n_bootstrap",
        }

    def test_ci_contains_mean(self):
        vals = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
        result = bootstrap_ci(vals, n_bootstrap=5000, seed=42)
        assert result["ci_lower"] <= result["mean"] <= result["ci_upper"]

    def test_single_value(self):
        result = bootstrap_ci(np.array([42.0]))
        assert result["mean"] == 42.0
        assert result["ci_lower"] == 42.0
        assert result["ci_upper"] == 42.0
        assert result["std"] == 0.0

    def test_empty_values(self):
        result = bootstrap_ci(np.array([]))
        assert np.isnan(result["mean"])
        assert np.isnan(result["ci_lower"])

    def test_nan_values_filtered(self):
        result = bootstrap_ci(np.array([1.0, np.nan, 3.0, np.nan, 5.0]))
        assert not np.isnan(result["mean"])
        assert result["ci_lower"] <= result["ci_upper"]

    def test_n_bootstrap_parameter(self):
        result = bootstrap_ci(np.array([1.0, 2.0, 3.0]), n_bootstrap=100)
        assert result["n_bootstrap"] == 100
