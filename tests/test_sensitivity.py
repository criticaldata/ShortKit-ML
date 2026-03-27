"""Tests for shortcut_detect.benchmark.sensitivity module."""

from __future__ import annotations

import tempfile
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from shortcut_detect.benchmark.sensitivity import (
    SensitivitySweep,
    SweepResult,
    _method_flag,
    _precision_recall_f1,
)

# Use small sizes / few seeds everywhere to keep tests fast.
_FAST_SEEDS = 2
_SMALL_SAMPLES = 60
_SMALL_DIM = 16


@pytest.fixture
def sweep():
    """Return a SensitivitySweep with all four methods."""
    return SensitivitySweep(
        methods=["hbac", "probe", "statistical", "geometric"],
        shortcut_dims=3,
        base_seed=0,
    )


# ------------------------------------------------------------------
# SweepResult tests
# ------------------------------------------------------------------


class TestSweepResult:
    def _make_result(self) -> SweepResult:
        rows = []
        for pval in [100, 200]:
            for method in ["hbac", "probe"]:
                for seed in [1, 2]:
                    rows.append(
                        {
                            "param_value": pval,
                            "method": method,
                            "seed": seed,
                            "detected": 1,
                            "precision": 0.8,
                            "recall": 0.6,
                            "f1": 0.685,
                            "convergence_count": 2,
                        }
                    )
        return SweepResult(
            sweep_param="sample_size",
            param_values=[100, 200],
            results_df=pd.DataFrame(rows),
        )

    def test_summary_shape(self):
        result = self._make_result()
        summary = result.summary()
        # 2 param_values x 2 methods = 4 rows
        assert len(summary) == 4
        # Should contain mean/std columns
        assert "detected_mean" in summary.columns
        assert "detected_std" in summary.columns
        assert "f1_mean" in summary.columns

    def test_summary_values(self):
        result = self._make_result()
        summary = result.summary()
        # All detected values are 1, so mean should be 1.0
        assert (summary["detected_mean"] == 1.0).all()

    def test_to_csv(self):
        result = self._make_result()
        with tempfile.TemporaryDirectory() as tmpdir:
            csv_path = Path(tmpdir) / "sub" / "results.csv"
            result.to_csv(csv_path)
            assert csv_path.exists()
            loaded = pd.read_csv(csv_path)
            assert len(loaded) == len(result.results_df)
            assert set(loaded.columns) == set(result.results_df.columns)


# ------------------------------------------------------------------
# SensitivitySweep integration tests
# ------------------------------------------------------------------


class TestSweepSampleSize:
    def test_basic(self, sweep):
        result = sweep.sweep_sample_size(
            sample_sizes=[_SMALL_SAMPLES, _SMALL_SAMPLES * 2],
            effect_size=1.5,
            embedding_dim=_SMALL_DIM,
            n_seeds=_FAST_SEEDS,
        )
        assert isinstance(result, SweepResult)
        assert result.sweep_param == "n_samples"
        assert result.param_values == [_SMALL_SAMPLES, _SMALL_SAMPLES * 2]
        # 2 sizes x 4 methods x 2 seeds = 16 rows
        assert len(result.results_df) == 2 * 4 * _FAST_SEEDS
        assert set(result.results_df.columns) >= {
            "param_value",
            "method",
            "seed",
            "detected",
            "precision",
            "recall",
            "f1",
            "convergence_count",
        }


class TestSweepImbalance:
    def test_basic(self, sweep):
        result = sweep.sweep_imbalance(
            group_ratios=[0.5, 0.7],
            effect_size=1.5,
            n_samples=_SMALL_SAMPLES,
            embedding_dim=_SMALL_DIM,
            n_seeds=_FAST_SEEDS,
        )
        assert isinstance(result, SweepResult)
        assert result.sweep_param == "group_ratio"
        assert len(result.results_df) == 2 * 4 * _FAST_SEEDS


class TestSweepDimensionality:
    def test_basic(self, sweep):
        result = sweep.sweep_dimensionality(
            embedding_dims=[_SMALL_DIM, _SMALL_DIM * 2],
            effect_size=1.5,
            n_samples=_SMALL_SAMPLES,
            n_seeds=_FAST_SEEDS,
        )
        assert isinstance(result, SweepResult)
        assert result.sweep_param == "embedding_dim"
        assert len(result.results_df) == 2 * 4 * _FAST_SEEDS


class TestSweepCustom:
    def test_basic(self, sweep):
        result = sweep.sweep_custom(
            param_name="effect_size",
            param_values=[0.5, 1.5],
            fixed_params={
                "n_samples": _SMALL_SAMPLES,
                "embedding_dim": _SMALL_DIM,
                "effect_size": 0.0,
                "group_ratio": 0.5,
            },
            n_seeds=_FAST_SEEDS,
        )
        assert isinstance(result, SweepResult)
        assert result.sweep_param == "effect_size"

    def test_missing_keys_raises(self, sweep):
        with pytest.raises(ValueError, match="missing required keys"):
            sweep.sweep_custom(
                param_name="n_samples",
                param_values=[100],
                fixed_params={"n_samples": 100},
                n_seeds=1,
            )


# ------------------------------------------------------------------
# Unit tests for helpers
# ------------------------------------------------------------------


class TestMethodFlag:
    def test_probe_detected(self):
        assert _method_flag("probe", {"success": True, "results": {"shortcut_detected": True}})

    def test_probe_not_detected(self):
        assert not _method_flag("probe", {"success": True, "results": {"shortcut_detected": False}})

    def test_failure_returns_false(self):
        assert not _method_flag("hbac", {"success": False})

    def test_unknown_method(self):
        assert not _method_flag("unknown", {"success": True})


class TestPrecisionRecallF1:
    def test_perfect(self):
        p, r, f = _precision_recall_f1(np.array([0, 1, 2]), np.array([0, 1, 2]))
        assert p == 1.0 and r == 1.0 and f == 1.0

    def test_empty_pred(self):
        p, r, f = _precision_recall_f1(np.array([]), np.array([0, 1]))
        assert p == 0.0 and r == 0.0 and f == 0.0


class TestUnsupportedMethod:
    def test_raises(self):
        with pytest.raises(ValueError, match="Unsupported method"):
            SensitivitySweep(methods=["nonexistent"])
