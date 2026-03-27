"""Tests for baseline comparison module (B11)."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from shortcut_detect.benchmark.baseline_comparison import (
    BaselineComparison,
    ComparisonResult,
    ToolkitResult,
    generate_feature_comparison_table,
)


@pytest.fixture()
def synthetic_data():
    rng = np.random.RandomState(42)
    n = 200
    d = 32
    X = rng.randn(n, d).astype(np.float32)
    y = (rng.rand(n) > 0.5).astype(int)
    g = (rng.rand(n) > 0.5).astype(int)
    X[y == 1, :3] += 1.0
    return X, y, g


class TestFeatureComparisonTable:
    def test_returns_dataframe(self):
        table = generate_feature_comparison_table()
        assert isinstance(table, pd.DataFrame)

    def test_has_three_toolkit_columns(self):
        table = generate_feature_comparison_table()
        assert "ShortKit-ML" in table.columns
        assert "AIF360" in table.columns
        assert "Fairlearn" in table.columns

    def test_has_expected_features(self):
        table = generate_feature_comparison_table()
        features = table.index.tolist()
        assert "Embedding-space analysis" in features
        assert "Multi-method convergence" in features

    def test_to_latex(self):
        table = generate_feature_comparison_table()
        latex = table.to_latex()
        assert "\\begin{tabular}" in latex


class TestBaselineComparison:
    def test_run_shortcutdetect_only(self, synthetic_data):
        X, y, g = synthetic_data
        comp = BaselineComparison(include_fairlearn=False, include_aif360=False)
        result = comp.run(X, y, g)
        assert isinstance(result, ComparisonResult)
        assert "ShortKit-ML" in result.toolkit_results

    def test_graceful_without_external(self, synthetic_data):
        X, y, g = synthetic_data
        comp = BaselineComparison(include_fairlearn=True, include_aif360=True)
        result = comp.run(X, y, g)
        assert isinstance(result, ComparisonResult)
        assert "ShortKit-ML" in result.toolkit_results

    def test_comparison_table(self, synthetic_data):
        X, y, g = synthetic_data
        comp = BaselineComparison(include_fairlearn=False, include_aif360=False)
        result = comp.run(X, y, g)
        table = result.comparison_table()
        assert isinstance(table, pd.DataFrame)

    def test_summary(self, synthetic_data):
        X, y, g = synthetic_data
        comp = BaselineComparison(include_fairlearn=False, include_aif360=False)
        result = comp.run(X, y, g)
        summary = result.summary()
        assert isinstance(summary, str)
        assert len(summary) > 0


class TestToolkitResult:
    def test_creation(self):
        tr = ToolkitResult(
            toolkit_name="test",
            metrics={"accuracy": 0.9},
            supported_features=["equalized_odds"],
            errors=[],
        )
        assert tr.toolkit_name == "test"
        assert tr.metrics["accuracy"] == 0.9
