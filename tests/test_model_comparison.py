"""Tests for model comparison runner."""

import numpy as np
import pandas as pd

from shortcut_detect import ComparisonResult, ModelComparisonRunner
from shortcut_detect.comparison.runner import _extract_summary_row


def _make_synthetic_embeddings(seed: int = 0, n: int = 120, d: int = 8):
    """Create synthetic embeddings and labels."""
    rng = np.random.RandomState(seed)
    emb = rng.randn(n, d).astype(np.float32)
    labels = rng.binomial(1, 0.5, n)
    groups = np.array(["A", "B"] * (n // 2))
    return emb, labels, groups


def test_model_comparison_runner_two_embeddings():
    """Test comparison with 2 precomputed embedding arrays."""
    emb1, labels, groups = _make_synthetic_embeddings(seed=1)
    emb2, _, _ = _make_synthetic_embeddings(seed=2)

    runner = ModelComparisonRunner(methods=["hbac", "probe", "statistical"])
    result = runner.run(
        model_sources=[
            ("model_a", emb1),
            ("model_b", emb2),
        ],
        labels=labels,
        group_labels=groups,
    )

    assert isinstance(result, ComparisonResult)
    assert result.model_ids == ["model_a", "model_b"]
    assert len(result.detectors) == 2
    assert "model_a" in result.detectors
    assert "model_b" in result.detectors
    assert result.summary_table.shape[0] == 2
    assert "model_id" in result.summary_table.columns
    assert list(result.summary_table["model_id"]) == ["model_a", "model_b"]


def test_comparison_result_to_dataframe():
    """ComparisonResult.to_dataframe returns copy of summary table."""
    emb1, labels, groups = _make_synthetic_embeddings(seed=3)
    emb2, _, _ = _make_synthetic_embeddings(seed=4)

    runner = ModelComparisonRunner(methods=["hbac", "probe"])
    result = runner.run(
        model_sources=[("a", emb1), ("b", emb2)],
        labels=labels,
        group_labels=groups,
    )

    df = result.to_dataframe()
    assert isinstance(df, pd.DataFrame)
    assert df.shape[0] == 2
    assert df is not result.summary_table


def test_extract_summary_row():
    """_extract_summary_row extracts key metrics from detector."""
    from shortcut_detect import ShortcutDetector

    emb, labels, groups = _make_synthetic_embeddings(seed=5)
    detector = ShortcutDetector(methods=["hbac", "probe"])
    detector.fit(embeddings=emb, labels=labels, group_labels=groups)

    row = _extract_summary_row(detector)
    assert "n_samples" in row
    assert "n_dimensions" in row
    assert row["n_samples"] == len(emb)
    assert row["n_dimensions"] == emb.shape[1]


def test_comparison_report_builder(tmp_path):
    """ComparisonReportBuilder generates HTML report."""
    from shortcut_detect.reporting.comparison_report import ComparisonReportBuilder

    emb1, labels, groups = _make_synthetic_embeddings(seed=10)
    emb2, _, _ = _make_synthetic_embeddings(seed=11)

    runner = ModelComparisonRunner(methods=["hbac", "probe"])
    result = runner.run(
        model_sources=[("m1", emb1), ("m2", emb2)],
        labels=labels,
        group_labels=groups,
    )

    builder = ComparisonReportBuilder(result)
    html_path = tmp_path / "comparison.html"
    content = builder.to_html(str(html_path))

    assert html_path.exists()
    assert "Model Comparison Report" in content
    assert "m1" in content
    assert "m2" in content


def test_export_comparison_to_csv(tmp_path):
    """export_comparison_to_csv creates comparison_summary.csv."""
    from shortcut_detect.reporting.csv_export import export_comparison_to_csv

    emb1, labels, groups = _make_synthetic_embeddings(seed=12)
    emb2, _, _ = _make_synthetic_embeddings(seed=13)

    runner = ModelComparisonRunner(methods=["hbac", "probe"])
    result = runner.run(
        model_sources=[("a", emb1), ("b", emb2)],
        labels=labels,
        group_labels=groups,
    )

    files = export_comparison_to_csv(result, str(tmp_path))

    assert "comparison_summary" in files
    summary_path = files["comparison_summary"]
    assert summary_path.endswith("comparison_summary.csv")
    import os

    assert os.path.exists(summary_path)
