"""Tests for embedding-space frequency shortcut detector."""

from __future__ import annotations

import numpy as np

from shortcut_detect import FrequencyDetector, ShortcutDetector
from shortcut_detect.reporting.csv_export import export_to_csv
from tests.fixtures.synthetic_data import generate_linear_shortcut


def test_frequency_detector_basic_fit():
    X, y = generate_linear_shortcut(n_samples=400, embedding_dim=24, shortcut_dims=4)
    det = FrequencyDetector(top_percent=0.1, probe_evaluation="holdout", probe_holdout_frac=0.2)
    det.fit(X.astype(np.float32), y.astype(np.int64))

    rep = det.get_report()
    assert rep["method"] == "frequency"
    assert isinstance(rep["shortcut_detected"], bool | np.bool_)
    assert rep["risk_level"] in {"low", "moderate", "high", "unknown"}
    assert "metrics" in rep and "report" in rep
    assert "class_rates" in rep["report"]
    assert "shortcut_classes" in rep["report"]


def test_frequency_top_dims_keys_use_true_class_labels():
    X, y = generate_linear_shortcut(n_samples=320, embedding_dim=18, shortcut_dims=3)
    # Shift labels to non-zero-based ids to catch row-index/key mismatches.
    y_shift = (y + 1).astype(np.int64)  # classes become {1, 2}
    det = FrequencyDetector(top_percent=0.1, probe_evaluation="train", random_state=11)
    det.fit(X.astype(np.float32), y_shift)
    rep = det.get_report()

    top_dims = rep["report"]["top_dims_by_class"]
    class_rate_keys = set(rep["report"]["class_rates"].keys())
    # For binary logistic, top dims may only exist for positive class, but key must be true class id.
    assert set(top_dims.keys()).issubset(class_rate_keys)
    assert "0" not in top_dims


def test_frequency_unified_integration_and_summary():
    X, y = generate_linear_shortcut(n_samples=300, embedding_dim=16, shortcut_dims=3)
    detector = ShortcutDetector(
        methods=["frequency"],
        seed=42,
        freq_probe_evaluation="holdout",
        freq_probe_holdout_frac=0.2,
    )
    detector.fit(X.astype(np.float32), y.astype(np.int64))

    assert "frequency" in detector.results_
    assert detector.results_["frequency"]["success"] is True
    # apply_standardized_risk should attach canonical fields
    assert "risk_value" in detector.results_["frequency"]
    assert "risk_reason" in detector.results_["frequency"]

    summary = detector.summary()
    assert "Embedding Frequency Shortcut" in summary


def test_frequency_unified_seed_propagates_to_detector():
    X, y = generate_linear_shortcut(n_samples=260, embedding_dim=14, shortcut_dims=2)
    detector = ShortcutDetector(methods=["frequency"], seed=123, freq_probe_evaluation="holdout")
    detector.fit(X.astype(np.float32), y.astype(np.int64))
    freq_report = detector.results_["frequency"]["report"]
    assert freq_report["metadata"]["random_state"] == 123


def test_frequency_csv_and_markdown_reporting(tmp_path):
    X, y = generate_linear_shortcut(n_samples=280, embedding_dim=20, shortcut_dims=3)
    detector = ShortcutDetector(methods=["frequency"], seed=0)
    detector.fit(X.astype(np.float32), y.astype(np.int64))

    csv_dir = tmp_path / "csv"
    exported = export_to_csv(detector, str(csv_dir))
    assert "overall_summary" in exported
    assert "frequency_class_rates" in exported

    md_path = tmp_path / "report.md"
    detector.generate_report(
        output_path=str(md_path),
        format="markdown",
        include_visualizations=False,
    )
    content = md_path.read_text()
    assert "Embedding Frequency Shortcut" in content
