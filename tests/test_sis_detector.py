"""Tests for SIS (Sufficient Input Subsets) detector."""

import numpy as np
import pytest

from shortcut_detect import ShortcutDetector
from shortcut_detect.xai import SISDetector


def _make_shortcut_dataset(seed=42, n_samples=200, n_dim=20, shortcut_dims=5):
    """Create dataset where first shortcut_dims correlate with labels (shortcut)."""
    rng = np.random.RandomState(seed)
    labels = rng.binomial(1, 0.5, n_samples)
    embeddings = rng.randn(n_samples, n_dim).astype(np.float32) * 0.5
    # Strong shortcut: first 5 dims fully predict label
    for d in range(shortcut_dims):
        embeddings[:, d] += labels * 3.0
    return embeddings, labels


def _make_no_shortcut_dataset(seed=42, n_samples=200, n_dim=20):
    """Create dataset with no clear shortcut (all dims weak)."""
    rng = np.random.RandomState(seed)
    labels = rng.binomial(1, 0.5, n_samples)
    embeddings = rng.randn(n_samples, n_dim).astype(np.float32) * 0.3
    embeddings[:, 0] += labels * 0.5  # weak signal
    return embeddings, labels


def test_sis_detector_fit_basic():
    """Test SIS detector fit with basic embeddings and labels."""
    embeddings, labels = _make_shortcut_dataset(n_samples=150, n_dim=16)
    detector = SISDetector(max_samples=50, test_size=0.3, seed=42)
    detector.fit(embeddings, labels)

    assert detector._is_fitted
    assert detector.results_ is not None
    assert "mean_sis_size" in detector.results_["metrics"]
    assert "risk_level" in detector.results_
    assert detector.results_["risk_level"] in ("low", "moderate", "high", "unknown")


def test_sis_detector_with_strong_shortcut():
    """Test that strong shortcut yields small SIS (potential shortcut signal)."""
    embeddings, labels = _make_shortcut_dataset(n_samples=300, n_dim=30, shortcut_dims=5)
    detector = SISDetector(
        max_samples=80,
        test_size=0.25,
        shortcut_threshold=0.3,
        seed=42,
    )
    detector.fit(embeddings, labels)

    metrics = detector.results_["metrics"]
    mean_sis = metrics.get("mean_sis_size")
    assert mean_sis is not None
    # With strong shortcut, SIS should be small (few dims suffice)
    assert mean_sis < 20
    assert len(detector.sis_sizes_) > 0


def test_sis_detector_with_group_labels():
    """Test SIS detector with group_labels for group-SIS overlap."""
    embeddings, labels = _make_shortcut_dataset(n_samples=200, n_dim=20)
    groups = np.array(["A"] * 100 + ["B"] * 100)
    detector = SISDetector(max_samples=60, seed=42)
    detector.fit(embeddings, labels, group_labels=groups)

    assert detector._is_fitted
    assert detector.results_["metrics"].get("n_computed") is not None


def test_sis_detector_results_schema():
    """Test that results_ has required schema."""
    embeddings, labels = _make_shortcut_dataset(n_samples=120, n_dim=12)
    detector = SISDetector(max_samples=50, seed=42)
    detector.fit(embeddings, labels)

    r = detector.results_
    assert "method" in r
    assert r["method"] == "sis"
    assert "shortcut_detected" in r
    assert "risk_level" in r
    assert "metrics" in r
    assert "report" in r
    assert "risk_level" in r
    assert r["risk_level"] in ("low", "moderate", "high", "unknown")


def test_sis_detector_unified():
    """Test SIS via ShortcutDetector unified API."""
    embeddings, labels = _make_shortcut_dataset(n_samples=150, n_dim=16)
    detector = ShortcutDetector(methods=["sis"], seed=42)
    detector.fit(embeddings, labels)

    assert "sis" in detector.results_
    assert detector.results_["sis"]["success"]
    assert "mean_sis_size" in detector.results_["sis"].get("metrics", {})


def test_sis_detector_small_dataset():
    """Test SIS with small dataset (edge case)."""
    embeddings, labels = _make_shortcut_dataset(n_samples=50, n_dim=8)
    detector = SISDetector(max_samples=20, test_size=0.3, seed=42)
    detector.fit(embeddings, labels)

    assert detector._is_fitted
    # May have 0 or few computed due to small test set
    n_computed = detector.results_["metrics"].get("n_computed", 0)
    assert n_computed >= 0


def test_sis_detector_invalid_inputs():
    """Test that invalid inputs raise."""
    embeddings, labels = _make_shortcut_dataset(n_samples=50, n_dim=10)

    with pytest.raises(ValueError):
        SISDetector(max_samples=0)

    with pytest.raises(ValueError):
        SISDetector(test_size=0.0)

    with pytest.raises(ValueError):
        detector = SISDetector(max_samples=50)
        detector.fit(embeddings[:10], labels[:5])  # shape mismatch
