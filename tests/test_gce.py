"""Unit and integration tests for GCE (Generalized Cross Entropy) bias detector."""

import numpy as np
import pytest

from shortcut_detect import ShortcutDetector
from shortcut_detect.detector_base import RiskLevel
from shortcut_detect.gce import GCEDetector
from shortcut_detect.gce.gce_detector import _assess_risk, _gce_loss_per_sample, _softmax_stable
from tests.fixtures.synthetic_data import (
    generate_linear_shortcut,
    generate_multiclass_shortcut,
)

# --- Unit tests: GCE loss logic ---


def test_gce_loss_per_sample_binary():
    """Per-sample GCE loss: high prob for true class => low loss."""
    probs = np.array([[0.9, 0.1], [0.1, 0.9], [0.5, 0.5]])
    y_true = np.array([0, 1, 0])
    loss = _gce_loss_per_sample(probs, y_true, q=0.7)
    assert loss.shape == (3,)
    # Rows 0 and 1: p_true=0.9 -> low loss; row 2: p_true=0.5 -> higher loss
    assert loss[0] < loss[2] and loss[1] < loss[2]
    assert np.all(loss >= 0)


def test_gce_loss_per_sample_q_limits():
    """GCE with q=1 gives (1 - p_true); q in (0,1] only."""
    probs = np.array([[0.5, 0.5]])
    y_true = np.array([0])
    loss_q1 = _gce_loss_per_sample(probs, y_true, q=1.0)
    # GCE at q=1: (1 - p^1)/1 = 1 - p_true
    assert np.isclose(loss_q1[0], 1.0 - 0.5, rtol=1e-5)
    with pytest.raises(ValueError, match="q must be in"):
        _gce_loss_per_sample(probs, y_true, q=0.0)
    with pytest.raises(ValueError, match="q must be in"):
        _gce_loss_per_sample(probs, y_true, q=1.5)


def test_assess_risk_returns_risk_level_enum():
    risk_level, notes = _assess_risk(minority_ratio=0.30, loss_mean=0.7, n_minority=120)
    assert risk_level == RiskLevel.HIGH
    assert "high-loss samples" in notes


def test_softmax_stable():
    """Stable softmax sums to 1 and handles large logits."""
    logits = np.array([[1.0, 2.0, 3.0], [100.0, 100.0, 100.0]])
    probs = _softmax_stable(logits)
    assert probs.shape == logits.shape
    np.testing.assert_array_almost_equal(probs.sum(axis=1), np.ones(2))
    assert np.all(probs > 0) and np.all(probs <= 1)


# --- Unit tests: GCEDetector fit and minority-flagging ---


def test_gce_detector_fit_binary():
    """GCEDetector fits on binary data and populates losses and report."""
    embeddings, labels = generate_linear_shortcut(
        n_samples=300, embedding_dim=20, shortcut_dims=3, seed=42
    )
    detector = GCEDetector(q=0.7, loss_percentile_threshold=90.0, random_state=42)
    detector.fit(embeddings, labels)

    assert detector.per_sample_losses_ is not None
    assert len(detector.per_sample_losses_) == 300
    assert detector.is_minority_ is not None
    assert detector.is_minority_.dtype == bool
    assert detector.report_ is not None
    assert detector.report_.n_samples == 300
    assert detector.report_.n_minority >= 0
    assert detector.report_.loss_mean >= detector.report_.loss_min
    assert detector.report_.loss_mean <= detector.report_.loss_max
    assert detector.report_.q == 0.7
    assert detector.report_.risk_level in ("low", "moderate", "high")


def test_gce_detector_minority_flagging_percentile():
    """Roughly (100 - percentile)% of samples are flagged as minority at that percentile."""
    np.random.seed(42)
    n = 500
    embeddings = np.random.randn(n, 10).astype(np.float64) * 0.5
    labels = (np.random.rand(n) > 0.5).astype(int)

    detector = GCEDetector(
        q=0.7,
        loss_percentile_threshold=90.0,
        max_iter=200,
        random_state=42,
    )
    detector.fit(embeddings, labels)

    n_minority = int(np.sum(detector.is_minority_))
    # At 90th percentile we expect ~10% above threshold (may be slightly more due to ties)
    assert 0.05 * n <= n_minority <= 0.20 * n
    assert detector.report_.n_minority == n_minority
    assert np.all(detector.per_sample_losses_[detector.is_minority_] >= detector.loss_threshold_)


def test_gce_detector_multiclass():
    """GCEDetector fits on multiclass data."""
    embeddings, labels = generate_multiclass_shortcut(n_samples=400, embedding_dim=15, n_classes=3)
    detector = GCEDetector(q=0.7, loss_percentile_threshold=85.0, random_state=0)
    detector.fit(embeddings, labels)

    assert len(detector.classes_) == 3
    assert detector.per_sample_losses_.shape[0] == 400
    preds = detector.predict(embeddings)
    assert preds.shape[0] == 400
    assert np.all(np.isin(preds, detector.classes_))


def test_gce_detector_get_minority_indices():
    """get_minority_indices returns indices of high-loss samples."""
    embeddings, labels = generate_linear_shortcut(n_samples=200, embedding_dim=12, seed=1)
    detector = GCEDetector(q=0.7, loss_percentile_threshold=90.0, random_state=1)
    detector.fit(embeddings, labels)

    indices = detector.get_minority_indices()
    assert isinstance(indices, np.ndarray)
    assert np.all(detector.is_minority_[indices])
    assert len(indices) == detector.report_.n_minority


def test_gce_detector_invalid_q():
    """Invalid q raises ValueError."""
    with pytest.raises(ValueError, match="q must be"):
        GCEDetector(q=0.0)
    with pytest.raises(ValueError, match="q must be"):
        GCEDetector(q=1.5)


def test_gce_detector_invalid_percentile():
    """Invalid loss_percentile_threshold raises ValueError."""
    with pytest.raises(ValueError, match="loss_percentile_threshold"):
        GCEDetector(loss_percentile_threshold=101.0)
    with pytest.raises(ValueError, match="loss_percentile_threshold"):
        GCEDetector(loss_percentile_threshold=-1.0)


def test_gce_detector_predict_before_fit():
    """Predict before fit raises."""
    detector = GCEDetector(q=0.7, random_state=42)
    X = np.random.randn(5, 10)
    with pytest.raises(ValueError, match="fitted"):
        detector.predict(X)


def test_gce_detector_single_class_raises():
    """Single unique label raises ValueError."""
    X = np.random.randn(50, 5)
    y = np.zeros(50, dtype=int)
    detector = GCEDetector(q=0.7, random_state=42)
    with pytest.raises(ValueError, match="At least 2 distinct"):
        detector.fit(X, y)


# --- Integration tests: ShortcutDetector with GCE ---


def test_shortcut_detector_with_gce():
    """ShortcutDetector runs GCE when 'gce' is in methods."""
    embeddings, labels = generate_linear_shortcut(
        n_samples=350, embedding_dim=18, shortcut_dims=2, seed=42
    )
    detector = ShortcutDetector(methods=["gce"], seed=42, gce_q=0.7)
    detector.fit(embeddings, labels)

    assert "gce" in detector.results_
    res = detector.results_["gce"]
    assert res["success"] is True
    assert "detector" in res
    assert "report" in res
    assert "per_sample_losses" in res
    assert "is_minority" in res
    assert "minority_indices" in res
    assert "summary_title" in res
    assert "summary_lines" in res
    assert "risk_indicators" in res
    assert res["report"].n_samples == 350
    assert len(res["per_sample_losses"]) == 350
    assert len(res["minority_indices"]) == res["report"].n_minority


def test_shortcut_detector_gce_plus_other_methods():
    """ShortcutDetector runs GCE alongside HBAC and probe."""
    embeddings, labels = generate_linear_shortcut(n_samples=400, embedding_dim=20, seed=7)
    detector = ShortcutDetector(
        methods=["hbac", "probe", "gce"],
        seed=7,
        gce_loss_percentile_threshold=90.0,
    )
    detector.fit(embeddings, labels)

    assert detector.results_["hbac"]["success"]
    assert detector.results_["probe"]["success"]
    assert detector.results_["gce"]["success"]
    summary = detector.summary()
    assert "GCE" in summary or "Generalized Cross Entropy" in summary
    assert "Minority" in summary or "minority" in summary


def test_shortcut_detector_gce_custom_kwargs():
    """ShortcutDetector passes gce_* kwargs to GCE detector."""
    embeddings, labels = generate_linear_shortcut(n_samples=150, embedding_dim=10, seed=0)
    detector = ShortcutDetector(
        methods=["gce"],
        seed=0,
        gce_q=0.5,
        gce_loss_percentile_threshold=80.0,
        gce_max_iter=100,
    )
    detector.fit(embeddings, labels)
    assert detector.results_["gce"]["success"]
    report = detector.results_["gce"]["report"]
    assert report.q == 0.5
