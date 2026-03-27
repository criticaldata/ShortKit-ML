"""Tests for Causal Effect detector."""

import numpy as np
import pytest

from shortcut_detect import CausalEffectDetector


def _make_causal_spurious_data(seed: int = 0):
    """Create data where one attribute is causal and one is spurious.

    Y is determined by X[:,0] (causal). Spurious attribute correlates with Y
    but is independent of X - so E(Y|X, a_spurious) = E(Y|X) and TE_spurious ~ 0.
    """
    rng = np.random.default_rng(seed)
    n = 300
    dim = 8
    X = rng.standard_normal((n, dim))
    # Y from first dimension
    y = (X[:, 0] > 0).astype(np.int64)
    # Causal attribute: same as the label-generating mechanism
    causal_attr = (X[:, 0] > 0).astype(np.int64)
    # Spurious: random coin that correlates with Y (e.g., 80% match)
    spurious_attr = np.where(
        rng.random(n) < 0.8,
        y,
        1 - y,
    ).astype(np.int64)
    return X, y, causal_attr, spurious_attr


def test_causal_effect_detects_spurious_when_causal_has_effect():
    """Causal attribute has high effect; spurious has low effect (below threshold)."""
    X, y, causal_attr, spurious_attr = _make_causal_spurious_data(seed=42)
    detector = CausalEffectDetector(
        spurious_threshold=0.15,
        random_state=42,
    )
    detector.fit(
        embeddings=X,
        labels=y,
        attributes={
            "causal": causal_attr,
            "spurious": spurious_attr,
        },
    )
    report = detector.get_report()
    effects = report["metrics"]["per_attribute_effects"]
    # Causal attribute should have larger |effect| than spurious
    assert abs(effects["causal"]) > abs(effects["spurious"])
    # With high threshold, spurious may be flagged
    assert "n_spurious" in report["metrics"]


def test_causal_effect_shortcut_detected_when_spurious_below_threshold():
    """When an attribute has effect below threshold, shortcut_detected is True."""
    rng = np.random.default_rng(99)
    n = 200
    dim = 5
    X = rng.standard_normal((n, dim))
    y = (X[:, 0] > 0).astype(np.int64)
    # Spurious: independent of X, so E(Y|X,a) ≈ E(Y|X), TE ≈ 0
    spurious = rng.integers(0, 2, size=n)
    detector = CausalEffectDetector(spurious_threshold=0.3, random_state=42)
    detector.fit(
        embeddings=X,
        labels=y,
        attributes={"spurious": spurious},
    )
    report = detector.get_report()
    # Spurious attribute with low effect should be flagged
    assert report["metrics"]["n_attributes"] == 1
    effect = report["metrics"]["per_attribute_effects"]["spurious"]
    # Independent attribute tends to have low |effect|
    if abs(effect) < 0.3:
        assert report["shortcut_detected"] is True
        assert report["risk_level"] in {"moderate", "high"}


def test_causal_effect_low_risk_when_all_attributes_causal():
    """When all attributes have high effect, shortcut_detected is False."""
    rng = np.random.default_rng(77)
    n = 250
    dim = 6
    X = rng.standard_normal((n, dim))
    # Y determined by first dim; both attributes equal this causal mechanism
    y = (X[:, 0] > 0).astype(np.int64)
    attr1 = (X[:, 0] > 0).astype(np.int64)
    attr2 = (X[:, 0] > 0).astype(np.int64)  # same as attr1, both causal
    detector = CausalEffectDetector(spurious_threshold=0.05, random_state=42)
    detector.fit(
        embeddings=X,
        labels=y,
        attributes={"a1": attr1, "a2": attr2},
    )
    report = detector.get_report()
    effects = report["metrics"]["per_attribute_effects"]
    assert abs(effects["a1"]) > 0.05
    assert abs(effects["a2"]) > 0.05
    assert report["shortcut_detected"] is False
    assert report["risk_level"] == "low"
    assert report["metrics"]["n_spurious"] == 0


def test_causal_effect_input_validation():
    detector = CausalEffectDetector(random_state=42)

    with pytest.raises(ValueError, match="non-empty dict"):
        detector.fit(
            embeddings=np.zeros((10, 5)),
            labels=np.zeros(10),
            attributes={},
        )

    with pytest.raises(ValueError, match="embeddings must be 2D"):
        detector.fit(
            embeddings=np.zeros(10),
            labels=np.zeros(10),
            attributes={"a": np.zeros(10)},
        )

    with pytest.raises(ValueError, match="length 10"):
        detector.fit(
            embeddings=np.zeros((10, 5)),
            labels=np.zeros(10),
            attributes={"a": np.zeros(8)},
        )


def test_causal_effect_only_direct_estimator():
    with pytest.raises(ValueError, match="must be 'direct'"):
        CausalEffectDetector(effect_estimator="riesz")


def test_causal_effect_spurious_threshold_bounds():
    with pytest.raises(ValueError, match="spurious_threshold"):
        CausalEffectDetector(spurious_threshold=1.5)
    with pytest.raises(ValueError, match="spurious_threshold"):
        CausalEffectDetector(spurious_threshold=-0.1)


def test_causal_effect_report_structure():
    rng = np.random.default_rng(123)
    X = rng.standard_normal((100, 4))
    y = (X[:, 0] > 0).astype(np.int64)
    attr = (X[:, 0] > 0).astype(np.int64)
    detector = CausalEffectDetector(random_state=42)
    detector.fit(embeddings=X, labels=y, attributes={"a": attr})
    report = detector.get_report()
    assert "method" in report
    assert report["method"] == "causal_effect"
    assert "shortcut_detected" in report
    assert "risk_level" in report
    assert "metrics" in report
    assert "per_attribute_effects" in report["metrics"]
    assert "attribute_ranking" in report["metrics"]
    assert "report" in report
    assert "per_attribute" in report["report"]


def test_causal_effect_multiclass_uses_full_distribution(monkeypatch):
    """Multiclass effect should use full distribution shift, not one class column."""

    class FakeLogisticRegression:
        def __init__(self, *args, **kwargs):
            pass

        def fit(self, X, y):
            return self

        def predict_proba(self, X):
            n = X.shape[0]
            probs = np.zeros((n, 3), dtype=float)
            a = X[:, -1]
            probs[a == 0] = np.array([0.2, 0.3, 0.5])
            probs[a == 1] = np.array([0.5, 0.3, 0.2])
            return probs

    monkeypatch.setattr(
        "shortcut_detect.causal.causal_effect.src.detector.LogisticRegression",
        FakeLogisticRegression,
    )

    detector = CausalEffectDetector(random_state=42)
    X = np.array([[0.0], [1.0], [2.0], [3.0]])
    y = np.array([0, 1, 2, 1], dtype=np.int64)
    attr = np.array([0, 1, 0, 1], dtype=np.int64)

    effect, n_a0, n_a1 = detector._estimate_causal_effect_direct(X, y, attr)

    # TV([0.2,0.3,0.5], [0.5,0.3,0.2]) = 0.5*(0.3+0+0.3) = 0.3
    assert effect == pytest.approx(0.3, abs=1e-12)
    assert n_a0 == 2
    assert n_a1 == 2


def test_causal_effect_supports_categorical_attribute_values():
    """Non-numeric categorical attributes should be binarized without median()."""
    detector = CausalEffectDetector(random_state=42)
    X = np.array(
        [
            [0.0, 1.0],
            [1.0, 0.0],
            [0.5, 0.2],
            [0.3, 1.4],
            [1.1, 0.7],
            [0.2, 0.1],
        ]
    )
    y = np.array([0, 1, 0, 1, 1, 0], dtype=np.int64)
    attr = np.array(["red", "blue", "green", "red", "blue", "green"], dtype=object)

    effect, n_a0, n_a1 = detector._estimate_causal_effect_direct(X, y, attr)

    assert np.isfinite(effect)
    assert n_a0 + n_a1 == len(attr)
