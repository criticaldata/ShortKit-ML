"""Tests for CAV detector."""

import numpy as np
import pytest

from shortcut_detect import CAVDetector


def _make_strong_signal(seed: int = 0):
    rng = np.random.default_rng(seed)
    dim = 12
    concept = rng.normal(0.0, 0.5, size=(80, dim))
    random = rng.normal(0.0, 0.5, size=(80, dim))
    concept[:, 0] += 2.0
    random[:, 0] -= 2.0

    target_acts = rng.normal(0.0, 1.0, size=(120, dim))
    target_dd = rng.normal(0.0, 0.2, size=(120, dim))
    target_dd[:, 0] += 1.0

    return concept, random, target_acts, target_dd


def test_cav_detects_strong_shortcut_concept():
    concept, random, target_acts, target_dd = _make_strong_signal()

    detector = CAVDetector(
        shortcut_threshold=0.6,
        quality_threshold=0.7,
        min_examples_per_set=20,
        random_state=7,
    )
    detector.fit(
        concept_sets={"shortcut": concept},
        random_set=random,
        target_activations=target_acts,
        target_directional_derivatives=target_dd,
    )

    report = detector.get_report()
    assert report["shortcut_detected"] is True
    assert report["risk_level"] in {"moderate", "high"}
    assert report["metrics"]["n_flagged"] >= 1
    assert report["metrics"]["max_tcav_score"] is not None


def test_cav_returns_unknown_without_directional_derivatives():
    concept, random, target_acts, _ = _make_strong_signal()

    detector = CAVDetector(min_examples_per_set=20)
    detector.fit(
        concept_sets={"shortcut": concept},
        random_set=random,
        target_activations=target_acts,
        target_directional_derivatives=None,
    )

    report = detector.get_report()
    assert report["shortcut_detected"] is None
    assert report["risk_level"] == "unknown"
    assert report["metrics"]["n_tested"] == 0


def test_cav_low_risk_when_unrelated_concept():
    rng = np.random.default_rng(4)
    dim = 10
    concept = rng.normal(0.0, 1.0, size=(60, dim))
    random = rng.normal(0.0, 1.0, size=(60, dim))

    # Force class separation on dim 0 but derivatives anti-align to keep TCAV low.
    concept[:, 0] += 2.0
    random[:, 0] -= 2.0

    target_dd = rng.normal(0.0, 0.2, size=(90, dim))
    target_dd[:, 0] -= 1.0

    detector = CAVDetector(shortcut_threshold=0.6, quality_threshold=0.7, min_examples_per_set=20)
    detector.fit(
        concept_sets={"artifact": concept},
        random_set=random,
        target_directional_derivatives=target_dd,
    )

    report = detector.get_report()
    assert report["shortcut_detected"] is False
    assert report["risk_level"] == "low"
    assert report["metrics"]["n_flagged"] == 0


def test_cav_input_validation_shapes_and_min_examples():
    detector = CAVDetector(min_examples_per_set=20)

    with pytest.raises(ValueError, match="non-empty dictionary"):
        detector.fit(concept_sets={}, random_set=np.zeros((30, 8)))

    with pytest.raises(ValueError, match="requires at least"):
        detector.fit(
            concept_sets={"small": np.zeros((5, 8))},
            random_set=np.zeros((30, 8)),
            target_directional_derivatives=np.zeros((10, 8)),
        )

    with pytest.raises(ValueError, match="feature dimension mismatch"):
        detector.fit(
            concept_sets={"ok": np.zeros((30, 8))},
            random_set=np.zeros((30, 8)),
            target_directional_derivatives=np.zeros((10, 7)),
        )
