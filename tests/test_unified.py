"""Tests for unified ShortcutDetector API."""

import numpy as np

from shortcut_detect import ShortcutDetector
from tests.fixtures.synthetic_data import generate_linear_shortcut, generate_no_shortcut


def test_unified_all_methods():
    """Test ShortcutDetector with all methods."""
    embeddings, labels = generate_linear_shortcut(n_samples=500, embedding_dim=30)

    detector = ShortcutDetector(methods=["hbac", "probe", "statistical"], seed=42)
    detector.fit(embeddings, labels)

    # Check all results are present
    assert "hbac" in detector.results_
    assert "probe" in detector.results_
    assert "statistical" in detector.results_

    # Check all succeeded
    assert detector.results_["hbac"]["success"]
    assert detector.results_["probe"]["success"]
    assert detector.results_["statistical"]["success"]

    # Check summary works
    summary = detector.summary()
    assert isinstance(summary, str)
    assert "UNIFIED SHORTCUT DETECTION SUMMARY" in summary
    assert "Risk: " in summary
    assert "Reason: " in summary
    assert "Risk level:" not in summary
    assert "Confidence:" not in summary


def test_unified_single_method():
    """Test ShortcutDetector with single method."""
    embeddings, labels = generate_linear_shortcut(n_samples=300, embedding_dim=20)

    detector = ShortcutDetector(methods=["hbac"])
    detector.fit(embeddings, labels)

    assert "hbac" in detector.results_
    assert "probe" not in detector.results_
    assert "statistical" not in detector.results_


def test_unified_probe_accuracy():
    """Test probe accuracy is reasonable."""
    embeddings, labels = generate_linear_shortcut(n_samples=600, embedding_dim=25, shortcut_dims=3)

    detector = ShortcutDetector(methods=["probe"])
    detector.fit(embeddings, labels)

    # Should detect shortcuts with high accuracy
    accuracy = detector.results_["probe"]["results"]["metrics"]["metric_value"]
    assert accuracy > 0.6  # Should be much better than random


def test_unified_no_shortcut():
    """Test on data without shortcuts."""
    embeddings, labels = generate_no_shortcut(n_samples=400, embedding_dim=20)

    detector = ShortcutDetector(methods=["hbac", "probe"])
    detector.fit(embeddings, labels)

    # Probe should have low accuracy
    if detector.results_["probe"]["success"]:
        accuracy = detector.results_["probe"]["results"]["metrics"]["metric_value"]
        assert 0.35 <= accuracy <= 0.65  # Random performance


def test_unified_with_group_labels():
    """Test ShortcutDetector with separate group labels."""
    embeddings, task_labels = generate_linear_shortcut(n_samples=500, embedding_dim=20)

    # Create separate group labels (e.g., demographic attributes)
    group_labels = np.random.randint(0, 3, len(task_labels))

    detector = ShortcutDetector(methods=["probe", "statistical"])
    detector.fit(embeddings, task_labels, group_labels=group_labels)

    # Probe and statistical should use group_labels
    assert detector.group_labels_ is not None
    assert len(detector.group_labels_) == len(task_labels)


def test_unified_get_results():
    """Test get_results method."""
    embeddings, labels = generate_linear_shortcut(n_samples=300, embedding_dim=15)

    detector = ShortcutDetector(methods=["hbac"])
    detector.fit(embeddings, labels)

    results = detector.get_results()
    assert isinstance(results, dict)
    assert "hbac" in results


def test_unified_custom_parameters():
    """Test ShortcutDetector with custom parameters."""
    embeddings, labels = generate_linear_shortcut(n_samples=400, embedding_dim=20)

    detector = ShortcutDetector(
        methods=["hbac"],
        hbac_max_iterations=5,
        hbac_min_cluster_size=0.05,
        seed=100,
    )

    detector.fit(embeddings, labels)

    # Check custom parameters were used
    hbac_detector = detector.detectors_["hbac"]
    assert hbac_detector.max_iterations == 5
    assert hbac_detector.min_cluster_size == 0.05


def test_unified_summary_before_fit():
    """Test summary raises appropriate message before fit."""
    detector = ShortcutDetector()
    summary = detector.summary()
    assert "No results available" in summary
