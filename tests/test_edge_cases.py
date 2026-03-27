"""Tests for degenerate, malformed, and stress inputs."""

import numpy as np

from shortcut_detect import ShortcutDetector


class TestDegenerateInputs:
    """Edge cases involving unusual but technically valid data."""

    def test_all_zero_embeddings(self):
        """All-zero embeddings should not crash; methods may report low risk."""
        X = np.zeros((100, 10))
        y = np.random.RandomState(0).randint(0, 2, 100)
        detector = ShortcutDetector(methods=["statistical"])
        detector.fit(X, y)
        assert "statistical" in detector.results_

    def test_constant_embeddings_per_class(self):
        """Each class maps to a single embedding vector."""
        X = np.vstack([np.ones((50, 10)) * 0.0, np.ones((50, 10)) * 1.0])
        y = np.array([0] * 50 + [1] * 50)
        detector = ShortcutDetector(methods=["probe"])
        detector.fit(X, y)
        assert detector.results_["probe"]["success"]

    def test_extremely_small_sample_size(self):
        """n=4 (minimum allowed) should not crash."""
        X = np.random.RandomState(42).randn(4, 5)
        y = np.array([0, 0, 1, 1])
        detector = ShortcutDetector(methods=["statistical"])
        detector.fit(X, y)
        assert "statistical" in detector.results_

    def test_single_feature(self):
        """Embeddings with d=1 should work."""
        X = np.random.RandomState(0).randn(100, 1)
        y = np.random.RandomState(0).randint(0, 2, 100)
        detector = ShortcutDetector(methods=["statistical"])
        detector.fit(X, y)
        assert "statistical" in detector.results_

    def test_multiclass_labels(self):
        """Labels with more than 2 classes should work."""
        X = np.random.RandomState(0).randn(150, 10)
        y = np.random.RandomState(0).randint(0, 5, 150)
        detector = ShortcutDetector(methods=["probe", "statistical"])
        detector.fit(X, y)
        assert detector.results_["probe"]["success"]

    def test_non_contiguous_array(self):
        """Sliced (non-contiguous) arrays should not crash."""
        X_full = np.random.RandomState(0).randn(200, 20)
        X = X_full[::2, ::2]  # every other row and col -> (100, 10)
        y = np.random.RandomState(0).randint(0, 2, 100)
        assert not X.flags["C_CONTIGUOUS"]
        detector = ShortcutDetector(methods=["probe"])
        detector.fit(X, y)
        assert detector.results_["probe"]["success"]

    def test_float32_embeddings(self):
        """float32 arrays should be accepted (common from PyTorch)."""
        X = np.random.RandomState(0).randn(100, 10).astype(np.float32)
        y = np.random.RandomState(0).randint(0, 2, 100)
        detector = ShortcutDetector(methods=["probe"])
        detector.fit(X, y)
        assert detector.results_["probe"]["success"]

    def test_string_labels(self):
        """String labels should not crash; probe may fail gracefully."""
        X = np.random.RandomState(0).randn(100, 10)
        y = np.array(["cat"] * 50 + ["dog"] * 50)
        detector = ShortcutDetector(methods=["statistical"])
        detector.fit(X, y)
        # statistical method handles arbitrary labels
        assert "statistical" in detector.results_


class TestHighDimensionalStress:
    """Stress tests for wide (d >> n) and large inputs."""

    def test_d_much_greater_than_n(self):
        """d=500, n=50: should not OOM or crash."""
        X = np.random.RandomState(0).randn(50, 500)
        y = np.random.RandomState(0).randint(0, 2, 50)
        detector = ShortcutDetector(methods=["probe", "statistical"])
        detector.fit(X, y)
        # Both should complete (success or graceful failure)
        assert "probe" in detector.results_
        assert "statistical" in detector.results_

    def test_d_equals_1000(self):
        """d=1000, n=50: extreme curse of dimensionality."""
        X = np.random.RandomState(0).randn(50, 1000)
        y = np.random.RandomState(0).randint(0, 2, 50)
        detector = ShortcutDetector(methods=["probe"])
        detector.fit(X, y)
        assert "probe" in detector.results_


class TestNegativeAndUnusualLabels:
    """Labels with negative values, floats, etc."""

    def test_negative_integer_labels(self):
        X = np.random.RandomState(0).randn(100, 10)
        y = np.array([-1] * 50 + [1] * 50)
        detector = ShortcutDetector(methods=["probe"])
        detector.fit(X, y)
        assert detector.results_["probe"]["success"]

    def test_large_label_ids(self):
        """Labels like 999, 1000 should not cause index errors."""
        X = np.random.RandomState(0).randn(100, 10)
        y = np.array([999] * 50 + [1000] * 50)
        detector = ShortcutDetector(methods=["statistical"])
        detector.fit(X, y)
        assert "statistical" in detector.results_
