"""Tests for input shape, type, and value validation."""

import numpy as np
import pytest

from shortcut_detect import ShortcutDetector
from shortcut_detect.utils import validate_embeddings_labels

# ---------------------------------------------------------------------------
# validate_embeddings_labels  (low-level utility)
# ---------------------------------------------------------------------------


class TestValidateEmbeddingsLabels:
    """Unit tests for the shared validation helper."""

    def test_valid_input(self):
        X, y = validate_embeddings_labels(np.random.randn(20, 5), np.zeros(20))
        assert X.shape == (20, 5)
        assert y.shape == (20,)

    @pytest.mark.parametrize(
        "n_samples, n_features",
        [(0, 10), (100, 0)],
    )
    def test_zero_dimension_inputs(self, n_samples, n_features):
        X = np.random.randn(n_samples, n_features) if n_features > 0 else np.empty((n_samples, 0))
        y = np.random.randint(0, 2, n_samples)
        with pytest.raises(ValueError):
            validate_embeddings_labels(X, y)

    def test_1d_embeddings_rejected(self):
        with pytest.raises(ValueError, match="2D"):
            validate_embeddings_labels(np.ones(10), np.zeros(10))

    def test_2d_labels_rejected(self):
        with pytest.raises(ValueError, match="1D"):
            validate_embeddings_labels(np.ones((10, 5)), np.zeros((10, 1)))

    def test_mismatched_lengths(self):
        with pytest.raises(ValueError, match="same number of samples"):
            validate_embeddings_labels(np.random.randn(100, 10), np.zeros(50))

    def test_too_few_samples(self):
        with pytest.raises(ValueError, match="at least 4"):
            validate_embeddings_labels(np.random.randn(2, 5), np.zeros(2), min_samples=4)

    def test_nan_in_embeddings(self):
        X = np.random.randn(20, 5)
        X[3, 2] = np.nan
        with pytest.raises(ValueError, match="NaN"):
            validate_embeddings_labels(X, np.zeros(20))

    def test_inf_in_embeddings(self):
        X = np.random.randn(20, 5)
        X[0, 0] = np.inf
        with pytest.raises(ValueError, match="NaN|inf"):
            validate_embeddings_labels(X, np.zeros(20))

    def test_neg_inf_in_embeddings(self):
        X = np.random.randn(20, 5)
        X[0, 0] = -np.inf
        with pytest.raises(ValueError, match="NaN|inf"):
            validate_embeddings_labels(X, np.zeros(20))

    def test_check_finite_disabled(self):
        X = np.random.randn(20, 5)
        X[0, 0] = np.nan
        # Should NOT raise when check is disabled
        X_out, y_out = validate_embeddings_labels(X, np.zeros(20), check_finite=False)
        assert np.isnan(X_out[0, 0])

    def test_min_classes_enforced(self):
        X = np.random.randn(20, 5)
        y = np.zeros(20)  # single class
        with pytest.raises(ValueError, match="At least 2"):
            validate_embeddings_labels(X, y, min_classes=2)

    def test_min_classes_passes_with_two(self):
        X = np.random.randn(20, 5)
        y = np.array([0] * 10 + [1] * 10)
        X_out, y_out = validate_embeddings_labels(X, y, min_classes=2)
        assert X_out.shape == (20, 5)


# ---------------------------------------------------------------------------
# ShortcutDetector.fit()  (integration-level validation)
# ---------------------------------------------------------------------------


class TestDetectorFitValidation:
    """Ensure ShortcutDetector.fit() rejects bad inputs with clear messages."""

    def test_empty_embeddings(self):
        detector = ShortcutDetector()
        with pytest.raises(ValueError):
            detector.fit(np.array([]).reshape(0, 10), np.array([]))

    def test_mismatched_lengths(self):
        detector = ShortcutDetector()
        with pytest.raises(ValueError, match="same number of samples"):
            detector.fit(np.random.randn(100, 10), np.zeros(50))

    def test_nan_embeddings_rejected(self):
        X = np.random.randn(100, 10)
        X[42, 7] = np.nan
        detector = ShortcutDetector()
        with pytest.raises(ValueError, match="NaN"):
            detector.fit(X, np.random.randint(0, 2, 100))

    def test_inf_embeddings_rejected(self):
        X = np.random.randn(100, 10)
        X[0, 0] = np.inf
        detector = ShortcutDetector()
        with pytest.raises(ValueError, match="NaN|inf"):
            detector.fit(X, np.random.randint(0, 2, 100))

    def test_single_class_labels_rejected(self):
        detector = ShortcutDetector()
        with pytest.raises(ValueError, match="At least 2"):
            detector.fit(np.random.randn(100, 10), np.zeros(100))

    def test_single_class_labels_ok_with_group_labels(self):
        """Single-class task labels are valid when group_labels has 2+ classes."""
        detector = ShortcutDetector(methods=["statistical"])
        X = np.random.RandomState(0).randn(100, 10)
        task_labels = np.zeros(100)
        group_labels = np.array([0] * 50 + [1] * 50)
        detector.fit(X, task_labels, group_labels=group_labels)
        assert "statistical" in detector.results_

    def test_1d_embeddings_rejected(self):
        detector = ShortcutDetector()
        with pytest.raises(ValueError, match="2D"):
            detector.fit(np.ones(50), np.zeros(50))

    def test_too_few_samples_rejected(self):
        detector = ShortcutDetector()
        with pytest.raises(ValueError, match="at least"):
            detector.fit(np.random.randn(2, 10), np.array([0, 1]))
