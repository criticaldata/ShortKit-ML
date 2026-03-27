# tests/test_causal.py
"""Tests for GenerativeCVEDetector (generative_cvae method)."""

import random

import numpy as np
import pytest
import torch
from sklearn.linear_model import LogisticRegression

from shortcut_detect.causal import GenerativeCVEDetector
from shortcut_detect.causal.generative_cvae.src.detector import (
    CVAE,
    AttrNet,
    _call_probe,
    _cosine_similarity_rows,
)
from shortcut_detect.unified import DetectorFactory
from tests.fixtures.synthetic_data import (
    generate_linear_shortcut,
    generate_linear_shortcut_with_group_labels,
    generate_no_shortcut,
)


@pytest.fixture(autouse=True)
def deterministic():
    random.seed(1)
    np.random.seed(1)
    torch.manual_seed(3)
    yield


def _assert_results_schema(results):
    """Minimal DetectorBase contract checks."""
    for key in [
        "shortcut_detected",
        "risk_level",
        "metrics",
        "notes",
        "metadata",
        "report",
        "details",
    ]:
        assert key in results, f"Missing key {key} in results"


# ---------------------------------------------------------------------------
# Registry / Builder tests
# ---------------------------------------------------------------------------
class TestGenerativeCVEDetectorBuilder:
    def test_registry_builder_registered(self):
        factory = DetectorFactory(seed=0, kwargs={})
        methods = factory.supported_methods()
        assert "generative_cvae" in methods

    def test_builder_build_returns_detector_instance(self):
        factory = DetectorFactory(seed=0, kwargs={})
        builder = factory.create("generative_cvae")
        detector = builder.build()
        assert isinstance(detector, GenerativeCVEDetector)

    def test_builder_build_passes_kwargs(self):
        factory = DetectorFactory(seed=0, kwargs={"epochs": 10, "hidden": 64})
        builder = factory.create("generative_cvae")
        detector = builder.build()
        assert detector.cfg.epochs == 10
        assert detector.cfg.hidden == 64

    def test_builder_run_returns_failure_on_bad_input(self):
        factory = DetectorFactory(seed=0, kwargs={})
        builder = factory.create("generative_cvae")
        # 1-D embeddings should raise
        out = builder.run(
            embeddings=np.array([1, 2, 3]),
            labels=np.array([0, 1, 0]),
            group_labels=np.array([0, 1, 0]),
        )
        assert out["success"] is False
        assert "error" in out


# ---------------------------------------------------------------------------
# Detection end-to-end tests
# ---------------------------------------------------------------------------
class TestGenerativeCVEDetector:
    def test_detects_shortcut_on_synthetic_data(self):
        X, y, g = generate_linear_shortcut_with_group_labels(n_samples=300, embedding_dim=128)
        factory = DetectorFactory(seed=0, kwargs={})
        builder = factory.create("generative_cvae")
        out = builder.run(
            embeddings=X,
            labels=y,
            group_labels=g,
        )
        assert out["success"] is True
        detector = out["detector"]
        results = detector.results_
        _assert_results_schema(results)
        assert results["shortcut_detected"] is True
        assert isinstance(results["metrics"], dict)

    def test_not_detected_on_no_shortcut(self):
        X, y = generate_no_shortcut(n_samples=300, embedding_dim=24)
        # Use random binary groups uncorrelated with labels
        rng = np.random.RandomState(99)
        g = rng.randint(0, 2, size=len(y))
        detector = GenerativeCVEDetector(epochs=30, random_state=42)
        # Raise thresholds to avoid false positives on pure noise —
        # default 1e-4 / 0.01 are too sensitive for small random data.
        detector.cfg.mean_delta_threshold = 0.05
        detector.cfg.frac_large_threshold = 0.15
        detector.fit(X, g, y)
        results = detector.results_
        _assert_results_schema(results)
        assert results["shortcut_detected"] is False

    def test_accepts_external_probe(self):
        X, y = generate_linear_shortcut(n_samples=300, embedding_dim=40, shortcut_dims=6)
        probe = LogisticRegression(max_iter=2000, random_state=0)
        probe.fit(X, y)
        detector = GenerativeCVEDetector(epochs=30, random_state=42, probe_classifier=probe)
        detector.fit(X, y, y)
        results = detector.results_
        _assert_results_schema(results)
        assert results["metadata"]["probe_trained_on"] == "raw"
        assert results["shortcut_detected"] is True

    def test_no_probe_path(self):
        X, y, g = generate_linear_shortcut_with_group_labels(n_samples=200, embedding_dim=20)
        detector = GenerativeCVEDetector(epochs=20, random_state=42)
        # labels=None -> no probe is trained
        detector.fit(X, g, labels=None)
        results = detector.results_
        _assert_results_schema(results)
        assert results["shortcut_detected"] is None
        assert results["risk_level"] == "unknown"
        assert "note" in results["metrics"]


# ---------------------------------------------------------------------------
# Input validation tests
# ---------------------------------------------------------------------------
class TestGenerativeCVEInputValidation:
    def test_rejects_1d_embeddings(self):
        detector = GenerativeCVEDetector()
        with pytest.raises(ValueError, match="2D array"):
            detector.fit(np.array([1, 2, 3]), np.array([0, 1, 0]))

    def test_rejects_none_group_labels(self):
        detector = GenerativeCVEDetector()
        with pytest.raises(ValueError, match="group_labels.*required"):
            detector.fit(np.zeros((5, 3)), None)

    def test_rejects_mismatched_group_labels(self):
        detector = GenerativeCVEDetector()
        with pytest.raises(ValueError, match="group_labels length"):
            detector.fit(np.zeros((5, 3)), np.array([0, 1]))

    def test_rejects_non_binary_group_labels(self):
        detector = GenerativeCVEDetector()
        with pytest.raises(ValueError, match="binary"):
            detector.fit(np.zeros((5, 3)), np.array([0, 1, 2, 0, 1]))

    def test_rejects_mismatched_labels(self):
        detector = GenerativeCVEDetector()
        with pytest.raises(ValueError, match="labels length"):
            detector.fit(np.zeros((5, 3)), np.array([0, 1, 0, 1, 0]), np.array([0, 1]))


# ---------------------------------------------------------------------------
# Results schema and metrics tests
# ---------------------------------------------------------------------------
class TestGenerativeCVEResults:
    @pytest.fixture()
    def fitted_detector(self):
        X, y, g = generate_linear_shortcut_with_group_labels(n_samples=200, embedding_dim=20)
        det = GenerativeCVEDetector(epochs=20, random_state=42)
        det.fit(X, g, y)
        return det

    def test_results_schema_keys(self, fitted_detector):
        results = fitted_detector.results_
        _assert_results_schema(results)

    def test_metrics_are_scalar(self, fitted_detector):
        metrics = fitted_detector.results_["metrics"]
        for k, v in metrics.items():
            assert isinstance(
                v, str | bool | int | float | type(None)
            ), f"metric {k!r} is {type(v)}, expected scalar"

    def test_metrics_contains_cosine_similarity(self, fitted_detector):
        metrics = fitted_detector.results_["metrics"]
        assert "mean_cosine_similarity" in metrics
        mcs = metrics["mean_cosine_similarity"]
        assert -1.0 <= float(mcs) <= 1.0

    def test_metadata_fields(self, fitted_detector):
        meta = fitted_detector.results_["metadata"]
        for key in [
            "n_samples",
            "embedding_dim",
            "device",
            "cvae_epochs",
            "probe_trained_on",
            "guidance_steps",
            "guidance_weight",
            "proximity_weight",
            "random_state",
        ]:
            assert key in meta, f"Missing metadata key: {key}"

    def test_details_contains_embeddings(self, fitted_detector):
        details = fitted_detector.results_["details"]
        for key in ["original_embeddings", "counterfactual_embeddings", "spurious_labels", "probe"]:
            assert key in details, f"Missing details key: {key}"

    def test_counterfactual_embeddings_shape(self, fitted_detector):
        details = fitted_detector.results_["details"]
        orig = details["original_embeddings"]
        cf = details["counterfactual_embeddings"]
        assert orig.shape == cf.shape

    def test_risk_level_high_when_detected(self, fitted_detector):
        results = fitted_detector.results_
        if results["shortcut_detected"] is True:
            assert results["risk_level"] == "high"

    def test_summary_after_fit(self, fitted_detector):
        s = fitted_detector.summary()
        assert isinstance(s, str)
        assert "generative_cvae" in s


# ---------------------------------------------------------------------------
# Helper function tests
# ---------------------------------------------------------------------------
class TestCosineSimilarity:
    def test_identical_vectors(self):
        A = np.array([[1.0, 0.0], [0.0, 1.0]])
        sim = _cosine_similarity_rows(A, A)
        np.testing.assert_allclose(sim, [1.0, 1.0], atol=1e-7)

    def test_orthogonal_vectors(self):
        A = np.array([[1.0, 0.0]])
        B = np.array([[0.0, 1.0]])
        sim = _cosine_similarity_rows(A, B)
        np.testing.assert_allclose(sim, [0.0], atol=1e-7)

    def test_zero_vector(self):
        A = np.array([[0.0, 0.0]])
        B = np.array([[1.0, 1.0]])
        sim = _cosine_similarity_rows(A, B)
        assert sim[0] == 0.0  # not NaN


class TestCallProbe:
    def test_predict_proba(self):
        class FakeProbe:
            def predict_proba(self, X):
                return np.column_stack([1 - X[:, 0], X[:, 0]])

        X = np.array([[0.3], [0.7]])
        out = _call_probe(FakeProbe(), X)
        np.testing.assert_allclose(out, [0.3, 0.7], atol=1e-7)

    def test_callable(self):
        out = _call_probe(lambda X: X[:, 0], np.array([[0.5], [0.8]]))
        np.testing.assert_allclose(out, [0.5, 0.8], atol=1e-7)

    def test_invalid_probe(self):
        with pytest.raises(ValueError, match="not callable"):
            _call_probe("not_a_probe", np.zeros((3, 2)))


# ---------------------------------------------------------------------------
# CVAE / AttrNet shape tests
# ---------------------------------------------------------------------------
class TestCVAEShapes:
    def test_forward_output_shapes(self):
        model = CVAE(dim=16, hidden=32, zdim=8)
        x = torch.randn(4, 16)
        s = torch.tensor([0.0, 1.0, 0.0, 1.0])
        xrec, mu, logvar = model(x, s)
        assert xrec.shape == (4, 16)
        assert mu.shape == (4, 8)
        assert logvar.shape == (4, 8)


class TestAttrNetShapes:
    def test_output_shape(self):
        net = AttrNet(dim=16)
        x = torch.randn(4, 16)
        out = net(x)
        assert out.shape == (4,)
