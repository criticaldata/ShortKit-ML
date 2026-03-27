"""Tests for DetectorFactory extensibility."""

import numpy as np

from shortcut_detect import ShortcutDetector
from shortcut_detect.unified import BaseDetector, DetectorFactory
from tests.fixtures.synthetic_data import generate_linear_shortcut


class _DummyDetector:
    def fit(self, *_args, **_kwargs):
        return self


class _DummyBuilder(BaseDetector):
    def build(self):
        return _DummyDetector()

    def run(
        self,
        embeddings: np.ndarray,
        labels: np.ndarray,
        group_labels: np.ndarray,
        feature_names,
        protected_labels,
        splits=None,
        extra_labels=None,
    ):
        return {
            "detector": self.build(),
            "summary_title": "Dummy Method",
            "summary_lines": ["Dummy summary line"],
            "risk_indicators": ["Dummy risk indicator"],
            "success": True,
        }


def test_custom_builder_summary_integration():
    registry_snapshot = DetectorFactory._registry.copy()
    DetectorFactory.register("dummy", _DummyBuilder)
    try:
        embeddings, labels = generate_linear_shortcut(n_samples=200, embedding_dim=10)
        detector = ShortcutDetector(methods=["dummy"])
        detector.fit(embeddings, labels)
        summary = detector.summary()
        assert "Dummy Method" in summary
        assert "Dummy summary line" in summary
        overall = detector._generate_overall_assessment()
        assert "Dummy risk indicator" in overall
    finally:
        DetectorFactory._registry = registry_snapshot


def test_loader_fallback_runs_builtin_method():
    embeddings, labels = generate_linear_shortcut(n_samples=200, embedding_dim=10)
    loader = {
        "embeddings": embeddings,
        "labels": labels,
        "group_labels": labels,
    }
    detector = ShortcutDetector.from_loaders({"probe": loader}, methods=["probe"])
    assert detector.results_["probe"]["success"]
    summary = detector.summary()
    assert "Probe-based Detection" in summary
