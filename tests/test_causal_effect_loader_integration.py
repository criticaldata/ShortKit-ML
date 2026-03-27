"""Unified loader-mode integration tests for Causal Effect detector."""

import numpy as np
import pytest

from shortcut_detect import ShortcutDetector
from shortcut_detect.unified import DetectorFactory


def test_causal_effect_loader_integration():
    rng = np.random.default_rng(10)
    n = 150
    dim = 8
    embeddings = rng.standard_normal((n, dim))
    labels = (embeddings[:, 0] > 0).astype(np.int64)
    # Causal-like attribute
    attr_causal = (embeddings[:, 0] > 0).astype(np.int64)
    # Spurious-like: random, independent
    attr_spurious = rng.integers(0, 2, size=n)

    loader = {
        "embeddings": embeddings,
        "labels": labels,
        "attributes": {
            "causal": attr_causal,
            "spurious": attr_spurious,
        },
    }

    detector = ShortcutDetector(
        methods=["causal_effect"],
        causal_effect_spurious_threshold=0.2,
    )
    detector.fit_from_loaders({"causal_effect": loader})

    result = detector.get_results().get("causal_effect")
    assert result is not None
    assert result["success"] is True
    assert result["metrics"]["n_attributes"] == 2
    assert "per_attribute_effects" in result["metrics"]
    assert "causal" in result["metrics"]["per_attribute_effects"]
    assert "spurious" in result["metrics"]["per_attribute_effects"]
    assert "per_attribute" in result["report"]
    assert result["summary_title"] == "Causal Effect Regularization"


def test_causal_effect_raises_without_loader():
    """run() raises ValueError directing user to fit_from_loaders."""
    factory = DetectorFactory(seed=42)
    builder = factory.create("causal_effect")
    with pytest.raises(ValueError, match="fit_from_loaders"):
        builder.run(
            embeddings=np.zeros((50, 4)),
            labels=np.zeros(50),
            group_labels=np.zeros(50),
            feature_names=None,
            protected_labels=None,
        )
