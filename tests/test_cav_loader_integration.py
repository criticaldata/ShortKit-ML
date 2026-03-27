"""Unified loader-mode integration tests for CAV."""

import numpy as np

from shortcut_detect import ShortcutDetector


def test_cav_loader_integration():
    rng = np.random.default_rng(10)
    dim = 8

    concept = rng.normal(0.0, 0.5, size=(50, dim))
    random = rng.normal(0.0, 0.5, size=(50, dim))
    concept[:, 0] += 1.8
    random[:, 0] -= 1.8

    target_dd = rng.normal(0.0, 0.2, size=(70, dim))
    target_dd[:, 0] += 1.0

    loader = {
        "concept_sets": {"shortcut": concept},
        "random_set": random,
        "target_directional_derivatives": target_dd,
    }

    detector = ShortcutDetector(
        methods=["cav"], cav_quality_threshold=0.7, cav_shortcut_threshold=0.6
    )
    detector.fit_from_loaders({"cav": loader})

    result = detector.get_results().get("cav")
    assert result is not None
    assert result["success"] is True
    assert result["metrics"]["n_concepts"] == 1
    assert "per_concept" in result["report"]
    assert result["summary_title"] == "CAV (Concept Activation Vectors)"
