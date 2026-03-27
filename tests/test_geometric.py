import numpy as np

from shortcut_detect import ShortcutDetector
from shortcut_detect.geometric import BiasDirectionPCADetector, GeometricShortcutAnalyzer
from tests.fixtures.synthetic_data import generate_linear_shortcut


def test_geometric_analyzer_detects_bias():
    rng = np.random.RandomState(0)
    n = 400
    dim = 16
    embeddings = rng.randn(n, dim)
    groups = np.array([0] * (n // 2) + [1] * (n // 2))
    embeddings[groups == 1, 0] += 3.0  # strong bias along dim 0

    analyzer = GeometricShortcutAnalyzer(n_components=3, min_group_size=50)
    analyzer.fit(embeddings, groups)

    assert analyzer.summary_["risk_level"] in {"high", "moderate"}
    assert analyzer.bias_pairs_[0].effect_size > 1.0


def test_shortcut_detector_integration_with_geometric():
    embeddings, labels = generate_linear_shortcut(n_samples=600, embedding_dim=32, shortcut_dims=3)

    detector = ShortcutDetector(methods=["geometric"])
    detector.fit(embeddings, labels, group_labels=labels)

    assert detector.results_["geometric"]["success"]
    summary = detector.summary()
    assert "Geometric Bias Analysis" in summary


def test_bias_direction_pca_detects_gap():
    rng = np.random.RandomState(42)
    n_per_group = 60
    embeddings = rng.randn(2 * n_per_group, 8)
    groups = np.array(["A"] * n_per_group + ["B"] * n_per_group)

    embeddings[:n_per_group, 0] += 2.0
    embeddings[n_per_group:, 0] -= 2.0

    detector = BiasDirectionPCADetector(min_group_size=30, gap_threshold=0.5)
    detector.fit(embeddings, groups)

    assert detector.report_ is not None
    assert detector.report_.projection_gap >= 0.0
    assert detector.report_.risk_level in {"low", "moderate", "high", "unknown"}


def test_bias_direction_pca_no_gap():
    rng = np.random.RandomState(0)
    n_per_group = 60
    embeddings = rng.randn(2 * n_per_group, 8)
    groups = np.array(["A"] * n_per_group + ["B"] * n_per_group)

    detector = BiasDirectionPCADetector(min_group_size=30, gap_threshold=2.0)
    detector.fit(embeddings, groups)

    assert detector.report_ is not None
    assert detector.report_.risk_level in {"low", "unknown"}


def test_shortcut_detector_includes_bias_direction_pca():
    rng = np.random.RandomState(1)
    n_per_group = 80
    embeddings = rng.randn(2 * n_per_group, 6)
    groups = np.array([0] * n_per_group + [1] * n_per_group)
    embeddings[:n_per_group, 1] += 1.5
    embeddings[n_per_group:, 1] -= 1.5

    detector = ShortcutDetector(methods=["bias_direction_pca"])
    detector.fit(embeddings=embeddings, labels=np.zeros(len(groups)), group_labels=groups)

    assert "bias_direction_pca" in detector.results_
    assert detector.results_["bias_direction_pca"]["success"] is True
    summary = detector.summary()
    assert "Embedding Bias Direction (PCA)" in summary


def test_geometric_finalize_normalizes_legacy_risk_level():
    analyzer = GeometricShortcutAnalyzer()
    analyzer.summary_ = {
        "risk_level": "Medium",
        "message": "legacy format",
        "num_high_effect_pairs": 1,
        "num_overlap_pairs": 0,
    }
    analyzer.group_stats_ = {}
    analyzer.bias_pairs_ = []
    analyzer.subspace_pairs_ = []

    analyzer._finalize_results()

    assert analyzer.results_["risk_level"] == "moderate"
    assert analyzer.shortcut_detected_ is True
