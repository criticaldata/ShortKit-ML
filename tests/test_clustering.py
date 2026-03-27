"""Tests for clustering-based detection (HBAC)."""

from shortcut_detect.clustering import HBACConfig, HBACDetector
from tests.fixtures.synthetic_data import generate_linear_shortcut, generate_no_shortcut


def test_hbac_detector_basic():
    """Test HBAC detector on simple linear shortcut."""
    # Generate data with clear shortcut
    embeddings, labels = generate_linear_shortcut(
        n_samples=1000, embedding_dim=50, shortcut_dims=3, seed=42
    )

    # Fit detector
    detector = HBACDetector(max_iterations=3, min_cluster_size=0.05)
    detector.fit(embeddings, labels)

    # Check shortcut detected
    assert detector.shortcut_report_ is not None
    assert "has_shortcut" in detector.shortcut_report_
    assert detector.shortcut_report_["has_shortcut"]["exists"] is True

    # Check important dimensions identified
    dim_importance = detector.shortcut_report_["dimension_importance"]
    top_dims = dim_importance.head(5)["dimension"].tolist()

    # At least one of the first 3 dimensions should be in top 5
    assert any(f"dim_{i}" in top_dims for i in range(3))


def test_hbac_detector_no_shortcut():
    """Test HBAC detector on random data (no shortcut)."""
    # Pure random data
    embeddings, labels = generate_no_shortcut(n_samples=500, embedding_dim=30, seed=42)

    detector = HBACDetector(max_iterations=2, min_cluster_size=0.1)
    detector.fit(embeddings, labels)

    # Should not detect strong shortcut
    # (May detect weak patterns due to random chance, so check confidence)
    if detector.shortcut_report_["has_shortcut"]["exists"]:
        assert detector.shortcut_report_["has_shortcut"]["confidence"] != "high"


def test_hbac_report_generation():
    """Test report generation."""
    embeddings, labels = generate_linear_shortcut(n_samples=500, embedding_dim=30)

    detector = HBACDetector()
    detector.fit(embeddings, labels)

    report = detector.get_report()
    report_text = detector.get_report_text()

    assert isinstance(report, dict)
    assert report["method"] == "hbac"
    assert "metrics" in report
    assert "report" in report

    assert isinstance(report_text, str)
    assert "SHORTCUT DETECTION RESULT" in report_text
    assert "CLUSTER ANALYSIS" in report_text
    assert "TOP IMPORTANT DIMENSIONS" in report_text


def test_hbac_with_feature_names():
    """Test HBAC detector with custom feature names."""
    embeddings, labels = generate_linear_shortcut(n_samples=500, embedding_dim=20)

    feature_names = [f"feature_{i}" for i in range(20)]
    detector = HBACDetector()
    detector.fit(embeddings, labels, feature_names=feature_names)

    dim_importance = detector.shortcut_report_["dimension_importance"]
    assert dim_importance["dimension"].iloc[0].startswith("feature_")


def test_hbac_multiclass():
    """Test HBAC on multi-class problem."""
    from tests.fixtures.synthetic_data import generate_multiclass_shortcut

    embeddings, labels = generate_multiclass_shortcut(n_samples=900, embedding_dim=40, n_classes=3)

    detector = HBACDetector(max_iterations=4)
    detector.fit(embeddings, labels)

    # Should detect shortcuts in multi-class setting
    assert detector.shortcut_report_ is not None
    assert len(detector.clusters_) >= 2  # Should form multiple clusters


def test_hbac_constructor_positional_backward_compatible():
    """HBACDetector should preserve legacy positional constructor behavior."""
    detector = HBACDetector(3, 0.05, 0.2, 42)
    assert detector.max_iterations == 3
    assert detector.min_cluster_size == 0.05
    assert detector.test_size == 0.2
    assert detector.random_state == 42


def test_hbac_constructor_accepts_config_object_as_first_arg():
    """HBACDetector should also accept an HBACConfig as the first argument."""
    cfg = HBACConfig(max_iterations=7, min_cluster_size=0.03, test_size=0.15, random_state=11)
    detector = HBACDetector(cfg)
    assert detector.max_iterations == 7
    assert detector.min_cluster_size == 0.03
    assert detector.test_size == 0.15
    assert detector.random_state == 11
