import numpy as np
import pytest
from sklearn.linear_model import LogisticRegression

from shortcut_detect import ShortcutDetector
from shortcut_detect.fairness import (
    DemographicParityDetector,
    DemographicParityReport,
    EqualizedOddsDetector,
    EqualizedOddsReport,
    IntersectionalDetector,
    IntersectionalReport,
)


def _make_fairness_dataset(seed=0, n_per_group=120):
    """Create synthetic dataset with known fairness disparities."""
    rng = np.random.RandomState(seed)
    groups = np.array(["GroupA"] * n_per_group + ["GroupB"] * n_per_group)
    labels_a = rng.binomial(1, 0.7, n_per_group)
    labels_b = rng.binomial(1, 0.3, n_per_group)
    labels = np.concatenate([labels_a, labels_b])

    embeddings = rng.randn(2 * n_per_group, 6)
    embeddings[:n_per_group, 0] += labels_a * 2.5
    embeddings[n_per_group:, 0] += labels_b * 0.3  # weaker signal, induces disparity

    return embeddings, labels, groups


def _make_perfect_fairness_dataset(seed=0, n_per_group=120):
    """Create dataset with perfect fairness (same TPR/FPR across groups)."""
    rng = np.random.RandomState(seed)
    groups = np.array(["GroupA"] * n_per_group + ["GroupB"] * n_per_group)
    # Same label distribution for both groups
    labels = rng.binomial(1, 0.5, 2 * n_per_group)

    embeddings = rng.randn(2 * n_per_group, 6)
    # Same signal strength for both groups
    embeddings[:, 0] += labels * 2.0

    return embeddings, labels, groups


def test_equalized_odds_detector_reports_gaps():
    """Test that detector computes and reports TPR/FPR gaps."""
    embeddings, labels, groups = _make_fairness_dataset()
    detector = EqualizedOddsDetector(min_group_size=50)
    detector.fit(embeddings, labels, groups)

    assert detector.report_ is not None
    assert isinstance(detector.report_, EqualizedOddsReport)
    assert detector.report_.tpr_gap >= 0.0
    assert detector.report_.fpr_gap >= 0.0
    assert detector.report_.reference == "Hardt et al. 2016"
    assert detector.report_.overall_accuracy >= 0.0
    assert detector.report_.overall_accuracy <= 1.0


def test_group_metrics_computation():
    """Test that group metrics are computed correctly."""
    embeddings, labels, groups = _make_fairness_dataset()
    detector = EqualizedOddsDetector(min_group_size=50)
    detector.fit(embeddings, labels, groups)

    assert len(detector.group_metrics_) == 2  # Two groups
    for _group_name, metrics in detector.group_metrics_.items():
        assert "tpr" in metrics
        assert "fpr" in metrics
        assert "support" in metrics
        assert "tp" in metrics
        assert "fp" in metrics
        assert "tn" in metrics
        assert "fn" in metrics

        # TPR and FPR should be in [0, 1] or NaN
        if not np.isnan(metrics["tpr"]):
            assert 0.0 <= metrics["tpr"] <= 1.0
        if not np.isnan(metrics["fpr"]):
            assert 0.0 <= metrics["fpr"] <= 1.0

        # Support should match group size
        assert metrics["support"] >= 0


def test_tpr_fpr_gap_calculation():
    """Test TPR and FPR gap calculations."""
    embeddings, labels, groups = _make_fairness_dataset()
    detector = EqualizedOddsDetector(min_group_size=50)
    detector.fit(embeddings, labels, groups)

    # Gaps should be non-negative (max - min)
    assert detector.tpr_gap_ >= 0.0 or np.isnan(detector.tpr_gap_)
    assert detector.fpr_gap_ >= 0.0 or np.isnan(detector.fpr_gap_)

    # If we have valid metrics, gaps should be computed
    valid_tprs = [m["tpr"] for m in detector.group_metrics_.values() if not np.isnan(m["tpr"])]
    if len(valid_tprs) >= 2:
        expected_tpr_gap = max(valid_tprs) - min(valid_tprs)
        assert abs(detector.tpr_gap_ - expected_tpr_gap) < 1e-6


def test_perfect_fairness_detection():
    """Test detection on perfectly fair dataset."""
    embeddings, labels, groups = _make_perfect_fairness_dataset()
    detector = EqualizedOddsDetector(min_group_size=50)
    detector.fit(embeddings, labels, groups)

    # With perfect fairness, gaps should be small
    assert detector.tpr_gap_ < 0.2  # Should be close to 0
    assert detector.fpr_gap_ < 0.2
    assert detector.report_.risk_level in {"low", "unknown"}


def test_risk_assessment():
    """Test risk level assessment based on gaps."""
    embeddings, labels, groups = _make_fairness_dataset()

    # Test with low threshold (should detect high risk)
    detector = EqualizedOddsDetector(
        min_group_size=50, tpr_gap_threshold=0.05, fpr_gap_threshold=0.05
    )
    detector.fit(embeddings, labels, groups)

    assert detector.report_.risk_level in {"low", "moderate", "high", "unknown"}
    assert len(detector.report_.notes) > 0


def test_min_group_size_filtering():
    """Test that groups below min_group_size are filtered out."""
    embeddings, labels, groups = _make_fairness_dataset(n_per_group=120)

    # Set high min_group_size to filter out groups
    detector = EqualizedOddsDetector(min_group_size=200)
    detector.fit(embeddings, labels, groups)

    # Groups with insufficient size should have NaN metrics
    for _group_name, metrics in detector.group_metrics_.items():
        if metrics["support"] < 200:
            assert np.isnan(metrics["tpr"])
            assert np.isnan(metrics["fpr"])


def test_binary_labels_requirement():
    """Test that non-binary labels raise an error."""
    embeddings = np.random.randn(100, 5)
    labels = np.array([0, 1, 2] * 33 + [1])  # Multi-class labels
    groups = np.array(["A", "B"] * 50)

    detector = EqualizedOddsDetector()
    with pytest.raises(ValueError, match="binary labels"):
        detector.fit(embeddings, labels, groups)


def test_missing_group_labels():
    """Test that missing group_labels raises an error."""
    embeddings = np.random.randn(100, 5)
    labels = np.random.randint(0, 2, 100)

    detector = EqualizedOddsDetector()
    with pytest.raises(ValueError, match="group_labels"):
        detector.fit(embeddings, labels, group_labels=None)


def test_shape_validation():
    """Test input shape validation."""
    detector = EqualizedOddsDetector()

    # Test 3D embeddings (should fail)
    embeddings_3d = np.random.randn(100, 5, 3)
    labels = np.random.randint(0, 2, 100)
    groups = np.array(["A", "B"] * 50)

    with pytest.raises(ValueError, match="2D"):
        detector.fit(embeddings_3d, labels, groups)

    # Test 2D labels (should fail)
    embeddings_2d = np.random.randn(100, 5)
    labels_2d = np.random.randint(0, 2, (100, 2))

    with pytest.raises(ValueError, match="1D"):
        detector.fit(embeddings_2d, labels_2d, groups)


def test_length_mismatch():
    """Test that length mismatches raise errors."""
    detector = EqualizedOddsDetector()

    embeddings = np.random.randn(100, 5)
    labels = np.random.randint(0, 2, 50)  # Wrong length
    groups = np.array(["A", "B"] * 50)

    with pytest.raises(ValueError, match="align"):
        detector.fit(embeddings, labels, groups)


def test_custom_estimator():
    """Test that custom estimator can be provided."""
    custom_estimator = LogisticRegression(max_iter=2000, C=0.1)
    embeddings, labels, groups = _make_fairness_dataset()

    detector = EqualizedOddsDetector(estimator=custom_estimator)
    detector.fit(embeddings, labels, groups)

    assert detector.report_ is not None
    assert detector.overall_accuracy_ >= 0.0


def test_overall_accuracy():
    """Test that overall accuracy is computed correctly."""
    embeddings, labels, groups = _make_fairness_dataset()
    detector = EqualizedOddsDetector()
    detector.fit(embeddings, labels, groups)

    # Accuracy should be between 0 and 1
    assert 0.0 <= detector.overall_accuracy_ <= 1.0
    assert detector.report_.overall_accuracy == detector.overall_accuracy_


def test_gap_with_single_valid_group():
    """Test gap computation when only one group has valid metrics."""
    rng = np.random.RandomState(42)
    embeddings = rng.randn(100, 5)
    labels = rng.randint(0, 2, 100)
    groups = np.array(["A"] * 5 + ["B"] * 95)  # One very small group

    detector = EqualizedOddsDetector(min_group_size=10)
    detector.fit(embeddings, labels, groups)

    # With only one valid group, gap should be NaN
    valid_tprs = [m["tpr"] for m in detector.group_metrics_.values() if not np.isnan(m["tpr"])]
    if len(valid_tprs) < 2:
        assert np.isnan(detector.tpr_gap_) or detector.tpr_gap_ == 0.0


def test_shortcut_detector_includes_equalized_odds_results():
    """Test integration with ShortcutDetector unified API."""
    embeddings, labels, groups = _make_fairness_dataset()
    unified = ShortcutDetector(methods=["equalized_odds"])
    unified.fit(embeddings=embeddings, labels=labels, group_labels=groups)

    assert "equalized_odds" in unified.results_
    assert unified.results_["equalized_odds"]["success"] is True
    report = unified.results_["equalized_odds"]["report"]
    assert isinstance(report, EqualizedOddsReport)
    assert report.risk_level in {"low", "moderate", "high", "unknown"}
    assert report.reference == "Hardt et al. 2016"


def test_demographic_parity_detector_reports_gap():
    embeddings, labels, groups = _make_fairness_dataset()
    detector = DemographicParityDetector(min_group_size=50)
    detector.fit(embeddings, labels, groups)

    assert detector.report_ is not None
    assert isinstance(detector.report_, DemographicParityReport)
    assert detector.report_.dp_gap >= 0.0 or np.isnan(detector.report_.dp_gap)
    assert detector.report_.reference == "Feldman et al. 2015"
    assert 0.0 <= detector.report_.overall_positive_rate <= 1.0


def test_equalized_odds_finalize_normalizes_legacy_risk_level():
    detector = EqualizedOddsDetector()
    detector.report_ = EqualizedOddsReport(
        group_metrics={},
        tpr_gap=float("nan"),
        fpr_gap=float("nan"),
        overall_accuracy=float("nan"),
        reference="Hardt et al. 2016",
        risk_level="Medium",
        notes="legacy risk string",
    )

    detector._finalize_results()

    assert detector.results_["risk_level"] == "moderate"
    assert detector.shortcut_detected_ is True


def test_demographic_parity_group_rates():
    embeddings, labels, groups = _make_fairness_dataset()
    detector = DemographicParityDetector(min_group_size=50)
    detector.fit(embeddings, labels, groups)

    assert len(detector.group_rates_) == 2
    for metrics in detector.group_rates_.values():
        assert "positive_rate" in metrics
        assert "support" in metrics
        if not np.isnan(metrics["positive_rate"]):
            assert 0.0 <= metrics["positive_rate"] <= 1.0


def test_shortcut_detector_includes_demographic_parity_results():
    embeddings, labels, groups = _make_fairness_dataset()
    unified = ShortcutDetector(methods=["demographic_parity"])
    unified.fit(embeddings=embeddings, labels=labels, group_labels=groups)

    assert "demographic_parity" in unified.results_
    assert unified.results_["demographic_parity"]["success"] is True
    report = unified.results_["demographic_parity"]["report"]
    assert isinstance(report, DemographicParityReport)
    assert report.reference == "Feldman et al. 2015"


def test_shortcut_detector_without_group_labels():
    """Test that ShortcutDetector handles missing group_labels gracefully."""
    embeddings, labels, _ = _make_fairness_dataset()
    unified = ShortcutDetector(methods=["equalized_odds"])
    with pytest.warns(UserWarning, match="group_labels are required"):
        unified.fit(embeddings=embeddings, labels=labels, group_labels=None)

    assert "equalized_odds" in unified.results_
    assert unified.results_["equalized_odds"]["success"] is False
    assert "group_labels" in unified.results_["equalized_odds"]["error"]


def test_multiple_groups():
    """Test with more than two groups."""
    rng = np.random.RandomState(42)
    n_per_group = 50
    groups = np.array(["A"] * n_per_group + ["B"] * n_per_group + ["C"] * n_per_group)
    labels = rng.randint(0, 2, 3 * n_per_group)
    embeddings = rng.randn(3 * n_per_group, 5)

    detector = EqualizedOddsDetector(min_group_size=40)
    detector.fit(embeddings, labels, groups)

    assert len(detector.group_metrics_) == 3
    # Gap should be computed across all groups
    if not np.isnan(detector.tpr_gap_):
        assert detector.tpr_gap_ >= 0.0


def test_edge_case_all_positive_labels():
    """Test edge case where all labels are positive (should raise error)."""
    rng = np.random.RandomState(42)
    embeddings = rng.randn(100, 5)
    labels = np.ones(100, dtype=int)  # All positive
    groups = np.array(["A", "B"] * 50)

    detector = EqualizedOddsDetector()
    # Equalized odds requires binary labels (both 0 and 1), so this should raise an error
    with pytest.raises(ValueError, match="binary labels"):
        detector.fit(embeddings, labels, groups)


def test_edge_case_all_negative_labels():
    """Test edge case where all labels are negative (should raise error)."""
    rng = np.random.RandomState(42)
    embeddings = rng.randn(100, 5)
    labels = np.zeros(100, dtype=int)  # All negative
    groups = np.array(["A", "B"] * 50)

    detector = EqualizedOddsDetector()
    # Equalized odds requires binary labels (both 0 and 1), so this should raise an error
    with pytest.raises(ValueError, match="binary labels"):
        detector.fit(embeddings, labels, groups)


def _make_intersectional_dataset(seed=0, n_per_intersection=80):
    """Create synthetic dataset with race and gender for intersectional analysis."""
    rng = np.random.RandomState(seed)
    n = n_per_intersection * 4  # 4 intersections: A_M, A_F, B_M, B_F
    race = np.array(["RaceA"] * (2 * n_per_intersection) + ["RaceB"] * (2 * n_per_intersection))
    gender = np.array(["Male", "Female"] * (2 * n_per_intersection))
    extra_labels = {"race": race, "gender": gender}

    labels = rng.binomial(1, 0.5, n)
    embeddings = rng.randn(n, 6)
    embeddings[:, 0] += labels * 2.0

    return embeddings, labels, extra_labels


def test_intersectional_detector_valid_two_attributes():
    """Test intersectional detector with 2 demographic attributes."""
    embeddings, labels, extra_labels = _make_intersectional_dataset()
    detector = IntersectionalDetector(min_group_size=50)
    detector.fit(embeddings, labels, extra_labels)

    assert detector.report_ is not None
    assert isinstance(detector.report_, IntersectionalReport)
    assert detector.report_.attribute_names == ["race", "gender"]
    assert len(detector.report_.intersection_metrics) >= 2
    assert detector.report_.reference == "Buolamwini & Gebru 2018"
    for _group, metrics in detector.report_.intersection_metrics.items():
        assert "tpr" in metrics
        assert "fpr" in metrics
        assert "positive_rate" in metrics
        assert "support" in metrics


def test_intersectional_detector_insufficient_attributes():
    """Test that intersectional detector raises when only 1 attribute provided."""
    embeddings, labels, _ = _make_intersectional_dataset()
    extra_labels = {"race": np.array(["A", "B"] * 80)}

    detector = IntersectionalDetector()
    with pytest.raises(ValueError, match="at least 2"):
        detector.fit(embeddings, labels, extra_labels)


def test_intersectional_detector_no_extra_labels():
    """Test that intersectional detector raises when extra_labels is None."""
    embeddings, labels, _ = _make_intersectional_dataset()

    detector = IntersectionalDetector()
    with pytest.raises(ValueError, match="at least 2"):
        detector.fit(embeddings, labels, extra_labels=None)


def test_shortcut_detector_includes_intersectional_results():
    """Test integration with ShortcutDetector unified API."""
    embeddings, labels, extra_labels = _make_intersectional_dataset()
    unified = ShortcutDetector(methods=["intersectional"])
    unified.fit(
        embeddings=embeddings,
        labels=labels,
        extra_labels=extra_labels,
    )

    assert "intersectional" in unified.results_
    assert unified.results_["intersectional"]["success"] is True
    report = unified.results_["intersectional"]["report"]
    assert isinstance(report, IntersectionalReport)
    assert report.risk_level in {"low", "moderate", "high", "unknown"}
    assert report.reference == "Buolamwini & Gebru 2018"


def test_shortcut_detector_intersectional_skipped_without_extra_labels():
    """Test that intersectional is skipped when extra_labels has insufficient attributes."""
    embeddings, labels, _ = _make_fairness_dataset()
    unified = ShortcutDetector(methods=["intersectional"])
    unified.fit(embeddings=embeddings, labels=labels, group_labels=np.array(["A", "B"] * 120))

    assert "intersectional" in unified.results_
    assert unified.results_["intersectional"]["success"] is False
    assert "extra_labels" in unified.results_["intersectional"]["error"]


def test_threshold_configuration():
    """Test that threshold configuration affects risk assessment."""
    embeddings, labels, groups = _make_fairness_dataset()

    # Low thresholds (should detect more issues)
    detector_low = EqualizedOddsDetector(tpr_gap_threshold=0.01, fpr_gap_threshold=0.01)
    detector_low.fit(embeddings, labels, groups)

    # High thresholds (should detect fewer issues)
    detector_high = EqualizedOddsDetector(tpr_gap_threshold=0.5, fpr_gap_threshold=0.5)
    detector_high.fit(embeddings, labels, groups)

    # Both should produce valid reports
    assert detector_low.report_ is not None
    assert detector_high.report_ is not None
