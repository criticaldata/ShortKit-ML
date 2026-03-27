"""Tests for shortcut_detect.benchmark.method_utils."""

import numpy as np

from shortcut_detect.benchmark.method_utils import (
    ALL_METHODS,
    DIM_SCORE_METHODS,
    EMBEDDING_METHODS,
    FAIRNESS_METHODS,
    SKIP_IN_SYNTHETIC,
    bias_direction_pca_dim_scores,
    convergence_bucket,
    method_flag,
    nan_dim_scores,
    sis_dim_scores,
)


class TestMethodFlag:
    """Test method_flag for all 10 methods."""

    def test_failed_result_returns_false(self):
        for method in ALL_METHODS:
            assert method_flag(method, {"success": False}) is False

    def test_probe_detected(self):
        result = {"success": True, "results": {"shortcut_detected": True}}
        assert method_flag("probe", result) is True

    def test_probe_not_detected(self):
        result = {"success": True, "results": {"shortcut_detected": False}}
        assert method_flag("probe", result) is False

    def test_hbac_detected(self):
        result = {"success": True, "report": {"has_shortcut": {"exists": True}}}
        assert method_flag("hbac", result) is True

    def test_hbac_not_detected(self):
        result = {"success": True, "report": {"has_shortcut": {"exists": False}}}
        assert method_flag("hbac", result) is False

    def test_statistical_detected(self):
        result = {"success": True, "significant_features": {"dim_0": [1, 2]}}
        assert method_flag("statistical", result) is True

    def test_statistical_not_detected(self):
        result = {"success": True, "significant_features": {"dim_0": []}}
        assert method_flag("statistical", result) is False

    def test_geometric_high(self):
        result = {"success": True, "summary": {"risk_level": "high"}}
        assert method_flag("geometric", result) is True

    def test_geometric_low(self):
        result = {"success": True, "summary": {"risk_level": "low"}}
        assert method_flag("geometric", result) is False

    def test_frequency_detected(self):
        result = {"success": True, "report": {"shortcut_detected": True}}
        assert method_flag("frequency", result) is True

    def test_frequency_not_detected(self):
        result = {"success": True, "report": {"shortcut_detected": False}}
        assert method_flag("frequency", result) is False

    def test_sis_detected(self):
        result = {"success": True, "shortcut_detected": True}
        assert method_flag("sis", result) is True

    def test_sis_not_detected(self):
        result = {"success": True, "shortcut_detected": False}
        assert method_flag("sis", result) is False

    def test_bias_direction_pca_via_risk_value(self):
        result = {"success": True, "risk_value": "high"}
        assert method_flag("bias_direction_pca", result) is True

    def test_bias_direction_pca_low(self):
        result = {"success": True, "risk_value": "low"}
        assert method_flag("bias_direction_pca", result) is False

    def test_demographic_parity_moderate(self):
        result = {"success": True, "risk_value": "moderate"}
        assert method_flag("demographic_parity", result) is True

    def test_equalized_odds_high(self):
        result = {"success": True, "risk_value": "high"}
        assert method_flag("equalized_odds", result) is True

    def test_intersectional_low(self):
        result = {"success": True, "risk_value": "low"}
        assert method_flag("intersectional", result) is False

    def test_groupdro_detected(self):
        result = {"success": True, "report": {"final": {"avg_acc": 0.90, "worst_group_acc": 0.70}}}
        assert method_flag("groupdro", result) is True

    def test_groupdro_not_detected(self):
        result = {"success": True, "report": {"final": {"avg_acc": 0.90, "worst_group_acc": 0.85}}}
        assert method_flag("groupdro", result) is False

    def test_gce_detected(self):
        result = {"success": True, "report": {"risk_level": "high"}}
        assert method_flag("gce", result) is True

    def test_gce_not_detected(self):
        result = {"success": True, "report": {"risk_level": "low"}}
        assert method_flag("gce", result) is False

    def test_ssa_detected(self):
        result = {"success": True, "shortcut_detected": True}
        assert method_flag("ssa", result) is True

    def test_ssa_not_detected(self):
        result = {"success": True, "shortcut_detected": False}
        assert method_flag("ssa", result) is False

    def test_unknown_method(self):
        assert method_flag("nonexistent", {"success": True}) is False


class TestConvergenceBucket:
    def test_all_agree(self):
        assert convergence_bucket(10, 10) == "high_confidence"

    def test_one_short(self):
        assert convergence_bucket(9, 10) == "moderate_confidence"

    def test_single_flag(self):
        assert convergence_bucket(1, 10) == "likely_false_alarm"

    def test_no_detection(self):
        assert convergence_bucket(0, 10) == "no_detection"

    def test_intermediate(self):
        assert convergence_bucket(5, 10) == "intermediate"

    def test_zero_methods(self):
        assert convergence_bucket(0, 0) == "no_detection"


class TestDimScores:
    def test_nan_dim_scores(self):
        scores = nan_dim_scores(128)
        assert scores.shape == (128,)
        assert np.all(np.isnan(scores))

    def test_bias_direction_pca_missing_report(self):
        scores = bias_direction_pca_dim_scores({}, 64)
        assert scores.shape == (64,)
        np.testing.assert_array_equal(scores, np.zeros(64))

    def test_sis_dim_scores_empty(self):
        scores = sis_dim_scores({}, 64)
        assert scores.shape == (64,)


class TestMethodSets:
    def test_all_methods_count(self):
        assert len(ALL_METHODS) == 13

    def test_dim_score_methods_subset(self):
        assert DIM_SCORE_METHODS.issubset(set(ALL_METHODS))

    def test_skip_in_synthetic(self):
        assert "intersectional" in SKIP_IN_SYNTHETIC

    def test_all_tiers_cover_all(self):
        from shortcut_detect.benchmark.method_utils import TRAINING_METHODS

        assert EMBEDDING_METHODS | FAIRNESS_METHODS | TRAINING_METHODS == set(ALL_METHODS)
