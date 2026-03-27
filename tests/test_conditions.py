"""Tests for pluggable overall assessment conditions."""

from __future__ import annotations

import pytest

from shortcut_detect import ShortcutDetector
from shortcut_detect.conditions import (
    ConditionContext,
    RiskCondition,
    available_conditions,
    create_condition,
    register_condition,
)
from shortcut_detect.conditions import registry as condition_registry


def test_indicator_count_low_when_no_successful_indicators():
    condition = create_condition("indicator_count")
    ctx = ConditionContext(
        methods=["probe", "statistical", "geometric"],
        results={
            "probe": {"success": True, "risk_indicators": []},
            "statistical": {"success": False, "risk_indicators": ["ignored"]},
        },
    )

    assert condition.assess(ctx) == "🟢 LOW RISK: No strong shortcut signals detected"


def test_indicator_count_moderate_with_one_indicator():
    condition = create_condition("indicator_count")
    ctx = ConditionContext(
        methods=["probe"],
        results={"probe": {"success": True, "risk_indicators": ["Probe signal"]}},
    )

    assert (
        condition.assess(ctx) == "🟡 MODERATE RISK: One method detected shortcuts\n  • Probe signal"
    )


def test_indicator_count_high_with_two_indicators_from_one_method():
    condition = create_condition("indicator_count")
    ctx = ConditionContext(
        methods=["probe", "statistical"],
        results={
            "probe": {"success": True, "risk_indicators": ["Signal A", "Signal B"]},
            "statistical": {"success": False, "risk_indicators": ["ignored"]},
        },
    )

    assessment = condition.assess(ctx)
    assert assessment.startswith("🔴 HIGH RISK")
    assert "Signal A" in assessment
    assert "Signal B" in assessment


def test_majority_vote_moderate_with_one_successful_method():
    condition = create_condition("majority_vote")
    ctx = ConditionContext(
        methods=["probe", "statistical"],
        results={
            "probe": {"success": True, "risk_indicators": ["Signal A", "Signal A"]},
            "statistical": {"success": False, "risk_indicators": ["ignored"]},
        },
    )

    assessment = condition.assess(ctx)
    assert assessment == "🟡 MODERATE RISK: One method detected shortcuts\n  • Signal A"


def test_majority_vote_high_uses_method_votes_and_deduplicates_output():
    condition = create_condition("majority_vote")
    ctx = ConditionContext(
        methods=["probe", "statistical", "geometric"],
        results={
            "probe": {"success": True, "risk_indicators": ["Shared signal", "Probe-only"]},
            "statistical": {"success": True, "risk_indicators": ["Shared signal"]},
            "geometric": {"success": False, "risk_indicators": ["ignored"]},
        },
    )

    assessment = condition.assess(ctx)
    assert assessment.startswith("🔴 HIGH RISK")
    assert assessment.count("Shared signal") == 1
    assert "Probe-only" in assessment


def test_majority_vote_respects_custom_threshold():
    condition = create_condition("majority_vote", high_threshold=3)
    ctx = ConditionContext(
        methods=["probe", "statistical"],
        results={
            "probe": {"success": True, "risk_indicators": ["Signal A"]},
            "statistical": {"success": True, "risk_indicators": ["Signal B"]},
        },
    )

    assessment = condition.assess(ctx)
    assert assessment == "🟢 LOW RISK: No strong shortcut signals detected"


def test_majority_vote_rejects_thresholds_below_one():
    with pytest.raises(ValueError, match="high_threshold must be >= 1"):
        create_condition("majority_vote", high_threshold=0)


def test_register_and_create_custom_condition():
    original = condition_registry._CONDITION_REGISTRY.copy()

    @register_condition("test_condition")
    class _TestCondition(RiskCondition):
        def assess(self, ctx: ConditionContext) -> str:
            return f"methods={len(ctx.methods)}"

    try:
        condition = create_condition("test_condition")
        result = condition.assess(ConditionContext(methods=["probe"], results={}))
        assert result == "methods=1"
        assert "test_condition" in available_conditions()
    finally:
        condition_registry._CONDITION_REGISTRY.clear()
        condition_registry._CONDITION_REGISTRY.update(original)


def test_duplicate_condition_registration_rejected():
    with pytest.raises(ValueError, match="Condition already registered: majority_vote"):

        @register_condition("majority_vote")
        class _DuplicateCondition(RiskCondition):
            def assess(self, ctx: ConditionContext) -> str:
                return "duplicate"


def test_unknown_condition_lists_available_names():
    with pytest.raises(ValueError, match="indicator_count"):
        create_condition("does_not_exist")


def test_shortcut_detector_defaults_to_indicator_count():
    detector = ShortcutDetector(methods=["probe", "statistical"])
    detector.results_ = {
        "probe": {"success": True, "risk_indicators": ["Signal A", "Signal B"]},
        "statistical": {"success": False, "risk_indicators": ["ignored"]},
    }

    assessment = detector._generate_overall_assessment()
    assert assessment.startswith("🔴 HIGH RISK")
    assert "Signal A" in assessment
    assert "Signal B" in assessment


def test_shortcut_detector_supports_majority_vote_condition():
    detector = ShortcutDetector(
        methods=["probe", "statistical"],
        condition_name="majority_vote",
    )
    detector.results_ = {
        "probe": {"success": True, "risk_indicators": ["Signal A", "Signal B"]},
        "statistical": {"success": False, "risk_indicators": ["ignored"]},
    }

    assessment = detector._generate_overall_assessment()
    assert assessment == "🟡 MODERATE RISK: One method detected shortcuts\n  • Signal A"


def test_shortcut_detector_condition_kwargs_do_not_leak_into_builder_kwargs():
    detector = ShortcutDetector(
        methods=["probe"],
        condition_name="majority_vote",
        condition_kwargs={"high_threshold": 3},
        probe_backend="sklearn",
    )

    assert detector.condition_kwargs == {"high_threshold": 3}
    assert "condition_kwargs" not in detector.kwargs
    assert "condition_name" not in detector.kwargs
    assert detector.kwargs["probe_backend"] == "sklearn"


def test_shortcut_detector_rejects_unknown_condition_at_init():
    with pytest.raises(ValueError, match="Available conditions"):
        ShortcutDetector(methods=["probe"], condition_name="does_not_exist")


# ──────────────────────────────────────────────────────────────────────────────
# C02: Weighted Risk Scoring
# ──────────────────────────────────────────────────────────────────────────────


class TestWeightedRisk:
    def test_low_when_no_successful_methods(self):
        condition = create_condition("weighted_risk")
        ctx = ConditionContext(
            methods=["probe", "statistical"],
            results={
                "probe": {"success": False},
                "statistical": {"success": False},
            },
        )
        assert "LOW RISK" in condition.assess(ctx)

    def test_high_when_all_methods_strong(self):
        condition = create_condition("weighted_risk")
        ctx = ConditionContext(
            methods=["probe", "statistical"],
            results={
                "probe": {
                    "success": True,
                    "risk_indicators": ["Probe high accuracy"],
                    "results": {
                        "metrics": {"metric_value": 0.95, "n_classes": 2},
                    },
                },
                "statistical": {
                    "success": True,
                    "risk_indicators": ["Many significant features"],
                    "significant_features": {
                        "A_vs_B": [0, 1, 2],
                        "A_vs_C": [3, 4],
                    },
                },
            },
        )
        assessment = condition.assess(ctx)
        assert "HIGH RISK" in assessment
        assert "Weighted score" in assessment

    def test_moderate_with_mixed_evidence(self):
        condition = create_condition("weighted_risk")
        ctx = ConditionContext(
            methods=["probe", "statistical"],
            results={
                "probe": {
                    "success": True,
                    "risk_indicators": ["Probe signal"],
                    "results": {
                        "metrics": {"metric_value": 0.85, "n_classes": 2},
                    },
                },
                "statistical": {
                    "success": True,
                    "risk_indicators": [],
                    "significant_features": {
                        "A_vs_B": [],
                        "A_vs_C": [],
                    },
                },
            },
        )
        assessment = condition.assess(ctx)
        assert "MODERATE RISK" in assessment

    def test_custom_thresholds(self):
        condition = create_condition("weighted_risk", high_threshold=0.9, moderate_threshold=0.5)
        ctx = ConditionContext(
            methods=["probe"],
            results={
                "probe": {
                    "success": True,
                    "risk_indicators": ["Signal"],
                    "results": {
                        "metrics": {"metric_value": 0.85, "n_classes": 2},
                    },
                },
            },
        )
        assessment = condition.assess(ctx)
        assert "MODERATE RISK" in assessment

    def test_invalid_thresholds_rejected(self):
        with pytest.raises(ValueError, match="Thresholds must satisfy"):
            create_condition("weighted_risk", high_threshold=0.2, moderate_threshold=0.5)

    def test_hbac_weight_uses_confidence(self):
        condition = create_condition("weighted_risk")
        ctx = ConditionContext(
            methods=["hbac"],
            results={
                "hbac": {
                    "success": True,
                    "risk_indicators": ["High purity clusters"],
                    "report": {
                        "has_shortcut": {"exists": True, "confidence": "high"},
                    },
                },
            },
        )
        assessment = condition.assess(ctx)
        assert "HIGH RISK" in assessment
        assert "hbac=1.00" in assessment

    def test_unknown_method_uses_indicator_fallback(self):
        condition = create_condition("weighted_risk")
        ctx = ConditionContext(
            methods=["custom_method"],
            results={
                "custom_method": {
                    "success": True,
                    "risk_indicators": ["Custom signal"],
                },
            },
        )
        assessment = condition.assess(ctx)
        assert "HIGH RISK" in assessment
        assert "custom_method=1.00" in assessment


# ──────────────────────────────────────────────────────────────────────────────
# C03: Multi-Attribute Intersection
# ──────────────────────────────────────────────────────────────────────────────


class TestMultiAttribute:
    def test_low_when_no_attributes_flagged(self):
        condition = create_condition("multi_attribute")
        ctx = ConditionContext(
            methods=["probe"],
            results={
                "probe": {
                    "success": True,
                    "risk_indicators": [],
                    "by_attribute": {
                        "race": {"risk_indicators": []},
                        "sex": {"risk_indicators": []},
                    },
                },
            },
        )
        assert "LOW RISK" in condition.assess(ctx)

    def test_high_when_multiple_attributes_flagged(self):
        condition = create_condition("multi_attribute")
        ctx = ConditionContext(
            methods=["probe"],
            results={
                "probe": {
                    "success": True,
                    "by_attribute": {
                        "race": {"risk_indicators": ["Race shortcut detected"]},
                        "sex": {"risk_indicators": ["Sex shortcut detected"]},
                    },
                },
            },
        )
        assessment = condition.assess(ctx)
        assert "HIGH RISK" in assessment
        assert "intersectional" in assessment
        assert "2 attributes" in assessment

    def test_moderate_when_single_attribute_flagged(self):
        condition = create_condition("multi_attribute")
        ctx = ConditionContext(
            methods=["probe"],
            results={
                "probe": {
                    "success": True,
                    "by_attribute": {
                        "race": {"risk_indicators": ["Race shortcut"]},
                        "sex": {"risk_indicators": []},
                    },
                },
            },
        )
        assessment = condition.assess(ctx)
        assert "MODERATE RISK" in assessment
        assert "race" in assessment

    def test_custom_high_threshold(self):
        condition = create_condition("multi_attribute", high_threshold=3)
        ctx = ConditionContext(
            methods=["probe"],
            results={
                "probe": {
                    "success": True,
                    "by_attribute": {
                        "race": {"risk_indicators": ["Race signal"]},
                        "sex": {"risk_indicators": ["Sex signal"]},
                    },
                },
            },
        )
        assessment = condition.assess(ctx)
        assert "MODERATE RISK" in assessment
        assert "below intersection threshold" in assessment

    def test_invalid_threshold_rejected(self):
        with pytest.raises(ValueError, match="high_threshold must be >= 1"):
            create_condition("multi_attribute", high_threshold=0)

    def test_falls_back_to_flat_results(self):
        """When results don't have by_attribute, uses flat risk_indicators."""
        condition = create_condition("multi_attribute")
        ctx = ConditionContext(
            methods=["probe", "statistical"],
            results={
                "probe": {
                    "success": True,
                    "risk_indicators": ["Probe signal"],
                },
                "statistical": {
                    "success": True,
                    "risk_indicators": ["Stat signal"],
                },
            },
        )
        assessment = condition.assess(ctx)
        assert "MODERATE RISK" in assessment

    def test_aggregates_across_methods(self):
        """Multiple methods contributing indicators to same attributes."""
        condition = create_condition("multi_attribute")
        ctx = ConditionContext(
            methods=["probe", "statistical"],
            results={
                "probe": {
                    "success": True,
                    "by_attribute": {
                        "race": {"risk_indicators": ["Probe race signal"]},
                        "sex": {"risk_indicators": ["Probe sex signal"]},
                    },
                },
                "statistical": {
                    "success": True,
                    "by_attribute": {
                        "race": {"risk_indicators": ["Stat race signal"]},
                        "sex": {"risk_indicators": []},
                    },
                },
            },
        )
        assessment = condition.assess(ctx)
        assert "HIGH RISK" in assessment
        assert "[race]: 2 indicator(s)" in assessment
        assert "[sex]: 1 indicator(s)" in assessment


# ──────────────────────────────────────────────────────────────────────────────
# C04: Meta-Classifier
# ──────────────────────────────────────────────────────────────────────────────


class TestMetaClassifier:
    def test_low_when_no_methods_succeed(self):
        condition = create_condition("meta_classifier")
        ctx = ConditionContext(
            methods=["probe", "statistical"],
            results={
                "probe": {"success": False},
                "statistical": {"success": False},
            },
        )
        assessment = condition.assess(ctx)
        assert "LOW RISK" in assessment

    def test_high_with_strong_signals(self):
        condition = create_condition("meta_classifier")
        # The trained model needs all 6 methods reporting high to exceed high_threshold
        ctx = ConditionContext(
            methods=["probe", "statistical", "hbac", "geometric", "frequency", "gce"],
            results={
                "probe": {
                    "success": True,
                    "risk_indicators": ["High probe accuracy", "Above threshold"],
                    "risk_value": "high",
                    "results": {"metrics": {"metric_value": 0.95, "threshold": 0.7}},
                },
                "statistical": {
                    "success": True,
                    "risk_indicators": ["Many significant features", "Strong signal"],
                    "risk_value": "high",
                    "significant_features": {"A_vs_B": [0, 1, 2]},
                },
                "hbac": {
                    "success": True,
                    "risk_indicators": ["High purity clusters", "Shortcut exists"],
                    "risk_value": "high",
                    "report": {
                        "has_shortcut": {"exists": True, "confidence": "high"},
                    },
                },
                "geometric": {
                    "success": True,
                    "risk_indicators": ["Large effect size", "Bias direction detected"],
                    "risk_value": "high",
                },
                "frequency": {
                    "success": True,
                    "risk_indicators": ["Frequency bias detected", "Critical drift"],
                    "risk_value": "high",
                },
                "gce": {
                    "success": True,
                    "risk_indicators": ["GCE outliers found", "High loss disparity"],
                    "risk_value": "high",
                },
            },
        )
        assessment = condition.assess(ctx)
        assert "HIGH RISK" in assessment
        assert "Meta-score" in assessment

    def test_moderate_with_mixed_signals(self):
        condition = create_condition("meta_classifier")
        # 3 methods reporting high produces a score in the moderate range
        ctx = ConditionContext(
            methods=["probe", "statistical", "hbac"],
            results={
                "probe": {
                    "success": True,
                    "risk_indicators": ["Probe signal", "Another signal"],
                    "risk_value": "high",
                    "results": {"metrics": {"metric_value": 0.92, "threshold": 0.7}},
                },
                "statistical": {
                    "success": True,
                    "risk_indicators": ["Significant features found"],
                    "risk_value": "high",
                    "significant_features": {"A_vs_B": [0, 1, 2]},
                },
                "hbac": {
                    "success": True,
                    "risk_indicators": ["High purity clusters"],
                    "risk_value": "high",
                    "report": {
                        "has_shortcut": {"exists": True, "confidence": "high"},
                    },
                },
            },
        )
        assessment = condition.assess(ctx)
        assert "MODERATE RISK" in assessment

    def test_custom_thresholds(self):
        condition = create_condition("meta_classifier", high_threshold=0.95, moderate_threshold=0.9)
        ctx = ConditionContext(
            methods=["probe"],
            results={
                "probe": {
                    "success": True,
                    "risk_indicators": ["Signal"],
                    "risk_value": "high",
                    "results": {"metrics": {"metric_value": 0.95, "threshold": 0.7}},
                },
            },
        )
        assessment = condition.assess(ctx)
        assert "LOW RISK" in assessment

    def test_invalid_thresholds_rejected(self):
        with pytest.raises(ValueError, match="Thresholds must satisfy"):
            create_condition("meta_classifier", high_threshold=0.2, moderate_threshold=0.5)

    def test_extract_features_returns_expected_keys(self):
        from shortcut_detect.conditions.meta_classifier import MetaClassifierCondition

        ctx = ConditionContext(
            methods=["probe", "statistical"],
            results={
                "probe": {
                    "success": True,
                    "risk_indicators": ["Signal"],
                    "risk_value": "high",
                    "results": {"metrics": {"metric_value": 0.9, "threshold": 0.7}},
                },
                "statistical": {
                    "success": True,
                    "risk_indicators": [],
                    "risk_value": "low",
                    "significant_features": {"A_vs_B": [1], "C_vs_D": []},
                },
            },
        )
        features = MetaClassifierCondition.extract_features(ctx)
        assert features["probe_success"] == 1.0
        assert features["probe_metric_value"] == 0.9
        assert features["statistical_sig_ratio"] == 0.5
        assert features["probe_risk_value"] == 1.0
        assert features["statistical_risk_value"] == 0.0

    def test_uses_trained_model_when_bundled(self):
        condition = create_condition("meta_classifier")
        # Bundled meta_model.joblib is shipped with the package
        assert condition._using_heuristic is False

    def test_detector_integration_with_weighted_risk(self):
        detector = ShortcutDetector(
            methods=["probe", "statistical"],
            condition_name="weighted_risk",
        )
        detector.results_ = {
            "probe": {
                "success": True,
                "risk_indicators": ["Signal A"],
                "results": {"metrics": {"metric_value": 0.95, "n_classes": 2}},
            },
            "statistical": {
                "success": True,
                "risk_indicators": ["Signal B"],
                "significant_features": {"A_vs_B": [0, 1, 2]},
            },
        }
        assessment = detector._generate_overall_assessment()
        assert "HIGH RISK" in assessment

    def test_detector_integration_with_meta_classifier(self):
        detector = ShortcutDetector(
            methods=["probe"],
            condition_name="meta_classifier",
        )
        detector.results_ = {
            "probe": {
                "success": True,
                "risk_indicators": ["High accuracy"],
                "risk_value": "high",
                "results": {"metrics": {"metric_value": 0.95, "threshold": 0.7}},
            },
        }
        assessment = detector._generate_overall_assessment()
        assert "Meta-score" in assessment

    def test_all_new_conditions_registered(self):
        conditions = available_conditions()
        assert "weighted_risk" in conditions
        assert "multi_attribute" in conditions
        assert "meta_classifier" in conditions
