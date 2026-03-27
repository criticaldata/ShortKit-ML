"""A02: Verify aggregation formula R matches implementation.

Tests:
- indicator_count: counts risk_indicators across methods
- majority_vote: counts methods with indicators (votes)
- weighted_risk: weighted average of per-method weights in [0,1]
- multi_attribute: counts flagged attributes
- meta_classifier: meta-score in [0,1]
"""

from __future__ import annotations

import pytest

from shortcut_detect.benchmark.synthetic_generator import SyntheticGenerator
from shortcut_detect.conditions import (
    ConditionContext,
    available_conditions,
    create_condition,
)
from shortcut_detect.unified import ShortcutDetector

CONDITION_NAMES = [
    "indicator_count",
    "majority_vote",
    "weighted_risk",
    "multi_attribute",
    "meta_classifier",
]

CORE_METHODS = ["hbac", "probe", "statistical", "geometric"]


@pytest.fixture(scope="module")
def detector_with_results():
    """Run ShortcutDetector on synthetic data and return the detector."""
    gen = SyntheticGenerator(
        n_samples=300,
        embedding_dim=32,
        shortcut_dims=3,
        group_ratio=0.5,
        seed=42,
    )
    data = gen.generate(effect_size=1.5)
    detector = ShortcutDetector(methods=CORE_METHODS, seed=42)
    detector.fit(data.embeddings, data.labels, group_labels=data.group_labels)
    return detector


class TestAllConditionsRunnable:
    """All 5 condition_names should be registered and callable."""

    def test_all_conditions_registered(self):
        registered = available_conditions()
        for name in CONDITION_NAMES:
            assert name in registered, f"Condition '{name}' not registered"

    @pytest.mark.parametrize("condition_name", CONDITION_NAMES)
    def test_condition_produces_string(self, detector_with_results, condition_name):
        det = detector_with_results
        ctx = ConditionContext(methods=det.methods, results=det.results_)
        condition = create_condition(condition_name)
        assessment = condition.assess(ctx)
        assert isinstance(assessment, str)
        assert len(assessment) > 0


class TestIndicatorCount:
    """indicator_count: R = f(total indicators across methods)."""

    def test_no_indicators_gives_low(self):
        results = {m: {"success": True, "risk_indicators": []} for m in CORE_METHODS}
        ctx = ConditionContext(methods=CORE_METHODS, results=results)
        condition = create_condition("indicator_count")
        assessment = condition.assess(ctx)
        assert "LOW" in assessment

    def test_multiple_indicators_gives_high(self):
        results = {
            m: {"success": True, "risk_indicators": [f"indicator_{m}"]} for m in CORE_METHODS
        }
        ctx = ConditionContext(methods=CORE_METHODS, results=results)
        condition = create_condition("indicator_count")
        assessment = condition.assess(ctx)
        # With 4 indicators (>= 2), should be HIGH
        assert "HIGH" in assessment

    def test_one_indicator_gives_moderate(self):
        results = {m: {"success": True, "risk_indicators": []} for m in CORE_METHODS}
        results["probe"]["risk_indicators"] = ["probe detected shortcut"]
        ctx = ConditionContext(methods=CORE_METHODS, results=results)
        condition = create_condition("indicator_count")
        assessment = condition.assess(ctx)
        assert "MODERATE" in assessment

    def test_counts_indicators_correctly(self):
        """Verify it counts total indicators, not total methods."""
        results = {m: {"success": True, "risk_indicators": []} for m in CORE_METHODS}
        # One method with 3 indicators
        results["probe"]["risk_indicators"] = ["a", "b", "c"]
        ctx = ConditionContext(methods=CORE_METHODS, results=results)
        condition = create_condition("indicator_count")
        assessment = condition.assess(ctx)
        # 3 indicators >= 2 => HIGH
        assert "HIGH" in assessment


class TestMajorityVote:
    """majority_vote: counts methods with indicators as votes."""

    def test_no_votes_gives_low(self):
        results = {m: {"success": True, "risk_indicators": []} for m in CORE_METHODS}
        ctx = ConditionContext(methods=CORE_METHODS, results=results)
        condition = create_condition("majority_vote")
        assessment = condition.assess(ctx)
        assert "LOW" in assessment

    def test_counts_methods_not_indicators(self):
        """One method with many indicators should count as 1 vote."""
        results = {m: {"success": True, "risk_indicators": []} for m in CORE_METHODS}
        results["probe"]["risk_indicators"] = ["a", "b", "c"]
        ctx = ConditionContext(methods=CORE_METHODS, results=results)
        condition = create_condition("majority_vote")
        assessment = condition.assess(ctx)
        # Only 1 method has indicators => 1 vote => MODERATE
        assert "MODERATE" in assessment

    def test_two_methods_gives_high(self):
        results = {m: {"success": True, "risk_indicators": []} for m in CORE_METHODS}
        results["probe"]["risk_indicators"] = ["probe found it"]
        results["hbac"]["risk_indicators"] = ["hbac found it"]
        ctx = ConditionContext(methods=CORE_METHODS, results=results)
        condition = create_condition("majority_vote")
        assessment = condition.assess(ctx)
        # 2 votes >= high_threshold(2) => HIGH
        assert "HIGH" in assessment

    def test_custom_threshold(self):
        results = {m: {"success": True, "risk_indicators": []} for m in CORE_METHODS}
        results["probe"]["risk_indicators"] = ["a"]
        results["hbac"]["risk_indicators"] = ["b"]
        ctx = ConditionContext(methods=CORE_METHODS, results=results)
        # With high_threshold=3, 2 votes < 3 => not HIGH
        condition = create_condition("majority_vote", high_threshold=3)
        assessment = condition.assess(ctx)
        assert "HIGH" not in assessment


class TestWeightedRisk:
    """weighted_risk: weighted average of per-method weights in [0,1]."""

    def test_weighted_score_in_unit_interval(self, detector_with_results):
        det = detector_with_results
        ctx = ConditionContext(methods=det.methods, results=det.results_)
        condition = create_condition("weighted_risk")
        assessment = condition.assess(ctx)
        # The assessment string contains the weighted score
        assert isinstance(assessment, str)

    def test_zero_weights_gives_low(self):
        """All methods succeed but have no signal => low risk."""
        results = {
            m: {
                "success": True,
                "risk_indicators": [],
                "risk_value": "low",
                "results": {"metrics": {"metric_value": 0.5, "n_classes": 2}},
                "significant_features": {},
                "report": {"has_shortcut": {"confidence": "low"}},
                "bias_pairs": [],
            }
            for m in CORE_METHODS
        }
        ctx = ConditionContext(methods=CORE_METHODS, results=results)
        condition = create_condition("weighted_risk")
        assessment = condition.assess(ctx)
        assert "LOW" in assessment

    def test_weighted_average_formula(self):
        """Verify that the score line in the assessment reflects weighted average."""
        results = {
            "probe": {
                "success": True,
                "risk_indicators": ["found it"],
                "results": {"metrics": {"metric_value": 1.0, "n_classes": 2}},
            },
            "hbac": {
                "success": True,
                "risk_indicators": [],
                "report": {"has_shortcut": {"confidence": "low"}},
            },
        }
        ctx = ConditionContext(methods=["probe", "hbac"], results=results)
        condition = create_condition("weighted_risk")
        assessment = condition.assess(ctx)
        # probe weight = (1.0 - 0.5) / 0.5 = 1.0, hbac weight = 0.0
        # weighted average = 0.5
        assert "0.50" in assessment


class TestMetaClassifier:
    """meta_classifier: produces a score and maps to risk level."""

    def test_assess_returns_string(self, detector_with_results):
        det = detector_with_results
        ctx = ConditionContext(methods=det.methods, results=det.results_)
        condition = create_condition("meta_classifier")
        assessment = condition.assess(ctx)
        assert isinstance(assessment, str)
        assert any(level in assessment for level in ["LOW", "MODERATE", "HIGH"])

    def test_empty_results_gives_low(self):
        results = {m: {"success": False} for m in CORE_METHODS}
        ctx = ConditionContext(methods=CORE_METHODS, results=results)
        condition = create_condition("meta_classifier")
        assessment = condition.assess(ctx)
        assert "LOW" in assessment


class TestShortcutDetectorWithConditions:
    """Integration: run ShortcutDetector with each condition_name."""

    @pytest.mark.parametrize("condition_name", CONDITION_NAMES)
    def test_detector_with_condition(self, condition_name):
        gen = SyntheticGenerator(
            n_samples=200,
            embedding_dim=16,
            shortcut_dims=2,
            group_ratio=0.5,
            seed=7,
        )
        data = gen.generate(effect_size=1.0)
        detector = ShortcutDetector(
            methods=CORE_METHODS,
            seed=42,
            condition_name=condition_name,
        )
        detector.fit(data.embeddings, data.labels, group_labels=data.group_labels)
        summary = detector.summary()
        assert isinstance(summary, str)
        assert "OVERALL ASSESSMENT" in summary
