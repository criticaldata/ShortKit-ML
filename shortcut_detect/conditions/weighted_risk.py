"""Weighted risk scoring condition.

Each detector gets a trust weight based on the strength of its evidence:
- Probe: weighted by how far above chance the metric value is
- Statistical: weighted by proportion of significant comparisons
- HBAC: weighted by confidence level from shortcut analysis
- Geometric: weighted by effect size magnitude
"""

from __future__ import annotations

from .base import ConditionContext, RiskCondition
from .registry import register_condition


def _probe_weight(result: dict) -> float:
    """Weight probe by how far above chance the metric value is."""
    results_inner = result.get("results", {})
    metrics = results_inner.get("metrics", result.get("metrics", {}))
    metric_value = metrics.get("metric_value")
    n_classes = metrics.get("n_classes", 2)
    if metric_value is None:
        return 0.0
    chance = 1.0 / max(n_classes, 2)
    above_chance = max(metric_value - chance, 0.0)
    max_above = 1.0 - chance
    if max_above <= 0:
        return 0.0
    return above_chance / max_above


def _statistical_weight(result: dict) -> float:
    """Weight statistical by proportion of comparisons with significant features."""
    sig_features = result.get("significant_features", {})
    if not sig_features:
        return 0.0
    total = len(sig_features)
    with_sig = sum(1 for feats in sig_features.values() if feats is not None and len(feats) > 0)
    return with_sig / total if total > 0 else 0.0


def _hbac_weight(result: dict) -> float:
    """Weight HBAC by shortcut detection confidence."""
    report = result.get("report", {})
    has_shortcut = report.get("has_shortcut", {})
    confidence = has_shortcut.get("confidence", "low")
    confidence_map = {"low": 0.0, "moderate": 0.5, "high": 1.0}
    return confidence_map.get(confidence, 0.0)


def _geometric_weight(result: dict) -> float:
    """Weight geometric by maximum effect size across bias pairs."""
    bias_pairs = result.get("bias_pairs", [])
    if not bias_pairs:
        return 0.0
    max_effect = max(
        (getattr(bp, "effect_size", 0.0) for bp in bias_pairs),
        default=0.0,
    )
    return min(max_effect / 1.0, 1.0)


_WEIGHT_FUNCTIONS = {
    "probe": _probe_weight,
    "statistical": _statistical_weight,
    "hbac": _hbac_weight,
    "geometric": _geometric_weight,
}


@register_condition("weighted_risk")
class WeightedRiskCondition(RiskCondition):
    """Weight each detector by its reliability/strength of evidence.

    The final score is the weighted average of individual detector weights,
    normalized to [0, 1]. Thresholds map this score to HIGH/MODERATE/LOW.
    """

    def __init__(
        self,
        high_threshold: float = 0.6,
        moderate_threshold: float = 0.3,
    ) -> None:
        if not 0 < moderate_threshold < high_threshold <= 1.0:
            raise ValueError(
                f"Thresholds must satisfy 0 < moderate ({moderate_threshold}) "
                f"< high ({high_threshold}) <= 1.0"
            )
        self.high_threshold = high_threshold
        self.moderate_threshold = moderate_threshold

    def assess(self, ctx: ConditionContext) -> str:
        weights: list[float] = []
        indicators: list[str] = []
        method_scores: list[str] = []

        for method in ctx.methods:
            result = ctx.results.get(method)
            if not result or not result.get("success"):
                continue

            weight_fn = _WEIGHT_FUNCTIONS.get(method)
            if weight_fn is not None:
                w = weight_fn(result)
            else:
                w = 1.0 if result.get("risk_indicators") else 0.0

            weights.append(w)
            method_scores.append(f"{method}={w:.2f}")
            indicators.extend(result.get("risk_indicators", []))

        if not weights:
            return "🟢 LOW RISK: No strong shortcut signals detected"

        score = sum(weights) / len(weights)
        score_line = f"  Weighted score: {score:.2f} ({', '.join(method_scores)})"

        seen: set[str] = set()
        deduped = [i for i in indicators if not (i in seen or seen.add(i))]

        if score >= self.high_threshold:
            lines = ["🔴 HIGH RISK: Strong weighted evidence of shortcuts"]
            lines.append(score_line)
            lines.extend(f"  • {ind}" for ind in deduped)
            return "\n".join(lines)
        if score >= self.moderate_threshold:
            lines = ["🟡 MODERATE RISK: Some weighted evidence of shortcuts"]
            lines.append(score_line)
            lines.extend(f"  • {ind}" for ind in deduped)
            return "\n".join(lines)
        return "🟢 LOW RISK: No strong shortcut signals detected\n" + score_line
