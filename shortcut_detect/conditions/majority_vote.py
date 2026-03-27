"""Majority-vote overall assessment condition."""

from __future__ import annotations

from .base import ConditionContext, RiskCondition
from .registry import register_condition


@register_condition("majority_vote")
class MajorityVoteCondition(RiskCondition):
    """Count successful methods with indicators as votes."""

    def __init__(self, high_threshold: int = 2) -> None:
        if high_threshold < 1:
            raise ValueError("high_threshold must be >= 1")
        self.high_threshold = high_threshold

    def assess(self, ctx: ConditionContext) -> str:
        indicators: list[str] = []
        votes = 0

        for method in ctx.methods:
            result = ctx.results.get(method)
            if not result or not result.get("success"):
                continue

            method_indicators = result.get("risk_indicators", []) or []
            if method_indicators:
                votes += 1
                indicators.extend(method_indicators)

        seen: set[str] = set()
        deduped_indicators = [
            indicator for indicator in indicators if not (indicator in seen or seen.add(indicator))
        ]

        if votes >= self.high_threshold:
            return "🔴 HIGH RISK: Multiple methods detected shortcuts\n" + "\n".join(
                f"  • {indicator}" for indicator in deduped_indicators
            )
        if votes == 1:
            if deduped_indicators:
                return (
                    "🟡 MODERATE RISK: One method detected shortcuts\n  • " + deduped_indicators[0]
                )
            return "🟡 MODERATE RISK: One method detected shortcuts"
        return "🟢 LOW RISK: No strong shortcut signals detected"
