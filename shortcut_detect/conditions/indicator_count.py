"""Legacy-compatible overall assessment condition."""

from __future__ import annotations

from .base import ConditionContext, RiskCondition
from .registry import register_condition


@register_condition("indicator_count")
class IndicatorCountCondition(RiskCondition):
    """Preserve the existing overall assessment semantics."""

    def assess(self, ctx: ConditionContext) -> str:
        risk_indicators: list[str] = []
        for method in ctx.methods:
            result = ctx.results.get(method)
            if not result or not result.get("success"):
                continue
            risk_indicators.extend(result.get("risk_indicators", []))

        if len(risk_indicators) >= 2:
            return "🔴 HIGH RISK: Multiple methods detected shortcuts\n" + "\n".join(
                f"  • {indicator}" for indicator in risk_indicators
            )
        if len(risk_indicators) == 1:
            return "🟡 MODERATE RISK: One method detected shortcuts\n  • " + risk_indicators[0]
        return "🟢 LOW RISK: No strong shortcut signals detected"
