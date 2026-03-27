"""Multi-attribute intersection condition.

Cross-references risk across different sensitive attributes. If shortcuts
are detected for BOTH race AND sex (for example), that's higher confidence
than a single attribute showing risk.
"""

from __future__ import annotations

from typing import Any

from .base import ConditionContext, RiskCondition
from .registry import register_condition


def _extract_attribute_risks(
    results: dict[str, dict[str, Any]],
    methods: list[str],
) -> dict[str, list[str]]:
    """Extract per-attribute risk indicators from results.

    Handles both single-attribute results (flat) and multi-attribute results
    (nested under "by_attribute").
    """
    attribute_indicators: dict[str, list[str]] = {}

    for method in methods:
        result = results.get(method)
        if not result or not result.get("success"):
            continue

        by_attr = result.get("by_attribute")
        if by_attr and isinstance(by_attr, dict):
            for attr_name, attr_result in by_attr.items():
                if not isinstance(attr_result, dict):
                    continue
                attr_indicators = attr_result.get("risk_indicators", []) or []
                if attr_indicators:
                    attribute_indicators.setdefault(attr_name, []).extend(
                        f"[{method}] {ind}" for ind in attr_indicators
                    )
        else:
            risk_inds = result.get("risk_indicators", []) or []
            if risk_inds:
                attribute_indicators.setdefault("_default", []).extend(
                    f"[{method}] {ind}" for ind in risk_inds
                )

    return attribute_indicators


@register_condition("multi_attribute")
class MultiAttributeCondition(RiskCondition):
    """Cross-reference risk across sensitive attributes.

    Risk escalates when multiple attributes independently show shortcut
    evidence, since this indicates a systemic issue rather than noise.
    """

    def __init__(self, high_threshold: int = 2) -> None:
        if high_threshold < 1:
            raise ValueError("high_threshold must be >= 1")
        self.high_threshold = high_threshold

    def assess(self, ctx: ConditionContext) -> str:
        attr_risks = _extract_attribute_risks(ctx.results, ctx.methods)
        flagged_attrs = {attr: inds for attr, inds in attr_risks.items() if inds}
        n_flagged = len(flagged_attrs)

        if n_flagged == 0:
            return "🟢 LOW RISK: No shortcut signals detected across attributes"

        all_indicators: list[str] = []
        attr_summaries: list[str] = []
        for attr, inds in sorted(flagged_attrs.items()):
            display_name = attr if attr != "_default" else "default attribute"
            attr_summaries.append(f"  [{display_name}]: {len(inds)} indicator(s)")
            all_indicators.extend(inds)

        seen: set[str] = set()
        deduped = [i for i in all_indicators if not (i in seen or seen.add(i))]

        if n_flagged >= self.high_threshold:
            lines = [
                f"🔴 HIGH RISK: Shortcuts detected across {n_flagged} attributes "
                f"(intersectional risk)",
            ]
            lines.extend(attr_summaries)
            lines.extend(f"  • {ind}" for ind in deduped)
            return "\n".join(lines)

        if n_flagged == 1:
            attr_name = next(iter(flagged_attrs))
            display = attr_name if attr_name != "_default" else "one attribute"
            lines = [
                f"🟡 MODERATE RISK: Shortcuts detected for {display} only",
            ]
            lines.extend(attr_summaries)
            lines.extend(f"  • {ind}" for ind in deduped)
            return "\n".join(lines)

        lines = [
            f"🟡 MODERATE RISK: Shortcuts detected for {n_flagged} attributes "
            f"(below intersection threshold of {self.high_threshold})",
        ]
        lines.extend(attr_summaries)
        lines.extend(f"  • {ind}" for ind in deduped)
        return "\n".join(lines)
