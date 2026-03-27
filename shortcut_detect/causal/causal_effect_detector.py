"""Backward-compatible exports for the Causal Effect detector.

This module preserves the historical import path:
`shortcut_detect.causal.causal_effect_detector`.
"""

from __future__ import annotations

from sklearn.linear_model import LogisticRegression

from .causal_effect.src.detector import AttributeEffectResult, CausalEffectDetector

__all__ = [
    "AttributeEffectResult",
    "CausalEffectDetector",
    "LogisticRegression",
]
