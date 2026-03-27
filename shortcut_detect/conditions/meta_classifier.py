"""Meta-classifier condition.

Trains a small classifier on the outputs of all detectors as features,
learning the interactions and trust levels between methods. When no
trained model is available, falls back to a heuristic ensemble score.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np

from .base import ConditionContext, RiskCondition
from .registry import register_condition


def _extract_features(ctx: ConditionContext) -> dict[str, float]:
    """Extract numeric features from detector results for the meta-classifier."""
    features: dict[str, float] = {}

    for method in ctx.methods:
        result = ctx.results.get(method)
        prefix = method

        if not result or not result.get("success"):
            features[f"{prefix}_success"] = 0.0
            features[f"{prefix}_n_indicators"] = 0.0
            features[f"{prefix}_risk_value"] = 0.0
            continue

        features[f"{prefix}_success"] = 1.0

        risk_indicators = result.get("risk_indicators", []) or []
        features[f"{prefix}_n_indicators"] = float(len(risk_indicators))

        risk_map = {"low": 0.0, "moderate": 0.5, "high": 1.0}
        risk_val = result.get("risk_value", "low")
        features[f"{prefix}_risk_value"] = risk_map.get(risk_val, 0.0)

        # Probe-specific features
        if method == "probe":
            inner = result.get("results", {})
            metrics = inner.get("metrics", result.get("metrics", {}))
            features[f"{prefix}_metric_value"] = float(metrics.get("metric_value", 0.0) or 0.0)
            features[f"{prefix}_threshold"] = float(metrics.get("threshold", 0.0) or 0.0)

        # Statistical-specific features
        if method == "statistical":
            sig = result.get("significant_features", {})
            total_comparisons = max(len(sig), 1)
            with_sig = sum(1 for v in sig.values() if v is not None and len(v) > 0)
            features[f"{prefix}_sig_ratio"] = with_sig / total_comparisons

        # HBAC-specific features
        if method == "hbac":
            report = result.get("report", {})
            has_shortcut = report.get("has_shortcut", {})
            features[f"{prefix}_shortcut_exists"] = 1.0 if has_shortcut.get("exists") else 0.0
            conf_map = {"low": 0.0, "moderate": 0.5, "high": 1.0}
            features[f"{prefix}_confidence"] = conf_map.get(
                has_shortcut.get("confidence", "low"), 0.0
            )

        # Geometric-specific features
        if method == "geometric":
            bias_pairs = result.get("bias_pairs", [])
            if bias_pairs:
                effects = [getattr(bp, "effect_size", 0.0) for bp in bias_pairs]
                features[f"{prefix}_max_effect"] = float(max(effects))
                features[f"{prefix}_mean_effect"] = float(np.mean(effects))
            else:
                features[f"{prefix}_max_effect"] = 0.0
                features[f"{prefix}_mean_effect"] = 0.0

    return features


def _heuristic_score(features: dict[str, float]) -> float:
    """Compute a heuristic ensemble score when no trained model is available.

    Combines risk values and indicator counts into a [0, 1] score.
    """
    risk_values = [v for k, v in features.items() if k.endswith("_risk_value")]
    n_indicators = [v for k, v in features.items() if k.endswith("_n_indicators")]

    if not risk_values:
        return 0.0

    avg_risk = sum(risk_values) / len(risk_values)
    total_indicators = sum(n_indicators)
    indicator_signal = min(total_indicators / 4.0, 1.0)

    return 0.6 * avg_risk + 0.4 * indicator_signal


_BUNDLED_MODEL_DIR = Path(__file__).parent


@register_condition("meta_classifier")
class MetaClassifierCondition(RiskCondition):
    """Use a trained meta-classifier or heuristic fallback for risk scoring.

    By default, loads the bundled model trained on synthetic benchmark data.
    A custom model path can be provided to override. Falls back to a heuristic
    ensemble if no model is available.
    """

    def __init__(
        self,
        model_path: str | None = None,
        high_threshold: float | None = None,
        moderate_threshold: float | None = None,
    ) -> None:
        self.model = None
        self.feature_names: list[str] | None = None
        self._using_heuristic = True

        # Try loading: explicit path > bundled model > heuristic fallback
        if model_path is not None:
            self._load_model(model_path)
        else:
            bundled = _BUNDLED_MODEL_DIR / "meta_model.joblib"
            if bundled.exists():
                self._load_model(str(bundled))

        # Thresholds: explicit > from model metadata > defaults
        meta_high = getattr(self, "_meta_high", None)
        meta_mod = getattr(self, "_meta_mod", None)
        default_high = meta_high or (0.6 if self._using_heuristic else 0.7)
        default_mod = meta_mod or (0.3 if self._using_heuristic else 0.35)
        self.high_threshold = high_threshold if high_threshold is not None else default_high
        self.moderate_threshold = (
            moderate_threshold if moderate_threshold is not None else default_mod
        )

        if not 0 < self.moderate_threshold < self.high_threshold <= 1.0:
            raise ValueError(
                f"Thresholds must satisfy 0 < moderate ({self.moderate_threshold}) "
                f"< high ({self.high_threshold}) <= 1.0"
            )

    def _load_model(self, model_path: str) -> None:
        """Load a pre-trained sklearn model and its feature metadata."""
        import joblib

        path = Path(model_path)
        self.model = joblib.load(path)

        meta_path = path.with_suffix(".meta.json")
        if meta_path.exists():
            with open(meta_path) as f:
                meta = json.load(f)
            self.feature_names = meta.get("feature_names", [])
            if "high_threshold" in meta:
                self._meta_high = meta["high_threshold"]
            if "moderate_threshold" in meta:
                self._meta_mod = meta["moderate_threshold"]

        self._using_heuristic = False

    def _predict_score(self, features: dict[str, float]) -> float:
        """Get risk score from the trained model."""
        if self.feature_names:
            X = np.array([[features.get(f, 0.0) for f in self.feature_names]])
        else:
            X = np.array([list(features.values())])

        if hasattr(self.model, "predict_proba"):
            proba = self.model.predict_proba(X)[0]
            return float(proba[-1])
        return float(self.model.predict(X)[0])

    def assess(self, ctx: ConditionContext) -> str:
        features = _extract_features(ctx)

        if self._using_heuristic:
            score = _heuristic_score(features)
            mode = "heuristic"
        else:
            score = self._predict_score(features)
            mode = "trained model"

        indicators: list[str] = []
        for method in ctx.methods:
            result = ctx.results.get(method)
            if result and result.get("success"):
                indicators.extend(result.get("risk_indicators", []))

        seen: set[str] = set()
        deduped = [i for i in indicators if not (i in seen or seen.add(i))]

        score_line = f"  Meta-score: {score:.2f} ({mode})"

        if score >= self.high_threshold:
            lines = ["🔴 HIGH RISK: Meta-classifier indicates strong shortcut evidence"]
            lines.append(score_line)
            lines.extend(f"  • {ind}" for ind in deduped)
            return "\n".join(lines)
        if score >= self.moderate_threshold:
            lines = ["🟡 MODERATE RISK: Meta-classifier indicates some shortcut evidence"]
            lines.append(score_line)
            lines.extend(f"  • {ind}" for ind in deduped)
            return "\n".join(lines)
        return "🟢 LOW RISK: Meta-classifier indicates no strong shortcuts\n" + score_line

    @staticmethod
    def extract_features(ctx: ConditionContext) -> dict[str, float]:
        """Public access to feature extraction for training pipelines."""
        return _extract_features(ctx)
