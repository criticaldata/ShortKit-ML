"""Shared constants and utilities for benchmark method detection.

Centralises the ``method_flag`` logic and method categorisation that was
previously duplicated across paper_runner, measurement, sensitivity, and
several benchmark scripts.
"""

from __future__ import annotations

from typing import Any

import numpy as np

# ── Method sets ──────────────────────────────────────────────────────────

ALL_METHODS: tuple[str, ...] = (
    "hbac",
    "probe",
    "statistical",
    "geometric",
    "frequency",
    "bias_direction_pca",
    "sis",
    "demographic_parity",
    "equalized_odds",
    "intersectional",
    "groupdro",
    "gce",
    "ssa",
)

#: Methods that produce dimension-level importance scores.
DIM_SCORE_METHODS: frozenset[str] = frozenset(
    {"hbac", "probe", "statistical", "geometric", "bias_direction_pca", "sis"}
)

#: Methods that must be skipped on synthetic data (need 2+ attributes).
SKIP_IN_SYNTHETIC: frozenset[str] = frozenset({"intersectional"})

#: Tier 1 – embedding-native methods (only need embeddings + labels + groups).
EMBEDDING_METHODS: frozenset[str] = frozenset(
    {"hbac", "probe", "statistical", "geometric", "frequency", "bias_direction_pca", "sis"}
)

#: Tier 2 – fairness-based methods (need binary labels + protected_labels).
FAIRNESS_METHODS: frozenset[str] = frozenset(
    {"demographic_parity", "equalized_odds", "intersectional"}
)

#: Tier 3 – training-dynamics methods (train a model internally on embeddings).
TRAINING_METHODS: frozenset[str] = frozenset({"groupdro", "gce", "ssa"})


# ── Detection flag ───────────────────────────────────────────────────────


def method_flag(method: str, result: dict[str, Any]) -> bool:
    """Determine whether *method* flagged a shortcut in *result*.

    This is the single canonical implementation.  All benchmark modules
    and scripts should delegate to this function.
    """
    if not result.get("success", False):
        return False

    # -- original 4 methods ------------------------------------------------
    if method == "probe":
        return bool(result.get("results", {}).get("shortcut_detected", False))

    if method == "hbac":
        return bool(result.get("report", {}).get("has_shortcut", {}).get("exists", False))

    if method == "statistical":
        sig = result.get("significant_features", {})
        return any(v is not None and len(v) > 0 for v in sig.values())

    if method == "geometric":
        risk = str(result.get("summary", {}).get("risk_level", "low")).lower()
        return risk in {"moderate", "high"}

    # -- new embedding-native methods --------------------------------------
    if method == "frequency":
        report = result.get("report", {})
        if isinstance(report, dict):
            return bool(report.get("shortcut_detected", False))
        return False

    if method == "sis":
        return bool(result.get("shortcut_detected", False))

    # -- training-dynamics methods ------------------------------------------
    if method == "groupdro":
        report = result.get("report", {})
        if isinstance(report, dict):
            final = report.get("final", {})
            avg = final.get("avg_acc", 0)
            worst = final.get("worst_group_acc", 0)
            # Flag if worst-group accuracy is >10% below average
            return avg - worst > 0.10
        return False

    if method == "gce":
        report = result.get("report", {})
        if isinstance(report, dict):
            risk = str(report.get("risk_level", "low")).lower()
            return risk in {"moderate", "high"}
        return False

    if method == "ssa":
        return bool(result.get("shortcut_detected", False))

    # -- methods that use risk_level from a report dataclass ---------------
    if method in {
        "bias_direction_pca",
        "demographic_parity",
        "equalized_odds",
        "intersectional",
    }:
        # Prefer the standardised top-level risk_value set by apply_standardized_risk
        risk = str(result.get("risk_value") or "").lower()
        if risk in {"moderate", "high"}:
            return True
        # Fallback: report may be a dataclass with .risk_level attribute
        report = result.get("report")
        if report is not None and hasattr(report, "risk_level"):
            return str(report.risk_level).lower() in {"moderate", "high"}
        return False

    return False


# ── Convergence bucket ───────────────────────────────────────────────────


def convergence_bucket(n_flagged: int, n_methods: int) -> str:
    """Map the number of agreeing methods to a confidence bucket.

    This is a dynamic version that works with any number of methods.
    """
    if n_methods == 0:
        return "no_detection"
    if n_flagged == n_methods:
        return "high_confidence"
    if n_flagged >= n_methods - 1 and n_methods > 1:
        return "moderate_confidence"
    if n_flagged == 1:
        return "likely_false_alarm"
    if n_flagged == 0:
        return "no_detection"
    return "intermediate"


# ── Dimension-level scoring helpers for new methods ──────────────────────


def bias_direction_pca_dim_scores(result: dict[str, Any], embedding_dim: int) -> np.ndarray:
    """Extract per-dimension scores from bias_direction_pca result."""
    report = result.get("report")
    if report is not None and hasattr(report, "bias_direction"):
        direction = np.asarray(report.bias_direction, dtype=float)
        if direction.shape == (embedding_dim,):
            return np.abs(direction)
    return np.zeros(embedding_dim, dtype=float)


def sis_dim_scores(result: dict[str, Any], embedding_dim: int) -> np.ndarray:
    """Compute per-dimension importance from SIS membership frequency.

    Dimensions that appear more frequently in Sufficient Input Subsets
    are scored higher (indicating they are more important for prediction).
    """
    scores = np.zeros(embedding_dim, dtype=float)
    report = result.get("report", {})
    if isinstance(report, dict):
        sis_sizes = report.get("sis_sizes")
        if sis_sizes is not None:
            # SIS reports sizes but not which dims — use metrics instead
            pass

    # Try the detector object for detailed per-sample SIS indices
    detector = result.get("detector")
    if detector is not None and hasattr(detector, "sis_indices_per_sample_"):
        for indices in detector.sis_indices_per_sample_:
            for idx in indices:
                if 0 <= idx < embedding_dim:
                    scores[idx] += 1.0
        total = len(detector.sis_indices_per_sample_)
        if total > 0:
            scores /= total
        return scores

    return scores


def nan_dim_scores(embedding_dim: int) -> np.ndarray:
    """Return NaN scores for methods without dimension-level analysis."""
    return np.full(embedding_dim, np.nan, dtype=float)
