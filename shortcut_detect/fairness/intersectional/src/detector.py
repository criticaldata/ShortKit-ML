"""Intersectional fairness detector for shortcut/fairness analysis.

Implements Buolamwini & Gebru (2018) intersectional analysis: evaluates fairness
metrics across intersections of demographic attributes (e.g., Black + Female,
White + Male). Single-attribute analysis can miss compounded disparities at
intersections.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Any

import numpy as np
from sklearn.base import clone
from sklearn.linear_model import LogisticRegression

from ....detector_base import DetectorBase, RiskLevel
from ...demographic_parity.src.detector import DemographicParityDetector, DemographicParityReport
from ...equalized_odds.src.detector import EqualizedOddsDetector, EqualizedOddsReport

# Reserved keys in extra_labels that are not demographic attributes
_RESERVED_EXTRA_LABELS = {"spurious", "early_epoch_reps"}


@dataclass
class IntersectionalReport:
    """Container for intersectional fairness metrics."""

    intersection_metrics: dict[str, dict[str, float]]
    attribute_names: list[str]
    tpr_gap: float
    fpr_gap: float
    dp_gap: float
    overall_accuracy: float
    overall_positive_rate: float
    reference: str
    risk_level: str
    notes: str


def _build_intersection_labels(
    extra_labels: dict[str, np.ndarray],
    attribute_names: list[str],
    separator: str = "_",
) -> tuple[np.ndarray, np.ndarray]:
    """Build intersection group labels from multiple attribute arrays.

    Returns:
        intersection_labels: str array of combined values (e.g., "Black_Female")
        valid_mask: bool array of samples with no missing values in any attribute
    """
    n = len(extra_labels[attribute_names[0]])
    valid_mask = np.ones(n, dtype=bool)

    parts: list[np.ndarray] = []
    for attr in attribute_names:
        arr = np.asarray(extra_labels[attr])
        if len(arr) != n:
            raise ValueError(f"extra_labels['{attr}'] has length {len(arr)}, expected {n}")
        # Convert to string for consistent joining
        str_arr = np.array([str(v) for v in arr])

        # Mask out missing values
        if np.issubdtype(arr.dtype, np.floating):
            valid_mask &= ~np.isnan(arr)
        else:
            # For object/string: exclude None, NaN, empty string
            invalid = np.array(
                [
                    v is None
                    or (isinstance(v, float) and np.isnan(v))
                    or (isinstance(v, str) and v.strip() == "")
                    for v in arr
                ]
            )
            valid_mask &= ~invalid

        parts.append(str_arr)

    # Build intersection labels
    intersection_labels = np.array(
        [separator.join(parts[i][j] for i in range(len(parts))).strip() for j in range(n)],
        dtype=object,
    )
    return intersection_labels, valid_mask


class IntersectionalDetector(DetectorBase):
    """Compute fairness metrics across intersections of demographic attributes."""

    def __init__(
        self,
        estimator: LogisticRegression | None = None,
        min_group_size: int = 10,
        tpr_gap_threshold: float = 0.1,
        fpr_gap_threshold: float = 0.1,
        dp_gap_threshold: float = 0.1,
        intersection_attributes: list[str] | None = None,
        separator: str = "_",
    ) -> None:
        super().__init__(method="intersectional")

        self.estimator = estimator or LogisticRegression(max_iter=1000)
        self.min_group_size = min_group_size
        self.tpr_gap_threshold = tpr_gap_threshold
        self.fpr_gap_threshold = fpr_gap_threshold
        self.dp_gap_threshold = dp_gap_threshold
        self.intersection_attributes = intersection_attributes
        self.separator = separator

        self.attribute_names_: list[str] = []
        self.intersection_metrics_: dict[str, dict[str, float]] = {}
        self.tpr_gap_: float = float("nan")
        self.fpr_gap_: float = float("nan")
        self.dp_gap_: float = float("nan")
        self.overall_accuracy_: float = float("nan")
        self.overall_positive_rate_: float = float("nan")
        self.report_: IntersectionalReport | None = None

    def fit(
        self,
        embeddings: np.ndarray,
        labels: np.ndarray,
        extra_labels: dict[str, np.ndarray],
    ) -> IntersectionalDetector:
        """Compute fairness metrics across demographic intersections."""
        if extra_labels is None or len(extra_labels) < 2:
            raise ValueError(
                "IntersectionalDetector requires extra_labels with at least "
                "2 demographic attribute arrays (e.g., {'race': ..., 'gender': ...})."
            )

        # Determine which attributes to use
        candidate_keys = [k for k in extra_labels.keys() if k not in _RESERVED_EXTRA_LABELS]
        if len(candidate_keys) < 2:
            raise ValueError(
                "Need at least 2 demographic attributes in extra_labels for "
                f"intersectional analysis. Found: {candidate_keys}."
            )

        if self.intersection_attributes is not None:
            attr_names = [a for a in self.intersection_attributes if a in extra_labels]
            if len(attr_names) < 2:
                raise ValueError(
                    f"intersection_attributes {self.intersection_attributes} "
                    f"must include at least 2 keys present in extra_labels: "
                    f"{list(extra_labels.keys())}."
                )
        else:
            attr_names = candidate_keys[:2]  # Use first 2

        self.attribute_names_ = attr_names

        if embeddings.ndim != 2:
            raise ValueError("Embeddings must be 2D (n_samples, embedding_dim).")
        if labels.ndim != 1:
            raise ValueError("Labels must be 1D.")
        if embeddings.shape[0] != labels.shape[0]:
            raise ValueError("Embeddings and labels must align.")

        unique_labels = np.unique(labels)
        if unique_labels.size != 2:
            raise ValueError("Intersectional analysis requires binary labels.")

        # Build intersection labels
        intersection_labels, valid_mask = _build_intersection_labels(
            extra_labels, attr_names, self.separator
        )

        # Count samples per intersection
        unique_intersections, counts = np.unique(
            intersection_labels[valid_mask], return_counts=True
        )
        large_groups = {
            u
            for u, c in zip(unique_intersections, counts, strict=False)
            if c >= self.min_group_size
        }

        if len(large_groups) < 2:
            self.shortcut_detected_ = None
            self.report_ = IntersectionalReport(
                intersection_metrics={},
                attribute_names=attr_names,
                tpr_gap=float("nan"),
                fpr_gap=float("nan"),
                dp_gap=float("nan"),
                overall_accuracy=float("nan"),
                overall_positive_rate=float("nan"),
                reference="Buolamwini & Gebru 2018",
                risk_level=RiskLevel.UNKNOWN.value,
                notes=(
                    f"Fewer than 2 intersection groups with >= {self.min_group_size} "
                    "samples. Cannot compute intersectional fairness metrics."
                ),
            )
            self._finalize_results()
            self._is_fitted = True
            return self

        # Build mask for samples in large groups only
        in_large = np.array([g in large_groups for g in intersection_labels]) & valid_mask

        X_sub = embeddings[in_large]
        y_sub = labels[in_large]
        groups_sub = intersection_labels[in_large]

        # Run EqualizedOddsDetector
        eo = EqualizedOddsDetector(
            estimator=clone(self.estimator),
            min_group_size=self.min_group_size,
            tpr_gap_threshold=self.tpr_gap_threshold,
            fpr_gap_threshold=self.fpr_gap_threshold,
        )
        eo.fit(X_sub, y_sub, groups_sub)

        # Run DemographicParityDetector
        dp = DemographicParityDetector(
            estimator=clone(self.estimator),
            min_group_size=self.min_group_size,
            dp_gap_threshold=self.dp_gap_threshold,
        )
        dp.fit(X_sub, y_sub, groups_sub)

        # Merge metrics into intersection_metrics
        eo_report: EqualizedOddsReport = eo.report_
        dp_report: DemographicParityReport = dp.report_

        self.intersection_metrics_ = {}
        for group in eo_report.group_metrics:
            eo_m = eo_report.group_metrics[group]
            dp_m = dp_report.group_rates.get(group, {})
            self.intersection_metrics_[group] = {
                "tpr": eo_m["tpr"],
                "fpr": eo_m["fpr"],
                "positive_rate": dp_m.get("positive_rate", float("nan")),
                "support": eo_m["support"],
            }

        self.tpr_gap_ = eo_report.tpr_gap
        self.fpr_gap_ = eo_report.fpr_gap
        self.dp_gap_ = dp_report.dp_gap
        self.overall_accuracy_ = eo_report.overall_accuracy
        self.overall_positive_rate_ = dp_report.overall_positive_rate

        risk_level, notes = self._assess_risk()
        self.report_ = IntersectionalReport(
            intersection_metrics=self.intersection_metrics_,
            attribute_names=self.attribute_names_,
            tpr_gap=self.tpr_gap_,
            fpr_gap=self.fpr_gap_,
            dp_gap=self.dp_gap_,
            overall_accuracy=self.overall_accuracy_,
            overall_positive_rate=self.overall_positive_rate_,
            reference="Buolamwini & Gebru 2018",
            risk_level=risk_level,
            notes=notes,
        )
        self._finalize_results()
        self._is_fitted = True
        return self

    def _assess_risk(self) -> tuple[str, str]:
        gaps = [self.tpr_gap_, self.fpr_gap_, self.dp_gap_]
        valid_gaps = [g for g in gaps if not np.isnan(g)]
        if not valid_gaps:
            return RiskLevel.UNKNOWN.value, "Insufficient data to assess intersectional fairness."

        thresh = max(
            self.tpr_gap_threshold,
            self.fpr_gap_threshold,
            self.dp_gap_threshold,
        )
        max_gap = max(valid_gaps)

        if max_gap >= 2 * thresh:
            return (
                RiskLevel.HIGH.value,
                "Large disparities detected across demographic intersections.",
            )
        if max_gap >= thresh:
            return (
                RiskLevel.MODERATE.value,
                "Moderate disparity detected in intersectional fairness metrics.",
            )
        return RiskLevel.LOW.value, "Intersectional fairness gaps within tolerance."

    def _finalize_results(self) -> None:
        risk_enum = RiskLevel.from_string(self.report_.risk_level if self.report_ else None)
        risk_level = risk_enum.value
        if self.report_ is not None:
            self.report_.risk_level = risk_level
        if risk_enum in {RiskLevel.HIGH, RiskLevel.MODERATE}:
            shortcut_detected = True
        elif risk_enum == RiskLevel.LOW:
            shortcut_detected = False
        else:
            shortcut_detected = None

        self.shortcut_detected_ = shortcut_detected
        metrics = {
            "tpr_gap": self.tpr_gap_,
            "fpr_gap": self.fpr_gap_,
            "dp_gap": self.dp_gap_,
            "overall_accuracy": self.overall_accuracy_,
            "overall_positive_rate": self.overall_positive_rate_,
        }
        metadata = {
            "min_group_size": self.min_group_size,
            "attribute_names": self.attribute_names_,
        }
        report = asdict(self.report_) if self.report_ else {}
        self._set_results(
            shortcut_detected=shortcut_detected,
            risk_level=risk_level,
            metrics=metrics,
            notes=self.report_.notes if self.report_ else "",
            metadata=metadata,
            report=report,
        )

    def get_report(self) -> dict[str, Any]:
        return super().get_report()
