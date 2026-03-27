"""Bias direction detector using PCA on group prototypes.

Implements the classic embedding bias direction approach (Bolukbasi et al. 2016)
by extracting a principal component from group prototype vectors and measuring
projection gaps across groups.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Any

import numpy as np
from sklearn.decomposition import PCA

from ....detector_base import DetectorBase


@dataclass
class BiasDirectionPCAReport:
    """Container for PCA bias direction metrics."""

    bias_direction: np.ndarray
    explained_variance: float
    group_projections: dict[str, dict[str, float]]
    projection_gap: float
    reference: str
    risk_level: str
    notes: str


@dataclass(frozen=True)
class BiasDirectionPCAConfig:
    n_components: int = 1
    min_group_size: int = 10
    gap_threshold: float = 0.5


class BiasDirectionPCADetector(DetectorBase):
    """Compute bias direction from group prototypes via PCA."""

    def __init__(
        self,
        n_components: int = 1,
        min_group_size: int = 10,
        gap_threshold: float = 0.5,
        config: BiasDirectionPCAConfig | None = None,
    ) -> None:
        super().__init__(method="bias_direction_pca")
        cfg = config or BiasDirectionPCAConfig(
            n_components=int(n_components),
            min_group_size=int(min_group_size),
            gap_threshold=float(gap_threshold),
        )
        self.config = cfg

        self.n_components = cfg.n_components
        self.min_group_size = cfg.min_group_size
        self.gap_threshold = cfg.gap_threshold

        self.bias_direction_: np.ndarray | None = None
        self.explained_variance_: float = float("nan")
        self.group_projections_: dict[str, dict[str, float]] = {}
        self.projection_gap_: float = float("nan")
        self.report_: BiasDirectionPCAReport | None = None

    def fit(self, embeddings: np.ndarray, group_labels: np.ndarray) -> BiasDirectionPCADetector:
        if group_labels is None:
            raise ValueError("BiasDirectionPCADetector requires group_labels.")

        if embeddings.ndim != 2:
            raise ValueError("Embeddings must be 2D (n_samples, embedding_dim).")
        if group_labels.ndim != 1:
            raise ValueError("group_labels must be 1D.")
        if embeddings.shape[0] != group_labels.shape[0]:
            raise ValueError("Embeddings and group_labels must align.")

        prototypes, supports = self._compute_group_prototypes(embeddings, group_labels)
        if len(prototypes) < 2:
            raise ValueError("Need at least two valid groups to compute bias direction.")

        pca = PCA(n_components=self.n_components)
        pca.fit(np.vstack(list(prototypes.values())))
        direction = pca.components_[0]
        direction = direction / (np.linalg.norm(direction) + 1e-12)

        self.bias_direction_ = direction
        self.explained_variance_ = float(pca.explained_variance_ratio_[0])
        self.group_projections_ = self._compute_projections(prototypes, supports, direction)
        self.projection_gap_ = self._compute_gap()

        risk_level, notes = self._assess_risk()
        self.report_ = BiasDirectionPCAReport(
            bias_direction=self.bias_direction_,
            explained_variance=self.explained_variance_,
            group_projections=self.group_projections_,
            projection_gap=self.projection_gap_,
            reference="Bolukbasi et al. 2016",
            risk_level=risk_level,
            notes=notes,
        )
        self._finalize_results()
        self._is_fitted = True
        return self

    def _compute_group_prototypes(
        self, embeddings: np.ndarray, group_labels: np.ndarray
    ) -> tuple[dict[str, np.ndarray], dict[str, float]]:
        prototypes: dict[str, np.ndarray] = {}
        supports: dict[str, float] = {}
        for group in np.unique(group_labels):
            mask = group_labels == group
            support = float(mask.sum())
            if support < self.min_group_size:
                continue
            prototypes[str(group)] = embeddings[mask].mean(axis=0)
            supports[str(group)] = support
        return prototypes, supports

    def _compute_projections(
        self,
        prototypes: dict[str, np.ndarray],
        supports: dict[str, float],
        direction: np.ndarray,
    ) -> dict[str, dict[str, float]]:
        projections: dict[str, dict[str, float]] = {}
        for group, proto in prototypes.items():
            projections[group] = {
                "projection": float(np.dot(proto, direction)),
                "support": supports[group],
            }
        return projections

    def _compute_gap(self) -> float:
        values = [m["projection"] for m in self.group_projections_.values()]
        if len(values) < 2:
            return float("nan")
        return float(np.max(values) - np.min(values))

    def _assess_risk(self) -> tuple[str, str]:
        gap = self.projection_gap_
        if np.isnan(gap):
            return "unknown", "Insufficient data to assess bias direction."

        if gap >= 2 * self.gap_threshold:
            return "high", "Large bias-direction projection gap detected."
        if gap >= self.gap_threshold:
            return "moderate", "Moderate bias-direction projection gap detected."
        return "low", "Bias-direction gap within tolerance."

    def _finalize_results(self) -> None:
        risk_level = self.report_.risk_level if self.report_ else "unknown"
        if risk_level in {"high", "moderate"}:
            shortcut_detected = True
        elif risk_level == "low":
            shortcut_detected = False
        else:
            shortcut_detected = None

        self.shortcut_detected_ = shortcut_detected
        metrics = {
            "projection_gap": self.projection_gap_,
            "explained_variance": self.explained_variance_,
            "n_groups": len(self.group_projections_),
        }
        metadata = {
            "min_group_size": self.min_group_size,
            "n_components": self.n_components,
            "gap_threshold": self.gap_threshold,
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
