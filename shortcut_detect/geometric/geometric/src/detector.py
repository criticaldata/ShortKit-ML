"""
Geometric shortcut analysis inspired by universal embedding geometry.

Implements two complementary tests:
1. Bias direction analysis – measures how strongly group centroids separate along a shared
   geometric axis (difference vectors and projections).
2. Prototype subspace analysis – compares low-dimensional subspaces (principal components)
   for each group to quantify how much their local geometry overlaps.
"""

from __future__ import annotations

import itertools
from collections.abc import Sequence
from dataclasses import dataclass
from typing import Any

import numpy as np

from ....detector_base import DetectorBase, RiskLevel


@dataclass
class BiasPairStats:
    groups: tuple[str, str]
    direction_norm: float
    effect_size: float
    projection_gap: float
    alignment_score: float


@dataclass
class SubspacePairStats:
    groups: tuple[str, str]
    mean_cosine: float
    min_angle_deg: float
    max_angle_deg: float


class GeometricShortcutAnalyzer(DetectorBase):
    """
    Analyze shortcut risk using bias directions and prototype subspaces.

    Parameters:
        n_components: Number of principal directions per group.
        min_group_size: Minimum samples per group required for analysis.
        effect_threshold: Threshold on standardized projection gap for high risk.
        subspace_cosine_threshold: Threshold on mean subspace cosine for overlap risk.
    """

    def __init__(
        self,
        n_components: int = 5,
        min_group_size: int = 20,
        effect_threshold: float = 0.8,
        subspace_cosine_threshold: float = 0.85,
    ):
        super().__init__(method="geometric")

        self.n_components = n_components
        self.min_group_size = min_group_size
        self.effect_threshold = effect_threshold
        self.subspace_cosine_threshold = subspace_cosine_threshold

        self.group_stats_: dict[str, dict[str, np.ndarray]] = {}
        self.bias_pairs_: list[BiasPairStats] = []
        self.subspace_pairs_: list[SubspacePairStats] = []
        self.summary_: dict[str, str] = {}

    def fit(self, embeddings: np.ndarray, group_labels: Sequence) -> GeometricShortcutAnalyzer:
        """
        Fit analyzer on embeddings.

        Args:
            embeddings: (n_samples, embedding_dim) array.
            group_labels: Sequence of group identifiers aligned with embeddings.
        """
        X = np.asarray(embeddings)
        if X.ndim != 2:
            raise ValueError("embeddings must be 2D (n_samples, embedding_dim)")

        labels = np.asarray(group_labels)
        if labels.shape[0] != X.shape[0]:
            raise ValueError("embeddings and group_labels must have the same length")

        unique_groups = np.unique(labels)
        if unique_groups.shape[0] < 2:
            raise ValueError("At least two groups are required for geometric analysis")

        self.group_stats_ = self._compute_group_stats(X, labels, unique_groups)
        self.bias_pairs_ = self._compute_bias_pairs(X, labels)
        self.subspace_pairs_ = self._compute_subspace_pairs()
        self.summary_ = self._assess_risk()
        self._finalize_results()
        self._is_fitted = True
        return self

    def _compute_group_stats(
        self, X: np.ndarray, labels: np.ndarray, unique_groups: np.ndarray
    ) -> dict[str, dict[str, np.ndarray]]:
        stats = {}
        for group in unique_groups:
            mask = labels == group
            count = int(mask.sum())
            if count < self.min_group_size:
                continue
            group_embeddings = X[mask]
            centroid = group_embeddings.mean(axis=0)
            centered = group_embeddings - centroid
            u, s, vh = np.linalg.svd(centered, full_matrices=False)
            components = vh[: self.n_components]
            variance = (s[: self.n_components] ** 2).sum() / max(1e-9, (s**2).sum())
            stats[str(group)] = {
                "count": count,
                "centroid": centroid,
                "components": components,
                "variance_fraction": variance,
                "mask": mask,
                "projection_std": self._safe_std(centered),
            }

        if len(stats) < 2:
            raise ValueError(
                "Not enough groups satisfy min_group_size for geometric analysis. "
                f"Available groups: {len(stats)}"
            )
        return stats

    def _compute_bias_pairs(self, X: np.ndarray, labels: np.ndarray) -> list[BiasPairStats]:
        pairs: list[BiasPairStats] = []
        stats = self.group_stats_
        for g1, g2 in itertools.combinations(stats.keys(), 2):
            c1, c2 = stats[g1]["centroid"], stats[g2]["centroid"]
            direction = c2 - c1
            norm = np.linalg.norm(direction)
            if norm < 1e-9:
                continue
            dir_hat = direction / norm
            proj = X @ dir_hat
            mask1, mask2 = stats[g1]["mask"], stats[g2]["mask"]
            proj_mean1 = proj[mask1].mean()
            proj_mean2 = proj[mask2].mean()
            pooled_std = self._pooled_std(proj[mask1], proj[mask2])
            effect = np.abs(proj_mean2 - proj_mean1) / max(pooled_std, 1e-9)
            gap = proj_mean2 - proj_mean1
            align1 = self._alignment_with_components(dir_hat, stats[g1]["components"])
            align2 = self._alignment_with_components(dir_hat, stats[g2]["components"])
            alignment = float((align1 + align2) / 2.0)
            pairs.append(
                BiasPairStats(
                    groups=(g1, g2),
                    direction_norm=float(norm),
                    effect_size=float(effect),
                    projection_gap=float(gap),
                    alignment_score=alignment,
                )
            )
        return sorted(pairs, key=lambda x: x.effect_size, reverse=True)

    def _compute_subspace_pairs(self) -> list[SubspacePairStats]:
        pairs: list[SubspacePairStats] = []
        stats = self.group_stats_
        for g1, g2 in itertools.combinations(stats.keys(), 2):
            basis1, basis2 = stats[g1]["components"], stats[g2]["components"]
            cosines = self._principal_cosines(basis1, basis2)
            mean_cos = float(np.mean(cosines)) if cosines.size else 0.0
            min_angle = (
                float(np.degrees(np.arccos(np.clip(cosines.max(), -1, 1))))
                if cosines.size
                else 90.0
            )
            max_angle = (
                float(np.degrees(np.arccos(np.clip(cosines.min(), -1, 1))))
                if cosines.size
                else 90.0
            )
            pairs.append(
                SubspacePairStats(
                    groups=(g1, g2),
                    mean_cosine=mean_cos,
                    min_angle_deg=min_angle,
                    max_angle_deg=max_angle,
                )
            )
        return sorted(pairs, key=lambda x: x.mean_cosine, reverse=True)

    def _assess_risk(self) -> dict[str, Any]:
        high_effect = [p for p in self.bias_pairs_ if p.effect_size >= self.effect_threshold]
        overlapping_subspaces = [
            p for p in self.subspace_pairs_ if p.mean_cosine >= self.subspace_cosine_threshold
        ]

        if high_effect and overlapping_subspaces:
            risk = RiskLevel.HIGH
            msg = "Bias direction and prototype subspace analysis both indicate shortcuts."
        elif high_effect or overlapping_subspaces:
            risk = RiskLevel.MODERATE
            msg = "One geometric test indicates shortcut risk."
        else:
            risk = RiskLevel.LOW
            msg = "No strong geometric shortcut indicators detected."

        return {
            "risk_level": risk.value,
            "message": msg,
            "num_high_effect_pairs": len(high_effect),
            "num_overlap_pairs": len(overlapping_subspaces),
        }

    def _finalize_results(self) -> None:
        risk_enum = RiskLevel.from_string(self.summary_.get("risk_level"))
        risk_level = risk_enum.value
        if risk_enum in {RiskLevel.HIGH, RiskLevel.MODERATE}:
            shortcut_detected = True
        elif risk_enum == RiskLevel.LOW:
            shortcut_detected = False
        else:
            shortcut_detected = None

        self.shortcut_detected_ = shortcut_detected
        metrics = {
            "risk_level": risk_level,
            "num_high_effect_pairs": self.summary_.get("num_high_effect_pairs"),
            "num_overlap_pairs": self.summary_.get("num_overlap_pairs"),
        }
        metadata = {
            "n_groups": len(self.group_stats_),
            "n_components": self.n_components,
            "min_group_size": self.min_group_size,
            "effect_threshold": self.effect_threshold,
            "subspace_cosine_threshold": self.subspace_cosine_threshold,
        }
        report = {
            "summary": self.summary_,
            "bias_pairs": [p.__dict__ for p in self.bias_pairs_],
            "subspace_pairs": [p.__dict__ for p in self.subspace_pairs_],
        }
        self._set_results(
            shortcut_detected=shortcut_detected,
            risk_level=risk_level,
            metrics=metrics,
            notes=self.summary_.get("message", ""),
            metadata=metadata,
            report=report,
        )

    def get_report(self) -> dict[str, Any]:
        return super().get_report()

    @staticmethod
    def _safe_std(centered: np.ndarray) -> float:
        values = np.linalg.norm(centered, axis=1)
        return float(values.std(ddof=1)) if values.size > 1 else 0.0

    @staticmethod
    def _pooled_std(x: np.ndarray, y: np.ndarray) -> float:
        n1, n2 = len(x), len(y)
        if n1 < 2 or n2 < 2:
            return float(np.std(np.concatenate([x, y])))
        var1, var2 = np.var(x, ddof=1), np.var(y, ddof=1)
        pooled = ((n1 - 1) * var1 + (n2 - 1) * var2) / (n1 + n2 - 2)
        return float(np.sqrt(max(pooled, 1e-12)))

    @staticmethod
    def _alignment_with_components(direction: np.ndarray, components: np.ndarray) -> float:
        if components.size == 0:
            return 0.0
        dir_norm = direction / max(np.linalg.norm(direction), 1e-9)
        scores = np.abs(components @ dir_norm)
        return float(scores.max())

    @staticmethod
    def _principal_cosines(basis_a: np.ndarray, basis_b: np.ndarray) -> np.ndarray:
        if basis_a.size == 0 or basis_b.size == 0:
            return np.array([])
        # Ensure bases are orthonormal
        Ua, _ = np.linalg.qr(basis_a.T)
        Ub, _ = np.linalg.qr(basis_b.T)
        M = Ua.T @ Ub
        _, s, _ = np.linalg.svd(M, full_matrices=False)
        return np.clip(s, -1.0, 1.0)
