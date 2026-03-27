"""Early-epoch clustering detector for shortcut bias (SPARE, Yang et al. 2023)."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from sklearn.cluster import KMeans


@dataclass
class EarlyEpochClusteringReport:
    """Container for early-epoch clustering metrics."""

    n_epochs: int
    cluster_method: str
    n_clusters: int
    cluster_sizes: dict[str, int]
    cluster_ratios: dict[str, float]
    size_entropy: float
    minority_ratio: float
    largest_gap: float
    cluster_label_agreement: float | None
    risk_level: str
    notes: str
    reference: str


class EarlyEpochClusteringDetector:
    """Detect shortcut bias using early-epoch clustering (SPARE 2023)."""

    def __init__(
        self,
        n_clusters: int = 4,
        cluster_method: str = "kmeans",
        min_cluster_ratio: float = 0.1,
        entropy_threshold: float = 0.7,
        random_state: int = 42,
    ) -> None:
        self.n_clusters = n_clusters
        self.cluster_method = cluster_method
        self.min_cluster_ratio = min_cluster_ratio
        self.entropy_threshold = entropy_threshold
        self.random_state = random_state

        self.cluster_labels_: np.ndarray | None = None
        self.report_: EarlyEpochClusteringReport | None = None

    def fit(
        self,
        representations: np.ndarray,
        labels: np.ndarray | None = None,
        n_epochs: int = 1,
    ) -> EarlyEpochClusteringDetector:
        """Cluster early-epoch representations and compute bias indicators."""
        if representations is None:
            raise ValueError("representations must be provided for early-epoch clustering")
        if representations.ndim != 2:
            raise ValueError("representations must be 2D (n_samples, n_features)")

        n_samples = representations.shape[0]
        if n_samples < 2:
            raise ValueError("Need at least 2 samples for clustering")
        if self.n_clusters < 2:
            raise ValueError("n_clusters must be >= 2")
        if self.n_clusters > n_samples:
            raise ValueError("n_clusters cannot exceed number of samples")

        if labels is not None:
            if labels.ndim != 1:
                raise ValueError("labels must be 1D")
            if labels.shape[0] != n_samples:
                raise ValueError("labels must align with representations")

        if self.cluster_method != "kmeans":
            raise ValueError(f"Unsupported cluster_method: {self.cluster_method}")

        kmeans = KMeans(n_clusters=self.n_clusters, n_init=10, random_state=self.random_state)
        cluster_labels = kmeans.fit_predict(representations)
        self.cluster_labels_ = cluster_labels

        counts = np.bincount(cluster_labels, minlength=self.n_clusters).astype(float)
        ratios = counts / float(n_samples)

        entropy = _normalized_entropy(ratios)
        minority_ratio = (
            float(np.min(ratios) / np.max(ratios)) if np.max(ratios) > 0 else float("nan")
        )
        largest_gap = float(np.max(ratios) - np.min(ratios)) if ratios.size else float("nan")

        agreement = None
        if labels is not None:
            agreement = _cluster_label_agreement(cluster_labels, labels, self.n_clusters)

        risk_level, notes = _assess_risk(
            minority_ratio=minority_ratio,
            entropy=entropy,
            min_cluster_ratio=self.min_cluster_ratio,
            entropy_threshold=self.entropy_threshold,
        )

        self.report_ = EarlyEpochClusteringReport(
            n_epochs=n_epochs,
            cluster_method=self.cluster_method,
            n_clusters=self.n_clusters,
            cluster_sizes={str(i): int(counts[i]) for i in range(self.n_clusters)},
            cluster_ratios={str(i): float(ratios[i]) for i in range(self.n_clusters)},
            size_entropy=float(entropy),
            minority_ratio=float(minority_ratio),
            largest_gap=float(largest_gap),
            cluster_label_agreement=agreement,
            risk_level=risk_level,
            notes=notes,
            reference="Yang et al. 2023 (SPARE)",
        )
        return self


def _normalized_entropy(ratios: np.ndarray) -> float:
    if ratios.size == 0:
        return float("nan")
    k = ratios.size
    if k <= 1:
        return 0.0
    eps = 1e-12
    probs = np.clip(ratios, eps, 1.0)
    entropy = -np.sum(probs * np.log(probs))
    return float(entropy / np.log(k))


def _cluster_label_agreement(
    cluster_labels: np.ndarray,
    labels: np.ndarray,
    n_clusters: int,
) -> float:
    total = labels.shape[0]
    if total == 0:
        return float("nan")
    agreement_sum = 0.0
    for cluster_id in range(n_clusters):
        mask = cluster_labels == cluster_id
        if not np.any(mask):
            continue
        cluster_labels_vals = labels[mask]
        values, counts = np.unique(cluster_labels_vals, return_counts=True)
        majority = counts.max()
        agreement_sum += majority
    return float(agreement_sum / total)


def _assess_risk(
    minority_ratio: float,
    entropy: float,
    min_cluster_ratio: float,
    entropy_threshold: float,
) -> tuple[str, str]:
    if np.isnan(minority_ratio) or np.isnan(entropy):
        return "unknown", "Insufficient data to assess early-epoch clustering risk."

    if minority_ratio <= min_cluster_ratio / 2 or entropy <= entropy_threshold / 2:
        return "high", "Early-epoch clusters are highly imbalanced, suggesting shortcut bias."
    if minority_ratio <= min_cluster_ratio or entropy <= entropy_threshold:
        return "moderate", "Early-epoch clusters show imbalance; possible shortcut bias."
    return "low", "Cluster sizes are balanced; no strong shortcut signal detected."
