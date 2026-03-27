"""SpRAy (Spectral Relevance Analysis) detector for heatmap clustering.

Based on: Lapuschkin et al., 2019 (Clever Hans Detection in Neural Networks).
"""

from __future__ import annotations

from collections.abc import Callable, Sequence
from dataclasses import dataclass
from typing import Any

import numpy as np
from scipy import ndimage
from sklearn.cluster import SpectralClustering
from sklearn.metrics import silhouette_score
from sklearn.metrics.pairwise import cosine_similarity, rbf_kernel
from sklearn.neighbors import kneighbors_graph

from ..detector_base import DetectorBase
from ..gradcam import GradCAMHeatmapGenerator

HeatmapArray = np.ndarray
HeatmapGenerator = Callable[[Any], HeatmapArray] | GradCAMHeatmapGenerator


@dataclass
class SpRAyClusterSummary:
    cluster_id: int
    size: int
    fraction: float
    label_purity: float | None
    group_purity: float | None
    focus_mean: float
    focus_std: float


class SpRAyDetector(DetectorBase):
    """Spectral clustering of explanation heatmaps for Clever Hans detection."""

    def __init__(
        self,
        *,
        n_clusters: int | None = None,
        cluster_selection: str = "auto",
        affinity: str = "cosine",
        nearest_neighbors: int = 10,
        rbf_gamma: float | None = None,
        min_clusters: int = 2,
        max_clusters: int = 10,
        downsample_size: int | tuple[int, int] | None = 32,
        random_state: int = 42,
        small_cluster_threshold: float = 0.05,
        purity_threshold: float = 0.8,
        focus_threshold: float = 0.6,
        separation_threshold: float = 0.3,
        alignment_threshold: float = 0.6,
        heatmap_generator: HeatmapGenerator | None = None,
    ) -> None:
        super().__init__(method="spray")
        self.n_clusters = n_clusters
        self.cluster_selection = cluster_selection
        self.affinity = affinity
        self.nearest_neighbors = nearest_neighbors
        self.rbf_gamma = rbf_gamma
        self.min_clusters = min_clusters
        self.max_clusters = max_clusters
        self.downsample_size = downsample_size
        self.random_state = random_state
        self.small_cluster_threshold = small_cluster_threshold
        self.purity_threshold = purity_threshold
        self.focus_threshold = focus_threshold
        self.separation_threshold = separation_threshold
        self.alignment_threshold = alignment_threshold
        self.heatmap_generator = heatmap_generator

        self.heatmaps_: np.ndarray | None = None
        self.cluster_labels_: np.ndarray | None = None
        self.cluster_summaries_: Sequence[SpRAyClusterSummary] = []
        self.representative_heatmaps_: dict[int, np.ndarray] = {}

    def fit(
        self,
        heatmaps: np.ndarray | None = None,
        *,
        labels: np.ndarray | None = None,
        group_labels: np.ndarray | None = None,
        inputs: Any | None = None,
        heatmap_generator: HeatmapGenerator | None = None,
        model: Any | None = None,
        target_layer: str | Any | None = None,
        head: str | int = "logits",
        target_index: int | None = None,
        batch_size: int = 16,
    ) -> SpRAyDetector:
        """Fit SpRAy detector on heatmaps or raw inputs.

        Parameters
        ----------
        heatmaps:
            Precomputed heatmaps shaped (N,H,W). If provided, inputs and model are ignored.
        labels:
            Optional task labels for cluster purity analysis.
        group_labels:
            Optional protected attribute labels for cluster purity analysis.
        inputs:
            Raw inputs to generate heatmaps from if heatmaps are not provided.
        heatmap_generator:
            Optional callable or GradCAMHeatmapGenerator used to produce heatmaps from inputs.
        model / target_layer:
            If provided (and heatmaps is None), a GradCAMHeatmapGenerator will be created internally.
        head / target_index:
            Parameters forwarded to GradCAM when generating heatmaps.
        batch_size:
            Batch size for heatmap generation when inputs are provided.
        """

        if heatmaps is None:
            heatmaps = self._generate_heatmaps(
                inputs=inputs,
                heatmap_generator=heatmap_generator,
                model=model,
                target_layer=target_layer,
                head=head,
                target_index=target_index,
                batch_size=batch_size,
            )

        heatmaps = self._validate_heatmaps(heatmaps)
        self.heatmaps_ = heatmaps
        n_samples = heatmaps.shape[0]

        labels_arr = np.asarray(labels) if labels is not None else None
        if labels_arr is not None and len(labels_arr) != n_samples:
            raise ValueError("labels length must match number of heatmaps")

        group_arr = np.asarray(group_labels) if group_labels is not None else None
        if group_arr is not None and len(group_arr) != n_samples:
            raise ValueError("group_labels length must match number of heatmaps")

        processed = self._preprocess_heatmaps(heatmaps)
        features = processed.reshape(n_samples, -1)

        affinity_matrix = None
        if self.affinity in {"cosine", "rbf"}:
            affinity_matrix = self._compute_affinity(features)

        if self.cluster_selection not in {"fixed", "eigengap", "auto"}:
            raise ValueError("cluster_selection must be 'fixed', 'eigengap', or 'auto'")

        n_clusters = self._resolve_cluster_count(
            features=features,
            affinity_matrix=affinity_matrix,
        )

        cluster_labels = self._cluster(features, affinity_matrix, n_clusters)

        summaries = self._summarize_clusters(
            cluster_labels,
            labels_arr,
            group_arr,
            processed,
        )
        representatives = self._compute_representative_heatmaps(processed, cluster_labels)

        clever_hans = self._evaluate_clever_hans(
            summaries=summaries,
            cluster_labels=cluster_labels,
            labels=labels_arr,
            features=features,
        )

        silhouette = self._compute_silhouette(features, cluster_labels)
        purity_values = [s.label_purity for s in summaries if s.label_purity is not None]
        mean_purity = float(np.mean(purity_values)) if purity_values else None
        max_purity = float(np.max(purity_values)) if purity_values else None
        focus_scores = [s.focus_mean for s in summaries]
        mean_focus = float(np.mean(focus_scores)) if focus_scores else None

        self.cluster_labels_ = cluster_labels
        self.cluster_summaries_ = summaries
        self.representative_heatmaps_ = representatives
        self.shortcut_detected_ = clever_hans["shortcut_detected"]

        metrics = {
            "n_clusters": int(n_clusters),
            "silhouette": silhouette,
            "mean_label_purity": mean_purity,
            "max_label_purity": max_purity,
            "mean_focus": mean_focus,
        }
        metadata = {
            "n_samples": int(n_samples),
            "heatmap_shape": heatmaps.shape[1:],
            "affinity": self.affinity,
            "cluster_selection": self.cluster_selection,
            "downsample_size": self.downsample_size,
        }
        report = {
            "clusters": [summary.__dict__ for summary in summaries],
            "clever_hans": clever_hans,
        }
        details = {
            "cluster_labels": cluster_labels,
            "representative_heatmaps": representatives,
        }

        risk_level = clever_hans.get("risk_level", "unknown")
        self._set_results(
            shortcut_detected=self.shortcut_detected_,
            risk_level=risk_level,
            metrics=metrics,
            notes="SpRAy spectral clustering on explanation heatmaps.",
            metadata=metadata,
            report=report,
            details=details,
        )

        self._is_fitted = True
        return self

    def _generate_heatmaps(
        self,
        *,
        inputs: Any | None,
        heatmap_generator: HeatmapGenerator | None,
        model: Any | None,
        target_layer: str | Any | None,
        head: str | int,
        target_index: int | None,
        batch_size: int,
    ) -> np.ndarray:
        if inputs is None:
            raise ValueError("Provide heatmaps or inputs to generate heatmaps.")

        generator = heatmap_generator or self.heatmap_generator

        if generator is None and model is not None and target_layer is not None:
            generator = GradCAMHeatmapGenerator(model, target_layer=target_layer)

        if generator is None:
            raise ValueError("No heatmap_generator or model/target_layer provided.")

        if isinstance(generator, GradCAMHeatmapGenerator):
            return self._generate_gradcam_heatmaps(
                generator, inputs, head, target_index, batch_size
            )

        if callable(generator):
            heatmaps = generator(inputs)
            return self._validate_heatmaps(heatmaps)

        raise TypeError("heatmap_generator must be callable or GradCAMHeatmapGenerator.")

    @staticmethod
    def _generate_gradcam_heatmaps(
        generator: GradCAMHeatmapGenerator,
        inputs: Any,
        head: str | int,
        target_index: int | None,
        batch_size: int,
    ) -> np.ndarray:
        if isinstance(inputs, list | tuple):
            if not inputs:
                raise ValueError("inputs list is empty.")
            first = inputs[0]
            if hasattr(first, "shape"):
                if hasattr(first, "dtype") and hasattr(first, "device"):
                    inputs = np.stack(
                        [
                            inp.detach().cpu().numpy() if hasattr(inp, "detach") else inp
                            for inp in inputs
                        ]
                    )
                else:
                    inputs = np.stack(inputs)
            else:
                raise TypeError("inputs list elements must be array-like or tensors.")

        if not hasattr(inputs, "__len__"):
            return generator.generate_heatmap(inputs, head=head, target_index=target_index)

        if hasattr(inputs, "shape"):
            try:
                ndim = inputs.ndim
            except AttributeError:
                ndim = inputs.dim()
            if ndim == 3:
                return generator.generate_heatmap(inputs, head=head, target_index=target_index)

        heatmaps: list[np.ndarray] = []
        n_total = len(inputs)
        for start in range(0, n_total, batch_size):
            batch = inputs[start : start + batch_size]
            heatmaps.append(generator.generate_heatmap(batch, head=head, target_index=target_index))
        return np.vstack(heatmaps)

    @staticmethod
    def _validate_heatmaps(heatmaps: np.ndarray) -> np.ndarray:
        if not isinstance(heatmaps, np.ndarray):
            heatmaps = np.asarray(heatmaps)
        if heatmaps.ndim == 4 and heatmaps.shape[-1] == 1:
            heatmaps = heatmaps.squeeze(-1)
        if heatmaps.ndim != 3:
            raise ValueError("Heatmaps must have shape (N,H,W).")
        return heatmaps.astype(np.float32)

    def _preprocess_heatmaps(self, heatmaps: np.ndarray) -> np.ndarray:
        processed = heatmaps.copy()
        if self.downsample_size is not None:
            processed = self._downsample(processed, self.downsample_size)
        processed = self._normalize(processed)
        return processed

    @staticmethod
    def _normalize(heatmaps: np.ndarray) -> np.ndarray:
        normed = heatmaps.copy()
        for idx, sample in enumerate(normed):
            sample = sample - np.min(sample)
            denom = np.max(sample) + 1e-8
            sample = sample / denom if denom > 0 else sample
            normed[idx] = sample
        return normed

    @staticmethod
    def _downsample(heatmaps: np.ndarray, size: int | tuple[int, int]) -> np.ndarray:
        if isinstance(size, int):
            target_h = target_w = size
        else:
            target_h, target_w = size
        n_samples, h, w = heatmaps.shape
        if (h, w) == (target_h, target_w):
            return heatmaps
        zoom_h = target_h / h
        zoom_w = target_w / w
        downsampled = np.stack(
            [ndimage.zoom(sample, (zoom_h, zoom_w), order=1) for sample in heatmaps]
        )
        return downsampled

    def _compute_affinity(self, features: np.ndarray) -> np.ndarray:
        if self.affinity == "cosine":
            return cosine_similarity(features)
        if self.affinity == "rbf":
            return rbf_kernel(features, gamma=self.rbf_gamma)
        raise ValueError("Affinity must be 'cosine' or 'rbf' for precomputed affinity.")

    def _resolve_cluster_count(
        self,
        *,
        features: np.ndarray,
        affinity_matrix: np.ndarray | None,
    ) -> int:
        if self.cluster_selection == "fixed":
            if self.n_clusters is None:
                raise ValueError("n_clusters must be provided when cluster_selection='fixed'.")
            return int(self.n_clusters)

        if self.n_clusters is not None:
            return int(self.n_clusters)

        if self.cluster_selection in {"eigengap", "auto"}:
            affinity = affinity_matrix
            if affinity is None:
                affinity = self._build_neighbor_affinity(features)
            return self._select_by_eigengap(affinity)

        return max(self.min_clusters, 2)

    def _build_neighbor_affinity(self, features: np.ndarray) -> np.ndarray:
        graph = kneighbors_graph(
            features,
            n_neighbors=min(self.nearest_neighbors, len(features) - 1),
            mode="connectivity",
            include_self=True,
        )
        affinity = graph.toarray()
        affinity = np.maximum(affinity, affinity.T)
        return affinity

    def _select_by_eigengap(self, affinity: np.ndarray) -> int:
        degrees = np.sum(affinity, axis=1)
        degree_inv_sqrt = np.diag(1.0 / np.sqrt(np.maximum(degrees, 1e-8)))
        laplacian = np.eye(affinity.shape[0]) - degree_inv_sqrt @ affinity @ degree_inv_sqrt

        max_k = min(self.max_clusters + 1, affinity.shape[0] - 1)
        eigvals = np.linalg.eigvalsh(laplacian)
        eigvals = np.sort(eigvals)[: max_k + 1]
        gaps = np.diff(eigvals[: max_k + 1])
        start = max(self.min_clusters, 2) - 1
        if start >= len(gaps):
            return max(self.min_clusters, 2)
        best = int(np.argmax(gaps[start:]) + start + 1)
        return max(best, self.min_clusters)

    def _cluster(
        self,
        features: np.ndarray,
        affinity_matrix: np.ndarray | None,
        n_clusters: int,
    ) -> np.ndarray:
        if self.affinity in {"cosine", "rbf"}:
            if affinity_matrix is None:
                affinity_matrix = self._compute_affinity(features)
            clustering = SpectralClustering(
                n_clusters=n_clusters,
                affinity="precomputed",
                assign_labels="kmeans",
                random_state=self.random_state,
            )
            return clustering.fit_predict(affinity_matrix)

        if self.affinity == "nearest_neighbors":
            clustering = SpectralClustering(
                n_clusters=n_clusters,
                affinity="nearest_neighbors",
                n_neighbors=min(self.nearest_neighbors, len(features) - 1),
                assign_labels="kmeans",
                random_state=self.random_state,
            )
            return clustering.fit_predict(features)

        raise ValueError("Unsupported affinity; use 'cosine', 'rbf', or 'nearest_neighbors'.")

    @staticmethod
    def _compute_focus_scores(heatmaps: np.ndarray) -> np.ndarray:
        n_samples = heatmaps.shape[0]
        flat = heatmaps.reshape(n_samples, -1)
        flat = np.maximum(flat, 0.0)
        sums = flat.sum(axis=1, keepdims=True) + 1e-8
        probs = flat / sums
        entropy = -np.sum(probs * np.log(probs + 1e-8), axis=1)
        max_entropy = np.log(probs.shape[1])
        focus = 1.0 - entropy / max_entropy if max_entropy > 0 else np.zeros_like(entropy)
        return np.clip(focus, 0.0, 1.0)

    def _summarize_clusters(
        self,
        cluster_labels: np.ndarray,
        labels: np.ndarray | None,
        group_labels: np.ndarray | None,
        heatmaps: np.ndarray,
    ) -> Sequence[SpRAyClusterSummary]:
        n_samples = len(cluster_labels)
        focus_scores = self._compute_focus_scores(heatmaps)

        summaries = []
        for cluster_id in np.unique(cluster_labels):
            idx = np.where(cluster_labels == cluster_id)[0]
            size = len(idx)
            fraction = size / n_samples if n_samples else 0.0

            label_purity = None
            if labels is not None:
                _, counts = np.unique(labels[idx], return_counts=True)
                label_purity = float(np.max(counts) / size) if size else None

            group_purity = None
            if group_labels is not None:
                _, counts = np.unique(group_labels[idx], return_counts=True)
                group_purity = float(np.max(counts) / size) if size else None

            focus_mean = float(np.mean(focus_scores[idx])) if size else 0.0
            focus_std = float(np.std(focus_scores[idx])) if size else 0.0

            summaries.append(
                SpRAyClusterSummary(
                    cluster_id=int(cluster_id),
                    size=int(size),
                    fraction=float(fraction),
                    label_purity=label_purity,
                    group_purity=group_purity,
                    focus_mean=focus_mean,
                    focus_std=focus_std,
                )
            )

        return summaries

    @staticmethod
    def _compute_representative_heatmaps(
        heatmaps: np.ndarray,
        cluster_labels: np.ndarray,
    ) -> dict[int, np.ndarray]:
        representatives = {}
        for cluster_id in np.unique(cluster_labels):
            idx = np.where(cluster_labels == cluster_id)[0]
            if len(idx) == 0:
                continue
            representatives[int(cluster_id)] = np.mean(heatmaps[idx], axis=0)
        return representatives

    def _evaluate_clever_hans(
        self,
        *,
        summaries: Sequence[SpRAyClusterSummary],
        cluster_labels: np.ndarray,
        labels: np.ndarray | None,
        features: np.ndarray,
    ) -> dict[str, Any]:
        flags = []

        for summary in summaries:
            if summary.fraction < self.small_cluster_threshold and summary.label_purity is not None:
                if summary.label_purity >= self.purity_threshold:
                    flags.append("small_high_purity_cluster")
                    break

        max_focus = max((s.focus_mean for s in summaries), default=0.0)
        max_purity = max((s.label_purity or 0.0 for s in summaries), default=0.0)
        if max_focus >= self.focus_threshold and max_purity >= self.purity_threshold:
            flags.append("highly_localized_attention")

        if labels is not None:
            avg_purity = np.mean([s.label_purity for s in summaries if s.label_purity is not None])
            silhouette = self._compute_silhouette(features, cluster_labels)
            if silhouette is not None and silhouette >= self.separation_threshold:
                if avg_purity < self.alignment_threshold:
                    flags.append("separation_without_label_alignment")

        shortcut_detected = len(flags) > 0
        if len(flags) >= 2:
            risk_level = "high"
        elif len(flags) == 1:
            risk_level = "moderate"
        else:
            risk_level = "low"

        return {
            "shortcut_detected": shortcut_detected,
            "risk_level": risk_level,
            "flags": flags,
        }

    def _compute_silhouette(
        self,
        features: np.ndarray,
        cluster_labels: np.ndarray,
    ) -> float | None:
        if len(np.unique(cluster_labels)) < 2:
            return None
        if len(cluster_labels) <= len(np.unique(cluster_labels)):
            return None
        metric = "cosine" if self.affinity == "cosine" else "euclidean"
        try:
            return float(silhouette_score(features, cluster_labels, metric=metric))
        except Exception:
            return None
