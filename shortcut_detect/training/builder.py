"""Builder for early-epoch clustering detector."""

import warnings
from typing import Any

import numpy as np

from ..base_builder import BaseDetector
from . import EarlyEpochClusteringDetector


class EarlyEpochClusteringDetectorBuilder(BaseDetector):
    def build(self):
        params = {
            "n_clusters": self.kwargs.get("eec_n_clusters", 4),
            "cluster_method": self.kwargs.get("eec_cluster_method", "kmeans"),
            "min_cluster_ratio": self.kwargs.get("eec_min_cluster_ratio", 0.1),
            "entropy_threshold": self.kwargs.get("eec_entropy_threshold", 0.7),
            "random_state": self.seed,
        }
        return EarlyEpochClusteringDetector(**params)

    def run(
        self,
        embeddings: np.ndarray,
        labels: np.ndarray,
        group_labels: np.ndarray,
        feature_names: list[str] | None,
        protected_labels: np.ndarray | None,
        splits: dict[str, np.ndarray] | None = None,
        extra_labels: dict[str, np.ndarray] | None = None,
    ) -> dict[str, Any]:
        print("Running early-epoch clustering (SPARE)...")
        detector = self.build()

        reps = None
        if extra_labels is not None:
            reps = extra_labels.get("early_epoch_reps")
        if reps is None:
            reps = embeddings

        try:
            n_epochs = self.kwargs.get("eec_n_epochs", 1)
            detector.fit(reps, labels=labels, n_epochs=n_epochs)
            report = detector.report_
            summary_lines = [
                f"Clusters: {report.n_clusters} | Method: {report.cluster_method}",
                f"Entropy: {report.size_entropy:.3f} | Minority ratio: {report.minority_ratio:.3f}",
                f"Largest gap: {report.largest_gap:.3f}",
            ]
            if report.cluster_label_agreement is not None:
                summary_lines.append(
                    f"Cluster label agreement: {report.cluster_label_agreement:.3f}"
                )
            risk_indicators = []
            if report.risk_level == "high":
                risk_indicators.append("Early-epoch clustering shows HIGH imbalance")
            elif report.risk_level == "moderate":
                risk_indicators.append("Early-epoch clustering shows moderate imbalance")
            return {
                "detector": detector,
                "report": report,
                "summary_title": "Early-Epoch Clustering (SPARE)",
                "summary_lines": summary_lines,
                "risk_indicators": risk_indicators,
                "success": True,
            }
        except Exception as exc:
            warnings.warn(f"Early-epoch clustering failed: {exc}", stacklevel=2)
            return {"success": False, "error": str(exc)}

    def run_from_loader(
        self,
        loader: Any,
        feature_names: list[str] | None = None,
        protected_labels: np.ndarray | None = None,
        splits: dict[str, np.ndarray] | None = None,
        extra_labels: dict[str, np.ndarray] | None = None,
    ) -> dict[str, Any]:
        data = loader() if callable(loader) else loader
        if isinstance(data, dict):
            reps = data.get("representations")
            if reps is None:
                reps = data.get("embeddings")
            labels = data.get("labels")
            if reps is None or labels is None:
                raise ValueError(
                    f"Loader for method '{self.method}' must provide 'representations' and 'labels'."
                )
            merged_extra = dict(extra_labels or {})
            merged_extra["early_epoch_reps"] = reps
            return self.run(
                embeddings=reps,
                labels=labels,
                group_labels=labels,
                feature_names=feature_names,
                protected_labels=protected_labels,
                splits=splits,
                extra_labels=merged_extra,
            )
        return super().run_from_loader(
            loader=loader,
            feature_names=feature_names,
            protected_labels=protected_labels,
            splits=splits,
            extra_labels=extra_labels,
        )
