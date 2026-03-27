"""Builder for GradCAM mask overlap detector."""

import warnings
from typing import Any

import numpy as np

from ...base_builder import BaseDetector
from .src.detector import GradCAMMaskOverlapDetector


class GradCAMMaskOverlapDetectorBuilder(BaseDetector):
    def build(self):
        return GradCAMMaskOverlapDetector(
            threshold=self.kwargs.get("gradcam_overlap_threshold", 0.5),
            mask_threshold=self.kwargs.get("gradcam_mask_threshold", 0.5),
            batch_size=self.kwargs.get("gradcam_overlap_batch_size", 16),
        )

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
        raise ValueError(
            "gradcam_mask_overlap requires inputs/masks. Use ShortcutDetector.fit_from_loaders."
        )

    def run_from_loader(
        self,
        loader: Any,
        feature_names: list[str] | None = None,
        protected_labels: np.ndarray | None = None,
        splits: dict[str, np.ndarray] | None = None,
        extra_labels: dict[str, np.ndarray] | None = None,
    ) -> dict[str, Any]:
        data = loader() if callable(loader) else loader
        if not isinstance(data, dict):
            raise ValueError("Loader for gradcam_mask_overlap must return a dict.")

        detector = self.build()
        try:
            detector.fit(
                heatmaps=data.get("heatmaps"),
                masks=data.get("masks"),
                inputs=data.get("inputs"),
                model=data.get("model"),
                target_layer=data.get("target_layer"),
                head=data.get("head", "logits"),
                target_index=data.get("target_index"),
                batch_size=data.get("batch_size", detector.batch_size),
                heatmap_generator=data.get("heatmap_generator"),
            )
            report = detector.get_report()
            summary = report.get("metrics", {})
            summary_lines = [
                f"Samples: {summary.get('n_samples', 0)}",
                f"Attention-in-mask (mean): {summary.get('attention_in_mask_mean', 0.0):.3f}",
                f"Dice (mean): {summary.get('dice_mean', 0.0):.3f}",
                f"IoU (mean): {summary.get('iou_mean', 0.0):.3f}",
            ]
            return {
                "detector": detector,
                "report": report.get("report", {}),
                "summary_title": "GradCAM Attention vs. GT Masks",
                "summary_lines": summary_lines,
                "risk_indicators": [],
                "success": True,
                "metrics": report.get("metrics", {}),
                "metadata": report.get("metadata", {}),
                "details": report.get("details", {}),
            }
        except Exception as exc:
            warnings.warn(f"GradCAM mask overlap failed: {exc}", stacklevel=2)
            return {"success": False, "error": str(exc)}
