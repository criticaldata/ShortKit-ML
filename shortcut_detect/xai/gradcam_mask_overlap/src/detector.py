"""GradCAM attention overlap with ground-truth masks."""

from __future__ import annotations

from collections.abc import Iterable, Sequence
from dataclasses import dataclass

import numpy as np
import torch

from ....detector_base import DetectorBase
from ....gradcam import GradCAMHeatmapGenerator

TensorOrArray = torch.Tensor | np.ndarray


@dataclass
class MaskOverlapSample:
    index: int
    attention_in_mask: float
    mask_coverage: float
    dice: float
    iou: float
    cosine: float


class GradCAMMaskOverlapDetector(DetectorBase):
    """Compute overlap between GradCAM attention maps and GT masks."""

    def __init__(
        self,
        *,
        threshold: float = 0.5,
        mask_threshold: float = 0.5,
        batch_size: int = 16,
        heatmap_generator: GradCAMHeatmapGenerator | None = None,
    ) -> None:
        super().__init__(method="gradcam_mask_overlap")
        self.threshold = float(threshold)
        self.mask_threshold = float(mask_threshold)
        self.batch_size = int(batch_size)
        self.heatmap_generator = heatmap_generator

        self.heatmaps_: np.ndarray | None = None
        self.masks_: np.ndarray | None = None
        self.sample_metrics_: list[MaskOverlapSample] = []

    def fit(
        self,
        *,
        heatmaps: np.ndarray | None = None,
        masks: TensorOrArray | None = None,
        inputs: TensorOrArray | None = None,
        model: torch.nn.Module | None = None,
        target_layer: str | torch.nn.Module | None = None,
        head: str | int = "logits",
        target_index: int | None = None,
        batch_size: int | None = None,
        heatmap_generator: GradCAMHeatmapGenerator | None = None,
    ) -> GradCAMMaskOverlapDetector:
        if masks is None:
            raise ValueError("masks are required for GradCAM mask overlap analysis.")

        if heatmaps is None:
            heatmaps = self._generate_heatmaps(
                inputs=inputs,
                model=model,
                target_layer=target_layer,
                head=head,
                target_index=target_index,
                batch_size=batch_size or self.batch_size,
                heatmap_generator=heatmap_generator or self.heatmap_generator,
            )
        heatmaps = self._validate_heatmaps(heatmaps)
        masks_arr = self._validate_masks(masks, heatmaps.shape[1:])

        self.heatmaps_ = heatmaps
        self.masks_ = masks_arr

        sample_metrics = self._compute_metrics(heatmaps, masks_arr)
        self.sample_metrics_ = sample_metrics

        summary = self._summarize(sample_metrics)
        metadata = {
            "n_samples": len(sample_metrics),
            "heatmap_shape": heatmaps.shape[1:],
            "mask_shape": masks_arr.shape[1:],
            "threshold": self.threshold,
            "mask_threshold": self.mask_threshold,
        }

        top_samples, bottom_samples = self._rank_samples(sample_metrics)
        report = {
            "summary": summary,
            "top_samples": top_samples,
            "bottom_samples": bottom_samples,
        }
        details = {
            "per_sample": [s.__dict__ for s in sample_metrics],
        }

        self._set_results(
            shortcut_detected=None,
            risk_level="unknown",
            metrics=summary,
            notes="GradCAM attention overlap with ground-truth masks.",
            metadata=metadata,
            report=report,
            details=details,
        )
        self._is_fitted = True
        return self

    def _generate_heatmaps(
        self,
        *,
        inputs: TensorOrArray | None,
        model: torch.nn.Module | None,
        target_layer: str | torch.nn.Module | None,
        head: str | int,
        target_index: int | None,
        batch_size: int,
        heatmap_generator: GradCAMHeatmapGenerator | None,
    ) -> np.ndarray:
        if inputs is None:
            raise ValueError("inputs are required to generate GradCAM heatmaps.")

        generator = heatmap_generator
        if generator is None:
            if model is None or target_layer is None:
                raise ValueError(
                    "Provide heatmap_generator or model + target_layer to generate heatmaps."
                )
            generator = GradCAMHeatmapGenerator(model, target_layer=target_layer)

        try:
            return self._generate_gradcam_heatmaps(
                generator, inputs, head, target_index, batch_size
            )
        finally:
            if heatmap_generator is None:
                generator.close()

    @staticmethod
    def _generate_gradcam_heatmaps(
        generator: GradCAMHeatmapGenerator,
        inputs: TensorOrArray,
        head: str | int,
        target_index: int | None,
        batch_size: int,
    ) -> np.ndarray:
        if isinstance(inputs, np.ndarray):
            tensor = torch.from_numpy(inputs)
        elif torch.is_tensor(inputs):
            tensor = inputs
        else:
            raise TypeError("inputs must be a torch.Tensor or np.ndarray")

        if tensor.ndim == 3:
            tensor = tensor.unsqueeze(0)
        if tensor.ndim != 4:
            raise ValueError("inputs must have shape (B,C,H,W) or (C,H,W)")

        heatmaps = []
        for i in range(0, tensor.shape[0], batch_size):
            batch = tensor[i : i + batch_size]
            heatmap = generator.generate_heatmap(batch, head=head, target_index=target_index)
            heatmaps.append(heatmap)
        return np.concatenate(heatmaps, axis=0)

    @staticmethod
    def _validate_heatmaps(heatmaps: np.ndarray) -> np.ndarray:
        arr = np.asarray(heatmaps, dtype=np.float32)
        if arr.ndim == 2:
            arr = arr[None, ...]
        if arr.ndim != 3:
            raise ValueError("heatmaps must have shape (N,H,W)")
        return arr

    @staticmethod
    def _validate_masks(masks: TensorOrArray, expected_shape: Sequence[int]) -> np.ndarray:
        arr = np.asarray(masks, dtype=np.float32)
        if arr.ndim == 2:
            arr = arr[None, ...]
        if arr.ndim != 3:
            raise ValueError("masks must have shape (N,H,W)")
        if arr.shape[1:] != tuple(expected_shape):
            raise ValueError(
                f"mask shape {arr.shape[1:]} does not match heatmap shape {tuple(expected_shape)}"
            )
        return arr

    def _compute_metrics(
        self,
        heatmaps: np.ndarray,
        masks: np.ndarray,
    ) -> list[MaskOverlapSample]:
        eps = 1e-8
        heatmaps = np.asarray(heatmaps, dtype=np.float32)
        masks = np.asarray(masks, dtype=np.float32)

        heat_bin = (heatmaps >= self.threshold).astype(np.float32)
        mask_bin = (masks >= self.mask_threshold).astype(np.float32)

        flat_heat = heatmaps.reshape(heatmaps.shape[0], -1)
        flat_mask = mask_bin.reshape(mask_bin.shape[0], -1)

        samples: list[MaskOverlapSample] = []
        for idx in range(heatmaps.shape[0]):
            hm = heatmaps[idx]
            hb = heat_bin[idx]
            mb = mask_bin[idx]

            intersection = float((hb * mb).sum())
            union = float((hb + mb - hb * mb).sum())
            dice_den = float(hb.sum() + mb.sum())

            dice = (2.0 * intersection / (dice_den + eps)) if dice_den > 0 else 0.0
            iou = (intersection / (union + eps)) if union > 0 else 0.0

            heat_sum = float(hm.sum())
            attention_in_mask = float((hm * mb).sum() / (heat_sum + eps)) if heat_sum > 0 else 0.0
            mask_sum = float(mb.sum())
            mask_coverage = float((hb * mb).sum() / (mask_sum + eps)) if mask_sum > 0 else 0.0

            d_vec = flat_heat[idx]
            m_vec = flat_mask[idx]
            d_norm = float(np.linalg.norm(d_vec))
            m_norm = float(np.linalg.norm(m_vec))
            cosine = (
                float(np.dot(d_vec, m_vec) / (d_norm * m_norm + eps))
                if d_norm > 0 and m_norm > 0
                else 0.0
            )

            samples.append(
                MaskOverlapSample(
                    index=idx,
                    attention_in_mask=attention_in_mask,
                    mask_coverage=mask_coverage,
                    dice=dice,
                    iou=iou,
                    cosine=cosine,
                )
            )

        return samples

    @staticmethod
    def _summarize(samples: Iterable[MaskOverlapSample]) -> dict[str, float]:
        samples_list = list(samples)
        if not samples_list:
            return {
                "n_samples": 0,
                "attention_in_mask_mean": 0.0,
                "attention_in_mask_median": 0.0,
                "mask_coverage_mean": 0.0,
                "dice_mean": 0.0,
                "iou_mean": 0.0,
                "cosine_mean": 0.0,
            }

        def _mean(values: list[float]) -> float:
            return float(np.mean(values)) if values else 0.0

        def _median(values: list[float]) -> float:
            return float(np.median(values)) if values else 0.0

        attention_vals = [s.attention_in_mask for s in samples_list]
        coverage_vals = [s.mask_coverage for s in samples_list]
        dice_vals = [s.dice for s in samples_list]
        iou_vals = [s.iou for s in samples_list]
        cosine_vals = [s.cosine for s in samples_list]

        return {
            "n_samples": len(samples_list),
            "attention_in_mask_mean": _mean(attention_vals),
            "attention_in_mask_median": _median(attention_vals),
            "mask_coverage_mean": _mean(coverage_vals),
            "dice_mean": _mean(dice_vals),
            "iou_mean": _mean(iou_vals),
            "cosine_mean": _mean(cosine_vals),
        }

    @staticmethod
    def _rank_samples(samples: list[MaskOverlapSample], top_k: int = 5):
        if not samples:
            return [], []
        ordered = sorted(samples, key=lambda s: s.attention_in_mask)
        bottom = ordered[:top_k]
        top = ordered[-top_k:][::-1]
        return [s.__dict__ for s in top], [s.__dict__ for s in bottom]
