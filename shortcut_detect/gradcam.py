"""GradCAM-based attention visualization utilities.

This module provides a lightweight GradCAM generator tailored for
shortcut analysis workflows.  The goal is to compare where a model
focuses when predicting a disease label versus a protected attribute
label so we can reason about attention overlap.
"""

from __future__ import annotations

from collections.abc import Callable, Sequence
from dataclasses import dataclass
from typing import Any

import numpy as np
import torch
import torch.nn.functional as F

TensorOrArray = torch.Tensor | np.ndarray


@dataclass
class AttentionOverlapResult:
    """Container with disease/attribute heatmaps and overlap metrics."""

    disease_heatmap: np.ndarray
    attribute_heatmap: np.ndarray
    overlap_score: float
    metrics: dict[str, float]


class GradCAMHeatmapGenerator:
    """Compute GradCAM heatmaps for multiple prediction heads.

    Parameters
    ----------
    model:
        ``torch.nn.Module`` that produces the predictions.  The module
        is always used in ``eval`` mode and gradients are enabled only
        for the current forward pass.
    target_layer:
        Layer whose activations will be used for GradCAM.  Can be the
        actual ``nn.Module`` instance or the dotted path to the module.
    head_mappings:
        Optional mapping from head names to either an index (for tuple
        outputs) or a callable ``fn(output) -> Tensor``.  Defaults to a
        mapping that treats ``("disease", "attribute")`` as the first
        two entries of a tuple/list output.
    device:
        Device to run inference on.  Defaults to the model's first
        parameter device or CPU.
    """

    def __init__(
        self,
        model: torch.nn.Module,
        target_layer: str | torch.nn.Module,
        head_mappings: dict[str, int | Callable[[Any], torch.Tensor]] | None = None,
        device: torch.device | None = None,
    ) -> None:
        self.model = model.eval()
        self.device = device or self._infer_device()
        self.model.to(self.device)

        self._target_layer = self._resolve_target_layer(target_layer)
        self._activations: torch.Tensor | None = None
        self._gradients: torch.Tensor | None = None
        self._handles: Sequence[torch.utils.hooks.RemovableHandle] = []

        default_mappings: dict[str, int | Callable[[Any], torch.Tensor]] = {
            "logits": None,
        }
        if head_mappings:
            default_mappings.update(head_mappings)
        self.head_mappings = default_mappings

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------
    def generate_heatmap(
        self,
        inputs: TensorOrArray,
        head: str | int,
        target_index: int | None = None,
    ) -> np.ndarray:
        """Generate GradCAM heatmap for a single prediction head.

        Parameters
        ----------
        inputs:
            Input image tensor/array shaped ``(B,C,H,W)`` or ``(C,H,W)``.
        head:
            Identifier for the prediction head.  Can be a string key,
            integer index, or alias defined in ``head_mappings``.
        target_index:
            Target class index.  If ``None`` the argmax of the head's
            logits for each sample is used.
        """

        inputs_tensor = self._prepare_inputs(inputs)
        with torch.enable_grad():
            return self._run_gradcam(inputs_tensor, head, target_index)

    def generate_attention_overlap(
        self,
        inputs: TensorOrArray,
        disease_target: int | None = None,
        attribute_target: int | None = None,
        disease_head: str | int = "disease",
        attribute_head: str | int = "attribute",
        threshold: float = 0.5,
    ) -> AttentionOverlapResult:
        """Generate disease/attribute heatmaps and overlap metrics."""

        disease_heatmap = self.generate_heatmap(inputs, disease_head, disease_target)
        attribute_heatmap = self.generate_heatmap(inputs, attribute_head, attribute_target)
        metrics = self.calculate_overlap(disease_heatmap, attribute_heatmap, threshold)
        overlap = metrics.get("dice", 0.0)
        return AttentionOverlapResult(
            disease_heatmap=disease_heatmap,
            attribute_heatmap=attribute_heatmap,
            overlap_score=overlap,
            metrics=metrics,
        )

    @staticmethod
    def calculate_overlap(
        disease_heatmap: np.ndarray,
        attribute_heatmap: np.ndarray,
        threshold: float = 0.5,
    ) -> dict[str, float]:
        """Compute overlap metrics between two normalized heatmaps."""

        dh = np.asarray(disease_heatmap, dtype=np.float32)
        ah = np.asarray(attribute_heatmap, dtype=np.float32)
        if dh.shape != ah.shape:
            raise ValueError("Heatmaps must share the same shape to compute overlap")

        if dh.ndim == 2:
            dh = dh[None, ...]
            ah = ah[None, ...]

        eps = 1e-8
        dh_bin = (dh >= threshold).astype(np.float32)
        ah_bin = (ah >= threshold).astype(np.float32)

        intersection = (dh_bin * ah_bin).sum(axis=(1, 2))
        union = (dh_bin + ah_bin - dh_bin * ah_bin).sum(axis=(1, 2))
        dice_den = dh_bin.sum(axis=(1, 2)) + ah_bin.sum(axis=(1, 2)) + eps

        dice = (2.0 * intersection / dice_den).mean().item() if dice_den.size else 0.0
        iou = (intersection / (union + eps)).mean().item() if union.size else 0.0

        flat_d = dh.reshape(dh.shape[0], -1)
        flat_a = ah.reshape(ah.shape[0], -1)
        cosines = []
        for d_vec, a_vec in zip(flat_d, flat_a, strict=False):
            d_norm = np.linalg.norm(d_vec)
            a_norm = np.linalg.norm(a_vec)
            if d_norm < eps or a_norm < eps:
                cosines.append(0.0)
            else:
                cosines.append(float(np.dot(d_vec, a_vec) / (d_norm * a_norm)))
        cosine = float(np.mean(cosines)) if cosines else 0.0

        return {"dice": float(dice), "iou": float(iou), "cosine": cosine}

    # ------------------------------------------------------------------
    # Internals
    # ------------------------------------------------------------------
    def _run_gradcam(
        self,
        inputs: torch.Tensor,
        head: str | int,
        target_index: int | None,
    ) -> np.ndarray:
        self._attach_hooks()
        try:
            outputs = self.model(inputs)
            head_tensor = self._extract_head_output(outputs, head)
            if head_tensor.dim() < 2:
                raise ValueError(
                    f"Head '{head}' must return a tensor with batch + class dimensions"
                )

            if target_index is None:
                target_idx = head_tensor.argmax(dim=1)
            else:
                target_idx = torch.full(
                    (head_tensor.shape[0],),
                    target_index,
                    device=head_tensor.device,
                    dtype=torch.long,
                )

            selected = head_tensor[torch.arange(head_tensor.shape[0]), target_idx]

            self.model.zero_grad(set_to_none=True)
            selected.sum().backward(retain_graph=False)

            if self._gradients is None or self._activations is None:
                raise RuntimeError(
                    "Gradients or activations were not captured. Check target_layer."
                )

            gradients = self._gradients
            activations = self._activations
            if gradients.shape[0] != inputs.shape[0]:
                raise RuntimeError("Batch size mismatch between gradients and inputs")

            weights = gradients.mean(dim=(2, 3), keepdim=True)
            cam = (weights * activations).sum(dim=1, keepdim=True)
            cam = F.relu(cam)
            cam = F.interpolate(cam, size=inputs.shape[-2:], mode="bilinear", align_corners=False)
            cam = cam.squeeze(1)

            heatmaps = []
            for sample in cam:
                sample_np = sample.detach().cpu().numpy()
                sample_np -= sample_np.min()
                denom = sample_np.max() + 1e-8
                sample_np = sample_np / denom if denom > 0 else sample_np
                heatmaps.append(sample_np)
            return np.stack(heatmaps)
        finally:
            self._detach_hooks()
            self._activations = None
            self._gradients = None

    def _prepare_inputs(self, inputs: TensorOrArray) -> torch.Tensor:
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

        tensor = tensor.to(self.device).float()
        tensor.requires_grad_(True)
        return tensor

    def _extract_head_output(self, outputs: Any, head: str | int) -> torch.Tensor:
        if isinstance(head, str) and head in self.head_mappings:
            mapping = self.head_mappings[head]
            return self._apply_mapping(outputs, mapping, head)

        if isinstance(head, int):
            return self._extract_by_index(outputs, head, head)

        if isinstance(outputs, dict) and isinstance(head, str) and head in outputs:
            return self._validate_tensor(outputs[head], head)

        if torch.is_tensor(outputs):
            if head in ("logits", None):
                return outputs
            raise ValueError(
                f"Model returned a tensor, but head '{head}' is unknown. Provide a head_mapping."
            )

        if isinstance(outputs, list | tuple):
            if isinstance(head, str):
                try:
                    idx = int(head)
                except (TypeError, ValueError) as exc:
                    raise ValueError(
                        f"Cannot infer output index for head '{head}'. Provide head_mappings."
                    ) from exc
            else:
                idx = head
            return self._validate_tensor(outputs[idx], head)

        raise ValueError(f"Unsupported model output type for head '{head}': {type(outputs)}")

    def _apply_mapping(self, outputs: Any, mapping: Any, head_name: str) -> torch.Tensor:
        if callable(mapping):
            head_tensor = mapping(outputs)
            return self._validate_tensor(head_tensor, head_name)
        if mapping is None:
            if torch.is_tensor(outputs):
                return outputs
            raise ValueError(
                f"Head '{head_name}' mapped to None but model output is not a tensor. Provide a callable."
            )
        if isinstance(mapping, int):
            return self._extract_by_index(outputs, mapping, head_name)
        if isinstance(mapping, str):
            return self._extract_head_output(outputs, mapping)
        return self._validate_tensor(mapping, head_name)

    def _extract_by_index(self, outputs: Any, index: int, head_name: str | int) -> torch.Tensor:
        if isinstance(outputs, list | tuple):
            if -len(outputs) <= index < len(outputs):
                return self._validate_tensor(outputs[index], head_name)
            raise IndexError(f"Head index {index} out of range for output of size {len(outputs)}")
        raise ValueError(
            f"Head '{head_name}' mapped to index {index}, but model output is not a list/tuple"
        )

    def _validate_tensor(self, tensor: Any, head: str) -> torch.Tensor:
        if not torch.is_tensor(tensor):
            raise TypeError(f"Output for head '{head}' must be a torch.Tensor, got {type(tensor)}")
        if tensor.dim() < 2:
            raise ValueError(
                f"Head '{head}' tensor must include batch dimension. Received shape {tuple(tensor.shape)}"
            )
        return tensor

    def _resolve_target_layer(self, target: str | torch.nn.Module) -> torch.nn.Module:
        if isinstance(target, torch.nn.Module):
            return target
        if isinstance(target, str):
            module: torch.nn.Module = self.model
            for attr in target.split("."):
                if not hasattr(module, attr):
                    raise AttributeError(f"Model has no layer '{target}'")
                module = getattr(module, attr)
            if not isinstance(module, torch.nn.Module):
                raise AttributeError(f"Target '{target}' is not a torch.nn.Module")
            return module
        raise TypeError("target_layer must be a module or dotted path string")

    def _attach_hooks(self) -> None:
        self._handles = [
            self._target_layer.register_forward_hook(self._save_activation),
            self._target_layer.register_full_backward_hook(self._save_gradient),
        ]

    def _detach_hooks(self) -> None:
        for handle in self._handles:
            handle.remove()
        self._handles = []

    def _save_activation(
        self, module: torch.nn.Module, inputs: tuple[torch.Tensor, ...], output: torch.Tensor
    ) -> None:
        self._activations = output.detach()

    def _save_gradient(
        self,
        module: torch.nn.Module,
        grad_input: tuple[torch.Tensor, ...],
        grad_output: tuple[torch.Tensor, ...],
    ) -> None:
        self._gradients = grad_output[0].detach()

    def _infer_device(self) -> torch.device:
        try:
            first_param = next(self.model.parameters())
            return first_param.device
        except StopIteration:
            return torch.device("cpu")

    def close(self) -> None:
        """Remove any lingering hooks (safe to call multiple times)."""

        self._detach_hooks()

    def __del__(self) -> None:  # pragma: no cover - best effort cleanup
        try:
            self.close()
        except Exception:
            pass
