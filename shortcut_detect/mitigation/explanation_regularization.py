"""
Explanation Regularization (M05 RRR) - Ross et al. 2017.

Right for Right Reasons: Penalize input gradients on shortcut regions during
training so the model relies on the right features.
"""

from __future__ import annotations

from typing import Any

import numpy as np
import torch
import torch.nn.functional as F


def _extract_logits(output: Any, head: str | int) -> torch.Tensor:
    """Extract logits tensor from model output."""
    if torch.is_tensor(output):
        return output
    if isinstance(output, list | tuple):
        idx = int(head) if isinstance(head, str | int) and str(head).lstrip("-").isdigit() else 0
        if -len(output) <= idx < len(output):
            return output[idx]
        return output[0]
    if isinstance(output, dict) and isinstance(head, str) and head in output:
        return output[head]
    raise ValueError(f"Cannot extract logits from output type {type(output)} with head={head}")


class ExplanationRegularization:
    """
    Right for Right Reasons (RRR) - Ross et al. 2017.

    Fine-tunes a model by penalizing input gradients on shortcut regions.
    Loss = L_task + lambda * sum(mask * (d log p(y|x)/dx)^2).

    Parameters
    ----------
    lambda_rrr : float
        Weight for the gradient penalty on shortcut regions.
    lr : float
        Learning rate for Adam optimizer.
    n_epochs : int
        Number of training epochs.
    batch_size : int
        Batch size for training.
    head : str or int
        How to extract logits from model output. "logits" or 0 for first output.
    device : str or torch.device, optional
        Device to train on.
    random_state : int, optional
        Seed for reproducibility.
    """

    def __init__(
        self,
        lambda_rrr: float = 1.0,
        lr: float = 1e-4,
        n_epochs: int = 10,
        batch_size: int = 8,
        head: str | int = "logits",
        device: str | torch.device | None = None,
        random_state: int | None = None,
    ):
        self.lambda_rrr = float(lambda_rrr)
        self.lr = float(lr)
        self.n_epochs = int(n_epochs)
        self.batch_size = int(batch_size)
        self.head = head
        self.device = device
        self.random_state = random_state

        self._device: torch.device = torch.device("cpu")
        self._history: list[dict] = []

    def _get_device(self) -> torch.device:
        if self.device is not None:
            return torch.device(self.device)
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")

    def fit(
        self,
        model: torch.nn.Module,
        images: torch.Tensor,
        labels: np.ndarray,
        shortcut_masks: np.ndarray,
    ) -> ExplanationRegularization:
        """
        Fine-tune model with RRR penalty. Model is updated in-place.

        Parameters
        ----------
        model : torch.nn.Module
            Differentiable model (e.g., CNN). Will be put in train mode.
        images : torch.Tensor
            Input images, shape (N, C, H, W). Will be moved to device.
        labels : np.ndarray
            Task labels, shape (N,), integer class indices.
        shortcut_masks : np.ndarray
            Masks where 1 = shortcut region to penalize. Shape (N, H, W) or (H, W).
            Will be resized to match input spatial size if needed.

        Returns
        -------
        self : ExplanationRegularization
        """
        dev = self._get_device()
        self._device = dev
        model = model.to(dev)
        model.train()

        if self.random_state is not None:
            torch.manual_seed(self.random_state)
            np.random.seed(self.random_state)

        images = images.to(dev).float()
        labels_t = torch.from_numpy(np.asarray(labels, dtype=np.int64)).to(dev)

        n = images.shape[0]
        if labels_t.shape[0] != n:
            raise ValueError(f"labels length {labels_t.shape[0]} must match images {n}")

        masks = np.asarray(shortcut_masks, dtype=np.float32)
        if masks.ndim == 2:
            masks = np.broadcast_to(masks, (n, masks.shape[0], masks.shape[1]))
        elif masks.shape[0] == 1 and n > 1:
            masks = np.broadcast_to(masks, (n, masks.shape[1], masks.shape[2])).copy()
        if masks.shape[0] != n:
            raise ValueError(f"shortcut_masks batch size {masks.shape[0]} must match images {n}")

        _, _, h_in, w_in = images.shape
        if masks.shape[1] != h_in or masks.shape[2] != w_in:
            masks_t = torch.from_numpy(masks).float().unsqueeze(1)
            masks_t = F.interpolate(
                masks_t, size=(h_in, w_in), mode="bilinear", align_corners=False
            )
            masks_t = masks_t.squeeze(1)
        else:
            masks_t = torch.from_numpy(masks).float().to(dev)

        optimizer = torch.optim.Adam(model.parameters(), lr=self.lr)
        self._history = []

        indices = np.arange(n)
        if self.random_state is not None:
            rng = np.random.default_rng(self.random_state)
        else:
            rng = np.random.default_rng()

        for epoch in range(self.n_epochs):
            rng.shuffle(indices)
            epoch_ce = 0.0
            epoch_penalty = 0.0
            n_batches = 0

            for start in range(0, n, self.batch_size):
                end = min(start + self.batch_size, n)
                batch_idx = indices[start:end]
                x = images[batch_idx].clone().detach().requires_grad_(True)
                y = labels_t[batch_idx]
                m = masks_t[batch_idx].to(dev)

                logits = model(x)
                logits = _extract_logits(logits, self.head)
                ce_loss = F.cross_entropy(logits, y)

                log_probs = F.log_softmax(logits, dim=1)
                sel = log_probs.gather(1, y.unsqueeze(1)).squeeze(1)
                grads = torch.autograd.grad(sel.sum(), x, create_graph=True, retain_graph=True)[0]
                if grads is None:
                    penalty = torch.tensor(0.0, device=dev)
                else:
                    m_expand = m.unsqueeze(1)
                    penalty = (m_expand * grads.pow(2)).sum()

                loss = ce_loss + self.lambda_rrr * penalty
                optimizer.zero_grad(set_to_none=True)
                loss.backward()
                optimizer.step()

                epoch_ce += ce_loss.item()
                epoch_penalty += penalty.item()
                n_batches += 1

            self._history.append(
                {
                    "epoch": epoch + 1,
                    "ce_loss": epoch_ce / max(n_batches, 1),
                    "penalty": epoch_penalty / max(n_batches, 1),
                }
            )

        return self
