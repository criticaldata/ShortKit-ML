"""
VAE-based shortcut detection (Müller et al., Fraunhofer-AISEC).

Uses Beta-VAE disentanglement to identify latent dimensions with high predictiveness
for the target label. High-predictiveness dimensions indicate candidate shortcuts.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np
import torch
from torch.utils.data import DataLoader, TensorDataset

from ..detector_base import DetectorBase
from .latent_analyzer import (
    compute_mpwd_per_dimension,
    compute_predictiveness_per_dimension,
    rank_candidate_dimensions,
)
from .vae_arch import ResnetVAE, VAEClassifier


def _to_tensor(x: np.ndarray | torch.Tensor) -> torch.Tensor:
    """Convert to torch tensor, ensure float32 and correct shape (N,C,H,W)."""
    if isinstance(x, torch.Tensor):
        t = x.float()
    else:
        t = torch.from_numpy(np.asarray(x, dtype=np.float32))
    if t.ndim == 3:
        t = t.unsqueeze(1)  # (N,H,W) -> (N,1,H,W)
    elif t.ndim == 4 and t.shape[-1] in (1, 3):
        t = t.permute(0, 3, 1, 2)  # (N,H,W,C) -> (N,C,H,W)
    return t


def _to_numpy(x: Any) -> np.ndarray:
    """Convert to numpy, handling tensors."""
    if isinstance(x, torch.Tensor):
        return x.detach().cpu().numpy()
    return np.asarray(x)


class VAEDetector(DetectorBase):
    """
    Detect shortcuts via VAE latent disentanglement and predictiveness analysis.

    Trains a Beta-VAE on images, then trains a classifier on frozen VAE encoder.
    Latent dimensions with high classifier weight (predictiveness) are candidate
    shortcuts. MPWD (max pairwise Wasserstein distance) per dimension indicates
    class separability.
    """

    def __init__(
        self,
        *,
        latent_dim: int = 10,
        kld_weight: float = 3.0,
        lr: float = 0.001,
        batch_size: int = 32,
        epochs: int = 50,
        classifier_epochs: int = 20,
        device: str | torch.device | None = None,
        predictiveness_threshold: float = 0.5,
        random_state: int = 42,
    ) -> None:
        """
        Args:
            latent_dim: VAE latent dimension.
            kld_weight: Beta-VAE KL weight (higher = more disentanglement).
            lr: VAE learning rate.
            batch_size: Training batch size.
            epochs: VAE training epochs.
            classifier_epochs: Classifier training epochs (frozen encoder).
            device: Device for training (cuda/cpu).
            predictiveness_threshold: Normalized predictiveness [0,1] above which
                a dimension is flagged as shortcut. Default 0.5 = top half.
            random_state: Random seed.
        """
        super().__init__(method="vae")
        self.latent_dim = int(latent_dim)
        self.kld_weight = float(kld_weight)
        self.lr = float(lr)
        self.batch_size = int(batch_size)
        self.epochs = int(epochs)
        self.classifier_epochs = int(classifier_epochs)
        self.device = torch.device(device or ("cuda" if torch.cuda.is_available() else "cpu"))
        self.predictiveness_threshold = float(predictiveness_threshold)
        self.random_state = int(random_state)

        self._vae: ResnetVAE | None = None
        self._classifier: VAEClassifier | None = None
        self._latents_: np.ndarray | None = None
        self._labels_: np.ndarray | None = None

    def fit(
        self,
        *,
        images: np.ndarray | torch.Tensor | None = None,
        labels: np.ndarray | torch.Tensor | None = None,
        img_size: int = 0,
        channels: int = 3,
        num_classes: int = 2,
        group_labels: np.ndarray | torch.Tensor | None = None,
        vae_checkpoint: str | None = None,
        train_dl: DataLoader | None = None,
        val_dl: DataLoader | None = None,
        test_dl: DataLoader | None = None,
    ) -> VAEDetector:
        """
        Fit VAE detector on images.

        Provide either (images, labels) or (train_dl, val_dl, test_dl).
        """
        if train_dl is not None and val_dl is not None:
            self._fit_from_dataloaders(
                train_dl=train_dl,
                val_dl=val_dl,
                test_dl=test_dl or val_dl,
                img_size=img_size,
                channels=channels,
                num_classes=num_classes,
                vae_checkpoint=vae_checkpoint,
            )
        else:
            if images is None or labels is None:
                raise ValueError("Provide images and labels, or train_dl/val_dl/test_dl.")
            self._fit_from_arrays(
                images=images,
                labels=labels,
                img_size=img_size,
                channels=channels,
                num_classes=num_classes,
                vae_checkpoint=vae_checkpoint,
            )

        latent_matrix = self._latents_
        labels_arr = self._labels_
        assert latent_matrix is not None
        assert labels_arr is not None

        mpwd = compute_mpwd_per_dimension(
            latent_matrix,
            labels_arr.astype(np.int64),
            self.latent_dim,
            num_classes,
        )

        predictiveness = compute_predictiveness_per_dimension(
            self._classifier,
            self.latent_dim,
        )

        # Normalize predictiveness to [0, 1] for thresholding
        pred_max = float(np.max(predictiveness))
        pred_norm = predictiveness / pred_max if pred_max > 0 else predictiveness

        _, flagged_indices = rank_candidate_dimensions(
            pred_norm,
            mpwd,
            self.predictiveness_threshold,
        )

        shortcut_detected = len(flagged_indices) > 0
        max_pred = float(np.max(pred_norm))
        n_flagged = len(flagged_indices)

        if shortcut_detected:
            risk_level = "high" if n_flagged >= self.latent_dim // 2 else "moderate"
            notes = f"{n_flagged} latent dimension(s) exceed predictiveness threshold."
        else:
            risk_level = "low"
            notes = "No latent dimension exceeded predictiveness threshold."

        per_dim = []
        for i in range(self.latent_dim):
            per_dim.append(
                {
                    "dimension": i,
                    "predictiveness": float(pred_norm[i]),
                    "mpwd": float(mpwd[i]),
                    "flagged": i in flagged_indices,
                }
            )

        metrics = {
            "n_candidate_dims": n_flagged,
            "max_predictiveness": max_pred,
            "n_flagged": n_flagged,
            "latent_dim": self.latent_dim,
        }

        metadata = {
            "kld_weight": self.kld_weight,
            "epochs": self.epochs,
            "predictiveness_threshold": self.predictiveness_threshold,
        }

        self._set_results(
            shortcut_detected=shortcut_detected,
            risk_level=risk_level,
            metrics=metrics,
            notes=notes,
            metadata=metadata,
            report={"per_dimension": per_dim},
            details={"mpwd": mpwd.tolist(), "predictiveness": pred_norm.tolist()},
        )
        self.shortcut_detected_ = shortcut_detected
        self._is_fitted = True
        return self

    def _fit_from_arrays(
        self,
        images: np.ndarray | torch.Tensor,
        labels: np.ndarray | torch.Tensor,
        img_size: int,
        channels: int,
        num_classes: int,
        vae_checkpoint: str | None,
    ) -> None:
        X = _to_tensor(images)
        y = _to_numpy(labels).astype(np.int64)
        y_t = torch.from_numpy(y)

        n = len(X)
        if n < 10:
            raise ValueError("Need at least 10 samples for VAE training.")

        torch.manual_seed(self.random_state)

        val_size = max(1, n // 5)
        train_size = n - val_size
        train_ds = TensorDataset(X[:train_size], y_t[:train_size])
        val_ds = TensorDataset(X[train_size:], y_t[train_size:])

        train_dl = DataLoader(train_ds, batch_size=self.batch_size, shuffle=True)
        val_dl = DataLoader(val_ds, batch_size=self.batch_size)

        self._fit_vae_and_classifier(
            train_dl=train_dl,
            val_dl=val_dl,
            img_size=img_size,
            channels=channels,
            num_classes=num_classes,
            vae_checkpoint=vae_checkpoint,
        )

        with torch.no_grad():
            all_latents = []
            self._vae.eval()
            for bx, _ in DataLoader(TensorDataset(X, y_t), batch_size=self.batch_size):
                bx = bx.to(self.device)
                mu, _ = self._vae.encode(bx)
                all_latents.append(mu.cpu().numpy())
            self._latents_ = np.vstack(all_latents)
            self._labels_ = y

    def _fit_from_dataloaders(
        self,
        train_dl: DataLoader,
        val_dl: DataLoader,
        test_dl: DataLoader,
        img_size: int,
        channels: int,
        num_classes: int,
        vae_checkpoint: str | None,
    ) -> None:
        self._fit_vae_and_classifier(
            train_dl=train_dl,
            val_dl=val_dl,
            img_size=img_size,
            channels=channels,
            num_classes=num_classes,
            vae_checkpoint=vae_checkpoint,
        )

        all_latents: list[np.ndarray] = []
        all_labels: list[np.ndarray] = []
        self._vae.eval()
        with torch.no_grad():
            for batch in test_dl:
                if isinstance(batch, list | tuple):
                    bx, by = batch[0], batch[1]
                else:
                    bx, by = batch.get("images", batch.get("x")), batch.get(
                        "labels", batch.get("y")
                    )
                bx = bx.to(self.device)
                mu, _ = self._vae.encode(bx)
                all_latents.append(mu.cpu().numpy())
                all_labels.append(_to_numpy(by))
        self._latents_ = np.vstack(all_latents)
        self._labels_ = np.concatenate(all_labels)

    def _fit_vae_and_classifier(
        self,
        train_dl: DataLoader,
        val_dl: DataLoader,
        img_size: int,
        channels: int,
        num_classes: int,
        vae_checkpoint: str | None,
    ) -> None:
        vae = ResnetVAE(
            input_size=img_size,
            latent_dim=self.latent_dim,
            input_channels=channels,
            kld_weight=self.kld_weight,
            cls_weight=0.0,
            num_classes=num_classes,
        )
        vae = vae.to(self.device)

        if vae_checkpoint and Path(vae_checkpoint).exists():
            ckpt = torch.load(vae_checkpoint, map_location=self.device)
            if isinstance(ckpt, dict) and "model_state_dict" in ckpt:
                vae.load_state_dict(ckpt["model_state_dict"])
            else:
                vae.load_state_dict(ckpt)
        else:
            self._train_vae(vae, train_dl, val_dl)

        self._vae = vae

        classifier = VAEClassifier(vae=vae, latent_dim=self.latent_dim, classes=num_classes)
        classifier = classifier.to(self.device)
        self._train_classifier(classifier, train_dl, val_dl)
        self._classifier = classifier

    def _train_vae(self, vae: ResnetVAE, train_dl: DataLoader, val_dl: DataLoader) -> None:
        opt = torch.optim.Adam(vae.parameters(), lr=self.lr)
        for ep in range(self.epochs):
            vae.train()
            train_loss = 0.0
            for batch in train_dl:
                if isinstance(batch, list | tuple):
                    bx, by = batch[0], batch[1]
                else:
                    bx = batch.get("images", batch.get("x"))
                    by = batch.get("labels", batch.get("y"))
                bx = bx.to(self.device)
                by = by.to(self.device).long()
                opt.zero_grad()
                loss, _ = vae(bx, by)
                loss.backward()
                opt.step()
                train_loss += loss.item()
            if (ep + 1) % 10 == 0:
                vae.eval()
                val_loss = 0.0
                with torch.no_grad():
                    for batch in val_dl:
                        if isinstance(batch, list | tuple):
                            bx, by = batch[0], batch[1]
                        else:
                            bx = batch.get("images", batch.get("x"))
                            by = batch.get("labels", batch.get("y"))
                        bx, by = bx.to(self.device), by.to(self.device).long()
                        loss, _ = vae(bx, by)
                        val_loss += loss.item()
                val_loss /= max(1, len(val_dl))

    def _train_classifier(
        self,
        classifier: VAEClassifier,
        train_dl: DataLoader,
        val_dl: DataLoader,
    ) -> None:
        opt = torch.optim.Adam(classifier.fc.parameters(), lr=self.lr * 2)
        crit = torch.nn.CrossEntropyLoss()
        for _ in range(self.classifier_epochs):
            classifier.train()
            for batch in train_dl:
                if isinstance(batch, list | tuple):
                    bx, by = batch[0], batch[1]
                else:
                    bx = batch.get("images", batch.get("x"))
                    by = batch.get("labels", batch.get("y"))
                bx = bx.to(self.device)
                by = by.to(self.device).long()
                opt.zero_grad()
                out = classifier(bx)
                loss = crit(out, by)
                loss.backward()
                opt.step()
