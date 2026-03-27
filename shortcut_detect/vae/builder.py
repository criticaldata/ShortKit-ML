"""Builder for VAE detector."""

import warnings
from typing import Any

import numpy as np

from ..base_builder import BaseDetector
from . import VAEDetector


class VAEDetectorBuilder(BaseDetector):
    def build(self):
        return VAEDetector(
            latent_dim=self.kwargs.get("vae_latent_dim", 10),
            kld_weight=self.kwargs.get("vae_kld_weight", 3.0),
            lr=self.kwargs.get("vae_lr", 0.001),
            batch_size=self.kwargs.get("vae_batch_size", 32),
            epochs=self.kwargs.get("vae_epochs", 50),
            classifier_epochs=self.kwargs.get("vae_classifier_epochs", 20),
            device=self.kwargs.get("vae_device"),
            predictiveness_threshold=self.kwargs.get("vae_predictiveness_threshold", 0.5),
            random_state=self.seed,
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
            "vae requires raw images/dataloaders. Use ShortcutDetector.fit_from_loaders."
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
            raise ValueError("Loader for vae must return a dict.")

        images = data.get("images")
        labels = data.get("labels")
        train_dl = data.get("train_dl")
        val_dl = data.get("val_dl")
        test_dl = data.get("test_dl")
        img_size = data.get("img_size")
        channels = data.get("channels", 3)
        num_classes = data.get("num_classes", 2)
        vae_checkpoint = data.get("vae_checkpoint")
        device = data.get("device")

        if img_size is None:
            raise ValueError("Loader for vae must provide 'img_size' (image height/width).")
        if labels is None and (train_dl is None or val_dl is None):
            raise ValueError("Loader for vae must provide 'labels' or 'train_dl' and 'val_dl'.")

        detector = self.build()
        if device is not None:
            import torch

            detector.device = torch.device(device) if isinstance(device, str) else device

        try:
            if train_dl is not None and val_dl is not None:
                detector.fit(
                    img_size=img_size,
                    channels=channels,
                    num_classes=num_classes,
                    train_dl=train_dl,
                    val_dl=val_dl,
                    test_dl=test_dl,
                    vae_checkpoint=vae_checkpoint,
                )
            else:
                if images is None:
                    raise ValueError(
                        "Loader for vae must provide 'images' when not using dataloaders."
                    )
                detector.fit(
                    images=images,
                    labels=labels,
                    img_size=img_size,
                    channels=channels,
                    num_classes=num_classes,
                    vae_checkpoint=vae_checkpoint,
                )

            report = detector.get_report()
            metrics = report.get("metrics", {})
            n_flagged = metrics.get("n_flagged", 0)
            max_pred = metrics.get("max_predictiveness")
            summary_lines = [
                f"Latent dims: {metrics.get('latent_dim', 0)}",
                f"Flagged shortcut dims: {n_flagged}",
                f"Max predictiveness: {max_pred:.3f}" if max_pred is not None else "",
            ]
            summary_lines = [s for s in summary_lines if s]

            risk_indicators = []
            if n_flagged > 0:
                risk_indicators.append(
                    f"VAE flagged {n_flagged} latent dimension(s) as shortcut candidates"
                )

            return {
                "detector": detector,
                "report": report.get("report", {}),
                "summary_title": "VAE (Variational Autoencoder) Shortcut Detection",
                "summary_lines": summary_lines,
                "risk_indicators": risk_indicators,
                "success": True,
                "metrics": metrics,
                "metadata": report.get("metadata", {}),
                "shortcut_detected": report.get("shortcut_detected"),
            }
        except Exception as exc:
            warnings.warn(f"VAE analysis failed: {exc}", stacklevel=2)
            return {"success": False, "error": str(exc)}
