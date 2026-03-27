"""Builder for generative CVAE counterfactual detector."""

import warnings
from typing import Any

import numpy as np

from ...base_builder import BaseDetector
from .src.detector import GenerativeCVEDetector


class GenerativeCVEDetectorBuilder(BaseDetector):
    def __init__(self, seed: int = 42, kwargs: dict | None = None, method: str = "generative_cvae"):
        super().__init__(seed=seed, kwargs=kwargs, method=method)

    def build(self):
        dim = self.kwargs.get("dim", self.kwargs.get("embedding_dim", 128))
        hidden = self.kwargs.get("hidden", 256)
        zdim = self.kwargs.get("zdim", 64)
        lr = self.kwargs.get("lr", 1e-3)
        batch_size = self.kwargs.get("batch_size", 256)
        epochs = self.kwargs.get("epochs", 50)
        device = self.kwargs.get("device", "cpu")
        recon_loss_weight = self.kwargs.get("recon_loss_weight", 1.0)
        kld_weight = self.kwargs.get("kld_weight", 1e-3)
        verbose = self.kwargs.get("verbose", False)
        random_state = self.kwargs.get("causal_random_state", self.seed)

        return GenerativeCVEDetector(
            dim=dim,
            hidden=hidden,
            zdim=zdim,
            lr=lr,
            batch_size=batch_size,
            epochs=epochs,
            device=device,
            recon_loss_weight=recon_loss_weight,
            kld_weight=kld_weight,
            verbose=verbose,
            random_state=random_state,
            probe_classifier=self.kwargs.get("probe_classifier"),
            method=self.method,
        )

    def run(
        self,
        embeddings: np.ndarray,
        labels: np.ndarray,
        group_labels: np.ndarray,
        feature_names: list[str] | None = None,
        protected_labels: np.ndarray | None = None,
        splits: dict[str, np.ndarray] | None = None,
        extra_labels: dict[str, np.ndarray] | None = None,
        **kwargs: Any,
    ) -> dict[str, Any]:
        # prefer explicit spurious/protected labels when provided
        spurious = protected_labels if protected_labels is not None else group_labels
        # If a probe_classifier is provided at run-time, inject into kwargs
        probe_classifier = kwargs.get("probe_classifier")
        if probe_classifier is not None:
            self.kwargs["probe_classifier"] = probe_classifier
        detector = self.build()
        try:
            detector.fit(embeddings, spurious, labels)
            return {
                "success": True,
                "detector": detector,
                "report": detector.results_,
                "summary_title": "Generative CVAE (embedding counterfactuals)",
                "summary_lines": [
                    f"Samples: {embeddings.shape[0]}",
                    "Generative counterfactuals produced",
                ],
                "risk_indicators": [],
            }
        except Exception as exc:
            warnings.warn(f"Generative CVAE detector failed: {exc}", stacklevel=2)
            return {"success": False, "error": str(exc)}
