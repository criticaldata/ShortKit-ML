# generative_detector.py
"""Generative embedding-space counterfactual detector.

Implements a conditional VAE (CVAE) trained on embeddings conditioned on a
binary spurious attribute. Generates counterfactual embeddings by encoding with
original attribute and decoding with the flipped attribute, then measures how a
probe classifier's predictions change.

This module registers a builder with the central DetectorFactory when imported
so it becomes available to ShortcutDetector via method name "generative_cvae".

Key points:
- Internal CVAE and (optional) internal probe use STANDARDIZED embeddings.
- External probe is assumed to expect RAW embeddings unless user wraps it.

Author: generated for user
"""

from __future__ import annotations

import random
from dataclasses import dataclass
from typing import Any

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

from ....detector_base import DetectorBase


@dataclass
class CVAEConfig:
    # dim is kept for configuration compatibility; the detector will infer dim from data.
    dim: int = 128
    hidden: int = 256
    zdim: int = 64
    lr: float = 1e-3
    batch_size: int = 256
    epochs: int = 50
    device: str = "cpu"
    recon_loss_weight: float = 1.0
    kld_weight: float = 1e-3
    verbose: bool = False
    random_state: int = 42

    # Latent guidance (ensures the spurious attribute is actually flipped)
    guidance_steps: int = 50
    guidance_lr: float = 5e-2
    guidance_weight: float = 5.0
    proximity_weight: float = 1.0

    # Detection rule thresholds (tunable)
    mean_delta_threshold: float = 1e-4
    frac_large_threshold: float = 0.01
    large_change_delta: float = 0.1


class CVAE(nn.Module):
    def __init__(self, dim: int, hidden: int = 256, zdim: int = 64):
        super().__init__()
        self.dim = dim
        self.zdim = zdim

        self.enc = nn.Sequential(
            nn.Linear(dim + 1, hidden),
            nn.ReLU(),
            nn.Linear(hidden, hidden),
            nn.ReLU(),
        )
        self.mu = nn.Linear(hidden, zdim)
        self.logvar = nn.Linear(hidden, zdim)

        self.dec = nn.Sequential(
            nn.Linear(zdim + 1, hidden),
            nn.ReLU(),
            nn.Linear(hidden, dim),
        )

    def encode(self, x: torch.Tensor, s: torch.Tensor):
        inp = torch.cat([x, s.unsqueeze(1)], dim=1)
        h = self.enc(inp)
        return self.mu(h), self.logvar(h)

    def reparam(self, mu: torch.Tensor, logvar: torch.Tensor) -> torch.Tensor:
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return mu + eps * std

    def decode(self, z: torch.Tensor, s: torch.Tensor) -> torch.Tensor:
        inp = torch.cat([z, s.unsqueeze(1)], dim=1)
        return self.dec(inp)

    def forward(self, x: torch.Tensor, s: torch.Tensor):
        mu, logvar = self.encode(x, s)
        z = self.reparam(mu, logvar)
        xrec = self.decode(z, s)
        return xrec, mu, logvar


class AttrNet(nn.Module):
    """Simple attribute predictor in embedding space (d -> 1 logit)."""

    def __init__(self, dim: int):
        super().__init__()
        self.linear = nn.Linear(dim, 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.linear(x).squeeze(1)


def _cosine_similarity_rows(A: np.ndarray, B: np.ndarray) -> np.ndarray:
    A = np.asarray(A, dtype=float)
    B = np.asarray(B, dtype=float)
    num = np.sum(A * B, axis=1)
    na = np.linalg.norm(A, axis=1)
    nb = np.linalg.norm(B, axis=1)
    denom = na * nb
    with np.errstate(divide="ignore", invalid="ignore"):
        sim = np.where(denom > 0, num / denom, 0.0)
    return sim


def _call_probe(probe: Any, X_np: np.ndarray) -> np.ndarray:
    """Return a 1D probability-like array for the positive class.

    Supported probe types:
    - sklearn-like objects with predict_proba
    - sklearn-like objects with predict
    - any callable f(X)->array

    Output coercion:
    - (N,2+) -> column 1
    - (N,) in [0,1] -> treat as prob
    - (N,) outside [0,1] -> treat as logits/scores and apply sigmoid
    """
    if hasattr(probe, "predict_proba"):
        out = probe.predict_proba(X_np)
    elif hasattr(probe, "predict"):
        out = probe.predict(X_np)
    elif callable(probe):
        out = probe(X_np)
    else:
        raise ValueError("Provided probe is not callable and lacks predict/predict_proba")

    out = np.asarray(out)
    if out.ndim == 2 and out.shape[1] >= 2:
        out = out[:, 1]
    else:
        out = out.ravel()

    if out.size != X_np.shape[0]:
        raise ValueError("Probe returned incompatible number of scores")

    # Coerce to probability if needed
    if np.nanmin(out) < 0.0 or np.nanmax(out) > 1.0:
        out = 1.0 / (1.0 + np.exp(-out))

    return out


class GenerativeCVEDetector(DetectorBase):
    """Generative CVAE detector for embedding counterfactuals."""

    def __init__(
        self,
        dim: int = 128,
        hidden: int = 256,
        zdim: int = 64,
        lr: float = 1e-3,
        batch_size: int = 256,
        epochs: int = 50,
        device: str = "cpu",
        recon_loss_weight: float = 1.0,
        kld_weight: float = 1e-3,
        verbose: bool = False,
        random_state: int = 42,
        probe_classifier: Any | None = None,
        method: str = "generative_cvae",
    ):
        super().__init__(method=method)

        self.cfg = CVAEConfig(
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
        )

        self.external_probe: Any | None = probe_classifier
        self.model: CVAE | None = None
        self.scaler: StandardScaler | None = None
        self.shortcut_detected_ = None

    def fit(
        self,
        embeddings: np.ndarray,
        group_labels: np.ndarray,
        labels: np.ndarray | None = None,
    ) -> GenerativeCVEDetector:
        embeddings = np.asarray(embeddings)
        if embeddings.ndim != 2:
            raise ValueError("embeddings must be a 2D array")
        n, d = embeddings.shape

        if group_labels is None:
            raise ValueError("group_labels (spurious attribute) required")
        s = np.asarray(group_labels).astype(int)
        if s.shape[0] != n:
            raise ValueError("group_labels length must match embeddings")
        if not np.all(np.isin(s, [0, 1])):
            raise ValueError("group_labels must be binary (0/1)")

        y = None if labels is None else np.asarray(labels).astype(int)
        if y is not None and y.shape[0] != n:
            raise ValueError("labels length must match embeddings")

        device = torch.device(self.cfg.device)

        # Seed all RNGs for reproducibility
        random.seed(self.cfg.random_state)
        np.random.seed(self.cfg.random_state)
        torch.manual_seed(self.cfg.random_state)

        # Standardize embeddings for CVAE training
        scaler = StandardScaler()
        E = scaler.fit_transform(embeddings)
        self.scaler = scaler

        X = torch.tensor(E, dtype=torch.float32, device=device)
        S = torch.tensor(s.astype(float), dtype=torch.float32, device=device)

        # Train CVAE
        model = CVAE(dim=d, hidden=self.cfg.hidden, zdim=self.cfg.zdim).to(device)
        opt = optim.Adam(model.parameters(), lr=self.cfg.lr)

        bs = min(self.cfg.batch_size, n)
        idx = np.arange(n)

        for epoch in range(self.cfg.epochs):
            model.train()
            np.random.shuffle(idx)
            total_loss = 0.0
            for i in range(0, n, bs):
                bidx = idx[i : i + bs]
                xb = X[bidx]
                sb = S[bidx]
                xrec, mu, logvar = model(xb, sb)

                rec_loss = ((xrec - xb) ** 2).mean()
                kld = -0.5 * torch.mean(1 + logvar - mu.pow(2) - logvar.exp())
                loss = self.cfg.recon_loss_weight * rec_loss + self.cfg.kld_weight * kld

                opt.zero_grad()
                loss.backward()
                opt.step()

                total_loss += float(loss.item()) * xb.shape[0]

            if self.cfg.verbose:
                if (
                    (epoch + 1) % max(1, (self.cfg.epochs // 5)) == 0
                    or epoch == 0
                    or (epoch + 1) == self.cfg.epochs
                ):
                    print(f"[CVAE] epoch {epoch+1}/{self.cfg.epochs} avg_loss={total_loss / n:.6f}")

        self.model = model

        # Prepare probe
        probe = None
        probe_trained_on = None  # "scaled" or "raw"
        probe_metrics = {}

        if self.external_probe is not None:
            probe = self.external_probe
            probe_trained_on = "raw"
            if y is not None and hasattr(probe, "score"):
                try:
                    probe_metrics["probe_accuracy"] = float(probe.score(embeddings, y))
                except Exception:
                    pass
        elif y is not None:
            # Internal probe: train on STANDARDIZED embeddings
            Xtr, Xte, ytr, yte = train_test_split(
                E, y, test_size=0.2, random_state=self.cfg.random_state
            )
            probe = LogisticRegression(max_iter=2000)
            probe.fit(Xtr, ytr)
            probe_trained_on = "scaled"
            probe_metrics["probe_accuracy"] = float(probe.score(Xte, yte))

        # Train an attribute predictor (on STANDARDIZED embeddings) to ensure s is actually flippable
        attr_net = AttrNet(dim=d).to(device)
        attr_opt = optim.Adam(attr_net.parameters(), lr=1e-2)
        bce = nn.BCEWithLogitsLoss()
        for epoch in range(50):
            attr_net.train()
            np.random.shuffle(idx)
            total = 0.0
            for i in range(0, n, bs):
                bidx = idx[i : i + bs]
                xb = X[bidx]
                sb = S[bidx]
                logits = attr_net(xb)
                loss = bce(logits, sb)
                attr_opt.zero_grad()
                loss.backward()
                attr_opt.step()
                total += float(loss.item()) * xb.shape[0]
            if self.cfg.verbose and (epoch + 1) in {1, 10, 25, 50}:
                print(f"[AttrNet] epoch {epoch+1}/50 avg_loss={total / n:.6f}")

        # Generate counterfactuals in STANDARDIZED space using latent guidance
        model.eval()
        attr_net.eval()
        with torch.no_grad():
            mu, _ = model.encode(X, S)
        # Optimize latent codes Z to (a) keep embeddings close and (b) flip attribute
        Z = mu.detach().clone().requires_grad_(True)
        S_cf = (1.0 - S).detach()
        z_opt = optim.Adam([Z], lr=self.cfg.guidance_lr)

        for _ in range(int(self.cfg.guidance_steps)):
            x_cf = model.decode(Z, S_cf)
            # proximity in standardized space
            prox = ((x_cf - X) ** 2).mean()
            # attribute flip loss
            logits_cf = attr_net(x_cf)
            attr_loss = bce(logits_cf, S_cf)
            loss = self.cfg.proximity_weight * prox + self.cfg.guidance_weight * attr_loss
            z_opt.zero_grad()
            loss.backward()
            z_opt.step()

        with torch.no_grad():
            X_cf = model.decode(Z.detach(), S_cf).cpu().numpy()  # scaled counterfactuals

        # Bring CFs back to raw space for storage / external probes
        E_cf = scaler.inverse_transform(X_cf)

        # Evaluate probe response
        eval_metrics = {}
        if probe is not None:
            try:
                if probe_trained_on == "scaled":
                    p_orig = _call_probe(probe, E)
                    p_cf = _call_probe(probe, X_cf)
                else:
                    p_orig = _call_probe(probe, embeddings)
                    p_cf = _call_probe(probe, E_cf)

                delta = np.asarray(p_orig).ravel() - np.asarray(p_cf).ravel()
                eval_metrics["mean_delta"] = float(np.mean(delta))
                eval_metrics["median_delta"] = float(np.median(delta))
                eval_metrics["std_delta"] = float(np.std(delta))
                eval_metrics["frac_large_change"] = float(
                    (np.abs(delta) > self.cfg.large_change_delta).mean()
                )
            except Exception as exc:
                eval_metrics["note"] = f"Probe failed during prediction: {exc}"
        else:
            eval_metrics["note"] = "No probe available (provide labels or external probe)"

        # Similarity diagnostics (in raw space)
        cos_sim = _cosine_similarity_rows(embeddings, E_cf)
        eval_metrics["mean_cosine_similarity"] = float(np.nanmean(cos_sim))
        eval_metrics["median_cosine_similarity"] = float(np.nanmedian(cos_sim))

        # Detection decision — use abs(mean_delta) since the sign depends on
        # which direction the counterfactual shifts predictions, not on whether
        # a shortcut exists.
        shortcut_detected = None
        if "mean_delta" in eval_metrics and "frac_large_change" in eval_metrics:
            mean_delta = float(eval_metrics["mean_delta"])
            frac_large = float(eval_metrics["frac_large_change"])
            shortcut_detected = (abs(mean_delta) > self.cfg.mean_delta_threshold) and (
                frac_large > self.cfg.frac_large_threshold
            )

        # Report
        metadata = {
            "n_samples": int(n),
            "embedding_dim": int(d),
            "device": str(device),
            "cvae_epochs": int(self.cfg.epochs),
            "probe_trained_on": probe_trained_on,
            "guidance_steps": int(self.cfg.guidance_steps),
            "guidance_weight": float(self.cfg.guidance_weight),
            "proximity_weight": float(self.cfg.proximity_weight),
            "random_state": self.cfg.random_state,
        }

        details = {
            "original_embeddings": embeddings,
            "counterfactual_embeddings": E_cf,
            "spurious_labels": s,
            "probe": probe,
        }

        self._set_results(
            shortcut_detected=bool(shortcut_detected) if shortcut_detected is not None else None,
            risk_level=(
                "unknown" if shortcut_detected is None else ("high" if shortcut_detected else "low")
            ),
            metrics={**eval_metrics, **probe_metrics},
            notes="Generative CVAE counterfactuals in embedding space.",
            metadata=metadata,
            report={"description": "Generative CVAE detector report summary"},
            details=details,
        )

        self.shortcut_detected_ = shortcut_detected
        self._is_fitted = True
        return self
