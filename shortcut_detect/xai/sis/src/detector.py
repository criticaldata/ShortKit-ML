"""Sufficient Input Subsets (SIS) detector for shortcut detection.

Carter et al. 2019: finds minimal embedding dimensions that suffice for prediction.
"""

from __future__ import annotations

from typing import Any

import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split

from ....detector_base import DetectorBase


class SISDetector(DetectorBase):
    """Find minimal sufficient input subsets for shortcut detection (Carter et al. 2019)."""

    def __init__(
        self,
        *,
        mask_value: float = 0.0,
        max_samples: int = 200,
        test_size: float = 0.2,
        shortcut_threshold: float = 0.15,
        seed: int = 42,
    ) -> None:
        super().__init__(method="sis")
        if max_samples < 1:
            raise ValueError("max_samples must be >= 1.")
        if not 0.0 < test_size < 1.0:
            raise ValueError("test_size must be in (0, 1).")
        if not 0.0 <= shortcut_threshold <= 1.0:
            raise ValueError("shortcut_threshold must be in [0, 1].")

        self.mask_value = float(mask_value)
        self.max_samples = int(max_samples)
        self.test_size = float(test_size)
        self.shortcut_threshold = float(shortcut_threshold)
        self.seed = int(seed)

        self.probe_ = None
        self.sis_sizes_: list[int] = []
        self.sis_indices_per_sample_: list[list[int]] | None = None

    def fit(
        self,
        embeddings: np.ndarray,
        labels: np.ndarray,
        group_labels: np.ndarray | None = None,
        *,
        probe: Any | None = None,
    ) -> SISDetector:
        """Fit SIS detector: find minimal sufficient subsets per sample.

        Args:
            embeddings: (n_samples, n_dim) array
            labels: (n_samples,) target labels (binary or multiclass)
            group_labels: Optional (n_samples,) for group-SIS overlap analysis
            probe: Optional sklearn-compatible classifier; default LogisticRegression

        Returns:
            self
        """
        embeddings = np.asarray(embeddings, dtype=float)
        labels = np.asarray(labels)

        if embeddings.ndim != 2:
            raise ValueError(f"embeddings must be 2D, got shape {embeddings.shape}")
        if labels.ndim != 1:
            raise ValueError(f"labels must be 1D, got shape {labels.shape}")
        if embeddings.shape[0] != labels.shape[0]:
            raise ValueError(
                f"embeddings and labels must have same n_samples: "
                f"{embeddings.shape[0]} != {labels.shape[0]}"
            )

        n_samples, n_dim = embeddings.shape
        rng = np.random.RandomState(self.seed)

        # Stratified train/test split with indices
        indices = np.arange(n_samples)
        try:
            train_idx, test_idx = train_test_split(
                indices,
                test_size=self.test_size,
                stratify=labels,
                random_state=self.seed,
            )
        except ValueError:
            train_idx, test_idx = train_test_split(
                indices,
                test_size=self.test_size,
                random_state=self.seed,
            )

        X_train = embeddings[train_idx]
        X_test = embeddings[test_idx]
        y_train = labels[train_idx]
        y_test = labels[test_idx]

        # Train probe
        if probe is None:
            probe = LogisticRegression(max_iter=1000, random_state=self.seed)
        probe.fit(X_train, y_train)
        self.probe_ = probe

        coef_arr = np.asarray(probe.coef_)
        if coef_arr.ndim == 2:
            coef = np.max(np.abs(coef_arr), axis=0)
        else:
            coef = np.abs(coef_arr).flatten()
        if len(coef) != n_dim:
            coef = np.broadcast_to(coef, (n_dim,))[:n_dim].copy()

        # Order dimensions by ascending |coef| (remove least important first)
        order = np.argsort(np.abs(coef))

        # Subsample test set for SIS computation
        n_test = X_test.shape[0]
        n_compute = min(n_test, self.max_samples)
        if n_compute < n_test:
            sub_idx = rng.choice(n_test, size=n_compute, replace=False)
            X_sub = X_test[sub_idx]
            y_sub = y_test[sub_idx]
            test_sub_idx = test_idx[sub_idx]
        else:
            X_sub = X_test
            y_sub = y_test
            test_sub_idx = test_idx

        sis_sizes: list[int] = []
        sis_indices_list: list[list[int]] = []

        for i in range(X_sub.shape[0]):
            x = X_sub[i].copy()
            y_true = y_sub[i]
            pred_full = probe.predict(x.reshape(1, -1))[0]

            if pred_full != y_true:
                continue

            # Backward selection: remove dims in order of least importance
            active = set(range(n_dim))
            x_masked = x.copy()

            for d in order:
                if d not in active:
                    continue
                x_masked[d] = self.mask_value
                pred = probe.predict(x_masked.reshape(1, -1))[0]
                if pred == y_true:
                    active.remove(d)
                else:
                    x_masked[d] = x[d]

            sis_sizes.append(len(active))
            sis_indices_list.append(sorted(active))

        if not sis_sizes:
            notes = "No test samples had correct predictions; SIS could not be computed."
            shortcut_detected = None
            risk_level = "unknown"
            metrics = {"n_computed": 0, "mean_sis_size": None, "median_sis_size": None}
        else:
            mean_sis = float(np.mean(sis_sizes))
            median_sis = float(np.median(sis_sizes))
            frac_dim = mean_sis / n_dim if n_dim > 0 else 0.0

            # Shortcut signal: small mean SIS (model uses few dims) or high group overlap
            group_overlap = None
            if group_labels is not None and len(sis_indices_list) > 0:
                g_sub = np.asarray(group_labels)[test_sub_idx]
                group_overlap = self._compute_group_sis_overlap(
                    g_sub, X_sub, sis_indices_list, probe
                )

            shortcut_detected = frac_dim <= self.shortcut_threshold
            if group_overlap is not None and group_overlap > 0.5:
                shortcut_detected = True

            if shortcut_detected:
                risk_level = "high" if frac_dim <= 0.1 else "moderate"
            else:
                risk_level = "low"

            notes = (
                f"Mean SIS size: {mean_sis:.1f} ({frac_dim:.1%} of dims). "
                f"Small SIS indicates model may rely on few dimensions (potential shortcut)."
            )
            if group_overlap is not None:
                notes += f" Group-SIS overlap: {group_overlap:.2f}."

            metrics = {
                "mean_sis_size": mean_sis,
                "median_sis_size": median_sis,
                "min_sis_size": int(min(sis_sizes)),
                "max_sis_size": int(max(sis_sizes)),
                "n_computed": len(sis_sizes),
                "n_dimensions": n_dim,
                "frac_dimensions": frac_dim,
                "group_sis_overlap": group_overlap,
            }

        self.sis_sizes_ = sis_sizes
        self.sis_indices_per_sample_ = sis_indices_list if sis_indices_list else None

        report = {
            "sis_sizes": sis_sizes,
            "distribution": {
                "mean": float(np.mean(sis_sizes)) if sis_sizes else None,
                "median": float(np.median(sis_sizes)) if sis_sizes else None,
                "min": int(min(sis_sizes)) if sis_sizes else None,
                "max": int(max(sis_sizes)) if sis_sizes else None,
            },
        }

        metadata = {
            "mask_value": self.mask_value,
            "max_samples": self.max_samples,
            "test_size": self.test_size,
            "shortcut_threshold": self.shortcut_threshold,
            "seed": self.seed,
        }

        self._set_results(
            shortcut_detected=shortcut_detected,
            risk_level=risk_level,
            metrics=metrics,
            notes=notes,
            metadata=metadata,
            report=report,
            details={"sis_indices_per_sample": sis_indices_list} if sis_indices_list else None,
        )
        self._is_fitted = True
        return self

    def _compute_group_sis_overlap(
        self,
        group_labels: np.ndarray,
        X_sub: np.ndarray,
        sis_indices_list: list[list[int]],
        probe: Any,
    ) -> float | None:
        """Compute overlap between SIS dimensions and group-discriminative dimensions."""
        group_labels = np.asarray(group_labels)
        if group_labels.shape[0] != X_sub.shape[0]:
            return None

        # Encode groups for binary/multiclass probe
        from sklearn.preprocessing import LabelEncoder

        le = LabelEncoder()
        try:
            g_enc = le.fit_transform(group_labels)
        except Exception:
            return None

        if len(np.unique(g_enc)) < 2:
            return None

        n_dim = X_sub.shape[1]
        group_probe = LogisticRegression(max_iter=1000, random_state=self.seed)
        group_probe.fit(X_sub, g_enc)
        gc_arr = np.asarray(group_probe.coef_)
        if gc_arr.ndim == 2:
            group_coef = np.max(np.abs(gc_arr), axis=0)
        else:
            group_coef = np.abs(gc_arr).flatten()
        if len(group_coef) != n_dim:
            return None

        # Top group-discriminative dims (top 25% by |coef|)
        n_top = max(1, n_dim // 4)
        top_dims = set(np.argsort(group_coef)[-n_top:])

        # Fraction of SIS dims that are in top group-discriminative
        overlaps = []
        for sis_dims in sis_indices_list:
            if not sis_dims:
                continue
            overlap = len(set(sis_dims) & top_dims) / len(sis_dims)
            overlaps.append(overlap)
        return float(np.mean(overlaps)) if overlaps else None
