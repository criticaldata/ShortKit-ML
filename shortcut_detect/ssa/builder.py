"""Builder for SSA (Spread Spurious Attribute) detector."""

import warnings
from typing import Any

import numpy as np

from ..base_builder import BaseDetector


class SSADetectorBuilder(BaseDetector):
    def build(self):
        from ..groupdro.groupdro import GroupDROConfig
        from . import SSAConfig, SSADetector

        gdro_cfg = GroupDROConfig(
            n_epochs=self.kwargs.get(
                "ssa_groupdro_n_epochs", self.kwargs.get("groupdro_n_epochs", 10)
            ),
            batch_size=self.kwargs.get(
                "ssa_groupdro_batch_size", self.kwargs.get("groupdro_batch_size", 128)
            ),
            lr=self.kwargs.get("ssa_groupdro_lr", self.kwargs.get("groupdro_lr", 1e-3)),
            weight_decay=self.kwargs.get(
                "ssa_groupdro_weight_decay", self.kwargs.get("groupdro_weight_decay", 5e-5)
            ),
            momentum=self.kwargs.get(
                "ssa_groupdro_momentum", self.kwargs.get("groupdro_momentum", 0.9)
            ),
            num_workers=self.kwargs.get(
                "ssa_groupdro_num_workers", self.kwargs.get("groupdro_num_workers", 0)
            ),
            loader_factory=self.kwargs.get(
                "ssa_groupdro_loader_factory",
                self.kwargs.get("groupdro_loader_factory", None),
            ),
            stage_loader_overrides=self.kwargs.get(
                "ssa_groupdro_stage_loader_overrides",
                self.kwargs.get("groupdro_stage_loader_overrides", None),
            ),
            robust=self.kwargs.get("ssa_groupdro_robust", self.kwargs.get("groupdro_robust", True)),
            alpha=self.kwargs.get("ssa_groupdro_alpha", self.kwargs.get("groupdro_alpha", 0.2)),
            gamma=self.kwargs.get("ssa_groupdro_gamma", self.kwargs.get("groupdro_gamma", 0.1)),
            robust_step_size=self.kwargs.get(
                "ssa_groupdro_robust_step_size", self.kwargs.get("groupdro_robust_step_size", 0.01)
            ),
            use_normalized_loss=self.kwargs.get(
                "ssa_groupdro_use_normalized_loss",
                self.kwargs.get("groupdro_use_normalized_loss", False),
            ),
            btl=self.kwargs.get("ssa_groupdro_btl", self.kwargs.get("groupdro_btl", False)),
            minimum_variational_weight=self.kwargs.get(
                "ssa_groupdro_minimum_variational_weight",
                self.kwargs.get("groupdro_minimum_variational_weight", 0.0),
            ),
            generalization_adjustment=self.kwargs.get(
                "ssa_groupdro_generalization_adjustment",
                self.kwargs.get("groupdro_generalization_adjustment", None),
            ),
            automatic_adjustment=self.kwargs.get(
                "ssa_groupdro_automatic_adjustment",
                self.kwargs.get("groupdro_automatic_adjustment", False),
            ),
            hidden_dim=self.kwargs.get(
                "ssa_groupdro_hidden_dim", self.kwargs.get("groupdro_hidden_dim", None)
            ),
            dropout=self.kwargs.get(
                "ssa_groupdro_dropout", self.kwargs.get("groupdro_dropout", 0.0)
            ),
            val_fraction=self.kwargs.get(
                "ssa_groupdro_val_fraction", self.kwargs.get("groupdro_val_fraction", 0.1)
            ),
            seed=self.seed,
            device=self.kwargs.get("ssa_groupdro_device", self.kwargs.get("groupdro_device", None)),
        )

        ssa_cfg = SSAConfig(
            K=self.kwargs.get("ssa_K", 3),
            T=self.kwargs.get("ssa_T", 2000),
            batch_size=self.kwargs.get("ssa_batch_size", 128),
            lr=self.kwargs.get("ssa_lr", 1e-3),
            weight_decay=self.kwargs.get("ssa_weight_decay", 1e-4),
            momentum=self.kwargs.get("ssa_momentum", 0.9),
            hidden_dim=self.kwargs.get("ssa_hidden_dim", None),
            dropout=self.kwargs.get("ssa_dropout", 0.0),
            tau_gmin=self.kwargs.get("ssa_tau_gmin", 0.95),
            threshold_update_every=self.kwargs.get("ssa_threshold_update_every", 200),
            threshold_update_max_items=self.kwargs.get("ssa_threshold_update_max_items", 20000),
            dl_val_fraction=self.kwargs.get("ssa_dl_val_fraction", 0.5),
            seed=self.seed,
            device=self.kwargs.get("ssa_device", None),
            loader_factory=self.kwargs.get("ssa_loader_factory"),
            stage_loader_overrides=self.kwargs.get("ssa_stage_loader_overrides"),
            groupdro=gdro_cfg,
        )

        return SSADetector(ssa_cfg)

    def _require_split(self, splits: dict[str, np.ndarray] | None, name: str, n: int) -> np.ndarray:
        if splits is None or name not in splits:
            raise ValueError(f"Missing splits['{name}'] required by selected method(s).")
        idx = np.asarray(splits[name], dtype=np.int64)
        if idx.ndim != 1:
            raise ValueError(f"splits['{name}'] must be 1D indices.")
        if idx.size == 0:
            raise ValueError(f"splits['{name}'] is empty.")
        if np.any(idx < 0) or np.any(idx >= n):
            raise ValueError(f"splits['{name}'] contains out-of-range indices.")
        return idx

    def _require_extra_label(
        self, extra_labels: dict[str, np.ndarray] | None, key: str, n: int
    ) -> np.ndarray:
        if extra_labels is None or key not in extra_labels:
            raise ValueError(f"Missing extra_labels['{key}'] required by selected method(s).")
        arr = np.asarray(extra_labels[key])
        if arr.ndim != 1 or arr.shape[0] != n:
            raise ValueError(f"extra_labels['{key}'] must be 1D of length n_samples.")
        return arr

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
        print("Running SSA (Spread Spurious Attribute)...")

        if splits is None:
            warnings.warn("SSA analysis skipped (splits parameter is required).", stacklevel=2)
            return {
                "success": False,
                "error": "splits parameter required for SSA analysis",
                "summary_title": "SSA (Spread Spurious Attribute)",
                "summary_lines": ["⚠️  SSA skipped: splits parameter required"],
                "risk_indicators": [],
            }

        detector = self.build()
        try:
            n = embeddings.shape[0]
            du_idx = self._require_split(splits, "train_u", n)
            dl_idx = self._require_split(splits, "train_l", n)

            if np.intersect1d(du_idx, dl_idx).size > 0:
                raise ValueError(
                    "splits['train_u'] and splits['train_l'] must be disjoint for SSA."
                )

            spurious_full = self._require_extra_label(extra_labels, "spurious", n).astype(np.int64)

            sp_dl = spurious_full[dl_idx]
            if np.any(sp_dl < 0):
                raise ValueError(
                    "extra_labels['spurious'] must be defined (>=0) for all train_l indices."
                )

            du_embeddings = embeddings[du_idx]
            du_labels = labels[du_idx]

            dl_embeddings = embeddings[dl_idx]
            dl_labels = labels[dl_idx]
            dl_spurious = sp_dl

            detector.fit(
                du_embeddings=du_embeddings,
                du_labels=du_labels,
                dl_embeddings=dl_embeddings,
                dl_labels=dl_labels,
                dl_spurious=dl_spurious,
            )

            rep = detector.get_report()
            detail = rep.get("report", {})

            n_labeled = int(len(splits["train_l"]))
            n_unlabeled = int(len(splits["train_u"]))
            shortcut_detected = detector.shortcut_detected_

            summary_lines = []
            summary_lines.append(
                f"Labeled samples: {n_labeled if n_labeled is not None else 'unknown'}"
            )
            summary_lines.append(
                f"Unlabeled samples: {n_unlabeled if n_unlabeled is not None else 'unknown'}"
            )

            gdro_final = detail.get("groupdro_report", {}).get("final", {})
            avg_acc = gdro_final.get("avg_acc")
            worst_acc = gdro_final.get("worst_group_acc")
            if avg_acc is not None and worst_acc is not None:
                summary_lines.append(f"Avg acc (val): {avg_acc:.2%}")
                summary_lines.append(f"Worst-group acc (val): {worst_acc:.2%}")

            if shortcut_detected is None:
                summary_lines.append(
                    "⚠️  SSA inconclusive: could not determine shortcut detection (insufficient metrics)"
                )
            else:
                if shortcut_detected:
                    summary_lines.append(
                        "🚨 Shortcut detected by SSA (worst-group gap exceeds threshold)"
                    )
                else:
                    summary_lines.append(
                        "✓ No shortcut detected by SSA (worst-group gap below threshold)"
                    )

            risk_indicators = []
            if shortcut_detected:
                risk_indicators.append("SSA detected large worst-group performance gap")

            result = {
                "detector": detector,
                "report": detail,
                "shortcut_detected": shortcut_detected,
                "summary_title": "SSA (Spread Spurious Attribute)",
                "summary_lines": summary_lines,
                "risk_indicators": risk_indicators,
                "success": True,
            }
            return result
        except Exception as exc:
            warnings.warn(f"SSA analysis failed: {exc}", stacklevel=2)
            return {"success": False, "error": str(exc)}
