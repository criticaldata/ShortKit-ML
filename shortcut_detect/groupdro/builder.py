"""Builder for GroupDRO detector."""

import warnings
from typing import Any

import numpy as np

from ..base_builder import BaseDetector


class GroupDRODetectorBuilder(BaseDetector):
    def build(self):
        from .groupdro import GroupDROConfig, GroupDRODetector

        cfg = GroupDROConfig(
            n_epochs=self.kwargs.get("groupdro_n_epochs", 10),
            batch_size=self.kwargs.get("groupdro_batch_size", 128),
            lr=self.kwargs.get("groupdro_lr", 1e-3),
            weight_decay=self.kwargs.get("groupdro_weight_decay", 5e-5),
            momentum=self.kwargs.get("groupdro_momentum", 0.9),
            num_workers=self.kwargs.get("groupdro_num_workers", 0),
            loader_factory=self.kwargs.get("groupdro_loader_factory"),
            stage_loader_overrides=self.kwargs.get("groupdro_stage_loader_overrides"),
            robust=self.kwargs.get("groupdro_robust", True),
            alpha=self.kwargs.get("groupdro_alpha", 0.2),
            gamma=self.kwargs.get("groupdro_gamma", 0.1),
            robust_step_size=self.kwargs.get("groupdro_robust_step_size", 0.01),
            use_normalized_loss=self.kwargs.get("groupdro_use_normalized_loss", False),
            btl=self.kwargs.get("groupdro_btl", False),
            minimum_variational_weight=self.kwargs.get("groupdro_minimum_variational_weight", 0.0),
            generalization_adjustment=self.kwargs.get("groupdro_generalization_adjustment", None),
            automatic_adjustment=self.kwargs.get("groupdro_automatic_adjustment", False),
            hidden_dim=self.kwargs.get("groupdro_hidden_dim", None),
            dropout=self.kwargs.get("groupdro_dropout", 0.0),
            val_fraction=self.kwargs.get("groupdro_val_fraction", 0.1),
            seed=self.seed,
            device=self.kwargs.get("groupdro_device", None),
        )
        return GroupDRODetector(cfg)

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
        print("Running GroupDRO...")
        detector = self.build()
        try:
            detector.fit(embeddings, labels, group_labels)
            report = detector.get_report().get("report", {})
            final = report.get("final", {})
            summary_lines = [
                f"Groups: {report.get('n_groups')}",
                f"Avg acc (full): {final.get('avg_acc', float('nan')):.2%}",
                f"Worst-group acc (full): {final.get('worst_group_acc', float('nan')):.2%}",
            ]
            return {
                "detector": detector,
                "report": report,
                "summary_title": "GroupDRO (Worst-Group Robustness)",
                "summary_lines": summary_lines,
                "risk_indicators": [],
                "success": True,
            }
        except Exception as exc:
            warnings.warn(f"GroupDRO analysis failed: {exc}", stacklevel=2)
            return {"success": False, "error": str(exc)}

    def run_from_loader(
        self,
        loader: Any,
        feature_names: list[str] | None = None,
        protected_labels: np.ndarray | None = None,
        splits: dict[str, np.ndarray] | None = None,
        extra_labels: dict[str, np.ndarray] | None = None,
    ) -> dict[str, Any]:
        data = loader() if callable(loader) else loader
        detector = self.build()

        if isinstance(data, dict):
            try:
                if "train_loader" in data:
                    detector.fit_loaders(
                        data["train_loader"],
                        val_loader=data["val_loader"],
                        target_extractor=data.get("target_extractor"),
                        group_extractor=data.get("group_extractor"),
                        data_spec=data.get("data_spec"),
                    )
                elif "train_dataset" in data:
                    detector.fit_dataset(
                        data["train_dataset"],
                        val_dataset=data.get("val_dataset"),
                        target_extractor=data.get("target_extractor"),
                        group_extractor=data.get("group_extractor"),
                        data_spec=data.get("data_spec"),
                    )
                else:
                    return super().run_from_loader(
                        loader=data,
                        feature_names=feature_names,
                        protected_labels=protected_labels,
                        splits=splits,
                        extra_labels=extra_labels,
                    )

                report = detector.get_report().get("report", {})
                final = report.get("final", {})
                summary_lines = [
                    f"Groups: {report.get('n_groups')}",
                    f"Avg acc (final): {final.get('avg_acc', float('nan')):.2%}",
                    f"Worst-group acc (final): {final.get('worst_group_acc', float('nan')):.2%}",
                ]
                return {
                    "detector": detector,
                    "report": report,
                    "summary_title": "GroupDRO (Worst-Group Robustness)",
                    "summary_lines": summary_lines,
                    "risk_indicators": [],
                    "success": True,
                }
            except Exception as exc:
                warnings.warn(f"GroupDRO analysis failed: {exc}", stacklevel=2)
                return {"success": False, "error": str(exc)}

        return super().run_from_loader(
            loader=data,
            feature_names=feature_names,
            protected_labels=protected_labels,
            splits=splits,
            extra_labels=extra_labels,
        )
