"""Builder for probe-based detector."""

import warnings
from typing import Any

import numpy as np

from ..base_builder import BaseDetector
from .probe_factory import ProbeDetectorFactory, ProbeFactoryContext


class ProbeDetectorBuilder(BaseDetector):
    def build(self):
        backend = self.kwargs.get("probe_backend", "sklearn")
        ctx = ProbeFactoryContext(seed=self.seed, kwargs=dict(self.kwargs))
        return ProbeDetectorFactory.create(backend=backend, ctx=ctx)

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
        print("Running probe-based detection...")

        detector = self.build()

        try:
            detector.fit(embeddings, group_labels)
            results = detector.get_report()

            metric_value = results["metrics"].get("metric_value")
            metric_name = results["metrics"].get("metric")
            shortcut = results.get("shortcut_detected")

            if shortcut is True:
                risk_line = "⚠️  Probe performance exceeds threshold — strong shortcut signal"
            elif shortcut is False:
                risk_line = "✓ Probe performance below threshold — weak shortcut signal"
            else:
                risk_line = "ℹ️  Probe signal inconclusive"

            risk_indicators = []
            if shortcut is True and isinstance(metric_value, int | float):
                risk_indicators.append(
                    f"{metric_name}={float(metric_value):.1%} (threshold exceeded)"
                )

            summary_metric = (
                f"Probe {metric_name}: {float(metric_value):.2%}"
                if isinstance(metric_value, int | float)
                else f"Probe {metric_name}: {metric_value}"
            )

            return {
                "detector": detector,
                "results": results,
                "summary_title": "Probe-based Detection",
                "summary_lines": [
                    summary_metric,
                    risk_line,
                ],
                "risk_indicators": risk_indicators,
                "success": True,
            }

        except Exception as exc:
            warnings.warn(f"Probe detection failed: {exc}", stacklevel=2)
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
        backend = self.kwargs.get("probe_backend", "sklearn")
        detector = self.build()

        # Native streaming path for torch backend.
        if backend == "torch" and isinstance(data, dict):
            try:
                if "train_loader" in data:
                    detector.fit_loaders(
                        data["train_loader"],
                        val_loader=data["val_loader"],
                        target_extractor=data.get("target_extractor"),
                        data_spec=data.get("data_spec"),
                    )
                elif "train_dataset" in data:
                    detector.fit_dataset(
                        data["train_dataset"],
                        val_dataset=data.get("val_dataset"),
                        target_extractor=data.get("target_extractor"),
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

                results = detector.get_report()
                metric_value = results["metrics"].get("metric_value")
                metric_name = results["metrics"].get("metric")
                shortcut = results.get("shortcut_detected")

                if shortcut is True:
                    risk_line = "⚠️  Probe performance exceeds threshold — strong shortcut signal"
                elif shortcut is False:
                    risk_line = "✓ Probe performance below threshold — weak shortcut signal"
                else:
                    risk_line = "ℹ️  Probe signal inconclusive"

                risk_indicators = []
                if shortcut is True and isinstance(metric_value, int | float):
                    risk_indicators.append(
                        f"{metric_name}={float(metric_value):.1%} (threshold exceeded)"
                    )

                summary_metric = (
                    f"Probe {metric_name}: {float(metric_value):.2%}"
                    if isinstance(metric_value, int | float)
                    else f"Probe {metric_name}: {metric_value}"
                )

                return {
                    "detector": detector,
                    "results": results,
                    "summary_title": "Probe-based Detection",
                    "summary_lines": [summary_metric, risk_line],
                    "risk_indicators": risk_indicators,
                    "success": True,
                }
            except Exception as exc:
                warnings.warn(f"Probe detection failed: {exc}", stacklevel=2)
                return {"success": False, "error": str(exc)}

        return super().run_from_loader(
            loader=data,
            feature_names=feature_names,
            protected_labels=protected_labels,
            splits=splits,
            extra_labels=extra_labels,
        )
