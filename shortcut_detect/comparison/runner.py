"""
Runner for comparing shortcut detection across multiple embedding models.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
from typing import Any

import numpy as np
import pandas as pd

from ..embedding_sources import EmbeddingSource
from ..unified import ShortcutDetector


@dataclass
class ComparisonResult:
    """
    Result of model comparison run. Holds detectors per model and summary table.
    """

    model_ids: list[str]
    detectors: dict[str, ShortcutDetector]
    summary_table: pd.DataFrame

    def to_dataframe(self) -> pd.DataFrame:
        """Return the summary table as a DataFrame."""
        return self.summary_table.copy()


def _extract_summary_row(detector: ShortcutDetector) -> dict[str, Any]:
    """Extract a single row of summary metrics from a fitted detector."""
    row: dict[str, Any] = {
        "n_samples": len(detector.embeddings_) if detector.embeddings_ is not None else None,
        "n_dimensions": (
            detector.embeddings_.shape[1] if detector.embeddings_ is not None else None
        ),
    }

    if "hbac" in detector.results_ and detector.results_["hbac"]["success"]:
        report = detector.results_["hbac"]["report"]
        row["hbac_shortcut_detected"] = report["has_shortcut"]["exists"]
        row["hbac_confidence"] = report["has_shortcut"]["confidence"]
        row["hbac_n_clusters"] = len(report["cluster_purities"])

    if "probe" in detector.results_ and detector.results_["probe"]["success"]:
        metrics = detector.results_["probe"]["results"]["metrics"]
        row["probe_metric"] = metrics.get("metric")
        row["probe_metric_value"] = metrics.get("metric_value")
        row["probe_auc"] = metrics.get("metric_value") if metrics.get("metric") == "auc" else None

    if "statistical" in detector.results_ and detector.results_["statistical"]["success"]:
        stat_result = detector.results_["statistical"]
        if "by_attribute" in stat_result:
            for attr_name, sub in stat_result["by_attribute"].items():
                if sub.get("success"):
                    sig = sub.get("significant_features", {})
                    row[f"statistical_{attr_name}_n_significant"] = sum(
                        1 for v in sig.values() if v is not None and len(v) > 0
                    )
        else:
            significant = stat_result["significant_features"]
            row["statistical_n_significant"] = sum(
                1 for v in significant.values() if v is not None and len(v) > 0
            )

    if "geometric" in detector.results_ and detector.results_["geometric"]["success"]:
        geo_result = detector.results_["geometric"]
        if "by_attribute" in geo_result:
            for attr_name, sub in geo_result["by_attribute"].items():
                if sub.get("success"):
                    s = sub.get("summary", {})
                    row[f"geometric_{attr_name}_risk_level"] = s.get("risk_level")
        else:
            geo_summary = geo_result.get("summary", {})
            row["geometric_risk_level"] = geo_summary.get("risk_level")
            row["geometric_num_high_effect_pairs"] = geo_summary.get("num_high_effect_pairs")

    if "equalized_odds" in detector.results_ and detector.results_["equalized_odds"]["success"]:
        eo_result = detector.results_["equalized_odds"]
        if "by_attribute" in eo_result:
            for attr_name, sub in eo_result["by_attribute"].items():
                if sub.get("success") and sub.get("report"):
                    r = sub["report"]
                    row[f"equalized_odds_{attr_name}_tpr_gap"] = r.tpr_gap
                    row[f"equalized_odds_{attr_name}_fpr_gap"] = r.fpr_gap
        else:
            r = eo_result["report"]
            row["equalized_odds_tpr_gap"] = r.tpr_gap
            row["equalized_odds_fpr_gap"] = r.fpr_gap
            row["equalized_odds_risk_level"] = r.risk_level

    if (
        "demographic_parity" in detector.results_
        and detector.results_["demographic_parity"]["success"]
    ):
        dp_result = detector.results_["demographic_parity"]
        if "by_attribute" in dp_result:
            for attr_name, sub in dp_result["by_attribute"].items():
                if sub.get("success") and sub.get("report"):
                    r = sub["report"]
                    row[f"demographic_parity_{attr_name}_gap"] = r.dp_gap
        else:
            r = dp_result["report"]
            row["demographic_parity_gap"] = r.dp_gap
            row["demographic_parity_risk_level"] = r.risk_level

    if "intersectional" in detector.results_ and detector.results_["intersectional"]["success"]:
        r = detector.results_["intersectional"]["report"]
        row["intersectional_tpr_gap"] = r.tpr_gap
        row["intersectional_fpr_gap"] = r.fpr_gap
        row["intersectional_dp_gap"] = r.dp_gap
        row["intersectional_risk_level"] = r.risk_level

    if (
        "bias_direction_pca" in detector.results_
        and detector.results_["bias_direction_pca"]["success"]
    ):
        bd_result = detector.results_["bias_direction_pca"]
        if "by_attribute" in bd_result:
            for attr_name, sub in bd_result["by_attribute"].items():
                if sub.get("success") and sub.get("report"):
                    r = sub["report"]
                    row[f"bias_direction_pca_{attr_name}_gap"] = r.projection_gap
        else:
            r = bd_result["report"]
            row["bias_direction_pca_gap"] = r.projection_gap
            row["bias_direction_pca_risk_level"] = r.risk_level

    if (
        "early_epoch_clustering" in detector.results_
        and detector.results_["early_epoch_clustering"]["success"]
    ):
        r = detector.results_["early_epoch_clustering"]["report"]
        row["early_epoch_risk_level"] = r.risk_level

    if "groupdro" in detector.results_ and detector.results_["groupdro"]["success"]:
        gdro_result = detector.results_["groupdro"]
        if "by_attribute" in gdro_result:
            for attr_name, sub in gdro_result["by_attribute"].items():
                if sub.get("success"):
                    rep = sub.get("report", {})
                    final = rep.get("final", {})
                    row[f"groupdro_{attr_name}_worst_group_acc"] = final.get("worst_group_acc")
        else:
            rep = gdro_result["report"]
            final = rep.get("final", {})
            row["groupdro_avg_acc"] = final.get("avg_acc")
            row["groupdro_worst_group_acc"] = final.get("worst_group_acc")

    if "gce" in detector.results_ and detector.results_["gce"]["success"]:
        r = detector.results_["gce"]["report"]
        row["gce_risk_level"] = r.risk_level
        row["gce_minority_ratio"] = r.minority_ratio

    if "sis" in detector.results_ and detector.results_["sis"]["success"]:
        sis_metrics = detector.results_["sis"].get("metrics", {})
        row["sis_mean_size"] = sis_metrics.get("mean_sis_size")
        row["sis_median_size"] = sis_metrics.get("median_sis_size")
        row["sis_risk_level"] = detector.results_["sis"].get("risk_level")

    return row


class ModelComparisonRunner:
    """
    Runs shortcut detection across multiple embedding models/sources and aggregates results.
    """

    def __init__(
        self,
        methods: list[str] | None = None,
        seed: int = 42,
        **detector_kwargs: Any,
    ) -> None:
        """
        Args:
            methods: Detection methods to run. Defaults to core methods.
            seed: Random seed.
            **detector_kwargs: Passed to ShortcutDetector (e.g., statistical_correction, etc.).
        """
        self.methods = methods or [
            "hbac",
            "probe",
            "statistical",
            "geometric",
            "equalized_odds",
            "demographic_parity",
            "bias_direction_pca",
        ]
        self.seed = seed
        self.detector_kwargs = detector_kwargs

    def run(
        self,
        model_sources: list[tuple[str, EmbeddingSource] | tuple[str, np.ndarray]],
        labels: np.ndarray,
        group_labels: np.ndarray | None = None,
        raw_inputs: Sequence[Any] | None = None,
        extra_labels: dict[str, np.ndarray] | None = None,
        splits: dict[str, np.ndarray] | None = None,
    ) -> ComparisonResult:
        """
        Run shortcut detection for each model and collect results.

        Args:
            model_sources: List of (model_id, EmbeddingSource) or (model_id, embeddings array).
                For EmbeddingSource: raw_inputs must be provided.
                For embeddings array: used directly.
            labels: Task labels (n_samples,).
            group_labels: Optional group labels. Defaults to labels.
            raw_inputs: Required when model_sources contain EmbeddingSource. Same order as labels.
            extra_labels: Optional extra labels for intersectional, SSA, etc.
            splits: Optional splits for SSA.

        Returns:
            ComparisonResult with detectors and summary table.
        """
        detectors: dict[str, ShortcutDetector] = {}
        rows: list[dict[str, Any]] = []

        for model_id, source in model_sources:
            if isinstance(source, np.ndarray):
                embeddings = np.asarray(source, dtype=np.float32)
                if embeddings.ndim != 2:
                    raise ValueError(
                        f"Embeddings for model '{model_id}' must be 2D, got shape {embeddings.shape}"
                    )
                if embeddings.shape[0] != len(labels):
                    raise ValueError(
                        f"Embeddings for model '{model_id}' must have {len(labels)} rows, got {embeddings.shape[0]}"
                    )
            else:
                if raw_inputs is None:
                    raise ValueError(
                        "raw_inputs required when model_sources contain EmbeddingSource"
                    )
                embeddings = source.generate(list(raw_inputs))
                if len(embeddings) != len(labels):
                    raise ValueError(
                        f"Embeddings from model '{model_id}' must have {len(labels)} rows, got {len(embeddings)}"
                    )

            detector = ShortcutDetector(
                methods=self.methods,
                seed=self.seed,
                **self.detector_kwargs,
            )
            detector.fit(
                embeddings=embeddings,
                labels=labels,
                group_labels=group_labels,
                splits=splits,
                extra_labels=extra_labels,
            )
            detectors[model_id] = detector
            row = _extract_summary_row(detector)
            row["model_id"] = model_id
            rows.append(row)

        df = pd.DataFrame(rows)
        # Move model_id to first column
        if "model_id" in df.columns:
            cols = ["model_id"] + [c for c in df.columns if c != "model_id"]
            df = df[cols]

        return ComparisonResult(
            model_ids=list(detectors.keys()),
            detectors=detectors,
            summary_table=df,
        )
