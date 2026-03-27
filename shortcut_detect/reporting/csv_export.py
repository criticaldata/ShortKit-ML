"""
CSV export functionality for shortcut detection results.
"""

import os
from typing import TYPE_CHECKING

import numpy as np
import pandas as pd

if TYPE_CHECKING:
    from ..comparison.runner import ComparisonResult
    from ..unified import ShortcutDetector


def export_to_csv(detector: "ShortcutDetector", output_dir: str) -> dict:
    """
    Export detection results to CSV files.

    Creates multiple CSV files in the output directory:
    - overall_summary.csv: Overall detection summary
    - hbac_cluster_purities.csv: HBAC cluster analysis
    - hbac_dimension_importance.csv: Important dimensions from HBAC
    - probe_results.csv: Probe accuracy and predictions
    - statistical_pvalues.csv: P-values from statistical tests

    Args:
        detector: Fitted ShortcutDetector instance
        output_dir: Directory to save CSV files

    Returns:
        Dictionary mapping file names to their paths
    """
    if not detector.results_:
        raise ValueError("Detector must be fitted before exporting to CSV")

    # Create output directory
    os.makedirs(output_dir, exist_ok=True)
    exported_files = {}

    # 0. Embeddings for mitigation (M01, M04, M05, M06)
    emb = detector.embeddings_
    labels = detector.labels_
    group_labels = detector.group_labels_ if detector.group_labels_ is not None else labels
    n, d = emb.shape
    cols = {f"embedding_{j}": emb[:, j] for j in range(d)}
    cols["task_label"] = labels
    cols["group_label"] = group_labels
    emb_df = pd.DataFrame(cols)
    emb_path = os.path.join(output_dir, "embeddings_for_mitigation.csv")
    emb_df.to_csv(emb_path, index=False)
    exported_files["embeddings_for_mitigation"] = emb_path

    # 1. Overall Summary
    summary_data = {
        "n_samples": [len(detector.embeddings_)],
        "n_dimensions": [detector.embeddings_.shape[1]],
        "n_unique_labels": [len(np.unique(detector.labels_))],
        "methods_used": [", ".join(detector.methods)],
        "overall_risk": [_extract_risk_level(detector)],
    }

    # Add method-specific summaries
    if "hbac" in detector.results_ and detector.results_["hbac"]["success"]:
        report = detector.results_["hbac"]["report"]
        summary_data["hbac_shortcut_detected"] = [report["has_shortcut"]["exists"]]
        summary_data["hbac_confidence"] = [report["has_shortcut"]["confidence"]]
        summary_data["hbac_n_clusters"] = [len(report["cluster_purities"])]

    if "probe" in detector.results_ and detector.results_["probe"]["success"]:
        metrics = detector.results_["probe"]["results"]["metrics"]
        summary_data["probe_metric"] = [metrics.get("metric")]
        summary_data["probe_metric_value"] = [metrics.get("metric_value")]

    if "statistical" in detector.results_ and detector.results_["statistical"]["success"]:
        stat_result = detector.results_["statistical"]
        if "by_attribute" in stat_result:
            for attr_name, sub in stat_result["by_attribute"].items():
                if sub.get("success"):
                    sig = sub.get("significant_features", {})
                    n = sum(1 for v in sig.values() if v is not None and len(v) > 0)
                    summary_data[f"statistical_{attr_name}_n_significant"] = [n]
        else:
            significant = stat_result["significant_features"]
            n_sig_comparisons = sum(1 for v in significant.values() if v is not None and len(v) > 0)
            summary_data["statistical_n_significant_comparisons"] = [n_sig_comparisons]

    if "geometric" in detector.results_ and detector.results_["geometric"]["success"]:
        geo_result = detector.results_["geometric"]
        if "by_attribute" in geo_result:
            for attr_name, sub in geo_result["by_attribute"].items():
                if sub.get("success"):
                    s = sub.get("summary", {})
                    summary_data[f"geometric_{attr_name}_risk_level"] = [s.get("risk_level")]
        else:
            geo_summary = geo_result.get("summary", {})
            summary_data["geometric_risk_level"] = [geo_summary.get("risk_level")]
            summary_data["geometric_num_high_effect_pairs"] = [
                geo_summary.get("num_high_effect_pairs")
            ]
            summary_data["geometric_num_overlap_pairs"] = [geo_summary.get("num_overlap_pairs")]

    if "equalized_odds" in detector.results_ and detector.results_["equalized_odds"]["success"]:
        eo_result = detector.results_["equalized_odds"]
        if "by_attribute" in eo_result:
            for attr_name, sub in eo_result["by_attribute"].items():
                if sub.get("success") and sub.get("report"):
                    r = sub["report"]
                    summary_data[f"equalized_odds_{attr_name}_tpr_gap"] = [r.tpr_gap]
                    summary_data[f"equalized_odds_{attr_name}_fpr_gap"] = [r.fpr_gap]
                    summary_data[f"equalized_odds_{attr_name}_risk_level"] = [r.risk_level]
        else:
            fairness_report = eo_result["report"]
            summary_data["equalized_odds_tpr_gap"] = [fairness_report.tpr_gap]
            summary_data["equalized_odds_fpr_gap"] = [fairness_report.fpr_gap]
            summary_data["equalized_odds_risk_level"] = [fairness_report.risk_level]

    if (
        "demographic_parity" in detector.results_
        and detector.results_["demographic_parity"]["success"]
    ):
        dp_result = detector.results_["demographic_parity"]
        if "by_attribute" in dp_result:
            for attr_name, sub in dp_result["by_attribute"].items():
                if sub.get("success") and sub.get("report"):
                    r = sub["report"]
                    summary_data[f"demographic_parity_{attr_name}_gap"] = [r.dp_gap]
                    summary_data[f"demographic_parity_{attr_name}_risk_level"] = [r.risk_level]
        else:
            dp_report = dp_result["report"]
            summary_data["demographic_parity_gap"] = [dp_report.dp_gap]
            summary_data["demographic_parity_overall_positive_rate"] = [
                dp_report.overall_positive_rate
            ]
            summary_data["demographic_parity_risk_level"] = [dp_report.risk_level]

    if "intersectional" in detector.results_ and detector.results_["intersectional"]["success"]:
        int_report = detector.results_["intersectional"]["report"]
        summary_data["intersectional_tpr_gap"] = [int_report.tpr_gap]
        summary_data["intersectional_fpr_gap"] = [int_report.fpr_gap]
        summary_data["intersectional_dp_gap"] = [int_report.dp_gap]
        summary_data["intersectional_risk_level"] = [int_report.risk_level]

    if (
        "bias_direction_pca" in detector.results_
        and detector.results_["bias_direction_pca"]["success"]
    ):
        bd_result = detector.results_["bias_direction_pca"]
        if "by_attribute" in bd_result:
            for attr_name, sub in bd_result["by_attribute"].items():
                if sub.get("success") and sub.get("report"):
                    r = sub["report"]
                    summary_data[f"bias_direction_{attr_name}_gap"] = [r.projection_gap]
        else:
            pca_report = bd_result["report"]
            summary_data["bias_direction_gap"] = [pca_report.projection_gap]
            summary_data["bias_direction_explained_variance"] = [pca_report.explained_variance]

    if (
        "gradcam_mask_overlap" in detector.results_
        and detector.results_["gradcam_mask_overlap"]["success"]
    ):
        gmo_summary = detector.results_["gradcam_mask_overlap"].get("metrics", {})
        summary_data["gradcam_mask_overlap_samples"] = [gmo_summary.get("n_samples")]
        summary_data["gradcam_mask_overlap_attention_in_mask_mean"] = [
            gmo_summary.get("attention_in_mask_mean")
        ]
        summary_data["gradcam_mask_overlap_dice_mean"] = [gmo_summary.get("dice_mean")]
        summary_data["gradcam_mask_overlap_iou_mean"] = [gmo_summary.get("iou_mean")]

    if (
        "early_epoch_clustering" in detector.results_
        and detector.results_["early_epoch_clustering"]["success"]
    ):
        eec_report = detector.results_["early_epoch_clustering"]["report"]
        summary_data["early_epoch_entropy"] = [eec_report.size_entropy]
        summary_data["early_epoch_minority_ratio"] = [eec_report.minority_ratio]
        summary_data["early_epoch_largest_gap"] = [eec_report.largest_gap]
        summary_data["early_epoch_risk_level"] = [eec_report.risk_level]

    if "cav" in detector.results_ and detector.results_["cav"]["success"]:
        cav_summary = detector.results_["cav"].get("metrics", {})
        summary_data["cav_n_concepts"] = [cav_summary.get("n_concepts")]
        summary_data["cav_n_tested"] = [cav_summary.get("n_tested")]
        summary_data["cav_max_tcav_score"] = [cav_summary.get("max_tcav_score")]
        summary_data["cav_max_concept_quality"] = [cav_summary.get("max_concept_quality")]
        summary_data["cav_n_flagged"] = [cav_summary.get("n_flagged")]

    if "sis" in detector.results_ and detector.results_["sis"]["success"]:
        sis_metrics = detector.results_["sis"].get("metrics", {})
        summary_data["sis_mean_size"] = [sis_metrics.get("mean_sis_size")]
        summary_data["sis_median_size"] = [sis_metrics.get("median_sis_size")]
        summary_data["sis_frac_dimensions"] = [sis_metrics.get("frac_dimensions")]
        summary_data["sis_risk_level"] = [detector.results_["sis"].get("risk_level")]

    if "groupdro" in detector.results_ and detector.results_["groupdro"]["success"]:
        gdro_result = detector.results_["groupdro"]
        if "by_attribute" in gdro_result:
            for attr_name, sub in gdro_result["by_attribute"].items():
                if sub.get("success"):
                    rep = sub.get("report", {})
                    final = rep.get("final", {})
                    summary_data[f"groupdro_{attr_name}_worst_group_acc"] = [
                        final.get("worst_group_acc")
                    ]
        else:
            rep = gdro_result["report"]
            final = rep.get("final", {})
            summary_data["groupdro_avg_acc"] = [final.get("avg_acc")]
            summary_data["groupdro_worst_group_acc"] = [final.get("worst_group_acc")]

    if "gce" in detector.results_ and detector.results_["gce"]["success"]:
        gce_report = detector.results_["gce"]["report"]
        summary_data["gce_n_minority"] = [gce_report.n_minority]
        summary_data["gce_minority_ratio"] = [gce_report.minority_ratio]
        summary_data["gce_loss_mean"] = [gce_report.loss_mean]
        summary_data["gce_loss_std"] = [gce_report.loss_std]
        summary_data["gce_risk_level"] = [gce_report.risk_level]

    if "causal_effect" in detector.results_ and detector.results_["causal_effect"]["success"]:
        ce_metrics = detector.results_["causal_effect"].get("metrics", {})
        summary_data["causal_effect_n_attributes"] = [ce_metrics.get("n_attributes")]
        summary_data["causal_effect_n_spurious"] = [ce_metrics.get("n_spurious")]
        summary_data["causal_effect_spurious_threshold"] = [ce_metrics.get("spurious_threshold")]

    if "vae" in detector.results_ and detector.results_["vae"]["success"]:
        vae_metrics = detector.results_["vae"].get("metrics", {})
        summary_data["vae_latent_dim"] = [vae_metrics.get("latent_dim")]
        summary_data["vae_n_flagged"] = [vae_metrics.get("n_flagged")]
        summary_data["vae_max_predictiveness"] = [vae_metrics.get("max_predictiveness")]

    if "frequency" in detector.results_ and detector.results_["frequency"]["success"]:
        freq_report = detector.results_["frequency"]["report"]
        freq_metrics = freq_report.get("metrics", {})
        freq_detail = freq_report.get("report", {})
        summary_data["frequency_probe_accuracy"] = [freq_metrics.get("probe_accuracy")]
        summary_data["frequency_n_shortcut_classes"] = [freq_metrics.get("n_shortcut_classes")]
        summary_data["frequency_top_percent"] = [freq_metrics.get("top_percent")]
        summary_data["frequency_shortcut_classes"] = [str(freq_detail.get("shortcut_classes", []))]

    summary_df = pd.DataFrame(summary_data)
    summary_path = os.path.join(output_dir, "overall_summary.csv")
    summary_df.to_csv(summary_path, index=False)
    exported_files["overall_summary"] = summary_path
    print(f"  ✅ Saved: {summary_path}")

    # 2. HBAC Results
    if "hbac" in detector.results_ and detector.results_["hbac"]["success"]:
        report = detector.results_["hbac"]["report"]

        # Cluster purities
        cluster_df = pd.DataFrame(report["cluster_purities"])
        cluster_path = os.path.join(output_dir, "hbac_cluster_purities.csv")
        cluster_df.to_csv(cluster_path, index=False)
        exported_files["hbac_cluster_purities"] = cluster_path
        print(f"  ✅ Saved: {cluster_path}")

        # Dimension importance
        dim_path = os.path.join(output_dir, "hbac_dimension_importance.csv")
        report["dimension_importance"].to_csv(dim_path, index=False)
        exported_files["hbac_dimension_importance"] = dim_path
        print(f"  ✅ Saved: {dim_path}")

    # 3. Probe Results
    if "probe" in detector.results_ and detector.results_["probe"]["success"]:
        probe_det = detector.results_["probe"]["detector"]
        try:
            predictions = probe_det.predict(detector.embeddings_)
        except Exception:
            predictions = None
        if predictions is not None:
            probe_data = {
                "sample_index": list(range(len(detector.embeddings_))),
                "true_label": detector.group_labels_,
                "predicted_label": predictions,
            }
            probe_df = pd.DataFrame(probe_data)
            probe_path = os.path.join(output_dir, "probe_predictions.csv")
            probe_df.to_csv(probe_path, index=False)
            exported_files["probe_predictions"] = probe_path
            print(f"  ✅ Saved: {probe_path}")

        # Probe summary
        metrics = detector.results_["probe"]["results"]["metrics"]
        probe_summary = pd.DataFrame(
            {"metric": [metrics.get("metric", "metric")], "value": [metrics.get("metric_value")]}
        )
        probe_summary_path = os.path.join(output_dir, "probe_summary.csv")
        probe_summary.to_csv(probe_summary_path, index=False)
        exported_files["probe_summary"] = probe_summary_path
        print(f"  ✅ Saved: {probe_summary_path}")

    # 4. Statistical Test Results
    if "statistical" in detector.results_ and detector.results_["statistical"]["success"]:
        p_values_dict = detector.results_["statistical"].get("p_values")
        significant_dict = detector.results_["statistical"].get("significant_features")
        if p_values_dict is None or significant_dict is None:
            print("  ⚠️  Skipping statistical CSV export (multi-attribute mode)")
            p_values_dict = None

        # Create a combined dataframe with all comparisons
        all_pvals = []
        if p_values_dict is None:
            p_values_dict = {}
        for comparison, p_vals in p_values_dict.items():
            if p_vals is not None:
                sig_features = significant_dict.get(comparison, [])
                for dim_idx, p_val in enumerate(p_vals):
                    all_pvals.append(
                        {
                            "comparison": comparison,
                            "dimension": dim_idx,
                            "p_value": p_val,
                            "significant_at_0.05": dim_idx
                            in (sig_features if sig_features else []),
                        }
                    )

        if all_pvals:
            pval_df = pd.DataFrame(all_pvals)
            pval_path = os.path.join(output_dir, "statistical_pvalues.csv")
            pval_df.to_csv(pval_path, index=False)
            exported_files["statistical_pvalues"] = pval_path
            print(f"  ✅ Saved: {pval_path}")

            # Create summary per comparison
            comparison_summary = []
            for comparison in p_values_dict.keys():
                if p_values_dict[comparison] is not None:
                    sig_feats = significant_dict.get(comparison, [])
                    comparison_summary.append(
                        {
                            "comparison": comparison,
                            "n_features_tested": len(p_values_dict[comparison]),
                            "n_significant_features": len(sig_feats) if sig_feats else 0,
                        }
                    )

            if comparison_summary:
                comp_summary_df = pd.DataFrame(comparison_summary)
                comp_summary_path = os.path.join(output_dir, "statistical_summary.csv")
                comp_summary_df.to_csv(comp_summary_path, index=False)
                exported_files["statistical_summary"] = comp_summary_path
                print(f"  ✅ Saved: {comp_summary_path}")

    # 5. Bias Direction (PCA) Results
    if (
        "bias_direction_pca" in detector.results_
        and detector.results_["bias_direction_pca"]["success"]
    ):
        bd_result = detector.results_["bias_direction_pca"]
        if "by_attribute" in bd_result:
            for attr_name, sub in bd_result["by_attribute"].items():
                if sub.get("success") and sub.get("report"):
                    report = sub["report"]
                    proj_rows = [
                        {
                            "attribute": attr_name,
                            "group": g,
                            "projection": m["projection"],
                            "support": m["support"],
                        }
                        for g, m in report.group_projections.items()
                    ]
                    if proj_rows:
                        proj_df = pd.DataFrame(proj_rows)
                        proj_path = os.path.join(
                            output_dir, f"bias_direction_{attr_name}_projections.csv"
                        )
                        proj_df.to_csv(proj_path, index=False)
                        exported_files[f"bias_direction_{attr_name}_projections"] = proj_path
            # Also write per-attribute summary
            summary_rows = []
            for attr_name, sub in bd_result["by_attribute"].items():
                if sub.get("success") and sub.get("report"):
                    r = sub["report"]
                    summary_rows.append(
                        {
                            "attribute": attr_name,
                            "projection_gap": r.projection_gap,
                            "explained_variance": r.explained_variance,
                            "risk_level": r.risk_level,
                        }
                    )
            if summary_rows:
                pd.DataFrame(summary_rows).to_csv(
                    os.path.join(output_dir, "bias_direction_per_attribute.csv"), index=False
                )
        else:
            report = bd_result["report"]
            proj_rows = [
                {"group": g, "projection": m["projection"], "support": m["support"]}
                for g, m in report.group_projections.items()
            ]
            if proj_rows:
                proj_df = pd.DataFrame(proj_rows)
                proj_path = os.path.join(output_dir, "bias_direction_projections.csv")
                proj_df.to_csv(proj_path, index=False)
                exported_files["bias_direction_projections"] = proj_path
                print(f"  ✅ Saved: {proj_path}")

    # 6. Geometric Analysis Results
    if "geometric" in detector.results_ and detector.results_["geometric"]["success"]:
        geo_result = detector.results_["geometric"]

        def _as_dict(item):
            if isinstance(item, dict):
                return item
            return item.__dict__ if hasattr(item, "__dict__") else {}

        if "by_attribute" in geo_result:
            for attr_name, sub in geo_result["by_attribute"].items():
                if sub.get("success"):
                    bias_pairs = sub.get("bias_pairs", [])
                    subspace_pairs = sub.get("subspace_pairs", [])
                    if bias_pairs:
                        rows = [dict(_as_dict(p), attribute=attr_name) for p in bias_pairs]
                        bias_df = pd.DataFrame(rows)
                        bias_path = os.path.join(
                            output_dir, f"geometric_{attr_name}_bias_pairs.csv"
                        )
                        bias_df.to_csv(bias_path, index=False)
                        exported_files[f"geometric_{attr_name}_bias_pairs"] = bias_path
                    if subspace_pairs:
                        rows = [dict(_as_dict(p), attribute=attr_name) for p in subspace_pairs]
                        subspace_df = pd.DataFrame(rows)
                        subspace_path = os.path.join(
                            output_dir, f"geometric_{attr_name}_subspace_pairs.csv"
                        )
                        subspace_df.to_csv(subspace_path, index=False)
                        exported_files[f"geometric_{attr_name}_subspace_pairs"] = subspace_path
        else:
            bias_pairs = geo_result.get("bias_pairs", [])
            subspace_pairs = geo_result.get("subspace_pairs", [])
            if bias_pairs:
                bias_df = pd.DataFrame([_as_dict(p) for p in bias_pairs])
                bias_path = os.path.join(output_dir, "geometric_bias_pairs.csv")
                bias_df.to_csv(bias_path, index=False)
                exported_files["geometric_bias_pairs"] = bias_path
                print(f"  ✅ Saved: {bias_path}")
            if subspace_pairs:
                subspace_df = pd.DataFrame([_as_dict(p) for p in subspace_pairs])
                subspace_path = os.path.join(output_dir, "geometric_subspace_pairs.csv")
                subspace_df.to_csv(subspace_path, index=False)
                exported_files["geometric_subspace_pairs"] = subspace_path
                print(f"  ✅ Saved: {subspace_path}")

    if (
        "gradcam_mask_overlap" in detector.results_
        and detector.results_["gradcam_mask_overlap"]["success"]
    ):
        gmo_details = detector.results_["gradcam_mask_overlap"].get("details", {})
        per_sample = gmo_details.get("per_sample", [])
        if per_sample:
            gmo_df = pd.DataFrame(per_sample)
            gmo_path = os.path.join(output_dir, "gradcam_mask_overlap_samples.csv")
            gmo_df.to_csv(gmo_path, index=False)
            exported_files["gradcam_mask_overlap_samples"] = gmo_path
            print(f"  ✅ Saved: {gmo_path}")

    if "cav" in detector.results_ and detector.results_["cav"]["success"]:
        cav_report = detector.results_["cav"].get("report", {})
        per_concept = cav_report.get("per_concept", [])
        if per_concept:
            cav_df = pd.DataFrame(per_concept)
            cav_path = os.path.join(output_dir, "cav_concept_scores.csv")
            cav_df.to_csv(cav_path, index=False)
            exported_files["cav_concept_scores"] = cav_path
            print(f"  ✅ Saved: {cav_path}")

    if "sis" in detector.results_ and detector.results_["sis"]["success"]:
        sis_report = detector.results_["sis"].get("report", {})
        sis_sizes = sis_report.get("sis_sizes", [])
        if sis_sizes:
            sis_df = pd.DataFrame({"sample_idx": range(len(sis_sizes)), "sis_size": sis_sizes})
            sis_path = os.path.join(output_dir, "sis_per_sample.csv")
            sis_df.to_csv(sis_path, index=False)
            exported_files["sis_per_sample"] = sis_path
            print(f"  ✅ Saved: {sis_path}")

    if (
        "early_epoch_clustering" in detector.results_
        and detector.results_["early_epoch_clustering"]["success"]
    ):
        eec_report = detector.results_["early_epoch_clustering"]["report"]
        rows = []
        for cluster_id, size in eec_report.cluster_sizes.items():
            rows.append(
                {
                    "cluster": cluster_id,
                    "size": size,
                    "ratio": eec_report.cluster_ratios.get(cluster_id, float("nan")),
                }
            )
        eec_df = pd.DataFrame(rows)
        eec_path = os.path.join(output_dir, "early_epoch_cluster_sizes.csv")
        eec_df.to_csv(eec_path, index=False)
        exported_files["early_epoch_cluster_sizes"] = eec_path
        print(f"  ✅ Saved: {eec_path}")

    if "equalized_odds" in detector.results_ and detector.results_["equalized_odds"]["success"]:
        eo_result = detector.results_["equalized_odds"]
        if "by_attribute" in eo_result:
            rows = []
            for attr_name, sub in eo_result["by_attribute"].items():
                if sub.get("success") and sub.get("report"):
                    r = sub["report"]
                    for group, metrics in r.group_metrics.items():
                        rows.append(
                            {
                                "attribute": attr_name,
                                "group": group,
                                "tpr": metrics["tpr"],
                                "fpr": metrics["fpr"],
                                "support": metrics["support"],
                            }
                        )
            if rows:
                eo_path = os.path.join(output_dir, "equalized_odds_per_attribute.csv")
                pd.DataFrame(rows).to_csv(eo_path, index=False)
                exported_files["equalized_odds_per_attribute"] = eo_path
        else:
            fairness_report = eo_result["report"]
            rows = [
                {
                    "group": g,
                    "tpr": m["tpr"],
                    "fpr": m["fpr"],
                    "support": m["support"],
                    "tp": m.get("tp"),
                    "fp": m.get("fp"),
                    "tn": m.get("tn"),
                    "fn": m.get("fn"),
                }
                for g, m in fairness_report.group_metrics.items()
            ]
            if rows:
                fairness_df = pd.DataFrame(rows)
                fairness_path = os.path.join(output_dir, "equalized_odds_metrics.csv")
                fairness_df.to_csv(fairness_path, index=False)
                exported_files["equalized_odds_metrics"] = fairness_path
                print(f"  ✅ Saved: {fairness_path}")

    if (
        "demographic_parity" in detector.results_
        and detector.results_["demographic_parity"]["success"]
    ):
        dp_result = detector.results_["demographic_parity"]
        if "by_attribute" in dp_result:
            rows = []
            for attr_name, sub in dp_result["by_attribute"].items():
                if sub.get("success") and sub.get("report"):
                    r = sub["report"]
                    for group, metrics in r.group_rates.items():
                        rows.append(
                            {
                                "attribute": attr_name,
                                "group": group,
                                "positive_rate": metrics["positive_rate"],
                                "support": metrics["support"],
                            }
                        )
            if rows:
                dp_df = pd.DataFrame(rows)
                dp_path = os.path.join(output_dir, "demographic_parity_per_attribute.csv")
                dp_df.to_csv(dp_path, index=False)
                exported_files["demographic_parity_per_attribute"] = dp_path
        else:
            dp_report = dp_result["report"]
            rows = [
                {"group": g, "positive_rate": m["positive_rate"], "support": m["support"]}
                for g, m in dp_report.group_rates.items()
            ]
            if rows:
                dp_df = pd.DataFrame(rows)
                dp_path = os.path.join(output_dir, "demographic_parity_metrics.csv")
                dp_df.to_csv(dp_path, index=False)
                exported_files["demographic_parity_metrics"] = dp_path
                print(f"  ✅ Saved: {dp_path}")

    if "intersectional" in detector.results_ and detector.results_["intersectional"]["success"]:
        int_report = detector.results_["intersectional"]["report"]
        rows = []
        for group, metrics in int_report.intersection_metrics.items():
            rows.append(
                {
                    "intersection": group,
                    "tpr": metrics["tpr"],
                    "fpr": metrics["fpr"],
                    "positive_rate": metrics.get("positive_rate", float("nan")),
                    "support": metrics["support"],
                }
            )
        int_df = pd.DataFrame(rows)
        int_path = os.path.join(output_dir, "intersectional_metrics.csv")
        int_df.to_csv(int_path, index=False)
        exported_files["intersectional_metrics"] = int_path
        print(f"  ✅ Saved: {int_path}")

    if "groupdro" in detector.results_ and detector.results_["groupdro"]["success"]:
        gdro_result = detector.results_["groupdro"]
        to_export = (
            gdro_result.get("by_attribute", {"": gdro_result})
            if "by_attribute" in gdro_result
            else {"": gdro_result}
        )
        for attr_name, sub in to_export.items():
            if not sub.get("success"):
                continue
            rep = sub.get("report", {})
            final = rep.get("final", {})
            adv_probs = rep.get("final_adv_probs", None)
            n_groups = rep.get("n_groups", 0)
            gid_map = (
                rep.get("group_id_map", {}) if isinstance(rep.get("group_id_map", {}), dict) else {}
            )
            idx_to_gid = {idx: gid for gid, idx in gid_map.items()} if gid_map else {}

            rows = []
            if isinstance(n_groups, int) and n_groups > 0:
                for i in range(n_groups):
                    gid = idx_to_gid.get(i, i)
                    row = {
                        "group": gid,
                        "avg_acc": final.get(f"avg_acc_group:{i}", float("nan")),
                        "avg_loss": final.get(f"avg_loss_group:{i}", float("nan")),
                        "adv_weight_q": (
                            float(adv_probs[i])
                            if adv_probs is not None and len(adv_probs) > i
                            else float("nan")
                        ),
                    }
                    if attr_name:
                        row["attribute"] = attr_name
                    rows.append(row)
            if rows:
                gd_df = pd.DataFrame(rows)
                gd_path = os.path.join(
                    output_dir,
                    f"groupdro_{attr_name}_metrics.csv" if attr_name else "groupdro_metrics.csv",
                )
                gd_df.to_csv(gd_path, index=False)
                exported_files["groupdro_metrics" + (f"_{attr_name}" if attr_name else "")] = (
                    gd_path
                )
                print(f"  ✅ Saved: {gd_path}")

    # GCE (Generalized Cross Entropy) Results
    if "gce" in detector.results_ and detector.results_["gce"]["success"]:
        gce_result = detector.results_["gce"]
        report = gce_result["report"]
        per_sample_losses = gce_result.get("per_sample_losses")
        is_minority = gce_result.get("is_minority")
        if per_sample_losses is not None:
            gce_df = pd.DataFrame(
                {
                    "sample_index": range(len(per_sample_losses)),
                    "gce_loss": per_sample_losses,
                    "is_minority": (
                        is_minority if is_minority is not None else [False] * len(per_sample_losses)
                    ),
                }
            )
            gce_path = os.path.join(output_dir, "gce_per_sample_losses.csv")
            gce_df.to_csv(gce_path, index=False)
            exported_files["gce_per_sample_losses"] = gce_path
            print(f"  ✅ Saved: {gce_path}")
        gce_summary = pd.DataFrame(
            {
                "metric": [
                    "n_minority",
                    "minority_ratio",
                    "loss_mean",
                    "loss_std",
                    "threshold",
                    "q",
                    "risk_level",
                ],
                "value": [
                    report.n_minority,
                    report.minority_ratio,
                    report.loss_mean,
                    report.loss_std,
                    report.threshold,
                    report.q,
                    report.risk_level,
                ],
            }
        )
        gce_summary_path = os.path.join(output_dir, "gce_summary.csv")
        gce_summary.to_csv(gce_summary_path, index=False)
        exported_files["gce_summary"] = gce_summary_path
        print(f"  ✅ Saved: {gce_summary_path}")

    if "causal_effect" in detector.results_ and detector.results_["causal_effect"]["success"]:
        ce_report = detector.results_["causal_effect"].get("report", {})
        per_attr = ce_report.get("per_attribute", [])
        if per_attr:
            ce_df = pd.DataFrame(per_attr)
            ce_path = os.path.join(output_dir, "causal_effect_per_attribute.csv")
            ce_df.to_csv(ce_path, index=False)
            exported_files["causal_effect_per_attribute"] = ce_path
            print(f"  ✅ Saved: {ce_path}")

    if "vae" in detector.results_ and detector.results_["vae"]["success"]:
        vae_report = detector.results_["vae"].get("report", {})
        per_dim = vae_report.get("per_dimension", [])
        if per_dim:
            vae_df = pd.DataFrame(per_dim)
            vae_path = os.path.join(output_dir, "vae_per_dimension.csv")
            vae_df.to_csv(vae_path, index=False)
            exported_files["vae_per_dimension"] = vae_path
            print(f"  ✅ Saved: {vae_path}")

    if "frequency" in detector.results_ and detector.results_["frequency"]["success"]:
        freq_report = detector.results_["frequency"]["report"]
        detail = freq_report.get("report", {})
        class_rates = detail.get("class_rates", {})
        if class_rates:
            freq_rows = []
            for cls, rates in class_rates.items():
                freq_rows.append(
                    {
                        "class": cls,
                        "tpr": rates.get("tpr"),
                        "fpr": rates.get("fpr"),
                        "support": rates.get("support"),
                        "is_shortcut_class": cls in detail.get("shortcut_classes", []),
                        "top_dims": str(detail.get("top_dims_by_class", {}).get(str(cls), [])),
                    }
                )
            freq_df = pd.DataFrame(freq_rows)
            freq_path = os.path.join(output_dir, "frequency_class_rates.csv")
            freq_df.to_csv(freq_path, index=False)
            exported_files["frequency_class_rates"] = freq_path
            print(f"  ✅ Saved: {freq_path}")

    return exported_files


def _extract_risk_level(detector: "ShortcutDetector") -> str:
    """Extract risk level from overall assessment."""
    assessment = detector._generate_overall_assessment()
    if "HIGH RISK" in assessment:
        return "HIGH"
    elif "MODERATE RISK" in assessment:
        return "MODERATE"
    else:
        return "LOW"


def export_comparison_to_csv(comparison_result: "ComparisonResult", output_dir: str) -> dict:
    """
    Export model comparison results to CSV files.

    Creates:
    - comparison_summary.csv: One row per model, columns for each method's summary metric
    - comparison_{model_id}_full/: Per-model full export (optional, via export_to_csv)

    Args:
        comparison_result: Result from ModelComparisonRunner.run()
        output_dir: Directory to save CSV files

    Returns:
        Dictionary mapping file names to their paths
    """
    os.makedirs(output_dir, exist_ok=True)
    exported_files: dict = {}

    # Summary table
    summary_path = os.path.join(output_dir, "comparison_summary.csv")
    comparison_result.summary_table.to_csv(summary_path, index=False)
    exported_files["comparison_summary"] = summary_path

    # Optional: per-model full export
    for model_id, detector in comparison_result.detectors.items():
        model_dir = os.path.join(output_dir, f"comparison_{model_id}")
        os.makedirs(model_dir, exist_ok=True)
        try:
            per_model = export_to_csv(detector, model_dir)
            for key, path in per_model.items():
                exported_files[f"comparison_{model_id}_{key}"] = path
        except Exception:
            pass  # Skip if export fails for a model

    return exported_files
