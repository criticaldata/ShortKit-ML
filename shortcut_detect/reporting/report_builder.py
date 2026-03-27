"""
Report generation for shortcut detection results.
"""

import os
import warnings
from typing import TYPE_CHECKING

import numpy as np

if TYPE_CHECKING:
    from ..unified import ShortcutDetector

from .csv_export import export_to_csv
from .reporters import REPORTER_CLASSES
from .risk_format import build_method_risk, normalize_risk_level
from .visualizations import generate_all_plots


class ReportBuilder:
    """
    Generate comprehensive reports from shortcut detection results.

    Supports HTML and PDF output formats with visualizations and CSV export.
    """

    GITHUB_REPO = "https://github.com/Kqp1227/Shortcut_Detect"

    def __init__(self, detector: "ShortcutDetector"):
        """
        Initialize report builder.

        Args:
            detector: Fitted ShortcutDetector instance
        """
        if not detector.results_:
            raise ValueError("Detector must be fitted before generating reports")

        self.detector = detector
        self.results = detector.results_
        self.plots = {}  # Cache for generated plots
        self._reporters = [cls() for cls in REPORTER_CLASSES]

    def to_html(self, output_path: str, include_visualizations: bool = True):
        """
        Generate interactive HTML report with visualizations.

        Args:
            output_path: Path to save HTML file
            include_visualizations: Whether to include plots
        """
        # Generate plots if needed
        if include_visualizations and not self.plots:
            print("Generating visualizations...")
            self.plots = generate_all_plots(self.detector)

        html_content = self._generate_html(include_visualizations)

        os.makedirs(os.path.dirname(os.path.abspath(output_path)), exist_ok=True)
        with open(output_path, "w") as f:
            f.write(html_content)

        print(f"✅ HTML report saved to: {output_path}")

    def to_csv(self, output_dir: str) -> dict[str, str]:
        """
        Export results to CSV files.

        Args:
            output_dir: Directory to save CSV files

        Returns:
            Dictionary mapping file names to their paths
        """
        print("Exporting to CSV...")
        return export_to_csv(self.detector, output_dir)

    @staticmethod
    def _ensure_homebrew_libs():
        """Ensure Homebrew system libraries are discoverable on macOS.

        weasyprint depends on native libraries (glib, pango, etc.) that
        Homebrew installs under its prefix.  Inside conda / venv
        environments the default DYLD_FALLBACK_LIBRARY_PATH may not
        include the Homebrew lib directory, so we add it once before the
        first dlopen call.
        """
        import sys

        if sys.platform != "darwin":
            return
        import subprocess

        try:
            brew_prefix = subprocess.check_output(
                ["brew", "--prefix"],
                stderr=subprocess.DEVNULL,
                text=True,
            ).strip()
        except (FileNotFoundError, subprocess.CalledProcessError):
            return
        brew_lib = os.path.join(brew_prefix, "lib")
        fallback = os.environ.get("DYLD_FALLBACK_LIBRARY_PATH", "")
        if brew_lib not in fallback:
            os.environ["DYLD_FALLBACK_LIBRARY_PATH"] = (
                f"{brew_lib}:{fallback}" if fallback else brew_lib
            )

    def to_pdf(self, output_path: str, include_visualizations: bool = True):
        """
        Generate PDF report using weasyprint.

        Args:
            output_path: Path to save PDF file
            include_visualizations: Whether to include plots
        """
        self._ensure_homebrew_libs()
        try:
            from weasyprint import HTML
        except ImportError:
            warnings.warn(
                "weasyprint not installed. PDF generation requires weasyprint.\n"
                "Install with: uv pip install weasyprint\n"
                "On macOS also run: brew install pango\n"
                "Generating HTML instead...",
                stacklevel=2,
            )
            html_path = output_path.replace(".pdf", ".html")
            self.to_html(html_path, include_visualizations)
            print(f"📄 HTML generated at: {html_path}")
            print("To convert to PDF, install weasyprint: uv pip install weasyprint")
            return

        # Generate HTML content
        if include_visualizations and not self.plots:
            print("Generating visualizations...")
            self.plots = generate_all_plots(self.detector)

        # Remove 3D plot from PDF version due to interactivity issues
        # 1. Back up the original plots
        original_plots = self.plots

        # 2. Filter out the interactive 3D plot for the PDF version
        if include_visualizations:
            self.plots = {k: v for k, v in original_plots.items() if k != "html_3d"}

        try:
            # 3. Generate content using the filtered plots
            html_content = self._generate_html(include_visualizations)
        finally:
            # 4. Restore original plots
            self.plots = original_plots

        # Convert HTML to PDF
        os.makedirs(os.path.dirname(os.path.abspath(output_path)) or ".", exist_ok=True)
        print("Converting HTML to PDF...")
        HTML(string=html_content).write_pdf(output_path)

        print(f"✅ PDF report saved to: {output_path}")

    def to_markdown(self, output_path: str, include_visualizations: bool = False):
        """
        Generate Markdown report summarizing detection results.

        Args:
            output_path: Path to save Markdown file
            include_visualizations: Placeholder for API parity (plots not embedded)
        """
        content = self._generate_markdown()
        os.makedirs(os.path.dirname(os.path.abspath(output_path)) or ".", exist_ok=True)
        with open(output_path, "w") as f:
            f.write(content)
        print(f"✅ Markdown report saved to: {output_path}")

    def _generate_html(self, include_viz: bool) -> str:
        """Generate HTML report content."""
        html = [
            "<!DOCTYPE html>",
            "<html>",
            "<head>",
            "  <meta charset='utf-8'>",
            "  <title>Shortcut Detection Report</title>",
            "  <style>",
            self._get_css(),
            "  </style>",
            "</head>",
            "<body class='shortcut-report'>",
            "  <div class='container'>",
            "    <h1>🔍 Shortcut Detection Report</h1>",
            self._generate_metadata_html(),
            self._generate_overall_assessment_html(),
        ]

        # Add visualizations early if available
        if include_viz and self.plots:
            html.append(self._generate_visualizations_html())

        html.extend(
            [
                self._generate_hbac_section_html(),
                self._generate_geometric_section_html(),
                self._generate_probe_section_html(),
                self._generate_frequency_section_html(),
                self._generate_statistical_section_html(),
                self._generate_gradcam_mask_overlap_section_html(),
                self._generate_cav_section_html(),
                self._generate_sis_section_html(),
                self._generate_bias_direction_section_html(),
                self._generate_groupdro_section_html(),
                self._generate_fairness_section_html(),
                self._generate_demographic_parity_section_html(),
                self._generate_intersectional_section_html(),
                self._generate_early_epoch_section_html(),
                self._generate_gce_section_html(),
                self._generate_causal_effect_section_html(),
                self._generate_vae_section_html(),
                self._generate_recommendations_html(),
                self._generate_footer_html(),
                "  </div>",
                "</body>",
                "</html>",
            ]
        )

        return "\n".join(html)

    def _generate_markdown(self) -> str:
        """Generate Markdown report content."""
        det = self.detector
        lines = [
            "# 🔍 Shortcut Detection Report",
            "",
            "## Dataset Overview",
            f"- Samples: {len(det.embeddings_):,}",
            f"- Embedding Dimensions: {det.embeddings_.shape[1]}",
            f"- Methods: {', '.join(det.methods)}",
            f"- Unique Labels: {len(np.unique(det.labels_))}",
            "",
            "## 📊 Overall Assessment",
            det._generate_overall_assessment(),
            "",
        ]

        if "hbac" in self.results and self.results["hbac"]["success"]:
            report = self.results["hbac"]["report"]
            shortcut_info = report["has_shortcut"]
            risk = self._risk_payload("hbac")
            lines.extend(
                [
                    "## HBAC (Clustering)",
                    f"- Risk: {risk['risk_label']}",
                    f"- Reason: {risk['risk_reason']}",
                    f"- Shortcut detected: {'YES' if shortcut_info['exists'] else 'NO'}",
                    f"- Types: {', '.join(shortcut_info['types']) if shortcut_info['types'] else 'N/A'}",
                    f"- Clusters found: {len(self.results['hbac']['detector'].clusters_)}",
                    "",
                ]
            )

        if "probe" in self.results and self.results["probe"]["success"]:
            metric_name = self.results["probe"]["results"]["metrics"]["metric"]
            acc = self.results["probe"]["results"]["metrics"]["metric_value"]
            risk = self._risk_payload("probe")
            lines.extend(
                [
                    "## Probe-based Detection",
                    f"- Risk: {risk['risk_label']}",
                    f"- Reason: {risk['risk_reason']}",
                    f"- {metric_name}: {acc:.2%}",
                    "",
                ]
            )

        if "frequency" in self.results and self.results["frequency"]["success"]:
            freq_report = self.results["frequency"]["report"]
            freq_metrics = freq_report.get("metrics", {})
            freq_detail = freq_report.get("report", {})
            risk = self._risk_payload("frequency")
            lines.extend(
                [
                    "## Embedding Frequency Shortcut",
                    f"- Risk: {risk['risk_label']}",
                    f"- Reason: {risk['risk_reason']}",
                    f"- Probe accuracy: {float(freq_metrics.get('probe_accuracy', float('nan'))):.2%}",
                    f"- Shortcut classes: {freq_detail.get('shortcut_classes', [])}",
                    f"- Top-percent: {freq_metrics.get('top_percent')}",
                    "",
                ]
            )

        if "statistical" in self.results and self.results["statistical"]["success"]:
            stat_result = self.results["statistical"]
            risk = self._risk_payload("statistical")
            lines.extend(
                [
                    "## Statistical Testing",
                    f"- Risk: {risk['risk_label']}",
                    f"- Reason: {risk['risk_reason']}",
                ]
            )
            if "by_attribute" in stat_result:
                for attr_name, sub in stat_result["by_attribute"].items():
                    if sub.get("success"):
                        sig = sub.get("significant_features", {})
                        details = [
                            f"  - {name}: {len(f) if f else 0} significant"
                            for name, f in sig.items()
                            if f is not None
                        ] or ["  - No significant group differences"]
                        lines.append(f"- **Attribute {attr_name}:**")
                        lines.extend(details)
            else:
                sig = stat_result["significant_features"]
                details = [
                    f"  - {name}: {len(features)} significant features"
                    for name, features in sig.items()
                    if features is not None
                ] or ["  - No significant group differences detected"]
                lines.extend(details)
            lines.append("")

        if (
            "gradcam_mask_overlap" in self.results
            and self.results["gradcam_mask_overlap"]["success"]
        ):
            report = self.results["gradcam_mask_overlap"].get("report", {})
            summary = self.results["gradcam_mask_overlap"].get("metrics", {})
            lines.extend(
                [
                    "## GradCAM Attention vs. GT Masks",
                    f"- Samples: {summary.get('n_samples', 0)}",
                    f"- Attention-in-mask (mean): {summary.get('attention_in_mask_mean', 0.0):.3f}",
                    f"- Dice (mean): {summary.get('dice_mean', 0.0):.3f}",
                    f"- IoU (mean): {summary.get('iou_mean', 0.0):.3f}",
                    "",
                ]
            )
            top = report.get("top_samples", [])
            if top:
                lines.append("- Top samples (attention-in-mask):")
                for sample in top:
                    lines.append(
                        f"  - idx {sample.get('index')}: {sample.get('attention_in_mask', 0.0):.3f}"
                    )
                lines.append("")

        if "cav" in self.results and self.results["cav"]["success"]:
            cav_report = self.results["cav"].get("report", {})
            cav_metrics = self.results["cav"].get("metrics", {})
            risk = self._risk_payload("cav")
            lines.extend(
                [
                    "## CAV (Concept Activation Vectors)",
                    f"- Risk: {risk['risk_label']}",
                    f"- Reason: {risk['risk_reason']}",
                    f"- Concepts: {cav_metrics.get('n_concepts', 0)} (tested: {cav_metrics.get('n_tested', 0)})",
                    f"- Max TCAV score: {cav_metrics.get('max_tcav_score')}",
                    f"- Max concept quality (AUC): {cav_metrics.get('max_concept_quality')}",
                    "",
                ]
            )
            per_concept = cav_report.get("per_concept", [])
            if per_concept:
                lines.append("- Per-concept summary:")
                for row in per_concept:
                    lines.append(
                        f"  - {row.get('concept_name')}: "
                        f"quality_auc={row.get('quality_auc')}, "
                        f"tcav_score={row.get('tcav_score')}, flagged={row.get('flagged')}"
                    )
                lines.append("")

        if "sis" in self.results and self.results["sis"]["success"]:
            sis_metrics = self.results["sis"].get("metrics", {})
            risk = self._risk_payload("sis")
            lines.extend(
                [
                    "## SIS (Sufficient Input Subsets)",
                    f"- Risk: {risk['risk_label']}",
                    f"- Reason: {risk['risk_reason']}",
                    f"- Mean SIS size: {sis_metrics.get('mean_sis_size')}",
                    f"- Median SIS size: {sis_metrics.get('median_sis_size')}",
                    f"- Fraction of dimensions: {sis_metrics.get('frac_dimensions')}",
                    f"- Samples computed: {sis_metrics.get('n_computed')}",
                    "",
                ]
            )

        if "geometric" in self.results and self.results["geometric"]["success"]:
            geo_result = self.results["geometric"]
            risk = self._risk_payload("geometric")
            lines.extend(
                [
                    "## Geometric Analysis",
                    f"- Risk: {risk['risk_label']}",
                    f"- Reason: {risk['risk_reason']}",
                ]
            )

            def _geo_pair(pair, *attrs):
                g = getattr(pair, "groups", None) or (
                    pair.get("groups", []) if isinstance(pair, dict) else []
                )
                vals = []
                for a in attrs:
                    v = (
                        pair.get(a, float("nan"))
                        if isinstance(pair, dict)
                        else getattr(pair, a, float("nan"))
                    )
                    vals.append(v)
                return " vs ".join(str(x) for x in g), vals

            to_render = (
                geo_result.get("by_attribute", {"": geo_result})
                if "by_attribute" in geo_result
                else {"": geo_result}
            )
            for attr_name, gr in to_render.items():
                if not gr.get("success"):
                    continue
                if attr_name:
                    lines.append(f"- **Attribute {attr_name}:**")
                summary = gr.get("summary", {})
                if summary:
                    lines.append(f"  - Details: {summary.get('message', '')}")
                for pair in (gr.get("bias_pairs") or [])[:3]:
                    gs, (es, al) = _geo_pair(pair, "effect_size", "alignment_score")
                    lines.append(f"  - {gs}: effect_size={es:.2f}, alignment={al:.2f}")
                for pair in (gr.get("subspace_pairs") or [])[:3]:
                    gs, (mc, ma) = _geo_pair(pair, "mean_cosine", "min_angle_deg")
                    lines.append(f"  - {gs}: mean_cosine={mc:.2f}, min_angle={ma:.1f}°")
            lines.append("")

        if "bias_direction_pca" in self.results and self.results["bias_direction_pca"]["success"]:
            bd_result = self.results["bias_direction_pca"]
            risk = self._risk_payload("bias_direction_pca")
            lines.extend(
                [
                    "## Embedding Bias Direction (PCA)",
                    f"- Risk: {risk['risk_label']}",
                    f"- Reason: {risk['risk_reason']}",
                ]
            )
            if "by_attribute" in bd_result:
                for attr_name, sub in bd_result["by_attribute"].items():
                    if sub.get("success") and sub.get("report"):
                        report = sub["report"]
                        lines.extend(
                            [
                                f"- **Attribute {attr_name}:** projection_gap={report.projection_gap:.3f}, explained_var={report.explained_variance:.3f}",
                                "  Group projections:",
                            ]
                        )
                        for group, metrics in report.group_projections.items():
                            lines.append(
                                f"    - {group}: projection={metrics['projection']:.3f}, support={metrics['support']:.0f}"
                            )
            else:
                report = bd_result["report"]
                lines.extend(
                    [
                        f"- Projection gap: {report.projection_gap:.3f}",
                        f"- Explained variance: {report.explained_variance:.3f}",
                        f"- Notes: {report.notes}",
                        f"- Reference: {report.reference}",
                        "- Group projections:",
                    ]
                )
                for group, metrics in report.group_projections.items():
                    lines.append(
                        f"  - {group}: projection={metrics['projection']:.3f}, support={metrics['support']:.0f}"
                    )
            lines.append("")

        if "groupdro" in self.results and self.results["groupdro"]["success"]:
            gdro_result = self.results["groupdro"]
            lines.append("## GroupDRO (Worst-Group Robustness)")
            if "by_attribute" in gdro_result:
                for attr_name, sub in gdro_result["by_attribute"].items():
                    if sub.get("success"):
                        rep = sub.get("report", {})
                        final = rep.get("final", {})
                        lines.extend(
                            [
                                f"- **Attribute {attr_name}:** Groups={rep.get('n_groups')}, "
                                f"avg_acc={final.get('avg_acc', float('nan')):.2%}, "
                                f"worst_acc={final.get('worst_group_acc', float('nan')):.2%}",
                            ]
                        )
            else:
                rep = gdro_result["report"]
                final = rep.get("final", {})
                lines.extend(
                    [
                        f"- Groups: {rep.get('n_groups')}",
                        f"- Average accuracy: {final.get('avg_acc', float('nan')):.2%}",
                        f"- Worst-group accuracy: {final.get('worst_group_acc', float('nan')):.2%}",
                    ]
                )
            lines.append("")

        if (
            "early_epoch_clustering" in self.results
            and self.results["early_epoch_clustering"]["success"]
        ):
            report = self.results["early_epoch_clustering"]["report"]
            risk = self._risk_payload("early_epoch_clustering")
            lines.extend(
                [
                    "## Early-Epoch Clustering (SPARE)",
                    f"- Risk: {risk['risk_label']}",
                    f"- Reason: {risk['risk_reason']}",
                    f"- Clusters: {report.n_clusters} | Method: {report.cluster_method}",
                    f"- Entropy: {report.size_entropy:.3f} | Minority ratio: {report.minority_ratio:.3f}",
                    f"- Largest gap: {report.largest_gap:.3f}",
                ]
            )
            if report.cluster_label_agreement is not None:
                lines.append(f"- Cluster label agreement: {report.cluster_label_agreement:.3f}")
            lines.append("")

        if "equalized_odds" in self.results and self.results["equalized_odds"]["success"]:
            eo_result = self.results["equalized_odds"]
            risk = self._risk_payload("equalized_odds")
            lines.extend(
                [
                    "## Fairness (Equalized Odds)",
                    f"- Risk: {risk['risk_label']}",
                    f"- Reason: {risk['risk_reason']}",
                ]
            )
            if "by_attribute" in eo_result:
                for attr_name, sub in eo_result["by_attribute"].items():
                    if sub.get("success") and sub.get("report"):
                        report = sub["report"]
                        lines.extend(
                            [
                                f"- **Attribute {attr_name}:** TPR gap={report.tpr_gap:.3f}, FPR gap={report.fpr_gap:.3f}",
                                "  Group metrics:",
                            ]
                        )
                        for group, metrics in report.group_metrics.items():
                            tpr = "nan" if np.isnan(metrics["tpr"]) else f"{metrics['tpr']:.3f}"
                            fpr = "nan" if np.isnan(metrics["fpr"]) else f"{metrics['fpr']:.3f}"
                            lines.append(
                                f"    - {group}: TPR={tpr}, FPR={fpr}, support={metrics['support']:.0f}"
                            )
            else:
                report = eo_result["report"]
                lines.extend(
                    [
                        f"- TPR gap: {report.tpr_gap:.3f}",
                        f"- FPR gap: {report.fpr_gap:.3f}",
                        f"- Notes: {report.notes}",
                        f"- Reference: {report.reference}",
                        "- Group metrics:",
                    ]
                )
                for group, metrics in report.group_metrics.items():
                    tpr = "nan" if np.isnan(metrics["tpr"]) else f"{metrics['tpr']:.3f}"
                    fpr = "nan" if np.isnan(metrics["fpr"]) else f"{metrics['fpr']:.3f}"
                    lines.append(
                        f"  - {group}: TPR={tpr}, FPR={fpr}, support={metrics['support']:.0f}"
                    )
            lines.append("")

        if "demographic_parity" in self.results and self.results["demographic_parity"]["success"]:
            dp_result = self.results["demographic_parity"]
            risk = self._risk_payload("demographic_parity")
            lines.extend(
                [
                    "## Fairness (Demographic Parity)",
                    f"- Risk: {risk['risk_label']}",
                    f"- Reason: {risk['risk_reason']}",
                ]
            )
            if "by_attribute" in dp_result:
                for attr_name, sub in dp_result["by_attribute"].items():
                    if sub.get("success") and sub.get("report"):
                        report = sub["report"]
                        lines.extend(
                            [
                                f"- **Attribute {attr_name}:** DP gap={report.dp_gap:.3f}, overall pos_rate={report.overall_positive_rate:.3f}",
                                "  Group rates:",
                            ]
                        )
                        for group, metrics in report.group_rates.items():
                            rate = (
                                "nan"
                                if np.isnan(metrics["positive_rate"])
                                else f"{metrics['positive_rate']:.3f}"
                            )
                            lines.append(
                                f"    - {group}: positive_rate={rate}, support={metrics['support']:.0f}"
                            )
            else:
                report = dp_result["report"]
                lines.extend(
                    [
                        f"- DP gap: {report.dp_gap:.3f}",
                        f"- Overall positive rate: {report.overall_positive_rate:.3f}",
                        f"- Notes: {report.notes}",
                        f"- Reference: {report.reference}",
                        "- Group rates:",
                    ]
                )
                for group, metrics in report.group_rates.items():
                    rate = (
                        "nan"
                        if np.isnan(metrics["positive_rate"])
                        else f"{metrics['positive_rate']:.3f}"
                    )
                    lines.append(
                        f"  - {group}: positive_rate={rate}, support={metrics['support']:.0f}"
                    )
            lines.append("")

        if "intersectional" in self.results and self.results["intersectional"]["success"]:
            report = self.results["intersectional"]["report"]
            risk = self._risk_payload("intersectional")
            lines.extend(
                [
                    "## Fairness (Intersectional)",
                    f"- Risk: {risk['risk_label']}",
                    f"- Reason: {risk['risk_reason']}",
                    f"- TPR gap: {report.tpr_gap:.3f} | FPR gap: {report.fpr_gap:.3f} | DP gap: {report.dp_gap:.3f}",
                    f"- Attributes: {', '.join(report.attribute_names)}",
                    f"- Notes: {report.notes}",
                    f"- Reference: {report.reference}",
                    "- Intersection metrics:",
                ]
            )
            for group, metrics in report.intersection_metrics.items():
                tpr = "nan" if np.isnan(metrics["tpr"]) else f"{metrics['tpr']:.3f}"
                fpr = "nan" if np.isnan(metrics["fpr"]) else f"{metrics['fpr']:.3f}"
                pr = (
                    "nan"
                    if np.isnan(metrics.get("positive_rate", float("nan")))
                    else f"{metrics['positive_rate']:.3f}"
                )
                lines.append(
                    f"  - {group}: TPR={tpr}, FPR={fpr}, pos_rate={pr}, support={metrics['support']:.0f}"
                )
            lines.append("")

        if "gce" in self.results and self.results["gce"]["success"]:
            report = self.results["gce"]["report"]
            risk = self._risk_payload("gce")
            lines.extend(
                [
                    "## GCE (Generalized Cross Entropy) Bias Detection",
                    f"- Risk: {risk['risk_label']}",
                    f"- Reason: {risk['risk_reason']}",
                    f"- Minority/bias-conflicting samples: {report.n_minority} ({report.minority_ratio:.1%})",
                    f"- Loss mean ± std: {report.loss_mean:.4f} ± {report.loss_std:.4f}",
                    f"- Loss range: [{report.loss_min:.4f}, {report.loss_max:.4f}]",
                    f"- Threshold (percentile): {report.threshold:.4f} (q={report.q})",
                    f"- Notes: {report.notes}",
                    "",
                ]
            )

        if "causal_effect" in self.results and self.results["causal_effect"]["success"]:
            result = self.results["causal_effect"]
            metrics = result.get("metrics", {})
            report = result.get("report", {})
            per_attr = report.get("per_attribute", [])
            risk = self._risk_payload("causal_effect")
            lines.extend(
                [
                    "## Causal Effect Regularization",
                    f"- Risk: {risk['risk_label']}",
                    f"- Reason: {risk['risk_reason']}",
                    f"- Attributes: {metrics.get('n_attributes', 0)} | Spurious: {metrics.get('n_spurious', 0)}",
                    "",
                ]
            )
            for row in per_attr:
                lines.append(
                    f"  - {row.get('attribute_name', '')}: effect={row.get('causal_effect', 0):.4f}, "
                    f"spurious={row.get('is_spurious', False)}"
                )
            lines.append("")

        if "vae" in self.results and self.results["vae"]["success"]:
            result = self.results["vae"]
            metrics = result.get("metrics", {})
            risk = self._risk_payload("vae")
            lines.extend(
                [
                    "## VAE (Variational Autoencoder) Shortcut Detection",
                    f"- Risk: {risk['risk_label']}",
                    f"- Reason: {risk['risk_reason']}",
                    f"- Latent dims: {metrics.get('latent_dim', 0)} | Flagged: {metrics.get('n_flagged', 0)}",
                    "",
                ]
            )

        return "\n".join(lines)

    def _get_css(self) -> str:
        """CSS styling for HTML reports."""
        return """
        /* Force black-on-white when embedded (e.g. in Gradio); exceptions for semantic colors */
        .shortcut-report, .shortcut-report .container { background: #ffffff !important; color: #000000 !important; }
        .shortcut-report *, .shortcut-report .container * { color: #000000 !important; }
        .shortcut-report th { background: #1d4ed8 !important; color: #ffffff !important; }
        .shortcut-report code { background: unset !important; background-color: rgba(255, 255, 255, 1) !important; color: rgba(21, 87, 36, 1) !important; background-clip: unset; -webkit-background-clip: unset; box-shadow: none !important; }
        .shortcut-report .risk-high { color: #991b1b !important; }
        .shortcut-report .risk-moderate { color: #92400e !important; }
        .shortcut-report .risk-low { color: #065f46 !important; }
        .shortcut-report .risk-unknown { color: #4b5563 !important; }
        .shortcut-report .recommendation { color: #14532d !important; }
        .shortcut-report .footer { color: #374151 !important; }
        .shortcut-report .footer a { color: #1d4ed8 !important; }
        .shortcut-report a { color: #1d4ed8 !important; }
        .shortcut-report::selection, .shortcut-report *::selection { background: #b4d5fe; color: #000; }
        body { font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif; margin: 0; padding: 20px; background: linear-gradient(180deg, #edf2f7 0%, #e2e8f0 100%); color: #111827; }
        .container { max-width: 1200px; margin: 0 auto; background: #ffffff; color: #111827; padding: 40px; border-radius: 12px; box-shadow: 0 12px 35px rgba(15, 23, 42, 0.15); }
        h1 { color: #0f172a; border-bottom: 3px solid #2563eb; padding-bottom: 10px; }
        h2 { color: #1f2933; margin-top: 30px; border-left: 4px solid #2563eb; padding-left: 15px; padding-top: 10px; padding-bottom: 10px; background: #f8fafc; border-radius: 6px; }
        h3 { color: #111827; }
        .metadata { background: #f1f5f9; padding: 20px; border-radius: 6px; margin: 20px 0; border: 1px solid #e2e8f0; }
        .risk-high { background: #fee2e2; color: #991b1b; padding: 20px; border-radius: 8px; margin: 20px 0; border-left: 5px solid #ef4444; }
        .risk-moderate { background: #fff4e6; color: #92400e; padding: 20px; border-radius: 8px; margin: 20px 0; border-left: 5px solid #f97316; }
        .risk-low { background: #ecfdf5; color: #065f46; padding: 20px; border-radius: 8px; margin: 20px 0; border-left: 5px solid #10b981; }
        .risk-unknown { background: #f3f4f6; color: #4b5563; padding: 20px; border-radius: 8px; margin: 20px 0; border-left: 5px solid #9ca3af; }
        .section { margin: 20px 0; padding: 20px; background: #ffffff; border-radius: 8px; border: 1px solid #e2e8f0; box-shadow: inset 0 1px 3px rgba(15, 23, 42, 0.05); }
        table { width: 100%; border-collapse: collapse; margin: 15px 0; }
        th { background: #1d4ed8 !important; color: #ffffff !important; padding: 12px; text-align: left; font-weight: 600; }
        td { padding: 10px; border-bottom: 1px solid #e2e8f0; }
        tr:hover { background: #f8fafc; }
        .recommendation { background: #f0fdf4; border-left: 5px solid #22c55e; padding: 18px; margin: 10px 0; border-radius: 6px; color: #14532d; }
        code { background: unset !important; background-color: rgba(255, 255, 255, 1) !important; color: rgba(21, 87, 36, 1) !important; background-clip: unset; -webkit-background-clip: unset; box-shadow: none !important; padding: 3px 7px; border-radius: 4px; font-family: 'Courier New', monospace; }
        .plot-container { margin: 20px 0; text-align: center; }
        .plot-container details { display: flex; flex-direction: column; align-items: center; }
        .plot-container details iframe { display: block; margin-left: auto; margin-right: auto; }
        .plot-container img { max-width: 100%; height: auto; border-radius: 8px; border: 1px solid #e2e8f0; box-shadow: 0 4px 12px rgba(15, 23, 42, 0.15); margin: 10px 0; }
        .plot-grid { display: grid; grid-template-columns: repeat(auto-fit, minmax(320px, 1fr)); gap: 20px; margin: 20px 0; }
        .footer { margin-top: 40px; padding-top: 20px; border-top: 2px solid #e2e8f0; text-align: center; color: #475569; font-size: 14px; }
        .footer a { color: #1d4ed8; text-decoration: none; font-weight: 600; }
        .footer a:hover { text-decoration: underline; }
        """

    def _generate_metadata_html(self) -> str:
        """Generate metadata section for HTML."""
        n_samples = len(self.detector.embeddings_)
        n_dims = self.detector.embeddings_.shape[1]
        methods = ", ".join(self.detector.methods)

        return f"""
    <div class='metadata'>
      <h3>📋 Dataset Information</h3>
      <table>
        <tr><td><strong>Samples:</strong></td><td>{n_samples:,}</td></tr>
        <tr><td><strong>Embedding Dimensions:</strong></td><td>{n_dims}</td></tr>
        <tr><td><strong>Detection Methods:</strong></td><td>{methods}</td></tr>
        <tr><td><strong>Unique Labels:</strong></td><td>{len(np.unique(self.detector.labels_))}</td></tr>
      </table>
    </div>
        """

    def _generate_overall_assessment_html(self) -> str:
        """Generate overall risk assessment for HTML."""
        assessment = self.detector._generate_overall_assessment()

        if "HIGH RISK" in assessment:
            css_class = "risk-high"
        elif "MODERATE RISK" in assessment:
            css_class = "risk-moderate"
        else:
            css_class = "risk-low"

        # Replace bullet points with HTML list
        parts = assessment.split("\n")
        risk_level = parts[0]
        details = parts[1:] if len(parts) > 1 else []

        html = f"<div class='{css_class}'><h3>{risk_level}</h3>"
        if details:
            html += "<ul>"
            for detail in details:
                if detail.strip():
                    html += f"<li>{detail.strip('•').strip()}</li>"
            html += "</ul>"
        html += "</div>"

        return html

    def _risk_payload(self, method: str) -> dict[str, str]:
        result = self.results.get(method, {})
        if not isinstance(result, dict):
            return {
                "risk_value": "unknown",
                "risk_label": "Unknown",
                "risk_reason": "Insufficient information to determine risk.",
            }
        if all(k in result for k in ("risk_value", "risk_label", "risk_reason")):
            return {
                "risk_value": normalize_risk_level(result.get("risk_value")),
                "risk_label": str(result.get("risk_label")),
                "risk_reason": str(result.get("risk_reason")),
            }
        if "by_attribute" in result:
            return {
                "risk_value": "unknown",
                "risk_label": "Unknown",
                "risk_reason": "Multi-attribute analysis across attributes.",
            }
        payload = build_method_risk(method, result)
        result.update(payload)
        return payload

    def _risk_color(self, risk_value: str) -> str:
        level = normalize_risk_level(risk_value)
        return {
            "high": "#b91c1c",
            "moderate": "#b45309",
            "low": "#047857",
            "unknown": "#4b5563",
        }.get(level, "#4b5563")

    def _generate_geometric_section_html(self) -> str:
        """Generate geometric analysis section for HTML."""
        if "geometric" not in self.results or not self.results["geometric"]["success"]:
            return ""

        geo_result = self.results["geometric"]

        def _pair_value(pair, key, default=float("nan")):
            if isinstance(pair, dict):
                return pair.get(key, default)
            return getattr(pair, key, default)

        def _render_geo(gr, attr_name: str | None = None) -> list:
            report = gr.get("report") or {}
            summary = report.get("summary") or gr.get("summary", {}) or {}
            bias_pairs = report.get("bias_pairs") or gr.get("bias_pairs", []) or []
            subspace_pairs = report.get("subspace_pairs") or gr.get("subspace_pairs", []) or []
            lines = []
            if attr_name:
                lines.append(f"<h3>Attribute: {attr_name}</h3>")
            if summary:
                lines.append(f"<p>{summary.get('message', '')}</p>")
            lines.append("<h4>Top Bias Direction Pairs</h4>")
            lines.append(
                "<table><tr><th>Groups</th><th>Effect Size</th><th>Projection Gap</th><th>Alignment</th></tr>"
            )
            for pair in bias_pairs[:5]:
                groups = " vs ".join(_pair_value(pair, "groups", []))
                lines.append(
                    f"<tr><td><code>{groups}</code></td>"
                    f"<td>{_pair_value(pair, 'effect_size'):.3f}</td>"
                    f"<td>{_pair_value(pair, 'projection_gap'):.3f}</td>"
                    f"<td>{_pair_value(pair, 'alignment_score'):.3f}</td></tr>"
                )
            if not bias_pairs:
                lines.append("<tr><td colspan='4'>No bias direction pairs available.</td></tr>")
            lines.append("</table>")
            lines.append("<h4>Top Subspace Overlap Pairs</h4>")
            lines.append(
                "<table><tr><th>Groups</th><th>Mean Cosine</th><th>Min Angle (°)</th><th>Max Angle (°)</th></tr>"
            )
            for pair in subspace_pairs[:5]:
                groups = " vs ".join(_pair_value(pair, "groups", []))
                lines.append(
                    f"<tr><td><code>{groups}</code></td>"
                    f"<td>{_pair_value(pair, 'mean_cosine'):.3f}</td>"
                    f"<td>{_pair_value(pair, 'min_angle_deg'):.1f}</td>"
                    f"<td>{_pair_value(pair, 'max_angle_deg'):.1f}</td></tr>"
                )
            if not subspace_pairs:
                lines.append("<tr><td colspan='4'>No subspace overlap pairs available.</td></tr>")
            lines.append("</table>")
            return lines

        html = [
            "<div class='section'>",
            "<h2>🧭 Geometric Analysis</h2>",
        ]
        risk = self._risk_payload("geometric")
        risk_color = self._risk_color(risk["risk_value"])
        html.append(
            f"<p><strong>Risk:</strong> <span style='color: {risk_color}; font-weight: 600;'>{risk['risk_label']}</span></p>"
        )
        html.append(f"<p><strong>Reason:</strong> {risk['risk_reason']}</p>")

        if "by_attribute" in geo_result:
            for attr_name, sub in geo_result["by_attribute"].items():
                if sub.get("success"):
                    html.extend(_render_geo(sub, attr_name))
        else:
            html.extend(_render_geo(geo_result))

        html.append("</div>")
        return "\n".join(html)

    def _generate_gradcam_mask_overlap_section_html(self) -> str:
        """Generate GradCAM GT mask overlap section for HTML."""
        if (
            "gradcam_mask_overlap" not in self.results
            or not self.results["gradcam_mask_overlap"]["success"]
        ):
            return ""

        summary = self.results["gradcam_mask_overlap"].get("metrics", {})
        report = self.results["gradcam_mask_overlap"].get("report", {})
        top_samples = report.get("top_samples", [])
        bottom_samples = report.get("bottom_samples", [])

        html = [
            "<div class='section'>",
            "<h2>🩺 GradCAM Attention vs. GT Masks</h2>",
            "<table>",
            "<tr><th>Metric</th><th>Value</th></tr>",
            f"<tr><td>Samples</td><td>{summary.get('n_samples', 0)}</td></tr>",
            f"<tr><td>Attention-in-mask (mean)</td><td>{summary.get('attention_in_mask_mean', 0.0):.3f}</td></tr>",
            f"<tr><td>Attention-in-mask (median)</td><td>{summary.get('attention_in_mask_median', 0.0):.3f}</td></tr>",
            f"<tr><td>Mask coverage (mean)</td><td>{summary.get('mask_coverage_mean', 0.0):.3f}</td></tr>",
            f"<tr><td>Dice (mean)</td><td>{summary.get('dice_mean', 0.0):.3f}</td></tr>",
            f"<tr><td>IoU (mean)</td><td>{summary.get('iou_mean', 0.0):.3f}</td></tr>",
            f"<tr><td>Cosine (mean)</td><td>{summary.get('cosine_mean', 0.0):.3f}</td></tr>",
            "</table>",
        ]

        def _render_table(title: str, rows: list):
            html.append(f"<h3>{title}</h3>")
            html.append("<table>")
            html.append(
                "<tr><th>Index</th><th>Attention-in-mask</th><th>Dice</th><th>IoU</th></tr>"
            )
            if rows:
                for sample in rows:
                    html.append(
                        "<tr>"
                        f"<td>{sample.get('index')}</td>"
                        f"<td>{sample.get('attention_in_mask', 0.0):.3f}</td>"
                        f"<td>{sample.get('dice', 0.0):.3f}</td>"
                        f"<td>{sample.get('iou', 0.0):.3f}</td>"
                        "</tr>"
                    )
            else:
                html.append("<tr><td colspan='4'>No samples available.</td></tr>")
            html.append("</table>")

        _render_table("Top Attention-in-mask Samples", top_samples)
        _render_table("Lowest Attention-in-mask Samples", bottom_samples)

        html.append("</div>")
        return "\n".join(html)

    def _generate_cav_section_html(self) -> str:
        """Generate CAV section for HTML."""
        if "cav" not in self.results or not self.results["cav"]["success"]:
            return ""

        summary = self.results["cav"].get("metrics", {})
        report = self.results["cav"].get("report", {})
        per_concept = report.get("per_concept", [])
        risk = self._risk_payload("cav")
        risk_color = self._risk_color(risk["risk_value"])

        html = [
            "<div class='section'>",
            "<h2>🧪 CAV (Concept Activation Vectors)</h2>",
            f"<p><strong>Risk:</strong> <span style='color: {risk_color}; font-weight: 600;'>{risk['risk_label']}</span></p>",
            f"<p><strong>Reason:</strong> {risk['risk_reason']}</p>",
            "<table>",
            "<tr><th>Metric</th><th>Value</th></tr>",
            f"<tr><td>Concepts</td><td>{summary.get('n_concepts', 0)}</td></tr>",
            f"<tr><td>Tested concepts</td><td>{summary.get('n_tested', 0)}</td></tr>",
            f"<tr><td>Max TCAV score</td><td>{summary.get('max_tcav_score')}</td></tr>",
            f"<tr><td>Mean TCAV score</td><td>{summary.get('mean_tcav_score')}</td></tr>",
            f"<tr><td>Max concept quality (AUC)</td><td>{summary.get('max_concept_quality')}</td></tr>",
            f"<tr><td>Flagged concepts</td><td>{summary.get('n_flagged', 0)}</td></tr>",
            "</table>",
            "<h3>Per-concept Results</h3>",
            "<table>",
            "<tr><th>Concept</th><th>Quality AUC</th><th>TCAV Score</th><th>Activation Mean</th><th>Activation P95</th><th>Flagged</th></tr>",
        ]

        if per_concept:
            for row in per_concept:
                html.append(
                    "<tr>"
                    f"<td><code>{row.get('concept_name')}</code></td>"
                    f"<td>{row.get('quality_auc')}</td>"
                    f"<td>{row.get('tcav_score')}</td>"
                    f"<td>{row.get('activation_mean')}</td>"
                    f"<td>{row.get('activation_p95')}</td>"
                    f"<td>{'Yes' if row.get('flagged') else 'No'}</td>"
                    "</tr>"
                )
        else:
            html.append("<tr><td colspan='6'>No concept rows available.</td></tr>")

        html.append("</table>")
        html.append("</div>")
        return "\n".join(html)

    def _generate_sis_section_html(self) -> str:
        """Generate SIS (Sufficient Input Subsets) section for HTML."""
        if "sis" not in self.results or not self.results["sis"]["success"]:
            return ""

        result = self.results["sis"]
        metrics = result.get("metrics", {})
        report = result.get("report") or {}
        dist = report.get("distribution", {})
        risk = self._risk_payload("sis")
        risk_color = self._risk_color(risk["risk_value"])

        mean_sis = metrics.get("mean_sis_size")
        median_sis = metrics.get("median_sis_size")
        frac_dim = metrics.get("frac_dimensions")
        n_computed = metrics.get("n_computed", 0)
        group_overlap = metrics.get("group_sis_overlap")

        rows = [
            (f"<tr><td>Mean SIS size</td><td>{mean_sis:.1f}</td></tr>", mean_sis is not None),
            (f"<tr><td>Median SIS size</td><td>{median_sis}</td></tr>", median_sis is not None),
            (
                f"<tr><td>Fraction of dimensions</td><td>{frac_dim:.1%}</td></tr>",
                frac_dim is not None,
            ),
            (f"<tr><td>Samples computed</td><td>{n_computed}</td></tr>", True),
            (
                f"<tr><td>Group-SIS overlap</td><td>{group_overlap:.2f}</td></tr>",
                group_overlap is not None,
            ),
        ]
        table_rows = "".join(r for r, ok in rows if ok)

        html = [
            "<div class='section'>",
            "<h2>🔍 SIS (Sufficient Input Subsets)</h2>",
            f"<p><strong>Risk:</strong> <span style='color: {risk_color}; font-weight: 600;'>{risk['risk_label']}</span></p>",
            f"<p><strong>Reason:</strong> {risk['risk_reason']}</p>",
            "<table>",
            "<tr><th>Metric</th><th>Value</th></tr>",
            table_rows,
            "</table>",
        ]
        sis_sizes = report.get("sis_sizes", [])
        if sis_sizes:
            html.append("<h3>SIS Size Distribution</h3>")
            html.append(
                f"<p>Min: {dist.get('min')} | Max: {dist.get('max')} | "
                f"Mean: {dist.get('mean', 0):.1f} | Median: {dist.get('median')}</p>"
            )
        html.append("</div>")
        return "\n".join(html)

    def _generate_hbac_section_html(self) -> str:
        """Generate HBAC results section for HTML."""
        if "hbac" not in self.results or not self.results["hbac"]["success"]:
            return ""

        report = self.results["hbac"]["report"]
        shortcut_info = report["has_shortcut"]
        risk = self._risk_payload("hbac")
        risk_color = self._risk_color(risk["risk_value"])

        html = ["<div class='section'>", "<h2>🧬 HBAC (Clustering-based Detection)</h2>"]
        html.append(
            f"<p><strong>Risk:</strong> <span style='color: {risk_color}; font-weight: 600;'>{risk['risk_label']}</span></p>"
        )
        html.append(f"<p><strong>Reason:</strong> {risk['risk_reason']}</p>")

        html.append(
            f"<p><strong>Shortcut Detected:</strong> {'YES' if shortcut_info['exists'] else 'NO'}</p>"
        )

        if shortcut_info["types"]:
            html.append(f"<p><strong>Types:</strong> {', '.join(shortcut_info['types'])}</p>")

        # Cluster purities table
        html.append("<h3>Cluster Analysis</h3>")
        html.append("<table>")
        html.append("<tr><th>Cluster</th><th>Size</th><th>Purity</th><th>Dominant Label</th></tr>")

        for cp in report["cluster_purities"]:
            html.append(
                f"<tr><td>{cp['cluster_id']}</td><td>{cp['size']}</td>"
                f"<td>{cp['purity']:.1%}</td><td>{cp['dominant_label']}</td></tr>"
            )

        html.append("</table>")

        # Top dimensions
        html.append("<h3>Top Important Dimensions</h3>")
        top_dims = report["dimension_importance"].head(5)
        html.append("<table>")
        html.append("<tr><th>Dimension</th><th>F-score</th><th>P-value</th></tr>")

        for _, row in top_dims.iterrows():
            html.append(
                f"<tr><td><code>{row['dimension']}</code></td><td>{row['f_score']:.2f}</td><td>{row['p_value']:.4f}</td></tr>"
            )

        html.append("</table>")
        html.append("</div>")

        return "\n".join(html)

    def _generate_probe_section_html(self) -> str:
        """Generate probe results section for HTML."""
        if "probe" not in self.results or not self.results["probe"]["success"]:
            return ""

        metric_name = self.results["probe"]["results"]["metrics"]["metric"]
        accuracy = self.results["probe"]["results"]["metrics"]["metric_value"]
        risk = self._risk_payload("probe")
        risk_color = self._risk_color(risk["risk_value"])

        html = [
            "<div class='section'>",
            "<h2>🔬 Probe-based Detection</h2>",
            f"<p><strong>Risk:</strong> <span style='color: {risk_color}; font-weight: 600;'>{risk['risk_label']}</span></p>",
            f"<p><strong>Reason:</strong> {risk['risk_reason']}</p>",
            f"<p><strong>{metric_name}:</strong> {accuracy:.2%}</p>",
        ]

        html.append("</div>")
        return "\n".join(html)

    def _generate_frequency_section_html(self) -> str:
        """Generate frequency shortcut section for HTML."""
        if "frequency" not in self.results or not self.results["frequency"]["success"]:
            return ""

        report = self.results["frequency"]["report"]
        metrics = report.get("metrics", {})
        detail = report.get("report", {})
        class_rates = detail.get("class_rates", {})
        risk = self._risk_payload("frequency")
        risk_color = self._risk_color(risk["risk_value"])

        html = [
            "<div class='section'>",
            "<h2>Embedding Frequency Shortcut</h2>",
            f"<p><strong>Risk:</strong> <span style='color: {risk_color}; font-weight: 600;'>{risk['risk_label']}</span></p>",
            f"<p><strong>Reason:</strong> {risk['risk_reason']}</p>",
            f"<p><strong>Probe accuracy:</strong> {float(metrics.get('probe_accuracy', float('nan'))):.2%}</p>",
            f"<p><strong>Shortcut classes:</strong> {detail.get('shortcut_classes', [])}</p>",
            f"<p><strong>Top-percent used:</strong> {metrics.get('top_percent')}</p>",
            "<h3>Class Rates</h3>",
            "<table>",
            "<tr><th>Class</th><th>TPR</th><th>FPR</th><th>Support</th><th>Top Dims</th></tr>",
        ]
        if class_rates:
            top_dims = detail.get("top_dims_by_class", {})
            for cls, rates in class_rates.items():
                html.append(
                    "<tr>"
                    f"<td><code>{cls}</code></td>"
                    f"<td>{float(rates.get('tpr', float('nan'))):.3f}</td>"
                    f"<td>{float(rates.get('fpr', float('nan'))):.3f}</td>"
                    f"<td>{rates.get('support')}</td>"
                    f"<td>{top_dims.get(str(cls), [])}</td>"
                    "</tr>"
                )
        else:
            html.append("<tr><td colspan='5'>No class-rate details available.</td></tr>")
        html.extend(["</table>", "</div>"])
        return "\n".join(html)

    def _generate_statistical_section_html(self) -> str:
        """Generate statistical results section for HTML."""
        if "statistical" not in self.results or not self.results["statistical"]["success"]:
            return ""

        result = self.results["statistical"]
        risk = self._risk_payload("statistical")
        risk_color = self._risk_color(risk["risk_value"])

        def _render_stat(sr) -> list:
            significant = sr.get("significant_features", {})
            lines = ["<table>", "<tr><th>Comparison</th><th>Significant Features</th></tr>"]
            for comparison, features in significant.items():
                n_sig = len(features) if features is not None else 0
                lines.append(f"<tr><td><code>{comparison}</code></td><td>{n_sig}</td></tr>")
            lines.append("</table>")
            return lines

        html = [
            "<div class='section'>",
            "<h2>📈 Statistical Testing</h2>",
            f"<p><strong>Risk:</strong> <span style='color: {risk_color}; font-weight: 600;'>{risk['risk_label']}</span></p>",
            f"<p><strong>Reason:</strong> {risk['risk_reason']}</p>",
            "<h3>Group Comparisons</h3>",
        ]
        if "by_attribute" in result:
            for attr_name, sub in result["by_attribute"].items():
                if sub.get("success"):
                    html.append(f"<h4>Attribute: {attr_name}</h4>")
                    html.extend(_render_stat(sub))
        else:
            html.extend(_render_stat(result))

        html.append("</div>")
        return "\n".join(html)

    def _generate_fairness_section_html(self) -> str:
        """Generate equalized odds fairness section."""
        if "equalized_odds" not in self.results or not self.results["equalized_odds"]["success"]:
            return ""

        result = self.results["equalized_odds"]
        risk = self._risk_payload("equalized_odds")
        risk_color = self._risk_color(risk["risk_value"])

        def _render_eo_report(report, attr_name: str | None = None) -> list:
            lines = []
            if attr_name:
                lines.append(f"<h3>Attribute: {attr_name}</h3>")
            lines.extend(
                [
                    f"<p><strong>TPR Gap:</strong> {report.tpr_gap:.3f} | <strong>FPR Gap:</strong> {report.fpr_gap:.3f}</p>",
                    f"<p><strong>Reference:</strong> {report.reference}</p>",
                    "<table>",
                    "<tr><th>Group</th><th>TPR</th><th>FPR</th><th>Support</th></tr>",
                ]
            )
            for group, metrics in report.group_metrics.items():
                tpr = "nan" if np.isnan(metrics["tpr"]) else f"{metrics['tpr']:.3f}"
                fpr = "nan" if np.isnan(metrics["fpr"]) else f"{metrics['fpr']:.3f}"
                lines.append(
                    f"<tr><td>{group}</td><td>{tpr}</td><td>{fpr}</td><td>{metrics['support']:.0f}</td></tr>"
                )
            lines.append("</table>")
            return lines

        html = [
            "<div class='section'>",
            "<h2>⚖️ Fairness (Equalized Odds)</h2>",
            f"<p><strong>Risk:</strong> <span style='color: {risk_color}; font-weight: 600;'>{risk['risk_label']}</span></p>",
            f"<p><strong>Reason:</strong> {risk['risk_reason']}</p>",
        ]
        if "by_attribute" in result:
            for attr_name, sub in result["by_attribute"].items():
                if sub.get("success") and sub.get("report"):
                    html.extend(_render_eo_report(sub["report"], attr_name))
        else:
            html.append("<h3>Group Metrics</h3>")
            html.extend(_render_eo_report(result["report"]))

        html.append("</div>")
        return "\n".join(html)

    def _generate_bias_direction_section_html(self) -> str:
        """Generate bias direction PCA section for HTML."""
        if (
            "bias_direction_pca" not in self.results
            or not self.results["bias_direction_pca"]["success"]
        ):
            return ""

        result = self.results["bias_direction_pca"]
        risk = self._risk_payload("bias_direction_pca")
        risk_color = self._risk_color(risk["risk_value"])

        def _render_bd(report, attr_name: str | None = None) -> list:
            lines = []
            if attr_name:
                lines.append(f"<h3>Attribute: {attr_name}</h3>")
            lines.extend(
                [
                    f"<p><strong>Projection gap:</strong> {report.projection_gap:.3f}</p>",
                    f"<p><strong>Explained variance:</strong> {report.explained_variance:.3f}</p>",
                    f"<p><strong>Reference:</strong> {report.reference}</p>",
                    "<table>",
                    "<tr><th>Group</th><th>Projection</th><th>Support</th></tr>",
                ]
            )
            for group, metrics in report.group_projections.items():
                lines.append(
                    f"<tr><td>{group}</td><td>{metrics['projection']:.3f}</td><td>{metrics['support']:.0f}</td></tr>"
                )
            lines.append("</table>")
            return lines

        html = [
            "<div class='section'>",
            "<h2>🧭 Embedding Bias Direction (PCA)</h2>",
            f"<p><strong>Risk:</strong> <span style='color: {risk_color}; font-weight: 600;'>{risk['risk_label']}</span></p>",
            f"<p><strong>Reason:</strong> {risk['risk_reason']}</p>",
        ]
        if "by_attribute" in result:
            for attr_name, sub in result["by_attribute"].items():
                if sub.get("success") and sub.get("report"):
                    html.extend(_render_bd(sub["report"], attr_name))
        else:
            html.extend(_render_bd(result["report"]))

        html.append("</div>")
        return "\n".join(html)

    def _generate_demographic_parity_section_html(self) -> str:
        """Generate demographic parity fairness section."""
        if (
            "demographic_parity" not in self.results
            or not self.results["demographic_parity"]["success"]
        ):
            return ""

        result = self.results["demographic_parity"]
        risk = self._risk_payload("demographic_parity")
        risk_color = self._risk_color(risk["risk_value"])

        def _render_dp_report(report, attr_name: str | None = None) -> list:
            lines = []
            if attr_name:
                lines.append(f"<h3>Attribute: {attr_name}</h3>")
            lines.extend(
                [
                    f"<p><strong>DP Gap:</strong> {report.dp_gap:.3f}</p>",
                    f"<p><strong>Overall positive rate:</strong> {report.overall_positive_rate:.3f}</p>",
                    f"<p><strong>Reference:</strong> {report.reference}</p>",
                    "<table>",
                    "<tr><th>Group</th><th>Positive Rate</th><th>Support</th></tr>",
                ]
            )
            for group, metrics in report.group_rates.items():
                rate = (
                    "nan"
                    if np.isnan(metrics["positive_rate"])
                    else f"{metrics['positive_rate']:.3f}"
                )
                lines.append(
                    f"<tr><td>{group}</td><td>{rate}</td><td>{metrics['support']:.0f}</td></tr>"
                )
            lines.append("</table>")
            return lines

        html = [
            "<div class='section'>",
            "<h2>⚖️ Fairness (Demographic Parity)</h2>",
            f"<p><strong>Risk:</strong> <span style='color: {risk_color}; font-weight: 600;'>{risk['risk_label']}</span></p>",
            f"<p><strong>Reason:</strong> {risk['risk_reason']}</p>",
        ]
        if "by_attribute" in result:
            for attr_name, sub in result["by_attribute"].items():
                if sub.get("success") and sub.get("report"):
                    html.extend(_render_dp_report(sub["report"], attr_name))
        else:
            html.append("<h3>Group Rates</h3>")
            html.extend(_render_dp_report(result["report"]))

        html.append("</div>")
        return "\n".join(html)

    def _generate_intersectional_section_html(self) -> str:
        """Generate intersectional fairness section."""
        if "intersectional" not in self.results or not self.results["intersectional"]["success"]:
            return ""

        report = self.results["intersectional"]["report"]
        risk = self._risk_payload("intersectional")
        risk_color = self._risk_color(risk["risk_value"])

        html = [
            "<div class='section'>",
            "<h2>⚖️ Fairness (Intersectional)</h2>",
            f"<p><strong>Risk:</strong> <span style='color: {risk_color}; font-weight: 600;'>{risk['risk_label']}</span></p>",
            f"<p><strong>Reason:</strong> {risk['risk_reason']}</p>",
            f"<p><strong>TPR Gap:</strong> {report.tpr_gap:.3f} | <strong>FPR Gap:</strong> {report.fpr_gap:.3f} | <strong>DP Gap:</strong> {report.dp_gap:.3f}</p>",
            f"<p><strong>Attributes:</strong> {', '.join(report.attribute_names)}</p>",
            f"<p><strong>Reference:</strong> {report.reference}</p>",
            "<h3>Intersection Metrics</h3>",
            "<table>",
            "<tr><th>Intersection</th><th>TPR</th><th>FPR</th><th>Positive Rate</th><th>Support</th></tr>",
        ]
        for group, metrics in report.intersection_metrics.items():
            tpr = "nan" if np.isnan(metrics["tpr"]) else f"{metrics['tpr']:.3f}"
            fpr = "nan" if np.isnan(metrics["fpr"]) else f"{metrics['fpr']:.3f}"
            pr = (
                "nan"
                if np.isnan(metrics.get("positive_rate", float("nan")))
                else f"{metrics['positive_rate']:.3f}"
            )
            html.append(
                f"<tr><td>{group}</td><td>{tpr}</td><td>{fpr}</td><td>{pr}</td><td>{metrics['support']:.0f}</td></tr>"
            )

        html.append("</table>")
        html.append("</div>")
        return "\n".join(html)

    def _generate_gce_section_html(self) -> str:
        """Generate GCE bias detection section for HTML."""
        if "gce" not in self.results or not self.results["gce"]["success"]:
            return ""

        result = self.results["gce"]
        # For by_attribute, use first successful sub for per-sample display (or skip)
        if "by_attribute" in result:
            first_ok = next((s for s in result["by_attribute"].values() if s.get("success")), None)
            report = first_ok["report"] if first_ok else None
            minority_indices = (
                first_ok.get("minority_indices", np.array([])) if first_ok else np.array([])
            )
            per_sample_losses = (
                first_ok.get("per_sample_losses", np.array([])) if first_ok else np.array([])
            )
        else:
            report = result["report"]
            minority_indices = result.get("minority_indices", np.array([]))
            per_sample_losses = result.get("per_sample_losses", np.array([]))
        risk = self._risk_payload("gce")
        risk_color = self._risk_color(risk["risk_value"])

        if report is None:
            html = [
                "<div class='section'>",
                "<h2>📊 GCE (Generalized Cross Entropy) Bias Detection</h2>",
                "<p>No successful GCE results.</p>",
                "</div>",
            ]
            return "\n".join(html)

        html = [
            "<div class='section'>",
            "<h2>📊 GCE (Generalized Cross Entropy) Bias Detection</h2>",
            f"<p><strong>Risk:</strong> <span style='color: {risk_color}; font-weight: 600;'>{risk['risk_label']}</span></p>",
            f"<p><strong>Reason:</strong> {risk['risk_reason']}</p>",
            f"<p><strong>Minority/bias-conflicting samples:</strong> {report.n_minority} ({report.minority_ratio:.1%})</p>",
            f"<p><strong>Loss statistics:</strong> mean={report.loss_mean:.4f}, std={report.loss_std:.4f}, "
            f"range=[{report.loss_min:.4f}, {report.loss_max:.4f}]</p>",
            f"<p><strong>Threshold (percentile):</strong> {report.threshold:.4f} (q={report.q})</p>",
            "<h3>Loss Statistics</h3>",
            "<table>",
            "<tr><th>Metric</th><th>Value</th></tr>",
            f"<tr><td>Mean loss</td><td>{report.loss_mean:.4f}</td></tr>",
            f"<tr><td>Std loss</td><td>{report.loss_std:.4f}</td></tr>",
            f"<tr><td>Min loss</td><td>{report.loss_min:.4f}</td></tr>",
            f"<tr><td>Max loss</td><td>{report.loss_max:.4f}</td></tr>",
            f"<tr><td>Threshold</td><td>{report.threshold:.4f}</td></tr>",
            "</table>",
        ]

        if len(per_sample_losses) > 0:
            # Top 20 high-loss samples (may be minority/bias-conflicting)
            top_n = min(20, len(per_sample_losses))
            sorted_idx = np.argsort(per_sample_losses)[::-1][:top_n]
            losses_at_top = per_sample_losses[sorted_idx]
            minority_set = set(np.atleast_1d(minority_indices).tolist())
            html.append("<h3>Top High-Loss (Minority/Bias-Conflicting) Samples</h3>")

            # Smart summarization: if all top samples share same loss (std < 1e-6), show summary
            losses_std = np.std(losses_at_top) if len(losses_at_top) > 1 else 0.0
            show_summary = len(losses_at_top) >= 5 and losses_std < 1e-6

            if show_summary:
                html.append(
                    f"<p><strong>{len(losses_at_top)} samples flagged</strong> with identical "
                    f"GCE Loss = {losses_at_top[0]:.4f}</p>"
                )
                html.append(
                    "<details style='margin-top: 8px;'>"
                    "<summary style='cursor: pointer; font-weight: 600;'>View full list</summary>"
                    "<table style='margin-top: 8px;'>"
                    "<tr><th>Sample Index</th><th>GCE Loss</th><th>Flagged</th></tr>"
                )
                for pos in sorted_idx:
                    sample_idx = int(pos)
                    loss = per_sample_losses[sample_idx]
                    flagged = "Yes" if sample_idx in minority_set else "No"
                    html.append(
                        f"<tr><td>{sample_idx}</td><td>{loss:.4f}</td><td>{flagged}</td></tr>"
                    )
                html.append("</table></details>")
            else:
                html.append("<table>")
                html.append("<tr><th>Sample Index</th><th>GCE Loss</th><th>Flagged</th></tr>")
                for pos in sorted_idx:
                    sample_idx = int(pos)
                    loss = per_sample_losses[sample_idx]
                    flagged = "Yes" if sample_idx in minority_set else "No"
                    html.append(
                        f"<tr><td>{sample_idx}</td><td>{loss:.4f}</td><td>{flagged}</td></tr>"
                    )
                html.append("</table>")

        html.append("</div>")
        return "\n".join(html)

    def _generate_causal_effect_section_html(self) -> str:
        """Generate Causal Effect section for HTML."""
        if "causal_effect" not in self.results or not self.results["causal_effect"]["success"]:
            return ""

        result = self.results["causal_effect"]
        metrics = result.get("metrics", {})
        report = result.get("report", {})
        per_attribute = report.get("per_attribute", [])
        risk = self._risk_payload("causal_effect")
        risk_color = self._risk_color(risk["risk_value"])

        html = [
            "<div class='section'>",
            "<h2>📐 Causal Effect Regularization</h2>",
            f"<p><strong>Risk:</strong> <span style='color: {risk_color}; font-weight: 600;'>{risk['risk_label']}</span></p>",
            f"<p><strong>Reason:</strong> {risk['risk_reason']}</p>",
            f"<p><strong>Attributes:</strong> {metrics.get('n_attributes', 0)} | "
            f"<strong>Spurious:</strong> {metrics.get('n_spurious', 0)} | "
            f"<strong>Threshold:</strong> {metrics.get('spurious_threshold')}</p>",
            "<h3>Per-Attribute Effects</h3>",
            "<table>",
            "<tr><th>Attribute</th><th>Causal Effect</th><th>Spurious</th><th>N (a=0)</th><th>N (a=1)</th></tr>",
        ]
        for row in per_attribute:
            spurious = "Yes" if row.get("is_spurious") else "No"
            html.append(
                f"<tr><td>{row.get('attribute_name', '')}</td>"
                f"<td>{row.get('causal_effect', 0):.4f}</td>"
                f"<td>{spurious}</td>"
                f"<td>{row.get('n_samples_a0', 0)}</td>"
                f"<td>{row.get('n_samples_a1', 0)}</td></tr>"
            )
        html.append("</table>")
        html.append("</div>")
        return "\n".join(html)

    def _generate_vae_section_html(self) -> str:
        """Generate VAE section for HTML."""
        if "vae" not in self.results or not self.results["vae"]["success"]:
            return ""

        result = self.results["vae"]
        metrics = result.get("metrics", {})
        report = result.get("report", {})
        per_dimension = report.get("per_dimension", [])
        risk = self._risk_payload("vae")
        risk_color = self._risk_color(risk["risk_value"])

        html = [
            "<div class='section'>",
            "<h2>🖼️ VAE (Variational Autoencoder) Shortcut Detection</h2>",
            f"<p><strong>Risk:</strong> <span style='color: {risk_color}; font-weight: 600;'>{risk['risk_label']}</span></p>",
            f"<p><strong>Reason:</strong> {risk['risk_reason']}</p>",
            f"<p><strong>Latent dims:</strong> {metrics.get('latent_dim', 0)} | "
            f"<strong>Flagged:</strong> {metrics.get('n_flagged', 0)} | "
            f"<strong>Max predictiveness:</strong> {metrics.get('max_predictiveness')}</p>",
            "<h3>Per-Dimension Analysis</h3>",
            "<table>",
            "<tr><th>Dimension</th><th>Predictiveness</th><th>MPWD</th><th>Flagged</th></tr>",
        ]
        for row in per_dimension:
            flagged = "Yes" if row.get("flagged") else "No"
            html.append(
                f"<tr><td>{row.get('dimension', 0)}</td>"
                f"<td>{row.get('predictiveness', 0):.4f}</td>"
                f"<td>{row.get('mpwd', 0):.4f}</td>"
                f"<td>{flagged}</td></tr>"
            )
        html.append("</table>")
        html.append("</div>")
        return "\n".join(html)

    def _generate_early_epoch_section_html(self) -> str:
        """Generate early-epoch clustering section."""
        if (
            "early_epoch_clustering" not in self.results
            or not self.results["early_epoch_clustering"]["success"]
        ):
            return ""

        report = self.results["early_epoch_clustering"]["report"]
        risk = self._risk_payload("early_epoch_clustering")
        risk_color = self._risk_color(risk["risk_value"])
        html = [
            "<div class='section'>",
            "<h2>⏱️ Early-Epoch Clustering (SPARE)</h2>",
            f"<p><strong>Risk:</strong> <span style='color: {risk_color}; font-weight: 600;'>{risk['risk_label']}</span></p>",
            f"<p><strong>Reason:</strong> {risk['risk_reason']}</p>",
            f"<p><strong>Clusters:</strong> {report.n_clusters} | <strong>Method:</strong> {report.cluster_method}</p>",
            f"<p><strong>Entropy:</strong> {report.size_entropy:.3f} | <strong>Minority ratio:</strong> {report.minority_ratio:.3f}</p>",
            f"<p><strong>Largest gap:</strong> {report.largest_gap:.3f}</p>",
            f"<p><strong>Reference:</strong> {report.reference}</p>",
            "<h3>Cluster Sizes</h3>",
            "<table>",
            "<tr><th>Cluster</th><th>Size</th><th>Ratio</th></tr>",
        ]

        for cluster_id, size in report.cluster_sizes.items():
            ratio = report.cluster_ratios.get(cluster_id, float("nan"))
            ratio_str = "nan" if np.isnan(ratio) else f"{ratio:.3f}"
            html.append(f"<tr><td>{cluster_id}</td><td>{size}</td><td>{ratio_str}</td></tr>")

        html.append("</table>")
        if report.cluster_label_agreement is not None:
            html.append(
                f"<p><strong>Cluster label agreement:</strong> {report.cluster_label_agreement:.3f}</p>"
            )
        html.append("</div>")
        return "\n".join(html)

    def _generate_visualizations_html(self) -> str:
        """Generate visualizations section for HTML with embedded plots."""
        if not self.plots:
            return ""

        html = ["<div class='section'>", "<h2>📊 Visualizations</h2>"]

        # PCA and t-SNE in a grid
        if "pca" in self.plots or "tsne" in self.plots:
            html.append("<div class='plot-grid'>")

            if "pca" in self.plots:
                html.append(
                    f"<div class='plot-container'>"
                    f"<h3>PCA Projection</h3>"
                    f"<img src='data:image/png;base64,{self.plots['pca']}' alt='PCA Plot'/>"
                    f"</div>"
                )

            if "tsne" in self.plots:
                html.append(
                    f"<div class='plot-container'>"
                    f"<h3>t-SNE Projection</h3>"
                    f"<img src='data:image/png;base64,{self.plots['tsne']}' alt='t-SNE Plot'/>"
                    f"</div>"
                )

            html.append("</div>")

        # 3D plots: show static image in app, keep interactive HTML for saved report
        if "static_3d" in self.plots:
            html.append(
                f"<div class='plot-container'>"
                f"<h3>3D Embedding Analysis</h3>"
                f"<img src='data:image/png;base64,{self.plots['static_3d']}' alt='3D Embedding Plot'/>"
                f"</div>"
            )
        if "html_3d" in self.plots:
            # Embed Plotly fragment in a full document and load in iframe so scripts run
            # (raw div+script often does not run when embedded in Gradio or saved report)
            fragment = self.plots["html_3d"]
            full_doc = (
                "<!DOCTYPE html><html><head><meta charset='utf-8'>"
                "</head><body style='margin:0; display:flex; justify-content:center; align-items:flex-start; min-height:100vh;'>"
                "<div style='display:inline-block;'>"
                f"{fragment}"
                "</div></body></html>"
            )
            # Escape for srcdoc attribute: " and & so the attribute value is valid
            srcdoc = full_doc.replace("&", "&amp;").replace('"', "&quot;")
            html.append("<details class='plot-container'>")
            html.append("<summary><strong>Interactive 3D</strong></summary>")
            html.append(
                f'<iframe srcdoc="{srcdoc}" '
                "style='width:100%; height:700px; border:1px solid #ddd; border-radius:5px;' "
                "title='Interactive 3D embedding plot'></iframe>"
            )
            # Fallback: if iframe is blocked (e.g. in Gradio), show static 3D so something appears
            if "static_3d" in self.plots:
                html.append(
                    "<p class='plot-container' style='margin-top:10px;'>"
                    "<strong>Static view (if interactive does not load):</strong></p>"
                )
                html.append(
                    f"<img src='data:image/png;base64,{self.plots['static_3d']}' alt='3D static' "
                    "style='max-width:100%; height:auto; border-radius:5px;'/>"
                )
            html.append("</details>")

        # HBAC-specific plots
        if "dimension_importance" in self.plots:
            html.append(
                f"<div class='plot-container'>"
                f"<h3>Dimension Importance (HBAC)</h3>"
                f"<img src='data:image/png;base64,{self.plots['dimension_importance']}' alt='Dimension Importance'/>"
                f"</div>"
            )

        if "cluster_purity" in self.plots:
            html.append(
                f"<div class='plot-container'>"
                f"<h3>Cluster Analysis</h3>"
                f"<img src='data:image/png;base64,{self.plots['cluster_purity']}' alt='Cluster Purity'/>"
                f"</div>"
            )

        # Statistical plots
        if "pvalue_heatmap" in self.plots:
            html.append(
                "<div class='plot-container'>"
                "<h3>Statistical Test P-values</h3>"
                "<p><em>Top 20 most significant dimensions shown by default.</em></p>"
                f"<img src='data:image/png;base64,{self.plots['pvalue_heatmap']}' alt='P-value Heatmap'/>"
            )
            if "pvalue_heatmap_full" in self.plots:
                html.append(
                    "<details style='margin-top: 12px;'>"
                    "<summary style='cursor: pointer; font-weight: 600;'>Expand to full heatmap</summary>"
                    "<div style='margin-top: 8px;'>"
                    f"<img src='data:image/png;base64,{self.plots['pvalue_heatmap_full']}' alt='P-value Heatmap (Full)'/>"
                    "<p style='margin-top: 8px;'>"
                    "<a href='data:image/png;base64,"
                    f"{self.plots['pvalue_heatmap_full']}'"
                    " download='pvalue_heatmap_full.png' "
                    "style='display: inline-block; padding: 6px 12px; background: #1d4ed8; color: white; "
                    "text-decoration: none; border-radius: 4px; font-size: 14px;'>Download full heatmap (PNG)</a>"
                    "</p>"
                    "</div>"
                    "</details>"
                )
            html.append("</div>")

        html.append("</div>")
        return "\n".join(html)

    def _generate_recommendations_html(self) -> str:
        """Generate recommendations section for HTML."""
        recommendations = self._get_recommendations()

        html = ["<div class='section'>", "<h2>💡 Recommendations</h2>"]

        for rec in recommendations:
            # Convert markdown bold (**text**) to HTML (<strong>text</strong>)
            rec_html = rec.replace("**", "BOLD_MARKER")
            parts = rec_html.split("BOLD_MARKER")
            formatted = ""
            for i, part in enumerate(parts):
                if i % 2 == 1:  # Odd indices are bolded text
                    formatted += f"<strong>{part}</strong>"
                else:
                    formatted += part

            html.append(f"<div class='recommendation'>{formatted}</div>")

        html.append("</div>")
        return "\n".join(html)

    def _generate_footer_html(self) -> str:
        """Generate footer section with GitHub link."""
        return f"""
    <div class='footer'>
      <p><strong>More Information</strong></p>
      <p>
        📚 Documentation & Source Code:
        <a href="{self.GITHUB_REPO}" target="_blank">{self.GITHUB_REPO}</a>
      </p>
      <p style="margin-top: 10px; font-size: 12px;">
        Generated with ShortKit-ML |
        Report issues or contribute on GitHub
      </p>
    </div>
        """

    def _get_recommendations(self) -> list:
        """Generate actionable recommendations based on results."""
        recommendations = []

        for reporter in self._reporters:
            reporter.extend_recommendations(self.results, recommendations)

        if "groupdro" in self.results and self.results["groupdro"]["success"]:
            rep = self.results["groupdro"].get("report")
            if rep is not None:
                final = rep.get("final", {})
                avg_acc = final.get("avg_acc", None)
                worst_acc = final.get("worst_group_acc", None)
                if avg_acc is not None and worst_acc is not None:
                    gap = avg_acc - worst_acc
                    if gap > 0.10:
                        recommendations.append(
                            f"**GroupDRO shows large worst-group gap ({gap:.1%}):** "
                            "This suggests severe subgroup performance disparity. Consider "
                            "data balancing, group-aware sampling, or robust training for mitigation."
                        )
                    elif gap > 0.05:
                        recommendations.append(
                            f"**GroupDRO shows moderate worst-group gap ({gap:.1%}):** "
                            "Potential subgroup shortcut/bias. Inspect groups with low accuracy and "
                            "analyze their embedding dimensions or data pipeline."
                        )

        # Check Demographic Parity
        if "demographic_parity" in self.results and self.results["demographic_parity"]["success"]:
            report = self.results["demographic_parity"].get("report")
            if report is not None and hasattr(report, "risk_level"):
                if report.risk_level in {"high", "moderate"}:
                    recommendations.append(
                        f"**Demographic parity gap ({report.dp_gap:.2%}) is {report.risk_level}:** "
                        "Consider mitigating subgroup imbalance via reweighting, "
                        "resampling, or group-aware calibration."
                    )

        # Check Intersectional
        if "intersectional" in self.results and self.results["intersectional"]["success"]:
            report = self.results["intersectional"].get("report")
            if report is not None and hasattr(report, "risk_level"):
                if report.risk_level in {"high", "moderate"}:
                    recommendations.append(
                        f"**Intersectional fairness gaps are {report.risk_level}:** "
                        f"Disparities across demographic intersections ({', '.join(report.attribute_names)}). "
                        "Consider intersection-aware data balancing or subgroup-specific calibration."
                    )

        if "bias_direction_pca" in self.results and self.results["bias_direction_pca"]["success"]:
            report = self.results["bias_direction_pca"].get("report")
            if report is not None and hasattr(report, "risk_level"):
                if report.risk_level in {"high", "moderate"}:
                    recommendations.append(
                        f"**Bias direction gap ({report.projection_gap:.2f}) is {report.risk_level}:** "
                        "Consider debiasing along the PCA direction or auditing group-specific embeddings."
                    )

        if "gce" in self.results and self.results["gce"]["success"]:
            report = self.results["gce"]["report"]
            if report.risk_level in {"high", "moderate"}:
                recommendations.append(
                    f"**GCE flagged {report.n_minority} high-loss samples ({report.risk_level} risk):** "
                    "Review minority/bias-conflicting samples and consider rebalancing or robust training."
                )

        if "cav" in self.results and self.results["cav"]["success"]:
            cav_metrics = self.results["cav"].get("metrics", {})
            n_flagged = cav_metrics.get("n_flagged", 0) or 0
            if n_flagged > 0:
                recommendations.append(
                    f"**CAV flagged {n_flagged} shortcut concepts:** "
                    "Review concept definitions and mitigate reliance on high-TCAV concepts "
                    "using data balancing, concept bottlenecks, or targeted augmentation."
                )

        if not recommendations:
            recommendations.append(
                "**No strong shortcuts detected.** Continue monitoring with larger/different datasets."
            )

        recommendations.append(
            "**General advice:** Always validate findings with domain experts and "
            "test mitigation strategies on held-out data."
        )

        return recommendations

    # GroupDRO
    def _generate_groupdro_section_html(self) -> str:
        """Generate GroupDRO results section for HTML."""
        if "groupdro" not in self.results or not self.results["groupdro"]["success"]:
            return ""

        result = self.results["groupdro"]

        def _render_groupdro(rep) -> list:
            report = rep.get("report", {})
            final = report.get("final", {})
            n_groups = report.get("n_groups", "N/A")
            avg_acc = final.get("avg_acc", None)
            worst_acc = final.get("worst_group_acc", None)
            adv_probs = report.get("final_adv_probs", None)
            lines = []
            lines.append(f"<p><strong>Groups:</strong> {n_groups}</p>")
            if avg_acc is not None:
                lines.append(f"<p><strong>Average accuracy:</strong> {avg_acc:.2%}</p>")
            if worst_acc is not None:
                lines.append(f"<p><strong>Worst-group accuracy:</strong> {worst_acc:.2%}</p>")
            lines.append("<table>")
            lines.append(
                "<tr><th>Group</th><th>Accuracy</th><th>Avg Loss</th><th>Adv Weight (q)</th></tr>"
            )
            n_groups_int = report.get("n_groups", 0)
            if isinstance(n_groups_int, int) and n_groups_int > 0:
                gid_map = report.get("group_id_map", {})
                idx_to_gid = (
                    {idx: gid for gid, idx in gid_map.items()} if isinstance(gid_map, dict) else {}
                )
                for i in range(n_groups_int):
                    gid = idx_to_gid.get(i, i)
                    acc_i = final.get(f"avg_acc_group:{i}", float("nan"))
                    loss_i = final.get(f"avg_loss_group:{i}", float("nan"))
                    q_i = (
                        float(adv_probs[i])
                        if adv_probs is not None and len(adv_probs) > i
                        else float("nan")
                    )
                    lines.append(
                        f"<tr><td><code>{gid}</code></td><td>{acc_i:.2%}</td>"
                        f"<td>{loss_i:.4f}</td><td>{q_i:.4f}</td></tr>"
                    )
            else:
                lines.append("<tr><td colspan='4'>No group statistics available.</td></tr>")
            lines.append("</table>")
            return lines

        html = ["<div class='section'>", "<h2>🛡️ GroupDRO (Worst-Group Robustness)</h2>"]
        if "by_attribute" in result:
            for attr_name, sub in result["by_attribute"].items():
                if sub.get("success"):
                    html.append(f"<h3>Attribute: {attr_name}</h3>")
                    html.extend(_render_groupdro(sub))
        else:
            html.extend(_render_groupdro(result))

        html.append("</div>")
        return "\n".join(html)
