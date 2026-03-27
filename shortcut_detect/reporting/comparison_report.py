"""
Comparison report generation for model comparison mode.
"""

from __future__ import annotations

import os
from typing import TYPE_CHECKING

import numpy as np

if TYPE_CHECKING:
    from ..comparison.runner import ComparisonResult


def _get_comparison_css() -> str:
    """CSS styling for comparison HTML reports."""
    return """
    .shortcut-report, .shortcut-report .container { background: #ffffff !important; color: #000000 !important; }
    .shortcut-report *, .shortcut-report .container * { color: #000000 !important; }
    .shortcut-report th { background: #1d4ed8 !important; color: #ffffff !important; }
    .shortcut-report code { background: unset !important; background-color: rgba(255, 255, 255, 1) !important; color: rgba(21, 87, 36, 1) !important; }
    .shortcut-report .risk-high { color: #991b1b !important; }
    .shortcut-report .risk-moderate { color: #92400e !important; }
    .shortcut-report .risk-low { color: #065f46 !important; }
    .shortcut-report .risk-unknown { color: #4b5563 !important; }
    body { font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif; margin: 0; padding: 20px; background: #edf2f7; color: #111827; }
    .container { max-width: 1200px; margin: 0 auto; background: #ffffff; padding: 40px; border-radius: 12px; box-shadow: 0 12px 35px rgba(15, 23, 42, 0.15); }
    h1 { color: #0f172a; border-bottom: 3px solid #2563eb; padding-bottom: 10px; }
    h2 { color: #1f2933; margin-top: 30px; border-left: 4px solid #2563eb; padding-left: 15px; padding-top: 10px; padding-bottom: 10px; background: #f8fafc; border-radius: 6px; }
    h3 { color: #111827; }
    .section { margin: 20px 0; padding: 20px; background: #ffffff; border-radius: 8px; border: 1px solid #e2e8f0; }
    table { width: 100%; border-collapse: collapse; margin: 15px 0; }
    th { background: #1d4ed8 !important; color: #ffffff !important; padding: 12px; text-align: left; font-weight: 600; }
    td { padding: 10px; border-bottom: 1px solid #e2e8f0; }
    tr:hover { background: #f8fafc; }
    details { margin: 10px 0; padding: 12px; background: #f8fafc; border-radius: 6px; border: 1px solid #e2e8f0; }
    summary { cursor: pointer; font-weight: 600; }
    .footer { margin-top: 40px; padding-top: 20px; border-top: 2px solid #e2e8f0; text-align: center; color: #475569; font-size: 14px; }
    """


def _fmt_val(val) -> str:
    """Format a value for HTML display."""
    if val is None or (isinstance(val, float) and np.isnan(val)):
        return "—"
    if isinstance(val, float):
        if abs(val) < 1e-3 or abs(val) > 1e6:
            return f"{val:.2e}"
        return f"{val:.3f}"
    if isinstance(val, bool):
        return "Yes" if val else "No"
    return str(val)


class ComparisonReportBuilder:
    """
    Generate side-by-side comparison reports from model comparison results.
    """

    def __init__(self, comparison_result: ComparisonResult):
        if not comparison_result.model_ids or not comparison_result.detectors:
            raise ValueError("ComparisonResult must have at least one model")
        self.result = comparison_result

    def to_html(self, output_path: str) -> str:
        """
        Generate HTML comparison report and save to file.

        Args:
            output_path: Path to save HTML file.

        Returns:
            HTML content string.
        """
        content = self._generate_html()
        os.makedirs(os.path.dirname(os.path.abspath(output_path)) or ".", exist_ok=True)
        with open(output_path, "w") as f:
            f.write(content)
        return content

    def _generate_html(self) -> str:
        """Generate HTML report content."""
        df = self.result.summary_table
        model_ids = self.result.model_ids

        html = [
            "<!DOCTYPE html>",
            "<html>",
            "<head>",
            "  <meta charset='utf-8'>",
            "  <title>Model Comparison Report</title>",
            "  <style>",
            _get_comparison_css(),
            "  </style>",
            "</head>",
            "<body class='shortcut-report'>",
            "  <div class='container'>",
            "    <h1>Model Comparison Report</h1>",
            f"    <p><strong>Models compared:</strong> {', '.join(model_ids)}</p>",
            "    <h2>Summary Table</h2>",
            "    <div class='section'>",
            "    <p>Side-by-side metrics for each model. Lower probe AUC and smaller fairness gaps indicate less shortcut bias.</p>",
            "    <table>",
        ]

        # Header row
        cols = list(df.columns)
        html.append("<tr>" + "".join(f"<th>{c}</th>" for c in cols) + "</tr>")

        # Data rows
        for _, row in df.iterrows():
            html.append("<tr>" + "".join(f"<td>{_fmt_val(row[c])}</td>" for c in cols) + "</tr>")

        html.extend(["</table>", "</div>"])

        # Per-model expandable sections
        html.append("<h2>Per-Model Details</h2>")
        for model_id in model_ids:
            detector = self.result.detectors.get(model_id)
            if detector is None:
                continue
            n_samples = len(detector.embeddings_) if detector.embeddings_ is not None else 0
            n_dims = detector.embeddings_.shape[1] if detector.embeddings_ is not None else 0
            methods_str = ", ".join(detector.methods)
            html.extend(
                [
                    "<details>",
                    f"<summary>{model_id} — {n_samples} samples, {n_dims} dims</summary>",
                    f"<p><strong>Methods:</strong> {methods_str}</p>",
                ]
            )
            # Brief summary of results
            summary_lines = []
            if "probe" in detector.results_ and detector.results_["probe"]["success"]:
                m = detector.results_["probe"]["results"]["metrics"]
                summary_lines.append(
                    f"Probe {m.get('metric', 'auc')}: {m.get('metric_value', 0):.3f}"
                )
            if "hbac" in detector.results_ and detector.results_["hbac"]["success"]:
                r = detector.results_["hbac"]["report"]
                summary_lines.append(
                    f"HBAC shortcut: {'Yes' if r['has_shortcut']['exists'] else 'No'}"
                )
            if (
                "equalized_odds" in detector.results_
                and detector.results_["equalized_odds"]["success"]
            ):
                eo = detector.results_["equalized_odds"]
                if "by_attribute" in eo:
                    for an, sub in eo["by_attribute"].items():
                        if sub.get("success") and sub.get("report"):
                            r = sub["report"]
                            summary_lines.append(f"Equalized Odds ({an}) TPR gap: {r.tpr_gap:.3f}")
                else:
                    r = eo["report"]
                    summary_lines.append(f"Equalized Odds TPR gap: {r.tpr_gap:.3f}")
            if (
                "demographic_parity" in detector.results_
                and detector.results_["demographic_parity"]["success"]
            ):
                dp = detector.results_["demographic_parity"]
                if "by_attribute" in dp:
                    for an, sub in dp["by_attribute"].items():
                        if sub.get("success") and sub.get("report"):
                            r = sub["report"]
                            summary_lines.append(f"Demographic Parity ({an}) gap: {r.dp_gap:.3f}")
                else:
                    r = dp["report"]
                    summary_lines.append(f"Demographic Parity gap: {r.dp_gap:.3f}")
            if "sis" in detector.results_ and detector.results_["sis"]["success"]:
                m = detector.results_["sis"].get("metrics", {})
                mean_sis = m.get("mean_sis_size")
                if mean_sis is not None:
                    summary_lines.append(f"SIS mean size: {mean_sis:.1f}")
            if summary_lines:
                html.append("<ul>")
                for line in summary_lines:
                    html.append(f"<li>{line}</li>")
                html.append("</ul>")
            html.append("</details>")

        html.extend(
            [
                "  <div class='footer'>",
                "    Generated by ShortKit-ML — Model Comparison Mode",
                "  </div>",
                "  </div>",
                "</body>",
                "</html>",
            ]
        )

        return "\n".join(html)
