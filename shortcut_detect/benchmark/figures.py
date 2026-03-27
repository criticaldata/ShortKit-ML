"""Publication-quality figures and tables for the paper.

Generates:
- F04: Synthetic benchmark results table (LaTeX)
- F06: Sensitivity analysis line plots

Usage:
    from shortcut_detect.benchmark.figures import (
        generate_synthetic_results_table,
        plot_sensitivity_analysis,
        plot_method_comparison_bar,
        generate_fp_rate_table,
    )
"""

from __future__ import annotations

import os
from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

# ---------------------------------------------------------------------------
# Publication style setup
# ---------------------------------------------------------------------------

_STYLE_APPLIED = False


def _apply_pub_style() -> None:
    """Apply a clean, publication-ready matplotlib style (idempotent)."""
    global _STYLE_APPLIED
    if _STYLE_APPLIED:
        return
    plt.rcParams.update(
        {
            "font.family": "sans-serif",
            "font.size": 10,
            "axes.titlesize": 12,
            "axes.labelsize": 11,
            "xtick.labelsize": 9,
            "ytick.labelsize": 9,
            "legend.fontsize": 9,
            "figure.dpi": 150,
            "savefig.dpi": 300,
            "savefig.bbox": "tight",
            "axes.grid": True,
            "grid.alpha": 0.3,
            "axes.spines.top": False,
            "axes.spines.right": False,
            "lines.linewidth": 1.8,
            "lines.markersize": 5,
        }
    )
    _STYLE_APPLIED = True


# Distinct colour palette for methods; convergence always black/dashed.
METHOD_COLORS: dict[str, str] = {
    "hbac": "#1f77b4",
    "probe": "#ff7f0e",
    "statistical": "#2ca02c",
    "geometric": "#d62728",
    "convergence": "#000000",
}

METHOD_LABELS: dict[str, str] = {
    "hbac": "HBAC",
    "probe": "Probe",
    "statistical": "Statistical",
    "geometric": "Geometric",
    "convergence": "Convergence",
}

# Default effect sizes matching paper grid
DEFAULT_EFFECT_SIZES = [0.2, 0.5, 0.8, 1.2, 2.0]
TABLE_METHODS_ORDER = ["hbac", "probe", "statistical", "geometric", "convergence"]

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _load_df(source: str | os.PathLike[str] | pd.DataFrame) -> pd.DataFrame:
    """Load a DataFrame from a path or pass through if already a DataFrame."""
    if isinstance(source, pd.DataFrame):
        return source
    path = Path(source)
    if not path.exists():
        raise FileNotFoundError(f"CSV not found: {path}")
    return pd.read_csv(path)


# ---------------------------------------------------------------------------
# F04 – Synthetic benchmark results table
# ---------------------------------------------------------------------------


def generate_synthetic_results_table(
    results_df: str | os.PathLike[str] | pd.DataFrame,
    output_path: str | os.PathLike[str] | None = None,
) -> str:
    """Generate a LaTeX-ready table of precision/recall/F1 per method and effect size.

    Parameters
    ----------
    results_df : path or DataFrame
        Must contain columns: effect_size, method, precision, recall, f1.
        Optionally ``fp_rate`` or ``flagged`` (used for FP-rate column).
        Rows with ``method == "convergence"`` are handled specially: they
        report the fraction of seeds reaching ``high_confidence``.
    output_path : path, optional
        If given, writes ``.tex`` and ``.md`` files (suffixes replaced).

    Returns
    -------
    str
        LaTeX table source.
    """
    df = _load_df(results_df)

    # Normalise method names to lowercase
    df = df.copy()
    df["method"] = df["method"].str.lower().str.strip()

    effect_sizes = sorted(df["effect_size"].dropna().unique())
    if not effect_sizes:
        effect_sizes = DEFAULT_EFFECT_SIZES

    methods_present = [m for m in TABLE_METHODS_ORDER if m in df["method"].unique()]

    # Build summary: mean P/R/F1 per (method, effect_size) across seeds/configs
    # Collect best-F1 per effect size for bolding
    best_f1: dict[float, float] = {}
    summary: dict[tuple[str, float], dict[str, float]] = {}

    for method in methods_present:
        for es in effect_sizes:
            sub = df[(df["method"] == method) & (df["effect_size"] == es)]
            if sub.empty:
                summary[(method, es)] = {"p": 0.0, "r": 0.0, "f1": 0.0}
                continue
            if method == "convergence":
                # For convergence, report fraction of runs at high/moderate confidence
                if "convergence_bucket" in sub.columns:
                    n_total = len(sub)
                    n_high = (sub["convergence_bucket"] == "high_confidence").sum()
                    rate = float(n_high / n_total) if n_total else 0.0
                elif "n_flagged_methods" in sub.columns:
                    n_total = len(sub)
                    n_high = (sub["n_flagged_methods"] >= 4).sum()
                    rate = float(n_high / n_total) if n_total else 0.0
                else:
                    rate = 0.0
                summary[(method, es)] = {"p": rate, "r": rate, "f1": rate}
            else:
                p = float(sub["precision"].mean())
                r = float(sub["recall"].mean())
                f = float(sub["f1"].mean())
                summary[(method, es)] = {"p": p, "r": r, "f1": f}
            cur_f1 = summary[(method, es)]["f1"]
            if es not in best_f1 or cur_f1 > best_f1[es]:
                best_f1[es] = cur_f1

    # Compute FP rate per method (from null-control or flagged column)
    fp_rates: dict[str, float] = {}
    if "fp_rate" in df.columns:
        for method in methods_present:
            sub = df[df["method"] == method]
            fp_rates[method] = float(sub["fp_rate"].mean())
    else:
        # Cannot compute per-method FP from runs_paper.csv directly; leave blank
        for method in methods_present:
            fp_rates[method] = float("nan")

    # --- LaTeX ---
    n_es = len(effect_sizes)
    col_spec = "l" + "c" * n_es + "c"  # method + effect sizes + FP rate
    header_es = " & ".join(f"$d={es}$" for es in effect_sizes)
    lines: list[str] = [
        r"\begin{table}[ht]",
        r"\centering",
        r"\caption{Synthetic benchmark: precision / recall / F1 per method and effect size.}",
        r"\label{tab:synthetic_benchmark}",
        rf"\begin{{tabular}}{{{col_spec}}}",
        r"\toprule",
        rf"Method & {header_es} & FP Rate \\",
        r"\midrule",
    ]

    for method in methods_present:
        label = METHOD_LABELS.get(method, method)
        cells: list[str] = []
        for es in effect_sizes:
            s = summary.get((method, es), {"p": 0, "r": 0, "f1": 0})
            cell = f"{s['p']:.2f}/{s['r']:.2f}/{s['f1']:.2f}"
            if abs(s["f1"] - best_f1.get(es, -1)) < 1e-9 and s["f1"] > 0:
                cell = rf"\textbf{{{cell}}}"
            cells.append(cell)
        fp_str = (
            f"{fp_rates[method]:.2f}" if not np.isnan(fp_rates.get(method, float("nan"))) else "--"
        )
        lines.append(f"{label} & " + " & ".join(cells) + f" & {fp_str} " + r"\\")

    lines += [r"\bottomrule", r"\end{tabular}", r"\end{table}"]
    latex = "\n".join(lines)

    # --- Markdown ---
    md_header = "| Method | " + " | ".join(f"d={es}" for es in effect_sizes) + " | FP Rate |"
    md_sep = "|---" * (n_es + 2) + "|"
    md_rows: list[str] = [md_header, md_sep]
    for method in methods_present:
        label = METHOD_LABELS.get(method, method)
        cells = []
        for es in effect_sizes:
            s = summary.get((method, es), {"p": 0, "r": 0, "f1": 0})
            cell = f"{s['p']:.2f}/{s['r']:.2f}/{s['f1']:.2f}"
            if abs(s["f1"] - best_f1.get(es, -1)) < 1e-9 and s["f1"] > 0:
                cell = f"**{cell}**"
            cells.append(cell)
        fp_str = (
            f"{fp_rates[method]:.2f}" if not np.isnan(fp_rates.get(method, float("nan"))) else "--"
        )
        md_rows.append(f"| {label} | " + " | ".join(cells) + f" | {fp_str} |")
    markdown = "\n".join(md_rows)

    # Write files
    if output_path is not None:
        out = Path(output_path)
        out.parent.mkdir(parents=True, exist_ok=True)
        tex_path = out.with_suffix(".tex")
        md_path = out.with_suffix(".md")
        tex_path.write_text(latex, encoding="utf-8")
        md_path.write_text(markdown, encoding="utf-8")

    return latex


# ---------------------------------------------------------------------------
# F06 – Sensitivity analysis plots
# ---------------------------------------------------------------------------

SweepInput = str | os.PathLike[str] | pd.DataFrame | dict[str, Any]


def plot_sensitivity_analysis(
    sweep_results: SweepInput,
    output_dir: str | os.PathLike[str] | None = None,
) -> list[matplotlib.figure.Figure]:
    """Create sensitivity analysis line plots.

    Parameters
    ----------
    sweep_results : DataFrame, path to CSV, or dict of DataFrames/paths
        If a single DataFrame/path, it must contain columns ``method``,
        ``flagged`` (or ``f1``), and one or more of ``n_samples``,
        ``imbalance_ratio``, ``embedding_dim``.  If a dict, keys are sweep
        type names (``"sample_size"``, ``"imbalance"``, ``"embedding_dim"``)
        mapped to DataFrames or CSV paths.

    output_dir : path, optional
        Directory to save PNG figures.

    Returns
    -------
    list[matplotlib.figure.Figure]
        Up to 3 figures (sample size, imbalance, embedding dim).
    """
    _apply_pub_style()

    # Normalise input to a single DataFrame
    if isinstance(sweep_results, dict):
        frames = []
        for _key, val in sweep_results.items():
            frames.append(_load_df(val))
        df = pd.concat(frames, ignore_index=True)
    else:
        df = _load_df(sweep_results)

    df = df.copy()
    if "method" in df.columns:
        df["method"] = df["method"].str.lower().str.strip()

    # Determine metric column: prefer detection_rate, then flagged, then f1
    if "detection_rate" in df.columns:
        metric_col = "detection_rate"
    elif "flagged" in df.columns:
        metric_col = "flagged"
    elif "f1" in df.columns:
        metric_col = "f1"
    else:
        raise ValueError("sweep_results must contain 'detection_rate', 'flagged', or 'f1' column")

    sweep_configs: list[tuple[str, str, str, str]] = [
        ("n_samples", "Sample size", "Detection rate", "sensitivity_sample_size.png"),
        ("imbalance_ratio", "Class imbalance ratio", "Detection rate", "sensitivity_imbalance.png"),
        (
            "embedding_dim",
            "Embedding dimensionality",
            "Detection rate",
            "sensitivity_embedding_dim.png",
        ),
    ]

    figures: list[matplotlib.figure.Figure] = []
    for x_col, xlabel, ylabel, fname in sweep_configs:
        if x_col not in df.columns:
            continue

        fig, ax = plt.subplots(figsize=(5, 3.5))
        methods = sorted(df["method"].unique())

        for method in methods:
            sub = df[df["method"] == method]
            grouped = sub.groupby(x_col)[metric_col]
            means = grouped.mean()
            stds = grouped.std().fillna(0)

            xs = means.index.to_numpy()
            ys = means.values
            yerr = stds.values

            color = METHOD_COLORS.get(method, None)
            label = METHOD_LABELS.get(method, method)
            ls = "--" if method == "convergence" else "-"
            marker = "s" if method == "convergence" else "o"

            ax.plot(xs, ys, color=color, linestyle=ls, marker=marker, label=label)
            ax.fill_between(xs, ys - yerr, ys + yerr, alpha=0.15, color=color)

        ax.set_xlabel(xlabel)
        ax.set_ylabel(ylabel)
        ax.set_ylim(-0.05, 1.05)
        ax.legend(frameon=True, framealpha=0.9)
        fig.tight_layout()
        figures.append(fig)

        if output_dir is not None:
            odir = Path(output_dir)
            odir.mkdir(parents=True, exist_ok=True)
            fig.savefig(str(odir / fname))

    return figures


# ---------------------------------------------------------------------------
# Helper – method comparison bar chart
# ---------------------------------------------------------------------------


def plot_method_comparison_bar(
    results_df: str | os.PathLike[str] | pd.DataFrame,
    metric: str = "f1",
    output_path: str | os.PathLike[str] | None = None,
) -> matplotlib.figure.Figure:
    """Grouped bar chart comparing methods across effect sizes.

    Parameters
    ----------
    results_df : path or DataFrame
        Must contain columns: ``effect_size``, ``method``, and *metric*.
    metric : str
        Column to plot (default ``"f1"``).
    output_path : path, optional
        If given, save figure to this path.

    Returns
    -------
    matplotlib.figure.Figure
    """
    _apply_pub_style()
    df = _load_df(results_df)
    df = df.copy()
    df["method"] = df["method"].str.lower().str.strip()

    if metric not in df.columns:
        raise ValueError(f"Metric column '{metric}' not found in DataFrame")

    effect_sizes = sorted(df["effect_size"].dropna().unique())
    methods = [m for m in TABLE_METHODS_ORDER if m in df["method"].unique() and m != "convergence"]

    agg = df.groupby(["method", "effect_size"])[metric].mean().reset_index()

    n_methods = len(methods)
    n_es = len(effect_sizes)
    x = np.arange(n_es)
    width = 0.8 / max(n_methods, 1)

    fig, ax = plt.subplots(figsize=(6, 4))
    for i, method in enumerate(methods):
        vals = []
        for es in effect_sizes:
            row = agg[(agg["method"] == method) & (agg["effect_size"] == es)]
            vals.append(float(row[metric].values[0]) if len(row) else 0.0)
        color = METHOD_COLORS.get(method, None)
        label = METHOD_LABELS.get(method, method)
        offset = (i - n_methods / 2 + 0.5) * width
        ax.bar(x + offset, vals, width, label=label, color=color, edgecolor="white", linewidth=0.5)

    ax.set_xlabel("Effect size")
    ax.set_ylabel(metric.upper())
    ax.set_xticks(x)
    ax.set_xticklabels([str(es) for es in effect_sizes])
    ax.set_ylim(0, 1.05)
    ax.legend(frameon=True, framealpha=0.9)
    fig.tight_layout()

    if output_path is not None:
        out = Path(output_path)
        out.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(str(out))

    return fig


# ---------------------------------------------------------------------------
# Null-data false positive rate table
# ---------------------------------------------------------------------------


def generate_fp_rate_table(
    fp_results: Any,
    output_path: str | os.PathLike[str] | None = None,
) -> str:
    """Generate a LaTeX table of false positive rates from FalsePositiveAnalyzer results.

    Parameters
    ----------
    fp_results : FPResult or dict
        Must contain ``method_fp_rates`` (dict mapping method -> FP rate) and
        ``convergence_fp_rate`` (float).  Optionally ``per_seed_results`` (a
        DataFrame with ``method`` and ``flagged`` columns) and additional
        per-correction FP rates stored in ``bonferroni_fp_rates`` and
        ``fdr_bh_fp_rates`` dicts.  If those are absent the single FP rate is
        used for both correction columns.
    output_path : path, optional
        If given, writes ``.tex`` and ``.md`` files (suffixes replaced).

    Returns
    -------
    str
        LaTeX table source.
    """
    if isinstance(fp_results, dict):
        method_fp_rates = fp_results.get("method_fp_rates", {})
        convergence_fp_rate = fp_results.get("convergence_fp_rate", 0.0)
        bonferroni_rates = fp_results.get("bonferroni_fp_rates", method_fp_rates)
        fdr_bh_rates = fp_results.get("fdr_bh_fp_rates", method_fp_rates)
    else:
        method_fp_rates = getattr(fp_results, "method_fp_rates", {})
        convergence_fp_rate = getattr(fp_results, "convergence_fp_rate", 0.0)
        bonferroni_rates = getattr(fp_results, "bonferroni_fp_rates", method_fp_rates)
        fdr_bh_rates = getattr(fp_results, "fdr_bh_fp_rates", method_fp_rates)

    methods_order = [m for m in TABLE_METHODS_ORDER if m in method_fp_rates and m != "convergence"]

    # --- LaTeX ---
    lines: list[str] = [
        r"\begin{table}[ht]",
        r"\centering",
        r"\caption{False positive rates on null (no-shortcut) data.}",
        r"\label{tab:fp_rates}",
        r"\begin{tabular}{lccc}",
        r"\toprule",
        r"Method & FP Rate (Bonferroni) & FP Rate (FDR-BH) & Convergence FP Rate \\",
        r"\midrule",
    ]

    for method in methods_order:
        label = METHOD_LABELS.get(method, method)
        bonf = bonferroni_rates.get(method, method_fp_rates.get(method, float("nan")))
        fdr = fdr_bh_rates.get(method, method_fp_rates.get(method, float("nan")))
        bonf_str = f"{bonf:.3f}" if not np.isnan(bonf) else "--"
        fdr_str = f"{fdr:.3f}" if not np.isnan(fdr) else "--"
        conv_str = f"{convergence_fp_rate:.3f}"
        lines.append(f"{label} & {bonf_str} & {fdr_str} & {conv_str} " + r"\\")

    lines += [r"\bottomrule", r"\end{tabular}", r"\end{table}"]
    latex = "\n".join(lines)

    # --- Markdown ---
    md_lines: list[str] = [
        "| Method | FP Rate (Bonferroni) | FP Rate (FDR-BH) | Convergence FP Rate |",
        "|---|---|---|---|",
    ]
    for method in methods_order:
        label = METHOD_LABELS.get(method, method)
        bonf = bonferroni_rates.get(method, method_fp_rates.get(method, float("nan")))
        fdr = fdr_bh_rates.get(method, method_fp_rates.get(method, float("nan")))
        bonf_str = f"{bonf:.3f}" if not np.isnan(bonf) else "--"
        fdr_str = f"{fdr:.3f}" if not np.isnan(fdr) else "--"
        conv_str = f"{convergence_fp_rate:.3f}"
        md_lines.append(f"| {label} | {bonf_str} | {fdr_str} | {conv_str} |")
    markdown = "\n".join(md_lines)

    if output_path is not None:
        out = Path(output_path)
        out.parent.mkdir(parents=True, exist_ok=True)
        out.with_suffix(".tex").write_text(latex, encoding="utf-8")
        out.with_suffix(".md").write_text(markdown, encoding="utf-8")

    return latex
