"""Convergence matrix visualization for multi-method shortcut detection.

Generates publication-quality heatmaps showing method agreement across experiments.
"""

from __future__ import annotations

import matplotlib

matplotlib.use("Agg")

from pathlib import Path  # noqa: E402

import matplotlib.colors as mcolors  # noqa: E402
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402


class ConvergenceMatrix:
    """Matrix of per-method detection results across experimental conditions.

    Rows correspond to detection methods and columns to experiments.  Each cell
    is a boolean indicating whether the method flagged a shortcut in that
    experiment.

    Parameters
    ----------
    methods : list[str]
        Ordered list of detection method names (row labels).
    """

    def __init__(self, methods: list[str]) -> None:
        self.methods = list(methods)
        # experiment_name -> {method: bool}
        self._experiments: dict[str, dict[str, bool]] = {}

    # -- building the matrix --------------------------------------------------

    def add_experiment(self, name: str, results: dict[str, bool]) -> None:
        """Add one experimental condition (column) with per-method flags.

        Parameters
        ----------
        name : str
            Label for the experiment / condition.
        results : dict[str, bool]
            Mapping of method name to detected flag.  Methods not present in
            ``self.methods`` are silently ignored; methods missing from
            *results* default to ``False``.
        """
        self._experiments[name] = {m: bool(results.get(m, False)) for m in self.methods}

    @classmethod
    def from_dataframe(
        cls,
        df: pd.DataFrame,
        method_col: str = "method",
        experiment_col: str = "experiment",
        detected_col: str = "flagged",
    ) -> ConvergenceMatrix:
        """Build a matrix from a long-form DataFrame.

        Parameters
        ----------
        df : DataFrame
            Must contain at least the three columns named by the remaining
            parameters.
        method_col : str
            Column holding detection method names.
        experiment_col : str
            Column holding experiment / condition labels.
        detected_col : str
            Column holding boolean-ish detection flags (truthy / falsy).
        """
        methods = list(df[method_col].unique())
        mat = cls(methods)
        for exp_name, grp in df.groupby(experiment_col):
            results = {row[method_col]: bool(row[detected_col]) for _, row in grp.iterrows()}
            mat.add_experiment(str(exp_name), results)
        return mat

    @classmethod
    def from_benchmark_results(cls, runs_df: pd.DataFrame) -> ConvergenceMatrix:
        """Build from :class:`PaperBenchmarkRunner` output.

        The runner produces rows with columns ``method``, ``flagged``, plus
        varying condition columns (``effect_size``, ``backbone``, ``attribute``,
        etc.).  This helper auto-detects the dataset type and constructs a
        human-readable experiment label for each unique condition.

        Parameters
        ----------
        runs_df : DataFrame
            The ``runs`` or ``chexpert_methods`` frame returned by
            :meth:`PaperBenchmarkRunner.run`.
        """
        # Filter to actual methods (exclude convergence summary rows)
        df = runs_df[runs_df["method"] != "convergence"].copy()
        methods = list(df["method"].unique())

        mat = cls(methods)

        # Determine condition columns based on what is available
        is_chexpert = "backbone" in df.columns and "attribute" in df.columns
        if is_chexpert:
            condition_cols = ["backbone", "attribute"]
        else:
            # Synthetic grid – use whatever numeric condition columns exist
            candidates = [
                "effect_size",
                "n_samples",
                "imbalance_ratio",
                "embedding_dim",
                "seed",
            ]
            condition_cols = [c for c in candidates if c in df.columns]

        if not condition_cols:
            # Fallback: use row index as experiment label
            for i, (_, row) in enumerate(df.iterrows()):
                name = f"exp_{i}"
                mat.add_experiment(name, {row["method"]: bool(row.get("flagged", False))})
            return mat

        for keys, grp in df.groupby(condition_cols):
            if not isinstance(keys, tuple):
                keys = (keys,)
            label_parts = [f"{col}={v}" for col, v in zip(condition_cols, keys, strict=False)]
            label = " | ".join(label_parts)
            results = {row["method"]: bool(row["flagged"]) for _, row in grp.iterrows()}
            mat.add_experiment(label, results)

        return mat

    # -- accessors ------------------------------------------------------------

    @property
    def experiment_names(self) -> list[str]:
        """Ordered list of experiment names (column labels)."""
        return list(self._experiments.keys())

    def to_dataframe(self) -> pd.DataFrame:
        """Return the binary detection matrix as a DataFrame.

        Rows are methods, columns are experiments.  Values are ``True`` /
        ``False``.
        """
        data = {
            exp: [self._experiments[exp].get(m, False) for m in self.methods]
            for exp in self._experiments
        }
        return pd.DataFrame(data, index=self.methods)

    def agreement_levels(self) -> dict[str, str]:
        """Per-experiment agreement string, e.g. ``"3/4"``.

        Returns a mapping from experiment name to a string like ``"3/4"``
        indicating how many of the ``n`` methods flagged a shortcut.
        """
        n = len(self.methods)
        result: dict[str, str] = {}
        for exp, method_flags in self._experiments.items():
            count = sum(1 for m in self.methods if method_flags.get(m, False))
            result[exp] = f"{count}/{n}"
        return result

    def agreement_counts(self) -> dict[str, int]:
        """Per-experiment agreement count (number of methods that flagged)."""
        result: dict[str, int] = {}
        for exp, method_flags in self._experiments.items():
            result[exp] = sum(1 for m in self.methods if method_flags.get(m, False))
        return result


# ---------------------------------------------------------------------------
# Plotting helpers
# ---------------------------------------------------------------------------

_DEFAULT_AGREEMENT_CMAP = {
    0: "#bfbfbf",  # medium gray (more visible than light gray)
    1: "#e8601c",  # dark orange
    2: "#f1c232",  # gold
    3: "#6aa84f",  # medium green
    4: "#0b6623",  # forest green
}


def _build_colormap(n_methods: int) -> tuple[mcolors.ListedColormap, mcolors.BoundaryNorm]:
    """Build a discrete colormap for *n_methods* agreement levels (0..n)."""
    if n_methods <= 4:
        colors = [_DEFAULT_AGREEMENT_CMAP.get(i, "#d9d9d9") for i in range(n_methods + 1)]
    else:
        # For >4 methods, interpolate a green ramp with gray at 0
        cmap_base = matplotlib.colormaps.get_cmap("RdYlGn").resampled(n_methods + 1)
        colors = [mcolors.to_hex(cmap_base(i / n_methods)) for i in range(n_methods + 1)]
        colors[0] = "#d9d9d9"
    cmap = mcolors.ListedColormap(colors)
    bounds = np.arange(-0.5, n_methods + 1.5, 1)
    norm = mcolors.BoundaryNorm(bounds, cmap.N)
    return cmap, norm


def plot_convergence_matrix(
    matrix: ConvergenceMatrix,
    *,
    title: str = "Method Convergence Matrix",
    figsize: tuple[float, float] | None = None,
    dpi: int = 150,
    font_size: int = 10,
    save_path: str | Path | None = None,
    show_agreement_row: bool = True,
) -> plt.Figure:
    """Render a convergence heatmap.

    Parameters
    ----------
    matrix : ConvergenceMatrix
        The convergence data to plot.
    title : str
        Figure title.
    figsize : tuple, optional
        ``(width, height)`` in inches.  If *None*, auto-sized from the matrix.
    dpi : int
        Resolution for raster output.
    font_size : int
        Base font size for labels and annotations.
    save_path : str or Path, optional
        If provided, the figure is saved to this path (format inferred from
        extension).
    show_agreement_row : bool
        Whether to add a summary row at the bottom showing agreement levels.

    Returns
    -------
    matplotlib.figure.Figure
    """
    df = matrix.to_dataframe()  # methods x experiments
    n_methods = len(matrix.methods)
    n_experiments = len(matrix.experiment_names)

    if n_experiments == 0:
        fig, ax = plt.subplots(1, 1, figsize=(4, 2), dpi=dpi)
        ax.text(0.5, 0.5, "No experiments", ha="center", va="center", fontsize=font_size)
        ax.set_axis_off()
        if save_path:
            fig.savefig(str(save_path), dpi=dpi, bbox_inches="tight")
        return fig

    # Numeric matrix: per-cell = agreement count for the column, but only
    # for rows where the method flagged.  For the heatmap coloring we color
    # each *column* by the total agreement count.
    agreement = matrix.agreement_counts()
    agreement_vals = np.array([agreement[e] for e in matrix.experiment_names])

    # Binary matrix (methods x experiments)
    bool_mat = df.values.astype(int)  # 1/0

    # Column-level color matrix: each cell in a column gets the column's
    # agreement level so the whole column has a uniform color.
    col_color = np.tile(agreement_vals, (n_methods, 1))

    extra_rows = 1 if show_agreement_row else 0
    if figsize is None:
        figsize = (max(6, n_experiments * 0.9 + 2), max(3, (n_methods + extra_rows) * 0.6 + 2))

    fig, ax = plt.subplots(1, 1, figsize=figsize, dpi=dpi)

    cmap, norm = _build_colormap(n_methods)

    # Plot heatmap
    im = ax.imshow(col_color, aspect="auto", cmap=cmap, norm=norm, origin="upper")

    # Annotate each cell with check / cross
    for i in range(n_methods):
        for j in range(n_experiments):
            marker = "\u2713" if bool_mat[i, j] else "\u2717"
            color = "white" if agreement_vals[j] >= n_methods else "black"
            ax.text(j, i, marker, ha="center", va="center", fontsize=font_size, color=color)

    # Agreement summary row
    if show_agreement_row:
        levels = matrix.agreement_levels()
        for j, exp in enumerate(matrix.experiment_names):
            ax.text(
                j,
                n_methods - 0.5 + 0.45,
                levels[exp],
                ha="center",
                va="center",
                fontsize=font_size - 1,
                fontweight="bold",
                color="#333333",
            )
        ax.set_ylim(n_methods - 0.5 + 0.8, -0.5)

    # Labels
    ax.set_yticks(range(n_methods))
    ax.set_yticklabels(matrix.methods, fontsize=font_size)
    ax.set_xticks(range(n_experiments))
    ax.set_xticklabels(matrix.experiment_names, fontsize=font_size - 1, rotation=45, ha="right")
    ax.set_title(title, fontsize=font_size + 2, pad=12)

    # Colorbar
    cbar = fig.colorbar(im, ax=ax, ticks=range(n_methods + 1), shrink=0.8)
    cbar.set_label("Methods agreeing", fontsize=font_size)
    cbar.ax.set_yticklabels(
        [f"{i}/{n_methods}" for i in range(n_methods + 1)], fontsize=font_size - 1
    )

    fig.tight_layout()

    if save_path:
        fig.savefig(str(save_path), dpi=dpi, bbox_inches="tight")

    return fig


def plot_agreement_summary(
    matrix: ConvergenceMatrix,
    *,
    title: str = "Agreement Level Distribution",
    figsize: tuple[float, float] | None = None,
    dpi: int = 150,
    font_size: int = 10,
    save_path: str | Path | None = None,
) -> plt.Figure:
    """Bar chart showing how many experiments fall into each agreement level.

    Parameters
    ----------
    matrix : ConvergenceMatrix
        The convergence data.
    title : str
        Figure title.
    figsize : tuple, optional
        ``(width, height)`` in inches.
    dpi : int
        Resolution.
    font_size : int
        Base font size.
    save_path : str or Path, optional
        Path to save figure.

    Returns
    -------
    matplotlib.figure.Figure
    """
    n_methods = len(matrix.methods)
    counts = matrix.agreement_counts()
    count_values = list(counts.values())

    # Tally: how many experiments have 0/n, 1/n, ..., n/n agreement
    bins = list(range(n_methods + 1))
    tallies = [count_values.count(b) for b in bins]
    labels = [f"{b}/{n_methods}" for b in bins]

    if figsize is None:
        figsize = (max(4, (n_methods + 1) * 0.8 + 1), 3.5)

    fig, ax = plt.subplots(1, 1, figsize=figsize, dpi=dpi)

    # Colors matching the heatmap
    if n_methods <= 4:
        bar_colors = [_DEFAULT_AGREEMENT_CMAP.get(i, "#d9d9d9") for i in bins]
    else:
        bar_colors = ["#d9d9d9"] * (n_methods + 1)

    ax.bar(labels, tallies, color=bar_colors, edgecolor="#555555", linewidth=0.5)
    ax.set_xlabel("Agreement level", fontsize=font_size)
    ax.set_ylabel("Number of experiments", fontsize=font_size)
    ax.set_title(title, fontsize=font_size + 2)
    ax.tick_params(labelsize=font_size - 1)

    # Integer y-ticks
    max_tally = max(tallies) if tallies else 1
    ax.set_yticks(range(0, max_tally + 2))

    fig.tight_layout()

    if save_path:
        fig.savefig(str(save_path), dpi=dpi, bbox_inches="tight")

    return fig
