"""
Visualization generation for shortcut detection reports.
"""

import matplotlib
import numpy as np

matplotlib.use("Agg")  # Non-interactive backend for server environments
import base64
from io import BytesIO
from typing import TYPE_CHECKING

import matplotlib.pyplot as plt
import pandas as pd
import plotly.express as px
import plotly.io as pio
import seaborn as sns
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE

if TYPE_CHECKING:
    from ..unified import ShortcutDetector


def generate_pca_plot(
    embeddings: np.ndarray, labels: np.ndarray, title: str = "PCA Visualization of Embeddings"
) -> str:
    """
    Generate PCA plot and return as base64 encoded PNG.

    Args:
        embeddings: (n_samples, embedding_dim) array
        labels: (n_samples,) labels for coloring
        title: Plot title

    Returns:
        Base64 encoded PNG image string
    """
    fig, ax = plt.subplots(figsize=(10, 7))

    # Apply PCA
    pca = PCA(n_components=2, random_state=42)
    embeddings_2d = pca.fit_transform(embeddings)

    # Plot
    unique_labels = np.unique(labels)
    colors = plt.cm.tab10(np.linspace(0, 1, len(unique_labels)))

    for label, color in zip(unique_labels, colors, strict=False):
        mask = labels == label
        ax.scatter(
            embeddings_2d[mask, 0],
            embeddings_2d[mask, 1],
            c=[color],
            label=f"Label {label}",
            alpha=0.6,
            s=50,
            edgecolors="white",
            linewidth=0.5,
        )

    ax.set_xlabel(f"PC1 ({pca.explained_variance_ratio_[0]:.1%} variance)", fontsize=12)
    ax.set_ylabel(f"PC2 ({pca.explained_variance_ratio_[1]:.1%} variance)", fontsize=12)
    ax.set_title(title, fontsize=14, fontweight="bold")
    ax.legend(loc="best", framealpha=0.9)
    ax.grid(True, alpha=0.3)

    plt.tight_layout()

    # Convert to base64
    buf = BytesIO()
    plt.savefig(buf, format="png", dpi=150, bbox_inches="tight")
    buf.seek(0)
    img_base64 = base64.b64encode(buf.read()).decode("utf-8")
    plt.close(fig)

    return img_base64


def generate_tsne_plot(
    embeddings: np.ndarray,
    labels: np.ndarray,
    title: str = "t-SNE Visualization of Embeddings",
    perplexity: int = 30,
) -> str:
    """
    Generate t-SNE plot and return as base64 encoded PNG.

    Args:
        embeddings: (n_samples, embedding_dim) array
        labels: (n_samples,) labels for coloring
        title: Plot title
        perplexity: t-SNE perplexity parameter

    Returns:
        Base64 encoded PNG image string
    """
    fig, ax = plt.subplots(figsize=(10, 7))

    # Apply t-SNE (limit perplexity if too few samples)
    n_samples = len(embeddings)
    perplexity = min(perplexity, max(5, n_samples // 4))

    tsne = TSNE(n_components=2, random_state=42, perplexity=perplexity)
    embeddings_2d = tsne.fit_transform(embeddings)

    # Plot
    unique_labels = np.unique(labels)
    colors = plt.cm.tab10(np.linspace(0, 1, len(unique_labels)))

    for label, color in zip(unique_labels, colors, strict=False):
        mask = labels == label
        ax.scatter(
            embeddings_2d[mask, 0],
            embeddings_2d[mask, 1],
            c=[color],
            label=f"Label {label}",
            alpha=0.6,
            s=50,
            edgecolors="white",
            linewidth=0.5,
        )

    ax.set_xlabel("t-SNE Dimension 1", fontsize=12)
    ax.set_ylabel("t-SNE Dimension 2", fontsize=12)
    ax.set_title(title, fontsize=14, fontweight="bold")
    ax.legend(loc="best", framealpha=0.9)
    ax.grid(True, alpha=0.3)

    plt.tight_layout()

    # Convert to base64
    buf = BytesIO()
    plt.savefig(buf, format="png", dpi=150, bbox_inches="tight")
    buf.seek(0)
    img_base64 = base64.b64encode(buf.read()).decode("utf-8")
    plt.close(fig)

    return img_base64


def generate_dimension_importance_plot(
    dimension_importance_df, title: str = "Top Important Dimensions (HBAC)", top_k: int = 15
) -> str:
    """
    Generate bar plot of dimension importance.

    Args:
        dimension_importance_df: DataFrame with 'dimension' and 'f_score' columns
        title: Plot title
        top_k: Number of top dimensions to show

    Returns:
        Base64 encoded PNG image string
    """
    fig, ax = plt.subplots(figsize=(10, 6))

    # Get top K dimensions
    top_dims = dimension_importance_df.head(top_k)

    # Plot
    ax.barh(
        range(len(top_dims)), top_dims["f_score"], color="steelblue", edgecolor="navy", alpha=0.7
    )

    ax.set_yticks(range(len(top_dims)))
    ax.set_yticklabels(top_dims["dimension"])
    ax.set_xlabel("F-score", fontsize=12)
    ax.set_ylabel("Dimension", fontsize=12)
    ax.set_title(title, fontsize=14, fontweight="bold")
    ax.grid(axis="x", alpha=0.3)

    # Invert y-axis so highest is at top
    ax.invert_yaxis()

    plt.tight_layout()

    # Convert to base64
    buf = BytesIO()
    plt.savefig(buf, format="png", dpi=150, bbox_inches="tight")
    buf.seek(0)
    img_base64 = base64.b64encode(buf.read()).decode("utf-8")
    plt.close(fig)

    return img_base64


def generate_pvalue_heatmap(
    p_values_dict: dict,
    corrected_pvalues_dict: dict | None = None,
    title: str = "P-values Heatmap (Statistical Tests)",
    top_k: int = 20,
    max_features_full: int = 100,
    show_all: bool = False,
) -> str | None:
    """
    Generate heatmap of p-values across features and comparisons.

    By default shows top-K most significant dimensions (smallest p-values).
    Use show_all=True for full heatmap (up to max_features_full dimensions).

    Args:
        p_values_dict: Dictionary mapping comparison names to p-value arrays
        corrected_pvalues_dict: If provided, used for significance ranking and display
        title: Plot title
        top_k: Number of most significant dimensions to show (when show_all=False)
        max_features_full: Maximum dimensions when show_all=True (avoids huge images)
        show_all: If True, show all dimensions (up to max_features_full) instead of top-K

    Returns:
        Base64 encoded PNG image string, or None if no data
    """
    pvals_to_use = corrected_pvalues_dict if corrected_pvalues_dict else p_values_dict
    if not pvals_to_use or all(v is None for v in pvals_to_use.values()):
        return None

    # Build full matrix and determine dimension indices to show
    comparison_names = []
    full_matrix_rows = []
    n_dims = 0

    for comp_name, p_vals in pvals_to_use.items():
        if p_vals is not None and len(p_vals) > 0:
            comparison_names.append(comp_name)
            full_matrix_rows.append(np.asarray(p_vals, dtype=np.float64))
            n_dims = max(n_dims, len(p_vals))

    if not full_matrix_rows:
        return None

    # Pad rows to same length
    full_matrix = np.array(
        [np.pad(row, (0, n_dims - len(row)), constant_values=np.nan) for row in full_matrix_rows]
    )

    # Per-dimension min p-value (most significant across comparisons)
    with np.errstate(invalid="ignore"):
        min_p_per_dim = np.nanmin(full_matrix, axis=0)
    valid_dims = np.isfinite(min_p_per_dim)
    if not np.any(valid_dims):
        return None

    if show_all:
        n_show = min(n_dims, max_features_full)
        dim_indices = np.arange(n_show)
    else:
        # Sort by min p-value ascending (most significant first), take top_k
        sorted_indices = np.argsort(min_p_per_dim)
        n_show = min(top_k, np.sum(valid_dims))
        dim_indices = sorted_indices[:n_show]

    # Slice matrix to selected dimensions
    p_value_matrix = full_matrix[:, dim_indices]

    # Abbreviated dimension labels
    n_cols = p_value_matrix.shape[1]
    xticklabels = [f"D{dim_indices[i]}" for i in range(n_cols)]

    # Plot heatmap
    fig_h = max(5, len(comparison_names) * 0.6)
    fig_w = max(10, min(14, n_cols * 0.35))
    fig, ax = plt.subplots(figsize=(fig_w, fig_h))

    with np.errstate(divide="ignore"):
        log_p_values = -np.log10(p_value_matrix + 1e-300)

    sns.heatmap(
        log_p_values,
        cmap="RdYlGn_r",
        cbar_kws={"label": "-log10(p-value)"},
        yticklabels=comparison_names,
        xticklabels=xticklabels,
        ax=ax,
        vmin=0,
        vmax=5,
        linewidths=0.5,
        linecolor="gray",
    )

    ax.set_xlabel("Feature Dimension", fontsize=12)
    ax.set_ylabel("Group Comparison", fontsize=12)
    ax.set_title(title, fontsize=14, fontweight="bold")
    plt.xticks(rotation=45, ha="right")

    plt.tight_layout()

    buf = BytesIO()
    plt.savefig(buf, format="png", dpi=150, bbox_inches="tight")
    buf.seek(0)
    img_base64 = base64.b64encode(buf.read()).decode("utf-8")
    plt.close(fig)

    return img_base64


def generate_cluster_purity_plot(
    cluster_purities: list, title: str = "Cluster Purity Analysis"
) -> str:
    """
    Generate bar plot of cluster purities.

    Args:
        cluster_purities: List of dicts with 'cluster_id', 'size', 'purity' keys
        title: Plot title

    Returns:
        Base64 encoded PNG image string
    """
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))

    cluster_ids = [cp["cluster_id"] for cp in cluster_purities]
    purities = [cp["purity"] for cp in cluster_purities]
    sizes = [cp["size"] for cp in cluster_purities]

    # Plot 1: Cluster purities
    ax1.bar(cluster_ids, purities, color="cornflowerblue", edgecolor="navy", alpha=0.7)
    ax1.axhline(
        y=0.8, color="red", linestyle="--", linewidth=2, label="High purity threshold (0.8)"
    )
    ax1.set_xlabel("Cluster ID", fontsize=12)
    ax1.set_ylabel("Purity", fontsize=12)
    ax1.set_title("Cluster Purity", fontsize=13, fontweight="bold")
    ax1.set_ylim([0, 1.05])
    ax1.legend()
    ax1.grid(axis="y", alpha=0.3)

    # Plot 2: Cluster sizes
    ax2.bar(cluster_ids, sizes, color="coral", edgecolor="darkred", alpha=0.7)
    ax2.set_xlabel("Cluster ID", fontsize=12)
    ax2.set_ylabel("Cluster Size", fontsize=12)
    ax2.set_title("Cluster Size Distribution", fontsize=13, fontweight="bold")
    ax2.grid(axis="y", alpha=0.3)

    plt.suptitle(title, fontsize=14, fontweight="bold", y=1.02)
    plt.tight_layout()

    # Convert to base64
    buf = BytesIO()
    plt.savefig(buf, format="png", dpi=150, bbox_inches="tight")
    buf.seek(0)
    img_base64 = base64.b64encode(buf.read()).decode("utf-8")
    plt.close(fig)

    return img_base64


def generate_interactive_3d_plot(
    embeddings: np.ndarray,
    labels: np.ndarray,
    title: str = "3D Embedding Visualization",
    method: str = "pca",
    sample_names=None,
) -> str:
    """
    Generate an interactive 3D Plotly visualization and return an embeddable HTML snippet.

    Args:
        embeddings: (n_samples, embedding_dim) array or pre-reduced to 3D
        labels: (n_samples,) labels for coloring
        title: Plot title
        method: Dimensionality reduction to use if embeddings > 3 dims: 'pca' or 'tsne'
        sample_names: Optional list of names/ids for hover labels

    Returns:
        HTML fragment (string) containing the Plotly div
    """
    # Reduce to 3D
    if method == "tsne" and embeddings.shape[1] > 3:
        pca = PCA(n_components=3, random_state=42)
        coords = pca.fit_transform(embeddings)
    elif method == "t-sne" and embeddings.shape[1] > 3:
        tsne = TSNE(n_components=3, random_state=42, perplexity=30)
        coords = tsne.fit_transform(embeddings)
    else:
        coords = embeddings

    data = {"x": coords[:, 0], "y": coords[:, 1], "z": coords[:, 2], "label": labels}
    if sample_names is not None:
        data["name"] = sample_names

    df = pd.DataFrame(data)

    fig = px.scatter_3d(
        df,
        x="x",
        y="y",
        z="z",
        color="label",
        hover_name=("name" if sample_names is not None else None),
        title=title,
        width=900,
        height=700,
        opacity=0.7,
    )

    # Return HTML string
    return pio.to_html(fig, full_html=False, include_plotlyjs="cdn")


def generate_static_3d_plot(
    embeddings: np.ndarray,
    labels: np.ndarray,
    title: str = "3D Embedding Visualization",
    method: str = "pca",
) -> str:
    """
    Generate a static 3D matplotlib plot and return base64-encoded PNG.

    Args:
        embeddings: (n_samples, embedding_dim) array or pre-reduced to 3D
        labels: (n_samples,) labels for coloring
        title: Plot title
        method: Dimensionality reduction to use if embeddings > 3 dims: 'pca' or 'tsne'
    """
    # Reduce to 3D if needed
    if method == "tsne" and embeddings.shape[1] > 3:
        tsne = TSNE(n_components=3, random_state=42, perplexity=30)
        coords = tsne.fit_transform(embeddings)
    elif embeddings.shape[1] > 3:
        pca = PCA(n_components=3, random_state=42)
        coords = pca.fit_transform(embeddings)
    else:
        coords = embeddings

    fig = plt.figure(figsize=(9, 6))
    ax = fig.add_subplot(111, projection="3d")
    labels_arr = np.asarray(labels)
    unique_labels = np.unique(labels_arr)
    cmap = plt.get_cmap("tab10")

    for i, label in enumerate(unique_labels):
        mask = labels_arr == label
        ax.scatter(
            coords[mask, 0],
            coords[mask, 1],
            coords[mask, 2],
            label=str(label),
            s=10,
            alpha=0.7,
            color=cmap(i % 10),
        )

    ax.set_title(title)
    ax.set_xlabel("Component 1")
    ax.set_ylabel("Component 2")
    ax.set_zlabel("Component 3")
    ax.legend(loc="best", fontsize=8)

    buf = BytesIO()
    plt.tight_layout()
    plt.savefig(buf, format="png", dpi=150, bbox_inches="tight")
    buf.seek(0)
    img_base64 = base64.b64encode(buf.read()).decode("utf-8")
    plt.close(fig)

    return img_base64


def generate_all_plots(detector: "ShortcutDetector") -> dict:
    """
    Generate all available plots for a fitted detector.

    Args:
        detector: Fitted ShortcutDetector instance

    Returns:
        Dictionary mapping plot names to base64 encoded PNG strings
    """
    plots = {}

    # Always generate PCA and t-SNE
    print("  Generating PCA plot...")
    plots["pca"] = generate_pca_plot(detector.embeddings_, detector.labels_)

    print("  Generating t-SNE plot...")
    plots["tsne"] = generate_tsne_plot(detector.embeddings_, detector.labels_)

    # 3D plots (static for app; interactive for HTML report)
    print("  Generating 3D plot...")
    plots["static_3d"] = generate_static_3d_plot(detector.embeddings_, detector.labels_)
    plots["html_3d"] = generate_interactive_3d_plot(detector.embeddings_, detector.labels_)

    # HBAC-specific plots
    if "hbac" in detector.results_ and detector.results_["hbac"]["success"]:
        report = detector.results_["hbac"]["report"]

        print("  Generating dimension importance plot...")
        plots["dimension_importance"] = generate_dimension_importance_plot(
            report["dimension_importance"]
        )

        print("  Generating cluster purity plot...")
        plots["cluster_purity"] = generate_cluster_purity_plot(report["cluster_purities"])

    # Statistical test plots
    if "statistical" in detector.results_ and detector.results_["statistical"]["success"]:
        stat_res = detector.results_["statistical"]
        # Multi-attribute results nest data under by_attribute; skip plots in that case
        p_values = stat_res.get("p_values")
        if p_values is None:
            print("  Skipping statistical plots (multi-attribute mode)")
        corrected_pvalues = stat_res.get("corrected_pvalues") if p_values else None

        if p_values is not None:
            print("  Generating p-value heatmap (top 20 dimensions)...")
            heatmap = generate_pvalue_heatmap(
                p_values,
                corrected_pvalues_dict=corrected_pvalues,
                top_k=20,
                show_all=False,
            )
            if heatmap:
                plots["pvalue_heatmap"] = heatmap

            print("  Generating full p-value heatmap...")
            heatmap_full = generate_pvalue_heatmap(
                p_values,
                corrected_pvalues_dict=corrected_pvalues,
                max_features_full=100,
                show_all=True,
            )
            if heatmap_full:
                plots["pvalue_heatmap_full"] = heatmap_full

    return plots
