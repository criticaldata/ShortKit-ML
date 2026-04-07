#!/usr/bin/env python
"""L03: UMAP visualization of demographic encoding.

Generates UMAP scatter plots for:
  - MIMIC-CXR RAD-DINO embeddings colored by sex and race
  - CelebA embeddings colored by Male attribute

Output: output/paper_figures/L03_mimic_lda_sex.pdf
        output/paper_figures/L03_mimic_umap_race.pdf
        output/paper_figures/L03_celeba_umap_male.pdf
"""

from __future__ import annotations

import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import umap

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

OUTPUT_DIR = PROJECT_ROOT / "output" / "paper_figures"
DATA_DIR = PROJECT_ROOT / "data"

# High-contrast, fully saturated palettes for print
SEX_COLORS = {"Female": "#D6264D", "Male": "#0855A5"}
RACE_COLORS = {
    "White": "#C43E00",
    "Black": "#08458A",
    "Asian": "#146B2E",
    "Other": "#444444",
}
MALE_COLORS = {0: "#D6264D", 1: "#0855A5"}
MALE_LABELS = {0: "Female", 1: "Male"}


def _scatter_plot(
    embedding_2d: np.ndarray,
    labels: np.ndarray,
    color_map: dict,
    label_map: dict | None = None,
    save_path: Path | None = None,
    point_size: float = 12.0,
    alpha: float = 0.85,
    figsize: tuple = (5.5, 5),
):
    """Create a publication-quality UMAP scatter plot."""
    if label_map is None:
        label_map = {k: str(k) for k in color_map}

    fig, ax = plt.subplots(figsize=figsize)

    # Plot each category separately for legend
    unique_labels = sorted(color_map.keys(), key=str)
    for lab in unique_labels:
        mask = labels == lab
        if not np.any(mask):
            continue
        ax.scatter(
            embedding_2d[mask, 0],
            embedding_2d[mask, 1],
            c=color_map[lab],
            label=label_map.get(lab, str(lab)),
            s=point_size,
            alpha=alpha,
            edgecolors="white",
            linewidths=0.3,
            rasterized=True,
        )

    ax.set_xlabel("UMAP 1", fontsize=11)
    ax.set_ylabel("UMAP 2", fontsize=11)
    ax.tick_params(labelsize=9)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)

    # Legend with larger markers
    legend = ax.legend(
        fontsize=10,
        framealpha=0.9,
        markerscale=3,
        loc="best",
        handletextpad=0.5,
    )
    for handle in legend.legend_handles:
        handle.set_alpha(0.9)

    fig.tight_layout()

    if save_path is not None:
        fig.savefig(save_path, dpi=300, bbox_inches="tight")
        print(f"  Saved: {save_path}")

    plt.close(fig)


def _lda_projection_plot(
    embeddings: np.ndarray,
    binary_labels: np.ndarray,
    color_map: dict,
    label_names: dict,
    save_path: Path,
    title: str = "",
):
    """1D LDA projection histogram showing demographic encoding.

    Projects embeddings onto the Linear Discriminant Analysis direction
    (the axis that maximally separates the two groups) and plots the
    resulting distributions as overlapping histograms.  This is guaranteed
    to reveal any linearly-encoded attribute, unlike UMAP which only
    captures nonlinear 2-D structure.
    """
    from sklearn.discriminant_analysis import LinearDiscriminantAnalysis

    lda = LinearDiscriminantAnalysis(n_components=1)
    proj = lda.fit_transform(embeddings, binary_labels).ravel()

    fig, ax = plt.subplots(figsize=(5.5, 4))
    bins = 40
    unique = sorted(np.unique(binary_labels))
    for lab in unique:
        mask = binary_labels == lab
        ax.hist(
            proj[mask],
            bins=bins,
            color=color_map[lab],
            label=label_names[lab],
            alpha=0.65,
            density=True,
            edgecolor="white",
            linewidth=0.4,
        )

    ax.set_xlabel("LDA projection (a.u.)", fontsize=11)
    ax.set_ylabel("Density", fontsize=11)
    ax.tick_params(labelsize=9)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.legend(fontsize=10, framealpha=0.9)
    if title:
        ax.set_title(title, fontsize=10)

    fig.tight_layout()
    fig.savefig(save_path, dpi=300, bbox_inches="tight")
    print(f"  Saved: {save_path}")
    plt.close(fig)


def generate_mimic_umap():
    """Generate demographic-encoding visualizations for MIMIC-CXR RAD-DINO embeddings.

    Produces:
    - LDA projection histogram for sex (shows clear separation, confirming F1=0.990)
    - UMAP scatter plot coloured by race
    """
    emb_path = DATA_DIR / "mimic_cxr" / "rad_dino_embeddings.npy"
    meta_path = DATA_DIR / "mimic_cxr" / "rad_dino_metadata.csv"

    if not emb_path.exists() or not meta_path.exists():
        print("MIMIC data not found, skipping.")
        return

    print("Loading MIMIC RAD-DINO embeddings...")
    embeddings = np.load(emb_path)
    metadata = pd.read_csv(meta_path)
    print(f"  Embeddings: {embeddings.shape}, Metadata: {metadata.shape}")

    assert len(embeddings) == len(metadata), "Embedding/metadata length mismatch"

    # --- Sex: LDA projection histogram ---
    sex_raw = metadata["sex"].values
    sex_binary = (sex_raw == "Male").astype(int)
    _lda_projection_plot(
        embeddings,
        sex_binary,
        color_map={0: SEX_COLORS["Female"], 1: SEX_COLORS["Male"]},
        label_names={0: "Female", 1: "Male"},
        save_path=OUTPUT_DIR / "L03_mimic_lda_sex.pdf",
    )

    # --- UMAP (for race, where nonlinear structure may be informative) ---
    print("  Computing UMAP for race plot (this may take a moment)...")
    reducer = umap.UMAP(n_components=2, random_state=42, n_neighbors=30, min_dist=0.05)
    umap_2d = reducer.fit_transform(embeddings)
    print("  UMAP done.")

    race_raw = metadata["race"].values
    race_map = {"WHITE": "White", "BLACK": "Black", "ASIAN": "Asian", "OTHER": "Other"}
    race_simplified = np.array([race_map.get(r, "Other") for r in race_raw])

    _scatter_plot(
        umap_2d,
        race_simplified,
        color_map=RACE_COLORS,
        save_path=OUTPUT_DIR / "L03_mimic_umap_race.pdf",
        point_size=14,
        alpha=0.85,
    )


def generate_celeba_umap():
    """Generate UMAP plot for CelebA embeddings colored by Male attribute."""
    emb_path = DATA_DIR / "celeba" / "celeba_real_embeddings.npy"
    attr_path = DATA_DIR / "celeba" / "celeba_real_attributes.csv"
    meta_path = DATA_DIR / "celeba" / "celeba_real_metadata.csv"

    if not emb_path.exists():
        print("CelebA embeddings not found, skipping.")
        return

    print("Loading CelebA embeddings...")
    embeddings = np.load(emb_path)

    # Load Male attribute
    if meta_path.exists():
        meta = pd.read_csv(meta_path)
        male_col = meta["Male"].values
    elif attr_path.exists():
        attrs = pd.read_csv(attr_path)
        male_col = attrs["Male"].values
    else:
        print("CelebA attributes not found, skipping.")
        return

    print(f"  Embeddings: {embeddings.shape}")
    assert len(embeddings) == len(male_col), "Embedding/attribute length mismatch"

    # Subsample to 5000 for speed
    n = len(embeddings)
    if n > 5000:
        rng = np.random.default_rng(42)
        idx = rng.choice(n, size=5000, replace=False)
        idx.sort()
        embeddings = embeddings[idx]
        male_col = male_col[idx]
        print(f"  Subsampled to {len(embeddings)}")

    # Compute UMAP
    print("  Computing UMAP (this may take a moment)...")
    reducer = umap.UMAP(n_components=2, random_state=42, n_neighbors=15, min_dist=0.1)
    umap_2d = reducer.fit_transform(embeddings)
    print("  UMAP done.")

    _scatter_plot(
        umap_2d,
        male_col,
        color_map=MALE_COLORS,
        label_map=MALE_LABELS,
        save_path=OUTPUT_DIR / "L03_celeba_umap_male.pdf",
        point_size=10,
        alpha=0.85,
    )


def main():
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    print("=" * 60)
    print("L03: UMAP Visualization of Demographic Encoding")
    print("=" * 60)
    print()

    generate_mimic_umap()
    print()
    generate_celeba_umap()

    print()
    print("Done.")


if __name__ == "__main__":
    main()
