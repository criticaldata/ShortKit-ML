#!/usr/bin/env python
"""Standalone script to generate the LDA sex-projection figure for the paper.

Loads MIMIC-CXR RAD-DINO embeddings (768-dim, ~1491 samples) and sex labels,
projects onto the maximally sex-discriminative LDA axis, and produces a
density histogram saved as L03_mimic_lda_sex.pdf.

Usage:
    python scripts/generate_lda_figure.py

Requires:
    - data/mimic_cxr/rad_dino_embeddings.npy   (768-dim float32 array)
    - data/mimic_cxr/rad_dino_metadata.csv      (must contain 'sex' column)
"""

from __future__ import annotations

import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis

PROJECT_ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = PROJECT_ROOT / "data" / "mimic_cxr"
OUTPUT_DIR = PROJECT_ROOT / "output" / "paper_figures"

# High-contrast palette suitable for IEEE print
COLOR_FEMALE = "#D6264D"
COLOR_MALE = "#0855A5"


def main() -> None:
    emb_path = DATA_DIR / "rad_dino_embeddings.npy"
    meta_path = DATA_DIR / "rad_dino_metadata.csv"

    for p in (emb_path, meta_path):
        if not p.exists():
            print(f"ERROR: Required file not found: {p}")
            print("Download MIMIC-CXR RAD-DINO data first (see data/README.md).")
            sys.exit(1)

    print("Loading MIMIC-CXR RAD-DINO embeddings...")
    embeddings = np.load(emb_path)
    metadata = pd.read_csv(meta_path)
    print(f"  Embeddings shape : {embeddings.shape}")
    print(f"  Metadata rows    : {len(metadata)}")
    assert len(embeddings) == len(metadata), "Embedding / metadata length mismatch"

    # Binary sex labels
    sex_raw = metadata["sex"].values
    sex_binary = (sex_raw == "Male").astype(int)

    # LDA projection to 1D
    lda = LinearDiscriminantAnalysis(n_components=1)
    proj = lda.fit_transform(embeddings, sex_binary).ravel()
    print(f"  LDA explained variance ratio: {lda.explained_variance_ratio_}")

    # -- Plot --
    fig, ax = plt.subplots(figsize=(5.5, 4))

    bins = 40
    for label, name, color in [(0, "Female", COLOR_FEMALE), (1, "Male", COLOR_MALE)]:
        mask = sex_binary == label
        ax.hist(
            proj[mask],
            bins=bins,
            color=color,
            label=name,
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

    fig.tight_layout()

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    save_path = OUTPUT_DIR / "L03_mimic_lda_sex.pdf"
    fig.savefig(save_path, dpi=300, bbox_inches="tight")
    print(f"  Saved: {save_path}")

    plt.close(fig)
    print("Done.")


if __name__ == "__main__":
    main()
