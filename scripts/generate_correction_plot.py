#!/usr/bin/env python
"""F07: Statistical correction comparison plot.

Generates a plot showing FPR across different correction methods as
embedding dimensionality increases. Uses null data (effect_size=0)
so any significant result is a false positive.

Output: output/paper_figures/F07_correction_comparison.pdf
"""

from __future__ import annotations

import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt  # noqa: E402
import numpy as np
from scipy.stats import mannwhitneyu

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from shortcut_detect.statistical import GroupDiffTest  # noqa: E402

OUTPUT_DIR = PROJECT_ROOT / "output" / "paper_figures"

# Parameters
DIMS = [32, 64, 128, 256, 512]
CORRECTIONS = ["bonferroni", "holm", "fdr_bh", "fdr_by"]
SEEDS = [0, 1, 2, 3, 4]
N_SAMPLES = 500
ALPHA = 0.05


def run_null_experiment(dim: int, seed: int, correction: str) -> bool:
    """Run a null experiment and return True if any feature is falsely significant."""
    rng = np.random.default_rng(seed)
    X = rng.standard_normal((N_SAMPLES, dim))
    Y = rng.choice([0, 1], size=N_SAMPLES)

    test = GroupDiffTest(test=mannwhitneyu)
    test.fit(X, Y)
    result = test.apply_correction(alpha=ALPHA, method=correction, verbose=False)

    sig = result["significant_features"]
    return any(v is not None and len(v) > 0 for v in sig.values())


def main():
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    # Collect FPR for each (correction, dim)
    fpr_data = {c: [] for c in CORRECTIONS}

    for dim in DIMS:
        print(f"dim={dim}")
        for correction in CORRECTIONS:
            false_positives = 0
            for seed in SEEDS:
                if run_null_experiment(dim, seed, correction):
                    false_positives += 1
            fpr = false_positives / len(SEEDS)
            fpr_data[correction].append(fpr)
            print(f"  {correction:15s}: FPR = {fpr:.2f}")

    # Also compute uncorrected FPR for reference
    fpr_uncorrected = []
    for dim in DIMS:
        false_positives = 0
        for seed in SEEDS:
            rng = np.random.default_rng(seed)
            X = rng.standard_normal((N_SAMPLES, dim))
            Y = rng.choice([0, 1], size=N_SAMPLES)
            test = GroupDiffTest(test=mannwhitneyu)
            test.fit(X, Y)
            sig = test.apply_threshold(alpha=ALPHA, verbose=False)
            if any(v is not None and len(v) > 0 for v in sig.values()):
                false_positives += 1
        fpr_uncorrected.append(false_positives / len(SEEDS))

    # --- Plot ---
    # Darker, high-contrast palette
    colors = {
        "uncorrected": "#555555",
        "bonferroni": "#D4760A",
        "holm": "#1F77B4",
        "fdr_bh": "#2CA02C",
        "fdr_by": "#B5338A",
    }
    markers = {
        "uncorrected": "s",
        "bonferroni": "o",
        "holm": "^",
        "fdr_bh": "D",
        "fdr_by": "v",
    }

    fig, (ax_top, ax_bot) = plt.subplots(
        2,
        1,
        figsize=(6, 5),
        height_ratios=[1, 2],
        sharex=True,
        gridspec_kw={"hspace": 0.12},
    )

    # Offset lines slightly so they don't perfectly overlap at 0
    offsets = {"bonferroni": -0.008, "holm": -0.003, "fdr_bh": 0.003, "fdr_by": 0.008}

    for ax in (ax_top, ax_bot):
        # Uncorrected
        ax.plot(
            DIMS,
            fpr_uncorrected,
            marker=markers["uncorrected"],
            color=colors["uncorrected"],
            linewidth=2.5,
            markersize=9,
            linestyle="--",
            label="Uncorrected",
            zorder=3,
        )
        # Corrected methods
        for correction in CORRECTIONS:
            label_map = {
                "bonferroni": "Bonferroni",
                "holm": "Holm",
                "fdr_bh": "FDR (BH)",
                "fdr_by": "FDR (BY)",
            }
            shifted = [v + offsets[correction] for v in fpr_data[correction]]
            ax.plot(
                DIMS,
                shifted,
                marker=markers[correction],
                color=colors[correction],
                linewidth=2.2,
                markersize=8,
                label=label_map[correction],
                zorder=3,
            )
        ax.axhline(y=ALPHA, color="black", linestyle=":", linewidth=1, alpha=0.5)
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)
        ax.tick_params(labelsize=10)

    # Top panel: full range for uncorrected
    ax_top.set_ylim(0.5, 1.08)
    ax_top.set_yticks([0.6, 0.8, 1.0])

    # Bottom panel: zoom into [0, 0.3] for corrected methods
    ax_bot.set_ylim(-0.02, 0.30)
    ax_bot.set_yticks([0.0, 0.05, 0.10, 0.15, 0.20, 0.25])
    ax_bot.text(DIMS[-1] + 15, ALPHA, r"$\alpha$=0.05", va="center", fontsize=9, alpha=0.7)

    # Broken axis markers
    d = 0.015
    kwargs = {"transform": ax_top.transAxes, "color": "k", "clip_on": False, "linewidth": 1}
    ax_top.plot((-d, +d), (-d, +d), **kwargs)
    ax_top.plot((1 - d, 1 + d), (-d, +d), **kwargs)
    kwargs.update(transform=ax_bot.transAxes)
    ax_bot.plot((-d, +d), (1 - d, 1 + d), **kwargs)
    ax_bot.plot((1 - d, 1 + d), (1 - d, 1 + d), **kwargs)

    ax_bot.set_xlabel("Embedding dimensionality", fontsize=12)
    fig.supylabel("False positive rate", fontsize=12, x=0.02)
    ax_bot.set_xscale("log", base=2)
    ax_bot.set_xticks(DIMS)
    ax_bot.set_xticklabels([str(d) for d in DIMS])

    # Legend outside
    handles, labels = ax_top.get_legend_handles_labels()
    ax_top.legend(
        handles,
        labels,
        fontsize=9,
        framealpha=0.95,
        loc="center left",
        bbox_to_anchor=(1.02, 0.0),
        edgecolor="0.8",
    )

    fig.tight_layout()
    fig.subplots_adjust(right=0.78)

    fig.tight_layout()

    out_path = OUTPUT_DIR / "F07_correction_comparison.pdf"
    fig.savefig(out_path, dpi=300, bbox_inches="tight")
    print(f"\nSaved: {out_path}")

    # Also save PNG for quick preview
    out_png = OUTPUT_DIR / "F07_correction_comparison.png"
    fig.savefig(out_png, dpi=300, bbox_inches="tight")
    print(f"Saved: {out_png}")

    plt.close(fig)


if __name__ == "__main__":
    main()
