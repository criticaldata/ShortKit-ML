#!/usr/bin/env python
"""Generate paper figures F02 (convergence matrix) and F03 (comparison table).

Outputs are saved to output/paper_figures/ as PDF, PNG, LaTeX, and Markdown.
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

# Ensure the project root is on sys.path
PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from shortcut_detect.benchmark.method_utils import ALL_METHODS  # noqa: E402
from shortcut_detect.benchmark.method_utils import method_flag as _method_flag  # noqa: E402

OUTPUT_DIR = PROJECT_ROOT / "output" / "paper_figures"
DATA_DIR = PROJECT_ROOT / "data"


def generate_f02_synthetic():
    """F02a: Convergence matrix from synthetic benchmark data."""
    from shortcut_detect.benchmark.convergence_viz import (
        ConvergenceMatrix,
        plot_agreement_summary,
        plot_convergence_matrix,
    )
    from shortcut_detect.benchmark.synthetic_generator import SyntheticGenerator
    from shortcut_detect.unified import ShortcutDetector

    effect_sizes = [0.0, 0.2, 0.5, 0.8, 1.2]
    methods = list(ALL_METHODS)
    n_samples = 1000
    embedding_dim = 128

    print("=" * 60)
    print("F02a: Synthetic Convergence Matrix")
    print("=" * 60)
    print(f"  Effect sizes: {effect_sizes}")
    print(f"  Methods: {methods}")
    print(f"  n_samples={n_samples}, embedding_dim={embedding_dim}")
    print()

    matrix = ConvergenceMatrix(methods=methods)

    for es in effect_sizes:
        print(f"  Running effect_size={es} ...", end=" ", flush=True)
        gen = SyntheticGenerator(
            n_samples=n_samples,
            embedding_dim=embedding_dim,
            shortcut_dims=5,
            seed=42,
        )
        data = gen.generate(effect_size=es)

        detector = ShortcutDetector(methods=methods, seed=42)
        detector.fit(data.embeddings, data.labels, group_labels=data.group_labels)
        results = detector.get_results()

        flags = {}
        for m in methods:
            r = results.get(m, {})
            flags[m] = _method_flag(m, r) if isinstance(r, dict) else False

        label = f"d={es}"
        matrix.add_experiment(label, flags)
        flagged = [m for m, v in flags.items() if v]
        print(f"flagged: {flagged if flagged else 'none'}")

    # Save convergence matrix heatmap
    for ext in ["pdf", "png"]:
        path = OUTPUT_DIR / f"F02_convergence_matrix_synthetic.{ext}"
        plot_convergence_matrix(
            matrix,
            title="Figure 2: Method Convergence Matrix (Synthetic)",
            save_path=path,
            dpi=300,
        )
        print(f"  Saved: {path}")

    # Save agreement summary bar chart
    for ext in ["pdf", "png"]:
        path = OUTPUT_DIR / f"F02_agreement_summary_synthetic.{ext}"
        plot_agreement_summary(
            matrix,
            title="Agreement Level Distribution (Synthetic)",
            save_path=path,
            dpi=300,
        )
        print(f"  Saved: {path}")

    # Save matrix as CSV
    df = matrix.to_dataframe()
    csv_path = OUTPUT_DIR / "F02_convergence_matrix_synthetic.csv"
    df.to_csv(csv_path)
    print(f"  Saved: {csv_path}")
    print()
    print("  Matrix:")
    print(df.to_string())
    print()
    print("  Agreement levels:", matrix.agreement_levels())
    print()

    return matrix


def generate_f02_chexpert():
    """F02b: Convergence matrix from CheXpert demo data."""
    from shortcut_detect.benchmark.convergence_viz import (
        ConvergenceMatrix,
        plot_agreement_summary,
        plot_convergence_matrix,
    )
    from shortcut_detect.unified import ShortcutDetector

    emb_path = DATA_DIR / "chest_embeddings.npy"
    labels_path = DATA_DIR / "chest_labels.npy"
    groups_path = DATA_DIR / "chest_group_labels.npy"

    if not all(p.exists() for p in [emb_path, labels_path, groups_path]):
        print("F02b: CheXpert data not found, skipping.")
        return None

    embeddings = np.load(emb_path)
    labels = np.load(labels_path)
    group_labels = np.load(groups_path)

    print("=" * 60)
    print("F02b: CheXpert Convergence Matrix")
    print("=" * 60)
    print(f"  Data shape: {embeddings.shape}")
    print(f"  Labels: {np.unique(labels, return_counts=True)}")
    print(f"  Groups: {np.unique(group_labels, return_counts=True)}")
    print()

    methods = list(ALL_METHODS)
    matrix = ConvergenceMatrix(methods=methods)

    print("  Running ShortcutDetector on CheXpert data ...", flush=True)
    detector = ShortcutDetector(methods=methods, seed=42)
    detector.fit(embeddings, labels, group_labels=group_labels)
    results = detector.get_results()

    flags = {}
    for m in methods:
        r = results.get(m, {})
        if isinstance(r, dict):
            flags[m] = bool(r.get("shortcut_detected", False))
        else:
            flags[m] = False

    matrix.add_experiment("CheXpert", flags)
    flagged = [m for m, v in flags.items() if v]
    print(f"  Flagged: {flagged if flagged else 'none'}")

    # If we have group sub-analyses, add per-group experiments
    unique_groups = np.unique(group_labels)
    if len(unique_groups) > 2:
        # Run pairwise for interesting subgroups
        for g in unique_groups[:3]:  # cap at 3 to keep figure readable
            if g == unique_groups[0]:
                continue  # skip self-vs-self
            mask = (group_labels == g) | (group_labels == unique_groups[0])
            if mask.sum() < 50:
                continue
            sub_emb = embeddings[mask]
            sub_labels = labels[mask]
            sub_groups_bin = (group_labels[mask] == g).astype(int)

            # Need at least 2 classes in both labels and groups
            if len(np.unique(sub_labels)) < 2 or len(np.unique(sub_groups_bin)) < 2:
                print(f"  Skipping group={g}: insufficient class diversity")
                continue

            try:
                det = ShortcutDetector(methods=methods, seed=42)
                det.fit(sub_emb, sub_labels, group_labels=sub_groups_bin)
                res = det.get_results()
                sub_flags = {}
                for m in methods:
                    r = res.get(m, {})
                    sub_flags[m] = _method_flag(m, r) if isinstance(r, dict) else False
                matrix.add_experiment(f"group={g}", sub_flags)
            except Exception as exc:
                print(f"  Skipping group={g}: {exc}")

    for ext in ["pdf", "png"]:
        path = OUTPUT_DIR / f"F02_convergence_matrix_chexpert.{ext}"
        plot_convergence_matrix(
            matrix,
            title="Figure 2: Method Convergence (CheXpert)",
            save_path=path,
            dpi=300,
        )
        print(f"  Saved: {path}")

    for ext in ["pdf", "png"]:
        path = OUTPUT_DIR / f"F02_agreement_summary_chexpert.{ext}"
        plot_agreement_summary(
            matrix,
            title="Agreement Level Distribution (CheXpert)",
            save_path=path,
            dpi=300,
        )
        print(f"  Saved: {path}")

    csv_path = OUTPUT_DIR / "F02_convergence_matrix_chexpert.csv"
    matrix.to_dataframe().to_csv(csv_path)
    print(f"  Saved: {csv_path}")
    print()
    print("  Matrix:")
    print(matrix.to_dataframe().to_string())
    print("  Agreement levels:", matrix.agreement_levels())
    print()

    return matrix


def generate_f03_feature_table():
    """F03: Static feature comparison table (AIF360 / Fairlearn)."""
    from shortcut_detect.benchmark.baseline_comparison import generate_feature_comparison_table

    print("=" * 60)
    print("F03: Feature Comparison Table")
    print("=" * 60)

    table = generate_feature_comparison_table()
    print()
    print(table.to_string())
    print()

    # LaTeX
    latex_str = table.to_latex()
    tex_path = OUTPUT_DIR / "F03_feature_comparison.tex"
    tex_path.write_text(latex_str)
    print(f"  Saved: {tex_path}")

    # Markdown
    md_str = table.to_markdown()
    md_path = OUTPUT_DIR / "F03_feature_comparison.md"
    md_path.write_text(md_str)
    print(f"  Saved: {md_path}")

    # CSV
    csv_path = OUTPUT_DIR / "F03_feature_comparison.csv"
    table.to_csv(csv_path)
    print(f"  Saved: {csv_path}")
    print()

    return table


def generate_f03_data_driven():
    """F03b: Data-driven metrics comparison on CheXpert demo data."""
    from shortcut_detect.benchmark.baseline_comparison import BaselineComparison

    emb_path = DATA_DIR / "chest_embeddings.npy"
    labels_path = DATA_DIR / "chest_labels.npy"
    groups_path = DATA_DIR / "chest_group_labels.npy"

    if not all(p.exists() for p in [emb_path, labels_path, groups_path]):
        print("F03b: CheXpert data not found, skipping data-driven comparison.")
        return None

    embeddings = np.load(emb_path)
    labels = np.load(labels_path)
    group_labels = np.load(groups_path)

    # Binarize group labels if needed (BaselineComparison expects binary-ish)
    unique_groups = np.unique(group_labels)
    if len(unique_groups) > 2:
        # Use first two groups for the comparison
        mask = np.isin(group_labels, unique_groups[:2])
        embeddings = embeddings[mask]
        labels = labels[mask]
        group_labels = group_labels[mask]
        print(f"  Binarized groups to {unique_groups[:2]}, n={mask.sum()}")

    # Binarize labels if needed
    unique_labels = np.unique(labels)
    if len(unique_labels) > 2:
        # Threshold at median
        median_label = np.median(labels)
        labels = (labels > median_label).astype(int)
        print(f"  Binarized labels at median={median_label}")

    print()
    print("=" * 60)
    print("F03b: Data-driven Metrics Comparison (CheXpert)")
    print("=" * 60)
    print(f"  Data shape: {embeddings.shape}")

    comp = BaselineComparison(include_fairlearn=True, include_aif360=True)
    result = comp.run(embeddings, labels, group_labels)

    print(result.summary())

    # Save metrics table
    metrics_df = result.metrics_table()
    csv_path = OUTPUT_DIR / "F03_metrics_chexpert.csv"
    metrics_df.to_csv(csv_path)
    print(f"  Saved: {csv_path}")

    # Save comparison table
    comp_df = result.comparison_table()
    tex_path = OUTPUT_DIR / "F03_comparison_chexpert.tex"
    tex_path.write_text(comp_df.to_latex())
    print(f"  Saved: {tex_path}")

    md_path = OUTPUT_DIR / "F03_comparison_chexpert.md"
    md_path.write_text(comp_df.to_markdown())
    print(f"  Saved: {md_path}")
    print()

    return result


def main():
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    print(f"Output directory: {OUTPUT_DIR}")
    print()

    # F02: Convergence matrices
    generate_f02_synthetic()
    generate_f02_chexpert()

    # F03: Feature comparison table
    generate_f03_feature_table()
    generate_f03_data_driven()

    # Final summary
    print("=" * 60)
    print("SUMMARY: Generated files")
    print("=" * 60)
    for f in sorted(OUTPUT_DIR.iterdir()):
        size_kb = f.stat().st_size / 1024
        print(f"  {f.name:50s} {size_kb:7.1f} KB")
    print()
    print("Done.")


if __name__ == "__main__":
    main()
