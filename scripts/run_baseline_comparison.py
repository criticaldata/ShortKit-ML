#!/usr/bin/env python
"""CLI script for running baseline comparison across toolkits.

Loads data (synthetic or from .npy files), runs the comparison, and saves
results to the specified output directory.

Examples:
    # Use synthetic data (default)
    python scripts/run_baseline_comparison.py --output-dir output/comparison

    # Use pre-saved numpy arrays
    python scripts/run_baseline_comparison.py \
        --embeddings data/embeddings.npy \
        --labels data/labels.npy \
        --group-labels data/groups.npy \
        --output-dir output/comparison
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np


def _parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run ShortKit-ML / Fairlearn / AIF360 comparison.",
    )
    parser.add_argument(
        "--embeddings",
        type=str,
        default=None,
        help="Path to embeddings .npy file. If omitted, synthetic data is generated.",
    )
    parser.add_argument(
        "--labels",
        type=str,
        default=None,
        help="Path to labels .npy file.",
    )
    parser.add_argument(
        "--group-labels",
        type=str,
        default=None,
        help="Path to group labels .npy file.",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default="output/baseline_comparison",
        help="Directory to write results to.",
    )
    parser.add_argument(
        "--no-fairlearn",
        action="store_true",
        help="Disable Fairlearn comparison.",
    )
    parser.add_argument(
        "--no-aif360",
        action="store_true",
        help="Disable AIF360 comparison.",
    )
    return parser.parse_args(argv)


def _generate_synthetic(
    n: int = 400,
    d: int = 64,
    seed: int = 42,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Generate simple synthetic data with an embedded shortcut."""
    rng = np.random.RandomState(seed)
    group_labels = rng.randint(0, 2, size=n)
    labels = rng.randint(0, 2, size=n)

    embeddings = rng.randn(n, d).astype(np.float32)
    # Inject a shortcut: first 5 dims shift by group
    embeddings[:, :5] += (group_labels * 1.5)[:, None]
    return embeddings, labels, group_labels


def main(argv: list[str] | None = None) -> None:
    args = _parse_args(argv)

    # Load or generate data
    if args.embeddings is not None:
        if args.labels is None or args.group_labels is None:
            print("ERROR: --labels and --group-labels are required when --embeddings is provided.")
            sys.exit(1)
        embeddings = np.load(args.embeddings)
        labels = np.load(args.labels)
        group_labels = np.load(args.group_labels)
        print(f"Loaded data: {embeddings.shape[0]} samples, {embeddings.shape[1]} dims")
    else:
        print("No data files specified; generating synthetic data.")
        embeddings, labels, group_labels = _generate_synthetic()
        print(f"Synthetic data: {embeddings.shape[0]} samples, {embeddings.shape[1]} dims")

    # Run comparison
    from shortcut_detect.benchmark.baseline_comparison import (
        BaselineComparison,
        generate_feature_comparison_table,
    )

    comp = BaselineComparison(
        include_fairlearn=not args.no_fairlearn,
        include_aif360=not args.no_aif360,
    )
    result = comp.run(embeddings, labels, group_labels)

    # Print summary
    print()
    print(result.summary())

    # Save outputs
    out = Path(args.output_dir)
    out.mkdir(parents=True, exist_ok=True)

    # Metrics table
    metrics_df = result.metrics_table()
    metrics_df.to_csv(out / "metrics_table.csv")
    print(f"\nMetrics table saved to {out / 'metrics_table.csv'}")

    # Feature comparison (dynamic, based on run)
    comp_df = result.comparison_table()
    comp_df.to_csv(out / "comparison_table.csv")
    print(f"Comparison table saved to {out / 'comparison_table.csv'}")

    # Static feature comparison
    static_df = generate_feature_comparison_table()
    static_df.to_csv(out / "feature_comparison.csv")
    print(f"Static feature comparison saved to {out / 'feature_comparison.csv'}")

    # LaTeX output
    latex = result.to_latex()
    (out / "comparison_table.tex").write_text(latex)
    print(f"LaTeX table saved to {out / 'comparison_table.tex'}")

    # Markdown output
    md = result.to_markdown()
    (out / "comparison_table.md").write_text(md)
    print(f"Markdown table saved to {out / 'comparison_table.md'}")

    print("\nDone.")


if __name__ == "__main__":
    main()
