#!/usr/bin/env python3
"""Run B09 (per-diagnosis) and B10 (per-attribute) CheXpert benchmarks.

This script uses the existing 512-dim embeddings (set up as 'medclip' backbone)
to run ShortcutDetector for each sensitive attribute (race, sex, age_bin).

Results are saved to output/paper_benchmark/chexpert_results/ as CSV and JSON.

B09: Per-diagnosis analysis (currently we have one binary task; extensible)
B10: Per-attribute analysis across race, sex, age_bin
"""

from __future__ import annotations

import argparse
import json
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from shortcut_detect.benchmark.method_utils import ALL_METHODS, convergence_bucket, method_flag
from shortcut_detect.unified import ShortcutDetector

METHODS = ALL_METHODS
ATTRIBUTES = ("race", "sex", "age_bin")
RACE_FOCUS = ("ASIAN", "BLACK", "WHITE")
AGE_BINS = [-np.inf, 40, 60, 80, np.inf]
AGE_BIN_LABELS = ["<40", "40-59", "60-79", ">=80"]


def load_backbone(embeddings_dir: Path, backbone: str) -> tuple[np.ndarray, pd.DataFrame]:
    """Load embeddings and metadata for a backbone."""
    emb_path = embeddings_dir / f"{backbone}_embeddings.npy"
    meta_path = embeddings_dir / f"{backbone}_metadata.csv"
    if not emb_path.exists() or not meta_path.exists():
        raise FileNotFoundError(f"Missing artifacts for {backbone}: {emb_path}, {meta_path}")
    X = np.load(str(emb_path))
    meta = pd.read_csv(meta_path)
    assert X.shape[0] == len(meta), f"Row mismatch: {X.shape[0]} vs {len(meta)}"
    return X, meta


def run_attribute_benchmark(
    X: np.ndarray,
    y: np.ndarray,
    meta: pd.DataFrame,
    backbone: str,
    seed: int = 42,
    alpha: float = 0.05,
) -> tuple[list[dict], list[dict]]:
    """Run ShortcutDetector for each attribute. Returns (method_rows, convergence_rows)."""
    method_rows: list[dict[str, Any]] = []
    conv_rows: list[dict[str, Any]] = []

    for attr in ATTRIBUTES:
        if attr not in meta.columns:
            print(f"  [SKIP] Attribute '{attr}' not in metadata")
            continue

        frame = meta.copy()
        if attr == "race":
            frame = frame[frame["race"].isin(RACE_FOCUS)].copy()
            if frame.empty:
                continue

        idx = frame.index.to_numpy()
        X_attr = X[idx]
        y_attr = y[idx]
        g_attr = frame[attr].to_numpy()
        n_groups = len(np.unique(g_attr))

        if n_groups < 2:
            print(f"  [SKIP] Attribute '{attr}': only {n_groups} group(s)")
            continue

        print(
            f"\n  Attribute: {attr} | n={len(X_attr)} | groups={n_groups} | unique={np.unique(g_attr)}"
        )

        flagged_methods = 0
        for method in METHODS:
            t0 = time.time()
            detector = ShortcutDetector(
                methods=[method],
                seed=seed,
                statistical_correction="fdr_bh",
                statistical_alpha=alpha,
            )
            try:
                detector.fit(X_attr, y_attr, group_labels=g_attr)
                result = detector.results_.get(method, {"success": False, "error": "missing"})
            except Exception as exc:
                result = {"success": False, "error": str(exc)}

            elapsed = time.time() - t0
            flag = int(method_flag(method, result))
            flagged_methods += flag
            risk = str(result.get("risk_value") or result.get("risk_level") or "unknown")

            row = {
                "backbone": backbone,
                "attribute": attr,
                "method": method,
                "n_samples": int(X_attr.shape[0]),
                "n_groups": int(n_groups),
                "flagged": flag,
                "risk_level": risk,
                "success": bool(result.get("success", False)),
                "error": result.get("error"),
                "elapsed_s": round(elapsed, 3),
            }
            method_rows.append(row)
            status = "FLAGGED" if flag else "ok"
            print(f"    {method:12s} -> {status:8s} (risk={risk}, {elapsed:.2f}s)")

        # Convergence assessment
        confidence = convergence_bucket(flagged_methods, len(METHODS))

        conv_rows.append(
            {
                "backbone": backbone,
                "attribute": attr,
                "n_samples": int(X_attr.shape[0]),
                "n_groups": int(n_groups),
                "n_flagged_methods": int(flagged_methods),
                "n_total_methods": len(METHODS),
                "confidence_level": confidence,
            }
        )
        print(f"    => Convergence: {flagged_methods}/{len(METHODS)} flagged -> {confidence}")

    return method_rows, conv_rows


def run_benchmark(
    repo_root: Path,
    backbone: str = "medclip",
    output_dir: Path | None = None,
    seed: int = 42,
) -> dict[str, Path]:
    """Run the full CheXpert benchmark and save results."""
    embeddings_dir = repo_root / "output" / "paper_benchmark" / "chexpert_embeddings"
    if output_dir is None:
        output_dir = repo_root / "output" / "paper_benchmark" / "chexpert_results"
    output_dir.mkdir(parents=True, exist_ok=True)

    print(f"{'='*60}")
    print(" CheXpert Shortcut Detection Benchmark")
    print(f"{'='*60}")
    print(f"  Backbone:    {backbone}")
    print(f"  Seed:        {seed}")
    print(f"  Methods:     {', '.join(METHODS)}")
    print(f"  Attributes:  {', '.join(ATTRIBUTES)}")
    print(f"  Output:      {output_dir}")
    print(f"{'='*60}")

    # Load data
    X, meta = load_backbone(embeddings_dir, backbone)
    y = (meta["task_label"].astype(float) > 0).astype(int).to_numpy()
    meta = meta.copy()
    meta["race"] = meta["race"].astype(str).str.upper()
    meta["sex"] = meta["sex"].astype(str)
    meta["age_bin"] = pd.cut(
        meta["age"].astype(float), bins=AGE_BINS, labels=AGE_BIN_LABELS, right=False
    ).astype(str)

    print(f"\nLoaded: {X.shape[0]} samples, {X.shape[1]}-dim embeddings")
    print(f"Label balance: {dict(pd.Series(y).value_counts().sort_index())}")

    t_start = time.time()

    # B10: Per-attribute analysis (all data, each sensitive attribute)
    print(f"\n{'='*60}")
    print(" B10: Per-Attribute Analysis")
    print(f"{'='*60}")
    method_rows, conv_rows = run_attribute_benchmark(X, y, meta, backbone, seed=seed)

    elapsed_total = time.time() - t_start

    # Save results
    methods_df = pd.DataFrame(method_rows)
    conv_df = pd.DataFrame(conv_rows)

    methods_path = output_dir / "chexpert_methods.csv"
    conv_path = output_dir / "chexpert_convergence.csv"
    summary_path = output_dir / "chexpert_summary.json"

    methods_df.to_csv(methods_path, index=False)
    conv_df.to_csv(conv_path, index=False)

    # Build summary JSON
    summary = {
        "benchmark": "chexpert_shortcut_detection",
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "backbone": backbone,
        "n_samples": int(X.shape[0]),
        "embedding_dim": int(X.shape[1]),
        "methods": list(METHODS),
        "attributes": list(ATTRIBUTES),
        "seed": seed,
        "elapsed_seconds": round(elapsed_total, 2),
        "results": {
            "per_attribute": {},
            "convergence_matrix": [],
        },
    }

    # Organize per-attribute results
    for _, row in conv_df.iterrows():
        attr = row["attribute"]
        summary["results"]["per_attribute"][attr] = {
            "n_samples": int(row["n_samples"]),
            "n_groups": int(row["n_groups"]),
            "n_flagged": int(row["n_flagged_methods"]),
            "n_total": int(row["n_total_methods"]),
            "confidence": row["confidence_level"],
        }

    # Convergence matrix (attribute x method)
    if not methods_df.empty:
        pivot = (
            methods_df.pivot_table(
                index="attribute", columns="method", values="flagged", aggfunc="max"
            )
            .fillna(0)
            .astype(int)
        )
        summary["results"]["convergence_matrix"] = pivot.reset_index().to_dict(orient="records")

    with open(summary_path, "w") as f:
        json.dump(summary, f, indent=2)

    print(f"\n{'='*60}")
    print(" Results Summary")
    print(f"{'='*60}")
    print(f"  Total time: {elapsed_total:.1f}s")
    print()

    if not conv_df.empty:
        print("  Convergence Matrix (attribute x flagged/total):")
        for _, row in conv_df.iterrows():
            print(
                f"    {row['attribute']:10s}: {row['n_flagged_methods']}/{row['n_total_methods']} "
                f"-> {row['confidence_level']}"
            )
        print()

    if not methods_df.empty:
        print("  Detailed Method Results:")
        print(methods_df.to_string(index=False))
        print()

    print("  Saved:")
    print(f"    Methods:     {methods_path}")
    print(f"    Convergence: {conv_path}")
    print(f"    Summary:     {summary_path}")
    print(f"{'='*60}")

    return {"methods": methods_path, "convergence": conv_path, "summary": summary_path}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run CheXpert shortcut detection benchmark")
    parser.add_argument(
        "--root",
        type=Path,
        default=Path(__file__).resolve().parent.parent,
        help="Repository root directory",
    )
    parser.add_argument("--backbone", default="medclip", help="Backbone name")
    parser.add_argument("--output-dir", type=Path, default=None, help="Override output directory")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    run_benchmark(
        repo_root=args.root,
        backbone=args.backbone,
        output_dir=args.output_dir,
        seed=args.seed,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
