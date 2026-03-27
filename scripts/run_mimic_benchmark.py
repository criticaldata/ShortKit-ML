#!/usr/bin/env python3
"""Run shortcut detection benchmark on MIMIC-CXR embeddings.

For each backbone (rad_dino, vit16_cls, vit32_cls, medsiglip) and each
sensitive attribute (race, sex, age_bin), runs ShortcutDetector with all
10 methods (hbac, probe, statistical, geometric, frequency,
bias_direction_pca, sis, demographic_parity, equalized_odds,
intersectional).

Saves per-method results, convergence matrices, and summary tables to
output/paper_benchmark/mimic_cxr_results/.
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

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------
ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = ROOT / "data" / "mimic_cxr"
OUTPUT_DIR = ROOT / "output" / "paper_benchmark" / "mimic_cxr_results"

BACKBONES = ["rad_dino", "vit16_cls", "vit32_cls", "medsiglip"]
METHODS = list(ALL_METHODS)
ATTRIBUTES = ["race", "sex", "age_bin"]
RACE_FOCUS = ["WHITE", "BLACK", "ASIAN"]


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------
def load_backbone(data_dir: Path, backbone: str) -> tuple[np.ndarray, pd.DataFrame]:
    """Load embeddings and metadata for a backbone."""
    emb_path = data_dir / f"{backbone}_embeddings.npy"
    meta_path = data_dir / f"{backbone}_metadata.csv"
    if not emb_path.exists() or not meta_path.exists():
        raise FileNotFoundError(f"Missing: {emb_path} or {meta_path}")
    X = np.load(str(emb_path))
    meta = pd.read_csv(meta_path)
    assert X.shape[0] == len(meta), f"Row mismatch: {X.shape[0]} vs {len(meta)}"
    return X, meta


# ---------------------------------------------------------------------------
# Core benchmark
# ---------------------------------------------------------------------------
def run_backbone_benchmark(
    X: np.ndarray,
    meta: pd.DataFrame,
    backbone: str,
    seed: int = 42,
    alpha: float = 0.05,
) -> tuple[list[dict], list[dict]]:
    """Run all methods x attributes for one backbone."""
    y = meta["task_label"].astype(int).to_numpy()
    method_rows: list[dict] = []
    conv_rows: list[dict] = []

    for attr in ATTRIBUTES:
        if attr not in meta.columns:
            print(f"  [SKIP] {attr} not in metadata")
            continue

        frame = meta.copy()
        if attr == "race":
            frame = frame[frame["race"].isin(RACE_FOCUS)].copy()
            if frame.empty:
                print("  [SKIP] No samples for race focus groups")
                continue

        idx = frame.index.to_numpy()
        X_attr = X[idx]
        y_attr = y[idx]
        g_attr = frame[attr].to_numpy()
        n_groups = len(np.unique(g_attr))

        if n_groups < 2:
            print(f"  [SKIP] {attr}: only {n_groups} group(s)")
            continue

        print(f"\n  Attribute: {attr} | n={len(X_attr)} | groups={n_groups}")

        # Build extra_labels for intersectional analysis (all 3 attributes)
        extra_labels: dict[str, np.ndarray] = {}
        for other_attr in ATTRIBUTES:
            if other_attr in frame.columns:
                extra_labels[other_attr] = frame[other_attr].to_numpy()

        flagged_count = 0
        for method in METHODS:
            t0 = time.time()
            detector = ShortcutDetector(
                methods=[method],
                seed=seed,
                statistical_correction="fdr_bh",
                statistical_alpha=alpha,
            )
            try:
                detector.fit(X_attr, y_attr, group_labels=g_attr, extra_labels=extra_labels)
                result = detector.results_.get(method, {"success": False, "error": "missing"})
            except Exception as exc:
                result = {"success": False, "error": str(exc)}

            elapsed = time.time() - t0
            flag = int(method_flag(method, result))
            flagged_count += flag
            risk = str(result.get("risk_value") or result.get("risk_level") or "unknown")

            method_rows.append(
                {
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
            )
            status = "FLAGGED" if flag else "ok"
            print(f"    {method:12s} -> {status:8s} (risk={risk}, {elapsed:.2f}s)")

        conf = convergence_bucket(flagged_count, len(METHODS))
        conv_rows.append(
            {
                "backbone": backbone,
                "attribute": attr,
                "n_samples": int(X_attr.shape[0]),
                "n_groups": int(n_groups),
                "n_flagged_methods": int(flagged_count),
                "n_total_methods": len(METHODS),
                "confidence_level": conf,
            }
        )
        print(f"    => Convergence: {flagged_count}/{len(METHODS)} -> {conf}")

    return method_rows, conv_rows


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main() -> int:
    parser = argparse.ArgumentParser(description="Run MIMIC-CXR shortcut detection benchmark")
    parser.add_argument("--data-dir", type=Path, default=DATA_DIR)
    parser.add_argument("--output-dir", type=Path, default=OUTPUT_DIR)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--backbones", nargs="+", default=BACKBONES)
    args = parser.parse_args()

    args.output_dir.mkdir(parents=True, exist_ok=True)

    print("=" * 60)
    print(" MIMIC-CXR Shortcut Detection Benchmark")
    print("=" * 60)
    print(f"  Data dir:   {args.data_dir}")
    print(f"  Output dir: {args.output_dir}")
    print(f"  Backbones:  {args.backbones}")
    print(f"  Methods:    {METHODS}")
    print(f"  Attributes: {ATTRIBUTES}")
    print("=" * 60)

    all_method_rows: list[dict] = []
    all_conv_rows: list[dict] = []
    t_start = time.time()

    for backbone in args.backbones:
        print(f"\n{'='*60}")
        print(f"  Backbone: {backbone}")
        print(f"{'='*60}")
        try:
            X, meta = load_backbone(args.data_dir, backbone)
        except FileNotFoundError as e:
            print(f"  [SKIP] {e}")
            continue

        print(f"  Loaded: {X.shape[0]} samples, {X.shape[1]}-dim")
        print(f"  Label balance: {dict(pd.Series(meta['task_label']).value_counts().sort_index())}")

        m_rows, c_rows = run_backbone_benchmark(X, meta, backbone, seed=args.seed)
        all_method_rows.extend(m_rows)
        all_conv_rows.extend(c_rows)

    elapsed_total = time.time() - t_start

    # Save results
    methods_df = pd.DataFrame(all_method_rows)
    conv_df = pd.DataFrame(all_conv_rows)

    methods_path = args.output_dir / "mimic_methods.csv"
    conv_path = args.output_dir / "mimic_convergence.csv"
    summary_path = args.output_dir / "mimic_summary.json"

    if not methods_df.empty:
        methods_df.to_csv(methods_path, index=False)
    if not conv_df.empty:
        conv_df.to_csv(conv_path, index=False)

    # Build summary JSON
    summary: dict[str, Any] = {
        "benchmark": "mimic_cxr_shortcut_detection",
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "backbones": args.backbones,
        "methods": METHODS,
        "attributes": ATTRIBUTES,
        "seed": args.seed,
        "elapsed_seconds": round(elapsed_total, 2),
    }

    # Convergence matrix per backbone
    if not methods_df.empty:
        for backbone in args.backbones:
            sub = methods_df[methods_df["backbone"] == backbone]
            if sub.empty:
                continue
            pivot = (
                sub.pivot_table(
                    index="attribute", columns="method", values="flagged", aggfunc="max"
                )
                .fillna(0)
                .astype(int)
            )
            summary[f"convergence_{backbone}"] = pivot.reset_index().to_dict(orient="records")

    with open(summary_path, "w") as f:
        json.dump(summary, f, indent=2)

    # Print final report
    print(f"\n{'='*60}")
    print(f" Results Summary  ({elapsed_total:.1f}s total)")
    print(f"{'='*60}")

    if not conv_df.empty:
        print("\n  Convergence Matrix:")
        for _, row in conv_df.iterrows():
            print(
                f"    {row['backbone']:12s} | {row['attribute']:8s}: "
                f"{row['n_flagged_methods']}/{row['n_total_methods']} -> {row['confidence_level']}"
            )

    if not methods_df.empty:
        print("\n  Full Method Results:")
        print(methods_df.to_string(index=False))

    print("\n  Saved:")
    print(f"    Methods:     {methods_path}")
    print(f"    Convergence: {conv_path}")
    print(f"    Summary:     {summary_path}")
    print(f"{'='*60}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
