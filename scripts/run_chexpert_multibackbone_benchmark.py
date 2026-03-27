#!/usr/bin/env python3
"""Run CheXpert multi-backbone benchmark (B08/B09/B10).

For each backbone (resnet50, densenet121, vit_b_16):
  - Per-attribute analysis: sex, age_bin
  - Per-diagnosis analysis: filter to each diagnosis subset, run sex detection

Results saved to output/paper_benchmark/chexpert_multibackbone_results/
LaTeX tables saved to output/paper_tables/
"""

from __future__ import annotations

import json
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from shortcut_detect.benchmark.method_utils import ALL_METHODS, convergence_bucket, method_flag
from shortcut_detect.unified import ShortcutDetector

PROJECT_ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = PROJECT_ROOT / "data" / "chexpert_multibackbone"
OUTPUT_DIR = PROJECT_ROOT / "output" / "paper_benchmark" / "chexpert_multibackbone_results"
TABLE_DIR = PROJECT_ROOT / "output" / "paper_tables"

BACKBONES = [
    "resnet50",
    "densenet121",
    "vit_b_16",
    "vit_b_32",
    "dinov2_base",
    "rad_dino",
    "medsiglip",
]
METHODS = list(ALL_METHODS)
ATTRIBUTES = ["sex", "age_bin", "race"]

METHOD_SHORT = {
    "hbac": "HBAC",
    "probe": "Probe",
    "statistical": "Stat.",
    "geometric": "Geom.",
    "frequency": "Freq.",
    "bias_direction_pca": "BiasDir",
    "sis": "SIS",
    "demographic_parity": "DP",
    "equalized_odds": "EO",
    "intersectional": "Inter.",
}

DIAGNOSIS_COLS = [
    "No Finding",
    "Enlarged Cardiomediastinum",
    "Cardiomegaly",
    "Lung Opacity",
    "Lung Lesion",
    "Edema",
    "Consolidation",
    "Pneumonia",
    "Atelectasis",
    "Pneumothorax",
    "Pleural Effusion",
    "Pleural Other",
    "Fracture",
    "Support Devices",
]

BACKBONE_LABELS = {
    "resnet50": "ResNet-50",
    "densenet121": "DenseNet-121",
    "vit_b_16": "ViT-B/16",
    "vit_b_32": "ViT-B/32",
    "dinov2_base": "DINOv2",
    "rad_dino": "RAD-DINO",
    "medsiglip": "MedSigLIP",
}


def load_backbone(backbone: str) -> tuple[np.ndarray, pd.DataFrame] | None:
    """Load embeddings and metadata for a backbone.

    Returns ``None`` when the required files are missing so callers can
    gracefully skip unavailable backbones.
    """
    emb_path = DATA_DIR / f"{backbone}_embeddings.npy"
    meta_path = DATA_DIR / f"{backbone}_metadata.csv"
    if not emb_path.exists() or not meta_path.exists():
        return None
    X = np.load(str(emb_path))
    meta = pd.read_csv(meta_path)
    assert X.shape[0] == len(meta), f"Row mismatch: {X.shape[0]} vs {len(meta)}"
    return X, meta


def run_detection(
    X: np.ndarray,
    y: np.ndarray,
    g: np.ndarray,
    seed: int = 42,
    extra_labels: dict[str, np.ndarray] | None = None,
) -> list[dict[str, Any]]:
    """Run all methods on a single (X, y, g) triple. Returns per-method results."""
    rows = []
    for method in METHODS:
        t0 = time.time()
        try:
            detector = ShortcutDetector(
                methods=[method],
                seed=seed,
                statistical_correction="fdr_bh",
                statistical_alpha=0.05,
            )
            detector.fit(X, y, group_labels=g, extra_labels=extra_labels)
            result = detector.results_.get(method, {"success": False, "error": "missing"})
        except Exception as exc:
            result = {"success": False, "error": str(exc)}
        elapsed = time.time() - t0
        flag = method_flag(method, result)
        rows.append(
            {
                "method": method,
                "flagged": int(flag),
                "success": bool(result.get("success", False)),
                "error": result.get("error"),
                "elapsed_s": round(elapsed, 3),
            }
        )
    return rows


# ============================================================================
# Task 1: Per-backbone per-attribute benchmark
# ============================================================================


def run_attribute_benchmark() -> pd.DataFrame:
    """Run per-backbone per-attribute analysis. Returns results DataFrame."""
    all_rows = []

    for bb in BACKBONES:
        print(f"\n{'='*60}")
        print(f"  Backbone: {bb}")
        print(f"{'='*60}")
        loaded = load_backbone(bb)
        if loaded is None:
            print(f"  [SKIP] {bb}: embedding/metadata files not found")
            continue
        X, meta = loaded
        y = meta["task_label"].astype(int).to_numpy()

        # Build extra_labels for intersectional support
        available_attrs: dict[str, np.ndarray] = {}
        for a in ATTRIBUTES:
            if a in meta.columns:
                available_attrs[a] = meta[a].astype(str).to_numpy()
        extra = available_attrs if len(available_attrs) >= 2 else None

        for attr in ATTRIBUTES:
            if attr not in meta.columns:
                print(f"  [SKIP] {attr} not in metadata")
                continue

            g = meta[attr].astype(str).to_numpy()
            n_groups = len(np.unique(g))
            if n_groups < 2:
                print(f"  [SKIP] {attr}: only {n_groups} group(s)")
                continue

            print(f"\n  Attribute: {attr} | n={len(X)} | groups={n_groups}")
            method_results = run_detection(X, y, g, extra_labels=extra)

            n_flagged = sum(r["flagged"] for r in method_results)
            for r in method_results:
                r.update(
                    {
                        "backbone": bb,
                        "attribute": attr,
                        "n_samples": len(X),
                        "n_groups": n_groups,
                    }
                )
                all_rows.append(r)
                status = "FLAGGED" if r["flagged"] else "ok"
                print(f"    {r['method']:12s} -> {status}")

            conf = convergence_bucket(n_flagged, len(METHODS))
            print(f"    => Convergence: {n_flagged}/{len(METHODS)} -> {conf}")

    return pd.DataFrame(all_rows)


# ============================================================================
# Task 1b: Per-diagnosis analysis (B09)
# ============================================================================


def run_diagnosis_benchmark(min_positive: int = 50) -> pd.DataFrame:
    """Run per-diagnosis per-backbone analysis for sex attribute."""
    all_rows = []

    for bb in BACKBONES:
        print(f"\n{'='*60}")
        print(f"  Per-Diagnosis Benchmark: {bb}")
        print(f"{'='*60}")
        loaded = load_backbone(bb)
        if loaded is None:
            print(f"  [SKIP] {bb}: embedding/metadata files not found")
            continue
        X, meta = loaded

        for dx in DIAGNOSIS_COLS:
            if dx not in meta.columns:
                continue
            # Filter to patients with this diagnosis = 1.0
            mask = meta[dx] == 1.0
            n_pos = mask.sum()
            if n_pos < min_positive:
                print(f"  [SKIP] {dx}: only {n_pos} positives (< {min_positive})")
                continue

            X_sub = X[mask.values]
            meta_sub = meta[mask].copy()
            # Use task_label as y (or could use the diagnosis itself)
            y_sub = meta_sub["task_label"].astype(int).to_numpy()
            g_sub = meta_sub["sex"].astype(str).to_numpy()

            n_groups = len(np.unique(g_sub))
            if n_groups < 2:
                continue

            print(f"\n  Diagnosis: {dx} | n={n_pos} | sex groups={n_groups}")
            # Build extra_labels from the subset for intersectional support
            dx_extra: dict[str, np.ndarray] = {}
            for a in ATTRIBUTES:
                if a in meta_sub.columns:
                    dx_extra[a] = meta_sub[a].astype(str).to_numpy()
            dx_el = dx_extra if len(dx_extra) >= 2 else None
            method_results = run_detection(X_sub, y_sub, g_sub, extra_labels=dx_el)

            n_flagged = sum(r["flagged"] for r in method_results)
            for r in method_results:
                r.update(
                    {
                        "backbone": bb,
                        "diagnosis": dx,
                        "attribute": "sex",
                        "n_samples": int(n_pos),
                        "n_groups": n_groups,
                    }
                )
                all_rows.append(r)
                status = "FLAGGED" if r["flagged"] else "ok"
                print(f"    {r['method']:12s} -> {status}")

            conf = convergence_bucket(n_flagged, len(METHODS))
            print(f"    => Convergence: {n_flagged}/{len(METHODS)} -> {conf}")

    return pd.DataFrame(all_rows)


# ============================================================================
# Task 2: Generate multi-backbone LaTeX table
# ============================================================================


def generate_multibackbone_table(df: pd.DataFrame) -> str:
    """Generate LaTeX table for multi-backbone per-attribute results."""
    n_methods = len(METHODS)
    col_spec = "ll" + "c" * n_methods + "|c"
    method_headers = " & ".join(rf"\textbf{{{METHOD_SHORT.get(m, m)}}}" for m in METHODS)

    lines = []
    lines.append(r"\begin{table*}[ht]")
    lines.append(r"\centering")
    lines.append(r"\scriptsize")
    lines.append(
        r"\caption{{\color{blue}CheXpert multi-backbone shortcut detection results. "
        r"Seven backbone architectures on CheXpert validation set. "
        r"$\checkmark$ = detected, $\cdot$ = not detected.}}"
    )
    lines.append(r"\label{tab:chexpert_results}")
    lines.append(rf"\begin{{tabular}}{{{col_spec}}}")
    lines.append(r"\hline")
    lines.append(
        rf"\textbf{{Backbone}} & \textbf{{Attribute}} & {method_headers} & \textbf{{Agree.}} \\"
    )
    lines.append(r"\hline")

    for bb in BACKBONES:
        bb_label = BACKBONE_LABELS[bb]
        bb_df = df[df["backbone"] == bb]
        attrs = [a for a in ATTRIBUTES if a in bb_df["attribute"].values]
        first = True
        for attr in attrs:
            attr_label = {"sex": "Sex", "age_bin": "Age Bin"}.get(attr, attr)
            if first:
                row = f"\\multirow{{{len(attrs)}}}{{*}}{{{bb_label}}} & {attr_label}"
                first = False
            else:
                row = f" & {attr_label}"

            n_flagged = 0
            for m in METHODS:
                sub = bb_df[(bb_df["attribute"] == attr) & (bb_df["method"] == m)]
                if len(sub) > 0 and sub["flagged"].values[0]:
                    row += r" & \checkmark"
                    n_flagged += 1
                else:
                    row += r" & $\cdot$"
            row += f" & {n_flagged}/{len(METHODS)}"
            row += r" \\"
            lines.append(row)
        lines.append(r"\hline")

    lines.append(r"\end{tabular}")
    lines.append(r"\end{table*}")

    return "\n".join(lines)


# ============================================================================
# Task 3: Generate per-diagnosis LaTeX table
# ============================================================================


def generate_diagnosis_table(df: pd.DataFrame, top_n: int = 5) -> str:
    """Generate per-diagnosis table (top N diagnoses by prevalence)."""
    if df.empty:
        return "% No per-diagnosis results available"

    # Get top diagnoses by sample count (across all backbones, take max)
    dx_counts = df.groupby("diagnosis")["n_samples"].max().sort_values(ascending=False)
    top_dx = dx_counts.head(top_n).index.tolist()

    n_methods = len(METHODS)
    col_spec = "llr" + "c" * n_methods + "|c"
    method_headers = " & ".join(rf"\textbf{{{METHOD_SHORT.get(m, m)}}}" for m in METHODS)

    lines = []
    lines.append(r"\begin{table*}[ht]")
    lines.append(r"\centering")
    lines.append(r"\scriptsize")
    lines.append(
        r"\caption{{\color{blue}Per-diagnosis shortcut detection on CheXpert "
        r"(attribute = sex). Top 5 diagnoses by prevalence. Each cell: "
        r"$\checkmark$ = method flagged shortcut, $\cdot$ = not flagged. "
        r"Subsets are filtered to patients with the given diagnosis.}}"
    )
    lines.append(r"\label{tab:per_diagnosis}")
    lines.append(rf"\begin{{tabular}}{{{col_spec}}}")
    lines.append(r"\hline")
    lines.append(
        rf"\textbf{{Diagnosis}} & \textbf{{Backbone}} & \textbf{{$n$}} & "
        rf"{method_headers} & \textbf{{Agree.}} \\"
    )
    lines.append(r"\hline")

    for dx in top_dx:
        dx_df = df[df["diagnosis"] == dx]
        bbs_in_dx = [b for b in BACKBONES if b in dx_df["backbone"].values]
        dx_display = dx.replace("_", r"\_")
        first = True
        for bb in bbs_in_dx:
            bb_label = BACKBONE_LABELS[bb]
            sub = dx_df[dx_df["backbone"] == bb]
            n_samp = sub["n_samples"].values[0] if len(sub) > 0 else 0

            if first:
                row = f"\\multirow{{{len(bbs_in_dx)}}}{{*}}{{{dx_display}}} & {bb_label} & {n_samp}"
                first = False
            else:
                row = f" & {bb_label} & {n_samp}"

            n_flagged = 0
            for m in METHODS:
                m_sub = sub[sub["method"] == m]
                if len(m_sub) > 0 and m_sub["flagged"].values[0]:
                    row += r" & \checkmark"
                    n_flagged += 1
                else:
                    row += r" & $\cdot$"
            row += f" & {n_flagged}/{len(METHODS)}"
            row += r" \\"
            lines.append(row)
        lines.append(r"\hline")

    lines.append(r"\end{tabular}")
    lines.append(r"\end{table*}")

    return "\n".join(lines)


# ============================================================================
# Main
# ============================================================================


def main():
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    TABLE_DIR.mkdir(parents=True, exist_ok=True)

    print(f"{'='*60}")
    print(" CheXpert Multi-Backbone Benchmark")
    print(f"{'='*60}")
    print(f"  Backbones:  {', '.join(BACKBONES)}")
    print(f"  Methods:    {', '.join(METHODS)}")
    print(f"  Attributes: {', '.join(ATTRIBUTES)}")
    print(f"  Output:     {OUTPUT_DIR}")
    print(f"{'='*60}")

    t_start = time.time()

    # --- Task 1a: Per-attribute benchmark ---
    print("\n" + "=" * 60)
    print(" TASK 1a: Per-Attribute Benchmark")
    print("=" * 60)
    attr_df = run_attribute_benchmark()
    attr_df.to_csv(OUTPUT_DIR / "multibackbone_attribute_results.csv", index=False)
    print(f"\nSaved attribute results: {OUTPUT_DIR / 'multibackbone_attribute_results.csv'}")

    # --- Task 1b: Per-diagnosis benchmark ---
    print("\n" + "=" * 60)
    print(" TASK 1b: Per-Diagnosis Benchmark (B09)")
    print("=" * 60)
    dx_df = run_diagnosis_benchmark(min_positive=50)
    dx_df.to_csv(OUTPUT_DIR / "multibackbone_diagnosis_results.csv", index=False)
    print(f"\nSaved diagnosis results: {OUTPUT_DIR / 'multibackbone_diagnosis_results.csv'}")

    elapsed = time.time() - t_start
    print(f"\nTotal elapsed: {elapsed:.1f}s")

    # --- Task 2: Multi-backbone LaTeX table ---
    print("\n" + "=" * 60)
    print(" TASK 2: Generating multi-backbone table")
    print("=" * 60)
    mb_latex = generate_multibackbone_table(attr_df)
    mb_path = TABLE_DIR / "table_chexpert_multibackbone.tex"
    mb_path.write_text(mb_latex)
    print(f"Saved: {mb_path}")
    print(mb_latex)

    # --- Task 3: Per-diagnosis LaTeX table ---
    print("\n" + "=" * 60)
    print(" TASK 3: Generating per-diagnosis table")
    print("=" * 60)
    dx_latex = generate_diagnosis_table(dx_df, top_n=5)
    dx_path = TABLE_DIR / "table_per_diagnosis.tex"
    dx_path.write_text(dx_latex)
    print(f"Saved: {dx_path}")
    print(dx_latex)

    # --- Summary JSON ---
    summary = {
        "benchmark": "chexpert_multibackbone",
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "backbones": BACKBONES,
        "methods": list(METHODS),
        "attributes": list(ATTRIBUTES),
        "elapsed_seconds": round(elapsed, 2),
        "n_attribute_rows": len(attr_df),
        "n_diagnosis_rows": len(dx_df),
    }
    with open(OUTPUT_DIR / "summary.json", "w") as f:
        json.dump(summary, f, indent=2)

    print(f"\n{'='*60}")
    print(" DONE")
    print(f"{'='*60}")


if __name__ == "__main__":
    main()
