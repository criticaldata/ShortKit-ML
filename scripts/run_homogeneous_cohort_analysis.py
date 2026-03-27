#!/usr/bin/env python3
"""Homogeneous cohort analysis: test for demographic encoding within diagnosis-matched subgroups.

Tests the "shortcut of a shortcut" hypothesis: is race/sex encoding in embeddings
direct (present even when pathology is controlled) or pathology-mediated (disappears
when diagnosis is held constant)?

Protocol:
  For each diagnosis with sufficient samples:
    1. Filter to patients with ONLY that diagnosis
    2. Split by demographic attribute (e.g., WHITE vs BLACK)
    3. Run all 13 detection methods
    4. If demographics are still detectable within a homogeneous cohort,
       the encoding is direct, not pathology-mediated

Uses 3 shared backbones (RAD-DINO, MedSigLIP, ViT-B/16) for cross-dataset comparison.
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
OUTPUT_DIR = PROJECT_ROOT / "output" / "paper_benchmark" / "homogeneous_cohort"

# Shared backbones for cross-dataset comparison
CHEXPERT_DIR = PROJECT_ROOT / "data" / "chexpert_multibackbone"
MIMIC_DIR = PROJECT_ROOT / "data" / "mimic_cxr"

SHARED_BACKBONES = {
    "RAD-DINO": {
        "chexpert": ("rad_dino_embeddings.npy", "rad_dino_metadata.csv", CHEXPERT_DIR),
        "mimic": ("rad_dino_embeddings.npy", "rad_dino_metadata.csv", MIMIC_DIR),
    },
    "MedSigLIP": {
        "chexpert": ("medsiglip_embeddings.npy", "medsiglip_metadata.csv", CHEXPERT_DIR),
        "mimic": ("medsiglip_embeddings.npy", "medsiglip_metadata.csv", MIMIC_DIR),
    },
    "ViT-B/16": {
        "chexpert": ("vit_b_16_embeddings.npy", "vit_b_16_metadata.csv", CHEXPERT_DIR),
        "mimic": ("vit16_cls_embeddings.npy", "vit16_cls_metadata.csv", MIMIC_DIR),
    },
}

DIAGNOSES = ["Lung Opacity", "Pleural Effusion", "Support Devices", "Atelectasis", "Edema"]
METHODS = list(ALL_METHODS)
MIN_PER_GROUP = 20  # Minimum samples per demographic group within a diagnosis


def load_data(
    emb_file: str, meta_file: str, data_dir: Path
) -> tuple[np.ndarray, pd.DataFrame] | None:
    """Load embeddings + metadata. Returns None if files missing."""
    emb_path = data_dir / emb_file
    meta_path = data_dir / meta_file
    if not emb_path.exists() or not meta_path.exists():
        return None
    X = np.load(str(emb_path))
    meta = pd.read_csv(meta_path)
    return X, meta


def run_detection(
    X: np.ndarray,
    y: np.ndarray,
    g: np.ndarray,
    seed: int = 42,
) -> list[dict[str, Any]]:
    """Run all methods on (X, y, g). Returns per-method results."""
    rows = []
    for method_name in METHODS:
        t0 = time.time()
        try:
            detector = ShortcutDetector(
                methods=[method_name],
                seed=seed,
                statistical_correction="fdr_bh",
                statistical_alpha=0.05,
            )
            detector.fit(X, y, group_labels=g)
            result = detector.results_.get(method_name, {"success": False, "error": "missing"})
        except Exception as exc:
            result = {"success": False, "error": str(exc)}
        elapsed = time.time() - t0
        flag = method_flag(method_name, result)
        rows.append(
            {
                "method": method_name,
                "flagged": int(flag),
                "success": bool(result.get("success", False)),
                "risk_level": str(result.get("risk_value", result.get("risk_level", "unknown"))),
                "elapsed_s": round(elapsed, 3),
            }
        )
    return rows


def run_cohort_analysis() -> pd.DataFrame:
    """Run the full homogeneous cohort analysis."""
    all_rows: list[dict[str, Any]] = []

    for backbone_name, datasets in SHARED_BACKBONES.items():
        for dataset_name, (emb_file, meta_file, data_dir) in datasets.items():
            data = load_data(emb_file, meta_file, data_dir)
            if data is None:
                print(f"  [SKIP] {backbone_name}/{dataset_name}: files not found")
                continue

            X_full, meta_full = data

            for dx in DIAGNOSES:
                if dx not in meta_full.columns:
                    continue

                # Filter to patients WITH this diagnosis
                dx_mask = meta_full[dx] == 1.0
                if dx_mask.sum() < MIN_PER_GROUP * 2:
                    continue

                X_dx = X_full[dx_mask.values]
                meta_dx = meta_full[dx_mask].copy()

                # Test sex encoding within this diagnosis
                if "sex" in meta_dx.columns:
                    g_sex = meta_dx["sex"].astype(str).to_numpy()
                    unique_sex = np.unique(g_sex)
                    min_group = (
                        min(np.sum(g_sex == v) for v in unique_sex) if len(unique_sex) >= 2 else 0
                    )

                    if len(unique_sex) >= 2 and min_group >= MIN_PER_GROUP:
                        # Use task_label as y (binary)
                        y = (
                            meta_dx["task_label"].astype(int).to_numpy()
                            if "task_label" in meta_dx.columns
                            else np.zeros(len(X_dx), dtype=int)
                        )

                        print(
                            f"\n  [{dataset_name}/{backbone_name}] {dx} | sex | n={len(X_dx)} | groups={dict(zip(*np.unique(g_sex, return_counts=True), strict=False))}"
                        )
                        method_results = run_detection(X_dx, y, g_sex)

                        n_flagged = sum(r["flagged"] for r in method_results)
                        bucket = convergence_bucket(n_flagged, len(METHODS))

                        for r in method_results:
                            r.update(
                                {
                                    "backbone": backbone_name,
                                    "dataset": dataset_name,
                                    "diagnosis": dx,
                                    "attribute": "sex",
                                    "n_samples": len(X_dx),
                                    "n_flagged_total": n_flagged,
                                    "convergence": bucket,
                                }
                            )
                            all_rows.append(r)
                            status = "FLAGGED" if r["flagged"] else "ok"
                            print(f"    {r['method']:20s} -> {status}")
                        print(f"    => {n_flagged}/{len(METHODS)} -> {bucket}")

                # Test race encoding within this diagnosis (MIMIC only, or CheXpert if available)
                if "race" in meta_dx.columns:
                    race_vals = meta_dx["race"].dropna().astype(str)
                    if len(race_vals) < MIN_PER_GROUP * 2:
                        continue
                    g_race = meta_dx.loc[race_vals.index, "race"].astype(str).to_numpy()
                    # Reindex properly using boolean mask
                    race_mask = meta_dx["race"].notna()
                    X_race = X_dx[race_mask.values]
                    meta_race = meta_dx[race_mask].copy()
                    g_race = meta_race["race"].astype(str).to_numpy()

                    unique_race = np.unique(g_race)
                    if len(unique_race) < 2:
                        continue
                    min_group = min(np.sum(g_race == v) for v in unique_race)
                    if min_group < MIN_PER_GROUP:
                        continue

                    y_race = (
                        meta_race["task_label"].astype(int).to_numpy()
                        if "task_label" in meta_race.columns
                        else np.zeros(len(X_race), dtype=int)
                    )

                    print(
                        f"\n  [{dataset_name}/{backbone_name}] {dx} | race | n={len(X_race)} | groups={dict(zip(*np.unique(g_race, return_counts=True), strict=False))}"
                    )
                    method_results = run_detection(X_race, y_race, g_race)

                    n_flagged = sum(r["flagged"] for r in method_results)
                    bucket = convergence_bucket(n_flagged, len(METHODS))

                    for r in method_results:
                        r.update(
                            {
                                "backbone": backbone_name,
                                "dataset": dataset_name,
                                "diagnosis": dx,
                                "attribute": "race",
                                "n_samples": len(X_race),
                                "n_flagged_total": n_flagged,
                                "convergence": bucket,
                            }
                        )
                        all_rows.append(r)
                        status = "FLAGGED" if r["flagged"] else "ok"
                        print(f"    {r['method']:20s} -> {status}")
                    print(f"    => {n_flagged}/{len(METHODS)} -> {bucket}")

    return pd.DataFrame(all_rows)


def generate_summary(df: pd.DataFrame) -> str:
    """Generate interpretive summary of homogeneous cohort results."""
    lines = [
        "=" * 70,
        "  HOMOGENEOUS COHORT ANALYSIS — SHORTCUT OF SHORTCUT TEST",
        "=" * 70,
        "",
        "Question: Is demographic encoding DIRECT or PATHOLOGY-MEDIATED?",
        "Method: Filter to same-diagnosis patients, test for demographic encoding.",
        "If encoding persists within a homogeneous cohort, it's DIRECT.",
        "If it disappears, it was PATHOLOGY-MEDIATED.",
        "",
    ]

    if df.empty:
        lines.append("No results available.")
        return "\n".join(lines)

    # Summary by diagnosis × attribute × dataset
    summary = (
        df.groupby(["diagnosis", "attribute", "dataset", "backbone"])
        .agg(
            n_samples=("n_samples", "first"),
            n_flagged=("n_flagged_total", "first"),
            convergence=("convergence", "first"),
        )
        .reset_index()
    )

    lines.append(
        f"{'Diagnosis':20s} | {'Attr':5s} | {'Dataset':10s} | {'Backbone':12s} | {'n':>5s} | {'Agree':>8s} | {'Verdict'}"
    )
    lines.append("-" * 90)

    for _, row in summary.iterrows():
        n_methods = len(METHODS)
        ratio = row["n_flagged"] / n_methods if n_methods > 0 else 0
        if ratio >= 0.3:
            verdict = "DIRECT encoding"
        elif ratio > 0:
            verdict = "Weak signal"
        else:
            verdict = "No encoding (pathology-mediated?)"

        lines.append(
            f"{row['diagnosis']:20s} | {row['attribute']:5s} | {row['dataset']:10s} | "
            f"{row['backbone']:12s} | {row['n_samples']:5d} | "
            f"{row['n_flagged']}/{n_methods:2d}    | {verdict}"
        )

    # Overall interpretation
    lines.append("")
    lines.append("=" * 70)
    lines.append("  INTERPRETATION")
    lines.append("=" * 70)

    sex_flags = summary[summary["attribute"] == "sex"]["n_flagged"]
    race_flags = summary[summary["attribute"] == "race"]["n_flagged"]

    if len(sex_flags) > 0:
        lines.append(
            f"  Sex encoding within diagnoses: mean {sex_flags.mean():.1f}/{len(METHODS)} methods flag"
        )
        if sex_flags.mean() >= 3:
            lines.append("  => SEX encoding is DIRECT (persists within homogeneous cohorts)")
        else:
            lines.append("  => SEX encoding is WEAK within homogeneous cohorts")

    if len(race_flags) > 0:
        lines.append(
            f"  Race encoding within diagnoses: mean {race_flags.mean():.1f}/{len(METHODS)} methods flag"
        )
        if race_flags.mean() >= 3:
            lines.append("  => RACE encoding is DIRECT (persists within homogeneous cohorts)")
        elif race_flags.mean() > 0:
            lines.append("  => RACE encoding is PARTIALLY direct, PARTIALLY pathology-mediated")
        else:
            lines.append(
                "  => RACE encoding appears PATHOLOGY-MEDIATED (disappears in homogeneous cohorts)"
            )

    return "\n".join(lines)


def main():
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    print("=" * 70)
    print("  Homogeneous Cohort Analysis")
    print("  Testing: Direct vs. Pathology-Mediated Demographic Encoding")
    print("=" * 70)
    print(f"  Shared backbones: {', '.join(SHARED_BACKBONES.keys())}")
    print(f"  Diagnoses: {', '.join(DIAGNOSES)}")
    print(f"  Methods: {len(METHODS)} ({', '.join(METHODS[:5])}...)")
    print(f"  Min per group: {MIN_PER_GROUP}")
    print(f"  Output: {OUTPUT_DIR}")
    print("=" * 70)

    t_start = time.time()
    df = run_cohort_analysis()
    elapsed = time.time() - t_start

    # Save results
    df.to_csv(OUTPUT_DIR / "homogeneous_cohort_results.csv", index=False)
    print(f"\nSaved: {OUTPUT_DIR / 'homogeneous_cohort_results.csv'}")

    # Generate and print summary
    summary_text = generate_summary(df)
    print(summary_text)
    (OUTPUT_DIR / "homogeneous_cohort_summary.txt").write_text(summary_text)

    # Save metadata
    meta = {
        "experiment": "homogeneous_cohort_analysis",
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "shared_backbones": list(SHARED_BACKBONES.keys()),
        "diagnoses": DIAGNOSES,
        "n_methods": len(METHODS),
        "min_per_group": MIN_PER_GROUP,
        "elapsed_seconds": round(elapsed, 2),
        "total_runs": len(df),
        "seed": 42,
    }
    with open(OUTPUT_DIR / "summary.json", "w") as f:
        json.dump(meta, f, indent=2)

    print(f"\nTotal elapsed: {elapsed:.1f}s")
    print("DONE.")


if __name__ == "__main__":
    main()
