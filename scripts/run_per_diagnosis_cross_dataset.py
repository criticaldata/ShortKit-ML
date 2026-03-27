#!/usr/bin/env python3
"""Per-diagnosis cross-dataset shortcut detection (L01 + L02).

For each diagnosis present with >= 50 positive cases in BOTH CheXpert and
MIMIC-CXR, run ShortcutDetector and compare results across datasets.

Analyses per diagnosis:
  - CheXpert subset: attribute = sex
  - MIMIC subset:    attribute = sex  (direct comparison)
  - MIMIC subset:    attribute = race (additional)

Outputs:
  - output/paper_benchmark/cross_dataset_diagnosis/  (CSVs + summary JSON)
  - output/paper_tables/table_cross_dataset_diagnosis.tex
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

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
PROJECT_ROOT = Path(__file__).resolve().parent.parent
CHEXPERT_DIR = PROJECT_ROOT / "data" / "chexpert_multibackbone"
MIMIC_DIR = PROJECT_ROOT / "data" / "mimic_cxr"
OUTPUT_DIR = PROJECT_ROOT / "output" / "paper_benchmark" / "cross_dataset_diagnosis"
TABLE_DIR = PROJECT_ROOT / "output" / "paper_tables"

METHODS = list(ALL_METHODS)
RACE_FOCUS = ["WHITE", "BLACK", "ASIAN"]

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

MIN_POSITIVE = 50
TOP_N = 5


def run_detection(
    X: np.ndarray,
    y: np.ndarray,
    g: np.ndarray,
    seed: int = 42,
    extra_labels: dict[str, np.ndarray] | None = None,
) -> list[dict[str, Any]]:
    """Run all methods on a single (X, y, g) triple."""
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


# ---------------------------------------------------------------------------
# Data loading
# ---------------------------------------------------------------------------
def load_chexpert() -> tuple[np.ndarray, pd.DataFrame]:
    X = np.load(str(CHEXPERT_DIR / "resnet50_embeddings.npy"))
    meta = pd.read_csv(CHEXPERT_DIR / "resnet50_metadata.csv")
    assert X.shape[0] == len(meta)
    return X, meta


def load_mimic() -> tuple[np.ndarray, pd.DataFrame]:
    X = np.load(str(MIMIC_DIR / "rad_dino_embeddings.npy"))
    meta = pd.read_csv(MIMIC_DIR / "rad_dino_metadata.csv")
    assert X.shape[0] == len(meta)
    return X, meta


# ---------------------------------------------------------------------------
# Step 1: identify eligible diagnoses (>= MIN_POSITIVE in BOTH datasets)
# ---------------------------------------------------------------------------
def eligible_diagnoses(
    chx_meta: pd.DataFrame, mimic_meta: pd.DataFrame
) -> list[tuple[str, int, int]]:
    """Return list of (diagnosis, chx_count, mimic_count) sorted by min count desc."""
    eligible = []
    for dx in DIAGNOSIS_COLS:
        if dx not in chx_meta.columns or dx not in mimic_meta.columns:
            continue
        n_chx = int((chx_meta[dx] == 1.0).sum())
        n_mimic = int((mimic_meta[dx] == 1.0).sum())
        if n_chx >= MIN_POSITIVE and n_mimic >= MIN_POSITIVE:
            eligible.append((dx, n_chx, n_mimic))
    # Sort by minimum across both datasets (descending) to get the most robust comparisons
    eligible.sort(key=lambda t: min(t[1], t[2]), reverse=True)
    return eligible[:TOP_N]


# ---------------------------------------------------------------------------
# Step 2: run per-diagnosis analysis
# ---------------------------------------------------------------------------
def run_cross_dataset_analysis() -> pd.DataFrame:
    """Run analysis and return combined results DataFrame."""
    print("Loading CheXpert (ResNet-50) ...")
    X_chx, meta_chx = load_chexpert()
    print(f"  CheXpert: {X_chx.shape[0]} samples, {X_chx.shape[1]}-dim")

    print("Loading MIMIC-CXR (RAD-DINO) ...")
    X_mimic, meta_mimic = load_mimic()
    print(f"  MIMIC-CXR: {X_mimic.shape[0]} samples, {X_mimic.shape[1]}-dim")

    top_dx = eligible_diagnoses(meta_chx, meta_mimic)
    print(f"\nTop {TOP_N} eligible diagnoses (>= {MIN_POSITIVE} in both):")
    for dx, nc, nm in top_dx:
        print(f"  {dx:30s}  CheXpert={nc:4d}  MIMIC={nm:4d}")

    all_rows: list[dict] = []

    for dx, _n_chx, _n_mimic in top_dx:
        print(f"\n{'='*65}")
        print(f"  DIAGNOSIS: {dx}")
        print(f"{'='*65}")

        # --- CheXpert subset (sex) ---
        mask_chx = meta_chx[dx] == 1.0
        X_sub = X_chx[mask_chx.values]
        meta_sub = meta_chx[mask_chx].copy()
        y_sub = meta_sub["task_label"].astype(int).to_numpy()
        g_sub = meta_sub["sex"].astype(str).to_numpy()
        n_groups = len(np.unique(g_sub))

        # Build extra_labels for intersectional (CheXpert only has sex)
        chx_extra: dict[str, np.ndarray] = {"sex": g_sub}

        print(f"\n  [CheXpert | sex] n={mask_chx.sum()}, groups={n_groups}")
        if n_groups >= 2:
            results = run_detection(X_sub, y_sub, g_sub, extra_labels=chx_extra)
            n_flagged = sum(r["flagged"] for r in results)
            for r in results:
                r.update(
                    {
                        "dataset": "CheXpert",
                        "backbone": "resnet50",
                        "diagnosis": dx,
                        "attribute": "sex",
                        "n_samples": int(mask_chx.sum()),
                        "n_groups": n_groups,
                        "convergence": convergence_bucket(n_flagged, len(METHODS)),
                    }
                )
                all_rows.append(r)
                status = "FLAGGED" if r["flagged"] else "ok"
                print(f"    {r['method']:12s} -> {status}")
            print(
                f"    => Convergence: {n_flagged}/{len(METHODS)} -> {convergence_bucket(n_flagged, len(METHODS))}"
            )
        else:
            print("    [SKIP] fewer than 2 sex groups")

        # --- MIMIC subset (sex) ---
        mask_mimic = meta_mimic[dx] == 1.0
        X_sub_m = X_mimic[mask_mimic.values]
        meta_sub_m = meta_mimic[mask_mimic].copy()
        y_sub_m = meta_sub_m["task_label"].astype(int).to_numpy()
        g_sub_m = meta_sub_m["sex"].astype(str).to_numpy()
        n_groups_m = len(np.unique(g_sub_m))

        # Build extra_labels for intersectional (MIMIC has sex, race, age_bin)
        mimic_extra: dict[str, np.ndarray] = {"sex": g_sub_m}
        for ecol in ("race", "age_bin"):
            if ecol in meta_sub_m.columns:
                mimic_extra[ecol] = meta_sub_m[ecol].to_numpy()

        print(f"\n  [MIMIC | sex] n={mask_mimic.sum()}, groups={n_groups_m}")
        if n_groups_m >= 2:
            results = run_detection(X_sub_m, y_sub_m, g_sub_m, extra_labels=mimic_extra)
            n_flagged = sum(r["flagged"] for r in results)
            for r in results:
                r.update(
                    {
                        "dataset": "MIMIC-CXR",
                        "backbone": "rad_dino",
                        "diagnosis": dx,
                        "attribute": "sex",
                        "n_samples": int(mask_mimic.sum()),
                        "n_groups": n_groups_m,
                        "convergence": convergence_bucket(n_flagged, len(METHODS)),
                    }
                )
                all_rows.append(r)
                status = "FLAGGED" if r["flagged"] else "ok"
                print(f"    {r['method']:12s} -> {status}")
            print(
                f"    => Convergence: {n_flagged}/{len(METHODS)} -> {convergence_bucket(n_flagged, len(METHODS))}"
            )
        else:
            print("    [SKIP] fewer than 2 sex groups")

        # --- MIMIC subset (race) ---
        meta_race = meta_sub_m[meta_sub_m["race"].isin(RACE_FOCUS)].copy()
        if len(meta_race) >= MIN_POSITIVE:
            idx_race = meta_race.index.to_numpy()
            # Compute position-based indices relative to the mask
            pos_in_mimic = np.where(mask_mimic.values)[0]
            # Map meta_race index back to position in full X_mimic
            np.array([np.searchsorted(pos_in_mimic, i) for i in idx_race if i in pos_in_mimic])
            # Simpler: just use the original index into X_mimic
            X_sub_r = X_mimic[idx_race]
            y_sub_r = meta_race["task_label"].astype(int).to_numpy()
            g_sub_r = meta_race["race"].astype(str).to_numpy()
            n_groups_r = len(np.unique(g_sub_r))

            # Build extra_labels for intersectional (race subset of MIMIC)
            race_extra: dict[str, np.ndarray] = {"race": g_sub_r}
            for ecol in ("sex", "age_bin"):
                if ecol in meta_race.columns:
                    race_extra[ecol] = meta_race[ecol].to_numpy()

            print(f"\n  [MIMIC | race] n={len(meta_race)}, groups={n_groups_r}")
            if n_groups_r >= 2:
                results = run_detection(X_sub_r, y_sub_r, g_sub_r, extra_labels=race_extra)
                n_flagged = sum(r["flagged"] for r in results)
                for r in results:
                    r.update(
                        {
                            "dataset": "MIMIC-CXR",
                            "backbone": "rad_dino",
                            "diagnosis": dx,
                            "attribute": "race",
                            "n_samples": len(meta_race),
                            "n_groups": n_groups_r,
                            "convergence": convergence_bucket(n_flagged, len(METHODS)),
                        }
                    )
                    all_rows.append(r)
                    status = "FLAGGED" if r["flagged"] else "ok"
                    print(f"    {r['method']:12s} -> {status}")
                print(
                    f"    => Convergence: {n_flagged}/{len(METHODS)} -> {convergence_bucket(n_flagged, len(METHODS))}"
                )
            else:
                print("    [SKIP] fewer than 2 race groups")
        else:
            print(f"\n  [MIMIC | race] n={len(meta_race)} < {MIN_POSITIVE}, SKIPPED")

    return pd.DataFrame(all_rows)


# ---------------------------------------------------------------------------
# Step 3: generate cross-dataset comparison table
# ---------------------------------------------------------------------------
def build_comparison_table(df: pd.DataFrame) -> pd.DataFrame:
    """Build a summary table: rows = diagnosis, cols = dataset+attribute agreement."""
    rows = []
    for dx in df["diagnosis"].unique():
        row: dict[str, Any] = {"diagnosis": dx}

        for dataset, attr, col_name in [
            ("CheXpert", "sex", "chexpert_sex"),
            ("MIMIC-CXR", "sex", "mimic_sex"),
            ("MIMIC-CXR", "race", "mimic_race"),
        ]:
            sub = df[
                (df["diagnosis"] == dx) & (df["dataset"] == dataset) & (df["attribute"] == attr)
            ]
            if sub.empty:
                row[f"{col_name}_agreement"] = "N/A"
                row[f"{col_name}_flagged"] = "N/A"
                row[f"{col_name}_n"] = "N/A"
                row[f"{col_name}_convergence"] = "N/A"
            else:
                n_flagged = int(sub["flagged"].sum())
                n_total = len(sub)
                row[f"{col_name}_agreement"] = f"{n_flagged}/{n_total}"
                row[f"{col_name}_flagged"] = n_flagged
                row[f"{col_name}_n"] = int(sub["n_samples"].iloc[0])
                row[f"{col_name}_convergence"] = sub["convergence"].iloc[0]
        rows.append(row)

    return pd.DataFrame(rows)


# ---------------------------------------------------------------------------
# Step 4: generate LaTeX table
# ---------------------------------------------------------------------------
METHOD_SHORT = {
    "hbac": "H",
    "probe": "P",
    "statistical": "S",
    "geometric": "G",
    "frequency": "Fr",
    "bias_direction_pca": "BD",
    "sis": "SI",
    "demographic_parity": "DP",
    "equalized_odds": "EO",
    "intersectional": "In",
}


def generate_latex_table(df: pd.DataFrame, comp: pd.DataFrame) -> str:
    """Generate LaTeX cross-dataset comparison table."""
    n_m = len(METHODS)
    # Column spec: n + 10 method flags + Agr. = 12 columns per dataset group
    col_per_group = f"r{'c' * n_m}c"
    lines = []
    lines.append(r"\begin{table*}[ht]")
    lines.append(r"\centering")
    lines.append(r"\scriptsize")
    lines.append(
        r"\caption{{\color{blue}Cross-dataset per-diagnosis shortcut detection. "
        r"CheXpert (ResNet-50, 2048-dim) vs MIMIC-CXR (RAD-DINO, 768-dim). "
        f"For each diagnosis, we filter to positive cases and run {n_m} detection "
        r"methods. Agreement = methods flagging shortcut / total methods. "
        r"$\checkmark$ = flagged, $\cdot$ = not flagged.}}"
    )
    lines.append(r"\label{tab:cross_dataset_diagnosis}")
    mc = n_m + 2  # n + methods + Agr.
    lines.append(
        r"\begin{tabular}{l|" + col_per_group + "|" + col_per_group + "|" + col_per_group + "}"
    )
    lines.append(r"\hline")
    lines.append(
        rf" & \multicolumn{{{mc}}}{{c|}}{{\textbf{{CheXpert (sex)}}}} "
        rf"& \multicolumn{{{mc}}}{{c|}}{{\textbf{{MIMIC-CXR (sex)}}}} "
        rf"& \multicolumn{{{mc}}}{{c}}{{\textbf{{MIMIC-CXR (race)}}}} \\"
    )
    method_headers = " & ".join(METHOD_SHORT.get(m, m[:2]) for m in METHODS)
    lines.append(
        r"\textbf{Diagnosis} "
        f"& $n$ & {method_headers} & Agr. "
        f"& $n$ & {method_headers} & Agr. "
        f"& $n$ & {method_headers} & Agr. \\\\"
    )
    lines.append(r"\hline")

    for _, crow in comp.iterrows():
        dx = crow["diagnosis"]
        dx_display = dx.replace("&", r"\&")
        row_str = f"{dx_display}"

        for prefix in ["chexpert_sex", "mimic_sex", "mimic_race"]:
            agr = crow.get(f"{prefix}_agreement", "N/A")
            n_samp = crow.get(f"{prefix}_n", "N/A")

            if agr == "N/A":
                row_str += r" & -- & -- & -- & -- & -- & --"
                continue

            row_str += f" & {n_samp}"

            # Get per-method flags
            sub = df[
                (df["diagnosis"] == dx)
                & (df["dataset"] == ("CheXpert" if "chexpert" in prefix else "MIMIC-CXR"))
                & (df["attribute"] == ("sex" if "sex" in prefix else "race"))
            ]
            for m in METHODS:
                m_sub = sub[sub["method"] == m]
                if len(m_sub) > 0 and m_sub["flagged"].values[0]:
                    row_str += r" & \checkmark"
                else:
                    row_str += r" & $\cdot$"

            row_str += f" & {agr}"

        row_str += r" \\"
        lines.append(row_str)

    lines.append(r"\hline")
    lines.append(r"\end{tabular}")
    lines.append(r"\end{table*}")

    return "\n".join(lines)


# ---------------------------------------------------------------------------
# Step 5: print summary
# ---------------------------------------------------------------------------
def print_summary(comp: pd.DataFrame) -> None:
    """Print a clear human-readable summary."""
    print(f"\n{'='*75}")
    print("  CROSS-DATASET SHORTCUT CONSISTENCY SUMMARY")
    print(f"{'='*75}")
    print(f"  {'Diagnosis':30s} | {'CheXpert sex':14s} | {'MIMIC sex':14s} | {'MIMIC race':14s}")
    print(f"  {'-'*30}-+-{'-'*14}-+-{'-'*14}-+-{'-'*14}")

    for _, row in comp.iterrows():
        dx = row["diagnosis"]
        chx = row.get("chexpert_sex_agreement", "N/A")
        ms = row.get("mimic_sex_agreement", "N/A")
        mr = row.get("mimic_race_agreement", "N/A")

        chx_conv = row.get("chexpert_sex_convergence", "")
        ms_conv = row.get("mimic_sex_convergence", "")
        mr_conv = row.get("mimic_race_convergence", "")

        print(
            f"  {dx:30s} | {chx:5s} ({chx_conv[:4]:4s}) | {ms:5s} ({ms_conv[:4]:4s}) | {mr:5s} ({mr_conv[:4]:4s})"
        )

    # Consistency analysis
    print("\n  CONSISTENCY ANALYSIS:")
    print(f"  {'='*70}")
    for _, row in comp.iterrows():
        dx = row["diagnosis"]
        chx_f = row.get("chexpert_sex_flagged", 0)
        ms_f = row.get("mimic_sex_flagged", 0)
        mr_f = row.get("mimic_race_flagged", 0)

        if chx_f == "N/A" or ms_f == "N/A":
            continue

        chx_f = int(chx_f)
        ms_f = int(ms_f)

        # Same shortcut pattern across datasets?
        both_sex = chx_f >= 2 and ms_f >= 2
        neither_sex = chx_f <= 1 and ms_f <= 1

        if both_sex:
            verdict = "CONSISTENT SHORTCUT (sex detected in both datasets)"
        elif neither_sex:
            verdict = "CONSISTENT: no sex shortcut in either dataset"
        else:
            verdict = "INCONSISTENT: shortcut in one dataset but not the other"

        race_note = ""
        if mr_f != "N/A":
            mr_f = int(mr_f)
            if mr_f >= 2:
                race_note = " | Race shortcut also detected in MIMIC"
            else:
                race_note = " | No race shortcut in MIMIC"

        print(f"  {dx:30s} -> {verdict}{race_note}")

    print(f"{'='*75}")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main():
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    TABLE_DIR.mkdir(parents=True, exist_ok=True)

    print(f"{'='*65}")
    print("  Per-Diagnosis Cross-Dataset Shortcut Detection (L01 + L02)")
    print(f"{'='*65}")
    print(f"  CheXpert:  {CHEXPERT_DIR}")
    print(f"  MIMIC-CXR: {MIMIC_DIR}")
    print(f"  Methods:   {', '.join(METHODS)}")
    print(f"  Min cases: {MIN_POSITIVE}")
    print(f"  Top N:     {TOP_N}")
    print(f"{'='*65}")

    t_start = time.time()

    # Run analysis
    df = run_cross_dataset_analysis()

    elapsed = time.time() - t_start
    print(f"\nDetection completed in {elapsed:.1f}s")

    # Save raw results
    df.to_csv(OUTPUT_DIR / "cross_dataset_results.csv", index=False)
    print(f"Saved: {OUTPUT_DIR / 'cross_dataset_results.csv'}")

    # Build comparison table
    comp = build_comparison_table(df)
    comp.to_csv(OUTPUT_DIR / "cross_dataset_comparison.csv", index=False)
    print(f"Saved: {OUTPUT_DIR / 'cross_dataset_comparison.csv'}")

    # Generate LaTeX
    latex = generate_latex_table(df, comp)
    tex_path = TABLE_DIR / "table_cross_dataset_diagnosis.tex"
    tex_path.write_text(latex)
    print(f"Saved: {tex_path}")
    print("\n" + latex)

    # Summary JSON
    summary = {
        "benchmark": "cross_dataset_per_diagnosis",
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "datasets": {
            "chexpert": {"backbone": "resnet50", "dim": 2048},
            "mimic_cxr": {"backbone": "rad_dino", "dim": 768},
        },
        "methods": METHODS,
        "min_positive": MIN_POSITIVE,
        "top_n": TOP_N,
        "elapsed_seconds": round(elapsed, 2),
        "n_result_rows": len(df),
        "diagnoses_analyzed": comp["diagnosis"].tolist(),
    }
    with open(OUTPUT_DIR / "summary.json", "w") as f:
        json.dump(summary, f, indent=2)

    # Print summary
    print_summary(comp)

    print(f"\nTotal elapsed: {elapsed:.1f}s")
    print("DONE.")


if __name__ == "__main__":
    main()
