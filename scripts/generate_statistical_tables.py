#!/usr/bin/env python3
"""Generate paper tables with bootstrap CIs and permutation p-values.

Produces:
  output/paper_tables/table_permutation_pvalues.tex   (Table A)
  output/paper_tables/table_demographics_effects.tex  (Table B)
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats as sp_stats

# Ensure project root is importable
PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from shortcut_detect.benchmark.measurement import (  # noqa: E402
    bootstrap_ci,
)

OUTPUT_DIR = PROJECT_ROOT / "output" / "paper_tables"
DATA_DIR = PROJECT_ROOT / "data"

# Maximum number of samples for the permutation test (subsample if larger)
_PERM_MAX_N = 2000


def _fast_probe_permutation(emb, labels, n_permutations=100, seed=42):
    """Fast permutation test using liblinear solver and subsampling."""
    from sklearn.linear_model import LogisticRegression
    from sklearn.model_selection import cross_val_score

    n = len(labels)
    rng = np.random.RandomState(seed)

    if n > _PERM_MAX_N:
        idx = rng.choice(n, _PERM_MAX_N, replace=False)
        X = emb[idx]
        y = labels[idx]
        print(f"    (subsampled {n} -> {_PERM_MAX_N} for permutation test)")
    else:
        X = np.asarray(emb, dtype=float)
        y = np.asarray(labels).ravel()

    # Use liblinear for speed (much faster than lbfgs on moderate-dim data)
    clf = LogisticRegression(max_iter=1000, solver="liblinear", random_state=seed)
    observed_accuracy = float(np.mean(cross_val_score(clf, X, y, cv=3, scoring="accuracy")))

    null_accuracies = np.empty(n_permutations, dtype=float)
    for i in range(n_permutations):
        y_perm = rng.permutation(y)
        null_accuracies[i] = float(
            np.mean(cross_val_score(clf, X, y_perm, cv=3, scoring="accuracy"))
        )
        if (i + 1) % 20 == 0:
            print(f"    permutation {i + 1}/{n_permutations}")

    p_value = float(np.mean(null_accuracies >= observed_accuracy))
    return {
        "observed_accuracy": observed_accuracy,
        "null_mean": float(np.mean(null_accuracies)),
        "null_std": float(np.std(null_accuracies)),
        "p_value": p_value,
        "n_permutations": n_permutations,
    }


# ---------------------------------------------------------------------------
# Dataset definitions
# ---------------------------------------------------------------------------


def _load_chexpert_race(backbone: str = "resnet50"):
    """CheXpert: WHITE vs non-WHITE binary probe (multi-backbone)."""
    mb_dir = DATA_DIR / "chexpert_multibackbone"
    emb = np.load(mb_dir / f"{backbone}_embeddings.npy")
    meta = pd.read_csv(mb_dir / f"{backbone}_metadata.csv")
    labels = (meta["race"] == "WHITE").astype(int).values  # 1=WHITE, 0=non-WHITE
    return emb, labels


def _load_chexpert_sex(backbone: str = "resnet50"):
    """CheXpert: Male vs Female (drop Unknown, multi-backbone)."""
    mb_dir = DATA_DIR / "chexpert_multibackbone"
    emb = np.load(mb_dir / f"{backbone}_embeddings.npy")
    meta = pd.read_csv(mb_dir / f"{backbone}_metadata.csv")
    mask = meta["sex"].isin(["Male", "Female"])
    labels = (meta.loc[mask, "sex"] == "Male").astype(int).values
    return emb[mask.values], labels


def _load_chexpert_age(backbone: str = "resnet50"):
    """CheXpert: Age >= 60 vs Age < 60 binary probe (multi-backbone)."""
    mb_dir = DATA_DIR / "chexpert_multibackbone"
    emb = np.load(mb_dir / f"{backbone}_embeddings.npy")
    meta = pd.read_csv(mb_dir / f"{backbone}_metadata.csv")
    mask = meta["age"].notna()
    labels = (meta.loc[mask, "age"] >= 60).astype(int).values  # 1=>=60, 0=<60
    return emb[mask.values], labels


def _load_mimic_age(backbone: str = "rad_dino"):
    """MIMIC-CXR: Age >= 60 vs Age < 60 binary probe."""
    emb_path = DATA_DIR / "mimic_cxr" / f"{backbone}_embeddings.npy"
    meta_path = DATA_DIR / "mimic_cxr" / f"{backbone}_metadata.csv"
    if not emb_path.exists() or not meta_path.exists():
        print(f"    [SKIP] MIMIC {backbone} data not found locally")
        return None, None
    emb = np.load(emb_path)
    meta = pd.read_csv(meta_path)
    if "age" not in meta.columns:
        print(f"    [SKIP] MIMIC {backbone} metadata has no 'age' column")
        return None, None
    mask = meta["age"].notna()
    labels = (meta.loc[mask, "age"] >= 60).astype(int).values
    return emb[mask.values], labels


# CheXpert backbones to include in Tables 12/13 (matching Table 11)
_CHEXPERT_BACKBONES = [
    ("resnet50", "ResNet-50"),
    ("densenet121", "DenseNet"),
    ("vit_b_16", "ViT-B/16"),
]


def _load_mimic(backbone: str, attribute: str):
    """MIMIC-CXR embeddings for a given backbone and attribute."""
    emb_path = DATA_DIR / "mimic_cxr" / f"{backbone}_embeddings.npy"
    meta_path = DATA_DIR / "mimic_cxr" / f"{backbone}_metadata.csv"
    if not emb_path.exists() or not meta_path.exists():
        print(f"    [SKIP] MIMIC {backbone} data not found locally")
        return None, None
    emb = np.load(emb_path)
    meta = pd.read_csv(meta_path)
    if attribute == "sex":
        labels = (meta["sex"] == "Male").astype(int).values
    elif attribute == "race":
        # Binary: WHITE vs non-WHITE (placeholder if race not available)
        if "race" in meta.columns:
            labels = (meta["race"].str.upper() == "WHITE").astype(int).values
        else:
            return None, None
    else:
        return None, None
    return emb, labels


def _load_celeba(shortcut_attr: str):
    """CelebA: Male as group label."""
    emb = np.load(DATA_DIR / "celeba" / "celeba_real_embeddings.npy")
    meta = pd.read_csv(DATA_DIR / "celeba" / "celeba_real_metadata.csv")
    group_labels = meta["Male"].values
    return emb, group_labels, shortcut_attr


# ---------------------------------------------------------------------------
# Table A: Probe Permutation P-values
# ---------------------------------------------------------------------------


def generate_table_a():
    """Generate permutation p-value table."""
    print("=" * 60)
    print("Table A: Probe Permutation P-values")
    print("=" * 60)

    rows = []

    # --- CheXpert (per-backbone, matching Table 11) ---
    for bb_key, bb_label in _CHEXPERT_BACKBONES:
        for attr in ["Race", "Sex", "Age"]:
            print(f"  CheXpert ({bb_label}) / {attr} ...")
            if attr == "Race":
                emb, labels = _load_chexpert_race(bb_key)
            elif attr == "Sex":
                emb, labels = _load_chexpert_sex(bb_key)
            else:
                emb, labels = _load_chexpert_age(bb_key)
            res = _fast_probe_permutation(emb, labels, n_permutations=100, seed=42)
            rows.append(
                {
                    "Dataset": f"CheXpert ({bb_label})",
                    "Attribute": attr,
                    "Observed Acc": res["observed_accuracy"],
                    "Null Mean": res["null_mean"],
                    "Null Std": res["null_std"],
                    "p-value": res["p_value"],
                }
            )
            print(
                f"    obs={res['observed_accuracy']:.4f}  null={res['null_mean']:.4f}+/-{res['null_std']:.4f}  p={res['p_value']:.4f}"
            )

    # --- MIMIC-CXR ---
    for backbone, bb_label in [
        ("rad_dino", "RAD-DINO"),
        ("vit16_cls", "ViT-16"),
        ("medsiglip", "MedSigLIP"),
    ]:
        for attr in ["race", "sex", "age"]:
            print(f"  MIMIC / {bb_label} / {attr} ...")
            if attr == "race":
                emb, labels = _load_mimic(backbone, "race")
            elif attr == "sex":
                emb, labels = _load_mimic(backbone, "sex")
            else:
                emb, labels = _load_mimic_age(backbone)
            if emb is None:
                continue
            res = _fast_probe_permutation(emb, labels, n_permutations=100, seed=42)
            rows.append(
                {
                    "Dataset": f"MIMIC ({bb_label})",
                    "Attribute": attr.capitalize(),
                    "Observed Acc": res["observed_accuracy"],
                    "Null Mean": res["null_mean"],
                    "Null Std": res["null_std"],
                    "p-value": res["p_value"],
                }
            )
            print(
                f"    obs={res['observed_accuracy']:.4f}  null={res['null_mean']:.4f}+/-{res['null_std']:.4f}  p={res['p_value']:.4f}"
            )

    # --- CelebA ---
    for attr in ["Blond_Hair", "Heavy_Makeup", "Attractive"]:
        print(f"  CelebA / Male vs {attr} ...")
        emb = np.load(DATA_DIR / "celeba" / "celeba_real_embeddings.npy")
        meta = pd.read_csv(DATA_DIR / "celeba" / "celeba_real_metadata.csv")
        labels = meta["Male"].values
        res = _fast_probe_permutation(emb, labels, n_permutations=100, seed=42)
        rows.append(
            {
                "Dataset": "CelebA",
                "Attribute": f"{attr} $\\leftrightarrow$ Male",
                "Observed Acc": res["observed_accuracy"],
                "Null Mean": res["null_mean"],
                "Null Std": res["null_std"],
                "p-value": res["p_value"],
            }
        )
        print(
            f"    obs={res['observed_accuracy']:.4f}  null={res['null_mean']:.4f}+/-{res['null_std']:.4f}  p={res['p_value']:.4f}"
        )
        break  # Male probe is the same for all shortcut pairs; report once

    df = pd.DataFrame(rows)

    # Format LaTeX table
    lines = []
    lines.append(r"\begin{table}[ht]")
    lines.append(r"\centering")
    lines.append(r"\scriptsize")
    lines.append(r"\renewcommand{\arraystretch}{1.2}")
    lines.append(
        r"\caption{Probe permutation test results ($n_{\mathrm{perm}}{=}100$, seed 42). "
        r"A logistic regression probe is trained on true group labels and on 100 "
        r"label-shuffled permutations. The empirical $p$-value is the fraction of "
        r"null accuracies $\geq$ the observed accuracy. Values $p{<}0.01$ indicate "
        r"the embedding encodes group membership beyond chance.}"
    )
    lines.append(r"\label{tab:permutation_pvalues}")
    lines.append(r"\begin{tabular}{llcccc}")
    lines.append(r"\hline")
    lines.append(
        r"\textbf{Dataset} & \textbf{Attribute} & \textbf{Obs.\ Acc} & "
        r"\textbf{Null Mean $\pm$ Std} & \textbf{$p$-value} \\"
    )
    lines.append(r"\hline")

    for _, row in df.iterrows():
        p_str = f"{row['p-value']:.2f}" if row["p-value"] >= 0.01 else "$<$0.01"
        null_str = f"{row['Null Mean']:.3f} $\\pm$ {row['Null Std']:.3f}"
        lines.append(
            f"{row['Dataset']} & {row['Attribute']} & "
            f"{row['Observed Acc']:.3f} & {null_str} & {p_str} \\\\"
        )

    lines.append(r"\hline")
    lines.append(r"\end{tabular}")
    lines.append(r"\end{table}")

    tex = "\n".join(lines)
    out_path = OUTPUT_DIR / "table_permutation_pvalues.tex"
    out_path.write_text(tex)
    print(f"\n  Saved: {out_path}")
    return tex


# ---------------------------------------------------------------------------
# Table B: Clinical Dataset Demographics & Effect Sizes
# ---------------------------------------------------------------------------


def _cohens_d(emb, labels):
    """Compute geometric Cohen's d between groups (mean across dimensions)."""
    g0 = emb[labels == 0]
    g1 = emb[labels == 1]
    n0, n1 = g0.shape[0], g1.shape[0]
    if n0 < 2 or n1 < 2:
        return float("nan")
    mean0 = np.mean(g0, axis=0)
    mean1 = np.mean(g1, axis=0)
    var0 = np.var(g0, axis=0, ddof=1)
    var1 = np.var(g1, axis=0, ddof=1)
    pooled_var = ((n0 - 1) * var0 + (n1 - 1) * var1) / (n0 + n1 - 2)
    pooled_std = np.sqrt(pooled_var)
    # Per-dimension Cohen's d, then average magnitude
    d_per_dim = np.abs(mean0 - mean1) / np.where(pooled_std > 0, pooled_std, 1.0)
    return float(np.mean(d_per_dim))


def _probe_accuracy_bootstrap(emb, labels, n_bootstrap=2000, seed=42):
    """Probe accuracy with bootstrap CI via cross-val scores."""
    from sklearn.linear_model import LogisticRegression
    from sklearn.model_selection import cross_val_score

    clf = LogisticRegression(max_iter=1000, solver="liblinear", random_state=seed)
    scores = cross_val_score(clf, emb, labels, cv=5, scoring="accuracy")
    ci = bootstrap_ci(scores, n_bootstrap=n_bootstrap, seed=seed)
    return ci


def _count_fdr_significant(emb, labels, alpha=0.05):
    """Count dimensions significant after FDR-BH correction (t-test)."""
    from statsmodels.stats.multitest import multipletests

    g0 = emb[labels == 0]
    g1 = emb[labels == 1]
    pvals = np.array(
        [
            sp_stats.ttest_ind(g0[:, j], g1[:, j], equal_var=False).pvalue
            for j in range(emb.shape[1])
        ]
    )
    reject, _, _, _ = multipletests(pvals, alpha=alpha, method="fdr_bh")
    return int(np.sum(reject))


def generate_table_b():
    """Generate demographics and effect sizes table."""
    print("\n" + "=" * 60)
    print("Table B: Clinical Dataset Demographics & Effect Sizes")
    print("=" * 60)

    rows = []

    # --- CheXpert (per-backbone, matching Table 11) ---
    for bb_key, bb_label in _CHEXPERT_BACKBONES:
        for attr in ["Race", "Sex", "Age"]:
            print(f"  CheXpert ({bb_label}) / {attr} ...")
            if attr == "Race":
                emb, labels = _load_chexpert_race(bb_key)
            elif attr == "Sex":
                emb, labels = _load_chexpert_sex(bb_key)
            else:
                emb, labels = _load_chexpert_age(bb_key)
            ci = _probe_accuracy_bootstrap(emb, labels)
            d = _cohens_d(emb, labels)
            n_sig = _count_fdr_significant(emb, labels)
            n0, n1 = int(np.sum(labels == 0)), int(np.sum(labels == 1))
            rows.append(
                {
                    "Dataset": f"CheXpert ({bb_label})",
                    "Attribute": attr,
                    "N": len(labels),
                    "N_g0": n0,
                    "N_g1": n1,
                    "Emb Dim": emb.shape[1],
                    "Cohen_d": d,
                    "Probe Mean": ci["mean"],
                    "CI Lower": ci["ci_lower"],
                    "CI Upper": ci["ci_upper"],
                    "FDR Sig": n_sig,
                }
            )
            print(
                f"    N={len(labels)}, d={d:.3f}, acc={ci['mean']:.3f} [{ci['ci_lower']:.3f}, {ci['ci_upper']:.3f}], sig={n_sig}"
            )

    # --- MIMIC-CXR ---
    for backbone, bb_label in [
        ("rad_dino", "RAD-DINO"),
        ("vit16_cls", "ViT-16"),
        ("medsiglip", "MedSigLIP"),
    ]:
        for attr, attr_label in [("race", "Race"), ("sex", "Sex"), ("age", "Age")]:
            print(f"  MIMIC / {bb_label} / {attr_label} ...")
            if attr == "race":
                emb, labels = _load_mimic(backbone, "race")
            elif attr == "sex":
                emb, labels = _load_mimic(backbone, "sex")
            else:
                emb, labels = _load_mimic_age(backbone)
            if emb is None:
                continue
            ci = _probe_accuracy_bootstrap(emb, labels)
            d = _cohens_d(emb, labels)
            n_sig = _count_fdr_significant(emb, labels)
            n0, n1 = int(np.sum(labels == 0)), int(np.sum(labels == 1))
            rows.append(
                {
                    "Dataset": f"MIMIC ({bb_label})",
                    "Attribute": attr_label,
                    "N": len(labels),
                    "N_g0": n0,
                    "N_g1": n1,
                    "Emb Dim": emb.shape[1],
                    "Cohen_d": d,
                    "Probe Mean": ci["mean"],
                    "CI Lower": ci["ci_lower"],
                    "CI Upper": ci["ci_upper"],
                    "FDR Sig": n_sig,
                }
            )
            print(
                f"    N={len(labels)}, d={d:.3f}, acc={ci['mean']:.3f} [{ci['ci_lower']:.3f}, {ci['ci_upper']:.3f}], sig={n_sig}"
            )

    # --- CelebA ---
    print("  CelebA / Male ...")
    emb = np.load(DATA_DIR / "celeba" / "celeba_real_embeddings.npy")
    meta = pd.read_csv(DATA_DIR / "celeba" / "celeba_real_metadata.csv")
    labels = meta["Male"].values
    ci = _probe_accuracy_bootstrap(emb, labels)
    d = _cohens_d(emb, labels)
    n_sig = _count_fdr_significant(emb, labels)
    n0, n1 = int(np.sum(labels == 0)), int(np.sum(labels == 1))
    rows.append(
        {
            "Dataset": "CelebA",
            "Attribute": "Male",
            "N": len(labels),
            "N_g0": n0,
            "N_g1": n1,
            "Emb Dim": emb.shape[1],
            "Cohen_d": d,
            "Probe Mean": ci["mean"],
            "CI Lower": ci["ci_lower"],
            "CI Upper": ci["ci_upper"],
            "FDR Sig": n_sig,
        }
    )
    print(
        f"    N={len(labels)}, d={d:.3f}, acc={ci['mean']:.3f} [{ci['ci_lower']:.3f}, {ci['ci_upper']:.3f}], sig={n_sig}"
    )

    df = pd.DataFrame(rows)

    # Format LaTeX table
    lines = []
    lines.append(r"\begin{table*}[ht]")
    lines.append(r"\centering")
    lines.append(r"\scriptsize")
    lines.append(r"\renewcommand{\arraystretch}{1.2}")
    lines.append(
        r"\caption{Clinical dataset demographics, geometric effect sizes, and "
        r"probe accuracy with 95\% bootstrap confidence intervals. "
        r"Cohen's $d$ is the mean absolute per-dimension standardized difference "
        r"between group centroids. FDR-BH Sig.\ reports the number of embedding "
        r"dimensions with significant group differences after Benjamini--Hochberg "
        r"correction ($\alpha{=}0.05$). Bootstrap CIs use 2{,}000 resamples of "
        r"5-fold cross-validation accuracy scores.}"
    )
    lines.append(r"\label{tab:demographics_effects}")
    lines.append(r"\begin{tabular}{llccccccc}")
    lines.append(r"\hline")
    lines.append(
        r"\textbf{Dataset} & \textbf{Attr.} & \textbf{$N$} & "
        r"\textbf{$N_{g_0}$/$N_{g_1}$} & \textbf{Emb.\ Dim} & "
        r"\textbf{Cohen's $d$} & \textbf{Probe Acc (95\% CI)} & "
        r"\textbf{FDR-BH Sig.} \\"
    )
    lines.append(r"\hline")

    for _, row in df.iterrows():
        group_str = f"{row['N_g0']}/{row['N_g1']}"
        ci_str = f"{row['Probe Mean']:.3f} " f"[{row['CI Lower']:.3f}, {row['CI Upper']:.3f}]"
        lines.append(
            f"{row['Dataset']} & {row['Attribute']} & "
            f"{row['N']:,} & {group_str} & {row['Emb Dim']} & "
            f"{row['Cohen_d']:.3f} & {ci_str} & "
            f"{row['FDR Sig']}/{row['Emb Dim']} \\\\"
        )

    lines.append(r"\hline")
    lines.append(r"\end{tabular}")
    lines.append(r"\end{table*}")

    tex = "\n".join(lines)
    out_path = OUTPUT_DIR / "table_demographics_effects.tex"
    out_path.write_text(tex)
    print(f"\n  Saved: {out_path}")
    return tex


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    print(f"Output directory: {OUTPUT_DIR}\n")

    generate_table_a()
    generate_table_b()

    print("\nDone. Tables written to:")
    print(f"  {OUTPUT_DIR / 'table_permutation_pvalues.tex'}")
    print(f"  {OUTPUT_DIR / 'table_demographics_effects.tex'}")
