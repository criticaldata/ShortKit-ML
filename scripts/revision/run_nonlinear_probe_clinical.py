#!/usr/bin/env python3
"""Squared-feature conditional probe on the clinical cohorts.

For every primary cell of the clinical audit (CheXpert 7 backbones x {sex, age,
race}; MIMIC-CXR 4 backbones x {sex, age, race}; CelebA left out) the conditional
probe is fitted on [E, E^2] through ``conditional_audit`` with
``detectors=("probe",)``, the audit's seed and K. The permutation rng, the
patient-grouped folds and the CV rng are spawned from the seed independently of
the detector set, and ``k_max`` is K either way, so the null draws and the folds
are the audit's own (``--self-check`` verifies this by refitting [E] on one cell
and comparing with clinical_audit/audit_rows.csv). Standardisation is in-fold,
through the probe's penalty weights, as in the audit.

The pairwise-product design of the synthetic check ([E, E^2, E_i E_j]) is not
run here: at d = 2048 it has d(d-1)/2 = 2,096,128 product columns, about
16 GB of float32 for 2,000 images, before the permutation batches.

Output: <results>/nonlinear_probe_clinical/rows.csv with the linear probe's
AUROC and flag taken from clinical_audit/audit_rows.csv (flag recomputed as
p < 0.05, the primary rule), the squared probe's AUROC, p and flag.
"""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
from common import (  # noqa: E402
    CHEXPERT_BACKBONES,
    CHEXPERT_JOIN_NOTE,
    DATA,
    MIMIC_BACKBONES,
    Manifest,
    attribute_cells,
    chexpert_paths,
    load_chexpert,
    load_mimic,
    mimic_paths,
    results_root,
)

K_PROBE = 500  # clinical audit, K for the probe
SEED = 42  # clinical audit seed
ALPHA = 0.05
PRODUCT_NOTE = (
    "[E, E^2, E_i E_j] not run: at d = 2048 the product block has 2,096,128 columns "
    "(about 16 GB of float32 for 2,000 images before the permutation batches)."
)


def probe(cell, X: np.ndarray, k: int, threads: int) -> dict:
    import torch

    torch.set_num_threads(threads)
    from shortcut_detect.audit import AuditConfig, conditional_audit

    cfg = AuditConfig(n_permutations=k, n_bootstrap=0, detectors=("probe",), seed=SEED)
    res = conditional_audit(X, cell.a, cell.y, cfg, groups=cell.groups)
    m = res.results["probe"]
    return {
        "statistic": m.statistic, "null_mean": m.null_mean, "null_sd": m.null_sd, "z": m.z,
        "p_value": m.p_value, "n_permutations": m.n_permutations, "repeat_sd": m.repeat_sd,
    }


def run_cell(cell, k: int, threads: int, self_check: bool) -> dict:
    t0 = time.time()
    base = {
        "dataset": cell.dataset, "encoder": cell.backbone, "attribute": cell.attribute,
        "task": cell.task, "n_images": int(len(cell.y)),
        "n_patients": int(len(set(cell.groups))) if cell.groups is not None else int(len(cell.y)),
        "d": int(cell.X.shape[1]),
    }
    if len(np.unique(cell.a)) < 2:
        return {**base, "evaluated": False}
    E = np.asarray(cell.X, dtype=np.float32)
    out = {**base, "n_features_sq": int(2 * E.shape[1]), "evaluated": True}
    if self_check:
        lin = probe(cell, E, k, threads)
        out["probe_auroc_linear_refit"] = lin["statistic"]
        out["p_linear_refit"] = lin["p_value"]
    sq = probe(cell, np.concatenate([E, E * E], axis=1), k, threads)
    out.update(
        probe_auroc_sq=sq["statistic"], p_sq=sq["p_value"], null_mean_sq=sq["null_mean"],
        null_sd_sq=sq["null_sd"], z_sq=sq["z"], repeat_sd_sq=sq["repeat_sd"],
        n_permutations=sq["n_permutations"], flagged_sq=bool(sq["p_value"] < ALPHA),
        seconds=round(time.time() - t0, 1),
    )
    print(f"{cell.key}: {out['seconds']:.0f}s, AUROC sq={sq['statistic']:.3f}, p={sq['p_value']:.4f}", flush=True)
    return out


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--datasets", nargs="+", default=["chexpert", "mimic"])
    ap.add_argument("--backbones", nargs="+", default=None)
    ap.add_argument("--attributes", nargs="+", default=None)
    ap.add_argument("--K", type=int, default=K_PROBE)
    ap.add_argument("--jobs", type=int, default=3)
    ap.add_argument("--threads", type=int, default=3)
    ap.add_argument("--self-check", action="store_true",
                    help="also refit the linear probe on every selected cell and compare with the audit")
    args = ap.parse_args()
    from joblib import Parallel, delayed

    root = results_root()
    out = root / "nonlinear_probe_clinical"
    out.mkdir(parents=True, exist_ok=True)
    man = Manifest(out, "squared-feature conditional probe on the clinical cohorts")
    man.config = {
        "K": args.K, "seed": SEED, "alpha": ALPHA, "C": 1.0, "n_folds": 5, "cv_repeats": 5,
        "design": "[E, E^2], standardised in-fold through the probe's penalty weights",
        "null": "A permuted within Y; patient-level within Y-pattern strata when grouped (the clinical audit's draws)",
        "datasets": args.datasets, "self_check": args.self_check,
    }
    man.notes += [PRODUCT_NOTE, CHEXPERT_JOIN_NOTE]
    if args.K != K_PROBE:
        man.notes.append(f"K = {args.K} differs from the clinical audit's {K_PROBE}: the permutation draws are "
                         "no longer the audit's own set (same seed, shorter stream).")
    cells = []
    if "chexpert" in args.datasets:
        man.add_data(DATA / "train.csv", DATA / "chexpert_multibackbone" / "race_mapping.csv",
                     DATA / "chexpert_multibackbone" / "chexpert_manifest.csv")
        for bb in args.backbones or CHEXPERT_BACKBONES:
            if bb not in CHEXPERT_BACKBONES:
                continue
            X, meta = load_chexpert(bb)
            man.add_data(chexpert_paths(bb)[0])
            cells += attribute_cells("chexpert", bb, X, meta, sensitivity=False)
    if "mimic" in args.datasets:
        for bb in args.backbones or MIMIC_BACKBONES:
            if bb not in MIMIC_BACKBONES:
                continue
            X, meta = load_mimic(bb)
            man.add_data(*mimic_paths(bb))
            cells += attribute_cells("mimic", bb, X, meta, sensitivity=False)
    if args.attributes:
        cells = [c for c in cells if c.attribute in args.attributes]
    cells.sort(key=lambda c: -c.X.shape[0] * c.X.shape[1])  # heaviest first
    t0 = time.time()
    rows = Parallel(n_jobs=args.jobs, verbose=0)(
        delayed(run_cell)(c, args.K, args.threads, args.self_check) for c in cells
    )
    man.time("all_cells", time.time() - t0)
    df = pd.DataFrame(rows)

    # linear probe of the clinical audit, primary cells, flag at p < alpha (the primary rule)
    aud = pd.read_csv(root / "clinical_audit" / "audit_rows.csv")
    aud = aud[(aud.detector == "probe") & (aud.variant == "primary") & aud.dataset.isin(args.datasets)]
    aud = aud.rename(columns={"backbone": "encoder", "statistic": "probe_auroc_linear", "p_value": "p_linear"})
    aud = aud[["dataset", "encoder", "attribute", "probe_auroc_linear", "p_linear"]].copy()
    aud["flagged_linear"] = aud.p_linear < ALPHA
    df = df.merge(aud, on=["dataset", "encoder", "attribute"], how="left", validate="one_to_one")
    if df.probe_auroc_linear.isna().any():
        missing = df[df.probe_auroc_linear.isna()][["dataset", "encoder", "attribute"]]
        raise SystemExit(f"cells without a linear audit row:\n{missing}")
    if args.self_check:
        diff = (df.probe_auroc_linear_refit - df.probe_auroc_linear).abs()
        man.notes.append(
            f"self-check ({int(diff.notna().sum())} cells): max |AUROC refit [E] - audit| = {diff.max():.2e}, "
            f"max |p refit - audit| = {(df.p_linear_refit - df.p_linear).abs().max():.2e}"
        )
    df["decision_changed"] = df.flagged_linear != df.flagged_sq
    df["auroc_gain"] = df.probe_auroc_sq - df.probe_auroc_linear
    front = ["dataset", "encoder", "attribute", "probe_auroc_linear", "probe_auroc_sq", "p_sq",
             "flagged_linear", "flagged_sq", "decision_changed", "auroc_gain"]
    df = df[front + [c for c in df.columns if c not in front]]
    path = out / "rows.csv"
    if path.exists():  # keep cells from datasets of an earlier partial run
        old = pd.read_csv(path)
        old = old[~old["dataset"].isin(args.datasets)]
        df = pd.concat([old, df], ignore_index=True)
    df.to_csv(path, index=False)
    man.notes.append(
        f"cells {len(df)}; flagged sq {int(df.flagged_sq.sum())}; decisions changed "
        f"{int(df.decision_changed.sum())}; max AUROC gain {df.auroc_gain.max():.4f}; "
        f"mean seconds per cell {df.seconds.mean():.1f}"
    )
    man.write()
    print(f"wrote {path} ({len(df)} rows) in {time.time() - t0:.0f}s")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
