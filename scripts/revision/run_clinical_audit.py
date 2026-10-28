#!/usr/bin/env python3
"""Conditional audit on CheXpert (7 backbones), MIMIC-CXR (4) and CelebA.

One attribute at a time, attributes per the revision decisions (section 5):
sex, age 60+ vs under 60, White vs non-White, plus sensitivity variants
(age in 4 bins; race named groups kept with 50+ images per group or per
(group, Y) stratum). CheXpert and CelebA are patient- (identity-) grouped.
Saves continuous scores, null summaries, p-values, BH flags, bootstrap CIs
and group sizes for every cell.
"""

from __future__ import annotations

import argparse
import json
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
    celeba_cells,
    celeba_paths,
    chexpert_paths,
    load_celeba,
    load_chexpert,
    load_mimic,
    mimic_paths,
    results_root,
)

K = 200
K_PROBE = 500  # revision decisions, section 9
N_BOOT = 2000
SEED = 42
PROFILE_NOTE = (
    "Profile before choosing K (28 Oct, one cell, CheXpert ResNet-50 d=2048, sex, K=200, "
    "6 torch threads): whole cell 40 s; probe 9.3 s, frequency_topk 7.3 s, bias_direction 4.0 s, "
    "geometric 2.3 s, hbac_nmi 0.5 s, statistical 0.2 s. K for the probe therefore set to 200, "
    "the same as the vectorised statistics. Section 9 (after verification) sets K = 500 for the "
    "probe and 5 x 5 repeated grouped CV for the observed probe statistics; the same cell then "
    "takes 76 s (probe 18.7 s, frequency_topk 19.1 s)."
)


def run_cell(cell, k: int, k_probe: int, n_boot: int, seed: int, threads: int, plain_probe: bool):
    import torch

    torch.set_num_threads(threads)
    from shortcut_detect.audit import AuditConfig, conditional_audit

    base = {
        "dataset": cell.dataset,
        "backbone": cell.backbone,
        "task": cell.task,
        "attribute": cell.attribute,
        "variant": cell.variant,
        "n_images": int(len(cell.y)),
        "n_patients": int(len(set(cell.groups))) if cell.groups is not None else int(len(cell.y)),
        "group_sizes": json.dumps(cell.group_sizes()),
        "note": cell.note,
    }
    if len(np.unique(cell.a)) < 2:
        return [{**base, "detector": "n/a", "evaluated": False}]
    t0 = time.time()
    cfg = AuditConfig(n_permutations=k, n_permutations_probe=k_probe, n_bootstrap=n_boot, seed=seed)
    res = conditional_audit(cell.X, cell.a, cell.y, cfg, groups=cell.groups)
    rows = [{**base, **r, "evaluated": True} for r in res.rows()]
    if plain_probe and cell.groups is not None:
        cfg2 = AuditConfig(
            n_permutations=k_probe, n_bootstrap=n_boot, detectors=("probe",), seed=seed
        )
        r2 = conditional_audit(cell.X, cell.a, cell.y, cfg2, groups=None)
        rows += [{**base, **r, "detector": "probe_plain_cv", "evaluated": True} for r in r2.rows()]
    for r in rows:
        r["cell_seconds"] = round(time.time() - t0, 1)
    print(f"{cell.key}: {time.time() - t0:.0f}s, n_flag={res.n_flagged}", flush=True)
    return rows


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--datasets", nargs="+", default=["chexpert", "mimic", "celeba"])
    ap.add_argument("--jobs", type=int, default=3)
    ap.add_argument("--threads", type=int, default=3)
    ap.add_argument("--K", type=int, default=K)
    ap.add_argument("--K-probe", type=int, default=K_PROBE)
    ap.add_argument("--bootstrap", type=int, default=N_BOOT)
    ap.add_argument("--backbones", nargs="+", default=None)
    args = ap.parse_args()
    from joblib import Parallel, delayed

    out = results_root() / "clinical_audit"
    out.mkdir(parents=True, exist_ok=True)
    man = Manifest(out, "conditional audit, clinical and CelebA")
    man.config = {
        "K": args.K,
        "K_probe": args.K_probe,
        "n_bootstrap": args.bootstrap,
        "seed": SEED,
        "alpha": 0.05,
        "C": 1.0,
        "n_folds": 5,
        "top_k_fraction": 0.05,
        "n_clusters": 8,
        "permutation": "within Y; patient-level within Y-pattern strata when grouped",
        "datasets": args.datasets,
    }
    man.notes += [PROFILE_NOTE, CHEXPERT_JOIN_NOTE]
    cells = []
    if "chexpert" in args.datasets:
        man.add_data(DATA / "train.csv", DATA / "chexpert_multibackbone" / "race_mapping.csv",
                     DATA / "chexpert_multibackbone" / "chexpert_manifest.csv")
        for bb in args.backbones or CHEXPERT_BACKBONES:
            if bb not in CHEXPERT_BACKBONES:
                continue
            X, meta = load_chexpert(bb)
            man.add_data(chexpert_paths(bb)[0])
            cells += [(c, c.attribute == "sex" and c.variant == "primary")
                      for c in attribute_cells("chexpert", bb, X, meta)]
    if "mimic" in args.datasets:
        for bb in args.backbones or MIMIC_BACKBONES:
            if bb not in MIMIC_BACKBONES:
                continue
            X, meta = load_mimic(bb)
            man.add_data(*mimic_paths(bb))
            cells += [(c, False) for c in attribute_cells("mimic", bb, X, meta)]
    if "celeba" in args.datasets:
        X, attrs = load_celeba()
        man.add_data(*celeba_paths(), DATA / "celeba_real" / "celeba" / "identity_CelebA.txt")
        cells += [(c, False) for c in celeba_cells(X, attrs)]
    # heaviest first so the pool stays busy
    cells.sort(key=lambda t: -t[0].X.shape[0] * t[0].X.shape[1])
    t0 = time.time()
    results = Parallel(n_jobs=args.jobs, verbose=0)(
        delayed(run_cell)(c, args.K, args.K_probe, args.bootstrap, SEED, args.threads, plain)
        for c, plain in cells
    )
    rows = [r for part in results for r in part]
    df = pd.DataFrame(rows)
    path = out / "audit_rows.csv"
    if path.exists():  # keep cells from other datasets of an earlier partial run
        old = pd.read_csv(path)
        old = old[~old["dataset"].isin(args.datasets)]
        df = pd.concat([old, df], ignore_index=True)
    df.to_csv(path, index=False)
    man.time("all_cells", time.time() - t0)
    man.write()
    print(f"wrote {path} ({len(df)} rows) in {time.time() - t0:.0f}s")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
