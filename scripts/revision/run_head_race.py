#!/usr/bin/env python3
"""Head experiment with race as the attribute (reviewer question: does the
audit-reliance relation hold for a third attribute?).

Reruns the reliance part of run_head_experiment.py (logistic head, same split
and subsampling code, same seeds, same probe audit per cell; no mitigations)
with A = race, White (0) against non-White (1), constant per patient. Images
without a recorded race are dropped before the split, as in the clinical audit.
A (cell, seed) draw that the smallest (A, Y) group cannot support at
rho_train = 0.9 is recorded with feasible = False and its n_train, not skipped
silently. Output columns match head_experiment/head_reliance.csv.
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
    Manifest,
    chexpert_paths,
    load_chexpert,
    results_root,
)
from run_head_experiment import (  # noqa: E402
    RHO_TRAIN,
    TASKS,
    attribute_vector,
    head,
    reliance,
    sample_cells,
)
from run_head_mlp import spearman_ci  # noqa: E402

ATTR = "race"


def run_cell(bb, task, seeds, k_audit, threads):
    import torch

    torch.set_num_threads(threads)
    from sklearn.model_selection import StratifiedGroupKFold

    from shortcut_detect.audit import AuditConfig, conditional_audit

    X, meta = load_chexpert(bb)
    col = TASKS[task]
    lab = meta[col].to_numpy(dtype=float)
    a_all = attribute_vector(meta, ATTR)
    keep = ((lab == 0) | (lab == 1)) & (a_all >= 0)
    n_dropped_uncertain = int((lab == -1).sum())
    n_dropped_blank = int(np.isnan(lab).sum())
    n_dropped_race = int((((lab == 0) | (lab == 1)) & (a_all < 0)).sum())
    rows_k = np.flatnonzero(keep)
    Xk, yk = X[rows_k], lab[rows_k].astype(int)
    ak = a_all[rows_k]
    pk = meta["patient_id"].to_numpy()[rows_k]
    info = dict(backbone=bb, attribute=ATTR, task=task, n_labelled=int(keep.sum()),
                n_dropped_uncertain=n_dropped_uncertain, n_dropped_blank=n_dropped_blank,
                n_patients=int(len(set(pk))))

    t0 = time.time()
    aud = conditional_audit(
        Xk, ak, yk,
        AuditConfig(n_permutations=200, n_permutations_probe=k_audit, n_bootstrap=0, detectors=("probe",), seed=42),
        groups=pk,
    ).results["probe"]
    audit = dict(probe_auroc=aud.statistic, probe_z=aud.z, probe_p=aud.p_value,
                 probe_null_mean=aud.null_mean, audit_seconds=round(time.time() - t0, 1))

    rel_rows = []
    for s in range(seeds):
        rng = np.random.default_rng(1000 * s + 7)
        key = ak * 2 + yk
        sgk = StratifiedGroupKFold(n_splits=2, shuffle=True, random_state=int(s))
        tr_pool, te_pool = next(sgk.split(Xk, key, pk))
        cells_tr = {(a, y): tr_pool[(ak[tr_pool] == a) & (yk[tr_pool] == y)] for a in (0, 1) for y in (0, 1)}
        cells_te = {(a, y): te_pool[(ak[te_pool] == a) & (yk[te_pool] == y)] for a in (0, 1) for y in (0, 1)}
        conf = {(1, 1): RHO_TRAIN / 2, (0, 0): RHO_TRAIN / 2, (1, 0): (1 - RHO_TRAIN) / 2, (0, 1): (1 - RHO_TRAIN) / 2}
        bal = dict.fromkeys(conf, 0.25)
        n_conf = min(len(cells_tr[c]) / f for c, f in conf.items())
        n_bal = min(len(cells_tr[c]) / f for c, f in bal.items())
        N = int(np.floor(min(n_conf, n_bal)))
        m_te = min(len(v) for v in cells_te.values())
        if N < 40 or m_te < 5:
            rel_rows.append({**info, **audit, "seed": s, "n_train": N, "n_test": 4 * m_te,
                             "feasible": False})
            continue
        i_conf = sample_cells(rng, cells_tr, conf, N)
        i_bal = sample_cells(rng, cells_tr, bal, N)
        i_te = np.concatenate([rng.choice(v, size=m_te, replace=False) for v in cells_te.values()])
        Xte, yte, ate = Xk[i_te], yk[i_te], ak[i_te]
        h_conf = head(Xk[i_conf], yk[i_conf], s)
        h_bal = head(Xk[i_bal], yk[i_bal], s)
        # A direction from the training pool, Y strata centred
        Xp_, ap_, yp_ = Xk[tr_pool], ak[tr_pool], yk[tr_pool]
        Xc = Xp_.copy().astype(np.float64)
        for v in (0, 1):
            Xc[yp_ == v] -= Xc[yp_ == v].mean(0)
        u = Xc[ap_ == 1].mean(0) - Xc[ap_ == 0].mean(0)
        u /= np.linalg.norm(u)
        rel = reliance(h_conf, Xte, yte, ate, h_bal, u, Xp_.mean(0))
        rel_rows.append({**info, **audit, "seed": s, "n_train": int(len(i_conf)),
                         "n_test": int(len(i_te)), "feasible": True, **rel})
    print(f"{bb} {ATTR} {task}: {time.time() - t0:.0f}s", flush=True)
    return rel_rows, n_dropped_race


def summarise(h: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    from scipy.stats import spearmanr

    key = ["backbone", "attribute", "task"]
    hf = h[h.feasible.astype(bool)]
    cell = hf.groupby(key).agg(probe_auroc=("probe_auroc", "first"), probe_p=("probe_p", "first"),
                               tpr_gap=("tpr_gap", "mean"), tpr_gap_sd=("tpr_gap", "std"),
                               auroc_drop=("auroc_drop", "mean"), n_train=("n_train", "mean"),
                               n_seeds=("seed", "nunique")).reset_index()
    x, y = cell.probe_auroc.to_numpy(), cell.tpr_gap.to_numpy()
    r, p = spearmanr(x, y)
    lo, hi = spearman_ci(x, y, 2000, 0)
    blo, bhi = spearman_ci(x, y, 2000, 1, groups=cell.backbone.to_numpy())
    n_cells_all = int(h.groupby(key).ngroups)
    out = [dict(stat="rho probe vs TPR gap, all cells", n=len(cell), value=r, p=p,
                ci_cells_lo=lo, ci_cells_hi=hi, ci_backbones_lo=blo, ci_backbones_hi=bhi),
           dict(stat="rho probe vs AUROC drop, all cells", n=len(cell), value=spearmanr(x, cell.auroc_drop)[0],
                p=spearmanr(x, cell.auroc_drop)[1]),
           dict(stat="mean tpr_gap", n=len(cell), value=cell.tpr_gap.mean()),
           dict(stat="mean auroc_drop", n=len(cell), value=cell.auroc_drop.mean()),
           dict(stat="probe_auroc min", n=len(cell), value=cell.probe_auroc.min()),
           dict(stat="probe_auroc max", n=len(cell), value=cell.probe_auroc.max()),
           dict(stat="n cells run", n=n_cells_all, value=n_cells_all),
           dict(stat="n cells with no feasible seed", n=n_cells_all, value=n_cells_all - len(cell)),
           dict(stat="n (cell, seed) draws skipped", n=len(h), value=int((~h.feasible.astype(bool)).sum()))]
    return cell, pd.DataFrame(out)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--seeds", type=int, default=10)
    ap.add_argument("--K", type=int, default=500)  # probe K, as in run_head_experiment.py
    ap.add_argument("--jobs", type=int, default=3)
    ap.add_argument("--threads", type=int, default=3)
    ap.add_argument("--backbones", nargs="+", default=CHEXPERT_BACKBONES)
    ap.add_argument("--tasks", nargs="+", default=list(TASKS))
    ap.add_argument("--out-name", default="head_race")
    args = ap.parse_args()
    from joblib import Parallel, delayed

    out = results_root() / args.out_name
    out.mkdir(parents=True, exist_ok=True)
    man = Manifest(out, "head reliance experiment with race as the attribute (reliance part only)")
    man.config = {"seeds": args.seeds, "K_audit": args.K, "rho_train": RHO_TRAIN, "rho_test": 0.5,
                  "head": "StandardScaler + LogisticRegression(C=1)", "threshold": 0.5, "tasks": args.tasks,
                  "attributes": ["race (non-White=1, White=0; missing race dropped)"],
                  "backbones": args.backbones,
                  "split": "patient-level StratifiedGroupKFold(2) on (A, Y), identical code to run_head_experiment.py",
                  "bootstrap": "2000 over cells (seed 0); 2000 over whole backbones (seed 1)"}
    man.notes += [CHEXPERT_JOIN_NOTE,
                  "n_labelled counts images with a 0/1 task label and a recorded race; images without a "
                  "recorded race are dropped before the audit and the split.",
                  "feasible = False rows: the (cell, seed) draw could not reach N >= 40 at rho_train = 0.9 "
                  "(or 5 test images per group); n_train holds the N the split allowed."]
    man.add_data(DATA / "train.csv", DATA / "chexpert_multibackbone" / "race_mapping.csv")
    for bb in args.backbones:
        man.add_data(chexpert_paths(bb)[0])
    jobs = [(bb, t) for bb in args.backbones for t in args.tasks]
    t0 = time.time()
    res = Parallel(n_jobs=args.jobs)(
        delayed(run_cell)(bb, t, args.seeds, args.K, args.threads) for bb, t in jobs
    )
    man.time("all_cells", time.time() - t0)
    h = pd.DataFrame([r for rr, _ in res for r in rr])
    h.to_csv(out / "head_reliance.csv", index=False)
    cell, summary = summarise(h)
    cell.to_csv(out / "cells_race.csv", index=False)
    summary.to_csv(out / "summary_race.csv", index=False)
    dropped = {t: n for (bb, t), (_, n) in zip(jobs, res, strict=True) if bb == args.backbones[0]}
    man.notes.append(f"labelled images dropped for missing race, per task: {dropped}")
    man.notes.append(f"(cell, seed) draws skipped at rho_train = {RHO_TRAIN}: "
                     f"{int((~h.feasible.astype(bool)).sum())} of {len(h)}")
    man.write()
    print(summary.to_string(index=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
