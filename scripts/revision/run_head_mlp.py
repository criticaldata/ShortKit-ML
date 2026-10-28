#!/usr/bin/env python3
"""Head experiment with an MLP head (reviewer question: does the finding hold
beyond a logistic head?).

Reruns only the reliance part of run_head_experiment.py (no mitigations, no
audit) with the head replaced by StandardScaler + MLPClassifier(256 hidden,
alpha = 1e-3, max_iter = 500, no early stopping, random_state = seed). Splits,
seeds and subsampling are drawn exactly as in run_head_experiment.py, so every
(cell, seed) is paired with the logistic run; the logistic head is refitted on
the same draw as a pairing check. The probe AUROC per cell is taken from the
logistic results (LOGISTIC_DIR/head_reliance.csv), since the
probe does not depend on the head.
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
    reliance,
    sample_cells,
)
from run_head_experiment import head as logistic_head  # noqa: E402

LOGISTIC_DIR = "head_experiment"  # logistic run whose probe values and splits this pairs with
CELL_COLS = ["n_labelled", "n_dropped_uncertain", "n_dropped_blank", "n_patients",
             "probe_auroc", "probe_z", "probe_p", "probe_null_mean"]  # head-independent
MLP = dict(hidden_layer_sizes=(256,), alpha=1e-3, early_stopping=False)


def mlp_head(Xtr, ytr, seed, max_iter):
    from sklearn.neural_network import MLPClassifier
    from sklearn.pipeline import make_pipeline
    from sklearn.preprocessing import StandardScaler

    return make_pipeline(StandardScaler(),
                         MLPClassifier(**MLP, max_iter=max_iter, random_state=seed)).fit(Xtr, ytr)


def draw(Xk, yk, ak, pk, s):
    """Same split and subsampling as run_head_experiment.run_cell, seed s."""
    from sklearn.model_selection import StratifiedGroupKFold

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
        return None
    i_conf = sample_cells(rng, cells_tr, conf, N)
    i_bal = sample_cells(rng, cells_tr, bal, N)
    i_te = np.concatenate([rng.choice(v, size=m_te, replace=False) for v in cells_te.values()])
    return tr_pool, i_conf, i_bal, i_te


def run_cell(bb, attr, task, seeds, max_iter, threads):
    from threadpoolctl import threadpool_limits

    threadpool_limits(threads)
    X, meta = load_chexpert(bb)
    lab = meta[TASKS[task]].to_numpy(dtype=float)
    rows_k = np.flatnonzero((lab == 0) | (lab == 1))
    Xk, yk = X[rows_k], lab[rows_k].astype(int)
    ak = attribute_vector(meta, attr)[rows_k]
    pk = meta["patient_id"].to_numpy()[rows_k]
    t0 = time.time()
    rows = []
    for s in range(seeds):
        d = draw(Xk, yk, ak, pk, s)
        if d is None:
            rows.append(dict(backbone=bb, attribute=attr, task=task, seed=s, feasible=False))
            continue
        tr_pool, i_conf, i_bal, i_te = d
        Xte, yte, ate = Xk[i_te], yk[i_te], ak[i_te]
        Xp_, ap_, yp_ = Xk[tr_pool], ak[tr_pool], yk[tr_pool]
        Xc = Xp_.copy().astype(np.float64)
        for v in (0, 1):
            Xc[yp_ == v] -= Xc[yp_ == v].mean(0)
        u = Xc[ap_ == 1].mean(0) - Xc[ap_ == 0].mean(0)
        u /= np.linalg.norm(u)
        t1 = time.time()
        h_conf = mlp_head(Xk[i_conf], yk[i_conf], s, max_iter)
        h_bal = mlp_head(Xk[i_bal], yk[i_bal], s, max_iter)
        n_iter = (h_conf[-1].n_iter_, h_bal[-1].n_iter_)
        fit_s = time.time() - t1
        rel = reliance(h_conf, Xte, yte, ate, h_bal, u, Xp_.mean(0))
        # pairing check: the logistic head on the same draw
        p_lr = logistic_head(Xk[i_conf], yk[i_conf], s).predict(Xte)
        lr_gap = float(p_lr[(yte == 1) & (ate == 1)].mean() - p_lr[(yte == 1) & (ate == 0)].mean())
        rows.append(dict(backbone=bb, attribute=attr, task=task, seed=s, n_train=int(len(i_conf)),
                         n_test=int(len(i_te)), feasible=True, **rel,
                         mlp_n_iter_conf=int(n_iter[0]), mlp_n_iter_unconf=int(n_iter[1]),
                         mlp_converged=bool(max(n_iter) < max_iter),
                         tpr_gap_logistic_rerun=lr_gap, fit_seconds=round(fit_s, 1)))
    print(f"{bb} {attr} {task}: {time.time() - t0:.0f}s", flush=True)
    return rows


def spearman_ci(x, y, n_boot, seed, groups=None):
    from scipy.stats import spearmanr

    rng = np.random.default_rng(seed)
    vals = []
    if groups is None:
        for _ in range(n_boot):
            i = rng.integers(0, len(x), len(x))
            vals.append(spearmanr(x[i], y[i])[0])
    else:
        ug = np.unique(groups)
        for _ in range(n_boot):
            g = rng.choice(ug, len(ug), replace=True)
            i = np.concatenate([np.flatnonzero(groups == k) for k in g])
            vals.append(spearmanr(x[i], y[i])[0])
    return np.nanpercentile(vals, [2.5, 97.5])


def summarise(mlp: pd.DataFrame, lr: pd.DataFrame) -> pd.DataFrame:
    from scipy.stats import spearmanr

    key = ["backbone", "attribute", "task"]
    cm = mlp[mlp.feasible].groupby(key)[["tpr_gap", "auroc_conf"]].mean()
    cl = lr[lr.feasible].groupby(key)[["probe_auroc", "tpr_gap", "auroc_conf"]].mean()
    c = cm.join(cl, lsuffix="_mlp", rsuffix="_lr", how="inner").reset_index()
    x, y = c.probe_auroc.to_numpy(), c.tpr_gap_mlp.to_numpy()
    out = []
    r, p = spearmanr(x, y)
    lo, hi = spearman_ci(x, y, 2000, 0)
    blo, bhi = spearman_ci(x, y, 2000, 1, groups=c.backbone.to_numpy())
    out.append(dict(stat="rho probe vs MLP TPR gap, all cells", n=len(c), value=r, p=p,
                    ci_cells_lo=lo, ci_cells_hi=hi, ci_backbones_lo=blo, ci_backbones_hi=bhi))
    for a in ("sex", "age"):
        s = c[c.attribute == a]
        r, p = spearmanr(s.probe_auroc, s.tpr_gap_mlp)
        out.append(dict(stat=f"rho probe vs MLP TPR gap, within {a}", n=len(s), value=r, p=p))
    r, p = spearmanr(x, c.tpr_gap_lr)
    out.append(dict(stat="rho probe vs logistic TPR gap, all cells (reference)", n=len(c), value=r, p=p))
    r, p = spearmanr(c.tpr_gap_mlp, c.tpr_gap_lr)
    out.append(dict(stat="rho MLP TPR gap vs logistic TPR gap", n=len(c), value=r, p=p))
    for col in ("tpr_gap", "auroc_conf"):
        for h in ("mlp", "lr"):
            out.append(dict(stat=f"mean {col} ({h})", n=len(c), value=c[f"{col}_{h}"].mean()))
    return c, pd.DataFrame(out)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--seeds", type=int, default=10)
    ap.add_argument("--max-iter", type=int, default=500)
    ap.add_argument("--jobs", type=int, default=3)
    ap.add_argument("--threads", type=int, default=2)
    ap.add_argument("--backbones", nargs="+", default=CHEXPERT_BACKBONES)
    ap.add_argument("--attributes", nargs="+", default=["sex", "age"])
    ap.add_argument("--tasks", nargs="+", default=list(TASKS))
    ap.add_argument("--out-name", default="head_mlp")
    args = ap.parse_args()
    from joblib import Parallel, delayed

    root = results_root()
    out = root / args.out_name
    out.mkdir(parents=True, exist_ok=True)
    man = Manifest(out, "head reliance with an MLP head (reliance part only)")
    man.config = {"seeds": args.seeds, "rho_train": RHO_TRAIN, "rho_test": 0.5, "threshold": 0.5,
                  "head": f"StandardScaler + MLPClassifier({MLP}, max_iter={args.max_iter}, random_state=seed)",
                  "splits": "identical to run_head_experiment.py (same seeds, split and subsampling code)",
                  "probe_values_from": f"{LOGISTIC_DIR}/head_reliance.csv", "tasks": TASKS,
                  "attributes": args.attributes, "backbones": args.backbones,
                  "bootstrap": "2000 over cells (seed 0); 2000 over whole backbones (seed 1)"}
    man.notes += [CHEXPERT_JOIN_NOTE,
                  "tpr_gap_logistic_rerun: logistic head refitted on the same draw; it should equal "
                  "tpr_gap in the logistic results for the same (cell, seed), which confirms pairing."]
    man.add_data(DATA / "train.csv", DATA / "chexpert_multibackbone" / "race_mapping.csv",
                 root / LOGISTIC_DIR / "head_reliance.csv")
    for bb in args.backbones:
        man.add_data(chexpert_paths(bb)[0])
    jobs = [(bb, a, t) for bb in args.backbones for a in args.attributes for t in args.tasks]
    t0 = time.time()
    res = Parallel(n_jobs=args.jobs)(
        delayed(run_cell)(bb, a, t, args.seeds, args.max_iter, args.threads) for bb, a, t in jobs
    )
    man.time("all_cells", time.time() - t0)
    mlp = pd.DataFrame([r for rr in res for r in rr])
    lr = pd.read_csv(root / LOGISTIC_DIR / "head_reliance.csv")
    key = ["backbone", "attribute", "task", "seed"]
    chk = mlp.merge(lr[key + ["tpr_gap", "n_train", "n_test"]], on=key, suffixes=("", "_lr"))
    n_mismatch = int(((chk.tpr_gap_logistic_rerun - chk.tpr_gap_lr).abs() > 1e-9).sum()
                     + (chk.n_train != chk.n_train_lr).sum() + (chk.n_test != chk.n_test_lr).sum())
    man.notes.append(f"pairing check against the logistic run: {n_mismatch} mismatches over {len(chk)} rows")
    probe = lr.groupby(["backbone", "attribute", "task"])[CELL_COLS].first().reset_index()
    mlp = probe.merge(mlp, on=["backbone", "attribute", "task"], how="right")
    mlp.to_csv(out / "head_reliance_mlp.csv", index=False)
    cells, summary = summarise(mlp, lr)
    cells.to_csv(out / "cells_mlp_vs_logistic.csv", index=False)
    summary.to_csv(out / "summary_mlp.csv", index=False)
    man.notes.append(f"MLP fits not converged at max_iter: {int((~mlp.mlp_converged.fillna(True)).sum())}"
                     f" of {int(mlp.feasible.sum())} seeds")
    man.write()
    print(summary.to_string(index=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
