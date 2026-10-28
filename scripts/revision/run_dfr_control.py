#!/usr/bin/env python3
"""DFR controls for the head experiment (reviewer check on the DFR AUROC drop).

In run_head_experiment.py the DFR head (LastLayerRetraining, groups 2A + Y)
cut the TPR gap but lost task AUROC. Its balanced subset is small (8 to 52
images) and its head differs from the confounded head (L1, liblinear,
class_weight balanced, against L2 lbfgs). This script reruns the same cells,
splits and the 3 mitigation seeds and fits five heads, all scored on the same
balanced test set:

- conf: the confounded head of the experiment (StandardScaler + L2 logistic, C = 1).
- dfr: LastLayerRetraining defaults on the confounded training set (reproduces
  the experiment's DFR row).
- dfr_l2: the confounded head's setting fitted on the same balanced subset
  (penalty control).
- random_dfr_head: LastLayerRetraining's head (same class, L1, C = 1) on a
  random subset of the confounded training set of the DFR subset's size, so
  the confounding is kept (size control).
- random_l2: the confounded head's setting on that same random subset.
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
from run_head_experiment import RHO_TRAIN, TASKS, attribute_vector, head, sample_cells  # noqa: E402

METHODS = ("conf", "dfr", "dfr_l2", "random_dfr_head", "random_l2")


def dfr_subset(g: np.ndarray, seed: int) -> np.ndarray:
    """Positions of the balanced subset LastLayerRetraining(random_state=seed) draws (same RNG calls)."""
    rng = np.random.default_rng(seed)
    g_idx = [np.flatnonzero(g == v) for v in np.unique(g)]
    min_g = min(len(gi) for gi in g_idx)
    out = []
    for gi in g_idx:
        idx = gi.copy()
        rng.shuffle(idx)
        out.extend(idx[:min_g])
    return np.asarray(out)


def scores(p, yte, ate):
    from sklearn.metrics import roc_auc_score

    pred = p >= 0.5
    return {"task_auroc": float(roc_auc_score(yte, p)),
            "tpr_gap": float(pred[(yte == 1) & (ate == 1)].mean() - pred[(yte == 1) & (ate == 0)].mean()),
            "fpr_gap": float(pred[(yte == 0) & (ate == 1)].mean() - pred[(yte == 0) & (ate == 0)].mean())}


def run_cell(bb, attr, task, mitig_seeds):
    from sklearn.model_selection import StratifiedGroupKFold

    from shortcut_detect.mitigation import LastLayerRetraining

    X, meta = load_chexpert(bb)
    lab = meta[TASKS[task]].to_numpy(dtype=float)
    rows_k = np.flatnonzero((lab == 0) | (lab == 1))
    Xk, yk = X[rows_k], lab[rows_k].astype(int)
    ak = attribute_vector(meta, attr)[rows_k]
    pk = meta["patient_id"].to_numpy()[rows_k]
    t0 = time.time()
    out = []
    for s in range(mitig_seeds):
        # identical to run_head_experiment.run_cell, same RNG call order
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
            continue
        i_conf = sample_cells(rng, cells_tr, conf, N)
        sample_cells(rng, cells_tr, bal, N)  # unconfounded set, drawn only to keep the RNG in step
        i_te = np.concatenate([rng.choice(v, size=m_te, replace=False) for v in cells_te.values()])
        Xte, yte, ate = Xk[i_te], yk[i_te], ak[i_te]
        Xte64 = Xte.astype(np.float64)

        g = 2 * ak[i_conf] + yk[i_conf]
        dfr = LastLayerRetraining(random_state=s).fit(Xk[i_conf], yk[i_conf], g)
        n_fit = int(dfr._n_balanced)
        i_dfr = i_conf[dfr_subset(g, s)]
        assert len(i_dfr) == n_fit
        p_dfr = dfr.classifier_.predict_proba(dfr.scaler_.transform(Xte64))[:, 1]
        # the reconstructed subset is the one DFR used: refitting on it gives the same head
        chk = LastLayerRetraining(random_state=s).fit(Xk[i_dfr], yk[i_dfr], np.zeros(n_fit, int))
        assert np.allclose(chk.classifier_.predict_proba(chk.scaler_.transform(Xte64))[:, 1], p_dfr, atol=1e-6)

        i_rnd = np.random.default_rng(20_000 + 100 * s).choice(i_conf, size=n_fit, replace=False)
        rnd_ok = len(np.unique(yk[i_rnd])) == 2
        preds = {"conf": head(Xk[i_conf], yk[i_conf], s).predict_proba(Xte)[:, 1],
                 "dfr": p_dfr,
                 "dfr_l2": head(Xk[i_dfr], yk[i_dfr], s).predict_proba(Xte)[:, 1]}
        if rnd_ok:
            # a single group label makes LastLayerRetraining fit on all rows given (a shuffle only)
            r = LastLayerRetraining(random_state=s).fit(Xk[i_rnd], yk[i_rnd], np.zeros(n_fit, int))
            preds["random_dfr_head"] = r.classifier_.predict_proba(r.scaler_.transform(Xte64))[:, 1]
            preds["random_l2"] = head(Xk[i_rnd], yk[i_rnd], s).predict_proba(Xte)[:, 1]
        base = dict(backbone=bb, attribute=attr, task=task, seed=s, n_train_conf=int(len(i_conf)),
                    n_test=int(len(i_te)), n_dfr_subset=n_fit,
                    n_random_pos=int(yk[i_rnd].sum()), n_random_A1=int(ak[i_rnd].sum()),
                    random_aligned_share=float((ak[i_rnd] == yk[i_rnd]).mean()))
        for meth in METHODS:
            sc = (scores(preds[meth], yte, ate) if meth in preds
                  else {"task_auroc": np.nan, "tpr_gap": np.nan, "fpr_gap": np.nan})
            n = {"conf": len(i_conf)}.get(meth, n_fit)
            out.append({**base, "method": meth, "n_head_fit": int(n), "fitted": meth in preds, **sc})
    print(f"{bb} {attr} {task}: {time.time() - t0:.0f}s", flush=True)
    return out


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--mitig-seeds", type=int, default=3)
    ap.add_argument("--jobs", type=int, default=3)
    ap.add_argument("--backbones", nargs="+", default=CHEXPERT_BACKBONES)
    args = ap.parse_args()
    from joblib import Parallel, delayed

    out = results_root() / "dfr_control"
    out.mkdir(parents=True, exist_ok=True)
    man = Manifest(out, "DFR size and penalty controls for the head experiment")
    man.config = {"mitigation_seeds": args.mitig_seeds, "rho_train": RHO_TRAIN, "rho_test": 0.5,
                  "threshold": 0.5, "tasks": TASKS, "attributes": ["sex (Male=1)", "age 60+ (=1)"],
                  "split": "as run_head_experiment.py (patient-level StratifiedGroupKFold(2) on (A, Y), same seeds)",
                  "conf / dfr_l2 / random_l2 head": "run_head_experiment.head: StandardScaler + LogisticRegression(C=1, L2, lbfgs)",
                  "dfr / random_dfr_head head": "LastLayerRetraining defaults: StandardScaler + LogisticRegression(C=1, L1, liblinear, class_weight=balanced)",
                  "dfr groups": "2A + Y on the confounded training set",
                  "random subset": "uniform draw without replacement from the confounded training set, size = DFR subset, seed 20000 + 100 * s"}
    man.notes += [CHEXPERT_JOIN_NOTE,
                  "dfr_l2 uses the exact DFR subset (reconstructed from LastLayerRetraining's RNG calls; checked by refitting)."]
    man.add_data(DATA / "train.csv", DATA / "chexpert_multibackbone" / "race_mapping.csv")
    for bb in args.backbones:
        man.add_data(chexpert_paths(bb)[0])
    jobs = [(bb, a, t) for bb in args.backbones for a in ("sex", "age") for t in TASKS]
    t0 = time.time()
    res = Parallel(n_jobs=args.jobs)(delayed(run_cell)(bb, a, t, args.mitig_seeds) for bb, a, t in jobs)
    df = pd.DataFrame([r for rr in res for r in rr])
    df.to_csv(out / "dfr_control_rows.csv", index=False)
    summ = (df.groupby("method", sort=False)
              .agg(n_rows=("task_auroc", "size"), n_fitted=("fitted", "sum"),
                   task_auroc_mean=("task_auroc", "mean"), task_auroc_sd=("task_auroc", "std"),
                   tpr_gap_mean=("tpr_gap", "mean"), tpr_gap_sd=("tpr_gap", "std"),
                   abs_tpr_gap_mean=("tpr_gap", lambda v: v.abs().mean()),
                   fpr_gap_mean=("fpr_gap", "mean"), n_head_fit_mean=("n_head_fit", "mean"))
              .reset_index())
    summ.to_csv(out / "dfr_control_summary.csv", index=False)
    print(summ.to_string(index=False))
    man.time("all_cells", time.time() - t0)
    man.write()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
