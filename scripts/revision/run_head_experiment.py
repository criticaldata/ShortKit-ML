#!/usr/bin/env python3
"""Head experiment (revision decisions, section 7, items 1 to 3).

For every CheXpert backbone, attribute A in {sex (Male = 1), age 60+ (= 1)} and
task Y in {Pneumothorax, Pleural Effusion, Cardiomegaly} (positive vs negative;
uncertain and blank rows dropped and counted):

1. Patients are split 50/50 into a training pool and a test pool. A confounded
   training set is drawn from the training pool with P(Y=1 | A=1) = 0.9 and
   P(Y=1 | A=0) = 0.1; an unconfounded one of the same size has all four (A, Y)
   cells equal. A balanced test set (rho = 0.5, the four cells equal) is drawn
   from the test pool. A logistic head (standardised inputs, C = 1) is trained
   on each. Reliance of the confounded head is measured on the test set as
   (a) the TPR and FPR gaps between A groups, (b) the AUROC drop against the
   unconfounded head, (c) the mean absolute change of its output when the A
   direction is projected out of the test embeddings, and, following Kina and
   Petersen (2026), (d) ECE per A group and AUROC on A-aligned against
   A-misaligned cases.
2. The conditional probe (A from [E, Y], patient-grouped, K = 200) is run on
   each task's labelled images; its score is rank-correlated across cells with
   each reliance measure.
3. Two embedding-level mitigations from the package (AdversarialDebiasing and
   ContrastiveDebiasing) are fitted on the confounded training embeddings,
   applied to training and test embeddings, and the audit (on the whole test
   pool) and the head are rerun. Each has a control: the same encoder trained
   identically with the debiasing term switched off (adversary_weight = 0,
   contrastive_weight = 0), which separates the effect of passing through a
   smaller learned encoder from the effect of the mitigation. Every
   mitigation seed is audited.
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

TASKS = {"pneumothorax": "Pneumothorax", "pleural_effusion": "Pleural Effusion",
         "cardiomegaly": "Cardiomegaly"}
RHO_TRAIN = 0.9
ATTR_REMOVAL = (
    "Unit direction u = difference of A-group means after centring each Y stratum, estimated "
    "on the whole training pool (raw embedding space); test embeddings are replaced by "
    "x - ((x - m) . u) u, with m the training-pool mean, and the confounded head is reapplied."
)


def attribute_vector(meta: pd.DataFrame, attr: str) -> np.ndarray:
    if attr == "sex":
        return (meta["sex"].to_numpy() == "Male").astype(int)
    return (meta["age_first_study"].to_numpy(dtype=float) >= 60).astype(int)  # constant per patient


def ece(p: np.ndarray, y: np.ndarray, bins: int = 10) -> float:
    edges = np.linspace(0, 1, bins + 1)
    idx = np.clip(np.digitize(p, edges[1:-1]), 0, bins - 1)
    tot = 0.0
    for b in range(bins):
        m = idx == b
        if m.any():
            tot += m.mean() * abs(p[m].mean() - y[m].mean())
    return float(tot)


def sample_cells(rng, cell_idx: dict, fracs: dict, N: int) -> np.ndarray:
    out = []
    for c, frac in fracs.items():
        k = int(round(frac * N))
        out.append(rng.choice(cell_idx[c], size=k, replace=False))
    return np.concatenate(out)


def head(Xtr, ytr, seed):
    from sklearn.linear_model import LogisticRegression
    from sklearn.pipeline import make_pipeline
    from sklearn.preprocessing import StandardScaler

    return make_pipeline(StandardScaler(), LogisticRegression(C=1.0, max_iter=3000)).fit(Xtr, ytr)


def reliance(model, Xte, yte, ate, model_unconf, u, m):
    from sklearn.metrics import roc_auc_score

    p = model.predict_proba(Xte)[:, 1]
    pred = (p >= 0.5).astype(int)
    out = {}
    for name, sel in (("tpr", yte == 1), ("fpr", yte == 0)):
        r1 = pred[sel & (ate == 1)].mean()
        r0 = pred[sel & (ate == 0)].mean()
        out[f"{name}_gap"] = float(r1 - r0)
    out["auroc_conf"] = float(roc_auc_score(yte, p))
    out["auroc_unconf"] = float(roc_auc_score(yte, model_unconf.predict_proba(Xte)[:, 1]))
    out["auroc_drop"] = out["auroc_unconf"] - out["auroc_conf"]
    Xp = Xte - np.outer((Xte - m) @ u, u)
    out["output_shift"] = float(np.mean(np.abs(model.predict_proba(Xp)[:, 1] - p)))
    aligned = ate == yte
    out["auroc_aligned"] = float(roc_auc_score(yte[aligned], p[aligned]))
    out["auroc_misaligned"] = float(roc_auc_score(yte[~aligned], p[~aligned]))
    for g in (0, 1):
        out[f"ece_A{g}"] = ece(p[ate == g], yte[ate == g])
    out["ece_gap"] = out["ece_A1"] - out["ece_A0"]
    pu = model_unconf.predict_proba(Xte)[:, 1]
    out["unconf_auroc_aligned"] = float(roc_auc_score(yte[aligned], pu[aligned]))
    out["unconf_auroc_misaligned"] = float(roc_auc_score(yte[~aligned], pu[~aligned]))
    return out


MITIGATIONS = {
    "adversarial": ("AdversarialDebiasing", {}),
    "adversarial_control": ("AdversarialDebiasing", {"adversary_weight": 0.0}),
    "contrastive": ("ContrastiveDebiasing", {}),
    "contrastive_control": ("ContrastiveDebiasing", {"contrastive_weight": 0.0}),
}


def run_cell(bb, attr, task, seeds, mitig_seeds, k_audit, k_mitig, threads):
    import torch

    torch.set_num_threads(threads)
    from sklearn.metrics import roc_auc_score
    from sklearn.model_selection import StratifiedGroupKFold

    from shortcut_detect.audit import AuditConfig, conditional_audit
    from shortcut_detect import mitigation as mitig_mod

    X, meta = load_chexpert(bb)
    col = TASKS[task]
    lab = meta[col].to_numpy(dtype=float)
    keep = (lab == 0) | (lab == 1)
    n_dropped_uncertain = int((lab == -1).sum())
    n_dropped_blank = int(np.isnan(lab).sum())
    rows_k = np.flatnonzero(keep)
    Xk, yk = X[rows_k], lab[rows_k].astype(int)
    ak = attribute_vector(meta, attr)[rows_k]
    pk = meta["patient_id"].to_numpy()[rows_k]
    info = dict(backbone=bb, attribute=attr, task=task, n_labelled=int(keep.sum()),
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

    rel_rows, mit_rows = [], []
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

        if s < mitig_seeds:
            cfg = AuditConfig(n_permutations=k_mitig, n_bootstrap=0, detectors=("probe",), seed=42)
            before = conditional_audit(Xk[te_pool], ak[te_pool], yk[te_pool], cfg,
                                       groups=pk[te_pool]).results["probe"]
            for name, (cls_name, kw) in MITIGATIONS.items():
                t1 = time.time()
                mit = getattr(mitig_mod, cls_name)(random_state=s, device="cpu", **kw)
                if cls_name == "AdversarialDebiasing":
                    mit.fit(Xk[i_conf], ak[i_conf], yk[i_conf])
                else:
                    mit.fit(Xk[i_conf], yk[i_conf], ak[i_conf])
                Ztr = mit.transform(Xk[i_conf])
                Zte = mit.transform(Xte)
                Zpool = mit.transform(Xk[te_pool])
                h_m = head(Ztr, yk[i_conf], s)
                pm = h_m.predict_proba(Zte)[:, 1]
                after = conditional_audit(Zpool, ak[te_pool], yk[te_pool], cfg, groups=pk[te_pool]).results["probe"]
                mit_rows.append({**info, "seed": s, "method": name, "n_train": int(len(i_conf)),
                                 "n_test": int(len(i_te)), "n_audit_pool": int(len(te_pool)),
                                 "task_auroc_before": rel["auroc_conf"],
                                 "task_auroc_after": float(roc_auc_score(yte, pm)),
                                 "tpr_gap_before": rel["tpr_gap"],
                                 "tpr_gap_after": float((pm[(yte == 1) & (ate == 1)] >= 0.5).mean()
                                                        - (pm[(yte == 1) & (ate == 0)] >= 0.5).mean()),
                                 "probe_auroc_before": before.statistic, "probe_p_before": before.p_value,
                                 "probe_auroc_after": after.statistic, "probe_p_after": after.p_value,
                                 "seconds": round(time.time() - t1, 1)})
    print(f"{bb} {attr} {task}: {time.time() - t0:.0f}s", flush=True)
    return rel_rows, mit_rows


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--seeds", type=int, default=10)
    ap.add_argument("--mitig-seeds", type=int, default=3)
    ap.add_argument("--K", type=int, default=500)  # probe K, section 9
    ap.add_argument("--K-mitig", type=int, default=200)
    ap.add_argument("--jobs", type=int, default=3)
    ap.add_argument("--threads", type=int, default=3)
    ap.add_argument("--backbones", nargs="+", default=CHEXPERT_BACKBONES)
    args = ap.parse_args()
    from joblib import Parallel, delayed

    out = results_root() / "head_experiment"
    out.mkdir(parents=True, exist_ok=True)
    man = Manifest(out, "head reliance experiment (section 7, items 1 to 3)")
    man.config = {"seeds": args.seeds, "mitigation_seeds": args.mitig_seeds, "K_audit": args.K,
                  "K_mitigation_audits": args.K_mitig, "age": "age at the first study (patient-constant)",
                  "rho_train": RHO_TRAIN, "rho_test": 0.5, "head": "StandardScaler + LogisticRegression(C=1)",
                  "threshold": 0.5, "tasks": TASKS, "attributes": ["sex (Male=1)", "age 60+ (=1)"],
                  "mitigations": {k: f"{c}({v or 'defaults'})" for k, (c, v) in MITIGATIONS.items()},
                  "split": "patient-level StratifiedGroupKFold(2) on (A, Y)"}
    man.notes += [CHEXPERT_JOIN_NOTE, "attribute removal: " + ATTR_REMOVAL]
    man.add_data(DATA / "train.csv", DATA / "chexpert_multibackbone" / "race_mapping.csv")
    for bb in args.backbones:
        man.add_data(chexpert_paths(bb)[0])
    jobs = [(bb, a, t) for bb in args.backbones for a in ("sex", "age") for t in TASKS]
    t0 = time.time()
    res = Parallel(n_jobs=args.jobs)(
        delayed(run_cell)(bb, a, t, args.seeds, args.mitig_seeds, args.K, args.K_mitig, args.threads)
        for bb, a, t in jobs
    )
    pd.DataFrame([r for rr, _ in res for r in rr]).to_csv(out / "head_reliance.csv", index=False)
    pd.DataFrame([r for _, mm in res for r in mm]).to_csv(out / "mitigation.csv", index=False)
    man.time("all_cells", time.time() - t0)
    man.write()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
