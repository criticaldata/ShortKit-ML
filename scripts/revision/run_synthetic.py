#!/usr/bin/env python3
"""Conditional synthetic benchmark (revision decisions, section 6).

Grid: variant {linear, xor, variance} x rho {0.5, 0.7, 0.9} x delta {0, 0.2, 0.5,
0.8, 1.2, 2.0} x n {200, 1000} x d {128, 512} x 10 seeds. Each dataset runs the
six-detector conditional audit (K = 200, no bootstrap) and, for comparison, the
marginal probe (E -> A, global shuffle) that ignores Y.

Also writes ``statistical_correction.csv``: on null data (delta = 0), the share of
datasets in which the per-dimension within-Y test finds at least one
significant dimension, uncorrected and under four corrections.
"""

from __future__ import annotations

import argparse
import itertools
import os
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
from common import Manifest, results_root  # noqa: E402

VARIANTS = ["linear", "xor", "variance"]
RHOS = [0.5, 0.7, 0.9]
DELTAS = [0.0, 0.2, 0.5, 0.8, 1.2, 2.0]
NS = [200, 1000]
DS = [128, 512]
TASK_DIMS, ATTR_DIMS, TASK_EFFECT = 4, 4, 1.0


def one(job: dict, K: int) -> list[dict]:
    import torch

    torch.set_num_threads(int(os.environ.get("AUDIT_THREADS", "2")))
    from shortcut_detect.audit import AuditConfig, conditional_audit
    from shortcut_detect.benchmark.conditional_synthetic import generate_conditional_synthetic

    t0 = time.time()
    ds = generate_conditional_synthetic(
        n_samples=job["n"],
        embedding_dim=job["d"],
        delta=job["delta"],
        rho=job["rho"],
        variant=job["variant"],
        attribute_dims=ATTR_DIMS,
        task_dims=TASK_DIMS,
        task_effect=TASK_EFFECT,
        seed=job["seed"],
    )
    seed = 10_000 + job["seed"]
    res = conditional_audit(
        ds.embeddings, ds.attribute, ds.task_labels, AuditConfig(n_permutations=K, n_bootstrap=0, seed=seed)
    )
    marg = conditional_audit(
        ds.embeddings,
        ds.attribute,
        ds.task_labels,
        AuditConfig(
            n_permutations=K,
            n_bootstrap=0,
            detectors=("probe",),
            permutation="global",
            include_y=False,
            seed=seed,
        ),
    )
    rows = []
    for r in res.rows():
        rows.append({**job, **r})
    m = marg.results["probe"]
    rows.append(
        {
            **job,
            "detector": "probe_unconditional",
            "statistic": m.statistic,
            "null_mean": m.null_mean,
            "null_sd": m.null_sd,
            "z": m.z,
            "p_value": m.p_value,
            "p_bh": m.p_value,
            "flag": m.p_value < 0.05,
            "n_permutations": m.n_permutations,
            "seconds": m.seconds,
        }
    )
    for r in rows:
        r["dataset_seconds"] = round(time.time() - t0, 2)
    return rows


def correction_null(n_reps: int = 200) -> pd.DataFrame:
    """Share of null datasets with any significant dimension, by correction."""
    from scipy.stats import mannwhitneyu
    from statsmodels.stats.multitest import multipletests

    from shortcut_detect.benchmark.conditional_synthetic import generate_conditional_synthetic

    rows = []
    for d in [16, 32, 64, 128, 256, 512]:
        hits = dict.fromkeys(["uncorrected", "bonferroni", "holm", "fdr_bh", "fdr_by"], 0)
        for rep in range(n_reps):
            ds = generate_conditional_synthetic(
                n_samples=1000, embedding_dim=d, delta=0.0, rho=0.7, seed=50_000 + rep,
                attribute_dims=ATTR_DIMS, task_dims=TASK_DIMS,
            )
            X, a, y = ds.embeddings, ds.attribute, ds.task_labels
            # within-Y test per dimension, the two strata combined by Fisher's method
            from scipy.stats import combine_pvalues

            ps = []
            for j in range(d):
                pj = [
                    mannwhitneyu(X[(y == v) & (a == 1), j], X[(y == v) & (a == 0), j]).pvalue
                    for v in (0, 1)
                ]
                ps.append(combine_pvalues(pj, method="fisher").pvalue)
            ps = np.array(ps)
            hits["uncorrected"] += bool((ps < 0.05).any())
            for meth in ["bonferroni", "holm", "fdr_bh", "fdr_by"]:
                hits[meth] += bool(multipletests(ps, alpha=0.05, method=meth)[0].any())
        for k, v in hits.items():
            rows.append({"d": d, "correction": k, "fpr": v / n_reps, "n_reps": n_reps})
    return pd.DataFrame(rows)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--seeds", type=int, default=10)
    ap.add_argument("--K", type=int, default=200)
    ap.add_argument("--jobs", type=int, default=4)
    ap.add_argument("--smoke", action="store_true", help="tiny grid for a quick check")
    args = ap.parse_args()
    from joblib import Parallel, delayed

    out = results_root() / "synthetic"
    out.mkdir(parents=True, exist_ok=True)
    man = Manifest(out, "conditional synthetic benchmark")
    grid = dict(variant=VARIANTS, rho=RHOS, delta=DELTAS, n=NS, d=DS, seed=list(range(args.seeds)))
    if args.smoke:
        grid = dict(variant=VARIANTS, rho=[0.9], delta=[0.0, 2.0], n=[200], d=[128], seed=[0])
    man.config = {
        **{k: v for k, v in grid.items() if k != "seed"},
        "seeds": len(grid["seed"]),
        "K": args.K,
        "task_dims": TASK_DIMS,
        "attribute_dims": ATTR_DIMS,
        "task_effect": TASK_EFFECT,
        "alpha": 0.05,
    }
    jobs = [dict(zip(grid, v, strict=True)) for v in itertools.product(*grid.values())]
    t0 = time.time()
    rows: list[dict] = []
    chunk = 120
    for c0 in range(0, len(jobs), chunk):
        part = Parallel(n_jobs=args.jobs)(delayed(one)(j, args.K) for j in jobs[c0 : c0 + chunk])
        for r in part:
            rows.extend(r)
        pd.DataFrame(rows).to_csv(out / "synthetic_runs.csv", index=False)
        print(f"{min(c0 + chunk, len(jobs))}/{len(jobs)} datasets, {time.time() - t0:.0f}s", flush=True)
    man.time("audit_grid", time.time() - t0)
    t1 = time.time()
    correction_null(20 if args.smoke else 200).to_csv(out / "statistical_correction.csv", index=False)
    man.time("statistical_correction", time.time() - t1)
    df = pd.DataFrame(rows)
    per = df.groupby(["n", "d"])["dataset_seconds"].mean().round(2).to_dict()
    man.notes.append(f"mean seconds per dataset by (n, d): {per}")
    man.write()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
