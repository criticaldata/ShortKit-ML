#!/usr/bin/env python3
"""Does a probe on nonlinear features, under the same conditional permutation
null, recover the two synthetic signals the linear probe misses?

One small setting of the conditional synthetic benchmark (n = 1000, d = 64,
rho = 0.5, delta in {0, 1.2, 2.0}, K = 200). For each dataset the conditional
probe is run three ways through ``conditional_audit`` with ``detectors=("probe",)``,
so the null (A permuted within Y), the folds, the in-fold standardisation and
the AUROC statistic are those of the paper's audit; only the design changes:

linear   [E]                     the paper's probe
sq       [E, E^2]                squared coordinates
prod     [E, E^2, E_i E_j, i<j]  all pairwise products, d(d-1)/2 of them

At delta = 0 the generator skips the signal block, so the three variants give
the same data for the same seed; the null is run once and labelled ``none``
(not ``null``, which pandas reads as a missing value).
The data seed and the audit seed (10000 + seed) match ``run_synthetic.py``.
"""

from __future__ import annotations

import argparse
import os
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
from common import Manifest, results_root  # noqa: E402

VARIANTS = ["linear", "xor", "variance"]
DELTAS = [1.2, 2.0]
N, D, RHO = 1000, 64, 0.5
TASK_DIMS, ATTR_DIMS, TASK_EFFECT = 4, 4, 1.0
PROBES = ["linear", "sq", "prod"]


def features(E: np.ndarray, probe: str) -> np.ndarray:
    if probe == "linear":
        return E
    sq = E * E
    if probe == "sq":
        return np.concatenate([E, sq], axis=1)
    iu = np.triu_indices(E.shape[1], 1)
    prod = (E[:, :, None] * E[:, None, :])[:, iu[0], iu[1]]
    return np.concatenate([E, sq, prod], axis=1).astype(np.float32)


def one(variant: str, delta: float, seed: int, K: int) -> list[dict]:
    from shortcut_detect.audit import AuditConfig, conditional_audit
    from shortcut_detect.benchmark.conditional_synthetic import generate_conditional_synthetic

    ds = generate_conditional_synthetic(
        n_samples=N, embedding_dim=D, delta=delta, rho=RHO,
        variant="linear" if variant == "none" else variant,
        attribute_dims=ATTR_DIMS, task_dims=TASK_DIMS, task_effect=TASK_EFFECT, seed=seed,
    )
    rows = []
    for probe in PROBES:
        t0 = time.time()
        F = features(ds.embeddings, probe)
        res = conditional_audit(
            F, ds.attribute, ds.task_labels,
            AuditConfig(n_permutations=K, n_bootstrap=0, detectors=("probe",), seed=10_000 + seed),
        )
        m = res.results["probe"]
        rows.append({
            "variant": variant, "delta": delta, "seed": seed, "probe": probe,
            "n_features": int(F.shape[1]), "statistic": m.statistic, "null_mean": m.null_mean,
            "null_sd": m.null_sd, "z": m.z, "p_value": m.p_value, "reject": m.p_value < 0.05,
            "n_permutations": m.n_permutations, "seconds": round(time.time() - t0, 2),
        })
    return rows


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--seeds", type=int, default=50)
    ap.add_argument("--K", type=int, default=200)
    args = ap.parse_args()
    import torch

    torch.set_num_threads(int(os.environ.get("AUDIT_THREADS", os.environ.get("OMP_NUM_THREADS", "3"))))
    out = results_root() / "nonlinear_probe"
    out.mkdir(parents=True, exist_ok=True)
    man = Manifest(out, "nonlinear probe check on the conditional synthetic benchmark")
    cells = [("none", 0.0)] + [(v, dl) for v in VARIANTS for dl in DELTAS]
    man.config = {
        "cells": [f"{v} delta={dl:g}" for v, dl in cells], "probes": PROBES, "n": N, "d": D, "rho": RHO,
        "seeds": args.seeds, "K": args.K, "task_dims": TASK_DIMS, "attribute_dims": ATTR_DIMS,
        "task_effect": TASK_EFFECT, "alpha": 0.05, "null": "A permuted within Y (image level)",
    }
    t0 = time.time()
    rows: list[dict] = []
    for v, dl in cells:
        for s in range(args.seeds):
            rows.extend(one(v, dl, s, args.K))
        pd.DataFrame(rows).to_csv(out / "rows.csv", index=False)
        print(f"{v} delta={dl:g}: {len(rows)} rows, {time.time() - t0:.0f}s", flush=True)
    man.time("audits", time.time() - t0)
    df = pd.DataFrame(rows)
    summ = (
        df.groupby(["variant", "delta", "probe"], sort=False)
        .agg(reject_rate=("reject", "mean"), auroc_mean=("statistic", "mean"), auroc_sd=("statistic", "std"),
             null_mean=("null_mean", "mean"), n_features=("n_features", "first"), n_seeds=("seed", "nunique"))
        .reset_index()
    )
    summ.to_csv(out / "summary.csv", index=False)
    man.notes.append(f"mean seconds per audit by probe: {df.groupby('probe').seconds.mean().round(2).to_dict()}")
    man.write()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
