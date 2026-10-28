#!/usr/bin/env python3
"""Large null check at the hardest setting of the synthetic grid.

The grid has only 40 null data sets per rho (20 at n = 1000), too few to tell a
rate of 0.15 from 0.05. This run draws many fresh null data sets (delta = 0) at
rho = 0.9, n = 1000, d = 128 and records each detector's permutation p-value.
"""

from __future__ import annotations

import argparse
import os
import sys
import time
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
from common import Manifest, results_root  # noqa: E402
from run_synthetic import ATTR_DIMS, TASK_DIMS, TASK_EFFECT  # noqa: E402

RHO, N, D, K = 0.9, 1000, 128, 200
SEED0 = 90_000  # disjoint from the grid's seeds


def one(s: int) -> list[dict]:
    import torch

    torch.set_num_threads(1)
    from shortcut_detect.audit import AuditConfig, conditional_audit
    from shortcut_detect.benchmark.conditional_synthetic import generate_conditional_synthetic

    ds = generate_conditional_synthetic(
        n_samples=N, embedding_dim=D, delta=0.0, rho=RHO, variant="linear",
        attribute_dims=ATTR_DIMS, task_dims=TASK_DIMS, task_effect=TASK_EFFECT, seed=SEED0 + s,
    )
    r = conditional_audit(ds.embeddings, ds.attribute, ds.task_labels,
                          AuditConfig(n_permutations=K, n_bootstrap=0, seed=s))
    return [{"seed": SEED0 + s, "detector": d, "p_value": v.p_value, "flag": v.flag} for d, v in r.results.items()]


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--datasets", type=int, default=500)
    ap.add_argument("--jobs", type=int, default=int(os.environ.get("NULLCHECK_JOBS", "6")))
    args = ap.parse_args()
    from joblib import Parallel, delayed

    out = results_root() / "synthetic"
    man = Manifest(out, "large null check, rho 0.9")
    man.config = {"rho": RHO, "n": N, "d": D, "K": K, "delta": 0.0, "variant": "linear",
                  "n_datasets": args.datasets, "seeds": f"{SEED0} to {SEED0 + args.datasets - 1}"}
    t0 = time.time()
    rows = [r for part in Parallel(n_jobs=args.jobs)(delayed(one)(s) for s in range(args.datasets)) for r in part]
    df = pd.DataFrame(rows)
    df.to_csv(out / "null_check_rho09.csv", index=False)
    man.time("all", time.time() - t0)
    man.out_dir = out / "null_check"
    man.write()
    print(df.assign(reject=df.p_value < 0.05).groupby("detector").reject.mean().round(3).to_string())
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
