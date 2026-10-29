#!/usr/bin/env python3
"""How the run time of the primary test (the conditional probe) grows with the
number of images.

Synthetic embeddings only: standard normal, d = 768 (the RAD-DINO width), a
weak linear attribute signal on a few coordinates, binary Y independent of A
(rho = 0.5), patients as groups of 4 images with A constant per patient. The
audit is the one ``run_clinical_audit.py`` runs for the probe alone:
``conditional_audit`` with ``detectors=("probe",)``, patient-grouped 5-fold CV,
5 CV repeats for the observed statistic, patient-level permutation within Y
strata, 2000 patient bootstrap resamples, seed 42.

One audit per n. K = 500 (the paper's probe setting) where the point is
predicted to fit the per-point budget, otherwise two smaller K, with a line in K
read at K = 500 (the observed statistic and the bootstrap do not depend on K,
so a single small-K time scaled by 500 / K would overstate it). The largest
n is added only when the previous point ran fast enough. The exponent is the
slope of a least-squares line of log(seconds at K = 500) on log(n).

At n = 2000 the K = 50 and K = 500 audits took 5.8 s and 29.7 s (3 threads),
so scaling a single small-K time by 500 / K would double the estimate; the
fallback therefore measures two small K and fits seconds = a + b K.
"""

from __future__ import annotations

import argparse
import os
import platform
import subprocess
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
from common import Manifest, results_root  # noqa: E402

D = 768
RHO = 0.5
IMAGES_PER_PATIENT = 4
ATTR_DIMS, ATTR_EFFECT = 4, 0.3  # weak linear signal: AUROC well above chance, far from 1
TASK_DIMS, TASK_EFFECT = 4, 1.0
K_PAPER = 500  # run_clinical_audit.K_PROBE
K_SMALL = (50, 150)  # two points, so the K-independent part (observed statistic, bootstrap) is fitted, not scaled
N_BOOT = 2000  # run_clinical_audit.N_BOOT
SEED = 42  # run_clinical_audit.SEED
NS = [2000, 10_000, 50_000]
N_EXTRA = 100_000
EXTRA_IF_UNDER = 480.0  # seconds the 50,000 point must stay under for the 100,000 point to run


def cpu_name() -> str:
    try:
        return subprocess.check_output(["sysctl", "-n", "machdep.cpu.brand_string"], text=True).strip()
    except Exception:  # not macOS
        return platform.processor() or platform.machine()


def make_data(n: int, seed: int) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    rng = np.random.default_rng(seed)
    n_pat = n // IMAGES_PER_PATIENT
    groups = np.repeat(np.arange(n_pat), IMAGES_PER_PATIENT)
    a_pat = (rng.random(n_pat) < 0.5).astype(np.int64)
    a = a_pat[groups]
    y = (rng.random(n) < 0.5).astype(np.int64)  # rho = 0.5: A carries nothing about Y
    E = rng.standard_normal((n, D)).astype(np.float32)
    E[:, :TASK_DIMS] += TASK_EFFECT * (2 * y[:, None] - 1)
    E[:, TASK_DIMS:TASK_DIMS + ATTR_DIMS] += ATTR_EFFECT * (2 * a[:, None] - 1)
    return E, a, y, groups


def one(n: int, k: int, threads: int) -> dict:
    from shortcut_detect.audit import AuditConfig, conditional_audit

    E, a, y, groups = make_data(n, SEED)
    cfg = AuditConfig(n_permutations=k, n_permutations_probe=k, n_bootstrap=N_BOOT,
                      detectors=("probe",), seed=SEED)
    t0 = time.time()
    res = conditional_audit(E, a, y, cfg, groups=groups)
    sec = time.time() - t0
    m = res.results["probe"]
    return {
        "n": n, "n_patients": int(len(np.unique(groups))), "d": D, "K": k,
        "seconds": round(sec, 1), "seconds_at_K500": round(sec, 1) if k == K_PAPER else float("nan"),
        "extrapolated": k != K_PAPER, "auroc": round(m.statistic, 4), "p_value": m.p_value,
        "fold_strata": res.fold_strata, "threads": threads, "platform": platform.platform(),
        "cpu": cpu_name(),
    }


def small_k_point(n: int, threads: int) -> list[dict]:
    """Two audits at the small K; seconds = a + b K fitted and read at K_PAPER."""
    rows = [one(n, k, threads) for k in K_SMALL]
    b = (rows[1]["seconds"] - rows[0]["seconds"]) / (K_SMALL[1] - K_SMALL[0])
    at_500 = rows[0]["seconds"] + b * (K_PAPER - K_SMALL[0])
    for r in rows:
        r["seconds_at_K500"] = round(at_500, 1)
    return rows


def fit_exponent(df: pd.DataFrame) -> float:
    g = df.groupby("n").seconds_at_K500.first()
    return float(np.polyfit(np.log(g.index.to_numpy(dtype=float)), np.log(g.to_numpy(dtype=float)), 1)[0])


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--point-budget", type=float, default=900.0,
                    help="predicted seconds above which a point runs at the small K")
    ap.add_argument("--total-budget", type=float, default=2400.0,
                    help="wall seconds for the whole run; a point predicted to cross it runs at the small K, "
                         "and the extra point is skipped")
    args = ap.parse_args()
    import torch

    threads = int(os.environ.get("AUDIT_THREADS", os.environ.get("OMP_NUM_THREADS", "3")))
    torch.set_num_threads(threads)
    out = results_root() / "probe_scaling"
    out.mkdir(parents=True, exist_ok=True)
    man = Manifest(out, "probe run time against sample size (synthetic embeddings)")
    man.config = {
        "n": NS, "n_extra": N_EXTRA, "extra_if_50k_under_seconds": EXTRA_IF_UNDER, "d": D, "rho": RHO,
        "images_per_patient": IMAGES_PER_PATIENT, "attribute_dims": ATTR_DIMS, "attribute_effect": ATTR_EFFECT,
        "task_dims": TASK_DIMS, "task_effect": TASK_EFFECT, "K_paper": K_PAPER, "K_small": K_SMALL,
        "point_budget_seconds": args.point_budget, "total_budget_seconds": args.total_budget, "n_bootstrap": N_BOOT, "n_folds": 5, "cv_repeats": 5,
        "seed": SEED, "detectors": ["probe"], "threads": threads, "platform": platform.platform(),
    }
    rows: list[dict] = []
    t0 = time.time()
    ns = list(NS)
    while ns:
        n = ns.pop(0)
        full = True
        if rows:  # predict the K = 500 time from the points so far
            df = pd.DataFrame(rows)
            slope = fit_exponent(df) if df.n.nunique() > 1 else 1.0
            pred = df.seconds_at_K500.iloc[-1] * (n / df.n.iloc[-1]) ** slope
            full = pred <= args.point_budget and time.time() - t0 + pred <= args.total_budget
            print(f"n={n}: predicted {pred:.0f}s at K={K_PAPER} (slope {slope:.2f}), "
                  f"running K={K_PAPER if full else K_SMALL}", flush=True)
        new = [one(n, K_PAPER, threads)] if full else small_k_point(n, threads)
        rows += new
        pd.DataFrame(rows).to_csv(out / "rows.csv", index=False)
        for r in new:
            man.time(f"n_{n}_K{r['K']}", r["seconds"])
            print(f"n={n} K={r['K']}: {r['seconds']:.1f}s (at K={K_PAPER}: {r['seconds_at_K500']:.1f}s), "
                  f"AUROC {r['auroc']:.3f}, p={r['p_value']:.3g}", flush=True)
        if n == NS[-1] and sum(r["seconds"] for r in new) < EXTRA_IF_UNDER:
            spent = time.time() - t0
            if spent + 2 * sum(r["seconds"] for r in new) <= args.total_budget:
                ns.append(N_EXTRA)
            else:
                print(f"n={N_EXTRA} skipped: {spent:.0f}s spent of the {args.total_budget:.0f}s budget", flush=True)
    df = pd.DataFrame(rows)
    exponent = fit_exponent(df)
    man.time("all_points", time.time() - t0)
    man.notes += [
        f"log-log slope of seconds at K={K_PAPER} on n over {len(df)} points: {exponent:.3f}",
        "points marked extrapolated were measured at the two small K and a line in K was read at K_paper",
        f"cpu: {cpu_name()}; torch threads: {threads}",
    ]
    man.write()
    print(f"exponent {exponent:.2f}; wrote {out / 'rows.csv'} in {time.time() - t0:.0f}s")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
