#!/usr/bin/env python3
"""Negative control for the audit on patient-clustered data (CheXpert).

A fake binary attribute is drawn per patient, independent of everything, so
I(E; A | Y) = 0 but images of one patient share A and have correlated
embeddings. The audit is run with patient-level permutation and grouped CV
(the revision protocol) and with image-level permutation and plain CV, and the
share of draws with p < 0.05 is recorded per detector.
"""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
from common import DATA, Manifest, chexpert_paths, load_chexpert, results_root  # noqa: E402


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--draws", type=int, default=30)
    ap.add_argument("--K", type=int, default=200)
    ap.add_argument("--backbone", default="rad_dino")
    ap.add_argument("--threads", type=int, default=4)
    args = ap.parse_args()
    import torch

    torch.set_num_threads(args.threads)
    from shortcut_detect.audit import AuditConfig, conditional_audit

    out = results_root() / "negative_control"
    man = Manifest(out, "patient-level fake attribute negative control")
    man.config = vars(args)
    man.add_data(chexpert_paths(args.backbone)[0], DATA / "train.csv",
                 DATA / "chexpert_multibackbone" / "race_mapping.csv")
    X, meta = load_chexpert(args.backbone)
    y = meta["task_label"].to_numpy()
    pid = meta["patient_id"].to_numpy()
    up = np.unique(pid)
    rows = []
    t0 = time.time()
    for s in range(args.draws):
        rng = np.random.default_rng(500 + s)
        fake = dict(zip(up, rng.integers(0, 2, len(up)), strict=True))
        a = np.array([fake[p] for p in pid])
        for mode, grp in (("patient_permutation_grouped_cv", pid), ("image_permutation_plain_cv", None)):
            r = conditional_audit(X, a, y, AuditConfig(n_permutations=args.K, n_bootstrap=0, seed=s), groups=grp)
            for d, v in r.results.items():
                rows.append({"draw": s, "mode": mode, "detector": d, "statistic": v.statistic,
                             "p_value": v.p_value, "flag_bh": v.flag})
        print(s, f"{time.time() - t0:.0f}s", flush=True)
    df = pd.DataFrame(rows)
    df.to_csv(out / "negative_control.csv", index=False)
    summ = df.assign(rej=df.p_value < 0.05).groupby(["mode", "detector"])["rej"].mean().unstack(0)
    summ.to_csv(out / "negative_control_fpr.csv")
    print(summ)
    man.time("all", time.time() - t0)
    man.write()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
