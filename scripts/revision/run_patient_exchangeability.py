#!/usr/bin/env python3
"""Check the exchangeability assumption behind the patient-level permutation on CheXpert.

Patients within one task stratum are treated as exchangeable, which assumes that the number of
images per patient and, among mixed patients, the share of positive images do not depend on the
attribute. Both are compared between the two groups of each binary attribute (sex, age 60 or
over at the first study) with a two-sided Mann-Whitney U test.

Usage: python scripts/revision/run_patient_exchangeability.py
"""

from __future__ import annotations

import sys
import time
from pathlib import Path

import pandas as pd
from scipy.stats import mannwhitneyu

sys.path.insert(0, str(Path(__file__).resolve().parent))
from common import DATA, Manifest, chexpert_meta, results_root  # noqa: E402


def main() -> int:
    out = results_root() / "patient_exchangeability"
    out.mkdir(parents=True, exist_ok=True)
    man = Manifest(out, "patient exchangeability check (images per patient, positive share) on CheXpert")
    man.add_data(DATA / "train.csv", DATA / "chexpert_multibackbone" / "race_mapping.csv")
    t0 = time.time()
    meta = chexpert_meta()
    pat = meta.groupby("patient_id").agg(n_images=("image_idx", "size"), sex=("sex", "first"),
                                         age_first=("age_first_study", "first"), pos_share=("task_label", "mean"))
    pat["female"] = (pat.sex == "Female").astype(int)
    pat["age60"] = (pat.age_first >= 60).astype(int)
    mixed = pat[(pat.pos_share > 0) & (pat.pos_share < 1)]
    rows = []
    for attr in ("female", "age60"):
        for quantity, frame, col in (("images_per_patient", pat, "n_images"), ("positive_share_mixed", mixed, "pos_share")):
            g0, g1 = frame[frame[attr] == 0][col], frame[frame[attr] == 1][col]
            p = mannwhitneyu(g0, g1, alternative="two-sided").pvalue
            rows.append({"attribute": attr, "quantity": quantity, "n0": len(g0), "n1": len(g1),
                         "median0": g0.median(), "median1": g1.median(), "mean0": g0.mean(), "mean1": g1.mean(),
                         "p_mannwhitney": p})
            print(f"{attr:7s} {quantity:22s} n {len(g0)}/{len(g1)} median {g0.median():.2f}/{g1.median():.2f} p={p:.3f}")
    pd.DataFrame(rows).to_csv(out / "patient_exchangeability.csv", index=False)
    man.time("wall_seconds", time.time() - t0)
    man.write()
    return 0


if __name__ == "__main__":
    sys.exit(main())
