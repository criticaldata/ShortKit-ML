#!/usr/bin/env python3
"""How representative the audited CheXpert sample is of the full training file.

The audited sample is the first 2,000 images in the order of the official
train.csv. This script compares it with the whole file on sex, age, view and
race, so the paper can state the difference. Sex, age and view are read from
train.csv on both sides; race per patient from the demographics file with the
same coding as the audit (common._code_chexpert_race, where Unknown and
Patient Refused are missing). Age is the per-image age in train.csv on both
sides, since the audit's per-patient age at first study has no counterpart in
the full file.
"""

from __future__ import annotations

import sys
import time
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
from common import (  # noqa: E402
    CHEXPERT_JOIN_NOTE,
    DATA,
    Manifest,
    _chexpert_demo,
    _code_chexpert_race,
    chexpert_meta,
    results_root,
)

N_SAMPLE = 2000


def full_train() -> pd.DataFrame:
    train = pd.read_csv(DATA / "train.csv")
    df = pd.DataFrame(
        {
            "path": train["Path"],
            "patient_id": train["Path"].str.extract(r"(patient\d+)")[0],
            "sex": train["Sex"],
            "age": train["Age"].astype(float),
            "view": train["Frontal/Lateral"],
        }
    )
    demo = _chexpert_demo().drop_duplicates("PATIENT")
    raw = df[["patient_id"]].merge(demo, left_on="patient_id", right_on="PATIENT", how="left", validate="many_to_one")
    assert len(raw) == len(df)
    df["race"] = [_code_chexpert_race(v) for v in raw["PRIMARY_RACE"]]
    return df


def summarise(df: pd.DataFrame) -> dict[str, float]:
    known = df["race"].notna()
    return {
        "n_images": int(len(df)),
        "n_patients": int(df["patient_id"].nunique()),
        "female_pct": float((df["sex"] == "Female").mean() * 100),
        "age_median": float(df["age"].median()),
        "age_mean": float(df["age"].mean()),
        "lateral_pct": float((df["view"] == "Lateral").mean() * 100),
        "race_known_n": int(known.sum()),
        "white_pct_of_known": float((df.loc[known, "race"] == "WHITE").mean() * 100),
    }


def main() -> int:
    out = results_root() / "sample_vs_full"
    out.mkdir(parents=True, exist_ok=True)
    man = Manifest(out, "Audited CheXpert sample against the full train.csv")
    man.config = {"n_sample": N_SAMPLE, "race_known": "coded by common._code_chexpert_race"}
    man.add_data(DATA / "train.csv", DATA / "CHEXPERT DEMO.xlsx",
                 DATA / "chexpert_multibackbone" / "chexpert_manifest.csv",
                 DATA / "chexpert_multibackbone" / "race_mapping.csv")
    t0 = time.time()
    sample = chexpert_meta()
    full = full_train()
    assert set(sample["path"]) == set(full["path"].head(N_SAMPLE)), "sample is not the first 2,000 rows of train.csv"
    s, f = summarise(sample), summarise(full)
    df = pd.DataFrame({"statistic": list(s), "sample": list(s.values()), "full": [f[k] for k in s]})
    df.to_csv(out / "sample_vs_full.csv", index=False)
    man.notes += [CHEXPERT_JOIN_NOTE,
                  "Age is the per-image Age column of train.csv on both sides, not the audit's "
                  "age at first study. Sex and view from train.csv; race per patient from "
                  "CHEXPERT DEMO.xlsx, white_pct_of_known over images with a coded race."]
    man.time("all", time.time() - t0)
    man.write()
    print(df.to_string(index=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
