#!/usr/bin/env python3
"""Rebuild two MIMIC-CXR inputs from the source parquet files (needs pyarrow).

The MIMIC-CXR embeddings come from the coreset parquet files kept under
``mimic_cxr/source/`` in the ShortKit-ML-data dataset, strategy 5 (PathologyStratifiedClean, 1,999 images, seed 0), minus rows without
an in-hospital mortality label (1,491 subjects, one image each).

1. ``vit32_cls_embeddings_realigned.npy``: the stored ViT-B/32 file holds the first
   1,491 parquet rows, taken before unlabelled rows were dropped, so it does not
   match its metadata. Rows are rebuilt in metadata order, matched by image path.
2. ``mimic_race_raw.csv``: image path and the raw MIMIC race string, so the audit
   can code unknown, unable-to-obtain and declined as missing.

Usage: python scripts/revision/prepare_mimic_source.py [--qml PATH]
(PATH is a local copy of the source folder; by default it is downloaded from the hub)
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from common import DATA  # noqa: E402
from mimic_source import mimic_source_dir  # noqa: E402

VIT32 = "vit-base-patch32-224-embeddings/data_type5_n1999_seed0_vit_base_patch32_224_cls_embedding.parquet"
SIGLIP = "medsiglip-448-embeddings/data_type5_n1999_seed0_medsiglip_448.parquet"


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--qml", type=Path, default=None)
    args = ap.parse_args()
    args.qml = mimic_source_dir(args.qml)
    d = DATA / "mimic_cxr"
    meta = pd.read_csv(d / "vit32_cls_metadata.csv")

    src = pd.read_parquet(args.qml / VIT32).drop_duplicates("dicom_path").set_index("dicom_path")
    missing = ~meta.image_path.isin(src.index)
    if missing.any():
        raise SystemExit(f"{int(missing.sum())} metadata paths not in the ViT-B/32 parquet")
    X = np.stack(src.loc[meta.image_path, "embedding"].to_numpy()).astype(np.float32)
    old = np.load(d / "vit32_cls_embeddings.npy")
    # the stored file is the first len(meta) parquet rows, taken before unlabelled rows were dropped
    first = np.stack(pd.read_parquet(args.qml / VIT32).embedding.to_numpy()[: len(old)]).astype(np.float32)
    assert np.abs(first - old).max() < 1e-5
    np.save(d / "vit32_cls_embeddings_realigned.npy", X)
    same = float((np.abs(X - old).max(1) < 1e-5).mean())
    print(f"ViT-B/32: {len(X)} rows rebuilt in metadata order; {same:.3f} of stored rows were already in place")

    race = pd.read_parquet(args.qml / SIGLIP, columns=["dicom_path", "race"]).drop_duplicates("dicom_path")
    out = meta[["image_path"]].merge(race, left_on="image_path", right_on="dicom_path", how="left")
    assert out.race.notna().all() and len(out) == len(meta)
    out[["image_path", "race"]].rename(columns={"race": "race_raw"}).to_csv(d / "mimic_race_raw.csv", index=False)
    print(f"race: {len(out)} rows written")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
