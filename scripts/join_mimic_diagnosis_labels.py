#!/usr/bin/env python3
"""Join MIMIC-CXR per-diagnosis labels onto existing embedding metadata.

The existing metadata CSV was produced by setup_mimic_cxr_data.py which
drops rows where hospital_expire_flag is NaN.  We must apply the same
filter to the parquet before joining so row counts align.

Reads:
  - data/mimic_cxr/mimic-cxr-2.0.0-chexpert.csv  (227k rows, 14 diagnoses)
  - qml-mimic-cxr-embeddings parquet files          (subject_id, study_id)
  - data/mimic_cxr/{backbone}_metadata.csv           (demographics + mortality)

Writes:
  - data/mimic_cxr/{backbone}_metadata_dx.csv        (+ 14 diagnosis columns)
  - data/mimic_cxr/{backbone}_metadata.csv            (updated in-place)
"""

from __future__ import annotations

import argparse
import shutil
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
MIMIC_DIR = ROOT / "data" / "mimic_cxr"
MIMIC_REPO = Path.home() / "Desktop" / "repositories" / "qml-mimic-cxr-embeddings"

BACKBONES = {
    "rad_dino": "rad-dino-embeddings/data_type5_n1998_seed0_rad_dino.parquet",
    "vit16_cls": "vit-base-patch16-224-embeddings/data_type5_n1999_seed0_vit_base_patch16_224_cls_embedding.parquet",
    "medsiglip": "medsiglip-448-embeddings/data_type5_n1999_seed0_medsiglip_448.parquet",
}

DIAGNOSES = [
    "Atelectasis",
    "Cardiomegaly",
    "Consolidation",
    "Edema",
    "Enlarged Cardiomediastinum",
    "Fracture",
    "Lung Lesion",
    "Lung Opacity",
    "No Finding",
    "Pleural Effusion",
    "Pleural Other",
    "Pneumonia",
    "Pneumothorax",
    "Support Devices",
]


def main() -> int:
    parser = argparse.ArgumentParser(description="Join MIMIC-CXR diagnosis labels")
    parser.add_argument("--mimic-dir", type=Path, default=MIMIC_DIR)
    parser.add_argument("--mimic-repo", type=Path, default=MIMIC_REPO)
    args = parser.parse_args()

    labels_path = args.mimic_dir / "mimic-cxr-2.0.0-chexpert.csv"
    if not labels_path.exists():
        print(f"ERROR: Labels file not found at {labels_path}")
        return 1

    # Load diagnosis labels (study-level: one row per study)
    dx = pd.read_csv(labels_path)
    print(f"Loaded diagnosis labels: {dx.shape}")

    # Show diagnosis prevalence
    print("\nDiagnosis prevalence (positive=1.0):")
    for col in DIAGNOSES:
        pos = (dx[col] == 1.0).sum()
        total = dx[col].notna().sum()
        print(f"  {col:30s}: {pos:6d} / {total:6d} ({100*pos/max(total,1):.1f}%)")

    for backbone, parquet_rel in BACKBONES.items():
        print(f"\n{'='*60}")
        print(f"  Processing: {backbone}")
        print(f"{'='*60}")

        parquet_path = args.mimic_repo / parquet_rel
        if not parquet_path.exists():
            print(f"  [SKIP] Parquet not found: {parquet_path}")
            continue

        # Read parquet: need subject_id, study_id, and hospital_expire_flag
        # to replicate the same filtering as setup_mimic_cxr_data.py
        pq = pd.read_parquet(
            parquet_path,
            columns=["subject_id", "study_id", "hospital_expire_flag"],
        )
        print(f"  Parquet rows (raw): {len(pq)}")

        # Apply same filter as setup_mimic_cxr_data.py
        pq = pq.dropna(subset=["hospital_expire_flag"]).reset_index(drop=True)
        print(f"  Parquet rows (after dropna): {len(pq)}")

        # Read existing metadata
        meta_path = args.mimic_dir / f"{backbone}_metadata.csv"
        if not meta_path.exists():
            print(f"  [SKIP] Metadata not found: {meta_path}")
            continue

        meta = pd.read_csv(meta_path)
        print(f"  Metadata rows: {len(meta)}")

        assert len(meta) == len(
            pq
        ), f"Length mismatch: meta={len(meta)}, filtered parquet={len(pq)}"

        # Join parquet (filtered) with diagnosis labels on study_id
        merged = pq.merge(dx, on=["subject_id", "study_id"], how="left")
        has_dx = merged[DIAGNOSES].notna().any(axis=1).sum()
        print(f"  Rows with diagnosis labels: {has_dx}/{len(merged)}")

        # Add diagnosis columns to metadata
        for col in DIAGNOSES:
            meta[col] = merged[col].values

        # Backup original
        backup = meta_path.with_name(f"{backbone}_metadata_orig.csv")
        if not backup.exists():
            shutil.copy2(meta_path, backup)
            print(f"  Backed up original to: {backup.name}")

        # Save with diagnoses
        dx_path = args.mimic_dir / f"{backbone}_metadata_dx.csv"
        meta.to_csv(dx_path, index=False)
        print(f"  Saved: {dx_path.name} ({len(meta)} rows, {len(meta.columns)} cols)")

        # Also update main metadata
        meta.to_csv(meta_path, index=False)
        print(f"  Updated: {meta_path.name}")

        # Per-diagnosis counts
        print(f"\n  Diagnosis counts (positive=1.0) in {backbone} subset:")
        for col in DIAGNOSES:
            pos = (meta[col] == 1.0).sum()
            print(f"    {col:30s}: {pos}")

    print("\nDone. Diagnosis labels joined successfully.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
