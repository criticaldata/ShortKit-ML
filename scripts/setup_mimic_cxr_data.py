#!/usr/bin/env python3
"""Prepare MIMIC-CXR embedding data for the shortcut detection benchmark.

Reads parquet files from the qml-mimic-cxr-embeddings repository and creates
standardised numpy arrays + metadata CSVs for three backbones:
  - RAD-DINO  (768-dim)
  - ViT-B/16 CLS  (768-dim)
  - MedSigLIP-448  (1152-dim)

Outputs are written to data/mimic_cxr/:
  {backbone}_embeddings.npy, {backbone}_metadata.csv, mimic_cxr_manifest.csv
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------
ROOT = Path(__file__).resolve().parent.parent
MIMIC_REPO = Path.home() / "Desktop" / "repositories" / "qml-mimic-cxr-embeddings"
OUTPUT_DIR = ROOT / "data" / "mimic_cxr"

BACKBONES = {
    "rad_dino": {
        "parquet": "rad-dino-embeddings/data_type5_n1998_seed0_rad_dino.parquet",
        "dim": 768,
    },
    "vit16_cls": {
        "parquet": "vit-base-patch16-224-embeddings/data_type5_n1999_seed0_vit_base_patch16_224_cls_embedding.parquet",
        "dim": 768,
    },
    "medsiglip": {
        "parquet": "medsiglip-448-embeddings/data_type5_n1999_seed0_medsiglip_448.parquet",
        "dim": 1152,
    },
}

AGE_BINS = [0, 40, 60, 80, 150]
AGE_BIN_LABELS = ["<40", "40-60", "60-80", "80+"]


# ---------------------------------------------------------------------------
# Race simplification
# ---------------------------------------------------------------------------
def simplify_race(race: str | None) -> str:
    """Map detailed MIMIC race strings to WHITE / BLACK / ASIAN / OTHER."""
    if race is None or pd.isna(race):
        return "OTHER"
    race = str(race).upper().strip()
    if race.startswith("WHITE"):
        return "WHITE"
    if race.startswith("BLACK"):
        return "BLACK"
    if race.startswith("ASIAN"):
        return "ASIAN"
    return "OTHER"


# ---------------------------------------------------------------------------
# Process one backbone
# ---------------------------------------------------------------------------
def process_backbone(
    name: str,
    parquet_path: Path,
    expected_dim: int,
    output_dir: Path,
) -> pd.DataFrame:
    """Read parquet, extract embeddings + metadata, save to disk."""
    print(f"\n{'─'*60}")
    print(f"  Processing backbone: {name}")
    print(f"  Parquet: {parquet_path}")
    print(f"{'─'*60}")

    df = pd.read_parquet(parquet_path)
    print(f"  Rows loaded: {len(df)}")

    # Drop rows with missing task label
    df = df.dropna(subset=["hospital_expire_flag"]).copy()
    print(f"  Rows after dropping NaN labels: {len(df)}")

    # Extract embeddings
    embeddings = np.stack(df["embedding"].values).astype(np.float32)
    assert (
        embeddings.shape[1] == expected_dim
    ), f"Expected {expected_dim}-dim, got {embeddings.shape[1]}-dim"
    print(f"  Embeddings shape: {embeddings.shape}")

    # Build metadata
    meta = pd.DataFrame(
        {
            "image_path": df["dicom_path"].values,
            "task_label": df["hospital_expire_flag"].astype(int).values,
            "race": df["race"].apply(simplify_race).values,
            "sex": df["gender"].map({"M": "Male", "F": "Female"}).values,
            "age": df["anchor_age"].values,
        }
    )
    meta["age_bin"] = pd.cut(
        meta["age"].astype(float),
        bins=AGE_BINS,
        labels=AGE_BIN_LABELS,
        right=False,
    ).astype(str)

    # Save
    emb_path = output_dir / f"{name}_embeddings.npy"
    meta_path = output_dir / f"{name}_metadata.csv"
    np.save(str(emb_path), embeddings)
    meta.to_csv(meta_path, index=False)

    print(f"  Saved embeddings -> {emb_path}")
    print(f"  Saved metadata   -> {meta_path}")

    # Summary
    print(f"\n  Label distribution:  {dict(meta['task_label'].value_counts().sort_index())}")
    print(f"  Race distribution:   {dict(meta['race'].value_counts())}")
    print(f"  Sex distribution:    {dict(meta['sex'].value_counts())}")
    print(f"  Age-bin distribution:{dict(meta['age_bin'].value_counts())}")

    # Return manifest rows
    meta_manifest = meta.copy()
    meta_manifest["backbone"] = name
    return meta_manifest


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main() -> int:
    parser = argparse.ArgumentParser(description="Prepare MIMIC-CXR data for benchmark")
    parser.add_argument(
        "--mimic-repo", type=Path, default=MIMIC_REPO, help="Path to qml-mimic-cxr-embeddings repo"
    )
    parser.add_argument("--output-dir", type=Path, default=OUTPUT_DIR)
    args = parser.parse_args()

    args.output_dir.mkdir(parents=True, exist_ok=True)

    print("=" * 60)
    print(" MIMIC-CXR Data Setup")
    print("=" * 60)
    print(f"  Source: {args.mimic_repo}")
    print(f"  Output: {args.output_dir}")

    manifest_parts = []
    for name, cfg in BACKBONES.items():
        parquet_path = args.mimic_repo / cfg["parquet"]
        if not parquet_path.exists():
            print(f"\n  [SKIP] {name}: parquet not found at {parquet_path}")
            continue
        part = process_backbone(name, parquet_path, cfg["dim"], args.output_dir)
        manifest_parts.append(part)

    if manifest_parts:
        manifest = pd.concat(manifest_parts, ignore_index=True)
        manifest_path = args.output_dir / "mimic_cxr_manifest.csv"
        manifest.to_csv(manifest_path, index=False)
        print(f"\nCombined manifest: {manifest_path}  ({len(manifest)} rows)")

    print("\nMIMIC-CXR data setup complete.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
