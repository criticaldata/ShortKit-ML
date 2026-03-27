#!/usr/bin/env python3
"""Reformat existing CheXpert data files for the paper benchmark pipeline.

Reads:
  - data/chest_embeddings.npy        (2000, 512) float32
  - data/chest_labels.npy            (2000,) int
  - data/chest_group_labels.npy      (2000,) int  (0=ASIAN,1=BLACK,2=OTHER,3=WHITE)
  - data/chexpert_manifest.csv       2000 rows

Writes:
  - output/paper_benchmark/chexpert_embeddings/{backbone}_embeddings.npy
  - output/paper_benchmark/chexpert_embeddings/{backbone}_metadata.csv

The metadata CSV has columns: image_path, task_label, race, sex, age
(matching what PaperBenchmarkRunner._load_backbone_embeddings expects).
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd

GROUP_TO_RACE = {0: "ASIAN", 1: "BLACK", 2: "OTHER", 3: "WHITE"}


def setup(
    *,
    repo_root: Path,
    backbone: str = "medclip",
    output_dir: Path | None = None,
) -> dict[str, Path]:
    """Create backbone embedding artifacts from existing data files."""
    data_dir = repo_root / "data"
    if output_dir is None:
        output_dir = repo_root / "output" / "paper_benchmark" / "chexpert_embeddings"
    output_dir.mkdir(parents=True, exist_ok=True)

    # Load existing data
    embeddings = np.load(str(data_dir / "chest_embeddings.npy"))
    labels = np.load(str(data_dir / "chest_labels.npy"))
    group_labels = np.load(str(data_dir / "chest_group_labels.npy"))
    manifest = pd.read_csv(data_dir / "chexpert_manifest.csv")

    print(f"Loaded embeddings: {embeddings.shape} {embeddings.dtype}")
    print(f"Loaded labels: {labels.shape}, unique={np.unique(labels)}")
    print(f"Loaded groups: {group_labels.shape}, unique={np.unique(group_labels)}")
    print(f"Manifest rows: {len(manifest)}, columns: {list(manifest.columns)}")

    # Validate alignment
    assert len(embeddings) == len(labels) == len(group_labels) == len(manifest), (
        f"Length mismatch: emb={len(embeddings)}, lab={len(labels)}, "
        f"grp={len(group_labels)}, manifest={len(manifest)}"
    )

    # Verify group_labels match manifest race column
    manifest_race_upper = manifest["race"].str.upper()
    for g, race in GROUP_TO_RACE.items():
        mask = group_labels == g
        manifest_races = manifest_race_upper[mask].unique()
        assert (
            len(manifest_races) == 1 and manifest_races[0] == race
        ), f"Group {g} expected race '{race}', got {manifest_races}"

    # Verify labels match manifest
    assert (labels == manifest["task_label"].to_numpy()).all(), "Label mismatch"

    # Create metadata CSV with exactly the columns the pipeline expects
    metadata = manifest[["image_path", "task_label", "race", "sex", "age"]].copy()
    metadata["race"] = metadata["race"].str.upper()

    # Save
    emb_path = output_dir / f"{backbone}_embeddings.npy"
    meta_path = output_dir / f"{backbone}_metadata.csv"

    np.save(str(emb_path), embeddings)
    metadata.to_csv(meta_path, index=False)

    print(f"\nSaved {backbone} artifacts:")
    print(f"  Embeddings: {emb_path}  ({embeddings.shape})")
    print(f"  Metadata:   {meta_path}  ({len(metadata)} rows)")

    # Print summary statistics
    print("\n--- Data Summary ---")
    print(f"Samples: {len(metadata)}")
    print(f"Embedding dim: {embeddings.shape[1]}")
    print(f"Label distribution: {dict(metadata['task_label'].value_counts().sort_index())}")
    print(f"Race distribution: {dict(metadata['race'].value_counts())}")
    print(f"Sex distribution: {dict(metadata['sex'].value_counts())}")

    # Age bins (matching paper_runner defaults: [40, 60, 80])
    bins = [-np.inf, 40, 60, 80, np.inf]
    bin_labels = ["<40", "40-59", "60-79", ">=80"]
    age_bin = pd.cut(metadata["age"].astype(float), bins=bins, labels=bin_labels, right=False)
    print(f"Age bin distribution: {dict(age_bin.value_counts())}")

    return {"embeddings": emb_path, "metadata": meta_path}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Set up CheXpert data for paper benchmark")
    parser.add_argument(
        "--root",
        type=Path,
        default=Path(__file__).resolve().parent.parent,
        help="Repository root directory",
    )
    parser.add_argument("--backbone", default="medclip", help="Backbone name for the output files")
    parser.add_argument("--output-dir", type=Path, default=None, help="Override output directory")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    setup(repo_root=args.root, backbone=args.backbone, output_dir=args.output_dir)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
