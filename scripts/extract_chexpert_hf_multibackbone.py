#!/usr/bin/env python3
"""Extract CheXpert embeddings from 3 backbones using HuggingFace dataset.

Phase 1: Stream 2k samples from HF, cache images + metadata to disk.
Phase 2: Extract embeddings from 3 backbones in parallel (multiprocessing).

NOTE: This dataset has sex + age + 14 diagnoses but NOT race.
      For race you need CheXpert Plus from Stanford AIMI.

Usage:
    python scripts/extract_chexpert_hf_multibackbone.py --n-samples 2000 --device mps
"""

from __future__ import annotations

import argparse
import pickle
from multiprocessing import Process
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
OUTPUT_DIR = ROOT / "data" / "chexpert_multibackbone"
CACHE_DIR = OUTPUT_DIR / "_cache"

DIAGNOSIS_COLS = [
    "No Finding",
    "Enlarged Cardiomediastinum",
    "Cardiomegaly",
    "Lung Opacity",
    "Lung Lesion",
    "Edema",
    "Consolidation",
    "Pneumonia",
    "Atelectasis",
    "Pneumothorax",
    "Pleural Effusion",
    "Pleural Other",
    "Fracture",
    "Support Devices",
]

LABEL_MAP = {0: float("nan"), 1: float("nan"), 2: 0.0, 3: 1.0}
BACKBONES = ["resnet50", "densenet121", "vit_b_16"]


# ── Phase 1: Download & cache ───────────────────────────────────────────


def download_and_cache(n_samples: int, seed: int, output_dir: Path) -> tuple[Path, Path]:
    """Stream CheXpert from HF, take first n_samples, cache to disk."""
    cache_dir = output_dir / "_cache"
    images_pkl = cache_dir / "images.pkl"
    meta_csv = cache_dir / "metadata.csv"

    if images_pkl.exists() and meta_csv.exists():
        meta = pd.read_csv(meta_csv)
        if len(meta) == n_samples:
            print(f"  Cache hit: {len(meta)} samples already downloaded")
            return images_pkl, meta_csv

    cache_dir.mkdir(parents=True, exist_ok=True)

    from datasets import load_dataset

    print(f"Streaming CheXpert from HuggingFace (first {n_samples} samples)...")
    ds = load_dataset("danjacobellis/chexpert", split="train", streaming=True)

    sex_map = {0: "Female", 1: "Male"}
    rows = []
    images = []
    for i, sample in enumerate(ds):
        if i >= n_samples:
            break
        images.append(sample["image"])
        row = {
            "image_idx": i,
            "task_label": 1 if sample.get("Pneumothorax", 0) == 3 else 0,
            "sex": sex_map.get(sample.get("Sex", -1), "Unknown"),
            "age": sample.get("Age", None),
        }
        for col in DIAGNOSIS_COLS:
            row[col] = LABEL_MAP.get(sample.get(col, 0), float("nan"))
        rows.append(row)
        if (i + 1) % 500 == 0:
            print(f"  Downloaded {i+1}/{n_samples}")

    print(f"  Downloaded {len(images)} samples")

    meta = pd.DataFrame(rows)
    meta["age_bin"] = pd.cut(
        meta["age"].astype(float),
        bins=[-np.inf, 40, 60, 80, np.inf],
        labels=["<40", "40-60", "60-80", "80+"],
        right=False,
    ).astype(str)

    # Cache
    with open(images_pkl, "wb") as f:
        pickle.dump(images, f)
    meta.to_csv(meta_csv, index=False)

    print(f"  Cached to {cache_dir}")
    print(f"  Sex: {dict(meta['sex'].value_counts())}")
    print(f"  Age bins: {dict(meta['age_bin'].value_counts())}")

    return images_pkl, meta_csv


# ── Phase 2: Extract one backbone ────────────────────────────────────────


def extract_one_backbone(
    backbone: str,
    images_pkl: Path,
    meta_csv: Path,
    output_dir: Path,
    device: str,
    batch_size: int,
) -> None:
    """Extract embeddings for a single backbone. Runs in its own process."""
    import pickle

    import torch
    from torchvision import models

    print(f"[{backbone}] Loading cached images...")
    with open(images_pkl, "rb") as f:
        images = pickle.load(f)
    meta = pd.read_csv(meta_csv)

    print(f"[{backbone}] Building model on {device}...")
    if backbone == "resnet50":
        model = models.resnet50(weights=models.ResNet50_Weights.DEFAULT)
        model.fc = torch.nn.Identity()
        transform = models.ResNet50_Weights.DEFAULT.transforms()
    elif backbone == "densenet121":
        model = models.densenet121(weights=models.DenseNet121_Weights.DEFAULT)
        model.classifier = torch.nn.Identity()
        transform = models.DenseNet121_Weights.DEFAULT.transforms()
    elif backbone == "vit_b_16":
        model = models.vit_b_16(weights=models.ViT_B_16_Weights.DEFAULT)
        model.heads = torch.nn.Identity()
        transform = models.ViT_B_16_Weights.DEFAULT.transforms()
    else:
        raise ValueError(f"Unknown backbone: {backbone}")

    model.eval()
    model.to(device)

    print(f"[{backbone}] Extracting {len(images)} samples...")
    all_feats = []
    n_batches = (len(images) - 1) // batch_size + 1
    for start in range(0, len(images), batch_size):
        end = min(start + batch_size, len(images))
        tensors = [transform(img.convert("RGB")) for img in images[start:end]]
        batch = torch.stack(tensors).to(device)
        with torch.no_grad():
            out = model(batch).detach().cpu().numpy()
        all_feats.append(out)
        batch_num = start // batch_size + 1
        if batch_num % 10 == 0 or batch_num == n_batches:
            print(f"[{backbone}] Batch {batch_num}/{n_batches}")

    embeddings = np.vstack(all_feats).astype(np.float32)
    print(f"[{backbone}] Embeddings shape: {embeddings.shape}")

    emb_path = output_dir / f"{backbone}_embeddings.npy"
    meta_path = output_dir / f"{backbone}_metadata.csv"
    np.save(str(emb_path), embeddings)
    meta.to_csv(meta_path, index=False)
    print(f"[{backbone}] DONE -> {emb_path.name}")


# ── Main ──────────────────────────────────────────────────────────────────


def main() -> int:
    parser = argparse.ArgumentParser(description="Extract CheXpert multi-backbone embeddings")
    parser.add_argument("--n-samples", type=int, default=2000)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--device", default="cpu", help="cpu, cuda, or mps")
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--output-dir", type=Path, default=OUTPUT_DIR)
    parser.add_argument("--backbones", default=",".join(BACKBONES))
    parser.add_argument("--parallel", action="store_true", help="Run backbones in parallel")
    args = parser.parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    backbones = [b.strip() for b in args.backbones.split(",")]

    # Phase 1: Download once
    images_pkl, meta_csv = download_and_cache(args.n_samples, args.seed, args.output_dir)

    # Phase 2: Extract backbones
    if args.parallel and len(backbones) > 1:
        print(f"\nLaunching {len(backbones)} backbone extractions in parallel...")
        procs = []
        for bb in backbones:
            p = Process(
                target=extract_one_backbone,
                args=(bb, images_pkl, meta_csv, args.output_dir, args.device, args.batch_size),
            )
            p.start()
            procs.append((bb, p))
        for bb, p in procs:
            p.join()
            if p.exitcode != 0:
                print(f"[{bb}] FAILED (exit code {p.exitcode})")
            else:
                print(f"[{bb}] completed successfully")
    else:
        for bb in backbones:
            extract_one_backbone(
                bb, images_pkl, meta_csv, args.output_dir, args.device, args.batch_size
            )

    # Save manifest
    meta = pd.read_csv(meta_csv)
    manifest_path = args.output_dir / "chexpert_manifest.csv"
    meta.to_csv(manifest_path, index=False)
    print(f"\nManifest: {manifest_path} ({len(meta)} rows)")
    print(f"Done. All backbones extracted to {args.output_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
