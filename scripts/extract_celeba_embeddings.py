#!/usr/bin/env python3
"""Extract real CelebA embeddings using a pretrained ResNet-50.

Loads CelebA via torchvision, extracts 2048-dim embeddings from ResNet-50's
avgpool layer for a random subset of 10,000 images, and saves embeddings
plus metadata/attributes.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torchvision
from torch.utils.data import DataLoader, Subset
from torchvision import models, transforms

ROOT = Path(__file__).resolve().parent.parent
DATA_ROOT = ROOT / "data" / "celeba_real"
OUTPUT_DIR = ROOT / "data" / "celeba"

SUBSET_SIZE = 10_000
BATCH_SIZE = 64
SEED = 42

# Key attributes to include in metadata
META_ATTRS = [
    "Male",
    "Blond_Hair",
    "Heavy_Makeup",
    "Smiling",
    "Young",
    "Eyeglasses",
    "Attractive",
    "Big_Nose",
]


def main() -> int:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    # Device
    if torch.backends.mps.is_available():
        device = torch.device("mps")
        print("Using MPS (Apple Silicon)")
    else:
        device = torch.device("cpu")
        print("Using CPU")

    # --- Load CelebA dataset ---
    print("Loading CelebA dataset...")
    transform = transforms.Compose(
        [
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ]
    )

    dataset = torchvision.datasets.CelebA(
        root=str(DATA_ROOT),
        split="all",
        target_type="attr",
        transform=transform,
        download=False,
    )

    total = len(dataset)
    print(f"Total CelebA images: {total}")

    # All 40 attribute names (torchvision may include a trailing empty string)
    attr_names = [n for n in dataset.attr_names if n]
    print(f"Attributes ({len(attr_names)}): {attr_names[:10]}...")

    # --- Random subset ---
    rng = np.random.RandomState(SEED)
    indices = rng.choice(total, size=SUBSET_SIZE, replace=False)
    indices.sort()
    subset = Subset(dataset, indices)
    print(f"Selected random subset of {SUBSET_SIZE} images")

    # --- Build ResNet-50 feature extractor ---
    print("Loading pretrained ResNet-50...")
    resnet = models.resnet50(weights=models.ResNet50_Weights.IMAGENET1K_V1)
    # Remove classification head: replace fc with identity
    resnet.fc = torch.nn.Identity()
    resnet = resnet.to(device)
    resnet.eval()

    # --- Extract embeddings ---
    loader = DataLoader(subset, batch_size=BATCH_SIZE, shuffle=False, num_workers=0)

    all_embeddings = []
    all_attrs = []
    count = 0

    print("Extracting embeddings...")
    with torch.no_grad():
        for images, attrs in loader:
            images = images.to(device)
            features = resnet(images)  # (B, 2048)
            all_embeddings.append(features.cpu().numpy())

            # attrs shape: (B, 40) with values -1/1 -> convert to 0/1
            attrs_np = attrs.numpy()
            attrs_01 = ((attrs_np + 1) // 2).astype(np.int32)
            all_attrs.append(attrs_01)

            count += len(images)
            if count % 500 < BATCH_SIZE:
                print(f"  Processed {count}/{SUBSET_SIZE} images")

    embeddings = np.concatenate(all_embeddings, axis=0).astype(np.float32)
    attributes = np.concatenate(all_attrs, axis=0)

    print(f"Embeddings shape: {embeddings.shape}")
    print(f"Attributes shape: {attributes.shape}")

    # --- Build filenames for selected indices ---
    filenames = [dataset.filename[i] for i in indices]

    # --- Save embeddings ---
    emb_path = OUTPUT_DIR / "celeba_real_embeddings.npy"
    np.save(emb_path, embeddings)
    print(f"Saved embeddings: {emb_path}")

    # --- Save full attributes ---
    attr_df = pd.DataFrame(attributes, columns=attr_names)
    attr_df.insert(0, "image_id", filenames)
    attr_path = OUTPUT_DIR / "celeba_real_attributes.csv"
    attr_df.to_csv(attr_path, index=False)
    print(f"Saved attributes: {attr_path}")

    # --- Save metadata ---
    meta = pd.DataFrame()
    meta["image_id"] = filenames
    for col in META_ATTRS:
        meta[col] = attr_df[col].values
    meta["task_label"] = attr_df["Smiling"].values
    meta["group_label"] = attr_df["Male"].values
    meta_path = OUTPUT_DIR / "celeba_real_metadata.csv"
    meta.to_csv(meta_path, index=False)
    print(f"Saved metadata: {meta_path}")

    # --- Print summary statistics ---
    print("\n--- Summary ---")
    for col in META_ATTRS:
        pos = attr_df[col].sum()
        print(f"  {col}: {pos}/{SUBSET_SIZE} ({100*pos/SUBSET_SIZE:.1f}%)")

    print("\nDone!")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
