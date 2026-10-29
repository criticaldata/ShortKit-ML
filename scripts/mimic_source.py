"""Locate the MIMIC-CXR source parquet files (coreset strategy 5, seed 0).

They live in the gated dataset MITCriticalData/ShortKit-ML-data under
``mimic_cxr/source/`` with the layout of the extraction repository they came from.
With no path given, that folder is downloaded (or reused from the hub cache).
"""

from __future__ import annotations

from pathlib import Path

REPO = "MITCriticalData/ShortKit-ML-data"
SOURCE_PREFIX = "mimic_cxr/source"
PARQUETS = [
    "rad-dino-embeddings/data_type5_n1998_seed0_rad_dino.parquet",
    "vit-base-patch16-224-embeddings/data_type5_n1999_seed0_vit_base_patch16_224_cls_embedding.parquet",
    "vit-base-patch32-224-embeddings/data_type5_n1999_seed0_vit_base_patch32_224_cls_embedding.parquet",
    "medsiglip-448-embeddings/data_type5_n1999_seed0_medsiglip_448.parquet",
]


def mimic_source_dir(path: Path | None = None) -> Path:
    if path is not None:
        return path
    from huggingface_hub import snapshot_download

    root = snapshot_download(REPO, repo_type="dataset", allow_patterns=[f"{SOURCE_PREFIX}/*"])
    return Path(root) / SOURCE_PREFIX
