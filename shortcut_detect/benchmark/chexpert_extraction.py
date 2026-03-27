"""CheXpert embedding extraction utilities for paper benchmarks."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

REQUIRED_MANIFEST_COLUMNS = ("image_path", "task_label", "race", "sex", "age")
SUPPORTED_BACKBONES = ("resnet50", "densenet121", "vit_b_16")


@dataclass(frozen=True)
class ExtractionConfig:
    manifest_path: str
    output_dir: str
    backbones: list[str]
    batch_size: int = 32
    num_workers: int = 0
    device: str = "cpu"


def load_and_validate_manifest(path: str | Path) -> pd.DataFrame:
    path_obj = Path(path)
    if not path_obj.exists():
        raise FileNotFoundError(f"Manifest not found: {path_obj}")
    df = pd.read_csv(path_obj)
    missing = sorted(set(REQUIRED_MANIFEST_COLUMNS) - set(df.columns))
    if missing:
        raise ValueError(f"Manifest missing required columns: {missing}")
    if df.empty:
        raise ValueError("Manifest has no rows")
    if df["image_path"].isna().any():
        raise ValueError("Manifest contains null image_path values")
    if df["task_label"].isna().any():
        raise ValueError("Manifest contains null task_label values")
    return df


def _get_torch_and_vision() -> tuple[Any, Any, Any]:
    try:
        import torch
        import torchvision.transforms as T
        from torchvision import models
    except Exception as exc:  # pragma: no cover
        raise ImportError(
            "CheXpert extraction requires torch, torchvision, and pillow dependencies"
        ) from exc
    return torch, T, models


def _build_model(backbone: str, device: str):
    torch, _, models = _get_torch_and_vision()
    backbone = backbone.lower()
    if backbone not in SUPPORTED_BACKBONES:
        raise ValueError(f"Unsupported backbone: {backbone}. Supported: {SUPPORTED_BACKBONES}")

    if backbone == "resnet50":
        model = models.resnet50(weights=models.ResNet50_Weights.DEFAULT)
        model.fc = torch.nn.Identity()
    elif backbone == "densenet121":
        model = models.densenet121(weights=models.DenseNet121_Weights.DEFAULT)
        model.classifier = torch.nn.Identity()
    else:  # vit_b_16
        model = models.vit_b_16(weights=models.ViT_B_16_Weights.DEFAULT)
        model.heads = torch.nn.Identity()

    model.eval()
    model.to(device)
    return model


def _build_transform(backbone: str):
    _, T, models = _get_torch_and_vision()
    backbone = backbone.lower()
    if backbone == "resnet50":
        return models.ResNet50_Weights.DEFAULT.transforms()
    if backbone == "densenet121":
        return models.DenseNet121_Weights.DEFAULT.transforms()
    return models.ViT_B_16_Weights.DEFAULT.transforms()


class _ManifestDataset:
    def __init__(self, df: pd.DataFrame, transform):
        from PIL import Image

        self.df = df.reset_index(drop=True)
        self.transform = transform
        self.Image = Image

    def __len__(self) -> int:
        return len(self.df)

    def __getitem__(self, idx: int):
        row = self.df.iloc[idx]
        path = str(row["image_path"])
        img = self.Image.open(path).convert("RGB")
        x = self.transform(img)
        return x, idx


def extract_embeddings(config: ExtractionConfig) -> dict[str, str]:
    torch, _, _ = _get_torch_and_vision()
    df = load_and_validate_manifest(config.manifest_path)

    out_dir = Path(config.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    outputs: dict[str, str] = {}

    for backbone in config.backbones:
        transform = _build_transform(backbone)
        model = _build_model(backbone, device=config.device)
        dataset = _ManifestDataset(df, transform)
        loader = torch.utils.data.DataLoader(
            dataset,
            batch_size=config.batch_size,
            shuffle=False,
            num_workers=config.num_workers,
        )

        feats: list[np.ndarray] = []
        order: list[int] = []
        with torch.no_grad():
            for xb, idx in loader:
                xb = xb.to(config.device)
                out = model(xb)
                if hasattr(out, "detach"):
                    out = out.detach().cpu().numpy()
                else:  # pragma: no cover
                    out = np.asarray(out)
                feats.append(out)
                order.extend(idx.cpu().numpy().tolist())

        emb = np.vstack(feats)
        order_arr = np.asarray(order)
        if emb.shape[0] != len(df):
            raise RuntimeError(f"Extraction row mismatch for {backbone}")

        inv = np.argsort(order_arr)
        emb = emb[inv]

        emb_path = out_dir / f"{backbone}_embeddings.npy"
        meta_path = out_dir / f"{backbone}_metadata.csv"

        np.save(str(emb_path), emb.astype(np.float32))
        meta_df = df.copy()
        meta_df.to_csv(meta_path, index=False)
        outputs[f"{backbone}_embeddings"] = str(emb_path)
        outputs[f"{backbone}_metadata"] = str(meta_path)

    return outputs
