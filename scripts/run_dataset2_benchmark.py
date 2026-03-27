#!/usr/bin/env python3
"""Bootstrap CheXpert embeddings and run the paper benchmark for Dataset 2."""

from __future__ import annotations

import argparse
import json
import pickle
from collections.abc import Sequence
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from shortcut_detect.benchmark.chexpert_extraction import load_and_validate_manifest
from shortcut_detect.benchmark.paper_runner import PaperBenchmarkConfig, PaperBenchmarkRunner


def _stack_embeddings(items: Sequence[Any]) -> np.ndarray:
    arrays = [np.asarray(item) for item in items]
    if not arrays:
        return np.empty((0, 0), dtype=np.float32)
    shapes = {arr.shape for arr in arrays}
    if len(shapes) != 1:
        raise ValueError(f"Embedding shapes disagree: {shapes}")
    return np.vstack(arrays).astype(np.float32)


def _canonical_path(value: Any) -> str:
    return Path(value).as_posix()


def _metadata_from_manifest(paths: Sequence[Any], manifest_df: pd.DataFrame) -> pd.DataFrame:
    df = pd.DataFrame({"image_path": [_canonical_path(p) for p in paths]})
    merged = df.merge(manifest_df, on="image_path", how="left", validate="one_to_one")
    required = ["image_path", "task_label", "race", "sex", "age"]
    missing = merged[required].isnull().any(axis=1)
    if missing.any():
        raise ValueError(
            f"Manifest lacks metadata for {missing.sum()} rows: {merged.loc[missing, 'image_path'].tolist()}"
        )
    return merged[required].reset_index(drop=True)


def _save_artifacts(X: np.ndarray, metadata: pd.DataFrame, backbone: str, output_dir: Path) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    np.save(output_dir / f"{backbone}_embeddings.npy", X)
    metadata.to_csv(output_dir / f"{backbone}_metadata.csv", index=False)


def load_chexpert_embeddings(root_dir: Path, model: str) -> dict[str, Any]:
    model = model.lower()
    base = root_dir / "data" / "chexpert"
    if model == "cxr-foundation":
        with open(base / "cxr-foundation" / "embed.pkl", "rb") as src:
            df_embed = pickle.load(src)
        return {"df_embed": df_embed}
    if model == "biomedclip":
        with open(base / "biomedclip" / "embedding_from_biomedclip_Chexpert.pkl", "rb") as src:
            payload = pickle.load(src)
        return {"embeddings_list": payload[0], "paths_list": payload[3]}
    if model == "medclip":
        with open(base / "medclip" / "embedding_from_medclip.pkl", "rb") as src:
            payload = pickle.load(src)
        return {"embeddings_list": payload[0], "paths_list": payload[3]}
    raise ValueError(f"Unknown CheXpert model: {model}")


def prepare_chexpert_artifacts(
    *, root_dir: Path, manifest_path: Path, model: str, output_dir: Path
) -> None:
    manifest_df = load_and_validate_manifest(manifest_path)
    data = load_chexpert_embeddings(root_dir, model)
    if "df_embed" in data:
        # Expect columns for image paths and embeddings
        df = data["df_embed"].copy()
        if "image_path" not in df.columns:
            raise ValueError("`df_embed` must contain an `image_path` column")
        if "embedding" not in df.columns:
            raise ValueError("`df_embed` must contain an `embedding` column")
        X = _stack_embeddings(df["embedding"].tolist())
        metadata = _metadata_from_manifest(df["image_path"].tolist(), manifest_df)
    else:
        X = _stack_embeddings(data["embeddings_list"])
        metadata = _metadata_from_manifest(data["paths_list"], manifest_df)
    _save_artifacts(X, metadata, model, output_dir)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Prepare CheXpert artifacts + run Dataset 2 benchmark"
    )
    parser.add_argument("--manifest", type=Path, required=True, help="CheXpert manifest CSV")
    parser.add_argument(
        "--model", choices=["cxr-foundation", "biomedclip", "medclip"], default="medclip"
    )
    parser.add_argument(
        "--root",
        type=Path,
        default=Path.cwd(),
        help="Project root containing the `data/chexpert` pickles",
    )
    parser.add_argument(
        "--artifacts-dir",
        type=Path,
        default=Path("output/paper_benchmark/chexpert_embeddings"),
        help="Where to write backbone embeddings + metadata",
    )
    parser.add_argument(
        "--config",
        type=Path,
        default=Path("examples/paper_benchmark_config.json"),
        help="Benchmark config template to use",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    prepare_chexpert_artifacts(
        root_dir=args.root,
        manifest_path=args.manifest,
        model=args.model,
        output_dir=args.artifacts_dir,
    )

    with args.config.open() as src:
        config_data = json.load(src)
    config_data.setdefault("chexpert", {})
    config_data["chexpert"]["enabled"] = True
    config_data["chexpert"]["manifest_path"] = str(args.manifest)
    config_data["chexpert"]["embeddings_dir"] = str(args.artifacts_dir)
    # Restrict benchmark loading to the backbone artifact generated in this run.
    config_data["chexpert"]["backbones"] = [args.model]
    config = PaperBenchmarkConfig.from_dict(config_data)
    runner = PaperBenchmarkRunner(config)
    runner.run()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
