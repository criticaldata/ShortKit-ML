#!/usr/bin/env python3
"""Extract CheXpert embeddings for paper benchmark backbones."""

from __future__ import annotations

import argparse
import json

from shortcut_detect.benchmark.chexpert_extraction import ExtractionConfig, extract_embeddings


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Extract CheXpert embeddings for ResNet/DenseNet/ViT"
    )
    parser.add_argument(
        "--manifest", required=True, help="Manifest CSV with image_path/task_label/race/sex/age"
    )
    parser.add_argument(
        "--backbones",
        default="resnet50,densenet121,vit_b_16",
        help="Comma-separated list of backbones",
    )
    parser.add_argument(
        "--output-dir", required=True, help="Output directory for *.npy and metadata CSV"
    )
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--device", default="cpu")
    return parser.parse_args()


def main() -> int:
    args = _parse_args()
    backbones = [x.strip() for x in args.backbones.split(",") if x.strip()]
    cfg = ExtractionConfig(
        manifest_path=args.manifest,
        output_dir=args.output_dir,
        backbones=backbones,
        batch_size=int(args.batch_size),
        num_workers=int(args.num_workers),
        device=str(args.device),
    )
    out = extract_embeddings(cfg)
    print(json.dumps(out, indent=2))
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
