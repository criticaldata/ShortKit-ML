#!/usr/bin/env python3
"""Re-extract the CheXpert backbones that had no extraction code (tracker X-6)
from the cached images and compare with the stored embeddings.

Candidates per backbone (pooling chosen to match the stored dimension):
  vit_b_32     torchvision ViT_B_32_Weights.DEFAULT, heads removed (class token, 768)
  dinov2_base  facebook/dinov2-base, pooler output (CLS after layernorm, 768)
  rad_dino     microsoft/rad-dino, pooler output (CLS after layernorm, 768)
  medsiglip    google/medsiglip-448 vision tower pooler output (1152); also tried
               google/siglip-so400m-patch14-384

Writes per-row cosine similarity between re-extracted and stored embeddings.
The stored files are never overwritten (new arrays go to the results folder).
"""

from __future__ import annotations

import argparse
import json
import pickle
import sys
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
from common import DATA, Manifest, results_root, update_numbers  # noqa: E402

CANDIDATES = {
    # control: vit_b_16 has extraction code in the repo
    "vit_b_16": [("torchvision", "vit_b_16", "class token (heads = Identity)")],
    "vit_b_32": [("torchvision", "vit_b_32", "class token (heads = Identity)"),
                 ("hf", "google/vit-base-patch32-224-in21k", "pooler_output"),
                 ("hf", "google/vit-base-patch32-224-in21k", "cls token"),
                 ("hf_clip", "openai/clip-vit-base-patch32", "vision pooler_output")],
    "dinov2_base": [("hf", "facebook/dinov2-base", "pooler_output"),
                    ("hf", "facebook/dinov2-base", "mean of patch tokens")],
    "rad_dino": [("hf", "microsoft/rad-dino", "pooler_output"),
                 ("hf", "microsoft/rad-dino", "mean of patch tokens")],
    "medsiglip": [("hf_siglip", "google/medsiglip-448", "vision pooler_output"),
                  ("hf_siglip", "google/siglip-so400m-patch14-384", "vision pooler_output")],
}


def embed(kind: str, model_id: str, pooling: str, images, device: str, bs: int):
    import torch

    feats = []
    revision = None
    if kind == "torchvision":
        from torchvision import models

        w = {"vit_b_32": models.ViT_B_32_Weights, "vit_b_16": models.ViT_B_16_Weights}[model_id].DEFAULT
        model = getattr(models, model_id)(weights=w)
        model.heads = torch.nn.Identity()
        tf = w.transforms()
        model.eval().to(device)
        revision = str(w)
        for s in range(0, len(images), bs):
            x = torch.stack([tf(im.convert("RGB")) for im in images[s : s + bs]]).to(device)
            with torch.no_grad():
                feats.append(model(x).cpu().numpy())
        return np.vstack(feats), revision
    from transformers import AutoImageProcessor, AutoModel

    proc = AutoImageProcessor.from_pretrained(model_id)
    model = AutoModel.from_pretrained(model_id)
    revision = getattr(model.config, "_commit_hash", None)
    if kind in {"hf_siglip", "hf_clip"}:
        model = model.vision_model
    model.eval().to(device)
    for s in range(0, len(images), bs):
        inp = proc(images=[im.convert("RGB") for im in images[s : s + bs]], return_tensors="pt").to(device)
        with torch.no_grad():
            out = model(pixel_values=inp["pixel_values"])
        if pooling.startswith("mean"):
            f = out.last_hidden_state[:, 1:].mean(1)
        elif pooling.startswith("cls"):
            f = out.last_hidden_state[:, 0]
        else:
            f = out.pooler_output
        feats.append(f.float().cpu().numpy())
    return np.vstack(feats), revision


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--backbones", nargs="+", default=[b for b in CANDIDATES if b != "vit_b_16"])
    ap.add_argument("--device", default="cpu")
    ap.add_argument("--batch-size", type=int, default=32)
    args = ap.parse_args()
    out = results_root() / "extraction_check"
    out.mkdir(parents=True, exist_ok=True)
    man = Manifest(out, "re-extraction of CheXpert backbones without code (X-6)")
    man.config = {"device": args.device, "candidates": CANDIDATES}
    cache = DATA / "chexpert_multibackbone" / "_cache" / "images.pkl"
    man.add_data(cache)
    with open(cache, "rb") as f:
        images = pickle.load(f)
    rows = []
    for bb in args.backbones:
        stored = np.load(DATA / "chexpert_multibackbone" / f"{bb}_embeddings.npy")
        man.add_data(DATA / "chexpert_multibackbone" / f"{bb}_embeddings.npy")
        for kind, mid, pooling in CANDIDATES[bb]:
            t0 = time.time()
            try:
                new, rev = embed(kind, mid, pooling, images, args.device, args.batch_size)
            except Exception as exc:  # recorded plainly
                rows.append({"backbone": bb, "model_id": mid, "pooling": pooling, "error": str(exc)[:300]})
                print(bb, mid, "FAILED", exc, flush=True)
                continue
            if new.shape != stored.shape:
                rows.append({"backbone": bb, "model_id": mid, "pooling": pooling,
                             "error": f"shape {new.shape} vs stored {stored.shape}"})
                continue
            cos = (new * stored).sum(1) / (np.linalg.norm(new, axis=1) * np.linalg.norm(stored, axis=1))
            np.save(out / f"{bb}__{mid.replace('/', '_')}__{pooling.split()[0]}.npy", new.astype(np.float32))
            rows.append({"backbone": bb, "model_id": mid, "revision": rev, "pooling": pooling,
                         "cos_mean": float(cos.mean()), "cos_min": float(cos.min()),
                         "max_abs_diff": float(np.abs(new - stored).max()), "seconds": round(time.time() - t0, 1)})
            print(rows[-1], flush=True)
    (out / "extraction_check.json").write_text(json.dumps(rows, indent=2))
    best = {}
    for r in rows:
        if "cos_mean" in r and (r["backbone"] not in best or r["cos_mean"] > best[r["backbone"]]["cos_mean"]):
            best[r["backbone"]] = r
    update_numbers({"model_ids_extraction_check": best})
    man.write()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
