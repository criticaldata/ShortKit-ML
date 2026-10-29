#!/usr/bin/env python3
"""Bring MITCriticalData/ShortKit-ML-data in line with the licence rule and the pipeline.

Run from the ShortKit-ML repo with its venv, logged in to Hugging Face as an owner of the dataset:

    cd ~/Desktop/repositories/ShortKit-ML
    .venv/bin/python scripts/hub_fix_labels.py --dry-run
    .venv/bin/python scripts/hub_fix_labels.py

What it does, as one hub commit (the hub keeps history, so it can be reverted):
  1. deletes train.csv and valid.csv (Stanford CheXpert label files; never to be redistributed)
  2. uploads the derived inputs common.py reads and the hub lacks:
       mimic_cxr/vit32_cls_embeddings_realigned.npy, mimic_cxr/mimic_race_raw.csv
  3. replaces the dataset card with data/README.md from the repo plus YAML front matter
It does NOT upload CHEXPERT DEMO.xlsx or any CelebA file, and it does not touch git history.
"""
import argparse
import hashlib
import os
import sys
from pathlib import Path

from huggingface_hub import CommitOperationAdd, CommitOperationDelete, HfApi

REPO = "MITCriticalData/ShortKit-ML-data"
LOCAL = Path.home() / "Documents/DATASETS/shortkit-ml"
REPO_DIR = Path.home() / "Desktop/repositories/ShortKit-ML"
DELETE = ["train.csv", "valid.csv"]
UPLOAD = ["mimic_cxr/vit32_cls_embeddings_realigned.npy", "mimic_cxr/mimic_race_raw.csv"]
FRONT = """---
license: cc-by-nc-4.0
pretty_name: ShortKit-ML derived embeddings and metadata
tags:
- medical-imaging
- chest-x-ray
- fairness
- shortcut-learning
extra_gated_prompt: "Access requires PhysioNet credentials for MIMIC-CXR. State your PhysioNet username; approval is manual after the requester emails proof to sebasmos@mit.edu."
extra_gated_fields:
  PhysioNet username: text
---
"""


def md5(p: Path) -> str:
    h = hashlib.md5()
    with open(p, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--dry-run", action="store_true")
    args = ap.parse_args()
    api = HfApi()
    who = api.whoami()
    print("logged in as", who.get("name"))
    files = set(api.list_repo_files(REPO, repo_type="dataset"))
    ops = []
    for f in DELETE:
        if f in files:
            ops.append(CommitOperationDelete(path_in_repo=f))
            print("delete ", f)
        else:
            print("absent ", f)
    for f in UPLOAD:
        src = LOCAL / f
        if not src.exists():
            print("MISSING locally:", src)
            return 1
        print("upload ", f, md5(src), f"{src.stat().st_size/1e6:.1f} MB", "(replaces)" if f in files else "(new)")
        ops.append(CommitOperationAdd(path_in_repo=f, path_or_fileobj=str(src)))
    card = REPO_DIR / "data/README.md"
    body = card.read_text()
    if body.startswith("---"):
        body = body.split("---", 2)[2]
    ops.append(CommitOperationAdd(path_in_repo="README.md", path_or_fileobj=(FRONT + body.lstrip()).encode()))
    print("replace README.md (dataset card) from", card)
    if args.dry_run:
        print("dry run: no commit")
        return 0
    info = api.create_commit(
        repo_id=REPO, repo_type="dataset", operations=ops,
        commit_message="Remove Stanford CheXpert label files; add realigned ViT-B/32 and MIMIC race inputs; current dataset card",
    )
    print("commit:", info.commit_url)
    after = set(api.list_repo_files(REPO, repo_type="dataset"))
    for f in DELETE:
        print("still present" if f in after else "removed", f)
    for f in UPLOAD:
        print("present" if f in after else "MISSING", f)
    return 0


if __name__ == "__main__":
    sys.exit(main())
