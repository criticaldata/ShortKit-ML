#!/usr/bin/env python3
"""Add the MIMIC-CXR source parquet files to MITCriticalData/ShortKit-ML-data.

Run from the ShortKit-ML repo with its venv, logged in to Hugging Face as an owner of the dataset:

    cd ~/Desktop/repositories/ShortKit-ML
    .venv/bin/python scripts/hub_add_mimic_source.py --dry-run
    .venv/bin/python scripts/hub_add_mimic_source.py

One hub commit: the four seed-0 coreset parquet files (about 33 MB) go under mimic_cxr/source/
with the layout the setup scripts expect, and the dataset card is refreshed from data/README.md.
After this the paper and the code cite only this dataset for MIMIC-CXR.
"""
import argparse
import sys
from pathlib import Path

from huggingface_hub import CommitOperationAdd, HfApi

sys.path.insert(0, str(Path(__file__).resolve().parent))
from hub_fix_labels import FRONT, REPO, REPO_DIR, md5  # noqa: E402
from mimic_source import PARQUETS, SOURCE_PREFIX  # noqa: E402

LOCAL = Path.home() / "Desktop/repositories/qml-mimic-cxr-embeddings"


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--dry-run", action="store_true")
    ap.add_argument("--source", type=Path, default=LOCAL, help="local checkout holding the parquet files")
    args = ap.parse_args()
    api = HfApi()
    print("logged in as", api.whoami().get("name"))
    files = set(api.list_repo_files(REPO, repo_type="dataset"))
    ops = []
    for f in PARQUETS:
        src = args.source / f
        if not src.exists():
            print("MISSING locally:", src)
            return 1
        dst = f"{SOURCE_PREFIX}/{f}"
        print("upload ", dst, md5(src), f"{src.stat().st_size/1e6:.1f} MB", "(replaces)" if dst in files else "(new)")
        ops.append(CommitOperationAdd(path_in_repo=dst, path_or_fileobj=str(src)))
    body = (REPO_DIR / "data/README.md").read_text()
    if body.startswith("---"):
        body = body.split("---", 2)[2]
    ops.append(CommitOperationAdd(path_in_repo="README.md", path_or_fileobj=(FRONT + body.lstrip()).encode()))
    print("replace README.md (dataset card)")
    if args.dry_run:
        print("dry run: no commit")
        return 0
    info = api.create_commit(
        repo_id=REPO, repo_type="dataset", operations=ops,
        commit_message="Add MIMIC-CXR coreset source parquet files (seed 0); current dataset card",
    )
    print("commit:", info.commit_url)
    after = set(api.list_repo_files(REPO, repo_type="dataset"))
    for f in PARQUETS:
        print("present" if f"{SOURCE_PREFIX}/{f}" in after else "MISSING", f)
    return 0


if __name__ == "__main__":
    sys.exit(main())
