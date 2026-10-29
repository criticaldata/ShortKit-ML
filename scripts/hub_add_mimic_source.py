#!/usr/bin/env python3
"""Add the MIMIC-CXR source parquet files to MITCriticalData/ShortKit-ML-data.

Run from the ShortKit-ML repo with its venv, logged in to Hugging Face as an owner of the dataset:

    cd ~/Desktop/repositories/ShortKit-ML
    .venv/bin/python scripts/hub_add_mimic_source.py --dry-run
    .venv/bin/python scripts/hub_add_mimic_source.py

One hub commit: the four seed-0 coreset parquet files (about 33 MB) go under mimic_cxr/source/
with the layout the setup scripts expect, and the dataset card is refreshed from data/README.md.
After this the paper and the code cite only this dataset for MIMIC-CXR.

The files are fetched from the extraction dataset on the hub (SOURCE_REPO, which the owner can
read); --source points at a local folder in the mimic_cxr/source/ layout instead. Git LFS pointer
files are refused by the size check.
"""
import argparse
import sys
from pathlib import Path

from huggingface_hub import CommitOperationAdd, HfApi, hf_hub_download

sys.path.insert(0, str(Path(__file__).resolve().parent))
from hub_fix_labels import FRONT, REPO, REPO_DIR, md5  # noqa: E402
from mimic_source import PARQUETS, SOURCE_PREFIX  # noqa: E402

SOURCE_REPO = "MITCriticalData/qml-mimic-cxr-embeddings"
# path in SOURCE_REPO for each destination path in PARQUETS (the RAD-DINO file moved there)
SOURCE_PATHS = {
    "rad-dino-embeddings/data_type5_n1998_seed0_rad_dino.parquet":
        "rad-dino-embeddings/20-seeds/seed_0/data_type5_n1998_seed0_rad_dino.parquet",
}
MIN_BYTES = 1_000_000


def fetch(dst: str, local: Path | None) -> Path:
    if local is not None:
        return local / dst
    return Path(hf_hub_download(SOURCE_REPO, SOURCE_PATHS.get(dst, dst), repo_type="dataset"))


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--dry-run", action="store_true")
    ap.add_argument("--source", type=Path, default=None, help="local folder in the mimic_cxr/source/ layout (default: fetch from the hub)")
    args = ap.parse_args()
    api = HfApi()
    print("logged in as", api.whoami().get("name"))
    files = set(api.list_repo_files(REPO, repo_type="dataset"))
    ops = []
    for f in PARQUETS:
        src = fetch(f, args.source)
        if not src.exists() or src.stat().st_size < MIN_BYTES:
            print("MISSING or LFS pointer:", src)
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
