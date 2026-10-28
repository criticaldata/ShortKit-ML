#!/usr/bin/env python3
"""How many audited images were in RAD-DINO's pretraining set.

microsoft/rad-dino publishes the list of its training images (training_images.csv).
CheXpert images are matched on patient, study and view; MIMIC-CXR images on the
DICOM id. RAD-DINO is self-supervised, so it saw the images but no labels.
"""

from __future__ import annotations

import re
import sys
import time
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
from common import Manifest, chexpert_meta, load_mimic, mimic_paths, results_root, update_numbers  # noqa: E402

REPO, FILE, REVISION = "microsoft/rad-dino", "training_images.csv", "110cbc18d5133582e320b43d53bf5c44e410c936"


def chexpert_key(path: str) -> str:
    m = re.search(r"patient(\d+)/study(\d+)/view(\d+)_(frontal|lateral)", path)
    return f"p{m.group(1)}_s{m.group(2)}_view{m.group(3)}_{m.group(4)}"


def main() -> int:
    from huggingface_hub import hf_hub_download

    out = results_root() / "pretraining_overlap"
    out.mkdir(parents=True, exist_ok=True)
    man = Manifest(out, "Audited images in the RAD-DINO pretraining set")
    man.config = {"repo": REPO, "file": FILE, "revision": REVISION}
    t0 = time.time()
    ids = set(pd.read_csv(hf_hub_download(REPO, FILE, revision=REVISION)).image_id)

    cm = chexpert_meta()
    c_in = cm.path.map(chexpert_key).isin(ids)
    _, mm = load_mimic("rad_dino")
    dicom = mm.image_path.str.extract(r"([0-9a-f]{8}(?:-[0-9a-f]{8}){4})")[0]
    m_in = dicom.isin(ids)
    man.add_data(*mimic_paths("rad_dino"))

    res = {"raddino_overlap_chexpert_images": int(c_in.sum()), "chexpert_n_images": len(cm),
           "raddino_overlap_chexpert_patients": int(cm.loc[c_in, "patient_id"].nunique()),
           "raddino_overlap_mimic_images": int(m_in.sum()), "mimic_n_images": len(mm)}
    pd.DataFrame([res]).to_csv(out / "pretraining_overlap.csv", index=False)
    update_numbers(res)
    man.time("all", time.time() - t0)
    man.write()
    print(res)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
