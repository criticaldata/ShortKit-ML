#!/usr/bin/env python3
"""Planted image marker as a known shortcut (revision decisions, section 7, item 4).

A white square (side 8 percent of the shorter image side, top-left corner at 30
percent of width and height so it survives the centre crop) is stamped into the
cached CheXpert images. Every image is embedded with and without the marker by
a frozen torchvision ResNet-50 (ImageNet weights, the paper's pipeline), so any
marker assignment M can be composed afterwards.

Audit: M on Y=1 (Pneumothorax) images at rate r_audit, never on Y=0; the
conditional audit with A = M, patient-grouped, must detect the marker.
Head: patients split 50/50; training images carry M on Y=1 at rate r_head and
never on Y=0; test images carry M on a random half regardless of Y. A logistic
head on the training embeddings is evaluated by its TPR and FPR gaps between
marked and unmarked test images (positive gap = the head uses the marker).
"""

from __future__ import annotations

import argparse
import pickle
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
from common import CHEXPERT_JOIN_NOTE, DATA, Manifest, chexpert_meta, results_root, update_numbers  # noqa: E402

R_AUDIT = 0.5
R_HEAD = 0.9


def stamp(img):
    from PIL import ImageDraw

    im = img.convert("RGB").copy()
    w, h = im.size
    s = int(round(0.08 * min(w, h)))
    x0, y0 = int(0.30 * w), int(0.30 * h)
    ImageDraw.Draw(im).rectangle([x0, y0, x0 + s, y0 + s], fill=(255, 255, 255))
    return im


def embed(images, threads: int):
    import torch
    from torchvision import models

    torch.set_num_threads(threads)
    w = models.ResNet50_Weights.DEFAULT
    model = models.resnet50(weights=w)
    model.fc = torch.nn.Identity()
    model.eval()
    tf = w.transforms()
    out = []
    for s in range(0, len(images), 32):
        x = torch.stack([tf(im.convert("RGB")) for im in images[s : s + 32]])
        with torch.no_grad():
            out.append(model(x).numpy())
    return np.vstack(out).astype(np.float32), str(w)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--threads", type=int, default=6)
    ap.add_argument("--K", type=int, default=200)
    ap.add_argument("--seeds", type=int, default=10)
    args = ap.parse_args()
    from sklearn.linear_model import LogisticRegression
    from sklearn.metrics import roc_auc_score
    from sklearn.model_selection import StratifiedGroupKFold
    from sklearn.pipeline import make_pipeline
    from sklearn.preprocessing import StandardScaler

    from shortcut_detect.audit import AuditConfig, conditional_audit

    out = results_root() / "planted_marker"
    out.mkdir(parents=True, exist_ok=True)
    man = Manifest(out, "planted marker (section 7 item 4)")
    man.config = {"r_audit": R_AUDIT, "r_head": R_HEAD, "backbone": "torchvision resnet50 IMAGENET1K_V2 (DEFAULT)",
                  "marker": "white square, side 8% of min(w,h), top-left at (0.3w, 0.3h)", "K": args.K,
                  "seeds": args.seeds, "task": "Pneumothorax (task_label)"}
    man.notes.append(CHEXPERT_JOIN_NOTE)
    cache = DATA / "chexpert_multibackbone" / "_cache" / "images.pkl"
    man.add_data(cache)
    meta = chexpert_meta()
    y = meta["task_label"].to_numpy()
    pid = meta["patient_id"].to_numpy()
    t0 = time.time()
    emb_clean, emb_mark = out / "emb_clean.npy", out / "emb_marked.npy"
    if not emb_mark.exists():
        with open(cache, "rb") as f:
            images = pickle.load(f)
        Ec, wname = embed(images, args.threads)
        Em, _ = embed([stamp(im) for im in images], args.threads)
        np.save(emb_clean, Ec)
        np.save(emb_mark, Em)
    Ec, Em = np.load(emb_clean), np.load(emb_mark)
    man.time("embedding", time.time() - t0)

    rows = []
    for s in range(args.seeds):
        rng = np.random.default_rng(s)
        m = np.zeros(len(y), dtype=int)
        pos = np.flatnonzero(y == 1)
        m[rng.choice(pos, size=int(round(R_AUDIT * len(pos))), replace=False)] = 1
        E = np.where(m[:, None] == 1, Em, Ec)
        res = conditional_audit(E, m, y, AuditConfig(n_permutations=args.K, n_bootstrap=0, seed=s), groups=pid)
        # head
        sgk = StratifiedGroupKFold(n_splits=2, shuffle=True, random_state=s)
        tr, te = next(sgk.split(E, y, pid))
        mtr = np.zeros(len(tr), dtype=int)
        ptr = np.flatnonzero(y[tr] == 1)
        mtr[rng.choice(ptr, size=int(round(R_HEAD * len(ptr))), replace=False)] = 1
        Etr = np.where(mtr[:, None] == 1, Em[tr], Ec[tr])
        mte = (rng.random(len(te)) < 0.5).astype(int)
        Ete = np.where(mte[:, None] == 1, Em[te], Ec[te])
        clf = make_pipeline(StandardScaler(), LogisticRegression(C=1.0, max_iter=3000, class_weight="balanced"))
        clf.fit(Etr, y[tr])
        p = clf.predict_proba(Ete)[:, 1]
        pred = p >= 0.5
        yt = y[te]
        row = {"seed": s, "n_marked_audit": int(m.sum()),
               **{f"{d}_stat": v.statistic for d, v in res.results.items()},
               **{f"{d}_p": v.p_value for d, v in res.results.items()},
               **{f"{d}_flag": v.flag for d, v in res.results.items()},
               "tpr_marked": float(pred[(yt == 1) & (mte == 1)].mean()),
               "tpr_unmarked": float(pred[(yt == 1) & (mte == 0)].mean()),
               "fpr_marked": float(pred[(yt == 0) & (mte == 1)].mean()),
               "fpr_unmarked": float(pred[(yt == 0) & (mte == 0)].mean()),
               "auroc_marked": float(roc_auc_score(yt[mte == 1], p[mte == 1])),
               "auroc_unmarked": float(roc_auc_score(yt[mte == 0], p[mte == 0]))}
        row["tpr_gap"] = row["tpr_marked"] - row["tpr_unmarked"]
        row["fpr_gap"] = row["fpr_marked"] - row["fpr_unmarked"]
        rows.append(row)
        print(row, flush=True)
    df = pd.DataFrame(rows)
    df.to_csv(out / "planted_marker.csv", index=False)
    update_numbers({})  # ensure file exists
    man.time("all", time.time() - t0)
    man.write()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
