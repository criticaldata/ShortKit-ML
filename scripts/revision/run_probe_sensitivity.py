#!/usr/bin/env python3
"""Reviewer sensitivity checks on the CheXpert conditional probe only.

--check provenance  probe on re-extracted arrays from public checkpoints
                    (extraction_check/), dinov2_base, rad_dino, vit_b_32
--check C           probe with C in {0.1, 1, 10} on the 7 stored backbones
                    (C = 1 reruns the main setting as a reproduction check)
--check frontal     probe on frontal images only (train.csv Frontal/Lateral)
--check labelled    probe on images with a definite Pneumothorax label (0 or 1),
                    dropping uncertain (-1) and blank labels as the head experiment does

Primary cells only (sex, age 60+, White vs non-White), task Pneumothorax,
the same seed, K_probe, CV repeats and patient-level permutation as
run_clinical_audit.py. Secondary detectors are not run, so the probe p_bh
written by conditional_audit equals the raw p; the paper's flag is BH over
the six detectors of a cell, and the comparison columns say how it is
judged (see BH_NOTE).
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
from common import (  # noqa: E402
    CHEXPERT_BACKBONES,
    CHEXPERT_JOIN_NOTE,
    DATA,
    Manifest,
    attribute_cells,
    chexpert_meta,
    chexpert_paths,
    load_chexpert,
    results_root,
)
from run_clinical_audit import K_PROBE, N_BOOT, SEED  # noqa: E402

ALPHA = 0.05
N_DET = 6
PROVENANCE = {
    "dinov2_base": ("dinov2_base__facebook_dinov2-base__pooler_output.npy", "facebook/dinov2-base pooler_output"),
    "rad_dino": ("rad_dino__microsoft_rad-dino__pooler_output.npy", "microsoft/rad-dino pooler_output"),
    "vit_b_32": ("vit_b_32__vit_b_32__class.npy", "torchvision vit_b_32 class token"),
}
OUT = {"provenance": "provenance_sensitivity", "C": "probe_C_sensitivity", "frontal": "frontal_only", "labelled": "labelled_only"}
BH_NOTE = (
    "Paper significance of the probe = its own permutation p < 0.05 (raw_sig; BH covers only the "
    "five secondary detectors of a cell, so the probe's decision does not depend on them). "
    "decision_changes compares raw_sig with the main audit's raw_sig. Kept for reference: "
    "decision_bound ('sig' if p < 0.05/6, 'not sig' if p >= 0.05, else 'indeterminate') and, for "
    "the C check, bh_with_main_secondaries (BH over this run's probe p and the five secondary "
    "p-values of the main audit), both from the earlier six-way rule."
)


def run_probe(cell, C: float, threads: int):
    import torch

    torch.set_num_threads(threads)
    from shortcut_detect.audit import AuditConfig, conditional_audit

    t0 = time.time()
    cfg = AuditConfig(n_permutations=K_PROBE, n_permutations_probe=K_PROBE, n_bootstrap=N_BOOT,
                      seed=SEED, C=C, detectors=("probe",))
    res = conditional_audit(cell.X, cell.a, cell.y, cfg, groups=cell.groups)
    r = res.results["probe"]
    print(f"{cell.key} C={C}: {time.time() - t0:.0f}s auroc={r.statistic:.4f} p={r.p_value:.4f}", flush=True)
    return {"backbone": cell.backbone, "attribute": cell.attribute, "C": C,
            "n_images": int(len(cell.y)), "n_patients": int(len(set(cell.groups))),
            "group_sizes": json.dumps(cell.group_sizes()), "auroc": r.statistic,
            "ci_low": r.ci_low, "ci_high": r.ci_high, "null_mean": r.null_mean, "p": r.p_value,
            "repeat_sd": r.repeat_sd, "n_permutations": r.n_permutations,
            "seconds": round(time.time() - t0, 1)}


def bh(p: np.ndarray) -> np.ndarray:
    from shortcut_detect.audit.conditional import benjamini_hochberg

    return benjamini_hochberg(p)


def decision_bound(p: float) -> str:
    return "sig" if p < ALPHA / N_DET else ("not sig" if p >= ALPHA else "indeterminate")


def main_rows() -> pd.DataFrame:
    d = pd.read_csv(results_root() / "clinical_audit" / "audit_rows.csv")
    return d[(d.dataset == "chexpert") & (d.variant == "primary") & d.evaluated.astype(bool)]


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--check", choices=list(OUT), required=True)
    ap.add_argument("--C", nargs="+", type=float, default=[0.1, 1.0, 10.0])
    ap.add_argument("--backbones", nargs="+", default=None)
    ap.add_argument("--jobs", type=int, default=3)
    ap.add_argument("--threads", type=int, default=3)
    args = ap.parse_args()
    from joblib import Parallel, delayed

    out = results_root() / OUT[args.check]
    out.mkdir(parents=True, exist_ok=True)
    man = Manifest(out, f"CheXpert probe sensitivity: {args.check}")
    man.config = {"check": args.check, "K_probe": K_PROBE, "n_bootstrap": N_BOOT, "seed": SEED,
                  "alpha": ALPHA, "n_folds": 5, "cv_repeats": 5, "detectors": ["probe"],
                  "C": args.C if args.check == "C" else [1.0],
                  "permutation": "within Y; patient-level within Y-pattern strata",
                  "cells": "primary: sex, age 60+ (first study), White vs non-White; task Pneumothorax"}
    man.notes += [CHEXPERT_JOIN_NOTE, BH_NOTE]
    man.add_data(DATA / "train.csv", DATA / "chexpert_multibackbone" / "race_mapping.csv",
                 DATA / "chexpert_multibackbone" / "chexpert_manifest.csv",
                 results_root() / "clinical_audit" / "audit_rows.csv")
    jobs = []
    if args.check == "provenance":
        meta = chexpert_meta()
        for bb in args.backbones or list(PROVENANCE):
            p = results_root() / "extraction_check" / PROVENANCE[bb][0]
            X = np.load(p).astype(np.float32)
            assert X.shape == (len(meta), 768), X.shape
            man.add_data(p)
            jobs += [(c, 1.0) for c in attribute_cells("chexpert", bb, X, meta, sensitivity=False)]
        man.config["arrays"] = {bb: PROVENANCE[bb] for bb in args.backbones or list(PROVENANCE)}
        man.notes.append("Row order: extract_missing_chexpert_backbones.py embeds the images in "
                         "_cache/images.pkl order and compares row by row with the stored array "
                         "(cosine in extraction_check.json); no reordering, so rows match chexpert_meta().")
    else:
        for bb in args.backbones or CHEXPERT_BACKBONES:
            X, meta = load_chexpert(bb)
            man.add_data(chexpert_paths(bb)[0])
            if args.check == "frontal":
                keep = (meta["view"] == "Frontal").to_numpy()
                X, meta = X[keep], meta[keep].reset_index(drop=True)
            if args.check == "labelled":
                keep = meta["Pneumothorax"].isin([0, 1]).to_numpy()
                X, meta = X[keep], meta[keep].reset_index(drop=True)
            cs = attribute_cells("chexpert", bb, X, meta, sensitivity=False)
            jobs += [(c, C) for c in cs for C in (args.C if args.check == "C" else [1.0])]
        if args.check == "labelled":
            man.notes.append("Labelled = train.csv Pneumothorax in {0, 1}; uncertain (-1) and blank "
                             "labels dropped, as in run_head_experiment.py.")
        if args.check == "frontal":
            man.notes.append("Frontal = train.csv Frontal/Lateral == 'Frontal'. Age attribute stays the "
                             "age at the patient's first study in the full 2,000-image cohort.")
    jobs.sort(key=lambda t: -t[0].X.shape[0] * t[0].X.shape[1] * t[1])  # heaviest first
    t0 = time.time()
    rows = Parallel(n_jobs=args.jobs)(delayed(run_probe)(c, C, args.threads) for c, C in jobs)
    man.time("all_cells", time.time() - t0)
    df = pd.DataFrame(rows)

    ref = main_rows()
    probe = ref[ref.detector == "probe"].set_index(["backbone", "attribute"])
    df["main_auroc"] = [probe.loc[(b, a), "statistic"] for b, a in zip(df.backbone, df.attribute)]
    df["main_p"] = [probe.loc[(b, a), "p_value"] for b, a in zip(df.backbone, df.attribute)]
    df["main_flag"] = [str(probe.loc[(b, a), "flag"]).lower() == "true" for b, a in zip(df.backbone, df.attribute)]
    df["main_n_images"] = [probe.loc[(b, a), "n_images"] for b, a in zip(df.backbone, df.attribute)]
    df["raw_sig"] = df.p < ALPHA
    df["decision_bound"] = df.p.map(decision_bound)
    if args.check == "C":
        q = []
        for b, a, p in zip(df.backbone, df.attribute, df.p):
            sec = ref[(ref.backbone == b) & (ref.attribute == a) & (ref.detector != "probe")]
            sec = sec[sec.detector.isin(["statistical", "geometric", "bias_direction", "hbac_nmi", "frequency_topk"])]
            assert len(sec) == 5
            q.append(float(bh(np.r_[p, sec.p_value.to_numpy(float)])[0]))
        df["q_bh_with_main_secondaries"] = q
        df["flag_bh_with_main_secondaries"] = df.q_bh_with_main_secondaries < ALPHA
    df["decision_changes"] = df.raw_sig != (df.main_p < ALPHA)  # the probe is judged at p < 0.05
    path = out / "probe_rows.csv"
    df.to_csv(path, index=False)
    man.write()
    print(f"wrote {path} ({len(df)} rows) in {time.time() - t0:.0f}s")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
