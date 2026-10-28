#!/usr/bin/env python3
"""Rerun the submitted 13-method pipeline on clinical data with the fixed flag reader.

The cohorts, extra_labels and detector settings match the submitted runners
(run_mimic_benchmark.py, run_chexpert_multibackbone_benchmark.py and
run_celeba_real_benchmark.py). Only the reading of each result changes:

* ``flag_legacy``: the submitted reading (top-level keys, GCE dict only).
* ``flag_fixed``: the tested attribute (``by_attribute['group']``), GCE dataclass.
* ``evaluated``: False when the method failed; such cells are n/a.

Every scalar of each result is also dumped to ``scalars.jsonl`` so later flag
questions can be answered offline.
"""

from __future__ import annotations

import argparse
import dataclasses
import json
import sys
import time
import warnings
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
from common import (  # noqa: E402
    CELEBA_TASKS,
    CHEXPERT_BACKBONES,
    MIMIC_BACKBONES,
    Manifest,
    celeba_paths,
    chexpert_paths,
    load_celeba,
    load_chexpert,
    load_mimic,
    mimic_paths,
    results_root,
    to_jsonable,
)

from shortcut_detect.benchmark.method_utils import (  # noqa: E402
    ALL_METHODS,
    _flag_single,
    method_evaluated,
    method_flag,
)
from shortcut_detect.unified import ShortcutDetector  # noqa: E402

warnings.filterwarnings("ignore")


def legacy_flag(method: str, result: dict) -> bool:
    """The submitted reading: top-level keys only, GCE read as a dict."""
    if not result.get("success", False):
        return False
    if method == "gce":
        rep = result.get("report", {})
        return isinstance(rep, dict) and str(rep.get("risk_level", "low")).lower() in {
            "moderate",
            "high",
        }
    return _flag_single(method, result)


def scalars(obj, prefix: str = "", out: dict | None = None, depth: int = 0) -> dict:
    out = {} if out is None else out
    if depth > 6:
        return out
    if dataclasses.is_dataclass(obj) and not isinstance(obj, type):
        obj = {f.name: getattr(obj, f.name) for f in dataclasses.fields(obj)}
    if isinstance(obj, dict):
        for k, v in obj.items():
            if k in {"detector", "per_sample_losses", "is_minority", "minority_indices"}:
                continue
            scalars(v, f"{prefix}.{k}" if prefix else str(k), out, depth + 1)
    elif isinstance(obj, bool | int | float | np.integer | np.floating | np.bool_):
        out[prefix] = to_jsonable(obj)
    elif isinstance(obj, str) and len(obj) < 200:
        out[prefix] = obj
    elif isinstance(obj, list | tuple) and len(obj) <= 20:
        if all(isinstance(v, int | float | str | np.integer | np.floating) for v in obj):
            out[prefix] = to_jsonable(list(obj))
    return out


def run_cell(X, y, g, extra, seed, meta_row, rows_out, scal_f):
    for method in ALL_METHODS:
        t0 = time.time()
        det = ShortcutDetector(
            methods=[method], seed=seed, statistical_correction="fdr_bh", statistical_alpha=0.05
        )
        try:
            det.fit(X, y, group_labels=g, extra_labels=extra)
            res = det.results_.get(method, {"success": False, "error": "missing"})
        except Exception as exc:  # recorded, never silent
            res = {"success": False, "error": str(exc)}
        el = time.time() - t0
        evaluated = method_evaluated(method, res, attribute="group")
        sub = res.get("by_attribute", {}).get("group", res) if "by_attribute" in res else res
        row = dict(meta_row)
        row.update(
            method=method,
            evaluated=bool(evaluated),
            flag_legacy=int(legacy_flag(method, res)),
            flag_any_attribute=int(method_flag(method, res)),
            flag_fixed=int(method_flag(method, res, attribute="group")) if evaluated else None,
            error=sub.get("error") or res.get("error"),
            elapsed_s=round(el, 2),
        )
        rows_out.append(row)
        scal_f.write(json.dumps({**meta_row, "method": method, "scalars": scalars(res)}) + "\n")
        scal_f.flush()
        print(
            f"  {meta_row['dataset']:8s} {meta_row['backbone']:12s} {meta_row['attribute']:8s} "
            f"{method:20s} eval={int(evaluated)} legacy={row['flag_legacy']} fixed={row['flag_fixed']} "
            f"({el:.1f}s)",
            flush=True,
        )


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--datasets", nargs="+", default=["mimic", "chexpert", "celeba"])
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--out", type=Path, default=None)
    args = ap.parse_args()

    out = args.out or results_root() / "old_pipeline_fixed_flag"
    out.mkdir(parents=True, exist_ok=True)
    man = Manifest(out, "old 13-method pipeline, fixed method_flag")
    man.config = {"datasets": args.datasets, "seed": args.seed, "methods": list(ALL_METHODS)}
    man.notes.append(
        "Pre-fix baseline: committed output/paper_benchmark/*_results CSVs; MIMIC also replayed "
        "on this laptop at ~/Documents/shortkit-ml-results/2026-10-28_mimic_replay_main-fe8de89 "
        "(matches the committed table in all 156 cells)."
    )
    rows: list[dict] = []
    scal_f = open(out / "scalars.jsonl", "a")

    if "mimic" in args.datasets:
        t0 = time.time()
        for bb in MIMIC_BACKBONES:
            X, meta = load_mimic(bb)
            man.add_data(*mimic_paths(bb))
            y = meta["task_label"].astype(int).to_numpy()
            for attr in ["race", "sex", "age_bin"]:
                frame = meta[meta["race"].isin(["WHITE", "BLACK", "ASIAN"])] if attr == "race" else meta
                idx = frame.index.to_numpy()
                extra = {a: frame[a].to_numpy() for a in ["race", "sex", "age_bin"]}
                info = {"dataset": "mimic", "backbone": bb, "attribute": attr, "task": "mortality",
                        "n": int(len(idx)), "n_groups": int(frame[attr].nunique())}
                run_cell(X[idx], y[idx], frame[attr].to_numpy(), extra, args.seed, info, rows, scal_f)
            pd.DataFrame(rows).to_csv(out / "old_pipeline_methods.csv", index=False)
        man.time("mimic", time.time() - t0)

    if "chexpert" in args.datasets:
        t0 = time.time()
        for bb in CHEXPERT_BACKBONES:
            # the submitted pipeline ran on the stored arrays
            X, meta = load_chexpert(bb, stored=True)
            man.add_data(*chexpert_paths(bb, stored=True))
            y = meta["task_label"].astype(int).to_numpy()
            extra = {a: meta[a].astype(str).to_numpy() for a in ["sex", "age_bin", "race"]}
            for attr in ["sex", "age_bin", "race"]:
                g = meta[attr].astype(str).to_numpy()
                info = {"dataset": "chexpert", "backbone": bb, "attribute": attr,
                        "task": "pneumothorax", "n": int(len(g)), "n_groups": int(len(set(g)))}
                run_cell(X, y, g, extra, args.seed, info, rows, scal_f)
            pd.DataFrame(rows).to_csv(out / "old_pipeline_methods.csv", index=False)
        man.time("chexpert", time.time() - t0)

    if "celeba" in args.datasets:
        t0 = time.time()
        X, attrs = load_celeba()
        man.add_data(*celeba_paths())
        for task in CELEBA_TASKS:
            y = attrs[task].to_numpy().astype(int)
            g = attrs["Male"].to_numpy().astype(int)
            extra = {c: attrs[c].to_numpy().astype(int)
                     for c in attrs.select_dtypes(include="number").columns if c != task}
            info = {"dataset": "celeba", "backbone": "resnet50_celeba", "attribute": f"Male|{task}",
                    "task": task, "n": int(len(y)), "n_groups": 2}
            run_cell(X, y, g, extra, args.seed, info, rows, scal_f)
            pd.DataFrame(rows).to_csv(out / "old_pipeline_methods.csv", index=False)
        man.time("celeba", time.time() - t0)

    scal_f.close()
    pd.DataFrame(rows).to_csv(out / "old_pipeline_methods.csv", index=False)
    man.write()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
