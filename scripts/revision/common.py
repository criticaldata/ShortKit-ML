"""Shared helpers for the CMPB revision scripts.

Paths, run manifests, data loaders and the attribute definitions fixed by the
revision method decisions (section 5). Every script under ``scripts/revision``
imports from here so that all runs use the same cohorts and the same folder.
"""

from __future__ import annotations

import hashlib
import json
import os
import platform
import subprocess
import sys
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

REPO = Path(__file__).resolve().parents[2]
DATA = REPO / "data"
RESULTS_PARENT = Path.home() / "Documents" / "shortkit-ml-results"

CHEXPERT_BACKBONES = [
    "resnet50",
    "densenet121",
    "vit_b_16",
    "vit_b_32",
    "dinov2_base",
    "rad_dino",
    "medsiglip",
]
MIMIC_BACKBONES = ["rad_dino", "vit16_cls", "vit32_cls", "medsiglip"]
BACKBONE_LABELS = {
    "resnet50": "ResNet-50",
    "densenet121": "DenseNet-121",
    "vit_b_16": "ViT-B/16",
    "vit_b_32": "ViT-B/32",
    "dinov2_base": "DINOv2",
    "rad_dino": "RAD-DINO",
    "medsiglip": "MedSigLIP",
    "vit16_cls": "ViT-B/16",
    "vit32_cls": "ViT-B/32",
    "resnet50_celeba": "ResNet-50",
}
CELEBA_TASKS = ["Blond_Hair", "Heavy_Makeup", "Attractive"]
RACE_NAMED = ["WHITE", "BLACK", "ASIAN"]


def git_sha(short: bool = True) -> str:
    """Commit of the code being run. ``SHORTKIT_COMMIT`` overrides it for runs
    made from a frozen ``git archive`` copy of a commit."""
    env = os.environ.get("SHORTKIT_COMMIT")
    if env:
        return env[:7] if short else env
    args = ["git", "-C", str(REPO), "rev-parse"] + (["--short"] if short else []) + ["HEAD"]
    return subprocess.check_output(args, text=True).strip()


def git_dirty() -> bool:
    if os.environ.get("SHORTKIT_COMMIT"):
        return False
    out = subprocess.check_output(["git", "-C", str(REPO), "status", "--porcelain"], text=True)
    return any(line and not line.startswith("??") for line in out.splitlines())


def results_root() -> Path:
    """Top results folder. Fixed by ``SHORTKIT_REVISION_DIR`` when set, so that
    all sub-runs of one revision pass land in the same folder."""
    env = os.environ.get("SHORTKIT_REVISION_DIR")
    root = Path(env) if env else RESULTS_PARENT / f"2026-10-28_revision_{git_sha()}"
    root.mkdir(parents=True, exist_ok=True)
    return root


def md5(path: Path, chunk: int = 1 << 20) -> str:
    h = hashlib.md5()
    with open(path, "rb") as f:
        while True:
            b = f.read(chunk)
            if not b:
                break
            h.update(b)
    return h.hexdigest()


@dataclass
class Manifest:
    """Collects what a sub-run needs to be reproduced and writes MANIFEST.txt."""

    out_dir: Path
    name: str
    config: dict[str, Any] = field(default_factory=dict)
    data_files: list[Path] = field(default_factory=list)
    timings: dict[str, float] = field(default_factory=dict)
    notes: list[str] = field(default_factory=list)
    t0: float = field(default_factory=time.time)

    def add_data(self, *paths: Path) -> None:
        for p in paths:
            p = Path(p)
            if p not in self.data_files:
                self.data_files.append(p)

    def time(self, key: str, seconds: float) -> None:
        self.timings[key] = round(float(seconds), 3)

    def write(self) -> Path:
        self.out_dir.mkdir(parents=True, exist_ok=True)
        lines = [
            f"run: {self.name}",
            f"commit: {git_sha(short=False)}",
            f"working tree has uncommitted changes: {git_dirty()}",
            f"command: {' '.join([Path(sys.executable).name] + sys.argv)}",
            f"python: {sys.version.split()[0]}  platform: {platform.platform()}",
            f"written: {time.strftime('%Y-%m-%d %H:%M:%S %Z')}",
            f"wall_seconds: {time.time() - self.t0:.1f}",
            "",
            "config:",
            json.dumps(self.config, indent=2, default=str),
            "",
            "data files (md5):",
        ]
        for p in self.data_files:
            real = Path(os.path.realpath(p))
            lines.append(f"  {md5(real)}  {p.relative_to(REPO) if p.is_relative_to(REPO) else p}")
        lines += ["", "timings (s):"]
        lines += [f"  {k}: {v}" for k, v in self.timings.items()]
        if self.notes:
            lines += ["", "notes:"] + [f"  - {n}" for n in self.notes]
        path = self.out_dir / "MANIFEST.txt"
        path.write_text("\n".join(lines) + "\n")
        return path


# ---------------------------------------------------------------------------
# Loaders
# ---------------------------------------------------------------------------


def chexpert_paths(backbone: str) -> tuple[Path, Path]:
    d = DATA / "chexpert_multibackbone"
    return d / f"{backbone}_embeddings.npy", d / f"{backbone}_metadata.csv"


def mimic_paths(backbone: str) -> tuple[Path, Path]:
    d = DATA / "mimic_cxr"
    return d / f"{backbone}_embeddings.npy", d / f"{backbone}_metadata.csv"


def celeba_paths() -> tuple[Path, Path]:
    d = DATA / "celeba"
    return d / "celeba_real_embeddings.npy", d / "celeba_real_attributes.csv"


def load_chexpert(backbone: str) -> tuple[np.ndarray, pd.DataFrame]:
    """CheXpert embeddings plus metadata. Race comes from the ``race`` column of
    ``chexpert_manifest.csv`` (same row order as every backbone's metadata)."""
    emb, meta_p = chexpert_paths(backbone)
    X = np.load(emb).astype(np.float32)
    meta = pd.read_csv(meta_p)
    man = pd.read_csv(DATA / "chexpert_multibackbone" / "chexpert_manifest.csv")
    assert len(man) == len(meta) == X.shape[0]
    assert (man["image_idx"].to_numpy() == meta["image_idx"].to_numpy()).all()
    meta["race"] = man["race"].to_numpy()
    return X, meta


def load_mimic(backbone: str) -> tuple[np.ndarray, pd.DataFrame]:
    emb, meta_p = mimic_paths(backbone)
    X = np.load(emb).astype(np.float32)
    meta = pd.read_csv(meta_p)
    assert X.shape[0] == len(meta)
    return X, meta


def load_celeba() -> tuple[np.ndarray, pd.DataFrame]:
    emb, attr_p = celeba_paths()
    X = np.load(emb).astype(np.float32)
    attrs = pd.read_csv(attr_p)
    assert X.shape[0] == len(attrs)
    return X, attrs


# ---------------------------------------------------------------------------
# Attribute definitions (spec section 5)
# ---------------------------------------------------------------------------


@dataclass
class Cell:
    """One audit cell: embeddings, task label Y and attribute A on a row subset."""

    dataset: str
    backbone: str
    attribute: str
    variant: str
    task: str
    X: np.ndarray
    y: np.ndarray
    a: np.ndarray
    rows: np.ndarray
    note: str = ""

    @property
    def key(self) -> str:
        return f"{self.dataset}|{self.backbone}|{self.task}|{self.attribute}|{self.variant}"

    def group_sizes(self) -> dict[str, int]:
        out: dict[str, int] = {}
        for yv in np.unique(self.y):
            for av in np.unique(self.a):
                out[f"Y={yv},A={av}"] = int(((self.y == yv) & (self.a == av)).sum())
        return out


def _binary_age(age: pd.Series) -> np.ndarray:
    return np.where(age.to_numpy(dtype=float) >= 60, "60+", "<60")


def _race_min_filter(
    race: np.ndarray, y: np.ndarray, min_n: int, per: str
) -> tuple[np.ndarray, str]:
    """Rows kept by the race sensitivity rule.

    per="group": keep named groups with at least ``min_n`` images overall.
    per="stratum": keep a (group, Y) cell only if it has at least ``min_n`` images.
    Returns a boolean mask and a note describing what was kept.
    """
    named = np.isin(race, RACE_NAMED)
    keep = np.zeros(len(race), dtype=bool)
    kept: list[str] = []
    if per == "group":
        for g in RACE_NAMED:
            m = named & (race == g)
            if m.sum() >= min_n:
                keep |= m
                kept.append(f"{g}({int(m.sum())})")
    else:
        for yv in np.unique(y):
            for g in RACE_NAMED:
                m = named & (race == g) & (y == yv)
                if m.sum() >= min_n:
                    keep |= m
                    kept.append(f"{g}|Y={yv}({int(m.sum())})")
    return keep, "kept " + ", ".join(kept) if kept else "no group qualifies"


def attribute_cells(
    dataset: str,
    backbone: str,
    X: np.ndarray,
    meta: pd.DataFrame,
    *,
    sensitivity: bool = True,
    task_col: str = "task_label",
    task_name: str | None = None,
) -> list[Cell]:
    """Primary attribute cells (sex, age 60+, White vs non-White) plus the
    sensitivity variants (age 4-bin, race named groups with 50+ images per group
    and per (group, Y) stratum)."""
    y_all = meta[task_col].to_numpy()
    valid = ~pd.isna(y_all)
    task = task_name or ("pneumothorax" if dataset == "chexpert" else "mortality")
    cells: list[Cell] = []

    def add(attr: str, variant: str, a: np.ndarray, mask: np.ndarray, note: str = "") -> None:
        m = mask & valid & ~pd.isna(a)
        rows = np.flatnonzero(m)
        a_sub = np.asarray(a)[rows].astype(str)
        y_sub = y_all[rows].astype(int)
        if len(np.unique(a_sub)) < 2:
            note = (note + "; " if note else "") + "fewer than 2 groups, not testable"
        cells.append(Cell(dataset, backbone, attr, variant, task, X[rows], y_sub, a_sub, rows, note))

    allrows = np.ones(len(meta), dtype=bool)
    add("sex", "primary", meta["sex"].to_numpy(), allrows)
    add("age", "primary", _binary_age(meta["age"]), meta["age"].notna().to_numpy())
    race = meta["race"].astype(str).to_numpy()
    add("race", "primary", np.where(race == "WHITE", "White", "non-White"), allrows)
    if sensitivity:
        add("age", "4bin", meta["age_bin"].astype(str).to_numpy(), meta["age_bin"].notna().to_numpy())
        for per in ("group", "stratum"):
            keep, note = _race_min_filter(race, np.nan_to_num(y_all.astype(float)).astype(int), 50, per)
            add("race", f"min50_per_{per}", race, keep, note)
    return cells


def celeba_cells(X: np.ndarray, attrs: pd.DataFrame) -> list[Cell]:
    out = []
    a = np.where(attrs["Male"].to_numpy() == 1, "Male", "Female")
    rows = np.arange(len(attrs))
    for t in CELEBA_TASKS:
        y = attrs[t].to_numpy().astype(int)
        out.append(Cell("celeba", "resnet50_celeba", "sex", "primary", t, X, y, a, rows))
    return out


def to_jsonable(obj: Any) -> Any:
    if isinstance(obj, dict):
        return {str(k): to_jsonable(v) for k, v in obj.items()}
    if isinstance(obj, list | tuple):
        return [to_jsonable(v) for v in obj]
    if isinstance(obj, np.integer):
        return int(obj)
    if isinstance(obj, np.floating):
        return None if not np.isfinite(obj) else float(obj)
    if isinstance(obj, float):
        return None if not np.isfinite(obj) else obj
    if isinstance(obj, np.bool_):
        return bool(obj)
    if isinstance(obj, np.ndarray):
        return to_jsonable(obj.tolist())
    return obj


def update_numbers(updates: dict[str, Any]) -> Path:
    """Merge headline numbers into ``numbers.json`` at the results root."""
    path = results_root() / "numbers.json"
    cur = json.loads(path.read_text()) if path.exists() else {}
    cur.update(to_jsonable(updates))
    path.write_text(json.dumps(dict(sorted(cur.items())), indent=2) + "\n")
    return path
