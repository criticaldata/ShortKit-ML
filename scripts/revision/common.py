"""Shared helpers for the CMPB revision scripts.

Paths, run manifests, data loaders and the attribute definitions fixed by the
revision method decisions (section 5). Every script under ``scripts/revision``
imports from here so that all runs use the same cohorts and the same folder.
"""

from __future__ import annotations

import functools
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


CHEXPERT_JOIN_NOTE = (
    "CheXpert sex in data/chexpert_multibackbone/*_metadata.csv is inverted (the HF mirror "
    "codes 0 = Male; extraction mapped 0 to Female). Sex, age, view, AP/PA and diagnosis "
    "labels are read from the official data/train.csv joined by image path via "
    "race_mapping.csv; patient_id from race_mapping.csv. Race is recoded from the CheXpert "
    "demographics file (data/CHEXPERT DEMO.xlsx, PRIMARY_RACE, joined by patient): images "
    "without a demographics record, or recorded as Unknown or Patient Refused, are missing "
    "(the shipped manifest coded them OTHER, so non-White), and "
    "Native Hawaiian or Pacific Islander is its own group NHPI (the manifest coded it ASIAN). "
    "Age as an attribute is the patient's age at their first study in the cohort, so it is "
    "constant per patient."
)


def _code_chexpert_race(v) -> str | None:
    if pd.isna(v):
        return None
    v = str(v)
    if "Unknown" in v or v == "Patient Refused":
        return None  # no race recorded: missing, like images without a demographics record
    for prefix, code in (("White", "WHITE"), ("Black", "BLACK"), ("Asian", "ASIAN"),
                         ("Native Hawaiian", "NHPI"), ("Pacific Islander", "NHPI")):
        if v.startswith(prefix):
            return code
    return "OTHER"  # Other, American Indian / Native American


@functools.lru_cache(maxsize=1)
def _chexpert_demo() -> pd.DataFrame:
    return pd.read_excel(DATA / "CHEXPERT DEMO.xlsx", usecols=["PATIENT", "PRIMARY_RACE"])
CHEXPERT_LABEL_COLS = [
    "No Finding", "Enlarged Cardiomediastinum", "Cardiomegaly", "Lung Opacity", "Lung Lesion",
    "Edema", "Consolidation", "Pneumonia", "Atelectasis", "Pneumothorax", "Pleural Effusion",
    "Pleural Other", "Fracture", "Support Devices",
]


def chexpert_meta() -> pd.DataFrame:
    """Per-image CheXpert metadata in embedding row order (see CHEXPERT_JOIN_NOTE)."""
    d = DATA / "chexpert_multibackbone"
    man = pd.read_csv(d / "chexpert_manifest.csv")
    rmap = pd.read_csv(d / "race_mapping.csv", dtype={"patient_id": str})
    train = pd.read_csv(DATA / "train.csv")
    assert (rmap["image_idx"].to_numpy() == man["image_idx"].to_numpy()).all()
    j = rmap[["image_idx", "path", "patient_id"]].merge(
        train, left_on="path", right_on="Path", how="left", validate="one_to_one"
    )
    assert j["Path"].notna().all() and len(j) == len(man)
    meta = pd.DataFrame(
        {
            "image_idx": man["image_idx"].to_numpy(),
            "path": j["path"].to_numpy(),
            "patient_id": j["patient_id"].to_numpy(),
            "sex": j["Sex"].to_numpy(),
            "age": j["Age"].to_numpy(dtype=float),
            "view": j["Frontal/Lateral"].to_numpy(),
            "ap_pa": j["AP/PA"].fillna("NA").to_numpy(),
            "race_shipped": man["race"].to_numpy(),
            "age_bin": man["age_bin"].to_numpy(),
            "sex_shipped_metadata": man["sex"].to_numpy(),
        }
    )
    for c in CHEXPERT_LABEL_COLS:
        meta[c] = j[c].to_numpy(dtype=float)  # 1 positive, 0 negative, -1 uncertain, NaN blank
    # race from the demographics file (see CHEXPERT_JOIN_NOTE)
    pat = meta["path"].str.extract(r"(patient\d+)")[0]
    demo = _chexpert_demo().drop_duplicates("PATIENT")
    raw = pd.DataFrame({"PATIENT": pat}).merge(demo, on="PATIENT", how="left", validate="many_to_one")
    meta["race_raw"] = raw["PRIMARY_RACE"].to_numpy()
    meta["race"] = [_code_chexpert_race(v) for v in meta["race_raw"]]
    changed = meta["race"].astype(str) != meta["race_shipped"].astype(str)
    expected = meta["race"].isna() & (meta["race_shipped"] == "OTHER") | (
        (meta["race"] == "NHPI") & (meta["race_shipped"] == "ASIAN"))
    assert (changed == expected).all(), "race recoding differs from the manifest beyond the two known cases"
    # age at the patient's first study in the cohort (patient-constant attribute)
    study = meta["path"].str.extract(r"/study(\d+)/")[0].astype(int)
    first = meta.assign(_s=study).sort_values(["patient_id", "_s"]).groupby("patient_id")["age"].first()
    meta["age_first_study"] = meta["patient_id"].map(first).to_numpy(dtype=float)
    meta["age_bin_first_study"] = pd.cut(meta["age_first_study"], [-np.inf, 40, 60, 80, np.inf], right=False,
                                         labels=["<40", "40-60", "60-80", "80+"]).astype(str)
    # audit task (spec section 5): the existing task label, Pneumothorax positive vs rest
    meta["task_label"] = (meta["Pneumothorax"] == 1).astype(int)
    assert (meta["task_label"].to_numpy() == man["task_label"].to_numpy()).all()
    assert (meta["age"].to_numpy() == man["age"].to_numpy()).all()
    return meta


def load_chexpert(backbone: str) -> tuple[np.ndarray, pd.DataFrame]:
    emb, _ = chexpert_paths(backbone)
    X = np.load(emb).astype(np.float32)
    meta = chexpert_meta()
    assert X.shape[0] == len(meta)
    return X, meta


def load_celeba_identities() -> np.ndarray:
    ids = pd.read_csv(
        DATA / "celeba_real" / "celeba" / "identity_CelebA.txt",
        sep=" ",
        header=None,
        names=["image_id", "identity"],
    )
    _, attrs = celeba_paths()
    a = pd.read_csv(attrs, usecols=["image_id"])
    return a.merge(ids, on="image_id", how="left")["identity"].to_numpy()


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
    groups: np.ndarray | None = None

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
        grp = meta["patient_id"].to_numpy()[rows] if "patient_id" in meta.columns else None
        cells.append(
            Cell(dataset, backbone, attr, variant, task, X[rows], y_sub, a_sub, rows, note, grp)
        )

    allrows = np.ones(len(meta), dtype=bool)
    add("sex", "primary", meta["sex"].to_numpy(), allrows)
    # CheXpert: age at the first study, constant per patient (section 9)
    age_col = "age_first_study" if "age_first_study" in meta.columns else "age"
    bin_col = "age_bin_first_study" if "age_bin_first_study" in meta.columns else "age_bin"
    add("age", "primary", _binary_age(meta[age_col]), meta[age_col].notna().to_numpy())
    race_s = meta["race"]
    race_bin = np.where(race_s.isna(), None, np.where(race_s.astype(str) == "WHITE", "White", "non-White"))
    add("race", "primary", race_bin, allrows, "missing race excluded" if race_s.isna().any() else "")
    race = race_s.astype(str).to_numpy()
    if sensitivity:
        add("age", "4bin", meta[bin_col].astype(str).to_numpy(), meta[bin_col].notna().to_numpy())
        for per in ("group", "stratum"):
            keep, note = _race_min_filter(race, np.nan_to_num(y_all.astype(float)).astype(int), 50, per)
            add("race", f"min50_per_{per}", race, keep, note)
    return cells


def celeba_cells(X: np.ndarray, attrs: pd.DataFrame) -> list[Cell]:
    """CelebA: A = Male (attribute 'Male' == 1), Y = each task attribute;
    grouped by celebrity identity (repeated identities exist)."""
    out = []
    ident = load_celeba_identities()
    a = np.where(attrs["Male"].to_numpy() == 1, "Male", "Female")
    rows = np.arange(len(attrs))
    for t in CELEBA_TASKS:
        y = attrs[t].to_numpy().astype(int)
        out.append(Cell("celeba", "resnet50_celeba", "sex", "primary", t, X, y, a, rows, "", ident))
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
