#!/usr/bin/env python3
"""Build the revision tables (LaTeX tabular, booktabs), figures (PDF) and
numbers.json from the result folders only. No number is typed by hand.

Output goes to <results>/tables, <results>/figures and <results>/numbers.json,
and is copied with the manifests to output/paper_tables/revision/ in the repo.
"""

from __future__ import annotations

import json
import os
import shutil
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
import matplotlib  # noqa: E402
from common import (  # noqa: E402
    BACKBONE_LABELS,
    CHEXPERT_PUBLIC,
    CHEXPERT_BACKBONES,
    MIMIC_BACKBONES,
    REPO,
    git_sha,
    results_root,
    to_jsonable,
)

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
from matplotlib.colors import LinearSegmentedColormap  # noqa: E402

from shortcut_detect.audit.conditional import benjamini_hochberg  # noqa: E402
from shortcut_detect.benchmark.figures import _apply_pub_style  # noqa: E402

ROOT = results_root()
TAB = ROOT / "tables"
FIG = ROOT / "figures"
TAB.mkdir(exist_ok=True)
FIG.mkdir(exist_ok=True)

DET = ["probe", "statistical", "geometric", "bias_direction", "hbac_nmi", "frequency_topk"]
DET_LABEL = {
    "probe": "Probe",
    "statistical": "Stat.",
    "geometric": "Geom.",
    "bias_direction": "BiasDir",
    "hbac_nmi": "HBAC-NMI",
    "frequency_topk": "Freq.",
    "probe_unconditional": "Probe, unconditional",
}
DET_KEY = {"probe": "probe", "statistical": "statistical", "geometric": "geometric",
           "bias_direction": "biasdir", "hbac_nmi": "hbac", "frequency_topk": "frequency"}
# Okabe-Ito, colour-blind safe
OI = {"orange": "#E69F00", "sky": "#56B4E9", "green": "#009E73", "yellow": "#F0E442",
      "blue": "#0072B2", "vermillion": "#D55E00", "purple": "#CC79A7", "black": "#000000"}
DETECTION_CMAP = LinearSegmentedColormap.from_list("det", ["#f7fbff", OI["blue"]])
FALSEPOS_CMAP = LinearSegmentedColormap.from_list("fp", ["#fff7ec", OI["vermillion"]])

NUM: dict[str, str] = {}
MISALIGNED: set[tuple[str, str]] = set()  # (dataset, backbone), filled by alignment()
# the MIMIC-CXR "medsiglip" files are google/medsiglip-448 vision pooler output (re-embedded on 50 images,
# cosine 1.000), although the source README names google/siglip-so400m-patch14-384; no relabel needed
DATASET_LABELS: dict[tuple[str, str], str] = {}


def bb_label(ds: str, bb: str) -> str:
    return DATASET_LABELS.get((ds, bb), BACKBONE_LABELS.get(bb, bb))
RAW: dict[str, object] = {}


def put(key: str, raw, fmt: str | None = "{:.2f}", text: str | None = None) -> None:
    RAW[key] = to_jsonable(raw)
    if text is not None:
        NUM[key] = text
    elif isinstance(raw, str) or fmt is None:
        NUM[key] = str(raw)
    else:
        NUM[key] = fmt.format(raw)
        if NUM[key].startswith("-0.") and float(NUM[key]) == 0.0:  # no "-0.00" in the paper
            NUM[key] = NUM[key][1:]


def rng_text(lo: float, hi: float, fmt="{:.2f}") -> str:
    return f"{fmt.format(lo)} to {fmt.format(hi)}"


def tex_escape(s) -> str:
    s = str(s)
    for a, b in (("\\", r"\textbackslash{}"), ("&", r"\&"), ("%", r"\%"), ("_", r"\_"), ("#", r"\#")):
        s = s.replace(a, b)
    return s


def write_tabular(name: str, df: pd.DataFrame, colfmt: str | None = None, escape_cells=True,
                  header: list[str] | None = None, midrules: list[int] | None = None,
                  prehead: list[str] | None = None) -> None:
    cols = header or [tex_escape(c) for c in df.columns]
    colfmt = colfmt or ("l" * 1 + "c" * (len(cols) - 1))
    lines = [f"\\begin{{tabular}}{{{colfmt}}}", "\\toprule", *(prehead or []),
             " & ".join(cols) + " \\\\", "\\midrule"]
    for i, (_, r) in enumerate(df.iterrows()):
        if midrules and i in midrules:
            lines.append("\\midrule")
        cells = [tex_escape(v) if escape_cells else str(v) for v in r.tolist()]
        lines.append(" & ".join(cells) + " \\\\")
    lines += ["\\bottomrule", "\\end{tabular}"]
    (TAB / f"{name}.tex").write_text("\n".join(lines) + "\n")


def load(rel: str) -> pd.DataFrame | None:
    p = ROOT / rel
    if not p.exists():
        print(f"[skip] {rel} not found")
        return None
    return pd.read_csv(p)


def f2(x, d=2):
    if x is None or (isinstance(x, float) and not np.isfinite(x)):
        return "n/a"
    t = f"{x:.{d}f}"
    return t[1:] if t.startswith("-0.") and float(t) == 0.0 else t  # no "-0.00"


def pfmt(p):
    return "<0.01" if p < 0.01 else f"{p:.2f}"


def apply_flag_rule(df: pd.DataFrame, cell_keys: list[str]) -> pd.DataFrame:
    """Decision rule: the probe is flagged at p < 0.05 on its own; the five secondary
    detectors are flagged by Benjamini-Hochberg across their five p-values within the
    cell (q < 0.05). Recomputed from p_value. The stored flag and q (BH across all six
    detectors) are kept as flag_bh6 and p_bh6 for the comparison in the letter. Rows of
    detectors outside DET keep their stored flag."""
    df = df.copy()
    df["flag_bh6"] = df["flag"].astype(str).str.lower().isin(["true", "1"])
    df["p_bh6"] = df["p_bh"].astype(float)
    df["flag"] = df["flag_bh6"]
    df["p_bh"] = df["p_bh6"]
    probe = df.detector == "probe"
    df.loc[probe, "flag"] = df.loc[probe, "p_value"] < 0.05
    df.loc[probe, "p_bh"] = np.nan
    sec = df[df.detector.isin(DET[1:])]
    for _, g in sec.groupby(cell_keys, dropna=False):
        q = benjamini_hochberg(g.p_value.to_numpy(float))
        df.loc[g.index, "p_bh"] = q
        df.loc[g.index, "flag"] = q < 0.05
    return df


# ---------------------------------------------------------------------------
# Settings
# ---------------------------------------------------------------------------


def clinical_config() -> dict:
    """Config block of the clinical audit manifest, so settings match the run."""
    text = (ROOT / "clinical_audit" / "MANIFEST.txt").read_text()
    block = text.split("config:\n", 1)[1].split("\n}\n", 1)[0] + "\n}"
    return json.loads(block)


def settings():
    cfg = clinical_config()
    put("K_probe", cfg["K_probe"], "{:d}")
    put("K_secondary", cfg["K"], "{:d}")
    put("probe_C", cfg["C"], "{:.0f}")
    put("hbac_k", cfg["n_clusters"], "{:d}")
    put("freq_topk", round(100 * cfg["top_k_fraction"]), "{:d}")
    RAW["freq_topk_fraction"] = cfg["top_k_fraction"]
    put("code_commit", git_sha())
    put("clinical_seed", cfg["seed"], "{:d}")
    put("n_bootstrap", cfg["n_bootstrap"], "{:d}")
    put("alpha", 0.05, "{:.2f}")
    put("synthetic_task_dims", 4, "{:d}")
    put("synthetic_attr_dims", 4, "{:d}")
    put("synthetic_task_shift", 1.0, "{:.1f}")
    put("xor_param", "pairs of attribute dimensions are shifted by (u1, u2) times delta/2, with u1 a random "
        "sign and u2 = u1 for A = 1 and u2 = -u1 for A = 0; both groups have zero mean in every "
        "dimension and, given the partner component, differ by delta standard deviations")
    put("varshift_param", "equal means; attribute dimensions have standard deviation 1 + delta for A = 1 "
        "and 1 for A = 0")
    put("celeba_mapping", "A = Male (CelebA attribute Male = 1); Y = Blond_Hair, Heavy_Makeup or "
        "Attractive (= 1); 10,000 images grouped by celebrity identity (6,066 identities)")


# ---------------------------------------------------------------------------
# Synthetic
# ---------------------------------------------------------------------------


def synthetic():
    df = load("synthetic/synthetic_runs.csv")
    if df is None:
        return
    df["reject"] = df["p_value"] < 0.05
    df = apply_flag_rule(df, ["variant", "rho", "delta", "n", "d", "seed"])
    null = df[(df.delta == 0) & (df.variant == "linear")]  # the three variants coincide at delta 0
    alt = df[df.delta > 0]

    # null FPR table, plus the large check at rho 0.9 (run_null_check.py) when present
    big = load("synthetic/null_check_rho09.csv")
    if big is not None:
        big["reject"] = big["p_value"] < 0.05
        put("n_nullcheck_datasets", int(big.seed.nunique()), "{:d}")
    put("n_null_datasets_per_rho", int(null[(null.detector == "probe") & (null.rho == 0.9)].shape[0]), "{:d}")
    rows = []
    for d in DET + ["probe_unconditional"]:
        sub = null[null.detector == d]
        r = {"Detector": DET_LABEL[d]}
        for rho in sorted(df.rho.unique()):
            r[f"rho={rho}"] = f2(sub[sub.rho == rho].reject.mean())
        r["All"] = f2(sub.reject.mean())
        r["Flag"] = f2(sub.flag.mean())
        if big is not None:
            b = big[big.detector == d]
            r["Large check"] = f2(b.reject.mean()) if len(b) else "--"
            if len(b) and d in DET_KEY:
                put(f"fpr_nullcheck_{DET_KEY[d]}", b.reject.mean())
        rows.append(r)
        if d in DET_KEY:
            put(f"fpr_null_{DET_KEY[d]}", sub.reject.mean())
            put(f"fpr_null_bhflag_{DET_KEY[d]}", sub.flag.mean())  # probe: raw p; secondaries: BH over five
            put(f"fpr_null_bh6flag_{DET_KEY[d]}", sub.flag_bh6.mean())  # BH across all six (earlier rule)
            for rho in sorted(df.rho.unique()):  # per-rho values next to the pooled one
                put(f"fpr_null_{DET_KEY[d]}_rho{round(10 * rho):02d}", sub[sub.rho == rho].reject.mean())
    put("fpr_null_probe_rho09", null[(null.detector == "probe") & (null.rho == 0.9)].reject.mean())
    put("fpr_null_unconditional_probe_rho09",
        null[(null.detector == "probe_unconditional") & (null.rho == 0.9)].reject.mean())
    RAW["n_null_datasets"] = int(null[null.detector == "probe"].shape[0])
    put("fpr_null_bhflag_secondary_max", max(null[null.detector == d].flag.mean() for d in DET[1:]))
    t = pd.DataFrame(rows)
    hdr = ["Detector"] + [f"$\\rho={r}$" for r in sorted(df.rho.unique())] + ["All", "Flag"]
    if big is not None:
        hdr.append("$\\rho=0.9$, large")
    write_tabular("synthetic_null_fpr", t, "l" + "c" * (len(hdr) - 1), header=hdr, midrules=[len(DET)])

    # sensitivity / specificity table (BH flags), pooled over rho, n, d
    rows = []
    deltas = sorted(alt.delta.unique())
    for v in ["linear", "xor", "variance"]:
        for d in DET:
            r = {"Variant": v if d == DET[0] else "", "Detector": DET_LABEL[d],
                 "Spec.": f2(1 - null[null.detector == d].flag.mean())}
            for dl in deltas:
                r[f"{dl}"] = f2(alt[(alt.variant == v) & (alt.detector == d) & (alt.delta == dl)].flag.mean())
            rows.append(r)
    t = pd.DataFrame(rows)
    hdr = ["Variant", "Detector", "Spec. ($\\delta=0$)"] + [f"$\\delta={x:g}$" for x in deltas]
    write_tabular("synthetic_sens_spec", t, "ll" + "c" * (len(hdr) - 2), header=hdr,
                  midrules=[len(DET), 2 * len(DET)])

    def sens(det, var, dl):
        return alt[(alt.variant == var) & (alt.detector == det) & (alt.delta == dl)].flag.mean()

    put("sens_probe_delta02_linear", sens("probe", "linear", 0.2))
    put("sens_probe_delta05_linear", sens("probe", "linear", 0.5))
    put("sens_probe_delta08_linear", sens("probe", "linear", 0.8))
    put("xor_delta_reported", 2.0, "{:.1f}")
    for det, k in (("probe", "probe"), ("hbac_nmi", "hbac"), ("statistical", "statistical")):
        put(f"sens_{k}_xor", sens(det, "xor", 2.0))
        put(f"sens_{k}_var", sens(det, "variance", 2.0))
    best_xor = max(DET, key=lambda d: sens(d, "xor", 2.0))
    put("best_detector_xor", DET_LABEL[best_xor])

    # heatmap: detection rate by delta, panels per variant; delta 0 in its own colour
    _apply_pub_style()
    fig, axes = plt.subplots(1, 3, figsize=(13, 3.6), sharey=True)
    for ax, v in zip(axes, ["linear", "xor", "variance"], strict=True):
        mat = np.array([[null[null.detector == d].flag.mean()] +
                        [sens(d, v, dl) for dl in deltas] for d in DET])
        fp = np.ma.masked_all(mat.shape)
        fp[:, 0] = mat[:, 0]
        det = np.ma.array(mat, mask=np.zeros_like(mat, bool))
        det.mask[:, 0] = True
        ax.imshow(det, cmap=DETECTION_CMAP, vmin=0, vmax=1, aspect="auto")
        ax.imshow(fp, cmap=FALSEPOS_CMAP, vmin=0, vmax=0.2, aspect="auto")
        for i in range(mat.shape[0]):
            for j in range(mat.shape[1]):
                ax.text(j, i, f"{mat[i, j]:.2f}", ha="center", va="center", fontsize=8.5,
                        color="white" if (j > 0 and mat[i, j] > 0.6) else "black")
        ax.axvline(0.5, color="black", lw=1.2)
        ax.set_xticks(range(len(deltas) + 1), ["0\n(FP)"] + [f"{x:g}" for x in deltas])
        ax.set_yticks(range(len(DET)), [DET_LABEL[d] for d in DET])
        ax.set_xlabel("Cohen's d of the attribute signal ($\\delta$)")
        ax.set_title(v)
        ax.grid(False)
    fig.suptitle("Flag rate (probe at $p<0.05$; secondaries at BH $q<0.05$ across the five); "
                 "$\\delta=0$ column (orange-red) is the false-positive rate", fontsize=10)
    fig.savefig(FIG / "synthetic_detection_heatmap.pdf")
    plt.close(fig)

    # aggregation sweep: >= k of 6 detectors flagged
    w = df[df.detector.isin(DET)].pivot_table(
        index=["variant", "rho", "delta", "n", "d", "seed"], columns="detector", values="flag", aggfunc="first"
    ).astype(float)
    cnt = w[DET].sum(1).reset_index(name="n_flag")
    cnt_null = cnt[(cnt.delta == 0) & (cnt.variant == "linear")]
    sweep = []
    for k in range(1, 7):
        r = {"k": k, "fpr": float((cnt_null.n_flag >= k).mean())}
        for v in ["linear", "xor", "variance"]:
            for dl in deltas:
                s = cnt[(cnt.variant == v) & (cnt.delta == dl)]
                r[f"sens_{v}_{dl:g}"] = float((s.n_flag >= k).mean())
        sweep.append(r)
    sw = pd.DataFrame(sweep)
    sw.to_csv(TAB / "aggregation_sweep.csv", index=False)
    rows = [{"k": int(r.k), "FPR": f2(r.fpr), **{f"{dl:g}": f2(r[f"sens_linear_{dl:g}"]) for dl in deltas}}
            for _, r in sw.iterrows()]
    write_tabular("aggregation_sweep", pd.DataFrame(rows),
                  header=["$k$", "FPR ($\\delta=0$)"] + [f"$\\delta={x:g}$" for x in deltas])
    fig, axes = plt.subplots(1, 3, figsize=(12, 3.5), sharey=True)
    colors = [OI["sky"], OI["blue"], OI["green"], OI["purple"], OI["black"]]
    for ax, v in zip(axes, ["linear", "xor", "variance"], strict=True):
        for c, dl in zip(colors, deltas):
            ax.plot(sw.k, sw[f"sens_{v}_{dl:g}"], marker="o", color=c, label=f"$\\delta={dl:g}$")
        ax.plot(sw.k, sw.fpr, marker="s", ls="--", color=OI["vermillion"], label="FPR ($\\delta=0$)")
        ax.set_xlabel("$k$: flagged by at least $k$ of 6 detectors")
        ax.set_title(v)
    axes[0].set_ylabel("Rate")
    axes[-1].legend(frameon=False, fontsize=8, loc="upper right")
    fig.savefig(FIG / "aggregation_sweep.pdf")
    plt.close(fig)

    # flag correlation and effective number of detectors
    def eff_n(frame):
        c = frame[DET].corr().to_numpy()
        # a detector whose flag never varies has no defined correlation; count it as
        # uncorrelated (off-diagonal 0) with unit variance (diagonal 1)
        c = np.nan_to_num(c)
        np.fill_diagonal(c, 1.0)
        lam = np.clip(np.linalg.eigvalsh(c), 0, None)
        return float(lam.sum() ** 2 / (lam ** 2).sum())

    corr = w[DET].corr()
    off = corr.to_numpy()[~np.eye(len(DET), dtype=bool)]
    put("mean_flag_correlation", float(np.nanmean(off)))
    put("effective_n_detectors", eff_n(w), "{:.1f}")
    wr = w.reset_index()
    for (v, dl), g in wr.groupby(["variant", "delta"]):
        put(f"effective_n_detectors_{v}_d{f'{dl:g}'.replace('.', '')}", eff_n(g), "{:.1f}")
    ct = corr.round(2)
    ct.index = [DET_LABEL[d] for d in ct.index]
    ct.columns = [DET_LABEL[d] for d in ct.columns]
    t = ct.reset_index().rename(columns={"index": ""})
    write_tabular("flag_correlation", t.astype(str))

    # multiple-testing correction on null data
    sc = load("synthetic/statistical_correction.csv")
    if sc is not None:
        fig, ax = plt.subplots(figsize=(5.5, 3.6))
        lab = {"uncorrected": "Uncorrected", "bonferroni": "Bonferroni", "holm": "Holm",
               "fdr_bh": "FDR (BH)", "fdr_by": "FDR (BY)"}
        cols = {"uncorrected": OI["vermillion"], "bonferroni": OI["blue"], "holm": OI["sky"],
                "fdr_bh": OI["green"], "fdr_by": OI["purple"]}
        for c, g in sc.groupby("correction"):
            ax.plot(g.d, g.fpr, marker="o", label=lab[c], color=cols[c])
        ax.axhline(0.05, color="grey", ls=":", lw=1)
        ax.set_xscale("log", base=2)
        ax.set_xlabel("Embedding dimension $d$")
        ax.set_ylabel("Share of null datasets with\nany significant dimension")
        ax.legend(frameon=False, fontsize=8)
        fig.savefig(FIG / "synthetic_statistical_correction.pdf")
        plt.close(fig)


# ---------------------------------------------------------------------------
# Clinical audit
# ---------------------------------------------------------------------------


def f3s(x):
    """Three decimals; a non-zero value that would print as 0.000 goes to scientific notation."""
    t = f"{x:.3f}"
    if x != 0 and float(t) == 0:
        m, e = f"{x:.1e}".split("e")
        return f"${m}{{\\times}}10^{{{int(e)}}}$"
    return t


def _cell_text(sub: pd.DataFrame, det: str, with_ci=False):
    r = sub[sub.detector == det]
    if r.empty:
        return "n/a"
    r = r.iloc[0]
    star = "$^{*}$" if bool(r.flag) else ""
    if with_ci:
        return f"{r.statistic:.2f} ({r.ci_low:.2f}, {r.ci_high:.2f}){star}"
    d = 3 if det in {"statistical", "hbac_nmi"} else 2
    return f"{r.statistic:.{d}f}{star}"


def clinical():
    df = load("clinical_audit/audit_rows.csv")
    if df is None:
        return None
    df = df[df.evaluated.astype(bool)]
    df = apply_flag_rule(df, ["dataset", "backbone", "task", "attribute", "variant"])
    prim = df[df.variant == "primary"]

    for ds, order in (("chexpert", ["resnet50", "densenet121", "vit_b_16", "vit_b_32", "dinov2_base",
                                     "rad_dino", "medsiglip"]),
                      ("mimic", ["rad_dino", "vit16_cls", "vit32_cls", "medsiglip"])):
        rows, mids = [], []
        for bb in order:
            for attr in ["sex", "age", "race"]:
                sub = prim[(prim.dataset == ds) & (prim.backbone == bb) & (prim.attribute == attr)]
                if sub.empty:
                    continue
                pr = sub[sub.detector == "probe"].iloc[0]
                rows.append({
                    "Backbone": bb_label(ds, bb) if attr == "sex" else "",
                    "Attribute": {"sex": "Sex", "age": "Age 60+", "race": "White vs non-White"}[attr],
                    "Probe AUROC (95% CI)": _cell_text(sub, "probe", True),
                    "p": pfmt(pr.p_value),
                    **{DET_LABEL[d]: _cell_text(sub, d) for d in DET[1:]},
                    "Flags": f"{int(sub[sub.detector.isin(DET)].flag.sum())}/6",
                })
            mids.append(len(rows))
        t = pd.DataFrame(rows)
        hdr = ["Encoder", "Attribute", "Probe AUROC (95\\% CI)", "$p$"] + [DET_LABEL[d] for d in DET[1:]] + ["Flags"]
        write_tabular(f"clinical_audit_{ds}", t, "ll" + "c" * (len(hdr) - 2), escape_cells=False,
                      header=hdr, midrules=mids[:-1])
        # every detector's raw statistics per cell (statistic, null mean and SD, raw p, BH q)
        attrs = ["sex", "age", "race"]
        rows, mids = [], []
        for bb in order:
            if prim[(prim.dataset == ds) & (prim.backbone == bb)].empty:
                continue
            for d in DET:
                r = {"Backbone": bb_label(ds, bb) if d == DET[0] else "", "Detector": DET_LABEL[d]}
                for attr in attrs:
                    s = prim[(prim.dataset == ds) & (prim.backbone == bb) & (prim.attribute == attr) & (prim.detector == d)]
                    if s.empty:
                        r.update({f"{attr}_{k}": "n/a" for k in ("s", "m", "sd", "p", "q")})
                        continue
                    s = s.iloc[0]
                    dd = 3 if d in {"statistical", "hbac_nmi"} else 2
                    r.update({f"{attr}_s": f"{s.statistic:.{dd}f}", f"{attr}_m": f3s(s.null_mean),
                              f"{attr}_sd": f3s(s.null_sd), f"{attr}_p": f"{s.p_value:.3f}",
                              f"{attr}_q": "--" if d == "probe" else f"{s.p_bh:.3f}"})  # the probe is judged on p
                rows.append(r)
            mids.append(len(rows))
        hdr = ["Encoder", "Detector"] + ["$s$", "Null mean", "Null SD", "$p$", "$q$"] * len(attrs)
        pre = [" & & " + " & ".join(f"\\multicolumn{{5}}{{c}}{{{lab}}}" for lab in
                                    ("Sex", "Age 60+", "White vs non-White")) + " \\\\",
               " ".join(f"\\cmidrule(lr){{{3 + 5 * i}-{7 + 5 * i}}}" for i in range(len(attrs)))]
        write_tabular(f"clinical_audit_raw_{ds}", pd.DataFrame(rows), "ll" + "c" * (len(hdr) - 2),
                      escape_cells=False, header=hdr, midrules=mids[:-1], prehead=pre)
        # summaries leave out backbones whose rows fail the alignment check (rows stay in the table)
        ok = ~prim.backbone.map(lambda b, ds=ds: (ds, b) in MISALIGNED)
        pp = prim[(prim.dataset == ds) & (prim.detector == "probe") & ok]
        put(f"n_cells_excluded_misaligned_{ds}",
            int(((prim.dataset == ds) & (prim.detector == "probe") & ~ok).sum()), "{:d}")
        put(f"n_cells_probe_sig_{ds}", int(pp.flag.sum()), "{:d}")  # probe at p < 0.05
        put(f"n_cells_probe_raw_sig_{ds}", int((pp.p_value < 0.05).sum()), "{:d}")
        put(f"n_cells_probe_sig_bh6_{ds}", int(pp.flag_bh6.sum()), "{:d}")  # BH across all six (earlier rule)
        if "cell_seconds" in pp:  # wall time of one cell, all six detectors, on the audit machine
            put(f"cell_seconds_{ds}", [pp.cell_seconds.min(), pp.cell_seconds.max()],
                text=rng_text(pp.cell_seconds.min(), pp.cell_seconds.max(), "{:.0f}"))
        put(f"n_cells_primary_{ds}", int(len(pp)), "{:d}")
        sx = pp[pp.attribute == "sex"].statistic
        put(f"probe_auroc_sex_min_{ds}", sx.min())
        put(f"probe_auroc_sex_max_{ds}", sx.max())
        for attr in ["age", "race"]:
            s = pp[pp.attribute == attr].statistic
            put(f"probe_auroc_{attr}_range_{ds}", [s.min(), s.max()], text=rng_text(s.min(), s.max()))
            put(f"n_cells_probe_sig_{attr}_{ds}", int(pp[pp.attribute == attr].flag.sum()), "{:d}")
        allflag = prim[(prim.dataset == ds) & prim.detector.isin(DET) & ok]
        put(f"n_secondary_flags_{ds}", int(allflag[allflag.detector != "probe"].flag.sum()), "{:d}")
    m = prim[(prim.dataset == "mimic") & (prim.backbone == "rad_dino") & (prim.attribute == "sex") & (prim.detector == "probe")]
    if not m.empty:
        r = m.iloc[0]
        put("probe_auroc_mimic_raddino_sex", r.statistic)
        put("probe_auroc_ci_mimic_raddino_sex", [r.ci_low, r.ci_high], text=rng_text(r.ci_low, r.ci_high))
    # MedSigLIP-448, re-extracted for the revision (the stored array was not this checkpoint)
    for attr in ["sex", "age", "race"]:
        m = prim[(prim.dataset == "chexpert") & (prim.backbone == "medsiglip") & (prim.attribute == attr)
                 & (prim.detector == "probe")]
        if not m.empty:
            put(f"probe_auroc_chexpert_medsiglip_{attr}", m.iloc[0].statistic)
            put(f"probe_p_chexpert_medsiglip_{attr}", m.iloc[0].p_value, "{:.3f}")
            put(f"probe_q_bh6_chexpert_medsiglip_{attr}", m.iloc[0].p_bh6)  # q under BH across all six
    # grouped vs plain CV on CheXpert
    for bb in df[df.dataset == "chexpert"].backbone.unique():
        g = prim[(prim.dataset == "chexpert") & (prim.backbone == bb) & (prim.attribute == "sex")]
        a, b = g[g.detector == "probe"], g[g.detector == "probe_plain_cv"]
        if len(a) and len(b):
            key = f"probe_auroc_sex_grouped_vs_plain_chexpert_{bb.replace('_', '')}"
            put(key, [a.statistic.iloc[0], b.statistic.iloc[0]],
                text=f"{a.statistic.iloc[0]:.3f} grouped vs {b.statistic.iloc[0]:.3f} plain")
    pr_all = prim[prim.detector == "probe"]
    if "repeat_sd" in pr_all.columns:
        for ds in ("chexpert", "mimic", "celeba"):
            v = pr_all[pr_all.dataset == ds].repeat_sd
            if v.notna().any():
                put(f"probe_repeat_sd_max_{ds}", float(v.max()), "{:.3f}")
    if "n_inconsistent_groups" in pr_all.columns:
        c_ = pr_all[pr_all.dataset == "celeba"]
        if not c_.empty:
            put("celeba_n_inconsistent_identities", int(c_.n_inconsistent_groups.max()), "{:d}")
    # CelebA
    c = prim[prim.dataset == "celeba"]
    if not c.empty:
        rows = []
        for task, key in (("Blond_Hair", "blond"), ("Heavy_Makeup", "makeup"), ("Attractive", "attractive")):
            sub = c[c.task == task]
            if sub.empty:
                continue
            gs = json.loads(sub.iloc[0].group_sizes)
            rows.append({"Task (Y)": task.replace("_", " "), "n": int(sub.iloc[0].n_images),
                         "Y=1 Male/Female": f"{gs.get('Y=1,A=Male', 0)}/{gs.get('Y=1,A=Female', 0)}",
                         "Probe AUROC (95% CI)": _cell_text(sub, "probe", True),
                         **{DET_LABEL[d]: _cell_text(sub, d) for d in DET[1:]},
                         "Flags": f"{int(sub[sub.detector.isin(DET)].flag.sum())}/6"})
            put(f"celeba_probe_auroc_{key}", sub[sub.detector == "probe"].statistic.iloc[0])
        hdr = ["Task ($Y$)", "$n$", "$Y{=}1$ M/F", "Probe AUROC (95\\% CI)"] + [DET_LABEL[d] for d in DET[1:]] + ["Flags"]
        write_tabular("celeba_audit", pd.DataFrame(rows), "l" + "c" * (len(hdr) - 1), escape_cells=False, header=hdr)

    # sensitivity tables
    sens = df[df.detector == "probe"]
    for attr, variants, name in (("race", ["primary", "min50_per_group", "min50_per_stratum"], "race_sensitivity"),
                                 ("age", ["primary", "4bin"], "age_sensitivity")):
        rows, mids = [], []
        for ds in ["chexpert", "mimic"]:
            for bb in sens[sens.dataset == ds].backbone.unique():
                r = {"Dataset": {"chexpert": "CheXpert", "mimic": "MIMIC-CXR"}[ds], "Backbone": bb_label(ds, bb)}
                for v in variants:
                    s = sens[(sens.dataset == ds) & (sens.backbone == bb) & (sens.attribute == attr) & (sens.variant == v)]
                    r[v] = "n/a" if s.empty else f"{s.statistic.iloc[0]:.2f} ($p$ {pfmt(s.p_value.iloc[0])})"
                    if not s.empty:
                        r[v + "_n"] = int(s.n_images.iloc[0])
                rows.append(r)
            mids.append(len(rows))
        t = pd.DataFrame(rows)
        if attr == "race":
            hdr = ["Dataset", "Encoder", "White vs non-White", "Groups with 50+ (W/B/A)", "Groups with 50+ per $Y$"]
            t = t[["Dataset", "Backbone"] + variants]
            notes = df[(df.attribute == "race") & (df.variant != "primary")][["dataset", "variant", "note", "n_images"]].drop_duplicates()
            notes.to_csv(TAB / "race_sensitivity_groups.csv", index=False)
        else:
            hdr = ["Dataset", "Encoder", "Age 60+ (binary)", "Age, 4 bins"]
            t = t[["Dataset", "Backbone"] + variants]
        write_tabular(name, t, "ll" + "c" * (len(hdr) - 2), escape_cells=False, header=hdr, midrules=mids[:-1])

    # continuous score heatmap (z against the null), primary cells
    _apply_pub_style()
    fig, axes = plt.subplots(1, 2, figsize=(13, 8.5), gridspec_kw={"width_ratios": [1, 1], "wspace": 0.6})
    for ax, ds in zip(axes, ["chexpert", "mimic"], strict=True):
        sub = prim[(prim.dataset == ds) & prim.detector.isin(DET)]
        bb_order = {b: i for i, b in enumerate(CHEXPERT_BACKBONES if ds == "chexpert" else MIMIC_BACKBONES)}
        attr_order = {"sex": 0, "age": 1, "race": 2}
        sub = sub.sort_values(["backbone", "attribute"], key=lambda c: c.map(bb_order if c.name == "backbone" else attr_order),
                              kind="stable")  # partial reruns append rows; keep backbone then attribute order
        labels, Z, F, S = [], [], [], []
        for (bb, attr), g in sub.groupby(["backbone", "attribute"], sort=False):
            g = g.set_index("detector").reindex(DET)
            lab = bb_label(ds, bb).replace(" (rows misaligned)", " (misaligned)")
            labels.append(f"{lab} | {attr}")
            Z.append(g.z.to_numpy(float))
            F.append(g.flag.to_numpy(bool))
            S.append(g.statistic.to_numpy(float))
        Z = np.clip(np.array(Z), -3, 30)
        im = ax.imshow(np.sign(Z) * np.log1p(np.abs(Z)), cmap="cividis", aspect="auto", vmin=0, vmax=np.log1p(30))
        for i in range(Z.shape[0]):
            for j in range(Z.shape[1]):
                ax.text(j, i, f"{S[i][j]:.2f}" + ("*" if F[i][j] else ""), ha="center", va="center",
                        fontsize=7.5, color="white" if Z[i, j] < 8 else "black")
        ax.set_xticks(range(len(DET)), [DET_LABEL[d] for d in DET], rotation=30, ha="right")
        ax.set_yticks(range(len(labels)), labels, fontsize=8)
        ax.set_title({"chexpert": "CheXpert (Y = Pneumothorax)", "mimic": "MIMIC-CXR (Y = mortality)"}[ds])
        ax.grid(False)
    cb = fig.colorbar(im, ax=axes, shrink=0.6)
    ticks = [0, 1, 3, 10, 30]
    cb.set_ticks(np.log1p(ticks), labels=[str(t) for t in ticks])
    cb.set_label("z = (score - null mean) / null SD")
    fig.text(0.5, 0.01, "Cell text: continuous score; * probe p < 0.05, secondary detectors BH q < 0.05 across the five",
             ha="center", fontsize=9)
    fig.savefig(FIG / "clinical_scores_heatmap.pdf")
    plt.close(fig)
    return df


# ---------------------------------------------------------------------------
# Cohorts
# ---------------------------------------------------------------------------


def cohorts():
    from common import load_chexpert, load_mimic

    _, mc = load_chexpert("rad_dino")
    _, mm = load_mimic("rad_dino")
    rows = []
    for name, m, task in (("CheXpert", mc, "Pneumothorax"), ("MIMIC-CXR", mm, "In-hospital mortality")):
        n = len(m)
        pats = m["patient_id"].nunique() if "patient_id" in m.columns else m["image_path"].str.extract(r"/p(\d+)/")[0].nunique()
        race = m["race"].fillna("MISSING").astype(str).value_counts()
        rows.append({"Cohort": name, "Images": n, "Patients": pats,
                     "Male": f"{(m.sex == 'Male').sum()} ({100 * (m.sex == 'Male').mean():.0f}\\%)",
                     "Age 60+": f"{(m.age >= 60).sum()} ({100 * (m.age >= 60).mean():.0f}\\%)",
                     "White": race.get("WHITE", 0), "Black": race.get("BLACK", 0), "Asian": race.get("ASIAN", 0),
                     "NHPI": race.get("NHPI", 0), "Other": race.get("OTHER", 0),
                     "Missing": race.get("MISSING", 0),
                     "Task": task, "Y=1": f"{int(m.task_label.sum())} ({100 * m.task_label.mean():.1f}\\%)"})
    put("chexpert_pneumothorax_prevalence", 100 * mc.task_label.mean(), "{:.1f}")
    put("mimic_mortality_prevalence", 100 * mm.task_label.mean(), "{:.1f}")
    put("chexpert_n_patients", int(mc.patient_id.nunique()), "{:d}")
    put("chexpert_n_lateral", int((mc.view == "Lateral").sum()), "{:d}")
    hdr = ["Cohort", "Images", "Patients", "Male", "Age 60+", "White", "Black", "Asian", "NHPI", "Other",
           "Missing", "Task ($Y$)", "$Y{=}1$"]
    write_tabular("cohort_demographics", pd.DataFrame(rows), "l" + "r" * 10 + "lr", escape_cells=False, header=hdr)
    put("chexpert_n_race_missing", int(mc.race.isna().sum()), "{:d}")
    put("chexpert_n_race_nhpi", int((mc.race == "NHPI").sum()), "{:d}")
    put("mimic_n_race_missing", int(mm.race.isna().sum()), "{:d}")


# ---------------------------------------------------------------------------
# Head experiment
# ---------------------------------------------------------------------------


def head():
    from scipy.stats import spearmanr

    h = load("head_experiment/head_reliance.csv")
    if h is None:
        return
    put("attr_removal_method",
        "unit direction u = difference of A-group means after centring each Y stratum, estimated on the "
        "training pool in raw embedding space; test embeddings x are replaced by x - ((x - m) . u) u with m "
        "the training-pool mean, and the confounded head is reapplied")
    hf = h[h.feasible.astype(bool)]
    cell = hf.groupby(["backbone", "attribute", "task"]).agg(
        probe_auroc=("probe_auroc", "first"), tpr_gap=("tpr_gap", "mean"), fpr_gap=("fpr_gap", "mean"),
        auroc_drop=("auroc_drop", "mean"), output_shift=("output_shift", "mean"),
        auroc_aligned=("auroc_aligned", "mean"), auroc_misaligned=("auroc_misaligned", "mean"),
        ece_A1=("ece_A1", "mean"), ece_A0=("ece_A0", "mean"), n_train=("n_train", "mean"),
        n_test=("n_test", "mean"), n_train_min=("n_train", "min"), n_train_max=("n_train", "max"),
        n_test_min=("n_test", "min"), n_test_max=("n_test", "max"), n_seeds=("seed", "nunique")).reset_index()
    cell.to_csv(TAB / "head_reliance_cells.csv", index=False)
    put("n_head_cells", int(len(cell)), "{:d}")
    put("head_n_seeds", int(h.seed.nunique()), "{:d}")
    put("n_head_cells_infeasible", int(h.groupby(["backbone", "attribute", "task"]).feasible.apply(lambda s: not s.astype(bool).any()).sum()), "{:d}")
    rho_vals = {}
    rng = np.random.default_rng(0)
    for col, key in (("tpr_gap", "tprgap"), ("auroc_drop", "aurocdrop"), ("output_shift", "outputshift"),
                     ("fpr_gap", "fprgap")):
        r, p = spearmanr(cell.probe_auroc, cell[col])
        rho_vals[col] = (r, p)
        put(f"spearman_probe_{key}_chexpert", r)
        put(f"spearman_probe_{key}_p_chexpert", p, "{:.3f}")
        if key == "tprgap":
            boot = []
            for _ in range(2000):
                i = rng.integers(0, len(cell), len(cell))
                boot.append(spearmanr(cell.probe_auroc.iloc[i], cell[col].iloc[i])[0])
            lo, hi = np.nanpercentile(boot, [2.5, 97.5])
            put("spearman_probe_tprgap_ci", [lo, hi], text=rng_text(lo, hi))
    for attr in ["sex", "age"]:
        s = cell[cell.attribute == attr]
        r, p = spearmanr(s.probe_auroc, s.tpr_gap)
        put(f"spearman_probe_tprgap_within_{attr}", r)
        put(f"spearman_probe_tprgap_within_{attr}_p", p, "{:.3f}")
        # whole backbones resampled, as for the pooled interval
        rng_w = np.random.default_rng(2)
        bw = []
        for _ in range(2000):
            pick = rng_w.choice(s.backbone.unique(), size=s.backbone.nunique(), replace=True)
            bs = pd.concat([s[s.backbone == b] for b in pick], ignore_index=True)
            bw.append(spearmanr(bs.probe_auroc, bs.tpr_gap)[0])
        lo, hi = np.nanpercentile(bw, [2.5, 97.5])
        put(f"spearman_probe_tprgap_within_{attr}_cluster_ci", [lo, hi], text=rng_text(lo, hi))
        # spread of the cell means against seed noise within a cell (range restriction check)
        g = hf[hf.attribute == attr].groupby(["backbone", "task"]).tpr_gap
        put(f"tprgap_cell_sd_{attr}", g.mean().std())
        put(f"tprgap_seed_sd_{attr}", g.std().mean())

    # the cells are not independent: the 7 backbones embed the same images, and the
    # attributes and tasks reuse the same patients
    def partial_spearman(frame):
        from scipy.stats import rankdata

        D = pd.get_dummies(frame.attribute + "|" + frame.task, drop_first=False).to_numpy(float)
        res = []
        for col in ("probe_auroc", "tpr_gap"):
            r_ = rankdata(frame[col].to_numpy())
            beta, *_ = np.linalg.lstsq(D, r_, rcond=None)
            res.append(r_ - D @ beta)
        return float(np.corrcoef(res[0], res[1])[0, 1])

    put("spearman_probe_tprgap_partial", partial_spearman(cell))
    bbs = cell.backbone.unique()
    rng_c = np.random.default_rng(1)
    boot_c = []
    for _ in range(2000):
        pick = rng_c.choice(bbs, size=len(bbs), replace=True)
        bc = pd.concat([cell[cell.backbone == b] for b in pick], ignore_index=True)
        boot_c.append(spearmanr(bc.probe_auroc, bc.tpr_gap)[0])
    lo, hi = np.nanpercentile(boot_c, [2.5, 97.5])
    put("spearman_probe_tprgap_cluster_ci", [lo, hi], text=rng_text(lo, hi))
    put("head_cells_share_image_sets",
        f"the {len(cell)} cells are not independent: the {len(bbs)} encoders embed the same "
        f"{int(h.n_labelled.max()) if 'n_labelled' in h.columns else 'CheXpert'}-or-fewer labelled images, "
        "and the two attributes and three tasks reuse the same patients; the partial Spearman (given "
        "attribute and task) and the bootstrap over encoders (whole encoders resampled) address this", None)
    put("head_n_train_min", int(hf.n_train.min()), "{:d}")
    put("head_n_train_max", int(hf.n_train.max()), "{:d}")
    put("head_n_test_min", int(hf.n_test.min()), "{:d}")
    put("head_n_test_max", int(hf.n_test.max()), "{:d}")
    put("head_auroc_aligned", hf.auroc_aligned.mean())
    put("head_auroc_misaligned", hf.auroc_misaligned.mean())
    put("head_ece_A1", hf.ece_A1.mean(), "{:.3f}")
    put("head_ece_A0", hf.ece_A0.mean(), "{:.3f}")
    put("head_unconf_auroc_aligned", hf.unconf_auroc_aligned.mean())
    put("head_unconf_auroc_misaligned", hf.unconf_auroc_misaligned.mean())
    put("head_tpr_gap_mean", hf.tpr_gap.mean())
    put("head_auroc_drop_mean", hf.auroc_drop.mean())

    rows = []
    for (attr, task), g in hf.groupby(["attribute", "task"]):
        rows.append({"A": {"sex": "Sex", "age": "Age 60+"}[attr], "Y": task.replace("_", " ").title(),
                     "n train": f"{int(g.n_train.min())}-{int(g.n_train.max())}",
                     "n test": f"{int(g.n_test.min())}-{int(g.n_test.max())}",
                     "Probe": f2(g.probe_auroc.mean()), "TPR gap": f2(g.tpr_gap.mean()),
                     "FPR gap": f2(g.fpr_gap.mean()), "AUROC drop": f2(g.auroc_drop.mean()),
                     "Output shift": f2(g.output_shift.mean()),
                     "AUROC aligned": f2(g.auroc_aligned.mean()), "AUROC misaligned": f2(g.auroc_misaligned.mean()),
                     "ECE A=1": f2(g.ece_A1.mean(), 3), "ECE A=0": f2(g.ece_A0.mean(), 3)})
    hdr = ["$A$", "$Y$", "$n$ train (min-max)", "$n$ test (min-max)", "Probe", "TPR gap", "FPR gap", "AUROC drop",
           "Output shift", "AUROC aligned", "AUROC misaligned", "ECE $A{=}1$", "ECE $A{=}0$"]
    write_tabular("head_reliance", pd.DataFrame(rows), "ll" + "c" * (len(hdr) - 2), escape_cells=False, header=hdr)

    _apply_pub_style()
    fig, axes = plt.subplots(1, 3, figsize=(13, 3.8), gridspec_kw={"wspace": 0.45})
    for ax, (col, lab) in zip(axes, [("tpr_gap", "TPR gap (A=1 minus A=0)"), ("auroc_drop", "AUROC drop vs unconfounded head"),
                                     ("output_shift", "Mean |output shift| after removing A")], strict=True):
        for attr, c, mk in (("sex", OI["blue"], "o"), ("age", OI["orange"], "s")):
            s = cell[cell.attribute == attr]
            ax.scatter(s.probe_auroc, s[col], color=c, marker=mk, s=28, label={"sex": "Sex", "age": "Age 60+"}[attr],
                       edgecolor="black", linewidth=0.4)
        r, p = rho_vals[col]
        ptxt = "p < 0.001" if p < 0.001 else f"p = {p:.3f}"
        ax.set_title(f"Spearman $\\rho$ = {r:.2f} ({ptxt})", fontsize=10)
        ax.set_xlabel("Conditional probe AUROC (audit)")
        ax.set_ylabel(lab)
    axes[0].legend(frameon=False)
    fig.savefig(FIG / "audit_vs_reliance.pdf")
    plt.close(fig)

    m = load("head_experiment/mitigation.csv")
    if m is not None and not m.empty:
        put("mitig1_name", "adversarial debiasing")
        put("mitig2_name", "contrastive debiasing")
        put("mitig3_name", "shortcut masking")
        put("mitig4_name", "last-layer retraining")
        put("mitig_n_seeds", int(m.seed.nunique()), "{:d}")
        rows = []
        label = {"adversarial": "Adversarial debiasing", "adversarial_control": "Same encoder, adversary weight 0",
                 "contrastive": "Contrastive debiasing", "contrastive_control": "Same encoder, contrastive weight 0",
                 "masking": "Shortcut masking", "masking_control": "Same number of random dimensions",
                 "dfr": "Last-layer retraining (DFR)"}
        mids = []
        for i, meth in ((1, "adversarial"), (2, "contrastive"), (3, "masking"), (4, "dfr")):
            g = m[m.method == meth]
            c = m[m.method == f"{meth}_control"]
            # all seeds (section 9); deltas against the raw embeddings
            put(f"mitig{i}_delta_probe_auroc", (g.probe_auroc_after - g.probe_auroc_before).mean())
            put(f"mitig{i}_delta_task_auroc", (g.task_auroc_after - g.task_auroc_before).mean())
            put(f"mitig{i}_delta_tpr_gap", (g.tpr_gap_after - g.tpr_gap_before).mean())
            put(f"mitig{i}_n_probe_sig_after", int((g.probe_p_after < 0.05).sum()), "{:d}")
            put(f"mitig{i}_n_probe_sig_before", int((g.probe_p_before < 0.05).sum()), "{:d}")
            put(f"mitig{i}_n_audits", int(len(g)), "{:d}")
            if not c.empty:
                put(f"mitig{i}_control_delta_probe_auroc", (c.probe_auroc_after - c.probe_auroc_before).mean())
                put(f"mitig{i}_control_delta_task_auroc", (c.task_auroc_after - c.task_auroc_before).mean())
                put(f"mitig{i}_control_delta_tpr_gap", (c.tpr_gap_after - c.tpr_gap_before).mean())
                put(f"mitig{i}_control_n_probe_sig_after", int((c.probe_p_after < 0.05).sum()), "{:d}")
                k = ["backbone", "attribute", "task", "seed"]
                j = g.set_index(k).join(c.set_index(k), rsuffix="_c")
                put(f"mitig{i}_vs_control_probe_auroc", (j.probe_auroc_after - j.probe_auroc_after_c).mean())
                put(f"mitig{i}_vs_control_task_auroc", (j.task_auroc_after - j.task_auroc_after_c).mean())
                put(f"mitig{i}_vs_control_tpr_gap", (j.tpr_gap_after - j.tpr_gap_after_c).mean())
            if meth == "dfr":
                # the head is refit, the embeddings are not changed
                put("mitig4_frac_tpr_gap_smaller", (g.tpr_gap_after.abs() < g.tpr_gap_before.abs()).mean())
                # seeds of a cell share images, so test on cell means
                from scipy.stats import wilcoxon
                dc = (g.tpr_gap_after - g.tpr_gap_before).groupby([g.backbone, g.attribute, g.task]).mean()
                put("mitig4_n_cells", int(len(dc)), "{:d}")
                put("mitig4_tpr_gap_before", g.tpr_gap_before.mean())
                put("mitig4_tpr_gap_after", g.tpr_gap_after.mean())
                put("mitig4_task_auroc_before", g.task_auroc_before.mean())
                put("mitig4_task_auroc_after", g.task_auroc_after.mean())
                if "n_head_fit" in g:
                    put("mitig4_n_fit_min", int(g.n_head_fit.min()), "{:d}")
                    put("mitig4_n_fit_max", int(g.n_head_fit.max()), "{:d}")
                put("mitig4_n_cells_gap_lower", int((dc < 0).sum()), "{:d}")
                put("mitig4_cells_wilcoxon_p", wilcoxon(dc).pvalue, "{:.3f}")
            for mm_ in (meth, f"{meth}_control"):
                gm = m[m.method == mm_]
                if gm.empty:
                    continue
                for attr in ["sex", "age"]:
                    ga = gm[gm.attribute == attr]
                    if ga.empty:
                        continue
                    rows.append({"Mitigation": label[mm_] if attr == "sex" else "",
                                 "A": {"sex": "Sex", "age": "Age 60+"}[attr],
                                 "Probe before": f2(ga.probe_auroc_before.mean()), "Probe after": f2(ga.probe_auroc_after.mean()),
                                 "Sig. before": f"{int((ga.probe_p_before < 0.05).sum())}/{len(ga)}",
                                 "Sig. after": f"{int((ga.probe_p_after < 0.05).sum())}/{len(ga)}",
                                 "Task before": f2(ga.task_auroc_before.mean()), "Task after": f2(ga.task_auroc_after.mean()),
                                 "TPR gap before": f2(ga.tpr_gap_before.mean()), "TPR gap after": f2(ga.tpr_gap_after.mean())})
                mids.append(len(rows))
        hdr = ["Method", "$A$", "Probe before", "Probe after", "Sig. before", "Sig. after",
               "Task AUROC before", "Task AUROC after", "TPR gap before", "TPR gap after"]
        write_tabular("mitigation_before_after", pd.DataFrame(rows), "ll" + "c" * 8, escape_cells=False,
                      header=hdr, midrules=mids[:-1])
        # per backbone: method minus its control on the same cell and seed (DFR: after minus before),
        # mean over attributes, tasks and seeds
        k = ["backbone", "attribute", "task", "seed"]
        meths = ["adversarial", "contrastive", "masking", "dfr"]
        rows = []
        for bb in CHEXPERT_BACKBONES + ["All"]:
            r = {"Backbone": bb_label("chexpert", bb) if bb != "All" else "All encoders"}
            for meth in meths:
                g = m[m.method == meth] if bb == "All" else m[(m.method == meth) & (m.backbone == bb)]
                if meth == "dfr":
                    dg, da = g.tpr_gap_after - g.tpr_gap_before, g.task_auroc_after - g.task_auroc_before
                else:
                    c = m[m.method == f"{meth}_control"]
                    j = g.set_index(k).join(c.set_index(k), rsuffix="_c", how="inner")
                    dg, da = j.tpr_gap_after - j.tpr_gap_after_c, j.task_auroc_after - j.task_auroc_after_c
                r[f"{meth}_gap"], r[f"{meth}_auc"] = f2(dg.mean()), f2(da.mean())
            rows.append(r)
        hdr = ["Encoder"] + ["$\\Delta$ TPR gap", "$\\Delta$ AUROC"] * len(meths)
        pre = [" & " + " & ".join(f"\\multicolumn{{2}}{{c}}{{{lab}}}" for lab in
                                  ("Adversarial", "Contrastive", "Masking", "DFR")) + " \\\\",
               " ".join(f"\\cmidrule(lr){{{2 + 2 * i}-{3 + 2 * i}}}" for i in range(len(meths)))]
        write_tabular("mitigation_by_backbone", pd.DataFrame(rows), "l" + "c" * (len(hdr) - 1),
                      escape_cells=False, header=hdr, midrules=[len(rows) - 1], prehead=pre)


# ---------------------------------------------------------------------------
# Old pipeline comparison, negative control, analyses
# ---------------------------------------------------------------------------

OLD_SHORT = {"hbac": "HBAC", "probe": "Probe", "statistical": "Stat.", "geometric": "Geom.", "frequency": "Freq.",
             "bias_direction_pca": "BiasDir", "sis": "SIS", "demographic_parity": "DP", "equalized_odds": "EO",
             "intersectional": "Inter.", "groupdro": "GDRO", "gce": "GCE", "ssa": "SSA"}


def old_pipeline():
    o = load("old_pipeline_fixed_flag/old_pipeline_methods.csv")
    if o is None:
        return
    o["evaluated"] = o["evaluated"].astype(bool)
    rows = []
    for ds in ["chexpert", "mimic", "celeba"]:
        g = o[o.dataset == ds]
        if g.empty:
            continue
        ncell = g.groupby(["backbone", "attribute"]).ngroups
        for meth in OLD_SHORT:
            s = g[g.method == meth]
            ev = s[s.evaluated]
            rows.append({"Dataset": {"chexpert": "CheXpert", "mimic": "MIMIC-CXR", "celeba": "CelebA"}[ds] if meth == "hbac" else "",
                         "Method": OLD_SHORT[meth],
                         "Submitted reading": f"{int(s.flag_legacy.sum())}/{ncell}",
                         "Fixed reading": f"{int(ev.flag_fixed.sum())}/{len(ev)}" if len(ev) else "n/a",
                         "Evaluated": f"{len(ev)}/{ncell}"})
        put(f"old_pipeline_mean_agreement_legacy_{ds}",
            g.groupby(["backbone", "attribute"]).flag_legacy.sum().mean(), "{:.1f}")
        put(f"old_pipeline_mean_agreement_fixed_{ds}",
            g[g.evaluated].groupby(["backbone", "attribute"]).flag_fixed.sum().mean(), "{:.1f}")
    write_tabular("old_pipeline_comparison", pd.DataFrame(rows), "llccc",
                  midrules=[13, 26] if len(rows) > 26 else [13])
    # Age and race identical vectors check (R2-27)
    for ds in ["chexpert", "mimic"]:
        g = o[(o.dataset == ds) & o.evaluated]
        same = 0
        tot = 0
        for bb, gb in g.groupby("backbone"):
            p = gb.pivot_table(index="method", columns="attribute", values="flag_fixed")
            if {"age_bin", "race"} <= set(p.columns):
                tot += 1
                same += int((p["age_bin"] == p["race"]).all())
        put(f"old_pipeline_age_race_identical_backbones_{ds}", f"{same} of {tot}", None)


def negative_control():
    n = load("negative_control/negative_control.csv")
    if n is None:
        return
    n["rej"] = n.p_value < 0.05
    s = n.groupby(["detector", "mode"]).rej.mean().unstack()
    rows = []
    for d in DET:
        rows.append({"Detector": DET_LABEL[d], "Patient-level (used)": f2(s.loc[d, "patient_permutation_grouped_cv"]),
                     "Image-level": f2(s.loc[d, "image_permutation_plain_cv"])})
        put(f"negctl_fpr_patient_{DET_KEY[d]}", s.loc[d, "patient_permutation_grouped_cv"])
        put(f"negctl_fpr_image_{DET_KEY[d]}", s.loc[d, "image_permutation_plain_cv"])
    put("negctl_n_draws", int(n.draw.nunique()), "{:d}")
    write_tabular("negative_control", pd.DataFrame(rows), "lcc",
                  header=["Detector", "Patient-level permutation, grouped CV", "Image-level permutation, plain CV"])


def analyses():
    c = load("analyses/cluster_metadata.csv")
    if c is not None:
        rows = []
        for (ds, var), g in c.groupby(["dataset", "variable"], sort=False):
            rows.append({"Dataset": {"chexpert": "CheXpert", "mimic": "MIMIC-CXR"}[ds], "Variable": var,
                         "NMI mean": f2(g.nmi.mean(), 3), "NMI range": rng_text(g.nmi.min(), g.nmi.max(), "{:.3f}"),
                         "Cramer's V mean": f2(g.cramers_v.mean()),
                         "chi2 p < 0.05": f"{int((g.chi2_p < 0.05).sum())}/{len(g)}"})
        t = pd.DataFrame(rows)
        mids = [int((t.Dataset == "CheXpert").sum())]
        write_tabular("cluster_metadata", t, "llcccc",
                      header=["Dataset", "Variable", "NMI (mean)", "NMI (range over encoders)",
                              "Cram\\'er's V (mean)", "$\\chi^2$ $p<0.05$"], midrules=mids)
        sd = c[(c.dataset == "chexpert") & (c.variable == "Support Devices")]
        put("cluster_support_devices_nmi_chexpert", sd.nmi.mean(), "{:.3f}")
        put("cluster_support_devices_nmi_range_chexpert", [sd.nmi.min(), sd.nmi.max()],
            text=rng_text(sd.nmi.min(), sd.nmi.max(), "{:.3f}"))
        vw = c[(c.dataset == "chexpert") & (c.variable == "view (frontal/lateral)")]
        put("cluster_view_nmi_chexpert", vw.nmi.mean(), "{:.3f}")
        top = c[c.dataset == "chexpert"].groupby("variable").nmi.mean().sort_values(ascending=False)
        put("cluster_top_variable_chexpert", top.index[0])
        topm = c[c.dataset == "mimic"].groupby("variable").nmi.mean().sort_values(ascending=False)
        put("cluster_top_variable_mimic", topm.index[0])
    g = load("analyses/group_size_stability.csv")
    if g is not None:
        rows = []
        for (ds, m), s in g.groupby(["dataset", "target_m"]):
            rows.append({"Dataset": {"chexpert": "CheXpert", "mimic": "MIMIC-CXR"}[ds], "m": int(m),
                         "images": f"{s.minority_images.mean():.0f}", "Y=1": f"{s.minority_Y1.mean():.1f}",
                         "AUROC": f"{s.probe_auroc.mean():.2f} $\\pm$ {s.probe_auroc.std():.2f}",
                         "Rejected": f"{int(s.reject.sum())}/{len(s)}"})
        write_tabular("group_size_stability", pd.DataFrame(rows), "lccccc", escape_cells=False,
                      header=["Dataset", "Target $m$", "Non-White images", "Non-White $Y{=}1$",
                              "Probe AUROC (mean $\\pm$ SD)", "$p<0.05$"], midrules=[5])
        for ds in ["chexpert", "mimic"]:
            s = g[(g.dataset == ds)]
            put(f"groupsize_reject_rate_m25_{ds}", s[s.target_m == 25].reject.mean())
            put(f"groupsize_reject_rate_m400_{ds}", s[s.target_m == 400].reject.mean())
            put(f"groupsize_auroc_sd_m25_{ds}", s[s.target_m == 25].probe_auroc.std())
    lda = load("analyses/lda_heldout_mimic_raddino_sex.csv")
    if lda is not None:
        from sklearn.metrics import roc_auc_score

        _apply_pub_style()
        fig, ax = plt.subplots(figsize=(5.5, 3.4))
        bins = np.linspace(lda.projection.quantile(0.005), lda.projection.quantile(0.995), 50)
        for sex, c in (("Female", OI["orange"]), ("Male", OI["blue"])):
            ax.hist(lda.projection[lda.sex == sex], bins=bins, alpha=0.6, color=c, label=sex, density=True)
        auc = roc_auc_score(lda.sex == "Male", lda.projection)
        ax.set_xlabel("Held-out LDA projection (fit on training folds)")
        ax.set_ylabel("Density")
        ax.set_title(f"MIMIC-CXR RAD-DINO, sex: held-out AUROC {auc:.3f}", fontsize=10)
        ax.legend(frameon=False)
        fig.savefig(FIG / "lda_sex_mimic_raddino_heldout.pdf")
        plt.close(fig)
        put("lda_heldout_auroc_mimic_raddino_sex", auc, "{:.3f}")
    um = load("analyses/umap_mimic_raddino_race.csv")
    if um is not None:
        _apply_pub_style()
        fig, ax = plt.subplots(figsize=(5.0, 4.2))
        for race, c in (("White", OI["sky"]), ("Black", OI["vermillion"]), ("Asian", OI["green"]),
                        ("Other", OI["purple"])):
            s = um[um.race == race]
            ax.scatter(s.x, s.y, s=6, alpha=0.7, color=c, label=f"{race} ({len(s)})", linewidths=0)
        ax.set_xticks([]), ax.set_yticks([])
        ax.set_title("MIMIC-CXR RAD-DINO, coloured by race", fontsize=10)
        ax.legend(frameon=False, fontsize=8, markerscale=2, loc="upper center", bbox_to_anchor=(0.5, -0.02), ncol=4)
        for sp in ax.spines.values():
            sp.set_visible(False)
        fig.savefig(FIG / "umap_mimic_raddino_race.pdf")
        plt.close(fig)
        put("umap_mimic_n_shown", len(um), "{:,.0f}")
    uc = load("analyses/umap_celeba_resnet50_male.csv")
    if uc is not None:
        _apply_pub_style()
        fig, ax = plt.subplots(figsize=(5.0, 4.2))
        for male, lab, c in ((False, "Not Male", OI["orange"]), (True, "Male", OI["blue"])):
            s = uc[uc.male == male]
            ax.scatter(s.x, s.y, s=4, alpha=0.6, color=c, label=f"{lab} ({len(s)})", linewidths=0)
        ax.set_xticks([]), ax.set_yticks([])
        ax.set_title("CelebA ResNet-50, coloured by Male", fontsize=10)
        ax.legend(frameon=False, fontsize=8, markerscale=2, loc="upper center", bbox_to_anchor=(0.5, -0.02), ncol=4)
        for sp in ax.spines.values():
            sp.set_visible(False)
        fig.savefig(FIG / "umap_celeba_resnet50_male.pdf")
        plt.close(fig)


def marker():
    mk = load("planted_marker/planted_marker.csv")
    if mk is None:
        return
    put("marker_rate", 0.5, "{:.1f}")
    put("marker_rate_head_train", 0.9, "{:.1f}")
    put("probe_auroc_marker", mk.probe_stat.mean(), "{:.3f}")  # 3 places, as in its table
    put("probe_p_marker", mk.probe_p.max(), "{:.3f}")
    put("head_tprgap_marker", mk.tpr_gap.mean())
    put("head_fprgap_marker", mk.fpr_gap.mean())
    put("marker_n_seeds", int(len(mk)), "{:d}")
    # flags under the decision rule, per seed: probe at p < 0.05, secondaries at BH q < 0.05 across the five
    flags = {"probe": (mk.probe_p < 0.05).to_numpy()}
    q5 = benjamini_hochberg(mk[[f"{d}_p" for d in DET[1:]]].to_numpy(float))
    for j, d in enumerate(DET[1:]):
        flags[d] = q5[:, j] < 0.05
    rows = []
    for d in DET:
        rows.append({"Detector": DET_LABEL[d], "Score": f"{mk[f'{d}_stat'].mean():.3f}",
                     "max p": f"{mk[f'{d}_p'].max():.3f}",
                     "Flagged": f"{int(flags[d].sum())}/{len(mk)}"})
    rows.append({"Detector": "Head TPR gap", "Score": f"{mk.tpr_gap.mean():.2f}", "max p": "n/a", "Flagged": "n/a"})
    rows.append({"Detector": "Head FPR gap", "Score": f"{mk.fpr_gap.mean():.2f}", "max p": "n/a", "Flagged": "n/a"})
    write_tabular("planted_marker", pd.DataFrame(rows), "lccc",
                  header=["", "Mean score", "Max $p$", "Flagged"], midrules=[len(DET)])


def _short_error(err: str) -> str:
    """One line for a table cell: gated repositories named as such, else the first line."""
    if "gated repo" in err:
        return "gated repository (no access)"
    line = next((x.strip() for x in str(err).splitlines() if x.strip()), "")
    return line[:60]


def sensitivity():
    """CheXpert probe under four changes: the arrays of the first submission, frontal
    images only, C = 0.1 and C = 10; and the head experiment with an MLP head."""
    main = load("clinical_audit/audit_rows.csv")
    if main is None:
        return
    # the probe is judged at p < 0.05 in every column (same rule as the main audit)
    sel = lambda d: d[(d.dataset == "chexpert") & (d.detector == "probe") & (d.variant == "primary")]
    main = sel(main).set_index(["backbone", "attribute"])
    cell = lambda a, sig: f"{a:.2f}{'*' if str(sig).lower() == 'true' else ''}"
    old = load("clinical_audit_stored_arrays/audit_rows.csv")
    old = sel(old).set_index(["backbone", "attribute"]) if old is not None else None
    fr = load("frontal_only/probe_rows.csv")
    fr = fr.set_index(["backbone", "attribute"]) if fr is not None else None
    lb = load("labelled_only/probe_rows.csv")
    lb = lb.set_index(["backbone", "attribute"]) if lb is not None else None
    cs = load("probe_C_sensitivity/probe_rows.csv")
    rows, diffs = [], {"stored": [], "frontal": [], "labelled": [], "C": []}
    for (bb, attr), r in main.iterrows():
        row = {"Encoder": BACKBONE_LABELS.get(bb, bb), "$A$": attr.capitalize(), "Main": cell(r.statistic, r.p_value < 0.05)}
        if old is not None and bb in CHEXPERT_PUBLIC and (bb, attr) in old.index:
            o = old.loc[(bb, attr)]
            row["Stored array"] = cell(o.statistic, o.p_value < 0.05)
            diffs["stored"].append(abs(o.statistic - r.statistic))
        else:
            row["Stored array"] = "same"
        if fr is not None and (bb, attr) in fr.index:
            f = fr.loc[(bb, attr)]
            row["Frontal only"] = cell(f.auroc, f.p < 0.05)
            diffs["frontal"].append(abs(f.auroc - r.statistic))
        if lb is not None and (bb, attr) in lb.index:
            f = lb.loc[(bb, attr)]
            row["Labelled only"] = cell(f.auroc, f.p < 0.05)
            diffs["labelled"].append(abs(f.auroc - r.statistic))
        elif lb is not None:
            row["Labelled only"] = "n/a"
        for C in (0.1, 10.0):
            c = cs[(cs.backbone == bb) & (cs.attribute == attr) & (cs.C == C)] if cs is not None else []
            if len(c):
                c = c.iloc[0]
                row[f"$C={C:g}$"] = cell(c.auroc, c.p < 0.05)
                diffs["C"].append(abs(c.auroc - r.statistic))
            else:
                row[f"$C={C:g}$"] = "n/a"
        rows.append(row)
    enc = list(dict.fromkeys(r["Encoder"] for r in rows))
    rows.sort(key=lambda r: (enc.index(r["Encoder"]), ["Sex", "Age", "Race"].index(r["$A$"])))
    cols = ["Encoder", "$A$", "Main", "Stored array", "Frontal only"] + (["Labelled only"] if lb is not None else []) + ["$C=0.1$", "$C=10$"]
    write_tabular("sensitivity_probe", pd.DataFrame(rows)[cols], "ll" + "c" * (len(cols) - 2), escape_cells=False, header=cols)
    for k, v in diffs.items():
        if v:
            put(f"sens_max_absdiff_{k}", max(v))
    if fr is not None:
        put("frontal_n_images", int(fr.n_images.max()), "{:d}")
        put("frontal_n_sig", int((fr.p < 0.05).sum()), "{:d}")
        put("frontal_n_cells", int(len(fr)), "{:d}")
    if lb is not None:
        put("labelled_n_images", int(lb.n_images.max()), "{:d}")
        put("labelled_n_sig", int((lb.p < 0.05).sum()), "{:d}")
        put("labelled_n_cells", int(len(lb)), "{:d}")
        put("labelled_n_raw_sig", int((lb.p < 0.05).sum()), "{:d}")
    if cs is not None and "main_p" in cs:
        x = cs[cs.C != 1.0]
        put("probe_C_n_cells", int(len(x)), "{:d}")
        # decision at p < 0.05 with this C against the main audit's decision at C = 1
        put("probe_C_n_decision_changes", int(((x.p < 0.05) != (x.main_p < 0.05)).sum()), "{:d}")
    mlp = load("head_mlp/summary_mlp.csv")
    if mlp is not None:
        g = mlp.set_index("stat")
        r = g.loc["rho probe vs MLP TPR gap, all cells"]
        put("mlp_spearman", r.value)
        put("mlp_spearman_ci", [r.ci_backbones_lo, r.ci_backbones_hi], text=rng_text(r.ci_backbones_lo, r.ci_backbones_hi))
        put("mlp_spearman_within_sex", g.loc["rho probe vs MLP TPR gap, within sex"].value)
        put("mlp_spearman_within_age", g.loc["rho probe vs MLP TPR gap, within age"].value)
        put("mlp_vs_logistic_gap_spearman", g.loc["rho MLP TPR gap vs logistic TPR gap"].value)
    race = load("head_race/summary_race.csv")
    if race is not None:
        g = race.set_index("stat")
        r = g.loc["rho probe vs TPR gap, all cells"]
        put("race_head_n_cells", int(r.n), "{:d}")
        put("race_head_spearman", r.value)
        put("race_head_spearman_ci", [r.ci_backbones_lo, r.ci_backbones_hi], text=rng_text(r.ci_backbones_lo, r.ci_backbones_hi))
        put("race_head_n_skipped", int(g.loc["n (cell, seed) draws skipped"].value), "{:d}")
        put("race_head_tpr_gap_mean", g.loc["mean tpr_gap"].value)
        lo, hi = g.loc["probe_auroc min"].value, g.loc["probe_auroc max"].value
        put("race_head_probe_auroc_range", [lo, hi], text=rng_text(lo, hi))
        hr = load("head_race/head_reliance.csv")
        if hr is not None:
            hr = hr[hr.feasible.astype(bool)]
            rows = []
            for bb in CHEXPERT_BACKBONES:
                for task in ("pneumothorax", "pleural_effusion", "cardiomegaly"):
                    c = hr[(hr.backbone == bb) & (hr.task == task)]
                    if c.empty:
                        continue
                    rows.append({"Encoder": bb_label("chexpert", bb), "Y": task.replace("_", " ").title(),
                                 "n train": f"{int(c.n_train.min())}-{int(c.n_train.max())}",
                                 "Probe": f2(c.probe_auroc.iloc[0]), "p": pfmt(c.probe_p.iloc[0]),
                                 "TPR gap": f"{f2(c.tpr_gap.mean())} ({f2(c.tpr_gap.std())})",
                                 "AUROC drop": f2(c.auroc_drop.mean())})
            hdr = ["Encoder", "$Y$", "$n$ train (min-max)", "Probe", "$p$", "TPR gap (SD)", "AUROC drop"]
            write_tabular("head_race", pd.DataFrame(rows), "ll" + "c" * (len(hdr) - 2), escape_cells=False, header=hdr)
    dc = load("dfr_control/dfr_control_summary.csv")
    if dc is not None:
        g = dc.set_index("method")
        for meth in ["dfr", "dfr_l2", "random_dfr_head", "random_l2"]:
            put(f"dfrctl_{meth}_task_auroc", g.loc[meth, "task_auroc_mean"])
            put(f"dfrctl_{meth}_tpr_gap", g.loc[meth, "tpr_gap_mean"])
    sv = load("sample_vs_full/sample_vs_full.csv")
    if sv is not None:  # audited CheXpert sample against the full train.csv
        g = sv.set_index("statistic")
        for side in ("sample", "full"):
            put(f"{side}_female_pct", g.loc["female_pct", side], "{:.0f}")
            put(f"{side}_age_median", g.loc["age_median", side], "{:.0f}")
            put(f"{side}_lateral_pct", g.loc["lateral_pct", side], "{:.0f}")
            put(f"{side}_white_pct", g.loc["white_pct_of_known", side], "{:.1f}")
    nl = load("nonlinear_probe/summary.csv")
    if nl is not None:  # probes on nonlinear features under the same null (run_nonlinear_probe_check.py)
        probes = ["linear", "sq", "prod"]
        g = nl.set_index(["variant", "delta", "probe"])
        rows = []
        label = {"none": "Null", "linear": "Linear", "xor": "XOR", "variance": "Variance"}
        for v in ["none", "linear", "xor", "variance"]:
            for dl in sorted(nl[nl.variant == v].delta.unique()):
                r = {"Variant": label[v], "delta": f"{dl:g}"}
                for p in probes:
                    c = g.loc[(v, dl, p)]
                    r[f"{p}_rej"] = f2(c.reject_rate)
                    r[f"{p}_auc"] = f2(c.auroc_mean)
                rows.append(r)
        hdr = ["Variant", "$\\delta$"] + ["Rej.", "AUROC"] * len(probes)
        prehead = ["\\multicolumn{2}{c}{} & \\multicolumn{2}{c}{$[E]$} & \\multicolumn{2}{c}{$[E,E^2]$} "
                   "& \\multicolumn{2}{c}{$[E,E^2,E_iE_j]$} \\\\",
                   "\\cmidrule(lr){3-4}\\cmidrule(lr){5-6}\\cmidrule(lr){7-8}"]
        write_tabular("nonlinear_probe", pd.DataFrame(rows), "lc" + "c" * (len(hdr) - 2), escape_cells=False,
                      header=hdr, prehead=prehead, midrules=[1, 3, 5])
        top = float(nl.delta.max())
        put("nl_probe_xor_reject_sq", g.loc[("xor", top, "sq")].reject_rate)
        put("nl_probe_var_reject_sq", g.loc[("variance", top, "sq")].reject_rate)
        put("nl_probe_xor_reject_prod", g.loc[("xor", top, "prod")].reject_rate)
        put("nl_probe_var_reject_prod", g.loc[("variance", top, "prod")].reject_rate)
        put("nl_probe_xor_reject_linear", g.loc[("xor", top, "linear")].reject_rate)
        put("nl_probe_var_reject_linear", g.loc[("variance", top, "linear")].reject_rate)
        put("nl_probe_xor_auroc_prod", g.loc[("xor", top, "prod")].auroc_mean)
        put("nl_probe_var_auroc_sq", g.loc[("variance", top, "sq")].auroc_mean)
        put("nl_probe_null_reject_linear", g.loc[("none", 0.0, "linear")].reject_rate)
        put("nl_probe_null_reject_sq", g.loc[("none", 0.0, "sq")].reject_rate)
        put("nl_probe_null_reject_prod", g.loc[("none", 0.0, "prod")].reject_rate)
        low = float(nl[nl.delta > 0].delta.min())
        put("nl_probe_delta_low", low, "{:.1f}")
        put("nl_probe_delta_high", top, "{:.1f}")
        put("nl_probe_xor_reject_prod_low", g.loc[("xor", low, "prod")].reject_rate)
        put("nl_probe_var_reject_sq_low", g.loc[("variance", low, "sq")].reject_rate)
        put("nl_probe_n_seeds", int(nl.n_seeds.max()), "{:d}")
        put("nl_probe_prod_features", int(g.loc[("none", 0.0, "prod")].n_features), "{:d}")
        man = ROOT / "nonlinear_probe" / "MANIFEST.txt"
        if man.exists():
            cfg = json.loads(man.read_text().split("config:\n", 1)[1].split("\n\n", 1)[0])
            put("nl_probe_n", int(cfg["n"]), "{:d}")
            put("nl_probe_d", int(cfg["d"]), "{:d}")
            put("nl_probe_rho", float(cfg["rho"]), "{:.1f}")
            put("nl_probe_K", int(cfg["K"]), "{:d}")
    nlc = load("nonlinear_probe_clinical/rows.csv")
    if nlc is not None:  # squared-feature probe on the clinical cells (run_nonlinear_probe_clinical.py)
        nlc = nlc[nlc.evaluated.astype(bool)].copy()
        nlc["flagged_sq"] = nlc.p_sq < 0.05
        nlc["flagged_linear"] = nlc.flagged_linear.astype(str).str.lower().isin(["true", "1"])
        nlc["changed"] = nlc.flagged_sq != nlc.flagged_linear
        rows, mids = [], []
        for ds, order in (("chexpert", ["resnet50", "densenet121", "vit_b_16", "vit_b_32", "dinov2_base",
                                         "rad_dino", "medsiglip"]),
                          ("mimic", ["rad_dino", "vit16_cls", "vit32_cls", "medsiglip"])):
            for bb in order:
                for attr in ["sex", "age", "race"]:
                    sub = nlc[(nlc.dataset == ds) & (nlc.encoder == bb) & (nlc.attribute == attr)]
                    if sub.empty:
                        continue
                    r = sub.iloc[0]
                    flag = "yes" if r.flagged_sq else "no"
                    if r.changed:
                        flag += " (linear: " + ("yes" if r.flagged_linear else "no") + ")"
                    rows.append({
                        "Dataset": {"chexpert": "CheXpert", "mimic": "MIMIC-CXR"}[ds]
                        if (bb == order[0] and attr == "sex") else "",
                        "Encoder": bb_label(ds, bb) if attr == "sex" else "",
                        "Attribute": {"sex": "Sex", "age": "Age 60+", "race": "White vs non-White"}[attr],
                        "linear": f2(r.probe_auroc_linear), "sq": f2(r.probe_auroc_sq),
                        # three decimals where two would print "0.05" beside a flag (p = 0.046)
                        "p": pfmt(r.p_sq) if pfmt(r.p_sq) != "0.05" else f"{r.p_sq:.3f}", "Flag": flag,
                    })
                mids.append(len(rows))
        write_tabular("nonlinear_probe_clinical", pd.DataFrame(rows), "lllcccc", escape_cells=False,
                      header=["Dataset", "Encoder", "Attribute", "AUROC $[E]$", "AUROC $[E,E^2]$", "$p$", "Flag"],
                      midrules=mids[:-1])
        put("nlc_n_cells", int(len(nlc)), "{:d}")
        put("nlc_n_cells_chexpert", int((nlc.dataset == "chexpert").sum()), "{:d}")
        put("nlc_n_cells_mimic", int((nlc.dataset == "mimic").sum()), "{:d}")
        put("nlc_n_sig_sq_chexpert", int(nlc[nlc.dataset == "chexpert"].flagged_sq.sum()), "{:d}")
        put("nlc_n_sig_sq_mimic", int(nlc[nlc.dataset == "mimic"].flagged_sq.sum()), "{:d}")
        put("nlc_n_sig_linear", int(nlc.flagged_linear.sum()), "{:d}")
        put("nlc_n_decisions_changed", int(nlc.changed.sum()), "{:d}")
        gain = nlc.probe_auroc_sq - nlc.probe_auroc_linear
        put("nlc_max_auroc_gain", float(gain.max()), "{:.2f}")
        put("nlc_min_auroc_gain", float(gain.min()), "{:.2f}")
        put("nlc_max_abs_auroc_diff", float(gain.abs().max()), "{:.2f}")
        put("nlc_K", int(nlc.n_permutations.max()), "{:d}")
    sc = load("probe_scaling/rows.csv")
    if sc is not None:  # probe run time against n on synthetic embeddings (run_probe_scaling.py)
        pts = sc.groupby("n", sort=True).agg(sec=("seconds_at_K500", "first"), extrap=("extrapolated", "any"))
        for n, r in pts.iterrows():
            key = f"probe_scaling_sec_{int(n) // 1000}k"
            put(key, float(r.sec), "{:.0f}", text=f"{r.sec / 60:.0f} min" if r.sec > 600 else None)
        put("probe_scaling_n_max", int(pts.index.max()), "{:,d}")
        slope = np.polyfit(np.log(pts.index.to_numpy(dtype=float)), np.log(pts.sec.to_numpy(dtype=float)), 1)[0]
        put("probe_scaling_exponent", float(slope), "{:.1f}")
        if pts.extrap.any():  # some points measured at smaller K and read at 500 from a line in K
            ks = sorted(int(k) for k in sc[sc.extrapolated].K.unique())
            ns_ext = ", ".join(f"{int(n):,d}" for n in pts.index[pts.extrap])
            put("probe_scaling_K", 500, text=f"500 (measured at K = {' and '.join(map(str, ks))} for n = {ns_ext} "
                                              "and extrapolated linearly in K)")
        else:
            put("probe_scaling_K", int(sc.K.max()), "{:d}")


def extraction():
    p = ROOT / "extraction_check" / "extraction_check.json"
    if not p.exists():
        return
    rows = json.loads(p.read_text())
    ids = {}
    for r in rows:
        if "cos_mean" not in r:
            continue
        b = r["backbone"]
        if b not in ids or r["cos_mean"] > ids[b]["cos_mean"]:
            ids[b] = {"model_id": r["model_id"], "revision": r.get("revision"), "pooling": r["pooling"],
                      "cos_mean": round(r["cos_mean"], 4), "cos_min": round(r["cos_min"], 4),
                      "reproduced": bool(r["cos_min"] > 0.99)}
    RAW["model_ids"] = ids
    NUM["model_ids"] = "; ".join(
        f"{b}: {v['model_id']} ({v['pooling']}), cosine mean {v['cos_mean']:.3f}, min {v['cos_min']:.3f}"
        + ("" if v["reproduced"] else ", NOT reproduced") for b, v in ids.items())
    # the re-extraction each backbone's results now use (common.CHEXPERT_PUBLIC); vit_b_16 is the pipeline control
    used = {("vit_b_32", "vit_b_32", "class token (heads = Identity)"),
            ("dinov2_base", "facebook/dinov2-base", "pooler_output"),
            ("rad_dino", "microsoft/rad-dino", "pooler_output"),
            ("medsiglip", "google/medsiglip-448", "vision pooler_output")}
    t = pd.DataFrame([{"Encoder": BACKBONE_LABELS.get(r["backbone"], r["backbone"]), "Candidate": r["model_id"],
                       "Pooling": r["pooling"],
                       "Used": "control" if r["backbone"] == "vit_b_16" else
                       ("yes" if (r["backbone"], r["model_id"], r["pooling"]) in used else ""),
                       "Cosine mean": f2(r.get("cos_mean"), 3) if "cos_mean" in r else "n/a",
                       "Cosine min": f2(r.get("cos_min"), 3) if "cos_mean" in r else "n/a",
                       "Note": "" if "cos_mean" in r else _short_error(r.get("error", ""))} for r in rows])
    write_tabular("extraction_check", t, "llllccl")


def alignment():
    """Misalignment check (section 9): a backbone is excluded only when two
    independent signs agree. (1) Its rows share no similarity structure with the
    other backbones of the same dataset (max Spearman of the cosine structure
    below 0.05, against about 0 for a shuffled control). (2) Its conditional sex
    probe is at chance (AUROC below 0.6) while the other backbones of the
    dataset are well above it (median above 0.75)."""
    a = load("alignment_check/alignment.csv")
    if a is None:
        return
    aud = load("clinical_audit/audit_rows.csv")
    sex_auc = {}
    if aud is not None:
        sx = aud[(aud.variant == "primary") & (aud.attribute == "sex") & (aud.detector == "probe")]
        sex_auc = {(r.dataset, r.backbone): float(r.statistic) for r in sx.itertuples()}
    sp = load("alignment_check/alignment_sexprobe.csv")
    if sp is not None:  # stored ViT-B/32 file: plain sex probe from the alignment check
        for r in sp[sp.backbone == "vit32_cls_stored"].itertuples():
            sex_auc[(r.dataset, r.backbone)] = float(r.sex_auroc)
        BACKBONE_LABELS.setdefault("vit32_cls_stored", "ViT-B/32 (stored file, replaced)")
    real = a[~a.b.str.contains("shuffled")]
    shuf = a[a.b.str.contains("shuffled")]
    rows, bad = [], []
    for ds, sub in real.groupby("dataset"):
        bbs = sorted(set(sub.a) | set(sub.b))
        for bb in bbs:
            mx = sub[(sub.a == bb) | (sub.b == bb)].spearman.max()
            ctl = shuf[(shuf.dataset == ds) & (shuf.a == bb)].spearman.iloc[0]
            auc = sex_auc.get((ds, bb), float("nan"))
            others = [sex_auc[(ds, b)] for b in bbs if b != bb and (ds, b) in sex_auc]
            sim_bad = bool(mx < 0.05)
            probe_bad = bool(np.isfinite(auc) and auc < 0.6 and others and np.median(others) > 0.75)
            excluded = sim_bad and probe_bad
            if excluded:
                bad.append(f"{ds}:{bb}")
                MISALIGNED.add((ds, bb))
                if not bb.endswith("_stored"):
                    BACKBONE_LABELS[bb] = BACKBONE_LABELS.get(bb, bb) + " (rows misaligned)"
            rows.append({"Dataset": {"chexpert": "CheXpert", "mimic": "MIMIC-CXR"}[ds],
                         "Backbone": bb_label(ds, bb), "Max Spearman": f2(mx, 3),
                         "Shuffled control": f2(ctl, 3), "Sex probe AUROC": f2(auc),
                         "Similarity fails": "yes" if sim_bad else "no",
                         "Probe at chance": "yes" if probe_bad else "no",
                         "Replaced": "yes" if excluded else "no"})
            put(f"alignment_max_spearman_{ds}_{bb}", float(mx), "{:.3f}")
    for key, bb in (("vit32_sex_auroc_mimic", "vit32_cls"), ("vit16_sex_auroc_mimic", "vit16_cls"),
                    ("vit32_stored_sex_auroc_mimic", "vit32_cls_stored")):
        if ("mimic", bb) in sex_auc:
            put(key, sex_auc[("mimic", bb)])
    put("alignment_misaligned_backbones", ", ".join(bad) if bad else "none", None)
    write_tabular("alignment_check", pd.DataFrame(rows), "llcccccc",
                  header=["Dataset", "Encoder", "Max Spearman", "Shuffled control", "Sex probe AUROC",
                          "Similarity fails", "Probe at chance", "Replaced"])


def finish():
    out = {k: NUM[k] for k in sorted(NUM)}
    out["raw"] = {k: RAW[k] for k in sorted(RAW)}
    path = ROOT / "numbers.json"
    old = json.loads(path.read_text()) if path.exists() else {}
    for k, v in old.items():  # keep keys written by other scripts (e.g. model_ids)
        if k == "raw":
            out["raw"] = {**v, **out["raw"]}
        elif k not in out:
            out[k] = v
    path.write_text(json.dumps(out, indent=2) + "\n")
    from export_numbers_tex import main as export_tex

    export_tex(str(path), str(TAB / "numbers.tex"))
    # top-level manifest and repo copy
    lines = [f"revision results folder: {ROOT}", f"tables/figures built at commit {git_sha(False)}", ""]
    for m in sorted(ROOT.glob("*/MANIFEST.txt")):
        lines += [f"===== {m.parent.name} =====", m.read_text()]
    (ROOT / "MANIFEST.txt").write_text("\n".join(lines))
    if os.environ.get("SHORTKIT_NO_REPO_COPY"):
        print(f"numbers: {len(NUM)} keys (repo copy skipped)")
        return
    dest = REPO / "output" / "paper_tables" / "revision"
    if dest.exists():  # replace, so files from an earlier build do not linger
        shutil.rmtree(dest)
    for sub in ("tables", "figures"):
        (dest / sub).mkdir(parents=True, exist_ok=True)
        for f in (ROOT / sub).iterdir():
            shutil.copy2(f, dest / sub / f.name)
    for f in ("numbers.json", "MANIFEST.txt"):
        shutil.copy2(ROOT / f, dest / f)
    print(f"numbers: {len(NUM)} keys; copied to {dest}")


# ---------------------------------------------------------------------------
# Figures, v2: polished variants written next to the originals as <name>_v2.pdf
# and <name>_v2.png (200 dpi). Same data and panels as the originals; sized for
# the printed width (elsarticle preprint, \textwidth = 390 pt) with 7 to 9 pt text.
# Reads only files the main pass has written (CSV results, tables/, numbers.json),
# so `--v2-only` redraws them without rebuilding tables or numbers.json.
# ---------------------------------------------------------------------------

V2_TEXTWIDTH_IN = 390 / 72.27
# categorical palette (validated colour-blind safe, dataviz reference palette)
V2C = {"blue": "#2a78d6", "orange": "#eb6834", "aqua": "#1baf7a", "violet": "#6250d6",
       "magenta": "#e87ba4", "red": "#e34948", "ink": "#0b0b0b", "ink2": "#52514e",
       "muted": "#898781", "grid": "#e1e0d9", "axis": "#c3c2b7"}
V2_BLUE_SEQ = LinearSegmentedColormap.from_list("v2blue", ["#ffffff", "#cde2fb", "#86b6ef", "#3987e5", "#1c5cab", "#0d366b"])
V2_ORANGE_SEQ = LinearSegmentedColormap.from_list("v2orange", ["#ffffff", "#fbd9c8", "#f4a27e", "#eb6834", "#b4451c"])
V2_ORDINAL_BLUE = ["#86b6ef", "#5598e7", "#2a78d6", "#1c5cab", "#0d366b"]  # light to dark, 5 steps
V2_RC = {
    "font.family": "sans-serif", "font.sans-serif": ["Arial", "Helvetica", "DejaVu Sans"],
    "font.size": 8, "axes.titlesize": 8.5, "axes.labelsize": 8, "xtick.labelsize": 8,
    "ytick.labelsize": 8, "legend.fontsize": 7.5, "pdf.fonttype": 42, "ps.fonttype": 42,
    "savefig.bbox": "tight", "savefig.pad_inches": 0.02, "figure.dpi": 100,
    "axes.grid": True, "grid.color": V2C["grid"], "grid.linewidth": 0.5, "grid.alpha": 1.0,
    "axes.spines.top": False, "axes.spines.right": False, "axes.edgecolor": V2C["axis"],
    "axes.linewidth": 0.6, "axes.labelcolor": V2C["ink"], "xtick.color": V2C["axis"],
    "ytick.color": V2C["axis"], "xtick.labelcolor": V2C["ink2"], "ytick.labelcolor": V2C["ink2"],
    "xtick.major.width": 0.6, "ytick.major.width": 0.6, "xtick.major.size": 2.5, "ytick.major.size": 2.5,
    "lines.linewidth": 1.4, "lines.markersize": 3.5, "legend.frameon": False, "legend.handlelength": 1.6,
    "axes.titlecolor": V2C["ink"], "axes.titleweight": "normal", "axes.unicode_minus": True,
    "axes.axisbelow": True,
}


def _v2_save(fig, name: str, tight: bool = True) -> None:
    # tight=False keeps the canvas at its designed printed size (bbox_inches=None falls back to the rc value)
    with plt.rc_context({"savefig.bbox": "tight" if tight else None}):
        fig.savefig(FIG / f"{name}_v2.pdf")
        fig.savefig(FIG / f"{name}_v2.png", dpi=200)
    plt.close(fig)
    print(f"[v2] {name}_v2.pdf/.png")


def _v2_text_colour(cmap, value) -> str:
    """Ink colour for text drawn over a colormap cell, by the cell's luminance."""
    r, g, b, _ = cmap(float(value))
    return "white" if 0.2126 * r + 0.7152 * g + 0.0722 * b < 0.5 else V2C["ink"]


def _v2_numbers() -> dict:
    p = ROOT / "numbers.json"
    return json.loads(p.read_text()) if p.exists() else {}


def v2_synthetic():
    df = load("synthetic/synthetic_runs.csv")
    if df is None:
        return
    df = apply_flag_rule(df, ["variant", "rho", "delta", "n", "d", "seed"])
    null = df[(df.delta == 0) & (df.variant == "linear")]
    alt = df[df.delta > 0]
    deltas = sorted(alt.delta.unique())
    variants = [("linear", "Linear"), ("xor", "XOR"), ("variance", "Variance")]

    def sens(det, var, dl):
        return alt[(alt.variant == var) & (alt.detector == det) & (alt.delta == dl)].flag.mean()

    fig = plt.figure(figsize=(V2_TEXTWIDTH_IN, 2.1))
    gs = fig.add_gridspec(1, 4, width_ratios=[1, 1, 1, 0.05], wspace=0.08, left=0.09, right=0.9,
                          top=0.9, bottom=0.24)
    axes = [fig.add_subplot(gs[0, i]) for i in range(3)]
    im_det = im_fp = None
    for ax, (v, title) in zip(axes, variants, strict=True):
        mat = np.array([[null[null.detector == d].flag.mean()] + [sens(d, v, dl) for dl in deltas] for d in DET])
        fp = np.ma.masked_all(mat.shape)
        fp[:, 0] = mat[:, 0]
        det = np.ma.array(mat, mask=np.zeros_like(mat, bool))
        det.mask[:, 0] = True
        im_det = ax.imshow(det, cmap=V2_BLUE_SEQ, vmin=0, vmax=1, aspect="auto")
        im_fp = ax.imshow(fp, cmap=V2_ORANGE_SEQ, vmin=0, vmax=0.2, aspect="auto")
        for i in range(mat.shape[0]):
            for j in range(mat.shape[1]):
                col = _v2_text_colour(V2_ORANGE_SEQ, mat[i, j] / 0.2) if j == 0 else _v2_text_colour(V2_BLUE_SEQ, mat[i, j])
                ax.text(j, i, f"{mat[i, j]:.2f}", ha="center", va="center", fontsize=7, color=col)
        ax.axvline(0.5, color="white", lw=2.0)  # surface gap between the null column and the rest
        ax.set_xticks(range(len(deltas) + 1), ["0"] + [f"{x:g}" for x in deltas])
        ax.set_title(title)
        ax.grid(False)
        ax.tick_params(length=0)
        for sp in ax.spines.values():
            sp.set_visible(False)
        if ax is axes[0]:
            ax.set_yticks(range(len(DET)), [DET_LABEL[d] for d in DET])
        else:
            ax.set_yticks(range(len(DET)), [""] * len(DET))
    fig.supxlabel("Attribute signal δ (Cohen's d); the δ = 0 column is the false positive rate", fontsize=8, y=0.03)
    col = fig.add_subplot(gs[0, 3])
    col.axis("off")
    cax2 = col.inset_axes([0.0, 0.56, 1.0, 0.44])
    cax1 = col.inset_axes([0.0, 0.0, 1.0, 0.44])
    cb1 = fig.colorbar(im_fp, cax=cax1)
    cb2 = fig.colorbar(im_det, cax=cax2)
    cb1.set_ticks([0, 0.1, 0.2])
    cb2.set_ticks([0, 0.5, 1])
    for cb in (cb1, cb2):
        cb.ax.tick_params(length=0, labelsize=6.5, pad=1.5)
        cb.outline.set_visible(False)
    cb1.set_label("FP rate (δ = 0)", fontsize=7, labelpad=4)
    cb2.set_label("Flag rate (δ > 0)", fontsize=7, labelpad=4)
    _v2_save(fig, "synthetic_detection_heatmap")


def v2_aggregation_sweep():
    p = TAB / "aggregation_sweep.csv"
    if not p.exists():
        print("[skip] tables/aggregation_sweep.csv not found (run the full build once)")
        return
    sw = pd.read_csv(p)
    deltas = sorted({float(c.split("_")[-1]) for c in sw.columns if c.startswith("sens_linear_")})
    fig, axes = plt.subplots(1, 3, figsize=(V2_TEXTWIDTH_IN, 1.9), sharey=True,
                             gridspec_kw={"wspace": 0.12, "left": 0.07, "right": 0.99, "top": 0.88, "bottom": 0.22})
    for ax, (v, title) in zip(axes, [("linear", "Linear"), ("xor", "XOR"), ("variance", "Variance")], strict=True):
        for c, dl in zip(V2_ORDINAL_BLUE, deltas, strict=True):
            y = sw[f"sens_{v}_{dl:g}"]
            ax.plot(sw.k, y, marker="o", color=c, label=f"δ = {dl:g}", markeredgecolor="white", markeredgewidth=0.5)
            if v == "linear" and dl >= 0.5:  # direct labels where the lines separate at the end
                ax.annotate(f"δ = {dl:g}", (sw.k.iloc[-1], y.iloc[-1]), xytext=(4, 0), textcoords="offset points",
                            fontsize=6.5, color=V2C["ink2"], va="center")
        ax.plot(sw.k, sw.fpr, marker="s", ls=(0, (3, 2)), color=V2C["red"], label="FPR (δ = 0)",
                markeredgecolor="white", markeredgewidth=0.5)
        ax.set_title(title)
        ax.set_xticks(range(1, 7))
        ax.set_xlim(0.7, 6.9 if v == "linear" else 6.3)
        ax.set_ylim(-0.03, 1.05)
        ax.grid(axis="x", visible=False)
    axes[0].set_ylabel("Rate")
    fig.supxlabel("k: flagged by at least k of the 6 detectors", fontsize=8, y=0.03)
    axes[1].legend(loc="upper right", ncol=2, columnspacing=0.8, handletextpad=0.4, borderaxespad=0.2)
    _v2_save(fig, "aggregation_sweep")


def v2_statistical_correction():
    sc = load("synthetic/statistical_correction.csv")
    if sc is None:
        return
    fig, ax = plt.subplots(figsize=(0.6 * V2_TEXTWIDTH_IN, 2.3))
    lab = {"uncorrected": "Uncorrected", "bonferroni": "Bonferroni", "holm": "Holm",
           "fdr_bh": "FDR (BH)", "fdr_by": "FDR (BY)"}
    # the caption names uncorrected and BH: those two carry colour, the rest stay grey
    style = {"uncorrected": dict(color=V2C["orange"], zorder=4), "fdr_bh": dict(color=V2C["blue"], zorder=5),
             "bonferroni": dict(color=V2C["muted"], ls=(0, (3, 2)), zorder=3),
             "holm": dict(color=V2C["muted"], ls=(0, (1, 1.5)), zorder=3),
             "fdr_by": dict(color=V2C["muted"], ls="-", zorder=2)}
    for c in ["uncorrected", "fdr_bh", "bonferroni", "holm", "fdr_by"]:
        g = sc[sc.correction == c].sort_values("d")
        ax.plot(g.d, g.fpr, marker="o", label=lab[c], markeredgecolor="white", markeredgewidth=0.5, **style[c])
    ax.axhline(0.05, color=V2C["ink"], ls=(0, (1, 2)), lw=0.8, zorder=1, label="Nominal level 0.05")
    ax.set_xscale("log", base=2)
    ds = sorted(sc.d.unique())
    ax.set_xticks(ds, [str(int(d)) for d in ds])
    ax.set_xlim(ds[0] / 1.2, ds[-1] * 1.2)
    ax.set_ylim(-0.03, 1.05)
    ax.set_xlabel("Embedding dimension d")
    ax.set_ylabel("Share of null data sets with\nany significant dimension")
    ax.legend(loc="center right", bbox_to_anchor=(1.0, 0.5), handlelength=2.0)
    ax.grid(axis="x", visible=False)
    _v2_save(fig, "synthetic_statistical_correction")


def v2_clinical_heatmap():
    df = load("clinical_audit/audit_rows.csv")
    if df is None:
        return
    df = df[df.evaluated.astype(bool)]
    df = apply_flag_rule(df, ["dataset", "backbone", "task", "attribute", "variant"])
    prim = df[(df.variant == "primary") & df.detector.isin(DET)]
    attr_lab = {"sex": "Sex", "age": "Age 60+", "race": "Race"}
    zmax = 100.0
    fig = plt.figure(figsize=(V2_TEXTWIDTH_IN, 5.6))
    gs = fig.add_gridspec(2, 2, height_ratios=[12, 9], width_ratios=[1, 1], wspace=0.72, hspace=0.35,
                          left=0.17, right=0.99, top=0.95, bottom=0.07)
    ax_c = fig.add_subplot(gs[:, 0])
    ax_m = fig.add_subplot(gs[0, 1])
    ax_leg = fig.add_subplot(gs[1, 1])
    ax_leg.axis("off")
    im = None
    for ax, ds in ((ax_c, "chexpert"), (ax_m, "mimic")):
        order = CHEXPERT_BACKBONES if ds == "chexpert" else MIMIC_BACKBONES
        sub = prim[prim.dataset == ds]
        rows, Z, F, S = [], [], [], []
        for bb in order:
            for attr in ["sex", "age", "race"]:
                g = sub[(sub.backbone == bb) & (sub.attribute == attr)]
                if g.empty:
                    continue
                g = g.set_index("detector").reindex(DET)
                rows.append((bb, attr))
                Z.append(g.z.to_numpy(float))
                F.append(g.flag.to_numpy(bool))
                S.append(g.statistic.to_numpy(float))
        Z = np.clip(np.array(Z), 0, zmax)
        L = np.log1p(Z)
        im = ax.imshow(L, cmap=V2_BLUE_SEQ, aspect="auto", vmin=0, vmax=np.log1p(zmax))
        for i in range(L.shape[0]):
            for j in range(L.shape[1]):
                ax.text(j, i, f"{S[i][j]:.2f}" + ("*" if F[i][j] else ""), ha="center", va="center", fontsize=7,
                        color=_v2_text_colour(V2_BLUE_SEQ, L[i, j] / np.log1p(zmax)))
        # backbone groups: name to the left of the attribute labels, white gap between groups
        ax.set_yticks(range(len(rows)), [attr_lab[a] for _, a in rows])
        starts = [i for i, (bb, _) in enumerate(rows) if i == 0 or rows[i - 1][0] != bb]
        for k, s0 in enumerate(starts):
            s1 = (starts[k + 1] if k + 1 < len(starts) else len(rows)) - 1
            ax.annotate(bb_label(ds, rows[s0][0]), xy=(0, (s0 + s1) / 2), xycoords=("axes fraction", "data"),
                        xytext=(-38, 0), textcoords="offset points", ha="right", va="center", fontsize=7.5,
                        color=V2C["ink"])
            if s0 > 0:
                ax.axhline(s0 - 0.5, color="white", lw=2.0)
        ax.set_xticks(range(len(DET)), [DET_LABEL[d] for d in DET], rotation=35, ha="right", rotation_mode="anchor")
        ax.set_title({"chexpert": "CheXpert, Y = Pneumothorax", "mimic": "MIMIC-CXR, Y = mortality"}[ds])
        ax.grid(False)
        ax.tick_params(length=0, pad=2)
        for sp in ax.spines.values():
            sp.set_visible(False)
    cax = ax_leg.inset_axes([0.0, 0.62, 1.0, 0.07])
    cb = fig.colorbar(im, cax=cax, orientation="horizontal")
    ticks = [0, 1, 3, 10, 30, 100]
    cb.set_ticks(np.log1p(ticks), labels=["0\n(null mean)", "1", "3", "10", "30", "100 or more"])
    cb.ax.tick_params(length=0, labelsize=6.5, pad=2)
    cb.outline.set_visible(False)
    cb.set_label("z = (score - null mean) / null SD, log scale", fontsize=7.5, labelpad=4)
    ax_leg.text(0.0, 0.3, "Cell text: continuous score\n* probe p < 0.05; secondary detectors\n  BH q < 0.05 across the five within the cell",
                transform=ax_leg.transAxes, fontsize=7.5, color=V2C["ink2"], va="top")
    _v2_save(fig, "clinical_scores_heatmap")


def v2_audit_vs_reliance():
    p = TAB / "head_reliance_cells.csv"
    num = _v2_numbers()
    if not p.exists() or "spearman_probe_tprgap_chexpert" not in num:
        print("[skip] head_reliance_cells.csv or numbers.json missing (run the full build once)")
        return
    cell = pd.read_csv(p)
    raw = num.get("raw", {})
    markers = dict(zip(CHEXPERT_BACKBONES, ["o", "s", "^", "D", "v", "P", "X"], strict=True))
    attr_style = {"sex": (V2C["blue"], "Sex"), "age": (V2C["orange"], "Age 60+")}
    panels = [("tpr_gap", "tprgap", "TPR gap\n(A=1 minus A=0)"),
              ("auroc_drop", "aurocdrop", "AUROC drop vs\nunconfounded head"),
              ("output_shift", "outputshift", "Mean |output shift|\nafter removing A")]
    fig, axes = plt.subplots(1, 3, figsize=(V2_TEXTWIDTH_IN, 2.6),
                             gridspec_kw={"wspace": 0.52, "left": 0.09, "right": 0.99, "top": 0.84, "bottom": 0.34})
    for ax, (col, key, lab) in zip(axes, panels, strict=True):
        for attr, (c, _) in attr_style.items():
            s = cell[cell.attribute == attr]
            for bb, mk in markers.items():
                b = s[s.backbone == bb]
                ax.scatter(b.probe_auroc, b[col], color=c, marker=mk, s=20, edgecolor="white", linewidth=0.5, zorder=3)
        rho = num[f"spearman_probe_{key}_chexpert"]
        pv = float(raw.get(f"spearman_probe_{key}_p_chexpert", float(num[f"spearman_probe_{key}_p_chexpert"])))
        ptxt = "p < 0.001" if pv < 0.001 else f"p = {pv:.3f}"
        title = f"Spearman ρ = {rho} ({ptxt})"
        if key == "tprgap" and "spearman_probe_tprgap_ci" in num:
            title += f"\n95% CI {num['spearman_probe_tprgap_ci']}"
        ax.set_title(title, fontsize=8)
        ax.set_ylabel(lab)
        ax.set_xticks([0.7, 0.8, 0.9, 1.0])
        if col in ("auroc_drop",):
            ax.axhline(0, color=V2C["axis"], lw=0.6, zorder=1)
    from matplotlib.lines import Line2D

    h_attr = [Line2D([], [], marker="o", ls="", color=c, markersize=5, label=l) for c, l in attr_style.values()]
    h_bb = [Line2D([], [], marker=mk, ls="", color=V2C["ink2"], markersize=4.5, label=bb_label("chexpert", bb))
            for bb, mk in markers.items()]
    fig.supxlabel("Conditional probe AUROC (audit)", fontsize=8, y=0.2)
    fig.legend(handles=h_attr + h_bb, loc="lower center", ncol=5, bbox_to_anchor=(0.53, 0.0), columnspacing=1.0,
               handletextpad=0.3, fontsize=7.5)
    _v2_save(fig, "audit_vs_reliance")


def v2_lda():
    lda = load("analyses/lda_heldout_mimic_raddino_sex.csv")
    if lda is None:
        return
    from sklearn.metrics import roc_auc_score

    auc = roc_auc_score(lda.sex == "Male", lda.projection)
    fig, ax = plt.subplots(figsize=(0.55 * V2_TEXTWIDTH_IN, 2.05))
    bins = np.linspace(lda.projection.quantile(0.005), lda.projection.quantile(0.995), 50)
    for sex, c in (("Female", V2C["orange"]), ("Male", V2C["blue"])):
        v = lda.projection[lda.sex == sex]
        ax.hist(v, bins=bins, alpha=0.75, color=c, label=f"{sex} (n = {len(v)})", density=True, linewidth=0)
    ax.set_xlabel("Held-out LDA projection (fit on training folds)")
    ax.set_ylabel("Density")
    ax.legend(loc="upper center", bbox_to_anchor=(0.5, 1.02))
    ax.text(0.5, 0.74, f"held-out AUROC {auc:.3f}", transform=ax.transAxes, ha="center", fontsize=7,
            color=V2C["ink2"])
    ax.grid(axis="x", visible=False)
    ax.set_ylim(0, ax.get_ylim()[1] * 1.08)
    _v2_save(fig, "lda_sex_mimic_raddino_heldout")


def v2_umaps():
    um = load("analyses/umap_mimic_raddino_race.csv")
    if um is not None:
        fig, ax = plt.subplots(figsize=(0.55 * V2_TEXTWIDTH_IN, 0.55 * V2_TEXTWIDTH_IN))
        # Okabe-Ito four, all pairs validated; the dominant group goes first, the smallest last and larger
        groups = [("White", "#56B4E9", 3.5), ("Other", "#CC79A7", 4.5), ("Black", "#D55E00", 4.5), ("Asian", "#009E73", 7)]
        for race, c, size in groups:
            s = um[um.race == race]
            ax.scatter(s.x, s.y, s=size, alpha=0.8, color=c, label=f"{race} ({len(s)})", linewidths=0)
        ax.set_xticks([]), ax.set_yticks([])
        ax.grid(False)
        handles, labels = ax.get_legend_handles_labels()
        order = [labels.index(next(l for l in labels if l.startswith(r))) for r in ("White", "Black", "Asian", "Other")]
        ax.legend([handles[i] for i in order], [labels[i] for i in order], loc="upper center",
                  bbox_to_anchor=(0.5, 0.0), ncol=4, markerscale=2.5, columnspacing=0.8,
                  handletextpad=0.1, borderaxespad=0.1, fontsize=7)
        for sp in ax.spines.values():
            sp.set_visible(False)
        fig.subplots_adjust(left=0.02, right=0.98, top=0.98, bottom=0.1)
        _v2_save(fig, "umap_mimic_raddino_race", tight=False)
    uc = load("analyses/umap_celeba_resnet50_male.csv")
    if uc is not None:
        fig, ax = plt.subplots(figsize=(0.55 * V2_TEXTWIDTH_IN, 0.55 * V2_TEXTWIDTH_IN))
        for male, lab, c in ((False, "Not Male", V2C["orange"]), (True, "Male", V2C["blue"])):
            s = uc[uc.male == male]
            ax.scatter(s.x, s.y, s=2.5, alpha=0.6, color=c, label=f"{lab} ({len(s)})", linewidths=0)
        ax.set_xticks([]), ax.set_yticks([])
        ax.grid(False)
        ax.legend(loc="upper center", bbox_to_anchor=(0.5, 0.0), ncol=2, markerscale=3, columnspacing=1.2,
                  handletextpad=0.1, borderaxespad=0.1, fontsize=7)
        for sp in ax.spines.values():
            sp.set_visible(False)
        fig.subplots_adjust(left=0.02, right=0.98, top=0.98, bottom=0.1)
        _v2_save(fig, "umap_celeba_resnet50_male", tight=False)


def figures_v2():
    with plt.rc_context(V2_RC):
        v2_synthetic()
        v2_aggregation_sweep()
        v2_statistical_correction()
        v2_clinical_heatmap()
        v2_audit_vs_reliance()
        v2_lda()
        v2_umaps()


def main(argv: list[str] | None = None) -> int:
    argv = sys.argv[1:] if argv is None else argv
    if "--v2-only" in argv:  # redraw the _v2 figures from the files of an earlier full build
        figures_v2()
        return 0
    settings()
    alignment()
    synthetic()
    clinical()
    cohorts()
    head()
    old_pipeline()
    negative_control()
    analyses()
    marker()
    extraction()
    sensitivity()
    figures_v2()
    finish()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
