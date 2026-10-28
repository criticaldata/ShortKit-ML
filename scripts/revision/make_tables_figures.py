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
from common import BACKBONE_LABELS, REPO, git_sha, results_root, to_jsonable  # noqa: E402

import matplotlib  # noqa: E402

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
from matplotlib.colors import LinearSegmentedColormap  # noqa: E402

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
RAW: dict[str, object] = {}


def put(key: str, raw, fmt: str | None = "{:.2f}", text: str | None = None) -> None:
    RAW[key] = to_jsonable(raw)
    if text is not None:
        NUM[key] = text
    elif isinstance(raw, str) or fmt is None:
        NUM[key] = str(raw)
    else:
        NUM[key] = fmt.format(raw)


def rng_text(lo: float, hi: float, fmt="{:.2f}") -> str:
    return f"{fmt.format(lo)} to {fmt.format(hi)}"


def tex_escape(s) -> str:
    s = str(s)
    for a, b in (("\\", r"\textbackslash{}"), ("&", r"\&"), ("%", r"\%"), ("_", r"\_"), ("#", r"\#")):
        s = s.replace(a, b)
    return s


def write_tabular(name: str, df: pd.DataFrame, colfmt: str | None = None, escape_cells=True,
                  header: list[str] | None = None, midrules: list[int] | None = None) -> None:
    cols = header or [tex_escape(c) for c in df.columns]
    colfmt = colfmt or ("l" * 1 + "c" * (len(cols) - 1))
    lines = [f"\\begin{{tabular}}{{{colfmt}}}", "\\toprule", " & ".join(cols) + " \\\\", "\\midrule"]
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
    return "n/a" if x is None or (isinstance(x, float) and not np.isfinite(x)) else f"{x:.{d}f}"


def pfmt(p):
    return "<0.01" if p < 0.01 else f"{p:.2f}"


# ---------------------------------------------------------------------------
# Settings
# ---------------------------------------------------------------------------


def settings():
    put("K_probe", 200, "{:d}")
    put("K_secondary", 200, "{:d}")
    put("probe_C", 1.0, "{:.0f}")
    put("hbac_k", 8, "{:d}")
    put("freq_topk", 5, "{:d}", text="5")
    RAW["freq_topk_fraction"] = 0.05
    put("code_commit", git_sha())
    put("clinical_seed", 42, "{:d}")
    put("n_bootstrap", 2000, "{:d}")
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
    df["flag"] = df["flag"].astype(bool)
    null = df[(df.delta == 0) & (df.variant == "linear")]  # the three variants coincide at delta 0
    alt = df[df.delta > 0]

    # null FPR table
    rows = []
    for d in DET + ["probe_unconditional"]:
        sub = null[null.detector == d]
        r = {"Detector": DET_LABEL[d]}
        for rho in sorted(df.rho.unique()):
            r[f"rho={rho}"] = f2(sub[sub.rho == rho].reject.mean())
        r["All"] = f2(sub.reject.mean())
        r["BH flag"] = f2(sub.flag.mean())
        rows.append(r)
        if d in DET_KEY:
            put(f"fpr_null_{DET_KEY[d]}", sub.reject.mean())
            put(f"fpr_null_bhflag_{DET_KEY[d]}", sub.flag.mean())
    put("fpr_null_probe_rho09", null[(null.detector == "probe") & (null.rho == 0.9)].reject.mean())
    put("fpr_null_unconditional_probe_rho09",
        null[(null.detector == "probe_unconditional") & (null.rho == 0.9)].reject.mean())
    RAW["n_null_datasets"] = int(null[null.detector == "probe"].shape[0])
    t = pd.DataFrame(rows)
    hdr = ["Detector"] + [f"$\\rho={r}$" for r in sorted(df.rho.unique())] + ["All", "BH flag"]
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
    fig.suptitle("Flag rate after BH; $\\delta=0$ column (orange-red) is the false-positive rate", fontsize=10)
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
    corr = w[DET].corr()
    off = corr.to_numpy()[~np.eye(len(DET), dtype=bool)]
    lam = np.clip(np.linalg.eigvalsh(np.nan_to_num(corr.to_numpy())), 0, None)
    put("mean_flag_correlation", float(np.nanmean(off)))
    put("effective_n_detectors", float(lam.sum() ** 2 / (lam ** 2).sum()), "{:.1f}")
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
    df["flag"] = df["flag"].astype(str).str.lower().isin(["true", "1"])
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
                    "Backbone": BACKBONE_LABELS[bb] if attr == "sex" else "",
                    "Attribute": {"sex": "Sex", "age": "Age 60+", "race": "White vs non-White"}[attr],
                    "Probe AUROC (95% CI)": _cell_text(sub, "probe", True),
                    "p": pfmt(pr.p_value),
                    **{DET_LABEL[d]: _cell_text(sub, d) for d in DET[1:]},
                    "Flags": f"{int(sub[sub.detector.isin(DET)].flag.sum())}/6",
                })
            mids.append(len(rows))
        t = pd.DataFrame(rows)
        hdr = ["Backbone", "Attribute", "Probe AUROC (95\\% CI)", "$p$"] + [DET_LABEL[d] for d in DET[1:]] + ["Flags"]
        write_tabular(f"clinical_audit_{ds}", t, "ll" + "c" * (len(hdr) - 2), escape_cells=False,
                      header=hdr, midrules=mids[:-1])
        pp = prim[(prim.dataset == ds) & (prim.detector == "probe")]
        put(f"n_cells_probe_sig_{ds}", int(pp.flag.sum()), "{:d}")
        put(f"n_cells_primary_{ds}", int(len(pp)), "{:d}")
        sx = pp[pp.attribute == "sex"].statistic
        put(f"probe_auroc_sex_min_{ds}", sx.min())
        put(f"probe_auroc_sex_max_{ds}", sx.max())
        for attr in ["age", "race"]:
            s = pp[pp.attribute == attr].statistic
            put(f"probe_auroc_{attr}_range_{ds}", [s.min(), s.max()], text=rng_text(s.min(), s.max()))
            put(f"n_cells_probe_sig_{attr}_{ds}", int(pp[pp.attribute == attr].flag.sum()), "{:d}")
        allflag = prim[(prim.dataset == ds) & prim.detector.isin(DET)]
        put(f"n_secondary_flags_{ds}", int(allflag[allflag.detector != "probe"].flag.sum()), "{:d}")
    m = prim[(prim.dataset == "mimic") & (prim.backbone == "rad_dino") & (prim.attribute == "sex") & (prim.detector == "probe")]
    if not m.empty:
        r = m.iloc[0]
        put("probe_auroc_mimic_raddino_sex", r.statistic)
        put("probe_auroc_ci_mimic_raddino_sex", [r.ci_low, r.ci_high], text=rng_text(r.ci_low, r.ci_high))
    # grouped vs plain CV on CheXpert
    for bb in df[df.dataset == "chexpert"].backbone.unique():
        g = prim[(prim.dataset == "chexpert") & (prim.backbone == bb) & (prim.attribute == "sex")]
        a, b = g[g.detector == "probe"], g[g.detector == "probe_plain_cv"]
        if len(a) and len(b):
            key = f"probe_auroc_sex_grouped_vs_plain_chexpert_{bb.replace('_', '')}"
            put(key, [a.statistic.iloc[0], b.statistic.iloc[0]],
                text=f"{a.statistic.iloc[0]:.3f} grouped vs {b.statistic.iloc[0]:.3f} plain")
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
                r = {"Dataset": {"chexpert": "CheXpert", "mimic": "MIMIC-CXR"}[ds], "Backbone": BACKBONE_LABELS[bb]}
                for v in variants:
                    s = sens[(sens.dataset == ds) & (sens.backbone == bb) & (sens.attribute == attr) & (sens.variant == v)]
                    r[v] = "n/a" if s.empty else f"{s.statistic.iloc[0]:.2f} ($p$ {pfmt(s.p_value.iloc[0])})"
                    if not s.empty:
                        r[v + "_n"] = int(s.n_images.iloc[0])
                rows.append(r)
            mids.append(len(rows))
        t = pd.DataFrame(rows)
        if attr == "race":
            hdr = ["Dataset", "Backbone", "White vs non-White", "W/B/A, 50+ per group", "W/B/A, 50+ per (group, $Y$)"]
            t = t[["Dataset", "Backbone"] + variants]
            notes = df[(df.attribute == "race") & (df.variant != "primary")][["dataset", "variant", "note", "n_images"]].drop_duplicates()
            notes.to_csv(TAB / "race_sensitivity_groups.csv", index=False)
        else:
            hdr = ["Dataset", "Backbone", "Age 60+ (binary)", "Age, 4 bins"]
            t = t[["Dataset", "Backbone"] + variants]
        write_tabular(name, t, "ll" + "c" * (len(hdr) - 2), escape_cells=False, header=hdr, midrules=mids[:-1])

    # continuous score heatmap (z against the null), primary cells
    _apply_pub_style()
    fig, axes = plt.subplots(1, 2, figsize=(11, 8.5), gridspec_kw={"width_ratios": [1, 1]})
    for ax, ds in zip(axes, ["chexpert", "mimic"], strict=True):
        sub = prim[(prim.dataset == ds) & prim.detector.isin(DET)]
        labels, Z, F, S = [], [], [], []
        for (bb, attr), g in sub.groupby(["backbone", "attribute"], sort=False):
            g = g.set_index("detector").reindex(DET)
            labels.append(f"{BACKBONE_LABELS[bb]} | {attr}")
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
    fig.text(0.5, 0.01, "Cell text: continuous score; * BH-adjusted p < 0.05", ha="center", fontsize=9)
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
        race = m["race"].astype(str).value_counts()
        rows.append({"Cohort": name, "Images": n, "Patients": pats,
                     "Male": f"{(m.sex == 'Male').sum()} ({100 * (m.sex == 'Male').mean():.0f}\\%)",
                     "Age 60+": f"{(m.age >= 60).sum()} ({100 * (m.age >= 60).mean():.0f}\\%)",
                     "White": race.get("WHITE", 0), "Black": race.get("BLACK", 0), "Asian": race.get("ASIAN", 0),
                     "Other": race.get("OTHER", 0),
                     "Task": task, "Y=1": f"{int(m.task_label.sum())} ({100 * m.task_label.mean():.1f}\\%)"})
    put("chexpert_pneumothorax_prevalence", 100 * mc.task_label.mean(), "{:.1f}")
    put("mimic_mortality_prevalence", 100 * mm.task_label.mean(), "{:.1f}")
    put("chexpert_n_patients", int(mc.patient_id.nunique()), "{:d}")
    put("chexpert_n_lateral", int((mc.view == "Lateral").sum()), "{:d}")
    hdr = ["Cohort", "Images", "Patients", "Male", "Age 60+", "White", "Black", "Asian", "Other", "Task ($Y$)", "$Y{=}1$"]
    write_tabular("cohort_demographics", pd.DataFrame(rows), "l" + "r" * 9 + "lr", escape_cells=False, header=hdr)


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
        n_test=("n_test", "mean"), n_seeds=("seed", "nunique")).reset_index()
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
        r, _ = spearmanr(s.probe_auroc, s.tpr_gap)
        put(f"spearman_probe_tprgap_within_{attr}", r)
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
                     "n train": f"{g.n_train.mean():.0f}", "n test": f"{g.n_test.mean():.0f}",
                     "Probe": f2(g.probe_auroc.mean()), "TPR gap": f2(g.tpr_gap.mean()),
                     "FPR gap": f2(g.fpr_gap.mean()), "AUROC drop": f2(g.auroc_drop.mean()),
                     "Output shift": f2(g.output_shift.mean()),
                     "AUROC aligned": f2(g.auroc_aligned.mean()), "AUROC misaligned": f2(g.auroc_misaligned.mean()),
                     "ECE A=1": f2(g.ece_A1.mean(), 3), "ECE A=0": f2(g.ece_A0.mean(), 3)})
    hdr = ["$A$", "$Y$", "$n$ train", "$n$ test", "Probe", "TPR gap", "FPR gap", "AUROC drop",
           "Output shift", "AUROC aligned", "AUROC misaligned", "ECE $A{=}1$", "ECE $A{=}0$"]
    write_tabular("head_reliance", pd.DataFrame(rows), "ll" + "c" * (len(hdr) - 2), escape_cells=False, header=hdr)

    _apply_pub_style()
    fig, axes = plt.subplots(1, 3, figsize=(12, 3.6))
    for ax, (col, lab) in zip(axes, [("tpr_gap", "TPR gap (A=1 minus A=0)"), ("auroc_drop", "AUROC drop vs unconfounded head"),
                                     ("output_shift", "Mean |output shift| after removing A")], strict=True):
        for attr, c, mk in (("sex", OI["blue"], "o"), ("age", OI["orange"], "s")):
            s = cell[cell.attribute == attr]
            ax.scatter(s.probe_auroc, s[col], color=c, marker=mk, s=28, label={"sex": "Sex", "age": "Age 60+"}[attr],
                       edgecolor="black", linewidth=0.4)
        r, p = rho_vals[col]
        ax.set_title(f"Spearman $\\rho$ = {r:.2f} (p = {p:.3f})", fontsize=10)
        ax.set_xlabel("Conditional probe AUROC (audit)")
        ax.set_ylabel(lab)
    axes[0].legend(frameon=False)
    fig.savefig(FIG / "audit_vs_reliance.pdf")
    plt.close(fig)

    m = load("head_experiment/mitigation.csv")
    if m is not None and not m.empty:
        put("mitig1_name", "adversarial debiasing")
        put("mitig2_name", "contrastive debiasing")
        rows = []
        for i, meth in ((1, "adversarial"), (2, "contrastive")):
            g = m[m.method == meth]
            a0 = g[g.seed == 0]
            put(f"mitig{i}_delta_probe_auroc", (a0.probe_auroc_after - a0.probe_auroc_before).mean())
            put(f"mitig{i}_delta_task_auroc", (g.task_auroc_after - g.task_auroc_before).mean())
            put(f"mitig{i}_delta_tpr_gap", (g.tpr_gap_after - g.tpr_gap_before).mean())
            put(f"mitig{i}_n_probe_sig_after", int((a0.probe_p_after < 0.05).sum()), "{:d}")
            put(f"mitig{i}_n_probe_sig_before", int((a0.probe_p_before < 0.05).sum()), "{:d}")
            for attr in ["sex", "age"]:
                ga, a0a = g[g.attribute == attr], a0[a0.attribute == attr]
                rows.append({"Mitigation": {"adversarial": "Adversarial debiasing", "contrastive": "Contrastive debiasing"}[meth] if attr == "sex" else "",
                             "A": {"sex": "Sex", "age": "Age 60+"}[attr],
                             "Probe before": f2(a0a.probe_auroc_before.mean()), "Probe after": f2(a0a.probe_auroc_after.mean()),
                             "Sig. before": f"{int((a0a.probe_p_before < 0.05).sum())}/{len(a0a)}",
                             "Sig. after": f"{int((a0a.probe_p_after < 0.05).sum())}/{len(a0a)}",
                             "Task before": f2(ga.task_auroc_before.mean()), "Task after": f2(ga.task_auroc_after.mean()),
                             "TPR gap before": f2(ga.tpr_gap_before.mean()), "TPR gap after": f2(ga.tpr_gap_after.mean())})
        hdr = ["Mitigation", "$A$", "Probe before", "Probe after", "Sig. before", "Sig. after",
               "Task AUROC before", "Task AUROC after", "TPR gap before", "TPR gap after"]
        write_tabular("mitigation_before_after", pd.DataFrame(rows), "ll" + "c" * 8, escape_cells=False,
                      header=hdr, midrules=[2])


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
                      header=["Dataset", "Variable", "NMI (mean)", "NMI (range over backbones)",
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


def marker():
    mk = load("planted_marker/planted_marker.csv")
    if mk is None:
        return
    put("marker_rate", 0.5, "{:.1f}")
    put("marker_rate_head_train", 0.9, "{:.1f}")
    put("probe_auroc_marker", mk.probe_stat.mean())
    put("probe_p_marker", mk.probe_p.max(), "{:.3f}")
    put("head_tprgap_marker", mk.tpr_gap.mean())
    put("head_fprgap_marker", mk.fpr_gap.mean())
    put("marker_n_seeds", int(len(mk)), "{:d}")
    rows = []
    for d in DET:
        rows.append({"Detector": DET_LABEL[d], "Score": f"{mk[f'{d}_stat'].mean():.3f}",
                     "max p": f"{mk[f'{d}_p'].max():.3f}",
                     "Flagged": f"{int(mk[f'{d}_flag'].astype(str).str.lower().isin(['true', '1']).sum())}/{len(mk)}"})
    rows.append({"Detector": "Head TPR gap", "Score": f"{mk.tpr_gap.mean():.2f}", "max p": "", "Flagged": ""})
    rows.append({"Detector": "Head FPR gap", "Score": f"{mk.fpr_gap.mean():.2f}", "max p": "", "Flagged": ""})
    write_tabular("planted_marker", pd.DataFrame(rows), "lccc",
                  header=["", "Mean score", "Max $p$", "Flagged (BH)"], midrules=[len(DET)])


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
    t = pd.DataFrame([{"Backbone": BACKBONE_LABELS.get(r["backbone"], r["backbone"]), "Candidate": r["model_id"],
                       "Pooling": r["pooling"],
                       "Cosine mean": f2(r.get("cos_mean"), 3) if "cos_mean" in r else "n/a",
                       "Cosine min": f2(r.get("cos_min"), 3) if "cos_mean" in r else "n/a",
                       "Note": "" if "cos_mean" in r else r.get("error", "")[:60]} for r in rows])
    write_tabular("extraction_check", t, "lllccl")


def alignment():
    """Mark backbones whose rows share no similarity structure with the other
    backbones of the same dataset (run_alignment_check.py)."""
    a = load("alignment_check/alignment.csv")
    if a is None:
        return
    real = a[~a.b.str.contains("shuffled")]
    shuf = a[a.b.str.contains("shuffled")]
    rows, bad = [], []
    for ds, sub in real.groupby("dataset"):
        for bb in sorted(set(sub.a) | set(sub.b)):
            mx = sub[(sub.a == bb) | (sub.b == bb)].spearman.max()
            ctl = shuf[(shuf.dataset == ds) & (shuf.a == bb)].spearman.iloc[0]
            ok = bool(mx > 0.05)
            if not ok:
                bad.append(f"{ds}:{bb}")
                BACKBONE_LABELS[bb] = BACKBONE_LABELS.get(bb, bb) + " (rows misaligned)"
            rows.append({"Dataset": {"chexpert": "CheXpert", "mimic": "MIMIC-CXR"}[ds],
                         "Backbone": BACKBONE_LABELS.get(bb, bb), "Max Spearman": f2(mx, 3),
                         "Shuffled control": f2(ctl, 3), "Aligned": "yes" if ok else "no"})
            put(f"alignment_max_spearman_{ds}_{bb}", float(mx), "{:.3f}")
    put("alignment_misaligned_backbones", ", ".join(bad) if bad else "none", None)
    write_tabular("alignment_check", pd.DataFrame(rows), "llccc")


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
    # top-level manifest and repo copy
    lines = [f"revision results folder: {ROOT}", f"tables/figures built at commit {git_sha(False)}", ""]
    for m in sorted(ROOT.glob("*/MANIFEST.txt")):
        lines += [f"===== {m.parent.name} =====", m.read_text()]
    (ROOT / "MANIFEST.txt").write_text("\n".join(lines))
    if os.environ.get("SHORTKIT_NO_REPO_COPY"):
        print(f"numbers: {len(NUM)} keys (repo copy skipped)")
        return
    dest = REPO / "output" / "paper_tables" / "revision"
    for sub in ("tables", "figures"):
        (dest / sub).mkdir(parents=True, exist_ok=True)
        for f in (ROOT / sub).iterdir():
            shutil.copy2(f, dest / sub / f.name)
    for f in ("numbers.json", "MANIFEST.txt"):
        shutil.copy2(ROOT / f, dest / f)
    print(f"numbers: {len(NUM)} keys; copied to {dest}")


def main() -> int:
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
    finish()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
