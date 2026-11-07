"""HTML and PDF report for one conditional audit (an :class:`AuditResult`).

The report states what the audit measured, the six detector statistics against
their conditional permutation null, and the decision rule used in the paper:
the conditional probe decides at p < alpha, the five secondary detectors are
flagged at Benjamini-Hochberg q < alpha across those five and are descriptive.

    from shortcut_detect.audit import conditional_audit
    from shortcut_detect.audit.report import audit_report

    result = conditional_audit(E, A, Y, groups=patient_ids)
    audit_report(result, "audit.html", attribute="sex", task="pneumothorax")
    audit_report(result, "audit.pdf", attribute="sex", task="pneumothorax")

PDF output uses matplotlib only, so no extra dependency is needed.
"""

from __future__ import annotations

import base64
import html
import io
import os
from datetime import datetime

import numpy as np

from .conditional import PRIMARY, AuditResult, DetectorResult

LABELS = {
    "probe": "Conditional probe (primary)",
    "statistical": "Statistical",
    "geometric": "Geometric",
    "bias_direction": "Bias direction PCA",
    "hbac_nmi": "HBAC (NMI)",
    "frequency_topk": "Frequency (top-k probe)",
}


def _label(name: str) -> str:
    return LABELS.get(name, name.replace("_", " "))


def _ordered(result: AuditResult) -> list[DetectorResult]:
    rows = list(result.results.values())
    rows.sort(key=lambda r: (r.detector != PRIMARY, r.detector))
    return rows


def _fmt(x: float, nd: int = 3) -> str:
    if x is None or (isinstance(x, float) and np.isnan(x)):
        return "--"
    return f"{x:.{nd}f}"


def summary_text(result: AuditResult, attribute: str = "the attribute", task: str = "the task label") -> str:
    """One paragraph a reader can act on."""
    probe = result.results.get(PRIMARY)
    alpha = result.config.get("alpha", 0.05)
    k = result.config.get("n_permutations_probe") or result.config.get("n_permutations")
    if probe is None:
        return "The conditional probe did not run, so no decision is made."
    ci = f" (95% CI {_fmt(probe.ci_low)} to {_fmt(probe.ci_high)})" if not np.isnan(probe.ci_low) else ""
    base = (
        f"The conditional probe predicts {attribute} from the embedding and {task}; its out-of-fold AUROC is "
        f"{_fmt(probe.statistic)}{ci} against a permutation null of {_fmt(probe.null_mean)} "
        f"(SD {_fmt(probe.null_sd)}, {k} permutations of {attribute} within levels of {task}), p = {_fmt(probe.p_value)}. "
    )
    if probe.flag:
        verdict = (
            f"At alpha = {alpha} the embedding carries information about {attribute} beyond what {task} explains: "
            f"its shortcut potential for this task is positive. A head trained on it could use {attribute}; "
            f"whether a given head does is a separate test."
        )
    else:
        verdict = (
            f"At alpha = {alpha} there is no evidence that the embedding carries {attribute} beyond what {task} explains. "
            f"A negative audit is weaker evidence than a positive one: the probe is linear and misses signals that leave group means unchanged."
        )
    secondary = [r for r in _ordered(result) if r.detector != PRIMARY]
    n_sec = sum(r.flag for r in secondary)
    tail = f" Of the {len(secondary)} secondary detectors, {n_sec} were flagged after Benjamini-Hochberg across them; they are descriptive and do not change the decision."
    return base + verdict + tail


def _z(r: DetectorResult, value: float) -> float:
    if r.null_sd <= 0 or np.isnan(value):
        return float("nan")
    return (value - r.null_mean) / r.null_sd


def _figure_png(result: AuditResult, E=None, A=None, Y=None, attribute: str = "attribute") -> bytes:
    """Left: how far each statistic sits from its permutation null, in null standard deviations.
    Right (when the arrays are given): the embeddings projected on the attribute direction within task strata."""
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    rows = _ordered(result)
    has_data = E is not None and A is not None and Y is not None
    fig = plt.figure(figsize=(12.0, 4.6) if has_data else (7.4, 4.6))
    gs = fig.add_gridspec(1, 2 if has_data else 1, width_ratios=[1.15, 1] if has_data else [1], wspace=0.35)
    ax = fig.add_subplot(gs[0, 0])
    cap = 40.0
    ys = list(range(len(rows)))[::-1]
    ax.axvspan(-2, 2, color="#e5e7eb", zorder=0)
    ax.axvline(0, color="#9ca3af", lw=1, zorder=1)
    xmax = 4.0
    for y, r in zip(ys, rows, strict=True):
        z = _z(r, r.statistic)
        color = "#e53935" if r.flag else "#0066f0"
        if np.isnan(z):
            ax.text(0.2, y, "null SD is 0: see table", va="center", fontsize=8, color="#6b7280")
            continue
        zc = min(z, cap)
        if r.detector == PRIMARY and not np.isnan(r.ci_low):
            lo, hi = min(_z(r, r.ci_low), cap), min(_z(r, r.ci_high), cap)
            ax.plot([lo, hi], [y, y], color=color, lw=3, solid_capstyle="round", zorder=2)
        ax.plot(zc, y, "o", color=color, ms=11 if r.detector == PRIMARY else 8, zorder=3)
        if z > cap:
            ax.text(zc + 0.9, y, f"{z:.0f} SD", va="center", fontsize=8, color=color)
        xmax = max(xmax, zc + 6)
    ax.set_yticks(ys)
    ax.set_yticklabels([f"{_label(r.detector)}\np = {_fmt(r.p_value)}" for r in rows], fontsize=8.5)
    ax.set_xlim(-3.5, xmax)
    ax.set_xlabel("standard deviations from the permutation null (grey band: within 2 SD)", fontsize=9)
    ax.set_title("How far each statistic sits from its null", fontsize=11, weight="bold", loc="left")
    for sp in ("top", "right"):
        ax.spines[sp].set_visible(False)
    ax.tick_params(axis="y", length=0)

    if has_data:
        E = np.asarray(E, dtype=np.float32)
        A = np.asarray(A)
        Y = np.asarray(Y)
        levels = np.unique(A)
        ax2 = fig.add_subplot(gs[0, 1])
        proj = np.zeros(len(E), dtype=np.float32)
        direction = np.zeros(E.shape[1], dtype=np.float32)
        for yv in np.unique(Y):
            m = Y == yv
            Xs = E[m] - E[m].mean(axis=0)
            if len(levels) == 2 and (A[m] == levels[0]).any() and (A[m] == levels[1]).any():
                d = Xs[A[m] == levels[1]].mean(axis=0) - Xs[A[m] == levels[0]].mean(axis=0)
                direction += d * m.sum()
        if np.linalg.norm(direction) > 0:
            direction /= np.linalg.norm(direction)
        for yv in np.unique(Y):
            m = Y == yv
            proj[m] = (E[m] - E[m].mean(axis=0)) @ direction
        palette = ["#6b7280", "#0066f0", "#f0b000", "#30a020", "#8050e0", "#00b0b0"]
        bins = np.linspace(np.percentile(proj, 0.5), np.percentile(proj, 99.5), 40)
        for i, lv in enumerate(levels[:6]):
            ax2.hist(proj[A == lv], bins=bins, alpha=0.65, color=palette[i % len(palette)], label=f"{attribute} = {lv}",
                     density=True)
        ax2.legend(frameon=False, fontsize=8)
        ax2.set_yticks([])
        ax2.set_xlabel("projection on the attribute direction, centred within each task label", fontsize=9)
        ax2.set_title("Where the two groups sit in the embedding", fontsize=11, weight="bold", loc="left")
        for sp in ("top", "right", "left"):
            ax2.spines[sp].set_visible(False)
    fig.tight_layout()
    buf = io.BytesIO()
    fig.savefig(buf, format="png", dpi=160, bbox_inches="tight")
    plt.close(fig)
    return buf.getvalue()


def _hero(result: AuditResult) -> tuple[str, str, str, bool]:
    """Big number, its subtitle, the badge text and the flag."""
    probe = result.results.get(PRIMARY)
    if probe is None:
        return "--", "no primary test", "NO DECISION", False
    sub = f"conditional probe AUROC, 95% CI {_fmt(probe.ci_low)} to {_fmt(probe.ci_high)}, p = {_fmt(probe.p_value)}"
    badge = "SHORTCUT POTENTIAL FLAGGED" if probe.flag else "NO EVIDENCE OF SHORTCUT POTENTIAL"
    return _fmt(probe.statistic, 2), sub, badge, bool(probe.flag)


def _table_rows(result: AuditResult) -> list[list[str]]:
    out = []
    for r in _ordered(result):
        ci = f"{_fmt(r.ci_low)} to {_fmt(r.ci_high)}" if not np.isnan(r.ci_low) else "--"
        q = _fmt(r.p_bh) if r.detector != PRIMARY else "--"
        out.append([_label(r.detector), _fmt(r.statistic), ci, _fmt(r.null_mean), _fmt(r.null_sd),
                    _fmt(r.p_value), q, "yes" if r.flag else "no", str(r.n_permutations)])
    return out


HEADERS = ["Detector", "Statistic", "95% CI", "Null mean", "Null SD", "p", "q (BH)", "Flag", "K"]


def _meta(result: AuditResult, attribute: str, task: str) -> list[tuple[str, str]]:
    sizes = ", ".join(f"{g}: {n}" for g, n in result.group_sizes.items())
    return [
        ("Attribute", attribute),
        ("Task label", task),
        ("Samples", str(result.n)),
        ("Attribute levels", str(result.n_groups)),
        ("Cells (Y, A)", sizes),
        ("Fold strata", result.fold_strata),
        ("Attribute level", result.attribute_level),
        ("Permutations", f"{result.config.get('n_permutations')} (probe: {result.config.get('n_permutations_probe') or result.config.get('n_permutations')})"),
        ("Alpha", str(result.config.get("alpha", 0.05))),
        ("Seed", str(result.config.get("seed"))),
        ("Generated", datetime.now().strftime("%Y-%m-%d %H:%M")),
    ]


def to_html(result: AuditResult, path: str, title: str = "ShortKit-ML audit", attribute: str = "the attribute",
            task: str = "the task label", E=None, A=None, Y=None) -> str:
    """Self-contained HTML (the figure is embedded as base64 PNG). Pass E, A, Y to add the embedding view."""
    png = base64.b64encode(_figure_png(result, E, A, Y, attribute)).decode()
    big, sub, badge, flagged = _hero(result)
    meta = "".join(f"<tr><th>{html.escape(k)}</th><td>{html.escape(v)}</td></tr>" for k, v in _meta(result, attribute, task))
    head = "".join(f"<th>{h}</th>" for h in HEADERS)
    body = "".join(
        f"<tr class='{'flag' if row[7] == 'yes' else ''}'>{''.join(f'<td>{html.escape(c)}</td>' for c in row)}</tr>"
        for row in _table_rows(result)
    )
    doc = f"""<!doctype html><html><head><meta charset="utf-8"><title>{html.escape(title)}</title>
<style>body{{font-family:-apple-system,Helvetica,Arial,sans-serif;max-width:960px;margin:2em auto;color:#222;line-height:1.45}}
table{{border-collapse:collapse;margin:1em 0;font-size:14px}}th,td{{border-bottom:1px solid #ddd;padding:6px 10px;text-align:left}}
th{{background:#f4f4f4}}tr.flag td{{color:#b00020}}.meta th{{width:12em}}p.summary{{background:#f6f8fa;padding:1em;border-left:4px solid #0066f0}}
img{{max-width:100%}}small{{color:#666}}
.hero{{display:flex;align-items:center;gap:28px;padding:22px 26px;border-radius:14px;background:#f0f5ff;border:1px solid #d6e4ff;margin:1em 0}}
.hero .big{{font-size:64px;font-weight:700;color:#0066f0;line-height:1}}.hero .sub{{color:#374151;font-size:14px}}
.hero .badge{{margin-left:auto;padding:10px 16px;border-radius:999px;font-weight:700;font-size:13px;letter-spacing:.04em;color:#fff;background:#30a020}}
.hero.flag .badge{{background:#e53935}}.hero.flag .big{{color:#e53935}}</style></head><body>
<h1>{html.escape(title)}</h1>
<div class="hero{' flag' if flagged else ''}"><div class="big">{big}</div><div class="sub">{html.escape(sub)}</div><div class="badge">{badge}</div></div>
<p class="summary">{html.escape(summary_text(result, attribute, task))}</p>
<h2>Setting</h2><table class="meta">{meta}</table>
<h2>Detectors</h2><table><thead><tr>{head}</tr></thead><tbody>{body}</tbody></table>
<small>Statistic: larger means more information about the attribute in the embedding given the task label. Null: the statistic recomputed on
{result.config.get('n_permutations')} permutations of the attribute within task strata. The probe decides at p &lt; alpha; the secondary detectors are
flagged at Benjamini-Hochberg q &lt; alpha across the five and are descriptive.</small>
<h2>Signal against the null</h2><img src="data:image/png;base64,{png}" alt="statistics against null and embedding view">
<p><small>Generated by ShortKit-ML (shortcut_detect.audit.report).</small></p></body></html>"""
    os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        f.write(doc)
    return path


def to_pdf(result: AuditResult, path: str, title: str = "ShortKit-ML audit", attribute: str = "the attribute",
           task: str = "the task label", E=None, A=None, Y=None) -> str:
    """Two-page PDF through matplotlib: verdict, summary and table, then the figure. Pass E, A, Y for the embedding view."""
    import textwrap

    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.backends.backend_pdf import PdfPages

    os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
    with PdfPages(path) as pdf:
        fig = plt.figure(figsize=(8.27, 11.69))
        fig.text(0.07, 0.95, title, fontsize=18, weight="bold")
        big, sub, badge, flagged = _hero(result)
        color = "#e53935" if flagged else "#0066f0"
        fig.patches.append(plt.Rectangle((0.07, 0.845), 0.86, 0.085, transform=fig.transFigure, color="#f0f5ff", zorder=0))
        fig.text(0.09, 0.862, big, fontsize=40, weight="bold", color=color)
        fig.text(0.36, 0.893, badge, fontsize=10.5, weight="bold", color="#ffffff",
                 bbox={"boxstyle": "round,pad=0.45", "facecolor": color if flagged else "#30a020", "edgecolor": "none"})
        fig.text(0.36, 0.862, sub, fontsize=9, color="#374151")
        y = 0.825
        for k, v in _meta(result, attribute, task):
            fig.text(0.07, y, f"{k}:", fontsize=9, weight="bold")
            fig.text(0.27, y, v, fontsize=9)
            y -= 0.018
        y -= 0.01
        for line in textwrap.wrap(summary_text(result, attribute, task), 105):
            fig.text(0.07, y, line, fontsize=9.5)
            y -= 0.018
        ax = fig.add_axes([0.07, y - 0.30, 0.86, 0.26])
        ax.axis("off")
        widths = [0.26, 0.09, 0.15, 0.1, 0.09, 0.08, 0.08, 0.06, 0.06]
        tbl = ax.table(cellText=_table_rows(result), colLabels=HEADERS, loc="upper center", cellLoc="left",
                       colLoc="left", colWidths=widths)
        tbl.auto_set_font_size(False)
        tbl.set_fontsize(7.5)
        tbl.scale(1, 1.35)
        for (row, _col), cell in tbl.get_celld().items():
            cell.set_edgecolor("#dddddd")
            if row == 0:
                cell.set_facecolor("#f4f4f4")
                cell.set_text_props(weight="bold")
        fig.text(0.07, y - 0.33, textwrap.fill(
            "Statistic: larger means more information about the attribute in the embedding given the task label. "
            "Null: the statistic recomputed on permutations of the attribute within task strata. The probe decides at p < alpha; "
            "the secondary detectors are flagged at Benjamini-Hochberg q < alpha across the five and are descriptive.", 110),
            fontsize=7.5, color="#555555", va="top")
        pdf.savefig(fig)
        plt.close(fig)
        png = _figure_png(result, E, A, Y, attribute)
        img = plt.imread(io.BytesIO(png), format="png")
        fig2 = plt.figure(figsize=(8.27, 11.69))
        h, w = img.shape[:2]
        box_w = 0.9
        box_h = box_w * (h / w) * (8.27 / 11.69)
        ax2 = fig2.add_axes([0.05, 0.92 - box_h, box_w, box_h])
        ax2.imshow(img)
        ax2.axis("off")
        fig2.text(0.07, 0.95, "Signal against the conditional permutation null", fontsize=12, weight="bold")
        fig2.text(0.07, 0.92 - box_h - 0.03, textwrap.fill(
            "Left: each detector's statistic as standard deviations from its permutation null (the attribute shuffled within "
            "task strata); the grey band is within 2 SD, the bar on the probe is its bootstrap 95% interval, red means flagged. "
            "Right, when the arrays were given: every embedding projected on the direction that separates the attribute groups "
            "within each task label; the further apart the two histograms, the stronger the signal.", 110),
            fontsize=8.5, va="top")
        pdf.savefig(fig2)
        plt.close(fig2)
    return path


def audit_report(result: AuditResult, path: str, **kwargs) -> str:
    """Write ``path`` as HTML or PDF depending on its extension.

    Keyword arguments: ``title``, ``attribute``, ``task``, and optionally the arrays ``E``, ``A``, ``Y`` that were
    audited, which add the embedding view to the figure.
    """
    ext = os.path.splitext(path)[1].lower()
    if ext == ".pdf":
        return to_pdf(result, path, **kwargs)
    if ext in (".html", ".htm"):
        return to_html(result, path, **kwargs)
    raise ValueError(f"unsupported report extension {ext!r}; use .html or .pdf")
