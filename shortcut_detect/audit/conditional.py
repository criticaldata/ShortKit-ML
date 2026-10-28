"""Attribute-aware shortcut-potential audit with a conditional permutation null.

The estimand is shortcut potential: for a task label Y and a sensitive attribute
A, does the embedding E carry information about A beyond what Y explains,
I(E; A | Y) > 0.

Every detector returns a continuous statistic s(E, A, Y), larger meaning more A
in E. It is calibrated by a conditional permutation null: A is shuffled within
each Y stratum, the statistic is recomputed K times and
p = (1 + #{s_null >= s_obs}) / (K + 1). Under the null each detector's false
positive rate is alpha by construction. Detectors in one cell are flagged when
their Benjamini-Hochberg adjusted p-value is below alpha.

Detectors
---------
probe (primary)
    L2 logistic regression predicting A from standardised [E, Y]; out-of-fold
    AUROC (macro one-vs-rest for more than two groups), 5-fold CV stratified on
    (A, Y), falling back to A when a joint cell is smaller than the fold count.
statistical
    Fraction of dimensions with a BH-significant A difference within Y strata
    (Kruskal-Wallis H summed over strata; Mann-Whitney for two groups).
geometric
    Mean pairwise distance between A-group centroids over the pooled
    within-group RMS spread, averaged over Y strata (weighted by stratum size).
bias_direction
    Gap between A-group means projected on the first principal direction of the
    centred group means, over the pooled within-group SD of that projection,
    averaged over Y strata.
hbac_nmi
    Normalised mutual information between k-means clusters of E (fitted inside
    each Y stratum, never seeing A) and A, averaged over Y strata.
frequency_topk
    The probe restricted to the top-k dimensions (k = 5 percent of d) ranked on
    the training folds by partial correlation with A given Y.

Bootstrap confidence intervals (bias-corrected percentile) resample images. For the two probes they
resample the out-of-fold predictions (the fitted models are not refitted).
For geometric, bias_direction and hbac_nmi they reweight samples. The
statistical detector has no CI because ranks would have to be recomputed for
every resample.
"""

from __future__ import annotations

import math
import time
from dataclasses import asdict, dataclass, field
from typing import Any

import numpy as np
import torch
from scipy.stats import chi2 as _chi2
from scipy.stats import rankdata

DETECTORS: tuple[str, ...] = (
    "probe",
    "statistical",
    "geometric",
    "bias_direction",
    "hbac_nmi",
    "frequency_topk",
)
PRIMARY = "probe"


@dataclass
class AuditConfig:
    """Settings fixed by the revision method decisions (sections 2 to 4)."""

    n_permutations: int = 200
    n_permutations_probe: int | None = None  # defaults to n_permutations
    alpha: float = 0.05
    n_folds: int = 5
    C: float = 1.0
    top_k_fraction: float = 0.05
    n_clusters: int = 8
    n_bootstrap: int = 2000
    max_iter: int = 100
    seed: int = 0
    permutation: str = "within_y"  # "within_y" (the audit) or "global" (for comparison only)
    include_y: bool = True  # probe features [E, Y]; False gives the marginal probe E -> A
    detectors: tuple[str, ...] = DETECTORS
    chunk_problems: int = 1200  # logistic problems fitted together


@dataclass
class DetectorResult:
    detector: str
    statistic: float
    null_mean: float
    null_sd: float
    z: float
    p_value: float
    p_bh: float = float("nan")
    flag: bool = False
    ci_low: float = float("nan")
    ci_high: float = float("nan")
    n_permutations: int = 0
    seconds: float = 0.0
    note: str = ""


@dataclass
class AuditResult:
    n: int
    n_groups: int
    groups: list[str]
    group_sizes: dict[str, int]
    fold_strata: str
    config: dict[str, Any]
    results: dict[str, DetectorResult] = field(default_factory=dict)

    def rows(self) -> list[dict[str, Any]]:
        out = []
        for r in self.results.values():
            d = asdict(r)
            d.update(n=self.n, n_groups=self.n_groups, fold_strata=self.fold_strata)
            out.append(d)
        return out

    @property
    def n_flagged(self) -> int:
        return int(sum(r.flag for r in self.results.values()))


# ---------------------------------------------------------------------------
# Small helpers
# ---------------------------------------------------------------------------


def benjamini_hochberg(p: np.ndarray) -> np.ndarray:
    """BH adjusted p-values along the last axis."""
    p = np.asarray(p, dtype=float)
    m = p.shape[-1]
    order = np.argsort(p, axis=-1)
    ranked = np.take_along_axis(p, order, axis=-1) * m / np.arange(1, m + 1)
    ranked = np.minimum.accumulate(ranked[..., ::-1], axis=-1)[..., ::-1]
    out = np.empty_like(ranked)
    np.put_along_axis(out, order, np.minimum(ranked, 1.0), axis=-1)
    return out


def _encode(a: np.ndarray) -> tuple[np.ndarray, list[str]]:
    names, codes = np.unique(np.asarray(a).astype(str), return_inverse=True)
    return codes.astype(np.int64), [str(x) for x in names]


def conditional_permutations(
    a: np.ndarray,
    y: np.ndarray,
    k: int,
    rng: np.random.Generator,
    mode: str = "within_y",
    groups: np.ndarray | None = None,
) -> np.ndarray:
    """Return a (k + 1, n) array whose row 0 is ``a`` and rows 1..k are ``a``
    shuffled within each Y stratum (or globally when ``mode='global'``).

    With ``groups`` (patient ids) the shuffle is at patient level: each
    patient's attribute (its most common value) is reassigned among patients
    with the same Y pattern (all negative, all positive, or mixed). Images of
    one patient are correlated, and an image-level shuffle then gives
    anti-conservative p-values.
    """
    n = len(a)
    out = np.empty((k + 1, n), dtype=a.dtype)
    out[0] = a
    if groups is not None:
        _, ginv = np.unique(np.asarray(groups).astype(str), return_inverse=True)
        n_g = ginv.max() + 1
        pa = np.empty(n_g, dtype=a.dtype)
        pstrat = np.empty(n_g, dtype=np.int64)
        y_hi = int(y.max())
        for g in range(n_g):
            idx = ginv == g
            vals, cnt = np.unique(a[idx], return_counts=True)
            pa[g] = vals[np.argmax(cnt)]
            ys = np.unique(y[idx])
            pstrat[g] = int(ys[0]) if len(ys) == 1 else y_hi + 1
        strata = [np.arange(n_g)] if mode == "global" else [np.flatnonzero(pstrat == s) for s in np.unique(pstrat)]
        for i in range(1, k + 1):
            prow = pa.copy()
            for idx in strata:
                prow[idx] = pa[idx][rng.permutation(len(idx))]
            out[i] = prow[ginv]
        return out
    strata = [np.arange(n)] if mode == "global" else [np.flatnonzero(y == v) for v in np.unique(y)]
    for i in range(1, k + 1):
        row = a.copy()
        for idx in strata:
            row[idx] = a[idx][rng.permutation(len(idx))]
        out[i] = row
    return out


def _fold_ids(
    a_rows: np.ndarray,
    y: np.ndarray,
    n_folds: int,
    rng: np.random.Generator,
    groups: np.ndarray | None = None,
) -> tuple[np.ndarray, str]:
    """Stratified fold ids for every row of ``a_rows`` (P, n).

    Stratifies on (A, Y). A permutation within Y keeps the joint (A, Y) counts,
    so the choice of strata is the same for every row of ``a_rows``. With
    ``groups`` (patient ids) the folds are patient-disjoint
    (``StratifiedGroupKFold``), recomputed for every permuted attribute.
    """
    joint = a_rows[0] * (int(y.max()) + 1) + y
    _, counts = np.unique(joint, return_counts=True)
    use_joint = counts.min() >= n_folds
    folds = np.empty(a_rows.shape, dtype=np.int64)
    ymult = int(y.max()) + 1
    if groups is not None:
        from sklearn.model_selection import StratifiedGroupKFold

        for i, a in enumerate(a_rows):
            key = a * ymult + y if use_joint else a
            sgk = StratifiedGroupKFold(
                n_splits=n_folds, shuffle=True, random_state=int(rng.integers(1 << 30))
            )
            for f, (_, te) in enumerate(sgk.split(np.zeros(len(a)), key, groups)):
                folds[i, te] = f
        return folds, ("A,Y" if use_joint else "A") + " grouped"
    for i, a in enumerate(a_rows):
        key = a * ymult + y if use_joint else a
        order = np.lexsort((rng.random(len(a)), key))
        sorted_key = key[order]
        starts = np.r_[0, np.flatnonzero(np.diff(sorted_key)) + 1]
        within = np.arange(len(a)) - np.repeat(starts, np.diff(np.r_[starts, len(a)]))
        offset = rng.integers(n_folds)
        folds[i, order] = (within + offset) % n_folds
    return folds, ("A,Y" if use_joint else "A")


def _auroc_columns(scores: np.ndarray, target: np.ndarray) -> np.ndarray:
    """AUROC of each column of ``scores`` (n, J) against binary ``target`` (n, J)."""
    ranks = rankdata(scores, axis=0)
    n1 = target.sum(axis=0)
    n0 = target.shape[0] - n1
    s1 = (ranks * target).sum(axis=0)
    with np.errstate(invalid="ignore", divide="ignore"):
        return (s1 - n1 * (n1 + 1) / 2.0) / (n1 * n0)


def _auroc_rows(scores: np.ndarray, target: np.ndarray) -> np.ndarray:
    """AUROC of each row of ``scores`` (B, n) against binary ``target`` (B, n)."""
    return _auroc_columns(scores.T, target.T)


# ---------------------------------------------------------------------------
# Batched L2 logistic regression (one shared design, many problems)
# ---------------------------------------------------------------------------


def _fit_logreg(
    Z: torch.Tensor, T: torch.Tensor, M: torch.Tensor, C: float, max_iter: int
) -> tuple[torch.Tensor, torch.Tensor]:
    """Fit J independent L2 logistic regressions that share a design.

    Z is (n, p) or (J, n, p); T and M are (n, J) targets and training-weight
    masks. Each problem minimises sum_i M_ij * logloss + ||w_j||^2 / (2 C),
    the objective of scikit-learn's ``LogisticRegression(C=C)`` restricted to
    the rows with M_ij = 1. The intercept is not penalised. The problems are
    separable, so minimising their mean finds every problem's optimum.
    """
    batched = Z.dim() == 3
    p = Z.shape[-1]
    J = T.shape[1]
    W = torch.zeros((J, p) if batched else (p, J), dtype=Z.dtype, requires_grad=True)
    b = torch.zeros(J, dtype=Z.dtype, requires_grad=True)
    opt = torch.optim.LBFGS(
        [W, b],
        lr=1.0,
        max_iter=max_iter,
        history_size=20,
        tolerance_grad=1e-6,
        tolerance_change=1e-10,
        line_search_fn="strong_wolfe",
    )
    n_train = M.sum(0).clamp(min=1.0)

    def logits() -> torch.Tensor:
        if batched:
            return torch.einsum("jnp,jp->nj", Z, W) + b
        return Z @ W + b

    def closure():
        opt.zero_grad()
        f = logits()
        loss = torch.nn.functional.binary_cross_entropy_with_logits(f, T, reduction="none")
        pen = 0.5 / C * (W * W).sum(dim=-1 if batched else 0)
        obj = ((loss * M).sum(0) + pen) / n_train
        total = obj.mean()
        total.backward()
        return total

    opt.step(closure)
    with torch.no_grad():
        return logits().detach(), W.detach()


def _oof_scores(
    Z: np.ndarray | list[np.ndarray],
    targets: np.ndarray,
    folds: np.ndarray,
    n_folds: int,
    C: float,
    max_iter: int,
    chunk: int,
) -> np.ndarray:
    """Out-of-fold logits for each column of ``targets`` (n, J).

    ``folds`` is (n, J). ``Z`` is a shared (n, p) design or a list of J designs
    of identical shape (used by the top-k probe).
    """
    n, J = targets.shape
    out = np.zeros((n, J), dtype=np.float32)
    shared = isinstance(Z, np.ndarray)
    Zt = torch.from_numpy(np.ascontiguousarray(Z, dtype=np.float32)) if shared else None
    # Each column is fitted n_folds times; chunk over columns.
    cols_per_chunk = max(1, chunk // n_folds)
    for c0 in range(0, J, cols_per_chunk):
        cols = np.arange(c0, min(J, c0 + cols_per_chunk))
        t = np.repeat(targets[:, cols], n_folds, axis=1)
        f = np.repeat(folds[:, cols], n_folds, axis=1)
        fid = np.tile(np.arange(n_folds), len(cols))
        train = (f != fid[None, :]).astype(np.float32)
        T = torch.from_numpy(t.astype(np.float32))
        M = torch.from_numpy(train)
        if shared:
            design = Zt
        else:
            design = torch.from_numpy(
                np.ascontiguousarray(np.repeat(np.stack([Z[c] for c in cols]), n_folds, axis=0))
            )
        lg, _ = _fit_logreg(design, T, M, C, max_iter)
        lg = lg.numpy()
        test = 1.0 - train
        # keep each sample's logit from the model that did not see it
        for j, col in enumerate(cols):
            block = slice(j * n_folds, (j + 1) * n_folds)
            out[:, col] = (lg[:, block] * test[:, block]).sum(1)
    return out


def _probe_targets(a_rows: np.ndarray, n_groups: int) -> tuple[np.ndarray, int]:
    """Binary targets (n, P * G') for every permutation row; G' = 1 for two groups."""
    P, n = a_rows.shape
    if n_groups == 2:
        return (a_rows == 1).T.astype(np.float32), 1
    t = np.stack([(a_rows == g) for g in range(n_groups)], axis=1)  # (P, G, n)
    return t.reshape(P * n_groups, n).T.astype(np.float32), n_groups


def _macro_auc(scores: np.ndarray, targets: np.ndarray, per: int) -> np.ndarray:
    auc = _auroc_columns(scores, targets)
    return auc.reshape(-1, per).mean(1)


def bc_interval(boot: np.ndarray, obs: float, level: float = 0.95) -> tuple[float, float]:
    """Bias-corrected percentile bootstrap interval.

    Resampling with replacement duplicates rows, which inflates distance-type
    statistics; the bias correction re-centres the percentiles on ``obs``.
    """
    from scipy.stats import norm

    boot = np.asarray(boot, dtype=float)
    boot = boot[np.isfinite(boot)]
    if boot.size == 0 or not np.isfinite(obs):
        return float("nan"), float("nan")
    frac = (np.sum(boot < obs) + 0.5 * np.sum(boot == obs)) / boot.size
    frac = min(max(frac, 0.5 / boot.size), 1 - 0.5 / boot.size)
    z0 = norm.ppf(frac)
    zq = norm.ppf(0.5 + level / 2)
    lo, hi = norm.cdf(2 * z0 - zq), norm.cdf(2 * z0 + zq)
    return float(np.percentile(boot, 100 * lo)), float(np.percentile(boot, 100 * hi))


def _resample_index(rng, n: int, groups_idx: list[np.ndarray] | None) -> np.ndarray:
    """Row indices of one bootstrap resample (images, or whole patients)."""
    if groups_idx is None:
        return rng.integers(0, n, size=n)
    pick = rng.integers(0, len(groups_idx), size=len(groups_idx))
    return np.concatenate([groups_idx[i] for i in pick])


def _bootstrap_auc(
    scores: np.ndarray, targets: np.ndarray, B: int, rng, groups_idx=None
) -> np.ndarray:
    """Bootstrap macro AUROC by resampling images (or patients) of
    out-of-fold predictions."""
    n, G = targets.shape
    if groups_idx is not None:
        out = np.empty(B)
        for b in range(B):
            idx = _resample_index(rng, n, groups_idx)
            out[b] = np.nanmean(
                [_auroc_columns(scores[idx, g : g + 1], targets[idx, g : g + 1])[0] for g in range(G)]
            )
        return out
    out = np.empty(B)
    step = max(1, int(2e7 // max(n * G, 1)))
    for s in range(0, B, step):
        idx = rng.integers(0, n, size=(min(step, B - s), n))
        aucs = []
        for g in range(G):
            aucs.append(_auroc_rows(scores[idx, g], targets[idx, g]))
        out[s : s + idx.shape[0]] = np.nanmean(np.stack(aucs), axis=0)
    return out


# ---------------------------------------------------------------------------
# Detectors on permutation rows. Each returns statistics of shape (P,).
# ---------------------------------------------------------------------------


def _stat_probe(ctx, a_rows, folds, cfg, features=None):
    tg, per = _probe_targets(a_rows, ctx["G"])
    fcols = np.repeat(folds, per, axis=0).T  # (n, P*per)
    Z = ctx["Z"] if features is None else features
    sc = _oof_scores(Z, tg, fcols, cfg.n_folds, cfg.C, cfg.max_iter, cfg.chunk_problems)
    return _macro_auc(sc, tg, per), sc[:, :per], tg[:, :per]


def _stat_frequency(ctx, a_rows, folds, cfg):
    """Probe on the top-k dimensions ranked on training folds by partial
    correlation with A given Y."""
    X = ctx["Xc"]  # features centred within Y strata
    n, d = X.shape
    k = max(1, int(math.ceil(cfg.top_k_fraction * d)))
    tg, per = _probe_targets(a_rows, ctx["G"])
    fcols = np.repeat(folds, per, axis=0).T
    J = tg.shape[1]
    ystd = ctx["ystd"][:, None]
    out = np.zeros((n, J), dtype=np.float32)
    Xt = torch.from_numpy(X)
    xnorm = None
    chunk_cols = max(1, cfg.chunk_problems // (cfg.n_folds * 8))
    for c0 in range(0, J, chunk_cols):
        cols = np.arange(c0, min(J, c0 + chunk_cols))
        for f in range(cfg.n_folds):
            train = (fcols[:, cols] != f).astype(np.float32)  # (n, c)
            t = tg[:, cols]
            # centre the target within Y strata on training rows
            tc = np.zeros_like(t)
            for idx in ctx["strata"]:
                m = train[idx]
                mu = (t[idx] * m).sum(0) / np.maximum(m.sum(0), 1)
                tc[idx] = (t[idx] - mu) * m
            num = (Xt.T @ torch.from_numpy(tc)).numpy()  # (d, c)
            if xnorm is None:
                xnorm = np.sqrt((X * X).sum(0))[:, None]
            corr = np.abs(num) / np.maximum(xnorm, 1e-12)
            top = np.argpartition(-corr, k - 1, axis=0)[:k]  # (k, c)
            designs = []
            for j in range(len(cols)):
                sel = ctx["Zx"][:, top[:, j]]
                designs.append(np.concatenate([sel, ystd], axis=1))
            design = torch.from_numpy(np.ascontiguousarray(np.stack(designs), dtype=np.float32))
            lg, _ = _fit_logreg(
                design,
                torch.from_numpy(t.astype(np.float32)),
                torch.from_numpy(train),
                cfg.C,
                cfg.max_iter,
            )
            lg = lg.numpy()
            out[:, cols] += lg * (1.0 - train)
    return _macro_auc(out, tg, per), out[:, :per], tg[:, :per]


def _onehot(a_rows: np.ndarray, G: int) -> np.ndarray:
    return (a_rows[:, :, None] == np.arange(G)[None, None, :]).astype(np.float32)


def _stat_statistical(ctx, a_rows, cfg):
    G = ctx["G"]
    P = a_rows.shape[0]
    d = ctx["d"]
    H = np.zeros((P, d))
    df = 0
    for idx, R in zip(ctx["strata"], ctx["ranks"], strict=True):
        Ns = len(idx)
        oh = _onehot(a_rows[:, idx], G)  # (P, Ns, G)
        # per-row counts: patient-level permutations change group sizes within a stratum
        ng = oh.sum(1)  # (P, G)
        present = (ng > 0).all(0)
        if present.sum() < 2:
            continue
        rs = np.einsum("png,nd->pgd", oh, R)  # rank sums
        h = (rs[:, present, :] ** 2 / ng[:, present][:, :, None]).sum(1)
        H += 12.0 / (Ns * (Ns + 1)) * h - 3.0 * (Ns + 1)
        df += int(present.sum()) - 1
    if df == 0:
        return np.zeros(P)
    p = _chi2.sf(H, df)
    q = benjamini_hochberg(p)
    return (q < cfg.alpha).mean(1)


def _group_moments(X, oh, w):
    """Weighted group counts, sums and total squared norm for one stratum.

    oh is (P, n, G) or (1, n, G); w is (P, n) or None."""
    ohw = oh if w is None else oh * w[:, :, None]
    cnt = ohw.sum(1)  # (P, G)
    sums = np.einsum("png,nd->pgd", ohw, X)  # (P, G, d)
    return cnt, sums


def _stat_geometric(ctx, a_rows, cfg, w=None, oh_fixed=None):
    G = ctx["G"]
    P = a_rows.shape[0] if w is None else w.shape[0]
    total = np.zeros(P)
    weight = 0.0
    for idx in ctx["strata"]:
        X = ctx["X"][idx]
        oh = _onehot(a_rows[:, idx], G) if oh_fixed is None else oh_fixed[:, idx]
        ws = None if w is None else w[:, idx]
        cnt, sums = _group_moments(X, oh, ws)
        present = (cnt > 0).all(0) if w is None else (oh[0].sum(0) > 0)
        if present.sum() < 2:
            continue
        cnt = np.maximum(cnt[:, present], 1e-12)
        mu = sums[:, present] / cnt[:, :, None]
        sq = (X * X).sum(1)
        tot_sq = sq.sum() if ws is None else (ws * sq[None, :]).sum(1)
        within = tot_sq - (cnt * (mu * mu).sum(2)).sum(1)
        n_eff = cnt.sum(1)
        spread = np.sqrt(np.maximum(within / np.maximum(n_eff - present.sum(), 1.0), 1e-12))
        gs = mu.shape[1]
        dist = []
        for i in range(gs):
            for j in range(i + 1, gs):
                dist.append(np.sqrt(((mu[:, i] - mu[:, j]) ** 2).sum(1)))
        total += len(idx) * np.mean(dist, axis=0) / spread
        weight += len(idx)
    return total / max(weight, 1.0)


def _stat_bias_direction(ctx, a_rows, cfg, w=None, oh_fixed=None):
    G = ctx["G"]
    P = a_rows.shape[0] if w is None else w.shape[0]
    total = np.zeros(P)
    weight = 0.0
    for idx in ctx["strata"]:
        X = ctx["X"][idx]
        oh = _onehot(a_rows[:, idx], G) if oh_fixed is None else oh_fixed[:, idx]
        ws = None if w is None else w[:, idx]
        cnt, sums = _group_moments(X, oh, ws)
        present = (cnt > 0).all(0) if w is None else (oh[0].sum(0) > 0)
        if present.sum() < 2:
            continue
        cnt = np.maximum(cnt[:, present], 1e-12)
        mu = sums[:, present] / cnt[:, :, None]  # (P, g, d)
        centred = mu - mu.mean(1, keepdims=True)
        if mu.shape[1] == 2:
            v = centred[:, 1] - centred[:, 0]
        else:
            _, _, vh = np.linalg.svd(centred, full_matrices=False)
            v = vh[:, 0, :]
        v = v / np.maximum(np.linalg.norm(v, axis=1, keepdims=True), 1e-12)
        proj = X @ v.T  # (n, P)
        ohp = oh[:, :, present] if oh.shape[0] == P else np.broadcast_to(oh[:, :, present], (P,) + oh.shape[1:2] + (int(present.sum()),))
        ohw = ohp if ws is None else ohp * ws[:, :, None]
        pm = np.einsum("png,np->pg", ohw, proj) / cnt  # projected group means
        ss = np.einsum("png,np->pg", ohw, proj * proj) - cnt * pm * pm
        sd = np.sqrt(np.maximum(ss.sum(1) / np.maximum(cnt.sum(1) - present.sum(), 1.0), 1e-12))
        total += len(idx) * (pm.max(1) - pm.min(1)) / sd
        weight += len(idx)
    return total / max(weight, 1.0)


def _nmi_from_tables(T: np.ndarray) -> np.ndarray:
    """Arithmetic-mean NMI of contingency tables T (P, C, G)."""
    n = T.sum((1, 2))
    pij = T / np.maximum(n[:, None, None], 1e-12)
    pi = pij.sum(2, keepdims=True)
    pj = pij.sum(1, keepdims=True)
    with np.errstate(divide="ignore", invalid="ignore"):
        mi = np.nansum(np.where(pij > 0, pij * np.log(pij / (pi * pj)), 0.0), axis=(1, 2))
        hi = -np.nansum(np.where(pi > 0, pi * np.log(pi), 0.0), axis=(1, 2))
        hj = -np.nansum(np.where(pj > 0, pj * np.log(pj), 0.0), axis=(1, 2))
        denom = 0.5 * (hi + hj)
        return np.where(denom > 0, mi / denom, 0.0)


def _stat_hbac_nmi(ctx, a_rows, cfg, w=None, oh_fixed=None):
    G = ctx["G"]
    P = a_rows.shape[0] if w is None else w.shape[0]
    total = np.zeros(P)
    weight = 0.0
    for idx, cl in zip(ctx["strata"], ctx["clusters"], strict=True):
        oh = _onehot(a_rows[:, idx], G) if oh_fixed is None else oh_fixed[:, idx]
        if (oh[0].sum(0) > 0).sum() < 2:
            continue
        cl_oh = (cl[:, None] == np.arange(cl.max() + 1)[None, :]).astype(np.float32)
        ohw = oh if w is None else oh * w[:, idx][:, :, None]
        T = np.einsum("nc,png->pcg", cl_oh, ohw)
        total += len(idx) * _nmi_from_tables(T)
        weight += len(idx)
    return total / max(weight, 1.0)


# ---------------------------------------------------------------------------
# Driver
# ---------------------------------------------------------------------------


def _prepare(E: np.ndarray, Y: np.ndarray, cfg: AuditConfig, needs: set[str], rng) -> dict:
    X = np.asarray(E, dtype=np.float32)
    n, d = X.shape
    mu, sd = X.mean(0), X.std(0)
    sd[sd < 1e-8] = 1.0
    Zx = ((X - mu) / sd).astype(np.float32)
    yv = Y.astype(np.float32)
    ystd = ((yv - yv.mean()) / (yv.std() if yv.std() > 0 else 1.0)).astype(np.float32)
    ctx: dict[str, Any] = {
        "X": X,
        "Zx": Zx,
        "ystd": ystd,
        "Z": np.concatenate([Zx, ystd[:, None]], axis=1) if cfg.include_y else Zx,
        "d": d,
        "strata": [np.flatnonzero(Y == v) for v in np.unique(Y)],
    }
    if "statistical" in needs:
        ctx["ranks"] = [rankdata(X[idx], axis=0).astype(np.float32) for idx in ctx["strata"]]
    if "frequency_topk" in needs:
        Xc = Zx.copy()
        for idx in ctx["strata"]:
            Xc[idx] -= Xc[idx].mean(0)
        ctx["Xc"] = Xc
    if "hbac_nmi" in needs:
        from sklearn.cluster import KMeans

        clusters = []
        for idx in ctx["strata"]:
            k = int(min(cfg.n_clusters, max(2, len(idx) // 20)))
            km = KMeans(n_clusters=k, n_init=3, random_state=int(rng.integers(1 << 30)))
            clusters.append(km.fit_predict(Zx[idx]).astype(np.int64))
        ctx["clusters"] = clusters
    return ctx


def conditional_audit(
    E: np.ndarray,
    A: np.ndarray,
    Y: np.ndarray,
    config: AuditConfig | None = None,
    groups: np.ndarray | None = None,
) -> AuditResult:
    """Run the attribute-aware audit on one (embedding, attribute, task) cell.

    ``groups`` (e.g. patient ids) makes every cross-validation split
    patient-disjoint, every bootstrap resample whole patients, and the
    permutation patient-level within Y-pattern strata (see
    ``conditional_permutations``).
    """
    cfg = config or AuditConfig()
    rng = np.random.default_rng(cfg.seed)
    E = np.asarray(E)
    Y = np.asarray(Y).astype(np.int64)
    a, names = _encode(A)
    G = len(names)
    if G < 2:
        raise ValueError("the attribute needs at least two groups")
    if E.shape[0] != len(a) or len(a) != len(Y):
        raise ValueError("E, A and Y must have the same number of rows")

    needs = set(cfg.detectors)
    ctx = _prepare(E, Y, cfg, needs, rng)
    ctx["G"] = G
    K = cfg.n_permutations
    Kp = cfg.n_permutations_probe or K
    a_rows = conditional_permutations(a, Y, max(K, Kp), rng, cfg.permutation, groups)
    groups_idx = None
    if groups is not None:
        groups = np.asarray(groups)
        if len(groups) != len(a):
            raise ValueError("groups must have one entry per row")
        _, ginv = np.unique(groups.astype(str), return_inverse=True)
        groups_idx = [np.flatnonzero(ginv == g) for g in range(ginv.max() + 1)]
    folds, fold_strata = _fold_ids(a_rows, Y, cfg.n_folds, rng, groups)
    sizes = {}
    for yv in np.unique(Y):
        for g, nm in enumerate(names):
            sizes[f"Y={yv},A={nm}"] = int(((Y == yv) & (a == g)).sum())

    res = AuditResult(
        n=len(a),
        n_groups=G,
        groups=names,
        group_sizes=sizes,
        fold_strata=fold_strata,
        config=asdict(cfg),
    )
    B = cfg.n_bootstrap
    for det in cfg.detectors:
        t0 = time.time()
        k_used = Kp if det in {"probe", "frequency_topk"} else K
        rows = a_rows[: k_used + 1]
        ci = (float("nan"), float("nan"))
        if det == "probe":
            stats, sc, tg = _stat_probe(ctx, rows, folds[: k_used + 1], cfg)
            if B:
                ci = bc_interval(_bootstrap_auc(sc, tg, B, rng, groups_idx), float(stats[0]))
        elif det == "frequency_topk":
            stats, sc, tg = _stat_frequency(ctx, rows, folds[: k_used + 1], cfg)
            if B:
                ci = bc_interval(_bootstrap_auc(sc, tg, B, rng, groups_idx), float(stats[0]))
        elif det == "statistical":
            stats = _stat_statistical(ctx, rows, cfg)
        else:
            fn = {
                "geometric": _stat_geometric,
                "bias_direction": _stat_bias_direction,
                "hbac_nmi": _stat_hbac_nmi,
            }[det]
            stats = fn(ctx, rows, cfg)
            if B:
                oh0 = _onehot(a[None, :], G)
                vals = []
                step = max(1, int(4e6 // max(len(a) * max(G, 2), 1)))
                for s in range(0, B, step):
                    m = min(step, B - s)
                    if groups_idx is None:
                        w = rng.multinomial(len(a), np.full(len(a), 1.0 / len(a)), size=m)
                    else:
                        gw = rng.multinomial(
                            len(groups_idx), np.full(len(groups_idx), 1.0 / len(groups_idx)), size=m
                        )
                        w = gw[:, ginv]
                    w = w.astype(np.float32)
                    vals.append(fn(ctx, rows[:1], cfg, w=w, oh_fixed=oh0))
                ci = bc_interval(np.concatenate(vals), float(stats[0]))
        obs, null = float(stats[0]), np.asarray(stats[1:], dtype=float)
        p = (1.0 + np.sum(null >= obs - 1e-12)) / (len(null) + 1.0)
        sdn = float(null.std(ddof=1)) if len(null) > 1 else float("nan")
        res.results[det] = DetectorResult(
            detector=det,
            statistic=obs,
            null_mean=float(null.mean()),
            null_sd=sdn,
            z=(obs - float(null.mean())) / sdn if sdn and sdn > 0 else float("nan"),
            p_value=float(p),
            ci_low=float(ci[0]),
            ci_high=float(ci[1]),
            n_permutations=int(len(null)),
            seconds=round(time.time() - t0, 3),
        )
    if res.results:
        names_d = list(res.results)
        q = benjamini_hochberg(np.array([res.results[d].p_value for d in names_d]))
        for d_, qv in zip(names_d, q, strict=True):
            res.results[d_].p_bh = float(qv)
            res.results[d_].flag = bool(qv < cfg.alpha)
    return res
