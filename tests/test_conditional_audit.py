"""Tests for the conditional shortcut-potential audit."""

from __future__ import annotations

import numpy as np
import pytest

from shortcut_detect.audit import (
    DETECTORS,
    AuditConfig,
    benjamini_hochberg,
    conditional_audit,
    conditional_permutations,
)
from shortcut_detect.benchmark.conditional_synthetic import generate_conditional_synthetic


def test_permutations_stay_within_y_strata():
    rng = np.random.default_rng(0)
    y = rng.integers(0, 2, 300)
    a = np.where(rng.random(300) < 0.8, y, 1 - y)
    rows = conditional_permutations(a, y, 20, rng)
    assert np.array_equal(rows[0], a)
    for r in rows[1:]:
        for v in (0, 1):
            assert np.sort(r[y == v]).tolist() == np.sort(a[y == v]).tolist()


def test_bh_matches_definition():
    p = np.array([0.01, 0.04, 0.03, 0.5])
    q = benjamini_hochberg(p)
    assert np.allclose(q, [0.04, 0.16 / 3, 0.16 / 3, 0.5])


def test_probe_matches_sklearn_cross_validated_auroc():
    from sklearn.linear_model import LogisticRegression
    from sklearn.metrics import roc_auc_score

    ds = generate_conditional_synthetic(n_samples=400, embedding_dim=20, delta=1.0, rho=0.5, seed=3)
    res = conditional_audit(
        ds.embeddings,
        ds.attribute,
        ds.task_labels,
        AuditConfig(n_permutations=5, n_bootstrap=0, detectors=("probe",), seed=0),
    )
    X = np.c_[ds.embeddings, ds.task_labels]
    X = (X - X.mean(0)) / X.std(0)
    clf = LogisticRegression(C=1.0, max_iter=2000).fit(X, ds.attribute)
    in_sample = roc_auc_score(ds.attribute, clf.decision_function(X))
    assert 0.6 < res.results["probe"].statistic <= in_sample + 1e-6


def test_null_fpr_near_alpha_per_detector():
    """A depends on Y (rho = 0.9) but E carries no A signal beyond Y."""
    n_seeds = 100
    rejections = {d: 0 for d in DETECTORS}
    rejections_global = 0
    for seed in range(n_seeds):
        ds = generate_conditional_synthetic(
            n_samples=160, embedding_dim=8, delta=0.0, rho=0.9, seed=1000 + seed
        )
        cfg = AuditConfig(n_permutations=39, n_bootstrap=0, n_clusters=4, seed=seed, max_iter=50)
        res = conditional_audit(ds.embeddings, ds.attribute, ds.task_labels, cfg)
        for d in DETECTORS:
            rejections[d] += res.results[d].p_value < 0.05
        glob = conditional_audit(
            ds.embeddings,
            ds.attribute,
            ds.task_labels,
            AuditConfig(
                n_permutations=39,
                n_bootstrap=0,
                detectors=("probe",),
                permutation="global",
                include_y=False,
                seed=seed,
                max_iter=50,
            ),
        )
        rejections_global += glob.results["probe"].p_value < 0.05
    for d, r in rejections.items():
        # binomial(100, 0.05): P(X >= 12) < 0.005
        assert r / n_seeds <= 0.12, f"{d}: FPR {r / n_seeds:.2f}"
    # the marginal test confuses A with Y and rejects almost always
    assert rejections_global / n_seeds > 0.8


def test_planted_linear_signal_detected():
    ds = generate_conditional_synthetic(
        n_samples=400, embedding_dim=16, delta=1.5, rho=0.7, variant="linear", seed=5
    )
    res = conditional_audit(
        ds.embeddings,
        ds.attribute,
        ds.task_labels,
        AuditConfig(n_permutations=49, n_bootstrap=0, n_clusters=4, seed=1),
    )
    for d in ("probe", "statistical", "geometric", "bias_direction", "frequency_topk"):
        assert res.results[d].flag, d
    assert res.results["probe"].p_value <= 0.02


def test_planted_xor_signal_detected_by_clustering():
    """XOR has zero mean difference; only the clustering statistic can see it."""
    ds = generate_conditional_synthetic(
        n_samples=600, embedding_dim=4, delta=5.0, rho=0.5, variant="xor",
        attribute_dims=2, task_dims=2, seed=7,
    )
    res = conditional_audit(
        ds.embeddings,
        ds.attribute,
        ds.task_labels,
        AuditConfig(n_permutations=199, n_bootstrap=0, n_clusters=8, seed=2),
    )
    # with K = 199 a lone significant detector survives BH across six
    assert res.results["hbac_nmi"].flag
    assert res.results["hbac_nmi"].p_value <= 0.02


def test_bootstrap_ci_brackets_probe_statistic():
    ds = generate_conditional_synthetic(n_samples=300, embedding_dim=10, delta=1.0, rho=0.5, seed=2)
    res = conditional_audit(
        ds.embeddings,
        ds.attribute,
        ds.task_labels,
        AuditConfig(n_permutations=19, n_bootstrap=200, detectors=("probe", "geometric"), seed=0),
    )
    for d in ("probe", "geometric"):
        r = res.results[d]
        assert r.ci_low <= r.statistic <= r.ci_high


def test_multigroup_attribute_runs():
    rng = np.random.default_rng(0)
    n = 300
    y = rng.integers(0, 2, n)
    a = rng.choice(["W", "B", "A"], n)
    X = rng.standard_normal((n, 6))
    X[a == "B", 0] += 1.5
    res = conditional_audit(X, a, y, AuditConfig(n_permutations=19, n_bootstrap=0, n_clusters=3))
    assert res.n_groups == 3
    assert res.results["probe"].p_value <= 0.1


def test_grouped_folds_are_patient_disjoint():
    from shortcut_detect.audit.conditional import _fold_ids

    rng = np.random.default_rng(0)
    n = 300
    groups = rng.integers(0, 60, n)
    y = rng.integers(0, 2, n)
    a = rng.integers(0, 2, n)
    rows = conditional_permutations(a, y, 3, rng)
    folds, how = _fold_ids(rows, y, 5, rng, groups)
    assert how.endswith("grouped")
    for f in folds:
        for g in np.unique(groups):
            assert len(np.unique(f[groups == g])) == 1
    res = conditional_audit(
        rng.standard_normal((n, 5)), a, y,
        AuditConfig(n_permutations=9, n_bootstrap=50, detectors=("probe",)), groups=groups,
    )
    assert res.fold_strata.endswith("grouped")


def test_statistical_batched_rows_match_single_rows_when_group_sizes_vary():
    """Patient-level permutations change the group counts inside each Y stratum,
    so every permutation row must use its own counts."""
    from shortcut_detect.audit.conditional import _prepare, _stat_statistical

    rng = np.random.default_rng(3)
    n, d = 240, 6
    groups = rng.integers(0, 40, n)
    y = rng.integers(0, 2, n)
    a = (rng.random(40) < 0.5).astype(int)[groups]
    X = rng.standard_normal((n, d)) + 0.8 * a[:, None]
    rows = conditional_permutations(a, y, 5, rng, groups=groups)
    assert len({tuple(np.bincount(r[y == 0], minlength=2)) for r in rows}) > 1
    cfg = AuditConfig()
    ctx = _prepare(X, y, cfg, {"statistical"}, rng)
    ctx["G"] = 2
    batched = _stat_statistical(ctx, rows, cfg)
    single = np.array([_stat_statistical(ctx, r[None, :], cfg)[0] for r in rows])
    np.testing.assert_allclose(batched, single)


# ---------------------------------------------------------------------------
# Section 9 corrections (28 Oct): permutation unit, in-fold preprocessing,
# observed statistic independent of K, repeated CV.
# ---------------------------------------------------------------------------


def _grouped_toy(seed=0, n_pat=60, mixed=0):
    rng = np.random.default_rng(seed)
    sizes = rng.integers(1, 5, n_pat)
    groups = np.repeat(np.arange(n_pat), sizes)
    n = len(groups)
    y = rng.integers(0, 2, n)
    a_pat = (rng.random(n_pat) < 0.5).astype(int)
    a = a_pat[groups].copy()
    flipped = []
    for g in np.flatnonzero(sizes >= 3)[:mixed]:  # one image of the patient gets the other value
        i = np.flatnonzero(groups == g)[0]
        a[i] = 1 - a[i]
        flipped.append(g)
    X = rng.standard_normal((n, 8)) + 0.6 * a_pat[groups][:, None]
    return X, a, y, groups, len(flipped)


def test_observed_statistic_does_not_depend_on_K():
    X, a, y, groups, _ = _grouped_toy(1)
    det = ("probe", "frequency_topk", "hbac_nmi", "geometric")
    r1 = conditional_audit(X, a, y, AuditConfig(n_permutations=9, n_bootstrap=0, detectors=det, seed=4), groups=groups)
    r2 = conditional_audit(X, a, y, AuditConfig(n_permutations=29, n_bootstrap=0, detectors=det, seed=4), groups=groups)
    for d in det:
        assert r1.results[d].statistic == pytest.approx(r2.results[d].statistic, abs=1e-9), d


def test_probe_reports_mean_and_sd_over_repeated_cv():
    X, a, y, groups, _ = _grouped_toy(2)
    r = conditional_audit(X, a, y, AuditConfig(n_permutations=9, n_bootstrap=100, detectors=("probe",)), groups=groups)
    p = r.results["probe"]
    assert p.n_repeats == 5
    assert np.isfinite(p.repeat_sd) and p.repeat_sd >= 0
    assert p.ci_low <= p.statistic <= p.ci_high


def test_probe_matches_in_fold_standardised_sklearn():
    """Scaling is fitted on the training folds only (StandardScaler inside the fold)."""
    from sklearn.linear_model import LogisticRegression
    from sklearn.metrics import roc_auc_score
    from sklearn.pipeline import make_pipeline
    from sklearn.preprocessing import StandardScaler

    from shortcut_detect.audit.conditional import _prepare, _stat_probe

    rng = np.random.default_rng(5)
    n = 300
    y = rng.integers(0, 2, n)
    a = rng.integers(0, 2, n)
    # heavy-tailed, shifted columns so in-fold and full-data scaling differ
    X = rng.standard_t(2, (n, 6)) * np.array([1, 5, 0.1, 20, 1, 3]) + 0.7 * a[:, None]
    X[:40] *= 6.0
    folds = np.tile(np.arange(n) % 5, (1, 1))
    rng.shuffle(folds[0])
    cfg = AuditConfig(n_permutations=1, max_iter=500)
    ctx = _prepare(X, y, cfg, {"probe"}, rng)
    ctx["G"] = 2
    stat, sc, _ = _stat_probe(ctx, a[None, :], folds, cfg)
    oof = np.zeros(n)
    F = np.c_[X, y]
    for f in range(5):
        tr, te = folds[0] != f, folds[0] == f
        m = make_pipeline(StandardScaler(), LogisticRegression(C=1.0, max_iter=5000, tol=1e-10)).fit(F[tr], a[tr])
        oof[te] = m.decision_function(F[te])
    assert stat[0] == pytest.approx(roc_auc_score(a, oof), abs=2e-3)
    np.testing.assert_allclose(sc[:, 0], oof, atol=2e-3 * (1 + np.abs(oof).max()))


def test_frequency_topk_matches_in_fold_reference():
    """Y-centring, the ranking norm and the scaling of the selected columns use training rows only."""
    from sklearn.linear_model import LogisticRegression
    from sklearn.metrics import roc_auc_score
    from sklearn.pipeline import make_pipeline
    from sklearn.preprocessing import StandardScaler

    from shortcut_detect.audit.conditional import _prepare, _stat_frequency

    rng = np.random.default_rng(6)
    n, d = 300, 60
    y = rng.integers(0, 2, n)
    a = rng.integers(0, 2, n)
    X = rng.standard_normal((n, d)) * rng.uniform(0.2, 5, d) + rng.uniform(-3, 3, d)
    X[:, :3] += 0.8 * a[:, None]
    X[:30] += 4.0  # rows that move full-data means and norms
    folds = (np.arange(n) % 5)[None, :].copy()
    rng.shuffle(folds[0])
    cfg = AuditConfig(n_permutations=1, max_iter=500)
    ctx = _prepare(X, y, cfg, {"frequency_topk"}, rng)
    ctx["G"] = 2
    stat, sc, _ = _stat_frequency(ctx, a[None, :], folds, cfg)
    k = int(np.ceil(cfg.top_k_fraction * d))
    oof = np.zeros(n)
    for f in range(5):
        tr, te = folds[0] != f, folds[0] == f
        Xt, yt, at = X[tr], y[tr], a[tr].astype(float)
        Xc, tc = Xt.copy(), at.copy()
        for v in (0, 1):
            Xc[yt == v] -= Xc[yt == v].mean(0)
            tc[yt == v] -= tc[yt == v].mean()
        corr = np.abs(Xc.T @ tc) / np.linalg.norm(Xc, axis=0)
        top = np.argsort(-corr)[:k]
        F = np.c_[X[:, top], y]
        m = make_pipeline(StandardScaler(), LogisticRegression(C=1.0, max_iter=5000, tol=1e-10)).fit(F[tr], a[tr])
        oof[te] = m.decision_function(F[te])
    assert stat[0] == pytest.approx(roc_auc_score(a, oof), abs=2e-3)
    np.testing.assert_allclose(sc[:, 0], oof, atol=2e-3 * (1 + np.abs(oof).max()))


def test_patient_level_attribute_is_used_in_the_observed_statistic():
    X, a, y, groups, n_mixed = _grouped_toy(3, mixed=5)
    assert n_mixed == 5
    maj = a.copy()
    for g in np.unique(groups):
        idx = groups == g
        vals, cnt = np.unique(a[idx], return_counts=True)
        maj[idx] = vals[np.argmax(cnt)]
    cfg = AuditConfig(n_permutations=9, n_bootstrap=0, seed=1)
    r_a = conditional_audit(X, a, y, cfg, groups=groups)
    r_m = conditional_audit(X, maj, y, cfg, groups=groups)
    assert r_a.n_inconsistent_groups == 5 and r_m.n_inconsistent_groups == 0
    for d in DETECTORS:
        assert r_a.results[d].statistic == pytest.approx(r_m.results[d].statistic, abs=1e-9), d
        assert r_a.results[d].p_value == pytest.approx(r_m.results[d].p_value), d


def test_image_level_block_permutation_moves_whole_patient_vectors():
    rng = np.random.default_rng(7)
    sizes = rng.integers(1, 4, 80)
    groups = np.repeat(np.arange(80), sizes)
    y = rng.integers(0, 2, len(groups))
    a = rng.integers(0, 2, len(groups))  # varies within patient
    rows = conditional_permutations(a, y, 20, rng, groups=groups, level="image")
    assert (rows[0] == a).all()

    def key(g):
        idx = np.flatnonzero(groups == g)
        o = np.argsort(y[idx], kind="stable")
        return tuple(y[idx][o]), idx[o]

    vec = {}
    for g in range(80):
        yk, idx = key(g)
        vec.setdefault(yk, set()).add(tuple(a[idx]))
    moved = 0
    for r in rows[1:]:
        assert ((r == 1) & (y == 1)).sum() == ((a == 1) & (y == 1)).sum()
        for g in range(80):
            yk, idx = key(g)
            assert tuple(r[idx]) in vec[yk]  # some patient's vector with the same Y pattern and size
            moved += tuple(r[idx]) != tuple(a[idx])
    assert moved > 0
