"""Regression tests for faults found in the CMPB revision audit.

Each test failed on the submitted code and passes after the fix.
"""

from __future__ import annotations

import numpy as np

from shortcut_detect.benchmark import method_utils
from shortcut_detect.benchmark.method_utils import method_flag
from shortcut_detect.gce.gce_detector import GCEDetectorReport


def _stat_result(flag: bool) -> dict:
    return {"success": True, "significant_features": {"0_vs_1": [3] if flag else None}}


def _geom_result(flag: bool) -> dict:
    return {"success": True, "summary": {"risk_level": "high" if flag else "low"}}


class TestMethodFlagNested:
    """Fault 3: statistical and geometric read top-level keys of nested results."""

    def test_statistical_reads_by_attribute(self):
        res = {
            "success": True,
            "by_attribute": {"group": _stat_result(True), "race": _stat_result(False)},
        }
        assert method_flag("statistical", res) is True
        assert method_flag("statistical", res, attribute="group") is True
        assert method_flag("statistical", res, attribute="race") is False

    def test_geometric_reads_by_attribute(self):
        res = {
            "success": True,
            "by_attribute": {"group": _geom_result(False), "sex": _geom_result(True)},
        }
        assert method_flag("geometric", res, attribute="group") is False
        assert method_flag("geometric", res, attribute="sex") is True

    def test_missing_attribute_is_not_flagged_nor_evaluated(self):
        res = {"success": True, "by_attribute": {"group": _geom_result(True)}}
        assert method_flag("geometric", res, attribute="age") is False
        assert method_utils.method_evaluated("geometric", res, attribute="age") is False

    def test_gce_reads_dataclass_report(self):
        kwargs = {k: 0 for k in GCEDetectorReport.__dataclass_fields__}
        kwargs.update(risk_level="high", notes="", reference="")
        report = GCEDetectorReport(**kwargs)
        assert method_flag("gce", {"success": True, "report": report}) is True

    def test_failed_result_is_not_evaluated(self):
        assert method_utils.method_evaluated("groupdro", {"success": False, "error": "x"}) is False
        res = {"success": True, "by_attribute": {"group": {"success": False}}}
        assert method_utils.method_evaluated("groupdro", res, attribute="group") is False


class TestGroupDROStringGroups:
    """Fault 4: GroupDRO crashed on string group labels (every clinical run)."""

    def test_string_group_labels(self):
        from shortcut_detect.unified import ShortcutDetector

        rng = np.random.RandomState(0)
        X = rng.randn(120, 8).astype(np.float32)
        y = rng.randint(0, 2, 120)
        g = np.where(rng.rand(120) > 0.5, "Female", "Male")
        det = ShortcutDetector(methods=["groupdro"], seed=0, groupdro_n_epochs=1)
        det.fit(X, y, group_labels=g)
        res = det.results_["groupdro"]
        assert res["success"] is True, res.get("error")
        assert method_utils.method_evaluated("groupdro", res)


class TestSSAClearError:
    """Fault 4: SSA failed silently in every clinical run; the error must say what it needs."""

    def test_ssa_without_splits_explains_requirements(self):
        from shortcut_detect.unified import ShortcutDetector

        rng = np.random.RandomState(0)
        X = rng.randn(60, 4)
        y = rng.randint(0, 2, 60)
        det = ShortcutDetector(methods=["ssa"], seed=0)
        det.fit(X, y, group_labels=y)
        res = det.results_["ssa"]
        assert res["success"] is False
        assert "train_u" in res["error"] and "spurious" in res["error"]
        assert not method_utils.method_evaluated("ssa", res)


def _cohens_d(x0: np.ndarray, x1: np.ndarray) -> float:
    pooled = np.sqrt((np.var(x0, ddof=1) + np.var(x1, ddof=1)) / 2)
    return float(abs(x1.mean() - x0.mean()) / pooled)


class TestSyntheticDeltaIsCohensD:
    """Fault 2: the shift was +/- delta, so the standardised difference was 2 * delta."""

    def test_parametric_dataset_delta(self):
        from shortcut_detect.benchmark.synthetic import generate_parametric_shortcut_dataset

        ds = generate_parametric_shortcut_dataset(
            n_samples=20000, embedding_dim=8, shortcut_dims=3, effect_size=0.8, seed=1
        )
        g = ds.group_labels
        d = np.mean([_cohens_d(ds.embeddings[g == 0, j], ds.embeddings[g == 1, j]) for j in range(3)])
        assert abs(d - 0.8) < 0.05

    def test_parametric_dataset_delta_scales_with_noise(self):
        from shortcut_detect.benchmark.synthetic import generate_parametric_shortcut_dataset

        ds = generate_parametric_shortcut_dataset(
            n_samples=20000, embedding_dim=8, shortcut_dims=3, effect_size=0.5, noise_std=2.0
        )
        g = ds.group_labels
        d = _cohens_d(ds.embeddings[g == 0, 0], ds.embeddings[g == 1, 0])
        assert abs(d - 0.5) < 0.05


class TestSyntheticTaskIsNotAttribute:
    """Fault 1: the synthetic benchmark passed the task label as the attribute."""

    def test_dataset_has_separate_attribute(self):
        from shortcut_detect.benchmark.synthetic import generate_parametric_shortcut_dataset

        ds = generate_parametric_shortcut_dataset(n_samples=2000, effect_size=1.0, seed=3)
        assert ds.group_labels is not None
        agreement = float(np.mean(ds.group_labels == ds.labels))
        assert 0.85 < agreement < 0.95

    def test_paper_runner_passes_attribute_not_task(self, monkeypatch):
        from shortcut_detect.benchmark import paper_runner as pr
        from shortcut_detect.unified import ShortcutDetector

        seen = {}

        def fake_fit(self, X, y, group_labels=None, **kw):
            seen["y"], seen["g"] = np.asarray(y), np.asarray(group_labels)
            self.results_ = {m: {"success": True} for m in self.methods}
            return self

        monkeypatch.setattr(ShortcutDetector, "fit", fake_fit)
        runner = pr.PaperBenchmarkRunner.__new__(pr.PaperBenchmarkRunner)
        X, y, g, _ = runner._generate_synthetic_dataset(
            n_samples=500, embedding_dim=16, shortcut_dims=3, effect_size=1.0,
            imbalance_ratio=0.5, seed=0, with_shortcut=True,
        )

        class _Cfg:
            methods = ["probe"]

            class synthetic:
                alpha = 0.05

        runner.config = _Cfg()
        runner._run_methods(X, y, g, seed=0)
        assert not np.array_equal(seen["g"], seen["y"])
        assert np.array_equal(seen["g"], g)


class TestLabelMethodsSeeTheAttribute:
    """Fault 5: HBAC, Frequency, GCE and SIS fitted on the task label only."""

    @staticmethod
    def _data(seed=0, n=400, d=16):
        rng = np.random.RandomState(seed)
        y = rng.randint(0, 2, n)
        a = rng.randint(0, 2, n)
        X = rng.randn(n, d)
        X[:, 0] += 3.0 * (a - 0.5)  # attribute signal only
        return X, y, np.where(a == 1, "M", "F")

    def test_builders_target_attribute(self, monkeypatch):
        from shortcut_detect.clustering.builder import HBACDetectorBuilder
        from shortcut_detect.frequency.builder import FrequencyShortcutBuilder
        from shortcut_detect.xai.sis.builder import SISDetectorBuilder

        X, y, a = self._data()
        for builder_cls in (HBACDetectorBuilder, FrequencyShortcutBuilder, SISDetectorBuilder):
            seen = {}
            det = builder_cls(0, {}).build()
            orig = type(det).fit

            def spy(self, *args, _orig=orig, **kw):
                lab = kw.get("labels", args[1] if len(args) > 1 else None)
                seen["labels"] = np.asarray(lab)
                return _orig(self, *args, **kw)

            monkeypatch.setattr(type(det), "fit", spy)
            res = builder_cls(0, {}).run(X, y, a, None, a)
            assert res["success"], res.get("error")
            assert res["target"] == "attribute"
            assert np.array_equal(seen["labels"], (a == "M").astype(int))
            monkeypatch.undo()

    def test_task_mode_keeps_old_behaviour(self):
        from shortcut_detect.frequency.builder import FrequencyShortcutBuilder

        X, y, a = self._data()
        res = FrequencyShortcutBuilder(0, {"freq_target": "task"}).run(X, y, a, None, a)
        assert res["target"] == "task"

    def test_gce_reports_attribute_enrichment(self):
        from shortcut_detect.gce.builder import GCEDetectorBuilder

        X, y, a = self._data()
        res = GCEDetectorBuilder(0, {}).run(X, y, a, None, a)
        assert res["success"]
        assert res["attribute_test"] is not None
        assert 0.0 <= res["attribute_test"]["p_value"] <= 1.0
