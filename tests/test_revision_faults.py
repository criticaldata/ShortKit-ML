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
