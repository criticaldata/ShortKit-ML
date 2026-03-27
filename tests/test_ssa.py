"""
Tests for SSA (Spread Spurious Attribute) detector and integration.

These tests are adapted to the *final* SSA interface:
- SSADetector.fit(du_embeddings, du_labels, dl_embeddings, dl_labels, dl_spurious)
- ShortcutDetector integrates SSA via (splits, extra_labels["spurious"]) where:
    splits["train_u"] indexes DU
    splits["train_l"] indexes DL
    extra_labels["spurious"] is full-length with -1 for unknown and >=0 for DL

Important: SSA training is iterative; tests configure very small T / epochs to keep runtime low.
"""

import numpy as np
import pytest

from shortcut_detect import ShortcutDetector
from shortcut_detect.groupdro.groupdro import GroupDROConfig
from shortcut_detect.ssa import SSAConfig, SSADetector

# -----------------------
# Helpers
# -----------------------


def _make_tiny_ssa(det_seed: int = 0) -> SSADetector:
    """Create an SSADetector configured to run fast in tests."""
    gdro_cfg = GroupDROConfig(
        n_epochs=1,
        batch_size=32,
        lr=1e-2,
        robust=True,
        alpha=0.2,
        robust_step_size=0.01,
        seed=det_seed,
        device=None,
    )
    cfg = SSAConfig(
        K=2,
        T=10,
        batch_size=32,
        lr=5e-2,
        weight_decay=0.0,
        momentum=0.0,
        tau_gmin=0.90,
        threshold_update_every=5,
        threshold_update_max_items=256,
        dl_val_fraction=0.5,
        seed=det_seed,
        device=None,
        groupdro=gdro_cfg,
    )
    return SSADetector(cfg)


def _toy_data(n_du=60, n_dl=40, d=8, n_y=2, n_a=2, seed=0):
    rng = np.random.RandomState(seed)
    X_du = rng.randn(n_du, d).astype(np.float32)
    y_du = rng.randint(0, n_y, size=n_du).astype(np.int64)

    X_dl = rng.randn(n_dl, d).astype(np.float32)
    y_dl = rng.randint(0, n_y, size=n_dl).astype(np.int64)
    a_dl = rng.randint(0, n_a, size=n_dl).astype(np.int64)
    return X_du, y_du, X_dl, y_dl, a_dl


# -----------------------
# SSADetector tests
# -----------------------


class TestSSADetector:
    """Test SSA detector class (direct usage)."""

    def test_ssa_detector_fit_requires_all_inputs(self):
        det = _make_tiny_ssa()
        X_du, y_du, X_dl, y_dl, a_dl = _toy_data()

        # Missing required positional arguments should raise TypeError
        with pytest.raises(TypeError):
            det.fit(X_du, y_du)  # missing DL triplet

        with pytest.raises(TypeError):
            det.fit(X_du, y_du, X_dl, y_dl)  # missing dl_spurious

    def test_ssa_detector_basic_fit(self):
        X_du, y_du, X_dl, y_dl, a_dl = _toy_data()
        det = _make_tiny_ssa(det_seed=123)

        det.fit(
            du_embeddings=X_du,
            du_labels=y_du,
            dl_embeddings=X_dl,
            dl_labels=y_dl,
            dl_spurious=a_dl,
        )

        rep = det.get_report()
        assert rep["method"] == "ssa"
        assert rep["metadata"]["K"] == 2
        assert rep["metadata"]["T"] == 10
        assert rep["metrics"]["n_unlabeled"] == X_du.shape[0]
        assert rep["metrics"]["n_labeled"] == X_dl.shape[0]

        # pseudo labels for DU
        detail = rep["report"]
        assert "pseudo_attr_hat" in detail
        assert isinstance(detail["pseudo_attr_hat"], np.ndarray)
        assert detail["pseudo_attr_hat"].shape[0] == X_du.shape[0]

        # Phase 2 report
        assert "groupdro_report" in detail
        gdro_rep = detail["groupdro_report"]
        assert gdro_rep.get("success", True) is True
        assert "final" in gdro_rep
        assert "avg_acc" in gdro_rep["final"]
        assert "worst_group_acc" in gdro_rep["final"]

    def test_ssa_detector_get_report_before_fit(self):
        """get_report() should raise error before fit()."""
        det = _make_tiny_ssa()
        with pytest.raises(ValueError, match="Must call fit"):
            det.get_report()


# -----------------------
# ShortcutDetector integration tests
# -----------------------


class TestShortcutDetectorSSAIntegration:
    """Test ShortcutDetector integration with SSA via splits/extra_labels."""

    def test_shortcut_detector_ssa_runs_with_splits_and_spurious(self):
        rng = np.random.RandomState(0)
        n = 100
        d = 8
        X = rng.randn(n, d).astype(np.float32)
        y = rng.randint(0, 2, n).astype(np.int64)

        # Define splits
        dl_idx = np.arange(0, 40, dtype=np.int64)  # labeled for spurious attribute
        du_idx = np.arange(40, 100, dtype=np.int64)  # unlabeled for spurious attribute
        splits = {"train_l": dl_idx, "train_u": du_idx}

        # Extra label "spurious" (full-length). -1 for unknown, >=0 for DL.
        spurious = np.full(n, -1, dtype=np.int64)
        spurious[dl_idx] = rng.randint(0, 2, size=len(dl_idx)).astype(np.int64)
        extra_labels = {"spurious": spurious}

        detector = ShortcutDetector(
            methods=["ssa"],
            seed=0,
            # Fast SSA + fast GroupDRO
            ssa_K=2,
            ssa_T=10,
            ssa_batch_size=32,
            ssa_lr=5e-2,
            ssa_weight_decay=0.0,
            ssa_momentum=0.0,
            ssa_tau_gmin=0.90,
            ssa_threshold_update_every=5,
            ssa_threshold_update_max_items=256,
            ssa_dl_val_fraction=0.5,
            ssa_groupdro_n_epochs=1,
            ssa_groupdro_batch_size=32,
            ssa_groupdro_lr=1e-2,
            ssa_groupdro_momentum=0.0,
        )

        detector.fit(X, y, splits=splits, extra_labels=extra_labels)

        assert "ssa" in detector.results_
        assert detector.results_["ssa"]["success"] is True

        rep = detector.results_["ssa"]["report"]
        assert rep["pseudo_attr_hat"].shape[0] == len(du_idx)

        gdro_final = rep["groupdro_report"]["final"]
        assert "avg_acc" in gdro_final
        assert "worst_group_acc" in gdro_final

    def test_shortcut_detector_validates_spurious_on_train_l(self):
        rng = np.random.RandomState(1)
        n = 80
        X = rng.randn(n, 6).astype(np.float32)
        y = rng.randint(0, 2, n).astype(np.int64)

        dl_idx = np.arange(0, 20, dtype=np.int64)
        du_idx = np.arange(20, 80, dtype=np.int64)
        splits = {"train_l": dl_idx, "train_u": du_idx}

        # Make spurious missing on DL -> should fail SSA
        spurious = np.full(n, -1, dtype=np.int64)
        extra_labels = {"spurious": spurious}

        detector = ShortcutDetector(
            methods=["ssa"],
            seed=0,
            ssa_K=2,
            ssa_T=10,
            ssa_groupdro_n_epochs=1,
        )

        with pytest.warns(UserWarning, match="SSA analysis failed"):
            detector.fit(X, y, splits=splits, extra_labels=extra_labels)

        assert detector.results_["ssa"]["success"] is False
        assert "spurious" in detector.results_["ssa"]["error"].lower()

    def test_shortcut_detector_summary_includes_ssa_when_successful(self):
        rng = np.random.RandomState(2)
        n = 90
        d = 8
        X = rng.randn(n, d).astype(np.float32)
        y = rng.randint(0, 2, n).astype(np.int64)

        dl_idx = np.arange(0, 30, dtype=np.int64)
        du_idx = np.arange(30, 90, dtype=np.int64)
        splits = {"train_l": dl_idx, "train_u": du_idx}

        spurious = np.full(n, -1, dtype=np.int64)
        spurious[dl_idx] = rng.randint(0, 2, size=len(dl_idx)).astype(np.int64)
        extra_labels = {"spurious": spurious}

        detector = ShortcutDetector(
            methods=["ssa"],
            seed=0,
            ssa_K=2,
            ssa_T=10,
            ssa_groupdro_n_epochs=1,
            ssa_groupdro_batch_size=32,
        )

        detector.fit(X, y, splits=splits, extra_labels=extra_labels)
        summary = detector.summary()

        assert "SSA (Spread Spurious Attribute)" in summary
        # Since summary formatting may vary, check for core fields rather than exact lines:
        assert "Worst-group acc" in summary or "worst-group" in summary.lower()

    def test_shortcut_detector_ssa_loader_hooks_invoked(self):
        rng = np.random.RandomState(7)
        n = 90
        d = 8
        X = rng.randn(n, d).astype(np.float32)
        y = rng.randint(0, 2, n).astype(np.int64)

        dl_idx = np.arange(0, 30, dtype=np.int64)
        du_idx = np.arange(30, 90, dtype=np.int64)
        splits = {"train_l": dl_idx, "train_u": du_idx}

        spurious = np.full(n, -1, dtype=np.int64)
        spurious[dl_idx] = rng.randint(0, 2, size=len(dl_idx)).astype(np.int64)
        extra_labels = {"spurious": spurious}

        ssa_calls = []
        gdro_calls = []

        def ssa_loader_factory(req):
            ssa_calls.append(req.stage)
            import torch

            return torch.utils.data.DataLoader(
                req.dataset,
                batch_size=req.batch_size,
                shuffle=req.shuffle,
                num_workers=req.num_workers,
                pin_memory=req.pin_memory,
                drop_last=req.drop_last,
                **req.extra_kwargs,
            )

        def ssa_groupdro_loader_factory(req):
            gdro_calls.append(req.stage)
            import torch

            return torch.utils.data.DataLoader(
                req.dataset,
                batch_size=req.batch_size,
                shuffle=req.shuffle,
                num_workers=req.num_workers,
                pin_memory=req.pin_memory,
                drop_last=req.drop_last,
                **req.extra_kwargs,
            )

        detector = ShortcutDetector(
            methods=["ssa"],
            seed=0,
            ssa_K=2,
            ssa_T=10,
            ssa_batch_size=32,
            ssa_threshold_update_every=5,
            ssa_threshold_update_max_items=256,
            ssa_groupdro_n_epochs=1,
            ssa_groupdro_batch_size=32,
            ssa_loader_factory=ssa_loader_factory,
            ssa_groupdro_loader_factory=ssa_groupdro_loader_factory,
        )

        detector.fit(X, y, splits=splits, extra_labels=extra_labels)
        assert detector.results_["ssa"]["success"] is True
        assert {"dl_train", "dl_val", "du_train", "du_eval", "du_bullet"}.issubset(set(ssa_calls))
        assert {"train", "val"}.issubset(set(gdro_calls))
