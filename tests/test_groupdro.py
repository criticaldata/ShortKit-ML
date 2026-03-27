# tests/test_groupdro.py
"""
Tests for GroupDRO detection method.

These tests validate that:
1) GroupDRO runs end-to-end through the ShortcutDetector interface on embeddings.
2) The report payload has the expected structure.
3) Group labels are required.
4) The adversarial weights (q) behave sensibly on a constructed "hard group" dataset.
"""

import numpy as np
import pytest
import torch

from shortcut_detect import ShortcutDetector
from shortcut_detect.groupdro.groupdro import GroupDROConfig, GroupDRODetector
from tests.fixtures.synthetic_data import generate_linear_shortcut_with_group_labels


class TestGroupDRO:
    def test_groupdro_basic_report_structure(self):
        X, y, g = generate_linear_shortcut_with_group_labels(n_samples=600, embedding_dim=16)
        n_epochs = 3

        det = ShortcutDetector(
            methods=["groupdro"],
            seed=0,
            groupdro_n_epochs=n_epochs,
            groupdro_batch_size=128,
            groupdro_lr=0.05,
            groupdro_weight_decay=0.0,
            groupdro_val_fraction=0.2,
            groupdro_device="cpu",
        )
        det.fit(X, y, group_labels=g)

        assert "groupdro" in det.results_
        assert det.results_["groupdro"]["success"] is True

        rep = det.results_["groupdro"]["report"]
        assert rep["success"] is True
        assert rep["method"] == "groupdro"
        assert "history" in rep
        assert isinstance(rep["history"], list)
        assert len(rep["history"]) == n_epochs

        assert "final" in rep
        final = rep["final"]
        assert "avg_acc" in final
        assert "worst_group_acc" in final
        assert 0.0 <= final["avg_acc"] <= 1.0
        assert 0.0 <= final["worst_group_acc"] <= 1.0
        assert final["worst_group_acc"] <= final["avg_acc"] + 1e-6

        assert "final_adv_probs" in rep
        q = np.asarray(rep["final_adv_probs"])
        assert q.ndim == 1
        assert np.isfinite(q).all()
        assert (q >= 0).all()
        assert abs(q.sum() - 1.0) < 1e-5

        assert "group_id_map" in rep
        assert isinstance(rep["group_id_map"], dict)
        # should map original group ids {0,1} -> contiguous {0,1}
        assert set(rep["group_id_map"].keys()) == {0, 1}
        assert set(rep["group_id_map"].values()) == {0, 1}

    def test_groupdro_adversarial_weights_focus_on_hard_group(self):
        """
        On a constructed dataset where group 1 is much noisier (harder),
        GroupDRO's adversarial q should tend to place more weight on that group.
        """
        X, y, g = generate_linear_shortcut_with_group_labels(
            n_samples=800,
            embedding_dim=20,
            hard_group_noise=3.0,
            easy_group_noise=0.2,
            seed=123,
        )

        det = ShortcutDetector(
            methods=["groupdro"],
            seed=0,
            groupdro_n_epochs=5,
            groupdro_batch_size=128,
            groupdro_lr=0.05,
            groupdro_weight_decay=0.0,
            groupdro_robust_step_size=0.05,
            groupdro_val_fraction=0.2,
            groupdro_device="cpu",
        )
        det.fit(X, y, group_labels=g)

        rep = det.results_["groupdro"]["report"]
        q = np.asarray(rep["final_adv_probs"])

        # Invert group_id_map to locate indices
        # group_id_map: original_gid -> idx
        gid_map = rep["group_id_map"]
        idx_easy = gid_map[0]
        idx_hard = gid_map[1]

        # Expect higher q on harder group (not guaranteed in every random draw, but should hold here)
        assert q[idx_hard] > q[idx_easy]

    def test_groupdro_loader_factory_hook_invoked(self):
        X, y, g = generate_linear_shortcut_with_group_labels(n_samples=400, embedding_dim=12)
        calls = []

        def loader_factory(req):
            calls.append(req.stage)
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

        det = ShortcutDetector(
            methods=["groupdro"],
            seed=0,
            groupdro_n_epochs=2,
            groupdro_batch_size=64,
            groupdro_lr=0.05,
            groupdro_weight_decay=0.0,
            groupdro_val_fraction=0.2,
            groupdro_device="cpu",
            groupdro_loader_factory=loader_factory,
        )
        det.fit(X, y, group_labels=g)

        assert det.results_["groupdro"]["success"] is True
        assert {"train", "val", "full"}.issubset(set(calls))

    def test_groupdro_fit_dataset_map_style(self):
        X, y, g = generate_linear_shortcut_with_group_labels(n_samples=320, embedding_dim=14)

        class DictMapDataset(torch.utils.data.Dataset):
            def __init__(self, X_arr, y_arr, g_arr):
                self.X_arr = X_arr.astype("float32")
                self.y_arr = y_arr.astype("int64")
                self.g_arr = g_arr.astype("int64")

            def __len__(self):
                return int(self.X_arr.shape[0])

            def __getitem__(self, idx):
                return {
                    "embeddings": self.X_arr[idx],
                    "labels": self.y_arr[idx],
                    "group_labels": self.g_arr[idx],
                }

        detector = GroupDRODetector(
            GroupDROConfig(
                n_epochs=2,
                batch_size=64,
                lr=0.05,
                weight_decay=0.0,
                val_fraction=0.2,
                device="cpu",
            )
        )
        detector.fit_dataset(DictMapDataset(X, y, g))

        report = detector.get_report()["report"]
        assert report["success"] is True
        assert report["method"] == "groupdro"
        assert report["split_mode"] == "external"
        assert "final" in report
        assert "avg_acc" in report["final"]
        assert "worst_group_acc" in report["final"]

    def test_groupdro_fit_dataset_iterable_requires_data_spec(self):
        X, y, g = generate_linear_shortcut_with_group_labels(n_samples=280, embedding_dim=10)
        split = 180

        class TupleIterableDataset(torch.utils.data.IterableDataset):
            def __init__(self, X_arr, y_arr, g_arr):
                self.X_arr = X_arr.astype("float32")
                self.y_arr = y_arr.astype("int64")
                self.g_arr = g_arr.astype("int64")

            def __iter__(self):
                for i in range(self.X_arr.shape[0]):
                    yield self.X_arr[i], self.y_arr[i], self.g_arr[i]

        train_ds = TupleIterableDataset(X[:split], y[:split], g[:split])
        val_ds = TupleIterableDataset(X[split:], y[split:], g[split:])
        detector = GroupDRODetector(
            GroupDROConfig(
                n_epochs=2,
                batch_size=64,
                lr=0.05,
                weight_decay=0.0,
                device="cpu",
            )
        )

        with pytest.raises(ValueError, match="IterableDataset loaders require `data_spec`"):
            detector.fit_dataset(train_ds, val_dataset=val_ds)

        detector.fit_dataset(
            train_ds,
            val_dataset=val_ds,
            data_spec={
                "n_features": 10,
                "n_classes": 2,
                "n_groups": 2,
                "train_size": split,
                "val_size": len(y) - split,
            },
        )
        report = detector.get_report()["report"]
        assert report["success"] is True
        assert report["method"] == "groupdro"

    def test_groupdro_shortcutdetector_loader_native_path(self):
        X, y, g = generate_linear_shortcut_with_group_labels(n_samples=300, embedding_dim=12)

        class DictMapDataset(torch.utils.data.Dataset):
            def __init__(self, X_arr, y_arr, g_arr):
                self.X_arr = X_arr.astype("float32")
                self.y_arr = y_arr.astype("int64")
                self.g_arr = g_arr.astype("int64")

            def __len__(self):
                return int(self.X_arr.shape[0])

            def __getitem__(self, idx):
                return {
                    "embeddings": self.X_arr[idx],
                    "labels": self.y_arr[idx],
                    "group_labels": self.g_arr[idx],
                }

        det = ShortcutDetector(
            methods=["groupdro"],
            seed=0,
            groupdro_n_epochs=2,
            groupdro_batch_size=64,
            groupdro_lr=0.05,
            groupdro_weight_decay=0.0,
            groupdro_device="cpu",
        )
        det.fit_from_loaders({"groupdro": {"train_dataset": DictMapDataset(X, y, g)}})
        rep = det.results_["groupdro"]["report"]
        assert rep["success"] is True
        assert rep["method"] == "groupdro"

    def test_groupdro_callable_fallback_called_once(self):
        X, y, g = generate_linear_shortcut_with_group_labels(n_samples=280, embedding_dim=12)
        calls = {"n": 0}

        def one_shot_loader():
            calls["n"] += 1
            return {
                "embeddings": X.astype("float32"),
                "labels": y.astype("int64"),
                "group_labels": g.astype("int64"),
            }

        det = ShortcutDetector(
            methods=["groupdro"],
            seed=0,
            groupdro_n_epochs=2,
            groupdro_batch_size=64,
            groupdro_lr=0.05,
            groupdro_weight_decay=0.0,
            groupdro_val_fraction=0.2,
            groupdro_device="cpu",
        )
        det.fit_from_loaders({"groupdro": one_shot_loader})

        assert calls["n"] == 1
        assert det.results_["groupdro"]["success"] is True
