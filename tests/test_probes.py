"""Tests for probe-based detection methods."""

import numpy as np
import pytest
import torch
import torch.nn as nn
from sklearn.linear_model import LogisticRegression

from shortcut_detect import ShortcutDetector
from shortcut_detect.probes import SKLearnProbe, TorchProbe
from tests.fixtures.synthetic_data import generate_linear_shortcut, generate_no_shortcut


class TestSKLearnProbe:
    """Tests for SKLearnProbe (DetectorBase-compatible)."""

    def test_sklearn_probe_basic_shortcut_detected(self):
        """
        On data with a strong linear shortcut, probe metric should exceed threshold,
        and shortcut should be detected.
        """
        # Determinism: numpy + torch
        np.random.seed(42)
        torch.manual_seed(42)
        torch.use_deterministic_algorithms(True)
        torch.set_num_threads(1)

        embeddings, group_labels = generate_linear_shortcut(
            n_samples=500, embedding_dim=30, shortcut_dims=3
        )

        lr = LogisticRegression(max_iter=1000, random_state=42)
        det = SKLearnProbe(
            estimator=lr,
            metric="accuracy",
            threshold=0.7,
            evaluation="holdout",
            test_size=0.2,
            random_state=42,
        )

        det.fit(embeddings, group_labels)

        assert det._is_fitted is True
        assert det.results_["method"] == "ml_probe" or det.results_["method"] == "sklearn_probe"
        assert "metrics" in det.results_
        assert det.results_["metrics"]["metric"] == "accuracy"
        assert isinstance(det.results_["metrics"]["metric_value"], float)

        # Should be high due to shortcut
        assert det.results_["metrics"]["metric_value"] > 0.7
        assert det.results_["shortcut_detected"] is True
        assert det.shortcut_detected_ is True
        assert det.results_["risk_level"] in {"moderate", "high"}

        # Standard schema required keys
        for k in ["method", "shortcut_detected", "risk_level", "metrics", "notes", "metadata"]:
            assert k in det.results_

    def test_sklearn_probe_cv_produces_fold_report(self):
        """CV mode should produce fold scores in report and fit a final estimator."""
        embeddings, group_labels = generate_linear_shortcut(
            n_samples=400, embedding_dim=15, shortcut_dims=2
        )

        det = SKLearnProbe(
            estimator=LogisticRegression(max_iter=500, random_state=0),
            metric="accuracy",
            threshold=0.6,
            evaluation="cv",
            cv_folds=3,
            random_state=0,
        )

        det.fit(embeddings, group_labels)
        report = det.results_.get("report", {})

        assert report.get("protocol") == "cv"
        assert report.get("cv_folds") == 3
        assert "fold_scores" in report
        assert isinstance(report["fold_scores"], list)
        assert len(report["fold_scores"]) == 3
        assert "mean_score" in report
        assert report["mean_score"] > 0.5

        # detector keeps a fitted estimator for later inspection/debugging
        assert det.estimator_ is not None

    def test_sklearn_probe_no_shortcut_not_detected(self):
        """
        On data without shortcuts, probe should perform near chance and not exceed threshold.
        """
        embeddings, group_labels = generate_no_shortcut(n_samples=400, embedding_dim=25)

        det = SKLearnProbe(
            estimator=LogisticRegression(max_iter=500, random_state=0),
            metric="accuracy",
            threshold=0.7,
            evaluation="holdout",
            test_size=0.2,
            random_state=0,
        )

        det.fit(embeddings, group_labels)

        mv = det.results_["metrics"]["metric_value"]
        assert 0.35 <= mv <= 0.65  # random performance with some variance
        assert det.results_["shortcut_detected"] is False
        assert det.results_["risk_level"] in {"low"}

    def test_get_report_requires_fit(self):
        """Calling get_report before fit should raise."""
        det = SKLearnProbe(
            estimator=LogisticRegression(max_iter=200),
            metric="accuracy",
            threshold=0.7,
            evaluation="holdout",
        )
        with pytest.raises(ValueError):
            _ = det.get_report()

    def test_summary_not_fitted(self):
        """summary() should indicate not fitted before fit."""
        det = SKLearnProbe(
            estimator=LogisticRegression(max_iter=200),
            metric="accuracy",
            threshold=0.7,
            evaluation="holdout",
        )
        s = det.summary()
        assert "not fitted" in s.lower()

    def test_results_metrics_are_scalar(self):
        """metrics should stay small/scalar (no large arrays/tables)."""
        embeddings, group_labels = generate_linear_shortcut(
            n_samples=300, embedding_dim=20, shortcut_dims=2
        )
        det = SKLearnProbe(
            estimator=LogisticRegression(max_iter=500, random_state=1),
            metric="f1",
            threshold=0.6,
            evaluation="holdout",
            test_size=0.25,
            random_state=1,
        )
        det.fit(embeddings, group_labels)

        metrics = det.results_["metrics"]
        assert isinstance(metrics, dict)
        for v in metrics.values():
            assert isinstance(v, str | bool | int | float | type(None))


class TestBuilderIntegration:
    """
    Optional integration-style test: if you have ProbeDetectorBuilder wired,
    validate it returns the standardized outputs. Skips if builder not available.
    """

    def test_builder_run_outputs(self):
        try:
            from shortcut_detect.builders.probe_builder import ProbeDetectorBuilder
        except Exception:
            pytest.skip("ProbeDetectorBuilder not available in this test environment.")

        embeddings, labels = generate_linear_shortcut(
            n_samples=300, embedding_dim=20, shortcut_dims=2
        )

        builder = ProbeDetectorBuilder(
            seed=42,
            probe_estimator=LogisticRegression(max_iter=500, random_state=42),
            metric="accuracy",
            threshold=0.7,
            evaluation="holdout",
        )

        out = builder.run(
            embeddings=embeddings,
            labels=labels,  # unused by this detector, kept for signature compatibility
            group_labels=labels,  # demographic target
            feature_names=None,
            protected_labels=None,
            splits=None,
            extra_labels=None,
        )

        assert out["success"] is True
        assert "detector" in out
        assert "results" in out
        assert out["results"]["method"] in {"ml_probe", "sklearn_probe"}
        assert "summary_lines" in out
        assert isinstance(out["summary_lines"], list)
        assert len(out["summary_lines"]) >= 2
        assert "risk_indicators" in out
        assert isinstance(out["risk_indicators"], list)


@pytest.mark.skipif(
    pytest.importorskip("torch", reason="torch not installed") is None,
    reason="torch not installed",
)
class TestTorchProbe:
    """Tests for TorchProbe (DetectorBase-compatible)."""

    def test_torch_probe_detector_basic_shortcut_detected(self):
        import torch.nn as nn

        embeddings, group_labels = generate_linear_shortcut(
            n_samples=500, embedding_dim=30, shortcut_dims=3
        )

        model = nn.Linear(30, 2)
        det = TorchProbe(
            model=model,
            loss_fn=nn.CrossEntropyLoss(),
            metric="accuracy",
            threshold=0.6,  # keep slightly lower for fast training stability
            test_size=0.2,
            random_state=42,
            epochs=5,
            batch_size=64,
            device="cpu",
            verbose=False,
        )

        det.fit(embeddings.astype("float32"), group_labels.astype("int64"))

        assert det._is_fitted is True
        assert det.results_["method"] == "torch_probe"
        for k in ["method", "shortcut_detected", "risk_level", "metrics", "notes", "metadata"]:
            assert k in det.results_

        mv = det.results_["metrics"]["metric_value"]
        assert isinstance(mv, float)
        assert mv > 0.5
        assert det.results_["shortcut_detected"] is True
        assert det.shortcut_detected_ is True
        assert det.results_["risk_level"] in {"moderate", "high"}

        report = det.results_.get("report", {})
        assert report.get("protocol") == "holdout"
        assert "eval_metrics" in report
        assert "train_loss_history" in report
        assert isinstance(report["train_loss_history"], list)

    def test_torch_probe_detector_no_shortcut_not_detected(self):
        import torch.nn as nn

        embeddings, group_labels = generate_no_shortcut(n_samples=500, embedding_dim=25)

        model = nn.Linear(25, 2)
        det = TorchProbe(
            model=model,
            loss_fn=nn.CrossEntropyLoss(),
            metric="accuracy",
            threshold=0.7,
            test_size=0.2,
            random_state=0,
            epochs=4,
            batch_size=64,
            device="cpu",
            verbose=False,
        )

        det.fit(embeddings.astype("float32"), group_labels.astype("int64"))

        mv = det.results_["metrics"]["metric_value"]
        # near chance
        assert 0.35 <= mv <= 0.65
        assert det.results_["shortcut_detected"] is False
        assert det.results_["risk_level"] == "low"

    def test_torch_probe_detector_get_report_requires_fit(self):
        import torch.nn as nn

        det = TorchProbe(
            model=nn.Linear(10, 2),
            loss_fn=nn.CrossEntropyLoss(),
            metric="accuracy",
            threshold=0.7,
            epochs=1,
            device="cpu",
        )
        with pytest.raises(ValueError):
            _ = det.get_report()

    def test_torch_probe_detector_metrics_scalar(self):
        import torch.nn as nn

        embeddings, group_labels = generate_linear_shortcut(
            n_samples=300, embedding_dim=20, shortcut_dims=2
        )

        det = TorchProbe(
            model=nn.Linear(20, 2),
            loss_fn=nn.CrossEntropyLoss(),
            metric="f1",
            threshold=0.55,
            epochs=4,
            batch_size=64,
            device="cpu",
            random_state=7,
            verbose=False,
        )

        det.fit(embeddings.astype("float32"), group_labels.astype("int64"))

        metrics = det.results_["metrics"]
        assert isinstance(metrics, dict)
        for v in metrics.values():
            assert isinstance(v, str | bool | int | float | type(None))

    def test_torch_probe_loader_factory_and_stage_overrides(self):
        embeddings, group_labels = generate_linear_shortcut(
            n_samples=240, embedding_dim=20, shortcut_dims=2
        )

        calls = []

        def loader_factory(req):
            calls.append((req.stage, req.batch_size, req.shuffle))
            return torch.utils.data.DataLoader(
                req.dataset,
                batch_size=req.batch_size,
                shuffle=req.shuffle,
                num_workers=req.num_workers,
                pin_memory=req.pin_memory,
                drop_last=req.drop_last,
                **req.extra_kwargs,
            )

        det = TorchProbe(
            model=nn.Linear(20, 2),
            loss_fn=nn.CrossEntropyLoss(),
            metric="accuracy",
            threshold=0.55,
            epochs=3,
            batch_size=48,
            device="cpu",
            random_state=3,
            verbose=False,
            loader_factory=loader_factory,
            stage_loader_overrides={"predict": {"batch_size": 17}},
        )

        det.fit(embeddings.astype("float32"), group_labels.astype("int64"))
        _ = det.predict(embeddings[:30].astype("float32"))

        seen_stages = {stage for stage, _, _ in calls}
        assert {"train", "val", "eval", "predict"}.issubset(seen_stages)

        predict_calls = [c for c in calls if c[0] == "predict"]
        assert predict_calls, "expected at least one predict-stage loader call"
        assert predict_calls[0][1] == 17

    def test_torch_probe_fit_dataset_map_style(self):
        embeddings, group_labels = generate_linear_shortcut(
            n_samples=220, embedding_dim=18, shortcut_dims=2
        )

        class DictMapDataset(torch.utils.data.Dataset):
            def __init__(self, X, y):
                self.X = X.astype("float32")
                self.y = y.astype("int64")

            def __len__(self):
                return int(self.X.shape[0])

            def __getitem__(self, idx):
                return {"embeddings": self.X[idx], "labels": self.y[idx]}

        ds = DictMapDataset(embeddings, group_labels)
        det = TorchProbe(
            model=nn.Linear(18, 2),
            loss_fn=nn.CrossEntropyLoss(),
            metric="accuracy",
            threshold=0.55,
            epochs=3,
            batch_size=32,
            random_state=3,
            device="cpu",
        )
        det.fit_dataset(ds)

        assert det._is_fitted is True
        assert det.results_["metrics"]["protocol"] == "loader"
        assert det.results_["report"]["protocol"] == "loader"

    def test_torch_probe_fit_dataset_iterable_with_val(self):
        embeddings, group_labels = generate_linear_shortcut(
            n_samples=200, embedding_dim=16, shortcut_dims=2
        )
        X = embeddings.astype("float32")
        y = group_labels.astype("int64")
        split = 140

        class TupleIterable(torch.utils.data.IterableDataset):
            def __init__(self, X_part, y_part):
                self.X_part = X_part
                self.y_part = y_part

            def __iter__(self):
                for i in range(self.X_part.shape[0]):
                    yield self.X_part[i], self.y_part[i]

        train_ds = TupleIterable(X[:split], y[:split])
        val_ds = TupleIterable(X[split:], y[split:])
        det = TorchProbe(
            model=nn.Linear(16, 2),
            loss_fn=nn.CrossEntropyLoss(),
            metric="accuracy",
            threshold=0.55,
            epochs=2,
            batch_size=32,
            random_state=3,
            device="cpu",
        )
        det.fit_dataset(
            train_ds,
            val_dataset=val_ds,
            data_spec={
                "n_features": 16,
                "n_classes": 2,
                "train_size": split,
                "val_size": len(y) - split,
            },
        )

        assert det._is_fitted is True
        assert det.results_["report"]["protocol"] == "loader"

    def test_shortcutdetector_probe_loader_native_path(self):
        embeddings, group_labels = generate_linear_shortcut(
            n_samples=220, embedding_dim=14, shortcut_dims=2
        )

        class DictMapDataset(torch.utils.data.Dataset):
            def __init__(self, X, y):
                self.X = X.astype("float32")
                self.y = y.astype("int64")

            def __len__(self):
                return int(self.X.shape[0])

            def __getitem__(self, idx):
                return {"embeddings": self.X[idx], "labels": self.y[idx]}

        det = ShortcutDetector(
            methods=["probe"],
            probe_backend="torch",
            probe_model=nn.Linear(14, 2),
            probe_loss_fn=nn.CrossEntropyLoss(),
            probe_threshold=0.55,
            probe_epochs=2,
            probe_batch_size=32,
            probe_device="cpu",
            seed=3,
        )
        det.fit_from_loaders({"probe": {"train_dataset": DictMapDataset(embeddings, group_labels)}})
        out = det.results_["probe"]
        assert out["success"] is True
        assert out["results"]["report"]["protocol"] == "loader"

    def test_shortcutdetector_probe_callable_fallback_called_once(self):
        embeddings, group_labels = generate_linear_shortcut(
            n_samples=200, embedding_dim=12, shortcut_dims=2
        )
        calls = {"n": 0}

        def one_shot_loader():
            calls["n"] += 1
            return {
                "embeddings": embeddings.astype("float32"),
                "labels": group_labels.astype("int64"),
            }

        det = ShortcutDetector(
            methods=["probe"],
            probe_backend="torch",
            probe_model=nn.Linear(12, 2),
            probe_loss_fn=nn.CrossEntropyLoss(),
            probe_threshold=0.55,
            probe_epochs=2,
            probe_batch_size=32,
            probe_device="cpu",
            seed=7,
        )
        det.fit_from_loaders({"probe": one_shot_loader})

        assert calls["n"] == 1
        assert det.results_["probe"]["success"] is True
