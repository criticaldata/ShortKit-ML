"""The package is input-agnostic: any embedding width, any modality.

The only shape contract is a 2-D float matrix ``(n_samples, embedding_dim)``
plus 1-D label arrays (``validate_embeddings_labels`` in ``utils.py``). These
tests pin that down for the legacy ``ShortcutDetector`` and for the paper's
``conditional_audit`` across widths from 1 to 3072, for float32, float64 and
plain nested lists, and check that every API which takes a second matrix after
``fit`` rejects a width mismatch with a clear ``ValueError``.
"""

import numpy as np
import pytest

from shortcut_detect import ShortcutDetector
from shortcut_detect.audit import AuditConfig, conditional_audit

WIDTHS = [1, 2, 3, 17, 129, 1024, 3072]
METHODS = ["hbac", "probe", "statistical", "geometric"]

# few permutations, no bootstrap, one CV repeat: the cheapest probe-only audit
CHEAP_AUDIT = AuditConfig(
    n_permutations=5, n_bootstrap=0, detectors=("probe",), cv_repeats=1, max_iter=50, seed=0
)


def _planted(width, n=300, shift=2.0, n_signal_dims=1, seed=0):
    """Random embeddings with a group shift planted in the first columns."""
    rng = np.random.default_rng(seed)
    y = rng.integers(0, 2, n)
    g = rng.integers(0, 2, n)
    X = rng.standard_normal((n, width))
    k = min(n_signal_dims, width)
    X[:, :k] += shift * g[:, None]
    return X, y, g


class TestShortcutDetectorAnyWidth:
    """(a) The four legacy methods fit and report success at every width."""

    @pytest.mark.parametrize("width", WIDTHS)
    def test_all_methods_succeed(self, width):
        X, y, g = _planted(width)
        detector = ShortcutDetector(methods=METHODS, seed=0)
        detector.fit(X, y, group_labels=g)
        assert detector.embeddings_.shape == (300, width)
        for method in METHODS:
            result = detector.results_[method]
            assert result["success"], (width, method, result.get("error"))
        assert f"{width} dimensions" in detector.summary()


class TestConditionalAuditAnyWidth:
    """(b) The paper's primary API returns a probe AUROC in [0, 1] at any width
    and dtype; a nested Python list is as good as an ndarray."""

    @pytest.mark.parametrize("dtype", [np.float32, np.float64])
    @pytest.mark.parametrize("width", [2, 64, 1500])
    def test_probe_statistic_in_unit_interval(self, width, dtype):
        X, y, g = _planted(width, shift=1.0)
        res = conditional_audit(X.astype(dtype), g, y, config=CHEAP_AUDIT)
        probe = res.results["probe"]
        assert 0.0 <= probe.statistic <= 1.0
        assert 0.0 < probe.p_value <= 1.0

    def test_float32_and_float64_agree(self):
        X, y, g = _planted(16, shift=1.0)
        s32 = conditional_audit(X.astype(np.float32), g, y, config=CHEAP_AUDIT)
        s64 = conditional_audit(X.astype(np.float64), g, y, config=CHEAP_AUDIT)
        assert s32.results["probe"].statistic == pytest.approx(
            s64.results["probe"].statistic, abs=1e-6
        )

    def test_nested_python_list_matches_ndarray(self):
        X, y, g = _planted(8, shift=1.0)
        as_list = X.tolist()
        assert isinstance(as_list, list) and isinstance(as_list[0], list)
        res_list = conditional_audit(as_list, g.tolist(), y.tolist(), config=CHEAP_AUDIT)
        res_arr = conditional_audit(X, g, y, config=CHEAP_AUDIT)
        assert res_list.results["probe"].statistic == pytest.approx(
            res_arr.results["probe"].statistic, abs=1e-6
        )


class TestProvenanceDoesNotMatter:
    """(c) A 384-wide 'sentence embedding' and an 8-wide 'tabular feature
    vector' carrying the same planted attribute shift are both flagged. The
    package never sees where the matrix came from."""

    @pytest.mark.parametrize("modality,width", [("text-like", 384), ("tabular-like", 8)])
    def test_planted_attribute_is_flagged(self, modality, width):
        X, y, g = _planted(width, shift=1.5, n_signal_dims=4, seed=1)
        cfg = AuditConfig(
            n_permutations=39, n_bootstrap=0, detectors=("probe",), cv_repeats=1, max_iter=50
        )
        res = conditional_audit(X, g, y, config=cfg)
        assert res.results["probe"].flag, modality
        # the legacy detector predicts the attribute from the matrix alone
        detector = ShortcutDetector(methods=["probe"], seed=0).fit(X, g)
        assert detector.results_["probe"]["results"]["metrics"]["metric_value"] > 0.7, modality


class TestWidthMismatchAfterFit:
    """(d) APIs that take a second matrix after fit reject a different width
    with a ValueError that names both widths, not a numpy or torch shape error."""

    TRAIN_WIDTH = 8
    HELD_OUT_WIDTH = 7

    def _train(self):
        return _planted(self.TRAIN_WIDTH, n=40, seed=2)

    def _held_out(self):
        return np.ones((5, self.HELD_OUT_WIDTH))

    def _assert_clear_error(self, call):
        with pytest.raises(ValueError, match=r"\b7\b.*\b8\b"):
            call()

    def test_adversarial_debiasing_transform(self):
        from shortcut_detect.mitigation import AdversarialDebiasing

        X, y, g = self._train()
        model = AdversarialDebiasing(hidden_dim=4, n_epochs=1, batch_size=16, random_state=0)
        model.fit(X, g, task_labels=y)
        self._assert_clear_error(lambda: model.transform(self._held_out()))

    def test_contrastive_debiasing_transform(self):
        from shortcut_detect.mitigation import ContrastiveDebiasing

        X, y, g = self._train()
        model = ContrastiveDebiasing(hidden_dim=4, n_epochs=1, batch_size=16, random_state=0)
        model.fit(X, y, g)
        self._assert_clear_error(lambda: model.transform(self._held_out()))

    def test_last_layer_retraining_predict(self):
        from shortcut_detect.mitigation import LastLayerRetraining

        X, y, g = self._train()
        model = LastLayerRetraining(random_state=0).fit(X, y, g)
        self._assert_clear_error(lambda: model.predict(self._held_out()))

    def test_gce_detector_predict(self):
        from shortcut_detect.gce import GCEDetector

        X, y, _ = self._train()
        model = GCEDetector(max_iter=50).fit(X, y)
        self._assert_clear_error(lambda: model.predict(self._held_out()))

    def test_sklearn_probe_predict(self):
        from sklearn.linear_model import LogisticRegression

        from shortcut_detect import SKLearnProbe

        X, _, g = self._train()
        probe = SKLearnProbe(LogisticRegression(max_iter=200)).fit(X, g)
        self._assert_clear_error(lambda: probe.predict(self._held_out()))

    def test_torch_probe_predict(self):
        import torch

        from shortcut_detect import TorchProbe

        X, _, g = self._train()
        probe = TorchProbe(
            torch.nn.Linear(self.TRAIN_WIDTH, 2),
            torch.nn.CrossEntropyLoss(),
            epochs=1,
            device="cpu",
        ).fit(X, g)
        self._assert_clear_error(lambda: probe.predict(self._held_out()))
