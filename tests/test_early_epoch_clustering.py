import numpy as np

from shortcut_detect.training import EarlyEpochClusteringDetector
from shortcut_detect.unified import ShortcutDetector


def _make_clustered_data(n_major=90, n_minor=10, dim=4, seed=0):
    rng = np.random.default_rng(seed)
    major = rng.normal(0.0, 0.5, size=(n_major, dim))
    minor = rng.normal(5.0, 0.5, size=(n_minor, dim))
    reps = np.vstack([major, minor])
    labels = np.array([0] * n_major + [1] * n_minor)
    return reps, labels


def test_early_epoch_clustering_detects_imbalance():
    reps, labels = _make_clustered_data()
    detector = EarlyEpochClusteringDetector(
        n_clusters=2,
        min_cluster_ratio=0.3,
        entropy_threshold=0.8,
        random_state=0,
    )
    detector.fit(reps, labels=labels, n_epochs=1)

    report = detector.report_
    assert report is not None
    assert report.n_clusters == 2
    assert report.minority_ratio < 0.3
    assert report.risk_level in {"moderate", "high"}


def test_early_epoch_clustering_balanced_low_risk():
    rng = np.random.default_rng(1)
    reps = np.vstack(
        [
            rng.normal(0.0, 0.5, size=(50, 3)),
            rng.normal(3.0, 0.5, size=(50, 3)),
        ]
    )
    labels = np.array([0] * 50 + [1] * 50)

    detector = EarlyEpochClusteringDetector(
        n_clusters=2,
        min_cluster_ratio=0.2,
        entropy_threshold=0.7,
        random_state=1,
    )
    detector.fit(reps, labels=labels, n_epochs=1)
    report = detector.report_
    assert report is not None
    assert report.risk_level == "low"


def test_shortcut_detector_integration():
    reps, labels = _make_clustered_data()
    detector = ShortcutDetector(methods=["early_epoch_clustering"], eec_n_clusters=2)
    detector.fit(reps, labels)

    results = detector.get_results()
    assert "early_epoch_clustering" in results
    assert results["early_epoch_clustering"]["success"] is True
