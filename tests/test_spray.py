"""Tests for SpRAy heatmap clustering."""

import numpy as np

from shortcut_detect import SpRAyDetector


def _make_blob(center, size=32, sigma=4.0):
    yy, xx = np.mgrid[0:size, 0:size]
    cy, cx = center
    blob = np.exp(-((yy - cy) ** 2 + (xx - cx) ** 2) / (2 * sigma**2))
    return blob.astype(np.float32)


def _synthetic_heatmaps(n_per_cluster=20, size=32):
    blob_a = _make_blob((8, 8), size=size)
    blob_b = _make_blob((24, 24), size=size)
    noise_a = np.random.normal(0, 0.05, size=(n_per_cluster, size, size)).astype(np.float32)
    noise_b = np.random.normal(0, 0.05, size=(n_per_cluster, size, size)).astype(np.float32)
    heatmaps_a = np.clip(blob_a + noise_a, 0.0, 1.0)
    heatmaps_b = np.clip(blob_b + noise_b, 0.0, 1.0)
    heatmaps = np.concatenate([heatmaps_a, heatmaps_b], axis=0)
    labels = np.array([0] * n_per_cluster + [1] * n_per_cluster)
    return heatmaps, labels


def test_spray_clustering_fixed_k():
    heatmaps, labels = _synthetic_heatmaps()
    detector = SpRAyDetector(
        n_clusters=2,
        cluster_selection="fixed",
        affinity="cosine",
        downsample_size=16,
        small_cluster_threshold=0.01,
    )
    detector.fit(heatmaps=heatmaps, labels=labels)
    report = detector.get_report()

    assert report["metrics"]["n_clusters"] == 2
    assert len(report["report"]["clusters"]) == 2
    assert len(detector.representative_heatmaps_) == 2


def test_spray_generator_path():
    heatmaps, labels = _synthetic_heatmaps()

    def generator(_inputs):
        return heatmaps

    detector = SpRAyDetector(
        n_clusters=2,
        cluster_selection="fixed",
        affinity="cosine",
        downsample_size=16,
    )
    detector.fit(inputs=[1, 2, 3], heatmap_generator=generator, labels=labels)
    report = detector.get_report()

    assert report["metrics"]["n_clusters"] == 2
