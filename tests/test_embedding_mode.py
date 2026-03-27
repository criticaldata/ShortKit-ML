import warnings

import numpy as np

from shortcut_detect import ShortcutDetector
from shortcut_detect.embedding_sources import EmbeddingSource


class DummySource(EmbeddingSource):
    def __init__(self):
        super().__init__(name="dummy")
        self.calls = 0

    def generate(self, inputs):
        self.calls += 1
        data = []
        for item in inputs:
            value = len(str(item))
            data.append([value, value % 2, float(value > 3)])
        return np.array(data, dtype=np.float32)


def test_embedding_only_mode(tmp_path):
    texts = ["aa", "bbb", "cccc", "ddddd"]
    labels = np.array([0, 1, 1, 0])
    source = DummySource()
    cache_path = tmp_path / "embeddings.npy"

    detector = ShortcutDetector(methods=["probe"])
    # Probe may warn due to tiny dataset (test_size < n_classes)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", UserWarning)
        detector.fit(
            embeddings=None,
            labels=labels,
            group_labels=labels,
            raw_inputs=texts,
            embedding_source=source,
            embedding_cache_path=str(cache_path),
        )

    assert detector.embeddings_.shape == (len(texts), 3)
    assert source.calls == 1
    assert cache_path.exists()
    assert detector.embedding_metadata_["mode"] == "embedding-only"
    assert detector.embedding_source_.name == "dummy"


def test_embedding_only_cache_reuse(tmp_path):
    texts = ["aa", "bbb", "cccc", "ddddd"]
    labels = np.array([0, 1, 1, 0])
    source = DummySource()
    cache_path = tmp_path / "embeddings.npy"

    # First run populates cache
    # Probe may warn due to tiny dataset (test_size < n_classes)
    detector = ShortcutDetector(methods=["probe"])
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", UserWarning)
        detector.fit(
            embeddings=None,
            labels=labels,
            group_labels=labels,
            raw_inputs=texts,
            embedding_source=source,
            embedding_cache_path=str(cache_path),
        )
    assert source.calls == 1

    # Second run should reuse cached embeddings
    detector2 = ShortcutDetector(methods=["probe"])

    with warnings.catch_warnings():
        warnings.simplefilter("ignore", UserWarning)
        detector2.fit(
            embeddings=None,
            labels=labels,
            group_labels=labels,
            raw_inputs=texts,
            embedding_source=source,
            embedding_cache_path=str(cache_path),
        )

    assert source.calls == 1  # cache hit, no new API call
    assert detector2.embedding_metadata_["cached"] is True
