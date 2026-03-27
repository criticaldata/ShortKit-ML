"""Tests for multi-attribute support."""

import os
import tempfile

import numpy as np

from shortcut_detect import ShortcutDetector
from shortcut_detect.reporting import ReportBuilder, export_to_csv
from shortcut_detect.unified import _get_attribute_sources


def test_get_attribute_sources_group_only():
    """Test _get_attribute_sources with group_labels only."""
    group_labels = np.array(["A", "B", "A", "B"])
    sources = _get_attribute_sources(group_labels, None)
    assert sources == {"group": group_labels}


def test_get_attribute_sources_extra_only():
    """Test _get_attribute_sources with extra_labels only (no group_labels)."""
    extra = {"race": np.array(["Black", "White", "Black"]), "gender": np.array(["M", "F", "M"])}
    sources = _get_attribute_sources(None, extra)
    assert set(sources.keys()) == {"race", "gender"}
    assert list(sources["race"]) == ["Black", "White", "Black"]


def test_get_attribute_sources_both():
    """Test _get_attribute_sources with both group_labels and extra_labels."""
    group_labels = np.array(["A", "B"])
    extra = {"race": np.array(["Black", "White"]), "gender": np.array(["M", "F"])}
    sources = _get_attribute_sources(group_labels, extra)
    assert set(sources.keys()) == {"group", "race", "gender"}


def test_get_attribute_sources_reserved_excluded():
    """Test that reserved keys (spurious, early_epoch_reps) are excluded."""
    extra = {
        "race": np.array([1, 2]),
        "spurious": np.array([0, 1]),
        "early_epoch_reps": np.zeros((2, 2)),
    }
    sources = _get_attribute_sources(None, extra)
    assert "race" in sources
    assert "spurious" not in sources
    assert "early_epoch_reps" not in sources


def _make_multi_attr_dataset(n=200, seed=42):
    """Create dataset with race and gender attributes for multi-attribute testing."""
    rng = np.random.RandomState(seed)
    race = np.array(["A"] * (n // 2) + ["B"] * (n // 2))
    gender = np.array(["M", "F"] * (n // 2))
    labels = rng.binomial(1, 0.5, n)
    group_labels = race  # primary group
    embeddings = rng.randn(n, 10)
    embeddings[:, 0] += (labels * 2) + (np.array(race == "A", dtype=float) * 0.5)
    extra_labels = {"race": race, "gender": gender}
    return embeddings, labels, group_labels, extra_labels


def test_multi_attribute_equalized_odds():
    """Test that with 2+ attributes, equalized_odds produces by_attribute structure."""
    embeddings, labels, group_labels, extra_labels = _make_multi_attr_dataset()
    detector = ShortcutDetector(methods=["equalized_odds"], seed=42)
    detector.fit(embeddings, labels, group_labels=group_labels, extra_labels=extra_labels)

    assert "equalized_odds" in detector.results_
    result = detector.results_["equalized_odds"]
    assert result["success"]
    assert "by_attribute" in result
    by_attr = result["by_attribute"]
    assert "group" in by_attr or "race" in by_attr or "gender" in by_attr
    for _attr_name, sub in by_attr.items():
        assert "success" in sub
        if sub["success"]:
            assert "report" in sub
            assert hasattr(sub["report"], "tpr_gap")


def test_single_attribute_unchanged():
    """Test that with 1 attribute, no by_attribute structure (backward compat)."""
    embeddings, labels, group_labels, _ = _make_multi_attr_dataset()
    detector = ShortcutDetector(methods=["equalized_odds"], seed=42)
    detector.fit(embeddings, labels, group_labels=group_labels)

    assert "equalized_odds" in detector.results_
    result = detector.results_["equalized_odds"]
    assert result["success"]
    assert "by_attribute" not in result
    assert "report" in result
    assert hasattr(result["report"], "tpr_gap")


def test_report_builder_by_attribute():
    """Test that ReportBuilder renders per-attribute sections."""
    embeddings, labels, group_labels, extra_labels = _make_multi_attr_dataset()
    detector = ShortcutDetector(methods=["equalized_odds"], seed=42)
    detector.fit(embeddings, labels, group_labels=group_labels, extra_labels=extra_labels)

    builder = ReportBuilder(detector)
    md = builder._generate_markdown()
    assert "Fairness (Equalized Odds)" in md
    assert "Attribute" in md or "group" in md or "race" in md


def test_csv_export_by_attribute():
    """Test that CSV export includes per-attribute columns."""
    embeddings, labels, group_labels, extra_labels = _make_multi_attr_dataset()
    detector = ShortcutDetector(methods=["equalized_odds"], seed=42)
    detector.fit(embeddings, labels, group_labels=group_labels, extra_labels=extra_labels)

    with tempfile.TemporaryDirectory() as tmp_dir:
        files = export_to_csv(detector, tmp_dir)
        summary_path = os.path.join(tmp_dir, "overall_summary.csv")
        assert os.path.exists(summary_path)
        import pandas as pd

        df = pd.read_csv(summary_path)
        cols = list(df.columns)
        assert any("equalized_odds" in c for c in cols)
        if "equalized_odds_per_attribute.csv" in str(files.values()):
            per_attr_path = [
                p
                for p in files.values()
                if "equalized_odds" in str(p) and "per_attribute" in str(p)
            ]
            if per_attr_path:
                per_df = pd.read_csv(per_attr_path[0])
                assert "attribute" in per_df.columns or "group" in per_df.columns
