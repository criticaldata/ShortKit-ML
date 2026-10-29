#!/usr/bin/env python3
"""
Quick test to verify dashboard sample data loading works
"""

import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from app import load_custom_csv, load_sample_data  # noqa: E402


def test_load_sample_data():
    """Verify dashboard sample data loads with correct shapes and columns."""
    embeddings, task_labels, group_labels, extra_labels, attributes, metadata_df = (
        load_sample_data()
    )

    assert len(embeddings) > 0
    assert embeddings.shape[1] > 0
    assert len(task_labels) == len(embeddings)
    assert set(task_labels) <= {0, 1, -1}  # -1 = uncertain in CheXpert
    assert len(group_labels) == len(embeddings)
    assert len(metadata_df) == len(embeddings)
    assert list(metadata_df.columns)

    unique, counts = np.unique(group_labels, return_counts=True)
    assert len(unique) >= 1
    for _group, count in zip(unique, counts, strict=False):
        assert count > 0
    assert attributes is not None
    assert "race" in attributes
    assert "gender" in attributes


def test_load_custom_csv_rejects_semicolon_delimiter(tmp_path):
    bad_csv = tmp_path / "bad_delimiter.csv"
    bad_csv.write_text("embedding_0;embedding_1;task_label;group_label\n0.1;0.2;1;A\n")

    with pytest.raises(ValueError, match="wrong delimiter|comma-separated"):
        load_custom_csv(str(bad_csv), is_raw_data=False)


def test_load_custom_csv_rejects_non_numeric_embeddings(tmp_path):
    bad_df = pd.DataFrame(
        {
            "embedding_0": [0.1, 0.2],
            "embedding_1": ["bad", "0.4"],
            "task_label": [1, 0],
            "group_label": ["A", "B"],
        }
    )
    bad_csv = tmp_path / "non_numeric_embeddings.csv"
    bad_df.to_csv(bad_csv, index=False)

    with pytest.raises(ValueError, match="Embedding columns must be numeric"):
        load_custom_csv(str(bad_csv), is_raw_data=False)


def test_load_custom_csv_rejects_empty_text_rows(tmp_path):
    bad_df = pd.DataFrame(
        {
            "text": ["valid row", "   "],
            "task_label": [1, 0],
            "group_label": ["A", "B"],
        }
    )
    bad_csv = tmp_path / "empty_text.csv"
    bad_df.to_csv(bad_csv, index=False)

    with pytest.raises(ValueError, match="text.*empty values"):
        load_custom_csv(str(bad_csv), is_raw_data=True)
