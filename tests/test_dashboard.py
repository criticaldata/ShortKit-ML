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

from app import load_audit_sample, load_custom_csv, load_sample_data, run_audit_tab  # noqa: E402


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


def _audit_inputs(tmp_path, n=300, d=16, seed=0):
    """Synthetic embeddings with the attribute planted in the first dims, plus a labels CSV."""
    rng = np.random.default_rng(seed)
    a = rng.integers(0, 2, n)
    y = np.where(rng.random(n) < 0.8, a, 1 - a)
    E = rng.standard_normal((n, d)).astype(np.float32)
    E[:, :3] += 1.5 * a[:, None]
    emb_path = tmp_path / "embeddings.npy"
    labels_path = tmp_path / "labels.csv"
    np.save(emb_path, E)
    pd.DataFrame({"sex": a, "disease": y}).to_csv(labels_path, index=False)
    return str(emb_path), str(labels_path)


def test_run_audit_tab_returns_summary_table_figure_and_reports(tmp_path):
    emb_path, labels_path = _audit_inputs(tmp_path)

    summary, table, png_path, html_path, pdf_path = run_audit_tab(
        emb_path,
        labels_path,
        attribute_col="sex",
        task_col="disease",
        n_permutations=8,
        n_permutations_probe=8,
        seed=1,
        attribute_name="patient sex",
        task_name="disease label",
    )

    assert "patient sex" in summary
    assert "disease label" in summary
    assert len(table) == 6
    assert list(table.columns) == [
        "detector",
        "statistic",
        "ci_low",
        "ci_high",
        "null_mean",
        "null_sd",
        "p_value",
        "p_bh",
        "flag",
    ]
    assert table["detector"].iloc[0] == "probe"
    assert Path(png_path).read_bytes()[:8] == b"\x89PNG\r\n\x1a\n"
    assert html_path.endswith(".html") and Path(html_path).stat().st_size > 0
    assert pdf_path.endswith(".pdf") and Path(pdf_path).read_bytes()[:4] == b"%PDF"
    assert "patient sex" in Path(html_path).read_text(encoding="utf-8")


def test_run_audit_tab_reads_labels_from_embeddings_csv(tmp_path):
    rng = np.random.default_rng(0)
    n = 120
    a = rng.integers(0, 2, n)
    df = pd.DataFrame(rng.standard_normal((n, 8)), columns=[f"embedding_{i}" for i in range(8)])
    df["embedding_0"] += 2.0 * a
    df["group_label"] = a
    df["task_label"] = np.where(rng.random(n) < 0.8, a, 1 - a)
    csv_path = tmp_path / "one_file.csv"
    df.to_csv(csv_path, index=False)

    summary, table, png_path, html_path, pdf_path = run_audit_tab(
        str(csv_path), None, n_permutations=4, n_permutations_probe=4, seed=0
    )

    assert "group_label" in summary
    assert len(table) == 6
    assert Path(pdf_path).exists()


def test_run_audit_tab_reports_errors_in_summary(tmp_path):
    emb_path, labels_path = _audit_inputs(tmp_path, n=50)

    summary, table, png_path, html_path, pdf_path = run_audit_tab(
        emb_path, labels_path, attribute_col="missing", task_col="disease", n_permutations=4
    )

    assert summary.startswith("**Error:**")
    assert "missing" in summary
    assert table is None and png_path is None and html_path is None and pdf_path is None


def test_load_audit_sample_fills_audit_inputs():
    emb_path, labels_path, attr_col, task_col, groups_col, attr_name, task_name, status = (
        load_audit_sample(n=40)
    )

    E = np.load(emb_path)
    labels = pd.read_csv(labels_path)
    assert E.shape[0] == 40 and E.ndim == 2
    assert len(labels) == 40
    assert attr_col in labels.columns and task_col in labels.columns
    assert groups_col == ""
    assert attr_name and task_name
    assert "Run audit" in status
