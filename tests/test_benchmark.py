"""Tests for benchmark runner and artifact generation."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from shortcut_detect.benchmark.runner import BenchmarkConfig, BenchmarkRunner


def _write_chest_arrays(tmp_path: Path, n: int = 64, d: int = 16) -> tuple[Path, Path, Path]:
    rng = np.random.RandomState(123)
    embeddings = rng.randn(n, d).astype(np.float32)
    labels = rng.randint(0, 2, size=n).astype(np.int64)
    groups = rng.randint(0, 2, size=n).astype(np.int64)

    emb_path = tmp_path / "chest_embeddings.npy"
    lbl_path = tmp_path / "chest_labels.npy"
    grp_path = tmp_path / "chest_groups.npy"
    np.save(emb_path, embeddings)
    np.save(lbl_path, labels)
    np.save(grp_path, groups)
    return emb_path, lbl_path, grp_path


def test_benchmark_config_requires_chest_paths(tmp_path: Path):
    cfg_dict = {
        "methods": ["probe"],
        "datasets": {
            "synthetic": {"enabled": False},
            "chest_xray": {"enabled": True, "n_seeds": 2},
        },
        "output_dir": str(tmp_path / "out"),
    }
    with pytest.raises(ValueError, match="embeddings_path"):
        BenchmarkConfig.from_dict(cfg_dict)


def test_benchmark_runner_smoke_outputs_artifacts(tmp_path: Path):
    emb_path, lbl_path, grp_path = _write_chest_arrays(tmp_path)
    out_dir = tmp_path / "benchmark_out"

    cfg = BenchmarkConfig.from_dict(
        {
            "benchmark_name": "smoke",
            "methods": ["hbac", "probe"],
            "primary_endpoint": "probe_metric_value",
            "random_seed": 7,
            "datasets": {
                "synthetic": {
                    "enabled": True,
                    "n_seeds": 3,
                    "n_samples": 80,
                    "embedding_dim": 12,
                    "shortcut_dims": 2,
                    "effect_size": 0.8,
                },
                "chest_xray": {
                    "enabled": True,
                    "n_seeds": 2,
                    "embeddings_path": str(emb_path),
                    "labels_path": str(lbl_path),
                    "group_labels_path": str(grp_path),
                },
            },
            "split_policy": {"policy": "seeded_holdout", "test_size": 0.25},
            "stats": {
                "paired_tests": True,
                "multiple_testing": "fdr_bh",
                "ci_method": "bootstrap",
                "bootstrap_samples": 200,
            },
            "failure_policy": "continue_record",
            "output_dir": str(out_dir),
        }
    )

    runner = BenchmarkRunner(cfg)
    result = runner.run()

    expected_runs = (3 + 2) * 2
    assert result["runs"].shape[0] == expected_runs
    assert (out_dir / "runs.jsonl").exists()
    assert (out_dir / "aggregate_by_method.csv").exists()
    assert (out_dir / "primary_endpoint_summary.csv").exists()
    assert (out_dir / "paired_tests.csv").exists()
    assert (out_dir / "run_manifest.json").exists()

    # JSONL line count matches expected rows.
    line_count = len((out_dir / "runs.jsonl").read_text().strip().splitlines())
    assert line_count == expected_runs

    # Verify manifest contains config and git_sha field.
    manifest = json.loads((out_dir / "run_manifest.json").read_text())
    assert manifest["benchmark_name"] == "smoke"
    assert "git_sha" in manifest
    assert manifest["config"]["split_policy"]["test_size"] == 0.25
    assert manifest["config"]["datasets"]["synthetic"]["effect_size"] == 0.8

    agg = pd.read_csv(out_dir / "aggregate_by_method.csv")
    assert set(agg["method"].unique()) == {"hbac", "probe"}
    assert set(agg["dataset_name"].unique()) == {"synthetic", "chest_xray"}
    assert "failure_rate" in agg.columns

    primary = pd.read_csv(out_dir / "primary_endpoint_summary.csv")
    assert set(primary["dataset_name"].unique()) == {"synthetic", "chest_xray"}
    assert {"mean", "std", "ci95_low", "ci95_high"}.issubset(primary.columns)

    paired = pd.read_csv(out_dir / "paired_tests.csv")
    assert {"dataset_name", "endpoint", "method_a", "method_b", "q_value"}.issubset(paired.columns)


def test_benchmark_no_probe_methods_does_not_crash_primary_summary(tmp_path: Path):
    out_dir = tmp_path / "benchmark_no_probe"
    cfg = BenchmarkConfig.from_dict(
        {
            "benchmark_name": "no_probe",
            "methods": ["hbac"],
            # Keep default endpoint (probe_metric_value) to verify empty probe stats do not crash.
            "random_seed": 11,
            "datasets": {
                "synthetic": {
                    "enabled": True,
                    "n_seeds": 2,
                    "n_samples": 64,
                    "embedding_dim": 10,
                    "shortcut_dims": 2,
                },
                "chest_xray": {"enabled": False},
            },
            "output_dir": str(out_dir),
        }
    )

    runner = BenchmarkRunner(cfg)
    result = runner.run()

    assert result["runs"].shape[0] == 2
    primary = pd.read_csv(out_dir / "primary_endpoint_summary.csv")
    assert set(primary["dataset_name"].unique()) == {"synthetic"}
    assert int(primary.loc[0, "n"]) == 0


def test_benchmark_primary_endpoint_and_paired_tests_flag(tmp_path: Path):
    out_dir = tmp_path / "benchmark_primary_and_pairs"
    cfg = BenchmarkConfig.from_dict(
        {
            "benchmark_name": "primary_hbac",
            "methods": ["hbac", "statistical"],
            "primary_endpoint": "hbac_confidence_score",
            "random_seed": 13,
            "datasets": {
                "synthetic": {
                    "enabled": True,
                    "n_seeds": 3,
                    "n_samples": 72,
                    "embedding_dim": 12,
                    "shortcut_dims": 2,
                },
                "chest_xray": {"enabled": False},
            },
            "stats": {"paired_tests": False, "bootstrap_samples": 100},
            "output_dir": str(out_dir),
        }
    )

    runner = BenchmarkRunner(cfg)
    result = runner.run()

    primary = pd.read_csv(out_dir / "primary_endpoint_summary.csv")
    assert set(primary["endpoint"].unique()) == {"hbac_confidence_score"}
    assert (primary["n"] > 0).all()
    assert result["paired_tests"].empty
    assert not (out_dir / "paired_tests.csv").exists()
