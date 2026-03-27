"""Tests for paper benchmark runner and CheXpert extraction helpers."""

from __future__ import annotations

from pathlib import Path

import pandas as pd
import pytest

from shortcut_detect.benchmark.chexpert_extraction import load_and_validate_manifest
from shortcut_detect.benchmark.paper_runner import PaperBenchmarkConfig, PaperBenchmarkRunner


def _write_manifest(path: Path) -> None:
    df = pd.DataFrame(
        {
            "image_path": ["/tmp/a.png", "/tmp/b.png"],
            "task_label": [0, 1],
            "race": ["ASIAN", "BLACK"],
            "sex": ["Male", "Female"],
            "age": [45, 67],
        }
    )
    df.to_csv(path, index=False)


def test_manifest_validation_happy_path(tmp_path: Path):
    manifest = tmp_path / "manifest.csv"
    _write_manifest(manifest)
    df = load_and_validate_manifest(manifest)
    assert len(df) == 2
    assert set(df.columns) >= {"image_path", "task_label", "race", "sex", "age"}


def test_manifest_validation_missing_columns(tmp_path: Path):
    manifest = tmp_path / "bad_manifest.csv"
    pd.DataFrame({"image_path": ["x"], "task_label": [1]}).to_csv(manifest, index=False)
    with pytest.raises(ValueError, match="missing required columns"):
        load_and_validate_manifest(manifest)


def test_paper_config_requires_chexpert_manifest_when_enabled():
    with pytest.raises(ValueError, match="manifest_path"):
        PaperBenchmarkConfig.from_dict(
            {
                "profile": "smoke",
                "methods": ["probe"],
                "chexpert": {"enabled": True},
            }
        )


def test_paper_runner_smoke_synthetic_outputs(tmp_path: Path):
    cfg = PaperBenchmarkConfig.from_dict(
        {
            "benchmark_name": "paper_smoke",
            "profile": "smoke",
            "methods": ["probe", "statistical"],
            "chexpert": {"enabled": False},
            "output_dir": str(tmp_path / "out"),
        }
    )
    runner = PaperBenchmarkRunner(cfg)
    result = runner.run()

    assert not result["synthetic_runs"].empty
    assert not result["synthetic_fp"].empty
    assert not result["correction"].empty

    out_dir = Path(cfg.output_dir)
    assert (out_dir / "runs_paper.csv").exists()
    assert (out_dir / "synthetic_dim_pr.csv").exists()
    assert (out_dir / "synthetic_fp_control.csv").exists()
    assert (out_dir / "synthetic_correction_control.csv").exists()
    assert (out_dir / "run_manifest.json").exists()
    assert (out_dir / "paper_summary.md").exists()
    assert (out_dir / "figures" / "synthetic_recall_by_effect_size.png").exists()
