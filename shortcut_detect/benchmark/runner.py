"""Benchmark runner for multi-seed detector evaluation."""

from __future__ import annotations

import argparse
import json
import os
import platform
import subprocess
import sys
import time
from collections.abc import Iterable
from dataclasses import dataclass, field
from datetime import datetime, timezone
from itertools import combinations
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from scipy.stats import wilcoxon
from statsmodels.stats.multitest import multipletests

from shortcut_detect.benchmark.method_utils import ALL_METHODS
from shortcut_detect.benchmark.synthetic import generate_parametric_shortcut_dataset
from shortcut_detect.unified import ShortcutDetector
from shortcut_detect.utils import set_seed

SUPPORTED_METHODS = set(ALL_METHODS)
RISK_ORDINAL = {"low": 0.0, "moderate": 1.0, "high": 2.0, "unknown": np.nan}
HBAC_CONFIDENCE = {"low": 0.0, "moderate": 1.0, "high": 2.0, "unknown": np.nan}


@dataclass
class DatasetConfig:
    enabled: bool = False
    n_seeds: int = 0
    n_samples: int = 2000
    embedding_dim: int = 128
    shortcut_dims: int = 5
    effect_size: float = 1.0
    embeddings_path: str | None = None
    labels_path: str | None = None
    group_labels_path: str | None = None


@dataclass
class StatsConfig:
    paired_tests: bool = True
    multiple_testing: str = "fdr_bh"
    ci_method: str = "bootstrap"
    bootstrap_samples: int = 2000


@dataclass
class SplitConfig:
    policy: str = "seeded_holdout"
    test_size: float = 0.2


@dataclass
class BenchmarkConfig:
    benchmark_name: str = "shortcut_benchmark"
    methods: list[str] = field(
        default_factory=lambda: ["hbac", "probe", "statistical", "geometric"]
    )
    primary_endpoint: str = "probe_metric_value"
    random_seed: int = 0
    synthetic: DatasetConfig = field(
        default_factory=lambda: DatasetConfig(enabled=True, n_seeds=30)
    )
    chest_xray: DatasetConfig = field(
        default_factory=lambda: DatasetConfig(enabled=True, n_seeds=20)
    )
    split: SplitConfig = field(default_factory=SplitConfig)
    stats: StatsConfig = field(default_factory=StatsConfig)
    failure_policy: str = "continue_record"
    output_dir: str = "output/benchmark"

    @staticmethod
    def _from_dataset_dict(raw: dict[str, Any]) -> DatasetConfig:
        return DatasetConfig(
            enabled=bool(raw.get("enabled", False)),
            n_seeds=int(raw.get("n_seeds", 0)),
            n_samples=int(raw.get("n_samples", 2000)),
            embedding_dim=int(raw.get("embedding_dim", 128)),
            shortcut_dims=int(raw.get("shortcut_dims", 5)),
            effect_size=float(raw.get("effect_size", 1.0)),
            embeddings_path=raw.get("embeddings_path"),
            labels_path=raw.get("labels_path"),
            group_labels_path=raw.get("group_labels_path"),
        )

    @classmethod
    def from_dict(cls, raw: dict[str, Any]) -> BenchmarkConfig:
        datasets = raw.get("datasets", {})
        split_raw = raw.get("split_policy", {})
        stats_raw = raw.get("stats", {})

        cfg = cls(
            benchmark_name=str(raw.get("benchmark_name", "shortcut_benchmark")),
            methods=[
                str(m) for m in raw.get("methods", ["hbac", "probe", "statistical", "geometric"])
            ],
            primary_endpoint=str(raw.get("primary_endpoint", "probe_metric_value")),
            random_seed=int(raw.get("random_seed", 0)),
            synthetic=cls._from_dataset_dict(
                datasets.get("synthetic", {"enabled": True, "n_seeds": 30})
            ),
            chest_xray=cls._from_dataset_dict(
                datasets.get("chest_xray", {"enabled": True, "n_seeds": 20})
            ),
            split=SplitConfig(
                policy=str(split_raw.get("policy", "seeded_holdout")),
                test_size=float(split_raw.get("test_size", 0.2)),
            ),
            stats=StatsConfig(
                paired_tests=bool(stats_raw.get("paired_tests", True)),
                multiple_testing=str(stats_raw.get("multiple_testing", "fdr_bh")),
                ci_method=str(stats_raw.get("ci_method", "bootstrap")),
                bootstrap_samples=int(stats_raw.get("bootstrap_samples", 2000)),
            ),
            failure_policy=str(raw.get("failure_policy", "continue_record")),
            output_dir=str(raw.get("output_dir", "output/benchmark")),
        )
        cfg.validate()
        return cfg

    @classmethod
    def from_path(cls, path: str | os.PathLike[str]) -> BenchmarkConfig:
        path_obj = Path(path)
        if not path_obj.exists():
            raise FileNotFoundError(f"Config not found: {path_obj}")

        suffix = path_obj.suffix.lower()
        raw: dict[str, Any]
        if suffix == ".json":
            raw = json.loads(path_obj.read_text())
        elif suffix in {".yaml", ".yml"}:
            try:
                import yaml  # type: ignore
            except Exception as exc:  # pragma: no cover
                raise ImportError(
                    "YAML config requires pyyaml; use JSON or install pyyaml."
                ) from exc
            raw = yaml.safe_load(path_obj.read_text()) or {}
        else:
            raise ValueError("Config file must be .json, .yaml, or .yml")
        if not isinstance(raw, dict):
            raise TypeError("Benchmark config root must be a JSON/YAML object")
        return cls.from_dict(raw)

    def validate(self) -> None:
        unknown = [m for m in self.methods if m not in SUPPORTED_METHODS]
        if unknown:
            raise ValueError(
                f"Unsupported methods in config: {unknown}. Supported: {sorted(SUPPORTED_METHODS)}"
            )
        if not self.methods:
            raise ValueError("methods must not be empty")
        if self.split.policy != "seeded_holdout":
            raise ValueError("Only split_policy 'seeded_holdout' is currently supported")
        if not (0.0 < self.split.test_size < 1.0):
            raise ValueError("split_policy.test_size must be in (0, 1)")
        if self.failure_policy != "continue_record":
            raise ValueError("Only failure_policy='continue_record' is currently supported")
        if self.stats.multiple_testing != "fdr_bh":
            raise ValueError("Only stats.multiple_testing='fdr_bh' is currently supported")

        any_enabled = False
        for name, ds in (("synthetic", self.synthetic), ("chest_xray", self.chest_xray)):
            if not ds.enabled:
                continue
            any_enabled = True
            if ds.n_seeds <= 0:
                raise ValueError(f"{name}.n_seeds must be > 0")
            if name == "synthetic":
                if ds.shortcut_dims <= 0:
                    raise ValueError("datasets.synthetic.shortcut_dims must be > 0")
                if ds.shortcut_dims > ds.embedding_dim:
                    raise ValueError("datasets.synthetic.shortcut_dims cannot exceed embedding_dim")
            if name == "chest_xray":
                required = [
                    ("embeddings_path", ds.embeddings_path),
                    ("labels_path", ds.labels_path),
                ]
                for field_name, value in required:
                    if not value:
                        raise ValueError(
                            f"datasets.chest_xray.{field_name} is required when enabled"
                        )
                    if not Path(value).exists():
                        raise FileNotFoundError(
                            f"datasets.chest_xray.{field_name} not found: {value}"
                        )
                if ds.group_labels_path and not Path(ds.group_labels_path).exists():
                    raise FileNotFoundError(
                        f"datasets.chest_xray.group_labels_path not found: {ds.group_labels_path}"
                    )
        if not any_enabled:
            raise ValueError("At least one dataset must be enabled")


def _bootstrap_ci(values: np.ndarray, n_bootstrap: int, alpha: float = 0.05) -> tuple[float, float]:
    vals = np.asarray(values, dtype=float)
    vals = vals[np.isfinite(vals)]
    if vals.size == 0:
        return (np.nan, np.nan)
    if vals.size == 1:
        v = float(vals[0])
        return (v, v)
    rng = np.random.RandomState(0)
    stats = np.empty(n_bootstrap, dtype=float)
    for i in range(n_bootstrap):
        sample = vals[rng.randint(0, vals.size, size=vals.size)]
        stats[i] = float(np.mean(sample))
    lo = float(np.percentile(stats, 100.0 * (alpha / 2.0)))
    hi = float(np.percentile(stats, 100.0 * (1.0 - alpha / 2.0)))
    return lo, hi


def _paired_rank_biserial(x: np.ndarray, y: np.ndarray) -> float:
    diff = np.asarray(x, dtype=float) - np.asarray(y, dtype=float)
    diff = diff[np.isfinite(diff)]
    diff = diff[diff != 0]
    if diff.size == 0:
        return np.nan
    n_pos = int(np.sum(diff > 0))
    n_neg = int(np.sum(diff < 0))
    return float((n_pos - n_neg) / diff.size)


def _paired_wilcoxon_pvalue(x: np.ndarray, y: np.ndarray) -> float:
    xx = np.asarray(x, dtype=float)
    yy = np.asarray(y, dtype=float)
    mask = np.isfinite(xx) & np.isfinite(yy)
    xx = xx[mask]
    yy = yy[mask]
    if xx.size < 2:
        return np.nan
    if np.allclose(xx, yy):
        return 1.0
    try:
        return float(wilcoxon(xx, yy, zero_method="wilcox", correction=False).pvalue)
    except ValueError:
        return np.nan


def _risk_ordinal_from_result(result: dict[str, Any]) -> float:
    level = str(result.get("risk_value") or result.get("risk_level") or "unknown").lower()
    return float(RISK_ORDINAL.get(level, np.nan))


def _extract_method_record(
    *,
    dataset_name: str,
    seed: int,
    method: str,
    detector_result: dict[str, Any],
    duration_sec: float,
    n_total: int,
    n_train: int,
    n_test: int,
) -> dict[str, Any]:
    success = bool(detector_result.get("success", False))
    row: dict[str, Any] = {
        "dataset_name": dataset_name,
        "seed": int(seed),
        "method": method,
        "status": "success" if success else "failed",
        "duration_sec": float(duration_sec),
        "n_total": int(n_total),
        "n_train": int(n_train),
        "n_test": int(n_test),
        "risk_level": str(
            detector_result.get("risk_value") or detector_result.get("risk_level") or "unknown"
        ),
        "risk_ordinal": np.nan,
        "error_message": None,
        "probe_metric_name": None,
        "probe_metric_value": np.nan,
        "hbac_shortcut_exists": None,
        "hbac_confidence_score": np.nan,
        "stat_n_significant_comparisons": np.nan,
        "geometric_num_high_effect_pairs": np.nan,
        "geometric_num_overlap_pairs": np.nan,
        "frequency_shortcut_detected": None,
        "bias_pca_projection_gap": np.nan,
        "sis_mean_sis_size": np.nan,
        "dp_gap": np.nan,
        "eo_tpr_gap": np.nan,
        "eo_fpr_gap": np.nan,
        "intersectional_tpr_gap": np.nan,
    }
    if not success:
        row["error_message"] = str(detector_result.get("error", "unknown error"))
        return row

    row["risk_ordinal"] = _risk_ordinal_from_result(detector_result)

    if method == "probe":
        results_payload = detector_result.get("results", {})
        metrics = results_payload.get("metrics", {})
        row["probe_metric_name"] = metrics.get("metric")
        row["probe_metric_value"] = float(metrics.get("metric_value", np.nan))
    elif method == "hbac":
        report = detector_result.get("report") or {}
        hs = report.get("has_shortcut", {}) if isinstance(report, dict) else {}
        confidence = str(hs.get("confidence", "unknown")).lower()
        row["hbac_shortcut_exists"] = hs.get("exists")
        row["hbac_confidence_score"] = float(HBAC_CONFIDENCE.get(confidence, np.nan))
    elif method == "statistical":
        sig = detector_result.get("significant_features", {})
        if isinstance(sig, dict):
            n_sig = sum(1 for val in sig.values() if val is not None and len(val) > 0)
            row["stat_n_significant_comparisons"] = float(n_sig)
    elif method == "geometric":
        summary = detector_result.get("summary", {})
        if isinstance(summary, dict):
            row["geometric_num_high_effect_pairs"] = float(
                summary.get("num_high_effect_pairs", np.nan)
            )
            row["geometric_num_overlap_pairs"] = float(summary.get("num_overlap_pairs", np.nan))
    elif method == "frequency":
        report = detector_result.get("report", {})
        if isinstance(report, dict):
            row["frequency_shortcut_detected"] = report.get("shortcut_detected")
    elif method == "bias_direction_pca":
        report = detector_result.get("report")
        if report is not None and hasattr(report, "projection_gap"):
            row["bias_pca_projection_gap"] = float(getattr(report, "projection_gap", np.nan))
    elif method == "sis":
        report = detector_result.get("report", {})
        if isinstance(report, dict):
            row["sis_mean_sis_size"] = float(report.get("mean_sis_size", np.nan))
    elif method == "demographic_parity":
        report = detector_result.get("report")
        if report is not None and hasattr(report, "dp_gap"):
            row["dp_gap"] = float(getattr(report, "dp_gap", np.nan))
    elif method == "equalized_odds":
        report = detector_result.get("report")
        if report is not None:
            if hasattr(report, "tpr_gap"):
                row["eo_tpr_gap"] = float(getattr(report, "tpr_gap", np.nan))
            if hasattr(report, "fpr_gap"):
                row["eo_fpr_gap"] = float(getattr(report, "fpr_gap", np.nan))
    elif method == "intersectional":
        report = detector_result.get("report")
        if report is not None and hasattr(report, "tpr_gap"):
            row["intersectional_tpr_gap"] = float(getattr(report, "tpr_gap", np.nan))

    return row


class BenchmarkRunner:
    """Runs benchmark experiments and writes standardized artifacts."""

    def __init__(self, config: BenchmarkConfig):
        self.config = config
        self.output_dir = Path(config.output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.runs_jsonl = self.output_dir / "runs.jsonl"
        self.aggregate_csv = self.output_dir / "aggregate_by_method.csv"
        self.primary_csv = self.output_dir / "primary_endpoint_summary.csv"
        self.paired_csv = self.output_dir / "paired_tests.csv"
        self.manifest_json = self.output_dir / "run_manifest.json"

    def _seed_values(self, count: int, offset: int = 0) -> list[int]:
        rng = np.random.RandomState(self.config.random_seed + offset)
        # Deterministic sampled seed values; use wide range to avoid trivial collisions.
        values = rng.choice(np.arange(1, 1_000_000, dtype=np.int64), size=count, replace=False)
        return [int(v) for v in values]

    def _split_indices(self, n: int, seed: int) -> tuple[np.ndarray, np.ndarray]:
        rng = np.random.RandomState(seed)
        idx = np.arange(n)
        rng.shuffle(idx)
        n_test = max(1, int(round(self.config.split.test_size * n)))
        test_idx = idx[:n_test]
        train_idx = idx[n_test:]
        if train_idx.size == 0:
            train_idx = idx
            test_idx = idx[:1]
        return train_idx, test_idx

    def _load_dataset(self, name: str) -> dict[str, np.ndarray]:
        if name == "synthetic":
            raise RuntimeError("Synthetic dataset must be generated per-seed")

        if name != "chest_xray":
            raise ValueError(f"Unknown dataset name: {name}")
        cfg = self.config.chest_xray
        embeddings = np.load(str(cfg.embeddings_path))
        labels = np.load(str(cfg.labels_path))
        if cfg.group_labels_path:
            group_labels = np.load(str(cfg.group_labels_path))
        else:
            group_labels = labels
        if embeddings.ndim != 2:
            raise ValueError("Chest embeddings must be 2D")
        if labels.ndim != 1:
            raise ValueError("Chest labels must be 1D")
        if group_labels.ndim != 1:
            raise ValueError("Chest group labels must be 1D")
        n = embeddings.shape[0]
        if labels.shape[0] != n or group_labels.shape[0] != n:
            raise ValueError("Chest arrays must have matching length")
        return {"embeddings": embeddings, "labels": labels, "group_labels": group_labels}

    def _generate_synthetic(self, seed: int) -> dict[str, np.ndarray]:
        cfg = self.config.synthetic
        dataset = generate_parametric_shortcut_dataset(
            n_samples=cfg.n_samples,
            embedding_dim=cfg.embedding_dim,
            shortcut_dims=cfg.shortcut_dims,
            effect_size=cfg.effect_size,
            seed=seed,
        )
        return {
            "embeddings": dataset.embeddings,
            "labels": dataset.labels,
            "group_labels": dataset.labels,
            "shortcut_dim_labels": dataset.shortcut_dim_labels,
        }

    def _iter_dataset_seed_inputs(self) -> Iterable[tuple[str, int, dict[str, np.ndarray]]]:
        if self.config.synthetic.enabled:
            for seed in self._seed_values(self.config.synthetic.n_seeds, offset=100):
                yield ("synthetic", seed, self._generate_synthetic(seed))
        if self.config.chest_xray.enabled:
            base = self._load_dataset("chest_xray")
            for seed in self._seed_values(self.config.chest_xray.n_seeds, offset=200):
                yield ("chest_xray", seed, base)

    def _run_single_method_with_split(
        self,
        *,
        dataset_name: str,
        seed: int,
        method: str,
        data: dict[str, np.ndarray],
    ) -> dict[str, Any]:
        X = data["embeddings"]
        y = data["labels"]
        g = data["group_labels"]
        tr_idx, te_idx = self._split_indices(X.shape[0], seed=seed)
        X_tr, y_tr, g_tr = X[tr_idx], y[tr_idx], g[tr_idx]
        start = time.perf_counter()
        try:
            detector = ShortcutDetector(methods=[method], seed=seed)
            detector.fit(X_tr, y_tr, group_labels=g_tr)
            result = detector.results_.get(
                method, {"success": False, "error": "missing method result"}
            )
        except Exception as exc:  # pragma: no cover
            result = {"success": False, "error": str(exc)}
        duration = time.perf_counter() - start
        return _extract_method_record(
            dataset_name=dataset_name,
            seed=seed,
            method=method,
            detector_result=result,
            duration_sec=duration,
            n_total=X.shape[0],
            n_train=X_tr.shape[0],
            n_test=te_idx.shape[0],
        )

    def _write_jsonl_row(self, row: dict[str, Any]) -> None:
        with self.runs_jsonl.open("a", encoding="utf-8") as f:
            f.write(json.dumps(row, default=str) + "\n")

    def _write_manifest(self) -> None:
        try:
            git_sha = (
                subprocess.check_output(["git", "rev-parse", "HEAD"], stderr=subprocess.DEVNULL)
                .decode("utf-8")
                .strip()
            )
        except Exception:
            git_sha = "unknown"
        manifest = {
            "benchmark_name": self.config.benchmark_name,
            "generated_at_utc": datetime.now(timezone.utc).isoformat(),
            "git_sha": git_sha,
            "python": sys.version,
            "platform": platform.platform(),
            "config": {
                "methods": self.config.methods,
                "primary_endpoint": self.config.primary_endpoint,
                "random_seed": self.config.random_seed,
                "split_policy": {
                    "policy": self.config.split.policy,
                    "test_size": self.config.split.test_size,
                },
                "stats": {
                    "paired_tests": self.config.stats.paired_tests,
                    "multiple_testing": self.config.stats.multiple_testing,
                    "ci_method": self.config.stats.ci_method,
                    "bootstrap_samples": self.config.stats.bootstrap_samples,
                },
                "datasets": {
                    "synthetic": {
                        "enabled": self.config.synthetic.enabled,
                        "n_seeds": self.config.synthetic.n_seeds,
                        "n_samples": self.config.synthetic.n_samples,
                        "embedding_dim": self.config.synthetic.embedding_dim,
                        "shortcut_dims": self.config.synthetic.shortcut_dims,
                        "effect_size": self.config.synthetic.effect_size,
                    },
                    "chest_xray": {
                        "enabled": self.config.chest_xray.enabled,
                        "n_seeds": self.config.chest_xray.n_seeds,
                        "embeddings_path": self.config.chest_xray.embeddings_path,
                        "labels_path": self.config.chest_xray.labels_path,
                        "group_labels_path": self.config.chest_xray.group_labels_path,
                    },
                },
                "failure_policy": self.config.failure_policy,
            },
        }
        self.manifest_json.write_text(json.dumps(manifest, indent=2))

    def _aggregate(self, rows: list[dict[str, Any]]) -> pd.DataFrame:
        df = pd.DataFrame(rows)
        endpoints = [
            "risk_ordinal",
            "probe_metric_value",
            "hbac_confidence_score",
            "stat_n_significant_comparisons",
            "geometric_num_high_effect_pairs",
            "geometric_num_overlap_pairs",
        ]
        records: list[dict[str, Any]] = []
        for (dataset_name, method), group in df.groupby(["dataset_name", "method"], sort=True):
            rec: dict[str, Any] = {
                "dataset_name": dataset_name,
                "method": method,
                "n_runs": int(group.shape[0]),
                "n_success": int((group["status"] == "success").sum()),
            }
            rec["failure_rate"] = (
                float((rec["n_runs"] - rec["n_success"]) / rec["n_runs"])
                if rec["n_runs"]
                else np.nan
            )
            for endpoint in endpoints:
                vals = pd.to_numeric(group[endpoint], errors="coerce").to_numpy(dtype=float)
                vals = vals[np.isfinite(vals)]
                if vals.size == 0:
                    rec[f"mean_{endpoint}"] = np.nan
                    rec[f"std_{endpoint}"] = np.nan
                    rec[f"median_{endpoint}"] = np.nan
                    rec[f"iqr_{endpoint}"] = np.nan
                    rec[f"ci95_low_{endpoint}"] = np.nan
                    rec[f"ci95_high_{endpoint}"] = np.nan
                    continue
                q1, q3 = np.percentile(vals, [25, 75])
                ci_lo, ci_hi = _bootstrap_ci(vals, self.config.stats.bootstrap_samples)
                rec[f"mean_{endpoint}"] = float(np.mean(vals))
                rec[f"std_{endpoint}"] = float(np.std(vals))
                rec[f"median_{endpoint}"] = float(np.median(vals))
                rec[f"iqr_{endpoint}"] = float(q3 - q1)
                rec[f"ci95_low_{endpoint}"] = ci_lo
                rec[f"ci95_high_{endpoint}"] = ci_hi
            records.append(rec)
        out = pd.DataFrame(records).sort_values(["dataset_name", "method"]).reset_index(drop=True)
        out.to_csv(self.aggregate_csv, index=False)
        return out

    def _primary_summary(self, df: pd.DataFrame) -> pd.DataFrame:
        endpoint = self.config.primary_endpoint
        rows = df[df["status"] == "success"].copy()
        if endpoint not in rows.columns:
            out = pd.DataFrame(
                columns=[
                    "dataset_name",
                    "endpoint",
                    "n",
                    "mean",
                    "std",
                    "median",
                    "ci95_low",
                    "ci95_high",
                ]
            )
            out.to_csv(self.primary_csv, index=False)
            return out

        records: list[dict[str, Any]] = []
        for dataset_name, group in rows.groupby("dataset_name", sort=True):
            vals = pd.to_numeric(group[endpoint], errors="coerce").to_numpy(dtype=float)
            vals = vals[np.isfinite(vals)]
            if vals.size == 0:
                records.append(
                    {
                        "dataset_name": dataset_name,
                        "endpoint": endpoint,
                        "n": 0,
                        "mean": np.nan,
                        "std": np.nan,
                        "median": np.nan,
                        "ci95_low": np.nan,
                        "ci95_high": np.nan,
                    }
                )
                continue
            ci_lo, ci_hi = _bootstrap_ci(vals, self.config.stats.bootstrap_samples)
            records.append(
                {
                    "dataset_name": dataset_name,
                    "endpoint": endpoint,
                    "n": int(vals.size),
                    "mean": float(np.mean(vals)),
                    "std": float(np.std(vals)),
                    "median": float(np.median(vals)),
                    "ci95_low": ci_lo,
                    "ci95_high": ci_hi,
                }
            )
        if records:
            out = pd.DataFrame(records).sort_values("dataset_name").reset_index(drop=True)
        else:
            out = pd.DataFrame(
                columns=[
                    "dataset_name",
                    "endpoint",
                    "n",
                    "mean",
                    "std",
                    "median",
                    "ci95_low",
                    "ci95_high",
                ]
            )
        out.to_csv(self.primary_csv, index=False)
        return out

    def _paired_tests(self, df: pd.DataFrame) -> pd.DataFrame:
        records: list[dict[str, Any]] = []
        endpoint = "risk_ordinal"
        success = df[df["status"] == "success"].copy()
        for dataset_name, ds_df in success.groupby("dataset_name", sort=True):
            methods = sorted(ds_df["method"].dropna().unique().tolist())
            for method_a, method_b in combinations(methods, 2):
                aa = ds_df[ds_df["method"] == method_a][["seed", endpoint]].rename(
                    columns={endpoint: "value_a"}
                )
                bb = ds_df[ds_df["method"] == method_b][["seed", endpoint]].rename(
                    columns={endpoint: "value_b"}
                )
                merged = aa.merge(bb, on="seed", how="inner")
                xa = pd.to_numeric(merged["value_a"], errors="coerce").to_numpy(dtype=float)
                xb = pd.to_numeric(merged["value_b"], errors="coerce").to_numpy(dtype=float)
                mask = np.isfinite(xa) & np.isfinite(xb)
                xa = xa[mask]
                xb = xb[mask]
                if xa.size < 2:
                    continue
                delta = xa - xb
                rec = {
                    "dataset_name": dataset_name,
                    "endpoint": endpoint,
                    "method_a": method_a,
                    "method_b": method_b,
                    "n_pairs": int(xa.size),
                    "mean_delta": float(np.mean(delta)),
                    "median_delta": float(np.median(delta)),
                    "effect_size_rank_biserial": _paired_rank_biserial(xa, xb),
                    "p_value": _paired_wilcoxon_pvalue(xa, xb),
                }
                records.append(rec)

        out = pd.DataFrame(records)
        if out.empty:
            out = pd.DataFrame(
                columns=[
                    "dataset_name",
                    "endpoint",
                    "method_a",
                    "method_b",
                    "n_pairs",
                    "mean_delta",
                    "median_delta",
                    "effect_size_rank_biserial",
                    "p_value",
                    "q_value",
                ]
            )
            out.to_csv(self.paired_csv, index=False)
            return out

        out["q_value"] = np.nan
        for (_dataset_name, _endpoint_name), idx in out.groupby(
            ["dataset_name", "endpoint"]
        ).groups.items():
            pvals = out.loc[idx, "p_value"].to_numpy(dtype=float)
            mask = np.isfinite(pvals)
            if not np.any(mask):
                continue
            qvals = np.full_like(pvals, fill_value=np.nan, dtype=float)
            _, qvals_masked, _, _ = multipletests(
                pvals[mask], alpha=0.05, method=self.config.stats.multiple_testing
            )
            qvals[mask] = qvals_masked
            out.loc[idx, "q_value"] = qvals
        out = out.sort_values(["dataset_name", "endpoint", "method_a", "method_b"]).reset_index(
            drop=True
        )
        out.to_csv(self.paired_csv, index=False)
        return out

    def run(self) -> dict[str, Any]:
        set_seed(self.config.random_seed)
        if self.runs_jsonl.exists():
            self.runs_jsonl.unlink()

        rows: list[dict[str, Any]] = []
        for dataset_name, seed, data in self._iter_dataset_seed_inputs():
            for method in self.config.methods:
                row = self._run_single_method_with_split(
                    dataset_name=dataset_name,
                    seed=seed,
                    method=method,
                    data=data,
                )
                self._write_jsonl_row(row)
                rows.append(row)

        df = pd.DataFrame(rows)
        aggregate_df = self._aggregate(rows)
        primary_df = self._primary_summary(df)
        if self.config.stats.paired_tests:
            paired_df = self._paired_tests(df)
        else:
            paired_df = pd.DataFrame(
                columns=[
                    "dataset_name",
                    "endpoint",
                    "method_a",
                    "method_b",
                    "n_pairs",
                    "mean_delta",
                    "median_delta",
                    "effect_size_rank_biserial",
                    "p_value",
                    "q_value",
                ]
            )
        self._write_manifest()

        expected = 0
        if self.config.synthetic.enabled:
            expected += self.config.synthetic.n_seeds * len(self.config.methods)
        if self.config.chest_xray.enabled:
            expected += self.config.chest_xray.n_seeds * len(self.config.methods)
        observed = int(df.shape[0])
        if observed != expected:
            raise RuntimeError(
                f"Unexpected number of runs: observed={observed}, expected={expected}"
            )
        if df.duplicated(subset=["dataset_name", "seed", "method"]).any():
            raise RuntimeError("Duplicate (dataset_name, seed, method) rows detected.")

        return {
            "runs": df,
            "aggregate": aggregate_df,
            "primary": primary_df,
            "paired_tests": paired_df,
            "output_dir": str(self.output_dir),
        }


def run_benchmark(config: BenchmarkConfig) -> dict[str, Any]:
    """Convenience function to execute benchmark run."""
    return BenchmarkRunner(config).run()


def _parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run multi-seed shortcut detector benchmark.")
    parser.add_argument(
        "--config", required=True, help="Path to benchmark config (.json/.yaml/.yml)"
    )
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = _parse_args(argv)
    cfg = BenchmarkConfig.from_path(args.config)
    runner = BenchmarkRunner(cfg)
    runner.run()
    print(f"Benchmark complete. Artifacts in: {runner.output_dir}")
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
