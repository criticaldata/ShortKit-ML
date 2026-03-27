"""Paper-focused benchmark runner for synthetic and CheXpert studies."""

from __future__ import annotations

import argparse
import json
import os
import platform
import subprocess
import sys
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from shortcut_detect.benchmark.method_utils import (
    ALL_METHODS,
    SKIP_IN_SYNTHETIC,
    bias_direction_pca_dim_scores,
    convergence_bucket,
    method_flag,
    nan_dim_scores,
    sis_dim_scores,
)
from shortcut_detect.benchmark.synthetic import generate_parametric_shortcut_dataset
from shortcut_detect.statistical import GroupDiffTest
from shortcut_detect.unified import ShortcutDetector
from shortcut_detect.utils import set_seed

SUPPORTED_METHODS = ALL_METHODS


@dataclass
class SyntheticGridConfig:
    effect_sizes: list[float] = field(default_factory=lambda: [0.2, 0.5, 0.8, 1.2, 2.0])
    sample_sizes: list[int] = field(default_factory=lambda: [200, 1000, 5000])
    imbalance_ratios: list[float] = field(default_factory=lambda: [0.5, 0.9])
    embedding_dims: list[int] = field(default_factory=lambda: [128, 256, 512])
    shortcut_dims: int = 5
    seeds: int = 10
    alpha: float = 0.05
    corrections: list[str] = field(default_factory=lambda: ["bonferroni", "fdr_bh"])


@dataclass
class CheXpertConfig:
    enabled: bool = False
    manifest_path: str | None = None
    embeddings_dir: str = "output/paper_benchmark/chexpert_embeddings"
    backbones: list[str] = field(default_factory=lambda: ["resnet50", "densenet121", "vit_b_16"])
    attributes: list[str] = field(default_factory=lambda: ["race", "sex", "age_bin"])
    race_focus: list[str] = field(default_factory=lambda: ["ASIAN", "BLACK", "WHITE"])
    age_bins: list[int] = field(default_factory=lambda: [40, 60, 80])


@dataclass
class FairnessBaselineConfig:
    enable_fairlearn: bool = True
    enable_aif360: bool = True


@dataclass
class PaperBenchmarkConfig:
    benchmark_name: str = "shortcut_paper_benchmark"
    profile: str = "default"
    random_seed: int = 42
    methods: list[str] = field(default_factory=lambda: list(ALL_METHODS))
    synthetic: SyntheticGridConfig = field(default_factory=SyntheticGridConfig)
    chexpert: CheXpertConfig = field(default_factory=CheXpertConfig)
    fairness_baselines: FairnessBaselineConfig = field(default_factory=FairnessBaselineConfig)
    output_dir: str = "output/paper_benchmark"

    @classmethod
    def from_dict(cls, raw: dict[str, Any]) -> PaperBenchmarkConfig:
        profile = str(raw.get("profile", "default"))

        synthetic_raw = dict(raw.get("synthetic", {}))
        chexpert_raw = dict(raw.get("chexpert", {}))
        fairness_raw = dict(raw.get("fairness_baselines", {}))

        synthetic = SyntheticGridConfig(
            effect_sizes=[
                float(x) for x in synthetic_raw.get("effect_sizes", [0.2, 0.5, 0.8, 1.2, 2.0])
            ],
            sample_sizes=[int(x) for x in synthetic_raw.get("sample_sizes", [200, 1000, 5000])],
            imbalance_ratios=[float(x) for x in synthetic_raw.get("imbalance_ratios", [0.5, 0.9])],
            embedding_dims=[int(x) for x in synthetic_raw.get("embedding_dims", [128, 256, 512])],
            shortcut_dims=int(synthetic_raw.get("shortcut_dims", 5)),
            seeds=int(synthetic_raw.get("seeds", 10)),
            alpha=float(synthetic_raw.get("alpha", 0.05)),
            corrections=[
                str(x) for x in synthetic_raw.get("corrections", ["bonferroni", "fdr_bh"])
            ],
        )

        chexpert = CheXpertConfig(
            enabled=bool(chexpert_raw.get("enabled", False)),
            manifest_path=chexpert_raw.get("manifest_path"),
            embeddings_dir=str(
                chexpert_raw.get("embeddings_dir", "output/paper_benchmark/chexpert_embeddings")
            ),
            backbones=[
                str(x)
                for x in chexpert_raw.get("backbones", ["resnet50", "densenet121", "vit_b_16"])
            ],
            attributes=[str(x) for x in chexpert_raw.get("attributes", ["race", "sex", "age_bin"])],
            race_focus=[
                str(x).upper() for x in chexpert_raw.get("race_focus", ["ASIAN", "BLACK", "WHITE"])
            ],
            age_bins=[int(x) for x in chexpert_raw.get("age_bins", [40, 60, 80])],
        )

        cfg = cls(
            benchmark_name=str(raw.get("benchmark_name", "shortcut_paper_benchmark")),
            profile=profile,
            random_seed=int(raw.get("random_seed", 42)),
            methods=[str(x) for x in raw.get("methods", list(SUPPORTED_METHODS))],
            synthetic=synthetic,
            chexpert=chexpert,
            fairness_baselines=FairnessBaselineConfig(
                enable_fairlearn=bool(fairness_raw.get("enable_fairlearn", True)),
                enable_aif360=bool(fairness_raw.get("enable_aif360", True)),
            ),
            output_dir=str(raw.get("output_dir", "output/paper_benchmark")),
        )
        cfg.apply_profile_defaults()
        cfg.validate()
        return cfg

    @classmethod
    def from_path(cls, path: str | os.PathLike[str]) -> PaperBenchmarkConfig:
        path_obj = Path(path)
        if not path_obj.exists():
            raise FileNotFoundError(f"Config not found: {path_obj}")
        suffix = path_obj.suffix.lower()
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
            raise TypeError("Paper benchmark config root must be an object")
        return cls.from_dict(raw)

    def apply_profile_defaults(self) -> None:
        if self.profile == "smoke":
            self.synthetic.effect_sizes = [0.2, 1.0]
            self.synthetic.sample_sizes = [200]
            self.synthetic.imbalance_ratios = [0.5]
            self.synthetic.embedding_dims = [128]
            self.synthetic.seeds = 2
        elif self.profile == "default":
            self.synthetic.effect_sizes = self.synthetic.effect_sizes or [0.2, 0.5, 0.8, 1.2]
            self.synthetic.sample_sizes = self.synthetic.sample_sizes or [200, 1000]
            self.synthetic.imbalance_ratios = self.synthetic.imbalance_ratios or [0.5, 0.9]
            self.synthetic.embedding_dims = self.synthetic.embedding_dims or [128, 256]
            self.synthetic.seeds = max(3, self.synthetic.seeds)
        elif self.profile == "full":
            # Keep user-provided values; defaults already reflect full paper intent.
            pass
        else:
            raise ValueError("profile must be one of: smoke, default, full")

    def validate(self) -> None:
        unknown = [m for m in self.methods if m not in ALL_METHODS]
        if unknown:
            raise ValueError(f"Unsupported methods: {unknown}. Supported: {list(ALL_METHODS)}")
        if self.synthetic.shortcut_dims <= 0:
            raise ValueError("synthetic.shortcut_dims must be > 0")
        if self.synthetic.seeds <= 0:
            raise ValueError("synthetic.seeds must be > 0")
        if not (0.0 < self.synthetic.alpha < 1.0):
            raise ValueError("synthetic.alpha must be in (0, 1)")
        if any(not (0.0 < x <= 1.0) for x in self.synthetic.imbalance_ratios):
            raise ValueError("All synthetic.imbalance_ratios must be in (0, 1]")
        if self.chexpert.enabled and not self.chexpert.manifest_path:
            raise ValueError("chexpert.manifest_path is required when chexpert.enabled=true")


@dataclass
class RunArtifacts:
    runs_path: Path
    synthetic_pr_path: Path
    synthetic_fp_path: Path
    correction_path: Path
    chexpert_methods_path: Path
    chexpert_convergence_path: Path
    baselines_path: Path
    manifest_path: Path
    summary_md_path: Path


class PaperBenchmarkRunner:
    """Runs paper-specific synthetic + CheXpert benchmark workflows."""

    def __init__(self, config: PaperBenchmarkConfig):
        self.config = config
        self.output_dir = Path(config.output_dir)
        self.figures_dir = self.output_dir / "figures"
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.figures_dir.mkdir(parents=True, exist_ok=True)
        self.artifacts = RunArtifacts(
            runs_path=self.output_dir / "runs_paper.csv",
            synthetic_pr_path=self.output_dir / "synthetic_dim_pr.csv",
            synthetic_fp_path=self.output_dir / "synthetic_fp_control.csv",
            correction_path=self.output_dir / "synthetic_correction_control.csv",
            chexpert_methods_path=self.output_dir / "chexpert_method_results.csv",
            chexpert_convergence_path=self.output_dir / "chexpert_convergence_matrix.csv",
            baselines_path=self.output_dir / "external_baseline_comparison.csv",
            manifest_path=self.output_dir / "run_manifest.json",
            summary_md_path=self.output_dir / "paper_summary.md",
        )

    def _seed_values(self, count: int, offset: int) -> list[int]:
        rng = np.random.RandomState(self.config.random_seed + offset)
        values = rng.choice(np.arange(1, 1_000_000, dtype=np.int64), size=count, replace=False)
        return [int(v) for v in values]

    def _generate_synthetic_dataset(
        self,
        *,
        n_samples: int,
        embedding_dim: int,
        shortcut_dims: int,
        effect_size: float,
        imbalance_ratio: float,
        seed: int,
        with_shortcut: bool,
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        if with_shortcut:
            dataset = generate_parametric_shortcut_dataset(
                n_samples=n_samples,
                embedding_dim=embedding_dim,
                shortcut_dims=shortcut_dims,
                effect_size=effect_size,
                positive_class_probability=max(1e-6, min(1.0 - 1e-6, 1.0 - imbalance_ratio)),
                seed=seed,
            )
            return dataset.embeddings, dataset.labels, dataset.shortcut_dim_indices

        rng = np.random.RandomState(seed)
        labels = (rng.rand(n_samples) < (1.0 - imbalance_ratio)).astype(np.int64)
        embeddings = rng.randn(n_samples, embedding_dim).astype(np.float32)
        true_dims = np.arange(shortcut_dims, dtype=int)
        return embeddings, labels, true_dims

    @staticmethod
    def _method_flag(method: str, result: dict[str, Any]) -> bool:
        return method_flag(method, result)

    @staticmethod
    def _precision_recall(pred: np.ndarray, truth: np.ndarray) -> tuple[float, float, float, float]:
        pred_set = {int(x) for x in pred.tolist()}
        true_set = {int(x) for x in truth.tolist()}
        tp = len(pred_set & true_set)
        fp = len(pred_set - true_set)
        fn = len(true_set - pred_set)
        precision = float(tp / (tp + fp)) if (tp + fp) else 0.0
        recall = float(tp / (tp + fn)) if (tp + fn) else 0.0
        f1 = float((2 * precision * recall) / (precision + recall)) if (precision + recall) else 0.0
        jaccard = float(tp / len(pred_set | true_set)) if (pred_set or true_set) else 1.0
        return precision, recall, f1, jaccard

    @staticmethod
    def _top_k_from_scores(scores: np.ndarray, k: int) -> np.ndarray:
        if scores.size == 0:
            return np.array([], dtype=int)
        k = max(1, min(k, scores.size))
        order = np.argsort(scores)[::-1]
        return order[:k]

    @staticmethod
    def _probe_dim_scores(X: np.ndarray, group_labels: np.ndarray, seed: int) -> np.ndarray:
        from sklearn.linear_model import LogisticRegression
        from sklearn.pipeline import Pipeline
        from sklearn.preprocessing import StandardScaler

        model = Pipeline(
            steps=[
                ("scaler", StandardScaler(with_mean=True, with_std=True)),
                ("clf", LogisticRegression(max_iter=2000, solver="lbfgs", random_state=seed)),
            ]
        )
        model.fit(X, group_labels)
        clf = model.named_steps["clf"]
        coef = np.asarray(clf.coef_, dtype=float)
        if coef.ndim == 2:
            coef = np.mean(np.abs(coef), axis=0)
        return np.asarray(coef, dtype=float)

    @staticmethod
    def _hbac_dim_scores(result: dict[str, Any], embedding_dim: int) -> np.ndarray:
        scores = np.zeros(embedding_dim, dtype=float)
        report = result.get("report", {})
        dim_importance = report.get("dimension_importance")
        if not isinstance(dim_importance, pd.DataFrame):
            return scores
        if "dimension" not in dim_importance.columns or "f_score" not in dim_importance.columns:
            return scores
        for _, row in dim_importance.iterrows():
            name = str(row["dimension"])
            if name.startswith("dim_"):
                try:
                    idx = int(name.split("_")[1])
                except (IndexError, ValueError):
                    continue
                if 0 <= idx < embedding_dim:
                    scores[idx] = float(row.get("f_score", 0.0))
        return scores

    @staticmethod
    def _statistical_dim_scores(result: dict[str, Any], embedding_dim: int) -> np.ndarray:
        corrected = result.get("corrected_pvalues", {})
        if not isinstance(corrected, dict) or not corrected:
            return np.zeros(embedding_dim, dtype=float)
        stack = []
        for arr in corrected.values():
            vals = np.asarray(arr, dtype=float)
            if vals.shape[0] == embedding_dim:
                stack.append(vals)
        if not stack:
            return np.zeros(embedding_dim, dtype=float)
        min_p = np.min(np.vstack(stack), axis=0)
        return -np.log10(np.clip(min_p, 1e-300, 1.0))

    @staticmethod
    def _geometric_dim_scores(X: np.ndarray, group_labels: np.ndarray) -> np.ndarray:
        groups = np.unique(group_labels)
        if groups.size < 2:
            return np.zeros(X.shape[1], dtype=float)
        centroids = []
        for g in groups:
            centroids.append(np.mean(X[group_labels == g], axis=0))
        centroids_arr = np.vstack(centroids)
        scores = np.zeros(X.shape[1], dtype=float)
        n_pairs = 0
        for i in range(centroids_arr.shape[0]):
            for j in range(i + 1, centroids_arr.shape[0]):
                diff = np.abs(centroids_arr[i] - centroids_arr[j])
                scores += diff
                n_pairs += 1
        if n_pairs > 0:
            scores /= float(n_pairs)
        return scores

    def _run_methods(self, X: np.ndarray, y: np.ndarray, seed: int) -> dict[str, dict[str, Any]]:
        out: dict[str, dict[str, Any]] = {}
        active_methods = [m for m in self.config.methods if m not in SKIP_IN_SYNTHETIC]
        for method in active_methods:
            detector = ShortcutDetector(
                methods=[method],
                seed=seed,
                statistical_correction="fdr_bh",
                statistical_alpha=self.config.synthetic.alpha,
            )
            try:
                detector.fit(X, y, group_labels=y)
                out[method] = detector.results_.get(
                    method, {"success": False, "error": "missing result"}
                )
            except Exception as exc:  # pragma: no cover
                out[method] = {"success": False, "error": str(exc)}
        return out

    def _run_synthetic(self) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
        rows: list[dict[str, Any]] = []
        fp_rows: list[dict[str, Any]] = []
        correction_rows: list[dict[str, Any]] = []
        seeds = self._seed_values(self.config.synthetic.seeds, offset=1000)

        for effect_size in self.config.synthetic.effect_sizes:
            for n_samples in self.config.synthetic.sample_sizes:
                for imb in self.config.synthetic.imbalance_ratios:
                    for dim in self.config.synthetic.embedding_dims:
                        for seed in seeds:
                            X, y, true_dims = self._generate_synthetic_dataset(
                                n_samples=n_samples,
                                embedding_dim=dim,
                                shortcut_dims=self.config.synthetic.shortcut_dims,
                                effect_size=effect_size,
                                imbalance_ratio=imb,
                                seed=seed,
                                with_shortcut=True,
                            )
                            method_results = self._run_methods(X, y, seed)
                            active_methods = [
                                m for m in self.config.methods if m not in SKIP_IN_SYNTHETIC
                            ]
                            dim_scores: dict[str, np.ndarray] = {}
                            for m in active_methods:
                                if m == "probe":
                                    dim_scores[m] = self._probe_dim_scores(X, y, seed)
                                elif m == "hbac":
                                    dim_scores[m] = self._hbac_dim_scores(
                                        method_results.get("hbac", {}), dim
                                    )
                                elif m == "statistical":
                                    dim_scores[m] = self._statistical_dim_scores(
                                        method_results.get("statistical", {}), dim
                                    )
                                elif m == "geometric":
                                    dim_scores[m] = self._geometric_dim_scores(X, y)
                                elif m == "bias_direction_pca":
                                    dim_scores[m] = bias_direction_pca_dim_scores(
                                        method_results.get("bias_direction_pca", {}), dim
                                    )
                                elif m == "sis":
                                    dim_scores[m] = sis_dim_scores(
                                        method_results.get("sis", {}), dim
                                    )
                                else:
                                    dim_scores[m] = nan_dim_scores(dim)
                            flags = []
                            for method in active_methods:
                                scores = dim_scores.get(method, nan_dim_scores(dim))
                                if np.all(np.isnan(scores)):
                                    precision, recall, f1, jaccard = np.nan, np.nan, np.nan, np.nan
                                    pred_dims = np.array([], dtype=int)
                                else:
                                    pred_dims = self._top_k_from_scores(
                                        scores, self.config.synthetic.shortcut_dims
                                    )
                                    precision, recall, f1, jaccard = self._precision_recall(
                                        pred_dims, true_dims
                                    )
                                flagged = self._method_flag(method, method_results.get(method, {}))
                                flags.append(int(flagged))
                                rows.append(
                                    {
                                        "dataset": "synthetic_shortcut",
                                        "seed": seed,
                                        "method": method,
                                        "effect_size": effect_size,
                                        "n_samples": n_samples,
                                        "imbalance_ratio": imb,
                                        "embedding_dim": dim,
                                        "precision": precision,
                                        "recall": recall,
                                        "f1": f1,
                                        "jaccard": jaccard,
                                        "flagged": flagged,
                                        "n_flagged_methods": np.nan,
                                        "convergence_bucket": None,
                                    }
                                )

                            n_flagged = int(sum(flags))
                            n_methods = len(active_methods)
                            bucket = convergence_bucket(n_flagged, n_methods)
                            rows.append(
                                {
                                    "dataset": "synthetic_shortcut",
                                    "seed": seed,
                                    "method": "convergence",
                                    "effect_size": effect_size,
                                    "n_samples": n_samples,
                                    "imbalance_ratio": imb,
                                    "embedding_dim": dim,
                                    "precision": np.nan,
                                    "recall": np.nan,
                                    "f1": np.nan,
                                    "jaccard": np.nan,
                                    "flagged": np.nan,
                                    "n_flagged_methods": n_flagged,
                                    "convergence_bucket": bucket,
                                }
                            )

                            # Null-control run at same configuration.
                            X0, y0, _ = self._generate_synthetic_dataset(
                                n_samples=n_samples,
                                embedding_dim=dim,
                                shortcut_dims=self.config.synthetic.shortcut_dims,
                                effect_size=effect_size,
                                imbalance_ratio=imb,
                                seed=seed + 17,
                                with_shortcut=False,
                            )
                            null_results = self._run_methods(X0, y0, seed + 17)
                            null_flags = [
                                int(self._method_flag(m, null_results.get(m, {})))
                                for m in active_methods
                            ]
                            n_null_flagged = int(sum(null_flags))
                            n_m = len(active_methods)
                            fp_rows.append(
                                {
                                    "seed": seed,
                                    "effect_size": effect_size,
                                    "n_samples": n_samples,
                                    "imbalance_ratio": imb,
                                    "embedding_dim": dim,
                                    "n_flagged_methods": n_null_flagged,
                                    "any_false_positive": int(n_null_flagged > 0),
                                    f"false_positive_1of{n_m}": int(n_null_flagged == 1),
                                    f"false_positive_{max(n_m-1,1)}of{n_m}": int(
                                        n_null_flagged >= max(n_m - 1, 1)
                                    ),
                                    f"false_positive_{n_m}of{n_m}": int(n_null_flagged == n_m),
                                }
                            )

                            # Multiple-testing calibration on null data.
                            for correction in self.config.synthetic.corrections:
                                from scipy.stats import mannwhitneyu

                                stat = GroupDiffTest(test=mannwhitneyu)
                                stat.fit(X0, y0)
                                corr = stat.apply_correction(
                                    alpha=self.config.synthetic.alpha,
                                    method=correction,
                                    verbose=False,
                                )
                                total_sig = 0
                                any_sig = 0
                                for sig in corr["significant_features"].values():
                                    if sig is not None and len(sig) > 0:
                                        any_sig = 1
                                        total_sig += len(sig)
                                correction_rows.append(
                                    {
                                        "seed": seed,
                                        "n_samples": n_samples,
                                        "embedding_dim": dim,
                                        "correction": correction,
                                        "alpha": self.config.synthetic.alpha,
                                        "any_significant": any_sig,
                                        "n_significant": total_sig,
                                        "false_discovery_rate_est": float(total_sig / max(1, dim)),
                                    }
                                )

        runs_df = pd.DataFrame(rows)
        fp_df = pd.DataFrame(fp_rows)
        correction_df = pd.DataFrame(correction_rows)
        return runs_df, fp_df, correction_df

    def _load_chexpert_manifest(self) -> pd.DataFrame:
        path = Path(str(self.config.chexpert.manifest_path))
        if not path.exists():
            raise FileNotFoundError(f"CheXpert manifest not found: {path}")
        df = pd.read_csv(path)
        required = {"image_path", "task_label", "race", "sex", "age"}
        missing = sorted(required - set(df.columns))
        if missing:
            raise ValueError(f"CheXpert manifest missing required columns: {missing}")
        return df

    def _make_age_bins(self, age: pd.Series) -> pd.Series:
        bins = [-np.inf] + list(self.config.chexpert.age_bins) + [np.inf]
        labels = ["<40", "40-59", "60-79", ">=80"]
        if len(labels) != len(bins) - 1:
            labels = [f"bin_{i}" for i in range(len(bins) - 1)]
        return pd.cut(age.astype(float), bins=bins, labels=labels, right=False).astype(str)

    def _load_backbone_embeddings(self, backbone: str) -> tuple[np.ndarray, pd.DataFrame]:
        emb_path = Path(self.config.chexpert.embeddings_dir) / f"{backbone}_embeddings.npy"
        meta_path = Path(self.config.chexpert.embeddings_dir) / f"{backbone}_metadata.csv"
        if not emb_path.exists() or not meta_path.exists():
            raise FileNotFoundError(
                f"Missing backbone artifacts for {backbone}. Expected {emb_path} and {meta_path}."
            )
        X = np.load(str(emb_path))
        meta = pd.read_csv(meta_path)
        if X.ndim != 2:
            raise ValueError(f"{emb_path} must be 2D embeddings")
        if len(meta) != X.shape[0]:
            raise ValueError(f"{meta_path} rows ({len(meta)}) != embeddings rows ({X.shape[0]})")
        return X, meta

    def _run_chexpert(self) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
        if not self.config.chexpert.enabled:
            return pd.DataFrame(), pd.DataFrame(), pd.DataFrame()

        _ = self._load_chexpert_manifest()  # validate contract presence
        method_rows: list[dict[str, Any]] = []
        conv_rows: list[dict[str, Any]] = []
        baseline_rows: list[dict[str, Any]] = []

        for backbone in self.config.chexpert.backbones:
            X, meta = self._load_backbone_embeddings(backbone)
            y = (meta["task_label"].astype(float) > 0).astype(int).to_numpy()
            meta = meta.copy()
            meta["race"] = meta["race"].astype(str).str.upper()
            meta["sex"] = meta["sex"].astype(str)
            meta["age_bin"] = self._make_age_bins(meta["age"])

            for attr in self.config.chexpert.attributes:
                if attr not in meta.columns:
                    continue
                frame = meta
                if attr == "race":
                    frame = frame[frame["race"].isin(self.config.chexpert.race_focus)].copy()
                    if frame.empty:
                        continue
                idx = frame.index.to_numpy()
                X_attr = X[idx]
                y_attr = y[idx]
                g_attr = frame[attr].to_numpy()
                if np.unique(g_attr).shape[0] < 2:
                    continue

                flagged_methods = 0
                for method in self.config.methods:
                    detector = ShortcutDetector(
                        methods=[method],
                        seed=self.config.random_seed,
                        statistical_correction="fdr_bh",
                        statistical_alpha=self.config.synthetic.alpha,
                    )
                    try:
                        detector.fit(X_attr, y_attr, group_labels=g_attr)
                        result = detector.results_.get(
                            method, {"success": False, "error": "missing"}
                        )
                    except Exception as exc:  # pragma: no cover
                        result = {"success": False, "error": str(exc)}

                    flag = int(self._method_flag(method, result))
                    flagged_methods += flag
                    risk = str(result.get("risk_value") or result.get("risk_level") or "unknown")
                    method_rows.append(
                        {
                            "backbone": backbone,
                            "attribute": attr,
                            "method": method,
                            "n_samples": int(X_attr.shape[0]),
                            "n_groups": int(np.unique(g_attr).shape[0]),
                            "flagged": flag,
                            "risk_level": risk,
                            "success": bool(result.get("success", False)),
                            "error": result.get("error"),
                        }
                    )

                n_methods = len(self.config.methods)
                confidence = convergence_bucket(flagged_methods, n_methods)
                conv_rows.append(
                    {
                        "backbone": backbone,
                        "attribute": attr,
                        "n_samples": int(X_attr.shape[0]),
                        "n_flagged_methods": int(flagged_methods),
                        "confidence_level": confidence,
                    }
                )

                baseline_rows.extend(
                    self._run_external_baselines(
                        backbone=backbone,
                        attribute=attr,
                        X=X_attr,
                        y=y_attr,
                        g=g_attr,
                    )
                )

        return pd.DataFrame(method_rows), pd.DataFrame(conv_rows), pd.DataFrame(baseline_rows)

    def _run_external_baselines(
        self,
        *,
        backbone: str,
        attribute: str,
        X: np.ndarray,
        y: np.ndarray,
        g: np.ndarray,
    ) -> list[dict[str, Any]]:
        rows: list[dict[str, Any]] = []
        from sklearn.linear_model import LogisticRegression
        from sklearn.model_selection import train_test_split
        from sklearn.pipeline import Pipeline
        from sklearn.preprocessing import StandardScaler

        X_tr, X_te, y_tr, y_te, g_tr, g_te = train_test_split(
            X,
            y,
            g,
            test_size=0.3,
            random_state=self.config.random_seed,
            stratify=y,
        )
        model = Pipeline(
            steps=[
                ("scaler", StandardScaler(with_mean=True, with_std=True)),
                (
                    "clf",
                    LogisticRegression(
                        max_iter=2000, solver="lbfgs", random_state=self.config.random_seed
                    ),
                ),
            ]
        )
        model.fit(X_tr, y_tr)
        y_pred = model.predict(X_te)

        if self.config.fairness_baselines.enable_fairlearn:
            try:
                from fairlearn.metrics import (
                    demographic_parity_difference,
                    equalized_odds_difference,
                )

                rows.append(
                    {
                        "tool": "fairlearn",
                        "backbone": backbone,
                        "attribute": attribute,
                        "status": "success",
                        "demographic_parity_difference": float(
                            demographic_parity_difference(
                                y_true=y_te, y_pred=y_pred, sensitive_features=g_te
                            )
                        ),
                        "equalized_odds_difference": float(
                            equalized_odds_difference(
                                y_true=y_te, y_pred=y_pred, sensitive_features=g_te
                            )
                        ),
                        "detail": None,
                    }
                )
            except Exception as exc:
                rows.append(
                    {
                        "tool": "fairlearn",
                        "backbone": backbone,
                        "attribute": attribute,
                        "status": "skipped_missing_dependency",
                        "demographic_parity_difference": np.nan,
                        "equalized_odds_difference": np.nan,
                        "detail": str(exc),
                    }
                )

        if self.config.fairness_baselines.enable_aif360:
            try:
                from aif360.datasets import BinaryLabelDataset
                from aif360.metrics import ClassificationMetric

                g_series = pd.Series(g_te)
                g_codes = pd.Categorical(g_series.astype(str))
                code_arr = g_codes.codes.astype(int)
                valid_mask = code_arr >= 0
                code_values = sorted(np.unique(code_arr[valid_mask]).tolist())
                if len(code_values) >= 2:
                    privileged = int(code_values[0])
                    df_eval = pd.DataFrame(
                        {
                            "label": y_te.astype(int),
                            "pred": np.asarray(y_pred).astype(int),
                            "group": code_arr,
                        }
                    )
                    bld_true = BinaryLabelDataset(
                        favorable_label=1,
                        unfavorable_label=0,
                        df=df_eval[["label", "group"]],
                        label_names=["label"],
                        protected_attribute_names=["group"],
                    )
                    bld_pred = BinaryLabelDataset(
                        favorable_label=1,
                        unfavorable_label=0,
                        df=df_eval[["pred", "group"]].rename(columns={"pred": "label"}),
                        label_names=["label"],
                        protected_attribute_names=["group"],
                    )
                    metric = ClassificationMetric(
                        bld_true,
                        bld_pred,
                        privileged_groups=[{"group": privileged}],
                        unprivileged_groups=[{"group": int(x)} for x in code_values[1:]],
                    )
                    rows.append(
                        {
                            "tool": "aif360",
                            "backbone": backbone,
                            "attribute": attribute,
                            "status": "success",
                            "demographic_parity_difference": float(
                                metric.statistical_parity_difference()
                            ),
                            "equalized_odds_difference": float(
                                metric.average_abs_odds_difference()
                            ),
                            "detail": None,
                        }
                    )
                else:
                    rows.append(
                        {
                            "tool": "aif360",
                            "backbone": backbone,
                            "attribute": attribute,
                            "status": "skipped_single_group",
                            "demographic_parity_difference": np.nan,
                            "equalized_odds_difference": np.nan,
                            "detail": "Need at least two groups",
                        }
                    )
            except Exception as exc:
                rows.append(
                    {
                        "tool": "aif360",
                        "backbone": backbone,
                        "attribute": attribute,
                        "status": "skipped_missing_dependency",
                        "demographic_parity_difference": np.nan,
                        "equalized_odds_difference": np.nan,
                        "detail": str(exc),
                    }
                )

        return rows

    def _write_manifest(self) -> None:
        try:
            git_sha = (
                subprocess.check_output(["git", "rev-parse", "HEAD"], stderr=subprocess.DEVNULL)
                .decode("utf-8")
                .strip()
            )
        except Exception:
            git_sha = "unknown"

        payload = {
            "benchmark_name": self.config.benchmark_name,
            "generated_at_utc": datetime.now(timezone.utc).isoformat(),
            "git_sha": git_sha,
            "python": sys.version,
            "platform": platform.platform(),
            "config": asdict(self.config),
        }
        self.artifacts.manifest_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")

    def _generate_plots(
        self,
        synthetic_runs: pd.DataFrame,
        synthetic_fp: pd.DataFrame,
        correction_df: pd.DataFrame,
        chexpert_conv: pd.DataFrame,
    ) -> None:
        try:
            import matplotlib.pyplot as plt
        except Exception:  # pragma: no cover
            return

        if not synthetic_runs.empty:
            d = synthetic_runs[synthetic_runs["method"].isin(self.config.methods)].copy()
            agg = d.groupby(["method", "effect_size"], as_index=False)["recall"].mean()
            fig, ax = plt.subplots(figsize=(8, 5))
            for method in self.config.methods:
                sub = agg[agg["method"] == method]
                if not sub.empty:
                    ax.plot(sub["effect_size"], sub["recall"], marker="o", label=method)
            ax.set_xlabel("Effect Size")
            ax.set_ylabel("Mean Recall (Shortcut Dims)")
            ax.set_title("Synthetic Shortcut-Dim Recall by Effect Size")
            ax.legend()
            fig.tight_layout()
            fig.savefig(self.figures_dir / "synthetic_recall_by_effect_size.png", dpi=160)
            plt.close(fig)

        if not synthetic_fp.empty:
            agg = (
                synthetic_fp.groupby("n_flagged_methods", as_index=False)["any_false_positive"]
                .mean()
                .sort_values("n_flagged_methods")
            )
            fig, ax = plt.subplots(figsize=(8, 5))
            ax.bar(agg["n_flagged_methods"].astype(str), agg["any_false_positive"])
            ax.set_xlabel("Number of Methods Flagging on Null Data")
            ax.set_ylabel("False Positive Rate")
            ax.set_title("Convergence vs False Positives (Synthetic Null)")
            fig.tight_layout()
            fig.savefig(self.figures_dir / "synthetic_convergence_false_positives.png", dpi=160)
            plt.close(fig)

        if not correction_df.empty:
            agg = correction_df.groupby(["correction", "embedding_dim"], as_index=False)[
                "any_significant"
            ].mean()
            fig, ax = plt.subplots(figsize=(8, 5))
            for corr in sorted(agg["correction"].unique().tolist()):
                sub = agg[agg["correction"] == corr]
                ax.plot(sub["embedding_dim"], sub["any_significant"], marker="o", label=corr)
            ax.set_xlabel("Embedding Dimensionality")
            ax.set_ylabel("Empirical FWER on Null")
            ax.set_title("Multiple Testing Calibration")
            ax.legend()
            fig.tight_layout()
            fig.savefig(self.figures_dir / "multiple_testing_calibration.png", dpi=160)
            plt.close(fig)

        if not chexpert_conv.empty:
            piv = chexpert_conv.pivot_table(
                index="backbone",
                columns="attribute",
                values="n_flagged_methods",
                aggfunc="mean",
            )
            fig, ax = plt.subplots(figsize=(7, 5))
            im = ax.imshow(piv.to_numpy(), aspect="auto")
            ax.set_xticks(np.arange(piv.shape[1]))
            ax.set_xticklabels(piv.columns.tolist(), rotation=45, ha="right")
            ax.set_yticks(np.arange(piv.shape[0]))
            ax.set_yticklabels(piv.index.tolist())
            ax.set_title("CheXpert Convergence Matrix (Mean # flagged methods)")
            fig.colorbar(im, ax=ax, label="# flagged methods")
            fig.tight_layout()
            fig.savefig(self.figures_dir / "chexpert_convergence_matrix.png", dpi=160)
            plt.close(fig)

    def _write_summary_markdown(
        self,
        synthetic_pr: pd.DataFrame,
        synthetic_fp: pd.DataFrame,
        chexpert_methods: pd.DataFrame,
        chexpert_conv: pd.DataFrame,
        baselines: pd.DataFrame,
    ) -> None:
        lines: list[str] = []
        lines.append("# Paper Benchmark Summary")
        lines.append("")
        lines.append(f"- Benchmark: `{self.config.benchmark_name}`")
        lines.append(f"- Profile: `{self.config.profile}`")
        lines.append(f"- Generated: `{datetime.now(timezone.utc).isoformat()}`")
        lines.append("")

        if not synthetic_pr.empty:
            core = synthetic_pr[synthetic_pr["method"].isin(self.config.methods)]
            agg = core.groupby("method", as_index=False)[["precision", "recall", "f1"]].mean()
            lines.append("## Synthetic Shortcut-Dim Recovery")
            lines.append("```text")
            lines.append(agg.to_string(index=False))
            lines.append("```")
            lines.append("")

        if not synthetic_fp.empty:
            n_m = len(self.config.methods)
            lines.append("## Synthetic False Positive Controls")
            # Build unique FP column mapping: label -> column name
            fp_entries: list[tuple[str, str]] = [
                (f"1/{n_m}", f"false_positive_1of{n_m}"),
            ]
            near_col = f"false_positive_{max(n_m - 1, 1)}of{n_m}"
            if near_col != fp_entries[0][1]:
                fp_entries.append((f">={n_m - 1}/{n_m}", near_col))
            all_col = f"false_positive_{n_m}of{n_m}"
            if all_col != fp_entries[-1][1]:
                fp_entries.append((f"{n_m}/{n_m}", all_col))
            for label, col in fp_entries:
                if col in synthetic_fp.columns:
                    val = synthetic_fp[col].mean()
                    lines.append(f"- Mean FP rate when {label} methods flag: {float(val):.4f}")
            lines.append("")

        if not chexpert_methods.empty:
            agg = (
                chexpert_methods.groupby(["backbone", "attribute"], as_index=False)["flagged"]
                .mean()
                .rename(columns={"flagged": "mean_flag_rate"})
            )
            lines.append("## CheXpert Method Flag Rates")
            lines.append("```text")
            lines.append(agg.to_string(index=False))
            lines.append("```")
            lines.append("")

        if not chexpert_conv.empty:
            lines.append("## CheXpert Convergence Buckets")
            conv = chexpert_conv.groupby("confidence_level", as_index=False).size()
            lines.append("```text")
            lines.append(conv.to_string(index=False))
            lines.append("```")
            lines.append("")

        if not baselines.empty:
            lines.append("## External Baseline Status")
            status = baselines.groupby(["tool", "status"], as_index=False).size()
            lines.append("```text")
            lines.append(status.to_string(index=False))
            lines.append("```")
            lines.append("")

        lines.append("## Artifacts")
        lines.append(f"- Runs table: `{self.artifacts.runs_path}`")
        lines.append(f"- Synthetic PR table: `{self.artifacts.synthetic_pr_path}`")
        lines.append(f"- Synthetic FP table: `{self.artifacts.synthetic_fp_path}`")
        lines.append(f"- Correction control table: `{self.artifacts.correction_path}`")
        lines.append(f"- CheXpert methods table: `{self.artifacts.chexpert_methods_path}`")
        lines.append(f"- CheXpert convergence table: `{self.artifacts.chexpert_convergence_path}`")
        lines.append(f"- Baseline comparison table: `{self.artifacts.baselines_path}`")
        lines.append(f"- Figures directory: `{self.figures_dir}`")

        self.artifacts.summary_md_path.write_text("\n".join(lines), encoding="utf-8")

    def run(self) -> dict[str, Any]:
        set_seed(self.config.random_seed)

        synthetic_runs, synthetic_fp, correction_df = self._run_synthetic()
        chexpert_methods, chexpert_conv, baselines = self._run_chexpert()

        all_runs = synthetic_runs.copy()
        if not chexpert_methods.empty:
            chex_runs = chexpert_methods.copy()
            chex_runs["dataset"] = "chexpert"
            all_runs = pd.concat([all_runs, chex_runs], ignore_index=True, sort=False)

        all_runs.to_csv(self.artifacts.runs_path, index=False)
        synthetic_runs.to_csv(self.artifacts.synthetic_pr_path, index=False)
        synthetic_fp.to_csv(self.artifacts.synthetic_fp_path, index=False)
        correction_df.to_csv(self.artifacts.correction_path, index=False)
        chexpert_methods.to_csv(self.artifacts.chexpert_methods_path, index=False)
        chexpert_conv.to_csv(self.artifacts.chexpert_convergence_path, index=False)
        baselines.to_csv(self.artifacts.baselines_path, index=False)

        self._generate_plots(synthetic_runs, synthetic_fp, correction_df, chexpert_conv)
        self._write_manifest()
        self._write_summary_markdown(
            synthetic_pr=synthetic_runs,
            synthetic_fp=synthetic_fp,
            chexpert_methods=chexpert_methods,
            chexpert_conv=chexpert_conv,
            baselines=baselines,
        )

        return {
            "runs": all_runs,
            "synthetic_runs": synthetic_runs,
            "synthetic_fp": synthetic_fp,
            "correction": correction_df,
            "chexpert_methods": chexpert_methods,
            "chexpert_convergence": chexpert_conv,
            "baselines": baselines,
            "output_dir": str(self.output_dir),
        }


def run_paper_benchmark(config: PaperBenchmarkConfig) -> dict[str, Any]:
    return PaperBenchmarkRunner(config).run()


def _parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run paper benchmark for shortcut_detect.")
    parser.add_argument("--config", required=True, help="Path to config (.json/.yaml/.yml)")
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = _parse_args(argv)
    cfg = PaperBenchmarkConfig.from_path(args.config)
    runner = PaperBenchmarkRunner(cfg)
    runner.run()
    print(f"Paper benchmark complete. Artifacts in: {runner.output_dir}")
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
