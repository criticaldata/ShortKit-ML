"""False positive rate analysis on clean (no-shortcut) data.

Generates embeddings with effect_size=0 and measures how often each method
falsely detects shortcuts. Key claim: multi-method convergence reduces false alarms.

Usage:
    from shortcut_detect.benchmark.fp_analysis import FalsePositiveAnalyzer

    analyzer = FalsePositiveAnalyzer(methods=["hbac", "probe", "statistical", "geometric"], n_seeds=20)
    results = analyzer.run(n_samples=1000, embedding_dim=128)
    print(results.summary())
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import pandas as pd

from shortcut_detect.benchmark.measurement import MeasurementHarness, method_detected
from shortcut_detect.benchmark.synthetic_generator import SyntheticGenerator


@dataclass
class FPResult:
    """Container for false positive analysis results.

    Attributes
    ----------
    method_fp_rates : dict mapping method name to its false positive rate.
    convergence_fp_rate : float
        Rate at which 2 or more methods simultaneously flag a clean dataset.
    n_seeds : int
        Number of seeds (independent runs) used in the analysis.
    per_seed_results : pd.DataFrame
        Detailed per-seed, per-method results.
    """

    method_fp_rates: dict[str, float]
    convergence_fp_rate: float
    n_seeds: int
    per_seed_results: pd.DataFrame

    def summary(self) -> str:
        """Return a human-readable summary of the false positive analysis."""
        lines = [
            f"False Positive Analysis ({self.n_seeds} seeds)",
            "=" * 50,
            "",
            "Per-method FP rates:",
        ]
        for method, rate in sorted(self.method_fp_rates.items()):
            lines.append(f"  {method:15s}: {rate:.3f}")
        lines.append("")
        lines.append(f"Convergence FP rate (2+ methods agree): {self.convergence_fp_rate:.3f}")
        return "\n".join(lines)


class FalsePositiveAnalyzer:
    """Measure false positive rates on clean (no-shortcut) data.

    Parameters
    ----------
    methods : list of str
        Detection methods to evaluate.
    n_seeds : int
        Number of independent random seeds to run.
    base_seed : int
        Starting seed; actual seeds are ``base_seed + i`` for ``i`` in
        ``range(n_seeds)``.
    """

    def __init__(
        self,
        methods: list[str] | None = None,
        n_seeds: int = 20,
        base_seed: int = 7000,
    ) -> None:
        if methods is None:
            methods = ["hbac", "probe", "statistical", "geometric"]
        self.methods = methods
        self.n_seeds = n_seeds
        self.base_seed = base_seed

    def run(
        self,
        n_samples: int = 1000,
        embedding_dim: int = 128,
        group_ratio: float = 0.5,
    ) -> FPResult:
        """Run the false positive analysis.

        Generates ``n_seeds`` clean datasets (``effect_size=0.0``) and runs
        all methods on each.  Returns per-method FP rates and the
        multi-method convergence FP rate (i.e., fraction of seeds where
        2 or more methods simultaneously flag a shortcut).

        Parameters
        ----------
        n_samples : int
            Number of samples per synthetic dataset.
        embedding_dim : int
            Embedding dimensionality.
        group_ratio : float
            Fraction of samples in the majority group.

        Returns
        -------
        FPResult
        """
        harness = MeasurementHarness(methods=self.methods, seed=self.base_seed)
        rows: list[dict[str, Any]] = []

        for i in range(self.n_seeds):
            seed = self.base_seed + i
            gen = SyntheticGenerator(
                n_samples=n_samples,
                embedding_dim=embedding_dim,
                shortcut_dims=5,
                group_ratio=group_ratio,
                seed=seed,
            )
            data = gen.generate(effect_size=0.0)

            # Run methods through the harness's internal runner.
            raw_results = harness._run_methods(
                data.embeddings, data.labels, data.group_labels, seed
            )

            seed_flags: dict[str, bool] = {}
            for method in self.methods:
                flagged = method_detected(method, raw_results[method])
                seed_flags[method] = flagged
                rows.append(
                    {
                        "seed": seed,
                        "method": method,
                        "flagged": flagged,
                    }
                )

            n_flagged = sum(seed_flags.values())
            rows.append(
                {
                    "seed": seed,
                    "method": "convergence",
                    "flagged": n_flagged >= 2,
                }
            )

        df = pd.DataFrame(rows)

        # Compute per-method FP rates.
        method_fp_rates: dict[str, float] = {}
        for method in self.methods:
            method_rows = df[df["method"] == method]
            method_fp_rates[method] = float(method_rows["flagged"].mean())

        # Convergence FP rate.
        conv_rows = df[df["method"] == "convergence"]
        convergence_fp_rate = float(conv_rows["flagged"].mean())

        return FPResult(
            method_fp_rates=method_fp_rates,
            convergence_fp_rate=convergence_fp_rate,
            n_seeds=self.n_seeds,
            per_seed_results=df,
        )
