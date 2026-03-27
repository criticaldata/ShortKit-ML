"""Benchmark utilities for multi-seed shortcut detector evaluation."""

from .baseline_comparison import (
    BaselineComparison,
    ComparisonResult,
    ToolkitResult,
    generate_feature_comparison_table,
)
from .convergence_viz import (
    ConvergenceMatrix,
    plot_agreement_summary,
    plot_convergence_matrix,
)
from .figures import (
    generate_fp_rate_table,
    generate_synthetic_results_table,
    plot_method_comparison_bar,
    plot_sensitivity_analysis,
)
from .fp_analysis import FalsePositiveAnalyzer, FPResult
from .measurement import (
    HarnessResult,
    MeasurementHarness,
    MethodResult,
    bootstrap_ci,
    method_detected,
    precision_recall_f1,
    probe_permutation_pvalue,
)
from .paper_runner import PaperBenchmarkConfig, PaperBenchmarkRunner, run_paper_benchmark
from .runner import BenchmarkConfig, BenchmarkRunner, run_benchmark
from .sensitivity import SensitivitySweep, SweepResult
from .synthetic import (
    MAX_SHORTCUT_EFFECT_SIZE,
    MIN_SHORTCUT_EFFECT_SIZE,
    SyntheticShortcutConfig,
    SyntheticShortcutDataset,
    generate_parametric_shortcut_dataset,
)
from .synthetic_generator import (
    SyntheticGenerator,
    SyntheticResult,
    generate_correlated_parametric,
    generate_distributed_parametric,
    generate_parametric,
)

__all__ = [
    "BenchmarkConfig",
    "BenchmarkRunner",
    "run_benchmark",
    "PaperBenchmarkConfig",
    "PaperBenchmarkRunner",
    "run_paper_benchmark",
    "SyntheticGenerator",
    "SyntheticResult",
    "generate_parametric",
    "generate_correlated_parametric",
    "generate_distributed_parametric",
    "SensitivitySweep",
    "SweepResult",
    "ConvergenceMatrix",
    "plot_convergence_matrix",
    "plot_agreement_summary",
    "MeasurementHarness",
    "HarnessResult",
    "MethodResult",
    "bootstrap_ci",
    "method_detected",
    "precision_recall_f1",
    "probe_permutation_pvalue",
    "FalsePositiveAnalyzer",
    "FPResult",
    "BaselineComparison",
    "ComparisonResult",
    "ToolkitResult",
    "generate_feature_comparison_table",
    "generate_fp_rate_table",
    "generate_synthetic_results_table",
    "plot_sensitivity_analysis",
    "plot_method_comparison_bar",
    "SyntheticShortcutConfig",
    "SyntheticShortcutDataset",
    "generate_parametric_shortcut_dataset",
    "MAX_SHORTCUT_EFFECT_SIZE",
    "MIN_SHORTCUT_EFFECT_SIZE",
]
