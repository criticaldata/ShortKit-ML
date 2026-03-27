"""Tests for figures module (F04 + F06)."""

from __future__ import annotations

import os
import tempfile

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import pytest

from shortcut_detect.benchmark.figures import (
    generate_fp_rate_table,
    generate_synthetic_results_table,
    plot_sensitivity_analysis,
)


@pytest.fixture()
def sample_results_df():
    """Mock benchmark results DataFrame."""
    rows = []
    for es in [0.2, 0.5, 0.8, 1.2, 2.0]:
        for method in ["hbac", "probe", "statistical", "geometric"]:
            rows.append(
                {
                    "effect_size": es,
                    "method": method,
                    "precision": np.random.uniform(0.3, 1.0),
                    "recall": np.random.uniform(0.3, 1.0),
                    "f1": np.random.uniform(0.3, 1.0),
                    "fp_rate": np.random.uniform(0.0, 0.2),
                }
            )
    return pd.DataFrame(rows)


@pytest.fixture()
def sample_sweep_data():
    """Mock sensitivity sweep data with correct column names for plot_sensitivity_analysis."""
    rows = []
    for n_samples in [100, 500, 1000, 5000]:
        for method in ["hbac", "probe", "statistical", "geometric"]:
            for seed in range(3):
                rows.append(
                    {
                        "n_samples": n_samples,
                        "method": method,
                        "seed": seed,
                        "detected": np.random.choice([True, False]),
                        "detection_rate": np.random.uniform(0.0, 1.0),
                    }
                )
    return {
        "sample_size": pd.DataFrame(rows),
    }


class TestSyntheticResultsTable:
    def test_returns_string(self, sample_results_df):
        latex = generate_synthetic_results_table(sample_results_df)
        assert isinstance(latex, str)
        assert len(latex) > 0

    def test_contains_methods(self, sample_results_df):
        latex = generate_synthetic_results_table(sample_results_df)
        # Should contain method names in some form
        assert "probe" in latex.lower() or "Probe" in latex

    def test_save_to_file(self, sample_results_df):
        with tempfile.NamedTemporaryFile(suffix=".tex", delete=False) as f:
            path = f.name
        try:
            generate_synthetic_results_table(sample_results_df, output_path=path)
            assert os.path.exists(path)
            with open(path) as f:
                content = f.read()
            assert len(content) > 0
        finally:
            os.unlink(path)


class TestSensitivityPlots:
    def test_returns_figures(self, sample_sweep_data):
        figs = plot_sensitivity_analysis(sample_sweep_data)
        assert isinstance(figs, list | dict)
        plt.close("all")

    def test_save_figures(self, sample_sweep_data):
        with tempfile.TemporaryDirectory() as tmpdir:
            plot_sensitivity_analysis(sample_sweep_data, output_dir=tmpdir)
            # Should have created at least one file
            files = os.listdir(tmpdir)
            assert len(files) >= 1
        plt.close("all")


# ---------------------------------------------------------------------------
# generate_fp_rate_table (S03)
# ---------------------------------------------------------------------------


class TestFPRateTable:
    @pytest.fixture()
    def sample_fp_results(self):
        """Mock FalsePositiveAnalyzer results."""
        return {
            "method_fp_rates": {
                "hbac": 0.05,
                "probe": 0.10,
                "statistical": 0.15,
                "geometric": 0.08,
            },
            "convergence_fp_rate": 0.02,
        }

    def test_returns_latex_string(self, sample_fp_results):
        latex = generate_fp_rate_table(sample_fp_results)
        assert isinstance(latex, str)
        assert r"\begin{table}" in latex
        assert r"\end{table}" in latex

    def test_contains_method_names(self, sample_fp_results):
        latex = generate_fp_rate_table(sample_fp_results)
        assert "Probe" in latex
        assert "HBAC" in latex

    def test_contains_fp_rates(self, sample_fp_results):
        latex = generate_fp_rate_table(sample_fp_results)
        assert "0.100" in latex
        assert "0.020" in latex

    def test_save_to_file(self, sample_fp_results):
        with tempfile.NamedTemporaryFile(suffix=".tex", delete=False) as f:
            path = f.name
        try:
            generate_fp_rate_table(sample_fp_results, output_path=path)
            assert os.path.exists(path)
            md_path = path.replace(".tex", ".md")
            assert os.path.exists(md_path)
            with open(path) as f:
                content = f.read()
            assert len(content) > 0
            assert "FP Rate" in content
        finally:
            os.unlink(path)
            md_path = path.replace(".tex", ".md")
            if os.path.exists(md_path):
                os.unlink(md_path)

    def test_with_separate_correction_rates(self):
        fp_results = {
            "method_fp_rates": {"hbac": 0.10, "probe": 0.12},
            "convergence_fp_rate": 0.03,
            "bonferroni_fp_rates": {"hbac": 0.04, "probe": 0.06},
            "fdr_bh_fp_rates": {"hbac": 0.08, "probe": 0.10},
        }
        latex = generate_fp_rate_table(fp_results)
        assert "0.040" in latex  # bonferroni hbac
        assert "0.100" in latex  # fdr_bh probe

    def test_with_fp_result_object(self):
        """Test with an object that has attributes (like FPResult)."""
        from shortcut_detect.benchmark.fp_analysis import FPResult

        fp_result = FPResult(
            method_fp_rates={"hbac": 0.05, "probe": 0.10},
            convergence_fp_rate=0.01,
            n_seeds=20,
            per_seed_results=pd.DataFrame(),
        )
        latex = generate_fp_rate_table(fp_result)
        assert isinstance(latex, str)
        assert "0.050" in latex
