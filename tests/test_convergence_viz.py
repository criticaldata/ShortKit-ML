"""Tests for shortcut_detect.benchmark.convergence_viz module."""

from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.figure  # noqa: E402
import pandas as pd  # noqa: E402

from shortcut_detect.benchmark.convergence_viz import (
    ConvergenceMatrix,
    plot_agreement_summary,
    plot_convergence_matrix,
)

# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

METHODS = ["hbac", "probe", "statistical", "geometric"]


def _make_matrix() -> ConvergenceMatrix:
    """Build a small hand-crafted convergence matrix."""
    mat = ConvergenceMatrix(METHODS)
    mat.add_experiment(
        "exp_A", {"hbac": True, "probe": True, "statistical": True, "geometric": True}
    )
    mat.add_experiment(
        "exp_B", {"hbac": True, "probe": True, "statistical": True, "geometric": False}
    )
    mat.add_experiment(
        "exp_C", {"hbac": True, "probe": False, "statistical": False, "geometric": False}
    )
    mat.add_experiment(
        "exp_D", {"hbac": False, "probe": False, "statistical": False, "geometric": False}
    )
    return mat


# ---------------------------------------------------------------------------
# ConvergenceMatrix construction
# ---------------------------------------------------------------------------


class TestConvergenceMatrixManual:
    def test_methods_preserved(self):
        mat = _make_matrix()
        assert mat.methods == METHODS

    def test_experiment_names(self):
        mat = _make_matrix()
        assert mat.experiment_names == ["exp_A", "exp_B", "exp_C", "exp_D"]

    def test_to_dataframe_shape(self):
        mat = _make_matrix()
        df = mat.to_dataframe()
        assert df.shape == (4, 4)
        assert list(df.index) == METHODS
        assert list(df.columns) == ["exp_A", "exp_B", "exp_C", "exp_D"]

    def test_to_dataframe_values(self):
        mat = _make_matrix()
        df = mat.to_dataframe()
        # exp_A: all True
        assert df["exp_A"].tolist() == [True, True, True, True]
        # exp_D: all False
        assert df["exp_D"].tolist() == [False, False, False, False]

    def test_missing_method_defaults_false(self):
        mat = ConvergenceMatrix(METHODS)
        mat.add_experiment("exp", {"hbac": True})
        df = mat.to_dataframe()
        assert df["exp"]["hbac"] == True  # noqa: E712
        assert df["exp"]["probe"] == False  # noqa: E712


class TestAgreementLevels:
    def test_agreement_levels_strings(self):
        mat = _make_matrix()
        levels = mat.agreement_levels()
        assert levels["exp_A"] == "4/4"
        assert levels["exp_B"] == "3/4"
        assert levels["exp_C"] == "1/4"
        assert levels["exp_D"] == "0/4"

    def test_agreement_counts(self):
        mat = _make_matrix()
        counts = mat.agreement_counts()
        assert counts["exp_A"] == 4
        assert counts["exp_B"] == 3
        assert counts["exp_C"] == 1
        assert counts["exp_D"] == 0


# ---------------------------------------------------------------------------
# from_dataframe
# ---------------------------------------------------------------------------


class TestFromDataFrame:
    def test_basic_roundtrip(self):
        df = pd.DataFrame(
            {
                "method": ["hbac", "probe", "hbac", "probe"],
                "experiment": ["e1", "e1", "e2", "e2"],
                "flagged": [True, False, True, True],
            }
        )
        mat = ConvergenceMatrix.from_dataframe(df)
        assert set(mat.methods) == {"hbac", "probe"}
        assert set(mat.experiment_names) == {"e1", "e2"}
        result_df = mat.to_dataframe()
        assert result_df.loc["hbac", "e1"] == True  # noqa: E712
        assert result_df.loc["probe", "e1"] == False  # noqa: E712
        assert result_df.loc["probe", "e2"] == True  # noqa: E712

    def test_custom_column_names(self):
        df = pd.DataFrame(
            {
                "detector": ["a", "b", "a", "b"],
                "condition": ["c1", "c1", "c2", "c2"],
                "detected": [1, 0, 1, 1],
            }
        )
        mat = ConvergenceMatrix.from_dataframe(
            df,
            method_col="detector",
            experiment_col="condition",
            detected_col="detected",
        )
        assert mat.agreement_counts()["c1"] == 1
        assert mat.agreement_counts()["c2"] == 2


# ---------------------------------------------------------------------------
# from_benchmark_results
# ---------------------------------------------------------------------------


class TestFromBenchmarkResults:
    def test_synthetic_format(self):
        rows = []
        for method in METHODS:
            rows.append(
                {
                    "method": method,
                    "flagged": method != "geometric",
                    "effect_size": 0.5,
                    "n_samples": 500,
                }
            )
        # convergence summary row (should be ignored)
        rows.append(
            {
                "method": "convergence",
                "flagged": float("nan"),
                "effect_size": 0.5,
                "n_samples": 500,
            }
        )
        df = pd.DataFrame(rows)
        mat = ConvergenceMatrix.from_benchmark_results(df)
        assert "convergence" not in mat.methods
        assert len(mat.experiment_names) == 1
        counts = mat.agreement_counts()
        assert list(counts.values())[0] == 3

    def test_chexpert_format(self):
        rows = []
        for method in METHODS:
            rows.append(
                {
                    "method": method,
                    "flagged": True,
                    "backbone": "densenet121",
                    "attribute": "race",
                }
            )
        df = pd.DataFrame(rows)
        mat = ConvergenceMatrix.from_benchmark_results(df)
        assert len(mat.experiment_names) == 1
        assert "densenet121" in mat.experiment_names[0]
        assert "race" in mat.experiment_names[0]


# ---------------------------------------------------------------------------
# Plotting
# ---------------------------------------------------------------------------


class TestPlotConvergenceMatrix:
    def test_returns_figure(self):
        mat = _make_matrix()
        fig = plot_convergence_matrix(mat)
        assert isinstance(fig, matplotlib.figure.Figure)
        plt_mod = matplotlib.pyplot
        plt_mod.close(fig)

    def test_empty_matrix(self):
        mat = ConvergenceMatrix(METHODS)
        fig = plot_convergence_matrix(mat)
        assert isinstance(fig, matplotlib.figure.Figure)
        matplotlib.pyplot.close(fig)

    def test_save_png(self, tmp_path: Path):
        mat = _make_matrix()
        save_path = tmp_path / "convergence.png"
        fig = plot_convergence_matrix(mat, save_path=save_path)
        assert save_path.exists()
        assert save_path.stat().st_size > 0
        matplotlib.pyplot.close(fig)

    def test_save_pdf(self, tmp_path: Path):
        mat = _make_matrix()
        save_path = tmp_path / "convergence.pdf"
        fig = plot_convergence_matrix(mat, save_path=save_path)
        assert save_path.exists()
        assert save_path.stat().st_size > 0
        matplotlib.pyplot.close(fig)

    def test_custom_params(self):
        mat = _make_matrix()
        fig = plot_convergence_matrix(
            mat,
            title="Custom Title",
            figsize=(10, 5),
            dpi=72,
            font_size=12,
            show_agreement_row=False,
        )
        assert isinstance(fig, matplotlib.figure.Figure)
        matplotlib.pyplot.close(fig)


class TestPlotAgreementSummary:
    def test_returns_figure(self):
        mat = _make_matrix()
        fig = plot_agreement_summary(mat)
        assert isinstance(fig, matplotlib.figure.Figure)
        matplotlib.pyplot.close(fig)

    def test_save(self, tmp_path: Path):
        mat = _make_matrix()
        save_path = tmp_path / "summary.png"
        fig = plot_agreement_summary(mat, save_path=save_path)
        assert save_path.exists()
        matplotlib.pyplot.close(fig)


# ---------------------------------------------------------------------------
# More than 4 methods
# ---------------------------------------------------------------------------


class TestManyMethods:
    def test_six_methods(self):
        methods = ["m1", "m2", "m3", "m4", "m5", "m6"]
        mat = ConvergenceMatrix(methods)
        mat.add_experiment("e1", {m: True for m in methods})
        mat.add_experiment("e2", {m: (i % 2 == 0) for i, m in enumerate(methods)})
        assert mat.agreement_levels()["e1"] == "6/6"
        assert mat.agreement_levels()["e2"] == "3/6"
        fig = plot_convergence_matrix(mat)
        assert isinstance(fig, matplotlib.figure.Figure)
        matplotlib.pyplot.close(fig)
