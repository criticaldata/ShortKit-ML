"""Reproducibility smoke tests (pytest version for CI).

A lightweight subset of scripts/test_reproducibility.py that can run in CI
without real data files. Tests synthetic generator determinism, harness
reproducibility, and basic detection correctness.

Run with:
    pytest tests/test_reproducibility_smoke.py -v
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

PROJECT_ROOT = Path(__file__).resolve().parent.parent


# ---------------------------------------------------------------------------
# 1. Synthetic Generator
# ---------------------------------------------------------------------------


class TestSyntheticGeneratorReproducibility:
    """Verify SyntheticGenerator produces identical output for the same seed."""

    def test_seed42_deterministic(self):
        from shortcut_detect.benchmark.synthetic_generator import SyntheticGenerator

        gen1 = SyntheticGenerator(n_samples=200, embedding_dim=64, shortcut_dims=3, seed=42)
        r1 = gen1.generate(effect_size=0.8)

        gen2 = SyntheticGenerator(n_samples=200, embedding_dim=64, shortcut_dims=3, seed=42)
        r2 = gen2.generate(effect_size=0.8)

        np.testing.assert_array_equal(r1.embeddings, r2.embeddings)
        np.testing.assert_array_equal(r1.labels, r2.labels)
        np.testing.assert_array_equal(r1.group_labels, r2.group_labels)
        assert r1.shortcut_dims == r2.shortcut_dims

    def test_different_seeds_differ(self):
        from shortcut_detect.benchmark.synthetic_generator import SyntheticGenerator

        r1 = SyntheticGenerator(seed=1).generate(effect_size=0.8)
        r2 = SyntheticGenerator(seed=2).generate(effect_size=0.8)
        assert not np.array_equal(r1.embeddings, r2.embeddings)

    def test_effect_size_zero_no_separation(self):
        from shortcut_detect.benchmark.synthetic_generator import SyntheticGenerator

        gen = SyntheticGenerator(n_samples=1000, embedding_dim=64, shortcut_dims=5, seed=42)
        r = gen.generate(effect_size=0.0)
        for dim in r.shortcut_dims:
            g0 = r.embeddings[r.group_labels == 0, dim]
            g1 = r.embeddings[r.group_labels == 1, dim]
            diff = abs(float(g0.mean() - g1.mean()))
            assert diff < 0.3, f"Dim {dim}: diff={diff:.3f} too large for delta=0"

    def test_effect_size_strong_separation(self):
        from shortcut_detect.benchmark.synthetic_generator import SyntheticGenerator

        gen = SyntheticGenerator(n_samples=1000, embedding_dim=64, shortcut_dims=5, seed=42)
        r = gen.generate(effect_size=2.0)
        for dim in r.shortcut_dims:
            g0 = r.embeddings[r.group_labels == 0, dim]
            g1 = r.embeddings[r.group_labels == 1, dim]
            diff = float(g1.mean() - g0.mean())
            assert diff > 3.0, f"Dim {dim}: diff={diff:.3f} too small for delta=2.0"


# ---------------------------------------------------------------------------
# 2. MeasurementHarness
# ---------------------------------------------------------------------------


class TestHarnessReproducibility:
    """Verify that harness results are deterministic and sensible."""

    def test_deterministic_results(self):
        from shortcut_detect.benchmark.measurement import MeasurementHarness
        from shortcut_detect.benchmark.synthetic_generator import SyntheticGenerator

        gen = SyntheticGenerator(n_samples=300, embedding_dim=64, shortcut_dims=3, seed=42)
        data = gen.generate(effect_size=1.0)

        harness = MeasurementHarness(methods=["probe", "geometric"], seed=42)

        r1 = harness.evaluate(
            data.embeddings, data.labels, data.group_labels, data.shortcut_dims, seed=42
        )
        r2 = harness.evaluate(
            data.embeddings, data.labels, data.group_labels, data.shortcut_dims, seed=42
        )

        for mr1, mr2 in zip(r1.method_results, r2.method_results, strict=False):
            assert mr1.method == mr2.method
            assert mr1.precision == mr2.precision
            assert mr1.recall == mr2.recall
            assert mr1.f1 == mr2.f1
            assert mr1.detected == mr2.detected

    def test_strong_effect_detected(self):
        from shortcut_detect.benchmark.measurement import MeasurementHarness
        from shortcut_detect.benchmark.synthetic_generator import SyntheticGenerator

        gen = SyntheticGenerator(n_samples=500, embedding_dim=64, shortcut_dims=5, seed=42)
        data = gen.generate(effect_size=2.0)

        harness = MeasurementHarness(methods=["probe", "geometric", "statistical"], seed=42)
        result = harness.evaluate(
            data.embeddings,
            data.labels,
            data.group_labels,
            data.shortcut_dims,
            seed=42,
        )

        n_detected = sum(1 for mr in result.method_results if mr.detected)
        assert (
            n_detected >= 1
        ), f"Expected >= 1 method to detect at effect_size=2.0, got {n_detected}"

    def test_four_method_smoke(self):
        """Run all 4 standard methods on small synthetic data."""
        from shortcut_detect.benchmark.measurement import MeasurementHarness
        from shortcut_detect.benchmark.synthetic_generator import SyntheticGenerator

        gen = SyntheticGenerator(n_samples=300, embedding_dim=64, shortcut_dims=3, seed=42)
        data = gen.generate(effect_size=1.0)

        harness = MeasurementHarness(
            methods=["hbac", "probe", "statistical", "geometric"],
            seed=42,
        )
        result = harness.evaluate(
            data.embeddings,
            data.labels,
            data.group_labels,
            data.shortcut_dims,
            seed=42,
        )

        assert len(result.method_results) == 4
        assert result.convergence_level is not None
        assert "/" in result.convergence_level


# ---------------------------------------------------------------------------
# 3. FalsePositiveAnalyzer
# ---------------------------------------------------------------------------


class TestFPAnalyzerSmoke:
    """Verify FP analyzer produces bounded rates."""

    def test_convergence_fp_bounded(self):
        from shortcut_detect.benchmark.fp_analysis import FalsePositiveAnalyzer

        analyzer = FalsePositiveAnalyzer(
            methods=["probe", "geometric"],
            n_seeds=5,
            base_seed=7000,
        )
        result = analyzer.run(n_samples=300, embedding_dim=64)

        # Convergence FP rate should not be extremely high on clean data
        assert (
            result.convergence_fp_rate <= 0.6
        ), f"Convergence FP rate {result.convergence_fp_rate:.3f} unexpectedly high"


# ---------------------------------------------------------------------------
# 4. Real data file existence (conditional -- skip if not available)
# ---------------------------------------------------------------------------

_CELEBA_DIR = PROJECT_ROOT / "data" / "celeba"
_MIMIC_DIR = PROJECT_ROOT / "data" / "mimic_cxr"
_CHEXPERT_EMB = PROJECT_ROOT / "data" / "chest_embeddings.npy"
_RESULTS_DIR = PROJECT_ROOT / "output" / "paper_benchmark"


@pytest.mark.skipif(
    not (_CELEBA_DIR / "celeba_real_embeddings.npy").exists(),
    reason="CelebA data not available",
)
class TestCelebADataFiles:
    def test_embeddings_shape(self):
        emb = np.load(_CELEBA_DIR / "celeba_real_embeddings.npy")
        assert emb.shape == (10000, 2048)

    def test_metadata_shape(self):
        meta = pd.read_csv(_CELEBA_DIR / "celeba_real_metadata.csv")
        assert len(meta) == 10000

    def test_attributes_exist(self):
        import pandas as pd

        attr = pd.read_csv(_CELEBA_DIR / "celeba_real_attributes.csv")
        assert "Male" in attr.columns
        assert "Blond_Hair" in attr.columns


@pytest.mark.skipif(
    not (_MIMIC_DIR / "rad_dino_embeddings.npy").exists(),
    reason="MIMIC data not available",
)
class TestMIMICDataFiles:
    @pytest.mark.parametrize("backbone", ["rad_dino", "vit16_cls", "medsiglip"])
    def test_backbone_files_exist(self, backbone):
        emb_path = _MIMIC_DIR / f"{backbone}_embeddings.npy"
        meta_path = _MIMIC_DIR / f"{backbone}_metadata.csv"
        assert emb_path.exists()
        assert meta_path.exists()
        emb = np.load(emb_path)
        assert emb.ndim == 2
        assert emb.shape[0] > 0


@pytest.mark.skipif(
    not _CHEXPERT_EMB.exists(),
    reason="CheXpert data not available",
)
class TestCheXpertDataFiles:
    def test_embeddings_shape(self):
        emb = np.load(_CHEXPERT_EMB)
        assert emb.ndim == 2
        assert emb.shape[1] == 512


# ---------------------------------------------------------------------------
# 5. Benchmark result files (conditional)
# ---------------------------------------------------------------------------


@pytest.mark.skipif(
    not (_RESULTS_DIR / "celeba_real_results" / "aggregate_report.json").exists(),
    reason="CelebA results not available",
)
class TestCelebAResults:
    def test_aggregate_report_structure(self):
        with open(_RESULTS_DIR / "celeba_real_results" / "aggregate_report.json") as f:
            data = json.load(f)
        pairs = data.get("shortcut_pairs", [])
        assert len(pairs) >= 2
        tasks = {p["task"] for p in pairs}
        assert "Blond_Hair" in tasks
        assert "Heavy_Makeup" in tasks

    def test_probe_detects_known_shortcuts(self):
        with open(_RESULTS_DIR / "celeba_real_results" / "aggregate_report.json") as f:
            data = json.load(f)
        for pair in data["shortcut_pairs"]:
            detected_by = pair.get("detected_by", {})
            assert detected_by.get("probe", False), f"Probe should detect {pair['task']} shortcut"


@pytest.mark.skipif(
    not (_RESULTS_DIR / "mimic_cxr_results" / "mimic_convergence.csv").exists(),
    reason="MIMIC results not available",
)
class TestMIMICResults:
    def test_all_backbone_attribute_combos_flagged(self):
        import pandas as pd

        conv = pd.read_csv(_RESULTS_DIR / "mimic_cxr_results" / "mimic_convergence.csv")
        assert len(conv) == 12, f"Expected 12 rows (4 backbones x 3 attrs), got {len(conv)}"
        for _, row in conv.iterrows():
            assert row["n_flagged_methods"] >= 2, (
                f"{row['backbone']}/{row['attribute']}: only "
                f"{row['n_flagged_methods']}/{row.get('n_total_methods', 10)} flagged"
            )

    def test_rad_dino_sex_four_flagged(self):
        import pandas as pd

        conv = pd.read_csv(_RESULTS_DIR / "mimic_cxr_results" / "mimic_convergence.csv")
        rd_sex = conv[(conv["backbone"] == "rad_dino") & (conv["attribute"] == "sex")]
        assert len(rd_sex) == 1
        assert int(rd_sex["n_flagged_methods"].values[0]) == 6


@pytest.mark.skipif(
    not (_RESULTS_DIR / "chexpert_results" / "chexpert_convergence.csv").exists(),
    reason="CheXpert results not available",
)
class TestCheXpertResults:
    def test_convergence_structure(self):
        import pandas as pd

        conv = pd.read_csv(_RESULTS_DIR / "chexpert_results" / "chexpert_convergence.csv")
        assert len(conv) == 3, f"Expected 3 rows, got {len(conv)}"
        attrs = set(conv["attribute"].unique())
        assert {"race", "sex", "age_bin"} == attrs


# ---------------------------------------------------------------------------
# 6. Paper table files (conditional)
# ---------------------------------------------------------------------------


@pytest.mark.skipif(
    not (PROJECT_ROOT / "output" / "paper_tables" / "table_a_synthetic_pr.tex").exists(),
    reason="Paper tables not generated (run scripts/generate_all_paper_tables.py first)",
)
class TestPaperTables:
    @pytest.mark.parametrize(
        "table_name",
        [
            "table_a_synthetic_pr.tex",
            "table_b_fp_rates.tex",
            "table_d_chexpert.tex",
            "table_e_mimic.tex",
            "table_f_celeba.tex",
        ],
    )
    def test_table_exists_and_valid(self, table_name):
        path = PROJECT_ROOT / "output" / "paper_tables" / table_name
        assert path.exists(), f"Missing: {path}"
        content = path.read_text()
        assert len(content) > 50, f"{table_name} appears empty"
        assert "\\begin{table" in content
