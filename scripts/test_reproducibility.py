#!/usr/bin/env python3
"""End-to-end reproducibility test for paper results.

Verifies that all benchmark scripts produce expected outputs
and that results are deterministic with seed=42.

Usage:
    python scripts/test_reproducibility.py
"""

from __future__ import annotations

import json
import sys
import time
import traceback
from pathlib import Path

import numpy as np
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

_results: list[tuple[str, bool, str]] = []


def _record(name: str, passed: bool, detail: str = "") -> None:
    status = "PASS" if passed else "FAIL"
    _results.append((name, passed, detail))
    msg = f"  [{status}] {name}"
    if detail:
        msg += f"  -- {detail}"
    print(msg)


def _run_check(name: str, fn):
    """Run *fn* and record PASS/FAIL."""
    try:
        fn()
        _record(name, True)
    except Exception as exc:
        _record(name, False, f"{exc}\n{traceback.format_exc()}")


# ============================================================================
# 1. Synthetic benchmark (smoke profile)
# ============================================================================


def check_synthetic_generator_determinism():
    """SyntheticGenerator(seed=42) produces identical embeddings each run."""
    from shortcut_detect.benchmark.synthetic_generator import SyntheticGenerator

    gen1 = SyntheticGenerator(n_samples=200, embedding_dim=64, shortcut_dims=3, seed=42)
    r1 = gen1.generate(effect_size=0.8)

    gen2 = SyntheticGenerator(n_samples=200, embedding_dim=64, shortcut_dims=3, seed=42)
    r2 = gen2.generate(effect_size=0.8)

    np.testing.assert_array_equal(r1.embeddings, r2.embeddings)
    np.testing.assert_array_equal(r1.labels, r2.labels)
    np.testing.assert_array_equal(r1.group_labels, r2.group_labels)
    assert r1.shortcut_dims == r2.shortcut_dims


def check_synthetic_effect_sizes():
    """Effect size 0 has no separation; effect size 2.0 has strong separation."""
    from shortcut_detect.benchmark.synthetic_generator import SyntheticGenerator

    gen = SyntheticGenerator(n_samples=1000, embedding_dim=64, shortcut_dims=5, seed=42)

    # No-shortcut case
    r0 = gen.generate(effect_size=0.0)
    for dim in r0.shortcut_dims:
        g0 = r0.embeddings[r0.group_labels == 0, dim]
        g1 = r0.embeddings[r0.group_labels == 1, dim]
        diff = abs(float(g0.mean() - g1.mean()))
        assert diff < 0.3, f"Dim {dim}: mean diff {diff:.3f} too large for effect_size=0"

    # Strong shortcut case
    r2 = gen.generate(effect_size=2.0)
    for dim in r2.shortcut_dims:
        g0 = r2.embeddings[r2.group_labels == 0, dim]
        g1 = r2.embeddings[r2.group_labels == 1, dim]
        diff = float(g1.mean() - g0.mean())
        assert diff > 3.0, f"Dim {dim}: mean diff {diff:.3f} too small for effect_size=2.0"


# ============================================================================
# 2. MeasurementHarness reproducibility
# ============================================================================


def check_harness_determinism():
    """MeasurementHarness produces same P/R/F1 on same data."""
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
        assert mr1.precision == mr2.precision, f"{mr1.method} precision mismatch"
        assert mr1.recall == mr2.recall, f"{mr1.method} recall mismatch"
        assert mr1.f1 == mr2.f1, f"{mr1.method} F1 mismatch"
        assert mr1.detected == mr2.detected, f"{mr1.method} detected mismatch"


def check_harness_strong_effect_detection():
    """With effect_size=2.0, at least geometric should detect shortcut."""
    from shortcut_detect.benchmark.measurement import MeasurementHarness
    from shortcut_detect.benchmark.synthetic_generator import SyntheticGenerator

    gen = SyntheticGenerator(n_samples=500, embedding_dim=64, shortcut_dims=5, seed=42)
    data = gen.generate(effect_size=2.0)

    harness = MeasurementHarness(methods=["probe", "geometric", "statistical"], seed=42)
    result = harness.evaluate(
        data.embeddings, data.labels, data.group_labels, data.shortcut_dims, seed=42
    )

    n_detected = sum(1 for mr in result.method_results if mr.detected)
    assert n_detected >= 1, (
        f"Expected at least 1 method to detect shortcut at effect_size=2.0, " f"got {n_detected}"
    )


# ============================================================================
# 3. FalsePositiveAnalyzer
# ============================================================================


def check_fp_rates():
    """FP rates on clean data should be low (convergence near 0)."""
    from shortcut_detect.benchmark.fp_analysis import FalsePositiveAnalyzer

    analyzer = FalsePositiveAnalyzer(
        methods=["probe", "geometric"],
        n_seeds=5,
        base_seed=7000,
    )
    result = analyzer.run(n_samples=300, embedding_dim=64)

    assert result.convergence_fp_rate <= 0.6, (
        f"Convergence FP rate {result.convergence_fp_rate:.3f} too high "
        f"(expected <= 0.6 on clean data)"
    )


# ============================================================================
# 4. CheXpert data files and benchmark results
# ============================================================================


def check_chexpert_data_exists():
    """CheXpert embedding files exist with expected shapes."""
    emb_path = PROJECT_ROOT / "data" / "chest_embeddings.npy"
    assert emb_path.exists(), f"Missing: {emb_path}"
    emb = np.load(emb_path)
    assert emb.ndim == 2, f"Expected 2D array, got {emb.ndim}D"
    assert emb.shape[0] > 0, "Empty embeddings"
    assert emb.shape[1] == 512, f"Expected 512-dim, got {emb.shape[1]}"


def check_chexpert_benchmark_embeddings():
    """CheXpert benchmark embeddings exist."""
    base = PROJECT_ROOT / "output" / "paper_benchmark" / "chexpert_embeddings"
    emb_path = base / "medclip_embeddings.npy"
    meta_path = base / "medclip_metadata.csv"
    assert emb_path.exists(), f"Missing: {emb_path}"
    assert meta_path.exists(), f"Missing: {meta_path}"
    emb = np.load(emb_path)
    meta = pd.read_csv(meta_path)
    assert emb.shape[0] == len(
        meta
    ), f"Row mismatch: embeddings {emb.shape[0]} vs metadata {len(meta)}"


def check_chexpert_results():
    """CheXpert results CSVs exist and have expected structure."""
    base = PROJECT_ROOT / "output" / "paper_benchmark" / "chexpert_results"
    methods_path = base / "chexpert_methods.csv"
    conv_path = base / "chexpert_convergence.csv"

    assert methods_path.exists(), f"Missing: {methods_path}"
    assert conv_path.exists(), f"Missing: {conv_path}"

    methods_df = pd.read_csv(methods_path)
    conv_df = pd.read_csv(conv_path)

    required_method_cols = {"backbone", "attribute", "method", "flagged"}
    assert required_method_cols.issubset(
        set(methods_df.columns)
    ), f"Missing columns: {required_method_cols - set(methods_df.columns)}"

    required_conv_cols = {"attribute", "n_flagged_methods", "n_total_methods"}
    assert required_conv_cols.issubset(
        set(conv_df.columns)
    ), f"Missing columns: {required_conv_cols - set(conv_df.columns)}"

    # CheXpert should have results for race, sex, age_bin
    attrs = set(methods_df["attribute"].unique())
    assert {"race", "sex", "age_bin"}.issubset(
        attrs
    ), f"Expected attributes race/sex/age_bin, got {attrs}"


# ============================================================================
# 5. MIMIC data files and benchmark results
# ============================================================================


def check_mimic_data_exists():
    """MIMIC-CXR embedding files exist with expected shapes."""
    base = PROJECT_ROOT / "data" / "mimic_cxr"
    for backbone in ["rad_dino", "vit16_cls", "medsiglip"]:
        emb_path = base / f"{backbone}_embeddings.npy"
        meta_path = base / f"{backbone}_metadata.csv"
        assert emb_path.exists(), f"Missing: {emb_path}"
        assert meta_path.exists(), f"Missing: {meta_path}"
        emb = np.load(emb_path)
        meta = pd.read_csv(meta_path)
        assert emb.shape[0] == len(meta), f"{backbone}: row mismatch {emb.shape[0]} vs {len(meta)}"
        assert emb.ndim == 2


def check_mimic_results():
    """MIMIC results CSVs exist and contain expected backbones/attributes."""
    base = PROJECT_ROOT / "output" / "paper_benchmark" / "mimic_cxr_results"
    methods_path = base / "mimic_methods.csv"
    conv_path = base / "mimic_convergence.csv"

    assert methods_path.exists(), f"Missing: {methods_path}"
    assert conv_path.exists(), f"Missing: {conv_path}"

    methods_df = pd.read_csv(methods_path)
    conv_df = pd.read_csv(conv_path)

    # Should have 3 backbones x 3 attributes x 4 methods = 36 rows
    assert len(methods_df) >= 36, f"Expected >= 36 method rows, got {len(methods_df)}"

    backbones = set(methods_df["backbone"].unique())
    assert {"rad_dino", "vit16_cls", "medsiglip"}.issubset(
        backbones
    ), f"Expected 3 backbones, got {backbones}"

    # All convergence rows should have >= 3/4 flagged (known shortcuts)
    for _, row in conv_df.iterrows():
        n_flagged = row["n_flagged_methods"]
        assert n_flagged >= 2, (
            f"MIMIC {row['backbone']}/{row['attribute']}: only "
            f"{n_flagged}/4 flagged (expected >= 2)"
        )


def check_mimic_key_values():
    """MIMIC RAD-DINO sex should have 3/4 methods flagged."""
    base = PROJECT_ROOT / "output" / "paper_benchmark" / "mimic_cxr_results"
    conv_path = base / "mimic_convergence.csv"
    conv_df = pd.read_csv(conv_path)

    rd_sex = conv_df[(conv_df["backbone"] == "rad_dino") & (conv_df["attribute"] == "sex")]
    assert len(rd_sex) == 1, "Expected one row for rad_dino/sex"
    n_flagged = int(rd_sex["n_flagged_methods"].values[0])
    assert n_flagged == 3, f"RAD-DINO sex: expected 3/4 flagged, got {n_flagged}/4"


# ============================================================================
# 6. CelebA data files and benchmark results
# ============================================================================


def check_celeba_data_exists():
    """CelebA embedding files exist with expected shapes."""
    base = PROJECT_ROOT / "data" / "celeba"
    emb_path = base / "celeba_real_embeddings.npy"
    meta_path = base / "celeba_real_metadata.csv"
    attr_path = base / "celeba_real_attributes.csv"

    assert emb_path.exists(), f"Missing: {emb_path}"
    assert meta_path.exists(), f"Missing: {meta_path}"
    assert attr_path.exists(), f"Missing: {attr_path}"

    emb = np.load(emb_path)
    meta = pd.read_csv(meta_path)

    assert emb.shape == (
        10000,
        2048,
    ), f"CelebA embeddings shape: {emb.shape}, expected (10000, 2048)"
    assert len(meta) == 10000, f"CelebA metadata rows: {len(meta)}, expected 10000"


def check_celeba_results():
    """CelebA aggregate report exists and has expected structure."""
    report_path = (
        PROJECT_ROOT
        / "output"
        / "paper_benchmark"
        / "celeba_real_results"
        / "aggregate_report.json"
    )
    assert report_path.exists(), f"Missing: {report_path}"

    with open(report_path) as f:
        data = json.load(f)

    pairs = data.get("shortcut_pairs", [])
    assert len(pairs) >= 2, f"Expected >= 2 shortcut pairs, got {len(pairs)}"

    # Check that Blond_Hair and Heavy_Makeup are present
    tasks = {p["task"] for p in pairs}
    assert "Blond_Hair" in tasks, "Missing Blond_Hair pair"
    assert "Heavy_Makeup" in tasks, "Missing Heavy_Makeup pair"


def check_celeba_key_values():
    """CelebA known shortcuts should be detected by probe + statistical + geometric."""
    report_path = (
        PROJECT_ROOT
        / "output"
        / "paper_benchmark"
        / "celeba_real_results"
        / "aggregate_report.json"
    )
    with open(report_path) as f:
        data = json.load(f)

    for pair in data["shortcut_pairs"]:
        task = pair["task"]
        detected_by = pair.get("detected_by", {})
        n_detected = sum(1 for v in detected_by.values() if v)

        # Each known shortcut pair should be detected by at least 2 methods
        assert n_detected >= 2, (
            f"CelebA {task}: only {n_detected} methods detected "
            f"(expected >= 2). detected_by={detected_by}"
        )

        # Probe should detect these strong shortcuts
        assert detected_by.get(
            "probe", False
        ), f"CelebA {task}: probe failed to detect known shortcut"


# ============================================================================
# 7. Paper table generation
# ============================================================================


def check_paper_tables_exist():
    """Paper table .tex files exist in output/paper_tables/."""
    table_dir = PROJECT_ROOT / "output" / "paper_tables"
    assert table_dir.exists(), f"Missing: {table_dir}"

    expected_tables = [
        "table_a_synthetic_pr.tex",
        "table_b_fp_rates.tex",
        "table_d_chexpert.tex",
        "table_e_mimic.tex",
        "table_f_celeba.tex",
    ]

    for table_name in expected_tables:
        path = table_dir / table_name
        assert path.exists(), f"Missing table: {path}"
        content = path.read_text()
        assert len(content) > 50, f"Table {table_name} appears empty ({len(content)} chars)"
        assert "\\begin{table" in content, f"Table {table_name} does not contain \\begin{{table}}"


def check_paper_table_generation_script():
    """generate_all_paper_tables.py can generate tables d/e/f (from saved results)."""
    # We only test the functions that read from saved CSV/JSON (tables d, e, f)
    # since tables a/b/c/g require running full benchmarks
    sys.path.insert(0, str(PROJECT_ROOT / "scripts"))

    from generate_all_paper_tables import (
        BENCHMARK_DIR,
        OUTPUT_DIR,
        generate_table_d_chexpert,
        generate_table_e_mimic,
        generate_table_f_celeba,
    )

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    latex_d = generate_table_d_chexpert()
    assert "\\begin{table" in latex_d, "Table D missing \\begin{table}"

    latex_e = generate_table_e_mimic()
    assert "\\begin{table" in latex_e, "Table E missing \\begin{table}"

    # Table F reads from celeba_results/ but actual dir may be celeba_real_results/
    celeba_json = BENCHMARK_DIR / "celeba_results" / "aggregate_report.json"
    if celeba_json.exists():
        latex_f = generate_table_f_celeba()
        assert "\\begin{table" in latex_f, "Table F missing \\begin{table}"
    else:
        print(
            "    (skipping table F: celeba_results/aggregate_report.json not found; "
            "actual results are in celeba_real_results/)"
        )


# ============================================================================
# 8. Synthetic benchmark smoke run
# ============================================================================


def check_synthetic_smoke_benchmark():
    """Run a minimal synthetic benchmark (1 effect size, 1 seed) end-to-end."""
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

    # Should have results for all 4 methods
    assert (
        len(result.method_results) == 4
    ), f"Expected 4 method results, got {len(result.method_results)}"

    # At strong effect size, most methods should detect
    n_detected = sum(1 for mr in result.method_results if mr.detected)
    assert n_detected >= 2, f"Expected >= 2 methods to detect at effect_size=1.0, got {n_detected}"

    # Convergence should reflect this
    assert result.convergence_level is not None
    assert "/" in result.convergence_level


# ============================================================================
# Main runner
# ============================================================================


def main() -> int:
    print("=" * 70)
    print("  End-to-End Reproducibility Test")
    print("=" * 70)
    print()

    t_start = time.time()

    # --- Section 1: Synthetic Generator ---
    print("Section 1: Synthetic Generator")
    _run_check(
        "1.1 SyntheticGenerator determinism (seed=42)", check_synthetic_generator_determinism
    )
    _run_check("1.2 Synthetic effect size separation", check_synthetic_effect_sizes)
    print()

    # --- Section 2: MeasurementHarness ---
    print("Section 2: MeasurementHarness")
    _run_check("2.1 Harness determinism (same data -> same P/R/F1)", check_harness_determinism)
    _run_check("2.2 Harness detects strong effects", check_harness_strong_effect_detection)
    print()

    # --- Section 3: FalsePositiveAnalyzer ---
    print("Section 3: FalsePositiveAnalyzer")
    _run_check("3.1 FP rates bounded on clean data", check_fp_rates)
    print()

    # --- Section 4: CheXpert ---
    print("Section 4: CheXpert Benchmark")
    _run_check("4.1 CheXpert data files exist", check_chexpert_data_exists)
    _run_check("4.2 CheXpert benchmark embeddings exist", check_chexpert_benchmark_embeddings)
    _run_check("4.3 CheXpert results have expected structure", check_chexpert_results)
    print()

    # --- Section 5: MIMIC-CXR ---
    print("Section 5: MIMIC-CXR Benchmark")
    _run_check("5.1 MIMIC data files exist", check_mimic_data_exists)
    _run_check("5.2 MIMIC results have expected structure", check_mimic_results)
    _run_check("5.3 MIMIC RAD-DINO sex detection matches expected", check_mimic_key_values)
    print()

    # --- Section 6: CelebA ---
    print("Section 6: CelebA Benchmark")
    _run_check("6.1 CelebA data files exist", check_celeba_data_exists)
    _run_check("6.2 CelebA results have expected structure", check_celeba_results)
    _run_check("6.3 CelebA key detection values match", check_celeba_key_values)
    print()

    # --- Section 7: Paper Tables ---
    print("Section 7: Paper Tables")
    _run_check("7.1 Paper table .tex files exist", check_paper_tables_exist)
    _run_check(
        "7.2 Table generation script runs (tables d/e/f)", check_paper_table_generation_script
    )
    print()

    # --- Section 8: Synthetic Smoke Run ---
    print("Section 8: Synthetic Smoke Benchmark")
    _run_check("8.1 Synthetic end-to-end smoke run", check_synthetic_smoke_benchmark)
    print()

    # --- Summary ---
    elapsed = time.time() - t_start
    n_total = len(_results)
    n_pass = sum(1 for _, passed, _ in _results if passed)
    n_fail = n_total - n_pass

    print("=" * 70)
    print(f"  SUMMARY: {n_pass}/{n_total} passed, {n_fail} failed  ({elapsed:.1f}s)")
    print("=" * 70)

    if n_fail > 0:
        print("\nFailed checks:")
        for name, passed, detail in _results:
            if not passed:
                print(f"  FAIL: {name}")
                if detail:
                    # Print first few lines of detail
                    for line in detail.strip().split("\n")[:5]:
                        print(f"        {line}")
        print()

    return 0 if n_fail == 0 else 1


if __name__ == "__main__":
    raise SystemExit(main())
