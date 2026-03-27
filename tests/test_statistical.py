"""Tests for statistical testing methods."""

import numpy as np
import pytest
from scipy.stats import mannwhitneyu, ttest_ind

from shortcut_detect.statistical import GroupDiffTest
from tests.fixtures.synthetic_data import generate_linear_shortcut, generate_multiclass_shortcut


def test_group_diff_test_binary():
    """Test GroupDiffTest on binary classification."""
    embeddings, labels = generate_linear_shortcut(n_samples=1000, embedding_dim=30, shortcut_dims=3)

    test = GroupDiffTest(test=mannwhitneyu)
    test.fit(embeddings, labels)
    p_values = test.get_pvalues()

    # Should have one comparison for binary case
    assert len(p_values) == 1
    assert "0_vs_1" in p_values

    # First few dimensions should have low p-values
    pvals = p_values["0_vs_1"]
    assert len(pvals) == 30
    assert np.min(pvals[:3]) < 0.01  # At least one shortcut dim significant


def test_group_diff_test_multiclass():
    """Test GroupDiffTest on multi-class problem."""
    embeddings, labels = generate_multiclass_shortcut(n_samples=900, embedding_dim=20, n_classes=3)

    test = GroupDiffTest(test=ttest_ind)
    test.fit(embeddings, labels)
    p_values = test.get_pvalues()

    # Should have pairwise and one-vs-rest comparisons
    # 3 classes: C(3,2) = 3 pairwise + 3 one-vs-rest = 6 total
    assert len(p_values) == 6

    # Check pairwise comparisons exist
    assert "0_vs_1" in p_values or "1_vs_0" in p_values
    assert "0_vs_2" in p_values or "2_vs_0" in p_values
    assert "1_vs_2" in p_values or "2_vs_1" in p_values

    # Check one-vs-rest comparisons
    assert "0_vs_rest" in p_values
    assert "1_vs_rest" in p_values
    assert "2_vs_rest" in p_values


def test_apply_threshold():
    """Test significance threshold application."""
    embeddings, labels = generate_linear_shortcut(n_samples=800, embedding_dim=25, shortcut_dims=2)

    test = GroupDiffTest(test=mannwhitneyu)
    test.fit(embeddings, labels)

    # Apply strict threshold
    significant = test.apply_threshold(alpha=0.01, verbose=False)

    # Should find significant features
    assert "0_vs_1" in significant
    if significant["0_vs_1"] is not None:
        # At least one of the shortcut dimensions should be significant
        assert len(significant["0_vs_1"]) > 0


def test_group_diff_no_difference():
    """Test on data with no group differences."""
    from tests.fixtures.synthetic_data import generate_no_shortcut

    embeddings, labels = generate_no_shortcut(n_samples=500, embedding_dim=20)

    test = GroupDiffTest(test=mannwhitneyu)
    test.fit(embeddings, labels)

    significant = test.apply_threshold(alpha=0.05, verbose=False)

    # Might have some false positives due to chance, but shouldn't be many
    if significant["0_vs_1"] is not None:
        # Expect < 5% false positives (1 feature out of 20)
        assert len(significant["0_vs_1"]) <= 2


def test_custom_test_function():
    """Test with custom statistical test."""

    def custom_test(x, y):
        """Simple difference in means test."""
        diff = abs(np.mean(x) - np.mean(y))
        # Return fake p-value based on difference
        p_value = max(0.001, 1.0 - diff)
        return (diff, p_value)

    embeddings, labels = generate_linear_shortcut(n_samples=500, embedding_dim=15)

    test = GroupDiffTest(test=custom_test)
    test.fit(embeddings, labels)
    p_values = test.get_pvalues()

    assert "0_vs_1" in p_values
    assert len(p_values["0_vs_1"]) == 15


# ============================================================================
# Multiple Testing Correction Tests
# ============================================================================


def test_multiple_testing_correction_fdr():
    """Test FDR correction reduces false positives."""
    # Generate data with known shortcuts in first 5 dimensions
    embeddings, labels = generate_linear_shortcut(
        n_samples=1000, embedding_dim=100, shortcut_dims=5
    )

    test = GroupDiffTest(test=mannwhitneyu)
    test.fit(embeddings, labels)

    # Raw threshold (many false positives expected)
    raw = test.apply_threshold(alpha=0.05, verbose=False)

    # FDR corrected
    corrected = test.apply_correction(method="fdr_bh", alpha=0.05, verbose=False)

    # Should have fewer significant features after correction
    n_raw = len(raw["0_vs_1"]) if raw["0_vs_1"] else 0
    n_corrected = (
        len(corrected["significant_features"]["0_vs_1"])
        if corrected["significant_features"]["0_vs_1"]
        else 0
    )

    assert n_corrected <= n_raw, "FDR should reduce or maintain number of significant features"
    # Should still detect some true shortcuts
    assert n_corrected >= 1, "Should still detect at least 1 true shortcut"


def test_bonferroni_vs_fdr():
    """Test that Bonferroni is more conservative than FDR."""
    embeddings, labels = generate_linear_shortcut(n_samples=800, embedding_dim=50, shortcut_dims=3)

    test = GroupDiffTest(test=mannwhitneyu)
    test.fit(embeddings, labels)

    fdr_results = test.apply_correction(method="fdr_bh", verbose=False)
    bonf_results = test.apply_correction(method="bonferroni", verbose=False)

    n_fdr = (
        len(fdr_results["significant_features"]["0_vs_1"])
        if fdr_results["significant_features"]["0_vs_1"]
        else 0
    )
    n_bonf = (
        len(bonf_results["significant_features"]["0_vs_1"])
        if bonf_results["significant_features"]["0_vs_1"]
        else 0
    )

    assert n_bonf <= n_fdr, "Bonferroni should be more conservative (or equal) than FDR"


def test_correction_methods_available():
    """Test all correction methods work."""
    embeddings, labels = generate_linear_shortcut(n_samples=500, embedding_dim=30)

    test = GroupDiffTest(test=mannwhitneyu)
    test.fit(embeddings, labels)

    for method in ["bonferroni", "fdr_bh", "fdr_by", "holm"]:
        results = test.apply_correction(method=method, verbose=False)
        assert "significant_features" in results
        assert "corrected_pvalues" in results
        assert "rejected" in results
        assert results["method"] == method
        assert results["alpha"] == 0.05


def test_correction_invalid_method():
    """Test that invalid correction method raises error."""
    embeddings, labels = generate_linear_shortcut(n_samples=200, embedding_dim=10)

    test = GroupDiffTest(test=mannwhitneyu)
    test.fit(embeddings, labels)

    with pytest.raises(ValueError, match="method must be one of"):
        test.apply_correction(method="invalid_method", verbose=False)


def test_correction_before_fit():
    """Test that apply_correction raises error if called before fit."""
    test = GroupDiffTest(test=mannwhitneyu)

    with pytest.raises(RuntimeError, match="Run `.fit"):
        test.apply_correction(method="fdr_bh", verbose=False)


def test_correction_results_structure():
    """Test that correction results have expected structure."""
    embeddings, labels = generate_linear_shortcut(n_samples=400, embedding_dim=20)

    test = GroupDiffTest(test=mannwhitneyu)
    test.fit(embeddings, labels)
    results = test.apply_correction(method="fdr_bh", alpha=0.01, verbose=False)

    # Check structure
    assert "significant_features" in results
    assert "corrected_pvalues" in results
    assert "rejected" in results
    assert "method" in results
    assert "alpha" in results

    # Check values
    assert results["method"] == "fdr_bh"
    assert results["alpha"] == 0.01

    # Check arrays
    for key in results["corrected_pvalues"]:
        pvals = results["corrected_pvalues"][key]
        rejected = results["rejected"][key]
        assert len(pvals) == 20  # embedding_dim
        assert len(rejected) == 20
        assert pvals.dtype == np.float64
        assert rejected.dtype == bool
