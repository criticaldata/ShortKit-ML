"""Statistical testing methods for shortcut detection."""

from .group_diff_test import FeatureGroupDiffTest

# Alias for better naming
GroupDiffTest = FeatureGroupDiffTest

# Available multiple testing correction methods
CORRECTION_METHODS = ["bonferroni", "holm", "fdr_bh", "fdr_by"]

__all__ = [
    "GroupDiffTest",
    "FeatureGroupDiffTest",
    "CORRECTION_METHODS",
]
