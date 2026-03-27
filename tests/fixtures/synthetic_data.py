"""
Synthetic dataset generators for testing shortcut detection.

NOTE: These functions are now part of the main package (shortcut_detect.datasets).
This module re-exports them for backward compatibility with existing tests.
"""

from shortcut_detect.benchmark.synthetic import generate_parametric_shortcut_dataset
from shortcut_detect.datasets import (
    generate_linear_shortcut,
    generate_linear_shortcut_with_group_labels,
    generate_multiclass_shortcut,
    generate_no_shortcut,
    generate_nonlinear_shortcut,
)

__all__ = [
    "generate_linear_shortcut",
    "generate_nonlinear_shortcut",
    "generate_multiclass_shortcut",
    "generate_no_shortcut",
    "generate_linear_shortcut_with_group_labels",
    "generate_parametric_shortcut_dataset",
]
