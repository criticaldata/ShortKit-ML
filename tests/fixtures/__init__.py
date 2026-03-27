"""Test fixtures and synthetic data generators."""

from .synthetic_data import (
    generate_linear_shortcut,
    generate_linear_shortcut_with_group_labels,
    generate_multiclass_shortcut,
    generate_no_shortcut,
    generate_nonlinear_shortcut,
    generate_parametric_shortcut_dataset,
)

__all__ = [
    "generate_linear_shortcut",
    "generate_nonlinear_shortcut",
    "generate_multiclass_shortcut",
    "generate_no_shortcut",
    "generate_linear_shortcut_with_group_labels",
    "generate_parametric_shortcut_dataset",
]
