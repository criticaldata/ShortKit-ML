"""Attribute-aware shortcut-potential audit (conditional permutation calibration)."""

from .conditional import (
    DETECTORS,
    PRIMARY,
    AuditConfig,
    AuditResult,
    DetectorResult,
    benjamini_hochberg,
    conditional_audit,
    conditional_permutations,
)

__all__ = [
    "DETECTORS",
    "PRIMARY",
    "AuditConfig",
    "AuditResult",
    "DetectorResult",
    "benjamini_hochberg",
    "conditional_audit",
    "conditional_permutations",
]
