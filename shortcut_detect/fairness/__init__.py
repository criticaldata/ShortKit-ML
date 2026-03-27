"""Fairness-related detectors."""

from .demographic_parity.src.detector import DemographicParityDetector, DemographicParityReport
from .equalized_odds.src.detector import EqualizedOddsDetector, EqualizedOddsReport
from .intersectional.src.detector import IntersectionalDetector, IntersectionalReport

__all__ = [
    "DemographicParityDetector",
    "DemographicParityReport",
    "EqualizedOddsDetector",
    "EqualizedOddsReport",
    "IntersectionalDetector",
    "IntersectionalReport",
]
