"""Compatibility module for frequency detector imports.

This keeps `shortcut_detect.frequency.frequency_detector` import paths working
while the canonical implementation lives in `shortcut_detect.frequency.detector`.
"""

from .detector import FrequencyConfig, FrequencyDetector

__all__ = ["FrequencyConfig", "FrequencyDetector"]
