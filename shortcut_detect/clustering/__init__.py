"""Clustering-based detection methods."""

from .hbac_detector import EmbeddingShortcutDetector, HBACConfig

# Alias for better naming
HBACDetector = EmbeddingShortcutDetector

__all__ = ["HBACDetector", "EmbeddingShortcutDetector", "HBACConfig"]
