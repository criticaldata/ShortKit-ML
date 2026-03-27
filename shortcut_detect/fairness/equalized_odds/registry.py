"""Register equalized_odds detector with DetectorFactory."""

from ...unified import DetectorFactory
from .builder import EqualizedOddsDetectorBuilder

DetectorFactory.register("equalized_odds", EqualizedOddsDetectorBuilder)
