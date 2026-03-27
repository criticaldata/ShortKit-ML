"""Register statistical detection builders with DetectorFactory."""

from ..unified import DetectorFactory
from .builder import StatisticalDetectorBuilder

DetectorFactory.register("statistical", StatisticalDetectorBuilder)
