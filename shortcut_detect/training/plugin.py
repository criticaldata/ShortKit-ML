"""Register training detection builders with DetectorFactory."""

from ..unified import DetectorFactory
from .builder import EarlyEpochClusteringDetectorBuilder

DetectorFactory.register("early_epoch_clustering", EarlyEpochClusteringDetectorBuilder)
