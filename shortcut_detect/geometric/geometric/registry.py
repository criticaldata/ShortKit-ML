"""Register geometric detector with DetectorFactory."""

from ...unified import DetectorFactory
from .builder import GeometricDetectorBuilder

DetectorFactory.register("geometric", GeometricDetectorBuilder)
