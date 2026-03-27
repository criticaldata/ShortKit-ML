"""Register cav detector with DetectorFactory."""

from ...unified import DetectorFactory
from .builder import CAVDetectorBuilder

DetectorFactory.register("cav", CAVDetectorBuilder)
