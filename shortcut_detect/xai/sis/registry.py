"""Register sis detector with DetectorFactory."""

from ...unified import DetectorFactory
from .builder import SISDetectorBuilder

DetectorFactory.register("sis", SISDetectorBuilder)
