"""Register intersectional detector with DetectorFactory."""

from ...unified import DetectorFactory
from .builder import IntersectionalDetectorBuilder

DetectorFactory.register("intersectional", IntersectionalDetectorBuilder)
