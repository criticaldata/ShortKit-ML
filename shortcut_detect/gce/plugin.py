"""Register GCE detection builders with DetectorFactory."""

from ..unified import DetectorFactory
from .builder import GCEDetectorBuilder

DetectorFactory.register("gce", GCEDetectorBuilder)
