"""Register generative_cvae detector with DetectorFactory."""

from ...unified import DetectorFactory
from .builder import GenerativeCVEDetectorBuilder

DetectorFactory.register("generative_cvae", GenerativeCVEDetectorBuilder)
