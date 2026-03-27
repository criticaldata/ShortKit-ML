"""Register probe detection builders with DetectorFactory."""

from ..unified import DetectorFactory
from .builder import ProbeDetectorBuilder

DetectorFactory.register("probe", ProbeDetectorBuilder)
