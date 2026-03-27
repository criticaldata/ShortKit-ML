"""Register SSA detection builders with DetectorFactory."""

from ..unified import DetectorFactory
from .builder import SSADetectorBuilder

DetectorFactory.register("ssa", SSADetectorBuilder)
