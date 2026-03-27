"""Register clustering detection builders with DetectorFactory."""

from ..unified import DetectorFactory
from .builder import HBACDetectorBuilder

DetectorFactory.register("hbac", HBACDetectorBuilder)
