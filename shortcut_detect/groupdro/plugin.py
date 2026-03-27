"""Register GroupDRO detection builders with DetectorFactory."""

from ..unified import DetectorFactory
from .builder import GroupDRODetectorBuilder

DetectorFactory.register("groupdro", GroupDRODetectorBuilder)
