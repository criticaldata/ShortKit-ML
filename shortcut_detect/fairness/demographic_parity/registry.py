"""Register demographic_parity detector with DetectorFactory."""

from ...unified import DetectorFactory
from .builder import DemographicParityDetectorBuilder

DetectorFactory.register("demographic_parity", DemographicParityDetectorBuilder)
