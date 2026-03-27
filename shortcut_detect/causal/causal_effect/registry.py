"""Register causal_effect detector with DetectorFactory."""

from ...unified import DetectorFactory
from .builder import CausalEffectDetectorBuilder

DetectorFactory.register("causal_effect", CausalEffectDetectorBuilder)
