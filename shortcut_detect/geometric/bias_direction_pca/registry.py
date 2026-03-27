"""Register bias_direction_pca detector with DetectorFactory."""

from ...unified import DetectorFactory
from .builder import BiasDirectionPCADetectorBuilder

DetectorFactory.register("bias_direction_pca", BiasDirectionPCADetectorBuilder)
