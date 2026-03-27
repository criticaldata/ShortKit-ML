"""Register gradcam_mask_overlap detector with DetectorFactory."""

from ...unified import DetectorFactory
from .builder import GradCAMMaskOverlapDetectorBuilder

DetectorFactory.register("gradcam_mask_overlap", GradCAMMaskOverlapDetectorBuilder)
