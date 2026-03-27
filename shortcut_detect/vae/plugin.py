"""Register VAE detection builders with DetectorFactory."""

from ..unified import DetectorFactory
from .builder import VAEDetectorBuilder

DetectorFactory.register("vae", VAEDetectorBuilder)
