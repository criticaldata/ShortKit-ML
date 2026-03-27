"""
VAE-based shortcut detection (Müller et al., Fraunhofer-AISEC).

Uses Beta-VAE disentanglement to identify latent dimensions with high predictiveness
for the target label. High-predictiveness dimensions indicate candidate shortcuts.

Reference: Müller et al., "Shortcut Detection with Variational Autoencoders",
ICML 2023 Workshop on Spurious Correlations, Invariance and Stability.
https://github.com/Fraunhofer-AISEC/shortcut-detection-vae
"""

from .vae_detector import VAEDetector

__all__ = ["VAEDetector"]
