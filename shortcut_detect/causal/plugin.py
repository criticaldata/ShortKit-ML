"""Register causal detection builders with DetectorFactory."""

from .causal_effect.registry import *  # noqa: F403  # Registers causal_effect
from .generative_cvae.registry import *  # noqa: F403  # Registers generative_cvae
