"""Register XAI detection builders with DetectorFactory."""

from .cav.registry import *  # noqa: F403  # Registers cav
from .gradcam_mask_overlap.registry import *  # noqa: F403  # Registers gradcam_mask_overlap
from .sis.registry import *  # noqa: F403  # Registers sis
