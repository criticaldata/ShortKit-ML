"""Register fairness detection builders with DetectorFactory."""

from .demographic_parity.registry import *  # noqa: F403  # Registers demographic_parity
from .equalized_odds.registry import *  # noqa: F403  # Registers equalized_odds
from .intersectional.registry import *  # noqa: F403  # Registers intersectional
