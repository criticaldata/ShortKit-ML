"""Register frequency detection builders with DetectorFactory."""

from ..unified import DetectorFactory
from .builder import FrequencyShortcutBuilder

DetectorFactory.register("frequency", FrequencyShortcutBuilder)
