"""AudioPrism 2.0: complex-domain, band-split music source separation."""

from .config import ExperimentConfig, load_config
from .model import AudioPrism

__all__ = ["AudioPrism", "ExperimentConfig", "load_config"]
__version__ = "2.0.0"
