"""TRM - Tiny Recursive Model for deep reasoning."""

from elara_core.trm.core import TRMCore
from elara_core.trm.block import TRMBlock
from elara_core.trm.halting import AdaptiveHalting
from elara_core.trm.state import LatentStateManager

__all__ = ["TRMCore", "TRMBlock", "AdaptiveHalting", "LatentStateManager"]
