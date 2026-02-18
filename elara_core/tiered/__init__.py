"""Tiered Inference - Orchestration engine for multi-tier inference."""


def __getattr__(name):
    """Lazy module-level imports to avoid torch dependency at import time."""
    if name == "TieredInferenceEngine":
        from elara_core.tiered.engine import TieredInferenceEngine
        return TieredInferenceEngine
    elif name == "TierRouter":
        from elara_core.tiered.router import TierRouter
        return TierRouter
    elif name == "InputMultiplexer":
        from elara_core.tiered.multiplexer import InputMultiplexer
        return InputMultiplexer
    elif name == "ModalityType":
        from elara_core.tiered.multiplexer import ModalityType
        return ModalityType
    elif name == "MetricsTracker":
        from elara_core.tiered.metrics import MetricsTracker
        return MetricsTracker
    raise AttributeError(f"module 'elara_core.tiered' has no attribute {name}")
