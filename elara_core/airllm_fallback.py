"""
AirLLM Fallback - Tier 3 catastrophic miss handler.
Layer-wise inference using memory-mapped model shards.
Based on AirLLM.md specification.
"""

import torch
import torch.nn as nn
import numpy as np
import time
from pathlib import Path
from typing import Dict, Any, Optional, List, Tuple
from functools import lru_cache
import warnings


class LayerShardConfig:
    """Configuration for layer-wise sharding."""
    def __init__(
        self,
        shard_dir: str = "models/airllm_shards",
        compression: str = "fp16",   # "fp16", "int8", "int4"
        prefetch: bool = True,
        cache_size: int = 4,           # Number of layers to keep in LRU cache
        max_memory_mb: int = 3500,
        cooldown_seconds: float = 5.0,
        max_calls_per_minute: int = 10,
    ):
        self.shard_dir = Path(shard_dir)
        self.compression = compression
        self.prefetch = prefetch
        self.cache_size = cache_size
        self.max_memory_mb = max_memory_mb
        self.cooldown_seconds = cooldown_seconds
        self.max_calls_per_minute = max_calls_per_minute


class LayerShardManager:
    """
    Manages layer-wise model shards for memory-efficient inference.
    Loads individual transformer layers one at a time, applying them
    sequentially to the hidden state and discarding after use.
    """

    def __init__(self, config: LayerShardConfig):
        self.config = config
        self._cache: Dict[int, torch.Tensor] = {}
        self._cache_order: List[int] = []
        self._total_layers = 0
        self._ready = False

    def prepare(self) -> bool:
        """Check shard directory and determine layer count."""
        if not self.config.shard_dir.exists():
            warnings.warn(f"Shard directory not found: {self.config.shard_dir}")
            self._ready = False
            return False

        # Count shard files
        shard_files = sorted(self.config.shard_dir.glob("layer_*.pt"))
        if not shard_files:
            shard_files = sorted(self.config.shard_dir.glob("layer_*.safetensors"))

        self._total_layers = len(shard_files)
        self._ready = self._total_layers > 0
        return self._ready

    def load_layer(self, layer_idx: int) -> Optional[Dict[str, torch.Tensor]]:
        """
        Load a single transformer layer from disk.
        Uses LRU caching to keep frequently-used layers in memory.
        """
        # Check cache
        if layer_idx in self._cache:
            return self._cache[layer_idx]

        # Load from disk
        shard_path = self.config.shard_dir / f"layer_{layer_idx:04d}.pt"
        if not shard_path.exists():
            shard_path = self.config.shard_dir / f"layer_{layer_idx:04d}.safetensors"
            if not shard_path.exists():
                return None

        try:
            weights = torch.load(shard_path, map_location="cpu", weights_only=True)
        except Exception:
            try:
                from safetensors.torch import load_file
                weights = load_file(str(shard_path))
            except ImportError:
                return None

        # Apply compression
        if self.config.compression == "int8":
            weights = self._quantize_int8(weights)
        elif self.config.compression == "int4":
            weights = self._quantize_int4(weights)

        # LRU cache management
        if len(self._cache) >= self.config.cache_size:
            evict = self._cache_order.pop(0)
            del self._cache[evict]

        self._cache[layer_idx] = weights
        self._cache_order.append(layer_idx)

        return weights

    def _quantize_int8(self, weights: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
        """Quantize weights to int8."""
        quantized = {}
        for name, tensor in weights.items():
            if tensor.dtype in (torch.float32, torch.float16):
                scale = tensor.abs().max() / 127.0
                quantized[name] = (tensor / scale).to(torch.int8)
                quantized[f"{name}_scale"] = scale
            else:
                quantized[name] = tensor
        return quantized

    def _quantize_int4(self, weights: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
        """Quantize weights to int4 (packed into int8)."""
        try:
            import bitsandbytes as bnb
            quantized = {}
            for name, tensor in weights.items():
                if tensor.dtype in (torch.float32, torch.float16):
                    q, state = bnb.functional.quantize_4bit(tensor)
                    quantized[name] = q
                    quantized[f"{name}_qstate"] = state
                else:
                    quantized[name] = tensor
            return quantized
        except ImportError:
            warnings.warn("bitsandbytes not available; falling back to int8")
            return self._quantize_int8(weights)

    @property
    def is_ready(self) -> bool:
        return self._ready


class AirLLMFallback:
    """
    Tier 3 fallback handler for catastrophic misses.

    Uses layer-wise streaming inference:
    1. Load layer weights from disk
    2. Apply to hidden state
    3. Discard weights
    4. Repeat for all layers

    Includes rate limiting and cooldown to manage resource usage.
    """

    def __init__(self, config: Optional[LayerShardConfig] = None):
        self.config = config or LayerShardConfig()
        self.shard_manager = LayerShardManager(self.config)

        # Rate limiting
        self._call_timestamps: List[float] = []
        self._last_call_time = 0.0
        self._total_calls = 0

    def is_available(self) -> bool:
        """Check if AirLLM fallback is ready."""
        return self.shard_manager.prepare()

    def _check_rate_limit(self) -> bool:
        """Check if we're within rate limits."""
        now = time.time()

        # Cooldown check
        if now - self._last_call_time < self.config.cooldown_seconds:
            return False

        # Calls per minute check
        recent = [t for t in self._call_timestamps if now - t < 60]
        self._call_timestamps = recent
        if len(recent) >= self.config.max_calls_per_minute:
            return False

        return True

    def generate(
        self,
        input_ids: torch.Tensor,
        max_new_tokens: int = 50,
        temperature: float = 0.7,
    ) -> Tuple[torch.Tensor, Dict[str, Any]]:
        """
        Layer-wise streaming inference.

        Args:
            input_ids: [B, L] prompt tokens.
            max_new_tokens: Maximum tokens to generate.
            temperature: Sampling temperature.

        Returns:
            output_ids: [B, L + new] generated tokens
            metadata: Dict with timing and stats.
        """
        if not self._check_rate_limit():
            return input_ids, {
                "error": "Rate limited",
                "cooldown_remaining": max(
                    0, self.config.cooldown_seconds - (time.time() - self._last_call_time)
                ),
            }

        start_time = time.time()
        self._last_call_time = start_time
        self._call_timestamps.append(start_time)
        self._total_calls += 1

        if not self.shard_manager.is_ready:
            if not self.shard_manager.prepare():
                return input_ids, {"error": "Shards not available"}

        # Placeholder: actual layer-wise inference would iterate here
        # For now, return input with metadata about capabilities
        metadata = {
            "tier": 3,
            "engine": "airllm",
            "total_layers": self.shard_manager._total_layers,
            "compression": self.config.compression,
            "latency_ms": (time.time() - start_time) * 1000,
            "total_calls": self._total_calls,
            "note": "Layer-wise inference: connect model shards for full operation",
        }

        return input_ids, metadata

    def get_stats(self) -> Dict[str, Any]:
        return {
            "available": self.is_available(),
            "total_calls": self._total_calls,
            "compression": self.config.compression,
            "cache_size": self.config.cache_size,
        }
