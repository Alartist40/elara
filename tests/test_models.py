"""
Tests for TRM and TiDAR components.
"""

import pytest
import torch
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))


class TestTRMBlock:
    """Test TRM transformer block."""

    def test_attention_forward(self):
        from elara_core.trm.block import TRMBlock
        block = TRMBlock(d_model=64, n_heads=4, d_head=16, d_ff=128)
        x = torch.randn(2, 10, 64)
        out = block(x)
        assert out.shape == (2, 10, 64)

    def test_mixer_forward(self):
        from elara_core.trm.block import TRMBlock
        block = TRMBlock(d_model=64, n_heads=4, d_head=16, d_ff=128, use_mixer=True)
        x = torch.randn(2, 10, 64)
        out = block(x)
        assert out.shape == (2, 10, 64)


class TestAdaptiveHalting:
    """Test adaptive halting mechanism."""

    def test_halting_decision(self):
        from elara_core.trm.halting import AdaptiveHalting
        halting = AdaptiveHalting(d_model=64, threshold=0.85)
        hidden = torch.randn(2, 10, 64)
        accum = torch.zeros(2)

        halt_prob, new_accum, should_halt = halting(hidden, accum, step=0, max_steps=6)
        assert halt_prob.shape == (2,)
        assert new_accum.shape == (2,)

    def test_forced_halt_at_max(self):
        from elara_core.trm.halting import AdaptiveHalting
        halting = AdaptiveHalting(d_model=64, threshold=0.85)
        hidden = torch.randn(2, 10, 64)
        accum = torch.tensor([0.5, 0.3])

        _, _, should_halt = halting(hidden, accum, step=5, max_steps=6)
        assert should_halt is True


class TestTRMCore:
    """Test full TRM core."""

    def setup_method(self):
        from elara_core.trm.core import TRMCore, TRMConfig
        self.config = TRMConfig(
            vocab_size=1000, d_model=64, n_layers=2,
            n_heads=4, d_head=16, d_ff=128,
            max_recursions=3, confidence_threshold=0.85,
        )
        self.model = TRMCore(self.config)

    def test_forward(self):
        input_ids = torch.randint(0, 1000, (2, 10))
        result = self.model(input_ids=input_ids)
        assert "logits" in result
        assert result["logits"].shape == (2, 10, 1000)
        assert result["n_recursions"] > 0

    def test_with_labels(self):
        input_ids = torch.randint(0, 1000, (2, 10))
        labels = torch.randint(0, 1000, (2, 10))
        result = self.model(input_ids=input_ids, labels=labels)
        assert "loss" in result
        assert result["loss"].requires_grad


class TestTiDARGenerator:
    """Test TiDAR hybrid generator."""

    def setup_method(self):
        from elara_core.tidar.generator import TiDARGenerator, TiDARConfig
        self.config = TiDARConfig(
            vocab_size=1000, d_model=64, n_layers=2,
            n_ar_heads=2, n_diff_heads=2, d_head=16,
            d_ff=128, block_size=4,
        )
        self.model = TiDARGenerator(self.config)

    def test_forward(self):
        input_ids = torch.randint(0, 1000, (2, 10))
        result = self.model(input_ids=input_ids)
        assert "logits" in result
        assert "ar_logits" in result
        assert "diff_logits" in result
        assert result["logits"].shape == (2, 10, 1000)

    def test_with_labels(self):
        input_ids = torch.randint(0, 1000, (2, 10))
        labels = torch.randint(0, 1000, (2, 10))
        result = self.model(input_ids=input_ids, labels=labels)
        assert "loss" in result
        assert "ar_loss" in result
        assert "diff_loss" in result


class TestHybridAttention:
    """Test hybrid attention module."""

    def test_forward(self):
        from elara_core.tidar.attention import HybridAttention
        attn = HybridAttention(d_model=64, n_ar_heads=2, n_diff_heads=2, d_head=16)
        x = torch.randn(2, 10, 64)
        out = attn(x)
        assert out.shape == (2, 10, 64)
