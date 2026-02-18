"""
TiDARGenerator - Full TiDAR model: Think in Diffusion, Talk in Autoregression.
Hybrid AR + diffusion generation with parallel pre-drafting.
Based on TiDAR.md specification.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, Optional, Tuple

from elara_core.tidar.attention import HybridAttention
from elara_core.tidar.sampler import DraftVerifySampler
from elara_core.clara.compressor import RMSNorm


class TiDARConfig:
    """Configuration for TiDAR Generator."""
    def __init__(
        self,
        vocab_size: int = 32000,
        d_model: int = 512,
        n_layers: int = 6,
        n_ar_heads: int = 4,
        n_diff_heads: int = 4,
        d_head: int = 64,
        d_ff: int = 1024,
        block_size: int = 8,
        beta: float = 0.5,
        temperature: float = 0.8,
        top_p: float = 0.9,
        dropout: float = 0.0,
        max_seq_len: int = 4096,
    ):
        self.vocab_size = vocab_size
        self.d_model = d_model
        self.n_layers = n_layers
        self.n_ar_heads = n_ar_heads
        self.n_diff_heads = n_diff_heads
        self.d_head = d_head
        self.d_ff = d_ff
        self.block_size = block_size
        self.beta = beta
        self.temperature = temperature
        self.top_p = top_p
        self.dropout = dropout
        self.max_seq_len = max_seq_len


class TiDARBlock(nn.Module):
    """Single TiDAR transformer block with hybrid attention."""

    def __init__(self, config: TiDARConfig):
        super().__init__()
        self.norm1 = RMSNorm(config.d_model)
        self.attn = HybridAttention(
            d_model=config.d_model,
            n_ar_heads=config.n_ar_heads,
            n_diff_heads=config.n_diff_heads,
            d_head=config.d_head,
            dropout=config.dropout,
            max_positions=config.max_seq_len,
        )
        self.norm2 = RMSNorm(config.d_model)

        # SwiGLU FFN
        self.gate_proj = nn.Linear(config.d_model, config.d_ff, bias=False)
        self.up_proj = nn.Linear(config.d_model, config.d_ff, bias=False)
        self.down_proj = nn.Linear(config.d_ff, config.d_model, bias=False)
        self.dropout = nn.Dropout(config.dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Hybrid attention
        residual = x
        x = self.norm1(x)
        x = self.attn(x)
        x = self.dropout(x)
        x = residual + x

        # SwiGLU FFN
        residual = x
        x = self.norm2(x)
        gate = F.silu(self.gate_proj(x))
        x = self.down_proj(gate * self.up_proj(x))
        x = self.dropout(x)
        x = residual + x

        return x


class TiDARGenerator(nn.Module):
    """
    Full TiDAR model.

    Architecture:
    - Hybrid attention layers (AR + diffusion heads)
    - Block-level parallel pre-drafting
    - Speculative decoding-style verification
    - Beta-blended output mixing
    """

    def __init__(self, config: TiDARConfig):
        super().__init__()
        self.config = config

        # Embedding
        self.embed_tokens = nn.Embedding(config.vocab_size, config.d_model)

        # Transformer blocks
        self.layers = nn.ModuleList([
            TiDARBlock(config) for _ in range(config.n_layers)
        ])

        self.norm = RMSNorm(config.d_model)

        # Separate LM heads for AR and diffusion
        self.ar_head = nn.Linear(config.d_model, config.vocab_size, bias=False)
        self.diff_head = nn.Linear(config.d_model, config.vocab_size, bias=False)

        # Sampler
        self.sampler = DraftVerifySampler(
            block_size=config.block_size,
            beta=config.beta,
            temperature=config.temperature,
            top_p=config.top_p,
        )

        # Tie AR head weights with embeddings
        self.ar_head.weight = self.embed_tokens.weight

    def forward(
        self,
        input_ids: Optional[torch.Tensor] = None,
        hidden_states: Optional[torch.Tensor] = None,
        labels: Optional[torch.Tensor] = None,
    ) -> Dict[str, torch.Tensor]:
        """
        Forward pass.

        Args:
            input_ids: [B, L] token ids.
            hidden_states: [B, L, D] pre-computed states.
            labels: [B, L] target tokens for loss.

        Returns:
            Dict with logits, loss, etc.
        """
        if hidden_states is None:
            if input_ids is None:
                raise ValueError("Either input_ids or hidden_states required")
            hidden_states = self.embed_tokens(input_ids)

        # Transform
        for layer in self.layers:
            hidden_states = layer(hidden_states)
        hidden_states = self.norm(hidden_states)

        # Dual heads
        ar_logits = self.ar_head(hidden_states)
        diff_logits = self.diff_head(hidden_states)

        # Blend
        blended = self.sampler.blend_outputs(ar_logits, diff_logits)

        result = {
            "logits": blended,
            "ar_logits": ar_logits,
            "diff_logits": diff_logits,
            "hidden_states": hidden_states,
        }

        if labels is not None:
            # AR loss (causal, shifted)
            shift_logits = ar_logits[:, :-1, :].contiguous()
            shift_labels = labels[:, 1:].contiguous()
            ar_loss = F.cross_entropy(
                shift_logits.view(-1, self.config.vocab_size),
                shift_labels.view(-1),
                ignore_index=-100,
            )

            # Diffusion loss (all positions, token prediction)
            diff_loss = F.cross_entropy(
                diff_logits[:, :-1, :].contiguous().view(-1, self.config.vocab_size),
                shift_labels.view(-1),
                ignore_index=-100,
            )

            total_loss = self.config.beta * ar_loss + (1 - self.config.beta) * diff_loss

            result["loss"] = total_loss
            result["ar_loss"] = ar_loss.item()
            result["diff_loss"] = diff_loss.item()

        return result

    @torch.no_grad()
    def generate(
        self,
        input_ids: torch.Tensor,
        max_new_tokens: int = 100,
    ) -> torch.Tensor:
        """
        Generate with draft-verify: draft block_size tokens, verify, repeat.

        Args:
            input_ids: [B, L] prompt tokens.
            max_new_tokens: Max tokens to generate.

        Returns:
            [B, L + new] completed sequence.
        """
        generated = input_ids.clone()
        tokens_generated = 0

        while tokens_generated < max_new_tokens:
            # Forward pass
            hidden = self.embed_tokens(generated)
            for layer in self.layers:
                hidden = layer(hidden)
            hidden = self.norm(hidden)

            # Get logits for next block_size positions
            remaining = min(self.config.block_size, max_new_tokens - tokens_generated)

            # Diffusion draft
            last_hidden = hidden[:, -1:, :]
            draft_hidden = last_hidden.expand(-1, remaining, -1)
            diff_logits = self.diff_head(draft_hidden)
            draft_tokens = self.sampler.draft_tokens(diff_logits)

            # AR verify
            # Append draft tokens and verify
            candidate = torch.cat([generated, draft_tokens], dim=-1)
            verify_hidden = self.embed_tokens(candidate)
            for layer in self.layers:
                verify_hidden = layer(verify_hidden)
            verify_hidden = self.norm(verify_hidden)

            # Get AR logits for the drafted positions
            start_pos = generated.shape[1]
            ar_logits = self.ar_head(verify_hidden[:, start_pos:start_pos + remaining, :])

            # Verify
            verified, accepted = self.sampler.verify_tokens(draft_tokens[:, :remaining], ar_logits)

            # Append verified tokens
            generated = torch.cat([generated, verified[:, :accepted]], dim=-1)
            tokens_generated += accepted

            # Check EOS
            if (verified[:, accepted - 1] == 2).all():
                break

        return generated
