"""
TRMBlock - Single layer of the Tiny Recursive Model.
2-layer core based on TRM.md specification.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional


class TRMBlock(nn.Module):
    """
    Single Tiny Recursive Model block.
    Operates on latent state with attention and SwiGLU FFN.
    Designed to be weight-shared across recursions for memory efficiency.
    """

    def __init__(
        self,
        d_model: int = 512,
        n_heads: int = 8,
        d_head: int = 64,
        d_ff: int = 1024,
        dropout: float = 0.0,
        use_mixer: bool = False,
    ):
        super().__init__()
        self.d_model = d_model
        self.use_mixer = use_mixer

        # Layer norm
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)

        if use_mixer:
            # MLP-Mixer style: token mixing + channel mixing
            self.token_mixer = nn.Sequential(
                nn.Linear(d_model, d_model),
                nn.GELU(),
                nn.Dropout(dropout),
                nn.Linear(d_model, d_model),
                nn.Dropout(dropout),
            )
        else:
            # Standard multi-head self-attention
            self.n_heads = n_heads
            self.d_head = d_head
            self.scale = d_head ** -0.5

            self.q_proj = nn.Linear(d_model, n_heads * d_head, bias=False)
            self.k_proj = nn.Linear(d_model, n_heads * d_head, bias=False)
            self.v_proj = nn.Linear(d_model, n_heads * d_head, bias=False)
            self.o_proj = nn.Linear(n_heads * d_head, d_model, bias=False)
            self.attn_dropout = nn.Dropout(dropout)

        # SwiGLU FFN
        self.gate_proj = nn.Linear(d_model, d_ff, bias=False)
        self.up_proj = nn.Linear(d_model, d_ff, bias=False)
        self.down_proj = nn.Linear(d_ff, d_model, bias=False)
        self.dropout = nn.Dropout(dropout)

    def _attention(self, x: torch.Tensor) -> torch.Tensor:
        """Standard multi-head self-attention."""
        B, L, _ = x.shape
        q = self.q_proj(x).view(B, L, self.n_heads, self.d_head).transpose(1, 2)
        k = self.k_proj(x).view(B, L, self.n_heads, self.d_head).transpose(1, 2)
        v = self.v_proj(x).view(B, L, self.n_heads, self.d_head).transpose(1, 2)

        attn = torch.matmul(q, k.transpose(-2, -1)) * self.scale
        attn = F.softmax(attn, dim=-1, dtype=torch.float32).to(q.dtype)
        attn = self.attn_dropout(attn)

        out = torch.matmul(attn, v)
        out = out.transpose(1, 2).contiguous().view(B, L, -1)
        return self.o_proj(out)

    def forward(
        self,
        hidden_states: torch.Tensor,
        recursion_step: int = 0,
    ) -> torch.Tensor:
        """
        Forward pass through TRM block.

        Args:
            hidden_states: [B, L, D] latent representations
            recursion_step: Current recursion index (for potential step embeddings)

        Returns:
            Refined hidden states [B, L, D]
        """
        # Attention / Mixer
        residual = hidden_states
        hidden_states = self.norm1(hidden_states)
        if self.use_mixer:
            hidden_states = self.token_mixer(hidden_states)
        else:
            hidden_states = self._attention(hidden_states)
        hidden_states = self.dropout(hidden_states)
        hidden_states = residual + hidden_states

        # SwiGLU FFN
        residual = hidden_states
        hidden_states = self.norm2(hidden_states)
        gate = F.silu(self.gate_proj(hidden_states))
        hidden_states = self.down_proj(gate * self.up_proj(hidden_states))
        hidden_states = self.dropout(hidden_states)
        hidden_states = residual + hidden_states

        return hidden_states
