"""
HybridAttention - Combined causal (AR) + bidirectional (diffusion) attention.
Core TiDAR innovation: simultaneous autoregressive + diffusion processing.
Based on TiDAR.md specification.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import math
from typing import Optional, Tuple


class RotaryPositionEmbedding(nn.Module):
    """Rotary Position Embedding (RoPE)."""

    def __init__(self, d_head: int, max_positions: int = 4096, base: float = 10000.0):
        super().__init__()
        inv_freq = 1.0 / (base ** (torch.arange(0, d_head, 2).float() / d_head))
        self.register_buffer("inv_freq", inv_freq)
        self.max_positions = max_positions

    def forward(self, seq_len: int, device: torch.device) -> Tuple[torch.Tensor, torch.Tensor]:
        t = torch.arange(seq_len, device=device, dtype=self.inv_freq.dtype)
        freqs = torch.einsum("i,j->ij", t, self.inv_freq)
        emb = torch.cat([freqs, freqs], dim=-1)
        return emb.cos(), emb.sin()


def rotate_half(x: torch.Tensor) -> torch.Tensor:
    """Rotate half of the hidden dims."""
    x1 = x[..., : x.shape[-1] // 2]
    x2 = x[..., x.shape[-1] // 2 :]
    return torch.cat((-x2, x1), dim=-1)


def apply_rotary_pos_emb(
    q: torch.Tensor, k: torch.Tensor,
    cos: torch.Tensor, sin: torch.Tensor,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Apply RoPE to query and key tensors."""
    q_embed = (q * cos) + (rotate_half(q) * sin)
    k_embed = (k * cos) + (rotate_half(k) * sin)
    return q_embed, k_embed


class HybridAttention(nn.Module):
    """
    TiDAR Hybrid Attention: AR + Diffusion attention in a single forward pass.

    Split strategy:
    - First half of heads: causal mask (autoregressive)
    - Second half of heads: bidirectional mask (diffusion drafting)

    This allows parallel token drafting (diffusion) while maintaining
    coherent autoregressive generation.
    """

    def __init__(
        self,
        d_model: int = 512,
        n_ar_heads: int = 4,
        n_diff_heads: int = 4,
        d_head: int = 64,
        dropout: float = 0.0,
        max_positions: int = 4096,
    ):
        super().__init__()
        self.d_model = d_model
        self.n_ar_heads = n_ar_heads
        self.n_diff_heads = n_diff_heads
        self.n_total_heads = n_ar_heads + n_diff_heads
        self.d_head = d_head
        self.scale = d_head ** -0.5

        total_dim = self.n_total_heads * d_head

        self.q_proj = nn.Linear(d_model, total_dim, bias=False)
        self.k_proj = nn.Linear(d_model, total_dim, bias=False)
        self.v_proj = nn.Linear(d_model, total_dim, bias=False)
        self.o_proj = nn.Linear(total_dim, d_model, bias=False)

        self.dropout = nn.Dropout(dropout)
        self.rope = RotaryPositionEmbedding(d_head, max_positions)

    def _build_hybrid_mask(
        self, seq_len: int, device: torch.device
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Build separate masks for AR and diffusion heads.

        Returns:
            ar_mask: [1, 1, L, L] causal mask
            diff_mask: [1, 1, L, L] bidirectional (zeros = no masking)
        """
        # Causal mask for AR heads
        ar_mask = torch.triu(
            torch.ones(seq_len, seq_len, device=device) * float('-inf'),
            diagonal=1,
        ).unsqueeze(0).unsqueeze(0)

        # Bidirectional mask for diffusion heads (no masking)
        diff_mask = torch.zeros(1, 1, seq_len, seq_len, device=device)

        return ar_mask, diff_mask

    def forward(
        self,
        hidden_states: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """
        Hybrid attention forward pass.

        Args:
            hidden_states: [B, L, D]
            attention_mask: Optional additional mask

        Returns:
            [B, L, D] attended states
        """
        B, L, _ = hidden_states.shape

        # Project Q, K, V
        q = self.q_proj(hidden_states).view(B, L, self.n_total_heads, self.d_head).transpose(1, 2)
        k = self.k_proj(hidden_states).view(B, L, self.n_total_heads, self.d_head).transpose(1, 2)
        v = self.v_proj(hidden_states).view(B, L, self.n_total_heads, self.d_head).transpose(1, 2)

        # Apply RoPE
        cos, sin = self.rope(L, hidden_states.device)
        cos = cos.unsqueeze(0).unsqueeze(0)  # [1, 1, L, D]
        sin = sin.unsqueeze(0).unsqueeze(0)
        q, k = apply_rotary_pos_emb(q, k, cos, sin)

        # Build hybrid masks
        ar_mask, diff_mask = self._build_hybrid_mask(L, hidden_states.device)

        # Split heads into AR and diffusion groups
        q_ar, q_diff = q[:, :self.n_ar_heads], q[:, self.n_ar_heads:]
        k_ar, k_diff = k[:, :self.n_ar_heads], k[:, self.n_ar_heads:]
        v_ar, v_diff = v[:, :self.n_ar_heads], v[:, self.n_ar_heads:]

        # AR attention (causal)
        attn_ar = torch.matmul(q_ar, k_ar.transpose(-2, -1)) * self.scale + ar_mask
        attn_ar = F.softmax(attn_ar, dim=-1, dtype=torch.float32).to(q.dtype)
        attn_ar = self.dropout(attn_ar)
        out_ar = torch.matmul(attn_ar, v_ar)

        # Diffusion attention (bidirectional)
        attn_diff = torch.matmul(q_diff, k_diff.transpose(-2, -1)) * self.scale + diff_mask
        attn_diff = F.softmax(attn_diff, dim=-1, dtype=torch.float32).to(q.dtype)
        attn_diff = self.dropout(attn_diff)
        out_diff = torch.matmul(attn_diff, v_diff)

        # Concatenate outputs
        out = torch.cat([out_ar, out_diff], dim=1)  # [B, total_heads, L, D]
        out = out.transpose(1, 2).contiguous().view(B, L, -1)

        return self.o_proj(out)
