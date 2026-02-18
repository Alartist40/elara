"""
SCPCompressor - Salient Compressor Pretraining module.
Compresses documents into continuous memory tokens.
Based on CLaRa: "Bridging Retrieval and Generation with Continuous Latent Reasoning" (Apple/Edinburgh, 2025)
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional, Dict
import math


class RMSNorm(nn.Module):
    """Root Mean Square Layer Normalization."""
    def __init__(self, dim: int, eps: float = 1e-6):
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(dim))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x * torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + self.eps) * self.weight


class LoRALinear(nn.Module):
    """LoRA adapter for efficient fine-tuning."""
    def __init__(self, in_features: int, out_features: int, r: int = 16, alpha: int = 32):
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.r = r
        self.scaling = alpha / r

        # Base weights (frozen during fine-tuning)
        self.weight = nn.Parameter(
            torch.zeros(out_features, in_features), requires_grad=False
        )
        # LoRA matrices
        self.lora_A = nn.Parameter(torch.randn(in_features, r))
        self.lora_B = nn.Parameter(torch.zeros(r, out_features))

        self.reset_parameters()

    def reset_parameters(self):
        nn.init.kaiming_uniform_(self.lora_A, a=math.sqrt(5))
        nn.init.zeros_(self.lora_B)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        base = F.linear(x, self.weight)
        lora = F.linear(F.linear(x, self.lora_A.t()), self.lora_B.t()) * self.scaling
        return base + lora


class CLaRaAttention(nn.Module):
    """Multi-head attention with optional LoRA."""
    def __init__(self, d_model: int, n_heads: int, d_head: int, dropout: float = 0.0, use_lora: bool = False):
        super().__init__()
        self.n_heads = n_heads
        self.d_head = d_head
        self.d_model = d_model
        self.scale = d_head ** -0.5

        Linear = LoRALinear if use_lora else lambda i, o: nn.Linear(i, o, bias=False)
        self.q_proj = Linear(d_model, n_heads * d_head)
        self.k_proj = Linear(d_model, n_heads * d_head)
        self.v_proj = Linear(d_model, n_heads * d_head)
        self.o_proj = Linear(n_heads * d_head, d_model)
        self.dropout = nn.Dropout(dropout)

    def forward(self, hidden_states: torch.Tensor, attention_mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        B, L, _ = hidden_states.shape
        q = self.q_proj(hidden_states).view(B, L, self.n_heads, self.d_head).transpose(1, 2)
        k = self.k_proj(hidden_states).view(B, L, self.n_heads, self.d_head).transpose(1, 2)
        v = self.v_proj(hidden_states).view(B, L, self.n_heads, self.d_head).transpose(1, 2)

        attn_weights = torch.matmul(q, k.transpose(-2, -1)) * self.scale
        if attention_mask is not None:
            attn_weights = attn_weights + attention_mask
        attn_weights = F.softmax(attn_weights, dim=-1, dtype=torch.float32).to(q.dtype)
        attn_weights = self.dropout(attn_weights)

        attn_output = torch.matmul(attn_weights, v)
        attn_output = attn_output.transpose(1, 2).contiguous().view(B, L, self.d_model)
        return self.o_proj(attn_output)


class CLaRaBlock(nn.Module):
    """Transformer block with optional LoRA adapters and SwiGLU FFN."""
    def __init__(self, d_model: int, n_heads: int, d_head: int, d_ff: int, dropout: float = 0.0, use_lora: bool = False):
        super().__init__()
        self.norm1 = RMSNorm(d_model)
        self.attn = CLaRaAttention(d_model, n_heads, d_head, dropout, use_lora)
        self.norm2 = RMSNorm(d_model)

        Linear = LoRALinear if use_lora else lambda i, o: nn.Linear(i, o, bias=False)
        self.gate_proj = Linear(d_model, d_ff)
        self.up_proj = Linear(d_model, d_ff)
        self.down_proj = Linear(d_ff, d_model)
        self.dropout = nn.Dropout(dropout)

    def forward(self, hidden_states: torch.Tensor, attention_mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        # Attention
        residual = hidden_states
        hidden_states = self.norm1(hidden_states)
        hidden_states = self.attn(hidden_states, attention_mask)
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


class SCPCompressorConfig:
    """Configuration for the SCP Compressor."""
    def __init__(
        self,
        vocab_size: int = 32000,
        d_model: int = 512,
        n_layers: int = 4,
        n_heads: int = 8,
        d_head: int = 64,
        d_ff: int = 1024,
        n_memory_tokens: int = 16,
        compression_rate: int = 16,
        dropout: float = 0.0,
        use_mse_alignment: bool = True,
        lambda_mse: float = 0.1,
    ):
        self.vocab_size = vocab_size
        self.d_model = d_model
        self.n_layers = n_layers
        self.n_heads = n_heads
        self.d_head = d_head
        self.d_ff = d_ff
        self.n_memory_tokens = n_memory_tokens
        self.compression_rate = compression_rate
        self.dropout = dropout
        self.use_mse_alignment = use_mse_alignment
        self.lambda_mse = lambda_mse


class SCPCompressor(nn.Module):
    """
    Salient Compressor Pretraining (SCP) module.
    Compresses documents into continuous memory tokens.
    """

    def __init__(self, config: SCPCompressorConfig):
        super().__init__()
        self.config = config

        # Embedding layer
        self.embed_tokens = nn.Embedding(config.vocab_size, config.d_model)

        # Transformer layers with LoRA
        self.layers = nn.ModuleList([
            CLaRaBlock(
                config.d_model, config.n_heads, config.d_head,
                config.d_ff, config.dropout, use_lora=True
            )
            for _ in range(config.n_layers)
        ])
        self.norm = RMSNorm(config.d_model)

        # Learnable memory tokens (m_1, ..., m_l)
        self.memory_tokens = nn.Parameter(
            torch.randn(1, config.n_memory_tokens, config.d_model)
        )

        self._init_weights()

    def _init_weights(self):
        for module in self.modules():
            if isinstance(module, (nn.Linear, nn.Embedding)):
                torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)

    def forward(
        self,
        document_ids: torch.Tensor,
        task_instruction: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """
        Compress document into memory tokens.

        Args:
            document_ids: [B, doc_len] tokenized document
            task_instruction: Optional instruction tensor

        Returns:
            memory_embeddings: [B, n_memory_tokens, d_model]
        """
        B, doc_len = document_ids.shape

        # Embed document
        doc_embeds = self.embed_tokens(document_ids)

        # Expand memory tokens for batch
        memory_tokens = self.memory_tokens.expand(B, -1, -1)

        # Concatenate: [doc_tokens | memory_tokens]
        combined = torch.cat([doc_embeds, memory_tokens], dim=1)

        # Pass through transformer
        hidden_states = combined
        for layer in self.layers:
            hidden_states = layer(hidden_states)

        hidden_states = self.norm(hidden_states)

        # Extract memory token representations (last l positions)
        memory_embeddings = hidden_states[:, -self.config.n_memory_tokens:, :]

        return memory_embeddings

    def compute_scp_loss(
        self,
        document_ids: torch.Tensor,
        target_ids: torch.Tensor,
    ) -> Dict[str, torch.Tensor]:
        """
        SCP pretraining loss: compress then generate target.
        L_total = L_CE + lambda * L_MSE
        """
        B = document_ids.shape[0]

        # Compress document
        memory_embeds = self.forward(document_ids)

        # Prepare for generation (teacher forcing)
        target_embeds = self.embed_tokens(target_ids)
        full_seq = torch.cat([memory_embeds, target_embeds], dim=1)

        # Causal mask
        seq_len = full_seq.size(1)
        causal_mask = torch.triu(
            torch.ones(seq_len, seq_len, device=full_seq.device) * float('-inf'),
            diagonal=1
        ).unsqueeze(0).unsqueeze(0)

        # Forward through generator
        hidden_states = full_seq
        for layer in self.layers:
            hidden_states = layer(hidden_states, causal_mask)
        hidden_states = self.norm(hidden_states)

        # Loss on target positions only
        logits = torch.matmul(hidden_states, self.embed_tokens.weight.t())
        target_logits = logits[:, self.config.n_memory_tokens:-1, :]
        target_labels = target_ids[:, 1:]

        ce_loss = F.cross_entropy(
            target_logits.reshape(-1, self.config.vocab_size),
            target_labels.reshape(-1),
            ignore_index=-100,
        )

        # MSE alignment loss
        mse_loss = torch.tensor(0.0, device=document_ids.device)
        if self.config.use_mse_alignment:
            doc_embeds = self.embed_tokens(document_ids)
            doc_avg = doc_embeds.mean(dim=1)
            mem_avg = memory_embeds.mean(dim=1)
            mse_loss = F.mse_loss(mem_avg, doc_avg)

        total_loss = ce_loss + self.config.lambda_mse * mse_loss

        return {
            'loss': total_loss,
            'ce_loss': ce_loss.item(),
            'mse_loss': mse_loss.item() if isinstance(mse_loss, torch.Tensor) else 0.0,
        }
