"""
QueryReasoner - Encodes queries into the same space as compressed documents.
Based on CLaRa: θ_qr (query reasoner) from Section 3 of the paper.
"""

import torch
import torch.nn as nn
from typing import Optional
from elara_core.clara.compressor import CLaRaBlock, RMSNorm, SCPCompressorConfig


class QueryReasoner(nn.Module):
    """
    Query Reasoner (θ_qr in paper).
    Encodes queries into the same embedding space as compressed documents.
    Initialized from SCP checkpoint for alignment.
    """

    def __init__(self, config: SCPCompressorConfig):
        super().__init__()
        self.config = config

        # Same architecture as compressor, initialized from SCP checkpoint
        self.embed_tokens = nn.Embedding(config.vocab_size, config.d_model)
        self.layers = nn.ModuleList([
            CLaRaBlock(
                config.d_model, config.n_heads, config.d_head,
                config.d_ff, config.dropout, use_lora=True
            )
            for _ in range(config.n_layers)
        ])
        self.norm = RMSNorm(config.d_model)

        # Learnable query memory tokens (same count as doc memory tokens)
        self.query_memory_tokens = nn.Parameter(
            torch.randn(1, config.n_memory_tokens, config.d_model)
        )

        self._init_weights()

    def _init_weights(self):
        for module in self.modules():
            if isinstance(module, (nn.Linear, nn.Embedding)):
                torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)

    def forward(self, query_ids: torch.Tensor) -> torch.Tensor:
        """
        Encode query into memory token representations.

        Args:
            query_ids: [B, query_len] tokenized query

        Returns:
            query_embedding: [B, n_memory_tokens, d_model]
        """
        B, query_len = query_ids.shape

        # Embed query
        query_embeds = self.embed_tokens(query_ids)

        # Expand query memory tokens
        query_memory = self.query_memory_tokens.expand(B, -1, -1)

        # Concatenate [query | memory_tokens]
        combined = torch.cat([query_embeds, query_memory], dim=1)

        # Forward
        hidden_states = combined
        for layer in self.layers:
            hidden_states = layer(hidden_states)
        hidden_states = self.norm(hidden_states)

        # Extract query memory representations
        query_embedding = hidden_states[:, -self.config.n_memory_tokens:, :]

        return query_embedding

    def load_from_compressor(self, compressor_state_dict: dict) -> None:
        """
        Initialize from a trained SCP compressor for alignment.
        Keys are mapped from compressor namespace.
        """
        own_state = self.state_dict()
        for name, param in compressor_state_dict.items():
            # Map compressor keys to query reasoner keys
            mapped_name = name.replace("memory_tokens", "query_memory_tokens")
            if mapped_name in own_state and own_state[mapped_name].shape == param.shape:
                own_state[mapped_name].copy_(param)
