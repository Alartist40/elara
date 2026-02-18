"""
LatentStateManager - Manages latent reasoning state across recursion steps.
Handles state initialization, mixing, and deep supervision targets.
"""

import torch
import torch.nn as nn
from typing import List, Optional


class LatentStateManager:
    """
    Manages the latent reasoning state for TRM recursions.
    Handles initialization, state accumulation, and deep supervision.
    """

    def __init__(self, d_model: int = 512, n_memory_tokens: int = 16):
        self.d_model = d_model
        self.n_memory_tokens = n_memory_tokens
        self.history: List[torch.Tensor] = []
        self.halt_probs: List[torch.Tensor] = []

    def init_state(
        self,
        context_embedding: torch.Tensor,
    ) -> torch.Tensor:
        """
        Initialize latent state from context embedding.

        Args:
            context_embedding: [B, L, D] from CLaRa or direct input.

        Returns:
            initial_state: [B, L, D] ready for recursion.
        """
        self.history = [context_embedding.clone()]
        self.halt_probs = []
        return context_embedding

    def update(
        self,
        new_state: torch.Tensor,
        halt_prob: Optional[torch.Tensor] = None,
    ) -> None:
        """Record a recursion step's output."""
        self.history.append(new_state.clone())
        if halt_prob is not None:
            self.halt_probs.append(halt_prob)

    def get_weighted_output(self) -> torch.Tensor:
        """
        Weighted combination of intermediate states using halt probs.
        ACT-style: output = Σ h_t * s_t
        """
        if not self.halt_probs or len(self.halt_probs) != len(self.history) - 1:
            # If no halt probs, return last state
            return self.history[-1]

        output = torch.zeros_like(self.history[0])
        for prob, state in zip(self.halt_probs, self.history[1:]):
            # Expand halt_prob from [B] to [B, 1, 1]
            weight = prob.unsqueeze(-1).unsqueeze(-1)
            output = output + weight * state

        return output

    def get_deep_supervision_targets(self) -> List[torch.Tensor]:
        """
        Return all intermediate states for deep supervision.
        Each state is a valid prediction target.
        """
        return self.history[1:]  # Skip initial state

    @property
    def num_steps(self) -> int:
        return len(self.history) - 1  # Subtract initial state

    def clear(self) -> None:
        self.history = []
        self.halt_probs = []
