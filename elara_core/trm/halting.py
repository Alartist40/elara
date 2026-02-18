"""
AdaptiveHalting - Learns when to stop recursive reasoning.
Based on ACT (Adaptive Computation Time) from TRM.md.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Tuple


class AdaptiveHalting(nn.Module):
    """
    Adaptive halting mechanism for TRM.
    Learns to stop recursive processing when confidence is high enough.
    Uses a halting probability network: h_t = σ(W_halt · s_t)
    """

    def __init__(self, d_model: int = 512, threshold: float = 0.85):
        super().__init__()
        self.threshold = threshold
        self.halt_net = nn.Sequential(
            nn.Linear(d_model, d_model // 4),
            nn.GELU(),
            nn.Linear(d_model // 4, 1),
            nn.Sigmoid(),
        )
        self._ponder_cost = 0.0

    def forward(
        self,
        hidden_states: torch.Tensor,
        accumulated_prob: torch.Tensor,
        step: int,
        max_steps: int,
    ) -> Tuple[torch.Tensor, torch.Tensor, bool]:
        """
        Compute halting probability and decide whether to stop.

        Args:
            hidden_states: [B, L, D] current latent state
            accumulated_prob: [B] running sum of halting probabilities
            step: current recursion step index
            max_steps: maximum allowed recursion steps

        Returns:
            halt_prob: [B] probability of halting at this step
            new_accumulated: [B] updated accumulated probability
            should_halt: bool indicating if all samples should halt
        """
        # Compute halting probability from pooled hidden state
        pooled = hidden_states.mean(dim=1)  # [B, D]
        halt_prob = self.halt_net(pooled).squeeze(-1)  # [B]

        # Force halt at max steps
        if step >= max_steps - 1:
            remainder = 1.0 - accumulated_prob
            return remainder, torch.ones_like(accumulated_prob), True

        new_accumulated = accumulated_prob + halt_prob

        # Check if all samples have crossed threshold
        should_halt = (new_accumulated >= self.threshold).all().item()

        # Clamp to prevent exceeding 1.0
        new_accumulated = torch.clamp(new_accumulated, max=1.0)

        return halt_prob, new_accumulated, should_halt

    def compute_ponder_loss(
        self,
        halt_probs: list,
        target_steps: int = 3,
    ) -> torch.Tensor:
        """
        Compute ponder loss to regularize computation depth.
        Encourages the model to halt early when possible.
        """
        actual_steps = len(halt_probs)
        # Simple L1 penalty on excess steps
        ponder_loss = torch.tensor(
            max(0, actual_steps - target_steps) * 0.01,
            requires_grad=True,
        )
        self._ponder_cost = actual_steps
        return ponder_loss

    @property
    def last_ponder_cost(self) -> float:
        """Return last recorded ponder cost (number of recursion steps)."""
        return self._ponder_cost
