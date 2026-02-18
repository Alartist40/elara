"""
DifferentiableTopK - Straight-Through estimator for differentiable top-k selection.
Key CLaRa innovation: allows gradients to flow from generator back to retriever.
Based on Algorithm 1 in CLaRa paper (Section 3).
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Tuple


class DifferentiableTopK(nn.Module):
    """
    Straight-Through (ST) estimator for differentiable top-k selection.
    Hard selection in forward pass, soft gradients in backward pass.
    Enables end-to-end joint training of retriever and generator.
    """

    def __init__(self, k: int = 5, temperature: float = 0.01):
        super().__init__()
        self.k = k
        self.temperature = temperature
        self.eps = 1e-10

    def forward(self, similarities: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Differentiable top-k selection.

        Args:
            similarities: [B, D] cosine similarities between query and documents.

        Returns:
            hard_indices: [B, k] indices of top-k documents (hard, for forward).
            soft_weights: [B, k, D] soft selection weights (for backward gradient).
        """
        B, D = similarities.shape

        # Hard selection (forward pass)
        hard_values, hard_indices = torch.topk(similarities, self.k, dim=-1)

        # Soft selection using temperature-scaled softmax (backward pass)
        logits = similarities / self.temperature

        # Iterative soft selection (Algorithm 1 from paper)
        mask = torch.ones_like(similarities)
        soft_selections = []

        for i in range(self.k):
            # Masked softmax
            masked_logits = logits + torch.log(mask + self.eps)
            probs = F.softmax(masked_logits, dim=-1)

            # Hard selection for this position
            selected_idx = hard_indices[:, i]

            # Create one-hot for hard selection
            hard_onehot = F.one_hot(selected_idx, num_classes=D).float()

            # ST estimator: hard forward, soft backward
            st_selection = hard_onehot + (probs - probs.detach())
            soft_selections.append(st_selection)

            # Mask out selected for next iteration
            mask = mask.scatter(1, selected_idx.unsqueeze(1), 0.0)

        # Stack selections: [B, k, D]
        soft_weights = torch.stack(soft_selections, dim=1)

        return hard_indices, soft_weights
