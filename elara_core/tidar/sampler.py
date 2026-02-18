"""
DraftVerifySampler - Two-phase sampling for TiDAR.
Phase 1: Diffusion heads draft multiple tokens in parallel.
Phase 2: AR heads verify and correct the draft.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Tuple


class DraftVerifySampler:
    """
    Two-phase TiDAR draft-verify sampling.

    Phase 1 (Diffusion): Draft block_size tokens in parallel
    Phase 2 (AR): Verify each drafted token, correct mismatches

    Result: Up to block_size tokens per step with quality guarantees.
    """

    def __init__(
        self,
        block_size: int = 8,
        beta: float = 0.5,
        temperature: float = 0.8,
        top_p: float = 0.9,
    ):
        self.block_size = block_size
        self.beta = beta
        self.temperature = temperature
        self.top_p = top_p
        self.acceptance_rate = 0.0

    def draft_tokens(
        self,
        diffusion_logits: torch.Tensor,
    ) -> torch.Tensor:
        """
        Draft tokens using diffusion heads (parallel sampling).

        Args:
            diffusion_logits: [B, block_size, V] logits from diffusion heads.

        Returns:
            draft_tokens: [B, block_size] drafted token indices.
        """
        probs = F.softmax(diffusion_logits / self.temperature, dim=-1)
        return torch.multinomial(probs.view(-1, probs.size(-1)), 1).view(
            probs.size(0), probs.size(1)
        )

    def verify_tokens(
        self,
        draft_tokens: torch.Tensor,
        ar_logits: torch.Tensor,
    ) -> Tuple[torch.Tensor, int]:
        """
        Verify drafted tokens using AR heads.
        Accept drafts that match AR predictions; reject and re-sample mismatches.

        Args:
            draft_tokens: [B, block_size] drafted token indices.
            ar_logits: [B, block_size, V] logits from AR verification pass.

        Returns:
            verified_tokens: [B, accepted_length] accepted token sequence
            accepted_length: number of tokens accepted (up to block_size)
        """
        B, L, V = ar_logits.shape

        # AR predictions
        ar_probs = F.softmax(ar_logits / self.temperature, dim=-1)

        # Get AR's top prediction for each position
        ar_top = ar_probs.argmax(dim=-1)  # [B, L]

        # Accept until first mismatch (speculative decoding style)
        # For each batch, find first position where draft != AR prediction
        matches = (draft_tokens == ar_top)  # [B, L]

        # Find first mismatch per batch
        accepted_lengths = []
        for b in range(B):
            match_seq = matches[b]
            first_mismatch = (~match_seq).nonzero(as_tuple=True)[0]
            if len(first_mismatch) == 0:
                accepted_lengths.append(L)  # All matched
            else:
                idx = first_mismatch[0].item()
                accepted_lengths.append(idx + 1)  # Accept up to mismatch, replace mismatch with AR

        max_accepted = max(accepted_lengths)

        # Build verified tokens
        verified = torch.zeros(B, max_accepted, dtype=draft_tokens.dtype, device=draft_tokens.device)
        for b in range(B):
            n = accepted_lengths[b]
            # Accept matching tokens
            if n > 0:
                verified[b, :n-1] = draft_tokens[b, :n-1]
                # Last accepted position: use AR prediction (in case of mismatch)
                verified[b, n-1] = ar_top[b, n-1]

        # Update acceptance rate
        total = B * L
        total_accepted = sum(accepted_lengths)
        self.acceptance_rate = total_accepted / total

        return verified, max_accepted

    def blend_outputs(
        self,
        ar_logits: torch.Tensor,
        diff_logits: torch.Tensor,
    ) -> torch.Tensor:
        """
        Blend AR and diffusion logits using beta mixing.

        Args:
            ar_logits: [B, L, V] from AR heads
            diff_logits: [B, L, V] from diffusion heads

        Returns:
            blended_logits: [B, L, V]
        """
        return self.beta * ar_logits + (1 - self.beta) * diff_logits
