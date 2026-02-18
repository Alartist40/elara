"""
TRMCore - Tiny Recursive Model core engine.
2-layer weight-shared network with adaptive halting and deep supervision.
Based on TRM.md specification.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, Optional, List, Tuple

from elara_core.trm.block import TRMBlock
from elara_core.trm.halting import AdaptiveHalting
from elara_core.trm.state import LatentStateManager


class TRMConfig:
    """Configuration for TRM Core."""
    def __init__(
        self,
        vocab_size: int = 32000,
        d_model: int = 512,
        n_layers: int = 2,
        n_heads: int = 8,
        d_head: int = 64,
        d_ff: int = 1024,
        max_recursions: int = 6,
        confidence_threshold: float = 0.85,
        use_mixer: bool = False,
        dropout: float = 0.0,
        deep_supervision: bool = True,
        lambda_ds: float = 0.1,
    ):
        self.vocab_size = vocab_size
        self.d_model = d_model
        self.n_layers = n_layers
        self.n_heads = n_heads
        self.d_head = d_head
        self.d_ff = d_ff
        self.max_recursions = max_recursions
        self.confidence_threshold = confidence_threshold
        self.use_mixer = use_mixer
        self.dropout = dropout
        self.deep_supervision = deep_supervision
        self.lambda_ds = lambda_ds


class TRMCore(nn.Module):
    """
    Tiny Recursive Model - Core reasoning engine.

    Architecture:
    - 2-layer core (weight-shared across recursions)
    - Adaptive halting (learned confidence scoring)
    - Deep supervision (loss on every intermediate state)
    - Optional MLP-Mixer variant
    """

    def __init__(self, config: TRMConfig):
        super().__init__()
        self.config = config

        # Embedding
        self.embed_tokens = nn.Embedding(config.vocab_size, config.d_model)

        # Core blocks (weight-shared across recursions)
        self.blocks = nn.ModuleList([
            TRMBlock(
                d_model=config.d_model,
                n_heads=config.n_heads,
                d_head=config.d_head,
                d_ff=config.d_ff,
                dropout=config.dropout,
                use_mixer=config.use_mixer,
            )
            for _ in range(config.n_layers)
        ])

        # Recursion step embedding (optional)
        self.step_embedding = nn.Embedding(config.max_recursions, config.d_model)

        # Adaptive halting
        self.halting = AdaptiveHalting(
            d_model=config.d_model,
            threshold=config.confidence_threshold,
        )

        # Output head
        self.output_norm = nn.LayerNorm(config.d_model)
        self.lm_head = nn.Linear(config.d_model, config.vocab_size, bias=False)

        # Tie weights
        self.lm_head.weight = self.embed_tokens.weight

    def recurse(
        self,
        hidden_states: torch.Tensor,
        state_manager: LatentStateManager,
    ) -> Tuple[torch.Tensor, int]:
        """
        Run recursive reasoning until halting.

        Args:
            hidden_states: [B, L, D] initial state (from context/CLaRa).
            state_manager: Tracks recursion states.

        Returns:
            final_state: [B, L, D] refined latent state
            n_steps: number of recursion steps taken
        """
        accumulated_prob = torch.zeros(hidden_states.shape[0], device=hidden_states.device)
        current = hidden_states

        for step in range(self.config.max_recursions):
            # Add step embedding
            step_emb = self.step_embedding(
                torch.tensor(step, device=hidden_states.device)
            ).unsqueeze(0).unsqueeze(0)
            current = current + step_emb

            # Pass through weight-shared blocks
            for block in self.blocks:
                current = block(current, recursion_step=step)

            # Halting check
            halt_prob, accumulated_prob, should_halt = self.halting(
                current, accumulated_prob, step, self.config.max_recursions
            )

            # Record state
            state_manager.update(current, halt_prob)

            if should_halt:
                break

        return state_manager.get_weighted_output(), step + 1

    def forward(
        self,
        input_ids: Optional[torch.Tensor] = None,
        hidden_states: Optional[torch.Tensor] = None,
        labels: Optional[torch.Tensor] = None,
    ) -> Dict[str, torch.Tensor]:
        """
        Forward pass.

        Args:
            input_ids: [B, L] token ids (used if hidden_states not provided).
            hidden_states: [B, L, D] pre-computed states (from CLaRa/TiDAR).
            labels: [B, L] target token ids for loss computation.

        Returns:
            Dictionary with logits, loss, and recursion metadata.
        """
        if hidden_states is None:
            if input_ids is None:
                raise ValueError("Either input_ids or hidden_states must be provided")
            hidden_states = self.embed_tokens(input_ids)

        state_manager = LatentStateManager(d_model=self.config.d_model)
        state_manager.init_state(hidden_states)

        # Recursive reasoning
        final_state, n_steps = self.recurse(hidden_states, state_manager)

        # Output
        normed = self.output_norm(final_state)
        logits = self.lm_head(normed)

        result = {
            "logits": logits,
            "n_recursions": n_steps,
            "final_state": final_state,
        }

        # Loss
        if labels is not None:
            shift_logits = logits[:, :-1, :].contiguous()
            shift_labels = labels[:, 1:].contiguous()

            ce_loss = F.cross_entropy(
                shift_logits.view(-1, self.config.vocab_size),
                shift_labels.view(-1),
                ignore_index=-100,
            )

            # Deep supervision loss
            ds_loss = torch.tensor(0.0, device=logits.device)
            if self.config.deep_supervision:
                intermediate_states = state_manager.get_deep_supervision_targets()
                for state in intermediate_states[:-1]:  # Skip last (already in main loss)
                    state_logits = self.lm_head(self.output_norm(state))
                    ds_loss += F.cross_entropy(
                        state_logits[:, :-1, :].contiguous().view(-1, self.config.vocab_size),
                        shift_labels.view(-1),
                        ignore_index=-100,
                    )
                if len(intermediate_states) > 1:
                    ds_loss = ds_loss / (len(intermediate_states) - 1)

            # Ponder loss
            ponder_loss = self.halting.compute_ponder_loss(
                state_manager.halt_probs,
                target_steps=3,
            )

            total_loss = ce_loss + self.config.lambda_ds * ds_loss + 0.01 * ponder_loss

            result["loss"] = total_loss
            result["ce_loss"] = ce_loss.item()
            result["ds_loss"] = ds_loss.item() if isinstance(ds_loss, torch.Tensor) else 0.0
            result["ponder_loss"] = ponder_loss.item()

        return result

    @torch.no_grad()
    def generate(
        self,
        input_ids: torch.Tensor,
        max_new_tokens: int = 100,
        temperature: float = 0.7,
        top_p: float = 0.9,
    ) -> torch.Tensor:
        """
        Autoregressive generation with recursive refinement at each step.

        Args:
            input_ids: [B, L] prompt token ids.
            max_new_tokens: Maximum tokens to generate.
            temperature: Sampling temperature.
            top_p: Nucleus sampling threshold.

        Returns:
            [B, L + new_tokens] completed sequence.
        """
        generated = input_ids.clone()

        for _ in range(max_new_tokens):
            hidden_states = self.embed_tokens(generated)

            state_manager = LatentStateManager(d_model=self.config.d_model)
            state_manager.init_state(hidden_states)
            final_state, _ = self.recurse(hidden_states, state_manager)

            # Get logits for last position
            logits = self.lm_head(self.output_norm(final_state[:, -1:, :]))
            logits = logits.squeeze(1) / max(temperature, 1e-8)

            # Top-p sampling
            sorted_logits, sorted_indices = torch.sort(logits, descending=True)
            cumulative_probs = torch.cumsum(F.softmax(sorted_logits, dim=-1), dim=-1)
            mask = cumulative_probs - F.softmax(sorted_logits, dim=-1) >= top_p
            sorted_logits[mask] = float('-inf')

            probs = F.softmax(sorted_logits, dim=-1)
            next_token_sorted = torch.multinomial(probs, num_samples=1)
            next_token = sorted_indices.gather(-1, next_token_sorted)

            generated = torch.cat([generated, next_token], dim=-1)

            # Check for EOS (token id 2 for Mistral)
            if (next_token == 2).all():
                break

        return generated
