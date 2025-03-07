"""
Rotary Positional Embeddings Module. This module implements the rotary positional embeddings as described in
https://arxiv.org/abs/2104.09864.
"""

import torch
import torch.nn as nn
from einops import repeat


class RotaryPositionalEmbedding(nn.Module):
    """
    Rotary Positional Embeddings as described in https://arxiv.org/abs/2104.09864.
    """

    def __init__(
        self, d_head: int, context_length: int = 2048, rope_base: int = 10000
    ) -> None:
        """
        Args:
            d_head: Dimension of each attention head (must be even).
            context_length: Model context length
            rope_base: Base for the frequency computation.
        """
        super().__init__()

        assert d_head % 2 == 0, "Dimension must be even for rotary embeddings."

        self.d_head = d_head
        self.context_length = context_length
        self.rope_base = rope_base

        # Compute the inverse frequency for each pair of dimensions.
        inv_freq = 1.0 / (
            rope_base ** (torch.arange(0, d_head, 2, dtype=torch.float32) / d_head)
        )

        # Create position indices [0, 1, ..., max_seq_len - 1]
        positions = torch.arange(context_length, dtype=torch.float32)

        # Precompute the sinusoidal frequencies. Shape: [context_length, d_head//2]
        freqs = torch.einsum("i,j->ij", positions, inv_freq)

        # Precompute cosine and sine caches. Shape: [context_length, d_head//2]
        self.cos_cached = torch.cos(freqs)
        self.sin_cached = torch.sin(freqs)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Applies rotary positional embeddings to tensor x.

        Args:
            x: Input tensor of shape [batch_size, seq_length, n_heads, d_head].

        Returns:
            Tensor after applying rotary positional embeddings.
        """
        # x should have head dimension equal to self.dim.
        batch_size, seq_len, n_heads = x.size(0), x.size(1), x.size(2)

        # Get the cached cosine and sine for the current sequence length.

        cos = self.cos_cached[:seq_len].to(x.device)  # shape: [seq_len, d_head//2]
        sin = self.sin_cached[:seq_len].to(x.device)  # shape: [seq_len, d_head//2]

        # Expand dimensions to match x's shape: [batch, seq_length, n_heads, d_head//2]
        cos = repeat(cos, "l d -> b l n d", b=batch_size, n=n_heads)
        sin = repeat(sin, "l d -> b l n d", b=batch_size, n=n_heads)

        # Split the input tensor into two halves along the head dimension.
        x1 = x[..., : self.d_head // 2]
        x2 = x[..., self.d_head // 2 :]

        # Apply rotary transformation.
        # For each half: [x1, x2] -> [x1*cos - x2*sin, x1*sin + x2*cos]
        x_first_half_rotated = x1 * cos - x2 * sin
        x_second_half_rotated = x1 * sin + x2 * cos

        # Concatenate the rotated halves.
        return torch.cat([x_first_half_rotated, x_second_half_rotated], dim=-1)
