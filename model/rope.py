"""
Rotary Positional Embeddings Module. This module implements the rotary positional embeddings as described in
https://arxiv.org/abs/2104.09864.
"""

import torch
import torch.nn as nn
from einops import rearrange


class RotaryEmbedding(nn.Module):
    """
    PyTorch module for Rotary Positional Embeddings (RoPE).

    This module precomputes the rotary embeddings during initialization and provides
    methods to apply them efficiently to query and key tensors in attention mechanisms.
    """

    def __init__(
        self,
        d_model: int,
        num_heads: int,
        model_context: int,
        rope_theta: float = 50000.0,
    ):
        """
        Initialize the RotaryEmbedding module.

        Args:
            d_model (int): Transformer model dimension
            num_heads (int): Number of attention heads
            model_context (int): Length of the max sequence that the model can handle.
            rope_theta (float): The base value for frequency computation.
        """
        super().__init__()

        if (d_model // num_heads) % 2 != 0:
            raise ValueError(f"Dimension must be even, got {d_model // num_heads}")

        self.dim = d_model // num_heads
        self.model_context = model_context
        self.rope_theta = rope_theta

        # Precompute frequencies during initialization
        self.register_buffer("freqs_cache", self._precompute_freqs())

    def _precompute_freqs(self) -> torch.Tensor:
        """
        Precompute the rotary positional frequencies for each position in the sequence.

        Returns:
            Tensor containing cached frequency values with shape [model_context, dim//2, 2]
        """
        # Compute the frequencies for each position in the sequence
        inv_freqs = 1.0 / (
            self.rope_theta
            ** (torch.arange(0, self.dim, 2)[: (self.dim // 2)].float() / self.dim)
        )

        # Compute the indices for each position in the sequence
        idxs = torch.arange(self.model_context, dtype=torch.float32)

        # Use torch.einsum to compute the outer product of idxs and freqs
        angles = torch.einsum("i,j->ij", idxs, inv_freqs)

        # Stack cos and sin values along the last dimension
        freqs_cache = torch.stack([torch.cos(angles), torch.sin(angles)], dim=-1)

        return freqs_cache

    def apply_embedding(self, x: torch.Tensor) -> torch.Tensor:
        """
        Apply rotary positional embeddings to a tensor.

        Args:
            x: Input tensor of shape [batch_size, num_heads or num_kv_heads, seq_len, d_head]

        Returns:
            Rotated tensor with the same shape as input
        """
        # Extract sequence length and verify it doesn't exceed cache length
        seq_len = x.shape[2]
        if seq_len > self.freqs_cache.shape[0]:
            raise ValueError(
                f"Input sequence length {seq_len} exceeds maximum cached length {self.freqs_cache.shape[0]}"
            )

        # Get cosine and sine values for the current sequence
        cos = self.freqs_cache[:seq_len, :, 0]  # [seq_len, dim//2]
        sin = self.freqs_cache[:seq_len, :, 1]  # [seq_len, dim//2]

        # Rearrange for broadcasting with einops
        # Transform to [1, 1, seq_len, d_head//2] for proper broadcasting
        cos = rearrange(cos, "s h -> 1 1 s h")
        sin = rearrange(sin, "s h -> 1 1 s h")

        # Split tensor along the last dimension
        x1, x2 = x.chunk(2, dim=-1)

        # Apply rotary embeddings to the two halves of the input tensor
        x1_rotated = x1 * cos - x2 * sin
        x2_rotated = x1 * sin + x2 * cos

        # Concatenate the rotated tensors
        x_rotated = torch.cat([x1_rotated, x2_rotated], dim=-1)

        return x_rotated

    def forward(
        self, q: torch.Tensor, k: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Apply rotary embeddings to both query and key tensors.

        Args:
            q: Query tensor of shape [batch_size, num_heads, seq_len, d_head]
            k: Key tensor of shape [batch_size, num_kv_heads, seq_len, d_head]

        Returns:
            Tuple of (rotated query tensor, rotated key tensor) with same shapes as inputs
        """
        q_rotated = self.apply_embedding(q)
        k_rotated = self.apply_embedding(k)

        return q_rotated, k_rotated
