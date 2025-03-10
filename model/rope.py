"""
Rotary Positional Embeddings Module. This module implements the rotary positional embeddings as described in
https://arxiv.org/abs/2104.09864.
"""

import torch
from einops import rearrange


def precompute_rotary_freqencies(
    dim: int, model_context: int, rope_theta: float = 50000.0
) -> torch.Tensor:
    """
    Precompute the rotary positional frequencies for each position in the sequence.

    Args:
        dim (int): The dimension of the positional embeddings. Must be equal to d_model // num_heads, where d_model is
            the transformer model dimension and num_heads is the number of attention heads.
        model_context (int): Length of the max sequence that the model can handle.
        rope_theta (float): The theta value for the rope positional embeddings.
    """
    # Compute the frequencies for each position in the sequence
    inv_freqs = 1.0 / (
        rope_theta ** (torch.arange(0, dim, 2)[: (dim // 2)].float() / dim)
    )

    # Compute the indices for each position in the sequence
    idxs = torch.arange(model_context, device=inv_freqs.device, dtype=torch.float32)

    # Use torch.einsum to compute the outer product of idxs and freqs.
    angles = torch.einsum("i,j->ij", idxs, inv_freqs)

    # Stack cos and sin values along the last dimension
    freqs_cache = torch.stack([torch.cos(angles), torch.sin(angles)], dim=-1)

    return freqs_cache


def apply_rotary_embedding(x: torch.Tensor, freqs_cache: torch.Tensor) -> torch.Tensor:
    """
    Apply rotary positional embeddings to a tensor.

    Args:
        x: Input tensor of shape [batch_size, num_heads, seq_len, d_head]
        freqs_cache: Precomputed rotary frequencies from precompute_rotary_freqencies
                    with shape [seq_len, d_head//2, 2] where the last dimension
                    contains [cos, sin] values
    """
    # Extract sequence length and verify it doesn't exceed cache length
    seq_len = x.shape[1]
    if seq_len > freqs_cache.shape[0]:
        raise ValueError(
            f"Input sequence length {seq_len} exceeds maximum cached length {freqs_cache.shape[0]}"
        )

    # Get cosine and sine values for the current sequence
    cos = freqs_cache[:seq_len, :, 0]  # [seq_len, d_head//2]
    sin = freqs_cache[:seq_len, :, 1]  # [seq_len, d_head//2]

    # Rearrange for broadcasting with einops
    cos = rearrange(cos, "s h -> 1 s 1 h")  # [1, seq_len, 1, d_head//2]
    sin = rearrange(sin, "s h -> 1 s 1 h")  # [1, seq_len, 1, d_head//2]

    # Split tensor along the last dimension
    x1, x2 = x.chunk(2, dim=-1)

    # Apply rotary embeddings to the two halves of the input tensor
    x1_rotated = x1 * cos - x2 * sin
    x2_rotated = x1 * sin + x2 * cos

    # Concatenate the rotated tensors
    x_rotated = torch.cat([x1_rotated, x2_rotated], dim=-1)

    return x_rotated
