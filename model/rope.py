"""
Rotary Positional Embeddings used in the LLM-TTS model, which is a LLaMA-style auto-regressive Transformer to
predict discrete acoustic tokens conditioned on text input.
"""

import torch
from einops import rearrange


def build_freqs_cache(
    d_head: int, model_context: int, rope_base: float = 500000.0
) -> torch.Tensor:
    """
    Precompute the frequency cache for the Rotary Positional Embeddings.

    Args:
        d_head (int): The dimension of the attention head.
        model_context (int): The length of the max sequence that the model can handle.
        rope_base (float): The base value for frequency computation.

    Returns:
        torch.Tensor: The frequency cache tensor of shape [model_context, d_head].
    """
    assert d_head % 2 == 0, "d_head must be divisible by 2"

    # Compute the inverse frequencies
    inv_freq = 1.0 / (
        rope_base ** (torch.arange(0, d_head, 2)[: (d_head // 2)].float() / d_head)
    )

    # Generate position indices
    position_idxs = torch.arange(model_context)

    # Compute the angles
    angles = torch.einsum("i,j->ij", position_idxs, inv_freq)

    # Compute the frequency cache
    freqs_cache = torch.cat([angles, angles], dim=1)

    return freqs_cache


def apply_rotary_embedding(x: torch.Tensor, freqs_cache: torch.Tensor) -> torch.Tensor:
    """
    Apply Rotary Positional Embeddings to the input tensor.

    Args:
        x (torch.Tensor): The input tensor of shape [batch_size, num_heads, seq_len, d_head].
        freqs_cache (torch.Tensor): The frequency cache tensor of shape [model_context, d_head].

    Returns:
        torch.Tensor: The output tensor of shape [batch_size, num_heads, seq_len, d_head].
    """
    seq_len, d_head = x.size(-2), x.size(-1)
    assert d_head % 2 == 0, "d_head must be divisible by 2"

    # Split x into two parts
    x_1 = x[..., : d_head // 2]
    x_2 = x[..., d_head // 2 :]

    # Get the cos and sin components of the frequencies [seq_len, d_head]
    cos = torch.cos(freqs_cache[:seq_len, :])
    sin = torch.sin(freqs_cache[:seq_len, :])

    # Expand the cos and sin components to match the input tensor shape
    cos = rearrange(cos, "s d -> () () s d")
    sin = rearrange(sin, "s d -> () () s d")

    # Apply the rotary positional embeddings
    x_rotated = torch.cat([-x_2, x_1], dim=-1)
    x_embed = (x * cos) + (x_rotated * sin)

    return x_embed
