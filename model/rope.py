"""
Rotary Positional Embeddings layer used in the LLM-TTS model, which is a LLaMA-style auto-regressive Transformer to
predict discrete acoustic tokens conditioned on text input.
"""

import torch
from einops import rearrange


def build_rope_cache(
    model_context: int, d_head: int, rope_base: float = 500000.0
) -> torch.Tensor:
    """
    Precompute the frequency cache for the Rotary Positional Embeddings.

    Args:
        model_context (int): The length of the max sequence that the model can handle.
        d_head (int): The dimension of the attention head.
        rope_base (float): The base value for frequency computation.

    Returns:
        torch.Tensor: The frequency cache tensor of shape [model_context, d_head//2, 2].
    """
    # Compute the theta values
    theta = 1.0 / (rope_base ** (torch.arange(0, d_head, 2) / d_head))

    # Compute the positions for the cache
    positions_idx = torch.arange(model_context)

    # Compute the angles; dot product of positions and theta
    angles = torch.einsum("i, j -> ij", positions_idx, theta)

    # Compute the frequencies
    freqs_cache = torch.stack([torch.cos(angles), torch.sin(angles)], dim=-1)

    return freqs_cache


def apply_rotary_embedding(x: torch.Tensor, freqs_cache: torch.Tensor) -> torch.Tensor:
    """
    Apply Rotary Positional Embeddings to the input tensor.

    Args:
        x (torch.Tensor): The input tensor of shape [batch_size, num_heads, seq_len, d_head].
        freqs_cache (torch.Tensor): The frequency cache tensor of shape [model_context, d_head//2, 2].

    Returns:
        torch.Tensor: The output tensor of shape [batch_size, num_heads, seq_len, d_head].
    """
    T = x.size(2)
    d_head = x.size(3)

    assert T <= freqs_cache.size(0), (
        "Input sequence length is greater than model context."
    )
    freqs_cache = freqs_cache[:T]

    # Convert to float for precision
    x = x.float()

    # Get cos and sin components
    cos = freqs_cache[..., 0]  # [T, d_head//2]
    sin = freqs_cache[..., 1]  # [T, d_head//2]

    # Reshape for broadcasting
    cos = rearrange(cos, "t d -> () () t d")
    sin = rearrange(sin, "t d -> () () t d")

    # Split the head dimension explicitly without using rearrange
    x1, x2 = x[..., : d_head // 2], x[..., d_head // 2 :]

    # Apply rotation
    x_out = torch.cat([x1 * cos - x2 * sin, x1 * sin + x2 * cos], dim=-1)

    return x_out.type_as(x)
