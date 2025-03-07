"""
Layers used in the LLM-TTS model. This module implements the core neural network layers and components used in the
LLM-TTS model which is a LLaMA-style auto-regressive Transformer to predict discrete acoustic tokens conditioned on text
input.
"""

import torch
import torch.nn as nn


def generate_causal_mask(seq_len: int, device: torch.device = None) -> torch.Tensor:
    """
    Generates a causal mask for autoregressive attention.

    Args:
        seq_len: The length of the input sequence.
        device: The device on which to create the mask.

    Returns:
        A [seq_len, seq_len] tensor where positions that should be masked are set to -1e9 and the rest are 0.
    """
    # Create an upper-triangular matrix filled with -1e9 for positions to be masked
    mask = torch.triu(torch.full((seq_len, seq_len), float("-1e9")), diagonal=1)

    if device is not None:
        mask = mask.to(device)

    return mask


class RMSNorm(nn.Module):
    """
    Root Mean Square Layer Normalization (RMSNorm) as described in https://arxiv.org/abs/1910.07467.
    """

    def __init__(self, d_model: int, eps: float = 1e-8) -> None:
        """
        Args:
            d_model: Transformer model dimension.
            eps (optional): A small value to avoid division by zero.
        """
        super().__init__()

        self.eps = eps

        # Learnable scaling parameter of shape [d_model]
        self.scale = nn.Parameter(torch.ones(d_model))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Applies RMSNorm to the input tensor.

        Args:
            x: Input tensor of shape [..., d_model].

        Returns:
            Normalized tensor of the same shape as x
        """
        # Compute the root mean square of the input tensor over the last dimension
        rms = torch.sqrt(torch.mean(x**2, dim=-1, keepdim=True) + self.eps)

        # Normalize the input tensor
        norm_x = self.scale * (x / rms)

        return norm_x
