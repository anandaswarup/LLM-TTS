"""
Attention layer used in the LLM-TTS model, which is a LLaMA-style auto-regressive Transformer to predict
discrete acoustic tokens conditioned on text input.
"""

import torch
import torch.nn as nn
from einops import rearrange

from model.rope import apply_rotary_embedding, build_freqs_cache


def build_attention_mask_cache(model_context: int) -> torch.Tensor:
    """
    Precompute the attention mask cache.

    Args:
        model_context (int): The length of the max sequence that the model can handle.

    Returns:
        torch.Tensor: The attention mask cache tensor of shape [model_context, model_context].
    """
    return torch.triu(torch.ones(model_context, model_context), diagonal=1)


class GroupedQueryAttention(nn.Module):
    """
    Attention layer used in the LLM-TTS model. The attention mechanism used is Grouped Query Attention which is a
    variant of Multi Head Attention where the number of key / value heads is less than the number of query heads.
    """

    def __init__(
        self,
        d_model: int,
        num_heads: int,
        num_kv_heads: int,
        model_context: int,
        rope_base: float,
        dtype: torch.dtype | None = None,
    ) -> None:
        """
        Instantiate the Grouped Query Attention layer.

        Args:
            d_model (int): Transformer model dimension.
            num_heads (int): Number of attention heads.
            num_kv_heads (int): Number of key / value heads for GQA.
            model_context (int): The length of the max sequence that the model can handle.
            rope_base (float): The base value for frequency computation.
            dtype (torch.dtype | None): Data type for the attention weights.
        """
        super().__init__()

        assert d_model % num_heads == 0, "d_model must be divisible by num_heads"
        assert num_heads % num_kv_heads == 0, (
            "num_heads must be divisible by num_kv_heads"
        )

        self.d_model = d_model
        self.num_heads = num_heads
        self.num_kv_heads = num_kv_heads

        self.d_head = d_model // num_heads
        self.group_size = num_heads // num_kv_heads

        # Linear projections for query, key, value, output
        self.q_proj = nn.Linear(
            self.d_model, self.d_head * self.num_heads, bias=False, dtype=dtype
        )
        self.k_proj = nn.Linear(
            self.d_model, self.d_head * self.num_kv_heads, bias=False, dtype=dtype
        )
        self.v_proj = nn.Linear(
            self.d_model, self.d_head * self.num_kv_heads, bias=False, dtype=dtype
        )
        self.o_proj = nn.Linear(
            self.d_head * self.num_heads, self.d_model, bias=False, dtype=dtype
        )

        # Frequency cache for rotary positional embeddings
        freqs_cache = build_freqs_cache(self.d_head, model_context, rope_base)
        if dtype is not None:
            freqs_cache = freqs_cache.to(dtype=dtype)
        self.register_buffer("freqs", freqs_cache)

        # Mask cache for attention scores masking
        mask_cache = build_attention_mask_cache(model_context)
        self.register_buffer("mask", mask_cache)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass of the Grouped Query Attention layer.

        Args:
            x (torch.Tensor): Input tensor of shape [batch_size, seq_len, d_model].

        Returns:
            torch.Tensor: Output tensor of shape [batch_size, seq_len, d_model].
        """
        # Get the number of tokens in the sequence
        num_tokens = x.size(1)

        # query projection [batch_size, num_tokens, d_head * num_heads or num_kv_heads]
        q = self.q_proj(x)
        k = self.k_proj(x)
        v = self.v_proj(x)

        # Reshape the query, key, value tensors to [batch_size, num_heads or num_kv_heads, num_tokens, d_head]
        q = rearrange(q, "b n (h d) -> b h n d", h=self.num_heads)
        k = rearrange(k, "b n (h d) -> b h n d", h=self.num_kv_heads)
        v = rearrange(v, "b n (h d) -> b h n d", h=self.num_kv_heads)

        # Apply RoPE to query and key tensors
        q = apply_rotary_embedding(q, self.freqs)
        k = apply_rotary_embedding(k, self.freqs)

        # Expand key and value tensors to match the query tensor shape
        k = k.repeat_interleave(self.group_size, dim=1)
        v = v.repeat_interleave(self.group_size, dim=1)

        # Compute the attention scores
        attn_scores = torch.einsum("b h i d, b h j d -> b h i j", q, k)

        # Apply causal masking to the attention scores
        mask = self.mask[:num_tokens, :num_tokens].bool()
        attn_scores.masked_fill_(mask, -torch.inf)

        # Normalize the attention scores
        attn_weights = torch.softmax(attn_scores / k.shape[-1] ** 0.5, dim=-1)

        # Compute attention context
        attn_context = torch.einsum("b h i j, b h j d -> b h i d", attn_weights, v)

        # Reshape the attention context to [batch_size, num_tokens, d_model]
        attn_context = rearrange(attn_context, "b h n d -> b n (h d)")

        # Project the attention context to the output dimension
        attn_context = self.o_proj(attn_context)

        return attn_context
