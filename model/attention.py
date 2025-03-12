"""
Attention layer used in the LLM-TTS model, which is a LLaMA-style auto-regressive Transformer to predict
discrete acoustic tokens conditioned on text input.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange

from model.rope import RotaryEmbedding


class Attention(nn.Module):
    """
    Attention layer used in the LLM-TTS model. The attention mechanism used is Grouped Query Attention which is a
    variant of Multi Head Attention where the number of key / value heads is less than the number of query heads.
    """

    def __init__(
        self,
        model_context: int,
        rope_theta: float,
        d_model: int,
        num_heads: int,
        num_kv_heads: int | None = None,
    ) -> None:
        """
        Instantiate the Attention layer.

        Args:
            model_context (int): The length of the max sequence that the model can handle.
            rope_theta (float): The base value for frequency computation.
            d_model (int): Transformer model dimension.
            num_heads (int): The number of attention heads (query heads).
            num_kv_heads (int, optional): The number of key / value heads. If none, it will be set to
                num_heads.
        """
        super().__init__()

        self.model_context = model_context
        self.rope_theta = rope_theta
        self.d_model = d_model
        self.num_heads = num_heads
        self.num_kv_heads = num_kv_heads if num_kv_heads is not None else num_heads

        self.d_head = d_model // num_heads
        self.num_groups = self.num_heads // self.num_kv_heads

        # Linear projections for query, key, value and output
        self.q_proj = nn.Linear(self.d_model, self.num_heads * self.d_head, bias=False)
        self.k_proj = nn.Linear(
            self.d_model, self.num_kv_heads * self.d_head, bias=False
        )
        self.v_proj = nn.Linear(
            self.d_model, self.num_kv_heads * self.d_head, bias=False
        )
        self.o_proj = nn.Linear(self.num_heads * self.d_head, self.d_model, bias=False)

        # Rotary Positional Embeddings
        self.rope = RotaryEmbedding(
            self.d_model, self.num_heads, self.model_context, self.rope_theta
        )

        self._init_weights()

    def _init_weights(self) -> None:
        """
        Initialize the weights of the linear projections.
        """
        nn.init.xavier_normal_(self.q_proj.weight, gain=1.0)
        if self.q_proj.bias is not None:
            nn.init.constant_(self.q_proj.bias, 0.0)

        nn.init.xavier_normal_(self.k_proj.weight, gain=1.0)
        if self.k_proj.bias is not None:
            nn.init.constant_(self.k_proj.bias, 0.0)

        nn.init.xavier_normal_(self.v_proj.weight, gain=1.0)
        if self.v_proj.bias is not None:
            nn.init.constant_(self.v_proj.bias, 0.0)

        nn.init.xavier_normal_(self.o_proj.weight, gain=1.0)
        if self.o_proj.bias is not None:
            nn.init.constant_(self.o_proj.bias, 0.0)

    def forward(
        self, x: torch.Tensor, mask: torch.Tensor | None = None
    ) -> torch.Tensor:
        """
        Forward pass of the Attention layer.

        Args:
            x (torch.Tensor): The input tensor of shape [batch_size, seq_len, d_model].
            mask (torch.Tensor, optional): The mask tensor of shape [seq_len, seq_len]. If provided, the attention
                scores will be masked.
        """
        # Apply linear projections to get query, key and value tensors
        q = self.q_proj(x)
        k = self.k_proj(x)
        v = self.v_proj(x)

        # Reshape query, key and value tensors
        q = rearrange(q, "b s (h d) -> b h s d", h=self.num_heads)
        k = rearrange(k, "b s (h d) -> b h s d", h=self.num_kv_heads)
        v = rearrange(v, "b s (h d) -> b h s d", h=self.num_kv_heads)

        # Apply rotary positional embeddings to query and key tensors
        q, k = self.rope(q, k)

        # Repeat k/v heads if num_kv_heads < num_heads
        if self.num_kv_heads < self.num_heads:
            k = torch.repeat_interleave(k, self.num_groups, dim=1)
            v = torch.repeat_interleave(v, self.num_groups, dim=1)

        # Compute the attention scores
        attn_scores = torch.einsum("b h s d, b h t d -> b h s t", q, k)

        # Apply mask if provided
        if mask is not None:
            attn_scores = attn_scores + mask

        # Normalize the attention scores
        attn_scores = F.softmax(attn_scores.float(), dim=-1).type_as(q)

        # Compute attention output
        output = torch.einsum("b h s t, b h t d -> b h s d", attn_scores, v)

        # Reshape output tensor and apply output projection
        output = rearrange(output, "b h s d -> b s (h d)")
        output = self.o_proj(output)

        return output
