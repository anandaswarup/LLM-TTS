"""
This module defines configuration parameters and settings for the LLM-TTS model. The configuration includes
hyperparameters, model architecture settings, training parameters and other settings that control the behavior
and performance of the text-to-speech system.
"""

from dataclasses import dataclass

import torch


@dataclass
class ModelConfig:
    """
    This class defines model architecture settings for the LLM-TTS model.
    """

    model_context: int = 8192  # Max context length for the model
    rope_base: float = 500000.0  # Base value for frequency computation
    d_model: int = 2048  # Transformer model dimension
    num_heads: int = 32  # Number of attention heads
    num_kv_heads: int = 8  # Number of key / value heads for GQA
    num_layers: int = 16  # Number of transformer layers
    d_ffn: int = 8192  # Feed Forward Network hidden dimension
    dtype: torch.dtype = (
        torch.bfloat16
    )  # Lower precision dtype to reduce model memory footprint
    device: torch.device = (
        torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
    )
