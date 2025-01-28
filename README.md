# LLM-TTS: Text-to-Speech with LLaMA-Style Transformers and Neural Codec Tokens

This project aims to build a text-to-speech (TTS) system by training a LLaMA-style auto-regressive Transformer to predict discrete acoustic tokens (from EnCodec or another neural vocoder) conditioned on text input. Once trained, the model’s outputs (acoustic tokens) can be decoded back into waveforms using the corresponding neural codec vocoder.

## Overview

LLM-TTS takes a large language model approach to speech synthesis:

1. Text Input: You provide a text prompt (e.g., “Hello world!”).
2. Transformer Model: A LLaMA-style model auto-regressively predicts the next acoustic token based on the text context and previously generated tokens.
3. Neural Codec Decoder: The generated acoustic tokens are decoded back into waveforms using a neural codec vocoder.

