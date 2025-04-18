# LLM-TTS: Text-to-Speech with Large Language Models

LLM-TTS is a single stage auto-regressive transformer model capable of generating speech samples conditioned on text prompts. The text prompts are passed through a tokenizer to obtain a sequence of text tokens. LLM-TTS is then trained to predict discrete audio tokens, or audio codes, conditioned on these text tokens. These audio tokens are then decoded using an audio compression model, such as EnCodec, to recover the audio waveform

## Overview

LLM-TTS takes a language modeling approach to speech synthesis:

1. Text Input: You provide a text prompt (e.g., “Hello world!”).
2. LLM Model: A transformer model auto-regressively predicts a sequence of acoustic tokens conditioned on the text prompt.
3. Neural Codec Decoder: The generated acoustic tokens are decoded back into waveforms using a neural codec vocoder.