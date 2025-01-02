# Multimodal LLM with Phi Architecture

A powerful multimodal Language Learning Model that processes text, images, and audio inputs. Based on Microsoft's Phi architecture with CLIP for vision and Whisper for audio processing.

## Architecture Overview

![Architecture Diagram](docs/architecture.png)

### Components
1. **Base Model**: Microsoft Phi-1.5
2. **Vision Processing**: CLIP ViT-B/32
3. **Audio Processing**: Whisper ASR
4. **Training Method**: QLoRA (Quantized Low-Rank Adaptation)

## Features

- **Multi-Input Processing**:
  - Text: Direct language processing
  - Image: Vision-language understanding
  - Audio: Speech-to-text conversion and processing
- **Efficient Training**:
  - 4-bit quantization
  - Low-rank adaptation
  - Gradient checkpointing
- **Optimized Inference**:
  - Real-time audio processing
  - Cached image embeddings
  - Interactive web interface

## Project Structure 