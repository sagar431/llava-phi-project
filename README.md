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

## Technical Details

### Image Processing
- **CLIP Integration**:
  - Model: ViT-B/32
  - Preprocessing: 224x224 resolution, normalized
  - Embedding dimension: 512
  - Storage: HDF5 format for preprocessed embeddings

### Audio Processing
- **Whisper Integration**:
  - Model: Whisper Base
  - Input: 16kHz audio
  - Output: Text transcription
  - Pipeline: Real-time processing

### Training Pipeline
1. **Data Preparation**:
   - Image preprocessing with CLIP
   - Audio transcription with Whisper
   - Text tokenization

2. **Model Architecture**:
   ```python
   class MultimodalPhiModel(nn.Module):
       def __init__(self):
           self.phi_model = AutoModelForCausalLM.from_pretrained("microsoft/phi-1_5")
           self.clip_projection = nn.Linear(512, 768)  # CLIP to Phi
           self.whisper_projection = nn.Linear(256, 768)  # Whisper to Phi
   ```

3. **QLoRA Configuration**:
   ```python
   lora_config = LoraConfig(
       r=16,
       lora_alpha=32,
       target_modules=["query_key_value"],
       lora_dropout=0.05,
       bias="none",
       task_type="CAUSAL_LM"
   )
   ```

## Setup and Installation

1. **Environment Setup**:
   ```bash
   python -m venv venv
   source venv/bin/activate  # Linux/Mac
   pip install -r requirements.txt
   ```

2. **Data Preparation**:
   ```bash
   # Preprocess images and generate CLIP embeddings
   python scripts/preprocess_images.py --data_dir data/images --output_dir data/embeddings

   # Prepare dataset
   python scripts/prepare_data.py
   ```

3. **Training**:
   ```bash
   python train.py --config configs/training_config.json
   ```

4. **Web Interface**:
   ```bash
   cd web
   python app.py
   ```

## Training Details

### Dataset
- **Instruct 150K**:
  - 150,000 instruction-following examples
  - Mixed text and image inputs
  - Curated for quality and diversity

### Training Parameters
- Epochs: 10
- Batch Size: 16
- Learning Rate: 2e-5
- Weight Decay: 0.01
- Warmup Ratio: 0.1
- Gradient Accumulation: 4

## Deployment

### Web Interface
- **Framework**: Flask/Streamlit
- **Features**:
  - Text input field
  - Image upload/drag-drop
  - Audio recording/upload
  - Real-time processing
  - Chat-like interface

### API Endpoints 