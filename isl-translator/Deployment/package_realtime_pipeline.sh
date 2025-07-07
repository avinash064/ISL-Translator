#!/bin/bash

# Define environment and app name
ENV_NAME="isl-translator"
APP_NAME="realtime_pipeline"

echo "✅ Creating virtual environment..."
conda create -y -n $ENV_NAME python=3.10
conda activate $ENV_NAME

echo "📦 Installing dependencies..."
pip install torch torchvision torchaudio
pip install opencv-python numpy pyttsx3 tqdm

# Optional: install Coqui-TTS (if needed)
# pip install TTS

echo "🚀 Packaging done. To run:"
echo "conda activate $ENV_NAME"
echo "python -m inference.realtime_pipeline --model_ckpt checkpoints/multitask/model.pt"
echo "You can also run the app with:"
echo "python -m inference.realtime_pipeline --model_ckpt checkpoints/multitask/model.pt --app_name $APP_NAME"