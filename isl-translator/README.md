# ISL Translator

The ISL Translator project is designed to translate Indian Sign Language (ISL) into text using advanced machine learning techniques. This project leverages keypoint extraction and multilingual decoding to facilitate real-time translation.

## Setup Instructions

To set up the ISL Translator project, follow these steps:

1. **Clone the repository**:
   ```bash
   git clone <repository-url>
   cd isl-translator
   ```

2. **Run the setup script**:
   The provided Bash script will create a virtual environment and install all necessary dependencies. Execute the following command:
   ```bash
   bash Deployment/package_realtime_pipeline.sh
   ```

## Deployment Script

The `package_realtime_pipeline.sh` script includes the following commands:

```bash
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
```

## Usage

After setting up the environment, you can run the application using the following command:

```bash
python -m inference.realtime_pipeline --model_ckpt checkpoints/multitask/model.pt
```

You can also specify an application name:

```bash
python -m inference.realtime_pipeline --model_ckpt checkpoints/multitask/model.pt --app_name $APP_NAME
```

## Contributing

Contributions are welcome! Please feel free to submit a pull request or open an issue for any suggestions or improvements.

## License

This project is licensed under the MIT License. See the LICENSE file for details.