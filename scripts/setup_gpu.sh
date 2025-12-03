#!/bin/bash
set -e

echo "Updating system packages..."
sudo apt update -y
sudo apt upgrade -y

echo "Installing required packages..."
sudo apt install -y git python3 python3-venv python3-pip build-essential unzip wget libjpeg-dev zlib1g-dev

VENV_DIR=".venv"
if [ ! -d "$VENV_DIR" ]; then
    echo "Creating virtual environment at $VENV_DIR..."
    python3 -m venv $VENV_DIR
fi

echo "Activating virtual environment..."
source $VENV_DIR/bin/activate


echo "Cleaning existing packages in virtual environment..."
pip freeze | xargs -r pip uninstall -y || true


echo "Installing PyTorch..."
pip install --upgrade pip
pip install torch torchvision torchaudio

echo "Installing remaining Python packages..."
pip install -r requirements.txt

echo "Verifying installation..."
python3 - <<EOF
import torch, torchvision, ftfy, transformers, PIL, numpy
print("All modules loaded successfully!")
print("PyTorch CUDA available:", torch.cuda.is_available())
EOF

echo "Setup complete"