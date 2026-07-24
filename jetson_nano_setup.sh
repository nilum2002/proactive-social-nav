#!/usr/bin/env bash
# ==============================================================================
# Jetson Nano Environment Setup Script
# Target Platform: NVIDIA Jetson Nano (JetPack 4.6 / Ubuntu 18.04 / Python 3.6 / ARM64)
# ==============================================================================

set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

echo "============================================================"
echo " Starting Jetson Nano Setup for Proactive Social Navigation"
echo "============================================================"

# Step 1: Install System Dependencies & Pre-compiled ARM64 Packages via APT
echo ""
echo "[1/6] Installing system prerequisites and pre-compiled packages via apt..."
sudo apt-get update
sudo apt-get install -y \
    libopenblas-base \
    libopenmpi-dev \
    libomp-dev \
    libfreetype6-dev \
    libpng-dev \
    python3-pip \
    python3-venv \
    python3-dev \
    python3-numpy \
    python3-scipy \
    python3-matplotlib \
    python3-yaml \
    python3-pil \
    python3-opencv \
    wget \
    curl \
    git

# Step 2: Create Python Virtual Environment with --system-site-packages
echo ""
echo "[2/6] Setting up Python virtual environment (with system-site-packages)..."
if [ -d "venv" ]; then
    echo "Recreating virtual environment to enable --system-site-packages..."
    rm -rf venv
fi

python3 -m venv --system-site-packages venv
echo "Created virtual environment in ./venv with system site packages enabled."

# Step 3: Activate Virtual Environment
echo ""
echo "[3/6] Activating virtual environment..."
source venv/bin/activate

# Upgrade pip inside venv
pip install --upgrade pip setuptools wheel
pip install Cython

# Step 4: Download and Install NVIDIA Pre-built PyTorch Wheel
PYTORCH_WHL="torch-1.10.0-cp36-cp36m-linux_aarch64.whl"
PYTORCH_URL="https://nvidia.box.com/shared/static/fjtbno0vpo676a25cgvuqc1wty0fkkg6.whl"

echo ""
echo "[4/6] Installing NVIDIA PyTorch 1.10.0 for JetPack 4.6 (aarch64)..."
if [ ! -f "$PYTORCH_WHL" ]; then
    echo "Downloading PyTorch wheel from NVIDIA..."
    wget -O "$PYTORCH_WHL" "$PYTORCH_URL"
else
    echo "Found existing wheel file $PYTORCH_WHL, skipping download."
fi

pip install "$PYTORCH_WHL"

# Step 5: Install Python Dependencies from req_jetson.txt
echo ""
echo "[5/6] Installing remaining Python dependencies..."
if [ -f "req_jetson.txt" ]; then
    pip install -r req_jetson.txt
else
    pip install tqdm scikit-learn tensorboardX python-lzf
fi

# Install local dr_spaam package
if [ -d "dr_spaam" ]; then
    echo "Installing dr_spaam package in editable mode..."
    pip install -e dr_spaam
fi

# Step 6: Verification
echo ""
echo "[6/6] Verifying PyTorch, Matplotlib, and CUDA availability..."
python3 -c "
import torch
import matplotlib
import scipy
import numpy

print('============================================================')
print('PyTorch Version    :', torch.__version__)
print('CUDA Available     :', torch.cuda.is_available())
if torch.cuda.is_available():
    print('Device Name        :', torch.cuda.get_device_name(0))
print('Matplotlib Version :', matplotlib.__version__)
print('NumPy Version      :', numpy.__version__)
print('SciPy Version      :', scipy.__version__)
print('============================================================')
"

echo ""
echo "Setup complete! To start working, activate the environment:"
echo "    source venv/bin/activate"
