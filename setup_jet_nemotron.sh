#!/bin/bash
# Jet-Nemotron Local Setup Script
# This script sets up Jet-Nemotron for local development and testing

set -e  # Exit on error

echo "=================================================="
echo "  Jet-Nemotron Local Setup"
echo "=================================================="
echo ""

# Check Python version
echo "[1/6] Checking Python version..."
PYTHON_VERSION=$(python3 --version 2>&1 | awk '{print $2}')
echo "  Found Python $PYTHON_VERSION"
if ! python3 -c 'import sys; sys.exit(0 if sys.version_info >= (3, 10) else 1)'; then
    echo "  ERROR: Python 3.10+ is required"
    exit 1
fi
echo "  ✓ Python version OK"
echo ""

# Check NVIDIA GPU
echo "[2/6] Checking for NVIDIA GPU..."
if ! command -v nvidia-smi &> /dev/null; then
    echo "  WARNING: nvidia-smi not found. GPU may not be available."
else
    nvidia-smi --query-gpu=name,driver_version,memory.total --format=csv,noheader
    echo "  ✓ NVIDIA GPU detected"
fi
echo ""

# Navigate to Jet-Nemotron directory
SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"
JET_DIR="$SCRIPT_DIR/Jet-Nemotron"

if [ ! -d "$JET_DIR" ]; then
    echo "  ERROR: Jet-Nemotron directory not found at $JET_DIR"
    exit 1
fi

cd "$JET_DIR"

# Create virtual environment
echo "[3/6] Creating Python virtual environment..."
if [ ! -d "venv" ]; then
    python3 -m venv venv
    echo "  ✓ Virtual environment created"
else
    echo "  ℹ Virtual environment already exists"
fi
echo ""

# Activate virtual environment
echo "[4/6] Activating virtual environment..."
source venv/bin/activate
echo "  ✓ Virtual environment activated"
echo ""

# Upgrade pip
echo "[5/6] Upgrading pip and installing dependencies..."
pip install --upgrade pip setuptools wheel

# Install PyTorch with CUDA support
echo "  Installing PyTorch with CUDA 12.1..."
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121

# Install basic dependencies
echo "  Installing Jet-Nemotron dependencies..."
pip install accelerate transformers datasets jieba fuzzywuzzy rouge python-Levenshtein

echo "  ✓ Dependencies installed"
echo ""

# Test installation
echo "[6/6] Testing installation..."
python3 << 'PYTHON_TEST'
import torch
print(f"  PyTorch version: {torch.__version__}")
print(f"  CUDA available: {torch.cuda.is_available()}")
if torch.cuda.is_available():
    print(f"  CUDA version: {torch.version.cuda}")
    print(f"  GPU: {torch.cuda.get_device_name(0)}")

from transformers import AutoTokenizer
print(f"  Transformers: OK")
PYTHON_TEST

echo ""
echo "=================================================="
echo "  Setup Complete!"
echo "=================================================="
echo ""
echo "Next steps:"
echo "  1. Activate the virtual environment:"
echo "     cd $JET_DIR"
echo "     source venv/bin/activate"
echo ""
echo "  2. Test with a simple example:"
echo "     python3 ../test_jet_nemotron.py"
echo ""
echo "  3. Download and test a model (requires internet):"
echo "     python3 -c 'from transformers import AutoModelForCausalLM; AutoModelForCausalLM.from_pretrained(\"jet-ai/Jet-Nemotron-2B\", trust_remote_code=True, torch_dtype=\"auto\", device_map=\"cuda\")'"
echo ""
echo "Note: flash-attention requires CUDA toolkit (nvcc) to build."
echo "      Models can run without it using eager attention (slower)."
echo ""
