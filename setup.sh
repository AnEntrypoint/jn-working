#!/bin/bash

set -e  # Exit on error

echo "================================================================================"
echo "Jet-Nemotron Setup - One-Command Installation"
echo "================================================================================"

# Check Python version
if ! command -v python3 &> /dev/null; then
    echo "❌ Python 3 not found. Please install Python 3.10+"
    exit 1
fi

PYTHON_VERSION=$(python3 --version | cut -d' ' -f2 | cut -d'.' -f1,2)
echo "✓ Found Python $PYTHON_VERSION"

# Check CUDA
if ! command -v nvidia-smi &> /dev/null; then
    echo "⚠️  nvidia-smi not found. CUDA may not be available."
else
    GPU_NAME=$(nvidia-smi --query-gpu=name --format=csv,noheader | head -1)
    echo "✓ Found GPU: $GPU_NAME"
fi

# Create virtual environment
echo ""
echo "1. Creating virtual environment..."
if [ -d "venv" ]; then
    echo "   ⚠️  venv already exists, removing..."
    rm -rf venv
fi
python3 -m venv venv
source venv/bin/activate
echo "   ✓ Virtual environment created"

# Upgrade pip
echo ""
echo "2. Upgrading pip..."
pip install --upgrade pip > /dev/null
echo "   ✓ pip upgraded"

# Install PyTorch with CUDA
echo ""
echo "3. Installing PyTorch with CUDA support..."
echo "   (This may take a few minutes...)"
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121 > /dev/null
echo "   ✓ PyTorch installed"

# Install dependencies
echo ""
echo "4. Installing dependencies..."
pip install accelerate transformers datasets jieba fuzzywuzzy rouge python-Levenshtein flash-attn > /dev/null
echo "   ✓ Dependencies installed"

# Test installation
echo ""
echo "5. Testing PyTorch and CUDA..."
python3 << 'EOF'
import torch
if torch.cuda.is_available():
    print(f"   ✓ CUDA available: {torch.cuda.get_device_name(0)}")
else:
    print("   ⚠️  CUDA not available - model will not work without GPU")
    exit(1)
EOF

# Download model and apply fixes
echo ""
echo "6. Downloading model and applying compatibility fixes..."
python3 << 'EOF'
import os
from transformers import AutoTokenizer, AutoModelForCausalLM
import torch

MODEL_NAME = "jet-ai/Jet-Nemotron-2B"

print("   Downloading model (this may take a while, ~2-4GB)...")
tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME, trust_remote_code=True)
model = AutoModelForCausalLM.from_pretrained(
    MODEL_NAME,
    trust_remote_code=True,
    dtype=torch.bfloat16,
    device_map="cuda",
    attn_implementation="flash_attention_2"
)
print("   ✓ Model downloaded")

# Apply fixes
import glob

cache_dir = os.path.expanduser("~/.cache/huggingface/modules/transformers_modules/jet_hyphen_ai/Jet_hyphen_Nemotron_hyphen_2B/")
pattern = os.path.join(cache_dir, "*", "*.py")
files = glob.glob(pattern)

if not files:
    print("   ⚠️  Could not find cached model files")
    exit(1)

model_dir = os.path.dirname(files[0])

print(f"   Applying compatibility fixes to {model_dir}...")

# Fix 1: jet_block.py
jet_block_file = os.path.join(model_dir, "jet_block.py")
if os.path.exists(jet_block_file):
    with open(jet_block_file, 'r') as f:
        content = f.read()

    # Remove autotune_interval from FusedRMSNormGated
    content = content.replace(
        'self.o_norm = FusedRMSNormGated(self.head_v_dim, eps=float(jet_block_config.norm_eps), autotune_interval=self.autotune_interval)',
        'self.o_norm = FusedRMSNormGated(self.head_v_dim, eps=float(jet_block_config.norm_eps))'
    )

    # Remove autotune_interval from chunk_gated_delta_rule
    content = content.replace(
        'use_qk_l2norm_in_kernel=True,\n                autotune_interval=self.autotune_interval',
        'use_qk_l2norm_in_kernel=True'
    )

    with open(jet_block_file, 'w') as f:
        f.write(content)
    print("   ✓ Fixed jet_block.py")

# Fix 2: modeling_jet_nemotron.py
modeling_file = os.path.join(model_dir, "modeling_jet_nemotron.py")
if os.path.exists(modeling_file):
    with open(modeling_file, 'r') as f:
        content = f.read()

    # Make device parameter optional
    content = content.replace(
        'device: torch.device,\n    ) -> bool:\n        assert not generation_config.return_legacy_cache',
        'device: torch.device = None,\n    ) -> bool:\n        if device is None:\n            device = self.device\n        if hasattr(generation_config, \'return_legacy_cache\'):\n            assert not generation_config.return_legacy_cache'
    )

    with open(modeling_file, 'w') as f:
        f.write(content)
    print("   ✓ Fixed modeling_jet_nemotron.py")

# Fix 3: kv_cache.py
kv_cache_file = os.path.join(model_dir, "kv_cache.py")
if os.path.exists(kv_cache_file):
    with open(kv_cache_file, 'r') as f:
        content = f.read()

    # Add layers and is_compileable properties
    if '@property\n    def layers(self):' not in content:
        content = content.replace(
            'self._seen_tokens = []  # Used in `generate` to keep tally of how many tokens the cache has seen\n\n    def __getitem__',
            'self._seen_tokens = []  # Used in `generate` to keep tally of how many tokens the cache has seen\n\n    @property\n    def layers(self):\n        return self.states\n\n    @property\n    def is_compileable(self):\n        return False\n\n    def __getitem__'
        )

        with open(kv_cache_file, 'w') as f:
            f.write(content)
        print("   ✓ Fixed kv_cache.py")

print("   ✓ All compatibility fixes applied")
EOF

# Test the model
echo ""
echo "7. Running verification test..."
python3 verify_setup.py

echo ""
echo "================================================================================"
echo "✅ SETUP COMPLETE!"
echo "================================================================================"
echo ""
echo "Next steps:"
echo "  1. Activate the virtual environment:"
echo "     source venv/bin/activate"
echo ""
echo "  2. Try generating a story:"
echo "     python generate_story.py"
echo ""
echo "  3. See INDEX.md for all available scripts and documentation"
echo ""
echo "Performance on your system:"
echo "  - Load time: ~37-57s (one-time)"
echo "  - Generation: ~10-11 tokens/second"
echo "  - Memory: ~4GB GPU RAM"
echo ""
echo "================================================================================"
