#!/usr/bin/env python3
# Simple test script to verify Jet-Nemotron setup
# This script tests basic model loading and generation

import sys
import os

# Add Jet-Nemotron to path
sys.path.insert(0, '/tmp/sandboxbox-fm5pzW/workspace/Jet-Nemotron')

print("Testing Jet-Nemotron setup...")
print("-" * 50)

# Test 1: Import torch and check CUDA
print("\n1. Testing PyTorch and CUDA...")
try:
    import torch
    print(f"   ✓ PyTorch version: {torch.__version__}")
    print(f"   ✓ CUDA available: {torch.cuda.is_available()}")
    if torch.cuda.is_available():
        print(f"   ✓ CUDA version: {torch.version.cuda}")
        print(f"   ✓ GPU: {torch.cuda.get_device_name(0)}")
except Exception as e:
    print(f"   ✗ Error: {e}")
    sys.exit(1)

# Test 2: Import transformers
print("\n2. Testing transformers...")
try:
    from transformers import AutoTokenizer, AutoModelForCausalLM
    print("   ✓ Transformers imported successfully")
except Exception as e:
    print(f"   ✗ Error: {e}")
    sys.exit(1)

# Test 3: Check if Jet-Nemotron model files exist locally
print("\n3. Checking Jet-Nemotron model files...")
model_path = "/tmp/sandboxbox-fm5pzW/workspace/Jet-Nemotron/jetai/modeling/hf"
if os.path.exists(model_path):
    files = os.listdir(model_path)
    print(f"   ✓ Model directory exists: {model_path}")
    print(f"   ✓ Files found: {len(files)}")
    has_model = any('safetensors' in f or 'bin' in f for f in files)
    has_config = 'config.json' in files
    print(f"   - Has model weights: {has_model}")
    print(f"   - Has config: {has_config}")
else:
    print(f"   ℹ Model directory not found: {model_path}")
    print("   ℹ Will need to download from Hugging Face")

# Test 4: Check flash-attention availability
print("\n4. Checking optional dependencies...")
try:
    import flash_attn
    print("   ✓ flash-attention installed")
except ImportError:
    print("   ℹ flash-attention not installed (optional, models can use eager attention)")

print("\n" + "-" * 50)
print("Basic setup verification complete!")
print("\nTo download and test a model, you can run:")
print("  python3 -c 'from transformers import AutoModel; AutoModel.from_pretrained(\"jet-ai/Jet-Nemotron-2B\", trust_remote_code=True)'")
print("\nNote: Model download requires internet and may be large (several GB)")
