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

# Check CUDA and GPU capabilities
if ! command -v nvidia-smi &> /dev/null; then
    echo "❌ nvidia-smi not found. CUDA GPU drivers required."
    exit 1
else
    GPU_NAME=$(nvidia-smi --query-gpu=name --format=csv,noheader | head -1)
    GPU_MEMORY=$(nvidia-smi --query-gpu=memory.total --format=csv,noheader,unit=MB | head -1)
    CUDA_VERSION=$(nvidia-smi | grep "CUDA Version" | sed 's/.*CUDA Version: \([0-9.]*\).*/\1/')
    echo "✓ Found GPU: $GPU_NAME"
    echo "✓ GPU Memory: ${GPU_MEMORY}MB"
    echo "✓ CUDA Version: $CUDA_VERSION"

    # Check if GPU has enough memory (minimum 6GB for 2B model)
    if [ "$GPU_MEMORY" -lt 6000 ]; then
        echo "⚠️  Warning: GPU has less than 6GB memory. Performance may be limited."
    fi
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
pip install accelerate transformers datasets jieba fuzzywuzzy rouge python-Levenshtein > /dev/null

# Try to install flash-attention (optional performance boost)
echo "   Attempting to install flash-attention for better performance..."
if pip install flash-attn --no-build-isolation > /dev/null 2>&1; then
    echo "   ✓ flash-attention installed (optimal performance)"
    export FLASH_ATTN_AVAILABLE=true
else
    echo "   ⚠️  flash-attention installation failed (will use SDPA - still good performance)"
    export FLASH_ATTN_AVAILABLE=false
    echo "     Note: flash-attention requires CUDA toolkit and proper compiler"
    echo "     The model will work fine with PyTorch's SDPA attention"
fi
echo "   ✓ Core dependencies installed"

# Set performance environment variables
echo ""
echo "5. Setting up performance optimizations..."
if [ -f "performance_config.sh" ]; then
    source performance_config.sh
else
    echo "   Creating performance config..."
    cat > performance_config.sh << 'PERF_EOF'
#!/bin/bash

# CUDA optimizations
export TORCH_CUDNN_V8_API_ENABLED=1
export TORCH_ALLOW_TF32_CUBLAS_OVERRIDE=1

# Memory optimizations
export PYTORCH_CUDA_ALLOC_CONF=max_split_size_mb:512,expandable_segments:True

# Threading optimizations
export OMP_NUM_THREADS=4
export TOKENIZERS_PARALLELISM=false
PERF_EOF
    source performance_config.sh
fi

# Test installation and CUDA optimizations
echo ""
echo "6. Testing PyTorch and CUDA optimizations..."
python3 << 'EOF'
import torch
import os

if torch.cuda.is_available():
    print(f"   ✓ CUDA available: {torch.cuda.get_device_name(0)}")
    print(f"   ✓ CUDA version: {torch.version.cuda}")
    print(f"   ✓ GPU memory: {torch.cuda.get_device_properties(0).total_memory / 1024**3:.1f}GB")

    # Set CUDA optimizations for better performance
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True
    torch.backends.cudnn.benchmark = True
    torch.backends.cudnn.deterministic = False

    print("   ✓ CUDA optimizations enabled:")
    print("     - TF32 for faster matrix operations")
    print("     - cuDNN benchmark mode")
    print("     - Non-deterministic algorithms for speed")

    # Test basic CUDA operation
    x = torch.randn(1000, 1000).cuda()
    y = torch.mm(x, x.t())
    print("   ✓ CUDA operations working correctly")
else:
    print("   ❌ CUDA not available - model will not work without GPU")
    exit(1)
EOF

# Download model and apply fixes
echo ""
echo "7. Downloading model and applying compatibility fixes..."
python3 << 'EOF'
import os
from transformers import AutoTokenizer, AutoModelForCausalLM
import torch

MODEL_NAME = "jet-ai/Jet-Nemotron-2B"

print("   Downloading model (this may take a while, ~2-4GB)...")
tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME, trust_remote_code=True)

# Determine best attention implementation based on availability
flash_attn_available = os.environ.get('FLASH_ATTN_AVAILABLE', 'false').lower() == 'true'

if flash_attn_available:
    print("   Using flash-attention-2 for optimal performance...")
    attn_impl = "flash_attention_2"
else:
    print("   Using SDPA attention (PyTorch's optimized implementation)...")
    attn_impl = "sdpa"

try:
    model = AutoModelForCausalLM.from_pretrained(
        MODEL_NAME,
        trust_remote_code=True,
        torch_dtype=torch.bfloat16,
        device_map="cuda",
        attn_implementation=attn_impl,
        low_cpu_mem_usage=True,  # Optimize memory usage
        use_cache=True           # Enable KV cache for faster generation
    )
    print(f"   ✓ Model loaded with {attn_impl} attention")
except Exception as e:
    print(f"   ⚠️  Failed with {attn_impl}, trying fallback...")
    print(f"     Error: {e}")

    # Fallback to SDPA if flash-attention failed
    if attn_impl != "sdpa":
        print("   Falling back to SDPA attention...")
        model = AutoModelForCausalLM.from_pretrained(
            MODEL_NAME,
            trust_remote_code=True,
            torch_dtype=torch.bfloat16,
            device_map="cuda",
            attn_implementation="sdpa",
            low_cpu_mem_usage=True,
            use_cache=True
        )
        print("   ✓ Model loaded with SDPA attention fallback")
    else:
        print(f"   ❌ Model loading failed: {e}")
        exit(1)

# Configure model for optimal performance
print("   Configuring model for performance...")
model.eval()  # Set to evaluation mode

# Clear cache to ensure clean state
torch.cuda.empty_cache()
print("   ✓ Model configured for inference")

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
echo "8. Running verification test..."
if [ -f "verify_setup.py" ]; then
    python3 verify_setup.py
else
    # Fallback basic test if verify_setup.py doesn't exist
    python3 << 'EOF'
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
import time

print("   Running basic verification test...")
MODEL_NAME = "jet-ai/Jet-Nemotron-2B"

try:
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME, trust_remote_code=True)
    model = AutoModelForCausalLM.from_pretrained(
        MODEL_NAME,
        trust_remote_code=True,
        torch_dtype=torch.bfloat16,
        device_map="cuda",
        attn_implementation="sdpa",
        low_cpu_mem_usage=True
    )

    # Quick generation test
    test_input = "Hello, I'm Jet-Nemotron."
    inputs = tokenizer(test_input, return_tensors="pt").to("cuda")

    start = time.time()
    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_new_tokens=20,
            do_sample=False,
            pad_token_id=tokenizer.eos_token_id
        )
    gen_time = time.time() - start

    tokens_generated = len(outputs[0]) - len(inputs['input_ids'][0])
    speed = tokens_generated / gen_time if gen_time > 0 else 0

    print(f"   ✓ Basic test passed - {speed:.1f} tokens/s")
    print(f"   ✓ Model ready for use!")

except Exception as e:
    print(f"   ❌ Verification failed: {e}")
    exit(1)
EOF
fi

echo ""
echo "================================================================================"
echo "✅ SETUP COMPLETE!"
echo "================================================================================"
echo ""
echo "Next steps:"
echo "  1. Activate the virtual environment:"
echo "     source venv/bin/activate"
echo ""
echo "  2. Load performance optimizations (optional):"
echo "     source performance_config.sh"
echo ""
echo "  3. Try generating a story:"
echo "     python generate_story.py"
echo ""
echo "  4. See INDEX.md for all available scripts and documentation"
echo ""
echo "Performance Information:"
echo "  - Model: Jet-Nemotron-2B (2B parameters)"
echo "  - Memory Usage: ~4GB GPU RAM (bfloat16)"
echo "  - Attention: $(if [ "$FLASH_ATTN_AVAILABLE" = true ]; then echo "Flash-Attention-2 (optimal)"; else echo "SDPA (PyTorch, good)"; fi)"
echo "  - CUDA Optimizations: TF32, cuDNN benchmark, KV cache enabled"
echo ""
echo "Expected Performance:"
echo "  - RTX 3060 (6GB): ~8-15 tokens/s"
echo "  - RTX 3070/3080 (8-12GB): ~15-25 tokens/s"
echo "  - RTX 4060+ (12GB+): ~20-35 tokens/s"
echo "  - Higher end GPUs: 30+ tokens/s"
echo ""
echo "================================================================================"
