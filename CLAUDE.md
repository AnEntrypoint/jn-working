# Jet-Nemotron Local Setup Documentation

Technical documentation for setting up and running Jet-Nemotron hybrid-architecture language models locally.

## Project Overview

**Repository**: https://github.com/NVlabs/Jet-Nemotron

Jet-Nemotron is a family of hybrid-architecture language models combining full-attention and linear attention mechanisms. Key innovations:
- **PostNAS**: Post-training architecture exploration and adaptation pipeline
- **JetBlock**: Novel linear attention module with dynamic convolution
- **Performance**: Up to 53.6× speedup on H100 GPUs (256K context) vs alternatives

Models:
- Jet-Nemotron-2B: 2 billion parameters
- Jet-Nemotron-4B: 4 billion parameters

## System Requirements

### Minimum
- Python 3.10+
- NVIDIA GPU with CUDA support (RTX 3060+)
- 8GB+ GPU memory (for 2B model)
- 16GB+ system RAM
- Ubuntu/Linux (tested on WSL2)

### Recommended
- Python 3.12
- NVIDIA GPU with 12GB+ VRAM (for 4B model)
- CUDA 12.1+
- 32GB+ system RAM

## Installation

### Quick Setup

```bash
cd /path/to/workspace
bash setup_jet_nemotron.sh
```

### Manual Setup

1. **Clone Repository** (already done):
```bash
cd /tmp/sandboxbox-fm5pzW/workspace
ls Jet-Nemotron/  # Should exist
```

2. **Create Virtual Environment**:
```bash
cd Jet-Nemotron
python3 -m venv venv
source venv/bin/activate
```

3. **Install PyTorch with CUDA**:
```bash
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121
```

4. **Install Dependencies**:
```bash
pip install accelerate transformers datasets jieba fuzzywuzzy rouge python-Levenshtein
```

5. **Verify Setup**:
```bash
python3 ../test_jet_nemotron.py
```

## Project Structure

```
Jet-Nemotron/
├── jetai/
│   ├── modeling/
│   │   ├── hf/              # HuggingFace model implementations
│   │   │   ├── modeling_jet_nemotron.py
│   │   │   ├── configuration_jet_nemotron.py
│   │   │   ├── jet_block.py           # JetBlock implementation
│   │   │   ├── dynamic_conv.py        # Dynamic convolution
│   │   │   └── kv_cache.py            # KV cache utilities
│   │   └── configs/         # Model configurations (2B, 4B)
│   ├── evaluation/          # Evaluation scripts
│   └── utils/               # Utility functions
├── scripts/                 # Evaluation scripts (MMLU, BBH, etc.)
├── requirements.txt         # Python dependencies
├── pyproject.toml          # Package configuration
└── README.md
```

## Key Technical Details

### Architecture Components

**JetBlock** (`jetai/modeling/hf/jet_block.py`):
- Linear attention module with hardware-aware design
- Integrates dynamic convolution
- Efficient KV cache management
- Significantly outperforms Mamba2

**Model Configuration** (`jetai/modeling/hf/configuration_jet_nemotron.py`):
- Hybrid attention: mix of full-attention and linear-attention layers
- PostNAS-discovered optimal layer placement
- Custom generation configuration

### Dependencies

**Core**:
- torch (2.5.1+cu121): Deep learning framework
- transformers (4.53.0): HuggingFace transformers
- accelerate (1.10.1): Training/inference acceleration

**Model-specific**:
- flash-attention: Optional, requires CUDA toolkit (nvcc)
  - Models can use eager attention without it (slower)
  - For optimal performance, need to build from source

**Evaluation** (optional):
- datasets (4.2.0): Dataset loading
- lm-evaluation-harness: Benchmark evaluation
- flash-linear-attention: Custom linear attention kernels

### Known Issues & Workarounds

1. **flash-attention installation fails**:
   - **Issue**: Requires CUDA toolkit (nvcc) not available in standard Python envs
   - **Workaround**: Models can run without flash-attention using `attn_implementation="eager"`
   - **Future**: Install CUDA toolkit or use pre-built wheels

2. **CPU execution not supported**:
   - Models use CUDA-specific kernels
   - Must run on NVIDIA GPU

3. **Complex dependency tree**:
   - Full evaluation harness has many dependencies
   - Basic inference only needs core packages

## Usage Examples

### Basic Model Loading

```python
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM

model_name = "jet-ai/Jet-Nemotron-2B"

# Load model (downloads first time, ~2-4GB)
model = AutoModelForCausalLM.from_pretrained(
    model_name,
    trust_remote_code=True,  # Required for custom model code
    torch_dtype=torch.bfloat16,
    device_map="cuda"
)

tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)

# Generate text
input_text = "Hello, I'm Jet-Nemotron from NVIDIA."
inputs = tokenizer(input_text, return_tensors="pt").to("cuda")
outputs = model.generate(**inputs, max_new_tokens=50)
print(tokenizer.decode(outputs[0], skip_special_tokens=True))
```

### Using Local Model Files

```python
# First download model weights
# huggingface-cli download jet-ai/Jet-Nemotron-2B --local-dir jetai/modeling/hf

model = AutoModelForCausalLM.from_pretrained(
    "jetai/modeling/hf",  # Local path
    trust_remote_code=True,
    torch_dtype=torch.bfloat16,
    device_map="cuda"
)
```

### Building Custom JetBlock

```python
from jetai.modeling.hf.jet_block import JetBlock, JetBlockConfig

config = JetBlockConfig(
    expand_v=2.0,
    num_heads=6,
    head_dim=256,
    conv_size=4,
)

jet_block = JetBlock(
    hidden_size=1536,
    initializer_range=0.02,
    jet_block_config=config,
).cuda().to(torch.bfloat16)

# Use in forward pass
hidden_states = torch.randn(16, 4096, 1536).cuda().to(torch.bfloat16)
output, _ = jet_block(hidden_states=hidden_states)
```

## Performance Notes

- **Memory**: 2B model requires ~4GB VRAM in bfloat16
- **Speed**: Without flash-attention, inference is slower but functional
- **Context Length**: Supports up to 256K tokens (with sufficient memory)
- **Batch Size**: Adjust based on available VRAM

## Testing

Run verification script:
```bash
python3 test_jet_nemotron.py
```

Expected output:
- ✓ PyTorch and CUDA working
- ✓ Transformers imported
- ℹ flash-attention optional (models work without it)

## Troubleshooting

### "No module named 'jetai'"
- Solution: Either install package or add to PYTHONPATH:
  ```bash
  export PYTHONPATH=/path/to/Jet-Nemotron:$PYTHONPATH
  ```

### "CUDA out of memory"
- Reduce batch size
- Use smaller model (2B instead of 4B)
- Use torch.float16 or torch.bfloat16

### "trust_remote_code=True required"
- Jet-Nemotron uses custom model code
- Always set `trust_remote_code=True`

## Development Workflow

1. **Setup**: Run setup script or manual installation
2. **Test**: Verify with test_jet_nemotron.py
3. **Experiment**: Load models, test generation
4. **Evaluate**: Use scripts in `scripts/eval/` for benchmarks

## References

- **Paper**: https://www.arxiv.org/abs/2508.15884
- **Models**: https://huggingface.co/jet-ai/
- **Demo**: https://youtu.be/qAQ5yMThhRY
- **Website**: https://hanlab.mit.edu/projects/jet-nemotron/

## File Specifications

**setup_jet_nemotron.sh**: Automated setup script
- Creates venv, installs PyTorch, installs dependencies
- Runs verification tests

**test_jet_nemotron.py**: Verification script
- Tests PyTorch/CUDA, transformers
- Checks model files
- Reports flash-attention status

**CLAUDE.md** (this file): Technical documentation
- Setup instructions, architecture details
- Usage examples, troubleshooting
- Development notes for future work
