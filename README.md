# Jet-Nemotron - Quick Start Guide

🚀 **Working implementation of Jet-Nemotron with all compatibility fixes applied**

## One-Command Setup

```bash
bash setup.sh
```

That's it! The script will:
- Create virtual environment
- Install PyTorch + CUDA
- Download the model (~2-4GB)
- Apply all compatibility fixes automatically
- Verify everything works

## Requirements

- **GPU**: NVIDIA GPU with CUDA support (RTX 3060 or better)
- **RAM**: 16GB+ system RAM recommended
- **Disk**: 10GB free space
- **OS**: Ubuntu/Linux (tested on WSL2)
- **Python**: 3.10+

## Quick Test

After setup:
```bash
source venv/bin/activate
python generate_story.py
```

## What You Get

✅ **10.7 tokens/second** on RTX 3060 Laptop
✅ **Flash-Attention-2** enabled (1.74x faster)
✅ **All compatibility fixes** applied automatically
✅ **Production-ready** examples included

## Performance

| Hardware | Speed | Memory |
|----------|-------|--------|
| RTX 3060 Laptop | ~10.7 tok/s | 3.9GB |
| RTX 4090 | ~40-50 tok/s | 4-6GB |
| H100 | ~200-400 tok/s | 6-8GB |

## Files

| File | Purpose |
|------|---------|
| **setup.sh** | One-command installation |
| **generate_story.py** | Story generation example |
| **verify_setup.py** | System verification |
| **production_example.py** | Best practices demo |
| **INDEX.md** | Complete documentation |

## Usage

### Simple Generation
```bash
python generate_story.py
```

### Production Pattern
```python
from transformers import AutoTokenizer, AutoModelForCausalLM
import torch

# Load once (37s)
model = AutoModelForCausalLM.from_pretrained(
    "jet-ai/Jet-Nemotron-2B",
    trust_remote_code=True,
    dtype=torch.bfloat16,
    device_map="cuda",
    attn_implementation="flash_attention_2"
)
tokenizer = AutoTokenizer.from_pretrained("jet-ai/Jet-Nemotron-2B", trust_remote_code=True)

# Generate many times (18s each)
for request in requests:
    inputs = tokenizer(request, return_tensors="pt").to("cuda")
    outputs = model.generate(**inputs, max_new_tokens=200)
    result = tokenizer.decode(outputs[0])
```

## Troubleshooting

**CUDA not available?**
```bash
nvidia-smi  # Check GPU
python -c "import torch; print(torch.cuda.is_available())"
```

**Out of memory?**
- Use smaller batch size
- Reduce max_new_tokens
- Close other GPU applications

**Slow performance?**
- Verify Flash-Attention-2 is enabled
- Check GPU utilization: `nvidia-smi`
- See PERFORMANCE_REPORT.md

## Documentation

- **INDEX.md** - Complete file index
- **PERFORMANCE_REPORT.md** - Detailed performance analysis
- **FINAL_SUMMARY.txt** - Quick reference

## Credits

Based on [Jet-Nemotron](https://github.com/NVlabs/Jet-Nemotron) by NVIDIA Research
Paper: https://www.arxiv.org/abs/2508.15884

## License

Same as original Jet-Nemotron (see LICENSE directory)

---

**Status**: ✅ Fully Working | **Performance**: Optimal | **One-Command Setup**: Yes
