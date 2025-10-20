#!/bin/bash

# Jet-Nemotron API startup script

echo "🚀 Starting Jet-Nemotron API..."
echo "================================"

# Check if CUDA is available
python3 -c "import torch; assert torch.cuda.is_available(), 'CUDA not available'; print(f'✓ CUDA available: {torch.cuda.get_device_name(0)}')" || {
    echo "❌ CUDA not available. Jet-Nemotron requires NVIDIA GPU."
    exit 1
}

# Check if model is accessible
echo "📦 Checking model access..."
python3 -c "
from transformers import AutoTokenizer
try:
    tokenizer = AutoTokenizer.from_pretrained('jet-ai/Jet-Nemotron-2B', trust_remote_code=True)
    print('✓ Model accessible')
except Exception as e:
    print(f'❌ Model access failed: {e}')
    print('Note: Model will be downloaded on first startup (~2-4GB)')
"

echo ""
echo "⚡ Starting API server..."
echo "   - Model will take 30-40 seconds to load"
echo "   - Server will be available at http://localhost:8000"
echo "   - API docs at http://localhost:8000/docs"
echo ""

# Start the API
python3 api.py