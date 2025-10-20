#!/bin/bash

# Jet-Nemotron Performance Optimization Configuration
# Source this file to set optimal environment variables for inference

echo "Setting up performance optimizations for Jet-Nemotron..."

# CUDA optimizations for faster matrix operations
export TORCH_CUDNN_V8_API_ENABLED=1
export TORCH_ALLOW_TF32_CUBLAS_OVERRIDE=1

# Memory optimizations
export PYTORCH_CUDA_ALLOC_CONF=max_split_size_mb:512,expandable_segments:True

# Transformer optimizations
export TRANSFORMERS_CACHE=/tmp/huggingface_cache
export HF_DATASETS_CACHE=/tmp/datasets_cache

# Flash attention optimizations (if available)
export FLASH_ATTENTION_FORCE_REBUILD=1
export NVTE_ALLOW_NONDETERMINISTIC_ALGO=1

# Set optimal number of threads for CPU preprocessing
export OMP_NUM_THREADS=4
export TOKENIZERS_PARALLELISM=false

echo "Performance optimizations configured:"
echo "  ✓ TF32 enabled for faster matrix operations"
echo "  ✓ cuDNN optimizations enabled"
echo "  ✓ Memory allocation optimized"
echo "  ✓ CPU threading configured"
echo "  ✓ Caching directories set to /tmp for faster access"