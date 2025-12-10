#!/usr/bin/env python3
"""
Test HuggingFace BitNet model structure by loading safetensors directly.
"""

import torch
from safetensors import safe_open
from huggingface_hub import hf_hub_download
import json

MODEL = 'microsoft/bitnet-b1.58-2B-4T-bf16'

# Download and read config
print(f"Downloading config from {MODEL}...")
config_path = hf_hub_download(MODEL, 'config.json')
with open(config_path) as f:
    config = json.load(f)
print(f"Config: {json.dumps(config, indent=2)}")

# Download safetensors
print(f"\nDownloading model weights...")
model_path = hf_hub_download(MODEL, 'model.safetensors')

# Inspect weight names and shapes
print("\nWeight tensors:")
with safe_open(model_path, framework="pt") as f:
    keys = list(f.keys())
    print(f"Total tensors: {len(keys)}")

    # Show first layer structure
    layer0_keys = [k for k in keys if 'layers.0.' in k]
    print(f"\nLayer 0 tensors:")
    for k in sorted(layer0_keys):
        tensor = f.get_tensor(k)
        print(f"  {k}: {tensor.shape} {tensor.dtype}")

    # Check a weight's value range (to see if it's already quantized)
    print("\nSample weight stats (q_proj):")
    w = f.get_tensor('model.layers.0.self_attn.q_proj.weight')
    print(f"  Shape: {w.shape}")
    print(f"  Dtype: {w.dtype}")
    print(f"  Min: {w.min().item():.6f}")
    print(f"  Max: {w.max().item():.6f}")
    print(f"  Mean: {w.mean().item():.6f}")
    print(f"  Unique values (sample): {torch.unique(w[:100,:100]).numel()}")
