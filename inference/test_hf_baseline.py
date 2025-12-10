#!/usr/bin/env python3
"""
Test HuggingFace BitNet model structure.
"""

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, AutoConfig

MODEL = 'microsoft/bitnet-b1.58-2B-4T-bf16'

print(f"Loading config: {MODEL}")
config = AutoConfig.from_pretrained(MODEL, trust_remote_code=True)
print(f"Config: {config}")

print(f"\nLoading model: {MODEL}")
model = AutoModelForCausalLM.from_pretrained(MODEL, trust_remote_code=True, torch_dtype=torch.bfloat16)

print("\nModel structure:")
print(f"  Layers: {len(model.model.layers)}")
layer = model.model.layers[0]
print(f"  Layer 0 children: {list(layer.named_children())}")
print(f"  Attention: {layer.self_attn}")
print(f"  MLP: {layer.mlp}")

# Check weight shapes
print("\nWeight shapes:")
print(f"  q_proj: {layer.self_attn.q_proj.weight.shape}")
print(f"  k_proj: {layer.self_attn.k_proj.weight.shape}")
print(f"  v_proj: {layer.self_attn.v_proj.weight.shape}")
print(f"  gate_proj: {layer.mlp.gate_proj.weight.shape}")
print(f"  up_proj: {layer.mlp.up_proj.weight.shape}")
print(f"  down_proj: {layer.mlp.down_proj.weight.shape}")

# Test generation
print("\nTesting generation...")
tokenizer = AutoTokenizer.from_pretrained(MODEL, trust_remote_code=True)
input_ids = tokenizer('The future of AI is', return_tensors='pt')['input_ids']
outputs = model.generate(input_ids, max_new_tokens=30, do_sample=False)
print(tokenizer.decode(outputs[0]))
