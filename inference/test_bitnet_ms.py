#!/usr/bin/env python3
"""
Test Microsoft BitNet b1.58-2B-4T with our ternary kernels.
"""

import torch
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from inference.ternary_models import load_ternary_model, get_arch_info

print("=" * 60)
print("Microsoft BitNet b1.58-2B-4T Test")
print("=" * 60)

# Architecture info
arch = get_arch_info()
print(f"Platform: {arch['machine']}")
print(f"Kernel: {arch['kernel_type']}")
print()

# Load model
print("Loading model...")
model, tokenizer = load_ternary_model('bitnet-2b')
model.eval()

print(f"\nModel config:")
print(f"  Hidden size: {model.config.hidden_size}")
print(f"  Layers: {model.config.num_layers}")
print(f"  Heads: {model.config.num_heads} (Q) / {model.config.num_kv_heads} (KV)")
print(f"  Intermediate: {model.config.intermediate_size}")
print(f"  Vocab: {model.config.vocab_size}")

# Test generation
print("\n" + "=" * 60)
print("Generation Test")
print("=" * 60)

prompts = [
    "The future of AI is",
    "Once upon a time",
    "The meaning of life is",
]

for prompt in prompts:
    input_ids = tokenizer.encode(prompt, return_tensors='pt')

    print(f"\nPrompt: {prompt}")
    print(f"Tokens: {input_ids.shape[1]}")

    # Generate
    generated = input_ids.clone()
    with torch.no_grad():
        for _ in range(20):
            logits = model(generated)
            next_token = logits[0, -1, :].argmax(keepdim=True)
            generated = torch.cat([generated, next_token.unsqueeze(0)], dim=1)

    output = tokenizer.decode(generated[0], skip_special_tokens=True)
    print(f"Output: {output}")

print("\n" + "=" * 60)
print("Test complete!")
