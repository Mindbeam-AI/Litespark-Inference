#!/usr/bin/env python3
"""Test HuggingFace BitNet model directly (no custom kernels)."""

import sys
from transformers import AutoModelForCausalLM, AutoTokenizer

# Try Microsoft's official model
MODEL = sys.argv[1] if len(sys.argv) > 1 else 'microsoft/bitnet-b1.58-2B-4T-bf16'

print(f"Loading model: {MODEL}")
model = AutoModelForCausalLM.from_pretrained(MODEL, trust_remote_code=True)
tokenizer = AutoTokenizer.from_pretrained(MODEL, trust_remote_code=True)

print("Generating...")
input_ids = tokenizer('The future of AI is', return_tensors='pt')['input_ids']
outputs = model.generate(input_ids, max_new_tokens=30, do_sample=False)
print(tokenizer.decode(outputs[0]))
