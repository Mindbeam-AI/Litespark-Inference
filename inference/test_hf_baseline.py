#!/usr/bin/env python3
"""
Test HuggingFace BitNet model directly (no custom kernels).

For Microsoft model, first install the required transformers fork:
  uv pip install git+https://github.com/huggingface/transformers.git@096f25ae1f501a084d8ff2dcaf25fbc2bd60eba4
"""

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

MODEL = 'microsoft/bitnet-b1.58-2B-4T'

print(f"Loading model: {MODEL}")
tokenizer = AutoTokenizer.from_pretrained(MODEL)
model = AutoModelForCausalLM.from_pretrained(MODEL, torch_dtype=torch.bfloat16)

print("Generating...")
input_ids = tokenizer('The future of AI is', return_tensors='pt')['input_ids']
outputs = model.generate(input_ids, max_new_tokens=30, do_sample=False)
print(tokenizer.decode(outputs[0]))
