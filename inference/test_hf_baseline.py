#!/usr/bin/env python3
"""Test HuggingFace BitNet model directly (no custom kernels)."""

from transformers import AutoModelForCausalLM, PreTrainedTokenizerFast

print("Loading model...")
model = AutoModelForCausalLM.from_pretrained('1bitLLM/bitnet_b1_58-3B', trust_remote_code=True)
tokenizer = PreTrainedTokenizerFast.from_pretrained('1bitLLM/bitnet_b1_58-3B')

print("Generating...")
inputs = tokenizer('The future of AI is', return_tensors='pt')
outputs = model.generate(**inputs, max_new_tokens=30)
print(tokenizer.decode(outputs[0]))
