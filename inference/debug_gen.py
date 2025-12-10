#!/usr/bin/env python3
"""Debug generation token by token."""
import torch
from ternary_models import load_ternary_model

model, tokenizer = load_ternary_model('bitnet-2b')
model.eval()

prompt = 'The'
input_ids = tokenizer.encode(prompt, return_tensors='pt')

generated = input_ids.clone()
print(f'Prompt: {prompt}')
for i in range(30):
    with torch.no_grad():
        logits = model(generated)
        next_token = logits[0, -1, :].argmax()
        tok_str = tokenizer.decode([next_token.item()])
        print(f'  {i}: token={next_token.item()} = {repr(tok_str)}')
        generated = torch.cat([generated, next_token.unsqueeze(0).unsqueeze(0)], dim=1)
