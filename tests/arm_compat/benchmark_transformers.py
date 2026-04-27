import time
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM

MODEL_ID = "microsoft/bitnet-b1.58-2B-4T-bf16"
PROMPT = "Hello, how are you?"
MAX_NEW_TOKENS = 21
WARMUP_TOKENS = 5
WARMUP_RUNS = 2

tokenizer = AutoTokenizer.from_pretrained(MODEL_ID)
model = AutoModelForCausalLM.from_pretrained(
    MODEL_ID,
    dtype=torch.bfloat16,
)
model.eval()

device = model.device
inputs = tokenizer(PROMPT, return_tensors="pt")
inputs = {k: v.to(device) for k, v in inputs.items()}

for _ in range(WARMUP_RUNS):
    with torch.no_grad():
        _ = model.generate(
            **inputs,
            max_new_tokens=WARMUP_TOKENS,
            do_sample=False,
            pad_token_id=tokenizer.eos_token_id,
        )

start = time.time()
with torch.no_grad():
    out = model.generate(
        **inputs,
        max_new_tokens=MAX_NEW_TOKENS,
        do_sample=False,
        pad_token_id=tokenizer.eos_token_id,
    )
elapsed = time.time() - start

n_tokens = out.shape[1] - inputs["input_ids"].shape[1]
print(f"Generated {n_tokens} tokens in {elapsed:.2f}s ({n_tokens/elapsed:.1f} tok/s)")
print(tokenizer.decode(out[0], skip_special_tokens=True))
