"""
Smoke test for the torchless BitNet forward pass.

Loads bitnet-2b, feeds a tiny synthetic token sequence, and checks:
  - forward_one returns finite logits of the right shape
  - different input tokens yield different logits (not a constant output)
  - generate() runs end-to-end for a few tokens without NaNs

No tokenizer yet, so token IDs are hand-picked. Semantic quality of the
greedy continuation is not asserted here -- that comes once the tokenizer
swap lands.
"""

from __future__ import annotations

import sys
import time

import numpy as np


def main() -> int:
    from litespark_inference.torchless import load_bitnet_2b
    from litespark_inference.torchless.runtime import forward_one, generate, init_state

    if "torch" in sys.modules:
        raise RuntimeError("torch imported - torchless invariant violated")

    embed_dtype = "int4" if "--int4" in sys.argv else ("int8" if "--int8" in sys.argv else "bf16")
    print(f"Loading model (embed_dtype={embed_dtype})...")
    t0 = time.perf_counter()
    model = load_bitnet_2b(embed_dtype=embed_dtype)
    print(f"  loaded in {time.perf_counter() - t0:.1f}s", flush=True)

    c = model.config
    assert c.hidden_size == 2560
    assert c.num_layers == 30

    # A tiny synthetic prompt: BOS-like token + three others.
    prompt = [1, 100, 200, 300]
    state = init_state(model, t_max=len(prompt) + 4)

    print("Prefill + one forward...")
    t0 = time.perf_counter()
    logits = None
    for tid in prompt:
        logits = forward_one(model, state, tid)
    elapsed = time.perf_counter() - t0
    print(f"  {len(prompt)} tokens in {elapsed:.2f}s "
          f"= {len(prompt)/elapsed:.3f} tok/s", flush=True)

    assert logits is not None
    assert logits.shape == (c.vocab_size,), logits.shape
    assert np.isfinite(logits).all(), "non-finite logits"
    print(f"  logits shape OK: {logits.shape}")
    print(f"  logits stats: min={logits.min():.3f} max={logits.max():.3f} "
          f"mean={logits.mean():.3f} std={logits.std():.3f}")

    # Different inputs should produce different outputs
    state2 = init_state(model, t_max=len(prompt) + 4)
    logits_alt = None
    for tid in [1, 500, 600, 700]:
        logits_alt = forward_one(model, state2, tid)
    diff = float(np.abs(logits - logits_alt).max())
    print(f"  max|logits - logits_alt| = {diff:.3f}")
    assert diff > 0.01, "logits identical across different prompts - kernel bug?"

    # End-to-end generate (greedy, 3 new tokens)
    print("Greedy generate (3 tokens)...")
    t0 = time.perf_counter()
    out = generate(model, prompt, max_new_tokens=3)
    elapsed = time.perf_counter() - t0
    print(f"  generated {out} in {elapsed:.2f}s "
          f"(prefill={len(prompt)}, gen=3, tok/s={(len(prompt)+3)/elapsed:.3f})")
    assert len(out) == 3
    assert all(0 <= t < c.vocab_size for t in out)

    print("OK")
    return 0


if __name__ == "__main__":
    sys.exit(main())
