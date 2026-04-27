"""
Semantic correctness check: torchless logits vs the torch-backed reference.

Loads the torch-hosted BitNet and the torchless BitNet in the same process,
feeds both the same prompt token IDs, and compares the final-position logits
that each produces.

Because the torchless invariant check in runtime.py guards against *auto*
imports of torch through the torchless path, explicitly importing torch
here is fine: this script is not one of the ones that asserts torchlessness.

What we check:
  - argmax(torch_logits) == argmax(torchless_logits) on the last position
  - top-k overlap at k in {1, 5, 20}
  - per-element max|diff| and relative error over the full logit vector

Expectations:
  - Top-1 should match. Tiny deviations in the tails are fine as long as
    the decoded top-1 stays stable.
  - Full-vector max|diff| may be several-to-tens of units because the
    two runtimes use slightly different activation quantization details
    (int8 absmax for torchless M=1 vs the torch per-row path) and
    different reduction orders. Relative error < ~0.05 is healthy.
"""

from __future__ import annotations

import os
import sys
import time

import numpy as np

os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")

# Prompts to evaluate. Intentionally short so the prefill cost stays low.
PROMPTS = [
    "Hello, how are you?",
    "The quick brown fox",
]


def _topk_overlap(a: np.ndarray, b: np.ndarray, k: int) -> float:
    ka = set(np.argpartition(-a, k - 1)[:k].tolist())
    kb = set(np.argpartition(-b, k - 1)[:k].tolist())
    return len(ka & kb) / k


def _diff_stats(a: np.ndarray, b: np.ndarray) -> tuple[float, float]:
    d = np.abs(a - b)
    scale = np.abs(a).max() + 1e-9
    return float(d.max()), float(d.max() / scale)


def run_torch(prompt_token_ids: list[int]) -> np.ndarray:
    """Return [vocab] fp32 logits from the torch-backed model at the last pos."""
    import torch
    from litespark_inference.models import load_ternary_model

    model, _tok = load_ternary_model("bitnet-2b", mode="neon")
    model.eval()
    input_ids = torch.tensor([prompt_token_ids], dtype=torch.long)
    with torch.no_grad():
        out = model(input_ids)
    # BitNet forward returns logits [B, T, V] (or a tuple containing them).
    if isinstance(out, torch.Tensor):
        logits = out
    elif hasattr(out, "logits"):
        logits = out.logits
    else:
        logits = out[0]
    return logits[0, -1, :].to(torch.float32).cpu().numpy()


def run_torchless(
    prompt_token_ids: list[int], embed_dtype: str = "int8",
) -> np.ndarray:
    """Return [vocab] fp32 logits from the torchless runtime at the last pos."""
    from litespark_inference.torchless import load_bitnet_2b
    from litespark_inference.torchless.runtime import forward_one, init_state

    model = load_bitnet_2b(embed_dtype=embed_dtype)
    state = init_state(model, t_max=len(prompt_token_ids) + 1)
    logits = None
    for tid in prompt_token_ids:
        logits = forward_one(model, state, int(tid))
    assert logits is not None
    return logits


def main() -> int:
    import argparse
    p = argparse.ArgumentParser()
    p.add_argument("--embed-dtype", choices=["bf16", "int8"], default="int8")
    args = p.parse_args()

    # Tokenize with the torch-side tokenizer so both runtimes see the same IDs.
    from transformers import AutoTokenizer
    tok = AutoTokenizer.from_pretrained("microsoft/bitnet-b1.58-2B-4T-bf16")

    all_ok = True
    for prompt in PROMPTS:
        token_ids = tok.encode(prompt)
        print(f"\nPROMPT: {prompt!r}  (torchless embed_dtype={args.embed_dtype})")
        print(f"  tokens ({len(token_ids)}): {token_ids}")

        t0 = time.perf_counter()
        torch_logits = run_torch(token_ids)
        t_torch = time.perf_counter() - t0
        print(f"  torch forward   : {t_torch:.2f}s")

        t0 = time.perf_counter()
        torchless_logits = run_torchless(token_ids, embed_dtype=args.embed_dtype)
        t_tl = time.perf_counter() - t0
        print(f"  torchless forward: {t_tl:.2f}s")

        max_abs, max_rel = _diff_stats(torch_logits, torchless_logits)
        top1_match = int(torch_logits.argmax()) == int(torchless_logits.argmax())
        overlap_5 = _topk_overlap(torch_logits, torchless_logits, 5)
        overlap_20 = _topk_overlap(torch_logits, torchless_logits, 20)

        torch_top1 = int(torch_logits.argmax())
        tl_top1 = int(torchless_logits.argmax())
        print(f"  top-1  torch    : {torch_top1!r}  ({tok.decode([torch_top1])!r})")
        print(f"  top-1  torchless: {tl_top1!r}  ({tok.decode([tl_top1])!r})")
        print(f"  top-1 match     : {top1_match}")
        print(f"  top-5 overlap   : {overlap_5:.2f}")
        print(f"  top-20 overlap  : {overlap_20:.2f}")
        print(f"  max|diff|       : {max_abs:.3f}")
        print(f"  max rel err     : {max_rel:.4f}")

        # Minimum bar: top-1 matches and top-20 overlap is decent.
        if not top1_match:
            print(f"  FAIL: top-1 differs")
            all_ok = False
        elif overlap_20 < 0.5:
            print(f"  FAIL: top-20 overlap < 0.5")
            all_ok = False

    print("\nOK" if all_ok else "\nFAIL")
    return 0 if all_ok else 1


if __name__ == "__main__":
    sys.exit(main())
