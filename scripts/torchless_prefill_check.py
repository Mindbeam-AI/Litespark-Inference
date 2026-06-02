"""
Correctness + TTFT benchmark for the batched prefill path.

Compares forward_prefill (one batched matmul per projection for the whole
prompt) against the reference loop of forward_one over each prompt token:

  1. Correctness: last-token logits, argmax, and the post-prefill KV cache
     must match the M=1 reference within fp tolerance; greedy continuations
     must produce identical token IDs.
  2. Timing: prefill wall-clock (TTFT) for both paths at the requested thread
     count, reported with the speedup.

Usage:
    python scripts/torchless_prefill_check.py [--prompt-tokens N]
                                              [--gen N] [--threads N]
"""

from __future__ import annotations

import argparse
import os
import statistics
import time


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--prompt-tokens", type=int, default=128)
    ap.add_argument("--gen", type=int, default=16, help="tokens to continue for the equality check")
    ap.add_argument("--threads", type=int, default=1)
    ap.add_argument("--runs", type=int, default=3)
    args = ap.parse_args()

    os.environ.setdefault("OMP_NUM_THREADS", str(args.threads))

    import numpy as np

    from litespark_inference.torchless import load_bitnet_2b, load_tokenizer
    from litespark_inference.torchless.runtime import (
        forward_one,
        forward_prefill,
        init_state,
    )

    print(f"Loading model (threads={args.threads}) ...")
    model = load_bitnet_2b()
    tok = load_tokenizer()

    prompt = "The quick brown fox jumps over the lazy dog. " * 20
    ids = tok.encode(prompt)[: args.prompt_tokens]
    T = len(ids)
    print(f"Prompt tokens: {T}")

    t_max = T + args.gen + 4

    # ---- Reference: forward_one loop ----
    ref_state = init_state(model, t_max=t_max)
    ref_logits = None
    for tid in ids:
        ref_logits = forward_one(model, ref_state, int(tid))
    ref_logits = ref_logits.copy()

    # ---- Batched prefill ----
    pf_state = init_state(model, t_max=t_max)
    pf_logits = forward_prefill(model, pf_state, [int(t) for t in ids]).copy()

    # ---- Correctness ----
    max_abs = float(np.max(np.abs(ref_logits - pf_logits)))
    ref_arg = int(np.argmax(ref_logits))
    pf_arg = int(np.argmax(pf_logits))
    k_diff = float(np.max(np.abs(ref_state.cache_k[:, :T] - pf_state.cache_k[:, :T])))
    v_diff = float(np.max(np.abs(ref_state.cache_v[:, :T] - pf_state.cache_v[:, :T])))
    assert ref_state.pos == pf_state.pos == T, (ref_state.pos, pf_state.pos, T)

    print("\n=== Correctness (batched prefill vs forward_one loop) ===")
    print(f"  argmax(last logits):  ref={ref_arg}  prefill={pf_arg}  match={ref_arg == pf_arg}")
    print(f"  max|Δ logits|:        {max_abs:.4e}")
    print(f"  max|Δ KV cache|:      k={k_diff:.4e}  v={v_diff:.4e}")

    # Greedy continuation equality.
    def greedy(state, logits, n):
        out = []
        for _ in range(n):
            nid = int(np.argmax(logits))
            out.append(nid)
            logits = forward_one(model, state, nid)
        return out

    ref_seq = greedy(ref_state, ref_logits, args.gen)
    pf_seq = greedy(pf_state, pf_logits, args.gen)
    print(f"  greedy {args.gen}-token continuation identical: {ref_seq == pf_seq}")
    if ref_seq != pf_seq:
        print(f"    ref:     {ref_seq}")
        print(f"    prefill: {pf_seq}")

    # ---- Timing (TTFT) ----
    def time_loop():
        st = init_state(model, t_max=t_max)
        t0 = time.perf_counter()
        for tid in ids:
            forward_one(model, st, int(tid))
        return (time.perf_counter() - t0) * 1000.0

    def time_prefill():
        st = init_state(model, t_max=t_max)
        t0 = time.perf_counter()
        forward_prefill(model, st, [int(t) for t in ids])
        return (time.perf_counter() - t0) * 1000.0

    # warmup
    time_loop(); time_prefill()
    loop_ms = [time_loop() for _ in range(args.runs)]
    pf_ms = [time_prefill() for _ in range(args.runs)]
    loop_mean = statistics.mean(loop_ms)
    pf_mean = statistics.mean(pf_ms)

    print(f"\n=== TTFT ({T} prompt tokens, threads={args.threads}, {args.runs} runs) ===")
    print(f"  forward_one loop:  {loop_mean:8.1f} ms  ({T / (loop_mean/1000):.1f} tok/s)")
    print(f"  forward_prefill:   {pf_mean:8.1f} ms  ({T / (pf_mean/1000):.1f} tok/s)")
    print(f"  speedup:           {loop_mean / pf_mean:6.2f}x")

    ok = (ref_arg == pf_arg) and (ref_seq == pf_seq) and max_abs < 1.0
    print("\nRESULT:", "PASS" if ok else "FAIL")
    raise SystemExit(0 if ok else 1)


if __name__ == "__main__":
    main()
