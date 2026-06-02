"""
Torchless CLI.

Usage:
    python -m litespark_inference.torchless generate "prompt" --max-tokens 32
    python -m litespark_inference.torchless info

Provides a minimal generate/info surface without any torch/transformers
imports. Uses the `tokenizers` Rust-backed package for encode/decode and
the extern "C" NEON packed matmul kernel for every projection.
"""

from __future__ import annotations

import argparse
import sys
import time


def cmd_info(args: argparse.Namespace) -> int:
    import platform

    from .kernel import ensure_built, has_omp, max_threads

    print(f"litespark_inference.torchless")
    print(f"  platform : {platform.system()} {platform.machine()}")
    print(f"  python   : {platform.python_version()}")
    print(f"  kernel   : {ensure_built()}")
    print(f"  OpenMP   : {has_omp()}  (max_threads={max_threads()})")
    return 0


def cmd_generate(args: argparse.Namespace) -> int:
    from . import load_bitnet_2b, load_tokenizer
    from .runtime import forward_one, forward_prefill, generate, init_state

    print(f"Loading tokenizer...", flush=True)
    t0 = time.perf_counter()
    tok = load_tokenizer()
    print(f"  {time.perf_counter() - t0:.2f}s")

    print(f"Loading model...", flush=True)
    t0 = time.perf_counter()
    model = load_bitnet_2b()
    print(f"  {time.perf_counter() - t0:.2f}s")

    # Build the prompt. `--raw` skips the chat wrapper.
    prompt_text = args.prompt if args.raw else tok.format_chat(args.prompt)
    prompt_ids = tok.encode(prompt_text)
    print(f"Prompt ({len(prompt_ids)} tokens): {prompt_text!r}", flush=True)

    t_max = len(prompt_ids) + args.max_tokens + 1
    state = init_state(model, t_max=t_max)

    # Prefill (batched for multi-token prompts; M=1 has nothing to batch).
    print("Prefill...", flush=True)
    t0 = time.perf_counter()
    if len(prompt_ids) > 1:
        logits = forward_prefill(model, state, [int(t) for t in prompt_ids])
    else:
        logits = forward_one(model, state, int(prompt_ids[0]))
    prefill_t = time.perf_counter() - t0
    print(f"  {len(prompt_ids)} tokens in {prefill_t:.2f}s "
          f"({len(prompt_ids)/prefill_t:.2f} tok/s)")

    # Decode (greedy; stop on eos)
    print("Generate:", flush=True)
    generated: list[int] = []
    t0 = time.perf_counter()
    for _ in range(args.max_tokens):
        import numpy as np
        next_id = int(np.argmax(logits))
        if next_id == tok.eos_token_id and not args.ignore_eos:
            break
        generated.append(next_id)
        # Stream one-token decode for readability (skip special only at end).
        piece = tok.decode([next_id], skip_special_tokens=False)
        print(piece, end="", flush=True)
        logits = forward_one(model, state, next_id)
    gen_t = time.perf_counter() - t0
    print()

    text = tok.decode(generated)
    if generated:
        print(f"\nGenerated {len(generated)} tokens in {gen_t:.2f}s "
              f"({len(generated)/gen_t:.2f} tok/s)")
    print(f"\n--- output ---")
    print(text)
    return 0


def cmd_benchmark(args: argparse.Namespace) -> int:
    """Mini benchmark: TTFT (single forward) + token generation throughput."""
    import numpy as np

    from . import load_bitnet_2b, load_tokenizer
    from .runtime import forward_one, forward_prefill, init_state

    print("Litespark-Inference Benchmark (torchless)")
    print("=" * 60)
    import platform
    print(f"\nArchitecture: {platform.machine()}")
    print(f"Platform: {platform.system()}")
    from .kernel import has_omp, max_threads
    print(f"Kernel: neon (torchless, extern \"C\" NEON SDOT, OMP={has_omp()} "
          f"max_threads={max_threads()})")

    print(f"\nLoading model: {args.model} (embed_dtype={args.embed_dtype})")
    t0 = time.perf_counter()
    model = load_bitnet_2b(embed_dtype=args.embed_dtype)
    tok = load_tokenizer()
    load_t = time.perf_counter() - t0

    tb = model.tensor_bytes()
    MB = 1024 * 1024
    total_tensor_mb = tb["total_incl_embedding"] / MB
    print(f"Memory: {total_tensor_mb:.0f} MB  "
          f"(packed={tb['packed_weights']/MB:.0f} MB, "
          f"embed={tb['embedding']/MB:.0f} MB)")
    print(f"Load time: {load_t:.2f}s")

    prompt = "The quick brown fox jumps over the lazy dog."
    ids = tok.encode(prompt)
    print(f"\nBenchmarking with prompt: '{prompt}' ({len(ids)} tokens)")
    print("-" * 60)

    # Warmup
    print("Warming up...", flush=True)
    for _ in range(3):
        state = init_state(model, t_max=len(ids) + 1)
        for tid in ids:
            _ = forward_one(model, state, int(tid))

    # TTFT: time a fresh batched prefill of the whole prompt.
    print("\nTime to First Token (TTFT):")
    num_runs = 10
    times = []
    prompt_token_ids = [int(t) for t in ids]
    for _ in range(num_runs):
        state = init_state(model, t_max=len(ids) + 1)
        t0 = time.perf_counter()
        forward_prefill(model, state, prompt_token_ids)
        times.append((time.perf_counter() - t0) * 1000)
    ttft_avg = sum(times) / len(times)
    ttft_min = min(times)
    print(f"  Average: {ttft_avg:.2f} ms")
    print(f"  Minimum: {ttft_min:.2f} ms")

    # Generation speed: generate args.tokens with greedy decode after prefill.
    print(f"\nGeneration Speed ({args.tokens} tokens):")
    num_runs = 3
    times = []
    for _ in range(num_runs):
        state = init_state(model, t_max=len(ids) + args.tokens + 1)
        # Prefill
        logits = forward_prefill(model, state, prompt_token_ids)
        # Generate
        t0 = time.perf_counter()
        for _ in range(args.tokens):
            next_id = int(np.argmax(logits))
            logits = forward_one(model, state, next_id)
        times.append(time.perf_counter() - t0)
    gen_avg = sum(times) / len(times)
    tps = args.tokens / gen_avg
    print(f"  Average: {gen_avg*1000:.0f} ms ({tps:.2f} tokens/sec)")

    print("\n" + "=" * 60)
    print("SUMMARY")
    print("=" * 60)
    print(f"Model:      {args.model}")
    print(f"Backend:    torchless (embed_dtype={args.embed_dtype})")
    print(f"Memory:     {total_tensor_mb:.0f} MB")
    print(f"TTFT:       {ttft_avg:.2f} ms")
    print(f"Throughput: {tps:.2f} tokens/sec")
    return 0


def cmd_chat(args: argparse.Namespace) -> int:
    """Interactive chat loop, greedy decoding, running history."""
    import numpy as np

    from . import load_bitnet_2b, load_tokenizer
    from .runtime import forward_one, forward_prefill, init_state

    print("Loading tokenizer + model...", flush=True)
    tok = load_tokenizer()
    model = load_bitnet_2b(embed_dtype=args.embed_dtype)

    print("\nLitespark-Inference Chat (torchless)")
    print("Type 'quit' or 'exit' to end the conversation")
    print("-" * 50)

    system_prompt = (
        "You are Litespark, a helpful AI assistant running locally. "
        "Provide accurate, concise, and practical answers."
    )
    history_user: list[str] = []
    history_assistant: list[str] = []

    while True:
        try:
            user_input = input("\nYou: ").strip()
        except (EOFError, KeyboardInterrupt):
            print("\nGoodbye!")
            break
        if user_input.lower() in ("quit", "exit", "q"):
            print("Goodbye!")
            break
        if not user_input:
            continue

        history_user.append(user_input)

        # Build a flattened prompt. format_chat handles the system + current
        # user turn; previous turns are rolled in as Question/Answer pairs
        # (matches the torch-backed CLI's non-template fallback).
        pieces = [system_prompt, ""]
        for u, a in zip(history_user[:-1], history_assistant):
            pieces.append(f"Question: {u}")
            pieces.append(f"Answer: {a}")
        pieces.append(f"Question: {user_input}")
        pieces.append("Answer:")
        prompt = "\n".join(pieces)

        prompt_ids = tok.encode(prompt)
        t_max = len(prompt_ids) + args.max_tokens + 1
        state = init_state(model, t_max=t_max)

        # Prefill (batched for multi-token prompts).
        if len(prompt_ids) > 1:
            logits = forward_prefill(model, state, [int(t) for t in prompt_ids])
        else:
            logits = forward_one(model, state, int(prompt_ids[0]))

        # Stream generate
        print("\nAssistant: ", end="", flush=True)
        gen_ids: list[int] = []
        for _ in range(args.max_tokens):
            next_id = int(np.argmax(logits))
            if next_id == tok.eos_token_id:
                break
            gen_ids.append(next_id)
            piece = tok.decode([next_id])
            print(piece, end="", flush=True)
            logits = forward_one(model, state, next_id)
        print()

        response = tok.decode(gen_ids)
        history_assistant.append(response)

    return 0


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(prog="litespark_inference.torchless")
    sub = p.add_subparsers(dest="cmd", required=True)

    pg = sub.add_parser("generate", help="Greedy-decode a continuation")
    pg.add_argument("prompt", type=str)
    pg.add_argument("--max-tokens", type=int, default=32)
    pg.add_argument("--raw", action="store_true",
                    help="Skip the chat-format wrapper around the prompt.")
    pg.add_argument("--ignore-eos", action="store_true",
                    help="Keep generating even after EOS is produced.")
    pg.set_defaults(func=cmd_generate)

    pi = sub.add_parser("info", help="Show build/runtime info")
    pi.set_defaults(func=cmd_info)

    pb = sub.add_parser("benchmark", help="Run a pp/tg mini benchmark")
    pb.add_argument("--model", type=str, default="bitnet-2b")
    pb.add_argument("--tokens", type=int, default=32,
                    help="Tokens to generate for the tg measurement.")
    pb.add_argument("--embed-dtype", choices=["bf16", "int8", "int4"], default="int4")
    pb.set_defaults(func=cmd_benchmark)

    pc = sub.add_parser("chat", help="Interactive chat (greedy decode)")
    pc.add_argument("--model", type=str, default="bitnet-2b")
    pc.add_argument("--max-tokens", type=int, default=256)
    pc.add_argument("--embed-dtype", choices=["bf16", "int8", "int4"], default="int4")
    pc.set_defaults(func=cmd_chat)

    args = p.parse_args(argv)
    return args.func(args)


if __name__ == "__main__":
    sys.exit(main())
