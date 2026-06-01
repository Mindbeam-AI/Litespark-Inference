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
import platform
import sys
import time


def _is_apple_silicon() -> bool:
    return platform.system() == "Darwin" and platform.machine().lower() in ("arm64", "aarch64")


def _kernel_label(mode: str) -> str:
    if mode == "accelerate":
        return "accelerate (torchless, Apple Accelerate)"
    machine = platform.machine().lower()
    if machine in ("arm64", "aarch64"):
        return "neon (torchless, extern \"C\" NEON SDOT)"
    if machine in ("x86_64", "amd64"):
        # The actual loaded kernel is decided at runtime in kernel.py based
        # on /proc/cpuinfo. Report whichever it picked rather than always
        # claiming AVX-512.
        from .kernel import _ARCH_TAG
        if _ARCH_TAG == "avx2":
            return "avx2 (torchless, extern \"C\" AVX2+FMA fallback)"
        return "avx512 (torchless, extern \"C\" AVX-512/VNNI)"
    return "torchless"


def cmd_info(args: argparse.Namespace) -> int:
    from .kernel import ensure_built, has_accelerate, has_omp, max_threads

    print(f"litespark_inference.torchless")
    print(f"  platform : {platform.system()} {platform.machine()}")
    print(f"  python   : {platform.python_version()}")
    print(f"  kernel   : {ensure_built()}")
    print(f"  OpenMP   : {has_omp()}  (max_threads={max_threads()})")
    print(f"  Accelerate: {has_accelerate()}")
    return 0


def cmd_generate(args: argparse.Namespace) -> int:
    model_name = getattr(args, "model", "bitnet-2b")
    embed_dtype = getattr(args, "embed_dtype", "int4")
    mode = getattr(args, "mode", "neon")

    print(f"Loading tokenizer...", flush=True)
    t0 = time.perf_counter()
    if model_name == "bitnet-2b":
        from . import load_bitnet_2b, load_tokenizer
        from .runtime import forward_one, forward_prefill, init_state

        tok = load_tokenizer()
        model = load_bitnet_2b(embed_dtype=embed_dtype, mode=mode)
    elif model_name.startswith("falcon-edge-"):
        from . import FALCON_TORCHLESS_REPOS, load_falcon_edge, load_tokenizer
        from .runtime import falcon_forward_one, falcon_forward_prefill, init_state

        if model_name not in FALCON_TORCHLESS_REPOS:
            raise ValueError(
                f"Unknown Falcon Edge model {model_name!r}. "
                f"Available: {sorted(FALCON_TORCHLESS_REPOS)}"
        )
        tok = load_tokenizer(FALCON_TORCHLESS_REPOS[model_name])
        model = load_falcon_edge(model_name, embed_dtype=embed_dtype, mode=mode)
    else:
        raise ValueError(f"Unsupported torchless model {model_name!r}.")
    print(f"  {time.perf_counter() - t0:.2f}s")

    # Build the prompt. `--raw` skips the chat wrapper.
    prompt_text = args.prompt if args.raw else tok.format_chat(args.prompt)
    prompt_ids = tok.encode(prompt_text)
    print(f"Prompt ({len(prompt_ids)} tokens): {prompt_text!r}", flush=True)

    t_max = len(prompt_ids) + args.max_tokens + 1
    state = init_state(model, t_max=t_max)

    # Prefill (batched M=T when the kernel supports it)
    print("Prefill...", flush=True)
    t0 = time.perf_counter()
    if model_name == "bitnet-2b":
        logits = forward_prefill(model, state, list(prompt_ids))
    else:
        logits = falcon_forward_prefill(model, state, list(prompt_ids))
    prefill_t = time.perf_counter() - t0
    print(f"  {len(prompt_ids)} tokens in {prefill_t:.2f}s "
          f"({len(prompt_ids)/prefill_t:.2f} tok/s)")

    # Decode (greedy; stop on eos)
    print("Generate:", flush=True)
    generated: list[int] = []
    streamed_text = ""
    t0 = time.perf_counter()
    for _ in range(args.max_tokens):
        import numpy as np
        next_id = int(np.argmax(logits))
        eos_ids = set(getattr(model.config, "eos_token_ids", (tok.eos_token_id,)))
        eos_ids.update(getattr(tok, "stop_token_ids", ()))
        if next_id in eos_ids and not args.ignore_eos:
            break
        generated.append(next_id)
        text_now = tok.decode(generated)
        if "\ufffd" not in text_now and text_now.startswith(streamed_text):
            print(text_now[len(streamed_text):], end="", flush=True)
            streamed_text = text_now
        if model_name == "bitnet-2b":
            logits = forward_one(model, state, next_id)
        else:
            logits = falcon_forward_one(model, state, next_id)
    gen_t = time.perf_counter() - t0
    text = tok.decode(generated)
    if text.startswith(streamed_text):
        print(text[len(streamed_text):], end="", flush=True)
    print()

    if generated:
        print(f"\nGenerated {len(generated)} tokens in {gen_t:.2f}s "
              f"({len(generated)/gen_t:.2f} tok/s)")
    print(f"\n--- output ---")
    print(text)
    return 0


def cmd_benchmark(args: argparse.Namespace) -> int:
    """Mini benchmark: TTFT (single forward) + token generation throughput."""
    import numpy as np

    print("Litespark-Inference Benchmark (torchless)")
    print("=" * 60)
    import platform
    print(f"\nArchitecture: {platform.machine()}")
    print(f"Platform: {platform.system()}")
    from .kernel import has_omp, max_threads

    mode = getattr(args, "mode", "neon")
    if mode == "accelerate":
        print(f"Kernel: {_kernel_label(mode)}")
    else:
        print(f"Kernel: {_kernel_label(mode)} (OMP={has_omp()} "
              f"max_threads={max_threads()})")

    print(f"\nLoading model: {args.model} (embed_dtype={args.embed_dtype})")
    t0 = time.perf_counter()
    if args.model == "bitnet-2b":
        from . import load_bitnet_2b, load_tokenizer
        from .runtime import forward_one, init_state

        model = load_bitnet_2b(embed_dtype=args.embed_dtype, mode=mode)
        tok = load_tokenizer()
    elif args.model.startswith("falcon-edge-"):
        from . import FALCON_TORCHLESS_REPOS, load_falcon_edge, load_tokenizer
        from .runtime import falcon_forward_one, init_state

        if args.model not in FALCON_TORCHLESS_REPOS:
            raise ValueError(
                f"Unknown Falcon Edge model {args.model!r}. "
                f"Available: {sorted(FALCON_TORCHLESS_REPOS)}"
        )
        model = load_falcon_edge(args.model, embed_dtype=args.embed_dtype, mode=mode)
        tok = load_tokenizer(FALCON_TORCHLESS_REPOS[args.model])
    else:
        raise ValueError(f"Unsupported torchless model {args.model!r}. ")
    load_t = time.perf_counter() - t0

    tb = model.tensor_bytes()
    MB = 1024 * 1024
    total_tensor_mb = tb["total_incl_embedding"] / MB
    weight_type = (
        f"float32={tb.get('float32_weights', 0)/MB:.0f} MB"
        if mode == "accelerate" else f"packed={tb['packed_weights']/MB:.0f} MB")
    print(f"Memory: {total_tensor_mb:.0f} MB  "f"({weight_type}, embed={tb['embedding']/MB:.0f} MB)")
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
            if args.model == "bitnet-2b":
                _ = forward_one(model, state, int(tid))
            else:
                _ = falcon_forward_one(model, state, int(tid))

    # TTFT: time a fresh prefill.
    print("\nTime to First Token (TTFT):")
    num_runs = 10
    times = []
    for _ in range(num_runs):
        state = init_state(model, t_max=len(ids) + 1)
        t0 = time.perf_counter()
        for tid in ids:
            if args.model == "bitnet-2b":
                _ = forward_one(model, state, int(tid))
            else:
                _ = falcon_forward_one(model, state, int(tid))
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
        logits = None
        for tid in ids:
            if args.model == "bitnet-2b":
                logits = forward_one(model, state, int(tid))
            else:
                logits = falcon_forward_one(model, state, int(tid))
        # Generate
        t0 = time.perf_counter()
        for _ in range(args.tokens):
            next_id = int(np.argmax(logits))
            if args.model == "bitnet-2b":
                logits = forward_one(model, state, next_id)
            else:
                logits = falcon_forward_one(model, state, next_id)
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

    print("Loading tokenizer and model...", flush=True)
    model_name = getattr(args, "model", "bitnet-2b")
    mode = getattr(args, "mode", "neon")
    if model_name == "bitnet-2b":
        from . import load_bitnet_2b, load_tokenizer
        from .runtime import forward_one, init_state

        model = load_bitnet_2b(embed_dtype=args.embed_dtype, mode=mode)
        tok = load_tokenizer()
    elif model_name.startswith("falcon-edge-"):
        from . import FALCON_TORCHLESS_REPOS, load_falcon_edge, load_tokenizer
        from .runtime import falcon_forward_one, init_state

        if model_name not in FALCON_TORCHLESS_REPOS:
            raise ValueError(
                f"Unknown Falcon Edge model {model_name!r}. "
                f"Available: {sorted(FALCON_TORCHLESS_REPOS)}"
        )
        model = load_falcon_edge(model_name, embed_dtype=args.embed_dtype, mode=mode)
        tok = load_tokenizer(FALCON_TORCHLESS_REPOS[model_name])
    else:
        raise ValueError(f"Unsupported torchless model {model_name!r}.")

    print("\nLitespark-Inference Chat (torchless)")
    print("Type 'quit' or 'exit' to end the conversation")
    print("-" * 50)

    system_prompt = (
        "You are Litespark, a helpful AI assistant running locally. "
        "Provide accurate, concise, and practical answers."
    )
    history_user: list[str] = []
    history_assistant: list[str] = []
    messages: list[dict[str, str]] = [{"role": "system", "content": system_prompt}]

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
        # Build a flattened prompt. format_chat handles the system + current
        # user turn; previous turns are rolled in as Question/Answer pairs
        # (matches the torch-backed CLI's non-template fallback).
        if not hasattr(tok, "format_messages"):
            history_user.append(user_input)

            pieces = [system_prompt, ""]
            for u, a in zip(history_user[:-1], history_assistant):
                pieces.append(f"Question: {u}")
                pieces.append(f"Answer: {a}")
            pieces.append(f"Question: {user_input}")
            pieces.append("Answer:")
            prompt = "\n".join(pieces)
        else:
            messages.append({"role": "user", "content": user_input})
            prompt = tok.format_messages(messages)

        prompt_ids = tok.encode(prompt)
        t_max = len(prompt_ids) + args.max_tokens + 1
        state = init_state(model, t_max=t_max)

        # Prefill
        logits = None
        for tid in prompt_ids:
            if model_name == "bitnet-2b":
                logits = forward_one(model, state, int(tid))
            else:
                logits = falcon_forward_one(model, state, int(tid))

        # Stream generate
        print("\nAssistant: ", end="", flush=True)
        gen_ids: list[int] = []
        streamed_text = ""
        for _ in range(args.max_tokens):
            next_id = int(np.argmax(logits))
            eos_ids = set(getattr(model.config, "eos_token_ids", (tok.eos_token_id,)))
            eos_ids.update(getattr(tok, "stop_token_ids", ()))
            if next_id in eos_ids:
                break
            gen_ids.append(next_id)
            text_now = tok.decode(gen_ids)
            if "\ufffd" not in text_now and text_now.startswith(streamed_text):
                print(text_now[len(streamed_text):], end="", flush=True)
                streamed_text = text_now
            if model_name == "bitnet-2b":
                logits = forward_one(model, state, next_id)
            else:
                logits = falcon_forward_one(model, state, next_id)
        response = tok.decode(gen_ids)
        if response.startswith(streamed_text):
            print(response[len(streamed_text):], end="", flush=True)
        print()

        if hasattr(tok, "format_messages"):
            messages.append({"role": "assistant", "content": response})
        else:
            history_assistant.append(response)

    return 0


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(prog="litespark_inference.torchless")
    sub = p.add_subparsers(dest="cmd", required=True)

    pg = sub.add_parser("generate", help="Greedy-decode a continuation")
    pg.add_argument("prompt", type=str)
    pg.add_argument("--model", "-m", type=str, default="bitnet-2b")
    pg.add_argument("--max-tokens", type=int, default=32)
    if _is_apple_silicon():
        pg.add_argument("--mode", choices=["neon", "accelerate"], default="neon",
                        help="Backend: packed NEON or Apple Accelerate.")
    pg.add_argument("--embed-dtype", choices=["bf16", "int8", "int4"], default="int4")
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
    if _is_apple_silicon():
        pb.add_argument("--mode", choices=["neon", "accelerate"], default="neon",
                        help="Backend: packed NEON or Apple Accelerate.")
    pb.add_argument("--embed-dtype", choices=["bf16", "int8", "int4"], default="int4")
    pb.set_defaults(func=cmd_benchmark)

    pc = sub.add_parser("chat", help="Interactive chat (greedy decode)")
    pc.add_argument("--model", type=str, default="bitnet-2b")
    pc.add_argument("--max-tokens", type=int, default=256)
    if _is_apple_silicon():
        pc.add_argument("--mode", choices=["neon", "accelerate"], default="neon",
                        help="Backend: packed NEON or Apple Accelerate.")
    pc.add_argument("--embed-dtype", choices=["bf16", "int8", "int4"], default="int4")
    pc.set_defaults(func=cmd_chat)

    args = p.parse_args(argv)
    if not hasattr(args, "mode"):
        args.mode = "neon"
    return args.func(args)


if __name__ == "__main__":
    sys.exit(main())
