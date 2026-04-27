#!/usr/bin/env python3
"""
Litespark-Inference Command Line Interface

Usage:
    litespark-inference generate "Your prompt here"
    litespark-inference chat
    litespark-inference benchmark
    litespark-inference info
"""

import argparse
import platform
import sys
import time

_is_apple_silicon = platform.machine() in ('arm64', 'aarch64') and platform.system() == 'Darwin'

_MODEL_CHOICES_HELP = (
    'bitnet-2b, falcon-edge-1b, falcon-edge-1b-instruct, '
    'falcon-edge-3b, falcon-edge-3b-instruct'
)

_CHAT_UNSUPPORTED_MODELS = {
    'falcon-edge-1b',
    'falcon-edge-3b',
}

# Models the torchless runtime can handle today. Extend as new families are
# ported. Anything outside this set silently falls back to the torch-backed
# path so existing Falcon Edge workflows keep working unchanged.
_TORCHLESS_SUPPORTED_MODELS = {'bitnet-2b'}


def _should_use_torchless(args) -> bool:
    """Route to torchless when the target model is supported there.

    Torchless currently does greedy decoding only; sampling args
    (`--temperature`/`--top-p`/`--top-k`) are accepted for compatibility
    with the torch-backed CLI but quietly ignored. Set LITESPARK_FORCE_TORCH=1
    if you need sampling-based decoding on a torchless-supported model.
    """
    import os
    if os.environ.get('LITESPARK_FORCE_TORCH'):
        return False
    return getattr(args, 'model', None) in _TORCHLESS_SUPPORTED_MODELS


def _build_prompt(tokenizer, messages):
    """Build a prompt, using chat templates when the tokenizer provides one."""
    chat_template = getattr(tokenizer, 'chat_template', None)
    if chat_template:
        return tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)

    system_messages = [m['content'] for m in messages if m['role'] == 'system']
    user_messages = [m['content'] for m in messages if m['role'] == 'user']
    assistant_messages = [m['content'] for m in messages if m['role'] == 'assistant']

    instruction = system_messages[-1] if system_messages else ""
    latest_user = user_messages[-1] if user_messages else ""

    if assistant_messages:
        history_lines = []
        for user_text, assistant_text in zip(user_messages[:-1], assistant_messages):
            history_lines.append(f"Question: {user_text}")
            history_lines.append(f"Answer: {assistant_text}")
        if latest_user:
            history_lines.append(f"Question: {latest_user}")
        history = "\n".join(history_lines)
        return f"{instruction}\n\n{history}\nAnswer:".strip()

    return f"{instruction}\n\nQuestion: {latest_user}\nAnswer:".strip()


def cmd_generate(args):
    """Generate text from a prompt."""
    if _should_use_torchless(args):
        from .torchless.__main__ import cmd_generate as _cmd_generate_torchless
        # torchless cmd_generate expects: prompt, max_tokens, raw, ignore_eos
        class _A: pass
        tl_args = _A()
        tl_args.prompt = args.prompt
        tl_args.max_tokens = args.max_tokens
        tl_args.raw = False   # use the chat-format wrapper, like the torch path does
        tl_args.ignore_eos = False
        print(f"[litespark-inference] torchless runtime (model={args.model}).")
        if getattr(args, 'temperature', 0.0) and args.temperature > 0.0:
            print(f"  note: --temperature/--top-p/--top-k ignored (torchless is "
                  f"greedy-only today; set LITESPARK_FORCE_TORCH=1 for sampling).")
        return _cmd_generate_torchless(tl_args)

    from .models import load_ternary_model, TextStreamer
    import torch

    model, tokenizer = load_ternary_model(args.model, mode=getattr(args, 'mode', 'neon'))

    print(f"\nPrompt: {args.prompt}")
    print("-" * 50)

    system_prompt = "You are Litespark, a helpful AI assistant running locally. Provide accurate, concise, and practical answers. If a request is ambiguous, ask a brief clarifying question. If you do not know something, say so plainly."
    messages = [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": args.prompt},
    ]
    prompt = _build_prompt(tokenizer, messages)
    input_ids = tokenizer.encode(prompt, return_tensors='pt')

    streamer = TextStreamer(tokenizer)

    start = time.perf_counter()
    with torch.no_grad():
        output_ids = model.generate(
            input_ids,
            max_new_tokens=args.max_tokens,
            temperature=args.temperature,
            top_p=args.top_p,
            top_k=args.top_k,
            streamer=streamer,
        )
    elapsed = time.perf_counter() - start

    num_tokens = output_ids.shape[1] - input_ids.shape[1]
    print(f"\n{'-' * 50}")
    print(f"Generated {num_tokens} tokens in {elapsed:.2f}s ({num_tokens/elapsed:.1f} tok/s)")


def cmd_chat(args):
    """Interactive chat mode."""
    if _should_use_torchless(args):
        from .torchless.__main__ import cmd_chat as _cmd_chat_torchless
        class _A: pass
        tl_args = _A()
        tl_args.model = args.model
        tl_args.max_tokens = args.max_tokens
        tl_args.embed_dtype = 'int4'
        print(f"[litespark-inference] routing to torchless runtime for '{args.model}'.")
        return _cmd_chat_torchless(tl_args)

    from .models import load_ternary_model, TextStreamer
    import torch

    if args.model in _CHAT_UNSUPPORTED_MODELS:
        raise SystemExit(
            f"Model '{args.model}' is generate-only. "
            "Use an instruct model for chat, such as "
            "'falcon-edge-1b-instruct' or 'falcon-edge-3b-instruct'."
        )

    model, tokenizer = load_ternary_model(args.model, mode=getattr(args, 'mode', 'neon'))

    print("\nLitespark-Inference Chat")
    print("Type 'quit' or 'exit' to end the conversation")
    print("-" * 50)

    system_prompt = "You are Litespark, a helpful AI assistant running locally. Provide accurate, concise, and practical answers. If a request is ambiguous, ask a brief clarifying question. If you do not know something, say so plainly."
    messages = [{"role": "system", "content": system_prompt}]

    while True:
        try:
            user_input = input("\nYou: ").strip()
        except (EOFError, KeyboardInterrupt):
            print("\nGoodbye!")
            break

        if user_input.lower() in ['quit', 'exit', 'q']:
            print("Goodbye!")
            break

        if not user_input:
            continue

        messages.append({"role": "user", "content": user_input})

        # Build prompt using the tokenizer's chat template
        prompt = _build_prompt(tokenizer, messages)
        input_ids = tokenizer.encode(prompt, return_tensors='pt')

        print("\nAssistant: ", end="", flush=True)

        streamer = TextStreamer(tokenizer)

        with torch.no_grad():
            output_ids = model.generate(
                input_ids,
                max_new_tokens=args.max_tokens,
                temperature=args.temperature,
                top_p=args.top_p,
                top_k=args.top_k,
                streamer=streamer,
            )

        # Extract just the new response for conversation history
        response = tokenizer.decode(output_ids[0][input_ids.shape[1]:], skip_special_tokens=True)
        print()  # newline after streamed output

        messages.append({"role": "assistant", "content": response})


def cmd_benchmark(args):
    """Run performance benchmark."""
    if _should_use_torchless(args):
        from .torchless.__main__ import cmd_benchmark as _cmd_benchmark_torchless
        class _A: pass
        tl_args = _A()
        tl_args.model = args.model
        tl_args.tokens = args.tokens
        tl_args.embed_dtype = 'int4'
        print(f"[litespark-inference] routing to torchless runtime for '{args.model}'.")
        return _cmd_benchmark_torchless(tl_args)

    from .models import load_ternary_model, get_arch_info, get_kernel_type
    import torch
    import gc

    print("Litespark-Inference Benchmark")
    print("=" * 60)

    # Show system info
    arch_info = get_arch_info()
    print(f"\nArchitecture: {arch_info['machine']}")
    if arch_info['is_apple_silicon']:
        print("Platform: Apple Silicon")
    elif arch_info['is_x86_64']:
        print(f"Platform: x86_64")
        if arch_info['x86_features']:
            features = arch_info['x86_features']
            if features['avx512vnni']:
                print("SIMD: AVX-512 VNNI")
            elif features['avx_vnni']:
                print("SIMD: AVX-VNNI (256-bit)")

    # Load model
    print(f"\nLoading model: {args.model}")
    gc.collect()

    import resource
    import platform
    def get_memory_mb():
        if platform.system() == 'Darwin':
            return resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / (1024 * 1024)
        else:
            return resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024

    mem_before = get_memory_mb()
    model, tokenizer = load_ternary_model(args.model, mode=getattr(args, 'mode', 'neon'))
    gc.collect()
    mem_after = get_memory_mb()

    print(f"Kernel: {get_kernel_type()}")
    print(f"Memory: {mem_after - mem_before:.0f} MB")

    # Benchmark
    prompt = "The quick brown fox jumps over the lazy dog."
    input_ids = tokenizer.encode(prompt, return_tensors='pt')

    print(f"\nBenchmarking with prompt: '{prompt}'")
    print("-" * 60)

    # Warmup
    print("Warming up...")
    with torch.no_grad():
        for _ in range(3):
            _ = model(input_ids)

    # TTFT (Time to First Token)
    print("\nTime to First Token (TTFT):")
    num_runs = 10
    times = []
    with torch.no_grad():
        for _ in range(num_runs):
            start = time.perf_counter()
            _ = model(input_ids)
            times.append((time.perf_counter() - start) * 1000)

    ttft_avg = sum(times) / len(times)
    ttft_min = min(times)
    print(f"  Average: {ttft_avg:.2f} ms")
    print(f"  Minimum: {ttft_min:.2f} ms")

    # Generation speed
    print(f"\nGeneration Speed ({args.tokens} tokens):")
    num_runs = 3
    times = []
    with torch.no_grad():
        for _ in range(num_runs):
            start = time.perf_counter()
            _ = model.generate(input_ids, max_new_tokens=args.tokens, temperature=0)
            times.append(time.perf_counter() - start)

    gen_avg = sum(times) / len(times)
    tps = args.tokens / gen_avg
    print(f"  Average: {gen_avg*1000:.0f} ms ({tps:.2f} tokens/sec)")

    # Summary
    print("\n" + "=" * 60)
    print("SUMMARY")
    print("=" * 60)
    print(f"Model:      {args.model}")
    print(f"Kernel:     {get_kernel_type()}")
    print(f"Memory:     {mem_after - mem_before:.0f} MB")
    print(f"TTFT:       {ttft_avg:.2f} ms")
    print(f"Throughput: {tps:.2f} tokens/sec")


def cmd_info(args):
    """Show system and model information."""
    from .models import get_arch_info
    from . import __version__

    print(f"Litespark-Inference v{__version__}")
    print("=" * 50)

    arch_info = get_arch_info()
    print(f"\nArchitecture: {arch_info['machine']}")

    if arch_info['is_apple_silicon']:
        print("Platform: Apple Silicon (M1/M2/M3/M4)")
        print("Kernel: NEON SDOT (128-bit)")
    elif arch_info['is_x86_64']:
        print("Platform: x86_64")
        if arch_info['x86_features']:
            features = arch_info['x86_features']
            print(f"  AVX-512F:    {'Yes' if features['avx512f'] else 'No'}")
            print(f"  AVX-512VNNI: {'Yes' if features['avx512vnni'] else 'No'}")
            print(f"  AVX-VNNI:    {'Yes' if features['avx_vnni'] else 'No'}")

            if features['avx512vnni']:
                print("Kernel: AVX-512 VNNI (512-bit)")
            elif features['avx_vnni']:
                print("Kernel: AVX-VNNI (256-bit)")
    else:
        print("Platform: Unknown")

    print("\nAvailable Models:")
    print("  bitnet-2b: BitNet b1.58 2B parameters, 4T tokens trained (~556 MB)")
    print("  falcon-edge-1b: Falcon Edge 1B base model")
    print("  falcon-edge-1b-instruct: Falcon Edge 1B instruct model")
    print("  falcon-edge-3b: Falcon Edge 3B base model")
    print("  falcon-edge-3b-instruct: Falcon Edge 3B instruct model")


def main():
    parser = argparse.ArgumentParser(
        prog='litespark-inference',
        description='Litespark-Inference - Efficient CPU inference for ternary language models'
    )
    parser.add_argument('--version', action='version', version='%(prog)s 0.1.5')

    subparsers = parser.add_subparsers(dest='command', help='Commands')

    # generate command
    gen_parser = subparsers.add_parser('generate', help='Generate text from a prompt')
    gen_parser.add_argument('prompt', type=str, help='Input prompt')
    gen_parser.add_argument('--model', '-m', type=str, default='bitnet-2b', help=f'Model name (available: {_MODEL_CHOICES_HELP})')
    if _is_apple_silicon:
        gen_parser.add_argument('--mode', type=str, default='neon', choices=['neon', 'accelerate'], help='Kernel mode: neon (fast, int8) or accelerate (accurate, float32)')
    gen_parser.add_argument('--max-tokens', '-n', type=int, default=100, help='Max tokens to generate')
    gen_parser.add_argument('--temperature', '-t', type=float, default=0.7, help='Sampling temperature')
    gen_parser.add_argument('--top-p', type=float, default=0.9, help='Nucleus sampling threshold')
    gen_parser.add_argument('--top-k', type=int, default=50, help='Top-k sampling')
    gen_parser.set_defaults(func=cmd_generate)

    # chat command
    chat_parser = subparsers.add_parser('chat', help='Interactive chat mode')
    chat_parser.add_argument('--model', '-m', type=str, default='bitnet-2b', help=f'Model name (available: {_MODEL_CHOICES_HELP})')
    if _is_apple_silicon:
        chat_parser.add_argument('--mode', type=str, default='neon', choices=['neon', 'accelerate'], help='Kernel mode: neon (fast, int8) or accelerate (accurate, float32)')
    chat_parser.add_argument('--max-tokens', '-n', type=int, default=200, help='Max tokens per response')
    chat_parser.add_argument('--temperature', '-t', type=float, default=0.7, help='Sampling temperature')
    chat_parser.add_argument('--top-p', type=float, default=0.9, help='Nucleus sampling threshold')
    chat_parser.add_argument('--top-k', type=int, default=50, help='Top-k sampling')
    chat_parser.set_defaults(func=cmd_chat)

    # benchmark command
    bench_parser = subparsers.add_parser('benchmark', help='Run performance benchmark')
    bench_parser.add_argument('--model', '-m', type=str, default='bitnet-2b', help=f'Model name (available: {_MODEL_CHOICES_HELP})')
    if _is_apple_silicon:
        bench_parser.add_argument('--mode', type=str, default='neon', choices=['neon', 'accelerate'], help='Kernel mode: neon (fast, int8) or accelerate (accurate, float32)')
    bench_parser.add_argument('--tokens', '-n', type=int, default=20, help='Tokens to generate')
    bench_parser.set_defaults(func=cmd_benchmark)

    # info command
    info_parser = subparsers.add_parser('info', help='Show system information')
    info_parser.set_defaults(func=cmd_info)

    args = parser.parse_args()

    if args.command is None:
        parser.print_help()
        sys.exit(1)

    args.func(args)


if __name__ == '__main__':
    main()
