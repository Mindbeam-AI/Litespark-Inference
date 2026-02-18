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
import sys
import time


def cmd_generate(args):
    """Generate text from a prompt."""
    from .models import load_ternary_model
    import torch

    model, tokenizer = load_ternary_model(args.model, mode=args.mode)

    print(f"\nPrompt: {args.prompt}")
    print("-" * 50)

    input_ids = tokenizer.encode(args.prompt, return_tensors='pt')

    start = time.perf_counter()
    with torch.no_grad():
        output_ids = model.generate(
            input_ids,
            max_new_tokens=args.max_tokens,
            temperature=args.temperature,
            top_p=args.top_p,
            top_k=args.top_k,
        )
    elapsed = time.perf_counter() - start

    output = tokenizer.decode(output_ids[0], skip_special_tokens=True)
    print(output)
    print("-" * 50)
    print(f"Generated in {elapsed:.2f}s")


def cmd_chat(args):
    """Interactive chat mode."""
    from .models import load_ternary_model
    import torch

    model, tokenizer = load_ternary_model(args.model, mode=args.mode)

    print("\nLitespark-Inference Chat")
    print("Type 'quit' or 'exit' to end the conversation")
    print("-" * 50)

    system_prompt = "You are a helpful AI assistant."
    conversation = []

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

        # Build chat prompt (simple format)
        conversation.append(f"User: {user_input}")
        prompt = f"{system_prompt}\n\n" + "\n".join(conversation) + "\nAssistant:"

        # Generate response
        input_ids = tokenizer.encode(prompt, return_tensors='pt')

        print("\nAssistant: ", end="", flush=True)

        with torch.no_grad():
            output_ids = model.generate(
                input_ids,
                max_new_tokens=args.max_tokens,
                temperature=args.temperature,
                top_p=args.top_p,
                top_k=args.top_k,
            )

        # Extract just the new response
        response = tokenizer.decode(output_ids[0][input_ids.shape[1]:], skip_special_tokens=True)
        print(response)

        conversation.append(f"Assistant: {response}")


def cmd_benchmark(args):
    """Run performance benchmark."""
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
    model, tokenizer = load_ternary_model(args.model, mode=args.mode)
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


def main():
    parser = argparse.ArgumentParser(
        prog='litespark-inference',
        description='Litespark-Inference - Efficient inference for BitNet 1.58-bit models'
    )
    parser.add_argument('--version', action='version', version='%(prog)s 0.1.0')

    subparsers = parser.add_subparsers(dest='command', help='Commands')

    # generate command
    gen_parser = subparsers.add_parser('generate', help='Generate text from a prompt')
    gen_parser.add_argument('prompt', type=str, help='Input prompt')
    gen_parser.add_argument('--model', '-m', type=str, default='bitnet-2b', help='Model name (available: bitnet-2b)')
    gen_parser.add_argument('--mode', type=str, default='neon', choices=['neon', 'accelerate'], help='Kernel mode: neon (fast, int8) or accelerate (accurate, float32)')
    gen_parser.add_argument('--max-tokens', '-n', type=int, default=100, help='Max tokens to generate')
    gen_parser.add_argument('--temperature', '-t', type=float, default=0.7, help='Sampling temperature')
    gen_parser.add_argument('--top-p', type=float, default=0.9, help='Nucleus sampling threshold')
    gen_parser.add_argument('--top-k', type=int, default=50, help='Top-k sampling')
    gen_parser.set_defaults(func=cmd_generate)

    # chat command
    chat_parser = subparsers.add_parser('chat', help='Interactive chat mode')
    chat_parser.add_argument('--model', '-m', type=str, default='bitnet-2b', help='Model name (available: bitnet-2b)')
    chat_parser.add_argument('--mode', type=str, default='neon', choices=['neon', 'accelerate'], help='Kernel mode: neon (fast, int8) or accelerate (accurate, float32)')
    chat_parser.add_argument('--max-tokens', '-n', type=int, default=200, help='Max tokens per response')
    chat_parser.add_argument('--temperature', '-t', type=float, default=0.7, help='Sampling temperature')
    chat_parser.add_argument('--top-p', type=float, default=0.9, help='Nucleus sampling threshold')
    chat_parser.add_argument('--top-k', type=int, default=50, help='Top-k sampling')
    chat_parser.set_defaults(func=cmd_chat)

    # benchmark command
    bench_parser = subparsers.add_parser('benchmark', help='Run performance benchmark')
    bench_parser.add_argument('--model', '-m', type=str, default='bitnet-2b', help='Model name (available: bitnet-2b)')
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
