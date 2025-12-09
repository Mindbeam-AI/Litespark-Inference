#!/usr/bin/env python3
"""
Interactive Demo for Ternary Model Inference

Supports:
- BitNet b1.58 models
- MatMul-free LM models
- Both x86_64 (VNNI) and ARM64 (Graviton/NEON) architectures

Example usage:
    # Interactive mode
    python inference/demo.py --model mmfreelm-370m

    # Single prompt
    python inference/demo.py --model bitnet-3b --prompt "The future of AI is"

    # With sampling
    python inference/demo.py --model mmfreelm-370m --prompt "Once upon a time" \
        --temperature 0.8 --top-k 40
"""

import argparse
import sys
import time
import platform
from pathlib import Path

# Add parent to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent))

import torch

from ternary_models import load_ternary_model, list_models, get_arch_info, AVAILABLE_MODELS
from generation import (
    generate,
    generate_with_metrics,
    GenerationConfig,
    benchmark_ttft,
    benchmark_throughput,
    stream_generate
)


def print_header():
    """Print demo header with system info."""
    arch_info = get_arch_info()

    print("=" * 70)
    print("Ternary Model Inference Demo")
    print("=" * 70)
    print(f"Platform: {platform.system()} {platform.machine()}")
    print(f"PyTorch: {torch.__version__}")
    print(f"Threads: {torch.get_num_threads()}")

    if arch_info['is_x86_64']:
        print(f"Kernel: AVX-512 VNNI (x86_64)")
    elif arch_info['is_graviton']:
        print(f"Kernel: NEON SDOT (AWS Graviton)")
    elif arch_info['is_apple_silicon']:
        print(f"Kernel: NEON SDOT (Apple Silicon)")
    else:
        print(f"Kernel: {arch_info['kernel_type']}")

    print("=" * 70)
    print()


def demo_single_prompt(model, tokenizer, args):
    """Generate for a single prompt with metrics."""
    print(f"\nPrompt: {args.prompt}")
    print("-" * 50)

    if args.stream:
        # Streaming mode
        print("Output: ", end="", flush=True)
        ttft = None
        token_count = 0

        for token_text, is_first, time_ms in stream_generate(
            model, tokenizer, args.prompt,
            max_new_tokens=args.max_tokens,
            temperature=args.temperature
        ):
            print(token_text, end="", flush=True)
            if is_first:
                ttft = time_ms
            token_count += 1

        print("\n")
        print(f"TTFT: {ttft:.1f} ms")
        print(f"Tokens: {token_count}")
    else:
        # Non-streaming mode with full metrics
        output, metrics = generate(
            model, tokenizer, args.prompt,
            max_new_tokens=args.max_tokens,
            temperature=args.temperature,
            top_k=args.top_k,
            top_p=args.top_p,
            do_sample=args.temperature > 0,
            return_metrics=True
        )

        print(f"Output: {output}")
        print()
        print(f"TTFT (Time to First Token): {metrics.ttft_ms:.2f} ms")
        print(f"Total time: {metrics.total_time_ms:.2f} ms")
        print(f"Tokens generated: {metrics.num_tokens}")
        print(f"Throughput: {metrics.tokens_per_sec:.1f} tokens/sec")


def demo_interactive(model, tokenizer, args):
    """Interactive chat mode."""
    print("\nInteractive mode (type 'quit' to exit, 'bench' for benchmark)")
    print("-" * 50)

    while True:
        try:
            prompt = input("\n> ").strip()
        except (EOFError, KeyboardInterrupt):
            print("\nExiting...")
            break

        if not prompt:
            continue

        if prompt.lower() == 'quit':
            break

        if prompt.lower() == 'bench':
            print("\nRunning quick benchmark...")
            test_prompts = [
                "The meaning of life is",
                "In the year 2050,",
                "Machine learning has"
            ]
            ttft_results = benchmark_ttft(model, tokenizer, test_prompts, warmup=2, num_runs=3)
            print(f"\nTTFT Results:")
            print(f"  Mean: {ttft_results['mean_ttft_ms']:.2f} ms")
            print(f"  P50:  {ttft_results['p50_ttft_ms']:.2f} ms")
            print(f"  P95:  {ttft_results['p95_ttft_ms']:.2f} ms")

            throughput_results = benchmark_throughput(
                model, tokenizer, test_prompts[0],
                num_tokens=30, warmup=2, num_runs=3
            )
            print(f"\nThroughput: {throughput_results['mean_tokens_per_sec']:.1f} tokens/sec")
            continue

        # Generate response
        if args.stream:
            print("\n", end="")
            for token_text, is_first, time_ms in stream_generate(
                model, tokenizer, prompt,
                max_new_tokens=args.max_tokens,
                temperature=args.temperature
            ):
                print(token_text, end="", flush=True)
            print()
        else:
            output, metrics = generate(
                model, tokenizer, prompt,
                max_new_tokens=args.max_tokens,
                temperature=args.temperature,
                top_k=args.top_k,
                top_p=args.top_p,
                do_sample=args.temperature > 0,
                return_metrics=True
            )
            print(f"\n{output}")
            print(f"\n[TTFT: {metrics.ttft_ms:.1f}ms | {metrics.tokens_per_sec:.1f} tok/s | {metrics.num_tokens} tokens]")


def demo_benchmark(model, tokenizer, args):
    """Run comprehensive benchmark."""
    print("\nRunning comprehensive benchmark...")
    print("=" * 70)

    # Test prompts with varying lengths
    test_prompts = [
        "The",  # Very short
        "The future of AI is",  # Short
        "In a world where technology has advanced beyond our wildest dreams,",  # Medium
        "The research paper presents a novel approach to natural language processing that combines " \
        "transformer architectures with efficient quantization techniques",  # Long
    ]

    # TTFT Benchmark
    print("\n1. Time-to-First-Token (TTFT) Benchmark")
    print("-" * 50)
    ttft_results = benchmark_ttft(model, tokenizer, test_prompts, warmup=3, num_runs=5)

    print(f"\n  Mean TTFT:  {ttft_results['mean_ttft_ms']:.2f} ms")
    print(f"  Min TTFT:   {ttft_results['min_ttft_ms']:.2f} ms")
    print(f"  Max TTFT:   {ttft_results['max_ttft_ms']:.2f} ms")
    print(f"  P50 TTFT:   {ttft_results['p50_ttft_ms']:.2f} ms")
    print(f"  P95 TTFT:   {ttft_results['p95_ttft_ms']:.2f} ms")
    print(f"  P99 TTFT:   {ttft_results['p99_ttft_ms']:.2f} ms")

    # Per prompt length analysis
    print("\n  By prompt length:")
    for measurement in ttft_results['measurements'][:len(test_prompts)]:
        print(f"    {measurement['prompt_tokens']:3d} tokens -> {measurement['ttft_ms']:.2f} ms")

    # Throughput Benchmark
    print("\n2. Throughput Benchmark")
    print("-" * 50)
    throughput_results = benchmark_throughput(
        model, tokenizer, test_prompts[1],
        num_tokens=args.max_tokens, warmup=3, num_runs=5
    )

    print(f"\n  Tokens per second: {throughput_results['mean_tokens_per_sec']:.1f}")
    print(f"  Mean total time:   {throughput_results['mean_total_time_ms']:.2f} ms")
    print(f"  Mean TTFT:         {throughput_results['mean_ttft_ms']:.2f} ms")
    print(f"  Tokens per run:    {throughput_results['num_tokens_per_run']}")

    # Generation quality check
    print("\n3. Sample Generations")
    print("-" * 50)
    for prompt in test_prompts[:2]:
        output = generate(
            model, tokenizer, prompt,
            max_new_tokens=30,
            temperature=0,
            do_sample=False
        )
        print(f"\n  Prompt: {prompt}")
        print(f"  Output: {output}")

    print("\n" + "=" * 70)
    print("Benchmark complete!")


def main():
    parser = argparse.ArgumentParser(
        description='Interactive demo for ternary model inference',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python demo.py --list                           # List available models
  python demo.py --model mmfreelm-370m            # Interactive mode
  python demo.py --model bitnet-3b --prompt "Hi"  # Single prompt
  python demo.py --model mmfreelm-370m --bench    # Run benchmark
  python demo.py --model mmfreelm-370m --stream   # Streaming output
"""
    )

    parser.add_argument('--list', action='store_true',
                        help='List available models')
    parser.add_argument('--model', type=str, default='mmfreelm-370m',
                        choices=list(AVAILABLE_MODELS.keys()),
                        help='Model to use (default: mmfreelm-370m)')
    parser.add_argument('--prompt', type=str, default=None,
                        help='Single prompt to generate (omit for interactive mode)')
    parser.add_argument('--max-tokens', type=int, default=50,
                        help='Maximum tokens to generate (default: 50)')
    parser.add_argument('--temperature', type=float, default=0.0,
                        help='Sampling temperature (0 = greedy, default: 0)')
    parser.add_argument('--top-k', type=int, default=50,
                        help='Top-k sampling parameter (default: 50)')
    parser.add_argument('--top-p', type=float, default=1.0,
                        help='Top-p (nucleus) sampling (default: 1.0)')
    parser.add_argument('--stream', action='store_true',
                        help='Stream output tokens')
    parser.add_argument('--bench', action='store_true',
                        help='Run comprehensive benchmark')
    parser.add_argument('--threads', type=int, default=None,
                        help='Number of threads to use')

    args = parser.parse_args()

    if args.list:
        list_models()
        return

    print_header()

    # Load model
    print(f"Loading model: {args.model}")
    print("This may take a moment for first-time downloads...")
    print()

    try:
        model, tokenizer = load_ternary_model(args.model, args.threads)
    except Exception as e:
        print(f"\nError loading model: {e}")
        print("\nTip: Make sure you have transformers and datasets installed:")
        print("  pip install transformers datasets")
        return 1

    print(f"\nModel loaded successfully!")
    print(f"Vocab size: {model.config.vocab_size}")
    print(f"Hidden size: {model.config.hidden_size}")
    print(f"Layers: {model.config.num_layers if hasattr(model.config, 'num_layers') else 'N/A'}")

    if args.bench:
        demo_benchmark(model, tokenizer, args)
    elif args.prompt:
        demo_single_prompt(model, tokenizer, args)
    else:
        demo_interactive(model, tokenizer, args)

    return 0


if __name__ == '__main__':
    sys.exit(main() or 0)
