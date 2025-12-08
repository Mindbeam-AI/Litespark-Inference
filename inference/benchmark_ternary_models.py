#!/usr/bin/env python3
"""
Benchmark Pre-trained Ternary Models

Tests inference speed and quality for:
1. MatMul-free LM (370M, 1.3B, 2.7B)
2. BitNet b1.58 (3B, Large)
3. Microsoft BitNet (2B)

Compares our VNNI kernels vs original HuggingFace implementation.
"""

import torch
import torch.nn.functional as F
import numpy as np
import time
import gc
import platform
import argparse
from pathlib import Path
from typing import Dict, List

from ternary_models import load_ternary_model, list_models, AVAILABLE_MODELS


# ============================================================================
# Memory Tracking
# ============================================================================

def get_memory_mb():
    """Get process RSS in MB."""
    try:
        with open('/proc/self/status', 'r') as f:
            for line in f:
                if line.startswith('VmRSS:'):
                    return int(line.split()[1]) / 1024.0
    except:
        pass
    try:
        import resource
        usage = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
        if platform.system() == 'Darwin':
            return usage / (1024 * 1024)
        return usage / 1024
    except:
        return 0.0


# ============================================================================
# Benchmarking
# ============================================================================

def benchmark_generation(
    model,
    tokenizer,
    prompts: List[str],
    max_new_tokens: int = 50,
    warmup: int = 2,
    num_runs: int = 3
) -> Dict:
    """Benchmark generation speed."""
    input_ids_list = [tokenizer.encode(p, return_tensors='pt') for p in prompts]

    # Warmup
    print("  Warming up...")
    for _ in range(warmup):
        for input_ids in input_ids_list[:1]:
            with torch.no_grad():
                for _ in range(5):
                    _ = model(input_ids)

    gc.collect()
    mem_before = get_memory_mb()

    # Benchmark forward passes
    print("  Benchmarking forward passes...")
    times = []
    total_tokens = 0

    for run in range(num_runs):
        for input_ids in input_ids_list:
            seq_len = input_ids.shape[1]
            start = time.perf_counter()
            with torch.no_grad():
                # Run multiple forward passes to simulate generation
                for _ in range(max_new_tokens):
                    logits = model(input_ids)
                    # Get next token (greedy)
                    next_token = logits[:, -1, :].argmax(dim=-1, keepdim=True)
                    input_ids = torch.cat([input_ids, next_token], dim=1)
            end = time.perf_counter()

            new_tokens = max_new_tokens
            times.append((end - start, new_tokens))
            total_tokens += new_tokens

    mem_after = get_memory_mb()

    total_time = sum(t for t, _ in times)
    tokens_per_sec = total_tokens / total_time

    return {
        'total_time_s': total_time,
        'total_tokens': total_tokens,
        'tokens_per_sec': tokens_per_sec,
        'avg_time_per_prompt_ms': (total_time / (num_runs * len(prompts))) * 1000,
        'memory_mb': mem_after,
    }


def benchmark_perplexity(
    model,
    tokenizer,
    texts: List[str],
    max_length: int = 512
) -> float:
    """Compute perplexity on texts."""
    total_loss = 0.0
    total_tokens = 0

    for text in texts:
        encodings = tokenizer(text, return_tensors='pt', truncation=True, max_length=max_length)
        input_ids = encodings.input_ids

        if input_ids.shape[1] < 2:
            continue

        with torch.no_grad():
            logits = model(input_ids)
            shift_logits = logits[..., :-1, :].contiguous()
            shift_labels = input_ids[..., 1:].contiguous()
            loss = F.cross_entropy(
                shift_logits.view(-1, shift_logits.size(-1)),
                shift_labels.view(-1)
            )

        total_loss += loss.item() * (input_ids.shape[1] - 1)
        total_tokens += input_ids.shape[1] - 1

    if total_tokens == 0:
        return float('inf')

    return np.exp(total_loss / total_tokens)


def load_validation_data(num_samples: int = 20) -> List[str]:
    """Load validation data."""
    print("Loading validation data...")
    try:
        from datasets import load_dataset
        dataset = load_dataset('wikitext', 'wikitext-2-raw-v1', split='validation')
        texts = [t for t in dataset['text'] if len(t.strip()) > 100][:num_samples]
        print(f"  Loaded {len(texts)} samples")
        return texts
    except Exception as e:
        print(f"  Failed to load dataset: {e}")
        return [
            "The quick brown fox jumps over the lazy dog. " * 10,
            "Machine learning is transforming the world. " * 10,
        ]


def sample_generation(model, tokenizer, prompt: str, max_tokens: int = 30) -> str:
    """Generate sample text."""
    input_ids = tokenizer.encode(prompt, return_tensors='pt')

    with torch.no_grad():
        for _ in range(max_tokens):
            logits = model(input_ids)
            next_token = logits[:, -1, :].argmax(dim=-1, keepdim=True)
            input_ids = torch.cat([input_ids, next_token], dim=1)

            if next_token.item() == tokenizer.eos_token_id:
                break

    return tokenizer.decode(input_ids[0], skip_special_tokens=True)


# ============================================================================
# Main Benchmark
# ============================================================================

def run_benchmark(
    model_key: str,
    num_threads: int = None,
    max_new_tokens: int = 30,
    num_runs: int = 3,
    num_ppl_samples: int = 20
):
    """Run benchmark for a single model."""
    print("=" * 70)
    print(f"Benchmarking: {model_key}")
    print("=" * 70)

    # Load model
    model, tokenizer = load_ternary_model(model_key, num_threads)
    mem_loaded = get_memory_mb()
    print(f"Model memory: {mem_loaded:.1f} MB")

    # Test prompts
    prompts = [
        "The meaning of life is",
        "In a galaxy far away,",
        "Machine learning has",
    ]

    # Generation benchmark
    print("\nGeneration benchmark:")
    gen_results = benchmark_generation(
        model, tokenizer, prompts,
        max_new_tokens=max_new_tokens,
        num_runs=num_runs
    )
    print(f"  Tokens/sec: {gen_results['tokens_per_sec']:.1f}")
    print(f"  Avg time per prompt: {gen_results['avg_time_per_prompt_ms']:.1f} ms")

    # Perplexity benchmark
    print("\nPerplexity benchmark:")
    val_texts = load_validation_data(num_ppl_samples)
    ppl = benchmark_perplexity(model, tokenizer, val_texts)
    print(f"  Perplexity: {ppl:.2f}")

    # Sample generation
    print("\nSample generation:")
    prompt = "The future of AI is"
    output = sample_generation(model, tokenizer, prompt, max_tokens=30)
    print(f"  Prompt: {prompt}")
    print(f"  Output: {output}")

    return {
        'model': model_key,
        'tokens_per_sec': gen_results['tokens_per_sec'],
        'avg_time_ms': gen_results['avg_time_per_prompt_ms'],
        'perplexity': ppl,
        'memory_mb': mem_loaded,
    }


def run_all_benchmarks(
    models: List[str] = None,
    num_threads: int = None,
    **kwargs
):
    """Run benchmarks for all specified models."""
    if models is None:
        models = list(AVAILABLE_MODELS.keys())

    results = []

    for model_key in models:
        try:
            result = run_benchmark(model_key, num_threads, **kwargs)
            results.append(result)
        except Exception as e:
            print(f"\nError benchmarking {model_key}: {e}")
            import traceback
            traceback.print_exc()

        gc.collect()
        print("\n")

    # Summary
    if results:
        print("=" * 70)
        print("SUMMARY")
        print("=" * 70)
        print(f"\n{'Model':<20} {'Tok/sec':<12} {'Time (ms)':<12} {'PPL':<12} {'Mem (MB)':<12}")
        print("-" * 70)
        for r in results:
            print(f"{r['model']:<20} {r['tokens_per_sec']:<12.1f} {r['avg_time_ms']:<12.1f} {r['perplexity']:<12.2f} {r['memory_mb']:<12.1f}")

    return results


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Benchmark ternary models')
    parser.add_argument('--model', type=str, default=None,
                        help='Model to benchmark (or "all" for all models)')
    parser.add_argument('--list', action='store_true', help='List available models')
    parser.add_argument('--threads', type=int, default=None)
    parser.add_argument('--max-tokens', type=int, default=30)
    parser.add_argument('--runs', type=int, default=3)
    parser.add_argument('--ppl-samples', type=int, default=20)

    args = parser.parse_args()

    if args.list:
        list_models()
    elif args.model == 'all':
        run_all_benchmarks(
            num_threads=args.threads,
            max_new_tokens=args.max_tokens,
            num_runs=args.runs,
            num_ppl_samples=args.ppl_samples
        )
    elif args.model:
        run_benchmark(
            args.model,
            num_threads=args.threads,
            max_new_tokens=args.max_tokens,
            num_runs=args.runs,
            num_ppl_samples=args.ppl_samples
        )
    else:
        # Default: try smallest model
        print("No model specified. Trying mmfreelm-370m...")
        run_benchmark(
            'mmfreelm-370m',
            num_threads=args.threads,
            max_new_tokens=args.max_tokens,
            num_runs=args.runs,
            num_ppl_samples=args.ppl_samples
        )
