#!/usr/bin/env python3
"""
Benchmark Ternary GPT-2 Inference

Compares:
1. Original GPT-2 (float32)
2. Ternary GPT-2 (our kernels)

Metrics:
- Tokens per second
- Memory usage
- Perplexity on validation set
"""

import torch
import torch.nn.functional as F
import numpy as np
import time
import gc
import platform
from pathlib import Path
from typing import Dict, List, Optional
import argparse

from transformers import GPT2LMHeadModel, GPT2Tokenizer
from datasets import load_dataset

from ternary_gpt2 import create_ternary_gpt2


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
# Benchmark Functions
# ============================================================================

def benchmark_generation(
    model,
    tokenizer,
    prompts: List[str],
    max_new_tokens: int = 50,
    warmup: int = 2,
    num_runs: int = 5
) -> Dict:
    """Benchmark generation speed."""
    # Encode prompts
    input_ids_list = [tokenizer.encode(p, return_tensors='pt') for p in prompts]

    # Warmup
    print("  Warming up...")
    for _ in range(warmup):
        for input_ids in input_ids_list:
            with torch.no_grad():
                if hasattr(model, 'generate'):
                    _ = model.generate(input_ids, max_new_tokens=10)
                else:
                    # HuggingFace model
                    _ = model.generate(input_ids, max_new_tokens=10, do_sample=False)

    gc.collect()
    mem_before = get_memory_mb()

    # Benchmark
    print("  Benchmarking...")
    times = []
    total_tokens = 0

    for run in range(num_runs):
        for input_ids in input_ids_list:
            start = time.perf_counter()
            with torch.no_grad():
                if hasattr(model, 'generate') and not isinstance(model, GPT2LMHeadModel):
                    output = model.generate(input_ids, max_new_tokens=max_new_tokens)
                else:
                    output = model.generate(
                        input_ids,
                        max_new_tokens=max_new_tokens,
                        do_sample=False,
                        pad_token_id=tokenizer.eos_token_id
                    )
            end = time.perf_counter()

            new_tokens = output.shape[1] - input_ids.shape[1]
            times.append((end - start, new_tokens))
            total_tokens += new_tokens

    mem_after = get_memory_mb()

    # Compute stats
    total_time = sum(t for t, _ in times)
    tokens_per_sec = total_tokens / total_time

    return {
        'total_time_s': total_time,
        'total_tokens': total_tokens,
        'tokens_per_sec': tokens_per_sec,
        'avg_time_per_prompt_ms': (total_time / (num_runs * len(prompts))) * 1000,
        'memory_mb': mem_after,
        'memory_delta_mb': mem_after - mem_before
    }


def compute_perplexity(
    model,
    tokenizer,
    texts: List[str],
    max_length: int = 512,
    stride: int = 256
) -> float:
    """Compute perplexity on a list of texts."""
    total_loss = 0.0
    total_tokens = 0

    for text in texts:
        encodings = tokenizer(text, return_tensors='pt', truncation=True, max_length=max_length)
        input_ids = encodings.input_ids

        if input_ids.shape[1] < 2:
            continue

        with torch.no_grad():
            if isinstance(model, GPT2LMHeadModel):
                outputs = model(input_ids, labels=input_ids)
                loss = outputs.loss
            else:
                # Our ternary model
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

    avg_loss = total_loss / total_tokens
    perplexity = np.exp(avg_loss)

    return perplexity


def load_validation_data(num_samples: int = 100) -> List[str]:
    """Load validation data from WikiText."""
    print("Loading validation data...")
    try:
        dataset = load_dataset('wikitext', 'wikitext-2-raw-v1', split='validation')
        texts = [t for t in dataset['text'] if len(t.strip()) > 100][:num_samples]
        print(f"  Loaded {len(texts)} samples")
        return texts
    except Exception as e:
        print(f"  Failed to load dataset: {e}")
        print("  Using fallback samples")
        return [
            "The quick brown fox jumps over the lazy dog. " * 10,
            "In a galaxy far far away, there lived a wise old wizard. " * 10,
            "Machine learning is a subset of artificial intelligence. " * 10,
        ]


# ============================================================================
# Main Benchmark
# ============================================================================

def run_benchmark(
    model_name: str = 'gpt2',
    ternarize_method: str = 'threshold',
    threshold_ratio: float = 0.05,
    num_threads: int = None,
    max_new_tokens: int = 50,
    num_gen_runs: int = 5,
    num_ppl_samples: int = 50
):
    """Run full benchmark comparison."""
    print("=" * 70)
    print("Ternary GPT-2 Inference Benchmark")
    print("=" * 70)
    print(f"Model: {model_name}")
    print(f"Ternarization: {ternarize_method} (threshold={threshold_ratio})")
    print(f"Threads: {num_threads or torch.get_num_threads()}")
    print()

    # Load tokenizer
    print("Loading tokenizer...")
    tokenizer = GPT2Tokenizer.from_pretrained(model_name)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    # Test prompts
    prompts = [
        "The meaning of life is",
        "In the beginning, there was",
        "Machine learning has revolutionized",
        "The future of artificial intelligence",
        "Once upon a time in a land far away",
    ]

    # -------------------------------------------------------------------------
    # Benchmark Original Model
    # -------------------------------------------------------------------------
    print("\n" + "=" * 70)
    print("Original GPT-2 (float32)")
    print("=" * 70)

    print("Loading model...")
    original_model = GPT2LMHeadModel.from_pretrained(model_name)
    original_model.eval()

    mem_original = get_memory_mb()
    print(f"Model memory: {mem_original:.1f} MB")

    print("\nGeneration benchmark:")
    original_gen = benchmark_generation(
        original_model, tokenizer, prompts,
        max_new_tokens=max_new_tokens,
        num_runs=num_gen_runs
    )
    print(f"  Tokens/sec: {original_gen['tokens_per_sec']:.1f}")
    print(f"  Avg time per prompt: {original_gen['avg_time_per_prompt_ms']:.1f} ms")

    print("\nPerplexity benchmark:")
    val_texts = load_validation_data(num_ppl_samples)
    original_ppl = compute_perplexity(original_model, tokenizer, val_texts)
    print(f"  Perplexity: {original_ppl:.2f}")

    # Free memory
    del original_model
    gc.collect()

    # -------------------------------------------------------------------------
    # Benchmark Ternary Model
    # -------------------------------------------------------------------------
    print("\n" + "=" * 70)
    print("Ternary GPT-2 (our kernels)")
    print("=" * 70)

    print("Creating ternary model...")
    ternary_model = create_ternary_gpt2(
        model_name=model_name,
        method=ternarize_method,
        threshold_ratio=threshold_ratio,
        num_threads=num_threads
    )
    ternary_model.eval()

    mem_ternary = get_memory_mb()
    print(f"Model memory: {mem_ternary:.1f} MB")

    print("\nGeneration benchmark:")
    ternary_gen = benchmark_generation(
        ternary_model, tokenizer, prompts,
        max_new_tokens=max_new_tokens,
        num_runs=num_gen_runs
    )
    print(f"  Tokens/sec: {ternary_gen['tokens_per_sec']:.1f}")
    print(f"  Avg time per prompt: {ternary_gen['avg_time_per_prompt_ms']:.1f} ms")

    print("\nPerplexity benchmark:")
    ternary_ppl = compute_perplexity(ternary_model, tokenizer, val_texts)
    print(f"  Perplexity: {ternary_ppl:.2f}")

    # -------------------------------------------------------------------------
    # Summary
    # -------------------------------------------------------------------------
    print("\n" + "=" * 70)
    print("SUMMARY")
    print("=" * 70)

    speedup = ternary_gen['tokens_per_sec'] / original_gen['tokens_per_sec']
    ppl_ratio = ternary_ppl / original_ppl

    print(f"\n{'Metric':<25} {'Original':<15} {'Ternary':<15} {'Ratio':<10}")
    print("-" * 70)
    print(f"{'Tokens/sec':<25} {original_gen['tokens_per_sec']:<15.1f} {ternary_gen['tokens_per_sec']:<15.1f} {speedup:<10.2f}x")
    print(f"{'Time/prompt (ms)':<25} {original_gen['avg_time_per_prompt_ms']:<15.1f} {ternary_gen['avg_time_per_prompt_ms']:<15.1f} {original_gen['avg_time_per_prompt_ms']/ternary_gen['avg_time_per_prompt_ms']:<10.2f}x")
    print(f"{'Perplexity':<25} {original_ppl:<15.2f} {ternary_ppl:<15.2f} {ppl_ratio:<10.2f}x")
    print(f"{'Memory (MB)':<25} {mem_original:<15.1f} {mem_ternary:<15.1f} {mem_ternary/mem_original:<10.2f}x")

    print("\n" + "=" * 70)

    # Sample generation
    print("\nSample generation comparison:")
    print("-" * 70)

    prompt = "The future of AI is"
    input_ids = tokenizer.encode(prompt, return_tensors='pt')

    print(f"Prompt: {prompt}\n")

    # Reload original for comparison
    original_model = GPT2LMHeadModel.from_pretrained(model_name)
    original_model.eval()

    with torch.no_grad():
        original_output = original_model.generate(
            input_ids, max_new_tokens=30, do_sample=False,
            pad_token_id=tokenizer.eos_token_id
        )
        ternary_output = ternary_model.generate(
            input_ids, max_new_tokens=30, temperature=1.0, top_k=0, top_p=1.0
        )

    print(f"Original: {tokenizer.decode(original_output[0], skip_special_tokens=True)}")
    print(f"\nTernary:  {tokenizer.decode(ternary_output[0], skip_special_tokens=True)}")

    return {
        'original': {
            'tokens_per_sec': original_gen['tokens_per_sec'],
            'perplexity': original_ppl,
            'memory_mb': mem_original
        },
        'ternary': {
            'tokens_per_sec': ternary_gen['tokens_per_sec'],
            'perplexity': ternary_ppl,
            'memory_mb': mem_ternary
        },
        'speedup': speedup,
        'ppl_ratio': ppl_ratio
    }


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Benchmark ternary GPT-2')
    parser.add_argument('--model', default='gpt2', help='Model name')
    parser.add_argument('--method', default='threshold', choices=['threshold', 'topk'])
    parser.add_argument('--threshold', type=float, default=0.05)
    parser.add_argument('--threads', type=int, default=None)
    parser.add_argument('--max-tokens', type=int, default=50)
    parser.add_argument('--gen-runs', type=int, default=5)
    parser.add_argument('--ppl-samples', type=int, default=50)

    args = parser.parse_args()

    run_benchmark(
        model_name=args.model,
        ternarize_method=args.method,
        threshold_ratio=args.threshold,
        num_threads=args.threads,
        max_new_tokens=args.max_tokens,
        num_gen_runs=args.gen_runs,
        num_ppl_samples=args.ppl_samples
    )
