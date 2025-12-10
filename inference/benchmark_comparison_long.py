#!/usr/bin/env python3
"""
Benchmark comparison with scaling tests: longer prompts and batch sizes.

Tests:
1. Prefill scaling: How does TTFT scale with prompt length?
2. Batch scaling: How does throughput scale with batch size?
3. Generation scaling: Throughput at different generation lengths

This complements benchmark_comparison_short.py which does quick comparisons.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import time
import sys
import gc
import platform
import resource
from pathlib import Path
from typing import Dict, List
import numpy as np

sys.path.insert(0, str(Path(__file__).parent.parent))

from inference.ternary_models import (
    BitNet, BitNetConfig, load_ternary_model, get_arch_info,
)

# Import baseline models from short benchmark
from inference.benchmark_comparison_short import (
    PyTorchBitNet, HuggingFaceModelWrapper,
    get_memory_mb, get_peak_memory_mb
)


# ============================================================================
# Configuration
# ============================================================================

# Prompt lengths to test (in tokens)
PROMPT_LENGTHS = [32, 128, 256, 512]

# Batch sizes to test
BATCH_SIZES = [1, 2, 4, 8, 16]

# Generation lengths to test
GEN_LENGTHS = [20, 50, 100]

# Warmup and benchmark iterations
WARMUP_ITERS = 2
BENCH_ITERS = 3


# ============================================================================
# Utility Functions
# ============================================================================

def create_prompt_of_length(tokenizer, target_length: int) -> torch.Tensor:
    """Create a prompt with approximately target_length tokens."""
    # Use a repeating text pattern
    base_text = "The quick brown fox jumps over the lazy dog. "
    text = base_text * (target_length // 10 + 1)

    input_ids = tokenizer.encode(text, return_tensors='pt', truncation=True, max_length=target_length)

    # Pad if needed
    if input_ids.shape[1] < target_length:
        pad_length = target_length - input_ids.shape[1]
        padding = torch.full((1, pad_length), tokenizer.pad_token_id or 0)
        input_ids = torch.cat([input_ids, padding], dim=1)

    return input_ids[:, :target_length]


def create_batch(input_ids: torch.Tensor, batch_size: int) -> torch.Tensor:
    """Replicate input_ids to create a batch."""
    return input_ids.repeat(batch_size, 1)


# ============================================================================
# Benchmark Functions
# ============================================================================

def benchmark_prefill(model, input_ids: torch.Tensor, warmup: int = WARMUP_ITERS, iters: int = BENCH_ITERS) -> Dict:
    """Benchmark prefill (forward pass) time."""
    model.eval()

    # Warmup
    with torch.no_grad():
        for _ in range(warmup):
            _ = model(input_ids)

    # Benchmark
    times = []
    with torch.no_grad():
        for _ in range(iters):
            start = time.perf_counter()
            _ = model(input_ids)
            times.append((time.perf_counter() - start) * 1000)

    return {
        'mean_ms': np.mean(times),
        'std_ms': np.std(times),
        'min_ms': np.min(times),
    }


def benchmark_generation(model, input_ids: torch.Tensor, num_tokens: int,
                         warmup: int = WARMUP_ITERS, iters: int = BENCH_ITERS) -> Dict:
    """Benchmark token generation throughput."""
    model.eval()
    batch_size = input_ids.shape[0]

    # Warmup
    with torch.no_grad():
        for _ in range(warmup):
            generated = input_ids.clone()
            for _ in range(min(5, num_tokens)):
                logits = model(generated)
                next_token = logits[:, -1, :].argmax(dim=-1, keepdim=True)
                generated = torch.cat([generated, next_token], dim=1)

    # Benchmark
    times = []
    with torch.no_grad():
        for _ in range(iters):
            generated = input_ids.clone()
            start = time.perf_counter()
            for _ in range(num_tokens):
                logits = model(generated)
                next_token = logits[:, -1, :].argmax(dim=-1, keepdim=True)
                generated = torch.cat([generated, next_token], dim=1)
            total_time = (time.perf_counter() - start) * 1000
            times.append(total_time)

    mean_time = np.mean(times)
    tokens_per_sec = (num_tokens * batch_size) / (mean_time / 1000)

    return {
        'mean_ms': mean_time,
        'std_ms': np.std(times),
        'tokens_per_sec': tokens_per_sec,
        'tokens_per_sec_per_batch': num_tokens / (mean_time / 1000),
    }


# ============================================================================
# Scaling Tests
# ============================================================================

def test_prefill_scaling(models: Dict, tokenizer, prompt_lengths: List[int]):
    """Test how prefill time scales with prompt length."""
    print("\n" + "=" * 80)
    print("PREFILL SCALING TEST (Time vs Prompt Length)")
    print("=" * 80)

    results = {name: {} for name in models.keys()}

    for length in prompt_lengths:
        print(f"\n  Prompt length: {length} tokens")
        input_ids = create_prompt_of_length(tokenizer, length)

        for name, model in models.items():
            try:
                r = benchmark_prefill(model, input_ids)
                results[name][length] = r
                print(f"    {name:<25}: {r['mean_ms']:>8.1f} ms")
            except Exception as e:
                print(f"    {name:<25}: FAILED ({e})")
                results[name][length] = None

    # Print summary table
    print("\n" + "-" * 80)
    print("Summary: Prefill Time (ms)")
    print("-" * 80)
    print(f"{'Model':<25}", end='')
    for length in prompt_lengths:
        print(f"{length:>10} tok", end='')
    print()
    print("-" * 80)

    for name in models.keys():
        print(f"{name:<25}", end='')
        for length in prompt_lengths:
            r = results[name].get(length)
            if r:
                print(f"{r['mean_ms']:>10.1f}  ", end='')
            else:
                print(f"{'N/A':>10}  ", end='')
        print()

    # Print speedups vs PyTorch
    if 'PyTorch (ternary)' in results and 'VNNI Kernels' in results:
        print("\n" + "-" * 80)
        print("Speedup: VNNI vs PyTorch")
        print("-" * 80)
        print(f"{'Prompt Length':<20}", end='')
        for length in prompt_lengths:
            print(f"{length:>10} tok", end='')
        print()
        print(f"{'Speedup':<20}", end='')
        for length in prompt_lengths:
            vnni = results['VNNI Kernels'].get(length)
            pytorch = results['PyTorch (ternary)'].get(length)
            if vnni and pytorch:
                speedup = pytorch['mean_ms'] / vnni['mean_ms']
                print(f"{speedup:>10.2f}x ", end='')
            else:
                print(f"{'N/A':>10}  ", end='')
        print()

    return results


def test_batch_scaling(models: Dict, tokenizer, batch_sizes: List[int], prompt_length: int = 32):
    """Test how throughput scales with batch size."""
    print("\n" + "=" * 80)
    print(f"BATCH SCALING TEST (Throughput vs Batch Size, prompt={prompt_length} tokens)")
    print("=" * 80)

    results = {name: {} for name in models.keys()}
    base_input = create_prompt_of_length(tokenizer, prompt_length)

    for batch_size in batch_sizes:
        print(f"\n  Batch size: {batch_size}")
        input_ids = create_batch(base_input, batch_size)

        for name, model in models.items():
            try:
                r = benchmark_prefill(model, input_ids)
                # Calculate throughput: tokens processed per second
                tokens_per_sec = (batch_size * prompt_length) / (r['mean_ms'] / 1000)
                r['tokens_per_sec'] = tokens_per_sec
                results[name][batch_size] = r
                print(f"    {name:<25}: {r['mean_ms']:>8.1f} ms, {tokens_per_sec:>8.0f} tok/s")
            except Exception as e:
                print(f"    {name:<25}: FAILED ({e})")
                results[name][batch_size] = None

    # Print summary table
    print("\n" + "-" * 80)
    print("Summary: Prefill Throughput (tokens/sec)")
    print("-" * 80)
    print(f"{'Model':<25}", end='')
    for bs in batch_sizes:
        print(f"{'B='+str(bs):>12}", end='')
    print()
    print("-" * 80)

    for name in models.keys():
        print(f"{name:<25}", end='')
        for bs in batch_sizes:
            r = results[name].get(bs)
            if r:
                print(f"{r['tokens_per_sec']:>12.0f}", end='')
            else:
                print(f"{'N/A':>12}", end='')
        print()

    return results


def test_generation_scaling(models: Dict, tokenizer, gen_lengths: List[int], prompt_length: int = 32):
    """Test generation throughput at different generation lengths."""
    print("\n" + "=" * 80)
    print(f"GENERATION SCALING TEST (prompt={prompt_length} tokens)")
    print("=" * 80)

    results = {name: {} for name in models.keys()}
    input_ids = create_prompt_of_length(tokenizer, prompt_length)

    for gen_length in gen_lengths:
        print(f"\n  Generate {gen_length} tokens:")

        for name, model in models.items():
            try:
                r = benchmark_generation(model, input_ids, gen_length)
                results[name][gen_length] = r
                print(f"    {name:<25}: {r['mean_ms']:>8.1f} ms, {r['tokens_per_sec']:>6.1f} tok/s")
            except Exception as e:
                print(f"    {name:<25}: FAILED ({e})")
                results[name][gen_length] = None

    # Print summary table
    print("\n" + "-" * 80)
    print("Summary: Generation Throughput (tokens/sec)")
    print("-" * 80)
    print(f"{'Model':<25}", end='')
    for gl in gen_lengths:
        print(f"{'Gen '+str(gl):>12}", end='')
    print()
    print("-" * 80)

    for name in models.keys():
        print(f"{name:<25}", end='')
        for gl in gen_lengths:
            r = results[name].get(gl)
            if r:
                print(f"{r['tokens_per_sec']:>12.1f}", end='')
            else:
                print(f"{'N/A':>12}", end='')
        print()

    return results


# ============================================================================
# Main
# ============================================================================

def main():
    print("=" * 80)
    print("BitNet Scaling Benchmark: Longer Prompts & Batch Sizes")
    print("=" * 80)

    arch = get_arch_info()
    print(f"Platform: {arch['machine']}")
    print(f"Kernel: {arch['kernel_type']}")
    print(f"Threads: {torch.get_num_threads()}")
    print()

    MODEL_NAME = 'microsoft/bitnet-b1.58-2B-4T-bf16'
    HF_BITNET_MODEL = 'microsoft/bitnet-b1.58-2B-4T'

    # Load tokenizer
    from transformers import AutoTokenizer
    print("Loading tokenizer...")
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token_id = tokenizer.eos_token_id

    # Load models
    models = {}

    print("\nLoading VNNI kernel model...")
    vnni_model, _ = load_ternary_model('bitnet-2b')
    vnni_model.eval()
    models['VNNI Kernels'] = vnni_model

    print("Loading PyTorch baseline model...")
    pytorch_model = PyTorchBitNet.from_safetensors(MODEL_NAME)
    pytorch_model.eval()
    models['PyTorch (ternary)'] = pytorch_model

    print("Loading HuggingFace BitNet model...")
    try:
        hf_model = HuggingFaceModelWrapper.from_pretrained(HF_BITNET_MODEL)
        hf_model.eval()
        models['HuggingFace (BitNet)'] = hf_model
    except Exception as e:
        print(f"  [!] Failed to load HuggingFace model: {e}")

    # Run scaling tests
    print("\n" + "=" * 80)
    print("RUNNING SCALING TESTS")
    print("=" * 80)

    # 1. Prefill scaling
    prefill_results = test_prefill_scaling(models, tokenizer, PROMPT_LENGTHS)

    # 2. Batch scaling
    batch_results = test_batch_scaling(models, tokenizer, BATCH_SIZES, prompt_length=128)

    # 3. Generation scaling
    gen_results = test_generation_scaling(models, tokenizer, GEN_LENGTHS, prompt_length=32)

    # Final summary
    print("\n" + "=" * 80)
    print("FINAL SUMMARY")
    print("=" * 80)

    print("\nKey findings:")

    # Prefill speedup at longest prompt
    if 'VNNI Kernels' in prefill_results and 'PyTorch (ternary)' in prefill_results:
        longest = PROMPT_LENGTHS[-1]
        vnni = prefill_results['VNNI Kernels'].get(longest)
        pytorch = prefill_results['PyTorch (ternary)'].get(longest)
        if vnni and pytorch:
            speedup = pytorch['mean_ms'] / vnni['mean_ms']
            print(f"  - Prefill @ {longest} tokens: {speedup:.2f}x faster than PyTorch")

    # Batch throughput at largest batch
    if 'VNNI Kernels' in batch_results and 'PyTorch (ternary)' in batch_results:
        largest_bs = BATCH_SIZES[-1]
        vnni = batch_results['VNNI Kernels'].get(largest_bs)
        pytorch = batch_results['PyTorch (ternary)'].get(largest_bs)
        if vnni and pytorch:
            speedup = vnni['tokens_per_sec'] / pytorch['tokens_per_sec']
            print(f"  - Batch={largest_bs} throughput: {speedup:.2f}x faster than PyTorch")

    # Generation throughput
    if 'VNNI Kernels' in gen_results and 'PyTorch (ternary)' in gen_results:
        gen_len = GEN_LENGTHS[1]  # Middle value
        vnni = gen_results['VNNI Kernels'].get(gen_len)
        pytorch = gen_results['PyTorch (ternary)'].get(gen_len)
        if vnni and pytorch:
            speedup = vnni['tokens_per_sec'] / pytorch['tokens_per_sec']
            print(f"  - Generation @ {gen_len} tokens: {speedup:.2f}x faster than PyTorch")

    print("\n" + "=" * 80)
    print("Benchmark complete!")
    print("=" * 80)


if __name__ == '__main__':
    main()
