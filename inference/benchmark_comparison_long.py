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
import json
from pathlib import Path
from datetime import datetime
from typing import Dict, List, Any
import numpy as np

sys.path.insert(0, str(Path(__file__).parent.parent))

# Try to import matplotlib for graphs
try:
    import matplotlib
    matplotlib.use('Agg')  # Non-interactive backend
    import matplotlib.pyplot as plt
    HAS_MATPLOTLIB = True
except ImportError:
    HAS_MATPLOTLIB = False
    print("Warning: matplotlib not available, skipping graphs")

from inference.ternary_models import (
    BitNet, BitNetConfig, load_ternary_model, get_arch_info,
)

# Import baseline models from short benchmark
from inference.benchmark_comparison_short import (
    PyTorchBitNet, HuggingFaceModelWrapper,
    get_memory_mb, get_peak_memory_mb
)


# ============================================================================
# Results Saving and Graph Generation
# ============================================================================

def save_results(results: Dict[str, Any], output_dir: Path):
    """Save benchmark results to JSON and text files."""
    output_dir.mkdir(parents=True, exist_ok=True)

    # Save JSON
    json_path = output_dir / 'benchmark_long_results.json'
    with open(json_path, 'w') as f:
        json.dump(results, f, indent=2, default=str)
    print(f"\nResults saved to: {json_path}")

    # Save text summary
    txt_path = output_dir / 'benchmark_long_summary.txt'
    with open(txt_path, 'w') as f:
        f.write("=" * 80 + "\n")
        f.write("BitNet Scaling Benchmark Results\n")
        f.write("=" * 80 + "\n")
        f.write(f"Date: {results.get('timestamp', 'N/A')}\n")
        f.write(f"Platform: {results.get('platform', {}).get('machine', 'N/A')}\n")
        f.write(f"Kernel: {results.get('platform', {}).get('kernel_type', 'N/A')}\n")
        f.write(f"Threads: {results.get('platform', {}).get('threads', 'N/A')}\n")
        f.write("\n")

        # Prefill scaling summary
        if 'prefill_scaling' in results:
            f.write("-" * 80 + "\n")
            f.write("PREFILL SCALING (Time in ms)\n")
            f.write("-" * 80 + "\n")
            prefill = results['prefill_scaling']
            models = list(prefill.keys())
            if models:
                lengths = list(prefill[models[0]].keys())
                f.write(f"{'Model':<25}")
                for length in lengths:
                    f.write(f"{length:>12}")
                f.write("\n")
                for model in models:
                    f.write(f"{model:<25}")
                    for length in lengths:
                        val = prefill[model].get(length, {})
                        if val:
                            f.write(f"{val.get('mean_ms', 'N/A'):>12.1f}")
                        else:
                            f.write(f"{'N/A':>12}")
                    f.write("\n")
            f.write("\n")

        # Batch scaling summary
        if 'batch_scaling' in results:
            f.write("-" * 80 + "\n")
            f.write("BATCH SCALING (Throughput in tokens/sec)\n")
            f.write("-" * 80 + "\n")
            batch = results['batch_scaling']
            models = list(batch.keys())
            if models:
                batch_sizes = list(batch[models[0]].keys())
                f.write(f"{'Model':<25}")
                for bs in batch_sizes:
                    f.write(f"{'B='+str(bs):>12}")
                f.write("\n")
                for model in models:
                    f.write(f"{model:<25}")
                    for bs in batch_sizes:
                        val = batch[model].get(bs, {})
                        if val:
                            f.write(f"{val.get('tokens_per_sec', 'N/A'):>12.0f}")
                        else:
                            f.write(f"{'N/A':>12}")
                    f.write("\n")
            f.write("\n")

        # Generation scaling summary
        if 'generation_scaling' in results:
            f.write("-" * 80 + "\n")
            f.write("GENERATION SCALING (Throughput in tokens/sec)\n")
            f.write("-" * 80 + "\n")
            gen = results['generation_scaling']
            models = list(gen.keys())
            if models:
                gen_lengths = list(gen[models[0]].keys())
                f.write(f"{'Model':<25}")
                for gl in gen_lengths:
                    f.write(f"{'Gen '+str(gl):>12}")
                f.write("\n")
                for model in models:
                    f.write(f"{model:<25}")
                    for gl in gen_lengths:
                        val = gen[model].get(gl, {})
                        if val:
                            f.write(f"{val.get('tokens_per_sec', 'N/A'):>12.1f}")
                        else:
                            f.write(f"{'N/A':>12}")
                    f.write("\n")
            f.write("\n")

        # Speedups summary
        if 'speedups' in results:
            f.write("-" * 80 + "\n")
            f.write("SPEEDUPS (VNNI vs PyTorch)\n")
            f.write("-" * 80 + "\n")
            for key, value in results['speedups'].items():
                f.write(f"  {key}: {value:.2f}x\n")

    print(f"Summary saved to: {txt_path}")


def create_scaling_graphs(results: Dict[str, Any], output_dir: Path, arch_info: Dict = None):
    """Create graphs for scaling benchmark results."""
    if not HAS_MATPLOTLIB:
        print("Skipping graphs (matplotlib not available)")
        return

    output_dir.mkdir(parents=True, exist_ok=True)

    # Determine display name based on architecture
    arch_display = "X86" if (arch_info and arch_info.get('machine') == 'x86_64') else "Graviton"

    # Color scheme
    colors = {
        'VNNI Kernels': '#2ecc71',
        'PyTorch (ternary)': '#3498db',
        'HuggingFace (BitNet)': '#e74c3c',
    }

    # Graph 1: Prefill Scaling (Time vs Prompt Length)
    if 'prefill_scaling' in results:
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))

        prefill = results['prefill_scaling']
        models = list(prefill.keys())

        if models and prefill[models[0]]:
            lengths = sorted([int(k) for k in prefill[models[0]].keys()])

            # Left: Absolute times
            for model in models:
                times = [prefill[model].get(str(l), {}).get('mean_ms', None) for l in lengths]
                if any(t is not None for t in times):
                    valid_lengths = [l for l, t in zip(lengths, times) if t is not None]
                    valid_times = [t for t in times if t is not None]
                    ax1.plot(valid_lengths, valid_times, 'o-', label=model,
                            color=colors.get(model, '#95a5a6'), linewidth=2, markersize=8)

            ax1.set_xlabel('Prompt Length (tokens)', fontsize=12)
            ax1.set_ylabel('Prefill Time (ms)', fontsize=12)
            ax1.set_title('Prefill Time vs Prompt Length', fontsize=14)
            ax1.legend()
            ax1.grid(True, alpha=0.3)
            ax1.set_xscale('log', base=2)

            # Right: Speedup vs PyTorch
            if 'VNNI Kernels' in prefill and 'PyTorch (ternary)' in prefill:
                speedups = []
                valid_lengths = []
                for l in lengths:
                    vnni = prefill['VNNI Kernels'].get(str(l), {}).get('mean_ms')
                    pytorch = prefill['PyTorch (ternary)'].get(str(l), {}).get('mean_ms')
                    if vnni and pytorch:
                        speedups.append(pytorch / vnni)
                        valid_lengths.append(l)

                if speedups:
                    bars = ax2.bar(range(len(valid_lengths)), speedups, color='#2ecc71')
                    ax2.set_xticks(range(len(valid_lengths)))
                    ax2.set_xticklabels([str(l) for l in valid_lengths])
                    ax2.set_xlabel('Prompt Length (tokens)', fontsize=12)
                    ax2.set_ylabel('Speedup (x)', fontsize=12)
                    ax2.set_title(f'{arch_display} Kernel Speedup vs PyTorch', fontsize=14)
                    ax2.axhline(y=1, color='red', linestyle='--', alpha=0.5, label='Baseline')
                    ax2.grid(True, alpha=0.3, axis='y')

                    for bar, speedup in zip(bars, speedups):
                        ax2.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.05,
                                f'{speedup:.2f}x', ha='center', va='bottom', fontsize=10)

        plt.tight_layout()
        plt.savefig(output_dir / 'prefill_scaling.png', dpi=150, bbox_inches='tight')
        plt.close()
        print(f"Saved: {output_dir / 'prefill_scaling.png'}")

    # Graph 2: Batch Scaling (Throughput vs Batch Size)
    if 'batch_scaling' in results:
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))

        batch = results['batch_scaling']
        models = list(batch.keys())

        if models and batch[models[0]]:
            batch_sizes = sorted([int(k) for k in batch[models[0]].keys()])

            # Left: Absolute throughput
            for model in models:
                throughputs = [batch[model].get(str(bs), {}).get('tokens_per_sec', None) for bs in batch_sizes]
                if any(t is not None for t in throughputs):
                    valid_bs = [bs for bs, t in zip(batch_sizes, throughputs) if t is not None]
                    valid_tp = [t for t in throughputs if t is not None]
                    ax1.plot(valid_bs, valid_tp, 'o-', label=model,
                            color=colors.get(model, '#95a5a6'), linewidth=2, markersize=8)

            ax1.set_xlabel('Batch Size', fontsize=12)
            ax1.set_ylabel('Throughput (tokens/sec)', fontsize=12)
            ax1.set_title('Throughput vs Batch Size', fontsize=14)
            ax1.legend()
            ax1.grid(True, alpha=0.3)
            ax1.set_xscale('log', base=2)

            # Right: Speedup vs PyTorch
            if 'VNNI Kernels' in batch and 'PyTorch (ternary)' in batch:
                speedups = []
                valid_bs = []
                for bs in batch_sizes:
                    vnni = batch['VNNI Kernels'].get(str(bs), {}).get('tokens_per_sec')
                    pytorch = batch['PyTorch (ternary)'].get(str(bs), {}).get('tokens_per_sec')
                    if vnni and pytorch:
                        speedups.append(vnni / pytorch)
                        valid_bs.append(bs)

                if speedups:
                    bars = ax2.bar(range(len(valid_bs)), speedups, color='#2ecc71')
                    ax2.set_xticks(range(len(valid_bs)))
                    ax2.set_xticklabels([str(bs) for bs in valid_bs])
                    ax2.set_xlabel('Batch Size', fontsize=12)
                    ax2.set_ylabel('Speedup (x)', fontsize=12)
                    ax2.set_title(f'{arch_display} Kernel Throughput Speedup vs PyTorch', fontsize=14)
                    ax2.axhline(y=1, color='red', linestyle='--', alpha=0.5)
                    ax2.grid(True, alpha=0.3, axis='y')

                    for bar, speedup in zip(bars, speedups):
                        ax2.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.05,
                                f'{speedup:.2f}x', ha='center', va='bottom', fontsize=10)

        plt.tight_layout()
        plt.savefig(output_dir / 'batch_scaling.png', dpi=150, bbox_inches='tight')
        plt.close()
        print(f"Saved: {output_dir / 'batch_scaling.png'}")

    # Graph 3: Generation Scaling
    if 'generation_scaling' in results:
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))

        gen = results['generation_scaling']
        models = list(gen.keys())

        if models and gen[models[0]]:
            gen_lengths = sorted([int(k) for k in gen[models[0]].keys()])

            # Left: Absolute throughput
            for model in models:
                throughputs = [gen[model].get(str(gl), {}).get('tokens_per_sec', None) for gl in gen_lengths]
                if any(t is not None for t in throughputs):
                    valid_gl = [gl for gl, t in zip(gen_lengths, throughputs) if t is not None]
                    valid_tp = [t for t in throughputs if t is not None]
                    ax1.plot(valid_gl, valid_tp, 'o-', label=model,
                            color=colors.get(model, '#95a5a6'), linewidth=2, markersize=8)

            ax1.set_xlabel('Generation Length (tokens)', fontsize=12)
            ax1.set_ylabel('Throughput (tokens/sec)', fontsize=12)
            ax1.set_title('Generation Throughput vs Length', fontsize=14)
            ax1.legend()
            ax1.grid(True, alpha=0.3)

            # Right: Speedup vs PyTorch
            if 'VNNI Kernels' in gen and 'PyTorch (ternary)' in gen:
                speedups = []
                valid_gl = []
                for gl in gen_lengths:
                    vnni = gen['VNNI Kernels'].get(str(gl), {}).get('tokens_per_sec')
                    pytorch = gen['PyTorch (ternary)'].get(str(gl), {}).get('tokens_per_sec')
                    if vnni and pytorch:
                        speedups.append(vnni / pytorch)
                        valid_gl.append(gl)

                if speedups:
                    bars = ax2.bar(range(len(valid_gl)), speedups, color='#2ecc71')
                    ax2.set_xticks(range(len(valid_gl)))
                    ax2.set_xticklabels([str(gl) for gl in valid_gl])
                    ax2.set_xlabel('Generation Length (tokens)', fontsize=12)
                    ax2.set_ylabel('Speedup (x)', fontsize=12)
                    ax2.set_title(f'{arch_display} Kernel Generation Speedup vs PyTorch', fontsize=14)
                    ax2.axhline(y=1, color='red', linestyle='--', alpha=0.5)
                    ax2.grid(True, alpha=0.3, axis='y')

                    for bar, speedup in zip(bars, speedups):
                        ax2.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.05,
                                f'{speedup:.2f}x', ha='center', va='bottom', fontsize=10)

        plt.tight_layout()
        plt.savefig(output_dir / 'generation_scaling.png', dpi=150, bbox_inches='tight')
        plt.close()
        print(f"Saved: {output_dir / 'generation_scaling.png'}")

    # Graph 4: Combined Summary
    fig, axes = plt.subplots(1, 3, figsize=(16, 5))

    speedups = results.get('speedups', {})

    # Prefill speedup
    if 'prefill_512_tokens' in speedups:
        axes[0].bar(['Prefill\n(512 tokens)'], [speedups['prefill_512_tokens']], color='#2ecc71', width=0.5)
        axes[0].set_ylabel('Speedup (x)', fontsize=12)
        axes[0].set_title('Prefill Speedup', fontsize=14)
        axes[0].axhline(y=1, color='red', linestyle='--', alpha=0.5)
        axes[0].text(0, speedups['prefill_512_tokens'] + 0.1, f"{speedups['prefill_512_tokens']:.2f}x",
                    ha='center', fontsize=12, fontweight='bold')

    # Batch throughput speedup
    if 'batch_16_throughput' in speedups:
        axes[1].bar(['Batch 16\nThroughput'], [speedups['batch_16_throughput']], color='#3498db', width=0.5)
        axes[1].set_ylabel('Speedup (x)', fontsize=12)
        axes[1].set_title('Batch Throughput Speedup', fontsize=14)
        axes[1].axhline(y=1, color='red', linestyle='--', alpha=0.5)
        axes[1].text(0, speedups['batch_16_throughput'] + 0.1, f"{speedups['batch_16_throughput']:.2f}x",
                    ha='center', fontsize=12, fontweight='bold')

    # Generation speedup
    if 'generation_50_tokens' in speedups:
        axes[2].bar(['Generation\n(50 tokens)'], [speedups['generation_50_tokens']], color='#e74c3c', width=0.5)
        axes[2].set_ylabel('Speedup (x)', fontsize=12)
        axes[2].set_title('Generation Speedup', fontsize=14)
        axes[2].axhline(y=1, color='red', linestyle='--', alpha=0.5)
        axes[2].text(0, speedups['generation_50_tokens'] + 0.1, f"{speedups['generation_50_tokens']:.2f}x",
                    ha='center', fontsize=12, fontweight='bold')

    for ax in axes:
        ax.set_ylim(0, max([speedups.get(k, 1) for k in speedups.keys()] + [1.5]) * 1.2)
        ax.grid(True, alpha=0.3, axis='y')

    plt.suptitle(f'{arch_display} Kernel Speedups vs PyTorch (Higher is Better)', fontsize=16, y=1.02)
    plt.tight_layout()
    plt.savefig(output_dir / 'speedup_summary_long.png', dpi=150, bbox_inches='tight')
    plt.close()
    print(f"Saved: {output_dir / 'speedup_summary_long.png'}")


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

    # Create output directory (architecture-specific)
    arch_suffix = "x86" if arch['machine'] == 'x86_64' else "graviton"
    output_dir = Path(__file__).parent / f'benchmark_inference_{arch_suffix}'
    output_dir.mkdir(parents=True, exist_ok=True)
    print(f"Results will be saved to: {output_dir}")

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

    # Compute speedups
    speedups = {}

    # Prefill speedup at longest prompt
    if 'VNNI Kernels' in prefill_results and 'PyTorch (ternary)' in prefill_results:
        longest = PROMPT_LENGTHS[-1]
        vnni = prefill_results['VNNI Kernels'].get(longest)
        pytorch = prefill_results['PyTorch (ternary)'].get(longest)
        if vnni and pytorch:
            speedups[f'prefill_{longest}_tokens'] = pytorch['mean_ms'] / vnni['mean_ms']

    # Batch throughput at largest batch
    if 'VNNI Kernels' in batch_results and 'PyTorch (ternary)' in batch_results:
        largest_bs = BATCH_SIZES[-1]
        vnni = batch_results['VNNI Kernels'].get(largest_bs)
        pytorch = batch_results['PyTorch (ternary)'].get(largest_bs)
        if vnni and pytorch:
            speedups[f'batch_{largest_bs}_throughput'] = vnni['tokens_per_sec'] / pytorch['tokens_per_sec']

    # Generation throughput
    if 'VNNI Kernels' in gen_results and 'PyTorch (ternary)' in gen_results:
        gen_len = GEN_LENGTHS[1]  # Middle value
        vnni = gen_results['VNNI Kernels'].get(gen_len)
        pytorch = gen_results['PyTorch (ternary)'].get(gen_len)
        if vnni and pytorch:
            speedups[f'generation_{gen_len}_tokens'] = vnni['tokens_per_sec'] / pytorch['tokens_per_sec']

    # Final summary
    print("\n" + "=" * 80)
    print("FINAL SUMMARY")
    print("=" * 80)

    print("\nKey findings:")
    for key, value in speedups.items():
        print(f"  - {key}: {value:.2f}x faster than PyTorch")

    # Convert results to JSON-serializable format (use string keys)
    def convert_keys_to_str(d):
        """Convert dict keys to strings for JSON serialization."""
        if not isinstance(d, dict):
            return d
        return {str(k): convert_keys_to_str(v) for k, v in d.items()}

    # Collect all results
    all_results = {
        'timestamp': datetime.now().isoformat(),
        'platform': {
            'machine': arch['machine'],
            'kernel_type': arch['kernel_type'],
            'threads': torch.get_num_threads(),
        },
        'config': {
            'prompt_lengths': PROMPT_LENGTHS,
            'batch_sizes': BATCH_SIZES,
            'gen_lengths': GEN_LENGTHS,
            'warmup_iters': WARMUP_ITERS,
            'bench_iters': BENCH_ITERS,
        },
        'prefill_scaling': convert_keys_to_str(prefill_results),
        'batch_scaling': convert_keys_to_str(batch_results),
        'generation_scaling': convert_keys_to_str(gen_results),
        'speedups': speedups,
    }

    # Save results and create graphs
    save_results(all_results, output_dir)
    create_scaling_graphs(all_results, output_dir, arch)

    print("\n" + "=" * 80)
    print("Benchmark complete!")
    print(f"Results saved to: {output_dir}")
    print("=" * 80)


if __name__ == '__main__':
    main()
