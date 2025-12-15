#!/usr/bin/env python3
"""
Generate benchmark results files and graphs for Graviton kernel performance.

This script:
1. Runs kernel-level benchmarks
2. Saves results to JSON files
3. Generates matplotlib graphs
4. Creates a summary report

Can be run standalone or imported.
"""

import torch
import json
import time
import sys
import os
import platform
from pathlib import Path
from datetime import datetime
from typing import Dict, List, Any

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent))

import numpy as np

# Try to import matplotlib
try:
    import matplotlib
    matplotlib.use('Agg')  # Non-interactive backend for server
    import matplotlib.pyplot as plt
    HAS_MATPLOTLIB = True
except ImportError:
    HAS_MATPLOTLIB = False
    print("Warning: matplotlib not available, skipping graphs")


def get_system_info() -> Dict[str, Any]:
    """Get system information."""
    return {
        'platform': platform.machine(),
        'system': platform.system(),
        'processor': platform.processor(),
        'python_version': platform.python_version(),
        'torch_version': torch.__version__,
        'num_threads': torch.get_num_threads(),
        'timestamp': datetime.now().isoformat(),
    }


def benchmark_kernel_batch_sizes(kernel_fn, pytorch_fn, K: int, N: int,
                                  batch_sizes: List[int], warmup: int = 3,
                                  iters: int = 5) -> Dict[str, List]:
    """Benchmark kernel at different batch sizes."""
    results = {
        'batch_sizes': batch_sizes,
        'kernel_times_ms': [],
        'pytorch_times_ms': [],
        'speedups': [],
    }

    for M in batch_sizes:
        # Create test data
        x = torch.randn(M, K, dtype=torch.float32)
        w_ternary = torch.randint(-1, 2, (N, K), dtype=torch.int8)
        w_float = w_ternary.float()
        scale = torch.ones(N, dtype=torch.float32)

        # Warmup kernel
        for _ in range(warmup):
            _ = kernel_fn(x, w_ternary, scale)

        # Benchmark kernel
        kernel_times = []
        for _ in range(iters):
            start = time.perf_counter()
            _ = kernel_fn(x, w_ternary, scale)
            kernel_times.append((time.perf_counter() - start) * 1000)

        # Warmup PyTorch
        for _ in range(warmup):
            _ = pytorch_fn(x, w_float.T)

        # Benchmark PyTorch
        pytorch_times = []
        for _ in range(iters):
            start = time.perf_counter()
            _ = pytorch_fn(x, w_float.T)
            pytorch_times.append((time.perf_counter() - start) * 1000)

        kernel_mean = np.mean(kernel_times)
        pytorch_mean = np.mean(pytorch_times)
        speedup = pytorch_mean / kernel_mean if kernel_mean > 0 else 0

        results['kernel_times_ms'].append(kernel_mean)
        results['pytorch_times_ms'].append(pytorch_mean)
        results['speedups'].append(speedup)

        print(f"  M={M:4d}: Kernel={kernel_mean:8.3f}ms, PyTorch={pytorch_mean:8.3f}ms, Speedup={speedup:.2f}x")

    return results


def benchmark_model_inference(model, tokenizer, prompt: str,
                               gen_tokens: int = 20, warmup: int = 2,
                               iters: int = 3) -> Dict[str, float]:
    """Benchmark model inference."""
    input_ids = tokenizer.encode(prompt, return_tensors='pt')

    # Warmup
    model.eval()
    with torch.no_grad():
        for _ in range(warmup):
            _ = model(input_ids)

    # TTFT benchmark
    ttft_times = []
    with torch.no_grad():
        for _ in range(iters):
            start = time.perf_counter()
            _ = model(input_ids)
            ttft_times.append((time.perf_counter() - start) * 1000)

    # Generation benchmark
    gen_times = []
    with torch.no_grad():
        for _ in range(iters):
            generated = input_ids.clone()
            start = time.perf_counter()
            for _ in range(gen_tokens):
                logits = model(generated)
                next_token = logits[:, -1, :].argmax(dim=-1, keepdim=True)
                generated = torch.cat([generated, next_token], dim=1)
            gen_times.append((time.perf_counter() - start) * 1000)

    mean_ttft = np.mean(ttft_times)
    mean_gen = np.mean(gen_times)
    tokens_per_sec = gen_tokens / (mean_gen / 1000)

    return {
        'ttft_ms': mean_ttft,
        'generation_time_ms': mean_gen,
        'tokens_per_sec': tokens_per_sec,
        'gen_tokens': gen_tokens,
    }


def create_kernel_speedup_graph(results: Dict, output_path: str):
    """Create kernel speedup vs batch size graph."""
    if not HAS_MATPLOTLIB:
        return

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))

    batch_sizes = results['batch_sizes']
    kernel_times = results['kernel_times_ms']
    pytorch_times = results['pytorch_times_ms']
    speedups = results['speedups']

    # Left plot: Execution time
    ax1.plot(batch_sizes, kernel_times, 'b-o', label='Graviton Kernel', linewidth=2, markersize=8)
    ax1.plot(batch_sizes, pytorch_times, 'r-s', label='PyTorch F.linear', linewidth=2, markersize=8)
    ax1.set_xlabel('Batch Size (M)', fontsize=12)
    ax1.set_ylabel('Execution Time (ms)', fontsize=12)
    ax1.set_title('Kernel Execution Time vs Batch Size', fontsize=14)
    ax1.legend(fontsize=11)
    ax1.grid(True, alpha=0.3)
    ax1.set_xscale('log', base=2)
    ax1.set_yscale('log')

    # Right plot: Speedup
    colors = ['green' if s >= 1 else 'red' for s in speedups]
    bars = ax2.bar(range(len(batch_sizes)), speedups, color=colors, alpha=0.7, edgecolor='black')
    ax2.axhline(y=1, color='black', linestyle='--', linewidth=1, label='Parity (1x)')
    ax2.set_xlabel('Batch Size (M)', fontsize=12)
    ax2.set_ylabel('Speedup (x)', fontsize=12)
    ax2.set_title('Graviton Kernel Speedup vs PyTorch', fontsize=14)
    ax2.set_xticks(range(len(batch_sizes)))
    ax2.set_xticklabels([str(b) for b in batch_sizes])
    ax2.grid(True, alpha=0.3, axis='y')

    # Add speedup labels on bars
    for i, (bar, speedup) in enumerate(zip(bars, speedups)):
        ax2.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.1,
                f'{speedup:.2f}x', ha='center', va='bottom', fontsize=10, fontweight='bold')

    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"Saved: {output_path}")


def create_comparison_graph(graviton_results: Dict, pytorch_results: Dict, output_path: str):
    """Create model comparison graph."""
    if not HAS_MATPLOTLIB:
        return

    fig, axes = plt.subplots(1, 3, figsize=(15, 5))

    models = ['Graviton Kernel', 'PyTorch']

    # TTFT comparison
    ttfts = [graviton_results['ttft_ms'], pytorch_results['ttft_ms']]
    colors = ['#2ecc71', '#e74c3c']
    bars1 = axes[0].bar(models, ttfts, color=colors, alpha=0.8, edgecolor='black')
    axes[0].set_ylabel('Time (ms)', fontsize=12)
    axes[0].set_title('Time to First Token (TTFT)', fontsize=14)
    for bar, val in zip(bars1, ttfts):
        axes[0].text(bar.get_x() + bar.get_width()/2, bar.get_height() + 5,
                    f'{val:.1f}ms', ha='center', va='bottom', fontsize=11, fontweight='bold')

    # Throughput comparison
    toks = [graviton_results['tokens_per_sec'], pytorch_results['tokens_per_sec']]
    bars2 = axes[1].bar(models, toks, color=colors, alpha=0.8, edgecolor='black')
    axes[1].set_ylabel('Tokens/sec', fontsize=12)
    axes[1].set_title('Generation Throughput', fontsize=14)
    for bar, val in zip(bars2, toks):
        axes[1].text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.02,
                    f'{val:.2f}', ha='center', va='bottom', fontsize=11, fontweight='bold')

    # Memory comparison (if available)
    if 'memory_mb' in graviton_results and 'memory_mb' in pytorch_results:
        mems = [graviton_results['memory_mb'], pytorch_results['memory_mb']]
        bars3 = axes[2].bar(models, mems, color=colors, alpha=0.8, edgecolor='black')
        axes[2].set_ylabel('Memory (MB)', fontsize=12)
        axes[2].set_title('Memory Usage', fontsize=14)
        for bar, val in zip(bars3, mems):
            axes[2].text(bar.get_x() + bar.get_width()/2, bar.get_height() + 100,
                        f'{val:.0f}', ha='center', va='bottom', fontsize=11, fontweight='bold')
    else:
        # Speedup summary
        speedup_ttft = pytorch_results['ttft_ms'] / graviton_results['ttft_ms']
        speedup_toks = graviton_results['tokens_per_sec'] / pytorch_results['tokens_per_sec']
        speedups = [speedup_ttft, speedup_toks]
        labels = ['TTFT', 'Throughput']
        colors_speedup = ['#3498db', '#9b59b6']
        bars3 = axes[2].bar(labels, speedups, color=colors_speedup, alpha=0.8, edgecolor='black')
        axes[2].axhline(y=1, color='black', linestyle='--', linewidth=1)
        axes[2].set_ylabel('Speedup (x)', fontsize=12)
        axes[2].set_title('Graviton vs PyTorch Speedup', fontsize=14)
        for bar, val in zip(bars3, speedups):
            axes[2].text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.02,
                        f'{val:.2f}x', ha='center', va='bottom', fontsize=11, fontweight='bold')

    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"Saved: {output_path}")


def main():
    """Main benchmark and results generation."""
    print("=" * 70)
    print("Graviton Benchmark Results Generator")
    print("=" * 70)

    # Create output directory
    output_dir = Path("benchmark_results")
    output_dir.mkdir(exist_ok=True)

    # System info
    sys_info = get_system_info()
    print(f"\nSystem: {sys_info['platform']}")
    print(f"Threads: {sys_info['num_threads']}")
    print(f"PyTorch: {sys_info['torch_version']}")

    # All results will be stored here
    all_results = {
        'system_info': sys_info,
        'kernel_benchmarks': {},
        'model_benchmarks': {},
    }

    # Import kernel functions
    try:
        from inference.ternary_models import load_ternary_model, get_arch_info
        from src.cpu_ops import get_ternary_matmul_fn

        arch = get_arch_info()
        print(f"Kernel type: {arch['kernel_type']}")
        all_results['system_info']['kernel_type'] = arch['kernel_type']

        kernel_fn = get_ternary_matmul_fn()

    except ImportError as e:
        print(f"Error importing kernel: {e}")
        print("Running in fallback mode with synthetic data")
        kernel_fn = None

    # Kernel benchmarks
    if kernel_fn is not None:
        print("\n" + "=" * 70)
        print("KERNEL BENCHMARKS")
        print("=" * 70)

        # Test different projection sizes (BitNet architecture)
        test_configs = [
            ("Q/K/V Projection", 2560, 2560),
            ("O Projection", 2560, 2560),
            ("Gate/Up Projection", 2560, 6912),
            ("Down Projection", 6912, 2560),
        ]

        batch_sizes = [1, 4, 8, 16, 32, 64, 128, 256]

        for name, K, N in test_configs:
            print(f"\n{name} (K={K} -> N={N}):")
            results = benchmark_kernel_batch_sizes(
                kernel_fn,
                lambda x, w: torch.mm(x, w),
                K, N, batch_sizes
            )
            all_results['kernel_benchmarks'][name] = results

            # Create graph for this config
            graph_name = name.replace("/", "_").replace(" ", "_").lower()
            create_kernel_speedup_graph(
                results,
                str(output_dir / f"kernel_{graph_name}.png")
            )

    # Model benchmarks
    print("\n" + "=" * 70)
    print("MODEL BENCHMARKS")
    print("=" * 70)

    try:
        from inference.ternary_models import load_ternary_model
        from inference.benchmark_comparison_short import PyTorchBitNet

        MODEL_NAME = 'microsoft/bitnet-b1.58-2B-4T-bf16'

        print("\nLoading Graviton kernel model...")
        graviton_model, tokenizer = load_ternary_model('bitnet-2b')
        graviton_model.eval()

        print("Loading PyTorch baseline model...")
        pytorch_model = PyTorchBitNet.from_safetensors(MODEL_NAME)
        pytorch_model.eval()

        prompt = "The future of artificial intelligence is"

        print(f"\nBenchmarking with prompt: '{prompt}'")

        print("\nGraviton Kernel model:")
        graviton_results = benchmark_model_inference(graviton_model, tokenizer, prompt)
        print(f"  TTFT: {graviton_results['ttft_ms']:.1f} ms")
        print(f"  Throughput: {graviton_results['tokens_per_sec']:.2f} tok/s")

        print("\nPyTorch baseline model:")
        pytorch_results = benchmark_model_inference(pytorch_model, tokenizer, prompt)
        print(f"  TTFT: {pytorch_results['ttft_ms']:.1f} ms")
        print(f"  Throughput: {pytorch_results['tokens_per_sec']:.2f} tok/s")

        all_results['model_benchmarks']['graviton'] = graviton_results
        all_results['model_benchmarks']['pytorch'] = pytorch_results

        # Calculate speedups
        speedup_ttft = pytorch_results['ttft_ms'] / graviton_results['ttft_ms']
        speedup_throughput = graviton_results['tokens_per_sec'] / pytorch_results['tokens_per_sec']

        all_results['model_benchmarks']['speedups'] = {
            'ttft': speedup_ttft,
            'throughput': speedup_throughput,
        }

        print(f"\nSpeedup vs PyTorch:")
        print(f"  TTFT: {speedup_ttft:.2f}x")
        print(f"  Throughput: {speedup_throughput:.2f}x")

        # Create comparison graph
        create_comparison_graph(
            graviton_results,
            pytorch_results,
            str(output_dir / "model_comparison.png")
        )

    except Exception as e:
        print(f"Error in model benchmarks: {e}")
        import traceback
        traceback.print_exc()

    # Save results to JSON
    results_file = output_dir / "benchmark_results.json"
    with open(results_file, 'w') as f:
        json.dump(all_results, f, indent=2, default=str)
    print(f"\nSaved results to: {results_file}")

    # Create summary report
    summary_file = output_dir / "benchmark_summary.txt"
    with open(summary_file, 'w') as f:
        f.write("=" * 70 + "\n")
        f.write("GRAVITON BENCHMARK SUMMARY\n")
        f.write("=" * 70 + "\n\n")

        f.write(f"Date: {sys_info['timestamp']}\n")
        f.write(f"Platform: {sys_info['platform']}\n")
        f.write(f"Threads: {sys_info['num_threads']}\n")
        f.write(f"PyTorch: {sys_info['torch_version']}\n")
        if 'kernel_type' in all_results['system_info']:
            f.write(f"Kernel: {all_results['system_info']['kernel_type']}\n")
        f.write("\n")

        if all_results['kernel_benchmarks']:
            f.write("-" * 70 + "\n")
            f.write("KERNEL PERFORMANCE (vs PyTorch F.linear)\n")
            f.write("-" * 70 + "\n\n")

            for name, results in all_results['kernel_benchmarks'].items():
                f.write(f"{name}:\n")
                for i, bs in enumerate(results['batch_sizes']):
                    speedup = results['speedups'][i]
                    f.write(f"  M={bs:4d}: {speedup:.2f}x speedup\n")
                f.write("\n")

        if 'graviton' in all_results['model_benchmarks']:
            f.write("-" * 70 + "\n")
            f.write("MODEL PERFORMANCE\n")
            f.write("-" * 70 + "\n\n")

            grav = all_results['model_benchmarks']['graviton']
            pyt = all_results['model_benchmarks']['pytorch']
            speedups = all_results['model_benchmarks']['speedups']

            f.write(f"{'Metric':<25} {'Graviton':>15} {'PyTorch':>15} {'Speedup':>10}\n")
            f.write("-" * 70 + "\n")
            f.write(f"{'TTFT (ms)':<25} {grav['ttft_ms']:>15.1f} {pyt['ttft_ms']:>15.1f} {speedups['ttft']:>10.2f}x\n")
            f.write(f"{'Throughput (tok/s)':<25} {grav['tokens_per_sec']:>15.2f} {pyt['tokens_per_sec']:>15.2f} {speedups['throughput']:>10.2f}x\n")

        f.write("\n" + "=" * 70 + "\n")

    print(f"Saved summary to: {summary_file}")

    # List all generated files
    print("\n" + "=" * 70)
    print("GENERATED FILES")
    print("=" * 70)
    for f in sorted(output_dir.iterdir()):
        print(f"  {f}")

    return all_results


if __name__ == '__main__':
    main()
