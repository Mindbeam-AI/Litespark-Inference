#!/usr/bin/env python3
"""
Simple Inference Performance Test

Tests the optimized VNNI kernels on realistic GPT-2 inference workloads.
Uses the same kernel loading as test_full_forward.py to ensure we test
the actual optimized kernels, not the fallback implementations.
"""

import torch
import time
import platform
import matplotlib.pyplot as plt
import numpy as np
import os
from torch.utils.cpp_extension import load

def test_inference_patterns():
    """Test performance on realistic inference patterns."""
    
    print("=" * 80)
    print("GPT-2 Inference Performance Test - Optimized VNNI Kernels")
    print("=" * 80)
    
    # Load kernels - EXACTLY the same as test_full_forward.py
    arch = platform.machine().lower()
    if arch not in ['x86_64', 'amd64']:
        print(f"Unsupported architecture: {arch}")
        return
        
    print("Loading optimized VNNI kernels...")
    
    # Use exact same loading as test_full_forward.py to ensure we get the optimized kernels
    kernel = load(
        name="vnni_kernel_inference_test",
        sources=["src/cpu_ops/kernels/x86_64/matmul_free_tmac_vnni.cpp"],
        extra_cflags=['-O3', '-march=native', '-ffast-math', '-fopenmp'],
        extra_ldflags=['-lgomp'],
        verbose=True  # Show compilation to verify correct kernel
    )
    
    print("✅ VNNI kernels loaded successfully")
    print("✅ Using optimized VNNI v2/v3/v4/v5/v6 kernels (same as transformer benchmark)")
    print()
    
    # GPT-2 layer dimensions - realistic inference patterns
    test_cases = [
        # (M, N, K, description)
        (1, 2048, 2048, "Single token - Attention QKV"),
        (1, 16384, 2048, "Single token - MLP up"),
        (1, 2048, 16384, "Single token - MLP down"),
        (1, 50257, 2048, "Single token - LM head (vocab)"),
        
        (8, 2048, 2048, "Small batch - Attention QKV"),
        (8, 16384, 2048, "Small batch - MLP up"),
        (8, 2048, 16384, "Small batch - MLP down"),
        (8, 50257, 2048, "Small batch - LM head (vocab)"),
        
        (32, 2048, 2048, "Medium batch - Attention QKV"),
        (32, 16384, 2048, "Medium batch - MLP up"),
        (32, 2048, 16384, "Medium batch - MLP down"),
        (32, 50257, 2048, "Medium batch - LM head (vocab)"),
        
        (128, 2048, 2048, "Large batch - Attention QKV"),
        (128, 16384, 2048, "Large batch - MLP up"),
        (128, 2048, 16384, "Large batch - MLP down"),
        (128, 50257, 2048, "Large batch - LM head (vocab)"),
    ]
    
    print(f"Testing {len(test_cases)} inference patterns...")
    print(f"{'Description':<35} {'M':<4} {'N':<6} {'K':<6} {'VNNI (ms)':<10} {'PyTorch (ms)':<12} {'Speedup':<8} {'Kernel'}")
    print("-" * 105)

    results = []
    
    for M, N, K, desc in test_cases:
        # Create test data - use EXACT same types as test_full_forward.py
        x_int8 = torch.randint(-128, 127, (M, K), dtype=torch.int8)
        scales = torch.randn(M, dtype=torch.float32)
        w_int8 = torch.randint(-1, 2, (N, K), dtype=torch.int8)  # Ternary weights
        w_sum = torch.sum(w_int8.int(), dim=1).to(torch.int32)  # Must be int32!
        y = torch.zeros(M, N, dtype=torch.float32)
        bias = torch.zeros(N, dtype=torch.float32)
        
        # Use same threading and kernel selection as test_full_forward.py
        base_threads = 8  # Same as transformer benchmark
        if M <= 8:
            threads = min(4, base_threads)
        elif M <= 32:
            threads = min(6, base_threads)
        else:
            threads = base_threads
            
        # Use same kernel selection logic as matmul_int32_out in test_full_forward.py
        if N >= 8192:
            # v4 kernel for large N - outputs int32
            y_out = torch.zeros(M, N, dtype=torch.int32)
            kernel_name = "v4"
            use_bias = False
        elif N >= 4096 and M >= 16:
            # v4 kernel for large N with medium+ M - outputs int32
            y_out = torch.zeros(M, N, dtype=torch.int32)
            kernel_name = "v4"
            use_bias = False
        elif M >= 64:
            # v5 kernel for large M - outputs int32
            y_out = torch.zeros(M, N, dtype=torch.int32)
            kernel_name = "v5"
            use_bias = False
        elif M >= 32 and N >= 2048:
            # v5 kernel for medium M with medium N - outputs int32
            y_out = torch.zeros(M, N, dtype=torch.int32)
            kernel_name = "v5"
            use_bias = False
        else:
            # v3 kernel for small operations - use int32 output version for consistency
            y_out = torch.zeros(M, N, dtype=torch.int32)
            kernel_name = "v3_int32"
            # v3_int32_out uses empty bias, not zeros bias
            bias_v3 = torch.empty(0)

        # Warmup
        for _ in range(3):
            if kernel_name == "v3_int32":
                kernel.matmul_free_vnni_v3_int32_out(x_int8, scales, w_int8, w_sum, y_out, bias_v3, M, N, K, threads)
            elif kernel_name == "v4":
                kernel.matmul_free_vnni_v4_large_n(x_int8, scales, w_int8, w_sum, y_out, M, N, K, threads)
            elif kernel_name == "v5":
                kernel.matmul_free_vnni_v5_large_m(x_int8, scales, w_int8, w_sum, y_out, M, N, K, threads)

        # Benchmark
        start_time = time.time()

        for _ in range(10):
            if kernel_name == "v3_int32":
                kernel.matmul_free_vnni_v3_int32_out(x_int8, scales, w_int8, w_sum, y_out, bias_v3, M, N, K, threads)
            elif kernel_name == "v4":
                kernel.matmul_free_vnni_v4_large_n(x_int8, scales, w_int8, w_sum, y_out, M, N, K, threads)
            elif kernel_name == "v5":
                kernel.matmul_free_vnni_v5_large_m(x_int8, scales, w_int8, w_sum, y_out, M, N, K, threads)

        end_time = time.time()
        vnni_time_ms = (end_time - start_time) * 1000 / 10

        # PyTorch baseline comparison
        x_float = torch.randn(M, K, dtype=torch.float32)
        w_float = torch.randn(N, K, dtype=torch.float32)

        # Warmup PyTorch
        for _ in range(3):
            torch.mm(x_float, w_float.t())

        # Benchmark PyTorch
        start_time = time.time()
        for _ in range(10):
            torch.mm(x_float, w_float.t())
        end_time = time.time()
        pytorch_time_ms = (end_time - start_time) * 1000 / 10

        # Calculate speedup
        speedup = pytorch_time_ms / vnni_time_ms

        kernel_display = kernel_name.replace("_int32", "")  # Clean up display name
        print(f"{desc:<35} {M:<4} {N:<6} {K:<6} {vnni_time_ms:<10.2f} {pytorch_time_ms:<12.2f} {speedup:<8.2f}x ({kernel_display})")

        results.append({
            'description': desc,
            'M': M, 'N': N, 'K': K,
            'vnni_time_ms': vnni_time_ms,
            'pytorch_time_ms': pytorch_time_ms,
            'speedup': speedup,
            'kernel': kernel_name
        })

    # Summary
    print("\n" + "=" * 80)
    print("INFERENCE PERFORMANCE SUMMARY")
    print("=" * 80)

    # Group by batch size
    batch_groups = {}
    for result in results:
        M = result['M']
        if M not in batch_groups:
            batch_groups[M] = []
        batch_groups[M].append(result)

    for M in sorted(batch_groups.keys()):
        group = batch_groups[M]
        avg_speedup = sum(r['speedup'] for r in group) / len(group)
        total_vnni_time = sum(r['vnni_time_ms'] for r in group)
        total_pytorch_time = sum(r['pytorch_time_ms'] for r in group)

        print(f"Batch M={M:<3}: {len(group)} operations, {avg_speedup:.2f}x avg speedup, {total_vnni_time:.1f}ms vs {total_pytorch_time:.1f}ms")

    overall_speedup = sum(r['speedup'] for r in results) / len(results)
    print(f"\n🚀 Overall average speedup: {overall_speedup:.2f}x faster than PyTorch!")

    # Highlight best performers
    best_speedups = sorted(results, key=lambda x: x['speedup'], reverse=True)[:3]
    print(f"\n🏆 Top 3 speedups:")
    for i, result in enumerate(best_speedups, 1):
        print(f"  {i}. {result['description']}: {result['speedup']:.2f}x ({result['kernel']})")

    # Create visualizations
    create_inference_plots(results, batch_groups)
    print("✅ Inference performance test completed successfully!")

def create_inference_plots(results, batch_groups):
    """Create comprehensive plots grouped by batch size M."""

    # Create output directory
    plot_dir = "benchmark_plots_inference"
    os.makedirs(plot_dir, exist_ok=True)

    # Set up matplotlib style
    plt.style.use('default')
    plt.rcParams['figure.figsize'] = (12, 8)
    plt.rcParams['font.size'] = 10

    # Define operation types and colors
    op_types = {
        'Attention QKV': '#1f77b4',
        'MLP up': '#ff7f0e',
        'MLP down': '#2ca02c',
        'LM head (vocab)': '#d62728'
    }

    # 1. Speedup comparison by batch size
    fig, axes = plt.subplots(2, 2, figsize=(16, 12))
    fig.suptitle('GPT-2 Inference Performance: VNNI vs PyTorch', fontsize=16, fontweight='bold')

    batch_sizes = sorted(batch_groups.keys())

    for idx, M in enumerate(batch_sizes):
        row = idx // 2
        col = idx % 2
        ax = axes[row, col]

        group = batch_groups[M]

        # Extract data for this batch size
        operations = []
        vnni_times = []
        pytorch_times = []
        speedups = []
        colors = []
        kernels = []

        for result in group:
            # Extract operation type from description
            desc = result['description']
            for op_type in op_types:
                if op_type in desc:
                    operations.append(op_type)
                    colors.append(op_types[op_type])
                    break
            else:
                operations.append(desc.split(' - ')[-1])
                colors.append('#gray')

            vnni_times.append(result['vnni_time_ms'])
            pytorch_times.append(result['pytorch_time_ms'])
            speedups.append(result['speedup'])
            kernels.append(result['kernel'].replace('_int32', ''))

        # Create grouped bar chart
        x = np.arange(len(operations))
        width = 0.35

        bars1 = ax.bar(x - width/2, vnni_times, width, label='VNNI Optimized',
                      color=colors, alpha=0.8)
        bars2 = ax.bar(x + width/2, pytorch_times, width, label='PyTorch Baseline',
                      color=colors, alpha=0.4)

        # Add speedup annotations
        for i, (speedup, kernel) in enumerate(zip(speedups, kernels)):
            height = max(vnni_times[i], pytorch_times[i])
            ax.annotate(f'{speedup:.2f}x\n({kernel})',
                       xy=(i, height), xytext=(0, 10),
                       textcoords='offset points', ha='center', va='bottom',
                       fontsize=8, fontweight='bold',
                       bbox=dict(boxstyle='round,pad=0.3', facecolor='yellow', alpha=0.7))

        ax.set_xlabel('Operation Type')
        ax.set_ylabel('Time (ms)')
        ax.set_title(f'Batch Size M={M}', fontweight='bold')
        ax.set_xticks(x)
        ax.set_xticklabels(operations, rotation=45, ha='right')
        ax.legend()
        ax.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(f'{plot_dir}/speedup_by_batch_size.png', dpi=300, bbox_inches='tight')
    plt.close()

    # 2. Overall speedup summary
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 6))

    # Speedup by batch size
    batch_speedups = [sum(r['speedup'] for r in batch_groups[M]) / len(batch_groups[M])
                     for M in batch_sizes]

    bars = ax1.bar(range(len(batch_sizes)), batch_speedups,
                   color=['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728'])
    ax1.set_xlabel('Batch Size (M)')
    ax1.set_ylabel('Average Speedup')
    ax1.set_title('Average Speedup by Batch Size', fontweight='bold')
    ax1.set_xticks(range(len(batch_sizes)))
    ax1.set_xticklabels([f'M={M}' for M in batch_sizes])
    ax1.grid(True, alpha=0.3)

    # Add value labels on bars
    for bar, speedup in zip(bars, batch_speedups):
        height = bar.get_height()
        ax1.annotate(f'{speedup:.2f}x',
                    xy=(bar.get_x() + bar.get_width() / 2, height),
                    xytext=(0, 3), textcoords="offset points",
                    ha='center', va='bottom', fontweight='bold')

    # Speedup by operation type
    op_speedups = {}
    for result in results:
        desc = result['description']
        for op_type in op_types:
            if op_type in desc:
                if op_type not in op_speedups:
                    op_speedups[op_type] = []
                op_speedups[op_type].append(result['speedup'])
                break

    op_names = list(op_speedups.keys())
    op_avg_speedups = [sum(op_speedups[op]) / len(op_speedups[op]) for op in op_names]
    op_colors = [op_types[op] for op in op_names]

    bars = ax2.bar(range(len(op_names)), op_avg_speedups, color=op_colors)
    ax2.set_xlabel('Operation Type')
    ax2.set_ylabel('Average Speedup')
    ax2.set_title('Average Speedup by Operation Type', fontweight='bold')
    ax2.set_xticks(range(len(op_names)))
    ax2.set_xticklabels(op_names, rotation=45, ha='right')
    ax2.grid(True, alpha=0.3)

    # Add value labels on bars
    for bar, speedup in zip(bars, op_avg_speedups):
        height = bar.get_height()
        ax2.annotate(f'{speedup:.2f}x',
                    xy=(bar.get_x() + bar.get_width() / 2, height),
                    xytext=(0, 3), textcoords="offset points",
                    ha='center', va='bottom', fontweight='bold')

    plt.tight_layout()
    plt.savefig(f'{plot_dir}/speedup_summary.png', dpi=300, bbox_inches='tight')
    plt.close()

    # 3. Kernel usage analysis
    fig, ax = plt.subplots(figsize=(10, 6))

    kernel_counts = {}
    kernel_speedups = {}
    for result in results:
        kernel = result['kernel'].replace('_int32', '')
        if kernel not in kernel_counts:
            kernel_counts[kernel] = 0
            kernel_speedups[kernel] = []
        kernel_counts[kernel] += 1
        kernel_speedups[kernel].append(result['speedup'])

    kernels = list(kernel_counts.keys())
    counts = [kernel_counts[k] for k in kernels]
    avg_speedups = [sum(kernel_speedups[k]) / len(kernel_speedups[k]) for k in kernels]

    # Create bubble chart
    colors = ['#1f77b4', '#ff7f0e', '#2ca02c'][:len(kernels)]
    scatter = ax.scatter(range(len(kernels)), avg_speedups, s=[c*100 for c in counts],
                        c=colors, alpha=0.6)

    for i, (kernel, count, speedup) in enumerate(zip(kernels, counts, avg_speedups)):
        ax.annotate(f'{kernel}\n({count} ops)\n{speedup:.2f}x avg',
                   xy=(i, speedup), xytext=(0, 0),
                   textcoords='offset points', ha='center', va='center',
                   fontweight='bold')

    ax.set_xlabel('Kernel Type')
    ax.set_ylabel('Average Speedup')
    ax.set_title('Kernel Performance Analysis', fontweight='bold')
    ax.set_xticks(range(len(kernels)))
    ax.set_xticklabels(kernels)
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(f'{plot_dir}/kernel_analysis.png', dpi=300, bbox_inches='tight')
    plt.close()

    print(f"\n📊 Plots saved to: {plot_dir}/")
    print(f"   - speedup_by_batch_size.png: Detailed comparison by batch size")
    print(f"   - speedup_summary.png: Overall speedup analysis")
    print(f"   - kernel_analysis.png: Kernel performance breakdown")

if __name__ == "__main__":
    test_inference_patterns()
