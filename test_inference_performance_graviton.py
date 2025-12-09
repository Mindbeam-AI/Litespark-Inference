#!/usr/bin/env python3
"""
Simple Inference Performance Test - Graviton

Tests the optimized Graviton NEON SDOT kernels on realistic GPT-2 inference workloads.
Uses the same kernel loading approach as test_inference_performance.py but for ARM64.
"""

import torch
import time
import platform
import os
from torch.utils.cpp_extension import load


def test_inference_patterns():
    """Test performance on realistic inference patterns."""

    print("=" * 80)
    print("GPT-2 Inference Performance Test - Graviton NEON SDOT Kernels")
    print("=" * 80)

    # Check architecture
    arch = platform.machine().lower()
    if arch not in ['aarch64', 'arm64']:
        print(f"Unsupported architecture: {arch}")
        print("This script requires ARM64 (Graviton) hardware.")
        return

    print("Loading optimized Graviton kernels...")

    # Load the Graviton kernel
    kernel = load(
        name="graviton_kernel_inference_test",
        sources=["src/cpu_ops/kernels/arm64/matmul_free_graviton.cpp"],
        extra_cflags=['-O3', '-march=armv8.2-a+dotprod', '-ffast-math', '-fopenmp'],
        extra_ldflags=['-fopenmp'],
        verbose=True  # Show compilation to verify correct kernel
    )

    print("Graviton kernels loaded successfully")
    print("Using optimized Graviton v2/v3 kernels with NEON SDOT")
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
    print(f"{'Description':<35} {'M':<4} {'N':<6} {'K':<6} {'Time (ms)':<10} {'GOPS':<8} {'Kernel'}")
    print("-" * 85)

    results = []

    # K padding for NEON (16 bytes)
    def get_k_padded(K):
        return ((K + 15) // 16) * 16

    for M, N, K, desc in test_cases:
        K_padded = get_k_padded(K)

        # Create test data - ensure correct types for Graviton kernels
        x_int8 = torch.randint(-128, 127, (M, K_padded), dtype=torch.int8).contiguous()
        scales = torch.randn(M, dtype=torch.float32).contiguous()
        w_int8 = torch.randint(-1, 2, (N, K_padded), dtype=torch.int8).contiguous()  # Ternary weights
        w_sum = torch.sum(w_int8.float(), dim=1).to(torch.int32).contiguous()
        y = torch.zeros(M, N, dtype=torch.float32).contiguous()
        bias = torch.Tensor()  # Empty bias

        # Thread selection based on batch size
        base_threads = os.cpu_count() or 4
        if M <= 8:
            threads = min(4, base_threads)
        elif M <= 32:
            threads = min(6, base_threads)
        else:
            threads = base_threads

        # Kernel selection:
        # - v2 (register-blocked) is better for small M (single token)
        # - v3 (cache-tiled) is better for larger M (batched)
        if M <= 4:
            kernel_name = "v2"
            kernel_func = kernel.matmul_free_graviton_v2
        else:
            kernel_name = "v3"
            kernel_func = kernel.matmul_free_graviton_v3

        # Warmup
        for _ in range(3):
            kernel_func(x_int8, scales, w_int8, w_sum, y, bias, M, N, K, threads)

        # Benchmark
        start_time = time.time()

        for _ in range(10):
            kernel_func(x_int8, scales, w_int8, w_sum, y, bias, M, N, K, threads)

        end_time = time.time()

        # Calculate metrics
        avg_time_ms = (end_time - start_time) * 1000 / 10
        ops = 2 * M * N * K  # Approximate ops for matmul (int8 operations)
        gops = ops / (avg_time_ms * 1e6)

        print(f"{desc:<35} {M:<4} {N:<6} {K:<6} {avg_time_ms:<10.2f} {gops:<8.1f} ({kernel_name})")

        results.append({
            'description': desc,
            'M': M, 'N': N, 'K': K,
            'time_ms': avg_time_ms,
            'gops': gops,
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
        avg_gops = sum(r['gops'] for r in group) / len(group)
        total_time = sum(r['time_ms'] for r in group)

        print(f"Batch M={M:<3}: {len(group)} operations, {avg_gops:.1f} avg GOPS, {total_time:.1f}ms total")

    print(f"\nOverall average: {sum(r['gops'] for r in results) / len(results):.1f} GOPS")
    print("Inference performance test completed successfully!")

    return results


def compare_with_pytorch():
    """Compare Graviton kernels with PyTorch baseline."""

    print("\n" + "=" * 80)
    print("COMPARISON: Graviton SDOT vs PyTorch FP32")
    print("=" * 80)

    arch = platform.machine().lower()
    if arch not in ['aarch64', 'arm64']:
        print(f"Unsupported architecture: {arch}")
        return

    # Load kernel
    kernel = load(
        name="graviton_kernel_compare",
        sources=["src/cpu_ops/kernels/arm64/matmul_free_graviton.cpp"],
        extra_cflags=['-O3', '-march=armv8.2-a+dotprod', '-ffast-math', '-fopenmp'],
        extra_ldflags=['-fopenmp'],
        verbose=False
    )

    test_sizes = [
        (1, 2048, 2048),
        (1, 8192, 2048),
        (32, 2048, 2048),
        (32, 8192, 2048),
        (128, 2048, 2048),
        (128, 8192, 2048),
    ]

    def get_k_padded(K):
        return ((K + 15) // 16) * 16

    print(f"{'Size (M×N×K)':<20} {'PyTorch (ms)':<15} {'Graviton (ms)':<15} {'Speedup':<10}")
    print("-" * 60)

    threads = os.cpu_count() or 4

    for M, N, K in test_sizes:
        K_padded = get_k_padded(K)

        # PyTorch FP32 baseline
        x_float = torch.randn(M, K, dtype=torch.float32)
        w_float = torch.randint(-1, 2, (N, K), dtype=torch.float32)

        # Warmup PyTorch
        for _ in range(3):
            _ = torch.mm(x_float, w_float.T)

        start = time.time()
        for _ in range(10):
            _ = torch.mm(x_float, w_float.T)
        pytorch_time = (time.time() - start) * 1000 / 10

        # Graviton kernel
        x_int8 = torch.randint(-128, 127, (M, K_padded), dtype=torch.int8).contiguous()
        scales = torch.randn(M, dtype=torch.float32).contiguous()
        w_int8 = w_float[:, :K_padded].to(torch.int8).contiguous() if K_padded <= K else torch.zeros(N, K_padded, dtype=torch.int8)
        w_int8[:, :K] = w_float.to(torch.int8)
        w_sum = torch.sum(w_int8.float(), dim=1).to(torch.int32).contiguous()
        y = torch.zeros(M, N, dtype=torch.float32).contiguous()
        bias = torch.Tensor()

        # Select kernel
        kernel_func = kernel.matmul_free_graviton_v3 if M > 4 else kernel.matmul_free_graviton_v2

        # Warmup
        for _ in range(3):
            kernel_func(x_int8, scales, w_int8, w_sum, y, bias, M, N, K, threads)

        start = time.time()
        for _ in range(10):
            kernel_func(x_int8, scales, w_int8, w_sum, y, bias, M, N, K, threads)
        graviton_time = (time.time() - start) * 1000 / 10

        speedup = pytorch_time / graviton_time
        size_str = f"{M}×{N}×{K}"
        print(f"{size_str:<20} {pytorch_time:<15.3f} {graviton_time:<15.3f} {speedup:<10.2f}x")


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description='Graviton Inference Performance Test')
    parser.add_argument('--compare', action='store_true', help='Compare with PyTorch baseline')
    args = parser.parse_args()

    # Print system info
    print("System Information:")
    print(f"  Platform: {platform.platform()}")
    print(f"  Machine: {platform.machine()}")
    print(f"  CPU count: {os.cpu_count()}")
    print(f"  PyTorch threads: {torch.get_num_threads()}")
    print()

    test_inference_patterns()

    if args.compare:
        compare_with_pytorch()
