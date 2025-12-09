#!/usr/bin/env python3
"""
Fused Kernel Inference Performance Test

Tests the fused VNNI kernels on realistic GPT-2 inference workloads.
Unlike test_inference_performance.py which tests raw matmul throughput,
this script tests the fused kernels that combine multiple operations:

- fused_softmax: matmul + softmax + quantize (for attention scores)
- fused_quantize: matmul + quantize (for down projection)
- fused_swiglu: matmul + SwiGLU activation + quantize (for MLP up)

These fused kernels reduce memory bandwidth by keeping intermediate
results in cache/registers instead of writing to RAM.
"""

import torch
import time
import platform
import os
from torch.utils.cpp_extension import load


def test_fused_inference_patterns():
    """Test performance of fused kernels on realistic inference patterns."""

    print("=" * 90)
    print("GPT-2 Inference Performance Test - Fused VNNI Kernels")
    print("=" * 90)

    # Check architecture
    arch = platform.machine().lower()
    if arch not in ['x86_64', 'amd64']:
        print(f"Unsupported architecture: {arch}")
        print("This script requires x86_64 with AVX-512 VNNI support.")
        return

    print("Loading fused VNNI kernels...")

    kernel = load(
        name="vnni_kernel_fused_test",
        sources=["src/cpu_ops/kernels/x86_64/matmul_free_tmac_vnni.cpp"],
        extra_cflags=['-O3', '-march=native', '-ffast-math', '-fopenmp'],
        extra_ldflags=['-lgomp'],
        verbose=True
    )

    print("Fused VNNI kernels loaded successfully")
    print("Testing: fused_softmax, fused_quantize, fused_swiglu_optimized")
    print()

    # K padding for AVX-512 (64 bytes)
    def get_k_padded(K):
        return ((K + 63) // 64) * 64

    # Thread selection
    base_threads = min(os.cpu_count() or 8, 8)

    def get_threads(M):
        if M <= 8:
            return min(4, base_threads)
        elif M <= 32:
            return min(6, base_threads)
        return base_threads

    results = []

    # =========================================================================
    # Test 1: Fused Softmax (Attention Scores: Q @ K^T -> softmax -> int8)
    # =========================================================================
    print("-" * 90)
    print("TEST 1: Fused Softmax (matmul + softmax + quantize)")
    print("Use case: Attention scores computation (Q @ K^T)")
    print("-" * 90)

    softmax_cases = [
        # (M, N, K, description) - N is sequence length for attention
        (1, 512, 64, "Single token, seq=512, head_dim=64"),
        (1, 1024, 64, "Single token, seq=1024, head_dim=64"),
        (1, 2048, 64, "Single token, seq=2048, head_dim=64"),
        (8, 512, 64, "Small batch, seq=512, head_dim=64"),
        (8, 1024, 64, "Small batch, seq=1024, head_dim=64"),
        (8, 2048, 64, "Small batch, seq=2048, head_dim=64"),
        (32, 512, 64, "Medium batch, seq=512, head_dim=64"),
        (32, 1024, 64, "Medium batch, seq=1024, head_dim=64"),
        (32, 2048, 64, "Medium batch, seq=2048, head_dim=64"),
    ]

    print(f"{'Description':<45} {'M':<5} {'N':<6} {'K':<5} {'Time (ms)':<12} {'GOPS':<10}")
    print("-" * 90)

    for M, N, K, desc in softmax_cases:
        K_padded = get_k_padded(K)
        threads = get_threads(M)

        # Prepare data
        x_int8 = torch.randint(-128, 127, (M, K_padded), dtype=torch.int8).contiguous()
        x_scales = torch.rand(M, dtype=torch.float32).contiguous() * 0.1
        w_int8 = torch.randint(-1, 2, (N, K_padded), dtype=torch.int8).contiguous()
        w_sum = torch.sum(w_int8.to(torch.int32), dim=1).to(torch.int32).contiguous()

        # Output tensors
        y_int8 = torch.zeros(M, N, dtype=torch.int8).contiguous()
        y_scales = torch.zeros(M, dtype=torch.float32).contiguous()

        # Warmup
        for _ in range(3):
            kernel.matmul_free_vnni_v3_fused_softmax(
                x_int8, x_scales, w_int8, w_sum, y_int8, y_scales, M, N, K, threads
            )

        # Benchmark
        start = time.time()
        for _ in range(10):
            kernel.matmul_free_vnni_v3_fused_softmax(
                x_int8, x_scales, w_int8, w_sum, y_int8, y_scales, M, N, K, threads
            )
        elapsed = (time.time() - start) * 1000 / 10

        ops = 2 * M * N * K + 4 * M * N  # matmul + softmax (exp, sum, div, quantize)
        gops = ops / (elapsed * 1e6)

        print(f"{desc:<45} {M:<5} {N:<6} {K:<5} {elapsed:<12.3f} {gops:<10.1f}")

        results.append({
            'test': 'fused_softmax',
            'description': desc,
            'M': M, 'N': N, 'K': K,
            'time_ms': elapsed,
            'gops': gops
        })

    # =========================================================================
    # Test 2: Fused Quantize (Down Projection: matmul -> quantize)
    # =========================================================================
    print()
    print("-" * 90)
    print("TEST 2: Fused Quantize (matmul + quantize)")
    print("Use case: MLP down projection, output layers")
    print("-" * 90)

    quantize_cases = [
        # (M, N, K, description)
        (1, 2048, 8192, "Single token - MLP down (8192->2048)"),
        (1, 2048, 16384, "Single token - MLP down (16384->2048)"),
        (1, 50257, 2048, "Single token - LM head (vocab)"),
        (8, 2048, 8192, "Small batch - MLP down"),
        (8, 2048, 16384, "Small batch - MLP down large"),
        (8, 50257, 2048, "Small batch - LM head"),
        (32, 2048, 8192, "Medium batch - MLP down"),
        (32, 2048, 16384, "Medium batch - MLP down large"),
        (32, 50257, 2048, "Medium batch - LM head"),
        (128, 2048, 8192, "Large batch - MLP down"),
        (128, 2048, 16384, "Large batch - MLP down large"),
    ]

    print(f"{'Description':<45} {'M':<5} {'N':<6} {'K':<6} {'Time (ms)':<12} {'GOPS':<10}")
    print("-" * 90)

    for M, N, K, desc in quantize_cases:
        K_padded = get_k_padded(K)
        threads = get_threads(M)

        # Prepare data
        x_int8 = torch.randint(-128, 127, (M, K_padded), dtype=torch.int8).contiguous()
        x_scales = torch.rand(M, dtype=torch.float32).contiguous() * 0.1
        w_int8 = torch.randint(-1, 2, (N, K_padded), dtype=torch.int8).contiguous()
        w_sum = torch.sum(w_int8.to(torch.int32), dim=1).to(torch.int32).contiguous()

        # Output tensors
        y_int8 = torch.zeros(M, N, dtype=torch.int8).contiguous()
        y_scales = torch.zeros(M, dtype=torch.float32).contiguous()

        # Warmup
        for _ in range(3):
            kernel.matmul_free_vnni_v3_fused_quantize(
                x_int8, x_scales, w_int8, w_sum, y_int8, y_scales, M, N, K, threads
            )

        # Benchmark
        start = time.time()
        for _ in range(10):
            kernel.matmul_free_vnni_v3_fused_quantize(
                x_int8, x_scales, w_int8, w_sum, y_int8, y_scales, M, N, K, threads
            )
        elapsed = (time.time() - start) * 1000 / 10

        ops = 2 * M * N * K + 2 * M * N  # matmul + quantize (scale, round)
        gops = ops / (elapsed * 1e6)

        print(f"{desc:<45} {M:<5} {N:<6} {K:<6} {elapsed:<12.3f} {gops:<10.1f}")

        results.append({
            'test': 'fused_quantize',
            'description': desc,
            'M': M, 'N': N, 'K': K,
            'time_ms': elapsed,
            'gops': gops
        })

    # =========================================================================
    # Test 3: Fused SwiGLU (MLP Up: matmul -> SwiGLU -> quantize)
    # =========================================================================
    print()
    print("-" * 90)
    print("TEST 3: Fused SwiGLU (matmul + SwiGLU activation + quantize)")
    print("Use case: MLP up projection with gated activation")
    print("Note: Input N is 2x output (gate + up weights combined)")
    print("-" * 90)

    swiglu_cases = [
        # (M, N, K, description) - N is total (gate + up), output is N/2
        (1, 16384, 2048, "Single token - MLP up (2048->8192 after SwiGLU)"),
        (1, 22016, 2048, "Single token - Llama MLP (2048->11008)"),
        (8, 16384, 2048, "Small batch - MLP up"),
        (8, 22016, 2048, "Small batch - Llama MLP"),
        (32, 16384, 2048, "Medium batch - MLP up"),
        (32, 22016, 2048, "Medium batch - Llama MLP"),
        (128, 16384, 2048, "Large batch - MLP up"),
        (128, 22016, 2048, "Large batch - Llama MLP"),
    ]

    print(f"{'Description':<45} {'M':<5} {'N':<6} {'K':<5} {'Out':<6} {'Time (ms)':<12} {'GOPS':<10}")
    print("-" * 90)

    for M, N, K, desc in swiglu_cases:
        K_padded = get_k_padded(K)
        threads = get_threads(M)
        out_dim = N // 2

        # Prepare data
        x_int8 = torch.randint(-128, 127, (M, K_padded), dtype=torch.int8).contiguous()
        x_scales = torch.rand(M, dtype=torch.float32).contiguous() * 0.1
        w_int8 = torch.randint(-1, 2, (N, K_padded), dtype=torch.int8).contiguous()
        w_sum = torch.sum(w_int8.to(torch.int32), dim=1).to(torch.int32).contiguous()

        # Output tensors (N/2 because SwiGLU halves the output)
        y_int8 = torch.zeros(M, out_dim, dtype=torch.int8).contiguous()
        y_scales = torch.zeros(M, dtype=torch.float32).contiguous()

        # Warmup
        for _ in range(3):
            kernel.matmul_free_vnni_v3_fused_swiglu_optimized(
                x_int8, x_scales, w_int8, w_sum, y_int8, y_scales, M, N, K, threads
            )

        # Benchmark
        start = time.time()
        for _ in range(10):
            kernel.matmul_free_vnni_v3_fused_swiglu_optimized(
                x_int8, x_scales, w_int8, w_sum, y_int8, y_scales, M, N, K, threads
            )
        elapsed = (time.time() - start) * 1000 / 10

        # 2 matmuls (gate + up) + SwiGLU ops + quantize
        ops = 2 * M * N * K + 5 * M * out_dim  # matmuls + sigmoid + mul + quantize
        gops = ops / (elapsed * 1e6)

        print(f"{desc:<45} {M:<5} {N:<6} {K:<5} {out_dim:<6} {elapsed:<12.3f} {gops:<10.1f}")

        results.append({
            'test': 'fused_swiglu',
            'description': desc,
            'M': M, 'N': N, 'K': K,
            'out_dim': out_dim,
            'time_ms': elapsed,
            'gops': gops
        })

    # =========================================================================
    # Summary
    # =========================================================================
    print()
    print("=" * 90)
    print("FUSED KERNEL PERFORMANCE SUMMARY")
    print("=" * 90)

    for test_name in ['fused_softmax', 'fused_quantize', 'fused_swiglu']:
        test_results = [r for r in results if r['test'] == test_name]
        if test_results:
            avg_gops = sum(r['gops'] for r in test_results) / len(test_results)
            total_time = sum(r['time_ms'] for r in test_results)
            print(f"{test_name:<20}: {len(test_results)} tests, {avg_gops:.1f} avg GOPS, {total_time:.1f}ms total")

    overall_avg = sum(r['gops'] for r in results) / len(results)
    print(f"\nOverall average: {overall_avg:.1f} GOPS")
    print("Fused kernel inference test completed successfully!")

    return results


def compare_fused_vs_separate():
    """Compare fused kernels against separate operations and PyTorch baseline."""

    print("\n" + "=" * 100)
    print("COMPARISON: PyTorch FP32 vs Separate VNNI vs Fused VNNI")
    print("=" * 100)

    arch = platform.machine().lower()
    if arch not in ['x86_64', 'amd64']:
        print(f"Unsupported architecture: {arch}")
        return

    kernel = load(
        name="vnni_kernel_compare_fused",
        sources=["src/cpu_ops/kernels/x86_64/matmul_free_tmac_vnni.cpp"],
        extra_cflags=['-O3', '-march=native', '-ffast-math', '-fopenmp'],
        extra_ldflags=['-lgomp'],
        verbose=False
    )

    def get_k_padded(K):
        return ((K + 63) // 64) * 64

    threads = min(os.cpu_count() or 8, 8)

    test_sizes = [
        (1, 2048, 2048, "Single token"),
        (1, 8192, 2048, "Single token - large N"),
        (8, 2048, 2048, "Small batch"),
        (8, 8192, 2048, "Small batch - large N"),
        (32, 2048, 2048, "Medium batch"),
        (32, 8192, 2048, "Medium batch - large N"),
        (128, 2048, 2048, "Large batch"),
        (128, 8192, 2048, "Large batch - large N"),
    ]

    print(f"\n{'Description':<25} {'PyTorch':<12} {'Separate':<12} {'Fused':<12} {'vs PyTorch':<12} {'vs Separate':<12}")
    print("-" * 100)

    results = []

    for M, N, K, desc in test_sizes:
        K_padded = get_k_padded(K)

        # =====================================================================
        # PyTorch FP32 baseline: matmul + softmax
        # =====================================================================
        x_float = torch.randn(M, K, dtype=torch.float32)
        w_float = torch.randint(-1, 2, (N, K), dtype=torch.float32)

        # Warmup PyTorch
        for _ in range(3):
            y_pt = torch.mm(x_float, w_float.T)
            y_pt = torch.softmax(y_pt, dim=-1)

        start = time.time()
        for _ in range(10):
            y_pt = torch.mm(x_float, w_float.T)
            y_pt = torch.softmax(y_pt, dim=-1)
        pytorch_time = (time.time() - start) * 1000 / 10

        # =====================================================================
        # Separate VNNI: matmul kernel + PyTorch softmax + quantize
        # =====================================================================
        x_int8 = torch.randint(-128, 127, (M, K_padded), dtype=torch.int8).contiguous()
        x_scales = torch.rand(M, dtype=torch.float32).contiguous() * 0.1
        w_int8 = torch.randint(-1, 2, (N, K_padded), dtype=torch.int8).contiguous()
        w_sum = torch.sum(w_int8.to(torch.int32), dim=1).to(torch.int32).contiguous()

        y_int32 = torch.zeros(M, N, dtype=torch.int32).contiguous()

        # Warmup separate
        for _ in range(3):
            kernel.matmul_free_vnni_v4_large_n(x_int8, x_scales, w_int8, w_sum, y_int32, M, N, K, threads)
            y_float = y_int32.float() * x_scales.unsqueeze(1)
            y_float = torch.softmax(y_float, dim=-1)
            y_int8_sep = (y_float * 127).clamp(-127, 127).to(torch.int8)

        start = time.time()
        for _ in range(10):
            kernel.matmul_free_vnni_v4_large_n(x_int8, x_scales, w_int8, w_sum, y_int32, M, N, K, threads)
            y_float = y_int32.float() * x_scales.unsqueeze(1)
            y_float = torch.softmax(y_float, dim=-1)
            y_int8_sep = (y_float * 127).clamp(-127, 127).to(torch.int8)
        separate_time = (time.time() - start) * 1000 / 10

        # =====================================================================
        # Fused VNNI: matmul + softmax + quantize in one kernel
        # =====================================================================
        y_int8_fused = torch.zeros(M, N, dtype=torch.int8).contiguous()
        y_scales_out = torch.zeros(M, dtype=torch.float32).contiguous()

        for _ in range(3):
            kernel.matmul_free_vnni_v3_fused_softmax(
                x_int8, x_scales, w_int8, w_sum, y_int8_fused, y_scales_out, M, N, K, threads
            )

        start = time.time()
        for _ in range(10):
            kernel.matmul_free_vnni_v3_fused_softmax(
                x_int8, x_scales, w_int8, w_sum, y_int8_fused, y_scales_out, M, N, K, threads
            )
        fused_time = (time.time() - start) * 1000 / 10

        # Calculate speedups
        speedup_vs_pytorch = pytorch_time / fused_time
        speedup_vs_separate = separate_time / fused_time

        print(f"{desc:<25} {pytorch_time:<12.3f} {separate_time:<12.3f} {fused_time:<12.3f} {speedup_vs_pytorch:<12.2f}x {speedup_vs_separate:<12.2f}x")

        results.append({
            'desc': desc, 'M': M, 'N': N, 'K': K,
            'pytorch_ms': pytorch_time,
            'separate_ms': separate_time,
            'fused_ms': fused_time,
            'speedup_vs_pytorch': speedup_vs_pytorch,
            'speedup_vs_separate': speedup_vs_separate
        })

    # Summary
    print()
    print("-" * 100)
    avg_vs_pytorch = sum(r['speedup_vs_pytorch'] for r in results) / len(results)
    avg_vs_separate = sum(r['speedup_vs_separate'] for r in results) / len(results)
    print(f"{'AVERAGE':<25} {'':<12} {'':<12} {'':<12} {avg_vs_pytorch:<12.2f}x {avg_vs_separate:<12.2f}x")

    # Best/worst cases
    best_pytorch = max(results, key=lambda x: x['speedup_vs_pytorch'])
    worst_pytorch = min(results, key=lambda x: x['speedup_vs_pytorch'])
    print()
    print(f"Best vs PyTorch:  {best_pytorch['speedup_vs_pytorch']:.2f}x at M={best_pytorch['M']}, N={best_pytorch['N']}")
    print(f"Worst vs PyTorch: {worst_pytorch['speedup_vs_pytorch']:.2f}x at M={worst_pytorch['M']}, N={worst_pytorch['N']}")

    print()
    print("Legend:")
    print("  PyTorch:  torch.mm() + torch.softmax() [FP32]")
    print("  Separate: VNNI matmul kernel + torch.softmax() + quantize [int8]")
    print("  Fused:    Single fused_softmax kernel [int8, all ops combined]")


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description='Fused Kernel Inference Performance Test')
    parser.add_argument('--compare', action='store_true', help='Compare fused vs separate operations')
    args = parser.parse_args()

    # Print system info
    print("System Information:")
    print(f"  Platform: {platform.platform()}")
    print(f"  Machine: {platform.machine()}")
    print(f"  CPU count: {os.cpu_count()}")
    print(f"  PyTorch threads: {torch.get_num_threads()}")
    print()

    test_fused_inference_patterns()

    if args.compare:
        compare_fused_vs_separate()
