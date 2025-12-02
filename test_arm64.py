#!/usr/bin/env python3
"""
ARM64 Benchmark Script for NEON Kernels

Tests ARM64 kernels against PyTorch baseline:
1. SDOT-based int8 kernels (fast, uses multiplication)
2. TBL-based 2-bit kernels (true MatMul-free, 16x memory savings)

Supports Apple Silicon (M1/M2/M3/M4) and generic ARM64 with dotprod extension.

Usage:
    python test_arm64.py              # Run all benchmarks
    python test_arm64.py --tbl-only   # Run only TBL (true MatMul-free) benchmarks
"""

import torch
import numpy as np
import time
import gc
import os
import platform
import sys
from pathlib import Path

# Check architecture
machine = platform.machine().lower()
if machine not in ['arm64', 'aarch64']:
    print(f"Warning: This script is designed for ARM64, detected: {machine}")
    print("Continuing anyway for development purposes...")

# PyTorch JIT compilation
from torch.utils.cpp_extension import load

# Paths
SCRIPT_DIR = Path(__file__).parent
KERNELS_DIR = SCRIPT_DIR / "src" / "cpu_ops" / "kernels"

# Detect platform for compiler flags
system = platform.system()
IS_APPLE_SILICON = False
if system == 'Darwin':
    # Apple Silicon (macOS)
    IS_APPLE_SILICON = True
    extra_cflags = [
        '-O3',
        '-std=c++17',
        '-mcpu=apple-m1',
        '-DAPPLE_SILICON',
        '-Xpreprocessor', '-fopenmp',
        '-I/opt/homebrew/opt/libomp/include'
    ]
    extra_ldflags = [
        '-L/opt/homebrew/opt/libomp/lib',
        '-lomp',
        '-framework', 'Accelerate'  # Link Apple Accelerate
    ]
else:
    # Generic ARM64 (Linux)
    extra_cflags = [
        '-O3',
        '-std=c++17',
        '-march=armv8.2-a+dotprod',
        '-fopenmp'
    ]
    extra_ldflags = ['-fopenmp', '-lgomp']

print("=" * 70)
print("ARM64 NEON Kernel Benchmarks")
print("=" * 70)
print(f"Platform: {system} ({machine})")
print(f"PyTorch version: {torch.__version__}")
print(f"Compiler flags: {' '.join(extra_cflags)}")
print()

# Parse command line args
import argparse
parser = argparse.ArgumentParser()
parser.add_argument('--tbl-only', action='store_true', help='Run only TBL (true MatMul-free) benchmarks')
args, _ = parser.parse_known_args()

# Load SDOT int8 kernel
sdot_kernel = None
if not args.tbl_only:
    print("Compiling ARM64 SDOT kernel...")
    try:
        sdot_kernel = load(
            name="neon_sdot_kernel",
            sources=[str(KERNELS_DIR / "arm64" / "matmul_free_neon_int8.cpp")],
            extra_cflags=extra_cflags,
            extra_ldflags=extra_ldflags,
            verbose=False
        )
        print("  SDOT kernel compiled successfully!")
    except Exception as e:
        print(f"  Failed to compile SDOT kernel: {e}")
        print("  Note: SDOT requires ARMv8.2+ with dotprod extension")

# Load TBL (true MatMul-free) kernel
print("Compiling ARM64 TBL kernel (true MatMul-free, 2-bit weights)...")
try:
    tbl_kernel = load(
        name="neon_tbl_kernel",
        sources=[str(KERNELS_DIR / "arm64" / "matmul_free_neon_tbl.cpp")],
        extra_cflags=extra_cflags,
        extra_ldflags=extra_ldflags,
        verbose=False
    )
    print("  TBL kernel compiled successfully!")
except Exception as e:
    print(f"  Failed to compile TBL kernel: {e}")
    tbl_kernel = None

# Load Popcount/Hybrid kernel (2-bit weights + SDOT)
print("Compiling ARM64 Hybrid kernel (2-bit weights + SDOT)...")
try:
    hybrid_kernel = load(
        name="neon_hybrid_kernel",
        sources=[str(KERNELS_DIR / "arm64" / "matmul_free_neon_popcount.cpp")],
        extra_cflags=extra_cflags,
        extra_ldflags=extra_ldflags,
        verbose=False
    )
    print("  Hybrid kernel compiled successfully!")
except Exception as e:
    print(f"  Failed to compile Hybrid kernel: {e}")
    hybrid_kernel = None

# Test configurations
TEST_CONFIGS = [
    ("Small",   128, 1024, 1024),
    ("Medium",  256, 1024, 1024),
    ("Large",   512, 1024, 1024),
    ("XLarge", 1024, 2048, 2048),
]

# Number of threads
num_threads = os.cpu_count() or 8
print(f"Using {num_threads} threads")
print()

# Benchmark parameters
WARMUP_ITERS = 5
BENCH_ITERS = 30


def create_ternary_weights(N, K):
    """Create random ternary weights (-1, 0, +1)"""
    w = torch.randint(-1, 2, (N, K), dtype=torch.float32)
    return w


def run_pytorch_baseline(x, w_ternary, bias):
    """Run PyTorch matmul baseline"""
    return torch.mm(x, w_ternary.t()) + bias


def benchmark_kernel(name, kernel_fn, warmup=WARMUP_ITERS, iters=BENCH_ITERS):
    """Benchmark a kernel and return timing stats"""
    # Warmup
    for _ in range(warmup):
        kernel_fn()

    # Benchmark
    gc.collect()
    times = []
    for _ in range(iters):
        start = time.perf_counter()
        kernel_fn()
        times.append(time.perf_counter() - start)

    return {
        'mean': np.mean(times) * 1000,  # ms
        'std': np.std(times) * 1000,
        'min': np.min(times) * 1000,
        'max': np.max(times) * 1000
    }


def run_benchmarks():
    """Run all benchmarks"""
    all_results = {}

    for config_name, M, N, K in TEST_CONFIGS:
        print("=" * 70)
        print(f"Test: {config_name} (M={M}, N={N}, K={K})")
        print("=" * 70)

        flops = 2.0 * M * N * K
        print(f"FLOPs: {flops / 1e9:.2f}B")

        # Create test data
        x = torch.randn(M, K, dtype=torch.float32)
        w_ternary = create_ternary_weights(N, K)
        bias = torch.randn(N, dtype=torch.float32)

        results = {}

        # PyTorch baseline
        print("\n[PyTorch Baseline]")
        y_pt = run_pytorch_baseline(x, w_ternary, bias)
        pt_stats = benchmark_kernel("PyTorch", lambda: run_pytorch_baseline(x, w_ternary, bias))
        pt_gflops = flops / (pt_stats['mean'] / 1000) / 1e9
        print(f"  Time: {pt_stats['mean']:.3f} ms (+/- {pt_stats['std']:.3f})")
        print(f"  GFLOPS: {pt_gflops:.1f}")
        results['pytorch'] = {'time': pt_stats['mean'], 'gflops': pt_gflops, 'y': y_pt}

        # SDOT Kernels (int8 multiply-based)
        if sdot_kernel is not None:
            # Prepare int8 data
            K_padded = ((K + 15) // 16) * 16
            x_int8 = torch.zeros(M, K, dtype=torch.int8)
            scales = torch.zeros(M, dtype=torch.float32)
            w_int8 = torch.zeros(N, K_padded, dtype=torch.int8)
            w_sum = torch.zeros(N, dtype=torch.int32)

            # Quantize and pack
            sdot_kernel.quantize_activations_int8_neon(x, x_int8, scales, M, K)
            sdot_kernel.pack_weights_neon_int8(w_ternary, w_int8, w_sum, N, K)

            # SDOT Simple
            print("\n[SDOT Simple]")
            y_sdot_simple = torch.zeros(M, N, dtype=torch.float32)

            def run_sdot_simple():
                sdot_kernel.matmul_free_neon_sdot_simple(
                    x_int8, scales, w_int8, w_sum, y_sdot_simple, bias,
                    M, N, K, num_threads
                )

            run_sdot_simple()  # First run for correctness
            sdot_simple_stats = benchmark_kernel("SDOT Simple", run_sdot_simple)
            sdot_simple_gflops = flops / (sdot_simple_stats['mean'] / 1000) / 1e9
            max_diff_simple = torch.max(torch.abs(y_sdot_simple - y_pt)).item()
            rel_error_simple = max_diff_simple / (torch.max(torch.abs(y_pt)).item() + 1e-6)
            print(f"  Time: {sdot_simple_stats['mean']:.3f} ms (+/- {sdot_simple_stats['std']:.3f})")
            print(f"  GFLOPS: {sdot_simple_gflops:.1f}")
            print(f"  Max diff: {max_diff_simple:.4f} (rel: {rel_error_simple*100:.2f}%)")
            print(f"  Speedup vs PyTorch: {pt_stats['mean'] / sdot_simple_stats['mean']:.2f}x")
            results['sdot_simple'] = {'time': sdot_simple_stats['mean'], 'gflops': sdot_simple_gflops, 'y': y_sdot_simple}

            # SDOT v2 (4-way blocking)
            print("\n[SDOT v2 (4-way blocking)]")
            y_sdot_v2 = torch.zeros(M, N, dtype=torch.float32)

            def run_sdot_v2():
                sdot_kernel.matmul_free_neon_sdot_v2(
                    x_int8, scales, w_int8, w_sum, y_sdot_v2, bias,
                    M, N, K, num_threads
                )

            run_sdot_v2()
            sdot_v2_stats = benchmark_kernel("SDOT v2", run_sdot_v2)
            sdot_v2_gflops = flops / (sdot_v2_stats['mean'] / 1000) / 1e9
            max_diff_v2 = torch.max(torch.abs(y_sdot_v2 - y_pt)).item()
            rel_error_v2 = max_diff_v2 / (torch.max(torch.abs(y_pt)).item() + 1e-6)
            print(f"  Time: {sdot_v2_stats['mean']:.3f} ms (+/- {sdot_v2_stats['std']:.3f})")
            print(f"  GFLOPS: {sdot_v2_gflops:.1f}")
            print(f"  Max diff: {max_diff_v2:.4f} (rel: {rel_error_v2*100:.2f}%)")
            print(f"  Speedup vs PyTorch: {pt_stats['mean'] / sdot_v2_stats['mean']:.2f}x")
            results['sdot_v2'] = {'time': sdot_v2_stats['mean'], 'gflops': sdot_v2_gflops, 'y': y_sdot_v2}

            # SDOT v3 (M+N tiling)
            print("\n[SDOT v3 (M+N tiling)]")
            y_sdot_v3 = torch.zeros(M, N, dtype=torch.float32)

            def run_sdot_v3():
                sdot_kernel.matmul_free_neon_sdot_v3(
                    x_int8, scales, w_int8, w_sum, y_sdot_v3, bias,
                    M, N, K, num_threads
                )

            run_sdot_v3()
            sdot_v3_stats = benchmark_kernel("SDOT v3", run_sdot_v3)
            sdot_v3_gflops = flops / (sdot_v3_stats['mean'] / 1000) / 1e9
            max_diff_v3 = torch.max(torch.abs(y_sdot_v3 - y_pt)).item()
            rel_error_v3 = max_diff_v3 / (torch.max(torch.abs(y_pt)).item() + 1e-6)
            print(f"  Time: {sdot_v3_stats['mean']:.3f} ms (+/- {sdot_v3_stats['std']:.3f})")
            print(f"  GFLOPS: {sdot_v3_gflops:.1f}")
            print(f"  Max diff: {max_diff_v3:.4f} (rel: {rel_error_v3*100:.2f}%)")
            print(f"  Speedup vs PyTorch: {pt_stats['mean'] / sdot_v3_stats['mean']:.2f}x")
            results['sdot_v3'] = {'time': sdot_v3_stats['mean'], 'gflops': sdot_v3_gflops, 'y': y_sdot_v3}

            # SDOT v4 (8-way blocking)
            print("\n[SDOT v4 (8-way blocking)]")
            y_sdot_v4 = torch.zeros(M, N, dtype=torch.float32)

            def run_sdot_v4():
                sdot_kernel.matmul_free_neon_sdot_v4(
                    x_int8, scales, w_int8, w_sum, y_sdot_v4, bias,
                    M, N, K, num_threads
                )

            run_sdot_v4()
            sdot_v4_stats = benchmark_kernel("SDOT v4", run_sdot_v4)
            sdot_v4_gflops = flops / (sdot_v4_stats['mean'] / 1000) / 1e9
            max_diff_v4 = torch.max(torch.abs(y_sdot_v4 - y_pt)).item()
            rel_error_v4 = max_diff_v4 / (torch.max(torch.abs(y_pt)).item() + 1e-6)
            print(f"  Time: {sdot_v4_stats['mean']:.3f} ms (+/- {sdot_v4_stats['std']:.3f})")
            print(f"  GFLOPS: {sdot_v4_gflops:.1f}")
            print(f"  Max diff: {max_diff_v4:.4f} (rel: {rel_error_v4*100:.2f}%)")
            print(f"  Speedup vs PyTorch: {pt_stats['mean'] / sdot_v4_stats['mean']:.2f}x")
            results['sdot_v4'] = {'time': sdot_v4_stats['mean'], 'gflops': sdot_v4_gflops, 'y': y_sdot_v4}

        # =====================================================================
        # TBL KERNELS (True MatMul-free, 2-bit weights)
        # =====================================================================
        if tbl_kernel is not None:
            print("\n" + "=" * 50)
            print("TBL KERNELS (True MatMul-free, 2-bit weights)")
            print("=" * 50)

            # Pack weights to 2-bit format (sign_plane + value_plane)
            K_groups = (K + 3) // 4
            sign_plane = torch.zeros(K_groups, N, dtype=torch.uint8)
            value_plane = torch.zeros(K_groups, N, dtype=torch.uint8)
            tbl_kernel.pack_ternary_bitplanes_tbl(w_ternary, sign_plane, value_plane, N, K)

            # Quantize activations for TBL kernel
            x_int8_tbl = torch.zeros(M, K, dtype=torch.int8)
            scales_tbl = torch.zeros(M, dtype=torch.float32)
            tbl_kernel.quantize_activations_int8_tbl(x, x_int8_tbl, scales_tbl, M, K)

            # Calculate memory savings
            memory_float32 = N * K * 4  # bytes
            memory_int8 = N * K * 1
            memory_2bit = K_groups * N * 2  # sign_plane + value_plane
            print(f"  Weight memory: float32={memory_float32/1024:.1f}KB, int8={memory_int8/1024:.1f}KB, 2-bit={memory_2bit/1024:.1f}KB")
            print(f"  Memory savings: {memory_float32/memory_2bit:.1f}x vs float32, {memory_int8/memory_2bit:.1f}x vs int8")

            # TBL v1 (basic)
            print("\n[TBL v1 (16 parallel lookups)]")
            y_tbl = torch.zeros(M, N, dtype=torch.float32)

            def run_tbl():
                tbl_kernel.matmul_free_neon_tbl(
                    x_int8_tbl, scales_tbl, sign_plane, value_plane, y_tbl, bias,
                    M, N, K, num_threads
                )

            run_tbl()  # First run for correctness
            tbl_stats = benchmark_kernel("TBL", run_tbl)
            tbl_gflops = flops / (tbl_stats['mean'] / 1000) / 1e9
            max_diff_tbl = torch.max(torch.abs(y_tbl - y_pt)).item()
            rel_error_tbl = max_diff_tbl / (torch.max(torch.abs(y_pt)).item() + 1e-6)
            print(f"  Time: {tbl_stats['mean']:.3f} ms (+/- {tbl_stats['std']:.3f})")
            print(f"  GFLOPS: {tbl_gflops:.1f}")
            print(f"  Max diff: {max_diff_tbl:.4f} (rel: {rel_error_tbl*100:.2f}%)")
            print(f"  Speedup vs PyTorch: {pt_stats['mean'] / tbl_stats['mean']:.2f}x")
            results['tbl'] = {'time': tbl_stats['mean'], 'gflops': tbl_gflops, 'y': y_tbl, 'memory_kb': memory_2bit/1024}

            # TBL v2 (N-blocked)
            print("\n[TBL v2 (N-blocked, 32 outputs)]")
            y_tbl_v2 = torch.zeros(M, N, dtype=torch.float32)

            def run_tbl_v2():
                tbl_kernel.matmul_free_neon_tbl_v2(
                    x_int8_tbl, scales_tbl, sign_plane, value_plane, y_tbl_v2, bias,
                    M, N, K, num_threads
                )

            run_tbl_v2()
            tbl_v2_stats = benchmark_kernel("TBL v2", run_tbl_v2)
            tbl_v2_gflops = flops / (tbl_v2_stats['mean'] / 1000) / 1e9
            max_diff_tbl_v2 = torch.max(torch.abs(y_tbl_v2 - y_pt)).item()
            rel_error_tbl_v2 = max_diff_tbl_v2 / (torch.max(torch.abs(y_pt)).item() + 1e-6)
            print(f"  Time: {tbl_v2_stats['mean']:.3f} ms (+/- {tbl_v2_stats['std']:.3f})")
            print(f"  GFLOPS: {tbl_v2_gflops:.1f}")
            print(f"  Max diff: {max_diff_tbl_v2:.4f} (rel: {rel_error_tbl_v2*100:.2f}%)")
            print(f"  Speedup vs PyTorch: {pt_stats['mean'] / tbl_v2_stats['mean']:.2f}x")
            results['tbl_v2'] = {'time': tbl_v2_stats['mean'], 'gflops': tbl_v2_gflops, 'y': y_tbl_v2, 'memory_kb': memory_2bit/1024}

            # TBL v3 (K-unrolled + register accumulators)
            print("\n[TBL v3 (K-unroll + reg accum)]")
            y_tbl_v3 = torch.zeros(M, N, dtype=torch.float32)

            def run_tbl_v3():
                tbl_kernel.matmul_free_neon_tbl_v3(
                    x_int8_tbl, scales_tbl, sign_plane, value_plane, y_tbl_v3, bias,
                    M, N, K, num_threads
                )

            run_tbl_v3()
            tbl_v3_stats = benchmark_kernel("TBL v3", run_tbl_v3)
            tbl_v3_gflops = flops / (tbl_v3_stats['mean'] / 1000) / 1e9
            max_diff_tbl_v3 = torch.max(torch.abs(y_tbl_v3 - y_pt)).item()
            rel_error_tbl_v3 = max_diff_tbl_v3 / (torch.max(torch.abs(y_pt)).item() + 1e-6)
            print(f"  Time: {tbl_v3_stats['mean']:.3f} ms (+/- {tbl_v3_stats['std']:.3f})")
            print(f"  GFLOPS: {tbl_v3_gflops:.1f}")
            print(f"  Max diff: {max_diff_tbl_v3:.4f} (rel: {rel_error_tbl_v3*100:.2f}%)")
            print(f"  Speedup vs PyTorch: {pt_stats['mean'] / tbl_v3_stats['mean']:.2f}x")
            results['tbl_v3'] = {'time': tbl_v3_stats['mean'], 'gflops': tbl_v3_gflops, 'y': y_tbl_v3, 'memory_kb': memory_2bit/1024}

            # TBL v4 (Cache-aware tiled)
            print("\n[TBL v4 (Cache-aware tiled)]")
            y_tbl_v4 = torch.zeros(M, N, dtype=torch.float32)

            def run_tbl_v4():
                tbl_kernel.matmul_free_neon_tbl_v4(
                    x_int8_tbl, scales_tbl, sign_plane, value_plane, y_tbl_v4, bias,
                    M, N, K, num_threads
                )

            run_tbl_v4()
            tbl_v4_stats = benchmark_kernel("TBL v4", run_tbl_v4)
            tbl_v4_gflops = flops / (tbl_v4_stats['mean'] / 1000) / 1e9
            max_diff_tbl_v4 = torch.max(torch.abs(y_tbl_v4 - y_pt)).item()
            rel_error_tbl_v4 = max_diff_tbl_v4 / (torch.max(torch.abs(y_pt)).item() + 1e-6)
            print(f"  Time: {tbl_v4_stats['mean']:.3f} ms (+/- {tbl_v4_stats['std']:.3f})")
            print(f"  GFLOPS: {tbl_v4_gflops:.1f}")
            print(f"  Max diff: {max_diff_tbl_v4:.4f} (rel: {rel_error_tbl_v4*100:.2f}%)")
            print(f"  Speedup vs PyTorch: {pt_stats['mean'] / tbl_v4_stats['mean']:.2f}x")
            results['tbl_v4'] = {'time': tbl_v4_stats['mean'], 'gflops': tbl_v4_gflops, 'y': y_tbl_v4, 'memory_kb': memory_2bit/1024}

            # TBL v5 (Vectorized LUT + 64-way N-blocking)
            print("\n[TBL v5 (Vectorized LUT + 64-way)]")
            y_tbl_v5 = torch.zeros(M, N, dtype=torch.float32)

            def run_tbl_v5():
                tbl_kernel.matmul_free_neon_tbl_v5(
                    x_int8_tbl, scales_tbl, sign_plane, value_plane, y_tbl_v5, bias,
                    M, N, K, num_threads
                )

            run_tbl_v5()
            tbl_v5_stats = benchmark_kernel("TBL v5", run_tbl_v5)
            tbl_v5_gflops = flops / (tbl_v5_stats['mean'] / 1000) / 1e9
            max_diff_tbl_v5 = torch.max(torch.abs(y_tbl_v5 - y_pt)).item()
            rel_error_tbl_v5 = max_diff_tbl_v5 / (torch.max(torch.abs(y_pt)).item() + 1e-6)
            print(f"  Time: {tbl_v5_stats['mean']:.3f} ms (+/- {tbl_v5_stats['std']:.3f})")
            print(f"  GFLOPS: {tbl_v5_gflops:.1f}")
            print(f"  Max diff: {max_diff_tbl_v5:.4f} (rel: {rel_error_tbl_v5*100:.2f}%)")
            print(f"  Speedup vs PyTorch: {pt_stats['mean'] / tbl_v5_stats['mean']:.2f}x")
            results['tbl_v5'] = {'time': tbl_v5_stats['mean'], 'gflops': tbl_v5_gflops, 'y': y_tbl_v5, 'memory_kb': memory_2bit/1024}

            # TBL v6 (M-batched weight loading)
            print("\n[TBL v6 (M-batched weight loading)]")
            y_tbl_v6 = torch.zeros(M, N, dtype=torch.float32)

            def run_tbl_v6():
                tbl_kernel.matmul_free_neon_tbl_v6(
                    x_int8_tbl, scales_tbl, sign_plane, value_plane, y_tbl_v6, bias,
                    M, N, K, num_threads
                )

            run_tbl_v6()
            tbl_v6_stats = benchmark_kernel("TBL v6", run_tbl_v6)
            tbl_v6_gflops = flops / (tbl_v6_stats['mean'] / 1000) / 1e9
            max_diff_tbl_v6 = torch.max(torch.abs(y_tbl_v6 - y_pt)).item()
            rel_error_tbl_v6 = max_diff_tbl_v6 / (torch.max(torch.abs(y_pt)).item() + 1e-6)
            print(f"  Time: {tbl_v6_stats['mean']:.3f} ms (+/- {tbl_v6_stats['std']:.3f})")
            print(f"  GFLOPS: {tbl_v6_gflops:.1f}")
            print(f"  Max diff: {max_diff_tbl_v6:.4f} (rel: {rel_error_tbl_v6*100:.2f}%)")
            print(f"  Speedup vs PyTorch: {pt_stats['mean'] / tbl_v6_stats['mean']:.2f}x")
            results['tbl_v6'] = {'time': tbl_v6_stats['mean'], 'gflops': tbl_v6_gflops, 'y': y_tbl_v6, 'memory_kb': memory_2bit/1024}

            # TBL v7 (Fused register accumulation)
            print("\n[TBL v7 (Fused register accum)]")
            y_tbl_v7 = torch.zeros(M, N, dtype=torch.float32)

            def run_tbl_v7():
                tbl_kernel.matmul_free_neon_tbl_v7(
                    x_int8_tbl, scales_tbl, sign_plane, value_plane, y_tbl_v7, bias,
                    M, N, K, num_threads
                )

            run_tbl_v7()
            tbl_v7_stats = benchmark_kernel("TBL v7", run_tbl_v7)
            tbl_v7_gflops = flops / (tbl_v7_stats['mean'] / 1000) / 1e9
            max_diff_tbl_v7 = torch.max(torch.abs(y_tbl_v7 - y_pt)).item()
            rel_error_tbl_v7 = max_diff_tbl_v7 / (torch.max(torch.abs(y_pt)).item() + 1e-6)
            print(f"  Time: {tbl_v7_stats['mean']:.3f} ms (+/- {tbl_v7_stats['std']:.3f})")
            print(f"  GFLOPS: {tbl_v7_gflops:.1f}")
            print(f"  Max diff: {max_diff_tbl_v7:.4f} (rel: {rel_error_tbl_v7*100:.2f}%)")
            print(f"  Speedup vs PyTorch: {pt_stats['mean'] / tbl_v7_stats['mean']:.2f}x")
            results['tbl_v7'] = {'time': tbl_v7_stats['mean'], 'gflops': tbl_v7_gflops, 'y': y_tbl_v7, 'memory_kb': memory_2bit/1024}

            # TBL v8 (Software pipelining + M-batching)
            print("\n[TBL v8 (SW pipeline + M-batch)]")
            y_tbl_v8 = torch.zeros(M, N, dtype=torch.float32)

            def run_tbl_v8():
                tbl_kernel.matmul_free_neon_tbl_v8(
                    x_int8_tbl, scales_tbl, sign_plane, value_plane, y_tbl_v8, bias,
                    M, N, K, num_threads
                )

            run_tbl_v8()
            tbl_v8_stats = benchmark_kernel("TBL v8", run_tbl_v8)
            tbl_v8_gflops = flops / (tbl_v8_stats['mean'] / 1000) / 1e9
            max_diff_tbl_v8 = torch.max(torch.abs(y_tbl_v8 - y_pt)).item()
            rel_error_tbl_v8 = max_diff_tbl_v8 / (torch.max(torch.abs(y_pt)).item() + 1e-6)
            print(f"  Time: {tbl_v8_stats['mean']:.3f} ms (+/- {tbl_v8_stats['std']:.3f})")
            print(f"  GFLOPS: {tbl_v8_gflops:.1f}")
            print(f"  Max diff: {max_diff_tbl_v8:.4f} (rel: {rel_error_tbl_v8*100:.2f}%)")
            print(f"  Speedup vs PyTorch: {pt_stats['mean'] / tbl_v8_stats['mean']:.2f}x")
            results['tbl_v8'] = {'time': tbl_v8_stats['mean'], 'gflops': tbl_v8_gflops, 'y': y_tbl_v8, 'memory_kb': memory_2bit/1024}

            # TBL v9 (N-first + register accumulators)
            print("\n[TBL v9 (N-first + reg accum)]")
            y_tbl_v9 = torch.zeros(M, N, dtype=torch.float32)

            def run_tbl_v9():
                tbl_kernel.matmul_free_neon_tbl_v9(
                    x_int8_tbl, scales_tbl, sign_plane, value_plane, y_tbl_v9, bias,
                    M, N, K, num_threads
                )

            run_tbl_v9()
            tbl_v9_stats = benchmark_kernel("TBL v9", run_tbl_v9)
            tbl_v9_gflops = flops / (tbl_v9_stats['mean'] / 1000) / 1e9
            max_diff_tbl_v9 = torch.max(torch.abs(y_tbl_v9 - y_pt)).item()
            rel_error_tbl_v9 = max_diff_tbl_v9 / (torch.max(torch.abs(y_pt)).item() + 1e-6)
            print(f"  Time: {tbl_v9_stats['mean']:.3f} ms (+/- {tbl_v9_stats['std']:.3f})")
            print(f"  GFLOPS: {tbl_v9_gflops:.1f}")
            print(f"  Max diff: {max_diff_tbl_v9:.4f} (rel: {rel_error_tbl_v9*100:.2f}%)")
            print(f"  Speedup vs PyTorch: {pt_stats['mean'] / tbl_v9_stats['mean']:.2f}x")
            results['tbl_v9'] = {'time': tbl_v9_stats['mean'], 'gflops': tbl_v9_gflops, 'y': y_tbl_v9, 'memory_kb': memory_2bit/1024}

            # TBL v10 (K-unroll x2 + M-batch=8)
            print("\n[TBL v10 (K-unroll x2 + M=8)]")
            y_tbl_v10 = torch.zeros(M, N, dtype=torch.float32)

            def run_tbl_v10():
                tbl_kernel.matmul_free_neon_tbl_v10(
                    x_int8_tbl, scales_tbl, sign_plane, value_plane, y_tbl_v10, bias,
                    M, N, K, num_threads
                )

            run_tbl_v10()
            tbl_v10_stats = benchmark_kernel("TBL v10", run_tbl_v10)
            tbl_v10_gflops = flops / (tbl_v10_stats['mean'] / 1000) / 1e9
            max_diff_tbl_v10 = torch.max(torch.abs(y_tbl_v10 - y_pt)).item()
            rel_error_tbl_v10 = max_diff_tbl_v10 / (torch.max(torch.abs(y_pt)).item() + 1e-6)
            print(f"  Time: {tbl_v10_stats['mean']:.3f} ms (+/- {tbl_v10_stats['std']:.3f})")
            print(f"  GFLOPS: {tbl_v10_gflops:.1f}")
            print(f"  Max diff: {max_diff_tbl_v10:.4f} (rel: {rel_error_tbl_v10*100:.2f}%)")
            print(f"  Speedup vs PyTorch: {pt_stats['mean'] / tbl_v10_stats['mean']:.2f}x")
            results['tbl_v10'] = {'time': tbl_v10_stats['mean'], 'gflops': tbl_v10_gflops, 'y': y_tbl_v10, 'memory_kb': memory_2bit/1024}

            # TBL v11 (M-batch=16 + K-unroll x4)
            print("\n[TBL v11 (M=16 + K-unroll x4)]")
            y_tbl_v11 = torch.zeros(M, N, dtype=torch.float32)

            def run_tbl_v11():
                tbl_kernel.matmul_free_neon_tbl_v11(
                    x_int8_tbl, scales_tbl, sign_plane, value_plane, y_tbl_v11, bias,
                    M, N, K, num_threads
                )

            run_tbl_v11()
            tbl_v11_stats = benchmark_kernel("TBL v11", run_tbl_v11)
            tbl_v11_gflops = flops / (tbl_v11_stats['mean'] / 1000) / 1e9
            max_diff_tbl_v11 = torch.max(torch.abs(y_tbl_v11 - y_pt)).item()
            rel_error_tbl_v11 = max_diff_tbl_v11 / (torch.max(torch.abs(y_pt)).item() + 1e-6)
            print(f"  Time: {tbl_v11_stats['mean']:.3f} ms (+/- {tbl_v11_stats['std']:.3f})")
            print(f"  GFLOPS: {tbl_v11_gflops:.1f}")
            print(f"  Max diff: {max_diff_tbl_v11:.4f} (rel: {rel_error_tbl_v11*100:.2f}%)")
            print(f"  Speedup vs PyTorch: {pt_stats['mean'] / tbl_v11_stats['mean']:.2f}x")
            results['tbl_v11'] = {'time': tbl_v11_stats['mean'], 'gflops': tbl_v11_gflops, 'y': y_tbl_v11, 'memory_kb': memory_2bit/1024}

            # TBL v12 (vqtbl2q dual-table)
            print("\n[TBL v12 (vqtbl2q dual-table)]")
            y_tbl_v12 = torch.zeros(M, N, dtype=torch.float32)

            def run_tbl_v12():
                tbl_kernel.matmul_free_neon_tbl_v12(
                    x_int8_tbl, scales_tbl, sign_plane, value_plane, y_tbl_v12, bias,
                    M, N, K, num_threads
                )

            run_tbl_v12()
            tbl_v12_stats = benchmark_kernel("TBL v12", run_tbl_v12)
            tbl_v12_gflops = flops / (tbl_v12_stats['mean'] / 1000) / 1e9
            max_diff_tbl_v12 = torch.max(torch.abs(y_tbl_v12 - y_pt)).item()
            rel_error_tbl_v12 = max_diff_tbl_v12 / (torch.max(torch.abs(y_pt)).item() + 1e-6)
            print(f"  Time: {tbl_v12_stats['mean']:.3f} ms (+/- {tbl_v12_stats['std']:.3f})")
            print(f"  GFLOPS: {tbl_v12_gflops:.1f}")
            print(f"  Max diff: {max_diff_tbl_v12:.4f} (rel: {rel_error_tbl_v12*100:.2f}%)")
            print(f"  Speedup vs PyTorch: {pt_stats['mean'] / tbl_v12_stats['mean']:.2f}x")
            results['tbl_v12'] = {'time': tbl_v12_stats['mean'], 'gflops': tbl_v12_gflops, 'y': y_tbl_v12, 'memory_kb': memory_2bit/1024}

            # TBL v13 (Direct ternary g=2 encoding - 2 TBL instead of 4!)
            print("\n[TBL v13 (Direct g=2, 2 TBL)]")
            K_groups_g2 = (K + 1) // 2
            packed_g2 = torch.zeros(K_groups_g2, N, dtype=torch.uint8)
            tbl_kernel.pack_ternary_direct_g2(w_ternary, packed_g2, N, K)
            y_tbl_v13 = torch.zeros(M, N, dtype=torch.float32)

            def run_tbl_v13():
                tbl_kernel.matmul_free_neon_tbl_v13(
                    x_int8_tbl, scales_tbl, packed_g2, y_tbl_v13, bias,
                    M, N, K, num_threads
                )

            run_tbl_v13()
            tbl_v13_stats = benchmark_kernel("TBL v13", run_tbl_v13)
            tbl_v13_gflops = flops / (tbl_v13_stats['mean'] / 1000) / 1e9
            max_diff_tbl_v13 = torch.max(torch.abs(y_tbl_v13 - y_pt)).item()
            rel_error_tbl_v13 = max_diff_tbl_v13 / (torch.max(torch.abs(y_pt)).item() + 1e-6)
            print(f"  Time: {tbl_v13_stats['mean']:.3f} ms (+/- {tbl_v13_stats['std']:.3f})")
            print(f"  GFLOPS: {tbl_v13_gflops:.1f}")
            print(f"  Max diff: {max_diff_tbl_v13:.4f} (rel: {rel_error_tbl_v13*100:.2f}%)")
            print(f"  Speedup vs PyTorch: {pt_stats['mean'] / tbl_v13_stats['mean']:.2f}x")
            results['tbl_v13'] = {'time': tbl_v13_stats['mean'], 'gflops': tbl_v13_gflops, 'y': y_tbl_v13, 'memory_kb': memory_2bit/1024}

            # TBL v14 (Direct ternary g=4 with vqtbl4q)
            print("\n[TBL v14 (Direct g=4, vqtbl4q)]")
            K_groups_g4 = (K + 3) // 4
            packed_g4 = torch.zeros(K_groups_g4, N, dtype=torch.uint8)
            tbl_kernel.pack_ternary_direct_g4(w_ternary, packed_g4, N, K)
            y_tbl_v14 = torch.zeros(M, N, dtype=torch.float32)

            def run_tbl_v14():
                tbl_kernel.matmul_free_neon_tbl_v14(
                    x_int8_tbl, scales_tbl, packed_g4, y_tbl_v14, bias,
                    M, N, K, num_threads
                )

            run_tbl_v14()
            tbl_v14_stats = benchmark_kernel("TBL v14", run_tbl_v14)
            tbl_v14_gflops = flops / (tbl_v14_stats['mean'] / 1000) / 1e9
            max_diff_tbl_v14 = torch.max(torch.abs(y_tbl_v14 - y_pt)).item()
            rel_error_tbl_v14 = max_diff_tbl_v14 / (torch.max(torch.abs(y_pt)).item() + 1e-6)
            print(f"  Time: {tbl_v14_stats['mean']:.3f} ms (+/- {tbl_v14_stats['std']:.3f})")
            print(f"  GFLOPS: {tbl_v14_gflops:.1f}")
            print(f"  Max diff: {max_diff_tbl_v14:.4f} (rel: {rel_error_tbl_v14*100:.2f}%)")
            print(f"  Speedup vs PyTorch: {pt_stats['mean'] / tbl_v14_stats['mean']:.2f}x")
            results['tbl_v14'] = {'time': tbl_v14_stats['mean'], 'gflops': tbl_v14_gflops, 'y': y_tbl_v14, 'memory_kb': memory_2bit/1024}

        # =====================================================================
        # HYBRID KERNELS (2-bit weights + SDOT computation)
        # =====================================================================
        if hybrid_kernel is not None:
            print("\n" + "=" * 50)
            print("HYBRID KERNELS (2-bit weights + SDOT computation)")
            print("=" * 50)

            # Pack weights to popcount format (val_plane + sign_plane)
            K_bytes = (K + 7) // 8
            val_plane_pc = torch.zeros(N, K_bytes, dtype=torch.uint8)
            sign_plane_pc = torch.zeros(N, K_bytes, dtype=torch.uint8)
            hybrid_kernel.pack_ternary_popcount(w_ternary, val_plane_pc, sign_plane_pc, N, K)

            # Quantize activations
            x_int8_hybrid = torch.zeros(M, K, dtype=torch.int8)
            scales_hybrid = torch.zeros(M, dtype=torch.float32)
            hybrid_kernel.quantize_activations_popcount(x, x_int8_hybrid, scales_hybrid, M, K)

            # Calculate memory
            memory_2bit_pc = K_bytes * N * 2
            print(f"  Weight memory: 2-bit={memory_2bit_pc/1024:.1f}KB (same as TBL)")

            # Hybrid v1 (2-bit weights expanded to int8 + SDOT)
            print("\n[Hybrid v1 (2-bit expand + SDOT)]")
            y_hybrid = torch.zeros(M, N, dtype=torch.float32)

            def run_hybrid():
                hybrid_kernel.matmul_free_neon_hybrid(
                    x_int8_hybrid, scales_hybrid, val_plane_pc, sign_plane_pc, y_hybrid, bias,
                    M, N, K, num_threads
                )

            run_hybrid()  # First run for correctness
            hybrid_stats = benchmark_kernel("Hybrid", run_hybrid)
            hybrid_gflops = flops / (hybrid_stats['mean'] / 1000) / 1e9
            max_diff_hybrid = torch.max(torch.abs(y_hybrid - y_pt)).item()
            rel_error_hybrid = max_diff_hybrid / (torch.max(torch.abs(y_pt)).item() + 1e-6)
            print(f"  Time: {hybrid_stats['mean']:.3f} ms (+/- {hybrid_stats['std']:.3f})")
            print(f"  GFLOPS: {hybrid_gflops:.1f}")
            print(f"  Max diff: {max_diff_hybrid:.4f} (rel: {rel_error_hybrid*100:.2f}%)")
            print(f"  Speedup vs PyTorch: {pt_stats['mean'] / hybrid_stats['mean']:.2f}x")
            results['hybrid'] = {'time': hybrid_stats['mean'], 'gflops': hybrid_gflops, 'y': y_hybrid, 'memory_kb': memory_2bit_pc/1024}

            # Hybrid v2 (optimized 2-bit expand + SDOT)
            print("\n[Hybrid v2 (optimized expand + SDOT)]")
            y_hybrid_v2 = torch.zeros(M, N, dtype=torch.float32)

            def run_hybrid_v2():
                hybrid_kernel.matmul_free_neon_hybrid_v2(
                    x_int8_hybrid, scales_hybrid, val_plane_pc, sign_plane_pc, y_hybrid_v2, bias,
                    M, N, K, num_threads
                )

            run_hybrid_v2()
            hybrid_v2_stats = benchmark_kernel("Hybrid v2", run_hybrid_v2)
            hybrid_v2_gflops = flops / (hybrid_v2_stats['mean'] / 1000) / 1e9
            max_diff_hybrid_v2 = torch.max(torch.abs(y_hybrid_v2 - y_pt)).item()
            rel_error_hybrid_v2 = max_diff_hybrid_v2 / (torch.max(torch.abs(y_pt)).item() + 1e-6)
            print(f"  Time: {hybrid_v2_stats['mean']:.3f} ms (+/- {hybrid_v2_stats['std']:.3f})")
            print(f"  GFLOPS: {hybrid_v2_gflops:.1f}")
            print(f"  Max diff: {max_diff_hybrid_v2:.4f} (rel: {rel_error_hybrid_v2*100:.2f}%)")
            print(f"  Speedup vs PyTorch: {pt_stats['mean'] / hybrid_v2_stats['mean']:.2f}x")
            results['hybrid_v2'] = {'time': hybrid_v2_stats['mean'], 'gflops': hybrid_v2_gflops, 'y': y_hybrid_v2, 'memory_kb': memory_2bit_pc/1024}

            # Hybrid v3 (LUT-based expand + SDOT)
            print("\n[Hybrid v3 (LUT expand + SDOT)]")
            y_hybrid_v3 = torch.zeros(M, N, dtype=torch.float32)

            def run_hybrid_v3():
                hybrid_kernel.matmul_free_neon_hybrid_v3(
                    x_int8_hybrid, scales_hybrid, val_plane_pc, sign_plane_pc, y_hybrid_v3, bias,
                    M, N, K, num_threads
                )

            run_hybrid_v3()
            hybrid_v3_stats = benchmark_kernel("Hybrid v3", run_hybrid_v3)
            hybrid_v3_gflops = flops / (hybrid_v3_stats['mean'] / 1000) / 1e9
            max_diff_hybrid_v3 = torch.max(torch.abs(y_hybrid_v3 - y_pt)).item()
            rel_error_hybrid_v3 = max_diff_hybrid_v3 / (torch.max(torch.abs(y_pt)).item() + 1e-6)
            print(f"  Time: {hybrid_v3_stats['mean']:.3f} ms (+/- {hybrid_v3_stats['std']:.3f})")
            print(f"  GFLOPS: {hybrid_v3_gflops:.1f}")
            print(f"  Max diff: {max_diff_hybrid_v3:.4f} (rel: {rel_error_hybrid_v3*100:.2f}%)")
            print(f"  Speedup vs PyTorch: {pt_stats['mean'] / hybrid_v3_stats['mean']:.2f}x")
            results['hybrid_v3'] = {'time': hybrid_v3_stats['mean'], 'gflops': hybrid_v3_gflops, 'y': y_hybrid_v3, 'memory_kb': memory_2bit_pc/1024}

            # Hybrid v4 (N-blocked expand + SDOT)
            print("\n[Hybrid v4 (N-blocked expand + SDOT)]")
            y_hybrid_v4 = torch.zeros(M, N, dtype=torch.float32)

            def run_hybrid_v4():
                hybrid_kernel.matmul_free_neon_hybrid_v4(
                    x_int8_hybrid, scales_hybrid, val_plane_pc, sign_plane_pc, y_hybrid_v4, bias,
                    M, N, K, num_threads
                )

            run_hybrid_v4()
            hybrid_v4_stats = benchmark_kernel("Hybrid v4", run_hybrid_v4)
            hybrid_v4_gflops = flops / (hybrid_v4_stats['mean'] / 1000) / 1e9
            max_diff_hybrid_v4 = torch.max(torch.abs(y_hybrid_v4 - y_pt)).item()
            rel_error_hybrid_v4 = max_diff_hybrid_v4 / (torch.max(torch.abs(y_pt)).item() + 1e-6)
            print(f"  Time: {hybrid_v4_stats['mean']:.3f} ms (+/- {hybrid_v4_stats['std']:.3f})")
            print(f"  GFLOPS: {hybrid_v4_gflops:.1f}")
            print(f"  Max diff: {max_diff_hybrid_v4:.4f} (rel: {rel_error_hybrid_v4*100:.2f}%)")
            print(f"  Speedup vs PyTorch: {pt_stats['mean'] / hybrid_v4_stats['mean']:.2f}x")
            results['hybrid_v4'] = {'time': hybrid_v4_stats['mean'], 'gflops': hybrid_v4_gflops, 'y': y_hybrid_v4, 'memory_kb': memory_2bit_pc/1024}

            # =====================================================================
            # DIRECT SDOT (Pre-expanded int8 weights - maximum speed)
            # =====================================================================
            print("\n" + "=" * 50)
            print("DIRECT SDOT (pre-expanded int8, max speed)")
            print("=" * 50)

            # Pack weights to int8 format
            K_padded = ((K + 15) // 16) * 16
            w_int8_direct = torch.zeros(N, K_padded, dtype=torch.int8)
            w_sum_direct = torch.zeros(N, dtype=torch.int32)
            hybrid_kernel.pack_ternary_int8(w_ternary, w_int8_direct, w_sum_direct, N, K)

            memory_int8 = N * K_padded
            print(f"  Weight memory: int8={memory_int8/1024:.1f}KB (4x more than 2-bit)")

            # Direct SDOT v1
            print("\n[Direct SDOT v1]")
            y_sdot_direct = torch.zeros(M, N, dtype=torch.float32)

            def run_sdot_direct():
                hybrid_kernel.matmul_free_neon_sdot_direct(
                    x_int8_hybrid, scales_hybrid, w_int8_direct, w_sum_direct, y_sdot_direct, bias,
                    M, N, K, num_threads
                )

            run_sdot_direct()
            sdot_direct_stats = benchmark_kernel("SDOT Direct", run_sdot_direct)
            sdot_direct_gflops = flops / (sdot_direct_stats['mean'] / 1000) / 1e9
            max_diff_sdot_direct = torch.max(torch.abs(y_sdot_direct - y_pt)).item()
            rel_error_sdot_direct = max_diff_sdot_direct / (torch.max(torch.abs(y_pt)).item() + 1e-6)
            print(f"  Time: {sdot_direct_stats['mean']:.3f} ms (+/- {sdot_direct_stats['std']:.3f})")
            print(f"  GFLOPS: {sdot_direct_gflops:.1f}")
            print(f"  Max diff: {max_diff_sdot_direct:.4f} (rel: {rel_error_sdot_direct*100:.2f}%)")
            print(f"  Speedup vs PyTorch: {pt_stats['mean'] / sdot_direct_stats['mean']:.2f}x")
            results['sdot_direct'] = {'time': sdot_direct_stats['mean'], 'gflops': sdot_direct_gflops, 'y': y_sdot_direct, 'memory_kb': memory_int8/1024}

            # Direct SDOT v2 (N-blocked)
            print("\n[Direct SDOT v2 (4-way N-blocking)]")
            y_sdot_direct_v2 = torch.zeros(M, N, dtype=torch.float32)

            def run_sdot_direct_v2():
                hybrid_kernel.matmul_free_neon_sdot_direct_v2(
                    x_int8_hybrid, scales_hybrid, w_int8_direct, w_sum_direct, y_sdot_direct_v2, bias,
                    M, N, K, num_threads
                )

            run_sdot_direct_v2()
            sdot_direct_v2_stats = benchmark_kernel("SDOT Direct v2", run_sdot_direct_v2)
            sdot_direct_v2_gflops = flops / (sdot_direct_v2_stats['mean'] / 1000) / 1e9
            max_diff_sdot_direct_v2 = torch.max(torch.abs(y_sdot_direct_v2 - y_pt)).item()
            rel_error_sdot_direct_v2 = max_diff_sdot_direct_v2 / (torch.max(torch.abs(y_pt)).item() + 1e-6)
            print(f"  Time: {sdot_direct_v2_stats['mean']:.3f} ms (+/- {sdot_direct_v2_stats['std']:.3f})")
            print(f"  GFLOPS: {sdot_direct_v2_gflops:.1f}")
            print(f"  Max diff: {max_diff_sdot_direct_v2:.4f} (rel: {rel_error_sdot_direct_v2*100:.2f}%)")
            print(f"  Speedup vs PyTorch: {pt_stats['mean'] / sdot_direct_v2_stats['mean']:.2f}x")
            results['sdot_direct_v2'] = {'time': sdot_direct_v2_stats['mean'], 'gflops': sdot_direct_v2_gflops, 'y': y_sdot_direct_v2, 'memory_kb': memory_int8/1024}

            # Direct SDOT v3 (8-way N-blocking + K unrolling)
            print("\n[Direct SDOT v3 (8-way + K unroll)]")
            y_sdot_direct_v3 = torch.zeros(M, N, dtype=torch.float32)

            def run_sdot_direct_v3():
                hybrid_kernel.matmul_free_neon_sdot_direct_v3(
                    x_int8_hybrid, scales_hybrid, w_int8_direct, w_sum_direct, y_sdot_direct_v3, bias,
                    M, N, K, num_threads
                )

            run_sdot_direct_v3()
            sdot_direct_v3_stats = benchmark_kernel("SDOT Direct v3", run_sdot_direct_v3)
            sdot_direct_v3_gflops = flops / (sdot_direct_v3_stats['mean'] / 1000) / 1e9
            max_diff_sdot_direct_v3 = torch.max(torch.abs(y_sdot_direct_v3 - y_pt)).item()
            rel_error_sdot_direct_v3 = max_diff_sdot_direct_v3 / (torch.max(torch.abs(y_pt)).item() + 1e-6)
            print(f"  Time: {sdot_direct_v3_stats['mean']:.3f} ms (+/- {sdot_direct_v3_stats['std']:.3f})")
            print(f"  GFLOPS: {sdot_direct_v3_gflops:.1f}")
            print(f"  Max diff: {max_diff_sdot_direct_v3:.4f} (rel: {rel_error_sdot_direct_v3*100:.2f}%)")
            print(f"  Speedup vs PyTorch: {pt_stats['mean'] / sdot_direct_v3_stats['mean']:.2f}x")
            results['sdot_direct_v3'] = {'time': sdot_direct_v3_stats['mean'], 'gflops': sdot_direct_v3_gflops, 'y': y_sdot_direct_v3, 'memory_kb': memory_int8/1024}

            # Direct SDOT v4 (Tiled for cache)
            print("\n[Direct SDOT v4 (Tiled)]")
            y_sdot_direct_v4 = torch.zeros(M, N, dtype=torch.float32)

            def run_sdot_direct_v4():
                hybrid_kernel.matmul_free_neon_sdot_direct_v4(
                    x_int8_hybrid, scales_hybrid, w_int8_direct, w_sum_direct, y_sdot_direct_v4, bias,
                    M, N, K, num_threads
                )

            run_sdot_direct_v4()
            sdot_direct_v4_stats = benchmark_kernel("SDOT Direct v4", run_sdot_direct_v4)
            sdot_direct_v4_gflops = flops / (sdot_direct_v4_stats['mean'] / 1000) / 1e9
            max_diff_sdot_direct_v4 = torch.max(torch.abs(y_sdot_direct_v4 - y_pt)).item()
            rel_error_sdot_direct_v4 = max_diff_sdot_direct_v4 / (torch.max(torch.abs(y_pt)).item() + 1e-6)
            print(f"  Time: {sdot_direct_v4_stats['mean']:.3f} ms (+/- {sdot_direct_v4_stats['std']:.3f})")
            print(f"  GFLOPS: {sdot_direct_v4_gflops:.1f}")
            print(f"  Max diff: {max_diff_sdot_direct_v4:.4f} (rel: {rel_error_sdot_direct_v4*100:.2f}%)")
            print(f"  Speedup vs PyTorch: {pt_stats['mean'] / sdot_direct_v4_stats['mean']:.2f}x")
            results['sdot_direct_v4'] = {'time': sdot_direct_v4_stats['mean'], 'gflops': sdot_direct_v4_gflops, 'y': y_sdot_direct_v4, 'memory_kb': memory_int8/1024}

        # Apple Accelerate (if available and SDOT kernel is loaded)
        if IS_APPLE_SILICON and sdot_kernel is not None:
            print("\n[Apple Accelerate (int8->f32 conversion)]")
            y_accel = torch.zeros(M, N, dtype=torch.float32)

            # Pack weights without padding for Accelerate
            w_int8_nk = torch.zeros(N, K, dtype=torch.int8)
            for n in range(N):
                for k in range(K):
                    val = w_ternary[n, k].item()
                    if val > 0.5:
                        w_int8_nk[n, k] = 1
                    elif val < -0.5:
                        w_int8_nk[n, k] = -1
                    else:
                        w_int8_nk[n, k] = 0

            def run_accelerate():
                sdot_kernel.matmul_free_accelerate(
                    x, w_int8_nk, y_accel, bias,
                    M, N, K, num_threads
                )

            run_accelerate()
            accel_stats = benchmark_kernel("Accelerate", run_accelerate)
            accel_gflops = flops / (accel_stats['mean'] / 1000) / 1e9
            max_diff_accel = torch.max(torch.abs(y_accel - y_pt)).item()
            rel_error_accel = max_diff_accel / (torch.max(torch.abs(y_pt)).item() + 1e-6)
            print(f"  Time: {accel_stats['mean']:.3f} ms (+/- {accel_stats['std']:.3f})")
            print(f"  GFLOPS: {accel_gflops:.1f}")
            print(f"  Max diff: {max_diff_accel:.6f} (rel: {rel_error_accel*100:.4f}%)")
            print(f"  Speedup vs PyTorch: {pt_stats['mean'] / accel_stats['mean']:.2f}x")
            results['accelerate'] = {'time': accel_stats['mean'], 'gflops': accel_gflops, 'y': y_accel}

            # Accelerate with pre-converted weights (no conversion overhead)
            print("\n[Apple Accelerate (pre-converted f32)]")
            y_accel_f32 = torch.zeros(M, N, dtype=torch.float32)
            w_f32 = torch.zeros(N, K, dtype=torch.float32)
            sdot_kernel.convert_weights_to_f32(w_int8_nk, w_f32, N, K)

            def run_accelerate_f32():
                sdot_kernel.matmul_free_accelerate_f32(
                    x, w_f32, y_accel_f32, bias,
                    M, N, K, num_threads
                )

            run_accelerate_f32()
            accel_f32_stats = benchmark_kernel("Accelerate f32", run_accelerate_f32)
            accel_f32_gflops = flops / (accel_f32_stats['mean'] / 1000) / 1e9
            max_diff_accel_f32 = torch.max(torch.abs(y_accel_f32 - y_pt)).item()
            rel_error_accel_f32 = max_diff_accel_f32 / (torch.max(torch.abs(y_pt)).item() + 1e-6)
            print(f"  Time: {accel_f32_stats['mean']:.3f} ms (+/- {accel_f32_stats['std']:.3f})")
            print(f"  GFLOPS: {accel_f32_gflops:.1f}")
            print(f"  Max diff: {max_diff_accel_f32:.6f} (rel: {rel_error_accel_f32*100:.4f}%)")
            print(f"  Speedup vs PyTorch: {pt_stats['mean'] / accel_f32_stats['mean']:.2f}x")
            results['accelerate_f32'] = {'time': accel_f32_stats['mean'], 'gflops': accel_f32_gflops, 'y': y_accel_f32}

        all_results[config_name] = results

    return all_results


def print_summary(all_results):
    """Print summary tables"""
    print("\n")
    print("=" * 80)
    print("SUMMARY: GFLOPS by Matrix Size")
    print("=" * 80)

    # Build kernel list based on what's available
    kernels = ['pytorch']
    kernel_names = ['PyTorch']

    # Check what kernels are available in first config
    first_config = list(all_results.values())[0]

    if 'sdot_simple' in first_config:
        kernels.extend(['sdot_simple', 'sdot_v2', 'sdot_v3', 'sdot_v4'])
        kernel_names.extend(['SDOT Simple', 'SDOT v2 (4-way)', 'SDOT v3 (unroll)', 'SDOT v4 (8-way)'])

    if 'tbl' in first_config:
        kernels.extend(['tbl', 'tbl_v2', 'tbl_v3', 'tbl_v4', 'tbl_v5', 'tbl_v6', 'tbl_v7', 'tbl_v8', 'tbl_v9', 'tbl_v10', 'tbl_v11', 'tbl_v12', 'tbl_v13', 'tbl_v14'])
        kernel_names.extend(['TBL (2-bit)', 'TBL v2 (N-block)', 'TBL v3 (K-unroll)', 'TBL v4 (Tiled)', 'TBL v5 (Vec LUT)', 'TBL v6 (M-batch)', 'TBL v7 (Fused)', 'TBL v8 (Pipeline)', 'TBL v9 (N-first)', 'TBL v10 (K2+M8)', 'TBL v11 (M16+K4)', 'TBL v12 (vqtbl2q)', 'TBL v13 (g=2)', 'TBL v14 (g=4)'])

    if 'hybrid' in first_config:
        kernels.extend(['hybrid', 'hybrid_v2', 'hybrid_v3', 'hybrid_v4'])
        kernel_names.extend(['Hybrid (2-bit+SDOT)', 'Hybrid v2', 'Hybrid v3 (LUT)', 'Hybrid v4 (N-block)'])

    if 'sdot_direct' in first_config:
        kernels.extend(['sdot_direct', 'sdot_direct_v2', 'sdot_direct_v3', 'sdot_direct_v4'])
        kernel_names.extend(['SDOT Direct (int8)', 'SDOT Direct v2', 'SDOT Direct v3 (8-way)', 'SDOT Direct v4 (Tiled)'])

    if 'accelerate' in first_config:
        kernels.extend(['accelerate', 'accelerate_f32'])
        kernel_names.extend(['Accelerate (conv)', 'Accelerate (f32)'])

    # Header
    print(f"{'Kernel':<20}", end="")
    for config_name, _, _, _ in TEST_CONFIGS:
        print(f"{config_name:>12}", end="")
    print()
    print("-" * 80)

    # Data rows
    for kernel, name in zip(kernels, kernel_names):
        print(f"{name:<20}", end="")
        for config_name, _, _, _ in TEST_CONFIGS:
            if kernel in all_results[config_name]:
                gflops = all_results[config_name][kernel]['gflops']
                print(f"{gflops:>12.1f}", end="")
            else:
                print(f"{'N/A':>12}", end="")
        print()

    # Speedup table
    print("\n")
    print("=" * 80)
    print("SUMMARY: Speedup vs PyTorch")
    print("=" * 80)

    print(f"{'Kernel':<20}", end="")
    for config_name, _, _, _ in TEST_CONFIGS:
        print(f"{config_name:>12}", end="")
    print()
    print("-" * 80)

    for kernel, name in zip(kernels[1:], kernel_names[1:]):
        print(f"{name:<20}", end="")
        for config_name, _, _, _ in TEST_CONFIGS:
            if kernel in all_results[config_name]:
                pt_time = all_results[config_name]['pytorch']['time']
                kernel_time = all_results[config_name][kernel]['time']
                speedup = pt_time / kernel_time
                print(f"{speedup:>11.2f}x", end="")
            else:
                print(f"{'N/A':>12}", end="")
        print()

    # Memory comparison table (for TBL kernels)
    if 'tbl' in first_config:
        print("\n")
        print("=" * 80)
        print("MEMORY COMPARISON: Weight Storage per Matrix")
        print("=" * 80)
        print(f"{'Format':<20}{'Bytes/Weight':<15}", end="")
        for config_name, _, N, K in TEST_CONFIGS:
            print(f"{config_name:>12}", end="")
        print()
        print("-" * 80)

        formats = [
            ('Float32', 4.0),
            ('Int8', 1.0),
            ('2-bit (TBL)', 0.25)
        ]
        for fmt_name, bytes_per in formats:
            print(f"{fmt_name:<20}{bytes_per:<15.2f}", end="")
            for _, _, N, K in TEST_CONFIGS:
                mem_kb = N * K * bytes_per / 1024
                print(f"{mem_kb:>10.1f}KB", end="")
            print()

    # Best kernel per config
    print("\n")
    print("=" * 80)
    print("BEST KERNEL per Configuration")
    print("=" * 80)

    for config_name, _, _, _ in TEST_CONFIGS:
        best_kernel = None
        best_gflops = 0
        for kernel in kernels[1:]:  # Exclude PyTorch
            if kernel in all_results[config_name]:
                gflops = all_results[config_name][kernel]['gflops']
                if gflops > best_gflops:
                    best_gflops = gflops
                    best_kernel = kernel

        pt_gflops = all_results[config_name]['pytorch']['gflops']
        speedup = best_gflops / pt_gflops
        print(f"{config_name:<10}: {best_kernel:<20} ({best_gflops:.1f} GFLOPS, {speedup:.2f}x vs PyTorch)")

    # True MatMul-free summary
    if 'tbl' in first_config:
        print("\n")
        print("=" * 80)
        print("TRUE MATMUL-FREE KERNELS (TBL) - Memory vs Speed Trade-off")
        print("=" * 80)
        print(f"{'Config':<12}{'TBL Best':>14}{'SDOT Direct':>14}{'Speed Ratio':>14}{'Memory Saved':>14}")
        print("-" * 80)
        for config_name, _, N, K in TEST_CONFIGS:
            # Get best TBL kernel
            tbl_gflops = max(
                all_results[config_name].get('tbl_v10', {}).get('gflops', 0),
                all_results[config_name].get('tbl_v9', {}).get('gflops', 0),
                all_results[config_name].get('tbl_v8', {}).get('gflops', 0),
                all_results[config_name].get('tbl_v7', {}).get('gflops', 0),
                all_results[config_name].get('tbl_v6', {}).get('gflops', 0),
                all_results[config_name].get('tbl_v5', {}).get('gflops', 0),
                all_results[config_name].get('tbl_v4', {}).get('gflops', 0),
                all_results[config_name].get('tbl_v3', {}).get('gflops', 0),
                all_results[config_name].get('tbl_v2', {}).get('gflops', 0),
                all_results[config_name].get('tbl', {}).get('gflops', 0)
            )
            # Get best SDOT Direct kernel
            sdot_gflops = max(
                all_results[config_name].get('sdot_direct_v4', {}).get('gflops', 0),
                all_results[config_name].get('sdot_direct_v3', {}).get('gflops', 0),
                all_results[config_name].get('sdot_v4', {}).get('gflops', 0)
            )
            if sdot_gflops > 0:
                speed_ratio = tbl_gflops / sdot_gflops
            else:
                speed_ratio = 0
            memory_saved = "4x vs int8"
            print(f"{config_name:<12}{tbl_gflops:>14.1f}{sdot_gflops:>14.1f}{speed_ratio:>13.2f}x{memory_saved:>14}")


def generate_plots(all_results):
    """Generate benchmark plots"""
    try:
        import matplotlib.pyplot as plt
    except ImportError:
        print("\nMatplotlib not installed, skipping plots")
        return

    os.makedirs("benchmark_plots_arm64", exist_ok=True)

    configs = [c[0] for c in TEST_CONFIGS]
    first_config = list(all_results.values())[0]

    # Build list of available kernels
    kernels = ['pytorch']
    colors = ['gray']
    labels = ['PyTorch']

    if 'sdot_simple' in first_config:
        kernels.extend(['sdot_simple', 'sdot_v2', 'sdot_v3', 'sdot_v4'])
        colors.extend(['blue', 'green', 'orange', 'red'])
        labels.extend(['SDOT Simple', 'SDOT v2', 'SDOT v3', 'SDOT v4'])

    if 'tbl' in first_config:
        kernels.extend(['tbl', 'tbl_v2'])
        colors.extend(['purple', 'magenta'])
        labels.extend(['TBL (2-bit)', 'TBL v2'])

    if 'hybrid' in first_config:
        kernels.extend(['hybrid', 'hybrid_v2'])
        colors.extend(['cyan', 'teal'])
        labels.extend(['Hybrid (2-bit+SDOT)', 'Hybrid v2'])

    # GFLOPS comparison
    fig, ax = plt.subplots(figsize=(14, 7))
    x = np.arange(len(configs))
    width = 0.8 / len(kernels)

    for i, (kernel, color, label) in enumerate(zip(kernels, colors, labels)):
        gflops = []
        for c in configs:
            if kernel in all_results[c]:
                gflops.append(all_results[c][kernel]['gflops'])
            else:
                gflops.append(0)
        ax.bar(x + i * width, gflops, width, label=label, color=color)

    ax.set_xlabel('Matrix Size')
    ax.set_ylabel('GFLOPS')
    ax.set_title('ARM64 NEON Kernel Performance (SDOT vs TBL)')
    ax.set_xticks(x + width * (len(kernels) - 1) / 2)
    ax.set_xticklabels(configs)
    ax.legend()
    ax.grid(axis='y', alpha=0.3)

    plt.tight_layout()
    plt.savefig('benchmark_plots_arm64/gflops_comparison.png', dpi=150)
    print("\nSaved: benchmark_plots_arm64/gflops_comparison.png")

    # Speedup vs PyTorch
    fig, ax = plt.subplots(figsize=(12, 6))

    for i, (kernel, color, label) in enumerate(zip(kernels[1:], colors[1:], labels[1:])):
        speedups = []
        for c in configs:
            if kernel in all_results[c]:
                pt_time = all_results[c]['pytorch']['time']
                k_time = all_results[c][kernel]['time']
                speedups.append(pt_time / k_time)
            else:
                speedups.append(0)
        ax.plot(configs, speedups, marker='o', color=color, label=label, linewidth=2)

    ax.axhline(y=1.0, color='gray', linestyle='--', alpha=0.5, label='PyTorch baseline')
    ax.set_xlabel('Matrix Size')
    ax.set_ylabel('Speedup vs PyTorch')
    ax.set_title('ARM64 NEON Speedup vs PyTorch')
    ax.legend()
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig('benchmark_plots_arm64/speedup_vs_pytorch.png', dpi=150)
    print("Saved: benchmark_plots_arm64/speedup_vs_pytorch.png")

    # Memory vs Performance trade-off plot (if TBL available)
    if 'tbl' in first_config and 'sdot_v2' in first_config:
        fig, ax = plt.subplots(figsize=(10, 6))

        # Data points: (memory per weight in bytes, GFLOPS)
        for c in configs:
            # Float32 baseline
            ax.scatter(4.0, all_results[c]['pytorch']['gflops'], s=100, c='gray', marker='s', label='Float32' if c == configs[0] else '')
            # Int8 (SDOT)
            if 'sdot_v2' in all_results[c]:
                ax.scatter(1.0, all_results[c]['sdot_v2']['gflops'], s=100, c='blue', marker='o', label='Int8 (SDOT)' if c == configs[0] else '')
            # 2-bit (TBL)
            if 'tbl_v2' in all_results[c]:
                ax.scatter(0.25, all_results[c]['tbl_v2']['gflops'], s=100, c='purple', marker='^', label='2-bit (TBL)' if c == configs[0] else '')

        ax.set_xlabel('Bytes per Weight')
        ax.set_ylabel('GFLOPS')
        ax.set_title('Memory vs Performance Trade-off')
        ax.set_xscale('log')
        ax.set_xticks([0.25, 1.0, 4.0])
        ax.set_xticklabels(['0.25 (2-bit)', '1.0 (int8)', '4.0 (float32)'])
        ax.legend()
        ax.grid(True, alpha=0.3)

        plt.tight_layout()
        plt.savefig('benchmark_plots_arm64/memory_vs_performance.png', dpi=150)
        print("Saved: benchmark_plots_arm64/memory_vs_performance.png")

    plt.close('all')


if __name__ == "__main__":
    all_results = run_benchmarks()
    print_summary(all_results)
    generate_plots(all_results)

    print("\n" + "=" * 70)
    print("Benchmark complete!")
    print("=" * 70)
