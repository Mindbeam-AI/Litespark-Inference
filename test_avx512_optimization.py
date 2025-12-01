"""
Test AVX-512 Optimized MatMul-Free Kernel

This script benchmarks AVX-512 kernels on x86_64 systems with AVX-512 support.
Compares:
1. AVX2 with float32 weights (current baseline)
2. AVX-512 with float32 weights
3. AVX-512 with 2-bit packed weights

Target: 150-250 GFLOPS (2-3x improvement over AVX2)
"""
import torch
import time
import numpy as np
import platform
import psutil
import os
from pathlib import Path
from torch.utils.cpp_extension import load


def get_process_memory_mb():
    """Get current process memory usage in MB."""
    process = psutil.Process(os.getpid())
    return process.memory_info().rss / (1024 * 1024)


# Check architecture
arch = platform.machine().lower()
if arch not in ['x86_64', 'amd64']:
    print(f"Warning: This script is for x86_64 systems. Current: {arch}")
    exit(1)

KERNELS_DIR = Path(__file__).parent / "src" / "cpu_ops" / "kernels"

print("="*70)
print("AVX-512 MatMul-Free Optimization Test")
print("="*70)
print(f"Architecture: {arch}")
print(f"System: {platform.system()}")

# Compile kernels
print("\nCompiling kernels...")

# AVX2 baseline
print("  [1/2] AVX2 kernel...", end=' ', flush=True)
try:
    avx2_kernel = load(
        name="avx2_matmul_free_test",
        sources=[
            str(KERNELS_DIR / "x86_64" / "matmul_free_avx2.cpp"),
            str(KERNELS_DIR / "generic" / "matmul_free_generic.cpp")
        ],
        extra_cflags=['-O3', '-mavx2', '-mfma', '-DHAS_AVX2_SUPPORT',
                      '-fopenmp', '-std=c++17', '-march=native'],
        extra_ldflags=['-fopenmp', '-lgomp'],
        verbose=False
    )
    print("OK")
except Exception as e:
    print(f"FAILED: {e}")
    avx2_kernel = None

# AVX-512 kernel
print("  [2/2] AVX-512 kernel...", end=' ', flush=True)
try:
    avx512_kernel = load(
        name="avx512_matmul_free_test4",
        sources=[str(KERNELS_DIR / "x86_64" / "matmul_free_avx512.cpp")],
        extra_cflags=['-O3', '-mavx512f', '-mavx512bw', '-mavx512dq', '-mbmi2',
                      '-fopenmp', '-std=c++17', '-march=native'],
        extra_ldflags=['-fopenmp', '-lgomp'],
        verbose=False
    )
    print("OK")

    # Check AVX-512 support
    has_avx512 = avx512_kernel.has_avx512_support()
    print(f"\nAVX-512 support detected: {has_avx512}")
    if not has_avx512:
        print("WARNING: AVX-512 not supported! Will skip AVX-512 tests.")
        avx512_kernel = None
except Exception as e:
    print(f"FAILED: {e}")
    avx512_kernel = None

if avx2_kernel is None and avx512_kernel is None:
    print("\nNo kernels compiled successfully!")
    exit(1)

# Test configurations
configs = [
    ("Small", 128, 1024, 1024),
    ("Medium", 256, 1024, 1024),
    ("Large", 512, 1024, 1024),
    ("XLarge", 1024, 2048, 2048),
]

num_threads = 8  # Adjust based on your CPU

print(f"\nTesting with {num_threads} threads")
print("="*70)

import gc

for name, M, N, K in configs:
    print(f"\n{name}: {M}x{K} @ {K}x{N}")
    print("-"*70)

    gc.collect()
    mem_baseline = get_process_memory_mb()

    # Create test data
    x = torch.randn(M, K, dtype=torch.float32)
    w_float = torch.randn(N, K, dtype=torch.float32)

    # Quantize to ternary
    w_ternary = torch.zeros_like(w_float)
    w_ternary[w_float > 0.5] = 1.0
    w_ternary[w_float < -0.5] = -1.0

    y = torch.zeros(M, N, dtype=torch.float32)
    bias = torch.zeros(N, dtype=torch.float32)

    mem_after_tensors = get_process_memory_mb()
    print(f"Memory: baseline={mem_baseline:.1f}MB, after tensors={mem_after_tensors:.1f}MB")

    results = {}

    # Benchmark AVX2 (baseline)
    if avx2_kernel is not None:
        print("\n  AVX2 (float32 weights):")
        for _ in range(10):  # Warmup
            avx2_kernel.matmul_free_avx2(x, w_ternary, y, bias, M, N, K, num_threads)

        gc.collect()
        mem_before = get_process_memory_mb()
        times = []
        for _ in range(50):
            start = time.perf_counter()
            avx2_kernel.matmul_free_avx2(x, w_ternary, y, bias, M, N, K, num_threads)
            times.append(time.perf_counter() - start)
        mem_after = get_process_memory_mb()

        y_avx2 = y.clone()
        avg_time = np.mean(times) * 1000
        std_time = np.std(times) * 1000
        gflops = (2.0 * M * N * K / 1e9) / (avg_time / 1000)

        print(f"    Time: {avg_time:.2f} +/- {std_time:.2f} ms")
        print(f"    GFLOPS: {gflops:.1f}")
        print(f"    Memory: {mem_before:.1f}MB -> {mem_after:.1f}MB")
        results['avx2'] = {'time': avg_time, 'gflops': gflops, 'y': y_avx2}

    # Benchmark AVX-512 with float32 weights
    if avx512_kernel is not None:
        print("\n  AVX-512 (float32 weights):")
        y_512 = torch.zeros(M, N, dtype=torch.float32)

        for _ in range(10):  # Warmup
            avx512_kernel.matmul_free_avx512_float(x, w_ternary, y_512, bias, M, N, K, num_threads)

        gc.collect()
        mem_before = get_process_memory_mb()
        times = []
        for _ in range(50):
            start = time.perf_counter()
            avx512_kernel.matmul_free_avx512_float(x, w_ternary, y_512, bias, M, N, K, num_threads)
            times.append(time.perf_counter() - start)
        mem_after = get_process_memory_mb()

        avg_time = np.mean(times) * 1000
        std_time = np.std(times) * 1000
        gflops = (2.0 * M * N * K / 1e9) / (avg_time / 1000)

        print(f"    Time: {avg_time:.2f} +/- {std_time:.2f} ms")
        print(f"    GFLOPS: {gflops:.1f}")
        print(f"    Memory: {mem_before:.1f}MB -> {mem_after:.1f}MB")
        results['avx512_float'] = {'time': avg_time, 'gflops': gflops, 'y': y_512.clone()}

        # Verify correctness against AVX2
        if 'avx2' in results:
            max_diff = torch.max(torch.abs(y_512 - results['avx2']['y'])).item()
            print(f"    Max diff vs AVX2: {max_diff:.6f} {'OK' if max_diff < 1e-3 else 'MISMATCH!'}")

    # Benchmark AVX-512 with 2-bit packed weights
    if avx512_kernel is not None:
        # Pack weights (shared between v1 and v2)
        gc.collect()
        K_packed = (K + 15) // 16
        w_packed = torch.zeros(N, K_packed * 4, dtype=torch.uint8)  # 4 bytes per uint32
        avx512_kernel.pack_weights_2bit_avx512(w_ternary, w_packed, N, K)

        float_mem_mb = N * K * 4 / (1024 * 1024)
        packed_mem_mb = N * K_packed * 4 / (1024 * 1024)

        # Test v1 (basic)
        print("\n  AVX-512 packed v1 (basic):")
        print(f"    Weight memory: {float_mem_mb:.2f}MB -> {packed_mem_mb:.2f}MB = {float_mem_mb/packed_mem_mb:.1f}x reduction")

        y_packed_v1 = torch.zeros(M, N, dtype=torch.float32)

        for _ in range(10):  # Warmup
            avx512_kernel.matmul_free_avx512_packed_v1(x, w_packed, y_packed_v1, bias, M, N, K, num_threads)

        gc.collect()
        mem_before = get_process_memory_mb()
        times = []
        for _ in range(50):
            start = time.perf_counter()
            avx512_kernel.matmul_free_avx512_packed_v1(x, w_packed, y_packed_v1, bias, M, N, K, num_threads)
            times.append(time.perf_counter() - start)
        mem_after = get_process_memory_mb()

        avg_time_v1 = np.mean(times) * 1000
        std_time_v1 = np.std(times) * 1000
        gflops_v1 = (2.0 * M * N * K / 1e9) / (avg_time_v1 / 1000)

        print(f"    Time: {avg_time_v1:.2f} +/- {std_time_v1:.2f} ms")
        print(f"    GFLOPS: {gflops_v1:.1f}")
        print(f"    Memory: {mem_before:.1f}MB -> {mem_after:.1f}MB")
        results['avx512_packed_v1'] = {'time': avg_time_v1, 'gflops': gflops_v1, 'y': y_packed_v1.clone()}

        if 'avx2' in results:
            max_diff = torch.max(torch.abs(y_packed_v1 - results['avx2']['y'])).item()
            print(f"    Max diff vs AVX2: {max_diff:.6f} {'OK' if max_diff < 1e-3 else 'MISMATCH!'}")

        # Test v2 (optimized with cache blocking + prefetching)
        print("\n  AVX-512 packed v2 (cache blocking + prefetch):")

        y_packed = torch.zeros(M, N, dtype=torch.float32)

        for _ in range(10):  # Warmup
            avx512_kernel.matmul_free_avx512_packed(x, w_packed, y_packed, bias, M, N, K, num_threads)

        gc.collect()
        mem_before = get_process_memory_mb()
        times = []
        for _ in range(50):
            start = time.perf_counter()
            avx512_kernel.matmul_free_avx512_packed(x, w_packed, y_packed, bias, M, N, K, num_threads)
            times.append(time.perf_counter() - start)
        mem_after = get_process_memory_mb()

        avg_time = np.mean(times) * 1000
        std_time = np.std(times) * 1000
        gflops = (2.0 * M * N * K / 1e9) / (avg_time / 1000)

        print(f"    Time: {avg_time:.2f} +/- {std_time:.2f} ms")
        print(f"    GFLOPS: {gflops:.1f}")
        print(f"    Memory: {mem_before:.1f}MB -> {mem_after:.1f}MB")
        speedup_vs_v1 = avg_time_v1 / avg_time
        print(f"    Speedup vs v1: {speedup_vs_v1:.2f}x")
        results['avx512_packed'] = {'time': avg_time, 'gflops': gflops, 'y': y_packed.clone()}

        # Verify correctness
        if 'avx2' in results:
            max_diff = torch.max(torch.abs(y_packed - results['avx2']['y'])).item()
            print(f"    Max diff vs AVX2: {max_diff:.6f} {'OK' if max_diff < 1e-3 else 'MISMATCH!'}")

    # PyTorch comparison
    print("\n  PyTorch (reference):")
    w_pt = w_ternary.clone()
    for _ in range(10):
        _ = torch.matmul(x, w_pt.t())

    gc.collect()
    mem_before = get_process_memory_mb()
    times = []
    for _ in range(50):
        start = time.perf_counter()
        y_pt = torch.matmul(x, w_pt.t())
        times.append(time.perf_counter() - start)
    mem_after = get_process_memory_mb()

    avg_time = np.mean(times) * 1000
    gflops = (2.0 * M * N * K / 1e9) / (avg_time / 1000)
    print(f"    Time: {avg_time:.2f} ms")
    print(f"    GFLOPS: {gflops:.1f}")
    print(f"    Memory: {mem_before:.1f}MB -> {mem_after:.1f}MB")
    results['pytorch'] = {'time': avg_time, 'gflops': gflops}

    # Summary
    print(f"\n  Summary for {name}:")
    if 'avx2' in results:
        print(f"    AVX2:              {results['avx2']['gflops']:.1f} GFLOPS")
    if 'avx512_float' in results:
        speedup = results['avx512_float']['gflops'] / results['avx2']['gflops'] if 'avx2' in results else 1.0
        print(f"    AVX-512 float:     {results['avx512_float']['gflops']:.1f} GFLOPS ({speedup:.2f}x vs AVX2)")
    if 'avx512_packed_v1' in results:
        speedup = results['avx512_packed_v1']['gflops'] / results['avx2']['gflops'] if 'avx2' in results else 1.0
        print(f"    AVX-512 packed v1: {results['avx512_packed_v1']['gflops']:.1f} GFLOPS ({speedup:.2f}x vs AVX2)")
    if 'avx512_packed' in results:
        speedup = results['avx512_packed']['gflops'] / results['avx2']['gflops'] if 'avx2' in results else 1.0
        v1_speedup = results['avx512_packed']['gflops'] / results['avx512_packed_v1']['gflops'] if 'avx512_packed_v1' in results else 1.0
        print(f"    AVX-512 packed v2: {results['avx512_packed']['gflops']:.1f} GFLOPS ({speedup:.2f}x vs AVX2, {v1_speedup:.2f}x vs v1)")
    print(f"    PyTorch:           {results['pytorch']['gflops']:.1f} GFLOPS")

print("\n" + "="*70)
print("Optimization Test Complete!")
print("="*70)
print("\nExpected improvements:")
print("  - AVX-512 float: ~1.5-2x vs AVX2 (wider SIMD)")
print("  - AVX-512 packed: ~2-4x vs AVX2 (reduced memory bandwidth)")
print("="*70)
