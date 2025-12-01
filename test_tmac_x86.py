"""
Test T-MAC Kernel for x86_64

This script benchmarks the T-MAC (Table-based Matrix-vector product Acceleration)
approach on x86_64 systems.

T-MAC key idea:
- Group 4 weights together
- Precompute all 16 possible partial sums for each group
- Use weight bit patterns as indices into lookup tables
- Reduces per-weight operations from compare+branch to simple lookup
"""
import torch
import time
import numpy as np
import platform
import psutil
import os
import gc
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
print("T-MAC MatMul-Free Kernel Test (x86_64)")
print("="*70)
print(f"Architecture: {arch}")
print(f"System: {platform.system()}")

# Compile T-MAC kernel
print("\nCompiling T-MAC kernel...", end=' ', flush=True)
try:
    tmac_kernel = load(
        name="tmac_x86_kernel",
        sources=[str(KERNELS_DIR / "x86_64" / "matmul_free_tmac_avx2.cpp")],
        extra_cflags=['-O3', '-mavx2', '-mfma', '-fopenmp', '-std=c++17', '-march=native'],
        extra_ldflags=['-fopenmp', '-lgomp'],
        verbose=False
    )
    print("OK")
except Exception as e:
    print(f"FAILED: {e}")
    exit(1)

# Also compile AVX-512 packed kernel for comparison
print("Compiling AVX-512 packed kernel...", end=' ', flush=True)
try:
    avx512_kernel = load(
        name="avx512_packed_comparison",
        sources=[str(KERNELS_DIR / "x86_64" / "matmul_free_avx512.cpp")],
        extra_cflags=['-O3', '-mavx512f', '-mavx512bw', '-mavx512dq', '-mbmi2',
                      '-fopenmp', '-std=c++17', '-march=native'],
        extra_ldflags=['-fopenmp', '-lgomp'],
        verbose=False
    )
    print("OK")
except Exception as e:
    print(f"FAILED: {e}")
    avx512_kernel = None

# Test configurations
configs = [
    ("Small", 128, 1024, 1024),
    ("Medium", 256, 1024, 1024),
    ("Large", 512, 1024, 1024),
    ("XLarge", 1024, 2048, 2048),
]

num_threads = 8

print(f"\nTesting with {num_threads} threads")
print("="*70)

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

    mem_after_tensors = get_process_memory_mb()
    print(f"Memory: baseline={mem_baseline:.1f}MB, after tensors={mem_after_tensors:.1f}MB")

    # Pack weights for T-MAC
    K_bytes = (K + 7) // 8
    w_packed_tmac = torch.zeros(N, 2 * K_bytes, dtype=torch.uint8)
    tmac_kernel.pack_ternary_tmac_x86(w_ternary, w_packed_tmac, N, K)

    # Pack weights for AVX-512
    if avx512_kernel is not None:
        K_packed = (K + 15) // 16
        w_packed_avx512 = torch.zeros(N, K_packed * 4, dtype=torch.uint8)
        avx512_kernel.pack_weights_2bit_avx512(w_ternary, w_packed_avx512, N, K)

    bias = torch.zeros(N, dtype=torch.float32)
    results = {}

    # Test T-MAC scalar version
    print("\n  T-MAC (scalar LUT):")
    y_tmac = torch.zeros(M, N, dtype=torch.float32)

    for _ in range(10):  # Warmup
        tmac_kernel.matmul_free_tmac_x86(x, w_packed_tmac, y_tmac, bias, M, N, K, num_threads)

    gc.collect()
    mem_before = get_process_memory_mb()
    times = []
    for _ in range(50):
        start = time.perf_counter()
        tmac_kernel.matmul_free_tmac_x86(x, w_packed_tmac, y_tmac, bias, M, N, K, num_threads)
        times.append(time.perf_counter() - start)
    mem_after = get_process_memory_mb()

    avg_time = np.mean(times) * 1000
    std_time = np.std(times) * 1000
    gflops = (2.0 * M * N * K / 1e9) / (avg_time / 1000)

    print(f"    Time: {avg_time:.2f} +/- {std_time:.2f} ms")
    print(f"    GFLOPS: {gflops:.1f}")
    print(f"    Memory: {mem_before:.1f}MB -> {mem_after:.1f}MB")
    results['tmac_scalar'] = {'time': avg_time, 'gflops': gflops, 'y': y_tmac.clone()}

    # Test T-MAC SIMD version
    print("\n  T-MAC (SIMD optimized):")
    y_tmac_simd = torch.zeros(M, N, dtype=torch.float32)

    for _ in range(10):  # Warmup
        tmac_kernel.matmul_free_tmac_simd_x86(x, w_packed_tmac, y_tmac_simd, bias, M, N, K, num_threads)

    gc.collect()
    mem_before = get_process_memory_mb()
    times = []
    for _ in range(50):
        start = time.perf_counter()
        tmac_kernel.matmul_free_tmac_simd_x86(x, w_packed_tmac, y_tmac_simd, bias, M, N, K, num_threads)
        times.append(time.perf_counter() - start)
    mem_after = get_process_memory_mb()

    avg_time = np.mean(times) * 1000
    std_time = np.std(times) * 1000
    gflops = (2.0 * M * N * K / 1e9) / (avg_time / 1000)

    print(f"    Time: {avg_time:.2f} +/- {std_time:.2f} ms")
    print(f"    GFLOPS: {gflops:.1f}")
    print(f"    Memory: {mem_before:.1f}MB -> {mem_after:.1f}MB")
    results['tmac_simd'] = {'time': avg_time, 'gflops': gflops, 'y': y_tmac_simd.clone()}

    # Test AVX-512 packed v1 for comparison
    if avx512_kernel is not None:
        print("\n  AVX-512 packed v1 (baseline):")
        y_avx512 = torch.zeros(M, N, dtype=torch.float32)

        for _ in range(10):  # Warmup
            avx512_kernel.matmul_free_avx512_packed_v1(x, w_packed_avx512, y_avx512, bias, M, N, K, num_threads)

        gc.collect()
        mem_before = get_process_memory_mb()
        times = []
        for _ in range(50):
            start = time.perf_counter()
            avx512_kernel.matmul_free_avx512_packed_v1(x, w_packed_avx512, y_avx512, bias, M, N, K, num_threads)
            times.append(time.perf_counter() - start)
        mem_after = get_process_memory_mb()

        avg_time = np.mean(times) * 1000
        std_time = np.std(times) * 1000
        gflops = (2.0 * M * N * K / 1e9) / (avg_time / 1000)

        print(f"    Time: {avg_time:.2f} +/- {std_time:.2f} ms")
        print(f"    GFLOPS: {gflops:.1f}")
        print(f"    Memory: {mem_before:.1f}MB -> {mem_after:.1f}MB")
        results['avx512_packed'] = {'time': avg_time, 'gflops': gflops, 'y': y_avx512.clone()}

    # PyTorch reference
    print("\n  PyTorch (reference):")
    w_pt = w_ternary.clone()
    for _ in range(10):
        _ = torch.matmul(x, w_pt.t())

    gc.collect()
    times = []
    for _ in range(50):
        start = time.perf_counter()
        y_pt = torch.matmul(x, w_pt.t())
        times.append(time.perf_counter() - start)

    avg_time = np.mean(times) * 1000
    gflops = (2.0 * M * N * K / 1e9) / (avg_time / 1000)
    print(f"    Time: {avg_time:.2f} ms")
    print(f"    GFLOPS: {gflops:.1f}")
    results['pytorch'] = {'time': avg_time, 'gflops': gflops, 'y': y_pt}

    # Verify correctness
    print("\n  Correctness check:")
    max_diff_tmac = torch.max(torch.abs(results['tmac_scalar']['y'] - y_pt)).item()
    max_diff_simd = torch.max(torch.abs(results['tmac_simd']['y'] - y_pt)).item()
    print(f"    T-MAC scalar vs PyTorch: {max_diff_tmac:.6f} {'OK' if max_diff_tmac < 1e-3 else 'MISMATCH!'}")
    print(f"    T-MAC SIMD vs PyTorch:   {max_diff_simd:.6f} {'OK' if max_diff_simd < 1e-3 else 'MISMATCH!'}")

    if 'avx512_packed' in results:
        max_diff_avx512 = torch.max(torch.abs(results['avx512_packed']['y'] - y_pt)).item()
        print(f"    AVX-512 vs PyTorch:      {max_diff_avx512:.6f} {'OK' if max_diff_avx512 < 1e-3 else 'MISMATCH!'}")

    # Summary
    print(f"\n  Summary for {name}:")
    print(f"    T-MAC scalar:    {results['tmac_scalar']['gflops']:.1f} GFLOPS")
    print(f"    T-MAC SIMD:      {results['tmac_simd']['gflops']:.1f} GFLOPS")
    if 'avx512_packed' in results:
        speedup_scalar = results['tmac_scalar']['gflops'] / results['avx512_packed']['gflops']
        speedup_simd = results['tmac_simd']['gflops'] / results['avx512_packed']['gflops']
        print(f"    AVX-512 packed:  {results['avx512_packed']['gflops']:.1f} GFLOPS")
        print(f"    T-MAC scalar vs AVX-512: {speedup_scalar:.2f}x")
        print(f"    T-MAC SIMD vs AVX-512:   {speedup_simd:.2f}x")
    print(f"    PyTorch:         {results['pytorch']['gflops']:.1f} GFLOPS")

print("\n" + "="*70)
print("T-MAC Test Complete!")
print("="*70)
print("\nKey insights:")
print("  - T-MAC uses lookup tables instead of masked operations")
print("  - Groups 4 weights -> 16-entry LUT (fits in cache)")
print("  - Eliminates PEXT instruction overhead")
print("="*70)
