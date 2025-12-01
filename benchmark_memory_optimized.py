"""
Comprehensive Memory-Optimized Benchmark

Compares:
1. Unpacked (float32) - Best performance, high memory
2. Packed 2-bit - Lower performance, minimal memory

Focus: Memory savings for deployment

Uses actual runtime memory measurement via psutil and tracemalloc.
"""
import torch
import time
import numpy as np
import psutil
import tracemalloc
import gc
import os
from pathlib import Path
from torch.utils.cpp_extension import load
import sys


def get_process_memory_mb():
    """Get current process memory usage in MB (RSS)."""
    process = psutil.Process(os.getpid())
    return process.memory_info().rss / (1024 * 1024)


def measure_actual_memory(create_tensors_func):
    """
    Measure actual memory used by tensor creation.
    Returns memory in MB.
    """
    gc.collect()
    mem_before = get_process_memory_mb()

    tensors = create_tensors_func()

    gc.collect()  # Don't collect the tensors we just created
    mem_after = get_process_memory_mb()

    return {
        'tensors': tensors,
        'memory_mb': mem_after - mem_before,
        'baseline_mb': mem_before,
        'after_mb': mem_after,
    }

KERNELS_DIR = Path(__file__).parent / "src" / "cpu_ops" / "kernels"

print("="*70)
print("MatMul-Free Memory-Optimized Benchmark")
print("="*70)

# Compile kernels
print("\nCompiling kernels...")
print("  [1/2] Unpacked kernel...", end=' ', flush=True)
unpacked = load(
    name="unpacked_kernel",
    sources=[
        str(KERNELS_DIR / "arm64" / "matmul_free_neon.cpp"),
        str(KERNELS_DIR / "generic" / "matmul_free_generic.cpp")
    ],
    extra_cflags=['-O3', '-DHAS_NEON_SUPPORT', '-Xpreprocessor', '-fopenmp',
                  '-std=c++17', '-mcpu=apple-m1', '-mtune=native'],
    extra_ldflags=['-L/opt/homebrew/lib', '-lomp'],
    verbose=False
)
print("✓")

print("  [2/2] Packed 2-bit kernel...", end=' ', flush=True)
packed = load(
    name="packed_2bit_kernel",
    sources=[str(KERNELS_DIR / "arm64" / "matmul_free_neon_packed2bit.cpp")],
    extra_cflags=['-O3', '-DHAS_NEON_SUPPORT', '-Xpreprocessor', '-fopenmp',
                  '-std=c++17', '-mcpu=apple-m1', '-mtune=native'],
    extra_ldflags=['-L/opt/homebrew/lib', '-lomp'],
    verbose=False
)
print("✓")

# Test configurations representing real model layers
configs = [
    ("Small FFN", 128, 1024, 1024),
    ("Medium FFN", 256, 2048, 2048),
    ("Large FFN", 512, 4096, 4096),
    ("LLM Layer (7B)", 1, 4096, 11008),  # Single token inference
    ("LLM Batch (7B)", 32, 4096, 11008),  # Small batch
]

num_threads = 10

print(f"\nTesting with {num_threads} threads")
print("="*70)

results = []

for name, M, N, K in configs:
    print(f"\n{name}: {M}x{K} @ {K}x{N}")
    print("-"*70)

    # Measure actual memory for unpacked weights
    gc.collect()
    mem_before_unpacked = get_process_memory_mb()

    x = torch.randn(M, K, dtype=torch.float32)
    w_float = torch.randn(N, K, dtype=torch.float32)
    w_ternary = torch.zeros_like(w_float)
    w_ternary[w_float > 0.5] = 1.0
    w_ternary[w_float < -0.5] = -1.0

    mem_after_unpacked = get_process_memory_mb()
    actual_unpacked_mem_mb = mem_after_unpacked - mem_before_unpacked

    print(f"Memory (Unpacked Weights):")
    print(f"  Actual measured:    {actual_unpacked_mem_mb:.2f} MB")

    # Benchmark unpacked
    y_unpacked = torch.zeros(M, N, dtype=torch.float32)
    bias = torch.zeros(N, dtype=torch.float32)

    for _ in range(10):  # Warmup
        unpacked.matmul_free_neon(x, w_ternary, y_unpacked, bias, M, N, K, num_threads)

    # Measure memory during unpacked kernel execution
    gc.collect()
    mem_before_unpacked_kernel = get_process_memory_mb()
    tracemalloc.start()

    times = []
    for _ in range(50):
        start = time.perf_counter()
        unpacked.matmul_free_neon(x, w_ternary, y_unpacked, bias, M, N, K, num_threads)
        times.append(time.perf_counter() - start)

    unpacked_current, unpacked_peak = tracemalloc.get_traced_memory()
    tracemalloc.stop()
    mem_after_unpacked_kernel = get_process_memory_mb()

    unpacked_time = np.mean(times) * 1000
    unpacked_std = np.std(times) * 1000
    unpacked_gflops = (2.0 * M * N * K / 1e9) / (unpacked_time / 1000)
    unpacked_kernel_mem_mb = mem_after_unpacked_kernel - mem_before_unpacked_kernel
    unpacked_peak_mb = unpacked_peak / (1024 * 1024)

    # Measure actual memory for packed weights
    gc.collect()
    mem_before_packed = get_process_memory_mb()

    K_packed = (K + 15) // 16
    w_packed = torch.zeros(N, K_packed, dtype=torch.uint32)
    packed.pack_weights_2bit(w_ternary, w_packed, N, K)

    mem_after_packed = get_process_memory_mb()
    actual_packed_mem_mb = mem_after_packed - mem_before_packed

    print(f"\nMemory (Packed Weights):")
    print(f"  Actual measured:    {actual_packed_mem_mb:.2f} MB")

    # Calculate actual memory reduction
    if actual_packed_mem_mb > 0 and actual_unpacked_mem_mb > 0:
        actual_mem_reduction = actual_unpacked_mem_mb / actual_packed_mem_mb
    else:
        # If memory delta is 0 or negative (unlikely), report 1.0
        actual_mem_reduction = 1.0
        print(f"  Warning: Could not measure memory reduction accurately")

    print(f"  Actual reduction:   {actual_mem_reduction:.2f}x")

    y_packed = torch.zeros(M, N, dtype=torch.float32)

    for _ in range(10):  # Warmup
        packed.matmul_free_neon_packed2bit(x, w_packed, y_packed, bias, M, N, K, num_threads)

    # Measure memory during packed kernel execution
    gc.collect()
    mem_before_packed_kernel = get_process_memory_mb()
    tracemalloc.start()

    times = []
    for _ in range(50):
        start = time.perf_counter()
        packed.matmul_free_neon_packed2bit(x, w_packed, y_packed, bias, M, N, K, num_threads)
        times.append(time.perf_counter() - start)

    packed_current, packed_peak = tracemalloc.get_traced_memory()
    tracemalloc.stop()
    mem_after_packed_kernel = get_process_memory_mb()

    packed_time = np.mean(times) * 1000
    packed_std = np.std(times) * 1000
    packed_gflops = (2.0 * M * N * K / 1e9) / (packed_time / 1000)
    packed_kernel_mem_mb = mem_after_packed_kernel - mem_before_packed_kernel
    packed_peak_mb = packed_peak / (1024 * 1024)

    # Verify correctness
    max_diff = torch.max(torch.abs(y_unpacked - y_packed)).item()

    print(f"\nPerformance:")
    print(f"  Unpacked: {unpacked_time:.2f} ± {unpacked_std:.2f} ms ({unpacked_gflops:.1f} GFLOPS)")
    print(f"  Packed:   {packed_time:.2f} ± {packed_std:.2f} ms ({packed_gflops:.1f} GFLOPS)")
    print(f"  Slowdown: {packed_time/unpacked_time:.2f}x")

    print(f"\nKernel Memory (during execution):")
    print(f"  Unpacked kernel delta: {unpacked_kernel_mem_mb:.2f} MB (peak: {unpacked_peak_mb:.2f} MB)")
    print(f"  Packed kernel delta:   {packed_kernel_mem_mb:.2f} MB (peak: {packed_peak_mb:.2f} MB)")

    print(f"\nCorrectness: max diff = {max_diff:.6f} {'✓' if max_diff < 1e-3 else '✗'}")

    results.append({
        'name': name,
        'M': M, 'N': N, 'K': K,
        'actual_unpacked_mem_mb': actual_unpacked_mem_mb,
        'actual_packed_mem_mb': actual_packed_mem_mb,
        'actual_mem_reduction': actual_mem_reduction,
        'unpacked_gflops': unpacked_gflops,
        'packed_gflops': packed_gflops,
        'slowdown': packed_time / unpacked_time,
        'unpacked_peak_mb': unpacked_peak_mb,
        'packed_peak_mb': packed_peak_mb,
        'max_diff': max_diff
    })

# Summary
print("\n" + "="*70)
print("SUMMARY (Actual Measured Memory)")
print("="*70)
print(f"{'Config':<20} {'Memory Reduction':<25} {'Performance':<25}")
print("-"*70)

for r in results:
    mem_str = f"{r['actual_mem_reduction']:.1f}x ({r['actual_unpacked_mem_mb']:.1f}→{r['actual_packed_mem_mb']:.1f} MB)"
    perf_str = f"{r['unpacked_gflops']:.0f}→{r['packed_gflops']:.0f} GFLOPS ({r['slowdown']:.2f}x slower)"
    print(f"{r['name']:<20} {mem_str:<25} {perf_str}")

print("="*70)
print("\nKey Findings (from actual measurements):")
avg_mem_reduction = np.mean([r['actual_mem_reduction'] for r in results])
avg_slowdown = np.mean([r['slowdown'] for r in results])
print(f"  • Average memory reduction: {avg_mem_reduction:.1f}x")
print(f"  • Average performance cost:  {avg_slowdown:.2f}x slower")
print(f"  • Trade-off: {avg_mem_reduction/avg_slowdown:.1f}x memory saved per 1x slowdown")

print("\nPer-config Memory Reduction:")
for r in results:
    print(f"  {r['name']}: {r['actual_mem_reduction']:.1f}x reduction")

print("\nRecommendation:")
if avg_mem_reduction > 10 and avg_slowdown < 3:
    print("  ✓ 2-bit packing is EXCELLENT for memory-constrained deployments!")
else:
    print("  ⚠ Use 2-bit packing only when memory is critical bottleneck")

print("="*70)
