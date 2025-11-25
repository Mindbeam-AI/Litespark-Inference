"""
Comprehensive Memory-Optimized Benchmark

Compares:
1. Unpacked (float32) - Best performance, high memory
2. Packed 2-bit - Lower performance, minimal memory

Focus: Memory savings for deployment
"""
import torch
import time
import numpy as np
from pathlib import Path
from torch.utils.cpp_extension import load
import sys

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

    # Create data
    x = torch.randn(M, K, dtype=torch.float32)
    w_float = torch.randn(N, K, dtype=torch.float32)
    w_ternary = torch.zeros_like(w_float)
    w_ternary[w_float > 0.5] = 1.0
    w_ternary[w_float < -0.5] = -1.0

    # Memory calculations
    unpacked_mem = N * K * 4  # float32
    packed_mem = packed.get_packed_memory_bytes(N, K)
    mem_reduction = unpacked_mem / packed_mem

    print(f"Memory:")
    print(f"  Unpacked (float32): {unpacked_mem / 1024 / 1024:.2f} MB")
    print(f"  Packed (2-bit):     {packed_mem / 1024 / 1024:.2f} MB")
    print(f"  Reduction:          {mem_reduction:.2f}x")

    # Benchmark unpacked
    y_unpacked = torch.zeros(M, N, dtype=torch.float32)
    bias = torch.zeros(N, dtype=torch.float32)

    for _ in range(10):  # Warmup
        unpacked.matmul_free_neon(x, w_ternary, y_unpacked, bias, M, N, K, num_threads)

    times = []
    for _ in range(50):
        start = time.perf_counter()
        unpacked.matmul_free_neon(x, w_ternary, y_unpacked, bias, M, N, K, num_threads)
        times.append(time.perf_counter() - start)

    unpacked_time = np.mean(times) * 1000
    unpacked_std = np.std(times) * 1000
    unpacked_gflops = (2.0 * M * N * K / 1e9) / (unpacked_time / 1000)

    # Benchmark packed
    K_packed = (K + 15) // 16
    w_packed = torch.zeros(N, K_packed, dtype=torch.uint32)
    packed.pack_weights_2bit(w_ternary, w_packed, N, K)

    y_packed = torch.zeros(M, N, dtype=torch.float32)

    for _ in range(10):  # Warmup
        packed.matmul_free_neon_packed2bit(x, w_packed, y_packed, bias, M, N, K, num_threads)

    times = []
    for _ in range(50):
        start = time.perf_counter()
        packed.matmul_free_neon_packed2bit(x, w_packed, y_packed, bias, M, N, K, num_threads)
        times.append(time.perf_counter() - start)

    packed_time = np.mean(times) * 1000
    packed_std = np.std(times) * 1000
    packed_gflops = (2.0 * M * N * K / 1e9) / (packed_time / 1000)

    # Verify correctness
    max_diff = torch.max(torch.abs(y_unpacked - y_packed)).item()

    print(f"\nPerformance:")
    print(f"  Unpacked: {unpacked_time:.2f} ± {unpacked_std:.2f} ms ({unpacked_gflops:.1f} GFLOPS)")
    print(f"  Packed:   {packed_time:.2f} ± {packed_std:.2f} ms ({packed_gflops:.1f} GFLOPS)")
    print(f"  Slowdown: {packed_time/unpacked_time:.2f}x")
    print(f"\nCorrectness: max diff = {max_diff:.6f} {'✓' if max_diff < 1e-3 else '✗'}")

    results.append({
        'name': name,
        'M': M, 'N': N, 'K': K,
        'unpacked_mem_mb': unpacked_mem / 1024 / 1024,
        'packed_mem_mb': packed_mem / 1024 / 1024,
        'mem_reduction': mem_reduction,
        'unpacked_gflops': unpacked_gflops,
        'packed_gflops': packed_gflops,
        'slowdown': packed_time / unpacked_time,
        'max_diff': max_diff
    })

# Summary
print("\n" + "="*70)
print("SUMMARY")
print("="*70)
print(f"{'Config':<20} {'Memory Reduction':<18} {'Performance':<25}")
print("-"*70)

for r in results:
    mem_str = f"{r['mem_reduction']:.1f}x ({r['unpacked_mem_mb']:.1f}→{r['packed_mem_mb']:.1f} MB)"
    perf_str = f"{r['unpacked_gflops']:.0f}→{r['packed_gflops']:.0f} GFLOPS ({r['slowdown']:.2f}x slower)"
    print(f"{r['name']:<20} {mem_str:<18} {perf_str}")

print("="*70)
print("\nKey Findings:")
avg_mem_reduction = np.mean([r['mem_reduction'] for r in results])
avg_slowdown = np.mean([r['slowdown'] for r in results])
print(f"  • Average memory reduction: {avg_mem_reduction:.1f}x")
print(f"  • Average performance cost:  {avg_slowdown:.2f}x slower")
print(f"  • Trade-off: {avg_mem_reduction/avg_slowdown:.1f}x memory saved per 1x slowdown")

print("\nRecommendation:")
if avg_mem_reduction > 10 and avg_slowdown < 3:
    print("  ✓ 2-bit packing is EXCELLENT for memory-constrained deployments!")
else:
    print("  ⚠ Use 2-bit packing only when memory is critical bottleneck")

print("="*70)
