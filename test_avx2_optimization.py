"""
Test AVX2 Optimized MatMul-Free Kernel

This script benchmarks the optimized AVX2 kernel on x86_64 systems.
Run this on Intel/AMD CPUs with AVX2 support.

Expected performance: 200-250 GFLOPS (2x ARM64 NEON)
"""
import torch
import time
import numpy as np
import platform
import psutil
import tracemalloc
import os
from pathlib import Path
from torch.utils.cpp_extension import load


def get_process_memory_mb():
    """Get current process memory usage in MB."""
    process = psutil.Process(os.getpid())
    return process.memory_info().rss / (1024 * 1024)


def measure_memory_usage(func, *args, **kwargs):
    """Measure peak memory usage during function execution."""
    # Force garbage collection before measurement
    import gc
    gc.collect()

    # Get baseline memory
    baseline_mb = get_process_memory_mb()

    # Start tracemalloc for Python allocations
    tracemalloc.start()

    # Run the function
    result = func(*args, **kwargs)

    # Get peak memory
    current, peak = tracemalloc.get_traced_memory()
    tracemalloc.stop()

    # Get process memory after
    after_mb = get_process_memory_mb()

    return {
        'result': result,
        'baseline_mb': baseline_mb,
        'after_mb': after_mb,
        'delta_mb': after_mb - baseline_mb,
        'python_peak_mb': peak / (1024 * 1024),
    }

# Check architecture
arch = platform.machine().lower()
if arch not in ['x86_64', 'amd64']:
    print(f"⚠️  Warning: This script is for x86_64 systems. Current: {arch}")
    print("AVX2 kernel will not be available on this architecture.")
    exit(1)

KERNELS_DIR = Path(__file__).parent / "src" / "cpu_ops" / "kernels"

print("="*70)
print("AVX2 MatMul-Free Optimization Test")
print("="*70)
print(f"Architecture: {arch}")
print(f"System: {platform.system()}")

# Compile AVX2 kernel
print("\nCompiling AVX2 kernel...", flush=True)
try:
    avx2_kernel = load(
        name="avx2_matmul_free",
        sources=[
            str(KERNELS_DIR / "x86_64" / "matmul_free_avx2.cpp"),
            str(KERNELS_DIR / "generic" / "matmul_free_generic.cpp")
        ],
        extra_cflags=['-O3', '-mavx2', '-mfma', '-DHAS_AVX2_SUPPORT',
                      '-fopenmp', '-std=c++17', '-march=native'],
        extra_ldflags=['-fopenmp', '-lgomp'],
        verbose=True
    )
    print("✓ AVX2 kernel compiled!")

    # Check AVX2 support
    if hasattr(avx2_kernel, 'has_avx2_support'):
        has_avx2 = avx2_kernel.has_avx2_support()
        print(f"AVX2 support detected: {has_avx2}")
        if not has_avx2:
            print("⚠️  Warning: AVX2 not supported on this CPU!")
            exit(1)

except Exception as e:
    print(f"❌ Failed to compile AVX2 kernel: {e}")
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

for name, M, N, K in configs:
    print(f"\n{name}: {M}x{K} @ {K}x{N}")
    print("-"*70)

    # Measure memory before tensor allocation
    import gc
    gc.collect()
    mem_baseline = get_process_memory_mb()

    # Create test data
    x = torch.randn(M, K, dtype=torch.float32)
    mem_after_x = get_process_memory_mb()

    w_float = torch.randn(N, K, dtype=torch.float32)
    mem_after_w_float = get_process_memory_mb()

    # Quantize to ternary
    w_ternary = torch.zeros_like(w_float)
    w_ternary[w_float > 0.5] = 1.0
    w_ternary[w_float < -0.5] = -1.0
    mem_after_w_ternary = get_process_memory_mb()

    y = torch.zeros(M, N, dtype=torch.float32)
    mem_after_y = get_process_memory_mb()

    bias = torch.zeros(N, dtype=torch.float32)
    mem_after_bias = get_process_memory_mb()

    # Calculate individual tensor sizes
    x_size_mb = M * K * 4 / (1024 * 1024)
    w_size_mb = N * K * 4 / (1024 * 1024)
    y_size_mb = M * N * 4 / (1024 * 1024)

    print(f"\nMemory Usage (Actual Process RSS):")
    print(f"  Baseline (before tensors):  {mem_baseline:.2f} MB")
    print(f"  After x [{M}x{K}]:          {mem_after_x:.2f} MB (+{mem_after_x - mem_baseline:.2f} MB, tensor: {x_size_mb:.2f} MB)")
    print(f"  After w_float [{N}x{K}]:    {mem_after_w_float:.2f} MB (+{mem_after_w_float - mem_after_x:.2f} MB, tensor: {w_size_mb:.2f} MB)")
    print(f"  After w_ternary [{N}x{K}]:  {mem_after_w_ternary:.2f} MB (+{mem_after_w_ternary - mem_after_w_float:.2f} MB)")
    print(f"  After y [{M}x{N}]:          {mem_after_y:.2f} MB (+{mem_after_y - mem_after_w_ternary:.2f} MB, tensor: {y_size_mb:.2f} MB)")
    print(f"  After bias [{N}]:           {mem_after_bias:.2f} MB")
    print(f"  Total tensors allocated:    {mem_after_bias - mem_baseline:.2f} MB")

    # Warmup
    for _ in range(10):
        avx2_kernel.matmul_free_avx2(x, w_ternary, y, bias, M, N, K, num_threads)

    # Measure memory during kernel execution
    gc.collect()
    mem_before_kernel = get_process_memory_mb()

    # Benchmark
    times = []
    peak_during_kernel = mem_before_kernel
    for _ in range(50):
        start = time.perf_counter()
        avx2_kernel.matmul_free_avx2(x, w_ternary, y, bias, M, N, K, num_threads)
        times.append(time.perf_counter() - start)
        # Sample memory during execution
        current = get_process_memory_mb()
        if current > peak_during_kernel:
            peak_during_kernel = current

    mem_after_kernel = get_process_memory_mb()

    avg_time = np.mean(times) * 1000
    std_time = np.std(times) * 1000
    gflops = (2.0 * M * N * K / 1e9) / (avg_time / 1000)

    print(f"\nAVX2 Kernel Performance:")
    print(f"  Time:   {avg_time:.2f} ± {std_time:.2f} ms")
    print(f"  GFLOPS: {gflops:.1f}")
    print(f"  Memory before kernel: {mem_before_kernel:.2f} MB")
    print(f"  Memory peak during:   {peak_during_kernel:.2f} MB")
    print(f"  Memory after kernel:  {mem_after_kernel:.2f} MB")

    # Compare with PyTorch
    print("\nPyTorch Comparison:")
    w_pytorch = w_ternary.clone()
    mem_after_clone = get_process_memory_mb()
    print(f"  Memory after w clone: {mem_after_clone:.2f} MB")

    for _ in range(10):
        _ = torch.matmul(x, w_pytorch.t())

    gc.collect()
    mem_before_pt = get_process_memory_mb()

    times_pt = []
    peak_during_pt = mem_before_pt
    for _ in range(50):
        start = time.perf_counter()
        y_pytorch = torch.matmul(x, w_pytorch.t())
        times_pt.append(time.perf_counter() - start)
        current = get_process_memory_mb()
        if current > peak_during_pt:
            peak_during_pt = current

    mem_after_pt = get_process_memory_mb()

    avg_time_pt = np.mean(times_pt) * 1000
    gflops_pt = (2.0 * M * N * K / 1e9) / (avg_time_pt / 1000)

    print(f"  Time:   {avg_time_pt:.2f} ms ({gflops_pt:.1f} GFLOPS)")
    print(f"  Memory before PyTorch: {mem_before_pt:.2f} MB")
    print(f"  Memory peak during:    {peak_during_pt:.2f} MB")
    print(f"  Memory after PyTorch:  {mem_after_pt:.2f} MB")

    print(f"\nComparison:")
    print(f"  Speedup: {avg_time_pt/avg_time:.3f}x vs PyTorch ({gflops/gflops_pt:.2%})")

    # Verify correctness
    y_verify = torch.zeros(M, N, dtype=torch.float32)
    avx2_kernel.matmul_free_avx2(x, w_ternary, y_verify, bias, M, N, K, num_threads)
    max_diff = torch.max(torch.abs(y_verify - y_pytorch)).item()
    print(f"  Max diff: {max_diff:.6f} {'✓' if max_diff < 1e-3 else '✗'}")

    # Final memory state
    gc.collect()
    mem_final = get_process_memory_mb()
    print(f"\nFinal process memory: {mem_final:.2f} MB")

print("\n" + "="*70)
print("AVX2 Optimization Complete!")
print("="*70)
print("\nExpected Performance Targets:")
print("  • AVX2 (8-wide):  200-250 GFLOPS")
print("  • vs PyTorch:     0.15-0.20x (MKL/OneDNN advantage)")
print("  • vs ARM64 NEON:  ~2x faster (wider SIMD)")
print("="*70)
