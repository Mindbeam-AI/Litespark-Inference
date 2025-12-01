"""
Quick test of T-MAC approach
"""
import torch
import time
import numpy as np
import psutil
import gc
from src.cpu_ops.setup_extensions import compile_cpu_kernels

# Compile with T-MAC kernel
import os
from pathlib import Path
from torch.utils.cpp_extension import load


def get_process_memory_mb():
    """Get current process memory usage in MB."""
    process = psutil.Process(os.getpid())
    return process.memory_info().rss / (1024 * 1024)

CURRENT_DIR = Path(__file__).parent
KERNELS_DIR = CURRENT_DIR / "src" / "cpu_ops" / "kernels"

print("Compiling T-MAC kernel...")
tmac_kernels = load(
    name="tmac_kernels",
    sources=[
        str(KERNELS_DIR / "arm64" / "matmul_free_neon_tmac.cpp"),
    ],
    extra_cflags=['-O3', '-DHAS_NEON_SUPPORT', '-Xpreprocessor', '-fopenmp',
                  '-std=c++17', '-mmacosx-version-min=10.9', '-mcpu=apple-m1'],
    extra_ldflags=['-L/opt/homebrew/lib', '-L/usr/local/lib', '-lomp'],
    verbose=True
)

print("✅ T-MAC kernel compiled!")

# Test configuration
M, N, K = 128, 1024, 1024
num_threads = 10

# Create test data
x = torch.randn(M, K, dtype=torch.float32)
w_float = torch.randn(N, K, dtype=torch.float32)

# Quantize to ternary
w_ternary = torch.zeros_like(w_float)
w_ternary[w_float > 0.5] = 1.0
w_ternary[w_float < -0.5] = -1.0

# Measure actual memory for unpacked weights
gc.collect()
mem_before_unpacked = get_process_memory_mb()
w_ternary_copy = w_ternary.clone()
mem_after_unpacked = get_process_memory_mb()
unpacked_mem_mb = mem_after_unpacked - mem_before_unpacked

# Pack weights T-MAC style
gc.collect()
mem_before_packed = get_process_memory_mb()
K_bytes = (K + 7) // 8
w_packed = torch.zeros(N, 2 * K_bytes, dtype=torch.uint8)
tmac_kernels.pack_ternary_tmac(w_ternary, w_packed, N, K)
mem_after_packed = get_process_memory_mb()
packed_mem_mb = mem_after_packed - mem_before_packed

print(f"Memory (actual):")
print(f"  Unpacked: {unpacked_mem_mb:.2f} MB")
print(f"  Packed:   {packed_mem_mb:.2f} MB")
if packed_mem_mb > 0:
    print(f"  Compression: {unpacked_mem_mb / packed_mem_mb:.2f}x")

# Allocate output
y = torch.zeros(M, N, dtype=torch.float32)
bias = torch.zeros(N, dtype=torch.float32)

# Warmup
for _ in range(5):
    tmac_kernels.matmul_free_neon_tmac(x, w_packed, y, bias, M, N, K, num_threads)

# Benchmark
num_runs = 100
times = []

for _ in range(num_runs):
    start = time.perf_counter()
    tmac_kernels.matmul_free_neon_tmac(x, w_packed, y, bias, M, N, K, num_threads)
    end = time.perf_counter()
    times.append(end - start)

avg_time = np.mean(times) * 1000  # ms
std_time = np.std(times) * 1000

# Calculate GFLOPS (2*M*N*K operations)
ops = 2.0 * M * N * K
gflops = (ops / 1e9) / (avg_time / 1000)

print(f"\n{'='*60}")
print(f"T-MAC Benchmark Results ({M}x{K} @ {K}x{N})")
print(f"{'='*60}")
print(f"Time: {avg_time:.2f} ± {std_time:.2f} ms")
print(f"GFLOPS: {gflops:.1f}")
print(f"Threads: {num_threads}")
print(f"{'='*60}")

# Verify correctness
y_expected = torch.matmul(x, w_ternary.T)
max_diff = torch.max(torch.abs(y - y_expected)).item()
rel_diff = max_diff / (torch.max(torch.abs(y_expected)).item() + 1e-8)
print(f"Max absolute difference: {max_diff:.6f}")
print(f"Max relative difference: {rel_diff:.6f}")
if max_diff < 1e-3:  # Relaxed tolerance for floating point
    print("✅ Results match!")
else:
    print("❌ Results don't match!")
