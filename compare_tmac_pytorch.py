"""
Compare T-MAC vs PyTorch vs Current NEON
"""
import torch
import torch.nn.functional as F
import time
import numpy as np
from pathlib import Path
from torch.utils.cpp_extension import load

CURRENT_DIR = Path(__file__).parent
KERNELS_DIR = CURRENT_DIR / "src" / "cpu_ops" / "kernels"

# Compile T-MAC kernel
print("Compiling T-MAC kernel...")
tmac_kernels = load(
    name="tmac_kernels",
    sources=[str(KERNELS_DIR / "arm64" / "matmul_free_neon_tmac.cpp")],
    extra_cflags=['-O3', '-DHAS_NEON_SUPPORT', '-Xpreprocessor', '-fopenmp',
                  '-std=c++17', '-mmacosx-version-min=10.9', '-mcpu=apple-m1'],
    extra_ldflags=['-L/opt/homebrew/lib', '-L/usr/local/lib', '-lomp'],
    verbose=False
)

# Compile current NEON kernel
print("Compiling current NEON kernel...")
neon_kernels = load(
    name="neon_kernels",
    sources=[
        str(KERNELS_DIR / "arm64" / "matmul_free_neon.cpp"),
        str(KERNELS_DIR / "generic" / "matmul_free_generic.cpp")
    ],
    extra_cflags=['-O3', '-DHAS_NEON_SUPPORT', '-Xpreprocessor', '-fopenmp',
                  '-std=c++17', '-mmacosx-version-min=10.9', '-mcpu=apple-m1'],
    extra_ldflags=['-L/opt/homebrew/lib', '-L/usr/local/lib', '-lomp'],
    verbose=False
)

# Test configuration
M, N, K = 128, 1024, 1024
num_threads = 10

print(f"\nBenchmarking {M}x{K} @ {K}x{N}...")

# Create test data
x = torch.randn(M, K, dtype=torch.float32)
w_float = torch.randn(N, K, dtype=torch.float32)
w_ternary = torch.zeros_like(w_float)
w_ternary[w_float > 0.5] = 1.0
w_ternary[w_float < -0.5] = -1.0

def benchmark(fn, name, num_runs=100, warmup=5):
    for _ in range(warmup):
        fn()

    times = []
    for _ in range(num_runs):
        start = time.perf_counter()
        fn()
        times.append(time.perf_counter() - start)

    avg_time = np.mean(times) * 1000
    std_time = np.std(times) * 1000
    ops = 2.0 * M * N * K
    gflops = (ops / 1e9) / (avg_time / 1000)

    print(f"\n{name}:")
    print(f"  Time: {avg_time:.2f} ± {std_time:.2f} ms")
    print(f"  GFLOPS: {gflops:.1f}")
    return gflops

# PyTorch baseline
pytorch_gflops = benchmark(
    lambda: F.linear(x, w_ternary),
    "PyTorch F.linear"
)

# Current NEON
y_neon = torch.zeros(M, N, dtype=torch.float32)
bias_neon = torch.zeros(N, dtype=torch.float32)
neon_gflops = benchmark(
    lambda: neon_kernels.matmul_free_neon(x, w_ternary, y_neon, bias_neon, M, N, K, num_threads),
    "Current NEON (8-way blocking)"
)

# T-MAC
K_bytes = (K + 7) // 8
w_packed = torch.zeros(N, 2 * K_bytes, dtype=torch.uint8)
tmac_kernels.pack_ternary_tmac(w_ternary, w_packed, N, K)
y_tmac = torch.zeros(M, N, dtype=torch.float32)
bias_tmac = torch.zeros(N, dtype=torch.float32)
tmac_gflops = benchmark(
    lambda: tmac_kernels.matmul_free_neon_tmac(x, w_packed, y_tmac, bias_tmac, M, N, K, num_threads),
    "T-MAC Lookup Table"
)

print(f"\n{'='*60}")
print("Summary:")
print(f"{'='*60}")
print(f"PyTorch:      {pytorch_gflops:6.1f} GFLOPS  (baseline)")
print(f"Current NEON: {neon_gflops:6.1f} GFLOPS  ({neon_gflops/pytorch_gflops:.3f}x)")
print(f"T-MAC:        {tmac_gflops:6.1f} GFLOPS  ({tmac_gflops/pytorch_gflops:.3f}x)")
print(f"\nT-MAC vs Current NEON: {tmac_gflops/neon_gflops:.2f}x speedup")
print(f"{'='*60}")
