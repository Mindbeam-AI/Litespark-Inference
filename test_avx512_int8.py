"""
Test AVX-512 Int8 T-MAC kernel vs AVX2 version
"""
import torch
import time
import numpy as np
from pathlib import Path
from torch.utils.cpp_extension import load

KERNELS_DIR = Path(__file__).parent / "src" / "cpu_ops" / "kernels" / "x86_64"

print("="*70)
print("AVX-512 Int8 T-MAC Kernel Test")
print("="*70)

# Compile AVX2 int8 kernel
print("\nCompiling AVX2 int8 kernel...", end=' ', flush=True)
try:
    tmac_int8_avx2 = load(
        name="tmac_int8_avx2_test",
        sources=[str(KERNELS_DIR / "matmul_free_tmac_int8.cpp")],
        extra_cflags=['-O3', '-mavx2', '-mfma', '-fopenmp', '-std=c++17', '-march=native'],
        extra_ldflags=['-fopenmp', '-lgomp'],
        verbose=False
    )
    print("OK")
except Exception as e:
    print(f"FAILED: {e}")
    exit(1)

# Compile AVX-512 int8 kernel
print("Compiling AVX-512 int8 kernel...", end=' ', flush=True)
try:
    tmac_int8_avx512 = load(
        name="tmac_int8_avx512_test",
        sources=[str(KERNELS_DIR / "matmul_free_tmac_int8_avx512.cpp")],
        extra_cflags=['-O3', '-mavx512f', '-mavx512bw', '-mavx512dq', '-mavx512vl',
                      '-fopenmp', '-std=c++17', '-march=native'],
        extra_ldflags=['-fopenmp', '-lgomp'],
        verbose=False
    )
    print("OK")
except Exception as e:
    print(f"FAILED: {e}")
    exit(1)

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

    # Create test data
    x = torch.randn(M, K, dtype=torch.float32)
    w_float = torch.randn(N, K, dtype=torch.float32)

    # Quantize to ternary
    w_ternary = torch.zeros_like(w_float)
    w_ternary[w_float > 0.5] = 1.0
    w_ternary[w_float < -0.5] = -1.0

    # Prepare int8 tensors
    x_int8 = torch.zeros(M, K, dtype=torch.int8)
    scales = torch.zeros(M, dtype=torch.float32)
    K_groups = (K + 3) // 4
    sign_plane = torch.zeros(K_groups, N, dtype=torch.uint8)
    value_plane = torch.zeros(K_groups, N, dtype=torch.uint8)
    bias = torch.Tensor()

    # Quantize and pack using AVX2 version
    tmac_int8_avx2.quantize_activations_int8(x, x_int8, scales, M, K)
    tmac_int8_avx2.pack_ternary_bitplanes_int8(w_ternary, sign_plane, value_plane, N, K)

    # Reference: PyTorch
    y_ref = torch.mm(x, w_ternary.t())

    # Test AVX2 int8
    y_avx2 = torch.zeros(M, N, dtype=torch.float32)
    for _ in range(5):  # warmup
        tmac_int8_avx2.matmul_free_tmac_int8(
            x_int8, scales, sign_plane, value_plane, y_avx2, bias, M, N, K, num_threads
        )

    times = []
    for _ in range(30):
        start = time.perf_counter()
        tmac_int8_avx2.matmul_free_tmac_int8(
            x_int8, scales, sign_plane, value_plane, y_avx2, bias, M, N, K, num_threads
        )
        times.append(time.perf_counter() - start)

    avx2_time = np.mean(times) * 1000
    avx2_gflops = (2.0 * M * N * K / 1e9) / (avx2_time / 1000)
    print(f"  AVX2 int8:   {avx2_time:.2f} ms, {avx2_gflops:.1f} GFLOPS")

    # Test AVX-512 int8
    y_avx512 = torch.zeros(M, N, dtype=torch.float32)
    for _ in range(5):  # warmup
        tmac_int8_avx512.matmul_free_tmac_int8_avx512(
            x_int8, scales, sign_plane, value_plane, y_avx512, bias, M, N, K, num_threads
        )

    times = []
    for _ in range(30):
        start = time.perf_counter()
        tmac_int8_avx512.matmul_free_tmac_int8_avx512(
            x_int8, scales, sign_plane, value_plane, y_avx512, bias, M, N, K, num_threads
        )
        times.append(time.perf_counter() - start)

    avx512_time = np.mean(times) * 1000
    avx512_gflops = (2.0 * M * N * K / 1e9) / (avx512_time / 1000)
    speedup = avx2_time / avx512_time
    print(f"  AVX-512 int8: {avx512_time:.2f} ms, {avx512_gflops:.1f} GFLOPS ({speedup:.2f}x vs AVX2)")

    # Correctness check
    diff_avx2 = torch.abs(y_avx2 - y_ref).max().item()
    diff_avx512 = torch.abs(y_avx512 - y_ref).max().item()
    rel_err_avx2 = (torch.abs(y_avx2 - y_ref) / (torch.abs(y_ref) + 1e-6)).mean().item()
    rel_err_avx512 = (torch.abs(y_avx512 - y_ref) / (torch.abs(y_ref) + 1e-6)).mean().item()

    print(f"  Correctness: AVX2 rel_err={rel_err_avx2:.4f}, AVX-512 rel_err={rel_err_avx512:.4f}")

print("\n" + "="*70)
print("Test Complete!")
print("="*70)
