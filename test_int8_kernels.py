"""
Test Int8 T-MAC and TL2 Kernels for x86_64

This is the CORRECT implementation following Microsoft's approach:
- Activations: int8 (per-row quantization)
- LUT: int8 values
- Accumulation: int32
- PSHUFB for 32 parallel lookups

Expected: Significant speedup over float32 versions due to PSHUFB!
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
    process = psutil.Process(os.getpid())
    return process.memory_info().rss / (1024 * 1024)


# Check architecture
arch = platform.machine().lower()
if arch not in ['x86_64', 'amd64']:
    print(f"Warning: This script is for x86_64 systems. Current: {arch}")
    exit(1)

KERNELS_DIR = Path(__file__).parent / "src" / "cpu_ops" / "kernels"

print("="*70)
print("Int8 T-MAC / TL2 Kernel Test (x86_64) - PSHUFB Implementation")
print("="*70)
print(f"Architecture: {arch}")
print(f"System: {platform.system()}")

# Compile Int8 T-MAC kernel
print("\nCompiling Int8 T-MAC kernel...", end=' ', flush=True)
try:
    tmac_int8_kernel = load(
        name="tmac_int8_kernel",
        sources=[str(KERNELS_DIR / "x86_64" / "matmul_free_tmac_int8.cpp")],
        extra_cflags=['-O3', '-mavx2', '-mfma', '-fopenmp', '-std=c++17', '-march=native'],
        extra_ldflags=['-fopenmp', '-lgomp'],
        verbose=False
    )
    print("OK")
except Exception as e:
    print(f"FAILED: {e}")
    import traceback
    traceback.print_exc()
    tmac_int8_kernel = None

# Compile Int8 TL2 kernel
print("Compiling Int8 TL2 kernel...", end=' ', flush=True)
try:
    tl2_int8_kernel = load(
        name="tl2_int8_kernel",
        sources=[str(KERNELS_DIR / "x86_64" / "matmul_free_tl2_int8.cpp")],
        extra_cflags=['-O3', '-mavx2', '-mfma', '-fopenmp', '-std=c++17', '-march=native'],
        extra_ldflags=['-fopenmp', '-lgomp'],
        verbose=False
    )
    print("OK")
except Exception as e:
    print(f"FAILED: {e}")
    import traceback
    traceback.print_exc()
    tl2_int8_kernel = None

# Compile float32 T-MAC for comparison
print("Compiling float32 T-MAC kernel...", end=' ', flush=True)
try:
    tmac_float_kernel = load(
        name="tmac_float_for_int8_test",
        sources=[str(KERNELS_DIR / "x86_64" / "matmul_free_tmac_avx2.cpp")],
        extra_cflags=['-O3', '-mavx2', '-mfma', '-fopenmp', '-std=c++17', '-march=native'],
        extra_ldflags=['-fopenmp', '-lgomp'],
        verbose=False
    )
    print("OK")
except Exception as e:
    print(f"FAILED: {e}")
    tmac_float_kernel = None

# Compile AVX-512 packed kernel for comparison
print("Compiling AVX-512 packed kernel...", end=' ', flush=True)
try:
    avx512_kernel = load(
        name="avx512_for_int8_test",
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

if tmac_int8_kernel is None and tl2_int8_kernel is None:
    print("\nNo int8 kernels compiled!")
    exit(1)

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

    # Create test data
    x = torch.randn(M, K, dtype=torch.float32)
    w_float = torch.randn(N, K, dtype=torch.float32)

    # Quantize weights to ternary
    w_ternary = torch.zeros_like(w_float)
    w_ternary[w_float > 0.5] = 1.0
    w_ternary[w_float < -0.5] = -1.0

    bias = torch.zeros(N, dtype=torch.float32)
    results = {}

    # Prepare int8 activations
    if tmac_int8_kernel is not None or tl2_int8_kernel is not None:
        x_int8 = torch.zeros(M, K, dtype=torch.int8)
        scales = torch.zeros(M, dtype=torch.float32)
        if tmac_int8_kernel is not None:
            tmac_int8_kernel.quantize_activations_int8(x, x_int8, scales, M, K)
        else:
            tl2_int8_kernel.quantize_activations_int8_tl2(x, x_int8, scales, M, K)

    # Pack weights for T-MAC int8
    if tmac_int8_kernel is not None:
        K_groups_tmac = (K + 3) // 4
        sign_plane_int8 = torch.zeros(K_groups_tmac, N, dtype=torch.uint8)
        value_plane_int8 = torch.zeros(K_groups_tmac, N, dtype=torch.uint8)
        tmac_int8_kernel.pack_ternary_bitplanes_int8(w_ternary, sign_plane_int8, value_plane_int8, N, K)

    # Pack weights for TL2 int8
    if tl2_int8_kernel is not None:
        K_groups_tl2 = (K + 2) // 3
        packed_tl2_int8 = torch.zeros(K_groups_tl2, N, dtype=torch.uint8)
        tl2_int8_kernel.pack_ternary_tl2_int8(w_ternary, packed_tl2_int8, N, K)

    # Pack weights for float32 T-MAC
    if tmac_float_kernel is not None:
        K_groups_tmac = (K + 3) // 4
        sign_plane_f32 = torch.zeros(K_groups_tmac, N, dtype=torch.uint8)
        value_plane_f32 = torch.zeros(K_groups_tmac, N, dtype=torch.uint8)
        tmac_float_kernel.pack_ternary_bitplanes(w_ternary, sign_plane_f32, value_plane_f32, N, K)

    # Pack weights for AVX-512
    if avx512_kernel is not None:
        K_packed_avx512 = (K + 15) // 16
        w_packed_avx512 = torch.zeros(N, K_packed_avx512 * 4, dtype=torch.uint8)
        avx512_kernel.pack_weights_2bit_avx512(w_ternary, w_packed_avx512, N, K)

    # Test T-MAC Int8 PSHUFB
    if tmac_int8_kernel is not None:
        print("\n  T-MAC Int8 PSHUFB:")
        y_tmac_int8 = torch.zeros(M, N, dtype=torch.float32)

        for _ in range(5):  # Warmup
            tmac_int8_kernel.matmul_free_tmac_int8(
                x_int8, scales, sign_plane_int8, value_plane_int8,
                y_tmac_int8, bias, M, N, K, num_threads
            )

        gc.collect()
        times = []
        for _ in range(30):
            start = time.perf_counter()
            tmac_int8_kernel.matmul_free_tmac_int8(
                x_int8, scales, sign_plane_int8, value_plane_int8,
                y_tmac_int8, bias, M, N, K, num_threads
            )
            times.append(time.perf_counter() - start)

        avg_time = np.mean(times) * 1000
        std_time = np.std(times) * 1000
        gflops = (2.0 * M * N * K / 1e9) / (avg_time / 1000)
        print(f"    Time: {avg_time:.2f} +/- {std_time:.2f} ms, GFLOPS: {gflops:.1f}")
        results['tmac_int8'] = {'time': avg_time, 'gflops': gflops, 'y': y_tmac_int8.clone()}

    # Test T-MAC Int8 Scalar (for correctness reference)
    if tmac_int8_kernel is not None:
        print("\n  T-MAC Int8 Scalar:")
        y_tmac_int8_scalar = torch.zeros(M, N, dtype=torch.float32)

        for _ in range(3):
            tmac_int8_kernel.matmul_free_tmac_int8_scalar(
                x_int8, scales, sign_plane_int8, value_plane_int8,
                y_tmac_int8_scalar, bias, M, N, K, num_threads
            )

        gc.collect()
        times = []
        for _ in range(10):
            start = time.perf_counter()
            tmac_int8_kernel.matmul_free_tmac_int8_scalar(
                x_int8, scales, sign_plane_int8, value_plane_int8,
                y_tmac_int8_scalar, bias, M, N, K, num_threads
            )
            times.append(time.perf_counter() - start)

        avg_time = np.mean(times) * 1000
        gflops = (2.0 * M * N * K / 1e9) / (avg_time / 1000)
        print(f"    Time: {avg_time:.2f} ms, GFLOPS: {gflops:.1f}")
        results['tmac_int8_scalar'] = {'time': avg_time, 'gflops': gflops, 'y': y_tmac_int8_scalar.clone()}

    # Test TL2 Int8 PSHUFB
    if tl2_int8_kernel is not None:
        print("\n  TL2 Int8 PSHUFB:")
        y_tl2_int8 = torch.zeros(M, N, dtype=torch.float32)

        for _ in range(5):
            tl2_int8_kernel.matmul_free_tl2_int8(
                x_int8, scales, packed_tl2_int8,
                y_tl2_int8, bias, M, N, K, num_threads
            )

        gc.collect()
        times = []
        for _ in range(30):
            start = time.perf_counter()
            tl2_int8_kernel.matmul_free_tl2_int8(
                x_int8, scales, packed_tl2_int8,
                y_tl2_int8, bias, M, N, K, num_threads
            )
            times.append(time.perf_counter() - start)

        avg_time = np.mean(times) * 1000
        std_time = np.std(times) * 1000
        gflops = (2.0 * M * N * K / 1e9) / (avg_time / 1000)
        print(f"    Time: {avg_time:.2f} +/- {std_time:.2f} ms, GFLOPS: {gflops:.1f}")
        results['tl2_int8'] = {'time': avg_time, 'gflops': gflops, 'y': y_tl2_int8.clone()}

    # Test TL2 Int8 Scalar
    if tl2_int8_kernel is not None:
        print("\n  TL2 Int8 Scalar:")
        y_tl2_int8_scalar = torch.zeros(M, N, dtype=torch.float32)

        gc.collect()
        times = []
        for _ in range(10):
            start = time.perf_counter()
            tl2_int8_kernel.matmul_free_tl2_int8_scalar(
                x_int8, scales, packed_tl2_int8,
                y_tl2_int8_scalar, bias, M, N, K, num_threads
            )
            times.append(time.perf_counter() - start)

        avg_time = np.mean(times) * 1000
        gflops = (2.0 * M * N * K / 1e9) / (avg_time / 1000)
        print(f"    Time: {avg_time:.2f} ms, GFLOPS: {gflops:.1f}")
        results['tl2_int8_scalar'] = {'time': avg_time, 'gflops': gflops, 'y': y_tl2_int8_scalar.clone()}

    # Test float32 T-MAC PSHUFB for comparison
    if tmac_float_kernel is not None:
        print("\n  T-MAC float32 PSHUFB (comparison):")
        y_tmac_f32 = torch.zeros(M, N, dtype=torch.float32)

        for _ in range(5):
            tmac_float_kernel.matmul_free_tmac_pshufb(
                x, sign_plane_f32, value_plane_f32,
                y_tmac_f32, bias, M, N, K, num_threads
            )

        gc.collect()
        times = []
        for _ in range(30):
            start = time.perf_counter()
            tmac_float_kernel.matmul_free_tmac_pshufb(
                x, sign_plane_f32, value_plane_f32,
                y_tmac_f32, bias, M, N, K, num_threads
            )
            times.append(time.perf_counter() - start)

        avg_time = np.mean(times) * 1000
        gflops = (2.0 * M * N * K / 1e9) / (avg_time / 1000)
        print(f"    Time: {avg_time:.2f} ms, GFLOPS: {gflops:.1f}")
        results['tmac_float32'] = {'time': avg_time, 'gflops': gflops, 'y': y_tmac_f32.clone()}

    # Test AVX-512 packed
    if avx512_kernel is not None:
        print("\n  AVX-512 packed (baseline):")
        y_avx512 = torch.zeros(M, N, dtype=torch.float32)

        for _ in range(5):
            avx512_kernel.matmul_free_avx512_packed_v1(
                x, w_packed_avx512, y_avx512, bias, M, N, K, num_threads
            )

        gc.collect()
        times = []
        for _ in range(30):
            start = time.perf_counter()
            avx512_kernel.matmul_free_avx512_packed_v1(
                x, w_packed_avx512, y_avx512, bias, M, N, K, num_threads
            )
            times.append(time.perf_counter() - start)

        avg_time = np.mean(times) * 1000
        gflops = (2.0 * M * N * K / 1e9) / (avg_time / 1000)
        print(f"    Time: {avg_time:.2f} ms, GFLOPS: {gflops:.1f}")
        results['avx512_packed'] = {'time': avg_time, 'gflops': gflops, 'y': y_avx512.clone()}

    # PyTorch reference
    print("\n  PyTorch (reference):")
    w_pt = w_ternary.clone()
    for _ in range(5):
        _ = torch.matmul(x, w_pt.t())

    gc.collect()
    times = []
    for _ in range(30):
        start = time.perf_counter()
        y_pt = torch.matmul(x, w_pt.t())
        times.append(time.perf_counter() - start)

    avg_time = np.mean(times) * 1000
    gflops = (2.0 * M * N * K / 1e9) / (avg_time / 1000)
    print(f"    Time: {avg_time:.2f} ms, GFLOPS: {gflops:.1f}")
    results['pytorch'] = {'time': avg_time, 'gflops': gflops, 'y': y_pt}

    # Correctness check
    print("\n  Correctness check vs PyTorch:")

    # Note: Int8 kernels will have quantization error, so use higher tolerance
    for key in ['tmac_int8', 'tmac_int8_scalar', 'tl2_int8', 'tl2_int8_scalar']:
        if key in results:
            max_diff = torch.max(torch.abs(results[key]['y'] - y_pt)).item()
            rel_error = max_diff / (torch.max(torch.abs(y_pt)).item() + 1e-6)
            # Int8 has quantization error, allow ~1% relative error
            status = 'OK' if rel_error < 0.02 else 'MISMATCH!'
            print(f"    {key}: max_diff={max_diff:.4f}, rel_err={rel_error:.4f} {status}")

    for key in ['tmac_float32', 'avx512_packed']:
        if key in results:
            max_diff = torch.max(torch.abs(results[key]['y'] - y_pt)).item()
            status = 'OK' if max_diff < 1e-3 else 'MISMATCH!'
            print(f"    {key}: max_diff={max_diff:.6f} {status}")

    # Summary
    print(f"\n  Summary for {name}:")
    print(f"    {'Kernel':<25} {'GFLOPS':>10} {'vs AVX-512':>12} {'vs float32':>12}")
    print(f"    {'-'*62}")

    avx512_gflops = results.get('avx512_packed', {}).get('gflops', 1.0)
    f32_gflops = results.get('tmac_float32', {}).get('gflops', 1.0)

    for key, label in [
        ('tmac_int8', 'T-MAC Int8 PSHUFB'),
        ('tmac_int8_scalar', 'T-MAC Int8 Scalar'),
        ('tl2_int8', 'TL2 Int8 PSHUFB'),
        ('tl2_int8_scalar', 'TL2 Int8 Scalar'),
        ('tmac_float32', 'T-MAC float32'),
        ('avx512_packed', 'AVX-512 packed'),
        ('pytorch', 'PyTorch'),
    ]:
        if key in results:
            gflops = results[key]['gflops']
            vs_avx512 = gflops / avx512_gflops if avx512_gflops > 0 else 0
            vs_f32 = gflops / f32_gflops if f32_gflops > 0 else 0
            print(f"    {label:<25} {gflops:>10.1f} {vs_avx512:>11.2f}x {vs_f32:>11.2f}x")

print("\n" + "="*70)
print("Int8 Kernel Test Complete!")
print("="*70)
print("\nKey insight: Int8 + PSHUFB enables 32 parallel lookups per instruction!")
print("Expected speedup over float32: 2-4x due to true SIMD LUT lookups")
print("="*70)
