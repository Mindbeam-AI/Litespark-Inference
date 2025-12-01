"""
Test T-MAC / TL2 Kernels for x86_64 (Microsoft approach)

This script benchmarks:
1. T-MAC float32 (LUT-based, K-first iteration)
2. TL2 float32 (element-wise LUT)
3. T-MAC int8 with PSHUFB (CORRECT Microsoft approach)
4. TL2 int8 with PSHUFB (CORRECT Microsoft approach)
5. AVX-512 packed (direct computation baseline)

Key insight from Microsoft:
- Int8 activations + int8 LUT + PSHUFB = 32 parallel lookups per instruction
- Float32 LUT = scalar lookups (slow!)
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
print("T-MAC MatMul-Free Kernel Test (x86_64) - K-First Iteration")
print("="*70)
print(f"Architecture: {arch}")
print(f"System: {platform.system()}")

# Compile T-MAC kernel
print("\nCompiling T-MAC kernel...", end=' ', flush=True)
try:
    tmac_kernel = load(
        name="tmac_x86_kfirst_kernel",
        sources=[str(KERNELS_DIR / "x86_64" / "matmul_free_tmac_avx2.cpp")],
        extra_cflags=['-O3', '-mavx2', '-mfma', '-fopenmp', '-std=c++17', '-march=native'],
        extra_ldflags=['-fopenmp', '-lgomp'],
        verbose=False
    )
    print("OK")
except Exception as e:
    print(f"FAILED: {e}")
    import traceback
    traceback.print_exc()
    exit(1)

# Compile TL2 float32 kernel
print("Compiling TL2 float32 kernel...", end=' ', flush=True)
try:
    tl2_kernel = load(
        name="tl2_x86_for_tmac_test",
        sources=[str(KERNELS_DIR / "x86_64" / "matmul_free_tl2_avx2.cpp")],
        extra_cflags=['-O3', '-mavx2', '-mfma', '-fopenmp', '-std=c++17', '-march=native'],
        extra_ldflags=['-fopenmp', '-lgomp'],
        verbose=False
    )
    print("OK")
except Exception as e:
    print(f"FAILED: {e}")
    tl2_kernel = None

# Compile T-MAC int8 kernel (CORRECT Microsoft approach)
print("Compiling T-MAC int8 kernel...", end=' ', flush=True)
try:
    tmac_int8_kernel = load(
        name="tmac_int8_for_main_test",
        sources=[str(KERNELS_DIR / "x86_64" / "matmul_free_tmac_int8.cpp")],
        extra_cflags=['-O3', '-mavx2', '-mfma', '-fopenmp', '-std=c++17', '-march=native'],
        extra_ldflags=['-fopenmp', '-lgomp'],
        verbose=False
    )
    print("OK")
except Exception as e:
    print(f"FAILED: {e}")
    tmac_int8_kernel = None

# Compile TL2 int8 kernel (CORRECT Microsoft approach)
print("Compiling TL2 int8 kernel...", end=' ', flush=True)
try:
    tl2_int8_kernel = load(
        name="tl2_int8_for_main_test",
        sources=[str(KERNELS_DIR / "x86_64" / "matmul_free_tl2_int8.cpp")],
        extra_cflags=['-O3', '-mavx2', '-mfma', '-fopenmp', '-std=c++17', '-march=native'],
        extra_ldflags=['-fopenmp', '-lgomp'],
        verbose=False
    )
    print("OK")
except Exception as e:
    print(f"FAILED: {e}")
    tl2_int8_kernel = None

# Compile AVX-512 packed kernel for comparison
print("Compiling AVX-512 packed kernel...", end=' ', flush=True)
try:
    avx512_kernel = load(
        name="avx512_packed_for_tmac_kfirst",
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

    # Pack weights for T-MAC K-first format [K_groups, N]
    K_groups = (K + 3) // 4
    sign_plane = torch.zeros(K_groups, N, dtype=torch.uint8)
    value_plane = torch.zeros(K_groups, N, dtype=torch.uint8)
    tmac_kernel.pack_ternary_bitplanes(w_ternary, sign_plane, value_plane, N, K)

    # Pack weights for TL2 float32 (3 weights per group)
    if tl2_kernel is not None:
        K_groups_tl2 = (K + 2) // 3
        w_packed_tl2 = torch.zeros(K_groups_tl2, N, dtype=torch.uint8)
        tl2_kernel.pack_ternary_tl2(w_ternary, w_packed_tl2, N, K)

    # Prepare int8 activations and weights for T-MAC int8
    if tmac_int8_kernel is not None:
        x_int8 = torch.zeros(M, K, dtype=torch.int8)
        scales = torch.zeros(M, dtype=torch.float32)
        tmac_int8_kernel.quantize_activations_int8(x, x_int8, scales, M, K)

        sign_plane_int8 = torch.zeros(K_groups, N, dtype=torch.uint8)
        value_plane_int8 = torch.zeros(K_groups, N, dtype=torch.uint8)
        tmac_int8_kernel.pack_ternary_bitplanes_int8(w_ternary, sign_plane_int8, value_plane_int8, N, K)

    # Prepare int8 weights for TL2 int8
    if tl2_int8_kernel is not None:
        if tmac_int8_kernel is None:
            # Need to quantize activations if not done above
            x_int8 = torch.zeros(M, K, dtype=torch.int8)
            scales = torch.zeros(M, dtype=torch.float32)
            tl2_int8_kernel.quantize_activations_int8_tl2(x, x_int8, scales, M, K)

        K_groups_tl2_int8 = (K + 2) // 3
        w_packed_tl2_int8 = torch.zeros(K_groups_tl2_int8, N, dtype=torch.uint8)
        tl2_int8_kernel.pack_ternary_tl2_int8(w_ternary, w_packed_tl2_int8, N, K)

    # Pack weights for AVX-512 (2-bit, 16 weights per uint32)
    if avx512_kernel is not None:
        K_packed_avx512 = (K + 15) // 16
        w_packed_avx512 = torch.zeros(N, K_packed_avx512 * 4, dtype=torch.uint8)
        avx512_kernel.pack_weights_2bit_avx512(w_ternary, w_packed_avx512, N, K)

    bias = torch.zeros(N, dtype=torch.float32)
    results = {}

    # Weight memory comparison
    float_mem = N * K * 4 / (1024 * 1024)
    # Bit-plane: 2 planes, K_groups * N bytes each
    bitplane_mem = K_groups * N * 2 / (1024 * 1024)
    print(f"\n  Weight memory: {float_mem:.2f}MB (float32) -> {bitplane_mem:.2f}MB (bit-planes) = {float_mem/bitplane_mem:.1f}x reduction")

    # Test T-MAC scalar (K-first)
    print("\n  T-MAC K-first scalar:")
    y_tmac = torch.zeros(M, N, dtype=torch.float32)

    for _ in range(5):  # Warmup
        tmac_kernel.matmul_free_tmac(x, sign_plane, value_plane, y_tmac, bias, M, N, K, num_threads)

    gc.collect()
    times = []
    for _ in range(30):
        start = time.perf_counter()
        tmac_kernel.matmul_free_tmac(x, sign_plane, value_plane, y_tmac, bias, M, N, K, num_threads)
        times.append(time.perf_counter() - start)

    avg_time = np.mean(times) * 1000
    std_time = np.std(times) * 1000
    gflops = (2.0 * M * N * K / 1e9) / (avg_time / 1000)
    print(f"    Time: {avg_time:.2f} +/- {std_time:.2f} ms, GFLOPS: {gflops:.1f}")
    results['tmac_scalar'] = {'time': avg_time, 'gflops': gflops, 'y': y_tmac.clone()}

    # Test T-MAC AVX2
    print("\n  T-MAC K-first AVX2:")
    y_tmac_avx2 = torch.zeros(M, N, dtype=torch.float32)

    for _ in range(5):
        tmac_kernel.matmul_free_tmac_avx2(x, sign_plane, value_plane, y_tmac_avx2, bias, M, N, K, num_threads)

    gc.collect()
    times = []
    for _ in range(30):
        start = time.perf_counter()
        tmac_kernel.matmul_free_tmac_avx2(x, sign_plane, value_plane, y_tmac_avx2, bias, M, N, K, num_threads)
        times.append(time.perf_counter() - start)

    avg_time = np.mean(times) * 1000
    std_time = np.std(times) * 1000
    gflops = (2.0 * M * N * K / 1e9) / (avg_time / 1000)
    print(f"    Time: {avg_time:.2f} +/- {std_time:.2f} ms, GFLOPS: {gflops:.1f}")
    results['tmac_avx2'] = {'time': avg_time, 'gflops': gflops, 'y': y_tmac_avx2.clone()}

    # Test T-MAC Tiled
    print("\n  T-MAC K-first Tiled:")
    y_tmac_tiled = torch.zeros(M, N, dtype=torch.float32)

    for _ in range(5):
        tmac_kernel.matmul_free_tmac_tiled(x, sign_plane, value_plane, y_tmac_tiled, bias, M, N, K, num_threads)

    gc.collect()
    times = []
    for _ in range(30):
        start = time.perf_counter()
        tmac_kernel.matmul_free_tmac_tiled(x, sign_plane, value_plane, y_tmac_tiled, bias, M, N, K, num_threads)
        times.append(time.perf_counter() - start)

    avg_time = np.mean(times) * 1000
    std_time = np.std(times) * 1000
    gflops = (2.0 * M * N * K / 1e9) / (avg_time / 1000)
    print(f"    Time: {avg_time:.2f} +/- {std_time:.2f} ms, GFLOPS: {gflops:.1f}")
    results['tmac_tiled'] = {'time': avg_time, 'gflops': gflops, 'y': y_tmac_tiled.clone()}

    # Test T-MAC PSHUFB
    print("\n  T-MAC K-first PSHUFB:")
    y_tmac_pshufb = torch.zeros(M, N, dtype=torch.float32)

    for _ in range(5):
        tmac_kernel.matmul_free_tmac_pshufb(x, sign_plane, value_plane, y_tmac_pshufb, bias, M, N, K, num_threads)

    gc.collect()
    times = []
    for _ in range(30):
        start = time.perf_counter()
        tmac_kernel.matmul_free_tmac_pshufb(x, sign_plane, value_plane, y_tmac_pshufb, bias, M, N, K, num_threads)
        times.append(time.perf_counter() - start)

    avg_time = np.mean(times) * 1000
    std_time = np.std(times) * 1000
    gflops = (2.0 * M * N * K / 1e9) / (avg_time / 1000)
    print(f"    Time: {avg_time:.2f} +/- {std_time:.2f} ms, GFLOPS: {gflops:.1f}")
    results['tmac_pshufb'] = {'time': avg_time, 'gflops': gflops, 'y': y_tmac_pshufb.clone()}

    # Test T-MAC Optimized
    print("\n  T-MAC K-first Optimized:")
    y_tmac_opt = torch.zeros(M, N, dtype=torch.float32)

    for _ in range(5):
        tmac_kernel.matmul_free_tmac_optimized(x, sign_plane, value_plane, y_tmac_opt, bias, M, N, K, num_threads)

    gc.collect()
    times = []
    for _ in range(30):
        start = time.perf_counter()
        tmac_kernel.matmul_free_tmac_optimized(x, sign_plane, value_plane, y_tmac_opt, bias, M, N, K, num_threads)
        times.append(time.perf_counter() - start)

    avg_time = np.mean(times) * 1000
    std_time = np.std(times) * 1000
    gflops = (2.0 * M * N * K / 1e9) / (avg_time / 1000)
    print(f"    Time: {avg_time:.2f} +/- {std_time:.2f} ms, GFLOPS: {gflops:.1f}")
    results['tmac_optimized'] = {'time': avg_time, 'gflops': gflops, 'y': y_tmac_opt.clone()}

    # Test T-MAC Gather (AVX2 gather instructions)
    print("\n  T-MAC K-first Gather:")
    y_tmac_gather = torch.zeros(M, N, dtype=torch.float32)

    for _ in range(5):
        tmac_kernel.matmul_free_tmac_gather(x, sign_plane, value_plane, y_tmac_gather, bias, M, N, K, num_threads)

    gc.collect()
    times = []
    for _ in range(30):
        start = time.perf_counter()
        tmac_kernel.matmul_free_tmac_gather(x, sign_plane, value_plane, y_tmac_gather, bias, M, N, K, num_threads)
        times.append(time.perf_counter() - start)

    avg_time = np.mean(times) * 1000
    std_time = np.std(times) * 1000
    gflops = (2.0 * M * N * K / 1e9) / (avg_time / 1000)
    print(f"    Time: {avg_time:.2f} +/- {std_time:.2f} ms, GFLOPS: {gflops:.1f}")
    results['tmac_gather'] = {'time': avg_time, 'gflops': gflops, 'y': y_tmac_gather.clone()}

    # Test TL2 float32 Optimized
    if tl2_kernel is not None:
        print("\n  TL2 float32 (element-wise LUT):")
        y_tl2 = torch.zeros(M, N, dtype=torch.float32)

        for _ in range(5):
            tl2_kernel.matmul_free_tl2_optimized(x, w_packed_tl2, y_tl2, bias, M, N, K, num_threads)

        gc.collect()
        times = []
        for _ in range(30):
            start = time.perf_counter()
            tl2_kernel.matmul_free_tl2_optimized(x, w_packed_tl2, y_tl2, bias, M, N, K, num_threads)
            times.append(time.perf_counter() - start)

        avg_time = np.mean(times) * 1000
        std_time = np.std(times) * 1000
        gflops = (2.0 * M * N * K / 1e9) / (avg_time / 1000)
        print(f"    Time: {avg_time:.2f} +/- {std_time:.2f} ms, GFLOPS: {gflops:.1f}")
        results['tl2_float32'] = {'time': avg_time, 'gflops': gflops, 'y': y_tl2.clone()}

    # Test T-MAC int8 PSHUFB (CORRECT Microsoft approach)
    if tmac_int8_kernel is not None:
        print("\n  T-MAC int8 PSHUFB:")
        y_tmac_int8 = torch.zeros(M, N, dtype=torch.float32)

        for _ in range(5):
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

    # Test TL2 int8 PSHUFB (CORRECT Microsoft approach)
    if tl2_int8_kernel is not None:
        print("\n  TL2 int8 PSHUFB:")
        y_tl2_int8 = torch.zeros(M, N, dtype=torch.float32)

        for _ in range(5):
            tl2_int8_kernel.matmul_free_tl2_int8(
                x_int8, scales, w_packed_tl2_int8,
                y_tl2_int8, bias, M, N, K, num_threads
            )

        gc.collect()
        times = []
        for _ in range(30):
            start = time.perf_counter()
            tl2_int8_kernel.matmul_free_tl2_int8(
                x_int8, scales, w_packed_tl2_int8,
                y_tl2_int8, bias, M, N, K, num_threads
            )
            times.append(time.perf_counter() - start)

        avg_time = np.mean(times) * 1000
        std_time = np.std(times) * 1000
        gflops = (2.0 * M * N * K / 1e9) / (avg_time / 1000)
        print(f"    Time: {avg_time:.2f} +/- {std_time:.2f} ms, GFLOPS: {gflops:.1f}")
        results['tl2_int8'] = {'time': avg_time, 'gflops': gflops, 'y': y_tl2_int8.clone()}

    # Test AVX-512 packed v1 for comparison
    if avx512_kernel is not None:
        print("\n  AVX-512 packed v1 (baseline):")
        y_avx512 = torch.zeros(M, N, dtype=torch.float32)

        for _ in range(5):
            avx512_kernel.matmul_free_avx512_packed_v1(x, w_packed_avx512, y_avx512, bias, M, N, K, num_threads)

        gc.collect()
        times = []
        for _ in range(30):
            start = time.perf_counter()
            avx512_kernel.matmul_free_avx512_packed_v1(x, w_packed_avx512, y_avx512, bias, M, N, K, num_threads)
            times.append(time.perf_counter() - start)

        avg_time = np.mean(times) * 1000
        std_time = np.std(times) * 1000
        gflops = (2.0 * M * N * K / 1e9) / (avg_time / 1000)
        print(f"    Time: {avg_time:.2f} +/- {std_time:.2f} ms, GFLOPS: {gflops:.1f}")
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

    # Verify correctness
    print("\n  Correctness check vs PyTorch:")

    # Float32 kernels - tight tolerance
    for key in ['tmac_scalar', 'tmac_avx2', 'tmac_tiled', 'tmac_pshufb', 'tmac_optimized', 'tmac_gather', 'tl2_float32', 'avx512_packed']:
        if key in results:
            max_diff = torch.max(torch.abs(results[key]['y'] - y_pt)).item()
            status = 'OK' if max_diff < 1e-3 else 'MISMATCH!'
            print(f"    {key}: max_diff={max_diff:.6f} {status}")

    # Int8 kernels - allow quantization error (~1-2% relative)
    for key in ['tmac_int8', 'tl2_int8']:
        if key in results:
            max_diff = torch.max(torch.abs(results[key]['y'] - y_pt)).item()
            rel_error = max_diff / (torch.max(torch.abs(y_pt)).item() + 1e-6)
            status = 'OK' if rel_error < 0.02 else 'MISMATCH!'
            print(f"    {key}: max_diff={max_diff:.4f}, rel_err={rel_error:.4f} {status}")

    # Summary
    print(f"\n  Summary for {name}:")
    print(f"    {'Kernel':<30} {'GFLOPS':>10} {'vs AVX-512':>12}")
    print(f"    {'-'*55}")

    avx512_gflops = results.get('avx512_packed', {}).get('gflops', 1.0)
    for key, label in [
        ('tmac_scalar', 'T-MAC float32 scalar'),
        ('tmac_avx2', 'T-MAC float32 AVX2'),
        ('tmac_tiled', 'T-MAC float32 Tiled'),
        ('tmac_pshufb', 'T-MAC float32 PSHUFB'),
        ('tmac_optimized', 'T-MAC float32 Optimized'),
        ('tmac_gather', 'T-MAC float32 Gather'),
        ('tl2_float32', 'TL2 float32'),
        ('tmac_int8', 'T-MAC int8 PSHUFB'),
        ('tl2_int8', 'TL2 int8 PSHUFB'),
        ('avx512_packed', 'AVX-512 packed'),
        ('pytorch', 'PyTorch'),
    ]:
        if key in results:
            gflops = results[key]['gflops']
            ratio = gflops / avx512_gflops if avx512_gflops > 0 else 0
            print(f"    {label:<30} {gflops:>10.1f} {ratio:>11.2f}x")

print("\n" + "="*70)
print("T-MAC / TL2 Test Complete!")
print("="*70)
print("\nKey Insights:")
print("  1. Float32 LUT = SLOW (scalar lookups, ~50 GFLOPS)")
print("  2. Int8 LUT + PSHUFB = FAST (32 parallel lookups)")
print("  3. AVX-512 packed (direct computation) = baseline (~145 GFLOPS)")
print("\nMicrosoft's approach (BitNet.cpp):")
print("  - Int8 activations (per-row quantization)")
print("  - Int8 LUT values")
print("  - PSHUFB for 32 parallel lookups per instruction")
print("  - Int32 accumulation, float32 output")
print("="*70)
