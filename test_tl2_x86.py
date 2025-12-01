"""
Test TL2 Element-wise LUT Kernel for x86_64

Based on Microsoft BitNet.cpp TL2 approach (arxiv 2502.11880):
- Groups 3 ternary weights instead of 4 (T-MAC)
- Uses element-wise LUT (3^3 = 27 entries) vs bit-wise (2^4 = 16)
- Mirror consolidation reduces to 14 entries + sign bit
- Achieves 1.67 bits per weight (vs 2.0 for T-MAC)

Key insight from paper:
"T-MAC employs bit-wise LUT methods, which for ternary weights exhibit
spatial inefficiencies, leading to a substantial performance decline."

TL2 achieves 2.32x speedup over T-MAC on Intel i7-13700H!
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
print("TL2 Element-wise LUT Kernel Test (x86_64)")
print("="*70)
print(f"Architecture: {arch}")
print(f"System: {platform.system()}")

# Compile TL2 kernel
print("\nCompiling TL2 kernel...", end=' ', flush=True)
try:
    tl2_kernel = load(
        name="tl2_x86_kernel",
        sources=[str(KERNELS_DIR / "x86_64" / "matmul_free_tl2_avx2.cpp")],
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

# Compile T-MAC kernel for comparison
print("Compiling T-MAC kernel...", end=' ', flush=True)
try:
    tmac_kernel = load(
        name="tmac_x86_for_tl2_compare",
        sources=[str(KERNELS_DIR / "x86_64" / "matmul_free_tmac_avx2.cpp")],
        extra_cflags=['-O3', '-mavx2', '-mfma', '-fopenmp', '-std=c++17', '-march=native'],
        extra_ldflags=['-fopenmp', '-lgomp'],
        verbose=False
    )
    print("OK")
except Exception as e:
    print(f"FAILED: {e}")
    tmac_kernel = None

# Compile AVX-512 packed kernel for comparison
print("Compiling AVX-512 packed kernel...", end=' ', flush=True)
try:
    avx512_kernel = load(
        name="avx512_packed_for_tl2",
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

    # Pack weights for TL2 (3 weights per group)
    K_groups_tl2 = (K + 2) // 3
    w_packed_tl2 = torch.zeros(K_groups_tl2, N, dtype=torch.uint8)
    tl2_kernel.pack_ternary_tl2(w_ternary, w_packed_tl2, N, K)

    # Pack weights for T-MAC (4 weights per group)
    if tmac_kernel is not None:
        K_groups_tmac = (K + 3) // 4
        sign_plane = torch.zeros(K_groups_tmac, N, dtype=torch.uint8)
        value_plane = torch.zeros(K_groups_tmac, N, dtype=torch.uint8)
        tmac_kernel.pack_ternary_bitplanes(w_ternary, sign_plane, value_plane, N, K)

    # Pack weights for AVX-512
    if avx512_kernel is not None:
        K_packed_avx512 = (K + 15) // 16
        w_packed_avx512 = torch.zeros(N, K_packed_avx512 * 4, dtype=torch.uint8)
        avx512_kernel.pack_weights_2bit_avx512(w_ternary, w_packed_avx512, N, K)

    bias = torch.zeros(N, dtype=torch.float32)
    results = {}

    # Weight memory comparison
    float_mem = N * K * 4 / (1024 * 1024)
    tl2_mem = K_groups_tl2 * N / (1024 * 1024)  # 1 byte per group (5 bits used)
    tmac_mem = K_groups_tmac * N * 2 / (1024 * 1024) if tmac_kernel else 0  # 2 planes
    print(f"\n  Weight memory:")
    print(f"    float32:  {float_mem:.2f}MB")
    print(f"    TL2:      {tl2_mem:.2f}MB ({float_mem/tl2_mem:.1f}x reduction, 1.67 bpw)")
    if tmac_kernel:
        print(f"    T-MAC:    {tmac_mem:.2f}MB ({float_mem/tmac_mem:.1f}x reduction, 2.0 bpw)")

    # Test TL2 scalar
    print("\n  TL2 scalar:")
    y_tl2 = torch.zeros(M, N, dtype=torch.float32)

    for _ in range(5):  # Warmup
        tl2_kernel.matmul_free_tl2(x, w_packed_tl2, y_tl2, bias, M, N, K, num_threads)

    gc.collect()
    times = []
    for _ in range(30):
        start = time.perf_counter()
        tl2_kernel.matmul_free_tl2(x, w_packed_tl2, y_tl2, bias, M, N, K, num_threads)
        times.append(time.perf_counter() - start)

    avg_time = np.mean(times) * 1000
    std_time = np.std(times) * 1000
    gflops = (2.0 * M * N * K / 1e9) / (avg_time / 1000)
    print(f"    Time: {avg_time:.2f} +/- {std_time:.2f} ms, GFLOPS: {gflops:.1f}")
    results['tl2_scalar'] = {'time': avg_time, 'gflops': gflops, 'y': y_tl2.clone()}

    # Test TL2 AVX2
    print("\n  TL2 AVX2:")
    y_tl2_avx2 = torch.zeros(M, N, dtype=torch.float32)

    for _ in range(5):
        tl2_kernel.matmul_free_tl2_avx2(x, w_packed_tl2, y_tl2_avx2, bias, M, N, K, num_threads)

    gc.collect()
    times = []
    for _ in range(30):
        start = time.perf_counter()
        tl2_kernel.matmul_free_tl2_avx2(x, w_packed_tl2, y_tl2_avx2, bias, M, N, K, num_threads)
        times.append(time.perf_counter() - start)

    avg_time = np.mean(times) * 1000
    std_time = np.std(times) * 1000
    gflops = (2.0 * M * N * K / 1e9) / (avg_time / 1000)
    print(f"    Time: {avg_time:.2f} +/- {std_time:.2f} ms, GFLOPS: {gflops:.1f}")
    results['tl2_avx2'] = {'time': avg_time, 'gflops': gflops, 'y': y_tl2_avx2.clone()}

    # Test TL2 Gather
    print("\n  TL2 Gather:")
    y_tl2_gather = torch.zeros(M, N, dtype=torch.float32)

    for _ in range(5):
        tl2_kernel.matmul_free_tl2_gather(x, w_packed_tl2, y_tl2_gather, bias, M, N, K, num_threads)

    gc.collect()
    times = []
    for _ in range(30):
        start = time.perf_counter()
        tl2_kernel.matmul_free_tl2_gather(x, w_packed_tl2, y_tl2_gather, bias, M, N, K, num_threads)
        times.append(time.perf_counter() - start)

    avg_time = np.mean(times) * 1000
    std_time = np.std(times) * 1000
    gflops = (2.0 * M * N * K / 1e9) / (avg_time / 1000)
    print(f"    Time: {avg_time:.2f} +/- {std_time:.2f} ms, GFLOPS: {gflops:.1f}")
    results['tl2_gather'] = {'time': avg_time, 'gflops': gflops, 'y': y_tl2_gather.clone()}

    # Test TL2 Optimized
    print("\n  TL2 Optimized:")
    y_tl2_opt = torch.zeros(M, N, dtype=torch.float32)

    for _ in range(5):
        tl2_kernel.matmul_free_tl2_optimized(x, w_packed_tl2, y_tl2_opt, bias, M, N, K, num_threads)

    gc.collect()
    times = []
    for _ in range(30):
        start = time.perf_counter()
        tl2_kernel.matmul_free_tl2_optimized(x, w_packed_tl2, y_tl2_opt, bias, M, N, K, num_threads)
        times.append(time.perf_counter() - start)

    avg_time = np.mean(times) * 1000
    std_time = np.std(times) * 1000
    gflops = (2.0 * M * N * K / 1e9) / (avg_time / 1000)
    print(f"    Time: {avg_time:.2f} +/- {std_time:.2f} ms, GFLOPS: {gflops:.1f}")
    results['tl2_optimized'] = {'time': avg_time, 'gflops': gflops, 'y': y_tl2_opt.clone()}

    # Test T-MAC best (PSHUFB) for comparison
    if tmac_kernel is not None:
        print("\n  T-MAC PSHUFB (comparison):")
        y_tmac = torch.zeros(M, N, dtype=torch.float32)

        for _ in range(5):
            tmac_kernel.matmul_free_tmac_pshufb(x, sign_plane, value_plane, y_tmac, bias, M, N, K, num_threads)

        gc.collect()
        times = []
        for _ in range(30):
            start = time.perf_counter()
            tmac_kernel.matmul_free_tmac_pshufb(x, sign_plane, value_plane, y_tmac, bias, M, N, K, num_threads)
            times.append(time.perf_counter() - start)

        avg_time = np.mean(times) * 1000
        std_time = np.std(times) * 1000
        gflops = (2.0 * M * N * K / 1e9) / (avg_time / 1000)
        print(f"    Time: {avg_time:.2f} +/- {std_time:.2f} ms, GFLOPS: {gflops:.1f}")
        results['tmac_pshufb'] = {'time': avg_time, 'gflops': gflops, 'y': y_tmac.clone()}

    # Test AVX-512 packed for comparison
    if avx512_kernel is not None:
        print("\n  AVX-512 packed (baseline):")
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
    for key in ['tl2_scalar', 'tl2_avx2', 'tl2_gather', 'tl2_optimized']:
        if key in results:
            max_diff = torch.max(torch.abs(results[key]['y'] - y_pt)).item()
            status = 'OK' if max_diff < 1e-3 else 'MISMATCH!'
            print(f"    {key}: max_diff={max_diff:.6f} {status}")

    if 'tmac_pshufb' in results:
        max_diff = torch.max(torch.abs(results['tmac_pshufb']['y'] - y_pt)).item()
        status = 'OK' if max_diff < 1e-3 else 'MISMATCH!'
        print(f"    tmac_pshufb: max_diff={max_diff:.6f} {status}")

    if 'avx512_packed' in results:
        max_diff = torch.max(torch.abs(results['avx512_packed']['y'] - y_pt)).item()
        status = 'OK' if max_diff < 1e-3 else 'MISMATCH!'
        print(f"    avx512_packed: max_diff={max_diff:.6f} {status}")

    # Summary
    print(f"\n  Summary for {name}:")
    print(f"    {'Kernel':<25} {'GFLOPS':>10} {'vs AVX-512':>12} {'vs T-MAC':>10}")
    print(f"    {'-'*60}")

    avx512_gflops = results.get('avx512_packed', {}).get('gflops', 1.0)
    tmac_gflops = results.get('tmac_pshufb', {}).get('gflops', 1.0)

    for key, label in [
        ('tl2_scalar', 'TL2 scalar'),
        ('tl2_avx2', 'TL2 AVX2'),
        ('tl2_gather', 'TL2 Gather'),
        ('tl2_optimized', 'TL2 Optimized'),
        ('tmac_pshufb', 'T-MAC PSHUFB'),
        ('avx512_packed', 'AVX-512 packed'),
        ('pytorch', 'PyTorch'),
    ]:
        if key in results:
            gflops = results[key]['gflops']
            vs_avx512 = gflops / avx512_gflops if avx512_gflops > 0 else 0
            vs_tmac = gflops / tmac_gflops if tmac_gflops > 0 else 0
            print(f"    {label:<25} {gflops:>10.1f} {vs_avx512:>11.2f}x {vs_tmac:>9.2f}x")

print("\n" + "="*70)
print("TL2 Test Complete!")
print("="*70)
print("\nTL2 vs T-MAC Key Differences:")
print("  - TL2: Groups 3 weights, element-wise LUT (3^3=27 -> 14 with mirror)")
print("  - T-MAC: Groups 4 weights, bit-wise LUT (2^4=16)")
print("  - TL2 achieves 1.67 bpw vs 2.0 bpw for T-MAC")
print("  - Paper claims TL2 is 2.32x faster than T-MAC on x86!")
print("="*70)
