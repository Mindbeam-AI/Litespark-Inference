"""
Test T-MAC Kernel for x86_64 (Microsoft bit-plane decomposition)

This script benchmarks the T-MAC (Table-based Matrix-vector product Acceleration)
approach on x86_64 systems using the actual Microsoft BitNet algorithm.

T-MAC key idea (from Microsoft):
- Decompose ternary weights into 2 bit-planes (sign + value)
- Group 4 weights → 4 bits → 16-entry LUT (fits in 128-bit register)
- Use PSHUFB for 32 parallel lookups per AVX2 instruction
- result = 2 * value_lookup - sign_lookup
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
print("T-MAC MatMul-Free Kernel Test (x86_64) - Microsoft Bit-Plane")
print("="*70)
print(f"Architecture: {arch}")
print(f"System: {platform.system()}")

# Compile T-MAC kernel
print("\nCompiling T-MAC kernel...", end=' ', flush=True)
try:
    tmac_kernel = load(
        name="tmac_x86_bitplane_kernel",
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

# Compile AVX-512 packed kernel for comparison
print("Compiling AVX-512 packed kernel...", end=' ', flush=True)
try:
    avx512_kernel = load(
        name="avx512_packed_for_tmac_comparison2",
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

    # Pack weights for T-MAC bit-plane format [N, K_packed] where K_packed = ceil(K/8)
    K_packed = (K + 7) // 8
    sign_plane = torch.zeros(N, K_packed, dtype=torch.uint8)
    value_plane = torch.zeros(N, K_packed, dtype=torch.uint8)
    tmac_kernel.pack_ternary_bitplanes(w_ternary, sign_plane, value_plane, N, K)

    # Pack weights for T-MAC transposed format [K_groups, N] where K_groups = ceil(K/4)
    K_groups = (K + 3) // 4
    sign_plane_t = torch.zeros(K_groups, N, dtype=torch.uint8)
    value_plane_t = torch.zeros(K_groups, N, dtype=torch.uint8)
    tmac_kernel.pack_ternary_bitplanes_transposed(w_ternary, sign_plane_t, value_plane_t, N, K)

    # Pack weights for AVX-512 (2-bit, 16 weights per uint32)
    if avx512_kernel is not None:
        K_packed_avx512 = (K + 15) // 16
        w_packed_avx512 = torch.zeros(N, K_packed_avx512 * 4, dtype=torch.uint8)
        avx512_kernel.pack_weights_2bit_avx512(w_ternary, w_packed_avx512, N, K)

    bias = torch.zeros(N, dtype=torch.float32)
    results = {}

    # Weight memory comparison
    float_mem = N * K * 4 / (1024 * 1024)
    # Bit-plane: 2 planes, each K_packed bytes per row
    bitplane_mem = N * K_packed * 2 / (1024 * 1024)
    print(f"\n  Weight memory: {float_mem:.2f}MB (float32) -> {bitplane_mem:.2f}MB (bit-planes) = {float_mem/bitplane_mem:.1f}x reduction")

    # Test T-MAC scalar (reference)
    print("\n  T-MAC scalar (reference):")
    y_tmac_scalar = torch.zeros(M, N, dtype=torch.float32)

    for _ in range(5):  # Warmup
        tmac_kernel.matmul_free_tmac(x, sign_plane, value_plane, y_tmac_scalar, bias, M, N, K, num_threads)

    gc.collect()
    times = []
    for _ in range(30):
        start = time.perf_counter()
        tmac_kernel.matmul_free_tmac(x, sign_plane, value_plane, y_tmac_scalar, bias, M, N, K, num_threads)
        times.append(time.perf_counter() - start)

    avg_time = np.mean(times) * 1000
    std_time = np.std(times) * 1000
    gflops = (2.0 * M * N * K / 1e9) / (avg_time / 1000)
    print(f"    Time: {avg_time:.2f} +/- {std_time:.2f} ms, GFLOPS: {gflops:.1f}")
    results['tmac_scalar'] = {'time': avg_time, 'gflops': gflops, 'y': y_tmac_scalar.clone()}

    # Test T-MAC AVX2
    print("\n  T-MAC AVX2 (float LUT):")
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

    # Test T-MAC PSHUFB (true parallel lookups)
    print("\n  T-MAC PSHUFB (32 parallel int8 lookups):")
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

    # Test T-MAC Transposed (best memory access + PSHUFB)
    print("\n  T-MAC Transposed (sequential access + PSHUFB):")
    y_tmac_trans = torch.zeros(M, N, dtype=torch.float32)

    for _ in range(5):
        tmac_kernel.matmul_free_tmac_transposed(x, sign_plane_t, value_plane_t, y_tmac_trans, bias, M, N, K, num_threads)

    gc.collect()
    times = []
    for _ in range(30):
        start = time.perf_counter()
        tmac_kernel.matmul_free_tmac_transposed(x, sign_plane_t, value_plane_t, y_tmac_trans, bias, M, N, K, num_threads)
        times.append(time.perf_counter() - start)

    avg_time = np.mean(times) * 1000
    std_time = np.std(times) * 1000
    gflops = (2.0 * M * N * K / 1e9) / (avg_time / 1000)
    print(f"    Time: {avg_time:.2f} +/- {std_time:.2f} ms, GFLOPS: {gflops:.1f}")
    results['tmac_transposed'] = {'time': avg_time, 'gflops': gflops, 'y': y_tmac_trans.clone()}

    # Test AVX-512 packed v1 for comparison
    if avx512_kernel is not None:
        print("\n  AVX-512 packed v1 (PEXT + masked adds):")
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
    for key in ['tmac_scalar', 'tmac_avx2', 'tmac_pshufb', 'tmac_transposed']:
        if key in results:
            max_diff = torch.max(torch.abs(results[key]['y'] - y_pt)).item()
            # Use slightly higher tolerance for int8 quantized versions
            tol = 0.1 if 'pshufb' in key or 'transposed' in key else 1e-3
            status = 'OK' if max_diff < tol else 'MISMATCH!'
            print(f"    {key}: max_diff={max_diff:.6f} {status}")

    if 'avx512_packed' in results:
        max_diff = torch.max(torch.abs(results['avx512_packed']['y'] - y_pt)).item()
        status = 'OK' if max_diff < 1e-3 else 'MISMATCH!'
        print(f"    avx512_packed: max_diff={max_diff:.6f} {status}")

    # Summary
    print(f"\n  Summary for {name}:")
    print(f"    {'Kernel':<30} {'GFLOPS':>10} {'vs AVX-512':>12}")
    print(f"    {'-'*55}")

    avx512_gflops = results.get('avx512_packed', {}).get('gflops', 1.0)
    for key, label in [
        ('tmac_scalar', 'T-MAC scalar'),
        ('tmac_avx2', 'T-MAC AVX2'),
        ('tmac_pshufb', 'T-MAC PSHUFB'),
        ('tmac_transposed', 'T-MAC Transposed'),
        ('avx512_packed', 'AVX-512 packed'),
        ('pytorch', 'PyTorch'),
    ]:
        if key in results:
            gflops = results[key]['gflops']
            ratio = gflops / avx512_gflops if avx512_gflops > 0 else 0
            print(f"    {label:<30} {gflops:>10.1f} {ratio:>11.2f}x")

print("\n" + "="*70)
print("T-MAC Test Complete!")
print("="*70)
print("\nMicrosoft T-MAC Algorithm:")
print("  - Bit-plane decomposition: sign + value planes")
print("  - 16-entry LUT fits in 128-bit register")
print("  - PSHUFB enables 32 parallel lookups per AVX2 instruction")
print("  - result = 2 * value_lookup - sign_lookup")
print("="*70)
