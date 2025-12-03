"""
Test T-MAC / TL2 Kernels for x86_64 (Microsoft approach)

This script benchmarks:
1. T-MAC float32 (LUT-based, K-first iteration)
2. TL2 float32 (element-wise LUT)
3. T-MAC int8 with PSHUFB (CORRECT Microsoft approach)
4. TL2 int8 with PSHUFB (CORRECT Microsoft approach)
5. T-MAC int8 AVX-512 with PSHUFB (64 parallel lookups)
6. T-MAC int8 AVX-512 v2 (optimized lane handling)
7. T-MAC int8 AVX-512 tiled (cache-aware)
8. VNNI simple (direct dot product)
9. VNNI v2 (register blocking)
10. VNNI v3 (M+N tiling for cache optimization on large matrices)
11. AVX-512 packed (direct computation baseline)

Key insight from Microsoft:
- Int8 activations + int8 LUT + PSHUFB = 32 parallel lookups per instruction (AVX2)
- AVX-512 PSHUFB = 64 parallel lookups per instruction
- VNNI dpbusd = 64 int8 multiplies per instruction
- Float32 LUT = scalar lookups (slow!)
"""
import torch
import time
import numpy as np
import platform
import psutil
import os
import gc
import json
from pathlib import Path
from torch.utils.cpp_extension import load
import matplotlib.pyplot as plt
from datetime import datetime


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

# Compile T-MAC int8 AVX-512 kernel
print("Compiling T-MAC int8 AVX-512 kernel...", end=' ', flush=True)
try:
    tmac_int8_avx512_kernel = load(
        name="tmac_int8_avx512_for_main_test",
        sources=[str(KERNELS_DIR / "x86_64" / "matmul_free_tmac_int8_avx512.cpp")],
        extra_cflags=['-O3', '-mavx512f', '-mavx512bw', '-mavx512dq', '-mavx512vl',
                      '-fopenmp', '-std=c++17', '-march=native'],
        extra_ldflags=['-fopenmp', '-lgomp'],
        verbose=False
    )
    print("OK")
except Exception as e:
    print(f"FAILED: {e}")
    tmac_int8_avx512_kernel = None

# Compile T-MAC int8 AVX-512 v2 kernel (optimized)
print("Compiling T-MAC int8 AVX-512 v2 kernel...", end=' ', flush=True)
try:
    tmac_int8_avx512_v2_kernel = load(
        name="tmac_int8_avx512_v2_for_main_test",
        sources=[str(KERNELS_DIR / "x86_64" / "matmul_free_tmac_int8_avx512_v2.cpp")],
        extra_cflags=['-O3', '-mavx512f', '-mavx512bw', '-mavx512dq', '-mavx512vl',
                      '-fopenmp', '-std=c++17', '-march=native'],
        extra_ldflags=['-fopenmp', '-lgomp'],
        verbose=False
    )
    print("OK")
except Exception as e:
    print(f"FAILED: {e}")
    tmac_int8_avx512_v2_kernel = None

# Compile VNNI kernel
print("Compiling VNNI kernel...", end=' ', flush=True)
try:
    vnni_kernel = load(
        name="vnni_for_main_test",
        sources=[str(KERNELS_DIR / "x86_64" / "matmul_free_tmac_vnni.cpp")],
        extra_cflags=['-O3', '-mavx512f', '-mavx512bw', '-mavx512dq', '-mavx512vl',
                      '-mavx512vnni', '-fopenmp', '-std=c++17', '-march=native'],
        extra_ldflags=['-fopenmp', '-lgomp'],
        verbose=False
    )
    print("OK")
except Exception as e:
    print(f"FAILED: {e}")
    vnni_kernel = None

# Test configurations
DEFAULT_CONFIGS = [
    ("Small", 128, 1024, 1024),
    ("Medium", 256, 1024, 1024),
    ("Large", 512, 1024, 1024),
    ("XLarge", 1024, 2048, 2048),
]

# Transformer layer configurations (realistic LLM shapes)
# Format: (name, M, N, K) where M=batch*seq, weights are [K, N]
TRANSFORMER_CONFIGS = [
    # QKV Projection: [2048, 2560] - Q+K+V fused
    ("QKV_M1",     1,   2560,  2048),   # Single token (autoregressive)
    ("QKV_M32",    32,  2560,  2048),   # Small batch
    ("QKV_M128",   128, 2560,  2048),   # Medium batch

    # Output Projection: [2048, 2048]
    ("Out_M1",     1,   2048,  2048),
    ("Out_M32",    32,  2048,  2048),
    ("Out_M128",   128, 2048,  2048),

    # MLP Gate-Up Projection: [2048, 16384] - Large N
    ("MLP_Up_M1",  1,   16384, 2048),
    ("MLP_Up_M32", 32,  16384, 2048),

    # MLP Down Projection: [8192, 2048] - Large K
    ("MLP_Down_M1",  1,   2048, 8192),
    ("MLP_Down_M32", 32,  2048, 8192),

    # LM Head: [2048, 32064] - Vocabulary size
    ("LMHead_M1",  1,   32064, 2048),
    ("LMHead_M32", 32,  32064, 2048),
]

# Select configs based on command line arg
import sys
import argparse
parser = argparse.ArgumentParser()
parser.add_argument('--transformer', action='store_true', help='Use transformer layer shapes')
parser.add_argument('--output-json', type=str, help='Save results to JSON file')
args, _ = parser.parse_known_args()

if args.transformer:
    configs = TRANSFORMER_CONFIGS
    print("Using TRANSFORMER layer configurations")
else:
    configs = DEFAULT_CONFIGS
    print("Using DEFAULT test configurations (use --transformer for LLM layer shapes)")

num_threads = 8

print(f"\nTesting with {num_threads} threads")
print("="*70)

# Store all results across configs for final summary
all_results = {}

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

    # Pack weights for VNNI kernel
    if vnni_kernel is not None:
        K_padded = ((K + 63) // 64) * 64
        w_int8_vnni = torch.zeros(N, K_padded, dtype=torch.int8)
        w_sum_vnni = torch.zeros(N, dtype=torch.int32)
        vnni_kernel.pack_weights_vnni(w_ternary, w_int8_vnni, w_sum_vnni, N, K)

        # Quantize activations for VNNI (reuse existing if available)
        if tmac_int8_kernel is None:
            x_int8 = torch.zeros(M, K, dtype=torch.int8)
            scales = torch.zeros(M, dtype=torch.float32)
            vnni_kernel.quantize_activations_int8_vnni(x, x_int8, scales, M, K)

    bias = torch.zeros(N, dtype=torch.float32)
    results = {}

    # Store config info
    config_info = {'M': M, 'N': N, 'K': K, 'flops': 2.0 * M * N * K}

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

    # Test T-MAC int8 AVX-512 PSHUFB (64 parallel lookups)
    if tmac_int8_avx512_kernel is not None and tmac_int8_kernel is not None:
        print("\n  T-MAC int8 AVX-512 PSHUFB:")
        y_tmac_int8_avx512 = torch.zeros(M, N, dtype=torch.float32)

        for _ in range(5):
            tmac_int8_avx512_kernel.matmul_free_tmac_int8_avx512(
                x_int8, scales, sign_plane_int8, value_plane_int8,
                y_tmac_int8_avx512, bias, M, N, K, num_threads
            )

        gc.collect()
        times = []
        for _ in range(30):
            start = time.perf_counter()
            tmac_int8_avx512_kernel.matmul_free_tmac_int8_avx512(
                x_int8, scales, sign_plane_int8, value_plane_int8,
                y_tmac_int8_avx512, bias, M, N, K, num_threads
            )
            times.append(time.perf_counter() - start)

        avg_time = np.mean(times) * 1000
        std_time = np.std(times) * 1000
        gflops = (2.0 * M * N * K / 1e9) / (avg_time / 1000)
        print(f"    Time: {avg_time:.2f} +/- {std_time:.2f} ms, GFLOPS: {gflops:.1f}")
        results['tmac_int8_avx512'] = {'time': avg_time, 'gflops': gflops, 'y': y_tmac_int8_avx512.clone()}

    # Test T-MAC int8 AVX-512 v2 (optimized lane handling)
    if tmac_int8_avx512_v2_kernel is not None and tmac_int8_kernel is not None:
        print("\n  T-MAC int8 AVX-512 v2:")
        y_avx512_v2 = torch.zeros(M, N, dtype=torch.float32)

        for _ in range(5):
            tmac_int8_avx512_v2_kernel.matmul_free_tmac_int8_avx512_v2(
                x_int8, scales, sign_plane_int8, value_plane_int8,
                y_avx512_v2, bias, M, N, K, num_threads
            )

        gc.collect()
        times = []
        for _ in range(30):
            start = time.perf_counter()
            tmac_int8_avx512_v2_kernel.matmul_free_tmac_int8_avx512_v2(
                x_int8, scales, sign_plane_int8, value_plane_int8,
                y_avx512_v2, bias, M, N, K, num_threads
            )
            times.append(time.perf_counter() - start)

        avg_time = np.mean(times) * 1000
        std_time = np.std(times) * 1000
        gflops = (2.0 * M * N * K / 1e9) / (avg_time / 1000)
        print(f"    Time: {avg_time:.2f} +/- {std_time:.2f} ms, GFLOPS: {gflops:.1f}")
        results['tmac_int8_avx512_v2'] = {'time': avg_time, 'gflops': gflops, 'y': y_avx512_v2.clone()}

        # Test tiled version
        print("\n  T-MAC int8 AVX-512 tiled:")
        y_avx512_tiled = torch.zeros(M, N, dtype=torch.float32)

        for _ in range(5):
            tmac_int8_avx512_v2_kernel.matmul_free_tmac_int8_avx512_tiled(
                x_int8, scales, sign_plane_int8, value_plane_int8,
                y_avx512_tiled, bias, M, N, K, num_threads
            )

        gc.collect()
        times = []
        for _ in range(30):
            start = time.perf_counter()
            tmac_int8_avx512_v2_kernel.matmul_free_tmac_int8_avx512_tiled(
                x_int8, scales, sign_plane_int8, value_plane_int8,
                y_avx512_tiled, bias, M, N, K, num_threads
            )
            times.append(time.perf_counter() - start)

        avg_time = np.mean(times) * 1000
        std_time = np.std(times) * 1000
        gflops = (2.0 * M * N * K / 1e9) / (avg_time / 1000)
        print(f"    Time: {avg_time:.2f} +/- {std_time:.2f} ms, GFLOPS: {gflops:.1f}")
        results['tmac_int8_avx512_tiled'] = {'time': avg_time, 'gflops': gflops, 'y': y_avx512_tiled.clone()}

    # Test VNNI kernel
    if vnni_kernel is not None:
        print("\n  VNNI simple:")
        y_vnni = torch.zeros(M, N, dtype=torch.float32)

        for _ in range(5):
            vnni_kernel.matmul_free_vnni_simple(
                x_int8, scales, w_int8_vnni, w_sum_vnni,
                y_vnni, bias, M, N, K, num_threads
            )

        gc.collect()
        times = []
        for _ in range(30):
            start = time.perf_counter()
            vnni_kernel.matmul_free_vnni_simple(
                x_int8, scales, w_int8_vnni, w_sum_vnni,
                y_vnni, bias, M, N, K, num_threads
            )
            times.append(time.perf_counter() - start)

        avg_time = np.mean(times) * 1000
        std_time = np.std(times) * 1000
        gflops = (2.0 * M * N * K / 1e9) / (avg_time / 1000)
        print(f"    Time: {avg_time:.2f} +/- {std_time:.2f} ms, GFLOPS: {gflops:.1f}")
        results['vnni'] = {'time': avg_time, 'gflops': gflops, 'y': y_vnni.clone()}

        # Test VNNI v2 (register blocking)
        print("\n  VNNI v2 (register blocking):")
        y_vnni_v2 = torch.zeros(M, N, dtype=torch.float32)

        for _ in range(5):
            vnni_kernel.matmul_free_vnni_v2(
                x_int8, scales, w_int8_vnni, w_sum_vnni,
                y_vnni_v2, bias, M, N, K, num_threads
            )

        gc.collect()
        times = []
        for _ in range(30):
            start = time.perf_counter()
            vnni_kernel.matmul_free_vnni_v2(
                x_int8, scales, w_int8_vnni, w_sum_vnni,
                y_vnni_v2, bias, M, N, K, num_threads
            )
            times.append(time.perf_counter() - start)

        avg_time = np.mean(times) * 1000
        std_time = np.std(times) * 1000
        gflops = (2.0 * M * N * K / 1e9) / (avg_time / 1000)
        print(f"    Time: {avg_time:.2f} +/- {std_time:.2f} ms, GFLOPS: {gflops:.1f}")
        results['vnni_v2'] = {'time': avg_time, 'gflops': gflops, 'y': y_vnni_v2.clone()}

        # Test VNNI v3 (cache-optimized tiled)
        print("\n  VNNI v3 (tiled):")
        y_vnni_v3 = torch.zeros(M, N, dtype=torch.float32)

        for _ in range(5):
            vnni_kernel.matmul_free_vnni_v3(
                x_int8, scales, w_int8_vnni, w_sum_vnni,
                y_vnni_v3, bias, M, N, K, num_threads
            )

        gc.collect()
        times = []
        for _ in range(30):
            start = time.perf_counter()
            vnni_kernel.matmul_free_vnni_v3(
                x_int8, scales, w_int8_vnni, w_sum_vnni,
                y_vnni_v3, bias, M, N, K, num_threads
            )
            times.append(time.perf_counter() - start)

        avg_time = np.mean(times) * 1000
        std_time = np.std(times) * 1000
        gflops = (2.0 * M * N * K / 1e9) / (avg_time / 1000)
        print(f"    Time: {avg_time:.2f} +/- {std_time:.2f} ms, GFLOPS: {gflops:.1f}")
        results['vnni_v3'] = {'time': avg_time, 'gflops': gflops, 'y': y_vnni_v3.clone()}

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
    for key in ['tmac_int8', 'tl2_int8', 'tmac_int8_avx512', 'tmac_int8_avx512_v2', 'tmac_int8_avx512_tiled', 'vnni', 'vnni_v2', 'vnni_v3']:
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
        ('tmac_int8_avx512', 'T-MAC int8 AVX-512'),
        ('tmac_int8_avx512_v2', 'T-MAC int8 AVX-512 v2'),
        ('tmac_int8_avx512_tiled', 'T-MAC int8 AVX-512 tiled'),
        ('vnni', 'VNNI simple'),
        ('vnni_v2', 'VNNI v2 (reg block)'),
        ('vnni_v3', 'VNNI v3 (tiled)'),
        ('tl2_int8', 'TL2 int8 PSHUFB'),
        ('avx512_packed', 'AVX-512 packed'),
        ('pytorch', 'PyTorch'),
    ]:
        if key in results:
            gflops = results[key]['gflops']
            ratio = gflops / avx512_gflops if avx512_gflops > 0 else 0
            print(f"    {label:<30} {gflops:>10.1f} {ratio:>11.2f}x")

    # Store results for this config (excluding large tensor data)
    all_results[name] = {
        'config': config_info,
        'kernels': {k: {'time': v['time'], 'gflops': v['gflops']} for k, v in results.items()}
    }

print("\n" + "="*70)
print("T-MAC / TL2 Test Complete!")
print("="*70)

# =============================================================================
# FINAL SUMMARY TABLE
# =============================================================================
print("\n")
print("="*100)
print("FINAL SUMMARY TABLE")
print("="*100)

# Define kernel display order and labels
kernel_order = [
    ('tmac_int8', 'T-MAC int8 PSHUFB'),
    ('tmac_int8_avx512_v2', 'T-MAC int8 AVX-512 v2'),
    ('vnni', 'VNNI simple'),
    ('vnni_v2', 'VNNI v2 (reg block)'),
    ('vnni_v3', 'VNNI v3 (tiled)'),
    ('pytorch', 'PyTorch (MKL)'),
]

# Print header
print(f"\n{'Kernel':<25}", end='')
for cfg_name, M, N, K in configs:
    print(f" | {cfg_name:^20}", end='')
print()
print("-"*25, end='')
for _ in configs:
    print("-"*23, end='')
print()

# Print each kernel row
for key, label in kernel_order:
    print(f"{label:<25}", end='')
    for cfg_name, M, N, K in configs:
        if cfg_name in all_results and key in all_results[cfg_name]['kernels']:
            data = all_results[cfg_name]['kernels'][key]
            gflops = data['gflops']
            time_ms = data['time']
            print(f" | {gflops:>6.0f} GFLOPS {time_ms:>5.2f}ms", end='')
        else:
            print(f" | {'N/A':^20}", end='')
    print()

# Print speedup vs PyTorch
print()
print(f"{'Speedup vs PyTorch':<25}", end='')
for cfg_name, M, N, K in configs:
    print(f" | {cfg_name:^20}", end='')
print()
print("-"*25, end='')
for _ in configs:
    print("-"*23, end='')
print()

for key, label in kernel_order:
    if key == 'pytorch':
        continue
    print(f"{label:<25}", end='')
    for cfg_name, M, N, K in configs:
        if cfg_name in all_results:
            kernels = all_results[cfg_name]['kernels']
            if key in kernels and 'pytorch' in kernels:
                speedup = kernels[key]['gflops'] / kernels['pytorch']['gflops']
                print(f" | {speedup:>10.2f}x         ", end='')
            else:
                print(f" | {'N/A':^20}", end='')
        else:
            print(f" | {'N/A':^20}", end='')
    print()

# =============================================================================
# GENERATE PLOTS
# =============================================================================
print("\n" + "="*70)
print("Generating plots...")
print("="*70)

# Create output directory for plots
plots_dir = Path(__file__).parent / "benchmark_plots"
plots_dir.mkdir(exist_ok=True)

# Color scheme for kernels
colors = {
    'tmac_int8': '#1f77b4',
    'tmac_int8_avx512_v2': '#ff7f0e',
    'vnni': '#2ca02c',
    'vnni_v2': '#d62728',
    'vnni_v3': '#9467bd',
    'pytorch': '#8c564b',
}

# Group configs by M value
from collections import defaultdict
configs_by_m = defaultdict(list)
for cfg in configs:
    cfg_name, M, N, K = cfg
    configs_by_m[M].append(cfg)

# Generate separate plots for each M value
for M_val, m_configs in configs_by_m.items():
    print(f"\n  Generating plots for M={M_val}...")

    # Plot 1: GFLOPS comparison for this M value
    n_configs = len(m_configs)
    fig, axes = plt.subplots(1, n_configs, figsize=(5 * n_configs, 6), sharey=True)
    if n_configs == 1:
        axes = [axes]
    fig.suptitle(f'GFLOPS Performance (M={M_val})', fontsize=14, fontweight='bold')

    for idx, (cfg_name, M, N, K) in enumerate(m_configs):
        ax = axes[idx]
        if cfg_name not in all_results:
            continue

        kernels_data = all_results[cfg_name]['kernels']
        kernel_names = []
        gflops_values = []
        bar_colors = []

        for key, label in kernel_order:
            if key in kernels_data:
                kernel_names.append(label)
                gflops_values.append(kernels_data[key]['gflops'])
                bar_colors.append(colors.get(key, '#333333'))

        bars = ax.bar(range(len(kernel_names)), gflops_values, color=bar_colors)
        ax.set_xticks(range(len(kernel_names)))
        ax.set_xticklabels(kernel_names, fontsize=9, rotation=45, ha='right')
        ax.set_title(f"{cfg_name}\n[K={K}, N={N}]", fontsize=11)
        ax.set_ylabel('GFLOPS' if idx == 0 else '')
        ax.grid(axis='y', alpha=0.3)

        # Add value labels on bars
        for bar, val in zip(bars, gflops_values):
            ax.text(bar.get_x() + bar.get_width()/2, bar.get_height(),
                    f'{val:.0f}', ha='center', va='bottom', fontsize=8)

    plt.tight_layout()
    plot_path = plots_dir / f'gflops_M{M_val}.png'
    plt.savefig(plot_path, dpi=150, bbox_inches='tight')
    print(f"    Saved: {plot_path}")
    plt.close()

    # Plot 2: Speedup vs PyTorch for this M value
    fig, ax = plt.subplots(figsize=(max(10, 3 * n_configs), 6))
    fig.suptitle(f'Speedup vs PyTorch (M={M_val})', fontsize=14, fontweight='bold')

    config_names = [cfg[0] for cfg in m_configs]
    x = np.arange(len(config_names))
    speedup_kernels = [(k, l) for k, l in kernel_order if k != 'pytorch']
    width = 0.8 / len(speedup_kernels)

    for i, (key, label) in enumerate(speedup_kernels):
        speedups = []
        for cfg_name, M, N, K in m_configs:
            if cfg_name in all_results:
                kernels = all_results[cfg_name]['kernels']
                if key in kernels and 'pytorch' in kernels:
                    speedups.append(kernels[key]['gflops'] / kernels['pytorch']['gflops'])
                else:
                    speedups.append(0)
            else:
                speedups.append(0)

        bars = ax.bar(x + (i - len(speedup_kernels)/2 + 0.5) * width, speedups,
                      width, label=label, color=colors.get(key, '#333333'))

    ax.axhline(y=1.0, color='black', linestyle='--', linewidth=1, label='PyTorch baseline')
    ax.set_xlabel('Configuration')
    ax.set_ylabel('Speedup (higher is better)')
    ax.set_xticks(x)
    ax.set_xticklabels([f"{cfg[0]}\n[K={cfg[3]}, N={cfg[2]}]" for cfg in m_configs])
    ax.legend(loc='upper right', fontsize=8)
    ax.grid(axis='y', alpha=0.3)

    plt.tight_layout()
    plot_path = plots_dir / f'speedup_M{M_val}.png'
    plt.savefig(plot_path, dpi=150, bbox_inches='tight')
    print(f"    Saved: {plot_path}")
    plt.close()

    # Plot 3: Time comparison for this M value
    fig, axes = plt.subplots(1, n_configs, figsize=(5 * n_configs, 6), sharey=False)
    if n_configs == 1:
        axes = [axes]
    fig.suptitle(f'Execution Time (M={M_val}) - Lower is Better', fontsize=14, fontweight='bold')

    for idx, (cfg_name, M, N, K) in enumerate(m_configs):
        ax = axes[idx]
        if cfg_name not in all_results:
            continue

        kernels_data = all_results[cfg_name]['kernels']
        kernel_names = []
        time_values = []
        bar_colors = []

        for key, label in kernel_order:
            if key in kernels_data:
                kernel_names.append(label)
                time_values.append(kernels_data[key]['time'])
                bar_colors.append(colors.get(key, '#333333'))

        bars = ax.bar(range(len(kernel_names)), time_values, color=bar_colors)
        ax.set_xticks(range(len(kernel_names)))
        ax.set_xticklabels(kernel_names, fontsize=9, rotation=45, ha='right')
        ax.set_title(f"{cfg_name}\n[K={K}, N={N}]", fontsize=11)
        ax.set_ylabel('Time (ms)' if idx == 0 else '')
        ax.grid(axis='y', alpha=0.3)

        # Add value labels on bars
        for bar, val in zip(bars, time_values):
            ax.text(bar.get_x() + bar.get_width()/2, bar.get_height(),
                    f'{val:.2f}', ha='center', va='bottom', fontsize=8)

    plt.tight_layout()
    plot_path = plots_dir / f'time_M{M_val}.png'
    plt.savefig(plot_path, dpi=150, bbox_inches='tight')
    print(f"    Saved: {plot_path}")
    plt.close()

# Plot 4: Overall scaling across all matrix sizes (line plot)
fig, ax = plt.subplots(figsize=(max(12, len(configs) * 0.8), 7))
fig.suptitle('GFLOPS Scaling Across All Configurations', fontsize=14, fontweight='bold')

for key, label in kernel_order:
    gflops_list = []
    for cfg_name, M, N, K in configs:
        if cfg_name in all_results and key in all_results[cfg_name]['kernels']:
            gflops_list.append(all_results[cfg_name]['kernels'][key]['gflops'])
        else:
            gflops_list.append(np.nan)

    ax.plot(range(len(configs)), gflops_list, 'o-', label=label,
            color=colors.get(key, '#333333'), linewidth=2, markersize=6)

ax.set_xticks(range(len(configs)))
ax.set_xticklabels([f"{cfg[0]}" for cfg in configs], rotation=45, ha='right', fontsize=9)
ax.set_xlabel('Configuration')
ax.set_ylabel('GFLOPS')
ax.legend(loc='best', fontsize=9)
ax.grid(True, alpha=0.3)

plt.tight_layout()
plot_path = plots_dir / 'gflops_scaling.png'
plt.savefig(plot_path, dpi=150, bbox_inches='tight')
print(f"  Saved: {plot_path}")
plt.close()

print(f"\nAll plots saved to: {plots_dir}")

# =============================================================================
# SAVE JSON RESULTS
# =============================================================================
if args.output_json:
    json_output = {
        'metadata': {
            'timestamp': datetime.now().isoformat(),
            'platform': platform.system(),
            'architecture': arch,
            'num_threads': num_threads,
            'config_type': 'transformer' if args.transformer else 'default'
        },
        'results': all_results
    }

    json_path = Path(args.output_json)
    with open(json_path, 'w') as f:
        json.dump(json_output, f, indent=2)
    print(f"\nResults saved to: {json_path}")

print("\n" + "="*70)
print("Benchmark Complete!")
print("="*70)
