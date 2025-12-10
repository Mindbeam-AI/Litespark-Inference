#!/usr/bin/env python3
"""
Benchmark kernels for Microsoft BitNet b1.58-2B-4T specific dimensions.

BitNet dimensions:
- hidden_size: 2560
- intermediate_size: 6912
- num_heads: 20 (Q), num_kv_heads: 5 (KV)
- head_dim: 128
"""

import torch
import numpy as np
import time
import platform
from pathlib import Path
from torch.utils.cpp_extension import load

# ============================================================================
# Configuration
# ============================================================================

BATCH_SIZES = [1, 4, 8, 16, 32, 64, 128]
WARMUP_ITERS = 5
BENCH_ITERS = 20

# BitNet actual dimensions
HIDDEN_SIZE = 2560
INTERMEDIATE_SIZE = 6912
NUM_KV_HEADS = 5
HEAD_DIM = 128
KV_DIM = NUM_KV_HEADS * HEAD_DIM  # 640

# ============================================================================
# Architecture Check
# ============================================================================

machine = platform.machine().lower()
if machine not in ['x86_64', 'amd64']:
    raise RuntimeError(f"This script requires x86_64. Got: {machine}")

print("Loading VNNI kernels...")
kernel = load(
    name='matmul_free_vnni',
    sources=[str(Path(__file__).parent / 'src/cpu_ops/kernels/x86_64/matmul_free_tmac_vnni.cpp')],
    extra_cflags=['-O3', '-march=native', '-fopenmp', '-mavx512f', '-mavx512bw', '-mavx512vnni'],
    extra_ldflags=['-fopenmp'],
    verbose=False
)

num_threads = torch.get_num_threads()
print(f"Using {num_threads} threads\n")

# ============================================================================
# Weight Preparation
# ============================================================================

def prepare_ternary_weights(N, K):
    """Create ternary weights and precompute sums"""
    K_padded = ((K + 63) // 64) * 64
    w_ternary = torch.randint(-1, 2, (N, K), dtype=torch.int8)
    w_int8 = torch.zeros(N, K_padded, dtype=torch.int8)
    w_int8[:, :K] = w_ternary
    w_sum = w_int8.sum(dim=1, dtype=torch.int32)
    return w_int8.contiguous(), w_sum.contiguous()

def quantize_input(x):
    """Quantize float input to int8 with per-row scaling"""
    M, K = x.shape
    K_padded = ((K + 63) // 64) * 64

    max_abs = x.abs().max(dim=1, keepdim=True).values
    scale = max_abs / 127.0
    scale = torch.where(scale == 0, torch.ones_like(scale), scale)

    x_int8 = torch.zeros(M, K_padded, dtype=torch.int8)
    x_int8[:, :K] = (x / scale).round().clamp(-127, 127).to(torch.int8)

    return x_int8.contiguous(), scale.squeeze(1).contiguous()

# ============================================================================
# Benchmark Function
# ============================================================================

def benchmark_kernel(fn, warmup=WARMUP_ITERS, iters=BENCH_ITERS):
    """Benchmark a kernel function"""
    for _ in range(warmup):
        fn()

    torch.cuda.synchronize() if torch.cuda.is_available() else None

    times = []
    for _ in range(iters):
        start = time.perf_counter()
        fn()
        end = time.perf_counter()
        times.append((end - start) * 1000)

    return {
        'mean_ms': np.mean(times),
        'std_ms': np.std(times),
        'min_ms': np.min(times),
    }

def benchmark_pytorch(x, w_float):
    """Benchmark PyTorch F.linear for comparison"""
    def run():
        return torch.nn.functional.linear(x, w_float)
    return benchmark_kernel(run)

# ============================================================================
# Test Cases
# ============================================================================

def test_projection(M, K, N, label):
    """Test different kernel variants for a projection"""
    print(f"\n  {label} [M={M}, K={K}, N={N}]")

    x = torch.randn(M, K, dtype=torch.float32)
    x_int8, x_scale = quantize_input(x)
    w_int8, w_sum = prepare_ternary_weights(N, K)
    bias = torch.Tensor()

    # Also create float weights for PyTorch comparison
    w_float = torch.randn(N, K, dtype=torch.float32)

    results = {}

    # PyTorch baseline
    results['PyTorch F.linear'] = benchmark_pytorch(x, w_float)

    # v2 (small layers)
    if K <= 1024 and N <= 4096:
        y_out = torch.zeros(M, N, dtype=torch.float32)
        def run_v2():
            kernel.matmul_free_vnni_v2(x_int8, x_scale, w_int8, w_sum, y_out, bias, M, N, K, num_threads)
        results['v2'] = benchmark_kernel(run_v2)

    # v3 (baseline for larger)
    y_out = torch.zeros(M, N, dtype=torch.float32)
    def run_v3():
        kernel.matmul_free_vnni_v3(x_int8, x_scale, w_int8, w_sum, y_out, bias, M, N, K, num_threads)
    results['v3'] = benchmark_kernel(run_v3)

    # v4 (large N) - outputs int32
    if N >= 1024:
        y_out_i32 = torch.zeros(M, N, dtype=torch.int32)
        def run_v4():
            kernel.matmul_free_vnni_v4_large_n(x_int8, x_scale, w_int8, w_sum, y_out_i32, M, N, K, num_threads)
        results['v4_large_n'] = benchmark_kernel(run_v4)

    # v5 (large M) - outputs int32
    if M >= 32:
        y_out_i32 = torch.zeros(M, N, dtype=torch.int32)
        def run_v5():
            kernel.matmul_free_vnni_v5_large_m(x_int8, x_scale, w_int8, w_sum, y_out_i32, M, N, K, num_threads)
        results['v5_large_m'] = benchmark_kernel(run_v5)

    # Print results
    pytorch_time = results['PyTorch F.linear']['mean_ms']
    print(f"    {'Kernel':<20} {'Time (ms)':>12} {'Speedup':>10}")
    print(f"    {'-'*44}")

    for name, r in sorted(results.items(), key=lambda x: x[1]['mean_ms']):
        speedup = pytorch_time / r['mean_ms']
        marker = " <-- BEST" if r['mean_ms'] == min(v['mean_ms'] for v in results.values()) else ""
        print(f"    {name:<20} {r['mean_ms']:>8.3f} ms   {speedup:>6.2f}x{marker}")

    return results

# ============================================================================
# Main
# ============================================================================

def main():
    print("=" * 70)
    print("BitNet b1.58-2B-4T Kernel Benchmark")
    print("=" * 70)
    print(f"\nBitNet dimensions:")
    print(f"  hidden_size: {HIDDEN_SIZE}")
    print(f"  intermediate_size: {INTERMEDIATE_SIZE}")
    print(f"  kv_dim: {KV_DIM}")
    print(f"\nThreads: {num_threads}")
    print(f"Batch sizes: {BATCH_SIZES}")

    all_results = {}

    for M in BATCH_SIZES:
        print(f"\n{'='*70}")
        print(f"BATCH SIZE M={M}")
        print(f"{'='*70}")

        all_results[f"M{M}"] = {}

        # Q projection: 2560 -> 2560
        all_results[f"M{M}"]['q_proj'] = test_projection(
            M, HIDDEN_SIZE, HIDDEN_SIZE, "Q Projection"
        )

        # K/V projection: 2560 -> 640
        all_results[f"M{M}"]['kv_proj'] = test_projection(
            M, HIDDEN_SIZE, KV_DIM, "K/V Projection"
        )

        # O projection: 2560 -> 2560
        all_results[f"M{M}"]['o_proj'] = test_projection(
            M, HIDDEN_SIZE, HIDDEN_SIZE, "O Projection"
        )

        # Gate/Up projection: 2560 -> 6912
        all_results[f"M{M}"]['gate_up'] = test_projection(
            M, HIDDEN_SIZE, INTERMEDIATE_SIZE, "Gate/Up Projection (MLP)"
        )

        # Down projection: 6912 -> 2560
        all_results[f"M{M}"]['down'] = test_projection(
            M, INTERMEDIATE_SIZE, HIDDEN_SIZE, "Down Projection (MLP)"
        )

    # Summary
    print("\n")
    print("=" * 70)
    print("SUMMARY: Best kernel for each operation")
    print("=" * 70)

    ops = ['q_proj', 'kv_proj', 'o_proj', 'gate_up', 'down']
    op_labels = {
        'q_proj': 'Q proj (2560->2560)',
        'kv_proj': 'K/V proj (2560->640)',
        'o_proj': 'O proj (2560->2560)',
        'gate_up': 'Gate/Up (2560->6912)',
        'down': 'Down (6912->2560)',
    }

    print(f"\n{'M':<6}", end='')
    for op in ops:
        print(f"{op_labels[op]:<22}", end='')
    print()
    print("-" * 120)

    for M in BATCH_SIZES:
        print(f"{M:<6}", end='')
        for op in ops:
            results = all_results[f"M{M}"][op]
            # Find best kernel (excluding PyTorch)
            our_kernels = {k: v for k, v in results.items() if k != 'PyTorch F.linear'}
            best = min(our_kernels.items(), key=lambda x: x[1]['mean_ms'])
            pytorch_time = results['PyTorch F.linear']['mean_ms']
            speedup = pytorch_time / best[1]['mean_ms']
            print(f"{best[0]:<12}({speedup:.2f}x)   ", end='')
        print()

    # Check if we beat PyTorch anywhere
    print("\n")
    print("=" * 70)
    print("COMPARISON VS PYTORCH F.linear")
    print("=" * 70)

    wins = 0
    losses = 0

    for M in BATCH_SIZES:
        for op in ops:
            results = all_results[f"M{M}"][op]
            our_kernels = {k: v for k, v in results.items() if k != 'PyTorch F.linear'}
            best = min(our_kernels.items(), key=lambda x: x[1]['mean_ms'])
            pytorch_time = results['PyTorch F.linear']['mean_ms']

            if best[1]['mean_ms'] < pytorch_time:
                wins += 1
            else:
                losses += 1

    total = wins + losses
    print(f"\nOur kernels beat PyTorch: {wins}/{total} ({100*wins/total:.1f}%)")
    print(f"PyTorch wins: {losses}/{total} ({100*losses/total:.1f}%)")

    print("\n" + "=" * 70)
    print("Benchmark Complete!")
    print("=" * 70)

if __name__ == '__main__':
    main()
