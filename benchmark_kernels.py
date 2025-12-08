#!/usr/bin/env python3
"""
Kernel Selection Benchmark

Tests all available kernel variants for each operation and recommends
the best one based on speed and memory usage.
"""

import torch
import numpy as np
import time
import gc
import tracemalloc
import platform
from pathlib import Path
from torch.utils.cpp_extension import load

# ============================================================================
# Configuration
# ============================================================================

BATCH_SIZES = [1, 32, 128]
WARMUP_ITERS = 3
BENCH_ITERS = 10

# Model dimensions
HIDDEN_DIM = 2048
MLP_HIDDEN = 16384
MLP_INTERMEDIATE = 8192

# ============================================================================
# Memory Utilities
# ============================================================================

def get_process_memory_mb():
    """Get current process RSS in MB"""
    try:
        with open('/proc/self/status', 'r') as f:
            for line in f:
                if line.startswith('VmRSS:'):
                    return int(line.split()[1]) / 1024.0
    except:
        pass
    try:
        import resource
        usage = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
        if platform.system() == 'Darwin':
            return usage / (1024 * 1024)
        return usage / 1024
    except:
        return 0.0

# ============================================================================
# Architecture Check
# ============================================================================

machine = platform.machine().lower()
if machine not in ['x86_64', 'amd64']:
    raise RuntimeError(f"This script requires x86_64. Got: {machine}")

print("Loading kernels...")
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
    # Warmup
    for _ in range(warmup):
        fn()

    gc.collect()
    mem_before = get_process_memory_mb()
    tracemalloc.start()

    times = []
    peak_mem = mem_before
    for _ in range(iters):
        start = time.perf_counter()
        fn()
        end = time.perf_counter()
        times.append((end - start) * 1000)
        peak_mem = max(peak_mem, get_process_memory_mb())

    _, python_peak = tracemalloc.get_traced_memory()
    tracemalloc.stop()

    return {
        'mean_ms': np.mean(times),
        'std_ms': np.std(times),
        'memory_mb': peak_mem,
    }

# ============================================================================
# Test Cases
# ============================================================================

def test_matmul_kernels(M, K, N, label):
    """Test different matmul kernel variants"""
    print(f"\n  {label} [M={M}, K={K}, N={N}]")

    x = torch.randn(M, K, dtype=torch.float32)
    x_int8, x_scale = quantize_input(x)
    w_int8, w_sum = prepare_ternary_weights(N, K)
    bias = torch.Tensor()  # empty tensor for no bias

    results = {}

    # v3 (baseline)
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
    best_time = min(r['mean_ms'] for r in results.values())
    for name, r in sorted(results.items(), key=lambda x: x[1]['mean_ms']):
        marker = " <-- BEST" if r['mean_ms'] == best_time else ""
        print(f"    {name:20s}: {r['mean_ms']:8.2f} ± {r['std_ms']:5.2f} ms, {r['memory_mb']:8.1f} MB{marker}")

    return results

def test_fused_softmax_kernels(M, K, N, label):
    """Test fused softmax kernel variants"""
    print(f"\n  {label} [M={M}, K={K}, N={N}]")

    x = torch.randn(M, K, dtype=torch.float32)
    x_int8, x_scale = quantize_input(x)
    w_int8, w_sum = prepare_ternary_weights(N, K)
    bias = torch.Tensor()  # empty tensor for no bias

    results = {}

    # Non-fused: matmul + separate softmax
    y_float = torch.zeros(M, N, dtype=torch.float32)
    y_int8 = torch.zeros(M, N, dtype=torch.int8)
    y_scale = torch.zeros(M, dtype=torch.float32)

    def run_separate():
        kernel.matmul_free_vnni_v3(x_int8, x_scale, w_int8, w_sum, y_float, bias, M, N, K, num_threads)
        # Softmax in PyTorch
        probs = torch.softmax(y_float, dim=-1)
        # Quantize
        y_int8.copy_((probs * 127).round().clamp(0, 127).to(torch.int8))
        y_scale.fill_(1.0 / 127.0)
    results['v3 + pytorch_softmax'] = benchmark_kernel(run_separate)

    # Fused softmax
    y_int8 = torch.zeros(M, N, dtype=torch.int8)
    y_scale = torch.zeros(M, dtype=torch.float32)
    def run_fused():
        kernel.matmul_free_vnni_v3_fused_softmax(x_int8, x_scale, w_int8, w_sum, y_int8, y_scale, M, N, K, num_threads)
    results['v3_fused_softmax'] = benchmark_kernel(run_fused)

    # Print results
    best_time = min(r['mean_ms'] for r in results.values())
    for name, r in sorted(results.items(), key=lambda x: x[1]['mean_ms']):
        marker = " <-- BEST" if r['mean_ms'] == best_time else ""
        print(f"    {name:25s}: {r['mean_ms']:8.2f} ± {r['std_ms']:5.2f} ms, {r['memory_mb']:8.1f} MB{marker}")

    return results

def test_fused_quantize_kernels(M, K, N, label):
    """Test fused quantize kernel variants (for down projection)"""
    print(f"\n  {label} [M={M}, K={K}, N={N}]")

    x = torch.randn(M, K, dtype=torch.float32)
    x_int8, x_scale = quantize_input(x)
    w_int8, w_sum = prepare_ternary_weights(N, K)
    bias = torch.Tensor()  # empty tensor for no bias

    results = {}

    # Non-fused: matmul + separate softmax + quantize
    y_float = torch.zeros(M, N, dtype=torch.float32)
    y_int8 = torch.zeros(M, N, dtype=torch.int8)
    y_scale = torch.zeros(M, dtype=torch.float32)

    def run_separate():
        kernel.matmul_free_vnni_v3(x_int8, x_scale, w_int8, w_sum, y_float, bias, M, N, K, num_threads)
        probs = torch.softmax(y_float, dim=-1)
        max_abs = probs.abs().max(dim=1, keepdim=True).values
        scale = max_abs / 127.0
        y_int8.copy_((probs / scale).round().clamp(-127, 127).to(torch.int8))
        y_scale.copy_(scale.squeeze())
    results['v3 + pytorch'] = benchmark_kernel(run_separate)

    # Fused quantize (includes softmax)
    y_int8 = torch.zeros(M, N, dtype=torch.int8)
    y_scale = torch.zeros(M, dtype=torch.float32)
    def run_fused():
        kernel.matmul_free_vnni_v3_fused_quantize(x_int8, x_scale, w_int8, w_sum, y_int8, y_scale, M, N, K, num_threads)
    results['v3_fused_quantize'] = benchmark_kernel(run_fused)

    # Print results
    best_time = min(r['mean_ms'] for r in results.values())
    for name, r in sorted(results.items(), key=lambda x: x[1]['mean_ms']):
        marker = " <-- BEST" if r['mean_ms'] == best_time else ""
        print(f"    {name:25s}: {r['mean_ms']:8.2f} ± {r['std_ms']:5.2f} ms, {r['memory_mb']:8.1f} MB{marker}")

    return results

def test_fused_qkv_kernels(M, K, N, label):
    """Test fused QKV kernel variants"""
    print(f"\n  {label} [M={M}, K={K}, N={N}] (3 weight matrices)")

    x = torch.randn(M, K, dtype=torch.float32)
    x_int8, x_scale = quantize_input(x)
    wq_int8, wq_sum = prepare_ternary_weights(N, K)
    wk_int8, wk_sum = prepare_ternary_weights(N, K)
    wv_int8, wv_sum = prepare_ternary_weights(N, K)
    bias = torch.Tensor()  # empty tensor for no bias

    results = {}

    # Non-fused: 3 separate matmuls + softmax
    q_float = torch.zeros(M, N, dtype=torch.float32)
    k_float = torch.zeros(M, N, dtype=torch.float32)
    v_float = torch.zeros(M, N, dtype=torch.float32)
    q_int8 = torch.zeros(M, N, dtype=torch.int8)
    k_int8 = torch.zeros(M, N, dtype=torch.int8)
    v_int8 = torch.zeros(M, N, dtype=torch.int8)
    q_scale = torch.zeros(M, dtype=torch.float32)
    k_scale = torch.zeros(M, dtype=torch.float32)
    v_scale = torch.zeros(M, dtype=torch.float32)

    def run_separate():
        kernel.matmul_free_vnni_v3(x_int8, x_scale, wq_int8, wq_sum, q_float, bias, M, N, K, num_threads)
        kernel.matmul_free_vnni_v3(x_int8, x_scale, wk_int8, wk_sum, k_float, bias, M, N, K, num_threads)
        kernel.matmul_free_vnni_v3(x_int8, x_scale, wv_int8, wv_sum, v_float, bias, M, N, K, num_threads)
        # Softmax + quantize
        for (f, i8, sc) in [(q_float, q_int8, q_scale), (k_float, k_int8, k_scale), (v_float, v_int8, v_scale)]:
            probs = torch.softmax(f, dim=-1)
            i8.copy_((probs * 127).round().clamp(0, 127).to(torch.int8))
            sc.fill_(1.0 / 127.0)
    results['3x v3 + pytorch'] = benchmark_kernel(run_separate)

    # Fused QKV
    q_int8 = torch.zeros(M, N, dtype=torch.int8)
    k_int8 = torch.zeros(M, N, dtype=torch.int8)
    v_int8 = torch.zeros(M, N, dtype=torch.int8)
    q_scale = torch.zeros(M, dtype=torch.float32)
    k_scale = torch.zeros(M, dtype=torch.float32)
    v_scale = torch.zeros(M, dtype=torch.float32)

    def run_fused():
        kernel.matmul_free_vnni_v3_fused_qkv_softmax(
            x_int8, x_scale,
            wq_int8, wq_sum, wk_int8, wk_sum, wv_int8, wv_sum,
            q_int8, q_scale, k_int8, k_scale, v_int8, v_scale,
            M, N, K, num_threads
        )
    results['v3_fused_qkv_softmax'] = benchmark_kernel(run_fused)

    # Print results
    best_time = min(r['mean_ms'] for r in results.values())
    for name, r in sorted(results.items(), key=lambda x: x[1]['mean_ms']):
        marker = " <-- BEST" if r['mean_ms'] == best_time else ""
        print(f"    {name:25s}: {r['mean_ms']:8.2f} ± {r['std_ms']:5.2f} ms, {r['memory_mb']:8.1f} MB{marker}")

    return results

def test_mlp_up_kernels(M, K, N, label):
    """Test MLP up projection kernels (large N)"""
    print(f"\n  {label} [M={M}, K={K}, N={N}]")

    x = torch.randn(M, K, dtype=torch.float32)
    x_int8, x_scale = quantize_input(x)
    w_int8, w_sum = prepare_ternary_weights(N, K)
    bias = torch.Tensor()  # empty tensor for no bias

    results = {}

    # v3
    y_out = torch.zeros(M, N, dtype=torch.float32)
    def run_v3():
        kernel.matmul_free_vnni_v3(x_int8, x_scale, w_int8, w_sum, y_out, bias, M, N, K, num_threads)
    results['v3'] = benchmark_kernel(run_v3)

    # v4 (designed for large N) - outputs int32
    y_out_i32 = torch.zeros(M, N, dtype=torch.int32)
    def run_v4():
        kernel.matmul_free_vnni_v4_large_n(x_int8, x_scale, w_int8, w_sum, y_out_i32, M, N, K, num_threads)
    results['v4_large_n'] = benchmark_kernel(run_v4)

    # v5 (if M is large enough) - outputs int32
    if M >= 32:
        y_out_i32 = torch.zeros(M, N, dtype=torch.int32)
        def run_v5():
            kernel.matmul_free_vnni_v5_large_m(x_int8, x_scale, w_int8, w_sum, y_out_i32, M, N, K, num_threads)
        results['v5_large_m'] = benchmark_kernel(run_v5)

    # Print results
    best_time = min(r['mean_ms'] for r in results.values())
    for name, r in sorted(results.items(), key=lambda x: x[1]['mean_ms']):
        marker = " <-- BEST" if r['mean_ms'] == best_time else ""
        print(f"    {name:20s}: {r['mean_ms']:8.2f} ± {r['std_ms']:5.2f} ms, {r['memory_mb']:8.1f} MB{marker}")

    return results

# ============================================================================
# Main
# ============================================================================

def main():
    print("=" * 80)
    print("Kernel Selection Benchmark")
    print("=" * 80)
    print(f"Threads: {num_threads}")
    print(f"Batch sizes: {BATCH_SIZES}")
    print()

    all_results = {}

    for M in BATCH_SIZES:
        print(f"\n{'='*80}")
        print(f"BATCH SIZE M={M}")
        print(f"{'='*80}")

        all_results[f"M{M}"] = {}

        # 1. QKV projection
        all_results[f"M{M}"]['qkv'] = test_fused_qkv_kernels(
            M, HIDDEN_DIM, HIDDEN_DIM, "QKV Projection"
        )

        # 2. Output projection (with softmax)
        all_results[f"M{M}"]['output'] = test_fused_softmax_kernels(
            M, HIDDEN_DIM, HIDDEN_DIM, "Output Projection"
        )

        # 3. MLP Up projection (large N)
        all_results[f"M{M}"]['mlp_up'] = test_mlp_up_kernels(
            M, HIDDEN_DIM, MLP_HIDDEN, "MLP Up (Gate+Up)"
        )

        # 4. MLP Down projection (with quantize)
        all_results[f"M{M}"]['mlp_down'] = test_fused_quantize_kernels(
            M, MLP_INTERMEDIATE, HIDDEN_DIM, "MLP Down"
        )

    # Print recommendations
    print("\n")
    print("=" * 80)
    print("RECOMMENDATIONS")
    print("=" * 80)

    for M in BATCH_SIZES:
        print(f"\nM={M}:")
        data = all_results[f"M{M}"]

        for op_name, op_results in data.items():
            best = min(op_results.items(), key=lambda x: x[1]['mean_ms'])
            print(f"  {op_name:12s}: {best[0]}")

    print("\n" + "=" * 80)
    print("Benchmark Complete!")
    print("=" * 80)

if __name__ == '__main__':
    main()
