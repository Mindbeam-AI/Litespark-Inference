#!/usr/bin/env python3
"""Synthetic benchmark matching BitNet 2B model dimensions for pp128/tg128"""

import time
import torch
import os
import platform
import subprocess
from litespark_inf.models import get_kernel, get_kernel_type, prepare_ternary_weight

# BitNet 2B model dimensions (matching MS BitNet paper)
MODEL_DIMS = {
    "hidden_size": 2560,
    "intermediate_size": 6912,
    "num_heads": 32,
    "num_layers": 32,
}

def benchmark_layer(M, N, K, kernel, num_threads=8, num_warmup=5, num_runs=20, use_m1_opt=False):
    """Benchmark a single matmul layer."""
    os.environ['OMP_NUM_THREADS'] = str(num_threads)

    w_float = torch.randn(N, K)
    w_int8, w_sum, w_scale = prepare_ternary_weight(w_float)
    x = torch.randn(M, K)
    y = torch.zeros(M, N, dtype=torch.float32)
    bias = torch.Tensor()

    # Use optimized M=1 kernel if available and M=1
    has_m1_opt = hasattr(kernel, 'matmul_free_vnni_m1_optimized')
    use_opt = use_m1_opt and M == 1 and has_m1_opt

    # Warmup
    for _ in range(num_warmup):
        if use_opt:
            kernel.matmul_free_vnni_m1_optimized(x, w_int8, w_sum, y, bias, w_scale, N, K, num_threads)
        else:
            kernel.matmul_free_vnni_v4_fused_wp(x, w_int8, w_sum, y, bias, w_scale, M, N, K, num_threads)

    # Benchmark
    start = time.perf_counter()
    for _ in range(num_runs):
        if use_opt:
            kernel.matmul_free_vnni_m1_optimized(x, w_int8, w_sum, y, bias, w_scale, N, K, num_threads)
        else:
            kernel.matmul_free_vnni_v4_fused_wp(x, w_int8, w_sum, y, bias, w_scale, M, N, K, num_threads)
    elapsed = (time.perf_counter() - start) / num_runs
    return elapsed

def benchmark_pp128_tg128(num_threads=8, use_m1_opt=False):
    """Simulate pp128 and tg128 benchmark for BitNet 2B."""
    kernel = get_kernel()
    H = MODEL_DIMS["hidden_size"]
    I = MODEL_DIMS["intermediate_size"]
    num_layers = MODEL_DIMS["num_layers"]

    opt_label = " [M1-OPT]" if use_m1_opt else ""
    print(f"\n=== BitNet 2B Synthetic Benchmark (threads={num_threads}){opt_label} ===")
    print(f"Hidden: {H}, Intermediate: {I}, Layers: {num_layers}")

    # Per-layer matmuls: q_proj, k_proj, v_proj, o_proj, gate, up, down
    layer_ops = [
        ("q_proj", H, H),
        ("k_proj", H, H),
        ("v_proj", H, H),
        ("o_proj", H, H),
        ("gate", I, H),
        ("up", I, H),
        ("down", H, I),
    ]

    results = {}
    for label, M_tokens in [("pp128", 128), ("tg128", 1)]:
        total_time = 0
        for name, N, K in layer_ops:
            t = benchmark_layer(M_tokens, N, K, kernel, num_threads, use_m1_opt=use_m1_opt)
            total_time += t

        # Scale by num_layers
        total_time_per_forward = total_time * num_layers
        # tg128 = 128 sequential tokens
        if label == "tg128":
            total_time_model = total_time_per_forward * 128
        else:
            total_time_model = total_time_per_forward

        tokens_per_sec = 128 / total_time_model
        results[label] = {
            "time_ms": total_time_model * 1000,
            "tokens_per_sec": tokens_per_sec
        }
        print(f"{label}: {total_time_model*1000:.2f} ms, {tokens_per_sec:.2f} tok/s")

    return results

if __name__ == '__main__':
    # Get CPU info
    try:
        cpu = subprocess.check_output("lscpu | grep 'Model name'", shell=True).decode().strip()
        print(f"CPU: {cpu}")
    except:
        print(f"Platform: {platform.machine()}")

    print(f"Kernel: {get_kernel_type()}")

    # Test with standard kernel
    print("\n" + "="*60)
    print("STANDARD KERNEL")
    print("="*60)
    all_results = {}
    for threads in [1, 2, 4, 8]:
        r = benchmark_pp128_tg128(threads, use_m1_opt=False)
        all_results[threads] = r

    print("\n" + "="*60)
    print("SUMMARY (Standard)")
    print("="*60)
    print(f"{'Threads':>8} {'pp128 (tok/s)':>15} {'tg128 (tok/s)':>15}")
    print("-"*45)
    for threads in [1, 2, 4, 8]:
        pp = all_results[threads]["pp128"]["tokens_per_sec"]
        tg = all_results[threads]["tg128"]["tokens_per_sec"]
        print(f"{threads:>8} {pp:>15.2f} {tg:>15.2f}")

    # Test with M1-optimized kernel
    print("\n" + "="*60)
    print("M1-OPTIMIZED KERNEL")
    print("="*60)
    all_results_opt = {}
    for threads in [1, 2, 4, 8]:
        r = benchmark_pp128_tg128(threads, use_m1_opt=True)
        all_results_opt[threads] = r

    print("\n" + "="*60)
    print("SUMMARY (M1-Optimized)")
    print("="*60)
    print(f"{'Threads':>8} {'pp128 (tok/s)':>15} {'tg128 (tok/s)':>15}")
    print("-"*45)
    for threads in [1, 2, 4, 8]:
        pp = all_results_opt[threads]["pp128"]["tokens_per_sec"]
        tg = all_results_opt[threads]["tg128"]["tokens_per_sec"]
        print(f"{threads:>8} {pp:>15.2f} {tg:>15.2f}")

    # Compare
    print("\n" + "="*60)
    print("SPEEDUP (M1-Opt vs Standard)")
    print("="*60)
    print(f"{'Threads':>8} {'tg128 speedup':>15}")
    print("-"*30)
    for threads in [1, 2, 4, 8]:
        std = all_results[threads]["tg128"]["tokens_per_sec"]
        opt = all_results_opt[threads]["tg128"]["tokens_per_sec"]
        speedup = opt / std
        print(f"{threads:>8} {speedup:>14.2f}x")

