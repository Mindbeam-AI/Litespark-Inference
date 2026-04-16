#!/usr/bin/env python3
"""Synthetic benchmark matching BitNet 2B model dimensions for pp128/tg128"""

import argparse
import json
import os
import platform
import subprocess
import time

import torch
from litespark_inference.models import (
    get_kernel,
    get_kernel_type,
    prepare_ternary_weight,
    quantize_activation,
)

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

    kernel_type = get_kernel_type()

    w_float = torch.randn(N, K)
    w_int8, w_sum, w_scale = prepare_ternary_weight(w_float)
    x = torch.randn(M, K)
    bias = torch.Tensor()

    has_m1_opt = kernel_type == 'vnni' and hasattr(kernel, 'matmul_free_vnni_m1_optimized')
    use_opt = use_m1_opt and M == 1 and has_m1_opt

    def run_benchmark():
        if kernel_type == 'vnni':
            if M <= 4 and num_threads > 1:
                y = torch.zeros(M, N, dtype=torch.float32)
                if use_opt:
                    kernel.matmul_free_vnni_m1_optimized(x.contiguous(), w_int8, w_sum, y, bias, w_scale, N, K, num_threads)
                else:
                    kernel.matmul_free_vnni_v4_fused_wp(x.contiguous(), w_int8, w_sum, y, bias, w_scale, M, N, K, num_threads)
                return

            x_int8, x_scale = quantize_activation(x)
            if N >= 1024:
                y_int32 = torch.zeros(M, N, dtype=torch.int32)
                kernel.matmul_free_vnni_v4_large_n(x_int8, x_scale, w_int8, w_sum, y_int32, M, N, K, num_threads)
            elif K <= 1024 and N <= 4096:
                y = torch.zeros(M, N, dtype=torch.float32)
                kernel.matmul_free_vnni_v2(x_int8, x_scale, w_int8, w_sum, y, bias, M, N, K, num_threads)
            else:
                y = torch.zeros(M, N, dtype=torch.float32)
                kernel.matmul_free_vnni_v3(x_int8, x_scale, w_int8, w_sum, y, bias, M, N, K, num_threads)
            return

        if kernel_type == 'avx_vnni':
            x_int8, x_scale = quantize_activation(x)
            if N >= 1024:
                y_int32 = torch.zeros(M, N, dtype=torch.int32)
                kernel.matmul_free_avx_vnni_v4_large_n(x_int8, x_scale, w_int8, w_sum, y_int32, M, N, K, num_threads)
            else:
                y = torch.zeros(M, N, dtype=torch.float32)
                kernel.matmul_free_avx_vnni_v3(x_int8, x_scale, w_int8, w_sum, y, bias, M, N, K, num_threads)
            return

        if kernel_type == 'graviton':
            y = torch.zeros(M, N, dtype=torch.float32)
            kernel.matmul_free_graviton_v4_fused(x.contiguous(), w_int8, w_sum, y, bias, w_scale, M, N, K, num_threads)
            return

        if kernel_type == 'neon':
            y = torch.zeros(M, N, dtype=torch.float32)
            if M <= 4 and num_threads > 1:
                kernel.matmul_free_neon_sdot_v4_fused_wp(x.contiguous(), w_int8, w_sum, y, bias, w_scale, M, N, K, num_threads)
            else:
                kernel.matmul_free_neon_sdot_v4_fused(x.contiguous(), w_int8, w_sum, y, bias, w_scale, M, N, K, num_threads)
            return

        raise RuntimeError(f"Unsupported kernel type: {kernel_type}")

    for _ in range(num_warmup):
        run_benchmark()

    start = time.perf_counter()
    for _ in range(num_runs):
        run_benchmark()
    return (time.perf_counter() - start) / num_runs

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
    parser = argparse.ArgumentParser(description='Synthetic pp128/tg128 benchmark')
    parser.add_argument(
        '--output',
        type=str,
        default=None,
        help='Optional JSON output path',
    )
    args = parser.parse_args()

    # Get CPU info
    try:
        system = platform.system()
        if system == 'Darwin':
            cpu = subprocess.check_output(['sysctl', '-n', 'machdep.cpu.brand_string']).decode().strip()
        elif system == 'Linux':
            cpu = subprocess.check_output(['sh', '-c', "lscpu | grep 'Model name'"]).decode().strip()
        elif system == 'Windows':
            cpu = subprocess.check_output(['powershell', '-Command', '(Get-CimInstance Win32_Processor).Name']).decode().strip()
        else:
            cpu = platform.machine()
        print(f"CPU: {cpu}")
    except Exception:
        print(f"Platform: {platform.machine()}")

    kernel = get_kernel()
    kernel_type = get_kernel_type()
    print(f"Kernel: {kernel_type}")

    # Test with standard kernel
    print("\n" + "="*60)
    print("STANDARD KERNEL")
    print("="*60)
    all_results = {
        "benchmark_name": "synthetic_pp128_tg128",
        "kernel": kernel_type,
        "results": {},
    }
    for threads in [1, 2, 4, 8]:
        r = benchmark_pp128_tg128(threads, use_m1_opt=False)
        all_results["results"][f"threads_{threads}"] = r

    print("\n" + "="*60)
    print("SUMMARY (Standard)")
    print("="*60)
    print(f"{'Threads':>8} {'pp128 (tok/s)':>15} {'tg128 (tok/s)':>15}")
    print("-"*45)
    for threads in [1, 2, 4, 8]:
        pp = all_results["results"][f"threads_{threads}"]["pp128"]["tokens_per_sec"]
        tg = all_results["results"][f"threads_{threads}"]["tg128"]["tokens_per_sec"]
        print(f"{threads:>8} {pp:>15.2f} {tg:>15.2f}")

    if kernel_type == 'vnni' and hasattr(kernel, 'matmul_free_vnni_m1_optimized'):
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

        print("\n" + "="*60)
        print("SPEEDUP (M1-Opt vs Standard)")
        print("="*60)
        print(f"{'Threads':>8} {'tg128 speedup':>15}")
        print("-"*30)
        for threads in [1, 2, 4, 8]:
            std = all_results["results"][f"threads_{threads}"]["tg128"]["tokens_per_sec"]
            opt = all_results_opt[threads]["tg128"]["tokens_per_sec"]
            speedup = opt / std
            print(f"{threads:>8} {speedup:>14.2f}x")

    if args.output:
        with open(args.output, "w", encoding="utf-8") as f:
            json.dump(all_results, f, indent=2)
        print(f"\nResults saved to: {args.output}")
