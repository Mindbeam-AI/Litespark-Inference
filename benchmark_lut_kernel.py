#!/usr/bin/env python3
"""Micro-benchmark: LUT kernel vs VNNI kernel for M=1 (token generation)"""

import time
import torch
import os
from litespark_inf.models import get_kernel, get_kernel_type, prepare_ternary_weight

def benchmark_kernels(N, K, num_threads=8, num_warmup=10, num_runs=100):
    """Compare LUT vs VNNI kernels for M=1 case."""
    kernel = get_kernel()
    kernel_type = get_kernel_type()
    print(f"\n=== N={N}, K={K}, threads={num_threads} ===")
    print(f"Kernel type: {kernel_type}")
    
    os.environ['OMP_NUM_THREADS'] = str(num_threads)
    
    # Create random ternary weights and activations
    w_float = torch.randn(N, K)
    w_int8, w_sum, w_scale = prepare_ternary_weight(w_float)
    x = torch.randn(1, K)  # M=1
    
    # Prepare LUT weights
    K2 = K // 2
    w_lut_idx = torch.zeros(N, K2, dtype=torch.uint8)
    kernel.pack_weights_lut_idx(w_int8, w_lut_idx, N, K)
    
    y_vnni = torch.zeros(1, N, dtype=torch.float32)
    y_lut = torch.zeros(1, N, dtype=torch.float32)
    bias = torch.Tensor()  # empty
    
    # Warmup VNNI
    for _ in range(num_warmup):
        kernel.matmul_free_vnni_v4_fused_wp(x, w_int8, w_sum, y_vnni, bias, w_scale, 1, N, K, num_threads)
    
    # Benchmark VNNI
    torch.cuda.synchronize() if torch.cuda.is_available() else None
    start = time.perf_counter()
    for _ in range(num_runs):
        kernel.matmul_free_vnni_v4_fused_wp(x, w_int8, w_sum, y_vnni, bias, w_scale, 1, N, K, num_threads)
    vnni_time = (time.perf_counter() - start) / num_runs * 1000
    vnni_gops = N * K * 2 / vnni_time / 1e6
    
    # Warmup LUT
    for _ in range(num_warmup):
        kernel.matmul_lut_m1_fused(x, w_lut_idx, y_lut, bias, w_scale, 1, N, K, num_threads)
    
    # Benchmark LUT
    start = time.perf_counter()
    for _ in range(num_runs):
        kernel.matmul_lut_m1_fused(x, w_lut_idx, y_lut, bias, w_scale, 1, N, K, num_threads)
    lut_time = (time.perf_counter() - start) / num_runs * 1000
    lut_gops = N * K * 2 / lut_time / 1e6
    
    # Compare outputs
    diff = (y_vnni - y_lut).abs().max().item()
    
    print(f"VNNI:  {vnni_time:.3f} ms ({vnni_gops:.1f} GOPS)")
    print(f"LUT:   {lut_time:.3f} ms ({lut_gops:.1f} GOPS)")
    print(f"Speedup: {vnni_time/lut_time:.2f}x")
    print(f"Max diff: {diff:.6f}")
    
    return {'vnni_ms': vnni_time, 'lut_ms': lut_time, 'speedup': vnni_time/lut_time}

if __name__ == '__main__':
    import platform
    import subprocess
    
    # Get CPU info
    try:
        cpu_info = subprocess.check_output("lscpu | grep 'Model name'", shell=True).decode().strip()
        print(f"CPU: {cpu_info}")
    except:
        print(f"Platform: {platform.machine()}")
    
    print("\n" + "="*60)
    print("LUT vs VNNI Kernel Benchmark (M=1)")
    print("="*60)
    
    # Test various sizes (typical transformer dimensions)
    sizes = [
        (2048, 2048),   # Small
        (4096, 4096),   # Medium (attention)
        (8192, 2048),   # MLP down
        (2048, 8192),   # MLP up
        (8192, 8192),   # Large
    ]
    
    results = []
    for N, K in sizes:
        r = benchmark_kernels(N, K, num_threads=8)
        results.append((N, K, r))
    
    print("\n" + "="*60)
    print("SUMMARY")
    print("="*60)
    print(f"{'N':>6} {'K':>6} {'VNNI(ms)':>10} {'LUT(ms)':>10} {'Speedup':>8}")
    print("-"*45)
    for N, K, r in results:
        print(f"{N:>6} {K:>6} {r['vnni_ms']:>10.3f} {r['lut_ms']:>10.3f} {r['speedup']:>7.2f}x")

