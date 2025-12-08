#!/usr/bin/env python3
"""
Benchmark Graviton Kernels

Tests the Graviton-optimized ternary matmul kernels:
1. matmul_free_graviton_v2 (register-blocked)
2. matmul_free_graviton_v3 (cache-tiled)
3. matmul_free_graviton_v3_fused_softmax
4. matmul_free_graviton_v3_fused_quantize
5. matmul_free_graviton_v3_fused_swiglu

Compares against:
- PyTorch float32 baseline
- PyTorch int8 (if available)
"""

import torch
import time
import gc
import platform
import argparse
import os
from pathlib import Path
from typing import Dict, List, Tuple
from torch.utils.cpp_extension import load


def get_memory_mb():
    """Get process RSS in MB."""
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


def load_graviton_kernel():
    """Load the Graviton kernel."""
    kernel_path = Path(__file__).parent.parent / 'src/cpu_ops/kernels/arm64/matmul_free_graviton.cpp'

    if not kernel_path.exists():
        raise FileNotFoundError(f"Kernel not found: {kernel_path}")

    print(f"Compiling Graviton kernel from {kernel_path}...")
    kernel = load(
        name='matmul_free_graviton',
        sources=[str(kernel_path)],
        extra_cflags=['-O3', '-march=armv8.2-a+dotprod', '-fopenmp'],
        extra_ldflags=['-fopenmp'],
        verbose=True
    )
    print("Kernel compiled successfully!")
    return kernel


def prepare_test_data(M: int, N: int, K: int, K_padded: int):
    """Prepare test data for benchmarking."""
    # Random input
    x = torch.randn(M, K, dtype=torch.float32)

    # Random ternary weights
    w = torch.randint(-1, 2, (N, K), dtype=torch.float32)

    # Quantize input
    max_abs = x.abs().max(dim=1, keepdim=True).values
    scale = max_abs / 127.0
    scale = torch.where(scale == 0, torch.ones_like(scale), scale)

    x_int8 = torch.zeros(M, K_padded, dtype=torch.int8)
    x_int8[:, :K] = (x / scale).round().clamp(-127, 127).to(torch.int8)

    # Prepare weights
    w_int8 = torch.zeros(N, K_padded, dtype=torch.int8)
    w_int8[:, :K] = w.round().clamp(-1, 1).to(torch.int8)
    w_sum = w_int8.sum(dim=1, dtype=torch.int32)

    return {
        'x': x,
        'x_int8': x_int8.contiguous(),
        'x_scale': scale.squeeze(1).contiguous(),
        'w': w,
        'w_int8': w_int8.contiguous(),
        'w_sum': w_sum.contiguous(),
    }


def benchmark_pytorch_f32(x: torch.Tensor, w: torch.Tensor, num_runs: int = 10) -> Dict:
    """Benchmark PyTorch float32 matmul."""
    # Warmup
    for _ in range(3):
        _ = torch.mm(x, w.T)

    gc.collect()

    # Benchmark
    start = time.perf_counter()
    for _ in range(num_runs):
        y = torch.mm(x, w.T)
    torch.cuda.synchronize() if torch.cuda.is_available() else None
    end = time.perf_counter()

    total_time = end - start
    M, K = x.shape
    N = w.shape[0]
    ops = 2 * M * N * K * num_runs  # multiply-add
    gflops = ops / (total_time * 1e9)

    return {
        'time_ms': total_time * 1000 / num_runs,
        'gflops': gflops,
    }


def benchmark_kernel(
    kernel,
    kernel_func_name: str,
    data: Dict,
    M: int, N: int, K: int,
    num_threads: int,
    num_runs: int = 10,
    **extra_args
) -> Dict:
    """Benchmark a kernel function."""
    kernel_func = getattr(kernel, kernel_func_name)

    # Prepare output tensor
    if 'fused' in kernel_func_name:
        y = torch.zeros(M, N, dtype=torch.int8)
        y_scale = torch.zeros(M, dtype=torch.float32)
    else:
        y = torch.zeros(M, N, dtype=torch.float32)

    bias = torch.Tensor()  # Empty bias

    # Warmup
    for _ in range(3):
        if 'fused_swiglu' in kernel_func_name:
            # SwiGLU needs gate and up weights
            kernel_func(
                data['x_int8'], data['x_scale'],
                data['w_int8'], data['w_sum'],
                data['w_int8'], data['w_sum'],  # Use same weights for gate/up
                y, y_scale if 'fused' in kernel_func_name else None,
                M, N, K, num_threads
            )
        elif 'fused' in kernel_func_name:
            kernel_func(
                data['x_int8'], data['x_scale'],
                data['w_int8'], data['w_sum'],
                y, y_scale,
                M, N, K, num_threads
            )
        else:
            kernel_func(
                data['x_int8'], data['x_scale'],
                data['w_int8'], data['w_sum'],
                y, bias,
                M, N, K, num_threads
            )

    gc.collect()

    # Benchmark
    start = time.perf_counter()
    for _ in range(num_runs):
        if 'fused_swiglu' in kernel_func_name:
            kernel_func(
                data['x_int8'], data['x_scale'],
                data['w_int8'], data['w_sum'],
                data['w_int8'], data['w_sum'],
                y, y_scale,
                M, N, K, num_threads
            )
        elif 'fused' in kernel_func_name:
            kernel_func(
                data['x_int8'], data['x_scale'],
                data['w_int8'], data['w_sum'],
                y, y_scale,
                M, N, K, num_threads
            )
        else:
            kernel_func(
                data['x_int8'], data['x_scale'],
                data['w_int8'], data['w_sum'],
                y, bias,
                M, N, K, num_threads
            )
    end = time.perf_counter()

    total_time = end - start
    ops = 2 * M * N * K * num_runs
    gops = ops / (total_time * 1e9)  # GOPS for int8

    return {
        'time_ms': total_time * 1000 / num_runs,
        'gops': gops,
    }


def run_benchmark(
    M: int, N: int, K: int,
    num_threads: int = None,
    num_runs: int = 10
):
    """Run benchmark for a specific size."""
    if num_threads is None:
        num_threads = os.cpu_count() or 1

    print(f"\n{'='*70}")
    print(f"Benchmark: M={M}, N={N}, K={K}, threads={num_threads}")
    print(f"{'='*70}")

    # Load kernel
    kernel = load_graviton_kernel()

    # K padding for NEON (16 bytes)
    K_padded = ((K + 15) // 16) * 16

    # Prepare data
    print("Preparing test data...")
    data = prepare_test_data(M, N, K, K_padded)

    # Benchmark PyTorch baseline
    print("\nPyTorch float32 baseline:")
    pytorch_result = benchmark_pytorch_f32(data['x'], data['w'], num_runs)
    print(f"  Time: {pytorch_result['time_ms']:.3f} ms")
    print(f"  GFLOPS: {pytorch_result['gflops']:.2f}")

    # Benchmark Graviton kernels
    kernels_to_test = [
        'matmul_free_graviton_v2',
        'matmul_free_graviton_v3',
        'matmul_free_graviton_v3_fused_softmax',
        'matmul_free_graviton_v3_fused_quantize',
        'matmul_free_graviton_v3_fused_swiglu',
    ]

    results = {'pytorch_f32': pytorch_result}

    for kernel_name in kernels_to_test:
        print(f"\n{kernel_name}:")
        try:
            result = benchmark_kernel(
                kernel, kernel_name, data,
                M, N, K, num_threads, num_runs
            )
            results[kernel_name] = result
            speedup = pytorch_result['time_ms'] / result['time_ms']
            print(f"  Time: {result['time_ms']:.3f} ms")
            print(f"  GOPS: {result['gops']:.2f}")
            print(f"  Speedup vs PyTorch: {speedup:.2f}x")
        except Exception as e:
            print(f"  Error: {e}")
            results[kernel_name] = None

    return results


def run_all_benchmarks(num_threads: int = None, num_runs: int = 10):
    """Run benchmarks for various sizes."""
    # Test sizes (M, N, K)
    sizes = [
        # Small (token-level)
        (1, 2048, 2048),
        (1, 8192, 2048),
        (1, 2048, 8192),

        # Medium (batch)
        (32, 2048, 2048),
        (32, 8192, 2048),
        (32, 2048, 8192),

        # Large (full batch)
        (128, 2048, 2048),
        (128, 8192, 2048),
        (128, 2048, 8192),

        # Very large
        (512, 4096, 4096),
    ]

    all_results = {}

    for M, N, K in sizes:
        key = f"M{M}_N{N}_K{K}"
        try:
            all_results[key] = run_benchmark(M, N, K, num_threads, num_runs)
        except Exception as e:
            print(f"Error for {key}: {e}")
            all_results[key] = None

    # Print summary
    print("\n" + "=" * 70)
    print("SUMMARY")
    print("=" * 70)
    print(f"\n{'Size':<25} {'PyTorch (ms)':<15} {'v2 (ms)':<15} {'v3 (ms)':<15} {'Speedup':<10}")
    print("-" * 80)

    for key, results in all_results.items():
        if results is None:
            continue

        pytorch_time = results['pytorch_f32']['time_ms']
        v2_time = results.get('matmul_free_graviton_v2', {}).get('time_ms', float('inf'))
        v3_time = results.get('matmul_free_graviton_v3', {}).get('time_ms', float('inf'))
        best_time = min(v2_time, v3_time)
        speedup = pytorch_time / best_time if best_time > 0 else 0

        print(f"{key:<25} {pytorch_time:<15.3f} {v2_time:<15.3f} {v3_time:<15.3f} {speedup:<10.2f}x")

    return all_results


def verify_correctness(M: int = 32, N: int = 256, K: int = 512, num_threads: int = 4):
    """Verify kernel correctness against PyTorch."""
    print("\n" + "=" * 70)
    print("Correctness Verification")
    print("=" * 70)

    kernel = load_graviton_kernel()
    K_padded = ((K + 15) // 16) * 16

    # Prepare data
    data = prepare_test_data(M, N, K, K_padded)

    # PyTorch reference (float32)
    # For ternary weights, result should be: x @ w.T
    # But we're using quantized x, so: (x_int8 * x_scale) @ w.T
    x_dequant = data['x_int8'][:, :K].float() * data['x_scale'].unsqueeze(1)
    w_float = data['w_int8'][:, :K].float()
    ref = torch.mm(x_dequant, w_float.T)

    # Test v2 kernel
    y_v2 = torch.zeros(M, N, dtype=torch.float32)
    bias = torch.Tensor()
    kernel.matmul_free_graviton_v2(
        data['x_int8'], data['x_scale'],
        data['w_int8'], data['w_sum'],
        y_v2, bias,
        M, N, K, num_threads
    )

    # Test v3 kernel
    y_v3 = torch.zeros(M, N, dtype=torch.float32)
    kernel.matmul_free_graviton_v3(
        data['x_int8'], data['x_scale'],
        data['w_int8'], data['w_sum'],
        y_v3, bias,
        M, N, K, num_threads
    )

    # Compare
    diff_v2 = (y_v2 - ref).abs().max().item()
    diff_v3 = (y_v3 - ref).abs().max().item()

    print(f"\nv2 max diff from reference: {diff_v2:.6f}")
    print(f"v3 max diff from reference: {diff_v3:.6f}")

    # Check relative error
    ref_scale = ref.abs().max().item()
    rel_err_v2 = diff_v2 / ref_scale if ref_scale > 0 else diff_v2
    rel_err_v3 = diff_v3 / ref_scale if ref_scale > 0 else diff_v3

    print(f"\nv2 relative error: {rel_err_v2:.6f}")
    print(f"v3 relative error: {rel_err_v3:.6f}")

    threshold = 1e-4
    v2_ok = rel_err_v2 < threshold
    v3_ok = rel_err_v3 < threshold

    print(f"\nv2 correctness: {'PASS' if v2_ok else 'FAIL'}")
    print(f"v3 correctness: {'PASS' if v3_ok else 'FAIL'}")

    return v2_ok and v3_ok


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Benchmark Graviton kernels')
    parser.add_argument('--threads', type=int, default=None,
                        help='Number of threads (default: all cores)')
    parser.add_argument('--runs', type=int, default=10,
                        help='Number of benchmark runs')
    parser.add_argument('--verify', action='store_true',
                        help='Run correctness verification')
    parser.add_argument('--single', action='store_true',
                        help='Run single benchmark (M=128, N=2048, K=2048)')
    parser.add_argument('-M', type=int, default=128)
    parser.add_argument('-N', type=int, default=2048)
    parser.add_argument('-K', type=int, default=2048)

    args = parser.parse_args()

    # Print system info
    print("System Information:")
    print(f"  Platform: {platform.platform()}")
    print(f"  Machine: {platform.machine()}")
    print(f"  CPU count: {os.cpu_count()}")
    print(f"  PyTorch threads: {torch.get_num_threads()}")

    if args.verify:
        verify_correctness(num_threads=args.threads or 4)
    elif args.single:
        run_benchmark(args.M, args.N, args.K, args.threads, args.runs)
    else:
        run_all_benchmarks(args.threads, args.runs)
