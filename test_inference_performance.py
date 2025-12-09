#!/usr/bin/env python3
"""
Simple Inference Performance Test

Tests the optimized VNNI kernels on realistic GPT-2 inference workloads.
Uses the same kernel loading as test_full_forward.py to ensure we test
the actual optimized kernels, not the fallback implementations.
"""

import torch
import time
import platform
from torch.utils.cpp_extension import load

def test_inference_patterns():
    """Test performance on realistic inference patterns."""
    
    print("=" * 80)
    print("GPT-2 Inference Performance Test - Optimized VNNI Kernels")
    print("=" * 80)
    
    # Load kernels - EXACTLY the same as test_full_forward.py
    arch = platform.machine().lower()
    if arch not in ['x86_64', 'amd64']:
        print(f"Unsupported architecture: {arch}")
        return
        
    print("Loading optimized VNNI kernels...")
    
    # Use exact same loading as test_full_forward.py to ensure we get the optimized kernels
    kernel = load(
        name="vnni_kernel_inference_test",
        sources=["src/cpu_ops/kernels/x86_64/matmul_free_tmac_vnni.cpp"],
        extra_cflags=['-O3', '-march=native', '-ffast-math', '-fopenmp'],
        extra_ldflags=['-lgomp'],
        verbose=True  # Show compilation to verify correct kernel
    )
    
    print("✅ VNNI kernels loaded successfully")
    print("✅ Using optimized VNNI v2/v3/v4/v5/v6 kernels (same as transformer benchmark)")
    print()
    
    # GPT-2 layer dimensions - realistic inference patterns
    test_cases = [
        # (M, N, K, description)
        (1, 2048, 2048, "Single token - Attention QKV"),
        (1, 16384, 2048, "Single token - MLP up"),
        (1, 2048, 16384, "Single token - MLP down"),
        (1, 50257, 2048, "Single token - LM head (vocab)"),
        
        (8, 2048, 2048, "Small batch - Attention QKV"),
        (8, 16384, 2048, "Small batch - MLP up"),
        (8, 2048, 16384, "Small batch - MLP down"),
        (8, 50257, 2048, "Small batch - LM head (vocab)"),
        
        (32, 2048, 2048, "Medium batch - Attention QKV"),
        (32, 16384, 2048, "Medium batch - MLP up"),
        (32, 2048, 16384, "Medium batch - MLP down"),
        (32, 50257, 2048, "Medium batch - LM head (vocab)"),
        
        (128, 2048, 2048, "Large batch - Attention QKV"),
        (128, 16384, 2048, "Large batch - MLP up"),
        (128, 2048, 16384, "Large batch - MLP down"),
        (128, 50257, 2048, "Large batch - LM head (vocab)"),
    ]
    
    print(f"Testing {len(test_cases)} inference patterns...")
    print(f"{'Description':<35} {'M':<4} {'N':<6} {'K':<6} {'Time (ms)':<10} {'GFLOPS':<8} {'Kernel'}")
    print("-" * 85)
    
    results = []
    
    for M, N, K, desc in test_cases:
        # Create test data - ensure correct types for VNNI kernels
        x_int8 = torch.randint(-128, 127, (M, K), dtype=torch.int8)
        scales = torch.randn(M, dtype=torch.float32)
        w_int8 = torch.randint(-1, 2, (N, K), dtype=torch.int8)  # Ternary weights
        w_sum = torch.sum(w_int8.float(), dim=1).to(torch.float32)  # Ensure float32
        y = torch.zeros(M, N, dtype=torch.float32)
        bias = torch.zeros(N, dtype=torch.float32)
        
        # Use same threading and kernel selection as test_full_forward.py
        base_threads = 8  # Same as transformer benchmark
        if M <= 8:
            threads = min(4, base_threads)
        elif M <= 32:
            threads = min(6, base_threads)
        else:
            threads = base_threads
            
        # Use same kernel selection logic as matmul_int32_out in test_full_forward.py
        if N >= 8192:
            # v4 kernel for large N - outputs int32
            y_out = torch.zeros(M, N, dtype=torch.int32)
            kernel_name = "v4"
            use_bias = False
        elif N >= 4096 and M >= 16:
            # v4 kernel for large N with medium+ M - outputs int32
            y_out = torch.zeros(M, N, dtype=torch.int32)
            kernel_name = "v4"
            use_bias = False
        elif M >= 64:
            # v5 kernel for large M - outputs int32
            y_out = torch.zeros(M, N, dtype=torch.int32)
            kernel_name = "v5"
            use_bias = False
        elif M >= 32 and N >= 2048:
            # v5 kernel for medium M with medium N - outputs int32
            y_out = torch.zeros(M, N, dtype=torch.int32)
            kernel_name = "v5"
            use_bias = False
        else:
            # v3 kernel for small operations - outputs float32
            y_out = y  # v3 outputs float32
            kernel_name = "v3"
            use_bias = True

        # Warmup
        for _ in range(3):
            if kernel_name == "v3":
                kernel.matmul_free_vnni_v3(x_int8, scales, w_int8, w_sum, y_out, bias, M, N, K, threads)
            elif kernel_name == "v4":
                kernel.matmul_free_vnni_v4_large_n(x_int8, scales, w_int8, w_sum, y_out, M, N, K, threads)
            elif kernel_name == "v5":
                kernel.matmul_free_vnni_v5_large_m(x_int8, scales, w_int8, w_sum, y_out, M, N, K, threads)

        # Benchmark
        start_time = time.time()

        for _ in range(10):
            if kernel_name == "v3":
                kernel.matmul_free_vnni_v3(x_int8, scales, w_int8, w_sum, y_out, bias, M, N, K, threads)
            elif kernel_name == "v4":
                kernel.matmul_free_vnni_v4_large_n(x_int8, scales, w_int8, w_sum, y_out, M, N, K, threads)
            elif kernel_name == "v5":
                kernel.matmul_free_vnni_v5_large_m(x_int8, scales, w_int8, w_sum, y_out, M, N, K, threads)

        end_time = time.time()

        # Calculate metrics
        avg_time_ms = (end_time - start_time) * 1000 / 10
        flops = 2 * M * N * K  # Approximate FLOPs for matmul
        gflops = flops / (avg_time_ms * 1e6)

        print(f"{desc:<35} {M:<4} {N:<6} {K:<6} {avg_time_ms:<10.2f} {gflops:<8.1f} ({kernel_name})")

        results.append({
            'description': desc,
            'M': M, 'N': N, 'K': K,
            'time_ms': avg_time_ms,
            'gflops': gflops,
            'kernel': kernel_name
        })

    # Summary
    print("\n" + "=" * 80)
    print("INFERENCE PERFORMANCE SUMMARY")
    print("=" * 80)

    # Group by batch size
    batch_groups = {}
    for result in results:
        M = result['M']
        if M not in batch_groups:
            batch_groups[M] = []
        batch_groups[M].append(result)

    for M in sorted(batch_groups.keys()):
        group = batch_groups[M]
        avg_gflops = sum(r['gflops'] for r in group) / len(group)
        total_time = sum(r['time_ms'] for r in group)

        print(f"Batch M={M:<3}: {len(group)} operations, {avg_gflops:.1f} avg GFLOPS, {total_time:.1f}ms total")

    print(f"\nOverall average: {sum(r['gflops'] for r in results) / len(results):.1f} GFLOPS")
    print("✅ Inference performance test completed successfully!")

if __name__ == "__main__":
    test_inference_patterns()
