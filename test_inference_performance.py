#!/usr/bin/env python3
"""
Simple Inference Performance Test

Tests the optimized kernels on realistic GPT-2 inference workloads
without requiring external dependencies.
"""

import torch
import time
import sys
import platform
from pathlib import Path

# Add src to path
sys.path.append('src')
from cpu_ops import load_kernels

def test_inference_patterns():
    """Test performance on realistic inference patterns."""
    
    print("=" * 70)
    print("GPT-2 Inference Performance Test")
    print("=" * 70)
    
    # Load kernels
    arch = platform.machine().lower()
    if arch in ['x86_64', 'amd64']:
        arch = 'x86_64'
    else:
        print(f"Unsupported architecture: {arch}")
        return
        
    kernel = load_kernels(arch)
    print(f"Loaded kernel for {arch}")
    
    # GPT-2 layer dimensions
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
    
    print(f"\nTesting {len(test_cases)} inference patterns...")
    print(f"{'Description':<35} {'M':<4} {'N':<6} {'K':<6} {'Time (ms)':<10} {'GFLOPS':<8}")
    print("-" * 70)
    
    results = []
    
    for M, N, K, desc in test_cases:
        # Create test data
        x_int8 = torch.randint(-128, 127, (M, K), dtype=torch.int8)
        scales = torch.randn(M, dtype=torch.float32)
        w_int8 = torch.randint(-1, 2, (N, K), dtype=torch.int8)  # Ternary weights
        w_sum = torch.sum(w_int8.float(), dim=1)
        y = torch.zeros(M, N, dtype=torch.float32)
        bias = torch.zeros(N, dtype=torch.float32)
        
        # Warmup
        for _ in range(3):
            kernel.matmul_free_vnni_v3(
                x_int8, scales, w_int8, w_sum, y, bias, M, N, K, 8
            )
        
        # Benchmark
        torch.cuda.synchronize() if torch.cuda.is_available() else None
        start_time = time.time()
        
        for _ in range(10):
            kernel.matmul_free_vnni_v3(
                x_int8, scales, w_int8, w_sum, y, bias, M, N, K, 8
            )
        
        torch.cuda.synchronize() if torch.cuda.is_available() else None
        end_time = time.time()
        
        # Calculate metrics
        avg_time_ms = (end_time - start_time) * 1000 / 10
        flops = 2 * M * N * K  # Approximate FLOPs for matmul
        gflops = flops / (avg_time_ms * 1e6)
        
        print(f"{desc:<35} {M:<4} {N:<6} {K:<6} {avg_time_ms:<10.2f} {gflops:<8.1f}")
        
        results.append({
            'description': desc,
            'M': M, 'N': N, 'K': K,
            'time_ms': avg_time_ms,
            'gflops': gflops
        })
    
    # Summary
    print("\n" + "=" * 70)
    print("INFERENCE PERFORMANCE SUMMARY")
    print("=" * 70)
    
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
    print("Test completed successfully!")

if __name__ == "__main__":
    test_inference_patterns()
