#!/usr/bin/env python3
"""
Test script for CPU-optimized MatMul-free operations

Verifies correctness and basic functionality of CPU implementations
across different architectures.
"""

import torch
import numpy as np
import sys
import psutil
import os
import gc
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.cpu_ops import (
    get_cpu_info,
    detect_simd_support, 
    get_optimal_implementation,
    CPUMatMulFreeLinear
)
from src.cpu_ops.matmul_free_cpu import (
    matmul_free_cpu,
    matmul_free_cpu_python,
    weight_quant_cpu,
    activation_quant_cpu
)


def test_weight_quantization():
    """Test ternary weight quantization."""
    print("Testing weight quantization...")
    
    # Create test weights
    w = torch.randn(64, 128)
    w_ternary = weight_quant_cpu(w)
    
    # Check that weights are ternary
    unique_vals = torch.unique(w_ternary)
    print(f"  Unique weight values: {unique_vals.numpy()}")
    
    # Should have at most 3 unique values (including 0)
    assert len(unique_vals) <= 3, f"Expected ≤3 unique values, got {len(unique_vals)}"
    
    # Check that non-zero values are symmetric
    non_zero = unique_vals[unique_vals != 0]
    if len(non_zero) > 0:
        assert torch.allclose(non_zero.abs(), non_zero.abs()[0:1]), "Non-zero values should be symmetric"
    
    print("  ✅ Weight quantization test passed")


def test_activation_quantization():
    """Test activation quantization."""
    print("Testing activation quantization...")
    
    # Create test activations
    x = torch.randn(32, 128)
    x_quant = activation_quant_cpu(x)
    
    # Check quantization range
    assert x_quant.abs().max() <= 127.0, "Quantized values should be ≤ 127"
    
    # Check that quantization preserves relative magnitudes
    correlation = torch.corrcoef(torch.stack([x.flatten(), x_quant.flatten()]))[0, 1]
    assert correlation > 0.9, f"Quantization should preserve correlation, got {correlation:.3f}"
    
    print("  ✅ Activation quantization test passed")


def test_matmul_free_correctness():
    """Test MatMul-free operation correctness."""
    print("Testing MatMul-free correctness...")
    
    # Create small test case for manual verification
    M, N, K = 4, 3, 5
    
    # Simple test case with known ternary weights
    x = torch.ones(M, K)  # All ones for easy calculation
    w = torch.tensor([
        [1.0, -1.0, 0.0, 1.0, -1.0],  # Should sum to 0
        [1.0, 1.0, 1.0, 0.0, 0.0],    # Should sum to 3
        [-1.0, -1.0, 0.0, 0.0, 1.0],  # Should sum to -1
    ], dtype=torch.float32)
    
    # Expected result: each row of x (all 1s) multiplied by each row of w
    expected = torch.tensor([
        [0.0, 3.0, -1.0],  # Row 0 of x
        [0.0, 3.0, -1.0],  # Row 1 of x  
        [0.0, 3.0, -1.0],  # Row 2 of x
        [0.0, 3.0, -1.0],  # Row 3 of x
    ], dtype=torch.float32)
    
    # Test Python implementation
    result_python = matmul_free_cpu_python(x, w)
    
    print(f"  Expected shape: {expected.shape}, got: {result_python.shape}")
    print(f"  Expected:\n{expected}")
    print(f"  Got (Python):\n{result_python}")
    
    assert torch.allclose(result_python, expected, atol=1e-5), "Python implementation incorrect"
    print("  ✅ Python implementation correct")
    
    # Test main interface (will use optimized if available, otherwise Python)
    result_main = matmul_free_cpu(x, w)
    assert torch.allclose(result_main, expected, atol=1e-5), "Main interface incorrect"
    print("  ✅ Main interface correct")


def test_cpu_linear_layer():
    """Test CPU MatMul-free linear layer."""
    print("Testing CPU linear layer...")
    
    # Create layer
    layer = CPUMatMulFreeLinear(128, 64, bias=True)
    
    # Test forward pass
    x = torch.randn(32, 128)
    output = layer(x)
    
    assert output.shape == (32, 64), f"Expected shape (32, 64), got {output.shape}"
    assert not torch.isnan(output).any(), "Output contains NaN values"
    assert not torch.isinf(output).any(), "Output contains infinite values"
    
    print(f"  Input shape: {x.shape}")
    print(f"  Output shape: {output.shape}")
    print(f"  Output range: [{output.min():.3f}, {output.max():.3f}]")
    print("  ✅ CPU linear layer test passed")


def test_performance_comparison():
    """Compare performance against PyTorch."""
    print("Testing performance comparison...")
    
    M, N, K = 128, 256, 512
    
    # Create test data
    x = torch.randn(M, K)
    w_full = torch.randn(N, K)
    w_ternary = weight_quant_cpu(w_full)
    
    # Time PyTorch F.linear
    import time
    
    # Warmup
    for _ in range(5):
        _ = torch.nn.functional.linear(x, w_full)
    
    start = time.perf_counter()
    for _ in range(10):
        result_pytorch = torch.nn.functional.linear(x, w_full)
    pytorch_time = (time.perf_counter() - start) / 10
    
    # Time MatMul-free
    for _ in range(5):
        _ = matmul_free_cpu(x, w_ternary)
    
    start = time.perf_counter()
    for _ in range(10):
        result_matmul_free = matmul_free_cpu(x, w_ternary)
    matmul_free_time = (time.perf_counter() - start) / 10
    
    speedup = pytorch_time / matmul_free_time
    
    print(f"  PyTorch time: {pytorch_time*1000:.2f}ms")
    print(f"  MatMul-free time: {matmul_free_time*1000:.2f}ms")
    print(f"  Speedup: {speedup:.2f}x")

    # Memory usage comparison - measure actual memory
    gc.collect()
    process = psutil.Process(os.getpid())
    mem_before = process.memory_info().rss / (1024 * 1024)

    # Create fresh tensors to measure actual allocation
    w_full_test = torch.randn(N, K)
    mem_after_full = process.memory_info().rss / (1024 * 1024)
    pytorch_mem = mem_after_full - mem_before

    gc.collect()
    mem_before_ternary = process.memory_info().rss / (1024 * 1024)
    w_ternary_test = weight_quant_cpu(torch.randn(N, K))
    mem_after_ternary = process.memory_info().rss / (1024 * 1024)
    matmul_free_mem = mem_after_ternary - mem_before_ternary

    if matmul_free_mem > 0:
        memory_reduction = pytorch_mem / matmul_free_mem
        print(f"  Memory (actual): PyTorch {pytorch_mem:.2f} MB, MatMul-free {matmul_free_mem:.2f} MB")
        print(f"  Memory reduction: {memory_reduction:.1f}x")
    else:
        print(f"  Memory measurement: Could not accurately measure (small tensors)")

    print("  ✅ Performance comparison completed")


def main():
    """Run all tests."""
    print("=" * 80)
    print("CPU MATMUL-FREE OPERATIONS TEST SUITE")
    print("=" * 80)
    
    # Print system info
    cpu_info = get_cpu_info()
    simd_caps = detect_simd_support()
    impl = get_optimal_implementation()
    
    print(f"System: {cpu_info['system']} {cpu_info['architecture']}")
    print(f"CPU: {cpu_info['processor']}")
    print(f"Cores: {cpu_info['cpu_count']}")
    print(f"SIMD: {simd_caps.architecture} (width: {simd_caps.vector_width})")
    print(f"Implementation: {impl}")
    print()
    
    # Run tests
    try:
        test_weight_quantization()
        test_activation_quantization()
        test_matmul_free_correctness()
        test_cpu_linear_layer()
        test_performance_comparison()
        
        print("\n" + "=" * 80)
        print("🎉 ALL TESTS PASSED!")
        print("=" * 80)
        
    except Exception as e:
        print(f"\n❌ TEST FAILED: {e}")
        import traceback
        traceback.print_exc()
        return 1
    
    return 0


if __name__ == "__main__":
    exit(main())
