import pytest
import torch
import numpy as np
import time
from memory_profiler import memory_usage

from src.cpu_ops.matmul_free_cpu import matmul_free_cpu, weight_quant_cpu

# List of all available kernels
KERNELS = [
    "python",
    "generic",
    "neon",
    "neon_tmac",
    "neon_packed",
    "neon_packed_simd",
    "neon_packed2bit",
    "neon_optimized",
    "neon_ultra",
    "avx2",
]

# Test parameters
@pytest.mark.parametrize("kernel", KERNELS)
@pytest.mark.parametrize("M", [1, 2, 4])
@pytest.mark.parametrize("N", [128, 256, 512])
@pytest.mark.parametrize("K", [256, 512, 1024])
def test_correctness(kernel, M, N, K):
    """
    Test the correctness of the matrix multiplication results for all kernels.
    """
    # Create random input tensors
    x = torch.randn(M, K)
    w = torch.randn(N, K)

    # Quantize weights to ternary values
    w_ternary = weight_quant_cpu(w)

    # Compute the expected result using torch.matmul
    expected_y = torch.matmul(x, w_ternary.T)

    # Compute the actual result using the custom kernel
    try:
        actual_y = matmul_free_cpu(x, w_ternary, kernel=kernel)
    except Exception as e:
        pytest.skip(f"Kernel {kernel} not available: {e}")

    # Compare the results
    assert torch.allclose(actual_y, expected_y, atol=1e-4), f"Kernel {kernel} failed correctness test for M={M}, N={N}, K={K}"

@pytest.mark.parametrize("kernel", KERNELS)
@pytest.mark.parametrize("M, N, K", [(1, 2048, 2048)])
def test_performance(kernel, M, N, K):
    """
    Test the performance of the matrix multiplication for all kernels.
    """
    # Create random input tensors
    x = torch.randn(M, K)
    w = torch.randn(N, K)
    
    # Quantize weights to ternary values
    w_ternary = weight_quant_cpu(w)

    # Warm-up run
    for _ in range(3):
        try:
            matmul_free_cpu(x, w_ternary, kernel=kernel)
        except Exception as e:
            pytest.skip(f"Kernel {kernel} not available: {e}")

    # Benchmark the kernel
    start_time = time.time()
    for _ in range(10):
        matmul_free_cpu(x, w_ternary, kernel=kernel)
    end_time = time.time()
    
    print(f"Kernel {kernel} performance for M={M}, N={N}, K={K}: {(end_time - start_time) / 10:.6f} seconds")

@pytest.mark.parametrize("kernel", KERNELS)
@pytest.mark.parametrize("M, N, K", [(1, 2048, 2048)])
def test_memory_usage(kernel, M, N, K):
    """
    Test the memory usage of the matrix multiplication for all kernels.
    """
    # Create random input tensors
    x = torch.randn(M, K)
    w = torch.randn(N, K)
    
    # Quantize weights to ternary values
    w_ternary = weight_quant_cpu(w)

    # Measure peak memory usage
    try:
        mem_usage = memory_usage((matmul_free_cpu, (x, w_ternary), {'kernel': kernel}), max_usage=True)
        print(f"Kernel {kernel} peak memory usage for M={M}, N={N}, K={K}: {mem_usage:.2f} MB")
    except Exception as e:
        pytest.skip(f"Kernel {kernel} not available: {e}")
