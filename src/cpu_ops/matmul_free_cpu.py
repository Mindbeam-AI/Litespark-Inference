"""
CPU-Optimized MatMul-Free Linear Operations

Implements true MatMul-free operations using only addition/subtraction
for ternary weights {-1, 0, 1}, optimized for CPU architectures.
"""

import torch
import torch.nn as nn
import numpy as np
import platform
import os
from typing import Optional, Tuple, Dict, Any

# Try to import CPU-optimized kernels (C++ extensions)
try:
    from .setup_extensions import get_cpu_kernels
    cpu_kernels = get_cpu_kernels()
    HAS_OPTIMIZED_KERNELS = cpu_kernels is not None
except ImportError:
    HAS_OPTIMIZED_KERNELS = False
    cpu_kernels = None
    print("Warning: Optimized CPU kernels not available, using fallback implementation")


def get_cpu_info() -> Dict[str, Any]:
    """Get detailed CPU information for optimization selection."""
    info = {
        'platform': platform.machine(),
        'processor': platform.processor(),
        'system': platform.system(),
        'architecture': platform.architecture()[0],
        'cpu_count': os.cpu_count(),
    }
    
    # Detect specific CPU features
    if info['system'] == 'Darwin' and 'arm' in info['platform'].lower():
        info['is_apple_silicon'] = True
        info['simd_support'] = 'NEON'
    elif 'x86' in info['platform'] or 'amd64' in info['platform']:
        info['is_apple_silicon'] = False
        # TODO: Detect AVX2/AVX-512 support
        info['simd_support'] = 'AVX2'  # Default assumption
    else:
        info['is_apple_silicon'] = False
        info['simd_support'] = 'generic'
    
    return info


def get_optimal_implementation() -> str:
    """Select the best CPU implementation based on hardware."""
    cpu_info = get_cpu_info()
    
    if not HAS_OPTIMIZED_KERNELS:
        return 'python_fallback'
    
    if cpu_info['is_apple_silicon']:
        return 'arm64_neon'
    elif cpu_info['simd_support'] == 'AVX2':
        return 'x86_64_avx2'
    else:
        return 'generic_cpp'


def weight_quant_cpu(w: torch.Tensor) -> torch.Tensor:
    """
    CPU-optimized ternary weight quantization to {-1, 0, 1}.
    
    Args:
        w: Weight tensor [out_features, in_features]
        
    Returns:
        Quantized weights in {-1, 0, 1}
    """
    # Compute scale factor
    scale = 1.0 / w.abs().mean().clamp_(min=1e-5)
    
    # Quantize to ternary values
    w_scaled = w * scale
    w_ternary = torch.sign(w_scaled) * (w_scaled.abs() > 0.5).float()
    
    return w_ternary


def activation_quant_cpu(x: torch.Tensor) -> torch.Tensor:
    """
    CPU-optimized activation quantization to 8-bit.
    
    Args:
        x: Input tensor [batch_size, features]
        
    Returns:
        Quantized activations
    """
    # Per-token quantization
    scale = 127.0 / x.abs().max(dim=-1, keepdim=True).values.clamp_(min=1e-5)
    x_quant = (x * scale).round().clamp_(-128, 127) / scale
    
    return x_quant


def matmul_free_cpu_python(x: torch.Tensor, w: torch.Tensor, bias: Optional[torch.Tensor] = None) -> torch.Tensor:
    """
    Pure Python implementation of MatMul-free operation.
    Uses only addition/subtraction for ternary weights.
    
    Args:
        x: Input tensor [M, K]
        w: Ternary weight tensor [N, K] with values in {-1, 0, 1}
        bias: Optional bias tensor [N]
        
    Returns:
        Output tensor [M, N]
    """
    M, K = x.shape
    N, K_w = w.shape
    assert K == K_w, f"Dimension mismatch: {K} != {K_w}"
    
    # Convert to numpy for efficient computation
    x_np = x.detach().cpu().numpy()
    w_np = w.detach().cpu().numpy()
    
    # Initialize output
    y_np = np.zeros((M, N), dtype=np.float32)
    
    # True MatMul-free: only add/subtract operations
    for m in range(M):
        for n in range(N):
            # Sum where weight is +1, subtract where weight is -1
            pos_mask = (w_np[n, :] > 0.5)
            neg_mask = (w_np[n, :] < -0.5)
            
            y_np[m, n] = np.sum(x_np[m, pos_mask]) - np.sum(x_np[m, neg_mask])
    
    # Convert back to torch tensor
    y = torch.from_numpy(y_np).to(x.device, x.dtype)
    
    if bias is not None:
        y += bias.unsqueeze(0)
    
    return y


def matmul_free_cpu_optimized(x: torch.Tensor, w: torch.Tensor, bias: Optional[torch.Tensor] = None) -> torch.Tensor:
    """
    Call optimized C++ kernels for MatMul-free operations.

    Args:
        x: Input tensor [M, K]
        w: Ternary weight tensor [N, K] with values in {-1, 0, 1}
        bias: Optional bias tensor [N]

    Returns:
        Output tensor [M, N]
    """
    if not HAS_OPTIMIZED_KERNELS:
        raise RuntimeError("Optimized kernels not available")

    M, K = x.shape
    N, K_w = w.shape
    assert K == K_w, f"Dimension mismatch: {K} != {K_w}"

    # Ensure contiguous memory layout
    x_contig = x.contiguous()
    w_contig = w.contiguous()

    # Allocate output tensor
    y = torch.empty((M, N), dtype=x.dtype, device=x.device)

    # Determine optimal number of threads
    num_threads = min(torch.get_num_threads(), os.cpu_count() or 1)

    # Call appropriate optimized kernel based on architecture
    impl = get_optimal_implementation()

    if impl == 'arm64_neon':
        cpu_kernels.matmul_free_neon(
            x_contig.data_ptr(), w_contig.data_ptr(), y.data_ptr(),
            bias.data_ptr() if bias is not None else None,
            M, N, K, num_threads
        )
    elif impl == 'x86_64_avx2':
        cpu_kernels.matmul_free_avx2(
            x_contig.data_ptr(), w_contig.data_ptr(), y.data_ptr(),
            bias.data_ptr() if bias is not None else None,
            M, N, K, num_threads
        )
    else:  # generic_cpp
        cpu_kernels.matmul_free_generic(
            x_contig.data_ptr(), w_contig.data_ptr(), y.data_ptr(),
            bias.data_ptr() if bias is not None else None,
            M, N, K, num_threads
        )

    return y


def matmul_free_cpu(x: torch.Tensor, w: torch.Tensor, bias: Optional[torch.Tensor] = None) -> torch.Tensor:
    """
    CPU-optimized MatMul-free operation with automatic implementation selection.

    Args:
        x: Input tensor [M, K]
        w: Ternary weight tensor [N, K] with values in {-1, 0, 1}
        bias: Optional bias tensor [N]

    Returns:
        Output tensor [M, N]
    """
    if HAS_OPTIMIZED_KERNELS:
        try:
            return matmul_free_cpu_optimized(x, w, bias)
        except Exception as e:
            print(f"Warning: Optimized kernel failed ({e}), falling back to Python implementation")

    return matmul_free_cpu_python(x, w, bias)


class CPUMatMulFreeLinear(nn.Module):
    """
    CPU-optimized MatMul-free linear layer.
    
    This layer replaces traditional matrix multiplication with addition/subtraction
    operations using ternary weights, optimized for CPU execution.
    """
    
    def __init__(self, in_features: int, out_features: int, bias: bool = True):
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        
        # Initialize weights normally, will be quantized during forward pass
        self.weight = nn.Parameter(torch.randn(out_features, in_features))
        if bias:
            self.bias = nn.Parameter(torch.zeros(out_features))
        else:
            self.register_parameter('bias', None)
        
        # Store CPU info for optimization
        self.cpu_info = get_cpu_info()
        self.implementation = get_optimal_implementation()
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Quantize weights to ternary values
        w_ternary = weight_quant_cpu(self.weight)
        
        # Apply MatMul-free operation
        return matmul_free_cpu(x, w_ternary, self.bias)
    
    def extra_repr(self) -> str:
        return f'in_features={self.in_features}, out_features={self.out_features}, ' \
               f'bias={self.bias is not None}, implementation={self.implementation}'
