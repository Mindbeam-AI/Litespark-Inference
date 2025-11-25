"""
CPU-Optimized MatMul-Free Operations

This module provides CPU-optimized implementations of MatMul-free operations
with support for multiple architectures:
- Intel/AMD x86_64 (AVX2, AVX-512)
- Apple Silicon M1/M2/M3 (ARM64 + NEON)
- ARM64 Linux (NEON)
- Generic fallback implementations
"""

from .matmul_free_cpu import (
    matmul_free_cpu,
    get_cpu_info,
    get_optimal_implementation,
    CPUMatMulFreeLinear
)

from .simd import (
    detect_simd_support,
    get_simd_width,
    SIMDCapabilities
)

from .setup_extensions import get_cpu_kernels

__all__ = [
    'matmul_free_cpu',
    'get_cpu_info', 
    'get_optimal_implementation',
    'CPUMatMulFreeLinear',
    'detect_simd_support',
    'get_simd_width',
    'SIMDCapabilities',
    'get_cpu_kernels'
]

# Version info
__version__ = '0.1.0'
__cpu_branch__ = True
