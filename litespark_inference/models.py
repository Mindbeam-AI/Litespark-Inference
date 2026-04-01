#!/usr/bin/env python3
"""
Ternary Model Loaders

Supports loading pre-trained ternary models:
1. BitNet b1.58 (1bitLLM/bitnet_b1_58-3B, microsoft/bitnet-b1.58-2B-4T)

Supported architectures:
- x86_64 with AVX-512 VNNI (Intel Ice Lake+, AMD Zen4+)
- x86_64 with AVX-VNNI on AVX2 (Intel Core Ultra 7/9, 12th gen+)
- aarch64/arm64 with NEON SDOT (AWS Graviton 2/3/4, Apple Silicon)
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import platform
import os
from pathlib import Path
from typing import Dict, Tuple, Optional, List
from torch.utils.cpp_extension import load
from dataclasses import dataclass
import json
import sys


class TextStreamer:
    """Streams generated tokens to stdout, decoding them incrementally."""

    def __init__(self, tokenizer, skip_special_tokens=True):
        self.tokenizer = tokenizer
        self.skip_special_tokens = skip_special_tokens
        self.token_cache = []
        self.print_len = 0

    def put(self, token_ids: torch.Tensor):
        """Decode and print new token(s) incrementally."""
        self.token_cache.extend(token_ids[0].tolist())
        text = self.tokenizer.decode(self.token_cache, skip_special_tokens=self.skip_special_tokens)
        # Only print new characters (handles multi-byte/BPE boundary issues)
        printable = text[self.print_len:]
        if printable:
            sys.stdout.write(printable)
            sys.stdout.flush()
            self.print_len = len(text)

    def end(self):
        """Flush any remaining text."""
        if self.token_cache:
            text = self.tokenizer.decode(self.token_cache, skip_special_tokens=self.skip_special_tokens)
            remaining = text[self.print_len:]
            if remaining:
                sys.stdout.write(remaining)
                sys.stdout.flush()
        self.token_cache = []
        self.print_len = 0


# ============================================================================
# Architecture Detection
# ============================================================================

def is_graviton() -> bool:
    """Detect if running on AWS Graviton."""
    try:
        # Check MIDR register for ARM implementer
        midr_path = '/sys/devices/system/cpu/cpu0/regs/identification/midr_el1'
        if os.path.exists(midr_path):
            with open(midr_path) as f:
                midr = int(f.read().strip(), 16)
                # Graviton uses ARM implementer (0x41)
                return (midr >> 24) == 0x41
    except:
        pass

    # Alternative: check /proc/cpuinfo
    try:
        with open('/proc/cpuinfo') as f:
            cpuinfo = f.read().lower()
            # Graviton 2 reports neoverse-n1, Graviton 3/4 reports neoverse-v1
            if 'neoverse' in cpuinfo:
                return True
    except:
        pass

    return False


def detect_x86_features() -> Dict[str, bool]:
    """Detect x86 CPU features (AVX-512 vs AVX-VNNI)."""
    features = {
        'avx512f': False,
        'avx512vnni': False,
        'avx_vnni': False,  # AVX-VNNI on AVX2 (Intel 12th gen+, Core Ultra)
    }

    system = platform.system()

    if system == 'Linux':
        try:
            with open('/proc/cpuinfo') as f:
                cpuinfo = f.read().lower()
                features['avx512f'] = 'avx512f' in cpuinfo
                features['avx512vnni'] = 'avx512vnni' in cpuinfo or 'avx512_vnni' in cpuinfo
                features['avx_vnni'] = 'avx_vnni' in cpuinfo
        except:
            pass
    elif system == 'Darwin':
        # macOS: use sysctl
        import subprocess
        try:
            result = subprocess.run(
                ['sysctl', '-a'],
                capture_output=True, text=True, timeout=5
            )
            sysctl_output = result.stdout.lower()
            # On macOS, CPU features are in hw.optional.*
            features['avx512f'] = 'hw.optional.avx512f: 1' in sysctl_output
            features['avx512vnni'] = 'hw.optional.avx512vnni: 1' in sysctl_output
            # AVX-VNNI (on AVX2) check - typically available on Intel 12th gen+
            features['avx_vnni'] = 'hw.optional.avx_vnni: 1' in sysctl_output
        except:
            pass
    elif system == 'Windows':
        # Windows: could use WMI or CPUID, for now assume AVX-512 if x86_64
        # TODO: add proper Windows detection
        features['avx512vnni'] = True  # Conservative assumption

    return features


def get_arch_info() -> Dict:
    """Get architecture information."""
    machine = platform.machine().lower()

    info = {
        'machine': machine,
        'is_x86_64': machine in ['x86_64', 'amd64'],
        'is_arm64': machine in ['aarch64', 'arm64'],
        'is_graviton': False,
        'is_apple_silicon': False,
        'kernel_type': None,
        'x86_features': None,
    }

    if info['is_x86_64']:
        # Detect specific x86 features
        info['x86_features'] = detect_x86_features()

        if info['x86_features']['avx512vnni']:
            info['kernel_type'] = 'vnni'  # Full AVX-512 VNNI
        elif info['x86_features']['avx_vnni']:
            info['kernel_type'] = 'avx_vnni'  # AVX-VNNI on AVX2 (Intel Ultra)
        else:
            # Fallback: assume AVX-512 VNNI (for older detection methods)
            info['kernel_type'] = 'vnni'

    elif info['is_arm64']:
        info['is_graviton'] = is_graviton()
        info['is_apple_silicon'] = platform.system() == 'Darwin'
        info['kernel_type'] = 'graviton' if info['is_graviton'] else 'neon'

    return info


# ============================================================================
# Load Architecture-Specific Kernel
# ============================================================================

_kernel = None
_kernel_type = None


def _get_litespark_cache_dir() -> Path:
    """Resolve the cache directory for quantized weights."""
    cache_override = os.environ.get('LITESPARK_CACHE_DIR')
    if cache_override:
        return Path(cache_override).expanduser()

    if platform.system() == 'Darwin':
        return Path.home() / 'Library' / 'Caches' / 'litespark'

    xdg_cache_home = os.environ.get('XDG_CACHE_HOME')
    if xdg_cache_home:
        return Path(xdg_cache_home).expanduser() / 'litespark'

    return Path.home() / '.cache' / 'litespark'


def _find_macos_openmp_paths() -> Tuple[Optional[str], Optional[str]]:
    """Find OpenMP headers and libraries on macOS."""
    candidate_prefixes = []
    for env_var in ('LITESPARK_LIBOMP_PREFIX', 'LIBOMP_PREFIX'):
        prefix = os.environ.get(env_var)
        if prefix:
            candidate_prefixes.append(Path(prefix).expanduser())

    # Next choice: PyTorch's bundled libomp (avoids duplicate runtime).
    try:
        import torch as _torch

        torch_dir = Path(_torch.__path__[0])
        candidate_prefixes.append(torch_dir)
    except ImportError:
        pass

    homebrew_prefix = os.environ.get('HOMEBREW_PREFIX')
    if homebrew_prefix:
        candidate_prefixes.append(Path(homebrew_prefix) / 'opt' / 'libomp')

    candidate_prefixes.extend([
        Path('/opt/homebrew/opt/libomp'),
        Path('/usr/local/opt/libomp'),
    ])

    for prefix in candidate_prefixes:
        omp_include = prefix / 'include' / 'omp.h'
        omp_library = prefix / 'lib' / 'libomp.dylib'
        if omp_include.exists() and omp_library.exists():
            return str(prefix / 'include'), str(prefix / 'lib')

    return None, None

def get_kernel():
    """Load the appropriate kernel for this architecture (cached)."""
    global _kernel, _kernel_type

    if _kernel is None:
        arch_info = get_arch_info()
        machine = arch_info['machine']

        if arch_info['is_x86_64']:
            if arch_info['kernel_type'] == 'avx_vnni':
                # Intel Core Ultra 7/9 and 12th gen+ with AVX-VNNI (256-bit)
                kernel_path = Path(__file__).parent / 'kernels/x86_64/matmul_free_avx_vnni.cpp'

                _kernel = load(
                    name='matmul_free_avx_vnni',
                    sources=[str(kernel_path)],
                    extra_cflags=['-O3', '-march=native', '-fopenmp', '-mavx2', '-mavxvnni'],
                    extra_ldflags=['-fopenmp'],
                    verbose=False
                )
                _kernel_type = 'avx_vnni'
            else:
                # Full AVX-512 VNNI (Intel Ice Lake+, AMD Zen4+)
                kernel_path = Path(__file__).parent / 'kernels/x86_64/matmul_free_tmac_vnni.cpp'

                _kernel = load(
                    name='matmul_free_vnni',
                    sources=[str(kernel_path)],
                    extra_cflags=['-O3', '-march=native', '-fopenmp', '-mavx512f', '-mavx512bw', '-mavx512vnni'],
                    extra_ldflags=['-fopenmp'],
                    verbose=False
                )
                _kernel_type = 'vnni'

        elif arch_info['is_arm64']:
            if arch_info['is_graviton']:
                # AWS Graviton with NEON SDOT
                kernel_path = Path(__file__).parent / 'kernels/arm64/matmul_free_graviton.cpp'

                _kernel = load(
                    name='matmul_free_graviton',
                    sources=[str(kernel_path)],
                    extra_cflags=['-O3', '-march=armv8.2-a+dotprod', '-fopenmp'],
                    extra_ldflags=['-fopenmp'],
                    verbose=False
                )
                _kernel_type = 'graviton'
            else:
                # Apple Silicon or other ARM64 with NEON SDOT
                kernel_path = Path(__file__).parent / 'kernels/arm64/matmul_free_neon_int8.cpp'

                # macOS with Apple Clang needs special OpenMP flags.
                # Always prefer PyTorch's bundled libomp to avoid duplicate
                # runtime crashes (two copies of libomp in one process segfault).
                if platform.system() == 'Darwin':
                    omp_include, omp_lib = _find_macos_openmp_paths()

                    if omp_include:
                        extra_cflags = [
                            '-O3', '-march=native',
                            '-Xclang', '-fopenmp',
                            f'-I{omp_include}',
                        ]
                        extra_ldflags = [
                            f'-L{omp_lib}',
                            '-lomp',
                            '-framework', 'Accelerate',
                        ]
                    else:
                        # No OpenMP available, compile without it (single-threaded)
                        extra_cflags = ['-O3', '-march=native', '-DDISABLE_OPENMP']
                        extra_ldflags = ['-framework', 'Accelerate']
                else:
                    # Linux ARM64 (non-Graviton)
                    extra_cflags = ['-O3', '-march=native', '-fopenmp']
                    extra_ldflags = ['-fopenmp']

                _kernel = load(
                    name='matmul_free_neon',
                    sources=[str(kernel_path)],
                    extra_cflags=extra_cflags,
                    extra_ldflags=extra_ldflags,
                    verbose=False
                )
                _kernel_type = 'neon'
        else:
            raise RuntimeError(f"Unsupported architecture: {machine}")

    return _kernel


def get_kernel_type() -> str:
    """Get the kernel type being used."""
    global _kernel_type
    if _kernel_type is None:
        get_kernel()  # This will set _kernel_type
    return _kernel_type


# ============================================================================
# Weight Preparation
# ============================================================================

def get_k_padding() -> int:
    """Get K padding alignment based on architecture."""
    kernel_type = get_kernel_type()
    if kernel_type == 'vnni':
        return 64  # AVX-512 = 512 bits = 64 bytes
    elif kernel_type == 'avx_vnni':
        return 32  # AVX2 = 256 bits = 32 bytes
    else:
        return 16  # NEON = 128 bits = 16 bytes


def quantize_activation(x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
    """Quantize float activation to int8 with per-row scaling.

    Optimized version that minimizes intermediate tensor allocations.
    """
    if x.dim() == 1:
        x = x.unsqueeze(0)

    M, K = x.shape
    k_pad = get_k_padding()
    K_padded = ((K + k_pad - 1) // k_pad) * k_pad

    # Compute scale: fused max and division
    max_abs = x.abs().max(dim=1, keepdim=True).values
    scale = max_abs / 127.0
    # Avoid division by zero - use clamp instead of where (faster)
    scale = scale.clamp(min=1e-10)

    # Fused quantization: multiply by inverse scale, round, clamp, convert
    # Using multiplication instead of division is faster
    inv_scale = 1.0 / scale

    if K_padded == K:
        # No padding needed - direct conversion
        x_int8 = (x * inv_scale).round().clamp(-127, 127).to(torch.int8)
    else:
        # Need padding - use F.pad which is optimized
        x_scaled = (x * inv_scale).round().clamp(-127, 127).to(torch.int8)
        x_int8 = F.pad(x_scaled, (0, K_padded - K), value=0)

    return x_int8.contiguous(), scale.squeeze(1).contiguous()


def prepare_ternary_weight(weight: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, float]:
    """
    Prepare weight tensor for kernel using BitNet-style quantization.

    BitNet stores weights as float and quantizes at runtime using:
        scale = 1 / weight.abs().mean()
        w_quant = round(weight * scale).clamp(-1, 1)
        output = w_quant / scale  (i.e., w_quant * (1/scale) = w_quant * weight.abs().mean())

    Args:
        weight: Float tensor [out_features, in_features]

    Returns:
        w_int8: Padded int8 tensor [N, K_padded] with values in {-1, 0, +1}
        w_sum: Row sums as int32 [N]
        scale: Scale factor = weight.abs().mean() for dequantization
    """
    N, K = weight.shape
    k_pad = get_k_padding()
    K_padded = ((K + k_pad - 1) // k_pad) * k_pad

    # BitNet quantization formula:
    # scale_inv = 1 / weight.abs().mean()
    # w_ternary = round(weight * scale_inv).clamp(-1, 1)
    # At inference: output = matmul(x, w_ternary) * (1 / scale_inv) = matmul(x, w_ternary) * scale
    weight_abs_mean = weight.abs().mean()
    if weight_abs_mean > 0:
        scale = weight_abs_mean.item()
        scale_inv = 1.0 / scale
    else:
        scale = 1.0
        scale_inv = 1.0

    # Quantize to ternary {-1, 0, +1}
    w_ternary = (weight * scale_inv).round().clamp(-1, 1).to(torch.int8)

    # Pad to kernel alignment
    w_int8 = torch.zeros(N, K_padded, dtype=torch.int8)
    w_int8[:, :K] = w_ternary

    w_sum = w_int8.sum(dim=1, dtype=torch.int32)

    return w_int8.contiguous(), w_sum.contiguous(), scale


# ============================================================================
# Ternary Linear Layer
# ============================================================================

class TernaryLinear(nn.Module):
    """Ternary linear layer with architecture-adaptive kernel selection.

    Kernel selection based on benchmarks:
    - v4_large_n: Best for large N (>= 4096), e.g., MLP up/gate projections
    - v4_fused (Graviton): Float input, fused quantize+matmul (eliminates Python overhead)
    - v3: Best for smaller N, e.g., attention projections
    - v2: Best for small layers (K <= 1024, N <= 4096)
    """

    # Pre-allocated output buffer cache (shared across instances)
    _output_cache = {}

    def __init__(
        self,
        in_features: int,
        out_features: int,
        bias: bool = True,
        num_threads: int = None
    ):
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        k_pad = get_k_padding()
        self.K_padded = ((in_features + k_pad - 1) // k_pad) * k_pad
        self.num_threads = num_threads or os.cpu_count()

        # Will be set by load_from_weight
        self.register_buffer('w_int8', None)
        self.register_buffer('w_sum', None)
        self.register_buffer('bias_tensor', None)
        self.scale = 1.0

        # Determine which kernel variant to use based on dimensions
        # v4_large_n is 4-8x faster for N >= 1024 (all major projections)
        self.use_v4_large_n = out_features >= 1024

    def load_from_weight(self, weight: torch.Tensor, bias: Optional[torch.Tensor] = None):
        """Load from a float weight tensor."""
        self.w_int8, self.w_sum, self.scale = prepare_ternary_weight(weight)
        if bias is not None:
            self.bias_tensor = bias.clone()

    def _get_output_buffer(self, M: int) -> torch.Tensor:
        """Get or create pre-allocated output buffer."""
        key = (M, self.out_features)
        if key not in TernaryLinear._output_cache:
            TernaryLinear._output_cache[key] = torch.zeros(M, self.out_features, dtype=torch.float32)
        return TernaryLinear._output_cache[key]

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass using architecture-appropriate kernel."""
        original_shape = x.shape
        x = x.view(-1, self.in_features)
        M = x.shape[0]

        # Select kernel based on architecture and dimensions
        kernel = get_kernel()
        kernel_type = get_kernel_type()

        if kernel_type == 'vnni':
            # x86_64 AVX-512 VNNI
            if M <= 4 and self.num_threads > 1:
                # Weight Parallel: fused kernel for small M (token generation)
                # Parallelizes over N instead of M for better thread utilization
                y = torch.zeros(M, self.out_features, dtype=torch.float32)
                bias = self.bias_tensor if self.bias_tensor is not None else torch.Tensor()
                kernel.matmul_free_vnni_v4_fused_wp(
                    x.contiguous(),  # Float input
                    self.w_int8, self.w_sum,
                    y, bias,
                    self.scale,  # Weight scale applied inside kernel
                    M, self.out_features, self.in_features, self.num_threads
                )
                # Note: scale and bias already applied in kernel
            else:
                # Activation Parallel: original logic for larger M
                # Quantize input
                x_int8, x_scale = quantize_activation(x)

                if self.use_v4_large_n:
                    # v4_large_n: outputs int32, ~17x faster for large N
                    y_int32 = torch.zeros(M, self.out_features, dtype=torch.int32)
                    kernel.matmul_free_vnni_v4_large_n(
                        x_int8, x_scale,
                        self.w_int8, self.w_sum,
                        y_int32,
                        M, self.out_features, self.in_features, self.num_threads
                    )
                    # Convert to float and apply scales
                    y = y_int32.float() * x_scale.unsqueeze(1) * self.scale
                elif self.in_features <= 1024 and self.out_features <= 4096:
                    y = torch.zeros(M, self.out_features, dtype=torch.float32)
                    bias = self.bias_tensor if self.bias_tensor is not None else torch.Tensor()
                    kernel.matmul_free_vnni_v2(
                        x_int8, x_scale,
                        self.w_int8, self.w_sum,
                        y, bias,
                        M, self.out_features, self.in_features, self.num_threads
                    )
                    y = y * self.scale
                else:
                    y = torch.zeros(M, self.out_features, dtype=torch.float32)
                    bias = self.bias_tensor if self.bias_tensor is not None else torch.Tensor()
                    kernel.matmul_free_vnni_v3(
                        x_int8, x_scale,
                        self.w_int8, self.w_sum,
                        y, bias,
                        M, self.out_features, self.in_features, self.num_threads
                    )
                    y = y * self.scale

                # Apply bias if present (for v4 path, bias wasn't applied in kernel)
                if self.use_v4_large_n and self.bias_tensor is not None:
                    y = y + self.bias_tensor

        elif kernel_type == 'graviton':
            # AWS Graviton NEON SDOT - use v4_fused (float input, fused quantize+matmul)
            # This eliminates Python quantization overhead
            # NOTE: Must create fresh tensor - buffer caching causes K/V to share buffers!
            y = torch.zeros(M, self.out_features, dtype=torch.float32)
            bias = self.bias_tensor if self.bias_tensor is not None else torch.Tensor()

            # v4_fused takes float input directly and handles quantization internally
            kernel.matmul_free_graviton_v4_fused(
                x.contiguous(),  # Float input
                self.w_int8, self.w_sum,
                y, bias,
                self.scale,  # Weight scale applied inside kernel
                M, self.out_features, self.in_features, self.num_threads
            )
            # Note: scale already applied in kernel, no need to multiply here

        elif kernel_type == 'avx_vnni':
            # Intel Core Ultra / 12th gen+ with AVX-VNNI (256-bit)
            # Same logic as AVX-512 VNNI but with 256-bit kernel
            x_int8, x_scale = quantize_activation(x)

            if self.use_v4_large_n:
                # v4_large_n: outputs int32, fastest for large N
                y_int32 = torch.zeros(M, self.out_features, dtype=torch.int32)
                kernel.matmul_free_avx_vnni_v4_large_n(
                    x_int8, x_scale,
                    self.w_int8, self.w_sum,
                    y_int32,
                    M, self.out_features, self.in_features, self.num_threads
                )
                # Convert to float and apply scales
                y = y_int32.float() * x_scale.unsqueeze(1) * self.scale
            else:
                y = torch.zeros(M, self.out_features, dtype=torch.float32)
                bias = self.bias_tensor if self.bias_tensor is not None else torch.Tensor()
                kernel.matmul_free_avx_vnni_v3(
                    x_int8, x_scale,
                    self.w_int8, self.w_sum,
                    y, bias,
                    M, self.out_features, self.in_features, self.num_threads
                )
                y = y * self.scale

            # Apply bias if present (for v4 path, bias wasn't applied in kernel)
            if self.use_v4_large_n and self.bias_tensor is not None:
                y = y + self.bias_tensor

        elif kernel_type == 'neon':
            # Apple Silicon / Generic ARM NEON SDOT
            # Use v4_fused kernel (float input, fused quantize+matmul)
            # This eliminates Python quantization overhead
            # NOTE: Must create fresh tensor - buffer caching causes K/V to share buffers!
            y = torch.zeros(M, self.out_features, dtype=torch.float32)
            bias = self.bias_tensor if self.bias_tensor is not None else torch.Tensor()

            if M <= 4 and self.num_threads > 1:
                # Weight Parallel: parallelize over N (output channels) for small M
                # This is faster for token generation (M=1) because we can't parallelize over M
                kernel.matmul_free_neon_sdot_v4_fused_wp(
                    x.contiguous(),  # Float input
                    self.w_int8, self.w_sum,
                    y, bias,
                    self.scale,  # Weight scale applied inside kernel
                    M, self.out_features, self.in_features, self.num_threads
                )
            else:
                # Activation Parallel: parallelize over M (rows) for large batches
                kernel.matmul_free_neon_sdot_v4_fused(
                    x.contiguous(),  # Float input
                    self.w_int8, self.w_sum,
                    y, bias,
                    self.scale,  # Weight scale applied inside kernel
                    M, self.out_features, self.in_features, self.num_threads
                )
            # Note: scale already applied in kernel, no need to multiply here
        else:
            raise RuntimeError(f"Unknown kernel type: {kernel_type}")

        # Reshape back
        output_shape = original_shape[:-1] + (self.out_features,)
        return y.view(output_shape)


# ============================================================================
# BitNet Linear Layer (with BitNet-specific architecture)
# ============================================================================

class BitNetLinear(nn.Module):
    """
    BitNet-specific linear layer.

    Same as TernaryLinear but designed for BitNet's architecture.
    Uses the same underlying VNNI/Graviton kernels.

    Kernel selection based on benchmarks:
    - v4_large_n: Best for large N (>= 4096), e.g., MLP up/gate projections
    - v4_fused (Graviton): Float input, fused quantize+matmul (eliminates Python overhead)
    - v3: Best for smaller N, e.g., attention projections
    - v2: Best for small layers (K <= 1024, N <= 4096)
    """

    # Pre-allocated output buffer cache (shared across instances)
    _output_cache = {}

    def __init__(
        self,
        in_features: int,
        out_features: int,
        bias: bool = False,
        num_threads: int = None
    ):
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        k_pad = get_k_padding()
        self.K_padded = ((in_features + k_pad - 1) // k_pad) * k_pad
        self.num_threads = num_threads or os.cpu_count()

        # Will be set by load_from_weight
        self.register_buffer('w_int8', None)
        self.register_buffer('w_sum', None)
        self.register_buffer('bias_tensor', None)
        self.scale = 1.0

        # Determine which kernel variant to use based on dimensions
        # v4_large_n is 4-8x faster for N >= 1024 (all major projections)
        self.use_v4_large_n = out_features >= 1024

    def load_from_weight(self, weight: torch.Tensor, bias: Optional[torch.Tensor] = None):
        """Load from a float weight tensor using BitNet quantization."""
        self.w_int8, self.w_sum, self.scale = prepare_ternary_weight(weight)
        if bias is not None:
            self.bias_tensor = bias.clone()

    def _get_output_buffer(self, M: int) -> torch.Tensor:
        """Get or create pre-allocated output buffer."""
        key = (M, self.out_features)
        if key not in BitNetLinear._output_cache:
            BitNetLinear._output_cache[key] = torch.zeros(M, self.out_features, dtype=torch.float32)
        return BitNetLinear._output_cache[key]

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass using architecture-appropriate kernel."""
        original_shape = x.shape
        x = x.view(-1, self.in_features)
        M = x.shape[0]

        # Select kernel based on architecture and dimensions
        kernel = get_kernel()
        kernel_type = get_kernel_type()

        if kernel_type == 'vnni':
            # x86_64 AVX-512 VNNI
            if M <= 4 and self.num_threads > 1:
                # Weight Parallel: fused kernel for small M (token generation)
                # Parallelizes over N instead of M for better thread utilization
                y = torch.zeros(M, self.out_features, dtype=torch.float32)
                bias = self.bias_tensor if self.bias_tensor is not None else torch.Tensor()
                kernel.matmul_free_vnni_v4_fused_wp(
                    x.contiguous(),  # Float input
                    self.w_int8, self.w_sum,
                    y, bias,
                    self.scale,  # Weight scale applied inside kernel
                    M, self.out_features, self.in_features, self.num_threads
                )
                # Note: scale and bias already applied in kernel
            else:
                # Activation Parallel: original logic for larger M
                # Quantize input
                x_int8, x_scale = quantize_activation(x)

                if self.use_v4_large_n:
                    # v4_large_n: outputs int32, ~17x faster for large N
                    y_int32 = torch.zeros(M, self.out_features, dtype=torch.int32)
                    kernel.matmul_free_vnni_v4_large_n(
                        x_int8, x_scale,
                        self.w_int8, self.w_sum,
                        y_int32,
                        M, self.out_features, self.in_features, self.num_threads
                    )
                    # Convert to float and apply scales
                    # v4 outputs raw int32 dot products, need to apply x_scale and weight scale
                    y = y_int32.float() * x_scale.unsqueeze(1) * self.scale
                elif self.in_features <= 1024 and self.out_features <= 4096:
                    y = torch.zeros(M, self.out_features, dtype=torch.float32)
                    bias = self.bias_tensor if self.bias_tensor is not None else torch.Tensor()
                    kernel.matmul_free_vnni_v2(
                        x_int8, x_scale,
                        self.w_int8, self.w_sum,
                        y, bias,
                        M, self.out_features, self.in_features, self.num_threads
                    )
                    y = y * self.scale
                else:
                    y = torch.zeros(M, self.out_features, dtype=torch.float32)
                    bias = self.bias_tensor if self.bias_tensor is not None else torch.Tensor()
                    kernel.matmul_free_vnni_v3(
                        x_int8, x_scale,
                        self.w_int8, self.w_sum,
                        y, bias,
                        M, self.out_features, self.in_features, self.num_threads
                    )
                    y = y * self.scale

                # Apply bias if present (for v4 path, bias wasn't applied in kernel)
                if self.use_v4_large_n and self.bias_tensor is not None:
                    y = y + self.bias_tensor

        elif kernel_type == 'graviton':
            # AWS Graviton NEON SDOT - use v4_fused (float input, fused quantize+matmul)
            # This eliminates Python quantization overhead
            # NOTE: Must create fresh tensor - buffer caching causes K/V to share buffers!
            y = torch.zeros(M, self.out_features, dtype=torch.float32)
            bias = self.bias_tensor if self.bias_tensor is not None else torch.Tensor()

            # v4_fused takes float input directly and handles quantization internally
            kernel.matmul_free_graviton_v4_fused(
                x.contiguous(),  # Float input
                self.w_int8, self.w_sum,
                y, bias,
                self.scale,  # Weight scale applied inside kernel
                M, self.out_features, self.in_features, self.num_threads
            )
            # Note: scale already applied in kernel, no need to multiply here

        elif kernel_type == 'avx_vnni':
            # Intel Core Ultra / 12th gen+ with AVX-VNNI (256-bit)
            # Same logic as AVX-512 VNNI but with 256-bit kernel
            x_int8, x_scale = quantize_activation(x)

            if self.use_v4_large_n:
                # v4_large_n: outputs int32, fastest for large N
                y_int32 = torch.zeros(M, self.out_features, dtype=torch.int32)
                kernel.matmul_free_avx_vnni_v4_large_n(
                    x_int8, x_scale,
                    self.w_int8, self.w_sum,
                    y_int32,
                    M, self.out_features, self.in_features, self.num_threads
                )
                # Convert to float and apply scales
                y = y_int32.float() * x_scale.unsqueeze(1) * self.scale
            else:
                y = torch.zeros(M, self.out_features, dtype=torch.float32)
                bias = self.bias_tensor if self.bias_tensor is not None else torch.Tensor()
                kernel.matmul_free_avx_vnni_v3(
                    x_int8, x_scale,
                    self.w_int8, self.w_sum,
                    y, bias,
                    M, self.out_features, self.in_features, self.num_threads
                )
                y = y * self.scale

            # Apply bias if present (for v4 path, bias wasn't applied in kernel)
            if self.use_v4_large_n and self.bias_tensor is not None:
                y = y + self.bias_tensor

        elif kernel_type == 'neon':
            # Apple Silicon / Generic ARM NEON SDOT
            # Use v4_fused kernel (float input, fused quantize+matmul)
            # This eliminates Python quantization overhead
            # NOTE: Must create fresh tensor - buffer caching causes K/V to share buffers!
            y = torch.zeros(M, self.out_features, dtype=torch.float32)
            bias = self.bias_tensor if self.bias_tensor is not None else torch.Tensor()

            if M <= 4 and self.num_threads > 1:
                # Weight Parallel: parallelize over N (output channels) for small M
                # This is faster for token generation (M=1) because we can't parallelize over M
                kernel.matmul_free_neon_sdot_v4_fused_wp(
                    x.contiguous(),  # Float input
                    self.w_int8, self.w_sum,
                    y, bias,
                    self.scale,  # Weight scale applied inside kernel
                    M, self.out_features, self.in_features, self.num_threads
                )
            else:
                # Activation Parallel: parallelize over M (rows) for large batches
                kernel.matmul_free_neon_sdot_v4_fused(
                    x.contiguous(),  # Float input
                    self.w_int8, self.w_sum,
                    y, bias,
                    self.scale,  # Weight scale applied inside kernel
                    M, self.out_features, self.in_features, self.num_threads
                )
            # Note: scale already applied in kernel, no need to multiply here
        else:
            raise RuntimeError(f"Unknown kernel type: {kernel_type}")

        # Reshape back
        output_shape = original_shape[:-1] + (self.out_features,)
        return y.view(output_shape)


# ============================================================================
# BitNet Linear with Apple Accelerate/AMX (float32, no quantization error)
# ============================================================================

class BitNetLinearAccelerate(nn.Module):
    """
    BitNet linear layer using Apple Accelerate framework with AMX.

    Uses float32 ternary weights (-1.0, 0.0, +1.0) and cblas_sgemm.
    - Pros: No quantization error, very fast (AMX), perfect accuracy
    - Cons: 4x memory for weights (float32 vs int8)

    Only available on Apple Silicon (macOS).
    """

    def __init__(
        self,
        in_features: int,
        out_features: int,
        bias: bool = False,
        num_threads: int = None
    ):
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.num_threads = num_threads or os.cpu_count()

        # Float32 ternary weights for Accelerate
        self.register_buffer('w_float32', None)
        self.register_buffer('bias_tensor', None)
        self.scale = 1.0

    def load_from_weight(self, weight: torch.Tensor, bias: Optional[torch.Tensor] = None):
        """Load from a float weight tensor using BitNet quantization to float32 ternary."""
        # Quantize to ternary but keep as float32
        weight_abs_mean = weight.abs().mean()
        self.scale = weight_abs_mean.item() if weight_abs_mean > 0 else 1.0

        scale_inv = 1.0 / self.scale
        w_ternary = (weight * scale_inv).round().clamp(-1, 1)

        # Store as float32 ternary (already scaled)
        self.w_float32 = (w_ternary * self.scale).float().contiguous()

        if bias is not None:
            self.bias_tensor = bias.clone()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass using Apple Accelerate cblas_sgemm."""
        original_shape = x.shape
        x = x.view(-1, self.in_features).contiguous()
        M = x.shape[0]

        kernel = get_kernel()

        # Use Accelerate with float32 weights (AMX accelerated)
        y = torch.zeros(M, self.out_features, dtype=torch.float32)
        bias = self.bias_tensor if self.bias_tensor is not None else torch.Tensor()

        kernel.matmul_free_accelerate_f32(
            x,
            self.w_float32,
            y, bias,
            M, self.out_features, self.in_features, self.num_threads
        )

        # Reshape back
        output_shape = original_shape[:-1] + (self.out_features,)
        return y.view(output_shape)


# Global mode setting for BitNet
_bitnet_mode = 'neon'  # 'neon' or 'accelerate'

def set_bitnet_mode(mode: str):
    """Set the BitNet kernel mode: 'neon' (memory efficient) or 'accelerate' (max accuracy)."""
    global _bitnet_mode
    if mode not in ('neon', 'accelerate'):
        raise ValueError(f"Invalid mode: {mode}. Must be 'neon' or 'accelerate'")
    _bitnet_mode = mode

def get_bitnet_mode() -> str:
    """Get the current BitNet kernel mode."""
    return _bitnet_mode

def create_bitnet_linear(in_features: int, out_features: int, num_threads: int = None) -> nn.Module:
    """Factory function to create BitNetLinear or BitNetLinearAccelerate based on mode."""
    if _bitnet_mode == 'accelerate' and get_kernel_type() == 'neon':
        return BitNetLinearAccelerate(in_features, out_features, num_threads=num_threads)
    else:
        return BitNetLinear(in_features, out_features, num_threads=num_threads)


# ============================================================================
# BitNet Loader (Microsoft BitNet b1.58-2B-4T architecture)
# ============================================================================

@dataclass
class BitNetConfig:
    """Configuration for Microsoft BitNet b1.58-2B-4T."""
    hidden_size: int = 2560
    num_layers: int = 30
    num_heads: int = 20  # Query heads
    num_kv_heads: int = 5  # Key/Value heads (GQA)
    intermediate_size: int = 6912
    vocab_size: int = 128256
    max_position_embeddings: int = 4096
    rms_norm_eps: float = 1e-5
    rope_theta: float = 500000.0


def relu2(x: torch.Tensor) -> torch.Tensor:
    """ReLU squared activation (BitNet uses this instead of SiLU)."""
    return F.relu(x) ** 2


class BitNetAttention(nn.Module):
    """BitNet attention with GQA, sub-norm, and KV cache support."""

    def __init__(self, config: BitNetConfig, num_threads: int):
        super().__init__()
        self.hidden_size = config.hidden_size
        self.num_heads = config.num_heads
        self.num_kv_heads = config.num_kv_heads
        self.head_dim = config.hidden_size // config.num_heads
        self.num_kv_groups = config.num_heads // config.num_kv_heads
        self.num_threads = num_threads

        # Q projection: [hidden_size, hidden_size]
        self.q_proj = create_bitnet_linear(config.hidden_size, config.hidden_size, num_threads=num_threads)
        # K/V projections: [hidden_size, num_kv_heads * head_dim] for GQA
        kv_dim = config.num_kv_heads * self.head_dim
        self.k_proj = create_bitnet_linear(config.hidden_size, kv_dim, num_threads=num_threads)
        self.v_proj = create_bitnet_linear(config.hidden_size, kv_dim, num_threads=num_threads)
        self.o_proj = create_bitnet_linear(config.hidden_size, config.hidden_size, num_threads=num_threads)

        # BitNet-specific sub-norm after attention
        self.attn_sub_norm = nn.RMSNorm(config.hidden_size, eps=config.rms_norm_eps)

        self.scale = 1.0 / (self.head_dim ** 0.5)

        # Use SDPA only on x86_64 where it provides speedup (hurts ARM performance)
        self.use_sdpa = platform.machine() in ('x86_64', 'AMD64')

        # Precompute RoPE frequencies
        self.rope_theta = config.rope_theta
        self._init_rope(config.max_position_embeddings)

    def _init_rope(self, max_seq_len: int):
        """Initialize rotary position embeddings."""
        inv_freq = 1.0 / (self.rope_theta ** (torch.arange(0, self.head_dim, 2).float() / self.head_dim))
        self.register_buffer('inv_freq', inv_freq, persistent=False)

        t = torch.arange(max_seq_len, dtype=torch.float32)
        freqs = torch.outer(t, inv_freq)
        emb = torch.cat((freqs, freqs), dim=-1)
        self.register_buffer('cos_cached', emb.cos(), persistent=False)
        self.register_buffer('sin_cached', emb.sin(), persistent=False)

    def _apply_rope(self, x: torch.Tensor, position_ids: torch.Tensor) -> torch.Tensor:
        """Apply rotary position embeddings with position offset support."""
        # position_ids: [seq_len] positions to index
        # x: [batch, num_heads, seq_len, head_dim]
        cos = self.cos_cached[position_ids].unsqueeze(0).unsqueeze(0)  # [1, 1, seq_len, head_dim]
        sin = self.sin_cached[position_ids].unsqueeze(0).unsqueeze(0)

        # Rotate half
        x1 = x[..., :self.head_dim // 2]
        x2 = x[..., self.head_dim // 2:]
        rotated = torch.cat((-x2, x1), dim=-1)

        return x * cos + rotated * sin

    def forward(
        self,
        x: torch.Tensor,
        past_key_value: Optional[Tuple[torch.Tensor, torch.Tensor]] = None,
        use_cache: bool = False
    ) -> Tuple[torch.Tensor, Optional[Tuple[torch.Tensor, torch.Tensor]]]:
        batch, seq_len, _ = x.shape

        # Project Q, K, V
        q = self.q_proj(x)
        k = self.k_proj(x)
        v = self.v_proj(x)

        # Reshape for multi-head attention
        q = q.view(batch, seq_len, self.num_heads, self.head_dim).transpose(1, 2)
        k = k.view(batch, seq_len, self.num_kv_heads, self.head_dim).transpose(1, 2)
        v = v.view(batch, seq_len, self.num_kv_heads, self.head_dim).transpose(1, 2)

        # Determine position offset from cache
        if past_key_value is not None:
            past_len = past_key_value[0].shape[2]
        else:
            past_len = 0

        # Create position IDs for RoPE
        position_ids = torch.arange(past_len, past_len + seq_len, device=x.device)

        # Apply RoPE with correct positions
        q = self._apply_rope(q, position_ids)
        k = self._apply_rope(k, position_ids)

        # Handle KV cache
        if past_key_value is not None:
            # Concatenate with cached K, V
            k = torch.cat([past_key_value[0], k], dim=2)
            v = torch.cat([past_key_value[1], v], dim=2)

        # Store for next iteration if using cache
        new_cache = (k, v) if use_cache else None

        # Expand K/V for GQA (repeat each KV head for its group of Q heads)
        k_expanded = k.repeat_interleave(self.num_kv_groups, dim=1)
        v_expanded = v.repeat_interleave(self.num_kv_groups, dim=1)

        if self.use_sdpa:
            # Use PyTorch's optimized scaled_dot_product_attention (faster on x86_64)
            # This fuses Q@K^T, scale, softmax, @V into a single optimized kernel
            out = F.scaled_dot_product_attention(
                q, k_expanded, v_expanded,
                attn_mask=None,
                dropout_p=0.0,
                is_causal=(seq_len > 1)
            )
        else:
            # Manual attention (faster on ARM)
            attn_weights = torch.matmul(q, k_expanded.transpose(-2, -1)) * self.scale

            # Apply causal mask if needed
            if seq_len > 1:
                causal_mask = torch.triu(
                    torch.ones(seq_len, k_expanded.shape[2], device=q.device, dtype=torch.bool),
                    diagonal=k_expanded.shape[2] - seq_len + 1
                )
                attn_weights = attn_weights.masked_fill(causal_mask, float('-inf'))

            attn_weights = F.softmax(attn_weights, dim=-1)
            out = torch.matmul(attn_weights, v_expanded)

        out = out.transpose(1, 2).contiguous().view(batch, seq_len, self.hidden_size)

        # BitNet: apply sub-norm before output projection
        out = self.attn_sub_norm(out)
        return self.o_proj(out), new_cache


class BitNetMLP(nn.Module):
    """BitNet MLP with relu2 activation and sub-norm."""

    def __init__(self, config: BitNetConfig, num_threads: int):
        super().__init__()
        self.gate_proj = create_bitnet_linear(config.hidden_size, config.intermediate_size, num_threads=num_threads)
        self.up_proj = create_bitnet_linear(config.hidden_size, config.intermediate_size, num_threads=num_threads)
        self.down_proj = create_bitnet_linear(config.intermediate_size, config.hidden_size, num_threads=num_threads)

        # BitNet-specific sub-norm after gated activation
        self.ffn_sub_norm = nn.RMSNorm(config.intermediate_size, eps=config.rms_norm_eps)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # BitNet uses relu2 instead of SiLU
        gate = relu2(self.gate_proj(x))
        up = self.up_proj(x)
        hidden = gate * up

        # Apply sub-norm before down projection
        hidden = self.ffn_sub_norm(hidden)
        return self.down_proj(hidden)


class BitNetBlock(nn.Module):
    """Single BitNet transformer block with KV cache support."""

    def __init__(self, config: BitNetConfig, num_threads: int):
        super().__init__()
        self.input_norm = nn.RMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        self.attn = BitNetAttention(config, num_threads)
        self.post_attn_norm = nn.RMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        self.mlp = BitNetMLP(config, num_threads)

    def forward(
        self,
        x: torch.Tensor,
        past_key_value: Optional[Tuple[torch.Tensor, torch.Tensor]] = None,
        use_cache: bool = False
    ) -> Tuple[torch.Tensor, Optional[Tuple[torch.Tensor, torch.Tensor]]]:
        attn_out, new_cache = self.attn(self.input_norm(x), past_key_value, use_cache)
        x = x + attn_out
        x = x + self.mlp(self.post_attn_norm(x))
        return x, new_cache


class BitNet(nn.Module):
    """Microsoft BitNet b1.58-2B-4T with VNNI/Graviton kernels and KV cache."""

    def __init__(self, config: BitNetConfig, num_threads: int = None):
        super().__init__()
        self.config = config
        self.num_threads = num_threads or os.cpu_count()

        self.embed_tokens = nn.Embedding(config.vocab_size, config.hidden_size)
        self.layers = nn.ModuleList([
            BitNetBlock(config, self.num_threads)
            for _ in range(config.num_layers)
        ])
        self.norm = nn.RMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        # Note: BitNet ties embeddings to lm_head
        self.lm_head = None  # Will share embed_tokens weight
        # Half-precision copy of embedding weights for faster LM head computation
        self._lm_head_initialized = False
        self._lm_head_weight_bf16 = None
        self._lm_head_dtype = None

    def forward(
        self,
        input_ids: torch.Tensor,
        past_key_values: Optional[List[Tuple[torch.Tensor, torch.Tensor]]] = None,
        use_cache: bool = False
    ):
        x = self.embed_tokens(input_ids)

        new_caches = [] if use_cache else None

        for i, layer in enumerate(self.layers):
            past_kv = past_key_values[i] if past_key_values is not None else None
            x, new_cache = layer(x, past_kv, use_cache)
            if use_cache:
                new_caches.append(new_cache)

        x = self.norm(x)
        # Tied embeddings: use embed_tokens weight for lm_head
        # Use bfloat16 for faster LM head on x86 (4x speedup), keep float32 on ARM
        if not self._lm_head_initialized:
            import platform
            if platform.machine() not in ('arm64', 'aarch64'):
                # x86: use bfloat16 for 4x faster LM head
                self._lm_head_weight_bf16 = self.embed_tokens.weight.to(torch.bfloat16)
                self._lm_head_dtype = torch.bfloat16
            self._lm_head_initialized = True

        if self._lm_head_dtype is not None:
            logits = F.linear(x.to(self._lm_head_dtype), self._lm_head_weight_bf16).to(x.dtype)
        else:
            logits = F.linear(x, self.embed_tokens.weight)

        # Return tuple only when using cache, otherwise just logits for backward compat
        if use_cache:
            return logits, new_caches
        return logits

    def generate(
        self,
        input_ids: torch.Tensor,
        max_new_tokens: int = 50,
        temperature: float = 1.0,
        top_k: int = 50,
        top_p: float = 0.9,
        do_sample: bool = None,
        no_repeat_ngram_size: int = 3,
        eos_token_id: int = 128009,
        streamer=None,
    ) -> torch.Tensor:
        """Generate tokens with KV cache for efficient inference."""
        # Clamp temperature to minimum 0.1 — greedy decoding degenerates with this model
        if temperature < 0.1:
            temperature = 0.1
        if do_sample is None:
            do_sample = True

        generated = input_ids.clone()
        past_key_values = None

        for _ in range(max_new_tokens):
            # Only pass the new token(s) if we have cache
            if past_key_values is not None:
                current_input = generated[:, -1:]
            else:
                current_input = generated

            # Forward pass with cache
            logits, past_key_values = self.forward(
                current_input,
                past_key_values=past_key_values,
                use_cache=True
            )

            # Get logits for last token
            next_logits = logits[:, -1, :]

            # Block repeated n-grams to prevent phrase-level repetition
            if no_repeat_ngram_size > 0 and generated.shape[1] >= no_repeat_ngram_size:
                gen_tokens = generated[0].tolist()
                ngram_len = no_repeat_ngram_size - 1
                # Get the last (n-1) tokens as the current context
                context = tuple(gen_tokens[-ngram_len:])
                # Find all previous n-grams matching this context
                for i in range(len(gen_tokens) - ngram_len):
                    if tuple(gen_tokens[i:i + ngram_len]) == context:
                        # Ban the token that would complete a repeated n-gram
                        next_logits[0, gen_tokens[i + ngram_len]] = float('-inf')

            # Apply temperature
            if temperature != 1.0:
                next_logits = next_logits / temperature

            if do_sample:
                # Top-k filtering
                if top_k > 0:
                    indices_to_remove = next_logits < torch.topk(next_logits, top_k)[0][..., -1, None]
                    next_logits[indices_to_remove] = float('-inf')

                # Top-p (nucleus) filtering
                if top_p < 1.0:
                    sorted_logits, sorted_indices = torch.sort(next_logits, descending=True)
                    cumulative_probs = torch.cumsum(F.softmax(sorted_logits, dim=-1), dim=-1)
                    sorted_indices_to_remove = cumulative_probs > top_p
                    sorted_indices_to_remove[..., 1:] = sorted_indices_to_remove[..., :-1].clone()
                    sorted_indices_to_remove[..., 0] = 0
                    indices_to_remove = sorted_indices_to_remove.scatter(1, sorted_indices, sorted_indices_to_remove)
                    next_logits[indices_to_remove] = float('-inf')

                # Sample
                probs = F.softmax(next_logits, dim=-1)
                next_token = torch.multinomial(probs, num_samples=1)
            else:
                # Greedy
                next_token = next_logits.argmax(dim=-1, keepdim=True)

            generated = torch.cat([generated, next_token], dim=1)

            if streamer is not None:
                streamer.put(next_token)

            # Stop on EOS — BitNet uses <|eot_id|> (128009) as end-of-turn
            if next_token.item() == eos_token_id:
                break

        if streamer is not None:
            streamer.end()

        return generated

    @classmethod
    def from_pretrained(cls, model_name: str, num_threads: int = None):
        """Load from safetensors (Microsoft BitNet bf16 format) with ternary cache."""
        from huggingface_hub import hf_hub_download
        import json

        print(f"Loading {model_name}...")

        # Download config
        config_path = hf_hub_download(model_name, 'config.json')
        with open(config_path) as f:
            hf_config = json.load(f)

        config = BitNetConfig(
            hidden_size=hf_config['hidden_size'],
            num_layers=hf_config['num_hidden_layers'],
            num_heads=hf_config['num_attention_heads'],
            num_kv_heads=hf_config.get('num_key_value_heads', hf_config['num_attention_heads']),
            intermediate_size=hf_config['intermediate_size'],
            vocab_size=hf_config['vocab_size'],
            max_position_embeddings=hf_config.get('max_position_embeddings', 4096),
            rms_norm_eps=hf_config.get('rms_norm_eps', 1e-5),
            rope_theta=hf_config.get('rope_theta', 500000.0),
        )

        model = cls(config, num_threads)

        # Check for cached ternary weights
        cache_dir = _get_litespark_cache_dir()
        model_slug = model_name.replace('/', '_')
        cache_path = cache_dir / f'{model_slug}_ternary.pt'

        if cache_path.exists():
            print("Loading cached ternary weights...")
            model._load_cached_weights(str(cache_path))
        else:
            print("Loading model weights...")
            model_path = hf_hub_download(model_name, 'model.safetensors')

            print("Converting weights to ternary format...")
            model._load_safetensors_weights(model_path)

            print("Caching ternary weights for faster loading next time...")
            cache_dir.mkdir(parents=True, exist_ok=True)
            model._save_cached_weights(str(cache_path))

        return model

    def _load_safetensors_weights(self, model_path: str):
        """Load weights from safetensors file and quantize to ternary."""
        from safetensors import safe_open

        with safe_open(model_path, framework="pt") as f:
            # Embeddings (kept as float)
            self.embed_tokens.weight.data = f.get_tensor('model.embed_tokens.weight').float()

            # Layers
            for i, layer in enumerate(self.layers):
                prefix = f'model.layers.{i}'

                # Attention projections (quantize to ternary)
                layer.attn.q_proj.load_from_weight(f.get_tensor(f'{prefix}.self_attn.q_proj.weight').float())
                layer.attn.k_proj.load_from_weight(f.get_tensor(f'{prefix}.self_attn.k_proj.weight').float())
                layer.attn.v_proj.load_from_weight(f.get_tensor(f'{prefix}.self_attn.v_proj.weight').float())
                layer.attn.o_proj.load_from_weight(f.get_tensor(f'{prefix}.self_attn.o_proj.weight').float())

                # Attention sub-norm
                layer.attn.attn_sub_norm.weight.data = f.get_tensor(f'{prefix}.self_attn.attn_sub_norm.weight').float()

                # MLP projections (quantize to ternary)
                layer.mlp.gate_proj.load_from_weight(f.get_tensor(f'{prefix}.mlp.gate_proj.weight').float())
                layer.mlp.up_proj.load_from_weight(f.get_tensor(f'{prefix}.mlp.up_proj.weight').float())
                layer.mlp.down_proj.load_from_weight(f.get_tensor(f'{prefix}.mlp.down_proj.weight').float())

                # MLP sub-norm
                layer.mlp.ffn_sub_norm.weight.data = f.get_tensor(f'{prefix}.mlp.ffn_sub_norm.weight').float()

                # Block norms
                layer.input_norm.weight.data = f.get_tensor(f'{prefix}.input_layernorm.weight').float()
                layer.post_attn_norm.weight.data = f.get_tensor(f'{prefix}.post_attention_layernorm.weight').float()

            # Final norm
            self.norm.weight.data = f.get_tensor('model.norm.weight').float()
            # lm_head is tied to embed_tokens, no separate weight

    def _save_cached_weights(self, cache_path: str):
        """Save quantized ternary weights to disk for fast reloading."""
        state = {
            'embed_tokens': self.embed_tokens.weight.data,
            'norm': self.norm.weight.data,
            'layers': [],
        }
        for layer in self.layers:
            layer_state = {
                'q_proj': (layer.attn.q_proj.w_int8, layer.attn.q_proj.w_sum, layer.attn.q_proj.scale),
                'k_proj': (layer.attn.k_proj.w_int8, layer.attn.k_proj.w_sum, layer.attn.k_proj.scale),
                'v_proj': (layer.attn.v_proj.w_int8, layer.attn.v_proj.w_sum, layer.attn.v_proj.scale),
                'o_proj': (layer.attn.o_proj.w_int8, layer.attn.o_proj.w_sum, layer.attn.o_proj.scale),
                'attn_sub_norm': layer.attn.attn_sub_norm.weight.data,
                'gate_proj': (layer.mlp.gate_proj.w_int8, layer.mlp.gate_proj.w_sum, layer.mlp.gate_proj.scale),
                'up_proj': (layer.mlp.up_proj.w_int8, layer.mlp.up_proj.w_sum, layer.mlp.up_proj.scale),
                'down_proj': (layer.mlp.down_proj.w_int8, layer.mlp.down_proj.w_sum, layer.mlp.down_proj.scale),
                'ffn_sub_norm': layer.mlp.ffn_sub_norm.weight.data,
                'input_norm': layer.input_norm.weight.data,
                'post_attn_norm': layer.post_attn_norm.weight.data,
            }
            state['layers'].append(layer_state)
        torch.save(state, cache_path)

    def _load_cached_weights(self, cache_path: str):
        """Load pre-quantized ternary weights from cache."""
        state = torch.load(cache_path, weights_only=True)

        self.embed_tokens.weight.data = state['embed_tokens']
        self.norm.weight.data = state['norm']

        for i, layer in enumerate(self.layers):
            ls = state['layers'][i]
            # Attention projections
            layer.attn.q_proj.w_int8, layer.attn.q_proj.w_sum, layer.attn.q_proj.scale = ls['q_proj']
            layer.attn.k_proj.w_int8, layer.attn.k_proj.w_sum, layer.attn.k_proj.scale = ls['k_proj']
            layer.attn.v_proj.w_int8, layer.attn.v_proj.w_sum, layer.attn.v_proj.scale = ls['v_proj']
            layer.attn.o_proj.w_int8, layer.attn.o_proj.w_sum, layer.attn.o_proj.scale = ls['o_proj']
            layer.attn.attn_sub_norm.weight.data = ls['attn_sub_norm']
            # MLP projections
            layer.mlp.gate_proj.w_int8, layer.mlp.gate_proj.w_sum, layer.mlp.gate_proj.scale = ls['gate_proj']
            layer.mlp.up_proj.w_int8, layer.mlp.up_proj.w_sum, layer.mlp.up_proj.scale = ls['up_proj']
            layer.mlp.down_proj.w_int8, layer.mlp.down_proj.w_sum, layer.mlp.down_proj.scale = ls['down_proj']
            layer.mlp.ffn_sub_norm.weight.data = ls['ffn_sub_norm']
            # Norms
            layer.input_norm.weight.data = ls['input_norm']
            layer.post_attn_norm.weight.data = ls['post_attn_norm']


# ============================================================================
# Model Registry
# ============================================================================

AVAILABLE_MODELS = {
    # Microsoft BitNet (official release, bf16 weights)
    'bitnet-2b': ('microsoft/bitnet-b1.58-2B-4T-bf16', BitNet),
}


def load_ternary_model(model_key: str, num_threads: int = None, mode: str = 'neon'):
    """
    Load a pre-trained ternary model.

    Args:
        model_key: One of the keys in AVAILABLE_MODELS
        num_threads: Number of threads for kernel
        mode: Kernel mode for Apple Silicon:
              - 'neon': NEON SDOT with int8 quantization (memory efficient, ~2GB)
              - 'accelerate': Apple Accelerate/AMX with float32 (max accuracy, ~8GB)
              Only affects Apple Silicon. x86 always uses VNNI.

    Returns:
        model: Loaded model with appropriate kernels
        tokenizer: HuggingFace tokenizer
    """
    import transformers
    from transformers import AutoTokenizer, LlamaTokenizer
    transformers.logging.set_verbosity_error()

    if model_key not in AVAILABLE_MODELS:
        raise ValueError(f"Unknown model: {model_key}. Available: {list(AVAILABLE_MODELS.keys())}")

    # Set mode before loading (affects which linear layers are created)
    if mode in ('neon', 'accelerate'):
        set_bitnet_mode(mode)
    else:
        raise ValueError(f"Invalid mode: {mode}. Must be 'neon' or 'accelerate'")

    hf_name, model_class = AVAILABLE_MODELS[model_key]

    mode_str = f", mode={mode}" if get_kernel_type() == 'neon' else ""
    print(f"Loading model: {model_key} ({hf_name}{mode_str})")
    model = model_class.from_pretrained(hf_name, num_threads)
    model.eval()

    print("Loading tokenizer...")
    # BitNet uses a custom tokenizer class that may not load with AutoTokenizer
    # Use LlamaTokenizer directly with the tokenizer.model file
    try:
        tokenizer = AutoTokenizer.from_pretrained(hf_name, trust_remote_code=True)
    except (ValueError, OSError, ImportError) as e:
        print(f"  AutoTokenizer failed: {e}")
        print("  Loading tokenizer.model directly with LlamaTokenizer...")
        try:
            # BitNet tokenizer is sentencepiece-based, compatible with LlamaTokenizer
            tokenizer = LlamaTokenizer.from_pretrained(hf_name)
        except Exception as e2:
            print(f"  LlamaTokenizer failed: {e2}")
            # Last resort: use the tokenizer.json with a generic tokenizer
            from transformers import PreTrainedTokenizerFast
            tokenizer = PreTrainedTokenizerFast.from_pretrained(hf_name)

    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    return model, tokenizer


def list_models():
    """List available models."""
    print("Available ternary models:")
    print("-" * 60)
    for key, (hf_name, model_class) in AVAILABLE_MODELS.items():
        print(f"  {key:20s} -> {hf_name}")
    print()


if __name__ == '__main__':
    list_models()
