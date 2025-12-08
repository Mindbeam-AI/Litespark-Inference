#!/usr/bin/env python3
"""
Full 24-Layer Transformer Forward Pass Benchmark - Native Kernels Version

Tests the complete forward pass through a transformer model with:
- 24 layers
- QKV projection + softmax (native kernel)
- Output projection + softmax (native kernel)
- MLP Gate-Up + SwiGLU (native kernel)
- MLP Down projection

All operations use native C++ kernels - NO PyTorch operations in the critical path.
Compares our ternary kernels vs PyTorch baseline.
"""

import torch
import torch.nn.functional as F
import numpy as np
import time
import json
import platform
import os
import gc
import tracemalloc
from pathlib import Path
from datetime import datetime
from torch.utils.cpp_extension import load


# ============================================================================
# Memory Tracking Utilities
# ============================================================================

def get_process_memory_mb():
    """Get current process memory usage in MB (RSS - Resident Set Size)"""
    try:
        # Linux: read from /proc/self/status
        with open('/proc/self/status', 'r') as f:
            for line in f:
                if line.startswith('VmRSS:'):
                    # Value is in kB
                    return int(line.split()[1]) / 1024.0
    except (FileNotFoundError, IOError):
        pass

    # Fallback: use resource module (works on Linux/macOS)
    try:
        import resource
        # ru_maxrss is in KB on Linux, bytes on macOS
        usage = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
        if platform.system() == 'Darwin':
            return usage / (1024 * 1024)  # bytes to MB
        else:
            return usage / 1024  # KB to MB
    except ImportError:
        return 0.0


def get_memory_stats():
    """Get comprehensive memory statistics"""
    stats = {
        'process_rss_mb': get_process_memory_mb(),
        'python_allocated_mb': 0.0,
        'python_peak_mb': 0.0,
    }

    # Get Python/tracemalloc stats if tracking is active
    if tracemalloc.is_tracing():
        current, peak = tracemalloc.get_traced_memory()
        stats['python_allocated_mb'] = current / (1024 * 1024)
        stats['python_peak_mb'] = peak / (1024 * 1024)

    return stats


class MemoryTracker:
    """Context manager for tracking memory during a code block"""

    def __init__(self, name=""):
        self.name = name
        self.start_rss = 0.0
        self.end_rss = 0.0
        self.peak_rss = 0.0
        self.start_python = 0.0
        self.peak_python = 0.0

    def __enter__(self):
        gc.collect()
        tracemalloc.start()
        self.start_rss = get_process_memory_mb()
        current, _ = tracemalloc.get_traced_memory()
        self.start_python = current / (1024 * 1024)
        return self

    def __exit__(self, *args):
        gc.collect()
        self.end_rss = get_process_memory_mb()
        current, peak = tracemalloc.get_traced_memory()
        self.peak_python = peak / (1024 * 1024)
        tracemalloc.stop()
        self.peak_rss = max(self.end_rss, self.start_rss)

    def get_results(self):
        return {
            'rss_start_mb': self.start_rss,
            'rss_end_mb': self.end_rss,
            'rss_delta_mb': self.end_rss - self.start_rss,
            'python_peak_mb': self.peak_python,
        }

# ============================================================================
# Configuration
# ============================================================================

N_LAYERS = 24

# Model dimensions (matching previous transformer configs)
HIDDEN_DIM = 2048
HEAD_DIM = 64       # Dimension per attention head
NUM_HEADS = 32      # Number of attention heads (2048 / 64 = 32)
QKV_DIM = 2560      # Q + K + V fused (could be 3*2048=6144 for equal Q,K,V or custom)
MLP_HIDDEN = 16384  # Gate-Up output (will be split for SwiGLU)
MLP_INTERMEDIATE = 8192  # After SwiGLU split

# For proper attention, we need Q, K, V dimensions
Q_DIM = HIDDEN_DIM
K_DIM = HIDDEN_DIM
V_DIM = HIDDEN_DIM

# Test batch sizes
BATCH_SIZES = [1, 32, 128]

# Number of warmup and benchmark iterations
WARMUP_ITERS = 5
BENCH_ITERS = 20

# ============================================================================
# Check Architecture
# ============================================================================

machine = platform.machine().lower()
if machine in ['arm64', 'aarch64']:
    ARCH = 'arm64'
    print("Detected ARM64 architecture (Apple Silicon)")
    print("WARNING: Native softmax/swiglu kernels only implemented for x86_64")
elif machine in ['x86_64', 'amd64']:
    ARCH = 'x86_64'
    print("Detected x86_64 architecture")
else:
    raise RuntimeError(f"Unsupported architecture: {machine}")

print(f"PyTorch version: {torch.__version__}")
print(f"Number of threads: {torch.get_num_threads()}")
print()

# ============================================================================
# Load Kernels
# ============================================================================

print("Loading kernels...")

if ARCH == 'arm64':
    raise RuntimeError("Native kernels not yet implemented for ARM64. Use test_full_forward_backup.py")
else:
    # Load x86_64 VNNI kernels with native softmax/swiglu
    extra_cflags = ['-O3', '-march=native', '-ffast-math', '-fopenmp']
    extra_ldflags = ['-lgomp']

    kernel = load(
        name="vnni_kernel_native",
        sources=["src/cpu_ops/kernels/x86_64/matmul_free_tmac_vnni.cpp"],
        extra_cflags=extra_cflags,
        extra_ldflags=extra_ldflags,
        verbose=False
    )

    KERNEL_NAME = "VNNI v3 Native"
    num_threads = torch.get_num_threads()

    # Limit threads for better small batch performance
    if num_threads > 8:
        num_threads = 8  # Reduce thread overhead

print(f"Loaded kernel: {KERNEL_NAME}")
print(f"Using {num_threads} threads")
print()

# ============================================================================
# Activation Functions (PyTorch baseline only)
# ============================================================================

def softmax_pytorch(x):
    """Standard softmax activation for PyTorch baseline"""
    return F.softmax(x, dim=-1)


def swiglu_pytorch(x):
    """SwiGLU activation for PyTorch baseline"""
    x1, x2 = x.chunk(2, dim=-1)
    return F.silu(x1) * x2


# ============================================================================
# Weight Preparation
# ============================================================================

def prepare_ternary_weights(N, K):
    """Create random ternary weights and prepare for kernel"""
    # Random ternary weights: -1, 0, +1 as float32
    w_float = torch.randn(K, N, dtype=torch.float32)
    w_ternary = torch.zeros_like(w_float)
    w_ternary[w_float > 0.5] = 1.0
    w_ternary[w_float < -0.5] = -1.0

    # For VNNI: kernel expects [N, K] layout, so transpose
    w_for_kernel = w_ternary.t().contiguous()  # [K, N] -> [N, K]
    K_padded = ((K + 63) // 64) * 64
    w_int8_vnni = torch.zeros(N, K_padded, dtype=torch.int8)
    w_sum_vnni = torch.zeros(N, dtype=torch.int32)
    kernel.pack_weights_vnni(w_for_kernel, w_int8_vnni, w_sum_vnni, N, K)
    return w_ternary, w_int8_vnni, w_sum_vnni


def quantize_activations_initial(x):
    """Quantize float activations to int8 (initial input only)"""
    M, K = x.shape
    scales = x.abs().max(dim=1, keepdim=True)[0] / 127.0
    scales = scales.clamp(min=1e-8)
    x_int8 = (x / scales).round().clamp(-127, 127).to(torch.int8)
    # Keep scales as 1D tensor [M], don't squeeze to scalar when M=1
    return x_int8, scales.view(M)


# ============================================================================
# Native Kernel Wrappers
# ============================================================================

def matmul_int32_out(x_int8, scales, w_int8, w_sum, M, N, K):
    """
    Run ternary matmul with int32 output (no float conversion).
    Output stays in int32 for chaining with softmax/swiglu.

    Kernel selection:
    - v4: Large N (parallelizes over N blocks) - good for up_matmul
    - v5: Large M (parallelizes over M×N tiles) - good for down_matmul
    - v3: Default for small M and small N
    """
    y_int32 = torch.zeros(M, N, dtype=torch.int32)

    if N >= 4096:
        # Large N: use v4 (parallelizes over N blocks)
        kernel.matmul_free_vnni_v4_large_n(
            x_int8, scales, w_int8, w_sum, y_int32, M, N, K, num_threads
        )
    elif M >= 64:
        # Large M: use v5 (parallelizes over M×N tiles)
        kernel.matmul_free_vnni_v5_large_m(
            x_int8, scales, w_int8, w_sum, y_int32, M, N, K, num_threads
        )
    else:
        # Small M and N: use v3
        bias = torch.empty(0)
        kernel.matmul_free_vnni_v3_int32_out(
            x_int8, scales, w_int8, w_sum, y_int32, bias, M, N, K, num_threads
        )

    return y_int32


def matmul_float_out(x_int8, scales, w_int8, w_sum, bias, M, N, K):
    """
    Run ternary matmul with float32 output (final output or residual).
    """
    y = torch.zeros(M, N, dtype=torch.float32)

    kernel.matmul_free_vnni_v3(
        x_int8, scales, w_int8, w_sum, y, bias, M, N, K, num_threads
    )

    return y


def native_softmax(x_int32, x_scales, M, N):
    """
    Native softmax: int32 input -> int8 output
    """
    y_int8 = torch.zeros(M, N, dtype=torch.int8)
    y_scales = torch.zeros(M, dtype=torch.float32)

    kernel.softmax_int8(x_int32, x_scales, y_int8, y_scales, M, N, num_threads)

    return y_int8, y_scales


def native_swiglu(x_int32, x_scales, M, N):
    """
    Native SwiGLU: int32 input [M, N] -> int8 output [M, N/2]
    """
    half_N = N // 2
    y_int8 = torch.zeros(M, half_N, dtype=torch.int8)
    y_scales = torch.zeros(M, dtype=torch.float32)

    kernel.swiglu_int8(x_int32, x_scales, y_int8, y_scales, M, N, num_threads)

    return y_int8, y_scales


def native_attention(q_int8, k_int8, v_int8, q_scales, k_scales, v_scales, M, num_heads, head_dim, scale):
    """
    Native multi-head attention in int8 using Flash Attention style tiling.
    Q @ K^T -> softmax -> @ V, all in int8/int32.
    """
    hidden_dim = num_heads * head_dim
    out_int8 = torch.zeros(M, hidden_dim, dtype=torch.int8)
    out_scales = torch.zeros(M, dtype=torch.float32)

    # For large M, use v2 kernel that parallelizes over M×heads
    # This gives better thread utilization (M×heads work units vs just M)
    if M >= 64:
        kernel.attention_int8_v2(
            q_int8, k_int8, v_int8,
            q_scales, k_scales, v_scales,
            out_int8, out_scales,
            M, num_heads, head_dim, scale, num_threads
        )
    else:
        # For small M, use Flash Attention style kernel
        kernel.attention_int8_flash(
            q_int8, k_int8, v_int8,
            q_scales, k_scales, v_scales,
            out_int8, out_scales,
            M, num_heads, head_dim, scale, num_threads
        )

    return out_int8, out_scales


def native_fused_residual(x1_int8, x1_scales, x2_int8, x2_scales, M, N):
    """
    Fused residual add + quantize: y = x1 + x2, quantized to int8.
    """
    y_int8 = torch.zeros(M, N, dtype=torch.int8)
    y_scales = torch.zeros(M, dtype=torch.float32)

    kernel.fused_residual_quantize(
        x1_int8, x1_scales, x2_int8, x2_scales,
        y_int8, y_scales, M, N, num_threads
    )

    return y_int8, y_scales


def native_quantize_int32(x_int32, x_scales, M, N):
    """
    Quantize int32 matmul output to int8.
    """
    y_int8 = torch.zeros(M, N, dtype=torch.int8)
    y_scales = torch.zeros(M, dtype=torch.float32)

    kernel.quantize_int32_to_int8(x_int32, x_scales, y_int8, y_scales, M, N, num_threads)

    return y_int8, y_scales


def native_fused_mlp(x_int8, x_scales, w_up, w_up_sum, w_down, w_down_sum, M, hidden_dim, mlp_hidden):
    """
    Fused MLP: up projection + SwiGLU + down projection in one kernel.
    Eliminates intermediate memory writes for better performance on large M.
    """
    y_int8 = torch.zeros(M, hidden_dim, dtype=torch.int8)
    y_scales = torch.zeros(M, dtype=torch.float32)

    kernel.fused_mlp_int8(
        x_int8, x_scales,
        w_up, w_up_sum,
        w_down, w_down_sum,
        y_int8, y_scales,
        M, hidden_dim, mlp_hidden,
        num_threads
    )

    return y_int8, y_scales


# ============================================================================
# Fused Kernel Wrappers (NEW - matmul+softmax, QKV fusion, matmul+quantize)
# ============================================================================

def fused_matmul_softmax(x_int8, x_scales, w_int8, w_sum, M, N, K):
    """
    Fused matmul + softmax: int8 input -> int8 output (never writes int32 to memory).
    Uses v3-based tiling for cache efficiency.
    """
    y_int8 = torch.zeros(M, N, dtype=torch.int8)
    y_scales = torch.zeros(M, dtype=torch.float32)

    kernel.matmul_free_vnni_v3_fused_softmax(
        x_int8, x_scales, w_int8, w_sum,
        y_int8, y_scales,
        M, N, K, num_threads
    )

    return y_int8, y_scales


def fused_qkv_softmax(x_int8, x_scales, wq, wq_sum, wk, wk_sum, wv, wv_sum, M, N, K):
    """
    Fused Q,K,V projection + softmax: reads input once, computes all three.
    Uses v3-based tiling for cache efficiency.
    Returns q, k, v as int8 with softmax already applied.
    """
    q_int8 = torch.zeros(M, N, dtype=torch.int8)
    q_scales = torch.zeros(M, dtype=torch.float32)
    k_int8 = torch.zeros(M, N, dtype=torch.int8)
    k_scales = torch.zeros(M, dtype=torch.float32)
    v_int8 = torch.zeros(M, N, dtype=torch.int8)
    v_scales = torch.zeros(M, dtype=torch.float32)

    kernel.matmul_free_vnni_v3_fused_qkv_softmax(
        x_int8, x_scales,
        wq, wq_sum, wk, wk_sum, wv, wv_sum,
        q_int8, q_scales, k_int8, k_scales, v_int8, v_scales,
        M, N, K, num_threads
    )

    return q_int8, q_scales, k_int8, k_scales, v_int8, v_scales


def fused_matmul_quantize(x_int8, x_scales, w_int8, w_sum, M, N, K):
    """
    Fused matmul + quantize: int8 input -> int8 output (never writes int32 to memory).
    Uses v3-based tiling for cache efficiency.
    """
    y_int8 = torch.zeros(M, N, dtype=torch.int8)
    y_scales = torch.zeros(M, dtype=torch.float32)

    kernel.matmul_free_vnni_v3_fused_quantize(
        x_int8, x_scales, w_int8, w_sum,
        y_int8, y_scales,
        M, N, K, num_threads
    )

    return y_int8, y_scales


def fused_matmul_swiglu(x_int8, x_scales, w_int8, w_sum, M, N, K):
    """
    Fused matmul + SwiGLU: int8 input -> int8 output (never writes int32 to memory).
    Input: x[M, K], weights W_up[N, K] where N = mlp_hidden (16384)
    Output: y[M, N/2] int8 (8192 outputs after SwiGLU)
    """
    half_N = N // 2
    y_int8 = torch.zeros(M, half_N, dtype=torch.int8)
    y_scales = torch.zeros(M, dtype=torch.float32)

    kernel.matmul_free_vnni_v3_fused_swiglu(
        x_int8, x_scales, w_int8, w_sum,
        y_int8, y_scales,
        M, N, K, num_threads
    )

    return y_int8, y_scales


# Flag to enable/disable fused MLP (for A/B testing)
# Fused MLP is better for large M (memory bound), separate kernels better for small M
USE_FUSED_MLP = False  # DISABLED - fused kernel is slower than separate kernels
FUSED_MLP_MIN_M = 32  # Only use fused kernel when M >= this threshold

# Flag to enable new fusion strategies
USE_FUSED_MATMUL_SOFTMAX = True  # Fuse matmul + softmax (eliminates int32 writes)
USE_FUSED_QKV = True  # Fuse Q,K,V projections (reads input once)
USE_FUSED_MATMUL_QUANTIZE = True  # Fuse down matmul + quantize
USE_FUSED_MATMUL_SWIGLU = False  # DISABLED - two-pass still slower than separate kernels


# ============================================================================
# Native Transformer Layer
# ============================================================================

class TransformerLayerNative:
    """
    Single transformer layer using ALL native kernels.
    No PyTorch operations in the forward pass except tensor reshaping.
    """

    def __init__(self, hidden_dim, qkv_dim, mlp_hidden, num_heads, head_dim):
        self.hidden_dim = hidden_dim
        self.qkv_dim = qkv_dim
        self.mlp_hidden = mlp_hidden
        self.mlp_intermediate = mlp_hidden // 2
        self.num_heads = num_heads
        self.head_dim = head_dim
        self.scale = 1.0 / (head_dim ** 0.5)

        # Create ternary weights for projections
        # Q projection: [hidden_dim, hidden_dim]
        self.w_q_ternary, self.w_q, self.w_q_sum = prepare_ternary_weights(hidden_dim, hidden_dim)
        # K projection: [hidden_dim, hidden_dim]
        self.w_k_ternary, self.w_k, self.w_k_sum = prepare_ternary_weights(hidden_dim, hidden_dim)
        # V projection: [hidden_dim, hidden_dim]
        self.w_v_ternary, self.w_v, self.w_v_sum = prepare_ternary_weights(hidden_dim, hidden_dim)
        # Output projection: [hidden_dim, hidden_dim]
        self.w_o_ternary, self.w_o, self.w_o_sum = prepare_ternary_weights(hidden_dim, hidden_dim)
        # MLP projections
        self.w_up_ternary, self.w_up, self.w_up_sum = prepare_ternary_weights(mlp_hidden, hidden_dim)
        self.w_down_ternary, self.w_down, self.w_down_sum = prepare_ternary_weights(hidden_dim, self.mlp_intermediate)

        # Biases (zeros)
        self.bias_o = torch.zeros(hidden_dim, dtype=torch.float32)
        self.bias_down = torch.zeros(hidden_dim, dtype=torch.float32)

    def forward(self, x_int8, x_scales):
        """
        Forward pass through one transformer layer.
        Input: x_int8 [M, hidden_dim] int8, x_scales [M] float32
        Output: y_int8 [M, hidden_dim] int8, y_scales [M] float32

        ALL operations in int8/int32 - NO float32 except for scales.
        Uses fused kernels when enabled to reduce memory traffic.
        """
        M = x_int8.shape[0]

        # === ATTENTION BLOCK ===

        if USE_FUSED_QKV:
            # Fused Q,K,V projection + softmax: reads input once
            q_int8_sm, q_scales_sm, k_int8_sm, k_scales_sm, v_int8_sm, v_scales_sm = fused_qkv_softmax(
                x_int8, x_scales,
                self.w_q, self.w_q_sum,
                self.w_k, self.w_k_sum,
                self.w_v, self.w_v_sum,
                M, self.hidden_dim, self.hidden_dim
            )
        elif USE_FUSED_MATMUL_SOFTMAX:
            # Fused matmul + softmax for each projection separately
            q_int8_sm, q_scales_sm = fused_matmul_softmax(
                x_int8, x_scales, self.w_q, self.w_q_sum,
                M, self.hidden_dim, self.hidden_dim
            )
            k_int8_sm, k_scales_sm = fused_matmul_softmax(
                x_int8, x_scales, self.w_k, self.w_k_sum,
                M, self.hidden_dim, self.hidden_dim
            )
            v_int8_sm, v_scales_sm = fused_matmul_softmax(
                x_int8, x_scales, self.w_v, self.w_v_sum,
                M, self.hidden_dim, self.hidden_dim
            )
        else:
            # Original path: separate matmul and softmax
            # Q Projection: [M, hidden_dim] -> [M, hidden_dim] int32
            q_int32 = matmul_int32_out(x_int8, x_scales, self.w_q, self.w_q_sum,
                                       M, self.hidden_dim, self.hidden_dim)

            # K Projection: [M, hidden_dim] -> [M, hidden_dim] int32
            k_int32 = matmul_int32_out(x_int8, x_scales, self.w_k, self.w_k_sum,
                                       M, self.hidden_dim, self.hidden_dim)

            # V Projection: [M, hidden_dim] -> [M, hidden_dim] int32
            v_int32 = matmul_int32_out(x_int8, x_scales, self.w_v, self.w_v_sum,
                                       M, self.hidden_dim, self.hidden_dim)

            # Apply softmax to Q, K, V (as per requirement: softmax after QKV)
            q_int8_sm, q_scales_sm = native_softmax(q_int32, x_scales, M, self.hidden_dim)
            k_int8_sm, k_scales_sm = native_softmax(k_int32, x_scales, M, self.hidden_dim)
            v_int8_sm, v_scales_sm = native_softmax(v_int32, x_scales, M, self.hidden_dim)

        # Native attention: Q @ K^T -> softmax -> @ V (all in int8/int32)
        attn_int8, attn_scales = native_attention(
            q_int8_sm, k_int8_sm, v_int8_sm,
            q_scales_sm, k_scales_sm, v_scales_sm,
            M, self.num_heads, self.head_dim, self.scale
        )

        # Output Projection: [M, hidden_dim] -> int8 with softmax
        if USE_FUSED_MATMUL_SOFTMAX:
            o_int8, o_scales = fused_matmul_softmax(
                attn_int8, attn_scales, self.w_o, self.w_o_sum,
                M, self.hidden_dim, self.hidden_dim
            )
        else:
            o_int32 = matmul_int32_out(attn_int8, attn_scales, self.w_o, self.w_o_sum,
                                       M, self.hidden_dim, self.hidden_dim)
            o_int8, o_scales = native_softmax(o_int32, attn_scales, M, self.hidden_dim)

        # Fused residual: x + o -> int8 (no float32 intermediate)
        x_res_int8, x_res_scales = native_fused_residual(
            x_int8, x_scales, o_int8, o_scales, M, self.hidden_dim
        )

        # === MLP BLOCK ===

        if USE_FUSED_MLP and M >= FUSED_MLP_MIN_M:
            # Fused MLP: up + swiglu + down in one kernel (better for large M)
            mlp_down_int8, mlp_down_scales = native_fused_mlp(
                x_res_int8, x_res_scales,
                self.w_up, self.w_up_sum,
                self.w_down, self.w_down_sum,
                M, self.hidden_dim, self.mlp_hidden
            )
        else:
            # Separate kernels (original path)
            if USE_FUSED_MATMUL_SWIGLU:
                # Fused matmul + SwiGLU: eliminates 16384 int32 write/read
                mlp_act_int8, mlp_act_scales = fused_matmul_swiglu(
                    x_res_int8, x_res_scales, self.w_up, self.w_up_sum,
                    M, self.mlp_hidden, self.hidden_dim
                )
            else:
                # MLP Gate-Up: [M, hidden_dim] -> [M, mlp_hidden] int32
                mlp_up_int32 = matmul_int32_out(x_res_int8, x_res_scales, self.w_up, self.w_up_sum,
                                                M, self.mlp_hidden, self.hidden_dim)

                # SwiGLU: [M, mlp_hidden] int32 -> [M, mlp_intermediate] int8
                mlp_act_int8, mlp_act_scales = native_swiglu(mlp_up_int32, x_res_scales,
                                                             M, self.mlp_hidden)

            # MLP Down: [M, mlp_intermediate] -> [M, hidden_dim]
            if USE_FUSED_MATMUL_QUANTIZE:
                # Fused matmul + quantize: eliminates int32 write
                mlp_down_int8, mlp_down_scales = fused_matmul_quantize(
                    mlp_act_int8, mlp_act_scales, self.w_down, self.w_down_sum,
                    M, self.hidden_dim, self.mlp_intermediate
                )
            else:
                mlp_down_int32 = matmul_int32_out(mlp_act_int8, mlp_act_scales, self.w_down, self.w_down_sum,
                                                  M, self.hidden_dim, self.mlp_intermediate)
                # Quantize int32 output to int8 (native kernel)
                mlp_down_int8, mlp_down_scales = native_quantize_int32(
                    mlp_down_int32, mlp_act_scales, M, self.hidden_dim
                )

        # Fused residual: x_res + mlp_down -> int8
        y_int8, y_scales = native_fused_residual(
            x_res_int8, x_res_scales, mlp_down_int8, mlp_down_scales, M, self.hidden_dim
        )

        return y_int8, y_scales


class TransformerLayerPyTorch:
    """Single transformer layer using PyTorch (float32 baseline)"""

    def __init__(self, hidden_dim, qkv_dim, mlp_hidden, num_heads, head_dim):
        self.hidden_dim = hidden_dim
        self.qkv_dim = qkv_dim
        self.mlp_hidden = mlp_hidden
        self.mlp_intermediate = mlp_hidden // 2
        self.num_heads = num_heads
        self.head_dim = head_dim
        self.scale = 1.0 / (head_dim ** 0.5)

        # Float32 weights
        self.w_q = torch.randn(hidden_dim, hidden_dim, dtype=torch.float32) * 0.02
        self.w_k = torch.randn(hidden_dim, hidden_dim, dtype=torch.float32) * 0.02
        self.w_v = torch.randn(hidden_dim, hidden_dim, dtype=torch.float32) * 0.02
        self.w_o = torch.randn(hidden_dim, hidden_dim, dtype=torch.float32) * 0.02
        self.w_up = torch.randn(hidden_dim, mlp_hidden, dtype=torch.float32) * 0.02
        self.w_down = torch.randn(self.mlp_intermediate, hidden_dim, dtype=torch.float32) * 0.02

    def forward(self, x):
        """Forward pass through one transformer layer"""
        M = x.shape[0]

        # === ATTENTION BLOCK ===
        q = torch.mm(x, self.w_q)
        k = torch.mm(x, self.w_k)
        v = torch.mm(x, self.w_v)

        # Reshape for multi-head attention
        q = q.view(M, self.num_heads, self.head_dim)
        k = k.view(M, self.num_heads, self.head_dim)
        v = v.view(M, self.num_heads, self.head_dim)

        # Attention
        attn_scores = torch.einsum('mhd,nhd->mhn', q, k) * self.scale
        attn_probs = F.softmax(attn_scores, dim=-1)
        attn_out = torch.einsum('mhn,nhd->mhd', attn_probs, v)
        attn_out = attn_out.reshape(M, self.hidden_dim)

        # Output Projection + Softmax
        out = torch.mm(attn_out, self.w_o)
        out = softmax_pytorch(out)

        # Residual
        x = x + out

        # === MLP BLOCK ===
        mlp_up = torch.mm(x, self.w_up)
        mlp_act = swiglu_pytorch(mlp_up)
        mlp_out = torch.mm(mlp_act, self.w_down)

        # Residual
        x = x + mlp_out

        return x


# ============================================================================
# Full Model (24 layers)
# ============================================================================

class FullTransformerNative:
    """Full 24-layer transformer using native kernels"""

    def __init__(self, n_layers, hidden_dim, qkv_dim, mlp_hidden, num_heads, head_dim):
        print(f"Initializing {n_layers}-layer native transformer...")
        self.layers = [
            TransformerLayerNative(hidden_dim, qkv_dim, mlp_hidden, num_heads, head_dim)
            for _ in range(n_layers)
        ]
        print(f"  Initialized {n_layers} layers")

    def forward(self, x):
        # Initial quantization
        x_int8, x_scales = quantize_activations_initial(x)

        for layer in self.layers:
            x_int8, x_scales = layer.forward(x_int8, x_scales)

        # Final dequantization
        return x_int8.float() * x_scales.unsqueeze(1)


class FullTransformerPyTorch:
    """Full 24-layer transformer using PyTorch baseline"""

    def __init__(self, n_layers, hidden_dim, qkv_dim, mlp_hidden, num_heads, head_dim):
        print(f"Initializing {n_layers}-layer PyTorch transformer...")
        self.layers = [
            TransformerLayerPyTorch(hidden_dim, qkv_dim, mlp_hidden, num_heads, head_dim)
            for _ in range(n_layers)
        ]
        print(f"  Initialized {n_layers} layers")

    def forward(self, x):
        for layer in self.layers:
            x = layer.forward(x)
        return x


# ============================================================================
# Benchmarking
# ============================================================================

def benchmark_forward(model, x, warmup=WARMUP_ITERS, iters=BENCH_ITERS, track_memory=True):
    """Benchmark forward pass with optional memory tracking"""
    # Warmup
    for _ in range(warmup):
        _ = model.forward(x.clone())

    gc.collect()

    # Get baseline memory before benchmark
    mem_before = get_process_memory_mb()

    # Start memory tracking if requested
    if track_memory:
        tracemalloc.start()

    # Benchmark
    times = []
    peak_mem = mem_before
    for _ in range(iters):
        x_input = x.clone()
        start = time.perf_counter()
        _ = model.forward(x_input)
        end = time.perf_counter()
        times.append((end - start) * 1000)  # ms

        # Track peak memory during iterations
        if track_memory:
            current_mem = get_process_memory_mb()
            peak_mem = max(peak_mem, current_mem)

    # Get memory stats
    mem_after = get_process_memory_mb()
    python_current, python_peak = (0, 0)
    if track_memory:
        python_current, python_peak = tracemalloc.get_traced_memory()
        tracemalloc.stop()

    return {
        'mean_ms': np.mean(times),
        'std_ms': np.std(times),
        'min_ms': np.min(times),
        'max_ms': np.max(times),
        'memory': {
            'rss_before_mb': mem_before,
            'rss_after_mb': mem_after,
            'rss_peak_mb': peak_mem,
            'rss_delta_mb': mem_after - mem_before,
            'python_peak_mb': python_peak / (1024 * 1024) if track_memory else 0,
        }
    }


def calculate_flops(M, n_layers, hidden_dim, mlp_hidden, num_heads, head_dim):
    """Calculate total FLOPs for forward pass"""
    mlp_intermediate = mlp_hidden // 2

    # Per layer FLOPs (2 * M * N * K for matmul)
    flops_qkv = 3 * 2 * M * hidden_dim * hidden_dim
    flops_attention = 2 * M * M * hidden_dim
    flops_o = 2 * M * hidden_dim * hidden_dim
    flops_up = 2 * M * mlp_hidden * hidden_dim
    flops_down = 2 * M * hidden_dim * mlp_intermediate

    flops_per_layer = flops_qkv + flops_attention + flops_o + flops_up + flops_down
    total_flops = flops_per_layer * n_layers

    return total_flops


# ============================================================================
# Main Benchmark
# ============================================================================

def run_benchmarks():
    """Run full benchmark suite"""
    print("=" * 70)
    print(f"Full {N_LAYERS}-Layer Transformer Forward Pass Benchmark")
    print("=" * 70)
    print()
    print(f"Architecture: {ARCH}")
    print(f"Kernel: {KERNEL_NAME}")
    print(f"Hidden dim: {HIDDEN_DIM}")
    print(f"Num heads: {NUM_HEADS}")
    print(f"Head dim: {HEAD_DIM}")
    print(f"MLP hidden: {MLP_HIDDEN}")
    print(f"Layers: {N_LAYERS}")
    print()
    print("Fusion settings:")
    print(f"  USE_FUSED_QKV: {USE_FUSED_QKV}")
    print(f"  USE_FUSED_MATMUL_SOFTMAX: {USE_FUSED_MATMUL_SOFTMAX}")
    print(f"  USE_FUSED_MATMUL_QUANTIZE: {USE_FUSED_MATMUL_QUANTIZE}")
    print(f"  USE_FUSED_MATMUL_SWIGLU: {USE_FUSED_MATMUL_SWIGLU}")
    print(f"  USE_FUSED_MLP: {USE_FUSED_MLP}")
    print()

    # Initialize models
    print("Initializing models...")
    model_native = FullTransformerNative(N_LAYERS, HIDDEN_DIM, QKV_DIM, MLP_HIDDEN, NUM_HEADS, HEAD_DIM)
    model_pytorch = FullTransformerPyTorch(N_LAYERS, HIDDEN_DIM, QKV_DIM, MLP_HIDDEN, NUM_HEADS, HEAD_DIM)
    print()

    results = {}

    for M in BATCH_SIZES:
        print(f"\n{'='*70}")
        print(f"Testing M={M} (batch size)")
        print(f"{'='*70}")

        # Create input
        x = torch.randn(M, HIDDEN_DIM, dtype=torch.float32)

        # Calculate FLOPs
        total_flops = calculate_flops(M, N_LAYERS, HIDDEN_DIM, MLP_HIDDEN, NUM_HEADS, HEAD_DIM)

        print(f"\nTotal FLOPs: {total_flops / 1e9:.2f} GFLOP")
        print()

        # Benchmark PyTorch
        print("Benchmarking PyTorch baseline...")
        pytorch_results = benchmark_forward(model_pytorch, x)
        pytorch_gflops = (total_flops / 1e9) / (pytorch_results['mean_ms'] / 1000)
        print(f"  Time: {pytorch_results['mean_ms']:.2f} ± {pytorch_results['std_ms']:.2f} ms")
        print(f"  Throughput: {pytorch_gflops:.1f} GFLOPS")
        print(f"  Memory (RSS peak): {pytorch_results['memory']['rss_peak_mb']:.1f} MB")

        # Benchmark Native
        print(f"\nBenchmarking Native ({KERNEL_NAME})...")
        native_results = benchmark_forward(model_native, x)
        native_gflops = (total_flops / 1e9) / (native_results['mean_ms'] / 1000)
        print(f"  Time: {native_results['mean_ms']:.2f} ± {native_results['std_ms']:.2f} ms")
        print(f"  Throughput: {native_gflops:.1f} GFLOPS")
        print(f"  Memory (RSS peak): {native_results['memory']['rss_peak_mb']:.1f} MB")

        # Speedup
        speedup = pytorch_results['mean_ms'] / native_results['mean_ms']
        print(f"\n  Speedup vs PyTorch: {speedup:.2f}x")

        results[f"M{M}"] = {
            'batch_size': M,
            'total_gflops': total_flops / 1e9,
            'pytorch': {
                'time_ms': pytorch_results['mean_ms'],
                'std_ms': pytorch_results['std_ms'],
                'gflops': pytorch_gflops,
                'memory_mb': pytorch_results['memory']['rss_peak_mb'],
            },
            'native': {
                'time_ms': native_results['mean_ms'],
                'std_ms': native_results['std_ms'],
                'gflops': native_gflops,
                'memory_mb': native_results['memory']['rss_peak_mb'],
            },
            'speedup': speedup,
        }

    return results


def print_summary(results):
    """Print summary table"""
    print("\n")
    print("=" * 90)
    print("SUMMARY: 24-Layer Transformer Forward Pass (Native Kernels)")
    print("=" * 90)
    print()
    print(f"{'Batch':<8} {'PyTorch (ms)':<14} {'Native (ms)':<14} {'Speedup':<10} {'GFLOPS':<10} {'PT Mem':<10} {'Native Mem':<10}")
    print("-" * 90)

    for key, data in results.items():
        M = data['batch_size']
        pt_time = data['pytorch']['time_ms']
        native_time = data['native']['time_ms']
        speedup = data['speedup']
        gflops = data['native']['gflops']
        pt_mem = data['pytorch'].get('memory_mb', 0)
        native_mem = data['native'].get('memory_mb', 0)
        print(f"M={M:<6} {pt_time:<14.2f} {native_time:<14.2f} {speedup:<10.2f}x {gflops:<10.1f} {pt_mem:<10.1f} {native_mem:<10.1f}")


def save_results(results):
    """Save results to JSON"""
    json_filename = f"{ARCH}_full_forward_native_{datetime.now().strftime('%m%d%y')}.json"

    output = {
        'metadata': {
            'timestamp': datetime.now().isoformat(),
            'platform': platform.system(),
            'architecture': ARCH,
            'kernel': KERNEL_NAME,
            'num_threads': num_threads,
            'n_layers': N_LAYERS,
            'hidden_dim': HIDDEN_DIM,
            'qkv_dim': QKV_DIM,
            'mlp_hidden': MLP_HIDDEN,
            'native_kernels': True,
        },
        'results': results
    }

    json_path = Path(__file__).parent / json_filename
    with open(json_path, 'w') as f:
        json.dump(output, f, indent=2)

    print(f"\nResults saved to: {json_path}")


def generate_plots(results):
    """Generate benchmark plots"""
    try:
        import matplotlib.pyplot as plt
    except ImportError:
        print("\nMatplotlib not installed, skipping plots")
        return

    plots_dir = Path("benchmark_plots_full_forward_native")
    plots_dir.mkdir(exist_ok=True)

    batch_sizes = [results[k]['batch_size'] for k in results]
    pytorch_times = [results[k]['pytorch']['time_ms'] for k in results]
    native_times = [results[k]['native']['time_ms'] for k in results]
    speedups = [results[k]['speedup'] for k in results]
    native_gflops = [results[k]['native']['gflops'] for k in results]
    pytorch_gflops = [results[k]['pytorch']['gflops'] for k in results]

    # Plot 1: Time comparison
    fig, ax = plt.subplots(figsize=(10, 6))
    x = np.arange(len(batch_sizes))
    width = 0.35

    bars1 = ax.bar(x - width/2, pytorch_times, width, label='PyTorch', color='#1f77b4')
    bars2 = ax.bar(x + width/2, native_times, width, label=f'Native ({KERNEL_NAME})', color='#2ca02c')

    ax.set_xlabel('Batch Size (M)')
    ax.set_ylabel('Time (ms)')
    ax.set_title(f'{N_LAYERS}-Layer Transformer Forward Pass Time (Native Kernels)')
    ax.set_xticks(x)
    ax.set_xticklabels([f'M={m}' for m in batch_sizes])
    ax.legend()
    ax.grid(axis='y', alpha=0.3)

    for bar in bars1:
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height(),
                f'{bar.get_height():.1f}', ha='center', va='bottom', fontsize=9)
    for bar in bars2:
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height(),
                f'{bar.get_height():.1f}', ha='center', va='bottom', fontsize=9)

    plt.tight_layout()
    plt.savefig(plots_dir / 'time_comparison.png', dpi=150)
    print(f"\nSaved: {plots_dir / 'time_comparison.png'}")
    plt.close()

    # Plot 2: Speedup
    fig, ax = plt.subplots(figsize=(10, 6))
    bars = ax.bar(x, speedups, color='#ff7f0e')
    ax.axhline(y=1.0, color='black', linestyle='--', linewidth=1, label='Baseline (1x)')

    ax.set_xlabel('Batch Size (M)')
    ax.set_ylabel('Speedup vs PyTorch')
    ax.set_title(f'{N_LAYERS}-Layer Transformer: Native Speedup vs PyTorch')
    ax.set_xticks(x)
    ax.set_xticklabels([f'M={m}' for m in batch_sizes])
    ax.grid(axis='y', alpha=0.3)

    for bar in bars:
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height(),
                f'{bar.get_height():.2f}x', ha='center', va='bottom', fontsize=10)

    plt.tight_layout()
    plt.savefig(plots_dir / 'speedup.png', dpi=150)
    print(f"Saved: {plots_dir / 'speedup.png'}")
    plt.close()

    # Plot 3: GFLOPS comparison
    fig, ax = plt.subplots(figsize=(10, 6))

    bars1 = ax.bar(x - width/2, pytorch_gflops, width, label='PyTorch', color='#1f77b4')
    bars2 = ax.bar(x + width/2, native_gflops, width, label=f'Native ({KERNEL_NAME})', color='#2ca02c')

    ax.set_xlabel('Batch Size (M)')
    ax.set_ylabel('GFLOPS')
    ax.set_title(f'{N_LAYERS}-Layer Transformer Throughput (Native Kernels)')
    ax.set_xticks(x)
    ax.set_xticklabels([f'M={m}' for m in batch_sizes])
    ax.legend()
    ax.grid(axis='y', alpha=0.3)

    plt.tight_layout()
    plt.savefig(plots_dir / 'gflops_comparison.png', dpi=150)
    print(f"Saved: {plots_dir / 'gflops_comparison.png'}")
    plt.close()

    print(f"\nAll plots saved to: {plots_dir}")


# ============================================================================
# Entry Point
# ============================================================================

if __name__ == "__main__":
    results = run_benchmarks()
    print_summary(results)
    save_results(results)
    generate_plots(results)

    print("\n" + "=" * 70)
    print("Benchmark Complete!")
    print("=" * 70)
