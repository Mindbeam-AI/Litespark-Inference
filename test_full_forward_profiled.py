#!/usr/bin/env python3
"""
Profiled 24-Layer Transformer Forward Pass

Adds detailed timing instrumentation to identify bottlenecks.
"""

import torch
import torch.nn.functional as F
import numpy as np
import time
import platform
from collections import defaultdict
from torch.utils.cpp_extension import load

# ============================================================================
# Configuration
# ============================================================================

N_LAYERS = 24
HIDDEN_DIM = 2048
HEAD_DIM = 64
NUM_HEADS = 32
MLP_HIDDEN = 16384
MLP_INTERMEDIATE = 8192

BATCH_SIZES = [1, 32, 128]
WARMUP_ITERS = 3
BENCH_ITERS = 10

# ============================================================================
# Check Architecture
# ============================================================================

machine = platform.machine().lower()
if machine in ['arm64', 'aarch64']:
    raise RuntimeError("Profiling only available on x86_64")
elif machine in ['x86_64', 'amd64']:
    ARCH = 'x86_64'
    print("Detected x86_64 architecture")

print(f"PyTorch version: {torch.__version__}")
num_threads = torch.get_num_threads()
print(f"Number of threads: {num_threads}")
print()

# ============================================================================
# Load Kernels
# ============================================================================

print("Loading kernels...")
extra_cflags = ['-O3', '-march=native', '-ffast-math', '-fopenmp']
extra_ldflags = ['-lgomp']

kernel = load(
    name="vnni_kernel_profiled",
    sources=["src/cpu_ops/kernels/x86_64/matmul_free_tmac_vnni.cpp"],
    extra_cflags=extra_cflags,
    extra_ldflags=extra_ldflags,
    verbose=False
)
print("Kernel loaded")
print()

# ============================================================================
# Profiling Infrastructure
# ============================================================================

class Profiler:
    """Simple profiler to track time spent in different operations"""

    def __init__(self):
        self.times = defaultdict(list)
        self.current_op = None
        self.start_time = None

    def start(self, op_name):
        self.current_op = op_name
        self.start_time = time.perf_counter()

    def stop(self):
        if self.current_op:
            elapsed = (time.perf_counter() - self.start_time) * 1000  # ms
            self.times[self.current_op].append(elapsed)
            self.current_op = None

    def reset(self):
        self.times = defaultdict(list)

    def report(self):
        """Print profiling report"""
        total = 0
        print("\n" + "=" * 70)
        print("PROFILING REPORT (per forward pass)")
        print("=" * 70)
        print(f"{'Operation':<35} {'Time (ms)':<12} {'%':<8} {'Calls':<8}")
        print("-" * 70)

        # Calculate totals
        totals = {}
        for op, times in self.times.items():
            totals[op] = sum(times)
            total += sum(times)

        # Sort by time
        sorted_ops = sorted(totals.items(), key=lambda x: -x[1])

        for op, t in sorted_ops:
            pct = (t / total) * 100 if total > 0 else 0
            calls = len(self.times[op])
            avg_per_call = t / calls if calls > 0 else 0
            print(f"{op:<35} {t:<12.2f} {pct:<8.1f} {calls:<8}")

        print("-" * 70)
        print(f"{'TOTAL':<35} {total:<12.2f} {'100.0':<8}")
        print()

        # Group by category
        categories = {
            'Matmul (QKV)': ['q_matmul', 'k_matmul', 'v_matmul'],
            'Matmul (Other)': ['o_matmul', 'up_matmul', 'down_matmul'],
            'Attention': ['attention'],
            'Softmax': ['softmax_q', 'softmax_k', 'softmax_v', 'softmax_o'],
            'SwiGLU': ['swiglu'],
            'Residual': ['residual_attn', 'residual_mlp'],
            'Quantize': ['quantize_down'],
        }

        print("BY CATEGORY:")
        print("-" * 70)
        for cat, ops in categories.items():
            cat_total = sum(totals.get(op, 0) for op in ops)
            pct = (cat_total / total) * 100 if total > 0 else 0
            print(f"{cat:<35} {cat_total:<12.2f} {pct:<8.1f}")

        return total

profiler = Profiler()

# ============================================================================
# Weight Preparation
# ============================================================================

def prepare_ternary_weights(N, K):
    w_float = torch.randn(K, N, dtype=torch.float32)
    w_ternary = torch.zeros_like(w_float)
    w_ternary[w_float > 0.5] = 1.0
    w_ternary[w_float < -0.5] = -1.0

    w_for_kernel = w_ternary.t().contiguous()
    K_padded = ((K + 63) // 64) * 64
    w_int8_vnni = torch.zeros(N, K_padded, dtype=torch.int8)
    w_sum_vnni = torch.zeros(N, dtype=torch.int32)
    kernel.pack_weights_vnni(w_for_kernel, w_int8_vnni, w_sum_vnni, N, K)
    return w_ternary, w_int8_vnni, w_sum_vnni


def quantize_activations_initial(x):
    M, K = x.shape
    scales = x.abs().max(dim=1, keepdim=True)[0] / 127.0
    scales = scales.clamp(min=1e-8)
    x_int8 = (x / scales).round().clamp(-127, 127).to(torch.int8)
    return x_int8, scales.view(M)


# ============================================================================
# Native Kernel Wrappers with Profiling
# ============================================================================

def matmul_int32_out(x_int8, scales, w_int8, w_sum, M, N, K, op_name):
    profiler.start(op_name)
    y_int32 = torch.zeros(M, N, dtype=torch.int32)

    # Use v4 kernel for large N or large K (better parallelization)
    # - Large N (up_matmul): parallelize over N
    # - Large K with small M: v4 still helps with better blocking
    if N >= 4096 or (K >= 4096 and M <= 32):
        kernel.matmul_free_vnni_v4_large_n(
            x_int8, scales, w_int8, w_sum, y_int32, M, N, K, num_threads
        )
    else:
        bias = torch.empty(0)
        kernel.matmul_free_vnni_v3_int32_out(
            x_int8, scales, w_int8, w_sum, y_int32, bias, M, N, K, num_threads
        )
    profiler.stop()
    return y_int32


def native_softmax(x_int32, x_scales, M, N, op_name):
    profiler.start(op_name)
    y_int8 = torch.zeros(M, N, dtype=torch.int8)
    y_scales = torch.zeros(M, dtype=torch.float32)
    kernel.softmax_int8(x_int32, x_scales, y_int8, y_scales, M, N, num_threads)
    profiler.stop()
    return y_int8, y_scales


def native_swiglu(x_int32, x_scales, M, N, op_name):
    profiler.start(op_name)
    half_N = N // 2
    y_int8 = torch.zeros(M, half_N, dtype=torch.int8)
    y_scales = torch.zeros(M, dtype=torch.float32)
    kernel.swiglu_int8(x_int32, x_scales, y_int8, y_scales, M, N, num_threads)
    profiler.stop()
    return y_int8, y_scales


def native_attention(q_int8, k_int8, v_int8, q_scales, k_scales, v_scales, M, num_heads, head_dim, scale, op_name):
    profiler.start(op_name)
    hidden_dim = num_heads * head_dim
    out_int8 = torch.zeros(M, hidden_dim, dtype=torch.int8)
    out_scales = torch.zeros(M, dtype=torch.float32)
    # For large M, use v2 kernel that parallelizes over M×heads
    if M >= 64:
        kernel.attention_int8_v2(
            q_int8, k_int8, v_int8,
            q_scales, k_scales, v_scales,
            out_int8, out_scales,
            M, num_heads, head_dim, scale, num_threads
        )
    else:
        kernel.attention_int8(
            q_int8, k_int8, v_int8,
            q_scales, k_scales, v_scales,
            out_int8, out_scales,
            M, num_heads, head_dim, scale, num_threads
        )
    profiler.stop()
    return out_int8, out_scales


def native_fused_residual(x1_int8, x1_scales, x2_int8, x2_scales, M, N, op_name):
    profiler.start(op_name)
    y_int8 = torch.zeros(M, N, dtype=torch.int8)
    y_scales = torch.zeros(M, dtype=torch.float32)
    kernel.fused_residual_quantize(
        x1_int8, x1_scales, x2_int8, x2_scales,
        y_int8, y_scales, M, N, num_threads
    )
    profiler.stop()
    return y_int8, y_scales


def native_quantize_int32(x_int32, x_scales, M, N, op_name):
    profiler.start(op_name)
    y_int8 = torch.zeros(M, N, dtype=torch.int8)
    y_scales = torch.zeros(M, dtype=torch.float32)
    kernel.quantize_int32_to_int8(x_int32, x_scales, y_int8, y_scales, M, N, num_threads)
    profiler.stop()
    return y_int8, y_scales


# ============================================================================
# Profiled Transformer Layer
# ============================================================================

class TransformerLayerProfiled:
    def __init__(self, hidden_dim, mlp_hidden, num_heads, head_dim):
        self.hidden_dim = hidden_dim
        self.mlp_hidden = mlp_hidden
        self.mlp_intermediate = mlp_hidden // 2
        self.num_heads = num_heads
        self.head_dim = head_dim
        self.scale = 1.0 / (head_dim ** 0.5)

        self.w_q_ternary, self.w_q, self.w_q_sum = prepare_ternary_weights(hidden_dim, hidden_dim)
        self.w_k_ternary, self.w_k, self.w_k_sum = prepare_ternary_weights(hidden_dim, hidden_dim)
        self.w_v_ternary, self.w_v, self.w_v_sum = prepare_ternary_weights(hidden_dim, hidden_dim)
        self.w_o_ternary, self.w_o, self.w_o_sum = prepare_ternary_weights(hidden_dim, hidden_dim)
        self.w_up_ternary, self.w_up, self.w_up_sum = prepare_ternary_weights(mlp_hidden, hidden_dim)
        self.w_down_ternary, self.w_down, self.w_down_sum = prepare_ternary_weights(hidden_dim, self.mlp_intermediate)

    def forward(self, x_int8, x_scales):
        M = x_int8.shape[0]

        # Q, K, V projections
        q_int32 = matmul_int32_out(x_int8, x_scales, self.w_q, self.w_q_sum,
                                   M, self.hidden_dim, self.hidden_dim, 'q_matmul')
        k_int32 = matmul_int32_out(x_int8, x_scales, self.w_k, self.w_k_sum,
                                   M, self.hidden_dim, self.hidden_dim, 'k_matmul')
        v_int32 = matmul_int32_out(x_int8, x_scales, self.w_v, self.w_v_sum,
                                   M, self.hidden_dim, self.hidden_dim, 'v_matmul')

        # Softmax on Q, K, V
        q_int8_sm, q_scales_sm = native_softmax(q_int32, x_scales, M, self.hidden_dim, 'softmax_q')
        k_int8_sm, k_scales_sm = native_softmax(k_int32, x_scales, M, self.hidden_dim, 'softmax_k')
        v_int8_sm, v_scales_sm = native_softmax(v_int32, x_scales, M, self.hidden_dim, 'softmax_v')

        # Attention
        attn_int8, attn_scales = native_attention(
            q_int8_sm, k_int8_sm, v_int8_sm,
            q_scales_sm, k_scales_sm, v_scales_sm,
            M, self.num_heads, self.head_dim, self.scale, 'attention'
        )

        # Output projection + softmax
        o_int32 = matmul_int32_out(attn_int8, attn_scales, self.w_o, self.w_o_sum,
                                   M, self.hidden_dim, self.hidden_dim, 'o_matmul')
        o_int8, o_scales = native_softmax(o_int32, attn_scales, M, self.hidden_dim, 'softmax_o')

        # Residual
        x_res_int8, x_res_scales = native_fused_residual(
            x_int8, x_scales, o_int8, o_scales, M, self.hidden_dim, 'residual_attn'
        )

        # MLP
        mlp_up_int32 = matmul_int32_out(x_res_int8, x_res_scales, self.w_up, self.w_up_sum,
                                        M, self.mlp_hidden, self.hidden_dim, 'up_matmul')
        mlp_act_int8, mlp_act_scales = native_swiglu(mlp_up_int32, x_res_scales,
                                                     M, self.mlp_hidden, 'swiglu')
        mlp_down_int32 = matmul_int32_out(mlp_act_int8, mlp_act_scales, self.w_down, self.w_down_sum,
                                          M, self.hidden_dim, self.mlp_intermediate, 'down_matmul')
        mlp_down_int8, mlp_down_scales = native_quantize_int32(
            mlp_down_int32, mlp_act_scales, M, self.hidden_dim, 'quantize_down'
        )

        # Final residual
        y_int8, y_scales = native_fused_residual(
            x_res_int8, x_res_scales, mlp_down_int8, mlp_down_scales, M, self.hidden_dim, 'residual_mlp'
        )

        return y_int8, y_scales


class FullTransformerProfiled:
    def __init__(self, n_layers, hidden_dim, mlp_hidden, num_heads, head_dim):
        print(f"Initializing {n_layers}-layer profiled transformer...")
        self.layers = [
            TransformerLayerProfiled(hidden_dim, mlp_hidden, num_heads, head_dim)
            for _ in range(n_layers)
        ]
        print(f"  Initialized {n_layers} layers")

    def forward(self, x):
        x_int8, x_scales = quantize_activations_initial(x)
        for layer in self.layers:
            x_int8, x_scales = layer.forward(x_int8, x_scales)
        return x_int8.float() * x_scales.unsqueeze(1)


# ============================================================================
# Main
# ============================================================================

def run_profiled_benchmark():
    print("=" * 70)
    print("PROFILED BENCHMARK")
    print("=" * 70)
    print()

    model = FullTransformerProfiled(N_LAYERS, HIDDEN_DIM, MLP_HIDDEN, NUM_HEADS, HEAD_DIM)
    print()

    for M in BATCH_SIZES:
        print(f"\n{'='*70}")
        print(f"Profiling M={M}")
        print(f"{'='*70}")

        x = torch.randn(M, HIDDEN_DIM, dtype=torch.float32)

        # Warmup
        print("Warming up...")
        for _ in range(WARMUP_ITERS):
            _ = model.forward(x.clone())

        # Profile
        profiler.reset()
        print(f"Running {BENCH_ITERS} iterations...")

        times = []
        for _ in range(BENCH_ITERS):
            start = time.perf_counter()
            _ = model.forward(x.clone())
            end = time.perf_counter()
            times.append((end - start) * 1000)

        avg_time = np.mean(times)
        print(f"\nTotal forward pass time: {avg_time:.2f} ms")

        # Report
        profiler.report()

        # Also show time per layer
        total_profiled = sum(sum(t) for t in profiler.times.values())
        print(f"Profiled time per layer: {total_profiled / N_LAYERS / BENCH_ITERS:.2f} ms")


if __name__ == "__main__":
    run_profiled_benchmark()
    print("\n" + "=" * 70)
    print("Profiling Complete!")
    print("=" * 70)
