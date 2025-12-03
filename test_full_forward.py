#!/usr/bin/env python3
"""
Full 24-Layer Transformer Forward Pass Benchmark

Tests the complete forward pass through a transformer model with:
- 24 layers
- QKV projection + softmax
- Output projection + softmax
- MLP Gate-Up + SwiGLU
- MLP Down projection

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
from pathlib import Path
from datetime import datetime
from torch.utils.cpp_extension import load

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
# QKV_DIM = 2560 suggests: Q=2048, K=256, V=256 (GQA style) or similar
# We'll use: Q_DIM = K_DIM = V_DIM = HIDDEN_DIM = 2048 for standard attention
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
    # Load ARM64 NEON kernels
    extra_cflags = ['-O3', '-ffast-math', '-mcpu=apple-m1', '-DACCELERATE_NEW_LAPACK']
    extra_ldflags = ['-L/opt/homebrew/opt/libomp/lib', '-lomp']

    kernel = load(
        name="neon_int8_kernel",
        sources=["src/cpu_ops/kernels/arm64/matmul_free_neon_int8.cpp"],
        extra_cflags=extra_cflags + ['-Xpreprocessor', '-fopenmp', '-I/opt/homebrew/opt/libomp/include'],
        extra_ldflags=extra_ldflags,
        verbose=False
    )

    KERNEL_NAME = "SDOT Direct v4"
    num_threads = 10

else:
    # Load x86_64 VNNI kernels
    extra_cflags = ['-O3', '-march=native', '-ffast-math', '-fopenmp']
    extra_ldflags = ['-lgomp']

    kernel = load(
        name="vnni_kernel",
        sources=["src/cpu_ops/kernels/x86_64/matmul_free_tmac_vnni.cpp"],
        extra_cflags=extra_cflags,
        extra_ldflags=extra_ldflags,
        verbose=False
    )

    KERNEL_NAME = "VNNI v3"
    # Use PyTorch's detected thread count for consistency
    num_threads = torch.get_num_threads()

print(f"Loaded kernel: {KERNEL_NAME}")
print(f"Using {num_threads} threads")
print()

# ============================================================================
# Activation Functions
# ============================================================================

def softmax(x):
    """Standard softmax activation"""
    return F.softmax(x, dim=-1)


def swiglu(x):
    """
    SwiGLU activation: splits input in half, applies SiLU to first half,
    multiplies with second half.

    SwiGLU(x, gate) = SiLU(x) * gate
    where SiLU(x) = x * sigmoid(x)
    """
    # Split in half along last dimension
    x1, x2 = x.chunk(2, dim=-1)
    # SiLU (Swish) on first half, multiply with second half
    return F.silu(x1) * x2


# ============================================================================
# Weight Preparation
# ============================================================================

def prepare_ternary_weights(N, K):
    """Create random ternary weights and prepare for kernel"""
    # Random ternary weights: -1, 0, +1
    w_ternary = torch.randint(-1, 2, (K, N), dtype=torch.int8)

    if ARCH == 'arm64':
        # For SDOT: just use int8 weights directly
        return w_ternary, w_ternary.clone()
    else:
        # For VNNI: convert to uint8 format (add 128 bias)
        w_int8 = w_ternary.clone()
        return w_ternary, w_int8


def quantize_activations(x):
    """Quantize float activations to int8"""
    M, K = x.shape
    scales = x.abs().max(dim=1, keepdim=True)[0] / 127.0
    scales = scales.clamp(min=1e-8)
    x_int8 = (x / scales).round().clamp(-127, 127).to(torch.int8)
    return x_int8, scales.squeeze()


# ============================================================================
# Single Layer Forward Pass
# ============================================================================

def matmul_ternary(x_int8, scales, w_int8, bias, M, N, K):
    """Run ternary matmul using our kernel"""
    y = torch.zeros(M, N, dtype=torch.float32)

    if ARCH == 'arm64':
        kernel.matmul_ternary_sdot_direct_v4(
            x_int8, scales, w_int8, y, bias, M, N, K, num_threads
        )
    else:
        kernel.matmul_free_vnni_v3(
            x_int8, scales, w_int8, y, bias, M, N, K, num_threads
        )

    return y


def matmul_pytorch(x, w):
    """Run matmul using PyTorch"""
    return torch.mm(x, w.float())


# ============================================================================
# Full Transformer Layer
# ============================================================================

class TransformerLayerTernary:
    """Single transformer layer using ternary kernels"""

    def __init__(self, hidden_dim, qkv_dim, mlp_hidden, num_heads, head_dim):
        self.hidden_dim = hidden_dim
        self.qkv_dim = qkv_dim
        self.mlp_hidden = mlp_hidden
        self.mlp_intermediate = mlp_hidden // 2  # After SwiGLU split
        self.num_heads = num_heads
        self.head_dim = head_dim
        self.scale = 1.0 / (head_dim ** 0.5)  # Attention scale factor

        # Create ternary weights for projections
        # Q projection: [hidden_dim, hidden_dim]
        self.w_q_ternary, self.w_q = prepare_ternary_weights(hidden_dim, hidden_dim)
        # K projection: [hidden_dim, hidden_dim]
        self.w_k_ternary, self.w_k = prepare_ternary_weights(hidden_dim, hidden_dim)
        # V projection: [hidden_dim, hidden_dim]
        self.w_v_ternary, self.w_v = prepare_ternary_weights(hidden_dim, hidden_dim)
        # Output projection: [hidden_dim, hidden_dim]
        self.w_o_ternary, self.w_o = prepare_ternary_weights(hidden_dim, hidden_dim)
        # MLP projections
        self.w_up_ternary, self.w_up = prepare_ternary_weights(mlp_hidden, hidden_dim)
        self.w_down_ternary, self.w_down = prepare_ternary_weights(hidden_dim, self.mlp_intermediate)

        # Biases (zeros for simplicity)
        self.bias_q = torch.zeros(hidden_dim, dtype=torch.float32)
        self.bias_k = torch.zeros(hidden_dim, dtype=torch.float32)
        self.bias_v = torch.zeros(hidden_dim, dtype=torch.float32)
        self.bias_o = torch.zeros(hidden_dim, dtype=torch.float32)
        self.bias_up = torch.zeros(mlp_hidden, dtype=torch.float32)
        self.bias_down = torch.zeros(hidden_dim, dtype=torch.float32)

    def forward(self, x):
        """Forward pass through one transformer layer"""
        M = x.shape[0]

        # === ATTENTION BLOCK ===

        # Q Projection: [M, hidden_dim] @ [hidden_dim, hidden_dim] -> [M, hidden_dim]
        x_int8, scales = quantize_activations(x)
        q = matmul_ternary(x_int8, scales, self.w_q, self.bias_q,
                          M, self.hidden_dim, self.hidden_dim)

        # K Projection: [M, hidden_dim] @ [hidden_dim, hidden_dim] -> [M, hidden_dim]
        k = matmul_ternary(x_int8, scales, self.w_k, self.bias_k,
                          M, self.hidden_dim, self.hidden_dim)

        # V Projection: [M, hidden_dim] @ [hidden_dim, hidden_dim] -> [M, hidden_dim]
        v = matmul_ternary(x_int8, scales, self.w_v, self.bias_v,
                          M, self.hidden_dim, self.hidden_dim)

        # Reshape for multi-head attention: [M, num_heads, head_dim]
        q = q.view(M, self.num_heads, self.head_dim)
        k = k.view(M, self.num_heads, self.head_dim)
        v = v.view(M, self.num_heads, self.head_dim)

        # Attention scores: Q @ K^T -> [M, num_heads, num_heads] per position
        # For single sequence: [num_heads, head_dim] @ [head_dim, num_heads] per batch
        # Simplified: compute attention per head
        attn_scores = torch.einsum('mhd,nhd->mhn', q, k) * self.scale

        # Softmax over keys
        attn_probs = F.softmax(attn_scores, dim=-1)

        # Apply attention to values: [M, num_heads, M] @ [M, num_heads, head_dim]
        attn_out = torch.einsum('mhn,nhd->mhd', attn_probs, v)

        # Reshape back: [M, hidden_dim]
        attn_out = attn_out.reshape(M, self.hidden_dim)

        # Output Projection + Softmax: [M, hidden_dim] @ [hidden_dim, hidden_dim] -> [M, hidden_dim]
        attn_int8, attn_scales = quantize_activations(attn_out)
        out = matmul_ternary(attn_int8, attn_scales, self.w_o, self.bias_o,
                           M, self.hidden_dim, self.hidden_dim)
        out = softmax(out)

        # Residual connection
        x = x + out

        # === MLP BLOCK ===

        # MLP Gate-Up + SwiGLU: [M, hidden_dim] @ [hidden_dim, mlp_hidden] -> [M, mlp_hidden]
        x_int8, scales = quantize_activations(x)
        mlp_up = matmul_ternary(x_int8, scales, self.w_up, self.bias_up,
                               M, self.mlp_hidden, self.hidden_dim)
        mlp_act = swiglu(mlp_up)  # Output: [M, mlp_intermediate]

        # MLP Down: [M, mlp_intermediate] @ [mlp_intermediate, hidden_dim] -> [M, hidden_dim]
        mlp_int8, mlp_scales = quantize_activations(mlp_act)
        mlp_out = matmul_ternary(mlp_int8, mlp_scales, self.w_down, self.bias_down,
                                M, self.hidden_dim, self.mlp_intermediate)

        # Residual connection
        x = x + mlp_out

        return x


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
        # Q, K, V projections: [hidden_dim, hidden_dim]
        self.w_q = torch.randn(hidden_dim, hidden_dim, dtype=torch.float32) * 0.02
        self.w_k = torch.randn(hidden_dim, hidden_dim, dtype=torch.float32) * 0.02
        self.w_v = torch.randn(hidden_dim, hidden_dim, dtype=torch.float32) * 0.02
        # Output: [hidden_dim, hidden_dim]
        self.w_o = torch.randn(hidden_dim, hidden_dim, dtype=torch.float32) * 0.02
        # MLP Up: [hidden_dim, mlp_hidden]
        self.w_up = torch.randn(hidden_dim, mlp_hidden, dtype=torch.float32) * 0.02
        # MLP Down: [mlp_intermediate, hidden_dim]
        self.w_down = torch.randn(self.mlp_intermediate, hidden_dim, dtype=torch.float32) * 0.02

    def forward(self, x):
        """Forward pass through one transformer layer"""
        M = x.shape[0]

        # === ATTENTION BLOCK ===

        # Q Projection: [M, hidden_dim] @ [hidden_dim, hidden_dim] -> [M, hidden_dim]
        q = torch.mm(x, self.w_q)

        # K Projection: [M, hidden_dim] @ [hidden_dim, hidden_dim] -> [M, hidden_dim]
        k = torch.mm(x, self.w_k)

        # V Projection: [M, hidden_dim] @ [hidden_dim, hidden_dim] -> [M, hidden_dim]
        v = torch.mm(x, self.w_v)

        # Reshape for multi-head attention: [M, num_heads, head_dim]
        q = q.view(M, self.num_heads, self.head_dim)
        k = k.view(M, self.num_heads, self.head_dim)
        v = v.view(M, self.num_heads, self.head_dim)

        # Attention scores: Q @ K^T
        attn_scores = torch.einsum('mhd,nhd->mhn', q, k) * self.scale

        # Softmax over keys
        attn_probs = F.softmax(attn_scores, dim=-1)

        # Apply attention to values
        attn_out = torch.einsum('mhn,nhd->mhd', attn_probs, v)

        # Reshape back: [M, hidden_dim]
        attn_out = attn_out.reshape(M, self.hidden_dim)

        # Output Projection + Softmax: [M, hidden_dim] @ [hidden_dim, hidden_dim] -> [M, hidden_dim]
        out = torch.mm(attn_out, self.w_o)
        out = softmax(out)

        # Residual
        x = x + out

        # === MLP BLOCK ===

        # MLP Gate-Up + SwiGLU: [M, hidden_dim] @ [hidden_dim, mlp_hidden] -> [M, mlp_hidden]
        mlp_up = torch.mm(x, self.w_up)
        mlp_act = swiglu(mlp_up)  # Output: [M, mlp_intermediate]

        # MLP Down: [M, mlp_intermediate] @ [mlp_intermediate, hidden_dim] -> [M, hidden_dim]
        mlp_out = torch.mm(mlp_act, self.w_down)

        # Residual
        x = x + mlp_out

        return x


# ============================================================================
# Full Model (24 layers)
# ============================================================================

class FullTransformerTernary:
    """Full 24-layer transformer using ternary kernels"""

    def __init__(self, n_layers, hidden_dim, qkv_dim, mlp_hidden, num_heads, head_dim):
        print(f"Initializing {n_layers}-layer ternary transformer...")
        self.layers = [
            TransformerLayerTernary(hidden_dim, qkv_dim, mlp_hidden, num_heads, head_dim)
            for _ in range(n_layers)
        ]
        print(f"  Initialized {n_layers} layers")

    def forward(self, x):
        for layer in self.layers:
            x = layer.forward(x)
        return x


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

def benchmark_forward(model, x, warmup=WARMUP_ITERS, iters=BENCH_ITERS):
    """Benchmark forward pass"""
    # Warmup
    for _ in range(warmup):
        _ = model.forward(x.clone())

    gc.collect()

    # Benchmark
    times = []
    for _ in range(iters):
        x_input = x.clone()
        start = time.perf_counter()
        _ = model.forward(x_input)
        end = time.perf_counter()
        times.append((end - start) * 1000)  # ms

    return {
        'mean_ms': np.mean(times),
        'std_ms': np.std(times),
        'min_ms': np.min(times),
        'max_ms': np.max(times),
    }


def calculate_flops(M, n_layers, hidden_dim, mlp_hidden, num_heads, head_dim):
    """Calculate total FLOPs for forward pass"""
    mlp_intermediate = mlp_hidden // 2

    # Per layer FLOPs (2 * M * N * K for matmul)
    # Q, K, V projections: 3 * (2 * M * hidden_dim * hidden_dim)
    flops_qkv = 3 * 2 * M * hidden_dim * hidden_dim
    # Attention: Q @ K^T and attn @ V
    # Q @ K^T: M * num_heads * head_dim * M (for each head)
    # attn @ V: M * num_heads * M * head_dim
    flops_attention = 2 * M * M * hidden_dim  # Simplified
    # Output projection
    flops_o = 2 * M * hidden_dim * hidden_dim
    # MLP
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

    # Initialize models
    print("Initializing models...")
    model_ternary = FullTransformerTernary(N_LAYERS, HIDDEN_DIM, QKV_DIM, MLP_HIDDEN, NUM_HEADS, HEAD_DIM)
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

        # Benchmark Ternary
        print(f"\nBenchmarking Ternary ({KERNEL_NAME})...")
        ternary_results = benchmark_forward(model_ternary, x)
        ternary_gflops = (total_flops / 1e9) / (ternary_results['mean_ms'] / 1000)
        print(f"  Time: {ternary_results['mean_ms']:.2f} ± {ternary_results['std_ms']:.2f} ms")
        print(f"  Throughput: {ternary_gflops:.1f} GFLOPS")

        # Speedup
        speedup = pytorch_results['mean_ms'] / ternary_results['mean_ms']
        print(f"\n  Speedup vs PyTorch: {speedup:.2f}x")

        results[f"M{M}"] = {
            'batch_size': M,
            'total_gflops': total_flops / 1e9,
            'pytorch': {
                'time_ms': pytorch_results['mean_ms'],
                'std_ms': pytorch_results['std_ms'],
                'gflops': pytorch_gflops,
            },
            'ternary': {
                'time_ms': ternary_results['mean_ms'],
                'std_ms': ternary_results['std_ms'],
                'gflops': ternary_gflops,
            },
            'speedup': speedup,
        }

    return results


def print_summary(results):
    """Print summary table"""
    print("\n")
    print("=" * 70)
    print("SUMMARY: 24-Layer Transformer Forward Pass")
    print("=" * 70)
    print()
    print(f"{'Batch Size':<12} {'PyTorch (ms)':<15} {'Ternary (ms)':<15} {'Speedup':<10} {'GFLOPS':<10}")
    print("-" * 70)

    for key, data in results.items():
        M = data['batch_size']
        pt_time = data['pytorch']['time_ms']
        tern_time = data['ternary']['time_ms']
        speedup = data['speedup']
        gflops = data['ternary']['gflops']
        print(f"M={M:<10} {pt_time:<15.2f} {tern_time:<15.2f} {speedup:<10.2f}x {gflops:<10.1f}")


def save_results(results):
    """Save results to JSON"""
    json_filename = f"{ARCH}_full_forward_{datetime.now().strftime('%m%d%y')}.json"

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

    plots_dir = Path("benchmark_plots_full_forward")
    plots_dir.mkdir(exist_ok=True)

    batch_sizes = [results[k]['batch_size'] for k in results]
    pytorch_times = [results[k]['pytorch']['time_ms'] for k in results]
    ternary_times = [results[k]['ternary']['time_ms'] for k in results]
    speedups = [results[k]['speedup'] for k in results]
    ternary_gflops = [results[k]['ternary']['gflops'] for k in results]
    pytorch_gflops = [results[k]['pytorch']['gflops'] for k in results]

    # Plot 1: Time comparison
    fig, ax = plt.subplots(figsize=(10, 6))
    x = np.arange(len(batch_sizes))
    width = 0.35

    bars1 = ax.bar(x - width/2, pytorch_times, width, label='PyTorch', color='#1f77b4')
    bars2 = ax.bar(x + width/2, ternary_times, width, label=f'Ternary ({KERNEL_NAME})', color='#2ca02c')

    ax.set_xlabel('Batch Size (M)')
    ax.set_ylabel('Time (ms)')
    ax.set_title(f'{N_LAYERS}-Layer Transformer Forward Pass Time')
    ax.set_xticks(x)
    ax.set_xticklabels([f'M={m}' for m in batch_sizes])
    ax.legend()
    ax.grid(axis='y', alpha=0.3)

    # Add value labels
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
    ax.set_title(f'{N_LAYERS}-Layer Transformer: Ternary Speedup vs PyTorch')
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
    bars2 = ax.bar(x + width/2, ternary_gflops, width, label=f'Ternary ({KERNEL_NAME})', color='#2ca02c')

    ax.set_xlabel('Batch Size (M)')
    ax.set_ylabel('GFLOPS')
    ax.set_title(f'{N_LAYERS}-Layer Transformer Throughput')
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
