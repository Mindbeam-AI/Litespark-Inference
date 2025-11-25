"""
Test different output blocking sizes to find optimal performance
"""
import torch
import time
import numpy as np
from pathlib import Path

# We'll test by modifying N_BLOCK in the kernel and recompiling
# For now, let's create variants with different blocking

def create_blocked_kernel(n_block):
    """Create a kernel with specific N_BLOCK size"""
    template = """
#include <arm_neon.h>
#include <torch/extension.h>
#include <omp.h>

#define PREFETCH(addr) __builtin_prefetch(addr, 0, 3)
#define N_BLOCK {N_BLOCK}

void matmul_free_neon_block{N_BLOCK}(
    torch::Tensor x_tensor,
    torch::Tensor w_tensor,
    torch::Tensor y_tensor,
    torch::Tensor bias_tensor,
    int M, int N, int K,
    int num_threads
) {{
    const float* x = x_tensor.data_ptr<float>();
    const float* w = w_tensor.data_ptr<float>();
    float* y = y_tensor.data_ptr<float>();
    const float* bias = bias_tensor.defined() ? bias_tensor.data_ptr<float>() : nullptr;

    const int K_SIMD = 4;
    const int K_SIMD_END = (K / K_SIMD) * K_SIMD;
    const int K_PREFETCH = 64;

    omp_set_num_threads(num_threads);

    #pragma omp parallel for schedule(dynamic, 8)
    for (int m = 0; m < M; m++) {{
        const float* x_row = x + m * K;
        float* y_row = y + m * N;

        int n = 0;
        for (; n <= N - N_BLOCK; n += N_BLOCK) {{
            // N_BLOCK separate pos/neg accumulators
            float32x4_t sum_pos[N_BLOCK], sum_neg[N_BLOCK];
            for (int i = 0; i < N_BLOCK; i++) {{
                sum_pos[i] = vdupq_n_f32(0.0f);
                sum_neg[i] = vdupq_n_f32(0.0f);
            }}
            float32x4_t zero = vdupq_n_f32(0.0f);

            for (int k = 0; k < K_SIMD_END; k += K_SIMD) {{
                if (k + K_PREFETCH < K) {{
                    PREFETCH(x_row + k + K_PREFETCH);
                    PREFETCH(w + (n + 0) * K + k + K_PREFETCH);
                }}

                float32x4_t x_vals = vld1q_f32(x_row + k);

                for (int i = 0; i < N_BLOCK; i++) {{
                    float32x4_t w_vals = vld1q_f32(w + (n + i) * K + k);
                    uint32x4_t mask_pos = vcgtq_f32(w_vals, zero);
                    uint32x4_t mask_neg = vcltq_f32(w_vals, zero);
                    sum_pos[i] = vaddq_f32(sum_pos[i], vbslq_f32(mask_pos, x_vals, zero));
                    sum_neg[i] = vaddq_f32(sum_neg[i], vbslq_f32(mask_neg, x_vals, zero));
                }}
            }}

            // Horizontal reduction and scalar tail
            float results[N_BLOCK];
            for (int i = 0; i < N_BLOCK; i++) {{
                results[i] = vaddvq_f32(sum_pos[i]) - vaddvq_f32(sum_neg[i]);
            }}

            for (int k = K_SIMD_END; k < K; k++) {{
                float x_val = x_row[k];
                for (int i = 0; i < N_BLOCK; i++) {{
                    float w_val = w[(n + i) * K + k];
                    if (w_val > 0.0f) results[i] += x_val;
                    else if (w_val < 0.0f) results[i] -= x_val;
                }}
            }}

            if (bias != nullptr) {{
                for (int i = 0; i < N_BLOCK; i++) {{
                    results[i] += bias[n + i];
                }}
            }}

            for (int i = 0; i < N_BLOCK; i++) {{
                y_row[n + i] = results[i];
            }}
        }}

        // Handle remaining outputs
        for (; n < N; n++) {{
            float32x4_t sum_pos = vdupq_n_f32(0.0f);
            float32x4_t sum_neg = vdupq_n_f32(0.0f);
            float32x4_t zero = vdupq_n_f32(0.0f);

            for (int k = 0; k < K_SIMD_END; k += K_SIMD) {{
                float32x4_t x_vals = vld1q_f32(x_row + k);
                float32x4_t w_vals = vld1q_f32(w + n * K + k);
                uint32x4_t mask_pos = vcgtq_f32(w_vals, zero);
                uint32x4_t mask_neg = vcltq_f32(w_vals, zero);
                sum_pos = vaddq_f32(sum_pos, vbslq_f32(mask_pos, x_vals, zero));
                sum_neg = vaddq_f32(sum_neg, vbslq_f32(mask_neg, x_vals, zero));
            }}

            float result = vaddvq_f32(sum_pos) - vaddvq_f32(sum_neg);

            for (int k = K_SIMD_END; k < K; k++) {{
                float x_val = x_row[k];
                float w_val = w[n * K + k];
                if (w_val > 0.0f) result += x_val;
                else if (w_val < 0.0f) result -= x_val;
            }}

            if (bias != nullptr) {{
                result += bias[n];
            }}

            y_row[n] = result;
        }}
    }}
}}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {{
    m.def("matmul_free", &matmul_free_neon_block{N_BLOCK}, "MatMul-free with N_BLOCK={N_BLOCK}");
}}
"""
    return template.format(N_BLOCK=n_block)

# Test different blocking sizes
from torch.utils.cpp_extension import load
import tempfile
import os

M, N, K = 256, 1024, 1024
num_threads = 10

x = torch.randn(M, K, dtype=torch.float32)
w = torch.randn(N, K, dtype=torch.float32)
w_ternary = torch.zeros_like(w)
w_ternary[w > 0.5] = 1.0
w_ternary[w < -0.5] = -1.0

print("Testing different output blocking sizes...")
print(f"Matrix size: {M}x{K} @ {K}x{N}")
print(f"Threads: {num_threads}")
print("="*70)

results = []

for n_block in [4, 8, 12, 16, 20, 24, 32]:
    # Create temporary file with kernel
    with tempfile.NamedTemporaryFile(mode='w', suffix='.cpp', delete=False) as f:
        f.write(create_blocked_kernel(n_block))
        kernel_file = f.name

    try:
        # Compile kernel
        print(f"\nCompiling N_BLOCK={n_block}...", end=' ', flush=True)
        kernel = load(
            name=f"block_{n_block}",
            sources=[kernel_file],
            extra_cflags=['-O3', '-DHAS_NEON_SUPPORT', '-Xpreprocessor', '-fopenmp',
                         '-std=c++17', '-mcpu=apple-m1', '-mtune=native'],
            extra_ldflags=['-L/opt/homebrew/lib', '-lomp'],
            verbose=False
        )
        print("✓")

        # Benchmark
        y = torch.zeros(M, N, dtype=torch.float32)
        bias = torch.zeros(N, dtype=torch.float32)

        # Warmup
        for _ in range(10):
            kernel.matmul_free(x, w_ternary, y, bias, M, N, K, num_threads)

        # Benchmark
        times = []
        for _ in range(50):
            start = time.perf_counter()
            kernel.matmul_free(x, w_ternary, y, bias, M, N, K, num_threads)
            times.append(time.perf_counter() - start)

        avg_time = np.mean(times) * 1000
        std_time = np.std(times) * 1000
        gflops = (2.0 * M * N * K / 1e9) / (avg_time / 1000)

        results.append((n_block, gflops, avg_time, std_time))
        print(f"  Time: {avg_time:.2f} ± {std_time:.2f} ms")
        print(f"  GFLOPS: {gflops:.1f}")

    finally:
        os.unlink(kernel_file)

print("\n" + "="*70)
print("Summary:")
print("="*70)
print(f"{'N_BLOCK':<10} {'GFLOPS':<12} {'Time (ms)':<15}")
print("-"*70)

best_gflops = max(results, key=lambda x: x[1])

for n_block, gflops, avg_time, std_time in results:
    marker = " ← BEST" if (n_block, gflops) == (best_gflops[0], best_gflops[1]) else ""
    print(f"{n_block:<10} {gflops:<12.1f} {avg_time:.2f} ± {std_time:.2f}{marker}")

print("="*70)
print(f"\nOptimal blocking: N_BLOCK={best_gflops[0]} ({best_gflops[1]:.1f} GFLOPS)")
