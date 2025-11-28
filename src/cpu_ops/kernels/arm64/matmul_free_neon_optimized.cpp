/*
 * Optimized NEON MatMul-Free Kernel
 *
 * TRUE MatMul-free with optimal 12-way blocking
 * Array-based for clean code (compiler unrolls automatically)
 */

#include <arm_neon.h>
#include <cstring>
#include <algorithm>
#include <torch/extension.h>
#include <omp.h>

#define PREFETCH(addr) __builtin_prefetch(addr, 0, 3)

void matmul_free_neon(
    torch::Tensor x_tensor,
    torch::Tensor w_tensor,
    torch::Tensor y_tensor,
    torch::Tensor bias_tensor,
    int M, int N, int K,
    int num_threads
) {
    const float* x = x_tensor.data_ptr<float>();
    const float* w = w_tensor.data_ptr<float>();
    float* y = y_tensor.data_ptr<float>();
    const float* bias = bias_tensor.defined() ? bias_tensor.data_ptr<float>() : nullptr;

    const int K_SIMD = 4;
    const int K_SIMD_END = (K / K_SIMD) * K_SIMD;
    const int N_BLOCK = 12;  // Optimal blocking
    const int K_PREFETCH = 64;

    omp_set_num_threads(num_threads);

    #pragma omp parallel for schedule(dynamic, 8)
    for (int m = 0; m < M; m++) {
        const float* x_row = x + m * K;
        float* y_row = y + m * N;

        // Process N_BLOCK outputs together
        int n = 0;
        for (; n <= N - N_BLOCK; n += N_BLOCK) {
            // N_BLOCK separate pos/neg accumulators
            float32x4_t sum_pos[N_BLOCK], sum_neg[N_BLOCK];
            for (int i = 0; i < N_BLOCK; i++) {
                sum_pos[i] = vdupq_n_f32(0.0f);
                sum_neg[i] = vdupq_n_f32(0.0f);
            }
            float32x4_t zero = vdupq_n_f32(0.0f);

            // Main SIMD loop with prefetching
            for (int k = 0; k < K_SIMD_END; k += K_SIMD) {
                // Prefetch future data
                if (k + K_PREFETCH < K) {
                    PREFETCH(x_row + k + K_PREFETCH);
                    for (int i = 0; i < N_BLOCK; i += 4) {
                        PREFETCH(w + (n + i) * K + k + K_PREFETCH);
                    }
                }

                // Load input (shared across all outputs)
                float32x4_t x_vals = vld1q_f32(x_row + k);

                // Process all N_BLOCK outputs
                for (int i = 0; i < N_BLOCK; i++) {
                    float32x4_t w_vals = vld1q_f32(w + (n + i) * K + k);

                    // Create masks - TRUE matmul-free!
                    uint32x4_t mask_pos = vcgtq_f32(w_vals, zero);
                    uint32x4_t mask_neg = vcltq_f32(w_vals, zero);

                    // Conditional accumulate (hardware predication)
                    sum_pos[i] = vaddq_f32(sum_pos[i], vbslq_f32(mask_pos, x_vals, zero));
                    sum_neg[i] = vaddq_f32(sum_neg[i], vbslq_f32(mask_neg, x_vals, zero));
                }
            }

            // Horizontal reduction
            float results[N_BLOCK];
            for (int i = 0; i < N_BLOCK; i++) {
                results[i] = vaddvq_f32(sum_pos[i]) - vaddvq_f32(sum_neg[i]);
            }

            // Scalar tail
            for (int k = K_SIMD_END; k < K; k++) {
                float x_val = x_row[k];
                for (int i = 0; i < N_BLOCK; i++) {
                    float w_val = w[(n + i) * K + k];
                    if (w_val > 0.0f) results[i] += x_val;
                    else if (w_val < 0.0f) results[i] -= x_val;
                }
            }

            // Add bias and store
            for (int i = 0; i < N_BLOCK; i++) {
                if (bias != nullptr) {
                    results[i] += bias[n + i];
                }
                y_row[n + i] = results[i];
            }
        }

        // Handle remaining outputs (same pattern)
        for (; n < N; n++) {
            float32x4_t sum_pos = vdupq_n_f32(0.0f);
            float32x4_t sum_neg = vdupq_n_f32(0.0f);
            float32x4_t zero = vdupq_n_f32(0.0f);

            for (int k = 0; k < K_SIMD_END; k += K_SIMD) {
                float32x4_t x_vals = vld1q_f32(x_row + k);
                float32x4_t w_vals = vld1q_f32(w + n * K + k);

                uint32x4_t mask_pos = vcgtq_f32(w_vals, zero);
                uint32x4_t mask_neg = vcltq_f32(w_vals, zero);

                sum_pos = vaddq_f32(sum_pos, vbslq_f32(mask_pos, x_vals, zero));
                sum_neg = vaddq_f32(sum_neg, vbslq_f32(mask_neg, x_vals, zero));
            }

            float result = vaddvq_f32(sum_pos) - vaddvq_f32(sum_neg);

            for (int k = K_SIMD_END; k < K; k++) {
                float x_val = x_row[k];
                float w_val = w[n * K + k];
                if (w_val > 0.0f) result += x_val;
                else if (w_val < 0.0f) result -= x_val;
            }

            if (bias != nullptr) {
                result += bias[n];
            }

            y_row[n] = result;
        }
    }
}

bool has_neon_support() {
    return true;
}
