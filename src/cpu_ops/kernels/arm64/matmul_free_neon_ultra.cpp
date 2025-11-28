/*
 * ULTRA-Optimized NEON MatMul-Free Kernel
 *
 * TRUE MatMul-free with maximum SIMD utilization
 * - Process 8 outputs simultaneously
 * - Prefetching for memory latency hiding
 * - Advanced loop unrolling
 * - Cache-optimized blocking
 */

#include <arm_neon.h>
#include <cstring>
#include <algorithm>
#include <torch/extension.h>
#include <omp.h>

// Prefetch data into cache
#define PREFETCH(addr) __builtin_prefetch(addr, 0, 3)

/**
 * Ultra-optimized NEON MatMul-free kernel
 */
void matmul_free_neon_ultra(
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
    const int N_BLOCK = 8;  // Process 8 outputs for maximum throughput
    const int K_PREFETCH = 64;  // Prefetch distance

    omp_set_num_threads(num_threads);

    #pragma omp parallel for schedule(dynamic, 8)
    for (int m = 0; m < M; m++) {
        const float* x_row = x + m * K;
        float* y_row = y + m * N;

        // Process 8 outputs together for maximum SIMD utilization
        int n = 0;
        for (; n <= N - N_BLOCK; n += N_BLOCK) {
            // 8 separate pos/neg accumulators
            float32x4_t sum_pos0 = vdupq_n_f32(0.0f), sum_neg0 = vdupq_n_f32(0.0f);
            float32x4_t sum_pos1 = vdupq_n_f32(0.0f), sum_neg1 = vdupq_n_f32(0.0f);
            float32x4_t sum_pos2 = vdupq_n_f32(0.0f), sum_neg2 = vdupq_n_f32(0.0f);
            float32x4_t sum_pos3 = vdupq_n_f32(0.0f), sum_neg3 = vdupq_n_f32(0.0f);
            float32x4_t sum_pos4 = vdupq_n_f32(0.0f), sum_neg4 = vdupq_n_f32(0.0f);
            float32x4_t sum_pos5 = vdupq_n_f32(0.0f), sum_neg5 = vdupq_n_f32(0.0f);
            float32x4_t sum_pos6 = vdupq_n_f32(0.0f), sum_neg6 = vdupq_n_f32(0.0f);
            float32x4_t sum_pos7 = vdupq_n_f32(0.0f), sum_neg7 = vdupq_n_f32(0.0f);

            float32x4_t zero = vdupq_n_f32(0.0f);

            // Main SIMD loop with prefetching
            for (int k = 0; k < K_SIMD_END; k += K_SIMD) {
                // Prefetch future data
                if (k + K_PREFETCH < K) {
                    PREFETCH(x_row + k + K_PREFETCH);
                    PREFETCH(w + (n + 0) * K + k + K_PREFETCH);
                    PREFETCH(w + (n + 4) * K + k + K_PREFETCH);
                }

                // Load input (shared across all outputs)
                float32x4_t x_vals = vld1q_f32(x_row + k);

                // Load weights for 8 outputs
                float32x4_t w0 = vld1q_f32(w + (n + 0) * K + k);
                float32x4_t w1 = vld1q_f32(w + (n + 1) * K + k);
                float32x4_t w2 = vld1q_f32(w + (n + 2) * K + k);
                float32x4_t w3 = vld1q_f32(w + (n + 3) * K + k);
                float32x4_t w4 = vld1q_f32(w + (n + 4) * K + k);
                float32x4_t w5 = vld1q_f32(w + (n + 5) * K + k);
                float32x4_t w6 = vld1q_f32(w + (n + 6) * K + k);
                float32x4_t w7 = vld1q_f32(w + (n + 7) * K + k);

                // Create masks - TRUE matmul-free!
                uint32x4_t mask_pos0 = vcgtq_f32(w0, zero);
                uint32x4_t mask_neg0 = vcltq_f32(w0, zero);
                uint32x4_t mask_pos1 = vcgtq_f32(w1, zero);
                uint32x4_t mask_neg1 = vcltq_f32(w1, zero);
                uint32x4_t mask_pos2 = vcgtq_f32(w2, zero);
                uint32x4_t mask_neg2 = vcltq_f32(w2, zero);
                uint32x4_t mask_pos3 = vcgtq_f32(w3, zero);
                uint32x4_t mask_neg3 = vcltq_f32(w3, zero);
                uint32x4_t mask_pos4 = vcgtq_f32(w4, zero);
                uint32x4_t mask_neg4 = vcltq_f32(w4, zero);
                uint32x4_t mask_pos5 = vcgtq_f32(w5, zero);
                uint32x4_t mask_neg5 = vcltq_f32(w5, zero);
                uint32x4_t mask_pos6 = vcgtq_f32(w6, zero);
                uint32x4_t mask_neg6 = vcltq_f32(w6, zero);
                uint32x4_t mask_pos7 = vcgtq_f32(w7, zero);
                uint32x4_t mask_neg7 = vcltq_f32(w7, zero);

                // Conditional accumulate (hardware predication, not multiplication!)
                sum_pos0 = vaddq_f32(sum_pos0, vbslq_f32(mask_pos0, x_vals, zero));
                sum_neg0 = vaddq_f32(sum_neg0, vbslq_f32(mask_neg0, x_vals, zero));
                sum_pos1 = vaddq_f32(sum_pos1, vbslq_f32(mask_pos1, x_vals, zero));
                sum_neg1 = vaddq_f32(sum_neg1, vbslq_f32(mask_neg1, x_vals, zero));
                sum_pos2 = vaddq_f32(sum_pos2, vbslq_f32(mask_pos2, x_vals, zero));
                sum_neg2 = vaddq_f32(sum_neg2, vbslq_f32(mask_neg2, x_vals, zero));
                sum_pos3 = vaddq_f32(sum_pos3, vbslq_f32(mask_pos3, x_vals, zero));
                sum_neg3 = vaddq_f32(sum_neg3, vbslq_f32(mask_neg3, x_vals, zero));
                sum_pos4 = vaddq_f32(sum_pos4, vbslq_f32(mask_pos4, x_vals, zero));
                sum_neg4 = vaddq_f32(sum_neg4, vbslq_f32(mask_neg4, x_vals, zero));
                sum_pos5 = vaddq_f32(sum_pos5, vbslq_f32(mask_pos5, x_vals, zero));
                sum_neg5 = vaddq_f32(sum_neg5, vbslq_f32(mask_neg5, x_vals, zero));
                sum_pos6 = vaddq_f32(sum_pos6, vbslq_f32(mask_pos6, x_vals, zero));
                sum_neg6 = vaddq_f32(sum_neg6, vbslq_f32(mask_neg6, x_vals, zero));
                sum_pos7 = vaddq_f32(sum_pos7, vbslq_f32(mask_pos7, x_vals, zero));
                sum_neg7 = vaddq_f32(sum_neg7, vbslq_f32(mask_neg7, x_vals, zero));
            }

            // Horizontal reduction
            float r0 = vaddvq_f32(sum_pos0) - vaddvq_f32(sum_neg0);
            float r1 = vaddvq_f32(sum_pos1) - vaddvq_f32(sum_neg1);
            float r2 = vaddvq_f32(sum_pos2) - vaddvq_f32(sum_neg2);
            float r3 = vaddvq_f32(sum_pos3) - vaddvq_f32(sum_neg3);
            float r4 = vaddvq_f32(sum_pos4) - vaddvq_f32(sum_neg4);
            float r5 = vaddvq_f32(sum_pos5) - vaddvq_f32(sum_neg5);
            float r6 = vaddvq_f32(sum_pos6) - vaddvq_f32(sum_neg6);
            float r7 = vaddvq_f32(sum_pos7) - vaddvq_f32(sum_neg7);

            // Scalar tail
            for (int k = K_SIMD_END; k < K; k++) {
                float x_val = x_row[k];
                float w0_val = w[(n + 0) * K + k];
                float w1_val = w[(n + 1) * K + k];
                float w2_val = w[(n + 2) * K + k];
                float w3_val = w[(n + 3) * K + k];
                float w4_val = w[(n + 4) * K + k];
                float w5_val = w[(n + 5) * K + k];
                float w6_val = w[(n + 6) * K + k];
                float w7_val = w[(n + 7) * K + k];

                // TRUE matmul-free
                if (w0_val > 0.0f) r0 += x_val; else if (w0_val < 0.0f) r0 -= x_val;
                if (w1_val > 0.0f) r1 += x_val; else if (w1_val < 0.0f) r1 -= x_val;
                if (w2_val > 0.0f) r2 += x_val; else if (w2_val < 0.0f) r2 -= x_val;
                if (w3_val > 0.0f) r3 += x_val; else if (w3_val < 0.0f) r3 -= x_val;
                if (w4_val > 0.0f) r4 += x_val; else if (w4_val < 0.0f) r4 -= x_val;
                if (w5_val > 0.0f) r5 += x_val; else if (w5_val < 0.0f) r5 -= x_val;
                if (w6_val > 0.0f) r6 += x_val; else if (w6_val < 0.0f) r6 -= x_val;
                if (w7_val > 0.0f) r7 += x_val; else if (w7_val < 0.0f) r7 -= x_val;
            }

            // Add bias
            if (bias != nullptr) {
                r0 += bias[n + 0];
                r1 += bias[n + 1];
                r2 += bias[n + 2];
                r3 += bias[n + 3];
                r4 += bias[n + 4];
                r5 += bias[n + 5];
                r6 += bias[n + 6];
                r7 += bias[n + 7];
            }

            // Store results
            y_row[n + 0] = r0;
            y_row[n + 1] = r1;
            y_row[n + 2] = r2;
            y_row[n + 3] = r3;
            y_row[n + 4] = r4;
            y_row[n + 5] = r5;
            y_row[n + 6] = r6;
            y_row[n + 7] = r7;
        }

        // Handle remaining outputs (same as before)
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

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("matmul_free_neon_ultra", &matmul_free_neon_ultra, "Ultra-optimized MatMul-free");
    m.def("has_neon_support", &has_neon_support, "Check NEON support");
}
