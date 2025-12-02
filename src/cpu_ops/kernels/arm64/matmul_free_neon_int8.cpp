/*
 * ARM NEON Int8 Kernels with SDOT (Dot Product Instructions)
 *
 * This is the ARM equivalent of matmul_free_tmac_vnni.cpp
 * Uses ARM's vdotq_s32 instruction for int8 dot products.
 *
 * Key difference from x86 VNNI:
 * - ARM SDOT does int8 x int8 directly (no uint8 offset trick needed!)
 * - NEON is 128-bit (16 int8) vs AVX-512's 512-bit (64 int8)
 * - Need 4x more instructions for same throughput
 *
 * Target: Apple Silicon (M1/M2/M3/M4) and ARMv8.2+ with dotprod
 */

#include <arm_neon.h>
#include <cstring>
#include <cstdint>
#include <cmath>
#include <algorithm>
#include <omp.h>
#include <torch/extension.h>

// Tile sizes for cache efficiency
constexpr int TILE_M = 32;   // Process 32 rows at a time
constexpr int TILE_N = 64;   // Process 64 columns at a time
constexpr int N_BLOCK = 4;   // Register blocking: 4 outputs at once

/**
 * Quantize activations to int8 with per-row scaling
 * Same logic as x86 version, but using NEON intrinsics
 */
void quantize_activations_int8_neon(
    torch::Tensor x_float,
    torch::Tensor x_int8,
    torch::Tensor scale_tensor,
    int M, int K
) {
    const float* x = x_float.data_ptr<float>();
    int8_t* x_q = x_int8.data_ptr<int8_t>();
    float* scales = scale_tensor.data_ptr<float>();

    #pragma omp parallel for
    for (int m = 0; m < M; m++) {
        const float* x_row = x + m * K;
        int8_t* xq_row = x_q + m * K;

        // Find max absolute value using NEON
        float32x4_t max_vec = vdupq_n_f32(0.0f);
        int k = 0;
        for (; k + 3 < K; k += 4) {
            float32x4_t x_vec = vld1q_f32(x_row + k);
            float32x4_t abs_vec = vabsq_f32(x_vec);
            max_vec = vmaxq_f32(max_vec, abs_vec);
        }

        // Horizontal max
        float max_abs = vmaxvq_f32(max_vec);

        // Handle remainder
        for (; k < K; k++) {
            float abs_val = std::abs(x_row[k]);
            if (abs_val > max_abs) max_abs = abs_val;
        }

        float scale = max_abs / 127.0f;
        if (scale == 0.0f) scale = 1.0f;
        scales[m] = scale;

        float inv_scale = 1.0f / scale;

        // Quantize using NEON
        float32x4_t inv_scale_vec = vdupq_n_f32(inv_scale);
        float32x4_t min_val = vdupq_n_f32(-127.0f);
        float32x4_t max_val = vdupq_n_f32(127.0f);

        k = 0;
        for (; k + 3 < K; k += 4) {
            float32x4_t x_vec = vld1q_f32(x_row + k);
            float32x4_t scaled = vmulq_f32(x_vec, inv_scale_vec);
            scaled = vmaxq_f32(min_val, vminq_f32(max_val, scaled));

            // Round and convert to int32
            int32x4_t rounded = vcvtnq_s32_f32(scaled);

            // Narrow to int16 then int8
            int16x4_t narrow16 = vmovn_s32(rounded);
            int8x8_t narrow8 = vmovn_s16(vcombine_s16(narrow16, narrow16));

            // Store first 4 elements
            vst1_lane_s8(xq_row + k + 0, narrow8, 0);
            vst1_lane_s8(xq_row + k + 1, narrow8, 1);
            vst1_lane_s8(xq_row + k + 2, narrow8, 2);
            vst1_lane_s8(xq_row + k + 3, narrow8, 3);
        }

        // Scalar remainder
        for (; k < K; k++) {
            float val = x_row[k] * inv_scale;
            val = std::max(-127.0f, std::min(127.0f, val));
            xq_row[k] = static_cast<int8_t>(std::round(val));
        }
    }
}

/**
 * Pack ternary weights to int8 format and compute weight sums
 *
 * Layout: [N, K_padded] int8 where K_padded = ceil(K/16)*16
 * Also computes sum of weights for each output (not needed for ARM SDOT,
 * but kept for API compatibility with x86)
 */
void pack_weights_neon_int8(
    torch::Tensor w_tensor,        // [N, K] float32 ternary
    torch::Tensor w_int8_tensor,   // [N, K_padded] int8 output
    torch::Tensor w_sum_tensor,    // [N] int32 output
    int N, int K
) {
    const float* w = w_tensor.data_ptr<float>();
    int8_t* w_int8 = w_int8_tensor.data_ptr<int8_t>();
    int32_t* w_sum = w_sum_tensor.data_ptr<int32_t>();

    const int K_padded = ((K + 15) / 16) * 16;

    #pragma omp parallel for
    for (int n = 0; n < N; n++) {
        int32_t sum = 0;
        for (int k = 0; k < K; k++) {
            float val = w[n * K + k];
            int8_t w_val;
            if (val > 0.5f) {
                w_val = 1;
            } else if (val < -0.5f) {
                w_val = -1;
            } else {
                w_val = 0;
            }
            w_int8[n * K_padded + k] = w_val;
            sum += w_val;
        }
        // Pad with zeros
        for (int k = K; k < K_padded; k++) {
            w_int8[n * K_padded + k] = 0;
        }
        w_sum[n] = sum;
    }
}

/**
 * Simple SDOT kernel - baseline implementation
 *
 * Uses vdotq_s32 for int8 dot products:
 *   acc[i] += sum_{j=0}^{3} (a[4i+j] * b[4i+j])
 *
 * Each vdotq_s32 processes 16 int8 pairs into 4 int32 accumulators.
 */
void matmul_free_neon_sdot_simple(
    torch::Tensor x_int8_tensor,    // [M, K] int8
    torch::Tensor scale_tensor,     // [M] float32
    torch::Tensor w_int8_tensor,    // [N, K_padded] int8
    torch::Tensor w_sum_tensor,     // [N] int32 (unused for ARM, kept for API)
    torch::Tensor y_tensor,         // [M, N] float32 output
    torch::Tensor bias_tensor,
    int M, int N, int K,
    int num_threads
) {
    const int8_t* __restrict__ x_int8 = x_int8_tensor.data_ptr<int8_t>();
    const float* __restrict__ scales = scale_tensor.data_ptr<float>();
    const int8_t* __restrict__ w_int8 = w_int8_tensor.data_ptr<int8_t>();
    float* __restrict__ y = y_tensor.data_ptr<float>();
    const float* bias = bias_tensor.defined() ? bias_tensor.data_ptr<float>() : nullptr;

    const int K_padded = ((K + 15) / 16) * 16;

    omp_set_num_threads(num_threads);

    #pragma omp parallel for schedule(static)
    for (int m = 0; m < M; m++) {
        const int8_t* __restrict__ x_row = x_int8 + m * K;
        float scale = scales[m];
        float* y_row = y + m * N;

        // Process each output
        for (int n = 0; n < N; n++) {
            const int8_t* __restrict__ w_row = w_int8 + n * K_padded;

            // Initialize accumulators (4 int32 lanes)
            int32x4_t acc = vdupq_n_s32(0);

            // Main SDOT loop - 16 elements per iteration
            int k = 0;
            for (; k + 15 < K; k += 16) {
                // Load 16 activations and weights
                int8x16_t x_vec = vld1q_s8(x_row + k);
                int8x16_t w_vec = vld1q_s8(w_row + k);

                // Dot product: acc += dot(x_vec, w_vec)
                // vdotq_s32 computes 4 dot products of 4 int8 pairs each
                acc = vdotq_s32(acc, x_vec, w_vec);
            }

            // Horizontal sum of 4 int32 values
            int32_t sum = vaddvq_s32(acc);

            // Handle remaining elements with scalar
            for (; k < K; k++) {
                sum += static_cast<int32_t>(x_row[k]) * static_cast<int32_t>(w_row[k]);
            }

            y_row[n] = static_cast<float>(sum) * scale + (bias ? bias[n] : 0.0f);
        }
    }
}

/**
 * SDOT v2 - Register-blocked kernel
 *
 * Key optimizations:
 * 1. Process 4 outputs simultaneously (register blocking)
 * 2. Reuse activation vector across 4 weight vectors
 * 3. Better instruction-level parallelism
 */
void matmul_free_neon_sdot_v2(
    torch::Tensor x_int8_tensor,    // [M, K] int8
    torch::Tensor scale_tensor,     // [M] float32
    torch::Tensor w_int8_tensor,    // [N, K_padded] int8
    torch::Tensor w_sum_tensor,     // [N] int32 (unused)
    torch::Tensor y_tensor,         // [M, N] float32 output
    torch::Tensor bias_tensor,
    int M, int N, int K,
    int num_threads
) {
    const int8_t* __restrict__ x_int8 = x_int8_tensor.data_ptr<int8_t>();
    const float* __restrict__ scales = scale_tensor.data_ptr<float>();
    const int8_t* __restrict__ w_int8 = w_int8_tensor.data_ptr<int8_t>();
    float* __restrict__ y = y_tensor.data_ptr<float>();
    const float* bias = bias_tensor.defined() ? bias_tensor.data_ptr<float>() : nullptr;

    const int K_padded = ((K + 15) / 16) * 16;

    omp_set_num_threads(num_threads);

    #pragma omp parallel for schedule(static)
    for (int m = 0; m < M; m++) {
        const int8_t* __restrict__ x_row = x_int8 + m * K;
        float scale = scales[m];
        float* y_row = y + m * N;

        // Process 4 outputs at a time (register blocking)
        int n = 0;
        for (; n + N_BLOCK - 1 < N; n += N_BLOCK) {
            // 4 independent accumulators
            int32x4_t acc0 = vdupq_n_s32(0);
            int32x4_t acc1 = vdupq_n_s32(0);
            int32x4_t acc2 = vdupq_n_s32(0);
            int32x4_t acc3 = vdupq_n_s32(0);

            const int8_t* w0 = w_int8 + (n + 0) * K_padded;
            const int8_t* w1 = w_int8 + (n + 1) * K_padded;
            const int8_t* w2 = w_int8 + (n + 2) * K_padded;
            const int8_t* w3 = w_int8 + (n + 3) * K_padded;

            // Main loop - process 16 elements at a time
            int k = 0;
            for (; k + 15 < K; k += 16) {
                // Load activations ONCE, use for all 4 outputs
                int8x16_t x_vec = vld1q_s8(x_row + k);

                // Load 4 weight vectors and compute dot products
                acc0 = vdotq_s32(acc0, x_vec, vld1q_s8(w0 + k));
                acc1 = vdotq_s32(acc1, x_vec, vld1q_s8(w1 + k));
                acc2 = vdotq_s32(acc2, x_vec, vld1q_s8(w2 + k));
                acc3 = vdotq_s32(acc3, x_vec, vld1q_s8(w3 + k));
            }

            // Horizontal sums
            int32_t sum0 = vaddvq_s32(acc0);
            int32_t sum1 = vaddvq_s32(acc1);
            int32_t sum2 = vaddvq_s32(acc2);
            int32_t sum3 = vaddvq_s32(acc3);

            // Handle remainder (scalar)
            for (; k < K; k++) {
                int8_t x_val = x_row[k];
                sum0 += static_cast<int32_t>(x_val) * static_cast<int32_t>(w0[k]);
                sum1 += static_cast<int32_t>(x_val) * static_cast<int32_t>(w1[k]);
                sum2 += static_cast<int32_t>(x_val) * static_cast<int32_t>(w2[k]);
                sum3 += static_cast<int32_t>(x_val) * static_cast<int32_t>(w3[k]);
            }

            // Store results
            y_row[n + 0] = static_cast<float>(sum0) * scale + (bias ? bias[n + 0] : 0.0f);
            y_row[n + 1] = static_cast<float>(sum1) * scale + (bias ? bias[n + 1] : 0.0f);
            y_row[n + 2] = static_cast<float>(sum2) * scale + (bias ? bias[n + 2] : 0.0f);
            y_row[n + 3] = static_cast<float>(sum3) * scale + (bias ? bias[n + 3] : 0.0f);
        }

        // Handle remaining outputs
        for (; n < N; n++) {
            const int8_t* w_row = w_int8 + n * K_padded;
            int32x4_t acc = vdupq_n_s32(0);

            int k = 0;
            for (; k + 15 < K; k += 16) {
                int8x16_t x_vec = vld1q_s8(x_row + k);
                acc = vdotq_s32(acc, x_vec, vld1q_s8(w_row + k));
            }

            int32_t sum = vaddvq_s32(acc);
            for (; k < K; k++) {
                sum += static_cast<int32_t>(x_row[k]) * static_cast<int32_t>(w_row[k]);
            }

            y_row[n] = static_cast<float>(sum) * scale + (bias ? bias[n] : 0.0f);
        }
    }
}

/**
 * SDOT v3 - Loop unrolled kernel with K unrolling
 *
 * Key optimization: Unroll K loop to reduce loop overhead and
 * enable better instruction scheduling.
 */
void matmul_free_neon_sdot_v3(
    torch::Tensor x_int8_tensor,    // [M, K] int8
    torch::Tensor scale_tensor,     // [M] float32
    torch::Tensor w_int8_tensor,    // [N, K_padded] int8
    torch::Tensor w_sum_tensor,     // [N] int32 (unused)
    torch::Tensor y_tensor,         // [M, N] float32 output
    torch::Tensor bias_tensor,
    int M, int N, int K,
    int num_threads
) {
    const int8_t* __restrict__ x_int8 = x_int8_tensor.data_ptr<int8_t>();
    const float* __restrict__ scales = scale_tensor.data_ptr<float>();
    const int8_t* __restrict__ w_int8 = w_int8_tensor.data_ptr<int8_t>();
    float* __restrict__ y = y_tensor.data_ptr<float>();
    const float* bias = bias_tensor.defined() ? bias_tensor.data_ptr<float>() : nullptr;

    const int K_padded = ((K + 15) / 16) * 16;

    omp_set_num_threads(num_threads);

    // Parallelize over M rows (simple, minimal overhead)
    #pragma omp parallel for schedule(static)
    for (int m = 0; m < M; m++) {
        const int8_t* __restrict__ x_row = x_int8 + m * K;
        float scale = scales[m];
        float* y_row = y + m * N;

        // Process 4 outputs at a time with K loop unrolling
        int n = 0;
        for (; n + 3 < N; n += 4) {
            int32x4_t acc0 = vdupq_n_s32(0);
            int32x4_t acc1 = vdupq_n_s32(0);
            int32x4_t acc2 = vdupq_n_s32(0);
            int32x4_t acc3 = vdupq_n_s32(0);

            const int8_t* w0 = w_int8 + (n + 0) * K_padded;
            const int8_t* w1 = w_int8 + (n + 1) * K_padded;
            const int8_t* w2 = w_int8 + (n + 2) * K_padded;
            const int8_t* w3 = w_int8 + (n + 3) * K_padded;

            // Unroll K by 4 (64 bytes = 4 vectors)
            int k = 0;
            for (; k + 63 < K; k += 64) {
                // Load 4 activation vectors
                int8x16_t x0 = vld1q_s8(x_row + k);
                int8x16_t x1 = vld1q_s8(x_row + k + 16);
                int8x16_t x2 = vld1q_s8(x_row + k + 32);
                int8x16_t x3 = vld1q_s8(x_row + k + 48);

                // Process output 0
                acc0 = vdotq_s32(acc0, x0, vld1q_s8(w0 + k));
                acc0 = vdotq_s32(acc0, x1, vld1q_s8(w0 + k + 16));
                acc0 = vdotq_s32(acc0, x2, vld1q_s8(w0 + k + 32));
                acc0 = vdotq_s32(acc0, x3, vld1q_s8(w0 + k + 48));

                // Process output 1
                acc1 = vdotq_s32(acc1, x0, vld1q_s8(w1 + k));
                acc1 = vdotq_s32(acc1, x1, vld1q_s8(w1 + k + 16));
                acc1 = vdotq_s32(acc1, x2, vld1q_s8(w1 + k + 32));
                acc1 = vdotq_s32(acc1, x3, vld1q_s8(w1 + k + 48));

                // Process output 2
                acc2 = vdotq_s32(acc2, x0, vld1q_s8(w2 + k));
                acc2 = vdotq_s32(acc2, x1, vld1q_s8(w2 + k + 16));
                acc2 = vdotq_s32(acc2, x2, vld1q_s8(w2 + k + 32));
                acc2 = vdotq_s32(acc2, x3, vld1q_s8(w2 + k + 48));

                // Process output 3
                acc3 = vdotq_s32(acc3, x0, vld1q_s8(w3 + k));
                acc3 = vdotq_s32(acc3, x1, vld1q_s8(w3 + k + 16));
                acc3 = vdotq_s32(acc3, x2, vld1q_s8(w3 + k + 32));
                acc3 = vdotq_s32(acc3, x3, vld1q_s8(w3 + k + 48));
            }

            // Handle remaining K (16 at a time)
            for (; k + 15 < K; k += 16) {
                int8x16_t x_vec = vld1q_s8(x_row + k);
                acc0 = vdotq_s32(acc0, x_vec, vld1q_s8(w0 + k));
                acc1 = vdotq_s32(acc1, x_vec, vld1q_s8(w1 + k));
                acc2 = vdotq_s32(acc2, x_vec, vld1q_s8(w2 + k));
                acc3 = vdotq_s32(acc3, x_vec, vld1q_s8(w3 + k));
            }

            int32_t sum0 = vaddvq_s32(acc0);
            int32_t sum1 = vaddvq_s32(acc1);
            int32_t sum2 = vaddvq_s32(acc2);
            int32_t sum3 = vaddvq_s32(acc3);

            // Scalar remainder
            for (; k < K; k++) {
                int8_t x_val = x_row[k];
                sum0 += static_cast<int32_t>(x_val) * static_cast<int32_t>(w0[k]);
                sum1 += static_cast<int32_t>(x_val) * static_cast<int32_t>(w1[k]);
                sum2 += static_cast<int32_t>(x_val) * static_cast<int32_t>(w2[k]);
                sum3 += static_cast<int32_t>(x_val) * static_cast<int32_t>(w3[k]);
            }

            y_row[n + 0] = static_cast<float>(sum0) * scale + (bias ? bias[n + 0] : 0.0f);
            y_row[n + 1] = static_cast<float>(sum1) * scale + (bias ? bias[n + 1] : 0.0f);
            y_row[n + 2] = static_cast<float>(sum2) * scale + (bias ? bias[n + 2] : 0.0f);
            y_row[n + 3] = static_cast<float>(sum3) * scale + (bias ? bias[n + 3] : 0.0f);
        }

        // Remainder outputs
        for (; n < N; n++) {
            const int8_t* w_row = w_int8 + n * K_padded;
            int32x4_t acc = vdupq_n_s32(0);

            int k = 0;
            for (; k + 15 < K; k += 16) {
                int8x16_t x_vec = vld1q_s8(x_row + k);
                acc = vdotq_s32(acc, x_vec, vld1q_s8(w_row + k));
            }

            int32_t sum = vaddvq_s32(acc);
            for (; k < K; k++) {
                sum += static_cast<int32_t>(x_row[k]) * static_cast<int32_t>(w_row[k]);
            }

            y_row[n] = static_cast<float>(sum) * scale + (bias ? bias[n] : 0.0f);
        }
    }
}

/**
 * SDOT v4 - Extended register blocking (8 outputs)
 *
 * ARM NEON has 32 registers, so we can afford more accumulators.
 * Process 8 outputs at a time for better throughput.
 */
void matmul_free_neon_sdot_v4(
    torch::Tensor x_int8_tensor,    // [M, K] int8
    torch::Tensor scale_tensor,     // [M] float32
    torch::Tensor w_int8_tensor,    // [N, K_padded] int8
    torch::Tensor w_sum_tensor,     // [N] int32 (unused)
    torch::Tensor y_tensor,         // [M, N] float32 output
    torch::Tensor bias_tensor,
    int M, int N, int K,
    int num_threads
) {
    const int8_t* __restrict__ x_int8 = x_int8_tensor.data_ptr<int8_t>();
    const float* __restrict__ scales = scale_tensor.data_ptr<float>();
    const int8_t* __restrict__ w_int8 = w_int8_tensor.data_ptr<int8_t>();
    float* __restrict__ y = y_tensor.data_ptr<float>();
    const float* bias = bias_tensor.defined() ? bias_tensor.data_ptr<float>() : nullptr;

    const int K_padded = ((K + 15) / 16) * 16;
    constexpr int N_BLOCK_8 = 8;

    omp_set_num_threads(num_threads);

    #pragma omp parallel for schedule(static)
    for (int m = 0; m < M; m++) {
        const int8_t* __restrict__ x_row = x_int8 + m * K;
        float scale = scales[m];
        float* y_row = y + m * N;

        // Process 8 outputs at a time
        int n = 0;
        for (; n + N_BLOCK_8 - 1 < N; n += N_BLOCK_8) {
            // 8 independent accumulators
            int32x4_t acc0 = vdupq_n_s32(0);
            int32x4_t acc1 = vdupq_n_s32(0);
            int32x4_t acc2 = vdupq_n_s32(0);
            int32x4_t acc3 = vdupq_n_s32(0);
            int32x4_t acc4 = vdupq_n_s32(0);
            int32x4_t acc5 = vdupq_n_s32(0);
            int32x4_t acc6 = vdupq_n_s32(0);
            int32x4_t acc7 = vdupq_n_s32(0);

            const int8_t* w0 = w_int8 + (n + 0) * K_padded;
            const int8_t* w1 = w_int8 + (n + 1) * K_padded;
            const int8_t* w2 = w_int8 + (n + 2) * K_padded;
            const int8_t* w3 = w_int8 + (n + 3) * K_padded;
            const int8_t* w4 = w_int8 + (n + 4) * K_padded;
            const int8_t* w5 = w_int8 + (n + 5) * K_padded;
            const int8_t* w6 = w_int8 + (n + 6) * K_padded;
            const int8_t* w7 = w_int8 + (n + 7) * K_padded;

            // Main loop
            int k = 0;
            for (; k + 15 < K; k += 16) {
                int8x16_t x_vec = vld1q_s8(x_row + k);

                acc0 = vdotq_s32(acc0, x_vec, vld1q_s8(w0 + k));
                acc1 = vdotq_s32(acc1, x_vec, vld1q_s8(w1 + k));
                acc2 = vdotq_s32(acc2, x_vec, vld1q_s8(w2 + k));
                acc3 = vdotq_s32(acc3, x_vec, vld1q_s8(w3 + k));
                acc4 = vdotq_s32(acc4, x_vec, vld1q_s8(w4 + k));
                acc5 = vdotq_s32(acc5, x_vec, vld1q_s8(w5 + k));
                acc6 = vdotq_s32(acc6, x_vec, vld1q_s8(w6 + k));
                acc7 = vdotq_s32(acc7, x_vec, vld1q_s8(w7 + k));
            }

            // Horizontal sums
            int32_t sum0 = vaddvq_s32(acc0);
            int32_t sum1 = vaddvq_s32(acc1);
            int32_t sum2 = vaddvq_s32(acc2);
            int32_t sum3 = vaddvq_s32(acc3);
            int32_t sum4 = vaddvq_s32(acc4);
            int32_t sum5 = vaddvq_s32(acc5);
            int32_t sum6 = vaddvq_s32(acc6);
            int32_t sum7 = vaddvq_s32(acc7);

            // Scalar remainder
            for (; k < K; k++) {
                int8_t x_val = x_row[k];
                sum0 += static_cast<int32_t>(x_val) * static_cast<int32_t>(w0[k]);
                sum1 += static_cast<int32_t>(x_val) * static_cast<int32_t>(w1[k]);
                sum2 += static_cast<int32_t>(x_val) * static_cast<int32_t>(w2[k]);
                sum3 += static_cast<int32_t>(x_val) * static_cast<int32_t>(w3[k]);
                sum4 += static_cast<int32_t>(x_val) * static_cast<int32_t>(w4[k]);
                sum5 += static_cast<int32_t>(x_val) * static_cast<int32_t>(w5[k]);
                sum6 += static_cast<int32_t>(x_val) * static_cast<int32_t>(w6[k]);
                sum7 += static_cast<int32_t>(x_val) * static_cast<int32_t>(w7[k]);
            }

            // Store results
            y_row[n + 0] = static_cast<float>(sum0) * scale + (bias ? bias[n + 0] : 0.0f);
            y_row[n + 1] = static_cast<float>(sum1) * scale + (bias ? bias[n + 1] : 0.0f);
            y_row[n + 2] = static_cast<float>(sum2) * scale + (bias ? bias[n + 2] : 0.0f);
            y_row[n + 3] = static_cast<float>(sum3) * scale + (bias ? bias[n + 3] : 0.0f);
            y_row[n + 4] = static_cast<float>(sum4) * scale + (bias ? bias[n + 4] : 0.0f);
            y_row[n + 5] = static_cast<float>(sum5) * scale + (bias ? bias[n + 5] : 0.0f);
            y_row[n + 6] = static_cast<float>(sum6) * scale + (bias ? bias[n + 6] : 0.0f);
            y_row[n + 7] = static_cast<float>(sum7) * scale + (bias ? bias[n + 7] : 0.0f);
        }

        // Process remaining with 4-way blocking
        for (; n + 3 < N; n += 4) {
            int32x4_t acc0 = vdupq_n_s32(0);
            int32x4_t acc1 = vdupq_n_s32(0);
            int32x4_t acc2 = vdupq_n_s32(0);
            int32x4_t acc3 = vdupq_n_s32(0);

            const int8_t* w0 = w_int8 + (n + 0) * K_padded;
            const int8_t* w1 = w_int8 + (n + 1) * K_padded;
            const int8_t* w2 = w_int8 + (n + 2) * K_padded;
            const int8_t* w3 = w_int8 + (n + 3) * K_padded;

            int k = 0;
            for (; k + 15 < K; k += 16) {
                int8x16_t x_vec = vld1q_s8(x_row + k);
                acc0 = vdotq_s32(acc0, x_vec, vld1q_s8(w0 + k));
                acc1 = vdotq_s32(acc1, x_vec, vld1q_s8(w1 + k));
                acc2 = vdotq_s32(acc2, x_vec, vld1q_s8(w2 + k));
                acc3 = vdotq_s32(acc3, x_vec, vld1q_s8(w3 + k));
            }

            int32_t sum0 = vaddvq_s32(acc0);
            int32_t sum1 = vaddvq_s32(acc1);
            int32_t sum2 = vaddvq_s32(acc2);
            int32_t sum3 = vaddvq_s32(acc3);

            for (; k < K; k++) {
                int8_t x_val = x_row[k];
                sum0 += static_cast<int32_t>(x_val) * static_cast<int32_t>(w0[k]);
                sum1 += static_cast<int32_t>(x_val) * static_cast<int32_t>(w1[k]);
                sum2 += static_cast<int32_t>(x_val) * static_cast<int32_t>(w2[k]);
                sum3 += static_cast<int32_t>(x_val) * static_cast<int32_t>(w3[k]);
            }

            y_row[n + 0] = static_cast<float>(sum0) * scale + (bias ? bias[n + 0] : 0.0f);
            y_row[n + 1] = static_cast<float>(sum1) * scale + (bias ? bias[n + 1] : 0.0f);
            y_row[n + 2] = static_cast<float>(sum2) * scale + (bias ? bias[n + 2] : 0.0f);
            y_row[n + 3] = static_cast<float>(sum3) * scale + (bias ? bias[n + 3] : 0.0f);
        }

        // Scalar remainder
        for (; n < N; n++) {
            const int8_t* w_row = w_int8 + n * K_padded;
            int32x4_t acc = vdupq_n_s32(0);

            int k = 0;
            for (; k + 15 < K; k += 16) {
                int8x16_t x_vec = vld1q_s8(x_row + k);
                acc = vdotq_s32(acc, x_vec, vld1q_s8(w_row + k));
            }

            int32_t sum = vaddvq_s32(acc);
            for (; k < K; k++) {
                sum += static_cast<int32_t>(x_row[k]) * static_cast<int32_t>(w_row[k]);
            }

            y_row[n] = static_cast<float>(sum) * scale + (bias ? bias[n] : 0.0f);
        }
    }
}

#ifdef APPLE_SILICON
#include <Accelerate/Accelerate.h>

/**
 * Apple Accelerate BLAS wrapper
 *
 * Uses cblas_sgemm which internally leverages AMX for maximum performance.
 * Converts int8 ternary weights to float32 on-the-fly.
 *
 * This achieves ~1500+ GFLOPS by using Apple's optimized BLAS.
 */
void matmul_free_accelerate(
    torch::Tensor x_tensor,         // [M, K] float32 activations
    torch::Tensor w_int8_tensor,    // [N, K] int8 ternary weights
    torch::Tensor y_tensor,         // [M, N] float32 output
    torch::Tensor bias_tensor,
    int M, int N, int K,
    int num_threads
) {
    const float* x = x_tensor.data_ptr<float>();
    const int8_t* w_int8 = w_int8_tensor.data_ptr<int8_t>();
    float* y = y_tensor.data_ptr<float>();
    const float* bias = bias_tensor.defined() ? bias_tensor.data_ptr<float>() : nullptr;

    // Convert int8 ternary weights to float32
    // This is a small overhead compared to the matmul itself
    std::vector<float> w_float(N * K);

    #pragma omp parallel for
    for (int i = 0; i < N * K; i++) {
        w_float[i] = static_cast<float>(w_int8[i]);
    }

    // Use Apple's optimized BLAS (uses AMX internally!)
    // Computes: Y = X * W^T (since W is [N, K] and we want [M, N] output)
    cblas_sgemm(CblasRowMajor, CblasNoTrans, CblasTrans,
                M, N, K,
                1.0f,                    // alpha
                x, K,                    // X [M, K]
                w_float.data(), K,       // W [N, K], transposed
                0.0f,                    // beta
                y, N);                   // Y [M, N]

    // Add bias if present
    if (bias != nullptr) {
        #pragma omp parallel for
        for (int m = 0; m < M; m++) {
            for (int n = 0; n < N; n++) {
                y[m * N + n] += bias[n];
            }
        }
    }
}

/**
 * Apple Accelerate with pre-converted float32 weights
 *
 * For repeated inference, convert weights once and reuse.
 * This avoids the int8->float32 conversion overhead.
 */
void matmul_free_accelerate_f32(
    torch::Tensor x_tensor,         // [M, K] float32 activations
    torch::Tensor w_f32_tensor,     // [N, K] float32 weights (pre-converted)
    torch::Tensor y_tensor,         // [M, N] float32 output
    torch::Tensor bias_tensor,
    int M, int N, int K,
    int num_threads
) {
    const float* x = x_tensor.data_ptr<float>();
    const float* w = w_f32_tensor.data_ptr<float>();
    float* y = y_tensor.data_ptr<float>();
    const float* bias = bias_tensor.defined() ? bias_tensor.data_ptr<float>() : nullptr;

    // Direct BLAS call - maximum performance!
    cblas_sgemm(CblasRowMajor, CblasNoTrans, CblasTrans,
                M, N, K,
                1.0f, x, K,
                w, K,
                0.0f, y, N);

    // Add bias
    if (bias != nullptr) {
        #pragma omp parallel for
        for (int m = 0; m < M; m++) {
            for (int n = 0; n < N; n++) {
                y[m * N + n] += bias[n];
            }
        }
    }
}

/**
 * Convert int8 ternary weights to float32 for Accelerate
 */
void convert_weights_to_f32(
    torch::Tensor w_int8_tensor,    // [N, K] int8
    torch::Tensor w_f32_tensor,     // [N, K] float32 output
    int N, int K
) {
    const int8_t* w_int8 = w_int8_tensor.data_ptr<int8_t>();
    float* w_f32 = w_f32_tensor.data_ptr<float>();

    #pragma omp parallel for
    for (int i = 0; i < N * K; i++) {
        w_f32[i] = static_cast<float>(w_int8[i]);
    }
}

#endif // APPLE_SILICON

// Python bindings
PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("quantize_activations_int8_neon", &quantize_activations_int8_neon,
          "Quantize activations to int8 (NEON)");
    m.def("pack_weights_neon_int8", &pack_weights_neon_int8,
          "Pack ternary weights for SDOT kernel");
    m.def("matmul_free_neon_sdot_simple", &matmul_free_neon_sdot_simple,
          "SDOT-based ternary matmul (simple)");
    m.def("matmul_free_neon_sdot_v2", &matmul_free_neon_sdot_v2,
          "SDOT v2: 4-way register blocking");
    m.def("matmul_free_neon_sdot_v3", &matmul_free_neon_sdot_v3,
          "SDOT v3: K-loop unrolling");
    m.def("matmul_free_neon_sdot_v4", &matmul_free_neon_sdot_v4,
          "SDOT v4: 8-way register blocking");

#ifdef APPLE_SILICON
    m.def("matmul_free_accelerate", &matmul_free_accelerate,
          "Apple Accelerate BLAS (converts int8 to float32)");
    m.def("matmul_free_accelerate_f32", &matmul_free_accelerate_f32,
          "Apple Accelerate BLAS (pre-converted float32 weights)");
    m.def("convert_weights_to_f32", &convert_weights_to_f32,
          "Convert int8 ternary weights to float32");
#endif
}
