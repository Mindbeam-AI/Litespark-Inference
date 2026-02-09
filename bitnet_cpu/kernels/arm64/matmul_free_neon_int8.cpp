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
#include <vector>
#ifndef DISABLE_OPENMP
#include <omp.h>
#endif
#include <torch/extension.h>

// Apple Silicon: Enable Accelerate framework
#ifdef __APPLE__
#include <Accelerate/Accelerate.h>
#define APPLE_SILICON 1
#endif

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

/**
 * SDOT v4 Fused - Float input, fused quantize + matmul
 *
 * This eliminates Python quantization overhead by doing everything in C++.
 * Takes float activations directly and outputs float results.
 *
 * Key optimization: Quantization is done per-row inline, avoiding
 * a separate Python call and intermediate tensor allocation.
 */
void matmul_free_neon_sdot_v4_fused(
    torch::Tensor x_float_tensor,   // [M, K] float32 input (NOT pre-quantized!)
    torch::Tensor w_int8_tensor,    // [N, K_padded] int8
    torch::Tensor w_sum_tensor,     // [N] int32 (unused)
    torch::Tensor y_tensor,         // [M, N] float32 output
    torch::Tensor bias_tensor,
    float weight_scale,             // Weight scale for dequantization
    int M, int N, int K,
    int num_threads
) {
    const float* __restrict__ x_float = x_float_tensor.data_ptr<float>();
    const int8_t* __restrict__ w_int8 = w_int8_tensor.data_ptr<int8_t>();
    float* __restrict__ y = y_tensor.data_ptr<float>();
    const float* bias = bias_tensor.defined() ? bias_tensor.data_ptr<float>() : nullptr;

    const int K_padded = ((K + 15) / 16) * 16;

#ifndef DISABLE_OPENMP
    omp_set_num_threads(num_threads);
#endif

    // Allocate thread-local quantized activation buffers
    #pragma omp parallel
    {
        // Thread-local buffer for quantized activations (reused per row)
        // Use dynamic allocation with 16-byte alignment for NEON
        std::vector<int8_t> x_q_storage(K_padded + 15);
        int8_t* x_q_local = reinterpret_cast<int8_t*>(
            (reinterpret_cast<uintptr_t>(x_q_storage.data()) + 15) & ~uintptr_t(15)
        );

        #pragma omp for schedule(static)
        for (int m = 0; m < M; m++) {
            const float* x_row = x_float + m * K;
            float* y_row = y + m * N;

            // === Step 1: Quantize this row to int8 ===
            // Find max absolute value using NEON
            float32x4_t max_vec = vdupq_n_f32(0.0f);
            int k = 0;
            for (; k + 3 < K; k += 4) {
                float32x4_t x_vec = vld1q_f32(x_row + k);
                float32x4_t abs_vec = vabsq_f32(x_vec);
                max_vec = vmaxq_f32(max_vec, abs_vec);
            }
            float max_abs = vmaxvq_f32(max_vec);
            for (; k < K; k++) {
                float abs_val = std::abs(x_row[k]);
                if (abs_val > max_abs) max_abs = abs_val;
            }

            float act_scale = max_abs / 127.0f;
            if (act_scale == 0.0f) act_scale = 1.0f;
            float inv_scale = 1.0f / act_scale;

            // Quantize to int8
            float32x4_t inv_scale_vec = vdupq_n_f32(inv_scale);
            float32x4_t min_val = vdupq_n_f32(-127.0f);
            float32x4_t max_val = vdupq_n_f32(127.0f);

            k = 0;
            for (; k + 3 < K; k += 4) {
                float32x4_t x_vec = vld1q_f32(x_row + k);
                float32x4_t scaled = vmulq_f32(x_vec, inv_scale_vec);
                scaled = vmaxq_f32(min_val, vminq_f32(max_val, scaled));
                int32x4_t rounded = vcvtnq_s32_f32(scaled);
                int16x4_t narrow16 = vmovn_s32(rounded);
                int8x8_t narrow8 = vmovn_s16(vcombine_s16(narrow16, narrow16));
                x_q_local[k + 0] = vget_lane_s8(narrow8, 0);
                x_q_local[k + 1] = vget_lane_s8(narrow8, 1);
                x_q_local[k + 2] = vget_lane_s8(narrow8, 2);
                x_q_local[k + 3] = vget_lane_s8(narrow8, 3);
            }
            for (; k < K; k++) {
                float val = x_row[k] * inv_scale;
                val = std::max(-127.0f, std::min(127.0f, val));
                x_q_local[k] = static_cast<int8_t>(std::round(val));
            }
            // Zero-pad
            for (k = K; k < K_padded; k++) {
                x_q_local[k] = 0;
            }

            // Combined scale: activation_scale * weight_scale
            float combined_scale = act_scale * weight_scale;

            // === Step 2: Compute matmul using SDOT ===
            // Process 8 outputs at a time (v4 style)
            int n = 0;
            for (; n + 7 < N; n += 8) {
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

                k = 0;
                for (; k + 15 < K; k += 16) {
                    int8x16_t x_vec = vld1q_s8(x_q_local + k);
                    acc0 = vdotq_s32(acc0, x_vec, vld1q_s8(w0 + k));
                    acc1 = vdotq_s32(acc1, x_vec, vld1q_s8(w1 + k));
                    acc2 = vdotq_s32(acc2, x_vec, vld1q_s8(w2 + k));
                    acc3 = vdotq_s32(acc3, x_vec, vld1q_s8(w3 + k));
                    acc4 = vdotq_s32(acc4, x_vec, vld1q_s8(w4 + k));
                    acc5 = vdotq_s32(acc5, x_vec, vld1q_s8(w5 + k));
                    acc6 = vdotq_s32(acc6, x_vec, vld1q_s8(w6 + k));
                    acc7 = vdotq_s32(acc7, x_vec, vld1q_s8(w7 + k));
                }

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
                    int8_t x_val = x_q_local[k];
                    sum0 += static_cast<int32_t>(x_val) * static_cast<int32_t>(w0[k]);
                    sum1 += static_cast<int32_t>(x_val) * static_cast<int32_t>(w1[k]);
                    sum2 += static_cast<int32_t>(x_val) * static_cast<int32_t>(w2[k]);
                    sum3 += static_cast<int32_t>(x_val) * static_cast<int32_t>(w3[k]);
                    sum4 += static_cast<int32_t>(x_val) * static_cast<int32_t>(w4[k]);
                    sum5 += static_cast<int32_t>(x_val) * static_cast<int32_t>(w5[k]);
                    sum6 += static_cast<int32_t>(x_val) * static_cast<int32_t>(w6[k]);
                    sum7 += static_cast<int32_t>(x_val) * static_cast<int32_t>(w7[k]);
                }

                y_row[n + 0] = static_cast<float>(sum0) * combined_scale + (bias ? bias[n + 0] : 0.0f);
                y_row[n + 1] = static_cast<float>(sum1) * combined_scale + (bias ? bias[n + 1] : 0.0f);
                y_row[n + 2] = static_cast<float>(sum2) * combined_scale + (bias ? bias[n + 2] : 0.0f);
                y_row[n + 3] = static_cast<float>(sum3) * combined_scale + (bias ? bias[n + 3] : 0.0f);
                y_row[n + 4] = static_cast<float>(sum4) * combined_scale + (bias ? bias[n + 4] : 0.0f);
                y_row[n + 5] = static_cast<float>(sum5) * combined_scale + (bias ? bias[n + 5] : 0.0f);
                y_row[n + 6] = static_cast<float>(sum6) * combined_scale + (bias ? bias[n + 6] : 0.0f);
                y_row[n + 7] = static_cast<float>(sum7) * combined_scale + (bias ? bias[n + 7] : 0.0f);
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

                k = 0;
                for (; k + 15 < K; k += 16) {
                    int8x16_t x_vec = vld1q_s8(x_q_local + k);
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
                    int8_t x_val = x_q_local[k];
                    sum0 += static_cast<int32_t>(x_val) * static_cast<int32_t>(w0[k]);
                    sum1 += static_cast<int32_t>(x_val) * static_cast<int32_t>(w1[k]);
                    sum2 += static_cast<int32_t>(x_val) * static_cast<int32_t>(w2[k]);
                    sum3 += static_cast<int32_t>(x_val) * static_cast<int32_t>(w3[k]);
                }

                y_row[n + 0] = static_cast<float>(sum0) * combined_scale + (bias ? bias[n + 0] : 0.0f);
                y_row[n + 1] = static_cast<float>(sum1) * combined_scale + (bias ? bias[n + 1] : 0.0f);
                y_row[n + 2] = static_cast<float>(sum2) * combined_scale + (bias ? bias[n + 2] : 0.0f);
                y_row[n + 3] = static_cast<float>(sum3) * combined_scale + (bias ? bias[n + 3] : 0.0f);
            }

            // Scalar remainder
            for (; n < N; n++) {
                const int8_t* w_row = w_int8 + n * K_padded;
                int32x4_t acc = vdupq_n_s32(0);

                k = 0;
                for (; k + 15 < K; k += 16) {
                    int8x16_t x_vec = vld1q_s8(x_q_local + k);
                    acc = vdotq_s32(acc, x_vec, vld1q_s8(w_row + k));
                }

                int32_t sum = vaddvq_s32(acc);
                for (; k < K; k++) {
                    sum += static_cast<int32_t>(x_q_local[k]) * static_cast<int32_t>(w_row[k]);
                }

                y_row[n] = static_cast<float>(sum) * combined_scale + (bias ? bias[n] : 0.0f);
            }
        }
    }
}

/**
 * SDOT v4 Fused - Weight Parallel version for M=1 (token generation)
 *
 * When M=1, parallelizing over M gives no benefit. Instead, we parallelize
 * over N (output channels). This is critical for efficient token generation.
 *
 * Key optimizations:
 * 1. Quantize the single input row ONCE (not per-thread)
 * 2. Parallelize the N loop across threads
 * 3. Each thread computes a chunk of output channels
 */
void matmul_free_neon_sdot_v4_fused_wp(
    torch::Tensor x_float_tensor,   // [M, K] float32 input
    torch::Tensor w_int8_tensor,    // [N, K_padded] int8
    torch::Tensor w_sum_tensor,     // [N] int32 (unused)
    torch::Tensor y_tensor,         // [M, N] float32 output
    torch::Tensor bias_tensor,
    float weight_scale,
    int M, int N, int K,
    int num_threads
) {
    const float* __restrict__ x_float = x_float_tensor.data_ptr<float>();
    const int8_t* __restrict__ w_int8 = w_int8_tensor.data_ptr<int8_t>();
    float* __restrict__ y = y_tensor.data_ptr<float>();
    const float* bias = bias_tensor.defined() ? bias_tensor.data_ptr<float>() : nullptr;

    const int K_padded = ((K + 15) / 16) * 16;

#ifndef DISABLE_OPENMP
    omp_set_num_threads(num_threads);
#endif

    // For M=1 (or small M), use Weight Parallel strategy
    if (M <= 4) {
        // Allocate quantized activation buffer (shared across threads)
        std::vector<int8_t> x_q_storage(M * K_padded + 15);
        int8_t* x_q = reinterpret_cast<int8_t*>(
            (reinterpret_cast<uintptr_t>(x_q_storage.data()) + 15) & ~uintptr_t(15)
        );

        // Store scales for each row
        std::vector<float> act_scales(M);

        // Step 1: Quantize all M rows (sequential, M is small)
        for (int m = 0; m < M; m++) {
            const float* x_row = x_float + m * K;
            int8_t* x_q_row = x_q + m * K_padded;

            // Find max absolute value
            float32x4_t max_vec = vdupq_n_f32(0.0f);
            int k = 0;
            for (; k + 3 < K; k += 4) {
                float32x4_t x_vec = vld1q_f32(x_row + k);
                float32x4_t abs_vec = vabsq_f32(x_vec);
                max_vec = vmaxq_f32(max_vec, abs_vec);
            }
            float max_abs = vmaxvq_f32(max_vec);
            for (; k < K; k++) {
                float abs_val = std::abs(x_row[k]);
                if (abs_val > max_abs) max_abs = abs_val;
            }

            float act_scale = max_abs / 127.0f;
            if (act_scale == 0.0f) act_scale = 1.0f;
            act_scales[m] = act_scale;

            float inv_scale = 1.0f / act_scale;
            float32x4_t inv_scale_vec = vdupq_n_f32(inv_scale);
            float32x4_t min_val = vdupq_n_f32(-127.0f);
            float32x4_t max_val_f = vdupq_n_f32(127.0f);

            k = 0;
            for (; k + 3 < K; k += 4) {
                float32x4_t x_vec = vld1q_f32(x_row + k);
                float32x4_t scaled = vmulq_f32(x_vec, inv_scale_vec);
                scaled = vmaxq_f32(min_val, vminq_f32(max_val_f, scaled));
                int32x4_t rounded = vcvtnq_s32_f32(scaled);
                int16x4_t narrow16 = vmovn_s32(rounded);
                int8x8_t narrow8 = vmovn_s16(vcombine_s16(narrow16, narrow16));
                x_q_row[k + 0] = vget_lane_s8(narrow8, 0);
                x_q_row[k + 1] = vget_lane_s8(narrow8, 1);
                x_q_row[k + 2] = vget_lane_s8(narrow8, 2);
                x_q_row[k + 3] = vget_lane_s8(narrow8, 3);
            }
            for (; k < K; k++) {
                float val = x_row[k] * inv_scale;
                val = std::max(-127.0f, std::min(127.0f, val));
                x_q_row[k] = static_cast<int8_t>(std::round(val));
            }
            for (k = K; k < K_padded; k++) {
                x_q_row[k] = 0;
            }
        }

        // Step 2: Parallel over N (Weight Parallel)
        constexpr int N_BLOCK = 8;

        #pragma omp parallel for schedule(static)
        for (int n_block = 0; n_block < N; n_block += N_BLOCK) {
            const int n_end = std::min(n_block + N_BLOCK, N);

            for (int m = 0; m < M; m++) {
                const int8_t* x_q_row = x_q + m * K_padded;
                float* y_row = y + m * N;
                float combined_scale = act_scales[m] * weight_scale;

                // Process outputs in this block
                for (int n = n_block; n < n_end; n++) {
                    const int8_t* w_row = w_int8 + n * K_padded;

                    int32x4_t acc = vdupq_n_s32(0);
                    int k = 0;
                    for (; k + 15 < K; k += 16) {
                        int8x16_t x_vec = vld1q_s8(x_q_row + k);
                        int8x16_t w_vec = vld1q_s8(w_row + k);
                        acc = vdotq_s32(acc, x_vec, w_vec);
                    }

                    int32_t sum = vaddvq_s32(acc);
                    for (; k < K; k++) {
                        sum += static_cast<int32_t>(x_q_row[k]) * static_cast<int32_t>(w_row[k]);
                    }

                    y_row[n] = static_cast<float>(sum) * combined_scale + (bias ? bias[n] : 0.0f);
                }
            }
        }
    } else {
        // For larger M, use the regular Activation Parallel (parallelize over M)
        #pragma omp parallel
        {
            std::vector<int8_t> x_q_storage(K_padded + 15);
            int8_t* x_q_local = reinterpret_cast<int8_t*>(
                (reinterpret_cast<uintptr_t>(x_q_storage.data()) + 15) & ~uintptr_t(15)
            );

            #pragma omp for schedule(static)
            for (int m = 0; m < M; m++) {
                const float* x_row = x_float + m * K;
                float* y_row = y + m * N;

                // Quantize
                float32x4_t max_vec = vdupq_n_f32(0.0f);
                int k = 0;
                for (; k + 3 < K; k += 4) {
                    float32x4_t x_vec = vld1q_f32(x_row + k);
                    max_vec = vmaxq_f32(max_vec, vabsq_f32(x_vec));
                }
                float max_abs = vmaxvq_f32(max_vec);
                for (; k < K; k++) {
                    float abs_val = std::abs(x_row[k]);
                    if (abs_val > max_abs) max_abs = abs_val;
                }

                float act_scale = max_abs / 127.0f;
                if (act_scale == 0.0f) act_scale = 1.0f;
                float inv_scale = 1.0f / act_scale;

                float32x4_t inv_scale_vec = vdupq_n_f32(inv_scale);
                float32x4_t min_val = vdupq_n_f32(-127.0f);
                float32x4_t max_val_f = vdupq_n_f32(127.0f);

                k = 0;
                for (; k + 3 < K; k += 4) {
                    float32x4_t x_vec = vld1q_f32(x_row + k);
                    float32x4_t scaled = vmulq_f32(x_vec, inv_scale_vec);
                    scaled = vmaxq_f32(min_val, vminq_f32(max_val_f, scaled));
                    int32x4_t rounded = vcvtnq_s32_f32(scaled);
                    int16x4_t narrow16 = vmovn_s32(rounded);
                    int8x8_t narrow8 = vmovn_s16(vcombine_s16(narrow16, narrow16));
                    x_q_local[k + 0] = vget_lane_s8(narrow8, 0);
                    x_q_local[k + 1] = vget_lane_s8(narrow8, 1);
                    x_q_local[k + 2] = vget_lane_s8(narrow8, 2);
                    x_q_local[k + 3] = vget_lane_s8(narrow8, 3);
                }
                for (; k < K; k++) {
                    float val = x_row[k] * inv_scale;
                    val = std::max(-127.0f, std::min(127.0f, val));
                    x_q_local[k] = static_cast<int8_t>(std::round(val));
                }
                for (k = K; k < K_padded; k++) {
                    x_q_local[k] = 0;
                }

                float combined_scale = act_scale * weight_scale;

                // Compute matmul
                int n = 0;
                for (; n + 7 < N; n += 8) {
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

                    k = 0;
                    for (; k + 15 < K; k += 16) {
                        int8x16_t x_vec = vld1q_s8(x_q_local + k);
                        acc0 = vdotq_s32(acc0, x_vec, vld1q_s8(w0 + k));
                        acc1 = vdotq_s32(acc1, x_vec, vld1q_s8(w1 + k));
                        acc2 = vdotq_s32(acc2, x_vec, vld1q_s8(w2 + k));
                        acc3 = vdotq_s32(acc3, x_vec, vld1q_s8(w3 + k));
                        acc4 = vdotq_s32(acc4, x_vec, vld1q_s8(w4 + k));
                        acc5 = vdotq_s32(acc5, x_vec, vld1q_s8(w5 + k));
                        acc6 = vdotq_s32(acc6, x_vec, vld1q_s8(w6 + k));
                        acc7 = vdotq_s32(acc7, x_vec, vld1q_s8(w7 + k));
                    }

                    int32_t sum0 = vaddvq_s32(acc0);
                    int32_t sum1 = vaddvq_s32(acc1);
                    int32_t sum2 = vaddvq_s32(acc2);
                    int32_t sum3 = vaddvq_s32(acc3);
                    int32_t sum4 = vaddvq_s32(acc4);
                    int32_t sum5 = vaddvq_s32(acc5);
                    int32_t sum6 = vaddvq_s32(acc6);
                    int32_t sum7 = vaddvq_s32(acc7);

                    for (; k < K; k++) {
                        int8_t x_val = x_q_local[k];
                        sum0 += static_cast<int32_t>(x_val) * static_cast<int32_t>(w0[k]);
                        sum1 += static_cast<int32_t>(x_val) * static_cast<int32_t>(w1[k]);
                        sum2 += static_cast<int32_t>(x_val) * static_cast<int32_t>(w2[k]);
                        sum3 += static_cast<int32_t>(x_val) * static_cast<int32_t>(w3[k]);
                        sum4 += static_cast<int32_t>(x_val) * static_cast<int32_t>(w4[k]);
                        sum5 += static_cast<int32_t>(x_val) * static_cast<int32_t>(w5[k]);
                        sum6 += static_cast<int32_t>(x_val) * static_cast<int32_t>(w6[k]);
                        sum7 += static_cast<int32_t>(x_val) * static_cast<int32_t>(w7[k]);
                    }

                    y_row[n + 0] = static_cast<float>(sum0) * combined_scale + (bias ? bias[n + 0] : 0.0f);
                    y_row[n + 1] = static_cast<float>(sum1) * combined_scale + (bias ? bias[n + 1] : 0.0f);
                    y_row[n + 2] = static_cast<float>(sum2) * combined_scale + (bias ? bias[n + 2] : 0.0f);
                    y_row[n + 3] = static_cast<float>(sum3) * combined_scale + (bias ? bias[n + 3] : 0.0f);
                    y_row[n + 4] = static_cast<float>(sum4) * combined_scale + (bias ? bias[n + 4] : 0.0f);
                    y_row[n + 5] = static_cast<float>(sum5) * combined_scale + (bias ? bias[n + 5] : 0.0f);
                    y_row[n + 6] = static_cast<float>(sum6) * combined_scale + (bias ? bias[n + 6] : 0.0f);
                    y_row[n + 7] = static_cast<float>(sum7) * combined_scale + (bias ? bias[n + 7] : 0.0f);
                }

                for (; n < N; n++) {
                    const int8_t* w_row = w_int8 + n * K_padded;
                    int32x4_t acc = vdupq_n_s32(0);

                    k = 0;
                    for (; k + 15 < K; k += 16) {
                        int8x16_t x_vec = vld1q_s8(x_q_local + k);
                        acc = vdotq_s32(acc, x_vec, vld1q_s8(w_row + k));
                    }

                    int32_t sum = vaddvq_s32(acc);
                    for (; k < K; k++) {
                        sum += static_cast<int32_t>(x_q_local[k]) * static_cast<int32_t>(w_row[k]);
                    }

                    y_row[n] = static_cast<float>(sum) * combined_scale + (bias ? bias[n] : 0.0f);
                }
            }
        }
    }
}

#ifdef APPLE_SILICON

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

// ============================================================================
// Fused Attention Kernel for Token Generation
// ============================================================================
// Fuses: RoPE + GQA expand + Q@K^T + scale + softmax + @V
// Optimized for single-token generation (seq_len=1)

/**
 * Apply RoPE to a single vector in-place
 * x: [head_dim] tensor
 * cos, sin: [head_dim] precomputed RoPE values for this position
 */
inline void apply_rope_single_neon(
    float* x,
    const float* cos_ptr,
    const float* sin_ptr,
    int head_dim
) {
    const int half_dim = head_dim / 2;

    // Process in chunks of 4 (NEON vector width)
    int i = 0;
    for (; i + 3 < half_dim; i += 4) {
        // Load x1 (first half) and x2 (second half)
        float32x4_t x1 = vld1q_f32(x + i);
        float32x4_t x2 = vld1q_f32(x + half_dim + i);

        // Load cos and sin
        float32x4_t cos_v = vld1q_f32(cos_ptr + i);
        float32x4_t sin_v = vld1q_f32(sin_ptr + i);

        // RoPE: [x1, x2] -> [x1*cos - x2*sin, x2*cos + x1*sin]
        // But the actual formula is: [-x2, x1] rotated
        // out[i] = x[i] * cos[i] - x[i + half] * sin[i]  (for i < half)
        // out[i] = x[i - half] * sin[i - half] + x[i] * cos[i - half]  (for i >= half)

        // Simpler: treat as complex rotation
        // For first half: out = x1*cos - x2*sin
        float32x4_t out1 = vmlsq_f32(vmulq_f32(x1, cos_v), x2, sin_v);
        // For second half: out = x2*cos + x1*sin
        float32x4_t out2 = vmlaq_f32(vmulq_f32(x2, cos_v), x1, sin_v);

        vst1q_f32(x + i, out1);
        vst1q_f32(x + half_dim + i, out2);
    }

    // Handle remainder
    for (; i < half_dim; i++) {
        float x1 = x[i];
        float x2 = x[half_dim + i];
        float c = cos_ptr[i];
        float s = sin_ptr[i];
        x[i] = x1 * c - x2 * s;
        x[half_dim + i] = x2 * c + x1 * s;
    }
}

/**
 * Fused attention for single-token generation
 *
 * Inputs:
 *   q: [batch, num_heads, 1, head_dim] - query (single token)
 *   k: [batch, num_kv_heads, seq_len, head_dim] - key cache (already has RoPE applied to past)
 *   v: [batch, num_kv_heads, seq_len, head_dim] - value cache
 *   cos: [head_dim] - RoPE cos for current position
 *   sin: [head_dim] - RoPE sin for current position
 *
 * Output:
 *   out: [batch, num_heads, 1, head_dim]
 *
 * Note: This kernel applies RoPE to q and k_new (the new K being added)
 * The KV cache should have RoPE already applied to past tokens.
 */
void fused_attention_neon(
    torch::Tensor q_tensor,      // [batch, num_heads, 1, head_dim]
    torch::Tensor k_tensor,      // [batch, num_kv_heads, kv_len, head_dim]
    torch::Tensor v_tensor,      // [batch, num_kv_heads, kv_len, head_dim]
    torch::Tensor out_tensor,    // [batch, num_heads, 1, head_dim]
    torch::Tensor cos_tensor,    // [head_dim] - for current position
    torch::Tensor sin_tensor,    // [head_dim] - for current position
    float scale,                 // 1/sqrt(head_dim)
    int num_threads
) {
    const int batch = q_tensor.size(0);
    const int num_heads = q_tensor.size(1);
    const int num_kv_heads = k_tensor.size(1);
    const int kv_len = k_tensor.size(2);
    const int head_dim = q_tensor.size(3);
    const int num_kv_groups = num_heads / num_kv_heads;

    float* q = q_tensor.data_ptr<float>();
    float* k = k_tensor.data_ptr<float>();
    float* v = v_tensor.data_ptr<float>();
    float* out = out_tensor.data_ptr<float>();
    const float* cos_ptr = cos_tensor.data_ptr<float>();
    const float* sin_ptr = sin_tensor.data_ptr<float>();

    #ifndef DISABLE_OPENMP
    omp_set_num_threads(num_threads);
    #endif

    // Process each batch and head
    #pragma omp parallel for collapse(2)
    for (int b = 0; b < batch; b++) {
        for (int h = 0; h < num_heads; h++) {
            // Get pointers for this head
            float* q_head = q + b * num_heads * head_dim + h * head_dim;
            float* out_head = out + b * num_heads * head_dim + h * head_dim;

            // Determine which KV head this Q head uses (GQA)
            int kv_h = h / num_kv_groups;
            const float* k_head = k + b * num_kv_heads * kv_len * head_dim + kv_h * kv_len * head_dim;
            const float* v_head = v + b * num_kv_heads * kv_len * head_dim + kv_h * kv_len * head_dim;

            // Apply RoPE to Q (in-place for this head)
            // Note: We make a copy to avoid modifying the original
            float q_rope[256];  // Assume head_dim <= 256
            memcpy(q_rope, q_head, head_dim * sizeof(float));
            apply_rope_single_neon(q_rope, cos_ptr, sin_ptr, head_dim);

            // Compute attention scores: Q @ K^T
            // scores[i] = dot(q_rope, k[i]) * scale
            float scores[4096];  // Assume kv_len <= 4096
            float max_score = -1e9f;

            for (int i = 0; i < kv_len; i++) {
                const float* k_vec = k_head + i * head_dim;

                // Dot product using NEON
                float32x4_t sum_vec = vdupq_n_f32(0.0f);
                int d = 0;
                for (; d + 3 < head_dim; d += 4) {
                    float32x4_t q_v = vld1q_f32(q_rope + d);
                    float32x4_t k_v = vld1q_f32(k_vec + d);
                    sum_vec = vmlaq_f32(sum_vec, q_v, k_v);
                }
                float dot = vaddvq_f32(sum_vec);
                for (; d < head_dim; d++) {
                    dot += q_rope[d] * k_vec[d];
                }

                scores[i] = dot * scale;
                if (scores[i] > max_score) max_score = scores[i];
            }

            // Softmax: exp(scores - max) / sum(exp(scores - max))
            float sum_exp = 0.0f;
            for (int i = 0; i < kv_len; i++) {
                scores[i] = expf(scores[i] - max_score);
                sum_exp += scores[i];
            }
            float inv_sum = 1.0f / sum_exp;
            for (int i = 0; i < kv_len; i++) {
                scores[i] *= inv_sum;
            }

            // Compute output: attention @ V
            // out = sum(scores[i] * v[i])
            memset(out_head, 0, head_dim * sizeof(float));

            for (int i = 0; i < kv_len; i++) {
                const float* v_vec = v_head + i * head_dim;
                float score = scores[i];

                // Accumulate: out += score * v
                float32x4_t score_vec = vdupq_n_f32(score);
                int d = 0;
                for (; d + 3 < head_dim; d += 4) {
                    float32x4_t out_v = vld1q_f32(out_head + d);
                    float32x4_t v_v = vld1q_f32(v_vec + d);
                    out_v = vmlaq_f32(out_v, score_vec, v_v);
                    vst1q_f32(out_head + d, out_v);
                }
                for (; d < head_dim; d++) {
                    out_head[d] += score * v_vec[d];
                }
            }
        }
    }
}

/**
 * Optimized version for longer sequences - tiles the KV dimension
 * to improve cache efficiency for the softmax computation
 */
void fused_attention_neon_tiled(
    torch::Tensor q_tensor,      // [batch, num_heads, seq_q, head_dim]
    torch::Tensor k_tensor,      // [batch, num_kv_heads, seq_kv, head_dim]
    torch::Tensor v_tensor,      // [batch, num_kv_heads, seq_kv, head_dim]
    torch::Tensor out_tensor,    // [batch, num_heads, seq_q, head_dim]
    float scale,
    bool is_causal,
    int num_threads
) {
    // For now, just call the simple version for single-token case
    // TODO: Implement tiled version for longer sequences
    const int seq_q = q_tensor.size(2);

    if (seq_q == 1) {
        // Single token case - use simple kernel without RoPE
        // (RoPE should be applied before calling this)
        const int batch = q_tensor.size(0);
        const int num_heads = q_tensor.size(1);
        const int num_kv_heads = k_tensor.size(1);
        const int kv_len = k_tensor.size(2);
        const int head_dim = q_tensor.size(3);
        const int num_kv_groups = num_heads / num_kv_heads;

        float* q = q_tensor.data_ptr<float>();
        float* k = k_tensor.data_ptr<float>();
        float* v = v_tensor.data_ptr<float>();
        float* out = out_tensor.data_ptr<float>();

        #ifndef DISABLE_OPENMP
        omp_set_num_threads(num_threads);
        #endif

        #pragma omp parallel for collapse(2)
        for (int b = 0; b < batch; b++) {
            for (int h = 0; h < num_heads; h++) {
                float* q_head = q + b * num_heads * head_dim + h * head_dim;
                float* out_head = out + b * num_heads * head_dim + h * head_dim;

                int kv_h = h / num_kv_groups;
                const float* k_head = k + b * num_kv_heads * kv_len * head_dim + kv_h * kv_len * head_dim;
                const float* v_head = v + b * num_kv_heads * kv_len * head_dim + kv_h * kv_len * head_dim;

                // Compute attention scores
                float scores[4096];
                float max_score = -1e9f;

                for (int i = 0; i < kv_len; i++) {
                    const float* k_vec = k_head + i * head_dim;
                    float32x4_t sum_vec = vdupq_n_f32(0.0f);
                    int d = 0;
                    for (; d + 3 < head_dim; d += 4) {
                        float32x4_t q_v = vld1q_f32(q_head + d);
                        float32x4_t k_v = vld1q_f32(k_vec + d);
                        sum_vec = vmlaq_f32(sum_vec, q_v, k_v);
                    }
                    float dot = vaddvq_f32(sum_vec);
                    for (; d < head_dim; d++) {
                        dot += q_head[d] * k_vec[d];
                    }
                    scores[i] = dot * scale;
                    if (scores[i] > max_score) max_score = scores[i];
                }

                // Softmax
                float sum_exp = 0.0f;
                for (int i = 0; i < kv_len; i++) {
                    scores[i] = expf(scores[i] - max_score);
                    sum_exp += scores[i];
                }
                float inv_sum = 1.0f / sum_exp;
                for (int i = 0; i < kv_len; i++) {
                    scores[i] *= inv_sum;
                }

                // Output: attention @ V
                memset(out_head, 0, head_dim * sizeof(float));
                for (int i = 0; i < kv_len; i++) {
                    const float* v_vec = v_head + i * head_dim;
                    float score = scores[i];
                    float32x4_t score_vec = vdupq_n_f32(score);
                    int d = 0;
                    for (; d + 3 < head_dim; d += 4) {
                        float32x4_t out_v = vld1q_f32(out_head + d);
                        float32x4_t v_v = vld1q_f32(v_vec + d);
                        out_v = vmlaq_f32(out_v, score_vec, v_v);
                        vst1q_f32(out_head + d, out_v);
                    }
                    for (; d < head_dim; d++) {
                        out_head[d] += score * v_vec[d];
                    }
                }
            }
        }
    } else {
        // TODO: Multi-token case (prompt processing)
        // For now, fall back to PyTorch
        throw std::runtime_error("fused_attention_neon_tiled: seq_q > 1 not yet implemented");
    }
}

/**
 * Apply RoPE to a batch of vectors (e.g., K tensor for all heads)
 * Input: x [batch, num_heads, seq_len, head_dim]
 * cos/sin: [seq_len, head_dim] or [head_dim] for single position
 * Modifies x in-place
 */
void apply_rope_batch_neon(
    torch::Tensor x_tensor,      // [batch, num_heads, seq_len, head_dim] - modified in-place
    torch::Tensor cos_tensor,    // [head_dim] for single position
    torch::Tensor sin_tensor,    // [head_dim] for single position
    int num_threads
) {
    const int batch = x_tensor.size(0);
    const int num_heads = x_tensor.size(1);
    const int seq_len = x_tensor.size(2);
    const int head_dim = x_tensor.size(3);
    const int half_dim = head_dim / 2;

    float* x = x_tensor.data_ptr<float>();
    const float* cos_ptr = cos_tensor.data_ptr<float>();
    const float* sin_ptr = sin_tensor.data_ptr<float>();

    #ifndef DISABLE_OPENMP
    omp_set_num_threads(num_threads);
    #endif

    // Process all heads in parallel
    #pragma omp parallel for collapse(3)
    for (int b = 0; b < batch; b++) {
        for (int h = 0; h < num_heads; h++) {
            for (int s = 0; s < seq_len; s++) {
                float* x_head = x + b * num_heads * seq_len * head_dim
                              + h * seq_len * head_dim
                              + s * head_dim;

                // Apply RoPE using NEON
                int i = 0;
                for (; i + 3 < half_dim; i += 4) {
                    float32x4_t x1 = vld1q_f32(x_head + i);
                    float32x4_t x2 = vld1q_f32(x_head + half_dim + i);
                    float32x4_t cos_v = vld1q_f32(cos_ptr + i);
                    float32x4_t sin_v = vld1q_f32(sin_ptr + i);

                    float32x4_t out1 = vmlsq_f32(vmulq_f32(x1, cos_v), x2, sin_v);
                    float32x4_t out2 = vmlaq_f32(vmulq_f32(x2, cos_v), x1, sin_v);

                    vst1q_f32(x_head + i, out1);
                    vst1q_f32(x_head + half_dim + i, out2);
                }

                // Remainder
                for (; i < half_dim; i++) {
                    float x1 = x_head[i];
                    float x2 = x_head[half_dim + i];
                    x_head[i] = x1 * cos_ptr[i] - x2 * sin_ptr[i];
                    x_head[half_dim + i] = x2 * cos_ptr[i] + x1 * sin_ptr[i];
                }
            }
        }
    }
}

/**
 * LM Head matmul optimized for single-token generation (M=1)
 * Computes: output = input @ weight.T where weight is (vocab_size, hidden_size)
 * Uses fp32 for ARM NEON (bf16 not widely supported on ARM)
 */
void lm_head_neon(
    torch::Tensor input_tensor,   // [batch, hidden_size] in fp32
    torch::Tensor weight_tensor,  // [vocab_size, hidden_size] in fp32
    torch::Tensor output_tensor,  // [batch, vocab_size] in fp32
    int num_threads
) {
    const int batch = input_tensor.dim() == 3 ? input_tensor.size(0) : input_tensor.size(0);
    const int hidden_size = input_tensor.dim() == 3 ? input_tensor.size(2) : input_tensor.size(1);
    const int vocab_size = weight_tensor.size(0);

    float* input = input_tensor.data_ptr<float>();
    float* weight = weight_tensor.data_ptr<float>();
    float* output = output_tensor.data_ptr<float>();

    omp_set_num_threads(num_threads);

    for (int b = 0; b < batch; b++) {
        const float* x = input + b * hidden_size;
        float* y = output + b * vocab_size;

        // Parallelize over vocab_size (output dimension)
        constexpr int N_BLOCK = 64;
        const int n_blocks = (vocab_size + N_BLOCK - 1) / N_BLOCK;

        #pragma omp parallel for schedule(dynamic, 4)
        for (int nb = 0; nb < n_blocks; nb++) {
            const int n_start = nb * N_BLOCK;
            const int n_end = std::min(n_start + N_BLOCK, vocab_size);

            for (int n = n_start; n < n_end; n++) {
                const float* w_row = weight + n * hidden_size;

                // Dot product using NEON
                float32x4_t sum_vec = vdupq_n_f32(0.0f);
                int k = 0;

                // Process 16 elements at a time (4 vectors of 4)
                for (; k + 15 < hidden_size; k += 16) {
                    float32x4_t x0 = vld1q_f32(x + k);
                    float32x4_t x1 = vld1q_f32(x + k + 4);
                    float32x4_t x2 = vld1q_f32(x + k + 8);
                    float32x4_t x3 = vld1q_f32(x + k + 12);

                    float32x4_t w0 = vld1q_f32(w_row + k);
                    float32x4_t w1 = vld1q_f32(w_row + k + 4);
                    float32x4_t w2 = vld1q_f32(w_row + k + 8);
                    float32x4_t w3 = vld1q_f32(w_row + k + 12);

                    sum_vec = vfmaq_f32(sum_vec, x0, w0);
                    sum_vec = vfmaq_f32(sum_vec, x1, w1);
                    sum_vec = vfmaq_f32(sum_vec, x2, w2);
                    sum_vec = vfmaq_f32(sum_vec, x3, w3);
                }

                // Process 4 elements at a time
                for (; k + 3 < hidden_size; k += 4) {
                    float32x4_t x_v = vld1q_f32(x + k);
                    float32x4_t w_v = vld1q_f32(w_row + k);
                    sum_vec = vfmaq_f32(sum_vec, x_v, w_v);
                }

                // Reduce
                float sum = vaddvq_f32(sum_vec);

                // Remainder
                for (; k < hidden_size; k++) {
                    sum += x[k] * w_row[k];
                }

                y[n] = sum;
            }
        }
    }
}

// ============================================================================
// C++ Transformer Forward Pass (Eliminates Python Overhead)
// ============================================================================

/**
 * RMSNorm: x = x * weight / sqrt(mean(x^2) + eps)
 * In-place operation on x
 */
inline void rms_norm_neon(
    float* x,           // [hidden_size] input/output
    const float* weight, // [hidden_size]
    int hidden_size,
    float eps = 1e-5f
) {
    // Compute sum of squares
    float32x4_t sum_sq = vdupq_n_f32(0.0f);
    int i = 0;
    for (; i + 3 < hidden_size; i += 4) {
        float32x4_t v = vld1q_f32(x + i);
        sum_sq = vfmaq_f32(sum_sq, v, v);
    }
    float ss = vaddvq_f32(sum_sq);
    for (; i < hidden_size; i++) {
        ss += x[i] * x[i];
    }

    // Compute scale: 1 / sqrt(mean + eps)
    float scale = 1.0f / sqrtf(ss / hidden_size + eps);
    float32x4_t scale_vec = vdupq_n_f32(scale);

    // Apply: x = x * scale * weight
    i = 0;
    for (; i + 3 < hidden_size; i += 4) {
        float32x4_t v = vld1q_f32(x + i);
        float32x4_t w = vld1q_f32(weight + i);
        v = vmulq_f32(vmulq_f32(v, scale_vec), w);
        vst1q_f32(x + i, v);
    }
    for (; i < hidden_size; i++) {
        x[i] = x[i] * scale * weight[i];
    }
}

/**
 * ReLU squared: relu2(x) = max(0, x)^2
 */
inline void relu2_inplace_neon(float* x, int size) {
    float32x4_t zero = vdupq_n_f32(0.0f);
    int i = 0;
    for (; i + 3 < size; i += 4) {
        float32x4_t v = vld1q_f32(x + i);
        v = vmaxq_f32(v, zero);  // ReLU
        v = vmulq_f32(v, v);      // Square
        vst1q_f32(x + i, v);
    }
    for (; i < size; i++) {
        float v = x[i];
        v = v > 0.0f ? v : 0.0f;
        x[i] = v * v;
    }
}

/**
 * Element-wise multiply: out = a * b
 */
inline void elementwise_mul_neon(
    const float* a,
    const float* b,
    float* out,
    int size
) {
    int i = 0;
    for (; i + 3 < size; i += 4) {
        float32x4_t va = vld1q_f32(a + i);
        float32x4_t vb = vld1q_f32(b + i);
        vst1q_f32(out + i, vmulq_f32(va, vb));
    }
    for (; i < size; i++) {
        out[i] = a[i] * b[i];
    }
}

/**
 * Single ternary matmul for M=1: y = x @ W^T * scale
 * Optimized for single-token inference
 */
inline void ternary_matmul_single_neon(
    const float* x,           // [K] input
    const int8_t* w_int8,     // [N, K_padded] weights
    float* y,                 // [N] output
    int N, int K, int K_padded,
    float weight_scale,
    int num_threads
) {
    // First quantize input to int8
    float max_abs = 0.0f;
    for (int k = 0; k < K; k++) {
        float abs_val = fabsf(x[k]);
        if (abs_val > max_abs) max_abs = abs_val;
    }
    float act_scale = max_abs / 127.0f;
    if (act_scale == 0.0f) act_scale = 1.0f;
    float inv_scale = 1.0f / act_scale;

    // Quantize with alignment
    std::vector<int8_t> x_q_storage(K_padded + 16);
    int8_t* x_q = reinterpret_cast<int8_t*>(
        (reinterpret_cast<uintptr_t>(x_q_storage.data()) + 15) & ~uintptr_t(15)
    );

    for (int k = 0; k < K; k++) {
        float val = x[k] * inv_scale;
        val = fmaxf(-127.0f, fminf(127.0f, val));
        x_q[k] = static_cast<int8_t>(roundf(val));
    }
    for (int k = K; k < K_padded; k++) {
        x_q[k] = 0;
    }

    float combined_scale = act_scale * weight_scale;

    // Parallel matmul over N
    #pragma omp parallel for num_threads(num_threads) schedule(static)
    for (int n = 0; n < N; n++) {
        const int8_t* w_row = w_int8 + n * K_padded;

        int32x4_t acc = vdupq_n_s32(0);
        int k = 0;
        for (; k + 15 < K_padded; k += 16) {
            int8x16_t x_vec = vld1q_s8(x_q + k);
            int8x16_t w_vec = vld1q_s8(w_row + k);
            acc = vdotq_s32(acc, x_vec, w_vec);
        }

        int32_t sum = vaddvq_s32(acc);
        for (; k < K; k++) {
            sum += static_cast<int32_t>(x_q[k]) * static_cast<int32_t>(w_row[k]);
        }

        y[n] = static_cast<float>(sum) * combined_scale;
    }
}

/**
 * Full transformer forward pass for single token generation
 * Runs all layers in C++ to eliminate Python overhead
 *
 * This function implements one full forward pass through all transformer layers
 * for a single token, updating the KV cache in place.
 */
void transformer_forward_single_neon(
    torch::Tensor hidden_tensor,      // [hidden_size] current hidden state (modified in-place)
    // Attention weights (per layer)
    std::vector<torch::Tensor> q_w_int8,
    std::vector<torch::Tensor> q_w_sum,
    std::vector<float> q_scale,
    std::vector<torch::Tensor> k_w_int8,
    std::vector<torch::Tensor> k_w_sum,
    std::vector<float> k_scale,
    std::vector<torch::Tensor> v_w_int8,
    std::vector<torch::Tensor> v_w_sum,
    std::vector<float> v_scale,
    std::vector<torch::Tensor> o_w_int8,
    std::vector<torch::Tensor> o_w_sum,
    std::vector<float> o_scale,
    // MLP weights (per layer)
    std::vector<torch::Tensor> gate_w_int8,
    std::vector<torch::Tensor> gate_w_sum,
    std::vector<float> gate_scale,
    std::vector<torch::Tensor> up_w_int8,
    std::vector<torch::Tensor> up_w_sum,
    std::vector<float> up_scale,
    std::vector<torch::Tensor> down_w_int8,
    std::vector<torch::Tensor> down_w_sum,
    std::vector<float> down_scale,
    // Norm weights (per layer)
    std::vector<torch::Tensor> input_norm_weights,
    std::vector<torch::Tensor> post_attn_norm_weights,
    std::vector<torch::Tensor> attn_sub_norm_weights,
    std::vector<torch::Tensor> ffn_sub_norm_weights,
    torch::Tensor final_norm_weight,
    // KV cache [num_layers, num_kv_heads, max_seq_len, head_dim]
    torch::Tensor k_cache_tensor,
    torch::Tensor v_cache_tensor,
    int cache_seq_len,  // Current position in cache
    // RoPE
    torch::Tensor cos_tensor,  // [head_dim] for current position
    torch::Tensor sin_tensor,
    // Config
    int num_layers,
    int hidden_size,
    int num_heads,
    int num_kv_heads,
    int head_dim,
    int intermediate_size,
    float attn_scale,
    int num_threads
) {
    float* hidden = hidden_tensor.data_ptr<float>();
    const int num_kv_groups = num_heads / num_kv_heads;
    const int kv_dim = num_kv_heads * head_dim;
    const int K_padded = ((hidden_size + 15) / 16) * 16;
    const int K_padded_inter = ((intermediate_size + 15) / 16) * 16;

    // Pre-allocated buffers for intermediate results
    std::vector<float> residual(hidden_size);
    std::vector<float> normed(hidden_size);
    std::vector<float> q_out(hidden_size);
    std::vector<float> k_out(kv_dim);
    std::vector<float> v_out(kv_dim);
    std::vector<float> attn_out(hidden_size);
    std::vector<float> o_out(hidden_size);
    std::vector<float> gate_out(intermediate_size);
    std::vector<float> up_out(intermediate_size);
    std::vector<float> down_out(hidden_size);
    std::vector<float> mlp_hidden(intermediate_size);

    const float* cos_ptr = cos_tensor.data_ptr<float>();
    const float* sin_ptr = sin_tensor.data_ptr<float>();

    #ifndef DISABLE_OPENMP
    omp_set_num_threads(num_threads);
    #endif

    for (int layer = 0; layer < num_layers; layer++) {
        // === Save residual ===
        memcpy(residual.data(), hidden, hidden_size * sizeof(float));

        // === Input Norm ===
        memcpy(normed.data(), hidden, hidden_size * sizeof(float));
        rms_norm_neon(normed.data(), input_norm_weights[layer].data_ptr<float>(), hidden_size);

        // === Attention ===
        // Q projection
        ternary_matmul_single_neon(
            normed.data(),
            q_w_int8[layer].data_ptr<int8_t>(),
            q_out.data(),
            hidden_size, hidden_size, K_padded,
            q_scale[layer], num_threads
        );

        // K projection
        ternary_matmul_single_neon(
            normed.data(),
            k_w_int8[layer].data_ptr<int8_t>(),
            k_out.data(),
            kv_dim, hidden_size, K_padded,
            k_scale[layer], num_threads
        );

        // V projection
        ternary_matmul_single_neon(
            normed.data(),
            v_w_int8[layer].data_ptr<int8_t>(),
            v_out.data(),
            kv_dim, hidden_size, K_padded,
            v_scale[layer], num_threads
        );

        // Apply RoPE to Q and K (per head)
        for (int h = 0; h < num_heads; h++) {
            apply_rope_single_neon(q_out.data() + h * head_dim, cos_ptr, sin_ptr, head_dim);
        }
        for (int h = 0; h < num_kv_heads; h++) {
            apply_rope_single_neon(k_out.data() + h * head_dim, cos_ptr, sin_ptr, head_dim);
        }

        // Update KV cache
        float* k_cache = k_cache_tensor.data_ptr<float>() +
                         layer * num_kv_heads * k_cache_tensor.size(2) * head_dim +
                         cache_seq_len * head_dim;
        float* v_cache = v_cache_tensor.data_ptr<float>() +
                         layer * num_kv_heads * v_cache_tensor.size(2) * head_dim +
                         cache_seq_len * head_dim;

        // Copy K and V to cache (interleaved by head)
        for (int h = 0; h < num_kv_heads; h++) {
            memcpy(k_cache + h * k_cache_tensor.size(2) * head_dim,
                   k_out.data() + h * head_dim,
                   head_dim * sizeof(float));
            memcpy(v_cache + h * v_cache_tensor.size(2) * head_dim,
                   v_out.data() + h * head_dim,
                   head_dim * sizeof(float));
        }

        // Compute attention for each head
        const int kv_len = cache_seq_len + 1;
        memset(attn_out.data(), 0, hidden_size * sizeof(float));

        #pragma omp parallel for num_threads(num_threads)
        for (int h = 0; h < num_heads; h++) {
            const int kv_h = h / num_kv_groups;
            const float* q_head = q_out.data() + h * head_dim;
            float* out_head = attn_out.data() + h * head_dim;

            // Get K and V for this head from cache
            const float* k_head_base = k_cache_tensor.data_ptr<float>() +
                                       layer * num_kv_heads * k_cache_tensor.size(2) * head_dim +
                                       kv_h * k_cache_tensor.size(2) * head_dim;
            const float* v_head_base = v_cache_tensor.data_ptr<float>() +
                                       layer * num_kv_heads * v_cache_tensor.size(2) * head_dim +
                                       kv_h * v_cache_tensor.size(2) * head_dim;

            // Compute attention scores
            float scores[4096];
            float max_score = -1e9f;

            for (int i = 0; i < kv_len; i++) {
                const float* k_vec = k_head_base + i * head_dim;
                float32x4_t sum_vec = vdupq_n_f32(0.0f);
                int d = 0;
                for (; d + 3 < head_dim; d += 4) {
                    float32x4_t q_v = vld1q_f32(q_head + d);
                    float32x4_t k_v = vld1q_f32(k_vec + d);
                    sum_vec = vfmaq_f32(sum_vec, q_v, k_v);
                }
                float dot = vaddvq_f32(sum_vec);
                for (; d < head_dim; d++) {
                    dot += q_head[d] * k_vec[d];
                }
                scores[i] = dot * attn_scale;
                if (scores[i] > max_score) max_score = scores[i];
            }

            // Softmax
            float sum_exp = 0.0f;
            for (int i = 0; i < kv_len; i++) {
                scores[i] = expf(scores[i] - max_score);
                sum_exp += scores[i];
            }
            float inv_sum = 1.0f / sum_exp;
            for (int i = 0; i < kv_len; i++) {
                scores[i] *= inv_sum;
            }

            // Weighted sum of V
            for (int i = 0; i < kv_len; i++) {
                const float* v_vec = v_head_base + i * head_dim;
                float score = scores[i];
                for (int d = 0; d < head_dim; d++) {
                    out_head[d] += score * v_vec[d];
                }
            }
        }

        // Attention sub-norm
        rms_norm_neon(attn_out.data(), attn_sub_norm_weights[layer].data_ptr<float>(), hidden_size);

        // O projection
        ternary_matmul_single_neon(
            attn_out.data(),
            o_w_int8[layer].data_ptr<int8_t>(),
            o_out.data(),
            hidden_size, hidden_size, K_padded,
            o_scale[layer], num_threads
        );

        // Residual connection
        for (int i = 0; i < hidden_size; i++) {
            hidden[i] = residual[i] + o_out[i];
        }

        // === MLP ===
        // Save residual
        memcpy(residual.data(), hidden, hidden_size * sizeof(float));

        // Post-attention norm
        memcpy(normed.data(), hidden, hidden_size * sizeof(float));
        rms_norm_neon(normed.data(), post_attn_norm_weights[layer].data_ptr<float>(), hidden_size);

        // Gate projection + relu2
        ternary_matmul_single_neon(
            normed.data(),
            gate_w_int8[layer].data_ptr<int8_t>(),
            gate_out.data(),
            intermediate_size, hidden_size, K_padded,
            gate_scale[layer], num_threads
        );
        relu2_inplace_neon(gate_out.data(), intermediate_size);

        // Up projection
        ternary_matmul_single_neon(
            normed.data(),
            up_w_int8[layer].data_ptr<int8_t>(),
            up_out.data(),
            intermediate_size, hidden_size, K_padded,
            up_scale[layer], num_threads
        );

        // Gate * Up
        elementwise_mul_neon(gate_out.data(), up_out.data(), mlp_hidden.data(), intermediate_size);

        // FFN sub-norm
        rms_norm_neon(mlp_hidden.data(), ffn_sub_norm_weights[layer].data_ptr<float>(), intermediate_size);

        // Down projection
        ternary_matmul_single_neon(
            mlp_hidden.data(),
            down_w_int8[layer].data_ptr<int8_t>(),
            down_out.data(),
            hidden_size, intermediate_size, K_padded_inter,
            down_scale[layer], num_threads
        );

        // Residual connection
        for (int i = 0; i < hidden_size; i++) {
            hidden[i] = residual[i] + down_out[i];
        }
    }

    // === Final Norm ===
    rms_norm_neon(hidden, final_norm_weight.data_ptr<float>(), hidden_size);
}

// Python bindings
PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("lm_head_neon", &lm_head_neon,
          "LM head matmul using NEON");
    m.def("quantize_activations_int8_neon", &quantize_activations_int8_neon,
          "Quantize activations to int8 (NEON)");
    m.def("apply_rope_batch_neon", &apply_rope_batch_neon,
          "Apply RoPE to batch of vectors (in-place)");
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
    m.def("matmul_free_neon_sdot_v4_fused", &matmul_free_neon_sdot_v4_fused,
          "SDOT v4 Fused: Float input, fused quantize+matmul");
    m.def("matmul_free_neon_sdot_v4_fused_wp", &matmul_free_neon_sdot_v4_fused_wp,
          "SDOT v4 Fused Weight Parallel: Optimized for M=1 token generation");
    m.def("fused_attention_neon", &fused_attention_neon,
          "Fused attention with RoPE for single-token generation");
    m.def("fused_attention_neon_tiled", &fused_attention_neon_tiled,
          "Fused attention without RoPE (RoPE pre-applied)");
    m.def("transformer_forward_single_neon", &transformer_forward_single_neon,
          "Full transformer forward pass for single token (eliminates Python overhead)");

#ifdef APPLE_SILICON
    m.def("matmul_free_accelerate", &matmul_free_accelerate,
          "Apple Accelerate BLAS (converts int8 to float32)");
    m.def("matmul_free_accelerate_f32", &matmul_free_accelerate_f32,
          "Apple Accelerate BLAS (pre-converted float32 weights)");
    m.def("convert_weights_to_f32", &convert_weights_to_f32,
          "Convert int8 ternary weights to float32");
#endif
}
