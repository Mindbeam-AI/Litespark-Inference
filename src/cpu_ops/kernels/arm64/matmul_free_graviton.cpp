/*
 * AWS Graviton Optimized Ternary MatMul Kernels
 *
 * Uses ARM NEON SDOT (Dot Product Instructions) for int8 dot products.
 * Optimized for Graviton 2/3/4 cache hierarchy.
 *
 * Key differences from Apple Silicon NEON kernel:
 * - Smaller M_TILE (16 vs 32) for Graviton's 64KB L1
 * - Different N_TILE for L2 optimization
 * - More aggressive parallelization for 64+ cores
 *
 * Graviton Architecture:
 * - Graviton 2: 64 cores, NEON only, 32KB L1D, 1MB L2
 * - Graviton 3: 64 cores, NEON + SVE2 (256-bit), 64KB L1D, 1MB L2
 * - Graviton 4: 96 cores, NEON + SVE2 (256-bit), 64KB L1D, 2MB L2
 *
 * ARM SDOT: vdotq_s32 does int8 x int8 directly (no uint8 offset needed!)
 * Each vdotq_s32 processes 16 int8 pairs into 4 int32 accumulators.
 */

#include <arm_neon.h>
#include <cstring>
#include <cstdint>
#include <cmath>
#include <algorithm>
#include <omp.h>
#include <torch/extension.h>

// Graviton-optimized tile sizes
constexpr int TILE_M = 16;   // Smaller for 64KB L1D
constexpr int TILE_N = 32;   // Fits weight tile in L2
constexpr int N_BLOCK = 4;   // Register blocking: 4 outputs at once
constexpr int N_BLOCK_8 = 8; // Extended register blocking

/**
 * Fast NEON exponential approximation for softmax
 * Accurate for x in [-88, 88]
 */
inline float32x4_t neon_exp_fast(float32x4_t x) {
    // Clamp to prevent overflow
    x = vmaxq_f32(x, vdupq_n_f32(-88.0f));
    x = vminq_f32(x, vdupq_n_f32(88.0f));

    // exp(x) = 2^(x * log2(e))
    const float32x4_t log2e = vdupq_n_f32(1.44269504088896341f);
    float32x4_t z = vmulq_f32(x, log2e);

    // Split into integer and fractional parts
    int32x4_t zi = vcvtq_s32_f32(z);
    float32x4_t zf = vsubq_f32(z, vcvtq_f32_s32(zi));

    // 2^frac using polynomial
    const float32x4_t c0 = vdupq_n_f32(1.0f);
    const float32x4_t c1 = vdupq_n_f32(0.693147180559945309f);
    const float32x4_t c2 = vdupq_n_f32(0.240226506959100712f);
    const float32x4_t c3 = vdupq_n_f32(0.0555041086648215799f);

    float32x4_t poly = vfmaq_f32(c2, c3, zf);
    poly = vfmaq_f32(c1, poly, zf);
    poly = vfmaq_f32(c0, poly, zf);

    // 2^int part via bit manipulation
    int32x4_t exp_int = vaddq_s32(zi, vdupq_n_s32(127));
    exp_int = vshlq_n_s32(exp_int, 23);
    float32x4_t pow2_int = vreinterpretq_f32_s32(exp_int);

    return vmulq_f32(pow2_int, poly);
}

/**
 * Quantize activations to int8 with per-row scaling
 */
void quantize_activations_int8_graviton(
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

            // Round and convert
            int32x4_t rounded = vcvtnq_s32_f32(scaled);
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
 * Also computes sum of weights for each output (unused for ARM SDOT,
 * but kept for API compatibility with x86 VNNI)
 */
void pack_weights_graviton(
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
 * Graviton v2 - Register-blocked SDOT kernel
 *
 * Key optimizations:
 * 1. Process 4 outputs simultaneously (register blocking)
 * 2. Reuse activation vector across 4 weight vectors
 * 3. Better instruction-level parallelism
 *
 * Note: ARM SDOT uses int8 x int8 directly (no uint8 offset trick needed!)
 */
void matmul_free_graviton_v2(
    torch::Tensor x_int8_tensor,    // [M, K] int8
    torch::Tensor scale_tensor,     // [M] float32
    torch::Tensor w_int8_tensor,    // [N, K_padded] int8
    torch::Tensor w_sum_tensor,     // [N] int32 (unused for ARM)
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
 * Graviton v3 - Optimized parallel kernel
 *
 * Key optimizations:
 * 1. Single parallel region (no thread pool recreation)
 * 2. Parallelize over M (rows) - each thread handles complete rows
 * 3. 8-way output blocking with K-loop unrolling
 * 4. Prefetch hints for weight data
 *
 * For M=1 (single token inference), this falls back to N-parallel
 */
void matmul_free_graviton_v3(
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

    if (M >= num_threads) {
        // Parallelize over M (rows) - best for batch processing
        #pragma omp parallel for schedule(static)
        for (int m = 0; m < M; m++) {
            const int8_t* __restrict__ x_row = x_int8 + m * K;
            float scale = scales[m];
            float* y_row = y + m * N;

            // Process 8 outputs at a time
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

                // K loop with 2x unrolling
                int k = 0;
                for (; k + 31 < K; k += 32) {
                    int8x16_t x0 = vld1q_s8(x_row + k);
                    int8x16_t x1 = vld1q_s8(x_row + k + 16);

                    acc0 = vdotq_s32(acc0, x0, vld1q_s8(w0 + k));
                    acc0 = vdotq_s32(acc0, x1, vld1q_s8(w0 + k + 16));
                    acc1 = vdotq_s32(acc1, x0, vld1q_s8(w1 + k));
                    acc1 = vdotq_s32(acc1, x1, vld1q_s8(w1 + k + 16));
                    acc2 = vdotq_s32(acc2, x0, vld1q_s8(w2 + k));
                    acc2 = vdotq_s32(acc2, x1, vld1q_s8(w2 + k + 16));
                    acc3 = vdotq_s32(acc3, x0, vld1q_s8(w3 + k));
                    acc3 = vdotq_s32(acc3, x1, vld1q_s8(w3 + k + 16));
                    acc4 = vdotq_s32(acc4, x0, vld1q_s8(w4 + k));
                    acc4 = vdotq_s32(acc4, x1, vld1q_s8(w4 + k + 16));
                    acc5 = vdotq_s32(acc5, x0, vld1q_s8(w5 + k));
                    acc5 = vdotq_s32(acc5, x1, vld1q_s8(w5 + k + 16));
                    acc6 = vdotq_s32(acc6, x0, vld1q_s8(w6 + k));
                    acc6 = vdotq_s32(acc6, x1, vld1q_s8(w6 + k + 16));
                    acc7 = vdotq_s32(acc7, x0, vld1q_s8(w7 + k));
                    acc7 = vdotq_s32(acc7, x1, vld1q_s8(w7 + k + 16));
                }

                // Handle remaining K
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

            // Handle remaining N with 4-way blocking
            for (; n + 3 < N; n += 4) {
                int32x4_t acc0 = vdupq_n_s32(0);
                int32x4_t acc1 = vdupq_n_s32(0);
                int32x4_t acc2 = vdupq_n_s32(0);
                int32x4_t acc3 = vdupq_n_s32(0);

                const int8_t* w0 = w_int8 + (n + 0) * K_padded;
                const int8_t* w1 = w_int8 + (n + 1) * K_padded;
                const int8_t* w2 = w_int8 + (n + 2) * K_padded;
                const int8_t* w3 = w_int8 + (n + 3) * K_padded;

                for (int k = 0; k + 15 < K; k += 16) {
                    int8x16_t x_vec = vld1q_s8(x_row + k);
                    acc0 = vdotq_s32(acc0, x_vec, vld1q_s8(w0 + k));
                    acc1 = vdotq_s32(acc1, x_vec, vld1q_s8(w1 + k));
                    acc2 = vdotq_s32(acc2, x_vec, vld1q_s8(w2 + k));
                    acc3 = vdotq_s32(acc3, x_vec, vld1q_s8(w3 + k));
                }

                y_row[n + 0] = static_cast<float>(vaddvq_s32(acc0)) * scale + (bias ? bias[n + 0] : 0.0f);
                y_row[n + 1] = static_cast<float>(vaddvq_s32(acc1)) * scale + (bias ? bias[n + 1] : 0.0f);
                y_row[n + 2] = static_cast<float>(vaddvq_s32(acc2)) * scale + (bias ? bias[n + 2] : 0.0f);
                y_row[n + 3] = static_cast<float>(vaddvq_s32(acc3)) * scale + (bias ? bias[n + 3] : 0.0f);
            }

            // Final remainder
            for (; n < N; n++) {
                const int8_t* w_row = w_int8 + n * K_padded;
                int32x4_t acc = vdupq_n_s32(0);

                for (int k = 0; k + 15 < K; k += 16) {
                    acc = vdotq_s32(acc, vld1q_s8(x_row + k), vld1q_s8(w_row + k));
                }

                y_row[n] = static_cast<float>(vaddvq_s32(acc)) * scale + (bias ? bias[n] : 0.0f);
            }
        }
    } else {
        // M < num_threads: parallelize over N instead (for single-token inference)
        // Split N into chunks, each thread handles a chunk of outputs for all M rows
        #pragma omp parallel
        {
            int tid = omp_get_thread_num();
            int nthreads = omp_get_num_threads();
            int n_per_thread = (N + nthreads - 1) / nthreads;
            int n_start = tid * n_per_thread;
            int n_end = std::min(n_start + n_per_thread, N);

            for (int m = 0; m < M; m++) {
                const int8_t* __restrict__ x_row = x_int8 + m * K;
                float scale = scales[m];
                float* y_row = y + m * N;

                // Process assigned N range with 8-way blocking
                int n = n_start;
                for (; n + 7 < n_end; n += 8) {
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

                    for (int k = 0; k + 15 < K; k += 16) {
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

                    y_row[n + 0] = static_cast<float>(vaddvq_s32(acc0)) * scale + (bias ? bias[n + 0] : 0.0f);
                    y_row[n + 1] = static_cast<float>(vaddvq_s32(acc1)) * scale + (bias ? bias[n + 1] : 0.0f);
                    y_row[n + 2] = static_cast<float>(vaddvq_s32(acc2)) * scale + (bias ? bias[n + 2] : 0.0f);
                    y_row[n + 3] = static_cast<float>(vaddvq_s32(acc3)) * scale + (bias ? bias[n + 3] : 0.0f);
                    y_row[n + 4] = static_cast<float>(vaddvq_s32(acc4)) * scale + (bias ? bias[n + 4] : 0.0f);
                    y_row[n + 5] = static_cast<float>(vaddvq_s32(acc5)) * scale + (bias ? bias[n + 5] : 0.0f);
                    y_row[n + 6] = static_cast<float>(vaddvq_s32(acc6)) * scale + (bias ? bias[n + 6] : 0.0f);
                    y_row[n + 7] = static_cast<float>(vaddvq_s32(acc7)) * scale + (bias ? bias[n + 7] : 0.0f);
                }

                // Handle remaining N
                for (; n < n_end; n++) {
                    const int8_t* w_row = w_int8 + n * K_padded;
                    int32x4_t acc = vdupq_n_s32(0);

                    for (int k = 0; k + 15 < K; k += 16) {
                        acc = vdotq_s32(acc, vld1q_s8(x_row + k), vld1q_s8(w_row + k));
                    }

                    y_row[n] = static_cast<float>(vaddvq_s32(acc)) * scale + (bias ? bias[n] : 0.0f);
                }
            }
        }
    }
}

/**
 * Graviton v3 Fused Softmax - Matmul + softmax + quantize in one kernel
 *
 * Eliminates intermediate float32 buffer write/read for attention scores.
 * Output is int8 with per-row scaling (softmax probabilities).
 */
void matmul_free_graviton_v3_fused_softmax(
    torch::Tensor x_int8_tensor,    // [M, K] int8
    torch::Tensor scale_tensor,     // [M] float32
    torch::Tensor w_int8_tensor,    // [N, K_padded] int8
    torch::Tensor w_sum_tensor,     // [N] int32 (unused)
    torch::Tensor y_tensor,         // [M, N] int8 output (softmax applied)
    torch::Tensor y_scale_tensor,   // [M] float32 output scales
    int M, int N, int K,
    int num_threads
) {
    const int8_t* __restrict__ x_int8 = x_int8_tensor.data_ptr<int8_t>();
    const float* __restrict__ scales = scale_tensor.data_ptr<float>();
    const int8_t* __restrict__ w_int8 = w_int8_tensor.data_ptr<int8_t>();
    int8_t* __restrict__ y = y_tensor.data_ptr<int8_t>();
    float* __restrict__ y_scales = y_scale_tensor.data_ptr<float>();

    const int K_padded = ((K + 15) / 16) * 16;
    const int N_padded = ((N + 3) / 4) * 4;

    // Adaptive M_TILE based on M
    const int M_TILE = (M >= 32) ? 32 : ((M >= 16) ? 16 : M);
    constexpr int N_TILE = 32;

    omp_set_num_threads(num_threads);

    // Process M in tiles
    for (int m_tile = 0; m_tile < M; m_tile += M_TILE) {
        const int m_end = std::min(m_tile + M_TILE, M);
        const int m_tile_size = m_end - m_tile;

        // Allocate row buffers for matmul results
        float* row_buffers = (float*)aligned_alloc(64, m_tile_size * N_padded * sizeof(float));

        // Step 1: Compute matmul for all rows in this M tile
        for (int n_tile = 0; n_tile < N; n_tile += N_TILE) {
            const int n_end_tile = std::min(n_tile + N_TILE, N);

            #pragma omp parallel for schedule(static)
            for (int m = m_tile; m < m_end; m++) {
                const int m_local = m - m_tile;
                const int8_t* __restrict__ x_row = x_int8 + m * K;
                float scale = scales[m];
                float* row_buffer = row_buffers + m_local * N_padded;

                // Process N tile with 8-way blocking
                int n = n_tile;
                for (; n + 7 < n_end_tile; n += 8) {
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

                    row_buffer[n + 0] = static_cast<float>(vaddvq_s32(acc0)) * scale;
                    row_buffer[n + 1] = static_cast<float>(vaddvq_s32(acc1)) * scale;
                    row_buffer[n + 2] = static_cast<float>(vaddvq_s32(acc2)) * scale;
                    row_buffer[n + 3] = static_cast<float>(vaddvq_s32(acc3)) * scale;
                    row_buffer[n + 4] = static_cast<float>(vaddvq_s32(acc4)) * scale;
                    row_buffer[n + 5] = static_cast<float>(vaddvq_s32(acc5)) * scale;
                    row_buffer[n + 6] = static_cast<float>(vaddvq_s32(acc6)) * scale;
                    row_buffer[n + 7] = static_cast<float>(vaddvq_s32(acc7)) * scale;
                }

                // Remainder with 4-way blocking
                for (; n + 3 < n_end_tile; n += 4) {
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

                    row_buffer[n + 0] = static_cast<float>(vaddvq_s32(acc0)) * scale;
                    row_buffer[n + 1] = static_cast<float>(vaddvq_s32(acc1)) * scale;
                    row_buffer[n + 2] = static_cast<float>(vaddvq_s32(acc2)) * scale;
                    row_buffer[n + 3] = static_cast<float>(vaddvq_s32(acc3)) * scale;
                }

                // Final remainder
                for (; n < n_end_tile; n++) {
                    const int8_t* w_row = w_int8 + n * K_padded;
                    int32x4_t acc = vdupq_n_s32(0);

                    int k = 0;
                    for (; k + 15 < K; k += 16) {
                        int8x16_t x_vec = vld1q_s8(x_row + k);
                        acc = vdotq_s32(acc, x_vec, vld1q_s8(w_row + k));
                    }

                    row_buffer[n] = static_cast<float>(vaddvq_s32(acc)) * scale;
                }
            }
        }

        // Step 2: Apply softmax and quantize
        #pragma omp parallel for schedule(static)
        for (int m = m_tile; m < m_end; m++) {
            const int m_local = m - m_tile;
            float* row_buffer = row_buffers + m_local * N_padded;
            int8_t* y_row = y + m * N;

            // Pad buffer for SIMD
            for (int n = N; n < N_padded; n++) {
                row_buffer[n] = -INFINITY;
            }

            // Find max (NEON)
            float32x4_t max_vec = vdupq_n_f32(-INFINITY);
            for (int n = 0; n < N_padded; n += 4) {
                float32x4_t vals = vld1q_f32(row_buffer + n);
                max_vec = vmaxq_f32(max_vec, vals);
            }
            float max_val = vmaxvq_f32(max_vec);

            // Compute exp(x - max) and sum
            float32x4_t sum_vec = vdupq_n_f32(0.0f);
            float32x4_t max_broadcast = vdupq_n_f32(max_val);
            for (int n = 0; n < N_padded; n += 4) {
                float32x4_t vals = vld1q_f32(row_buffer + n);
                float32x4_t shifted = vsubq_f32(vals, max_broadcast);
                float32x4_t exp_v = neon_exp_fast(shifted);
                vst1q_f32(row_buffer + n, exp_v);
                sum_vec = vaddq_f32(sum_vec, exp_v);
            }
            float sum = vaddvq_f32(sum_vec);

            // Normalize and quantize to int8
            float inv_sum = 1.0f / sum;
            y_scales[m] = 1.0f / 127.0f;  // Fixed scale for softmax output

            float32x4_t inv_sum_vec = vdupq_n_f32(inv_sum);
            float32x4_t scale_127 = vdupq_n_f32(127.0f);
            float32x4_t zero_vec = vdupq_n_f32(0.0f);
            float32x4_t max_127 = vdupq_n_f32(127.0f);

            int n = 0;
            for (; n + 3 < N; n += 4) {
                float32x4_t exp_v = vld1q_f32(row_buffer + n);
                float32x4_t prob = vmulq_f32(exp_v, inv_sum_vec);
                float32x4_t scaled = vmulq_f32(prob, scale_127);
                scaled = vmaxq_f32(scaled, zero_vec);
                scaled = vminq_f32(scaled, max_127);

                int32x4_t int_vals = vcvtnq_s32_f32(scaled);
                int16x4_t narrow16 = vmovn_s32(int_vals);
                int8x8_t narrow8 = vmovn_s16(vcombine_s16(narrow16, narrow16));

                y_row[n + 0] = vget_lane_s8(narrow8, 0);
                y_row[n + 1] = vget_lane_s8(narrow8, 1);
                y_row[n + 2] = vget_lane_s8(narrow8, 2);
                y_row[n + 3] = vget_lane_s8(narrow8, 3);
            }
            for (; n < N; n++) {
                float prob = row_buffer[n] * inv_sum;
                int8_t q_val = static_cast<int8_t>(std::min(127.0f, std::round(prob * 127.0f)));
                y_row[n] = q_val;
            }
        }

        free(row_buffers);
    }
}

/**
 * Graviton v3 Fused Quantize - Matmul + quantize (for down projection)
 *
 * Output is int8 with per-row scaling.
 */
void matmul_free_graviton_v3_fused_quantize(
    torch::Tensor x_int8_tensor,    // [M, K] int8
    torch::Tensor scale_tensor,     // [M] float32
    torch::Tensor w_int8_tensor,    // [N, K_padded] int8
    torch::Tensor w_sum_tensor,     // [N] int32 (unused)
    torch::Tensor y_tensor,         // [M, N] int8 output
    torch::Tensor y_scale_tensor,   // [M] float32 output scales
    int M, int N, int K,
    int num_threads
) {
    const int8_t* __restrict__ x_int8 = x_int8_tensor.data_ptr<int8_t>();
    const float* __restrict__ scales = scale_tensor.data_ptr<float>();
    const int8_t* __restrict__ w_int8 = w_int8_tensor.data_ptr<int8_t>();
    int8_t* __restrict__ y = y_tensor.data_ptr<int8_t>();
    float* __restrict__ y_scales = y_scale_tensor.data_ptr<float>();

    const int K_padded = ((K + 15) / 16) * 16;
    const int N_padded = ((N + 3) / 4) * 4;

    const int M_TILE = (M >= 32) ? 32 : ((M >= 16) ? 16 : M);
    constexpr int N_TILE = 32;

    omp_set_num_threads(num_threads);

    for (int m_tile = 0; m_tile < M; m_tile += M_TILE) {
        const int m_end = std::min(m_tile + M_TILE, M);
        const int m_tile_size = m_end - m_tile;

        float* row_buffers = (float*)aligned_alloc(64, m_tile_size * N_padded * sizeof(float));

        // Step 1: Compute matmul
        for (int n_tile = 0; n_tile < N; n_tile += N_TILE) {
            const int n_end_tile = std::min(n_tile + N_TILE, N);

            #pragma omp parallel for schedule(static)
            for (int m = m_tile; m < m_end; m++) {
                const int m_local = m - m_tile;
                const int8_t* __restrict__ x_row = x_int8 + m * K;
                float scale = scales[m];
                float* row_buffer = row_buffers + m_local * N_padded;

                int n = n_tile;
                for (; n + 7 < n_end_tile; n += 8) {
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

                    row_buffer[n + 0] = static_cast<float>(vaddvq_s32(acc0)) * scale;
                    row_buffer[n + 1] = static_cast<float>(vaddvq_s32(acc1)) * scale;
                    row_buffer[n + 2] = static_cast<float>(vaddvq_s32(acc2)) * scale;
                    row_buffer[n + 3] = static_cast<float>(vaddvq_s32(acc3)) * scale;
                    row_buffer[n + 4] = static_cast<float>(vaddvq_s32(acc4)) * scale;
                    row_buffer[n + 5] = static_cast<float>(vaddvq_s32(acc5)) * scale;
                    row_buffer[n + 6] = static_cast<float>(vaddvq_s32(acc6)) * scale;
                    row_buffer[n + 7] = static_cast<float>(vaddvq_s32(acc7)) * scale;
                }

                for (; n + 3 < n_end_tile; n += 4) {
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

                    row_buffer[n + 0] = static_cast<float>(vaddvq_s32(acc0)) * scale;
                    row_buffer[n + 1] = static_cast<float>(vaddvq_s32(acc1)) * scale;
                    row_buffer[n + 2] = static_cast<float>(vaddvq_s32(acc2)) * scale;
                    row_buffer[n + 3] = static_cast<float>(vaddvq_s32(acc3)) * scale;
                }

                for (; n < n_end_tile; n++) {
                    const int8_t* w_row = w_int8 + n * K_padded;
                    int32x4_t acc = vdupq_n_s32(0);

                    int k = 0;
                    for (; k + 15 < K; k += 16) {
                        int8x16_t x_vec = vld1q_s8(x_row + k);
                        acc = vdotq_s32(acc, x_vec, vld1q_s8(w_row + k));
                    }

                    row_buffer[n] = static_cast<float>(vaddvq_s32(acc)) * scale;
                }
            }
        }

        // Step 2: Quantize to int8
        #pragma omp parallel for schedule(static)
        for (int m = m_tile; m < m_end; m++) {
            const int m_local = m - m_tile;
            float* row_buffer = row_buffers + m_local * N_padded;
            int8_t* y_row = y + m * N;

            // Find max abs
            float32x4_t max_abs_vec = vdupq_n_f32(0.0f);
            for (int n = 0; n < N; n += 4) {
                float32x4_t vals = vld1q_f32(row_buffer + n);
                float32x4_t abs_vals = vabsq_f32(vals);
                max_abs_vec = vmaxq_f32(max_abs_vec, abs_vals);
            }
            float max_abs = vmaxvq_f32(max_abs_vec);

            // Handle remainder for max
            for (int n = (N / 4) * 4; n < N; n++) {
                float abs_val = std::abs(row_buffer[n]);
                if (abs_val > max_abs) max_abs = abs_val;
            }

            float out_scale = max_abs / 127.0f;
            if (out_scale == 0.0f) out_scale = 1.0f;
            y_scales[m] = out_scale;

            float inv_scale = 1.0f / out_scale;
            float32x4_t inv_scale_vec = vdupq_n_f32(inv_scale);
            float32x4_t min_val = vdupq_n_f32(-127.0f);
            float32x4_t max_val = vdupq_n_f32(127.0f);

            int n = 0;
            for (; n + 3 < N; n += 4) {
                float32x4_t vals = vld1q_f32(row_buffer + n);
                float32x4_t scaled = vmulq_f32(vals, inv_scale_vec);
                scaled = vmaxq_f32(min_val, vminq_f32(max_val, scaled));

                int32x4_t int_vals = vcvtnq_s32_f32(scaled);
                int16x4_t narrow16 = vmovn_s32(int_vals);
                int8x8_t narrow8 = vmovn_s16(vcombine_s16(narrow16, narrow16));

                y_row[n + 0] = vget_lane_s8(narrow8, 0);
                y_row[n + 1] = vget_lane_s8(narrow8, 1);
                y_row[n + 2] = vget_lane_s8(narrow8, 2);
                y_row[n + 3] = vget_lane_s8(narrow8, 3);
            }
            for (; n < N; n++) {
                float val = row_buffer[n] * inv_scale;
                val = std::max(-127.0f, std::min(127.0f, val));
                y_row[n] = static_cast<int8_t>(std::round(val));
            }
        }

        free(row_buffers);
    }
}

/**
 * SwiGLU activation: swish(gate) * up
 * swish(x) = x * sigmoid(x) = x / (1 + exp(-x))
 */
inline float32x4_t neon_swish(float32x4_t x) {
    float32x4_t neg_x = vnegq_f32(x);
    float32x4_t exp_neg_x = neon_exp_fast(neg_x);
    float32x4_t one = vdupq_n_f32(1.0f);
    float32x4_t denom = vaddq_f32(one, exp_neg_x);
    return vdivq_f32(x, denom);
}

/**
 * Graviton v3 Fused SwiGLU - Matmul + SwiGLU activation
 *
 * Computes: output = swish(x @ W_gate) * (x @ W_up)
 * Both gate and up projections done in one pass over input.
 */
void matmul_free_graviton_v3_fused_swiglu(
    torch::Tensor x_int8_tensor,      // [M, K] int8
    torch::Tensor scale_tensor,       // [M] float32
    torch::Tensor w_gate_tensor,      // [N, K_padded] int8
    torch::Tensor w_gate_sum_tensor,  // [N] int32 (unused)
    torch::Tensor w_up_tensor,        // [N, K_padded] int8
    torch::Tensor w_up_sum_tensor,    // [N] int32 (unused)
    torch::Tensor y_tensor,           // [M, N] int8 output
    torch::Tensor y_scale_tensor,     // [M] float32 output scales
    int M, int N, int K,
    int num_threads
) {
    const int8_t* __restrict__ x_int8 = x_int8_tensor.data_ptr<int8_t>();
    const float* __restrict__ scales = scale_tensor.data_ptr<float>();
    const int8_t* __restrict__ w_gate = w_gate_tensor.data_ptr<int8_t>();
    const int8_t* __restrict__ w_up = w_up_tensor.data_ptr<int8_t>();
    int8_t* __restrict__ y = y_tensor.data_ptr<int8_t>();
    float* __restrict__ y_scales = y_scale_tensor.data_ptr<float>();

    const int K_padded = ((K + 15) / 16) * 16;
    const int N_padded = ((N + 3) / 4) * 4;

    const int M_TILE = (M >= 32) ? 32 : ((M >= 16) ? 16 : M);
    constexpr int N_TILE = 32;

    omp_set_num_threads(num_threads);

    for (int m_tile = 0; m_tile < M; m_tile += M_TILE) {
        const int m_end = std::min(m_tile + M_TILE, M);
        const int m_tile_size = m_end - m_tile;

        // Buffers for gate and up projections
        float* gate_buffers = (float*)aligned_alloc(64, m_tile_size * N_padded * sizeof(float));
        float* up_buffers = (float*)aligned_alloc(64, m_tile_size * N_padded * sizeof(float));

        // Step 1: Compute both gate and up projections
        for (int n_tile = 0; n_tile < N; n_tile += N_TILE) {
            const int n_end_tile = std::min(n_tile + N_TILE, N);

            #pragma omp parallel for schedule(static)
            for (int m = m_tile; m < m_end; m++) {
                const int m_local = m - m_tile;
                const int8_t* __restrict__ x_row = x_int8 + m * K;
                float scale = scales[m];
                float* gate_buffer = gate_buffers + m_local * N_padded;
                float* up_buffer = up_buffers + m_local * N_padded;

                int n = n_tile;
                for (; n + 3 < n_end_tile; n += 4) {
                    // Gate accumulators
                    int32x4_t acc_g0 = vdupq_n_s32(0);
                    int32x4_t acc_g1 = vdupq_n_s32(0);
                    int32x4_t acc_g2 = vdupq_n_s32(0);
                    int32x4_t acc_g3 = vdupq_n_s32(0);

                    // Up accumulators
                    int32x4_t acc_u0 = vdupq_n_s32(0);
                    int32x4_t acc_u1 = vdupq_n_s32(0);
                    int32x4_t acc_u2 = vdupq_n_s32(0);
                    int32x4_t acc_u3 = vdupq_n_s32(0);

                    const int8_t* wg0 = w_gate + (n + 0) * K_padded;
                    const int8_t* wg1 = w_gate + (n + 1) * K_padded;
                    const int8_t* wg2 = w_gate + (n + 2) * K_padded;
                    const int8_t* wg3 = w_gate + (n + 3) * K_padded;

                    const int8_t* wu0 = w_up + (n + 0) * K_padded;
                    const int8_t* wu1 = w_up + (n + 1) * K_padded;
                    const int8_t* wu2 = w_up + (n + 2) * K_padded;
                    const int8_t* wu3 = w_up + (n + 3) * K_padded;

                    int k = 0;
                    for (; k + 15 < K; k += 16) {
                        int8x16_t x_vec = vld1q_s8(x_row + k);

                        // Gate
                        acc_g0 = vdotq_s32(acc_g0, x_vec, vld1q_s8(wg0 + k));
                        acc_g1 = vdotq_s32(acc_g1, x_vec, vld1q_s8(wg1 + k));
                        acc_g2 = vdotq_s32(acc_g2, x_vec, vld1q_s8(wg2 + k));
                        acc_g3 = vdotq_s32(acc_g3, x_vec, vld1q_s8(wg3 + k));

                        // Up
                        acc_u0 = vdotq_s32(acc_u0, x_vec, vld1q_s8(wu0 + k));
                        acc_u1 = vdotq_s32(acc_u1, x_vec, vld1q_s8(wu1 + k));
                        acc_u2 = vdotq_s32(acc_u2, x_vec, vld1q_s8(wu2 + k));
                        acc_u3 = vdotq_s32(acc_u3, x_vec, vld1q_s8(wu3 + k));
                    }

                    gate_buffer[n + 0] = static_cast<float>(vaddvq_s32(acc_g0)) * scale;
                    gate_buffer[n + 1] = static_cast<float>(vaddvq_s32(acc_g1)) * scale;
                    gate_buffer[n + 2] = static_cast<float>(vaddvq_s32(acc_g2)) * scale;
                    gate_buffer[n + 3] = static_cast<float>(vaddvq_s32(acc_g3)) * scale;

                    up_buffer[n + 0] = static_cast<float>(vaddvq_s32(acc_u0)) * scale;
                    up_buffer[n + 1] = static_cast<float>(vaddvq_s32(acc_u1)) * scale;
                    up_buffer[n + 2] = static_cast<float>(vaddvq_s32(acc_u2)) * scale;
                    up_buffer[n + 3] = static_cast<float>(vaddvq_s32(acc_u3)) * scale;
                }

                // Remainder
                for (; n < n_end_tile; n++) {
                    const int8_t* wg_row = w_gate + n * K_padded;
                    const int8_t* wu_row = w_up + n * K_padded;
                    int32x4_t acc_g = vdupq_n_s32(0);
                    int32x4_t acc_u = vdupq_n_s32(0);

                    int k = 0;
                    for (; k + 15 < K; k += 16) {
                        int8x16_t x_vec = vld1q_s8(x_row + k);
                        acc_g = vdotq_s32(acc_g, x_vec, vld1q_s8(wg_row + k));
                        acc_u = vdotq_s32(acc_u, x_vec, vld1q_s8(wu_row + k));
                    }

                    gate_buffer[n] = static_cast<float>(vaddvq_s32(acc_g)) * scale;
                    up_buffer[n] = static_cast<float>(vaddvq_s32(acc_u)) * scale;
                }
            }
        }

        // Step 2: Apply SwiGLU and quantize
        #pragma omp parallel for schedule(static)
        for (int m = m_tile; m < m_end; m++) {
            const int m_local = m - m_tile;
            float* gate_buffer = gate_buffers + m_local * N_padded;
            float* up_buffer = up_buffers + m_local * N_padded;
            int8_t* y_row = y + m * N;

            // Compute swish(gate) * up and find max abs
            float max_abs = 0.0f;
            int n = 0;
            for (; n + 3 < N; n += 4) {
                float32x4_t gate = vld1q_f32(gate_buffer + n);
                float32x4_t up = vld1q_f32(up_buffer + n);
                float32x4_t swish_gate = neon_swish(gate);
                float32x4_t result = vmulq_f32(swish_gate, up);
                vst1q_f32(gate_buffer + n, result);  // Reuse buffer

                float32x4_t abs_result = vabsq_f32(result);
                float local_max = vmaxvq_f32(abs_result);
                if (local_max > max_abs) max_abs = local_max;
            }
            for (; n < N; n++) {
                float g = gate_buffer[n];
                float u = up_buffer[n];
                float swish_g = g / (1.0f + std::exp(-g));
                float result = swish_g * u;
                gate_buffer[n] = result;
                if (std::abs(result) > max_abs) max_abs = std::abs(result);
            }

            // Quantize
            float out_scale = max_abs / 127.0f;
            if (out_scale == 0.0f) out_scale = 1.0f;
            y_scales[m] = out_scale;

            float inv_scale = 1.0f / out_scale;
            float32x4_t inv_scale_vec = vdupq_n_f32(inv_scale);
            float32x4_t min_val = vdupq_n_f32(-127.0f);
            float32x4_t max_val = vdupq_n_f32(127.0f);

            n = 0;
            for (; n + 3 < N; n += 4) {
                float32x4_t vals = vld1q_f32(gate_buffer + n);
                float32x4_t scaled = vmulq_f32(vals, inv_scale_vec);
                scaled = vmaxq_f32(min_val, vminq_f32(max_val, scaled));

                int32x4_t int_vals = vcvtnq_s32_f32(scaled);
                int16x4_t narrow16 = vmovn_s32(int_vals);
                int8x8_t narrow8 = vmovn_s16(vcombine_s16(narrow16, narrow16));

                y_row[n + 0] = vget_lane_s8(narrow8, 0);
                y_row[n + 1] = vget_lane_s8(narrow8, 1);
                y_row[n + 2] = vget_lane_s8(narrow8, 2);
                y_row[n + 3] = vget_lane_s8(narrow8, 3);
            }
            for (; n < N; n++) {
                float val = gate_buffer[n] * inv_scale;
                val = std::max(-127.0f, std::min(127.0f, val));
                y_row[n] = static_cast<int8_t>(std::round(val));
            }
        }

        free(gate_buffers);
        free(up_buffers);
    }
}

// PyBind11 module
PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("quantize_activations_int8_graviton", &quantize_activations_int8_graviton,
          "Quantize activations to int8 (Graviton)");
    m.def("pack_weights_graviton", &pack_weights_graviton,
          "Pack ternary weights for Graviton SDOT kernel");
    m.def("matmul_free_graviton_v2", &matmul_free_graviton_v2,
          "Graviton v2: register-blocked SDOT");
    m.def("matmul_free_graviton_v3", &matmul_free_graviton_v3,
          "Graviton v3: cache-optimized tiled");
    m.def("matmul_free_graviton_v3_fused_softmax", &matmul_free_graviton_v3_fused_softmax,
          "Graviton v3 with fused softmax");
    m.def("matmul_free_graviton_v3_fused_quantize", &matmul_free_graviton_v3_fused_quantize,
          "Graviton v3 with fused quantize");
    m.def("matmul_free_graviton_v3_fused_swiglu", &matmul_free_graviton_v3_fused_swiglu,
          "Graviton v3 with fused SwiGLU");
}
