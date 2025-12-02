/*
 * True MatMul-Free ARM NEON TBL Kernel
 *
 * This is the ARM equivalent of matmul_free_tmac_int8.cpp on x86.
 * Uses ARM's vqtbl1q_u8 instruction (equivalent to x86 PSHUFB) for
 * 16 parallel table lookups.
 *
 * Key features:
 * - TRUE MatMul-free: No multiplication in inner loop
 * - 2-bit packed weights: 16x memory savings vs float32
 * - LUT-based computation: sum = 2*value_lookup - sign_lookup
 *
 * Based on Microsoft T-MAC and BitNet.cpp approaches.
 */

#include <arm_neon.h>
#include <cstring>
#include <cstdint>
#include <cmath>
#include <algorithm>
#include <omp.h>
#include <torch/extension.h>

/**
 * Quantize float activations to int8
 *
 * Per-row symmetric quantization:
 *   scale = max(abs(x)) / 127
 *   x_int8 = round(x / scale)
 */
void quantize_activations_int8_tbl(
    torch::Tensor x_float,      // [M, K] float32
    torch::Tensor x_int8,       // [M, K] int8
    torch::Tensor scale_tensor, // [M] float32 - one scale per row
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

        float max_abs = vmaxvq_f32(max_vec);
        for (; k < K; k++) {
            float abs_val = std::abs(x_row[k]);
            if (abs_val > max_abs) max_abs = abs_val;
        }

        float scale = max_abs / 127.0f;
        if (scale == 0.0f) scale = 1.0f;
        scales[m] = scale;

        float inv_scale = 1.0f / scale;

        // Quantize
        for (k = 0; k < K; k++) {
            float val = x_row[k] * inv_scale;
            val = std::max(-127.0f, std::min(127.0f, val));
            xq_row[k] = static_cast<int8_t>(std::round(val));
        }
    }
}

/**
 * Pack ternary weights into bit-plane format for T-MAC
 *
 * Layout: [K_groups, N] where K_groups = ceil(K/4)
 * Each byte stores a 4-bit nibble index for 16-entry LUT
 *
 * Weight encoding (bit-plane decomposition):
 *   sign_plane[i] = 1 if weight[i] != 0 (i.e., is +1 or -1)
 *   value_plane[i] = 1 if weight[i] == +1
 *
 * The result formula: weight[i] = 2*value[i] - sign[i]
 *   - If weight = +1: sign=1, value=1 -> 2*1 - 1 = +1 ✓
 *   - If weight = -1: sign=1, value=0 -> 2*0 - 1 = -1 ✓
 *   - If weight =  0: sign=0, value=0 -> 2*0 - 0 =  0 ✓
 */
void pack_ternary_bitplanes_tbl(
    torch::Tensor w_tensor,
    torch::Tensor sign_plane_tensor,   // [K_groups, N]
    torch::Tensor value_plane_tensor,  // [K_groups, N]
    int N, int K
) {
    const float* w = w_tensor.data_ptr<float>();
    uint8_t* sign_plane = sign_plane_tensor.data_ptr<uint8_t>();
    uint8_t* value_plane = value_plane_tensor.data_ptr<uint8_t>();

    const int K_groups = (K + 3) / 4;

    #pragma omp parallel for collapse(2)
    for (int kg = 0; kg < K_groups; kg++) {
        for (int n = 0; n < N; n++) {
            uint8_t sign_nibble = 0;
            uint8_t value_nibble = 0;

            for (int i = 0; i < 4 && (kg * 4 + i) < K; i++) {
                float val = w[n * K + kg * 4 + i];
                if (val > 0.5f) {
                    // +1: sign=1, value=1
                    sign_nibble |= (1 << i);
                    value_nibble |= (1 << i);
                } else if (val < -0.5f) {
                    // -1: sign=1, value=0
                    sign_nibble |= (1 << i);
                }
                // 0: sign=0, value=0 (default)
            }
            sign_plane[kg * N + n] = sign_nibble;
            value_plane[kg * N + n] = value_nibble;
        }
    }
}

/**
 * Build 16-entry int16 LUT from 4 int8 activations
 *
 * Each entry is the sum of selected activations based on bit pattern.
 * Entry i = sum of activations where bit j of i is set.
 *
 * Uses int16 to avoid overflow (4 int8 values can sum to ±508).
 */
inline void build_lut_16_int16(const int8_t* x, int16_t* lut) {
    const int16_t x0 = x[0], x1 = x[1], x2 = x[2], x3 = x[3];

    lut[0]  = 0;
    lut[1]  = x0;
    lut[2]  = x1;
    lut[3]  = x0 + x1;
    lut[4]  = x2;
    lut[5]  = x0 + x2;
    lut[6]  = x1 + x2;
    lut[7]  = x0 + x1 + x2;
    lut[8]  = x3;
    lut[9]  = x0 + x3;
    lut[10] = x1 + x3;
    lut[11] = x0 + x1 + x3;
    lut[12] = x2 + x3;
    lut[13] = x0 + x2 + x3;
    lut[14] = x1 + x2 + x3;
    lut[15] = x0 + x1 + x2 + x3;
}

/**
 * Build split low/high byte LUTs for TBL pack-and-unpack technique
 *
 * Same as x86 BitNet.cpp approach: split int16 LUT into low and high bytes,
 * do TBL twice (once for low, once for high), then unpack to get int16 results.
 */
inline void build_lut_16_split(const int8_t* x, uint8_t* lut_lo, uint8_t* lut_hi) {
    const int16_t x0 = x[0], x1 = x[1], x2 = x[2], x3 = x[3];

    int16_t sums[16];
    sums[0]  = 0;
    sums[1]  = x0;
    sums[2]  = x1;
    sums[3]  = x0 + x1;
    sums[4]  = x2;
    sums[5]  = x0 + x2;
    sums[6]  = x1 + x2;
    sums[7]  = x0 + x1 + x2;
    sums[8]  = x3;
    sums[9]  = x0 + x3;
    sums[10] = x1 + x3;
    sums[11] = x0 + x1 + x3;
    sums[12] = x2 + x3;
    sums[13] = x0 + x2 + x3;
    sums[14] = x1 + x2 + x3;
    sums[15] = x0 + x1 + x2 + x3;

    // Split each int16 into low byte and high byte
    for (int i = 0; i < 16; i++) {
        lut_lo[i] = static_cast<uint8_t>(sums[i] & 0xFF);
        lut_hi[i] = static_cast<uint8_t>((sums[i] >> 8) & 0xFF);
    }
}

/**
 * Vectorized LUT construction using NEON
 *
 * Builds the 16-entry LUT more efficiently using NEON operations.
 * Returns the LUT as two NEON registers (lo/hi bytes).
 */
inline void build_lut_16_split_neon(const int8_t* x, uint8x16_t& lut_lo_vec, uint8x16_t& lut_hi_vec) {
    // Load 4 int8 values and extend to int16
    const int16_t x0 = x[0], x1 = x[1], x2 = x[2], x3 = x[3];

    // Build LUT using NEON - compute all 16 sums in parallel where possible
    // Pattern: entry i = sum of x[j] where bit j of i is set

    // First, build the 16 int16 values
    // Using addition chains to minimize operations
    const int16_t x01 = static_cast<int16_t>(x0 + x1);
    const int16_t x02 = static_cast<int16_t>(x0 + x2);
    const int16_t x12 = static_cast<int16_t>(x1 + x2);
    const int16_t x012 = static_cast<int16_t>(x01 + x2);

    // Create int16 vectors for efficient storage
    // sums[0..7] = combinations without x3
    // sums[8..15] = combinations with x3
    int16x8_t sums_lo, sums_hi;

    // Lower 8: {0, x0, x1, x0+x1, x2, x0+x2, x1+x2, x0+x1+x2}
    alignas(16) int16_t lo_arr[8] = {0, x0, x1, x01, x2, x02, x12, x012};
    sums_lo = vld1q_s16(lo_arr);

    // Upper 8: add x3 to each of the lower 8
    int16x8_t x3_vec = vdupq_n_s16(x3);
    sums_hi = vaddq_s16(sums_lo, x3_vec);

    // Extract low and high bytes using uzp (unzip)
    // Convert to uint8 for TBL
    uint8x16_t sums_lo_u8 = vreinterpretq_u8_s16(sums_lo);
    uint8x16_t sums_hi_u8 = vreinterpretq_u8_s16(sums_hi);

    // Combine into single 16-element result (16 int16 = 32 bytes)
    // uzp1 extracts even bytes (low bytes of int16)
    // uzp2 extracts odd bytes (high bytes of int16)
    lut_lo_vec = vuzp1q_u8(sums_lo_u8, sums_hi_u8);
    lut_hi_vec = vuzp2q_u8(sums_lo_u8, sums_hi_u8);
}

/**
 * True MatMul-Free TBL Kernel
 *
 * Uses ARM NEON vqtbl1q_u8 for 16 parallel table lookups.
 * This is the ARM equivalent of x86 PSHUFB (_mm_shuffle_epi8).
 *
 * Memory: 2 bits per weight (16x savings vs float32)
 * Computation: Pure lookup + addition (no multiplication)
 */
void matmul_free_neon_tbl(
    torch::Tensor x_int8_tensor,       // [M, K] int8
    torch::Tensor scale_tensor,        // [M] float32
    torch::Tensor sign_plane_tensor,   // [K_groups, N] uint8
    torch::Tensor value_plane_tensor,  // [K_groups, N] uint8
    torch::Tensor y_tensor,            // [M, N] float32 output
    torch::Tensor bias_tensor,
    int M, int N, int K,
    int num_threads
) {
    const int8_t* __restrict__ x_int8 = x_int8_tensor.data_ptr<int8_t>();
    const float* __restrict__ scales = scale_tensor.data_ptr<float>();
    const uint8_t* __restrict__ sign_plane = sign_plane_tensor.data_ptr<uint8_t>();
    const uint8_t* __restrict__ value_plane = value_plane_tensor.data_ptr<uint8_t>();
    float* __restrict__ y = y_tensor.data_ptr<float>();
    const float* bias = bias_tensor.defined() ? bias_tensor.data_ptr<float>() : nullptr;

    const int K_groups = (K + 3) / 4;

    omp_set_num_threads(num_threads);

    #pragma omp parallel
    {
        // Per-thread accumulator buffer
        alignas(64) int32_t y_acc[N];

        #pragma omp for schedule(static)
        for (int m = 0; m < M; m++) {
            const int8_t* __restrict__ x_row = x_int8 + m * K;
            float scale = scales[m];

            // Initialize accumulator
            memset(y_acc, 0, N * sizeof(int32_t));

            // K-first iteration: build LUT once, use for all N outputs
            for (int kg = 0; kg < K_groups; kg++) {
                const int k_base = kg * 4;

                // Get 4 int8 activations
                int8_t x_vals[4] = {0, 0, 0, 0};
                for (int i = 0; i < 4 && k_base + i < K; i++) {
                    x_vals[i] = x_row[k_base + i];
                }

                // Build split LUTs (low byte and high byte)
                alignas(16) uint8_t lut_lo[16];
                alignas(16) uint8_t lut_hi[16];
                build_lut_16_split(x_vals, lut_lo, lut_hi);

                // Load LUTs into NEON registers
                uint8x16_t lut_lo_vec = vld1q_u8(lut_lo);
                uint8x16_t lut_hi_vec = vld1q_u8(lut_hi);

                const uint8_t* __restrict__ sign_row = sign_plane + kg * N;
                const uint8_t* __restrict__ value_row = value_plane + kg * N;

                // Mask for 4-bit indices
                uint8x16_t mask_0f = vdupq_n_u8(0x0F);

                // Process 16 outputs at a time with TBL
                int n = 0;
                for (; n + 15 < N; n += 16) {
                    // Load 16 sign and value indices
                    uint8x16_t sign_idx = vld1q_u8(sign_row + n);
                    uint8x16_t value_idx = vld1q_u8(value_row + n);

                    // Mask to 4 bits (indices 0-15)
                    sign_idx = vandq_u8(sign_idx, mask_0f);
                    value_idx = vandq_u8(value_idx, mask_0f);

                    // TBL lookups for sign (low and high bytes)
                    uint8x16_t sign_lo = vqtbl1q_u8(lut_lo_vec, sign_idx);
                    uint8x16_t sign_hi = vqtbl1q_u8(lut_hi_vec, sign_idx);

                    // TBL lookups for value (low and high bytes)
                    uint8x16_t value_lo = vqtbl1q_u8(lut_lo_vec, value_idx);
                    uint8x16_t value_hi = vqtbl1q_u8(lut_hi_vec, value_idx);

                    // Unpack to int16: interleave low and high bytes
                    // vzip1q gives elements 0,2,4,6,8,10,12,14 interleaved
                    // vzip2q gives elements 1,3,5,7,9,11,13,15 interleaved
                    // We need proper byte ordering: lo[0],hi[0],lo[1],hi[1],...
                    int16x8_t sign_16_lo = vreinterpretq_s16_u8(vzip1q_u8(sign_lo, sign_hi));
                    int16x8_t sign_16_hi = vreinterpretq_s16_u8(vzip2q_u8(sign_lo, sign_hi));
                    int16x8_t value_16_lo = vreinterpretq_s16_u8(vzip1q_u8(value_lo, value_hi));
                    int16x8_t value_16_hi = vreinterpretq_s16_u8(vzip2q_u8(value_lo, value_hi));

                    // Compute 2*value - sign for each int16
                    // Use shift left by 1 instead of multiply
                    int16x8_t res_16_lo = vsubq_s16(vshlq_n_s16(value_16_lo, 1), sign_16_lo);
                    int16x8_t res_16_hi = vsubq_s16(vshlq_n_s16(value_16_hi, 1), sign_16_hi);

                    // Extend to int32 and accumulate
                    // First 4 results (from res_16_lo low half)
                    int32x4_t res_32_0 = vmovl_s16(vget_low_s16(res_16_lo));
                    int32x4_t res_32_1 = vmovl_s16(vget_high_s16(res_16_lo));
                    int32x4_t res_32_2 = vmovl_s16(vget_low_s16(res_16_hi));
                    int32x4_t res_32_3 = vmovl_s16(vget_high_s16(res_16_hi));

                    // Load current accumulators and add
                    int32x4_t acc0 = vld1q_s32(y_acc + n);
                    int32x4_t acc1 = vld1q_s32(y_acc + n + 4);
                    int32x4_t acc2 = vld1q_s32(y_acc + n + 8);
                    int32x4_t acc3 = vld1q_s32(y_acc + n + 12);

                    acc0 = vaddq_s32(acc0, res_32_0);
                    acc1 = vaddq_s32(acc1, res_32_1);
                    acc2 = vaddq_s32(acc2, res_32_2);
                    acc3 = vaddq_s32(acc3, res_32_3);

                    vst1q_s32(y_acc + n, acc0);
                    vst1q_s32(y_acc + n + 4, acc1);
                    vst1q_s32(y_acc + n + 8, acc2);
                    vst1q_s32(y_acc + n + 12, acc3);
                }

                // Handle remaining outputs with scalar code
                alignas(32) int16_t lut[16];
                build_lut_16_int16(x_vals, lut);
                for (; n < N; n++) {
                    uint8_t s = sign_row[n] & 0x0F;
                    uint8_t v = value_row[n] & 0x0F;
                    y_acc[n] += 2 * static_cast<int32_t>(lut[v]) - static_cast<int32_t>(lut[s]);
                }
            }

            // Convert int32 accumulator to float32 output with scale
            float* y_row = y + m * N;
            if (bias != nullptr) {
                for (int n = 0; n < N; n++) {
                    y_row[n] = static_cast<float>(y_acc[n]) * scale + bias[n];
                }
            } else {
                // Vectorized conversion
                float32x4_t scale_vec = vdupq_n_f32(scale);
                int n = 0;
                for (; n + 3 < N; n += 4) {
                    int32x4_t acc = vld1q_s32(y_acc + n);
                    float32x4_t acc_f = vcvtq_f32_s32(acc);
                    float32x4_t result = vmulq_f32(acc_f, scale_vec);
                    vst1q_f32(y_row + n, result);
                }
                for (; n < N; n++) {
                    y_row[n] = static_cast<float>(y_acc[n]) * scale;
                }
            }
        }
    }
}

/**
 * Optimized TBL kernel with N-blocking
 *
 * Processes 32 outputs at a time to maximize LUT reuse.
 * Each LUT build is amortized over more outputs.
 */
void matmul_free_neon_tbl_v2(
    torch::Tensor x_int8_tensor,       // [M, K] int8
    torch::Tensor scale_tensor,        // [M] float32
    torch::Tensor sign_plane_tensor,   // [K_groups, N] uint8
    torch::Tensor value_plane_tensor,  // [K_groups, N] uint8
    torch::Tensor y_tensor,            // [M, N] float32 output
    torch::Tensor bias_tensor,
    int M, int N, int K,
    int num_threads
) {
    const int8_t* __restrict__ x_int8 = x_int8_tensor.data_ptr<int8_t>();
    const float* __restrict__ scales = scale_tensor.data_ptr<float>();
    const uint8_t* __restrict__ sign_plane = sign_plane_tensor.data_ptr<uint8_t>();
    const uint8_t* __restrict__ value_plane = value_plane_tensor.data_ptr<uint8_t>();
    float* __restrict__ y = y_tensor.data_ptr<float>();
    const float* bias = bias_tensor.defined() ? bias_tensor.data_ptr<float>() : nullptr;

    const int K_groups = (K + 3) / 4;

    omp_set_num_threads(num_threads);

    #pragma omp parallel
    {
        alignas(64) int32_t y_acc[N];

        #pragma omp for schedule(static)
        for (int m = 0; m < M; m++) {
            const int8_t* __restrict__ x_row = x_int8 + m * K;
            float scale = scales[m];

            memset(y_acc, 0, N * sizeof(int32_t));

            // K-first iteration
            for (int kg = 0; kg < K_groups; kg++) {
                const int k_base = kg * 4;

                int8_t x_vals[4] = {0, 0, 0, 0};
                for (int i = 0; i < 4 && k_base + i < K; i++) {
                    x_vals[i] = x_row[k_base + i];
                }

                alignas(16) uint8_t lut_lo[16];
                alignas(16) uint8_t lut_hi[16];
                build_lut_16_split(x_vals, lut_lo, lut_hi);

                uint8x16_t lut_lo_vec = vld1q_u8(lut_lo);
                uint8x16_t lut_hi_vec = vld1q_u8(lut_hi);

                const uint8_t* __restrict__ sign_row = sign_plane + kg * N;
                const uint8_t* __restrict__ value_row = value_plane + kg * N;

                uint8x16_t mask_0f = vdupq_n_u8(0x0F);

                // Process 32 outputs at a time (2 x 16)
                int n = 0;
                for (; n + 31 < N; n += 32) {
                    // First batch of 16
                    uint8x16_t sign_idx_0 = vandq_u8(vld1q_u8(sign_row + n), mask_0f);
                    uint8x16_t value_idx_0 = vandq_u8(vld1q_u8(value_row + n), mask_0f);

                    uint8x16_t sign_lo_0 = vqtbl1q_u8(lut_lo_vec, sign_idx_0);
                    uint8x16_t sign_hi_0 = vqtbl1q_u8(lut_hi_vec, sign_idx_0);
                    uint8x16_t value_lo_0 = vqtbl1q_u8(lut_lo_vec, value_idx_0);
                    uint8x16_t value_hi_0 = vqtbl1q_u8(lut_hi_vec, value_idx_0);

                    int16x8_t sign_16_0a = vreinterpretq_s16_u8(vzip1q_u8(sign_lo_0, sign_hi_0));
                    int16x8_t sign_16_0b = vreinterpretq_s16_u8(vzip2q_u8(sign_lo_0, sign_hi_0));
                    int16x8_t value_16_0a = vreinterpretq_s16_u8(vzip1q_u8(value_lo_0, value_hi_0));
                    int16x8_t value_16_0b = vreinterpretq_s16_u8(vzip2q_u8(value_lo_0, value_hi_0));

                    int16x8_t res_0a = vsubq_s16(vshlq_n_s16(value_16_0a, 1), sign_16_0a);
                    int16x8_t res_0b = vsubq_s16(vshlq_n_s16(value_16_0b, 1), sign_16_0b);

                    // Second batch of 16
                    uint8x16_t sign_idx_1 = vandq_u8(vld1q_u8(sign_row + n + 16), mask_0f);
                    uint8x16_t value_idx_1 = vandq_u8(vld1q_u8(value_row + n + 16), mask_0f);

                    uint8x16_t sign_lo_1 = vqtbl1q_u8(lut_lo_vec, sign_idx_1);
                    uint8x16_t sign_hi_1 = vqtbl1q_u8(lut_hi_vec, sign_idx_1);
                    uint8x16_t value_lo_1 = vqtbl1q_u8(lut_lo_vec, value_idx_1);
                    uint8x16_t value_hi_1 = vqtbl1q_u8(lut_hi_vec, value_idx_1);

                    int16x8_t sign_16_1a = vreinterpretq_s16_u8(vzip1q_u8(sign_lo_1, sign_hi_1));
                    int16x8_t sign_16_1b = vreinterpretq_s16_u8(vzip2q_u8(sign_lo_1, sign_hi_1));
                    int16x8_t value_16_1a = vreinterpretq_s16_u8(vzip1q_u8(value_lo_1, value_hi_1));
                    int16x8_t value_16_1b = vreinterpretq_s16_u8(vzip2q_u8(value_lo_1, value_hi_1));

                    int16x8_t res_1a = vsubq_s16(vshlq_n_s16(value_16_1a, 1), sign_16_1a);
                    int16x8_t res_1b = vsubq_s16(vshlq_n_s16(value_16_1b, 1), sign_16_1b);

                    // Extend and accumulate all 32
                    int32x4_t acc0 = vaddq_s32(vld1q_s32(y_acc + n), vmovl_s16(vget_low_s16(res_0a)));
                    int32x4_t acc1 = vaddq_s32(vld1q_s32(y_acc + n + 4), vmovl_s16(vget_high_s16(res_0a)));
                    int32x4_t acc2 = vaddq_s32(vld1q_s32(y_acc + n + 8), vmovl_s16(vget_low_s16(res_0b)));
                    int32x4_t acc3 = vaddq_s32(vld1q_s32(y_acc + n + 12), vmovl_s16(vget_high_s16(res_0b)));
                    int32x4_t acc4 = vaddq_s32(vld1q_s32(y_acc + n + 16), vmovl_s16(vget_low_s16(res_1a)));
                    int32x4_t acc5 = vaddq_s32(vld1q_s32(y_acc + n + 20), vmovl_s16(vget_high_s16(res_1a)));
                    int32x4_t acc6 = vaddq_s32(vld1q_s32(y_acc + n + 24), vmovl_s16(vget_low_s16(res_1b)));
                    int32x4_t acc7 = vaddq_s32(vld1q_s32(y_acc + n + 28), vmovl_s16(vget_high_s16(res_1b)));

                    vst1q_s32(y_acc + n, acc0);
                    vst1q_s32(y_acc + n + 4, acc1);
                    vst1q_s32(y_acc + n + 8, acc2);
                    vst1q_s32(y_acc + n + 12, acc3);
                    vst1q_s32(y_acc + n + 16, acc4);
                    vst1q_s32(y_acc + n + 20, acc5);
                    vst1q_s32(y_acc + n + 24, acc6);
                    vst1q_s32(y_acc + n + 28, acc7);
                }

                // Handle 16 at a time
                for (; n + 15 < N; n += 16) {
                    uint8x16_t sign_idx = vandq_u8(vld1q_u8(sign_row + n), mask_0f);
                    uint8x16_t value_idx = vandq_u8(vld1q_u8(value_row + n), mask_0f);

                    uint8x16_t sign_lo = vqtbl1q_u8(lut_lo_vec, sign_idx);
                    uint8x16_t sign_hi = vqtbl1q_u8(lut_hi_vec, sign_idx);
                    uint8x16_t value_lo = vqtbl1q_u8(lut_lo_vec, value_idx);
                    uint8x16_t value_hi = vqtbl1q_u8(lut_hi_vec, value_idx);

                    int16x8_t sign_16_lo = vreinterpretq_s16_u8(vzip1q_u8(sign_lo, sign_hi));
                    int16x8_t sign_16_hi = vreinterpretq_s16_u8(vzip2q_u8(sign_lo, sign_hi));
                    int16x8_t value_16_lo = vreinterpretq_s16_u8(vzip1q_u8(value_lo, value_hi));
                    int16x8_t value_16_hi = vreinterpretq_s16_u8(vzip2q_u8(value_lo, value_hi));

                    int16x8_t res_16_lo = vsubq_s16(vshlq_n_s16(value_16_lo, 1), sign_16_lo);
                    int16x8_t res_16_hi = vsubq_s16(vshlq_n_s16(value_16_hi, 1), sign_16_hi);

                    int32x4_t acc0 = vaddq_s32(vld1q_s32(y_acc + n), vmovl_s16(vget_low_s16(res_16_lo)));
                    int32x4_t acc1 = vaddq_s32(vld1q_s32(y_acc + n + 4), vmovl_s16(vget_high_s16(res_16_lo)));
                    int32x4_t acc2 = vaddq_s32(vld1q_s32(y_acc + n + 8), vmovl_s16(vget_low_s16(res_16_hi)));
                    int32x4_t acc3 = vaddq_s32(vld1q_s32(y_acc + n + 12), vmovl_s16(vget_high_s16(res_16_hi)));

                    vst1q_s32(y_acc + n, acc0);
                    vst1q_s32(y_acc + n + 4, acc1);
                    vst1q_s32(y_acc + n + 8, acc2);
                    vst1q_s32(y_acc + n + 12, acc3);
                }

                // Scalar remainder
                alignas(32) int16_t lut[16];
                build_lut_16_int16(x_vals, lut);
                for (; n < N; n++) {
                    uint8_t s = sign_row[n] & 0x0F;
                    uint8_t v = value_row[n] & 0x0F;
                    y_acc[n] += 2 * static_cast<int32_t>(lut[v]) - static_cast<int32_t>(lut[s]);
                }
            }

            // Convert to float
            float* y_row = y + m * N;
            if (bias != nullptr) {
                for (int n = 0; n < N; n++) {
                    y_row[n] = static_cast<float>(y_acc[n]) * scale + bias[n];
                }
            } else {
                float32x4_t scale_vec = vdupq_n_f32(scale);
                int n = 0;
                for (; n + 3 < N; n += 4) {
                    int32x4_t acc = vld1q_s32(y_acc + n);
                    float32x4_t acc_f = vcvtq_f32_s32(acc);
                    vst1q_f32(y_row + n, vmulq_f32(acc_f, scale_vec));
                }
                for (; n < N; n++) {
                    y_row[n] = static_cast<float>(y_acc[n]) * scale;
                }
            }
        }
    }
}

/**
 * Scalar reference implementation for correctness testing
 */
void matmul_free_neon_tbl_scalar(
    torch::Tensor x_int8_tensor,
    torch::Tensor scale_tensor,
    torch::Tensor sign_plane_tensor,
    torch::Tensor value_plane_tensor,
    torch::Tensor y_tensor,
    torch::Tensor bias_tensor,
    int M, int N, int K,
    int num_threads
) {
    const int8_t* x_int8 = x_int8_tensor.data_ptr<int8_t>();
    const float* scales = scale_tensor.data_ptr<float>();
    const uint8_t* sign_plane = sign_plane_tensor.data_ptr<uint8_t>();
    const uint8_t* value_plane = value_plane_tensor.data_ptr<uint8_t>();
    float* y = y_tensor.data_ptr<float>();
    const float* bias = bias_tensor.defined() ? bias_tensor.data_ptr<float>() : nullptr;

    const int K_groups = (K + 3) / 4;

    omp_set_num_threads(num_threads);

    #pragma omp parallel for schedule(static)
    for (int m = 0; m < M; m++) {
        const int8_t* x_row = x_int8 + m * K;
        float scale = scales[m];
        float* y_row = y + m * N;

        // Accumulate in int32 to avoid overflow
        std::vector<int32_t> y_acc(N, 0);

        // K-first iteration
        for (int kg = 0; kg < K_groups; kg++) {
            const int k_base = kg * 4;

            // Build LUT
            int8_t x_vals[4] = {0, 0, 0, 0};
            for (int i = 0; i < 4 && k_base + i < K; i++) {
                x_vals[i] = x_row[k_base + i];
            }

            int16_t lut[16];
            build_lut_16_int16(x_vals, lut);

            const uint8_t* sign_row = sign_plane + kg * N;
            const uint8_t* value_row = value_plane + kg * N;

            for (int n = 0; n < N; n++) {
                uint8_t s = sign_row[n] & 0x0F;
                uint8_t v = value_row[n] & 0x0F;
                // True MatMul-free: result = 2*value_sum - sign_sum
                y_acc[n] += 2 * static_cast<int32_t>(lut[v]) - static_cast<int32_t>(lut[s]);
            }
        }

        // Convert to float
        for (int n = 0; n < N; n++) {
            y_row[n] = static_cast<float>(y_acc[n]) * scale + (bias ? bias[n] : 0.0f);
        }
    }
}

/**
 * Highly optimized TBL v3 with:
 * - K-loop unrolling (2 LUTs at a time)
 * - Register-based accumulators (avoid memory round-trips)
 * - Prefetching for weight planes
 * - 64-output N-blocking
 */
void matmul_free_neon_tbl_v3(
    torch::Tensor x_int8_tensor,       // [M, K] int8
    torch::Tensor scale_tensor,        // [M] float32
    torch::Tensor sign_plane_tensor,   // [K_groups, N] uint8
    torch::Tensor value_plane_tensor,  // [K_groups, N] uint8
    torch::Tensor y_tensor,            // [M, N] float32 output
    torch::Tensor bias_tensor,
    int M, int N, int K,
    int num_threads
) {
    const int8_t* __restrict__ x_int8 = x_int8_tensor.data_ptr<int8_t>();
    const float* __restrict__ scales = scale_tensor.data_ptr<float>();
    const uint8_t* __restrict__ sign_plane = sign_plane_tensor.data_ptr<uint8_t>();
    const uint8_t* __restrict__ value_plane = value_plane_tensor.data_ptr<uint8_t>();
    float* __restrict__ y = y_tensor.data_ptr<float>();
    const float* bias = bias_tensor.defined() ? bias_tensor.data_ptr<float>() : nullptr;

    const int K_groups = (K + 3) / 4;
    const int N_BLOCK = 32;  // Process 32 outputs with register accumulators

    omp_set_num_threads(num_threads);

    #pragma omp parallel
    {
        // Per-thread buffer for outputs that don't fit in registers
        alignas(64) int32_t y_acc[N];

        #pragma omp for schedule(static)
        for (int m = 0; m < M; m++) {
            const int8_t* __restrict__ x_row = x_int8 + m * K;
            float scale = scales[m];

            memset(y_acc, 0, N * sizeof(int32_t));

            uint8x16_t mask_0f = vdupq_n_u8(0x0F);

            // Process N in blocks of 32 with register accumulators
            for (int n_block = 0; n_block + N_BLOCK - 1 < N; n_block += N_BLOCK) {
                // Register accumulators for 32 outputs (8 x int32x4_t)
                int32x4_t acc0 = vdupq_n_s32(0);
                int32x4_t acc1 = vdupq_n_s32(0);
                int32x4_t acc2 = vdupq_n_s32(0);
                int32x4_t acc3 = vdupq_n_s32(0);
                int32x4_t acc4 = vdupq_n_s32(0);
                int32x4_t acc5 = vdupq_n_s32(0);
                int32x4_t acc6 = vdupq_n_s32(0);
                int32x4_t acc7 = vdupq_n_s32(0);

                // K-loop with 2x unrolling
                int kg = 0;
                for (; kg + 1 < K_groups; kg += 2) {
                    // Build 2 LUTs
                    int8_t x_vals0[4] = {0, 0, 0, 0};
                    int8_t x_vals1[4] = {0, 0, 0, 0};

                    int k_base0 = kg * 4;
                    int k_base1 = (kg + 1) * 4;

                    for (int i = 0; i < 4 && k_base0 + i < K; i++) {
                        x_vals0[i] = x_row[k_base0 + i];
                    }
                    for (int i = 0; i < 4 && k_base1 + i < K; i++) {
                        x_vals1[i] = x_row[k_base1 + i];
                    }

                    // Build split LUTs for both
                    alignas(16) uint8_t lut_lo0[16], lut_hi0[16];
                    alignas(16) uint8_t lut_lo1[16], lut_hi1[16];
                    build_lut_16_split(x_vals0, lut_lo0, lut_hi0);
                    build_lut_16_split(x_vals1, lut_lo1, lut_hi1);

                    uint8x16_t lut_lo_vec0 = vld1q_u8(lut_lo0);
                    uint8x16_t lut_hi_vec0 = vld1q_u8(lut_hi0);
                    uint8x16_t lut_lo_vec1 = vld1q_u8(lut_lo1);
                    uint8x16_t lut_hi_vec1 = vld1q_u8(lut_hi1);

                    // Prefetch next weight plane rows
                    if (kg + 2 < K_groups) {
                        __builtin_prefetch(sign_plane + (kg + 2) * N + n_block, 0, 3);
                        __builtin_prefetch(value_plane + (kg + 2) * N + n_block, 0, 3);
                    }

                    // Process first K group
                    const uint8_t* sign_row0 = sign_plane + kg * N + n_block;
                    const uint8_t* value_row0 = value_plane + kg * N + n_block;

                    // First 16 outputs
                    uint8x16_t sign_idx0 = vandq_u8(vld1q_u8(sign_row0), mask_0f);
                    uint8x16_t value_idx0 = vandq_u8(vld1q_u8(value_row0), mask_0f);

                    uint8x16_t sign_lo0 = vqtbl1q_u8(lut_lo_vec0, sign_idx0);
                    uint8x16_t sign_hi0 = vqtbl1q_u8(lut_hi_vec0, sign_idx0);
                    uint8x16_t value_lo0 = vqtbl1q_u8(lut_lo_vec0, value_idx0);
                    uint8x16_t value_hi0 = vqtbl1q_u8(lut_hi_vec0, value_idx0);

                    int16x8_t sign_16_0a = vreinterpretq_s16_u8(vzip1q_u8(sign_lo0, sign_hi0));
                    int16x8_t sign_16_0b = vreinterpretq_s16_u8(vzip2q_u8(sign_lo0, sign_hi0));
                    int16x8_t value_16_0a = vreinterpretq_s16_u8(vzip1q_u8(value_lo0, value_hi0));
                    int16x8_t value_16_0b = vreinterpretq_s16_u8(vzip2q_u8(value_lo0, value_hi0));

                    int16x8_t res_0a = vsubq_s16(vshlq_n_s16(value_16_0a, 1), sign_16_0a);
                    int16x8_t res_0b = vsubq_s16(vshlq_n_s16(value_16_0b, 1), sign_16_0b);

                    acc0 = vaddq_s32(acc0, vmovl_s16(vget_low_s16(res_0a)));
                    acc1 = vaddq_s32(acc1, vmovl_s16(vget_high_s16(res_0a)));
                    acc2 = vaddq_s32(acc2, vmovl_s16(vget_low_s16(res_0b)));
                    acc3 = vaddq_s32(acc3, vmovl_s16(vget_high_s16(res_0b)));

                    // Second 16 outputs for first K group
                    uint8x16_t sign_idx1 = vandq_u8(vld1q_u8(sign_row0 + 16), mask_0f);
                    uint8x16_t value_idx1 = vandq_u8(vld1q_u8(value_row0 + 16), mask_0f);

                    uint8x16_t sign_lo1 = vqtbl1q_u8(lut_lo_vec0, sign_idx1);
                    uint8x16_t sign_hi1 = vqtbl1q_u8(lut_hi_vec0, sign_idx1);
                    uint8x16_t value_lo1 = vqtbl1q_u8(lut_lo_vec0, value_idx1);
                    uint8x16_t value_hi1 = vqtbl1q_u8(lut_hi_vec0, value_idx1);

                    int16x8_t sign_16_1a = vreinterpretq_s16_u8(vzip1q_u8(sign_lo1, sign_hi1));
                    int16x8_t sign_16_1b = vreinterpretq_s16_u8(vzip2q_u8(sign_lo1, sign_hi1));
                    int16x8_t value_16_1a = vreinterpretq_s16_u8(vzip1q_u8(value_lo1, value_hi1));
                    int16x8_t value_16_1b = vreinterpretq_s16_u8(vzip2q_u8(value_lo1, value_hi1));

                    int16x8_t res_1a = vsubq_s16(vshlq_n_s16(value_16_1a, 1), sign_16_1a);
                    int16x8_t res_1b = vsubq_s16(vshlq_n_s16(value_16_1b, 1), sign_16_1b);

                    acc4 = vaddq_s32(acc4, vmovl_s16(vget_low_s16(res_1a)));
                    acc5 = vaddq_s32(acc5, vmovl_s16(vget_high_s16(res_1a)));
                    acc6 = vaddq_s32(acc6, vmovl_s16(vget_low_s16(res_1b)));
                    acc7 = vaddq_s32(acc7, vmovl_s16(vget_high_s16(res_1b)));

                    // Process second K group
                    const uint8_t* sign_row1 = sign_plane + (kg + 1) * N + n_block;
                    const uint8_t* value_row1 = value_plane + (kg + 1) * N + n_block;

                    // First 16 outputs for second K group
                    sign_idx0 = vandq_u8(vld1q_u8(sign_row1), mask_0f);
                    value_idx0 = vandq_u8(vld1q_u8(value_row1), mask_0f);

                    sign_lo0 = vqtbl1q_u8(lut_lo_vec1, sign_idx0);
                    sign_hi0 = vqtbl1q_u8(lut_hi_vec1, sign_idx0);
                    value_lo0 = vqtbl1q_u8(lut_lo_vec1, value_idx0);
                    value_hi0 = vqtbl1q_u8(lut_hi_vec1, value_idx0);

                    sign_16_0a = vreinterpretq_s16_u8(vzip1q_u8(sign_lo0, sign_hi0));
                    sign_16_0b = vreinterpretq_s16_u8(vzip2q_u8(sign_lo0, sign_hi0));
                    value_16_0a = vreinterpretq_s16_u8(vzip1q_u8(value_lo0, value_hi0));
                    value_16_0b = vreinterpretq_s16_u8(vzip2q_u8(value_lo0, value_hi0));

                    res_0a = vsubq_s16(vshlq_n_s16(value_16_0a, 1), sign_16_0a);
                    res_0b = vsubq_s16(vshlq_n_s16(value_16_0b, 1), sign_16_0b);

                    acc0 = vaddq_s32(acc0, vmovl_s16(vget_low_s16(res_0a)));
                    acc1 = vaddq_s32(acc1, vmovl_s16(vget_high_s16(res_0a)));
                    acc2 = vaddq_s32(acc2, vmovl_s16(vget_low_s16(res_0b)));
                    acc3 = vaddq_s32(acc3, vmovl_s16(vget_high_s16(res_0b)));

                    // Second 16 outputs for second K group
                    sign_idx1 = vandq_u8(vld1q_u8(sign_row1 + 16), mask_0f);
                    value_idx1 = vandq_u8(vld1q_u8(value_row1 + 16), mask_0f);

                    sign_lo1 = vqtbl1q_u8(lut_lo_vec1, sign_idx1);
                    sign_hi1 = vqtbl1q_u8(lut_hi_vec1, sign_idx1);
                    value_lo1 = vqtbl1q_u8(lut_lo_vec1, value_idx1);
                    value_hi1 = vqtbl1q_u8(lut_hi_vec1, value_idx1);

                    sign_16_1a = vreinterpretq_s16_u8(vzip1q_u8(sign_lo1, sign_hi1));
                    sign_16_1b = vreinterpretq_s16_u8(vzip2q_u8(sign_lo1, sign_hi1));
                    value_16_1a = vreinterpretq_s16_u8(vzip1q_u8(value_lo1, value_hi1));
                    value_16_1b = vreinterpretq_s16_u8(vzip2q_u8(value_lo1, value_hi1));

                    res_1a = vsubq_s16(vshlq_n_s16(value_16_1a, 1), sign_16_1a);
                    res_1b = vsubq_s16(vshlq_n_s16(value_16_1b, 1), sign_16_1b);

                    acc4 = vaddq_s32(acc4, vmovl_s16(vget_low_s16(res_1a)));
                    acc5 = vaddq_s32(acc5, vmovl_s16(vget_high_s16(res_1a)));
                    acc6 = vaddq_s32(acc6, vmovl_s16(vget_low_s16(res_1b)));
                    acc7 = vaddq_s32(acc7, vmovl_s16(vget_high_s16(res_1b)));
                }

                // Handle odd K group
                for (; kg < K_groups; kg++) {
                    int8_t x_vals[4] = {0, 0, 0, 0};
                    int k_base = kg * 4;
                    for (int i = 0; i < 4 && k_base + i < K; i++) {
                        x_vals[i] = x_row[k_base + i];
                    }

                    alignas(16) uint8_t lut_lo[16], lut_hi[16];
                    build_lut_16_split(x_vals, lut_lo, lut_hi);

                    uint8x16_t lut_lo_vec = vld1q_u8(lut_lo);
                    uint8x16_t lut_hi_vec = vld1q_u8(lut_hi);

                    const uint8_t* sign_row = sign_plane + kg * N + n_block;
                    const uint8_t* value_row = value_plane + kg * N + n_block;

                    // First 16
                    uint8x16_t sign_idx = vandq_u8(vld1q_u8(sign_row), mask_0f);
                    uint8x16_t value_idx = vandq_u8(vld1q_u8(value_row), mask_0f);

                    uint8x16_t sign_lo = vqtbl1q_u8(lut_lo_vec, sign_idx);
                    uint8x16_t sign_hi = vqtbl1q_u8(lut_hi_vec, sign_idx);
                    uint8x16_t value_lo = vqtbl1q_u8(lut_lo_vec, value_idx);
                    uint8x16_t value_hi = vqtbl1q_u8(lut_hi_vec, value_idx);

                    int16x8_t sign_16_a = vreinterpretq_s16_u8(vzip1q_u8(sign_lo, sign_hi));
                    int16x8_t sign_16_b = vreinterpretq_s16_u8(vzip2q_u8(sign_lo, sign_hi));
                    int16x8_t value_16_a = vreinterpretq_s16_u8(vzip1q_u8(value_lo, value_hi));
                    int16x8_t value_16_b = vreinterpretq_s16_u8(vzip2q_u8(value_lo, value_hi));

                    int16x8_t res_a = vsubq_s16(vshlq_n_s16(value_16_a, 1), sign_16_a);
                    int16x8_t res_b = vsubq_s16(vshlq_n_s16(value_16_b, 1), sign_16_b);

                    acc0 = vaddq_s32(acc0, vmovl_s16(vget_low_s16(res_a)));
                    acc1 = vaddq_s32(acc1, vmovl_s16(vget_high_s16(res_a)));
                    acc2 = vaddq_s32(acc2, vmovl_s16(vget_low_s16(res_b)));
                    acc3 = vaddq_s32(acc3, vmovl_s16(vget_high_s16(res_b)));

                    // Second 16
                    sign_idx = vandq_u8(vld1q_u8(sign_row + 16), mask_0f);
                    value_idx = vandq_u8(vld1q_u8(value_row + 16), mask_0f);

                    sign_lo = vqtbl1q_u8(lut_lo_vec, sign_idx);
                    sign_hi = vqtbl1q_u8(lut_hi_vec, sign_idx);
                    value_lo = vqtbl1q_u8(lut_lo_vec, value_idx);
                    value_hi = vqtbl1q_u8(lut_hi_vec, value_idx);

                    sign_16_a = vreinterpretq_s16_u8(vzip1q_u8(sign_lo, sign_hi));
                    sign_16_b = vreinterpretq_s16_u8(vzip2q_u8(sign_lo, sign_hi));
                    value_16_a = vreinterpretq_s16_u8(vzip1q_u8(value_lo, value_hi));
                    value_16_b = vreinterpretq_s16_u8(vzip2q_u8(value_lo, value_hi));

                    res_a = vsubq_s16(vshlq_n_s16(value_16_a, 1), sign_16_a);
                    res_b = vsubq_s16(vshlq_n_s16(value_16_b, 1), sign_16_b);

                    acc4 = vaddq_s32(acc4, vmovl_s16(vget_low_s16(res_a)));
                    acc5 = vaddq_s32(acc5, vmovl_s16(vget_high_s16(res_a)));
                    acc6 = vaddq_s32(acc6, vmovl_s16(vget_low_s16(res_b)));
                    acc7 = vaddq_s32(acc7, vmovl_s16(vget_high_s16(res_b)));
                }

                // Store accumulators to output buffer
                vst1q_s32(y_acc + n_block, acc0);
                vst1q_s32(y_acc + n_block + 4, acc1);
                vst1q_s32(y_acc + n_block + 8, acc2);
                vst1q_s32(y_acc + n_block + 12, acc3);
                vst1q_s32(y_acc + n_block + 16, acc4);
                vst1q_s32(y_acc + n_block + 20, acc5);
                vst1q_s32(y_acc + n_block + 24, acc6);
                vst1q_s32(y_acc + n_block + 28, acc7);
            }

            // Handle remaining N with the v2 approach
            int n_remaining = N - (N / N_BLOCK) * N_BLOCK;
            if (n_remaining > 0) {
                int n_start = (N / N_BLOCK) * N_BLOCK;

                for (int kg = 0; kg < K_groups; kg++) {
                    int8_t x_vals[4] = {0, 0, 0, 0};
                    int k_base = kg * 4;
                    for (int i = 0; i < 4 && k_base + i < K; i++) {
                        x_vals[i] = x_row[k_base + i];
                    }

                    alignas(32) int16_t lut[16];
                    build_lut_16_int16(x_vals, lut);

                    const uint8_t* sign_row = sign_plane + kg * N;
                    const uint8_t* value_row = value_plane + kg * N;

                    for (int n = n_start; n < N; n++) {
                        uint8_t s = sign_row[n] & 0x0F;
                        uint8_t v = value_row[n] & 0x0F;
                        y_acc[n] += 2 * static_cast<int32_t>(lut[v]) - static_cast<int32_t>(lut[s]);
                    }
                }
            }

            // Convert to float
            float* y_row = y + m * N;
            if (bias != nullptr) {
                for (int n = 0; n < N; n++) {
                    y_row[n] = static_cast<float>(y_acc[n]) * scale + bias[n];
                }
            } else {
                float32x4_t scale_vec = vdupq_n_f32(scale);
                int n = 0;
                for (; n + 3 < N; n += 4) {
                    int32x4_t acc = vld1q_s32(y_acc + n);
                    float32x4_t acc_f = vcvtq_f32_s32(acc);
                    vst1q_f32(y_row + n, vmulq_f32(acc_f, scale_vec));
                }
                for (; n < N; n++) {
                    y_row[n] = static_cast<float>(y_acc[n]) * scale;
                }
            }
        }
    }
}

/**
 * TBL v4: Cache-aware tiled version
 *
 * Uses M/N tiling to improve cache locality, similar to SDOT v4.
 * Processes tiles that fit in L2 cache.
 */
void matmul_free_neon_tbl_v4(
    torch::Tensor x_int8_tensor,       // [M, K] int8
    torch::Tensor scale_tensor,        // [M] float32
    torch::Tensor sign_plane_tensor,   // [K_groups, N] uint8
    torch::Tensor value_plane_tensor,  // [K_groups, N] uint8
    torch::Tensor y_tensor,            // [M, N] float32 output
    torch::Tensor bias_tensor,
    int M, int N, int K,
    int num_threads
) {
    const int8_t* __restrict__ x_int8 = x_int8_tensor.data_ptr<int8_t>();
    const float* __restrict__ scales = scale_tensor.data_ptr<float>();
    const uint8_t* __restrict__ sign_plane = sign_plane_tensor.data_ptr<uint8_t>();
    const uint8_t* __restrict__ value_plane = value_plane_tensor.data_ptr<uint8_t>();
    float* __restrict__ y = y_tensor.data_ptr<float>();
    const float* bias = bias_tensor.defined() ? bias_tensor.data_ptr<float>() : nullptr;

    const int K_groups = (K + 3) / 4;

    // Tile sizes for L2 cache (Apple M1/M2 has ~12MB L2)
    const int M_TILE = 64;
    const int N_TILE = 128;

    omp_set_num_threads(num_threads);

    // Parallelize over M tiles
    #pragma omp parallel for schedule(dynamic)
    for (int m_tile = 0; m_tile < M; m_tile += M_TILE) {
        const int m_end = std::min(m_tile + M_TILE, M);

        // Per-tile accumulator (thread-local due to omp parallel for)
        alignas(64) int32_t y_acc[N_TILE];

        for (int n_tile = 0; n_tile < N; n_tile += N_TILE) {
            const int n_end = std::min(n_tile + N_TILE, N);
            const int n_size = n_end - n_tile;

            for (int m = m_tile; m < m_end; m++) {
                const int8_t* __restrict__ x_row = x_int8 + m * K;
                float scale = scales[m];

                memset(y_acc, 0, n_size * sizeof(int32_t));

                uint8x16_t mask_0f = vdupq_n_u8(0x0F);

                // K-first iteration with 2x unrolling
                int kg = 0;
                for (; kg + 1 < K_groups; kg += 2) {
                    // Build 2 LUTs
                    int8_t x_vals0[4] = {0, 0, 0, 0};
                    int8_t x_vals1[4] = {0, 0, 0, 0};

                    int k_base0 = kg * 4;
                    int k_base1 = (kg + 1) * 4;

                    for (int i = 0; i < 4 && k_base0 + i < K; i++) {
                        x_vals0[i] = x_row[k_base0 + i];
                    }
                    for (int i = 0; i < 4 && k_base1 + i < K; i++) {
                        x_vals1[i] = x_row[k_base1 + i];
                    }

                    alignas(16) uint8_t lut_lo0[16], lut_hi0[16];
                    alignas(16) uint8_t lut_lo1[16], lut_hi1[16];
                    build_lut_16_split(x_vals0, lut_lo0, lut_hi0);
                    build_lut_16_split(x_vals1, lut_lo1, lut_hi1);

                    uint8x16_t lut_lo_vec0 = vld1q_u8(lut_lo0);
                    uint8x16_t lut_hi_vec0 = vld1q_u8(lut_hi0);
                    uint8x16_t lut_lo_vec1 = vld1q_u8(lut_lo1);
                    uint8x16_t lut_hi_vec1 = vld1q_u8(lut_hi1);

                    const uint8_t* sign_row0 = sign_plane + kg * N + n_tile;
                    const uint8_t* value_row0 = value_plane + kg * N + n_tile;
                    const uint8_t* sign_row1 = sign_plane + (kg + 1) * N + n_tile;
                    const uint8_t* value_row1 = value_plane + (kg + 1) * N + n_tile;

                    int n = 0;
                    for (; n + 31 < n_size; n += 32) {
                        // Prefetch next K rows
                        if (kg + 2 < K_groups) {
                            __builtin_prefetch(sign_plane + (kg + 2) * N + n_tile + n, 0, 3);
                            __builtin_prefetch(value_plane + (kg + 2) * N + n_tile + n, 0, 3);
                        }

                        // First K group, first 16
                        uint8x16_t sign_idx = vandq_u8(vld1q_u8(sign_row0 + n), mask_0f);
                        uint8x16_t value_idx = vandq_u8(vld1q_u8(value_row0 + n), mask_0f);

                        uint8x16_t sign_lo = vqtbl1q_u8(lut_lo_vec0, sign_idx);
                        uint8x16_t sign_hi = vqtbl1q_u8(lut_hi_vec0, sign_idx);
                        uint8x16_t value_lo = vqtbl1q_u8(lut_lo_vec0, value_idx);
                        uint8x16_t value_hi = vqtbl1q_u8(lut_hi_vec0, value_idx);

                        int16x8_t sign_16_a = vreinterpretq_s16_u8(vzip1q_u8(sign_lo, sign_hi));
                        int16x8_t sign_16_b = vreinterpretq_s16_u8(vzip2q_u8(sign_lo, sign_hi));
                        int16x8_t value_16_a = vreinterpretq_s16_u8(vzip1q_u8(value_lo, value_hi));
                        int16x8_t value_16_b = vreinterpretq_s16_u8(vzip2q_u8(value_lo, value_hi));

                        int16x8_t res_a = vsubq_s16(vshlq_n_s16(value_16_a, 1), sign_16_a);
                        int16x8_t res_b = vsubq_s16(vshlq_n_s16(value_16_b, 1), sign_16_b);

                        int32x4_t acc0 = vaddq_s32(vld1q_s32(y_acc + n), vmovl_s16(vget_low_s16(res_a)));
                        int32x4_t acc1 = vaddq_s32(vld1q_s32(y_acc + n + 4), vmovl_s16(vget_high_s16(res_a)));
                        int32x4_t acc2 = vaddq_s32(vld1q_s32(y_acc + n + 8), vmovl_s16(vget_low_s16(res_b)));
                        int32x4_t acc3 = vaddq_s32(vld1q_s32(y_acc + n + 12), vmovl_s16(vget_high_s16(res_b)));

                        // First K group, second 16
                        sign_idx = vandq_u8(vld1q_u8(sign_row0 + n + 16), mask_0f);
                        value_idx = vandq_u8(vld1q_u8(value_row0 + n + 16), mask_0f);

                        sign_lo = vqtbl1q_u8(lut_lo_vec0, sign_idx);
                        sign_hi = vqtbl1q_u8(lut_hi_vec0, sign_idx);
                        value_lo = vqtbl1q_u8(lut_lo_vec0, value_idx);
                        value_hi = vqtbl1q_u8(lut_hi_vec0, value_idx);

                        sign_16_a = vreinterpretq_s16_u8(vzip1q_u8(sign_lo, sign_hi));
                        sign_16_b = vreinterpretq_s16_u8(vzip2q_u8(sign_lo, sign_hi));
                        value_16_a = vreinterpretq_s16_u8(vzip1q_u8(value_lo, value_hi));
                        value_16_b = vreinterpretq_s16_u8(vzip2q_u8(value_lo, value_hi));

                        res_a = vsubq_s16(vshlq_n_s16(value_16_a, 1), sign_16_a);
                        res_b = vsubq_s16(vshlq_n_s16(value_16_b, 1), sign_16_b);

                        int32x4_t acc4 = vaddq_s32(vld1q_s32(y_acc + n + 16), vmovl_s16(vget_low_s16(res_a)));
                        int32x4_t acc5 = vaddq_s32(vld1q_s32(y_acc + n + 20), vmovl_s16(vget_high_s16(res_a)));
                        int32x4_t acc6 = vaddq_s32(vld1q_s32(y_acc + n + 24), vmovl_s16(vget_low_s16(res_b)));
                        int32x4_t acc7 = vaddq_s32(vld1q_s32(y_acc + n + 28), vmovl_s16(vget_high_s16(res_b)));

                        // Second K group, first 16
                        sign_idx = vandq_u8(vld1q_u8(sign_row1 + n), mask_0f);
                        value_idx = vandq_u8(vld1q_u8(value_row1 + n), mask_0f);

                        sign_lo = vqtbl1q_u8(lut_lo_vec1, sign_idx);
                        sign_hi = vqtbl1q_u8(lut_hi_vec1, sign_idx);
                        value_lo = vqtbl1q_u8(lut_lo_vec1, value_idx);
                        value_hi = vqtbl1q_u8(lut_hi_vec1, value_idx);

                        sign_16_a = vreinterpretq_s16_u8(vzip1q_u8(sign_lo, sign_hi));
                        sign_16_b = vreinterpretq_s16_u8(vzip2q_u8(sign_lo, sign_hi));
                        value_16_a = vreinterpretq_s16_u8(vzip1q_u8(value_lo, value_hi));
                        value_16_b = vreinterpretq_s16_u8(vzip2q_u8(value_lo, value_hi));

                        res_a = vsubq_s16(vshlq_n_s16(value_16_a, 1), sign_16_a);
                        res_b = vsubq_s16(vshlq_n_s16(value_16_b, 1), sign_16_b);

                        acc0 = vaddq_s32(acc0, vmovl_s16(vget_low_s16(res_a)));
                        acc1 = vaddq_s32(acc1, vmovl_s16(vget_high_s16(res_a)));
                        acc2 = vaddq_s32(acc2, vmovl_s16(vget_low_s16(res_b)));
                        acc3 = vaddq_s32(acc3, vmovl_s16(vget_high_s16(res_b)));

                        // Second K group, second 16
                        sign_idx = vandq_u8(vld1q_u8(sign_row1 + n + 16), mask_0f);
                        value_idx = vandq_u8(vld1q_u8(value_row1 + n + 16), mask_0f);

                        sign_lo = vqtbl1q_u8(lut_lo_vec1, sign_idx);
                        sign_hi = vqtbl1q_u8(lut_hi_vec1, sign_idx);
                        value_lo = vqtbl1q_u8(lut_lo_vec1, value_idx);
                        value_hi = vqtbl1q_u8(lut_hi_vec1, value_idx);

                        sign_16_a = vreinterpretq_s16_u8(vzip1q_u8(sign_lo, sign_hi));
                        sign_16_b = vreinterpretq_s16_u8(vzip2q_u8(sign_lo, sign_hi));
                        value_16_a = vreinterpretq_s16_u8(vzip1q_u8(value_lo, value_hi));
                        value_16_b = vreinterpretq_s16_u8(vzip2q_u8(value_lo, value_hi));

                        res_a = vsubq_s16(vshlq_n_s16(value_16_a, 1), sign_16_a);
                        res_b = vsubq_s16(vshlq_n_s16(value_16_b, 1), sign_16_b);

                        acc4 = vaddq_s32(acc4, vmovl_s16(vget_low_s16(res_a)));
                        acc5 = vaddq_s32(acc5, vmovl_s16(vget_high_s16(res_a)));
                        acc6 = vaddq_s32(acc6, vmovl_s16(vget_low_s16(res_b)));
                        acc7 = vaddq_s32(acc7, vmovl_s16(vget_high_s16(res_b)));

                        // Store
                        vst1q_s32(y_acc + n, acc0);
                        vst1q_s32(y_acc + n + 4, acc1);
                        vst1q_s32(y_acc + n + 8, acc2);
                        vst1q_s32(y_acc + n + 12, acc3);
                        vst1q_s32(y_acc + n + 16, acc4);
                        vst1q_s32(y_acc + n + 20, acc5);
                        vst1q_s32(y_acc + n + 24, acc6);
                        vst1q_s32(y_acc + n + 28, acc7);
                    }

                    // Handle remaining N with 16-element blocks
                    for (; n + 15 < n_size; n += 16) {
                        // First K group
                        uint8x16_t sign_idx = vandq_u8(vld1q_u8(sign_row0 + n), mask_0f);
                        uint8x16_t value_idx = vandq_u8(vld1q_u8(value_row0 + n), mask_0f);

                        uint8x16_t sign_lo = vqtbl1q_u8(lut_lo_vec0, sign_idx);
                        uint8x16_t sign_hi = vqtbl1q_u8(lut_hi_vec0, sign_idx);
                        uint8x16_t value_lo = vqtbl1q_u8(lut_lo_vec0, value_idx);
                        uint8x16_t value_hi = vqtbl1q_u8(lut_hi_vec0, value_idx);

                        int16x8_t sign_16_a = vreinterpretq_s16_u8(vzip1q_u8(sign_lo, sign_hi));
                        int16x8_t sign_16_b = vreinterpretq_s16_u8(vzip2q_u8(sign_lo, sign_hi));
                        int16x8_t value_16_a = vreinterpretq_s16_u8(vzip1q_u8(value_lo, value_hi));
                        int16x8_t value_16_b = vreinterpretq_s16_u8(vzip2q_u8(value_lo, value_hi));

                        int16x8_t res_a = vsubq_s16(vshlq_n_s16(value_16_a, 1), sign_16_a);
                        int16x8_t res_b = vsubq_s16(vshlq_n_s16(value_16_b, 1), sign_16_b);

                        int32x4_t acc0 = vaddq_s32(vld1q_s32(y_acc + n), vmovl_s16(vget_low_s16(res_a)));
                        int32x4_t acc1 = vaddq_s32(vld1q_s32(y_acc + n + 4), vmovl_s16(vget_high_s16(res_a)));
                        int32x4_t acc2 = vaddq_s32(vld1q_s32(y_acc + n + 8), vmovl_s16(vget_low_s16(res_b)));
                        int32x4_t acc3 = vaddq_s32(vld1q_s32(y_acc + n + 12), vmovl_s16(vget_high_s16(res_b)));

                        // Second K group
                        sign_idx = vandq_u8(vld1q_u8(sign_row1 + n), mask_0f);
                        value_idx = vandq_u8(vld1q_u8(value_row1 + n), mask_0f);

                        sign_lo = vqtbl1q_u8(lut_lo_vec1, sign_idx);
                        sign_hi = vqtbl1q_u8(lut_hi_vec1, sign_idx);
                        value_lo = vqtbl1q_u8(lut_lo_vec1, value_idx);
                        value_hi = vqtbl1q_u8(lut_hi_vec1, value_idx);

                        sign_16_a = vreinterpretq_s16_u8(vzip1q_u8(sign_lo, sign_hi));
                        sign_16_b = vreinterpretq_s16_u8(vzip2q_u8(sign_lo, sign_hi));
                        value_16_a = vreinterpretq_s16_u8(vzip1q_u8(value_lo, value_hi));
                        value_16_b = vreinterpretq_s16_u8(vzip2q_u8(value_lo, value_hi));

                        res_a = vsubq_s16(vshlq_n_s16(value_16_a, 1), sign_16_a);
                        res_b = vsubq_s16(vshlq_n_s16(value_16_b, 1), sign_16_b);

                        acc0 = vaddq_s32(acc0, vmovl_s16(vget_low_s16(res_a)));
                        acc1 = vaddq_s32(acc1, vmovl_s16(vget_high_s16(res_a)));
                        acc2 = vaddq_s32(acc2, vmovl_s16(vget_low_s16(res_b)));
                        acc3 = vaddq_s32(acc3, vmovl_s16(vget_high_s16(res_b)));

                        vst1q_s32(y_acc + n, acc0);
                        vst1q_s32(y_acc + n + 4, acc1);
                        vst1q_s32(y_acc + n + 8, acc2);
                        vst1q_s32(y_acc + n + 12, acc3);
                    }

                    // Scalar remainder
                    alignas(32) int16_t lut0[16], lut1[16];
                    build_lut_16_int16(x_vals0, lut0);
                    build_lut_16_int16(x_vals1, lut1);
                    for (; n < n_size; n++) {
                        uint8_t s0 = sign_row0[n] & 0x0F;
                        uint8_t v0 = value_row0[n] & 0x0F;
                        uint8_t s1 = sign_row1[n] & 0x0F;
                        uint8_t v1 = value_row1[n] & 0x0F;
                        y_acc[n] += 2 * static_cast<int32_t>(lut0[v0]) - static_cast<int32_t>(lut0[s0]);
                        y_acc[n] += 2 * static_cast<int32_t>(lut1[v1]) - static_cast<int32_t>(lut1[s1]);
                    }
                }

                // Handle remaining K groups
                for (; kg < K_groups; kg++) {
                    int8_t x_vals[4] = {0, 0, 0, 0};
                    int k_base = kg * 4;
                    for (int i = 0; i < 4 && k_base + i < K; i++) {
                        x_vals[i] = x_row[k_base + i];
                    }

                    alignas(32) int16_t lut[16];
                    build_lut_16_int16(x_vals, lut);

                    const uint8_t* sign_row = sign_plane + kg * N + n_tile;
                    const uint8_t* value_row = value_plane + kg * N + n_tile;

                    for (int n = 0; n < n_size; n++) {
                        uint8_t s = sign_row[n] & 0x0F;
                        uint8_t v = value_row[n] & 0x0F;
                        y_acc[n] += 2 * static_cast<int32_t>(lut[v]) - static_cast<int32_t>(lut[s]);
                    }
                }

                // Convert to float output
                float* y_row = y + m * N + n_tile;
                float32x4_t scale_vec = vdupq_n_f32(scale);

                if (bias != nullptr) {
                    const float* bias_row = bias + n_tile;
                    int n = 0;
                    for (; n + 3 < n_size; n += 4) {
                        int32x4_t acc = vld1q_s32(y_acc + n);
                        float32x4_t acc_f = vcvtq_f32_s32(acc);
                        float32x4_t bias_v = vld1q_f32(bias_row + n);
                        vst1q_f32(y_row + n, vaddq_f32(vmulq_f32(acc_f, scale_vec), bias_v));
                    }
                    for (; n < n_size; n++) {
                        y_row[n] = static_cast<float>(y_acc[n]) * scale + bias_row[n];
                    }
                } else {
                    int n = 0;
                    for (; n + 3 < n_size; n += 4) {
                        int32x4_t acc = vld1q_s32(y_acc + n);
                        float32x4_t acc_f = vcvtq_f32_s32(acc);
                        vst1q_f32(y_row + n, vmulq_f32(acc_f, scale_vec));
                    }
                    for (; n < n_size; n++) {
                        y_row[n] = static_cast<float>(y_acc[n]) * scale;
                    }
                }
            }
        }
    }
}

/**
 * TBL v5: Optimized with vectorized LUT construction
 *
 * Uses NEON-accelerated LUT building and 64-way N-blocking.
 * Focuses on minimizing overhead while maintaining correctness.
 */
void matmul_free_neon_tbl_v5(
    torch::Tensor x_int8_tensor,       // [M, K] int8
    torch::Tensor scale_tensor,        // [M] float32
    torch::Tensor sign_plane_tensor,   // [K_groups, N] uint8
    torch::Tensor value_plane_tensor,  // [K_groups, N] uint8
    torch::Tensor y_tensor,            // [M, N] float32 output
    torch::Tensor bias_tensor,
    int M, int N, int K,
    int num_threads
) {
    const int8_t* __restrict__ x_int8 = x_int8_tensor.data_ptr<int8_t>();
    const float* __restrict__ scales = scale_tensor.data_ptr<float>();
    const uint8_t* __restrict__ sign_plane = sign_plane_tensor.data_ptr<uint8_t>();
    const uint8_t* __restrict__ value_plane = value_plane_tensor.data_ptr<uint8_t>();
    float* __restrict__ y = y_tensor.data_ptr<float>();
    const float* bias = bias_tensor.defined() ? bias_tensor.data_ptr<float>() : nullptr;

    const int K_groups = (K + 3) / 4;

    omp_set_num_threads(num_threads);

    #pragma omp parallel
    {
        alignas(64) int32_t y_acc[N];

        #pragma omp for schedule(static)
        for (int m = 0; m < M; m++) {
            const int8_t* __restrict__ x_row = x_int8 + m * K;
            float scale = scales[m];

            memset(y_acc, 0, N * sizeof(int32_t));

            uint8x16_t mask_0f = vdupq_n_u8(0x0F);

            // K-first iteration with vectorized LUT construction
            for (int kg = 0; kg < K_groups; kg++) {
                const int k_base = kg * 4;

                // Get 4 int8 activations
                int8_t x_vals[4] = {0, 0, 0, 0};
                for (int i = 0; i < 4 && k_base + i < K; i++) {
                    x_vals[i] = x_row[k_base + i];
                }

                // Build LUT using vectorized function (returns NEON registers directly)
                uint8x16_t lut_lo_vec, lut_hi_vec;
                build_lut_16_split_neon(x_vals, lut_lo_vec, lut_hi_vec);

                const uint8_t* __restrict__ sign_row = sign_plane + kg * N;
                const uint8_t* __restrict__ value_row = value_plane + kg * N;

                // Process 64 outputs at a time (4 x 16)
                int n = 0;
                for (; n + 63 < N; n += 64) {
                    // Prefetch next K group's data
                    if (kg + 1 < K_groups) {
                        __builtin_prefetch(sign_plane + (kg + 1) * N + n, 0, 3);
                        __builtin_prefetch(value_plane + (kg + 1) * N + n, 0, 3);
                    }

                    // Process 4 batches of 16 outputs each
                    #pragma unroll 4
                    for (int batch = 0; batch < 4; batch++) {
                        const int offset = n + batch * 16;

                        uint8x16_t sign_idx = vandq_u8(vld1q_u8(sign_row + offset), mask_0f);
                        uint8x16_t value_idx = vandq_u8(vld1q_u8(value_row + offset), mask_0f);

                        uint8x16_t sign_lo = vqtbl1q_u8(lut_lo_vec, sign_idx);
                        uint8x16_t sign_hi = vqtbl1q_u8(lut_hi_vec, sign_idx);
                        uint8x16_t value_lo = vqtbl1q_u8(lut_lo_vec, value_idx);
                        uint8x16_t value_hi = vqtbl1q_u8(lut_hi_vec, value_idx);

                        int16x8_t sign_16_a = vreinterpretq_s16_u8(vzip1q_u8(sign_lo, sign_hi));
                        int16x8_t sign_16_b = vreinterpretq_s16_u8(vzip2q_u8(sign_lo, sign_hi));
                        int16x8_t value_16_a = vreinterpretq_s16_u8(vzip1q_u8(value_lo, value_hi));
                        int16x8_t value_16_b = vreinterpretq_s16_u8(vzip2q_u8(value_lo, value_hi));

                        int16x8_t res_a = vsubq_s16(vshlq_n_s16(value_16_a, 1), sign_16_a);
                        int16x8_t res_b = vsubq_s16(vshlq_n_s16(value_16_b, 1), sign_16_b);

                        int32x4_t acc0 = vaddq_s32(vld1q_s32(y_acc + offset), vmovl_s16(vget_low_s16(res_a)));
                        int32x4_t acc1 = vaddq_s32(vld1q_s32(y_acc + offset + 4), vmovl_s16(vget_high_s16(res_a)));
                        int32x4_t acc2 = vaddq_s32(vld1q_s32(y_acc + offset + 8), vmovl_s16(vget_low_s16(res_b)));
                        int32x4_t acc3 = vaddq_s32(vld1q_s32(y_acc + offset + 12), vmovl_s16(vget_high_s16(res_b)));

                        vst1q_s32(y_acc + offset, acc0);
                        vst1q_s32(y_acc + offset + 4, acc1);
                        vst1q_s32(y_acc + offset + 8, acc2);
                        vst1q_s32(y_acc + offset + 12, acc3);
                    }
                }

                // Handle remaining 16-output blocks
                for (; n + 15 < N; n += 16) {
                    uint8x16_t sign_idx = vandq_u8(vld1q_u8(sign_row + n), mask_0f);
                    uint8x16_t value_idx = vandq_u8(vld1q_u8(value_row + n), mask_0f);

                    uint8x16_t sign_lo = vqtbl1q_u8(lut_lo_vec, sign_idx);
                    uint8x16_t sign_hi = vqtbl1q_u8(lut_hi_vec, sign_idx);
                    uint8x16_t value_lo = vqtbl1q_u8(lut_lo_vec, value_idx);
                    uint8x16_t value_hi = vqtbl1q_u8(lut_hi_vec, value_idx);

                    int16x8_t sign_16_a = vreinterpretq_s16_u8(vzip1q_u8(sign_lo, sign_hi));
                    int16x8_t sign_16_b = vreinterpretq_s16_u8(vzip2q_u8(sign_lo, sign_hi));
                    int16x8_t value_16_a = vreinterpretq_s16_u8(vzip1q_u8(value_lo, value_hi));
                    int16x8_t value_16_b = vreinterpretq_s16_u8(vzip2q_u8(value_lo, value_hi));

                    int16x8_t res_a = vsubq_s16(vshlq_n_s16(value_16_a, 1), sign_16_a);
                    int16x8_t res_b = vsubq_s16(vshlq_n_s16(value_16_b, 1), sign_16_b);

                    int32x4_t acc0 = vaddq_s32(vld1q_s32(y_acc + n), vmovl_s16(vget_low_s16(res_a)));
                    int32x4_t acc1 = vaddq_s32(vld1q_s32(y_acc + n + 4), vmovl_s16(vget_high_s16(res_a)));
                    int32x4_t acc2 = vaddq_s32(vld1q_s32(y_acc + n + 8), vmovl_s16(vget_low_s16(res_b)));
                    int32x4_t acc3 = vaddq_s32(vld1q_s32(y_acc + n + 12), vmovl_s16(vget_high_s16(res_b)));

                    vst1q_s32(y_acc + n, acc0);
                    vst1q_s32(y_acc + n + 4, acc1);
                    vst1q_s32(y_acc + n + 8, acc2);
                    vst1q_s32(y_acc + n + 12, acc3);
                }

                // Scalar remainder
                alignas(32) int16_t lut[16];
                build_lut_16_int16(x_vals, lut);
                for (; n < N; n++) {
                    uint8_t s = sign_row[n] & 0x0F;
                    uint8_t v = value_row[n] & 0x0F;
                    y_acc[n] += 2 * static_cast<int32_t>(lut[v]) - static_cast<int32_t>(lut[s]);
                }
            }

            // Convert to float
            float* y_row = y + m * N;
            if (bias != nullptr) {
                for (int n = 0; n < N; n++) {
                    y_row[n] = static_cast<float>(y_acc[n]) * scale + bias[n];
                }
            } else {
                float32x4_t scale_vec = vdupq_n_f32(scale);
                int n = 0;
                for (; n + 3 < N; n += 4) {
                    int32x4_t acc = vld1q_s32(y_acc + n);
                    float32x4_t acc_f = vcvtq_f32_s32(acc);
                    vst1q_f32(y_row + n, vmulq_f32(acc_f, scale_vec));
                }
                for (; n < N; n++) {
                    y_row[n] = static_cast<float>(y_acc[n]) * scale;
                }
            }
        }
    }
}

/**
 * TBL v6: M-batched to amortize weight loading
 *
 * Key insight: Load weight indices once, use for multiple M rows.
 * Each M row has a different LUT but same weight indices.
 */
void matmul_free_neon_tbl_v6(
    torch::Tensor x_int8_tensor,       // [M, K] int8
    torch::Tensor scale_tensor,        // [M] float32
    torch::Tensor sign_plane_tensor,   // [K_groups, N] uint8
    torch::Tensor value_plane_tensor,  // [K_groups, N] uint8
    torch::Tensor y_tensor,            // [M, N] float32 output
    torch::Tensor bias_tensor,
    int M, int N, int K,
    int num_threads
) {
    const int8_t* __restrict__ x_int8 = x_int8_tensor.data_ptr<int8_t>();
    const float* __restrict__ scales = scale_tensor.data_ptr<float>();
    const uint8_t* __restrict__ sign_plane = sign_plane_tensor.data_ptr<uint8_t>();
    const uint8_t* __restrict__ value_plane = value_plane_tensor.data_ptr<uint8_t>();
    float* __restrict__ y = y_tensor.data_ptr<float>();
    const float* bias = bias_tensor.defined() ? bias_tensor.data_ptr<float>() : nullptr;

    const int K_groups = (K + 3) / 4;
    const int M_BATCH = 4;  // Process 4 M rows together

    omp_set_num_threads(num_threads);

    #pragma omp parallel
    {
        // Accumulators for M_BATCH rows
        alignas(64) int32_t y_acc[M_BATCH][2048];  // Max N = 2048

        #pragma omp for schedule(static)
        for (int m_base = 0; m_base < M; m_base += M_BATCH) {
            const int m_end = std::min(m_base + M_BATCH, M);
            const int m_count = m_end - m_base;

            // Initialize accumulators for this batch
            for (int mi = 0; mi < m_count; mi++) {
                memset(y_acc[mi], 0, N * sizeof(int32_t));
            }

            uint8x16_t mask_0f = vdupq_n_u8(0x0F);

            // K-first iteration
            for (int kg = 0; kg < K_groups; kg++) {
                const int k_base = kg * 4;

                // Build LUTs for all M rows in this batch
                uint8x16_t lut_lo[M_BATCH], lut_hi[M_BATCH];

                for (int mi = 0; mi < m_count; mi++) {
                    const int8_t* x_row = x_int8 + (m_base + mi) * K;
                    int8_t x_vals[4] = {0, 0, 0, 0};
                    for (int i = 0; i < 4 && k_base + i < K; i++) {
                        x_vals[i] = x_row[k_base + i];
                    }
                    build_lut_16_split_neon(x_vals, lut_lo[mi], lut_hi[mi]);
                }

                const uint8_t* __restrict__ sign_row = sign_plane + kg * N;
                const uint8_t* __restrict__ value_row = value_plane + kg * N;

                // Process N in chunks of 16
                int n = 0;
                for (; n + 15 < N; n += 16) {
                    // Load weight indices ONCE for all M rows
                    uint8x16_t sign_idx = vandq_u8(vld1q_u8(sign_row + n), mask_0f);
                    uint8x16_t value_idx = vandq_u8(vld1q_u8(value_row + n), mask_0f);

                    // Process each M row with these same indices
                    for (int mi = 0; mi < m_count; mi++) {
                        uint8x16_t sign_lo = vqtbl1q_u8(lut_lo[mi], sign_idx);
                        uint8x16_t sign_hi = vqtbl1q_u8(lut_hi[mi], sign_idx);
                        uint8x16_t value_lo = vqtbl1q_u8(lut_lo[mi], value_idx);
                        uint8x16_t value_hi = vqtbl1q_u8(lut_hi[mi], value_idx);

                        int16x8_t sign_16_a = vreinterpretq_s16_u8(vzip1q_u8(sign_lo, sign_hi));
                        int16x8_t sign_16_b = vreinterpretq_s16_u8(vzip2q_u8(sign_lo, sign_hi));
                        int16x8_t value_16_a = vreinterpretq_s16_u8(vzip1q_u8(value_lo, value_hi));
                        int16x8_t value_16_b = vreinterpretq_s16_u8(vzip2q_u8(value_lo, value_hi));

                        int16x8_t res_a = vsubq_s16(vshlq_n_s16(value_16_a, 1), sign_16_a);
                        int16x8_t res_b = vsubq_s16(vshlq_n_s16(value_16_b, 1), sign_16_b);

                        int32x4_t acc0 = vaddq_s32(vld1q_s32(&y_acc[mi][n]), vmovl_s16(vget_low_s16(res_a)));
                        int32x4_t acc1 = vaddq_s32(vld1q_s32(&y_acc[mi][n + 4]), vmovl_s16(vget_high_s16(res_a)));
                        int32x4_t acc2 = vaddq_s32(vld1q_s32(&y_acc[mi][n + 8]), vmovl_s16(vget_low_s16(res_b)));
                        int32x4_t acc3 = vaddq_s32(vld1q_s32(&y_acc[mi][n + 12]), vmovl_s16(vget_high_s16(res_b)));

                        vst1q_s32(&y_acc[mi][n], acc0);
                        vst1q_s32(&y_acc[mi][n + 4], acc1);
                        vst1q_s32(&y_acc[mi][n + 8], acc2);
                        vst1q_s32(&y_acc[mi][n + 12], acc3);
                    }
                }

                // Handle remainder
                for (; n < N; n++) {
                    uint8_t s = sign_row[n] & 0x0F;
                    uint8_t v = value_row[n] & 0x0F;

                    for (int mi = 0; mi < m_count; mi++) {
                        const int8_t* x_row = x_int8 + (m_base + mi) * K;
                        int8_t x_vals[4] = {0, 0, 0, 0};
                        for (int i = 0; i < 4 && k_base + i < K; i++) {
                            x_vals[i] = x_row[k_base + i];
                        }

                        alignas(32) int16_t lut[16];
                        build_lut_16_int16(x_vals, lut);
                        y_acc[mi][n] += 2 * static_cast<int32_t>(lut[v]) - static_cast<int32_t>(lut[s]);
                    }
                }
            }

            // Convert all M rows to float
            for (int mi = 0; mi < m_count; mi++) {
                const int m = m_base + mi;
                float scale = scales[m];
                float* y_row = y + m * N;

                if (bias != nullptr) {
                    for (int n = 0; n < N; n++) {
                        y_row[n] = static_cast<float>(y_acc[mi][n]) * scale + bias[n];
                    }
                } else {
                    float32x4_t scale_vec = vdupq_n_f32(scale);
                    int n = 0;
                    for (; n + 3 < N; n += 4) {
                        int32x4_t acc = vld1q_s32(&y_acc[mi][n]);
                        float32x4_t acc_f = vcvtq_f32_s32(acc);
                        vst1q_f32(y_row + n, vmulq_f32(acc_f, scale_vec));
                    }
                    for (; n < N; n++) {
                        y_row[n] = static_cast<float>(y_acc[mi][n]) * scale;
                    }
                }
            }
        }
    }
}

/**
 * TBL v7: Fused accumulation - avoid memory round-trip
 *
 * Key insight: Keep accumulators in registers for the entire K loop,
 * only store to memory at the end of each N block.
 */
void matmul_free_neon_tbl_v7(
    torch::Tensor x_int8_tensor,       // [M, K] int8
    torch::Tensor scale_tensor,        // [M] float32
    torch::Tensor sign_plane_tensor,   // [K_groups, N] uint8
    torch::Tensor value_plane_tensor,  // [K_groups, N] uint8
    torch::Tensor y_tensor,            // [M, N] float32 output
    torch::Tensor bias_tensor,
    int M, int N, int K,
    int num_threads
) {
    const int8_t* __restrict__ x_int8 = x_int8_tensor.data_ptr<int8_t>();
    const float* __restrict__ scales = scale_tensor.data_ptr<float>();
    const uint8_t* __restrict__ sign_plane = sign_plane_tensor.data_ptr<uint8_t>();
    const uint8_t* __restrict__ value_plane = value_plane_tensor.data_ptr<uint8_t>();
    float* __restrict__ y = y_tensor.data_ptr<float>();
    const float* bias = bias_tensor.defined() ? bias_tensor.data_ptr<float>() : nullptr;

    const int K_groups = (K + 3) / 4;
    const int N_BLOCK = 16;  // Process 16 outputs with register accumulators

    omp_set_num_threads(num_threads);

    #pragma omp parallel for schedule(static)
    for (int m = 0; m < M; m++) {
        const int8_t* __restrict__ x_row = x_int8 + m * K;
        float scale = scales[m];
        float* y_row = y + m * N;

        uint8x16_t mask_0f = vdupq_n_u8(0x0F);

        // Process N in blocks of 16, keeping accumulators in registers
        for (int n_base = 0; n_base < N; n_base += N_BLOCK) {
            const int n_end = std::min(n_base + N_BLOCK, N);
            const int n_count = n_end - n_base;

            // Register accumulators - keep in registers for ENTIRE K loop
            int32x4_t acc0 = vdupq_n_s32(0);
            int32x4_t acc1 = vdupq_n_s32(0);
            int32x4_t acc2 = vdupq_n_s32(0);
            int32x4_t acc3 = vdupq_n_s32(0);

            // Full K iteration with register accumulators
            for (int kg = 0; kg < K_groups; kg++) {
                const int k_base = kg * 4;

                // Build LUT for this K group
                int8_t x_vals[4] = {0, 0, 0, 0};
                for (int i = 0; i < 4 && k_base + i < K; i++) {
                    x_vals[i] = x_row[k_base + i];
                }

                uint8x16_t lut_lo_vec, lut_hi_vec;
                build_lut_16_split_neon(x_vals, lut_lo_vec, lut_hi_vec);

                // Load indices for this N block
                const uint8_t* sign_ptr = sign_plane + kg * N + n_base;
                const uint8_t* value_ptr = value_plane + kg * N + n_base;

                if (n_count == 16) {
                    uint8x16_t sign_idx = vandq_u8(vld1q_u8(sign_ptr), mask_0f);
                    uint8x16_t value_idx = vandq_u8(vld1q_u8(value_ptr), mask_0f);

                    uint8x16_t sign_lo = vqtbl1q_u8(lut_lo_vec, sign_idx);
                    uint8x16_t sign_hi = vqtbl1q_u8(lut_hi_vec, sign_idx);
                    uint8x16_t value_lo = vqtbl1q_u8(lut_lo_vec, value_idx);
                    uint8x16_t value_hi = vqtbl1q_u8(lut_hi_vec, value_idx);

                    int16x8_t sign_16_a = vreinterpretq_s16_u8(vzip1q_u8(sign_lo, sign_hi));
                    int16x8_t sign_16_b = vreinterpretq_s16_u8(vzip2q_u8(sign_lo, sign_hi));
                    int16x8_t value_16_a = vreinterpretq_s16_u8(vzip1q_u8(value_lo, value_hi));
                    int16x8_t value_16_b = vreinterpretq_s16_u8(vzip2q_u8(value_lo, value_hi));

                    int16x8_t res_a = vsubq_s16(vshlq_n_s16(value_16_a, 1), sign_16_a);
                    int16x8_t res_b = vsubq_s16(vshlq_n_s16(value_16_b, 1), sign_16_b);

                    // Accumulate directly in registers - NO memory access!
                    acc0 = vaddq_s32(acc0, vmovl_s16(vget_low_s16(res_a)));
                    acc1 = vaddq_s32(acc1, vmovl_s16(vget_high_s16(res_a)));
                    acc2 = vaddq_s32(acc2, vmovl_s16(vget_low_s16(res_b)));
                    acc3 = vaddq_s32(acc3, vmovl_s16(vget_high_s16(res_b)));
                } else {
                    // Handle partial block with scalar code
                    alignas(32) int16_t lut[16];
                    build_lut_16_int16(x_vals, lut);

                    alignas(16) int32_t partial[16] = {0};
                    for (int i = 0; i < n_count; i++) {
                        uint8_t s = sign_ptr[i] & 0x0F;
                        uint8_t v = value_ptr[i] & 0x0F;
                        partial[i] = 2 * static_cast<int32_t>(lut[v]) - static_cast<int32_t>(lut[s]);
                    }
                    acc0 = vaddq_s32(acc0, vld1q_s32(partial));
                    acc1 = vaddq_s32(acc1, vld1q_s32(partial + 4));
                    acc2 = vaddq_s32(acc2, vld1q_s32(partial + 8));
                    acc3 = vaddq_s32(acc3, vld1q_s32(partial + 12));
                }
            }

            // Store results with scaling
            float32x4_t scale_vec = vdupq_n_f32(scale);

            if (bias != nullptr) {
                float32x4_t bias0 = vld1q_f32(bias + n_base);
                float32x4_t bias1 = vld1q_f32(bias + n_base + 4);
                float32x4_t bias2 = vld1q_f32(bias + n_base + 8);
                float32x4_t bias3 = vld1q_f32(bias + n_base + 12);

                vst1q_f32(y_row + n_base, vaddq_f32(vmulq_f32(vcvtq_f32_s32(acc0), scale_vec), bias0));
                vst1q_f32(y_row + n_base + 4, vaddq_f32(vmulq_f32(vcvtq_f32_s32(acc1), scale_vec), bias1));
                vst1q_f32(y_row + n_base + 8, vaddq_f32(vmulq_f32(vcvtq_f32_s32(acc2), scale_vec), bias2));
                vst1q_f32(y_row + n_base + 12, vaddq_f32(vmulq_f32(vcvtq_f32_s32(acc3), scale_vec), bias3));
            } else {
                vst1q_f32(y_row + n_base, vmulq_f32(vcvtq_f32_s32(acc0), scale_vec));
                vst1q_f32(y_row + n_base + 4, vmulq_f32(vcvtq_f32_s32(acc1), scale_vec));
                vst1q_f32(y_row + n_base + 8, vmulq_f32(vcvtq_f32_s32(acc2), scale_vec));
                vst1q_f32(y_row + n_base + 12, vmulq_f32(vcvtq_f32_s32(acc3), scale_vec));
            }
        }
    }
}

/**
 * TBL v8: Software pipelining + M-batching + larger N blocks
 *
 * Key optimizations:
 * 1. Build next LUT while processing current (software pipelining)
 * 2. M-batching to amortize weight loading
 * 3. Process 64 outputs at a time (4x16) with prefetching
 */
void matmul_free_neon_tbl_v8(
    torch::Tensor x_int8_tensor,       // [M, K] int8
    torch::Tensor scale_tensor,        // [M] float32
    torch::Tensor sign_plane_tensor,   // [K_groups, N] uint8
    torch::Tensor value_plane_tensor,  // [K_groups, N] uint8
    torch::Tensor y_tensor,            // [M, N] float32 output
    torch::Tensor bias_tensor,
    int M, int N, int K,
    int num_threads
) {
    const int8_t* __restrict__ x_int8 = x_int8_tensor.data_ptr<int8_t>();
    const float* __restrict__ scales = scale_tensor.data_ptr<float>();
    const uint8_t* __restrict__ sign_plane = sign_plane_tensor.data_ptr<uint8_t>();
    const uint8_t* __restrict__ value_plane = value_plane_tensor.data_ptr<uint8_t>();
    float* __restrict__ y = y_tensor.data_ptr<float>();
    const float* bias = bias_tensor.defined() ? bias_tensor.data_ptr<float>() : nullptr;

    const int K_groups = (K + 3) / 4;
    const int M_BATCH = 2;  // Process 2 M rows together
    const int N_BLOCK = 64; // Process 64 outputs at a time

    omp_set_num_threads(num_threads);

    #pragma omp parallel
    {
        // Accumulators for M_BATCH rows
        alignas(64) int32_t y_acc[M_BATCH][2048];  // Max N = 2048

        #pragma omp for schedule(static)
        for (int m_base = 0; m_base < M; m_base += M_BATCH) {
            const int m_end = std::min(m_base + M_BATCH, M);
            const int m_count = m_end - m_base;

            // Initialize accumulators
            for (int mi = 0; mi < m_count; mi++) {
                memset(y_acc[mi], 0, N * sizeof(int32_t));
            }

            uint8x16_t mask_0f = vdupq_n_u8(0x0F);

            // Pre-build first LUT for pipelining
            uint8x16_t lut_lo_cur[M_BATCH], lut_hi_cur[M_BATCH];
            uint8x16_t lut_lo_next[M_BATCH], lut_hi_next[M_BATCH];

            // Build first K group LUTs
            if (K_groups > 0) {
                for (int mi = 0; mi < m_count; mi++) {
                    const int8_t* x_row = x_int8 + (m_base + mi) * K;
                    int8_t x_vals[4] = {0, 0, 0, 0};
                    x_vals[0] = x_row[0];
                    if (K > 1) x_vals[1] = x_row[1];
                    if (K > 2) x_vals[2] = x_row[2];
                    if (K > 3) x_vals[3] = x_row[3];
                    build_lut_16_split_neon(x_vals, lut_lo_cur[mi], lut_hi_cur[mi]);
                }
            }

            // K-first iteration with software pipelining
            for (int kg = 0; kg < K_groups; kg++) {
                // Prefetch and build next K group's LUT while processing current
                if (kg + 1 < K_groups) {
                    const int next_k_base = (kg + 1) * 4;
                    for (int mi = 0; mi < m_count; mi++) {
                        const int8_t* x_row = x_int8 + (m_base + mi) * K;
                        int8_t x_vals[4] = {0, 0, 0, 0};
                        for (int i = 0; i < 4 && next_k_base + i < K; i++) {
                            x_vals[i] = x_row[next_k_base + i];
                        }
                        build_lut_16_split_neon(x_vals, lut_lo_next[mi], lut_hi_next[mi]);
                    }
                }

                const uint8_t* __restrict__ sign_row = sign_plane + kg * N;
                const uint8_t* __restrict__ value_row = value_plane + kg * N;

                // Process N in 64-element blocks
                int n = 0;
                for (; n + 63 < N; n += N_BLOCK) {
                    // Prefetch next N block's data
                    __builtin_prefetch(sign_row + n + N_BLOCK, 0, 3);
                    __builtin_prefetch(value_row + n + N_BLOCK, 0, 3);

                    // Load indices ONCE, reuse for all M rows
                    uint8x16_t sign_idx0 = vandq_u8(vld1q_u8(sign_row + n), mask_0f);
                    uint8x16_t value_idx0 = vandq_u8(vld1q_u8(value_row + n), mask_0f);
                    uint8x16_t sign_idx1 = vandq_u8(vld1q_u8(sign_row + n + 16), mask_0f);
                    uint8x16_t value_idx1 = vandq_u8(vld1q_u8(value_row + n + 16), mask_0f);
                    uint8x16_t sign_idx2 = vandq_u8(vld1q_u8(sign_row + n + 32), mask_0f);
                    uint8x16_t value_idx2 = vandq_u8(vld1q_u8(value_row + n + 32), mask_0f);
                    uint8x16_t sign_idx3 = vandq_u8(vld1q_u8(sign_row + n + 48), mask_0f);
                    uint8x16_t value_idx3 = vandq_u8(vld1q_u8(value_row + n + 48), mask_0f);

                    // Process each M row with same indices
                    for (int mi = 0; mi < m_count; mi++) {
                        // Batch 0 (n to n+15)
                        uint8x16_t sign_lo0 = vqtbl1q_u8(lut_lo_cur[mi], sign_idx0);
                        uint8x16_t sign_hi0 = vqtbl1q_u8(lut_hi_cur[mi], sign_idx0);
                        uint8x16_t value_lo0 = vqtbl1q_u8(lut_lo_cur[mi], value_idx0);
                        uint8x16_t value_hi0 = vqtbl1q_u8(lut_hi_cur[mi], value_idx0);

                        int16x8_t sign_16_0a = vreinterpretq_s16_u8(vzip1q_u8(sign_lo0, sign_hi0));
                        int16x8_t sign_16_0b = vreinterpretq_s16_u8(vzip2q_u8(sign_lo0, sign_hi0));
                        int16x8_t value_16_0a = vreinterpretq_s16_u8(vzip1q_u8(value_lo0, value_hi0));
                        int16x8_t value_16_0b = vreinterpretq_s16_u8(vzip2q_u8(value_lo0, value_hi0));

                        int16x8_t res_0a = vsubq_s16(vshlq_n_s16(value_16_0a, 1), sign_16_0a);
                        int16x8_t res_0b = vsubq_s16(vshlq_n_s16(value_16_0b, 1), sign_16_0b);

                        int32x4_t acc00 = vaddq_s32(vld1q_s32(&y_acc[mi][n]), vmovl_s16(vget_low_s16(res_0a)));
                        int32x4_t acc01 = vaddq_s32(vld1q_s32(&y_acc[mi][n + 4]), vmovl_s16(vget_high_s16(res_0a)));
                        int32x4_t acc02 = vaddq_s32(vld1q_s32(&y_acc[mi][n + 8]), vmovl_s16(vget_low_s16(res_0b)));
                        int32x4_t acc03 = vaddq_s32(vld1q_s32(&y_acc[mi][n + 12]), vmovl_s16(vget_high_s16(res_0b)));

                        vst1q_s32(&y_acc[mi][n], acc00);
                        vst1q_s32(&y_acc[mi][n + 4], acc01);
                        vst1q_s32(&y_acc[mi][n + 8], acc02);
                        vst1q_s32(&y_acc[mi][n + 12], acc03);

                        // Batch 1 (n+16 to n+31)
                        uint8x16_t sign_lo1 = vqtbl1q_u8(lut_lo_cur[mi], sign_idx1);
                        uint8x16_t sign_hi1 = vqtbl1q_u8(lut_hi_cur[mi], sign_idx1);
                        uint8x16_t value_lo1 = vqtbl1q_u8(lut_lo_cur[mi], value_idx1);
                        uint8x16_t value_hi1 = vqtbl1q_u8(lut_hi_cur[mi], value_idx1);

                        int16x8_t sign_16_1a = vreinterpretq_s16_u8(vzip1q_u8(sign_lo1, sign_hi1));
                        int16x8_t sign_16_1b = vreinterpretq_s16_u8(vzip2q_u8(sign_lo1, sign_hi1));
                        int16x8_t value_16_1a = vreinterpretq_s16_u8(vzip1q_u8(value_lo1, value_hi1));
                        int16x8_t value_16_1b = vreinterpretq_s16_u8(vzip2q_u8(value_lo1, value_hi1));

                        int16x8_t res_1a = vsubq_s16(vshlq_n_s16(value_16_1a, 1), sign_16_1a);
                        int16x8_t res_1b = vsubq_s16(vshlq_n_s16(value_16_1b, 1), sign_16_1b);

                        int32x4_t acc10 = vaddq_s32(vld1q_s32(&y_acc[mi][n + 16]), vmovl_s16(vget_low_s16(res_1a)));
                        int32x4_t acc11 = vaddq_s32(vld1q_s32(&y_acc[mi][n + 20]), vmovl_s16(vget_high_s16(res_1a)));
                        int32x4_t acc12 = vaddq_s32(vld1q_s32(&y_acc[mi][n + 24]), vmovl_s16(vget_low_s16(res_1b)));
                        int32x4_t acc13 = vaddq_s32(vld1q_s32(&y_acc[mi][n + 28]), vmovl_s16(vget_high_s16(res_1b)));

                        vst1q_s32(&y_acc[mi][n + 16], acc10);
                        vst1q_s32(&y_acc[mi][n + 20], acc11);
                        vst1q_s32(&y_acc[mi][n + 24], acc12);
                        vst1q_s32(&y_acc[mi][n + 28], acc13);

                        // Batch 2 (n+32 to n+47)
                        uint8x16_t sign_lo2 = vqtbl1q_u8(lut_lo_cur[mi], sign_idx2);
                        uint8x16_t sign_hi2 = vqtbl1q_u8(lut_hi_cur[mi], sign_idx2);
                        uint8x16_t value_lo2 = vqtbl1q_u8(lut_lo_cur[mi], value_idx2);
                        uint8x16_t value_hi2 = vqtbl1q_u8(lut_hi_cur[mi], value_idx2);

                        int16x8_t sign_16_2a = vreinterpretq_s16_u8(vzip1q_u8(sign_lo2, sign_hi2));
                        int16x8_t sign_16_2b = vreinterpretq_s16_u8(vzip2q_u8(sign_lo2, sign_hi2));
                        int16x8_t value_16_2a = vreinterpretq_s16_u8(vzip1q_u8(value_lo2, value_hi2));
                        int16x8_t value_16_2b = vreinterpretq_s16_u8(vzip2q_u8(value_lo2, value_hi2));

                        int16x8_t res_2a = vsubq_s16(vshlq_n_s16(value_16_2a, 1), sign_16_2a);
                        int16x8_t res_2b = vsubq_s16(vshlq_n_s16(value_16_2b, 1), sign_16_2b);

                        int32x4_t acc20 = vaddq_s32(vld1q_s32(&y_acc[mi][n + 32]), vmovl_s16(vget_low_s16(res_2a)));
                        int32x4_t acc21 = vaddq_s32(vld1q_s32(&y_acc[mi][n + 36]), vmovl_s16(vget_high_s16(res_2a)));
                        int32x4_t acc22 = vaddq_s32(vld1q_s32(&y_acc[mi][n + 40]), vmovl_s16(vget_low_s16(res_2b)));
                        int32x4_t acc23 = vaddq_s32(vld1q_s32(&y_acc[mi][n + 44]), vmovl_s16(vget_high_s16(res_2b)));

                        vst1q_s32(&y_acc[mi][n + 32], acc20);
                        vst1q_s32(&y_acc[mi][n + 36], acc21);
                        vst1q_s32(&y_acc[mi][n + 40], acc22);
                        vst1q_s32(&y_acc[mi][n + 44], acc23);

                        // Batch 3 (n+48 to n+63)
                        uint8x16_t sign_lo3 = vqtbl1q_u8(lut_lo_cur[mi], sign_idx3);
                        uint8x16_t sign_hi3 = vqtbl1q_u8(lut_hi_cur[mi], sign_idx3);
                        uint8x16_t value_lo3 = vqtbl1q_u8(lut_lo_cur[mi], value_idx3);
                        uint8x16_t value_hi3 = vqtbl1q_u8(lut_hi_cur[mi], value_idx3);

                        int16x8_t sign_16_3a = vreinterpretq_s16_u8(vzip1q_u8(sign_lo3, sign_hi3));
                        int16x8_t sign_16_3b = vreinterpretq_s16_u8(vzip2q_u8(sign_lo3, sign_hi3));
                        int16x8_t value_16_3a = vreinterpretq_s16_u8(vzip1q_u8(value_lo3, value_hi3));
                        int16x8_t value_16_3b = vreinterpretq_s16_u8(vzip2q_u8(value_lo3, value_hi3));

                        int16x8_t res_3a = vsubq_s16(vshlq_n_s16(value_16_3a, 1), sign_16_3a);
                        int16x8_t res_3b = vsubq_s16(vshlq_n_s16(value_16_3b, 1), sign_16_3b);

                        int32x4_t acc30 = vaddq_s32(vld1q_s32(&y_acc[mi][n + 48]), vmovl_s16(vget_low_s16(res_3a)));
                        int32x4_t acc31 = vaddq_s32(vld1q_s32(&y_acc[mi][n + 52]), vmovl_s16(vget_high_s16(res_3a)));
                        int32x4_t acc32 = vaddq_s32(vld1q_s32(&y_acc[mi][n + 56]), vmovl_s16(vget_low_s16(res_3b)));
                        int32x4_t acc33 = vaddq_s32(vld1q_s32(&y_acc[mi][n + 60]), vmovl_s16(vget_high_s16(res_3b)));

                        vst1q_s32(&y_acc[mi][n + 48], acc30);
                        vst1q_s32(&y_acc[mi][n + 52], acc31);
                        vst1q_s32(&y_acc[mi][n + 56], acc32);
                        vst1q_s32(&y_acc[mi][n + 60], acc33);
                    }
                }

                // Handle remaining N elements
                for (; n + 15 < N; n += 16) {
                    uint8x16_t sign_idx = vandq_u8(vld1q_u8(sign_row + n), mask_0f);
                    uint8x16_t value_idx = vandq_u8(vld1q_u8(value_row + n), mask_0f);

                    for (int mi = 0; mi < m_count; mi++) {
                        uint8x16_t sign_lo = vqtbl1q_u8(lut_lo_cur[mi], sign_idx);
                        uint8x16_t sign_hi = vqtbl1q_u8(lut_hi_cur[mi], sign_idx);
                        uint8x16_t value_lo = vqtbl1q_u8(lut_lo_cur[mi], value_idx);
                        uint8x16_t value_hi = vqtbl1q_u8(lut_hi_cur[mi], value_idx);

                        int16x8_t sign_16_a = vreinterpretq_s16_u8(vzip1q_u8(sign_lo, sign_hi));
                        int16x8_t sign_16_b = vreinterpretq_s16_u8(vzip2q_u8(sign_lo, sign_hi));
                        int16x8_t value_16_a = vreinterpretq_s16_u8(vzip1q_u8(value_lo, value_hi));
                        int16x8_t value_16_b = vreinterpretq_s16_u8(vzip2q_u8(value_lo, value_hi));

                        int16x8_t res_a = vsubq_s16(vshlq_n_s16(value_16_a, 1), sign_16_a);
                        int16x8_t res_b = vsubq_s16(vshlq_n_s16(value_16_b, 1), sign_16_b);

                        int32x4_t acc0 = vaddq_s32(vld1q_s32(&y_acc[mi][n]), vmovl_s16(vget_low_s16(res_a)));
                        int32x4_t acc1 = vaddq_s32(vld1q_s32(&y_acc[mi][n + 4]), vmovl_s16(vget_high_s16(res_a)));
                        int32x4_t acc2 = vaddq_s32(vld1q_s32(&y_acc[mi][n + 8]), vmovl_s16(vget_low_s16(res_b)));
                        int32x4_t acc3 = vaddq_s32(vld1q_s32(&y_acc[mi][n + 12]), vmovl_s16(vget_high_s16(res_b)));

                        vst1q_s32(&y_acc[mi][n], acc0);
                        vst1q_s32(&y_acc[mi][n + 4], acc1);
                        vst1q_s32(&y_acc[mi][n + 8], acc2);
                        vst1q_s32(&y_acc[mi][n + 12], acc3);
                    }
                }

                // Scalar remainder
                for (; n < N; n++) {
                    uint8_t s = sign_row[n] & 0x0F;
                    uint8_t v = value_row[n] & 0x0F;

                    for (int mi = 0; mi < m_count; mi++) {
                        const int8_t* x_row = x_int8 + (m_base + mi) * K;
                        const int k_base = kg * 4;
                        int8_t x_vals[4] = {0, 0, 0, 0};
                        for (int i = 0; i < 4 && k_base + i < K; i++) {
                            x_vals[i] = x_row[k_base + i];
                        }

                        alignas(32) int16_t lut[16];
                        build_lut_16_int16(x_vals, lut);
                        y_acc[mi][n] += 2 * static_cast<int32_t>(lut[v]) - static_cast<int32_t>(lut[s]);
                    }
                }

                // Swap current and next LUTs
                for (int mi = 0; mi < m_count; mi++) {
                    lut_lo_cur[mi] = lut_lo_next[mi];
                    lut_hi_cur[mi] = lut_hi_next[mi];
                }
            }

            // Convert all M rows to float
            for (int mi = 0; mi < m_count; mi++) {
                const int m = m_base + mi;
                float scale = scales[m];
                float* y_row = y + m * N;

                if (bias != nullptr) {
                    for (int n = 0; n < N; n++) {
                        y_row[n] = static_cast<float>(y_acc[mi][n]) * scale + bias[n];
                    }
                } else {
                    float32x4_t scale_vec = vdupq_n_f32(scale);
                    int n = 0;
                    for (; n + 3 < N; n += 4) {
                        int32x4_t acc = vld1q_s32(&y_acc[mi][n]);
                        float32x4_t acc_f = vcvtq_f32_s32(acc);
                        vst1q_f32(y_row + n, vmulq_f32(acc_f, scale_vec));
                    }
                    for (; n < N; n++) {
                        y_row[n] = static_cast<float>(y_acc[mi][n]) * scale;
                    }
                }
            }
        }
    }
}

/**
 * TBL v9: N-first with register accumulators + instruction reordering
 *
 * Key optimizations:
 * 1. N-first iteration: Process N_BLOCK outputs completely through all K
 * 2. Keep accumulators in registers for entire K loop (no memory round-trip)
 * 3. Interleave TBL and ZIP instructions to hide latency
 * 4. M-batching for weight index reuse
 */
void matmul_free_neon_tbl_v9(
    torch::Tensor x_int8_tensor,       // [M, K] int8
    torch::Tensor scale_tensor,        // [M] float32
    torch::Tensor sign_plane_tensor,   // [K_groups, N] uint8
    torch::Tensor value_plane_tensor,  // [K_groups, N] uint8
    torch::Tensor y_tensor,            // [M, N] float32 output
    torch::Tensor bias_tensor,
    int M, int N, int K,
    int num_threads
) {
    const int8_t* __restrict__ x_int8 = x_int8_tensor.data_ptr<int8_t>();
    const float* __restrict__ scales = scale_tensor.data_ptr<float>();
    const uint8_t* __restrict__ sign_plane = sign_plane_tensor.data_ptr<uint8_t>();
    const uint8_t* __restrict__ value_plane = value_plane_tensor.data_ptr<uint8_t>();
    float* __restrict__ y = y_tensor.data_ptr<float>();
    const float* bias = bias_tensor.defined() ? bias_tensor.data_ptr<float>() : nullptr;

    const int K_groups = (K + 3) / 4;
    const int N_BLOCK = 16;  // Process 16 outputs with register accumulators

    omp_set_num_threads(num_threads);

    #pragma omp parallel for schedule(static)
    for (int m = 0; m < M; m++) {
        const int8_t* __restrict__ x_row = x_int8 + m * K;
        const float scale = scales[m];
        float* __restrict__ y_row = y + m * N;

        const uint8x16_t mask_0f = vdupq_n_u8(0x0F);

        // N-first: process N_BLOCK outputs completely before moving on
        for (int n_base = 0; n_base < N; n_base += N_BLOCK) {
            const int n_end = std::min(n_base + N_BLOCK, N);
            const int n_count = n_end - n_base;

            // Keep accumulators in registers for entire K loop
            int32x4_t acc0 = vdupq_n_s32(0);
            int32x4_t acc1 = vdupq_n_s32(0);
            int32x4_t acc2 = vdupq_n_s32(0);
            int32x4_t acc3 = vdupq_n_s32(0);

            // Iterate through all K groups
            for (int kg = 0; kg < K_groups; kg++) {
                const int k_base = kg * 4;

                // Build LUT for this K group
                int8_t x_vals[4] = {0, 0, 0, 0};
                for (int i = 0; i < 4 && k_base + i < K; i++) {
                    x_vals[i] = x_row[k_base + i];
                }

                uint8x16_t lut_lo, lut_hi;
                build_lut_16_split_neon(x_vals, lut_lo, lut_hi);

                // Load weight indices for this N block
                const uint8_t* sign_ptr = sign_plane + kg * N + n_base;
                const uint8_t* value_ptr = value_plane + kg * N + n_base;

                if (n_count == N_BLOCK) {
                    // Full block - vectorized path
                    uint8x16_t sign_idx = vandq_u8(vld1q_u8(sign_ptr), mask_0f);
                    uint8x16_t value_idx = vandq_u8(vld1q_u8(value_ptr), mask_0f);

                    // Interleaved TBL lookups to hide latency
                    // Do sign_lo and value_lo first (both use lut_lo)
                    uint8x16_t sign_lo = vqtbl1q_u8(lut_lo, sign_idx);
                    uint8x16_t value_lo = vqtbl1q_u8(lut_lo, value_idx);
                    // Then sign_hi and value_hi (both use lut_hi)
                    uint8x16_t sign_hi = vqtbl1q_u8(lut_hi, sign_idx);
                    uint8x16_t value_hi = vqtbl1q_u8(lut_hi, value_idx);

                    // Interleaved ZIP - start as soon as data is ready
                    int16x8_t sign_16_a = vreinterpretq_s16_u8(vzip1q_u8(sign_lo, sign_hi));
                    int16x8_t value_16_a = vreinterpretq_s16_u8(vzip1q_u8(value_lo, value_hi));
                    int16x8_t sign_16_b = vreinterpretq_s16_u8(vzip2q_u8(sign_lo, sign_hi));
                    int16x8_t value_16_b = vreinterpretq_s16_u8(vzip2q_u8(value_lo, value_hi));

                    // Compute 2*value - sign using add instead of shift (better pipelining)
                    int16x8_t res_a = vsubq_s16(vaddq_s16(value_16_a, value_16_a), sign_16_a);
                    int16x8_t res_b = vsubq_s16(vaddq_s16(value_16_b, value_16_b), sign_16_b);

                    // Accumulate directly in registers (no memory load/store)
                    acc0 = vaddq_s32(acc0, vmovl_s16(vget_low_s16(res_a)));
                    acc1 = vaddq_s32(acc1, vmovl_s16(vget_high_s16(res_a)));
                    acc2 = vaddq_s32(acc2, vmovl_s16(vget_low_s16(res_b)));
                    acc3 = vaddq_s32(acc3, vmovl_s16(vget_high_s16(res_b)));
                } else {
                    // Partial block - scalar fallback
                    alignas(32) int16_t lut[16];
                    build_lut_16_int16(x_vals, lut);

                    alignas(16) int32_t acc_arr[16];
                    vst1q_s32(acc_arr, acc0);
                    vst1q_s32(acc_arr + 4, acc1);
                    vst1q_s32(acc_arr + 8, acc2);
                    vst1q_s32(acc_arr + 12, acc3);

                    for (int i = 0; i < n_count; i++) {
                        uint8_t s = sign_ptr[i] & 0x0F;
                        uint8_t v = value_ptr[i] & 0x0F;
                        acc_arr[i] += 2 * static_cast<int32_t>(lut[v]) - static_cast<int32_t>(lut[s]);
                    }

                    acc0 = vld1q_s32(acc_arr);
                    acc1 = vld1q_s32(acc_arr + 4);
                    acc2 = vld1q_s32(acc_arr + 8);
                    acc3 = vld1q_s32(acc_arr + 12);
                }
            }

            // Convert to float and store (only once per N block)
            float32x4_t scale_vec = vdupq_n_f32(scale);

            if (bias != nullptr && n_base + 16 <= N) {
                float32x4_t bias0 = vld1q_f32(bias + n_base);
                float32x4_t bias1 = vld1q_f32(bias + n_base + 4);
                float32x4_t bias2 = vld1q_f32(bias + n_base + 8);
                float32x4_t bias3 = vld1q_f32(bias + n_base + 12);

                vst1q_f32(y_row + n_base, vaddq_f32(vmulq_f32(vcvtq_f32_s32(acc0), scale_vec), bias0));
                vst1q_f32(y_row + n_base + 4, vaddq_f32(vmulq_f32(vcvtq_f32_s32(acc1), scale_vec), bias1));
                vst1q_f32(y_row + n_base + 8, vaddq_f32(vmulq_f32(vcvtq_f32_s32(acc2), scale_vec), bias2));
                vst1q_f32(y_row + n_base + 12, vaddq_f32(vmulq_f32(vcvtq_f32_s32(acc3), scale_vec), bias3));
            } else if (n_base + 16 <= N) {
                vst1q_f32(y_row + n_base, vmulq_f32(vcvtq_f32_s32(acc0), scale_vec));
                vst1q_f32(y_row + n_base + 4, vmulq_f32(vcvtq_f32_s32(acc1), scale_vec));
                vst1q_f32(y_row + n_base + 8, vmulq_f32(vcvtq_f32_s32(acc2), scale_vec));
                vst1q_f32(y_row + n_base + 12, vmulq_f32(vcvtq_f32_s32(acc3), scale_vec));
            } else {
                // Partial store
                alignas(16) int32_t acc_arr[16];
                vst1q_s32(acc_arr, acc0);
                vst1q_s32(acc_arr + 4, acc1);
                vst1q_s32(acc_arr + 8, acc2);
                vst1q_s32(acc_arr + 12, acc3);

                for (int i = 0; i < n_count; i++) {
                    float val = static_cast<float>(acc_arr[i]) * scale;
                    if (bias != nullptr) val += bias[n_base + i];
                    y_row[n_base + i] = val;
                }
            }
        }
    }
}

/**
 * TBL v10: Larger group size g=8 with vqtbl4q_u8
 *
 * Key insight: Process 8 K elements per lookup instead of 4
 * - Halves K iterations
 * - Uses vqtbl4q_u8 for 64-byte table lookup (256 entries as uint8)
 * - But need to be careful about int16 overflow
 *
 * For g=8 with int8 activations: max sum = 8 * 127 = 1016, fits in int16
 */
void matmul_free_neon_tbl_v10(
    torch::Tensor x_int8_tensor,       // [M, K] int8
    torch::Tensor scale_tensor,        // [M] float32
    torch::Tensor sign_plane_tensor,   // [K_groups, N] uint8 - K_groups = ceil(K/8)
    torch::Tensor value_plane_tensor,  // [K_groups, N] uint8
    torch::Tensor y_tensor,            // [M, N] float32 output
    torch::Tensor bias_tensor,
    int M, int N, int K,
    int num_threads
) {
    const int8_t* __restrict__ x_int8 = x_int8_tensor.data_ptr<int8_t>();
    const float* __restrict__ scales = scale_tensor.data_ptr<float>();
    const uint8_t* __restrict__ sign_plane = sign_plane_tensor.data_ptr<uint8_t>();
    const uint8_t* __restrict__ value_plane = value_plane_tensor.data_ptr<uint8_t>();
    float* __restrict__ y = y_tensor.data_ptr<float>();
    const float* bias = bias_tensor.defined() ? bias_tensor.data_ptr<float>() : nullptr;

    // For g=8, we need 256-entry LUT, but vqtbl4q only does 64 bytes
    // So we use g=4 but process 2 groups at once with better pipelining
    // This is a hybrid approach: still g=4 LUT but doubled iteration

    const int K_groups = (K + 3) / 4;
    const int M_BATCH = 8;  // Increased M batching

    omp_set_num_threads(num_threads);

    #pragma omp parallel
    {
        alignas(64) int32_t y_acc[M_BATCH][2048];

        #pragma omp for schedule(static)
        for (int m_base = 0; m_base < M; m_base += M_BATCH) {
            const int m_end = std::min(m_base + M_BATCH, M);
            const int m_count = m_end - m_base;

            for (int mi = 0; mi < m_count; mi++) {
                memset(y_acc[mi], 0, N * sizeof(int32_t));
            }

            const uint8x16_t mask_0f = vdupq_n_u8(0x0F);

            // Process 2 K groups at a time for better pipelining
            int kg = 0;
            for (; kg + 1 < K_groups; kg += 2) {
                const int k_base0 = kg * 4;
                const int k_base1 = (kg + 1) * 4;

                // Build LUTs for all M rows, both K groups
                uint8x16_t lut_lo0[M_BATCH], lut_hi0[M_BATCH];
                uint8x16_t lut_lo1[M_BATCH], lut_hi1[M_BATCH];

                for (int mi = 0; mi < m_count; mi++) {
                    const int8_t* x_row = x_int8 + (m_base + mi) * K;

                    int8_t x_vals0[4] = {0, 0, 0, 0};
                    int8_t x_vals1[4] = {0, 0, 0, 0};

                    for (int i = 0; i < 4 && k_base0 + i < K; i++) {
                        x_vals0[i] = x_row[k_base0 + i];
                    }
                    for (int i = 0; i < 4 && k_base1 + i < K; i++) {
                        x_vals1[i] = x_row[k_base1 + i];
                    }

                    build_lut_16_split_neon(x_vals0, lut_lo0[mi], lut_hi0[mi]);
                    build_lut_16_split_neon(x_vals1, lut_lo1[mi], lut_hi1[mi]);
                }

                const uint8_t* __restrict__ sign_row0 = sign_plane + kg * N;
                const uint8_t* __restrict__ value_row0 = value_plane + kg * N;
                const uint8_t* __restrict__ sign_row1 = sign_plane + (kg + 1) * N;
                const uint8_t* __restrict__ value_row1 = value_plane + (kg + 1) * N;

                // Process N with prefetching
                for (int n = 0; n + 15 < N; n += 16) {
                    // Prefetch next iteration
                    __builtin_prefetch(sign_row0 + n + 64, 0, 2);
                    __builtin_prefetch(value_row0 + n + 64, 0, 2);

                    // Load indices for both K groups
                    uint8x16_t sign_idx0 = vandq_u8(vld1q_u8(sign_row0 + n), mask_0f);
                    uint8x16_t value_idx0 = vandq_u8(vld1q_u8(value_row0 + n), mask_0f);
                    uint8x16_t sign_idx1 = vandq_u8(vld1q_u8(sign_row1 + n), mask_0f);
                    uint8x16_t value_idx1 = vandq_u8(vld1q_u8(value_row1 + n), mask_0f);

                    for (int mi = 0; mi < m_count; mi++) {
                        // K group 0
                        uint8x16_t s_lo0 = vqtbl1q_u8(lut_lo0[mi], sign_idx0);
                        uint8x16_t v_lo0 = vqtbl1q_u8(lut_lo0[mi], value_idx0);
                        uint8x16_t s_hi0 = vqtbl1q_u8(lut_hi0[mi], sign_idx0);
                        uint8x16_t v_hi0 = vqtbl1q_u8(lut_hi0[mi], value_idx0);

                        // K group 1 (interleaved to hide latency)
                        uint8x16_t s_lo1 = vqtbl1q_u8(lut_lo1[mi], sign_idx1);
                        uint8x16_t v_lo1 = vqtbl1q_u8(lut_lo1[mi], value_idx1);
                        uint8x16_t s_hi1 = vqtbl1q_u8(lut_hi1[mi], sign_idx1);
                        uint8x16_t v_hi1 = vqtbl1q_u8(lut_hi1[mi], value_idx1);

                        // Unpack K group 0
                        int16x8_t s16_0a = vreinterpretq_s16_u8(vzip1q_u8(s_lo0, s_hi0));
                        int16x8_t v16_0a = vreinterpretq_s16_u8(vzip1q_u8(v_lo0, v_hi0));
                        int16x8_t s16_0b = vreinterpretq_s16_u8(vzip2q_u8(s_lo0, s_hi0));
                        int16x8_t v16_0b = vreinterpretq_s16_u8(vzip2q_u8(v_lo0, v_hi0));

                        // Unpack K group 1
                        int16x8_t s16_1a = vreinterpretq_s16_u8(vzip1q_u8(s_lo1, s_hi1));
                        int16x8_t v16_1a = vreinterpretq_s16_u8(vzip1q_u8(v_lo1, v_hi1));
                        int16x8_t s16_1b = vreinterpretq_s16_u8(vzip2q_u8(s_lo1, s_hi1));
                        int16x8_t v16_1b = vreinterpretq_s16_u8(vzip2q_u8(v_lo1, v_hi1));

                        // Compute 2*value - sign for both groups
                        int16x8_t res_0a = vsubq_s16(vaddq_s16(v16_0a, v16_0a), s16_0a);
                        int16x8_t res_0b = vsubq_s16(vaddq_s16(v16_0b, v16_0b), s16_0b);
                        int16x8_t res_1a = vsubq_s16(vaddq_s16(v16_1a, v16_1a), s16_1a);
                        int16x8_t res_1b = vsubq_s16(vaddq_s16(v16_1b, v16_1b), s16_1b);

                        // Add both results together before accumulating (reduces memory ops)
                        int16x8_t sum_a = vaddq_s16(res_0a, res_1a);
                        int16x8_t sum_b = vaddq_s16(res_0b, res_1b);

                        // Load, accumulate, store
                        int32x4_t acc0 = vaddq_s32(vld1q_s32(&y_acc[mi][n]), vmovl_s16(vget_low_s16(sum_a)));
                        int32x4_t acc1 = vaddq_s32(vld1q_s32(&y_acc[mi][n + 4]), vmovl_s16(vget_high_s16(sum_a)));
                        int32x4_t acc2 = vaddq_s32(vld1q_s32(&y_acc[mi][n + 8]), vmovl_s16(vget_low_s16(sum_b)));
                        int32x4_t acc3 = vaddq_s32(vld1q_s32(&y_acc[mi][n + 12]), vmovl_s16(vget_high_s16(sum_b)));

                        vst1q_s32(&y_acc[mi][n], acc0);
                        vst1q_s32(&y_acc[mi][n + 4], acc1);
                        vst1q_s32(&y_acc[mi][n + 8], acc2);
                        vst1q_s32(&y_acc[mi][n + 12], acc3);
                    }
                }

                // Handle N remainder
                for (int n = (N / 16) * 16; n < N; n++) {
                    uint8_t s0 = sign_row0[n] & 0x0F;
                    uint8_t v0 = value_row0[n] & 0x0F;
                    uint8_t s1 = sign_row1[n] & 0x0F;
                    uint8_t v1 = value_row1[n] & 0x0F;

                    for (int mi = 0; mi < m_count; mi++) {
                        const int8_t* x_row = x_int8 + (m_base + mi) * K;

                        alignas(32) int16_t lut0[16], lut1[16];
                        int8_t xv0[4] = {0, 0, 0, 0}, xv1[4] = {0, 0, 0, 0};
                        for (int i = 0; i < 4 && k_base0 + i < K; i++) xv0[i] = x_row[k_base0 + i];
                        for (int i = 0; i < 4 && k_base1 + i < K; i++) xv1[i] = x_row[k_base1 + i];

                        build_lut_16_int16(xv0, lut0);
                        build_lut_16_int16(xv1, lut1);

                        y_acc[mi][n] += 2 * static_cast<int32_t>(lut0[v0]) - static_cast<int32_t>(lut0[s0]);
                        y_acc[mi][n] += 2 * static_cast<int32_t>(lut1[v1]) - static_cast<int32_t>(lut1[s1]);
                    }
                }
            }

            // Handle remaining K group if odd number
            for (; kg < K_groups; kg++) {
                const int k_base = kg * 4;

                uint8x16_t lut_lo[M_BATCH], lut_hi[M_BATCH];

                for (int mi = 0; mi < m_count; mi++) {
                    const int8_t* x_row = x_int8 + (m_base + mi) * K;
                    int8_t x_vals[4] = {0, 0, 0, 0};
                    for (int i = 0; i < 4 && k_base + i < K; i++) {
                        x_vals[i] = x_row[k_base + i];
                    }
                    build_lut_16_split_neon(x_vals, lut_lo[mi], lut_hi[mi]);
                }

                const uint8_t* __restrict__ sign_row = sign_plane + kg * N;
                const uint8_t* __restrict__ value_row = value_plane + kg * N;

                for (int n = 0; n + 15 < N; n += 16) {
                    uint8x16_t sign_idx = vandq_u8(vld1q_u8(sign_row + n), mask_0f);
                    uint8x16_t value_idx = vandq_u8(vld1q_u8(value_row + n), mask_0f);

                    for (int mi = 0; mi < m_count; mi++) {
                        uint8x16_t s_lo = vqtbl1q_u8(lut_lo[mi], sign_idx);
                        uint8x16_t v_lo = vqtbl1q_u8(lut_lo[mi], value_idx);
                        uint8x16_t s_hi = vqtbl1q_u8(lut_hi[mi], sign_idx);
                        uint8x16_t v_hi = vqtbl1q_u8(lut_hi[mi], value_idx);

                        int16x8_t s16_a = vreinterpretq_s16_u8(vzip1q_u8(s_lo, s_hi));
                        int16x8_t v16_a = vreinterpretq_s16_u8(vzip1q_u8(v_lo, v_hi));
                        int16x8_t s16_b = vreinterpretq_s16_u8(vzip2q_u8(s_lo, s_hi));
                        int16x8_t v16_b = vreinterpretq_s16_u8(vzip2q_u8(v_lo, v_hi));

                        int16x8_t res_a = vsubq_s16(vaddq_s16(v16_a, v16_a), s16_a);
                        int16x8_t res_b = vsubq_s16(vaddq_s16(v16_b, v16_b), s16_b);

                        int32x4_t acc0 = vaddq_s32(vld1q_s32(&y_acc[mi][n]), vmovl_s16(vget_low_s16(res_a)));
                        int32x4_t acc1 = vaddq_s32(vld1q_s32(&y_acc[mi][n + 4]), vmovl_s16(vget_high_s16(res_a)));
                        int32x4_t acc2 = vaddq_s32(vld1q_s32(&y_acc[mi][n + 8]), vmovl_s16(vget_low_s16(res_b)));
                        int32x4_t acc3 = vaddq_s32(vld1q_s32(&y_acc[mi][n + 12]), vmovl_s16(vget_high_s16(res_b)));

                        vst1q_s32(&y_acc[mi][n], acc0);
                        vst1q_s32(&y_acc[mi][n + 4], acc1);
                        vst1q_s32(&y_acc[mi][n + 8], acc2);
                        vst1q_s32(&y_acc[mi][n + 12], acc3);
                    }
                }
            }

            // Convert to float
            for (int mi = 0; mi < m_count; mi++) {
                const int m = m_base + mi;
                const float scale = scales[m];
                float* y_row = y + m * N;

                float32x4_t scale_vec = vdupq_n_f32(scale);

                if (bias != nullptr) {
                    for (int n = 0; n + 3 < N; n += 4) {
                        int32x4_t acc = vld1q_s32(&y_acc[mi][n]);
                        float32x4_t b = vld1q_f32(bias + n);
                        vst1q_f32(y_row + n, vaddq_f32(vmulq_f32(vcvtq_f32_s32(acc), scale_vec), b));
                    }
                    for (int n = (N / 4) * 4; n < N; n++) {
                        y_row[n] = static_cast<float>(y_acc[mi][n]) * scale + bias[n];
                    }
                } else {
                    for (int n = 0; n + 3 < N; n += 4) {
                        int32x4_t acc = vld1q_s32(&y_acc[mi][n]);
                        vst1q_f32(y_row + n, vmulq_f32(vcvtq_f32_s32(acc), scale_vec));
                    }
                    for (int n = (N / 4) * 4; n < N; n++) {
                        y_row[n] = static_cast<float>(y_acc[mi][n]) * scale;
                    }
                }
            }
        }
    }
}

/**
 * TBL v11: M-batch=16 + K-unroll x4 + aggressive prefetching
 *
 * Maximum parallelism with deep K-unrolling.
 * Uses 4 K groups at once to maximize LUT reuse.
 */
void matmul_free_neon_tbl_v11(
    torch::Tensor x_int8_tensor,
    torch::Tensor scale_tensor,
    torch::Tensor sign_plane_tensor,
    torch::Tensor value_plane_tensor,
    torch::Tensor y_tensor,
    torch::Tensor bias_tensor,
    int M, int N, int K,
    int num_threads
) {
    const int8_t* __restrict__ x_int8 = x_int8_tensor.data_ptr<int8_t>();
    const float* __restrict__ scales = scale_tensor.data_ptr<float>();
    const uint8_t* __restrict__ sign_plane = sign_plane_tensor.data_ptr<uint8_t>();
    const uint8_t* __restrict__ value_plane = value_plane_tensor.data_ptr<uint8_t>();
    float* __restrict__ y = y_tensor.data_ptr<float>();
    const float* bias = bias_tensor.defined() ? bias_tensor.data_ptr<float>() : nullptr;

    const int K_groups = (K + 3) / 4;
    const int M_BATCH = 16;

    omp_set_num_threads(num_threads);

    #pragma omp parallel
    {
        alignas(64) int32_t y_acc[M_BATCH][2048];

        #pragma omp for schedule(static)
        for (int m_base = 0; m_base < M; m_base += M_BATCH) {
            const int m_end = std::min(m_base + M_BATCH, M);
            const int m_count = m_end - m_base;

            for (int mi = 0; mi < m_count; mi++) {
                memset(y_acc[mi], 0, N * sizeof(int32_t));
            }

            const uint8x16_t mask_0f = vdupq_n_u8(0x0F);

            // Process 4 K groups at once
            int kg = 0;
            for (; kg + 3 < K_groups; kg += 4) {
                const int k_base0 = kg * 4;
                const int k_base1 = (kg + 1) * 4;
                const int k_base2 = (kg + 2) * 4;
                const int k_base3 = (kg + 3) * 4;

                // Build LUTs for all M rows, all 4 K groups
                uint8x16_t lut_lo0[M_BATCH], lut_hi0[M_BATCH];
                uint8x16_t lut_lo1[M_BATCH], lut_hi1[M_BATCH];
                uint8x16_t lut_lo2[M_BATCH], lut_hi2[M_BATCH];
                uint8x16_t lut_lo3[M_BATCH], lut_hi3[M_BATCH];

                for (int mi = 0; mi < m_count; mi++) {
                    const int8_t* x_row = x_int8 + (m_base + mi) * K;

                    int8_t xv0[4] = {0}, xv1[4] = {0}, xv2[4] = {0}, xv3[4] = {0};
                    for (int i = 0; i < 4 && k_base0 + i < K; i++) xv0[i] = x_row[k_base0 + i];
                    for (int i = 0; i < 4 && k_base1 + i < K; i++) xv1[i] = x_row[k_base1 + i];
                    for (int i = 0; i < 4 && k_base2 + i < K; i++) xv2[i] = x_row[k_base2 + i];
                    for (int i = 0; i < 4 && k_base3 + i < K; i++) xv3[i] = x_row[k_base3 + i];

                    build_lut_16_split_neon(xv0, lut_lo0[mi], lut_hi0[mi]);
                    build_lut_16_split_neon(xv1, lut_lo1[mi], lut_hi1[mi]);
                    build_lut_16_split_neon(xv2, lut_lo2[mi], lut_hi2[mi]);
                    build_lut_16_split_neon(xv3, lut_lo3[mi], lut_hi3[mi]);
                }

                const uint8_t* sign_row0 = sign_plane + kg * N;
                const uint8_t* value_row0 = value_plane + kg * N;
                const uint8_t* sign_row1 = sign_plane + (kg + 1) * N;
                const uint8_t* value_row1 = value_plane + (kg + 1) * N;
                const uint8_t* sign_row2 = sign_plane + (kg + 2) * N;
                const uint8_t* value_row2 = value_plane + (kg + 2) * N;
                const uint8_t* sign_row3 = sign_plane + (kg + 3) * N;
                const uint8_t* value_row3 = value_plane + (kg + 3) * N;

                for (int n = 0; n + 15 < N; n += 16) {
                    // Prefetch next cache lines
                    __builtin_prefetch(sign_row0 + n + 128, 0, 1);
                    __builtin_prefetch(value_row0 + n + 128, 0, 1);

                    // Load all indices
                    uint8x16_t s_idx0 = vandq_u8(vld1q_u8(sign_row0 + n), mask_0f);
                    uint8x16_t v_idx0 = vandq_u8(vld1q_u8(value_row0 + n), mask_0f);
                    uint8x16_t s_idx1 = vandq_u8(vld1q_u8(sign_row1 + n), mask_0f);
                    uint8x16_t v_idx1 = vandq_u8(vld1q_u8(value_row1 + n), mask_0f);
                    uint8x16_t s_idx2 = vandq_u8(vld1q_u8(sign_row2 + n), mask_0f);
                    uint8x16_t v_idx2 = vandq_u8(vld1q_u8(value_row2 + n), mask_0f);
                    uint8x16_t s_idx3 = vandq_u8(vld1q_u8(sign_row3 + n), mask_0f);
                    uint8x16_t v_idx3 = vandq_u8(vld1q_u8(value_row3 + n), mask_0f);

                    for (int mi = 0; mi < m_count; mi++) {
                        // TBL lookups for all 4 K groups
                        uint8x16_t s_lo0 = vqtbl1q_u8(lut_lo0[mi], s_idx0);
                        uint8x16_t s_hi0 = vqtbl1q_u8(lut_hi0[mi], s_idx0);
                        uint8x16_t v_lo0 = vqtbl1q_u8(lut_lo0[mi], v_idx0);
                        uint8x16_t v_hi0 = vqtbl1q_u8(lut_hi0[mi], v_idx0);

                        uint8x16_t s_lo1 = vqtbl1q_u8(lut_lo1[mi], s_idx1);
                        uint8x16_t s_hi1 = vqtbl1q_u8(lut_hi1[mi], s_idx1);
                        uint8x16_t v_lo1 = vqtbl1q_u8(lut_lo1[mi], v_idx1);
                        uint8x16_t v_hi1 = vqtbl1q_u8(lut_hi1[mi], v_idx1);

                        uint8x16_t s_lo2 = vqtbl1q_u8(lut_lo2[mi], s_idx2);
                        uint8x16_t s_hi2 = vqtbl1q_u8(lut_hi2[mi], s_idx2);
                        uint8x16_t v_lo2 = vqtbl1q_u8(lut_lo2[mi], v_idx2);
                        uint8x16_t v_hi2 = vqtbl1q_u8(lut_hi2[mi], v_idx2);

                        uint8x16_t s_lo3 = vqtbl1q_u8(lut_lo3[mi], s_idx3);
                        uint8x16_t s_hi3 = vqtbl1q_u8(lut_hi3[mi], s_idx3);
                        uint8x16_t v_lo3 = vqtbl1q_u8(lut_lo3[mi], v_idx3);
                        uint8x16_t v_hi3 = vqtbl1q_u8(lut_hi3[mi], v_idx3);

                        // Unpack all to int16 and compute 2*v - s
                        int16x8_t s16_0a = vreinterpretq_s16_u8(vzip1q_u8(s_lo0, s_hi0));
                        int16x8_t v16_0a = vreinterpretq_s16_u8(vzip1q_u8(v_lo0, v_hi0));
                        int16x8_t s16_0b = vreinterpretq_s16_u8(vzip2q_u8(s_lo0, s_hi0));
                        int16x8_t v16_0b = vreinterpretq_s16_u8(vzip2q_u8(v_lo0, v_hi0));

                        int16x8_t s16_1a = vreinterpretq_s16_u8(vzip1q_u8(s_lo1, s_hi1));
                        int16x8_t v16_1a = vreinterpretq_s16_u8(vzip1q_u8(v_lo1, v_hi1));
                        int16x8_t s16_1b = vreinterpretq_s16_u8(vzip2q_u8(s_lo1, s_hi1));
                        int16x8_t v16_1b = vreinterpretq_s16_u8(vzip2q_u8(v_lo1, v_hi1));

                        int16x8_t s16_2a = vreinterpretq_s16_u8(vzip1q_u8(s_lo2, s_hi2));
                        int16x8_t v16_2a = vreinterpretq_s16_u8(vzip1q_u8(v_lo2, v_hi2));
                        int16x8_t s16_2b = vreinterpretq_s16_u8(vzip2q_u8(s_lo2, s_hi2));
                        int16x8_t v16_2b = vreinterpretq_s16_u8(vzip2q_u8(v_lo2, v_hi2));

                        int16x8_t s16_3a = vreinterpretq_s16_u8(vzip1q_u8(s_lo3, s_hi3));
                        int16x8_t v16_3a = vreinterpretq_s16_u8(vzip1q_u8(v_lo3, v_hi3));
                        int16x8_t s16_3b = vreinterpretq_s16_u8(vzip2q_u8(s_lo3, s_hi3));
                        int16x8_t v16_3b = vreinterpretq_s16_u8(vzip2q_u8(v_lo3, v_hi3));

                        // Compute 2*v - s for all
                        int16x8_t r0a = vsubq_s16(vaddq_s16(v16_0a, v16_0a), s16_0a);
                        int16x8_t r0b = vsubq_s16(vaddq_s16(v16_0b, v16_0b), s16_0b);
                        int16x8_t r1a = vsubq_s16(vaddq_s16(v16_1a, v16_1a), s16_1a);
                        int16x8_t r1b = vsubq_s16(vaddq_s16(v16_1b, v16_1b), s16_1b);
                        int16x8_t r2a = vsubq_s16(vaddq_s16(v16_2a, v16_2a), s16_2a);
                        int16x8_t r2b = vsubq_s16(vaddq_s16(v16_2b, v16_2b), s16_2b);
                        int16x8_t r3a = vsubq_s16(vaddq_s16(v16_3a, v16_3a), s16_3a);
                        int16x8_t r3b = vsubq_s16(vaddq_s16(v16_3b, v16_3b), s16_3b);

                        // Sum all 4 K groups in int16 (safe: 4*508 = 2032 fits in int16)
                        int16x8_t sum_a = vaddq_s16(vaddq_s16(r0a, r1a), vaddq_s16(r2a, r3a));
                        int16x8_t sum_b = vaddq_s16(vaddq_s16(r0b, r1b), vaddq_s16(r2b, r3b));

                        // Widen and accumulate
                        int32x4_t acc0 = vaddq_s32(vld1q_s32(&y_acc[mi][n]), vmovl_s16(vget_low_s16(sum_a)));
                        int32x4_t acc1 = vaddq_s32(vld1q_s32(&y_acc[mi][n + 4]), vmovl_s16(vget_high_s16(sum_a)));
                        int32x4_t acc2 = vaddq_s32(vld1q_s32(&y_acc[mi][n + 8]), vmovl_s16(vget_low_s16(sum_b)));
                        int32x4_t acc3 = vaddq_s32(vld1q_s32(&y_acc[mi][n + 12]), vmovl_s16(vget_high_s16(sum_b)));

                        vst1q_s32(&y_acc[mi][n], acc0);
                        vst1q_s32(&y_acc[mi][n + 4], acc1);
                        vst1q_s32(&y_acc[mi][n + 8], acc2);
                        vst1q_s32(&y_acc[mi][n + 12], acc3);
                    }
                }

                // N remainder scalar
                for (int n = (N / 16) * 16; n < N; n++) {
                    for (int mi = 0; mi < m_count; mi++) {
                        const int8_t* x_row = x_int8 + (m_base + mi) * K;
                        alignas(32) int16_t lut0[16], lut1[16], lut2[16], lut3[16];
                        int8_t xv[4];

                        for (int i = 0; i < 4 && k_base0 + i < K; i++) xv[i] = x_row[k_base0 + i];
                        build_lut_16_int16(xv, lut0);
                        for (int i = 0; i < 4; i++) xv[i] = 0;
                        for (int i = 0; i < 4 && k_base1 + i < K; i++) xv[i] = x_row[k_base1 + i];
                        build_lut_16_int16(xv, lut1);
                        for (int i = 0; i < 4; i++) xv[i] = 0;
                        for (int i = 0; i < 4 && k_base2 + i < K; i++) xv[i] = x_row[k_base2 + i];
                        build_lut_16_int16(xv, lut2);
                        for (int i = 0; i < 4; i++) xv[i] = 0;
                        for (int i = 0; i < 4 && k_base3 + i < K; i++) xv[i] = x_row[k_base3 + i];
                        build_lut_16_int16(xv, lut3);

                        y_acc[mi][n] += 2 * lut0[value_row0[n] & 0x0F] - lut0[sign_row0[n] & 0x0F];
                        y_acc[mi][n] += 2 * lut1[value_row1[n] & 0x0F] - lut1[sign_row1[n] & 0x0F];
                        y_acc[mi][n] += 2 * lut2[value_row2[n] & 0x0F] - lut2[sign_row2[n] & 0x0F];
                        y_acc[mi][n] += 2 * lut3[value_row3[n] & 0x0F] - lut3[sign_row3[n] & 0x0F];
                    }
                }
            }

            // Handle remaining K groups
            for (; kg < K_groups; kg++) {
                const int k_base = kg * 4;
                uint8x16_t lut_lo[M_BATCH], lut_hi[M_BATCH];

                for (int mi = 0; mi < m_count; mi++) {
                    const int8_t* x_row = x_int8 + (m_base + mi) * K;
                    int8_t xv[4] = {0};
                    for (int i = 0; i < 4 && k_base + i < K; i++) xv[i] = x_row[k_base + i];
                    build_lut_16_split_neon(xv, lut_lo[mi], lut_hi[mi]);
                }

                const uint8_t* sign_row = sign_plane + kg * N;
                const uint8_t* value_row = value_plane + kg * N;

                for (int n = 0; n + 15 < N; n += 16) {
                    uint8x16_t s_idx = vandq_u8(vld1q_u8(sign_row + n), mask_0f);
                    uint8x16_t v_idx = vandq_u8(vld1q_u8(value_row + n), mask_0f);

                    for (int mi = 0; mi < m_count; mi++) {
                        uint8x16_t s_lo = vqtbl1q_u8(lut_lo[mi], s_idx);
                        uint8x16_t s_hi = vqtbl1q_u8(lut_hi[mi], s_idx);
                        uint8x16_t v_lo = vqtbl1q_u8(lut_lo[mi], v_idx);
                        uint8x16_t v_hi = vqtbl1q_u8(lut_hi[mi], v_idx);

                        int16x8_t s16_a = vreinterpretq_s16_u8(vzip1q_u8(s_lo, s_hi));
                        int16x8_t v16_a = vreinterpretq_s16_u8(vzip1q_u8(v_lo, v_hi));
                        int16x8_t s16_b = vreinterpretq_s16_u8(vzip2q_u8(s_lo, s_hi));
                        int16x8_t v16_b = vreinterpretq_s16_u8(vzip2q_u8(v_lo, v_hi));

                        int16x8_t res_a = vsubq_s16(vaddq_s16(v16_a, v16_a), s16_a);
                        int16x8_t res_b = vsubq_s16(vaddq_s16(v16_b, v16_b), s16_b);

                        int32x4_t acc0 = vaddq_s32(vld1q_s32(&y_acc[mi][n]), vmovl_s16(vget_low_s16(res_a)));
                        int32x4_t acc1 = vaddq_s32(vld1q_s32(&y_acc[mi][n + 4]), vmovl_s16(vget_high_s16(res_a)));
                        int32x4_t acc2 = vaddq_s32(vld1q_s32(&y_acc[mi][n + 8]), vmovl_s16(vget_low_s16(res_b)));
                        int32x4_t acc3 = vaddq_s32(vld1q_s32(&y_acc[mi][n + 12]), vmovl_s16(vget_high_s16(res_b)));

                        vst1q_s32(&y_acc[mi][n], acc0);
                        vst1q_s32(&y_acc[mi][n + 4], acc1);
                        vst1q_s32(&y_acc[mi][n + 8], acc2);
                        vst1q_s32(&y_acc[mi][n + 12], acc3);
                    }
                }
            }

            // Convert to float
            for (int mi = 0; mi < m_count; mi++) {
                const int m = m_base + mi;
                const float scale = scales[m];
                float* y_row = y + m * N;
                float32x4_t scale_vec = vdupq_n_f32(scale);

                if (bias != nullptr) {
                    for (int n = 0; n + 3 < N; n += 4) {
                        int32x4_t acc = vld1q_s32(&y_acc[mi][n]);
                        float32x4_t b = vld1q_f32(bias + n);
                        vst1q_f32(y_row + n, vaddq_f32(vmulq_f32(vcvtq_f32_s32(acc), scale_vec), b));
                    }
                    for (int n = (N / 4) * 4; n < N; n++) {
                        y_row[n] = static_cast<float>(y_acc[mi][n]) * scale + bias[n];
                    }
                } else {
                    for (int n = 0; n + 3 < N; n += 4) {
                        int32x4_t acc = vld1q_s32(&y_acc[mi][n]);
                        vst1q_f32(y_row + n, vmulq_f32(vcvtq_f32_s32(acc), scale_vec));
                    }
                    for (int n = (N / 4) * 4; n < N; n++) {
                        y_row[n] = static_cast<float>(y_acc[mi][n]) * scale;
                    }
                }
            }
        }
    }
}

/**
 * TBL v12: True g=8 using vqtbl2q_u8 for 32-byte table
 *
 * Processes 8 K elements at once instead of 4.
 * Uses vqtbl2q_u8 which can handle 32-byte tables (indices 0-31).
 *
 * For 8 elements, we need 2^8 = 256 entries, but we only have 32.
 * So we split into two g=4 lookups but with better memory access pattern.
 *
 * Alternative approach: Use two cascaded tables for g=8.
 */
void matmul_free_neon_tbl_v12(
    torch::Tensor x_int8_tensor,
    torch::Tensor scale_tensor,
    torch::Tensor sign_plane_tensor,
    torch::Tensor value_plane_tensor,
    torch::Tensor y_tensor,
    torch::Tensor bias_tensor,
    int M, int N, int K,
    int num_threads
) {
    const int8_t* __restrict__ x_int8 = x_int8_tensor.data_ptr<int8_t>();
    const float* __restrict__ scales = scale_tensor.data_ptr<float>();
    const uint8_t* __restrict__ sign_plane = sign_plane_tensor.data_ptr<uint8_t>();
    const uint8_t* __restrict__ value_plane = value_plane_tensor.data_ptr<uint8_t>();
    float* __restrict__ y = y_tensor.data_ptr<float>();
    const float* bias = bias_tensor.defined() ? bias_tensor.data_ptr<float>() : nullptr;

    const int K_groups = (K + 3) / 4;
    const int M_BATCH = 8;

    omp_set_num_threads(num_threads);

    #pragma omp parallel
    {
        // Use register-resident accumulators for hot N blocks
        alignas(64) int32_t y_acc[M_BATCH][2048];

        #pragma omp for schedule(static)
        for (int m_base = 0; m_base < M; m_base += M_BATCH) {
            const int m_end = std::min(m_base + M_BATCH, M);
            const int m_count = m_end - m_base;

            for (int mi = 0; mi < m_count; mi++) {
                memset(y_acc[mi], 0, N * sizeof(int32_t));
            }

            const uint8x16_t mask_0f = vdupq_n_u8(0x0F);

            // Process K groups with dual-table approach
            // Pack 2 g=4 LUTs into a single vqtbl2q operation for efficiency
            int kg = 0;
            for (; kg + 1 < K_groups; kg += 2) {
                const int k_base0 = kg * 4;
                const int k_base1 = (kg + 1) * 4;

                // Build combined LUTs for vqtbl2q (32 entries per LUT pair)
                uint8x16x2_t lut_lo_pair[M_BATCH], lut_hi_pair[M_BATCH];

                for (int mi = 0; mi < m_count; mi++) {
                    const int8_t* x_row = x_int8 + (m_base + mi) * K;

                    int8_t xv0[4] = {0}, xv1[4] = {0};
                    for (int i = 0; i < 4 && k_base0 + i < K; i++) xv0[i] = x_row[k_base0 + i];
                    for (int i = 0; i < 4 && k_base1 + i < K; i++) xv1[i] = x_row[k_base1 + i];

                    uint8x16_t lo0, hi0, lo1, hi1;
                    build_lut_16_split_neon(xv0, lo0, hi0);
                    build_lut_16_split_neon(xv1, lo1, hi1);

                    // Pack into 2-register table for vqtbl2q
                    lut_lo_pair[mi].val[0] = lo0;
                    lut_lo_pair[mi].val[1] = lo1;
                    lut_hi_pair[mi].val[0] = hi0;
                    lut_hi_pair[mi].val[1] = hi1;
                }

                const uint8_t* sign_row0 = sign_plane + kg * N;
                const uint8_t* value_row0 = value_plane + kg * N;
                const uint8_t* sign_row1 = sign_plane + (kg + 1) * N;
                const uint8_t* value_row1 = value_plane + (kg + 1) * N;

                for (int n = 0; n + 15 < N; n += 16) {
                    __builtin_prefetch(sign_row0 + n + 64, 0, 2);
                    __builtin_prefetch(value_row0 + n + 64, 0, 2);

                    // Load indices
                    uint8x16_t s_idx0 = vandq_u8(vld1q_u8(sign_row0 + n), mask_0f);
                    uint8x16_t v_idx0 = vandq_u8(vld1q_u8(value_row0 + n), mask_0f);
                    uint8x16_t s_idx1_raw = vandq_u8(vld1q_u8(sign_row1 + n), mask_0f);
                    uint8x16_t v_idx1_raw = vandq_u8(vld1q_u8(value_row1 + n), mask_0f);

                    // For vqtbl2q, indices 0-15 go to first table, 16-31 to second
                    // Add 16 to second group indices
                    uint8x16_t offset_16 = vdupq_n_u8(16);
                    uint8x16_t s_idx1 = vaddq_u8(s_idx1_raw, offset_16);
                    uint8x16_t v_idx1 = vaddq_u8(v_idx1_raw, offset_16);

                    for (int mi = 0; mi < m_count; mi++) {
                        // Use vqtbl2q for both K groups in single lookup
                        uint8x16_t s_lo0 = vqtbl2q_u8(lut_lo_pair[mi], s_idx0);
                        uint8x16_t s_hi0 = vqtbl2q_u8(lut_hi_pair[mi], s_idx0);
                        uint8x16_t v_lo0 = vqtbl2q_u8(lut_lo_pair[mi], v_idx0);
                        uint8x16_t v_hi0 = vqtbl2q_u8(lut_hi_pair[mi], v_idx0);

                        uint8x16_t s_lo1 = vqtbl2q_u8(lut_lo_pair[mi], s_idx1);
                        uint8x16_t s_hi1 = vqtbl2q_u8(lut_hi_pair[mi], s_idx1);
                        uint8x16_t v_lo1 = vqtbl2q_u8(lut_lo_pair[mi], v_idx1);
                        uint8x16_t v_hi1 = vqtbl2q_u8(lut_hi_pair[mi], v_idx1);

                        // Unpack to int16
                        int16x8_t s16_0a = vreinterpretq_s16_u8(vzip1q_u8(s_lo0, s_hi0));
                        int16x8_t v16_0a = vreinterpretq_s16_u8(vzip1q_u8(v_lo0, v_hi0));
                        int16x8_t s16_0b = vreinterpretq_s16_u8(vzip2q_u8(s_lo0, s_hi0));
                        int16x8_t v16_0b = vreinterpretq_s16_u8(vzip2q_u8(v_lo0, v_hi0));

                        int16x8_t s16_1a = vreinterpretq_s16_u8(vzip1q_u8(s_lo1, s_hi1));
                        int16x8_t v16_1a = vreinterpretq_s16_u8(vzip1q_u8(v_lo1, v_hi1));
                        int16x8_t s16_1b = vreinterpretq_s16_u8(vzip2q_u8(s_lo1, s_hi1));
                        int16x8_t v16_1b = vreinterpretq_s16_u8(vzip2q_u8(v_lo1, v_hi1));

                        // Compute 2*v - s
                        int16x8_t r0a = vsubq_s16(vaddq_s16(v16_0a, v16_0a), s16_0a);
                        int16x8_t r0b = vsubq_s16(vaddq_s16(v16_0b, v16_0b), s16_0b);
                        int16x8_t r1a = vsubq_s16(vaddq_s16(v16_1a, v16_1a), s16_1a);
                        int16x8_t r1b = vsubq_s16(vaddq_s16(v16_1b, v16_1b), s16_1b);

                        // Sum both K groups
                        int16x8_t sum_a = vaddq_s16(r0a, r1a);
                        int16x8_t sum_b = vaddq_s16(r0b, r1b);

                        // Accumulate
                        int32x4_t acc0 = vaddq_s32(vld1q_s32(&y_acc[mi][n]), vmovl_s16(vget_low_s16(sum_a)));
                        int32x4_t acc1 = vaddq_s32(vld1q_s32(&y_acc[mi][n + 4]), vmovl_s16(vget_high_s16(sum_a)));
                        int32x4_t acc2 = vaddq_s32(vld1q_s32(&y_acc[mi][n + 8]), vmovl_s16(vget_low_s16(sum_b)));
                        int32x4_t acc3 = vaddq_s32(vld1q_s32(&y_acc[mi][n + 12]), vmovl_s16(vget_high_s16(sum_b)));

                        vst1q_s32(&y_acc[mi][n], acc0);
                        vst1q_s32(&y_acc[mi][n + 4], acc1);
                        vst1q_s32(&y_acc[mi][n + 8], acc2);
                        vst1q_s32(&y_acc[mi][n + 12], acc3);
                    }
                }

                // N remainder
                for (int n = (N / 16) * 16; n < N; n++) {
                    for (int mi = 0; mi < m_count; mi++) {
                        const int8_t* x_row = x_int8 + (m_base + mi) * K;
                        alignas(32) int16_t lut0[16], lut1[16];
                        int8_t xv0[4] = {0}, xv1[4] = {0};
                        for (int i = 0; i < 4 && k_base0 + i < K; i++) xv0[i] = x_row[k_base0 + i];
                        for (int i = 0; i < 4 && k_base1 + i < K; i++) xv1[i] = x_row[k_base1 + i];
                        build_lut_16_int16(xv0, lut0);
                        build_lut_16_int16(xv1, lut1);
                        y_acc[mi][n] += 2 * lut0[value_row0[n] & 0x0F] - lut0[sign_row0[n] & 0x0F];
                        y_acc[mi][n] += 2 * lut1[value_row1[n] & 0x0F] - lut1[sign_row1[n] & 0x0F];
                    }
                }
            }

            // Handle remaining K group
            for (; kg < K_groups; kg++) {
                const int k_base = kg * 4;
                uint8x16_t lut_lo[M_BATCH], lut_hi[M_BATCH];

                for (int mi = 0; mi < m_count; mi++) {
                    const int8_t* x_row = x_int8 + (m_base + mi) * K;
                    int8_t xv[4] = {0};
                    for (int i = 0; i < 4 && k_base + i < K; i++) xv[i] = x_row[k_base + i];
                    build_lut_16_split_neon(xv, lut_lo[mi], lut_hi[mi]);
                }

                const uint8_t* sign_row = sign_plane + kg * N;
                const uint8_t* value_row = value_plane + kg * N;

                for (int n = 0; n + 15 < N; n += 16) {
                    uint8x16_t s_idx = vandq_u8(vld1q_u8(sign_row + n), mask_0f);
                    uint8x16_t v_idx = vandq_u8(vld1q_u8(value_row + n), mask_0f);

                    for (int mi = 0; mi < m_count; mi++) {
                        uint8x16_t s_lo = vqtbl1q_u8(lut_lo[mi], s_idx);
                        uint8x16_t s_hi = vqtbl1q_u8(lut_hi[mi], s_idx);
                        uint8x16_t v_lo = vqtbl1q_u8(lut_lo[mi], v_idx);
                        uint8x16_t v_hi = vqtbl1q_u8(lut_hi[mi], v_idx);

                        int16x8_t s16_a = vreinterpretq_s16_u8(vzip1q_u8(s_lo, s_hi));
                        int16x8_t v16_a = vreinterpretq_s16_u8(vzip1q_u8(v_lo, v_hi));
                        int16x8_t s16_b = vreinterpretq_s16_u8(vzip2q_u8(s_lo, s_hi));
                        int16x8_t v16_b = vreinterpretq_s16_u8(vzip2q_u8(v_lo, v_hi));

                        int16x8_t res_a = vsubq_s16(vaddq_s16(v16_a, v16_a), s16_a);
                        int16x8_t res_b = vsubq_s16(vaddq_s16(v16_b, v16_b), s16_b);

                        int32x4_t acc0 = vaddq_s32(vld1q_s32(&y_acc[mi][n]), vmovl_s16(vget_low_s16(res_a)));
                        int32x4_t acc1 = vaddq_s32(vld1q_s32(&y_acc[mi][n + 4]), vmovl_s16(vget_high_s16(res_a)));
                        int32x4_t acc2 = vaddq_s32(vld1q_s32(&y_acc[mi][n + 8]), vmovl_s16(vget_low_s16(res_b)));
                        int32x4_t acc3 = vaddq_s32(vld1q_s32(&y_acc[mi][n + 12]), vmovl_s16(vget_high_s16(res_b)));

                        vst1q_s32(&y_acc[mi][n], acc0);
                        vst1q_s32(&y_acc[mi][n + 4], acc1);
                        vst1q_s32(&y_acc[mi][n + 8], acc2);
                        vst1q_s32(&y_acc[mi][n + 12], acc3);
                    }
                }
            }

            // Convert to float
            for (int mi = 0; mi < m_count; mi++) {
                const int m = m_base + mi;
                const float scale = scales[m];
                float* y_row = y + m * N;
                float32x4_t scale_vec = vdupq_n_f32(scale);

                if (bias != nullptr) {
                    for (int n = 0; n + 3 < N; n += 4) {
                        int32x4_t acc = vld1q_s32(&y_acc[mi][n]);
                        float32x4_t b = vld1q_f32(bias + n);
                        vst1q_f32(y_row + n, vaddq_f32(vmulq_f32(vcvtq_f32_s32(acc), scale_vec), b));
                    }
                    for (int n = (N / 4) * 4; n < N; n++) {
                        y_row[n] = static_cast<float>(y_acc[mi][n]) * scale + bias[n];
                    }
                } else {
                    for (int n = 0; n + 3 < N; n += 4) {
                        int32x4_t acc = vld1q_s32(&y_acc[mi][n]);
                        vst1q_f32(y_row + n, vmulq_f32(vcvtq_f32_s32(acc), scale_vec));
                    }
                    for (int n = (N / 4) * 4; n < N; n++) {
                        y_row[n] = static_cast<float>(y_acc[mi][n]) * scale;
                    }
                }
            }
        }
    }
}

/**
 * TBL v13: Direct ternary encoding with single LUT
 *
 * Instead of separate sign/value planes with 2*v - s computation,
 * encode ternary weights directly as 2-bit values:
 *   0 -> 0, 1 -> +1, 2 -> -1
 *
 * This reduces TBL lookups from 4 to 2 per 16 outputs!
 * LUT directly contains partial sums for each combination.
 *
 * For g=4, each LUT entry is: sum of x[i] * w[i] for 4 weights
 * Index encodes 4 weights as: idx = w0 + 3*w1 + 9*w2 + 27*w3 (base-3)
 * So we need 3^4 = 81 entries (fits in vqtbl4q with 64 + scatter for 17 more)
 *
 * Simplified approach: use g=2 with only 9 entries (fits in single TBL)
 */
void pack_ternary_direct_g2(
    torch::Tensor w_tensor,
    torch::Tensor packed_tensor,  // [K_groups, N] uint8, K_groups = ceil(K/2)
    int N, int K
) {
    const float* w = w_tensor.data_ptr<float>();
    uint8_t* packed = packed_tensor.data_ptr<uint8_t>();

    const int K_groups = (K + 1) / 2;

    #pragma omp parallel for collapse(2)
    for (int kg = 0; kg < K_groups; kg++) {
        for (int n = 0; n < N; n++) {
            uint8_t idx = 0;
            for (int i = 0; i < 2 && (kg * 2 + i) < K; i++) {
                float val = w[n * K + kg * 2 + i];
                uint8_t w_enc;
                if (val > 0.5f) {
                    w_enc = 1;  // +1
                } else if (val < -0.5f) {
                    w_enc = 2;  // -1
                } else {
                    w_enc = 0;  // 0
                }
                // Base-3 encoding
                if (i == 0) idx += w_enc;
                else idx += 3 * w_enc;
            }
            packed[kg * N + n] = idx;
        }
    }
}

void matmul_free_neon_tbl_v13(
    torch::Tensor x_int8_tensor,
    torch::Tensor scale_tensor,
    torch::Tensor packed_tensor,   // [K_groups, N] uint8 - direct ternary g=2
    torch::Tensor y_tensor,
    torch::Tensor bias_tensor,
    int M, int N, int K,
    int num_threads
) {
    const int8_t* __restrict__ x_int8 = x_int8_tensor.data_ptr<int8_t>();
    const float* __restrict__ scales = scale_tensor.data_ptr<float>();
    const uint8_t* __restrict__ packed = packed_tensor.data_ptr<uint8_t>();
    float* __restrict__ y = y_tensor.data_ptr<float>();
    const float* bias = bias_tensor.defined() ? bias_tensor.data_ptr<float>() : nullptr;

    const int K_groups = (K + 1) / 2;
    const int M_BATCH = 8;

    omp_set_num_threads(num_threads);

    #pragma omp parallel
    {
        alignas(64) int32_t y_acc[M_BATCH][2048];

        #pragma omp for schedule(static)
        for (int m_base = 0; m_base < M; m_base += M_BATCH) {
            const int m_end = std::min(m_base + M_BATCH, M);
            const int m_count = m_end - m_base;

            for (int mi = 0; mi < m_count; mi++) {
                memset(y_acc[mi], 0, N * sizeof(int32_t));
            }

            for (int kg = 0; kg < K_groups; kg++) {
                const int k_base = kg * 2;

                // Build 9-entry LUT for each M row
                // LUT[idx] = x[0]*w[0] + x[1]*w[1] where idx = w[0] + 3*w[1]
                // w encoding: 0->0, 1->+1, 2->-1
                uint8x16_t lut_lo[M_BATCH], lut_hi[M_BATCH];

                for (int mi = 0; mi < m_count; mi++) {
                    const int8_t* x_row = x_int8 + (m_base + mi) * K;
                    int16_t x0 = (k_base < K) ? x_row[k_base] : 0;
                    int16_t x1 = (k_base + 1 < K) ? x_row[k_base + 1] : 0;

                    // Build 9-entry LUT (only indices 0-8 used)
                    // idx = w0 + 3*w1
                    // w0/w1 in {0, 1, 2} -> {0, +1, -1}
                    alignas(16) int16_t lut[16] = {0};

                    // w0=0, w1=0: idx=0, sum=0
                    lut[0] = 0;
                    // w0=1, w1=0: idx=1, sum=x0
                    lut[1] = x0;
                    // w0=2, w1=0: idx=2, sum=-x0
                    lut[2] = -x0;
                    // w0=0, w1=1: idx=3, sum=x1
                    lut[3] = x1;
                    // w0=1, w1=1: idx=4, sum=x0+x1
                    lut[4] = x0 + x1;
                    // w0=2, w1=1: idx=5, sum=-x0+x1
                    lut[5] = -x0 + x1;
                    // w0=0, w1=2: idx=6, sum=-x1
                    lut[6] = -x1;
                    // w0=1, w1=2: idx=7, sum=x0-x1
                    lut[7] = x0 - x1;
                    // w0=2, w1=2: idx=8, sum=-x0-x1
                    lut[8] = -x0 - x1;

                    // Split into lo/hi bytes
                    alignas(16) uint8_t lo[16], hi[16];
                    for (int i = 0; i < 16; i++) {
                        lo[i] = static_cast<uint8_t>(lut[i] & 0xFF);
                        hi[i] = static_cast<uint8_t>((lut[i] >> 8) & 0xFF);
                    }
                    lut_lo[mi] = vld1q_u8(lo);
                    lut_hi[mi] = vld1q_u8(hi);
                }

                const uint8_t* packed_row = packed + kg * N;

                // Single TBL lookup per 16 outputs (instead of 4)!
                for (int n = 0; n + 15 < N; n += 16) {
                    uint8x16_t idx = vld1q_u8(packed_row + n);
                    // No need to mask - indices are already 0-8

                    for (int mi = 0; mi < m_count; mi++) {
                        // Only 2 TBL lookups instead of 4!
                        uint8x16_t res_lo = vqtbl1q_u8(lut_lo[mi], idx);
                        uint8x16_t res_hi = vqtbl1q_u8(lut_hi[mi], idx);

                        // Unpack to int16
                        int16x8_t res_a = vreinterpretq_s16_u8(vzip1q_u8(res_lo, res_hi));
                        int16x8_t res_b = vreinterpretq_s16_u8(vzip2q_u8(res_lo, res_hi));

                        // Widen and accumulate
                        int32x4_t acc0 = vaddq_s32(vld1q_s32(&y_acc[mi][n]), vmovl_s16(vget_low_s16(res_a)));
                        int32x4_t acc1 = vaddq_s32(vld1q_s32(&y_acc[mi][n + 4]), vmovl_s16(vget_high_s16(res_a)));
                        int32x4_t acc2 = vaddq_s32(vld1q_s32(&y_acc[mi][n + 8]), vmovl_s16(vget_low_s16(res_b)));
                        int32x4_t acc3 = vaddq_s32(vld1q_s32(&y_acc[mi][n + 12]), vmovl_s16(vget_high_s16(res_b)));

                        vst1q_s32(&y_acc[mi][n], acc0);
                        vst1q_s32(&y_acc[mi][n + 4], acc1);
                        vst1q_s32(&y_acc[mi][n + 8], acc2);
                        vst1q_s32(&y_acc[mi][n + 12], acc3);
                    }
                }

                // N remainder
                for (int n = (N / 16) * 16; n < N; n++) {
                    uint8_t idx = packed_row[n];
                    for (int mi = 0; mi < m_count; mi++) {
                        const int8_t* x_row = x_int8 + (m_base + mi) * K;
                        int16_t x0 = (k_base < K) ? x_row[k_base] : 0;
                        int16_t x1 = (k_base + 1 < K) ? x_row[k_base + 1] : 0;

                        int16_t lut[9];
                        lut[0] = 0; lut[1] = x0; lut[2] = -x0;
                        lut[3] = x1; lut[4] = x0 + x1; lut[5] = -x0 + x1;
                        lut[6] = -x1; lut[7] = x0 - x1; lut[8] = -x0 - x1;

                        y_acc[mi][n] += lut[idx];
                    }
                }
            }

            // Convert to float
            for (int mi = 0; mi < m_count; mi++) {
                const int m = m_base + mi;
                const float scale = scales[m];
                float* y_row = y + m * N;
                float32x4_t scale_vec = vdupq_n_f32(scale);

                if (bias != nullptr) {
                    for (int n = 0; n + 3 < N; n += 4) {
                        int32x4_t acc = vld1q_s32(&y_acc[mi][n]);
                        float32x4_t b = vld1q_f32(bias + n);
                        vst1q_f32(y_row + n, vaddq_f32(vmulq_f32(vcvtq_f32_s32(acc), scale_vec), b));
                    }
                    for (int n = (N / 4) * 4; n < N; n++) {
                        y_row[n] = static_cast<float>(y_acc[mi][n]) * scale + bias[n];
                    }
                } else {
                    for (int n = 0; n + 3 < N; n += 4) {
                        int32x4_t acc = vld1q_s32(&y_acc[mi][n]);
                        vst1q_f32(y_row + n, vmulq_f32(vcvtq_f32_s32(acc), scale_vec));
                    }
                    for (int n = (N / 4) * 4; n < N; n++) {
                        y_row[n] = static_cast<float>(y_acc[mi][n]) * scale;
                    }
                }
            }
        }
    }
}

/**
 * TBL v14: Direct ternary g=4 with vqtbl4q_u8 (81 entries in 64+17)
 *
 * Uses 4 K elements per group, requiring 3^4=81 LUT entries.
 * vqtbl4q can handle 64 entries (indices 0-63), so we split:
 * - First 64 entries via vqtbl4q
 * - Remaining 17 entries via separate lookup
 */
void pack_ternary_direct_g4(
    torch::Tensor w_tensor,
    torch::Tensor packed_tensor,  // [K_groups, N] uint8, K_groups = ceil(K/4)
    int N, int K
) {
    const float* w = w_tensor.data_ptr<float>();
    uint8_t* packed = packed_tensor.data_ptr<uint8_t>();

    const int K_groups = (K + 3) / 4;

    #pragma omp parallel for collapse(2)
    for (int kg = 0; kg < K_groups; kg++) {
        for (int n = 0; n < N; n++) {
            uint8_t idx = 0;
            uint8_t base = 1;
            for (int i = 0; i < 4 && (kg * 4 + i) < K; i++) {
                float val = w[n * K + kg * 4 + i];
                uint8_t w_enc;
                if (val > 0.5f) {
                    w_enc = 1;  // +1
                } else if (val < -0.5f) {
                    w_enc = 2;  // -1
                } else {
                    w_enc = 0;  // 0
                }
                idx += base * w_enc;
                base *= 3;
            }
            packed[kg * N + n] = idx;
        }
    }
}

void matmul_free_neon_tbl_v14(
    torch::Tensor x_int8_tensor,
    torch::Tensor scale_tensor,
    torch::Tensor packed_tensor,   // [K_groups, N] uint8 - direct ternary g=4
    torch::Tensor y_tensor,
    torch::Tensor bias_tensor,
    int M, int N, int K,
    int num_threads
) {
    const int8_t* __restrict__ x_int8 = x_int8_tensor.data_ptr<int8_t>();
    const float* __restrict__ scales = scale_tensor.data_ptr<float>();
    const uint8_t* __restrict__ packed = packed_tensor.data_ptr<uint8_t>();
    float* __restrict__ y = y_tensor.data_ptr<float>();
    const float* bias = bias_tensor.defined() ? bias_tensor.data_ptr<float>() : nullptr;

    const int K_groups = (K + 3) / 4;
    const int M_BATCH = 8;

    omp_set_num_threads(num_threads);

    #pragma omp parallel
    {
        alignas(64) int32_t y_acc[M_BATCH][2048];

        #pragma omp for schedule(static)
        for (int m_base = 0; m_base < M; m_base += M_BATCH) {
            const int m_end = std::min(m_base + M_BATCH, M);
            const int m_count = m_end - m_base;

            for (int mi = 0; mi < m_count; mi++) {
                memset(y_acc[mi], 0, N * sizeof(int32_t));
            }

            for (int kg = 0; kg < K_groups; kg++) {
                const int k_base = kg * 4;

                // Build 81-entry LUT for each M row
                // Split into vqtbl4q (64 entries) + overflow (17 entries)
                uint8x16x4_t lut_lo_4[M_BATCH], lut_hi_4[M_BATCH];
                uint8x16_t lut_lo_overflow[M_BATCH], lut_hi_overflow[M_BATCH];

                for (int mi = 0; mi < m_count; mi++) {
                    const int8_t* x_row = x_int8 + (m_base + mi) * K;
                    int16_t x0 = (k_base < K) ? x_row[k_base] : 0;
                    int16_t x1 = (k_base + 1 < K) ? x_row[k_base + 1] : 0;
                    int16_t x2 = (k_base + 2 < K) ? x_row[k_base + 2] : 0;
                    int16_t x3 = (k_base + 3 < K) ? x_row[k_base + 3] : 0;

                    // Weight decoding: 0->0, 1->+1, 2->-1
                    int16_t w_val[3] = {0, 1, -1};

                    // Build all 81 entries
                    alignas(64) int16_t lut[81];
                    for (int w3 = 0; w3 < 3; w3++) {
                        for (int w2 = 0; w2 < 3; w2++) {
                            for (int w1 = 0; w1 < 3; w1++) {
                                for (int w0 = 0; w0 < 3; w0++) {
                                    int idx = w0 + 3*w1 + 9*w2 + 27*w3;
                                    lut[idx] = x0*w_val[w0] + x1*w_val[w1] + x2*w_val[w2] + x3*w_val[w3];
                                }
                            }
                        }
                    }

                    // Pack first 64 entries into vqtbl4q format
                    alignas(16) uint8_t lo_64[64], hi_64[64];
                    for (int i = 0; i < 64; i++) {
                        lo_64[i] = static_cast<uint8_t>(lut[i] & 0xFF);
                        hi_64[i] = static_cast<uint8_t>((lut[i] >> 8) & 0xFF);
                    }
                    lut_lo_4[mi].val[0] = vld1q_u8(lo_64);
                    lut_lo_4[mi].val[1] = vld1q_u8(lo_64 + 16);
                    lut_lo_4[mi].val[2] = vld1q_u8(lo_64 + 32);
                    lut_lo_4[mi].val[3] = vld1q_u8(lo_64 + 48);
                    lut_hi_4[mi].val[0] = vld1q_u8(hi_64);
                    lut_hi_4[mi].val[1] = vld1q_u8(hi_64 + 16);
                    lut_hi_4[mi].val[2] = vld1q_u8(hi_64 + 32);
                    lut_hi_4[mi].val[3] = vld1q_u8(hi_64 + 48);

                    // Pack overflow entries (64-80) into single register
                    alignas(16) uint8_t lo_of[16] = {0}, hi_of[16] = {0};
                    for (int i = 64; i < 81; i++) {
                        lo_of[i - 64] = static_cast<uint8_t>(lut[i] & 0xFF);
                        hi_of[i - 64] = static_cast<uint8_t>((lut[i] >> 8) & 0xFF);
                    }
                    lut_lo_overflow[mi] = vld1q_u8(lo_of);
                    lut_hi_overflow[mi] = vld1q_u8(hi_of);
                }

                const uint8_t* packed_row = packed + kg * N;
                uint8x16_t threshold_64 = vdupq_n_u8(64);

                for (int n = 0; n + 15 < N; n += 16) {
                    uint8x16_t idx = vld1q_u8(packed_row + n);

                    for (int mi = 0; mi < m_count; mi++) {
                        // Check which indices are >= 64
                        uint8x16_t is_overflow = vcgeq_u8(idx, threshold_64);

                        // Main lookup (indices 0-63)
                        uint8x16_t res_lo_main = vqtbl4q_u8(lut_lo_4[mi], idx);
                        uint8x16_t res_hi_main = vqtbl4q_u8(lut_hi_4[mi], idx);

                        // Overflow lookup (indices 64-80)
                        uint8x16_t idx_overflow = vsubq_u8(idx, threshold_64);
                        uint8x16_t res_lo_of = vqtbl1q_u8(lut_lo_overflow[mi], idx_overflow);
                        uint8x16_t res_hi_of = vqtbl1q_u8(lut_hi_overflow[mi], idx_overflow);

                        // Select based on overflow mask
                        uint8x16_t res_lo = vbslq_u8(is_overflow, res_lo_of, res_lo_main);
                        uint8x16_t res_hi = vbslq_u8(is_overflow, res_hi_of, res_hi_main);

                        // Unpack to int16
                        int16x8_t res_a = vreinterpretq_s16_u8(vzip1q_u8(res_lo, res_hi));
                        int16x8_t res_b = vreinterpretq_s16_u8(vzip2q_u8(res_lo, res_hi));

                        // Accumulate
                        int32x4_t acc0 = vaddq_s32(vld1q_s32(&y_acc[mi][n]), vmovl_s16(vget_low_s16(res_a)));
                        int32x4_t acc1 = vaddq_s32(vld1q_s32(&y_acc[mi][n + 4]), vmovl_s16(vget_high_s16(res_a)));
                        int32x4_t acc2 = vaddq_s32(vld1q_s32(&y_acc[mi][n + 8]), vmovl_s16(vget_low_s16(res_b)));
                        int32x4_t acc3 = vaddq_s32(vld1q_s32(&y_acc[mi][n + 12]), vmovl_s16(vget_high_s16(res_b)));

                        vst1q_s32(&y_acc[mi][n], acc0);
                        vst1q_s32(&y_acc[mi][n + 4], acc1);
                        vst1q_s32(&y_acc[mi][n + 8], acc2);
                        vst1q_s32(&y_acc[mi][n + 12], acc3);
                    }
                }

                // N remainder
                for (int n = (N / 16) * 16; n < N; n++) {
                    uint8_t idx = packed_row[n];
                    for (int mi = 0; mi < m_count; mi++) {
                        const int8_t* x_row = x_int8 + (m_base + mi) * K;
                        int16_t x0 = (k_base < K) ? x_row[k_base] : 0;
                        int16_t x1 = (k_base + 1 < K) ? x_row[k_base + 1] : 0;
                        int16_t x2 = (k_base + 2 < K) ? x_row[k_base + 2] : 0;
                        int16_t x3 = (k_base + 3 < K) ? x_row[k_base + 3] : 0;

                        int16_t w_val[3] = {0, 1, -1};
                        int w0 = idx % 3;
                        int w1 = (idx / 3) % 3;
                        int w2 = (idx / 9) % 3;
                        int w3 = (idx / 27) % 3;

                        y_acc[mi][n] += x0*w_val[w0] + x1*w_val[w1] + x2*w_val[w2] + x3*w_val[w3];
                    }
                }
            }

            // Convert to float
            for (int mi = 0; mi < m_count; mi++) {
                const int m = m_base + mi;
                const float scale = scales[m];
                float* y_row = y + m * N;
                float32x4_t scale_vec = vdupq_n_f32(scale);

                if (bias != nullptr) {
                    for (int n = 0; n + 3 < N; n += 4) {
                        int32x4_t acc = vld1q_s32(&y_acc[mi][n]);
                        float32x4_t b = vld1q_f32(bias + n);
                        vst1q_f32(y_row + n, vaddq_f32(vmulq_f32(vcvtq_f32_s32(acc), scale_vec), b));
                    }
                    for (int n = (N / 4) * 4; n < N; n++) {
                        y_row[n] = static_cast<float>(y_acc[mi][n]) * scale + bias[n];
                    }
                } else {
                    for (int n = 0; n + 3 < N; n += 4) {
                        int32x4_t acc = vld1q_s32(&y_acc[mi][n]);
                        vst1q_f32(y_row + n, vmulq_f32(vcvtq_f32_s32(acc), scale_vec));
                    }
                    for (int n = (N / 4) * 4; n < N; n++) {
                        y_row[n] = static_cast<float>(y_acc[mi][n]) * scale;
                    }
                }
            }
        }
    }
}

// Python bindings
PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("quantize_activations_int8_tbl", &quantize_activations_int8_tbl,
          "Quantize float32 activations to int8 with per-row scale");
    m.def("pack_ternary_bitplanes_tbl", &pack_ternary_bitplanes_tbl,
          "Pack ternary weights to bit-plane format (2-bit)");
    m.def("matmul_free_neon_tbl", &matmul_free_neon_tbl,
          "True MatMul-free TBL kernel (16 parallel lookups)");
    m.def("matmul_free_neon_tbl_v2", &matmul_free_neon_tbl_v2,
          "TBL v2: N-blocked (32 outputs at a time)");
    m.def("matmul_free_neon_tbl_v3", &matmul_free_neon_tbl_v3,
          "TBL v3: K-unrolled + register accumulators");
    m.def("matmul_free_neon_tbl_v4", &matmul_free_neon_tbl_v4,
          "TBL v4: Cache-aware tiled + K-unrolled");
    m.def("matmul_free_neon_tbl_v5", &matmul_free_neon_tbl_v5,
          "TBL v5: Vectorized LUT + 64-way N-blocking");
    m.def("matmul_free_neon_tbl_v6", &matmul_free_neon_tbl_v6,
          "TBL v6: M-batched weight loading");
    m.def("matmul_free_neon_tbl_v7", &matmul_free_neon_tbl_v7,
          "TBL v7: Fused register accumulation");
    m.def("matmul_free_neon_tbl_v8", &matmul_free_neon_tbl_v8,
          "TBL v8: Software pipelining + M-batching");
    m.def("matmul_free_neon_tbl_v9", &matmul_free_neon_tbl_v9,
          "TBL v9: N-first + register accumulators");
    m.def("matmul_free_neon_tbl_v10", &matmul_free_neon_tbl_v10,
          "TBL v10: K-unroll x2 + M-batch=8");
    m.def("matmul_free_neon_tbl_v11", &matmul_free_neon_tbl_v11,
          "TBL v11: M-batch=16 + K-unroll x4");
    m.def("matmul_free_neon_tbl_v12", &matmul_free_neon_tbl_v12,
          "TBL v12: vqtbl2q dual-table optimization");
    m.def("pack_ternary_direct_g2", &pack_ternary_direct_g2,
          "Pack ternary weights to direct g=2 format");
    m.def("matmul_free_neon_tbl_v13", &matmul_free_neon_tbl_v13,
          "TBL v13: Direct ternary g=2 (2 TBL instead of 4)");
    m.def("pack_ternary_direct_g4", &pack_ternary_direct_g4,
          "Pack ternary weights to direct g=4 format");
    m.def("matmul_free_neon_tbl_v14", &matmul_free_neon_tbl_v14,
          "TBL v14: Direct ternary g=4 with vqtbl4q");
    m.def("matmul_free_neon_tbl_scalar", &matmul_free_neon_tbl_scalar,
          "TBL scalar reference for correctness testing");
}
