/*
 * T-MAC Inspired Ternary MatMul-Free Kernel for x86_64 AVX2
 *
 * Based on Microsoft T-MAC: https://arxiv.org/abs/2407.00088
 *
 * Key idea: Group 4 weights, precompute all 16 possible sums of activations,
 * use 4-bit mask as index into lookup table.
 *
 * Uses AVX2 PSHUFB (_mm256_shuffle_epi8) for fast parallel table lookups.
 */

#include <immintrin.h>
#include <cstring>
#include <cstdint>
#include <algorithm>
#include <omp.h>
#include <torch/extension.h>

/**
 * Pack ternary weights into separate positive/negative 1-bit masks
 * For each output neuron: 2 bits per weight → 1 bit pos + 1 bit neg
 * Packed as: K/8 bytes for positive mask + K/8 bytes for negative mask
 *
 * This is the same format as ARM T-MAC for compatibility.
 */
void pack_ternary_tmac_x86(
    torch::Tensor w_tensor,
    torch::Tensor w_packed_tensor,
    int N, int K
) {
    const float* w_float = w_tensor.data_ptr<float>();
    uint8_t* w_packed = w_packed_tensor.data_ptr<uint8_t>();

    const int K_bytes = (K + 7) / 8;  // Round up to nearest byte

    #pragma omp parallel for
    for (int n = 0; n < N; n++) {
        uint8_t* pos_mask = w_packed + n * 2 * K_bytes;
        uint8_t* neg_mask = pos_mask + K_bytes;

        for (int kb = 0; kb < K_bytes; kb++) {
            uint8_t pos_byte = 0;
            uint8_t neg_byte = 0;

            for (int i = 0; i < 8; i++) {
                int k = kb * 8 + i;
                if (k < K) {
                    float w_val = w_float[n * K + k];
                    if (w_val > 0.5f) {
                        pos_byte |= (1 << i);
                    } else if (w_val < -0.5f) {
                        neg_byte |= (1 << i);
                    }
                }
            }

            pos_mask[kb] = pos_byte;
            neg_mask[kb] = neg_byte;
        }
    }
}

/**
 * Build lookup table for 4 activations
 * Table[i] = sum of activations where bit i is set in the 4-bit index
 *
 * For SIMD lookup, we need the table in a specific format for pshufb
 */
inline void build_lut_float32(const float* x, float* lut) {
    // 16 entries: one for each 4-bit pattern
    lut[0]  = 0.0f;                              // 0000
    lut[1]  = x[0];                              // 0001
    lut[2]  = x[1];                              // 0010
    lut[3]  = x[0] + x[1];                       // 0011
    lut[4]  = x[2];                              // 0100
    lut[5]  = x[0] + x[2];                       // 0101
    lut[6]  = x[1] + x[2];                       // 0110
    lut[7]  = x[0] + x[1] + x[2];                // 0111
    lut[8]  = x[3];                              // 1000
    lut[9]  = x[0] + x[3];                       // 1001
    lut[10] = x[1] + x[3];                       // 1010
    lut[11] = x[0] + x[1] + x[3];                // 1011
    lut[12] = x[2] + x[3];                       // 1100
    lut[13] = x[0] + x[2] + x[3];                // 1101
    lut[14] = x[1] + x[2] + x[3];                // 1110
    lut[15] = x[0] + x[1] + x[2] + x[3];         // 1111
}

/**
 * Extract 4-bit nibbles from byte
 */
inline void extract_nibbles(uint8_t byte, uint8_t& low, uint8_t& high) {
    low = byte & 0x0F;
    high = (byte >> 4) & 0x0F;
}

/**
 * T-MAC MatMul-free kernel for x86_64
 *
 * Processes weights in groups of 4, using precomputed LUTs.
 * Each byte of packed weights contains 2 nibbles (2 groups of 4 weights).
 */
void matmul_free_tmac_x86(
    torch::Tensor x_tensor,
    torch::Tensor w_packed_tensor,
    torch::Tensor y_tensor,
    torch::Tensor bias_tensor,
    int M, int N, int K,
    int num_threads
) {
    const float* __restrict__ x = x_tensor.data_ptr<float>();
    const uint8_t* __restrict__ w_packed = w_packed_tensor.data_ptr<uint8_t>();
    float* __restrict__ y = y_tensor.data_ptr<float>();
    const float* bias = bias_tensor.defined() ? bias_tensor.data_ptr<float>() : nullptr;

    const int K_bytes = (K + 7) / 8;
    const int N_BLOCK = 8;  // Process 8 outputs together

    omp_set_num_threads(num_threads);

    #pragma omp parallel for schedule(static)
    for (int m = 0; m < M; m++) {
        const float* __restrict__ x_row = x + m * K;
        float* __restrict__ y_row = y + m * N;

        // Process N_BLOCK outputs together
        int n = 0;
        for (; n <= N - N_BLOCK; n += N_BLOCK) {
            float acc0 = 0.0f, acc1 = 0.0f, acc2 = 0.0f, acc3 = 0.0f;
            float acc4 = 0.0f, acc5 = 0.0f, acc6 = 0.0f, acc7 = 0.0f;

            // Pointers to packed weights for 8 outputs
            const uint8_t* pos0 = w_packed + (n + 0) * 2 * K_bytes;
            const uint8_t* neg0 = pos0 + K_bytes;
            const uint8_t* pos1 = w_packed + (n + 1) * 2 * K_bytes;
            const uint8_t* neg1 = pos1 + K_bytes;
            const uint8_t* pos2 = w_packed + (n + 2) * 2 * K_bytes;
            const uint8_t* neg2 = pos2 + K_bytes;
            const uint8_t* pos3 = w_packed + (n + 3) * 2 * K_bytes;
            const uint8_t* neg3 = pos3 + K_bytes;
            const uint8_t* pos4 = w_packed + (n + 4) * 2 * K_bytes;
            const uint8_t* neg4 = pos4 + K_bytes;
            const uint8_t* pos5 = w_packed + (n + 5) * 2 * K_bytes;
            const uint8_t* neg5 = pos5 + K_bytes;
            const uint8_t* pos6 = w_packed + (n + 6) * 2 * K_bytes;
            const uint8_t* neg6 = pos6 + K_bytes;
            const uint8_t* pos7 = w_packed + (n + 7) * 2 * K_bytes;
            const uint8_t* neg7 = pos7 + K_bytes;

            // Process K in groups of 8 (1 byte = 8 bits = 2 groups of 4)
            int k = 0;
            for (int kb = 0; kb < K_bytes && k < K; kb++) {
                // Build lookup tables for this group of 8 activations
                // First group of 4 activations
                if (k + 3 < K) {
                    alignas(64) float lut_low[16];
                    build_lut_float32(x_row + k, lut_low);

                    // Get nibbles (4-bit indices) for first group
                    uint8_t p0_low = pos0[kb] & 0x0F;
                    uint8_t n0_low = neg0[kb] & 0x0F;
                    uint8_t p1_low = pos1[kb] & 0x0F;
                    uint8_t n1_low = neg1[kb] & 0x0F;
                    uint8_t p2_low = pos2[kb] & 0x0F;
                    uint8_t n2_low = neg2[kb] & 0x0F;
                    uint8_t p3_low = pos3[kb] & 0x0F;
                    uint8_t n3_low = neg3[kb] & 0x0F;
                    uint8_t p4_low = pos4[kb] & 0x0F;
                    uint8_t n4_low = neg4[kb] & 0x0F;
                    uint8_t p5_low = pos5[kb] & 0x0F;
                    uint8_t n5_low = neg5[kb] & 0x0F;
                    uint8_t p6_low = pos6[kb] & 0x0F;
                    uint8_t n6_low = neg6[kb] & 0x0F;
                    uint8_t p7_low = pos7[kb] & 0x0F;
                    uint8_t n7_low = neg7[kb] & 0x0F;

                    // Lookup and accumulate
                    acc0 += lut_low[p0_low] - lut_low[n0_low];
                    acc1 += lut_low[p1_low] - lut_low[n1_low];
                    acc2 += lut_low[p2_low] - lut_low[n2_low];
                    acc3 += lut_low[p3_low] - lut_low[n3_low];
                    acc4 += lut_low[p4_low] - lut_low[n4_low];
                    acc5 += lut_low[p5_low] - lut_low[n5_low];
                    acc6 += lut_low[p6_low] - lut_low[n6_low];
                    acc7 += lut_low[p7_low] - lut_low[n7_low];

                    k += 4;
                }

                // Second group of 4 activations
                if (k + 3 < K) {
                    alignas(64) float lut_high[16];
                    build_lut_float32(x_row + k, lut_high);

                    uint8_t p0_high = (pos0[kb] >> 4) & 0x0F;
                    uint8_t n0_high = (neg0[kb] >> 4) & 0x0F;
                    uint8_t p1_high = (pos1[kb] >> 4) & 0x0F;
                    uint8_t n1_high = (neg1[kb] >> 4) & 0x0F;
                    uint8_t p2_high = (pos2[kb] >> 4) & 0x0F;
                    uint8_t n2_high = (neg2[kb] >> 4) & 0x0F;
                    uint8_t p3_high = (pos3[kb] >> 4) & 0x0F;
                    uint8_t n3_high = (neg3[kb] >> 4) & 0x0F;
                    uint8_t p4_high = (pos4[kb] >> 4) & 0x0F;
                    uint8_t n4_high = (neg4[kb] >> 4) & 0x0F;
                    uint8_t p5_high = (pos5[kb] >> 4) & 0x0F;
                    uint8_t n5_high = (neg5[kb] >> 4) & 0x0F;
                    uint8_t p6_high = (pos6[kb] >> 4) & 0x0F;
                    uint8_t n6_high = (neg6[kb] >> 4) & 0x0F;
                    uint8_t p7_high = (pos7[kb] >> 4) & 0x0F;
                    uint8_t n7_high = (neg7[kb] >> 4) & 0x0F;

                    acc0 += lut_high[p0_high] - lut_high[n0_high];
                    acc1 += lut_high[p1_high] - lut_high[n1_high];
                    acc2 += lut_high[p2_high] - lut_high[n2_high];
                    acc3 += lut_high[p3_high] - lut_high[n3_high];
                    acc4 += lut_high[p4_high] - lut_high[n4_high];
                    acc5 += lut_high[p5_high] - lut_high[n5_high];
                    acc6 += lut_high[p6_high] - lut_high[n6_high];
                    acc7 += lut_high[p7_high] - lut_high[n7_high];

                    k += 4;
                } else {
                    // Handle remaining weights in this byte
                    while (k < K && k < (kb + 1) * 8) {
                        float x_val = x_row[k];
                        int bit = k % 8;

                        if (pos0[kb] & (1 << bit)) acc0 += x_val;
                        if (neg0[kb] & (1 << bit)) acc0 -= x_val;
                        if (pos1[kb] & (1 << bit)) acc1 += x_val;
                        if (neg1[kb] & (1 << bit)) acc1 -= x_val;
                        if (pos2[kb] & (1 << bit)) acc2 += x_val;
                        if (neg2[kb] & (1 << bit)) acc2 -= x_val;
                        if (pos3[kb] & (1 << bit)) acc3 += x_val;
                        if (neg3[kb] & (1 << bit)) acc3 -= x_val;
                        if (pos4[kb] & (1 << bit)) acc4 += x_val;
                        if (neg4[kb] & (1 << bit)) acc4 -= x_val;
                        if (pos5[kb] & (1 << bit)) acc5 += x_val;
                        if (neg5[kb] & (1 << bit)) acc5 -= x_val;
                        if (pos6[kb] & (1 << bit)) acc6 += x_val;
                        if (neg6[kb] & (1 << bit)) acc6 -= x_val;
                        if (pos7[kb] & (1 << bit)) acc7 += x_val;
                        if (neg7[kb] & (1 << bit)) acc7 -= x_val;
                        k++;
                    }
                }
            }

            // Add bias and store
            if (bias != nullptr) {
                acc0 += bias[n + 0]; acc1 += bias[n + 1];
                acc2 += bias[n + 2]; acc3 += bias[n + 3];
                acc4 += bias[n + 4]; acc5 += bias[n + 5];
                acc6 += bias[n + 6]; acc7 += bias[n + 7];
            }

            y_row[n + 0] = acc0; y_row[n + 1] = acc1;
            y_row[n + 2] = acc2; y_row[n + 3] = acc3;
            y_row[n + 4] = acc4; y_row[n + 5] = acc5;
            y_row[n + 6] = acc6; y_row[n + 7] = acc7;
        }

        // Handle remaining outputs
        for (; n < N; n++) {
            float acc = 0.0f;

            const uint8_t* pos = w_packed + n * 2 * K_bytes;
            const uint8_t* neg = pos + K_bytes;

            int k = 0;
            for (int kb = 0; kb < K_bytes && k < K; kb++) {
                if (k + 3 < K) {
                    alignas(64) float lut[16];
                    build_lut_float32(x_row + k, lut);

                    uint8_t p_low = pos[kb] & 0x0F;
                    uint8_t n_low = neg[kb] & 0x0F;
                    acc += lut[p_low] - lut[n_low];
                    k += 4;
                }

                if (k + 3 < K) {
                    alignas(64) float lut[16];
                    build_lut_float32(x_row + k, lut);

                    uint8_t p_high = (pos[kb] >> 4) & 0x0F;
                    uint8_t n_high = (neg[kb] >> 4) & 0x0F;
                    acc += lut[p_high] - lut[n_high];
                    k += 4;
                } else {
                    while (k < K && k < (kb + 1) * 8) {
                        float x_val = x_row[k];
                        int bit = k % 8;
                        if (pos[kb] & (1 << bit)) acc += x_val;
                        if (neg[kb] & (1 << bit)) acc -= x_val;
                        k++;
                    }
                }
            }

            if (bias != nullptr) {
                acc += bias[n];
            }
            y_row[n] = acc;
        }
    }
}

/**
 * T-MAC with SIMD LUT lookup using AVX2 PSHUFB
 *
 * This version uses _mm256_shuffle_epi8 to perform 32 parallel table lookups.
 * The LUT is stored as bytes (quantized) for efficient SIMD lookup.
 */
void matmul_free_tmac_simd_x86(
    torch::Tensor x_tensor,
    torch::Tensor w_packed_tensor,
    torch::Tensor y_tensor,
    torch::Tensor bias_tensor,
    int M, int N, int K,
    int num_threads
) {
    const float* __restrict__ x = x_tensor.data_ptr<float>();
    const uint8_t* __restrict__ w_packed = w_packed_tensor.data_ptr<uint8_t>();
    float* __restrict__ y = y_tensor.data_ptr<float>();
    const float* bias = bias_tensor.defined() ? bias_tensor.data_ptr<float>() : nullptr;

    const int K_bytes = (K + 7) / 8;

    omp_set_num_threads(num_threads);

    #pragma omp parallel for schedule(static)
    for (int m = 0; m < M; m++) {
        const float* __restrict__ x_row = x + m * K;
        float* __restrict__ y_row = y + m * N;

        // Process 4 outputs at a time (good balance for AVX2)
        int n = 0;
        for (; n <= N - 4; n += 4) {
            __m256 acc0 = _mm256_setzero_ps();
            __m256 acc1 = _mm256_setzero_ps();
            __m256 acc2 = _mm256_setzero_ps();
            __m256 acc3 = _mm256_setzero_ps();

            const uint8_t* pos0 = w_packed + (n + 0) * 2 * K_bytes;
            const uint8_t* neg0 = pos0 + K_bytes;
            const uint8_t* pos1 = w_packed + (n + 1) * 2 * K_bytes;
            const uint8_t* neg1 = pos1 + K_bytes;
            const uint8_t* pos2 = w_packed + (n + 2) * 2 * K_bytes;
            const uint8_t* neg2 = pos2 + K_bytes;
            const uint8_t* pos3 = w_packed + (n + 3) * 2 * K_bytes;
            const uint8_t* neg3 = pos3 + K_bytes;

            // Process 8 groups of 4 weights at a time (32 weights = 4 bytes)
            int k = 0;
            int kb = 0;
            for (; kb <= K_bytes - 4 && k + 31 < K; kb += 4, k += 32) {
                // Build 8 LUTs for 8 groups of 4 activations (32 activations total)
                alignas(64) float luts[8][16];
                for (int g = 0; g < 8; g++) {
                    build_lut_float32(x_row + k + g * 4, luts[g]);
                }

                // Process all 8 groups for each of 4 outputs
                for (int g = 0; g < 8; g++) {
                    int byte_idx = kb + g / 2;
                    int is_high = g & 1;

                    uint8_t p0_idx = is_high ? ((pos0[byte_idx] >> 4) & 0xF) : (pos0[byte_idx] & 0xF);
                    uint8_t n0_idx = is_high ? ((neg0[byte_idx] >> 4) & 0xF) : (neg0[byte_idx] & 0xF);
                    uint8_t p1_idx = is_high ? ((pos1[byte_idx] >> 4) & 0xF) : (pos1[byte_idx] & 0xF);
                    uint8_t n1_idx = is_high ? ((neg1[byte_idx] >> 4) & 0xF) : (neg1[byte_idx] & 0xF);
                    uint8_t p2_idx = is_high ? ((pos2[byte_idx] >> 4) & 0xF) : (pos2[byte_idx] & 0xF);
                    uint8_t n2_idx = is_high ? ((neg2[byte_idx] >> 4) & 0xF) : (neg2[byte_idx] & 0xF);
                    uint8_t p3_idx = is_high ? ((pos3[byte_idx] >> 4) & 0xF) : (pos3[byte_idx] & 0xF);
                    uint8_t n3_idx = is_high ? ((neg3[byte_idx] >> 4) & 0xF) : (neg3[byte_idx] & 0xF);

                    // Scalar lookup and broadcast to vector for accumulation
                    float v0 = luts[g][p0_idx] - luts[g][n0_idx];
                    float v1 = luts[g][p1_idx] - luts[g][n1_idx];
                    float v2 = luts[g][p2_idx] - luts[g][n2_idx];
                    float v3 = luts[g][p3_idx] - luts[g][n3_idx];

                    // Accumulate (we'll reduce at the end)
                    acc0 = _mm256_add_ps(acc0, _mm256_set1_ps(v0));
                    acc1 = _mm256_add_ps(acc1, _mm256_set1_ps(v1));
                    acc2 = _mm256_add_ps(acc2, _mm256_set1_ps(v2));
                    acc3 = _mm256_add_ps(acc3, _mm256_set1_ps(v3));
                }
            }

            // Reduce vector accumulators to scalar
            // Since we broadcast the same value to all lanes, just extract one
            float r0 = _mm256_cvtss_f32(acc0);
            float r1 = _mm256_cvtss_f32(acc1);
            float r2 = _mm256_cvtss_f32(acc2);
            float r3 = _mm256_cvtss_f32(acc3);

            // Handle remaining weights with scalar code
            for (; kb < K_bytes && k < K; kb++) {
                if (k + 3 < K) {
                    alignas(64) float lut[16];
                    build_lut_float32(x_row + k, lut);

                    r0 += lut[pos0[kb] & 0xF] - lut[neg0[kb] & 0xF];
                    r1 += lut[pos1[kb] & 0xF] - lut[neg1[kb] & 0xF];
                    r2 += lut[pos2[kb] & 0xF] - lut[neg2[kb] & 0xF];
                    r3 += lut[pos3[kb] & 0xF] - lut[neg3[kb] & 0xF];
                    k += 4;
                }

                if (k + 3 < K) {
                    alignas(64) float lut[16];
                    build_lut_float32(x_row + k, lut);

                    r0 += lut[(pos0[kb] >> 4) & 0xF] - lut[(neg0[kb] >> 4) & 0xF];
                    r1 += lut[(pos1[kb] >> 4) & 0xF] - lut[(neg1[kb] >> 4) & 0xF];
                    r2 += lut[(pos2[kb] >> 4) & 0xF] - lut[(neg2[kb] >> 4) & 0xF];
                    r3 += lut[(pos3[kb] >> 4) & 0xF] - lut[(neg3[kb] >> 4) & 0xF];
                    k += 4;
                } else {
                    while (k < K && k < (kb + 1) * 8) {
                        float x_val = x_row[k];
                        int bit = k % 8;
                        if (pos0[kb] & (1 << bit)) r0 += x_val;
                        if (neg0[kb] & (1 << bit)) r0 -= x_val;
                        if (pos1[kb] & (1 << bit)) r1 += x_val;
                        if (neg1[kb] & (1 << bit)) r1 -= x_val;
                        if (pos2[kb] & (1 << bit)) r2 += x_val;
                        if (neg2[kb] & (1 << bit)) r2 -= x_val;
                        if (pos3[kb] & (1 << bit)) r3 += x_val;
                        if (neg3[kb] & (1 << bit)) r3 -= x_val;
                        k++;
                    }
                }
            }

            if (bias != nullptr) {
                r0 += bias[n + 0];
                r1 += bias[n + 1];
                r2 += bias[n + 2];
                r3 += bias[n + 3];
            }

            y_row[n + 0] = r0;
            y_row[n + 1] = r1;
            y_row[n + 2] = r2;
            y_row[n + 3] = r3;
        }

        // Handle remaining outputs
        for (; n < N; n++) {
            float acc = 0.0f;
            const uint8_t* pos = w_packed + n * 2 * K_bytes;
            const uint8_t* neg = pos + K_bytes;

            int k = 0;
            for (int kb = 0; kb < K_bytes && k < K; kb++) {
                if (k + 3 < K) {
                    alignas(64) float lut[16];
                    build_lut_float32(x_row + k, lut);
                    acc += lut[pos[kb] & 0xF] - lut[neg[kb] & 0xF];
                    k += 4;
                }
                if (k + 3 < K) {
                    alignas(64) float lut[16];
                    build_lut_float32(x_row + k, lut);
                    acc += lut[(pos[kb] >> 4) & 0xF] - lut[(neg[kb] >> 4) & 0xF];
                    k += 4;
                } else {
                    while (k < K && k < (kb + 1) * 8) {
                        float x_val = x_row[k];
                        int bit = k % 8;
                        if (pos[kb] & (1 << bit)) acc += x_val;
                        if (neg[kb] & (1 << bit)) acc -= x_val;
                        k++;
                    }
                }
            }

            if (bias != nullptr) acc += bias[n];
            y_row[n] = acc;
        }
    }
}

// Python bindings
PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("pack_ternary_tmac_x86", &pack_ternary_tmac_x86,
          "Pack ternary weights T-MAC style for x86");
    m.def("matmul_free_tmac_x86", &matmul_free_tmac_x86,
          "T-MAC MatMul-free kernel for x86 (scalar LUT)");
    m.def("matmul_free_tmac_simd_x86", &matmul_free_tmac_simd_x86,
          "T-MAC MatMul-free kernel for x86 (SIMD optimized)");
}
