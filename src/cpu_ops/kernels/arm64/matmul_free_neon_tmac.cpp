/*
 * T-MAC Inspired Ternary MatMul-Free Kernel for ARM NEON
 *
 * TRUE MatMul-free using lookup tables and NEON vtbl
 * Based on Microsoft T-MAC: https://arxiv.org/abs/2407.00088
 *
 * Key idea: Group 4 weights, precompute all 16 possible sums of activations,
 * use 4-bit mask as index into lookup table (fits in NEON register!)
 */

#include <arm_neon.h>
#include <cstring>
#include <algorithm>
#include <torch/extension.h>
#include <omp.h>

#define PREFETCH(addr) __builtin_prefetch(addr, 0, 3)

/**
 * Pack ternary weights into separate positive/negative 1-bit masks
 * For each output neuron: 2 bits per weight → 1 bit pos + 1 bit neg
 * Packed as: K/8 bytes for positive mask + K/8 bytes for negative mask
 */
void pack_ternary_tmac(
    torch::Tensor w_tensor,
    torch::Tensor w_packed_tensor,
    int N, int K
) {
    const float* w_float = w_tensor.data_ptr<float>();
    uint8_t* w_packed = w_packed_tensor.data_ptr<uint8_t>();

    int K_bytes = (K + 7) / 8;  // Round up to nearest byte

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
 * Build lookup table for 4 activations and 4-bit mask
 * Table[i] = sum of activations where bit i is set
 */
inline void build_lut_float32(const float* x, float* lut) {
    // 16 entries: one for each 4-bit pattern
    lut[0] = 0.0f;                              // 0000
    lut[1] = x[0];                              // 0001
    lut[2] = x[1];                              // 0010
    lut[3] = x[0] + x[1];                       // 0011
    lut[4] = x[2];                              // 0100
    lut[5] = x[0] + x[2];                       // 0101
    lut[6] = x[1] + x[2];                       // 0110
    lut[7] = x[0] + x[1] + x[2];                // 0111
    lut[8] = x[3];                              // 1000
    lut[9] = x[0] + x[3];                       // 1001
    lut[10] = x[1] + x[3];                      // 1010
    lut[11] = x[0] + x[1] + x[3];               // 1011
    lut[12] = x[2] + x[3];                      // 1100
    lut[13] = x[0] + x[2] + x[3];               // 1101
    lut[14] = x[1] + x[2] + x[3];               // 1110
    lut[15] = x[0] + x[1] + x[2] + x[3];        // 1111
}

/**
 * Extract 4-bit nibbles from byte
 */
inline void extract_nibbles(uint8_t byte, uint8_t& low, uint8_t& high) {
    low = byte & 0x0F;
    high = (byte >> 4) & 0x0F;
}

/**
 * T-MAC inspired MatMul-free kernel
 */
void matmul_free_neon_tmac(
    torch::Tensor x_tensor,
    torch::Tensor w_packed_tensor,
    torch::Tensor y_tensor,
    torch::Tensor bias_tensor,
    int M, int N, int K,
    int num_threads
) {
    const float* x = x_tensor.data_ptr<float>();
    const uint8_t* w_packed = w_packed_tensor.data_ptr<uint8_t>();
    float* y = y_tensor.data_ptr<float>();
    const float* bias = bias_tensor.defined() ? bias_tensor.data_ptr<float>() : nullptr;

    const int K_bytes = (K + 7) / 8;
    const int N_BLOCK = 8;  // Process 8 outputs together

    omp_set_num_threads(num_threads);

    #pragma omp parallel for schedule(dynamic, 8)
    for (int m = 0; m < M; m++) {
        const float* x_row = x + m * K;
        float* y_row = y + m * N;

        // Process 8 outputs together
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
            for (int kb = 0; kb < K_bytes; kb++) {
                // Build lookup tables for this group of 8 activations
                // Each byte has 2 nibbles (4 bits each), so 2 LUTs
                // IMPORTANT: LUT is based on ACTIVATIONS, same for all outputs!
                if (k + 7 < K) {
                    // First group of 4 - build ONCE, use for all 8 outputs
                    float lut_low[16];
                    build_lut_float32(x_row + k, lut_low);

                    // Get nibbles (4-bit indices) for first group
                    uint8_t p0_low, p0_high, n0_low, n0_high;
                    extract_nibbles(pos0[kb], p0_low, p0_high);
                    extract_nibbles(neg0[kb], n0_low, n0_high);

                    uint8_t p1_low, p1_high, n1_low, n1_high;
                    extract_nibbles(pos1[kb], p1_low, p1_high);
                    extract_nibbles(neg1[kb], n1_low, n1_high);

                    uint8_t p2_low, p2_high, n2_low, n2_high;
                    extract_nibbles(pos2[kb], p2_low, p2_high);
                    extract_nibbles(neg2[kb], n2_low, n2_high);

                    uint8_t p3_low, p3_high, n3_low, n3_high;
                    extract_nibbles(pos3[kb], p3_low, p3_high);
                    extract_nibbles(neg3[kb], n3_low, n3_high);

                    uint8_t p4_low, p4_high, n4_low, n4_high;
                    extract_nibbles(pos4[kb], p4_low, p4_high);
                    extract_nibbles(neg4[kb], n4_low, n4_high);

                    uint8_t p5_low, p5_high, n5_low, n5_high;
                    extract_nibbles(pos5[kb], p5_low, p5_high);
                    extract_nibbles(neg5[kb], n5_low, n5_high);

                    uint8_t p6_low, p6_high, n6_low, n6_high;
                    extract_nibbles(pos6[kb], p6_low, p6_high);
                    extract_nibbles(neg6[kb], n6_low, n6_high);

                    uint8_t p7_low, p7_high, n7_low, n7_high;
                    extract_nibbles(pos7[kb], p7_low, p7_high);
                    extract_nibbles(neg7[kb], n7_low, n7_high);

                    // Lookup and accumulate - TRUE matmul-free!
                    // All outputs use the SAME LUT (built from activations)
                    acc0 += lut_low[p0_low] - lut_low[n0_low];
                    acc1 += lut_low[p1_low] - lut_low[n1_low];
                    acc2 += lut_low[p2_low] - lut_low[n2_low];
                    acc3 += lut_low[p3_low] - lut_low[n3_low];
                    acc4 += lut_low[p4_low] - lut_low[n4_low];
                    acc5 += lut_low[p5_low] - lut_low[n5_low];
                    acc6 += lut_low[p6_low] - lut_low[n6_low];
                    acc7 += lut_low[p7_low] - lut_low[n7_low];

                    // Second group of 4 (next 4 activations)
                    k += 4;
                    if (k + 3 < K) {
                        float lut_high[16];
                        build_lut_float32(x_row + k, lut_high);

                        acc0 += lut_high[p0_high] - lut_high[n0_high];
                        acc1 += lut_high[p1_high] - lut_high[n1_high];
                        acc2 += lut_high[p2_high] - lut_high[n2_high];
                        acc3 += lut_high[p3_high] - lut_high[n3_high];
                        acc4 += lut_high[p4_high] - lut_high[n4_high];
                        acc5 += lut_high[p5_high] - lut_high[n5_high];
                        acc6 += lut_high[p6_high] - lut_high[n6_high];
                        acc7 += lut_high[p7_high] - lut_high[n7_high];
                    }
                    k += 4;
                } else {
                    // Scalar tail
                    for (int i = 0; i < 8 && k < K; i++, k++) {
                        float x_val = x_row[k];
                        int bit = i % 8;

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
                    }
                }
            }

            // Add bias
            if (bias != nullptr) {
                acc0 += bias[n + 0];
                acc1 += bias[n + 1];
                acc2 += bias[n + 2];
                acc3 += bias[n + 3];
                acc4 += bias[n + 4];
                acc5 += bias[n + 5];
                acc6 += bias[n + 6];
                acc7 += bias[n + 7];
            }

            y_row[n + 0] = acc0;
            y_row[n + 1] = acc1;
            y_row[n + 2] = acc2;
            y_row[n + 3] = acc3;
            y_row[n + 4] = acc4;
            y_row[n + 5] = acc5;
            y_row[n + 6] = acc6;
            y_row[n + 7] = acc7;
        }

        // Handle remaining outputs
        for (; n < N; n++) {
            float acc = 0.0f;

            const uint8_t* pos = w_packed + n * 2 * K_bytes;
            const uint8_t* neg = pos + K_bytes;

            int k = 0;
            for (int kb = 0; kb < K_bytes; kb++) {
                if (k + 7 < K) {
                    // Use LUT for groups of 4
                    float lut[16];

                    uint8_t p_low, p_high, n_low, n_high;
                    extract_nibbles(pos[kb], p_low, p_high);
                    extract_nibbles(neg[kb], n_low, n_high);

                    build_lut_float32(x_row + k, lut);
                    acc += lut[p_low] - lut[n_low];
                    k += 4;

                    if (k + 3 < K) {
                        build_lut_float32(x_row + k, lut);
                        acc += lut[p_high] - lut[n_high];
                        k += 4;
                    }
                } else {
                    // Scalar tail
                    for (int i = 0; i < 8 && k < K; i++, k++) {
                        float x_val = x_row[k];
                        if (pos[kb] & (1 << i)) acc += x_val;
                        if (neg[kb] & (1 << i)) acc -= x_val;
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

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("pack_ternary_tmac", &pack_ternary_tmac, "Pack ternary weights T-MAC style");
    m.def("matmul_free_neon_tmac", &matmul_free_neon_tmac, "T-MAC inspired MatMul-free");
}
