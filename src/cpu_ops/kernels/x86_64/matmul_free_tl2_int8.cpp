/*
 * TL2 Int8 Element-wise LUT Kernel for x86_64 AVX2
 *
 * Based on Microsoft BitNet.cpp TL2 approach with int8:
 * - Activations: int8 (per-tensor quantization)
 * - LUT: int8 values (14 entries for 3 weights)
 * - Accumulation: int32
 * - PSHUFB for parallel lookups
 *
 * TL2 groups 3 weights (3^3 = 27 combinations)
 * Mirror consolidation reduces to 14 entries + sign bit
 * Achieves 1.67 bits per weight
 */

#include <immintrin.h>
#include <cstring>
#include <cstdint>
#include <cmath>
#include <algorithm>
#include <omp.h>
#include <torch/extension.h>

constexpr int TL2_GROUP_SIZE = 3;

/**
 * Quantize float activations to int8 (same as T-MAC)
 */
void quantize_activations_int8_tl2(
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

        float max_abs = 0.0f;
        for (int k = 0; k < K; k++) {
            float abs_val = std::abs(x_row[k]);
            if (abs_val > max_abs) max_abs = abs_val;
        }

        float scale = max_abs / 127.0f;
        if (scale == 0.0f) scale = 1.0f;
        scales[m] = scale;

        float inv_scale = 1.0f / scale;
        for (int k = 0; k < K; k++) {
            float val = x_row[k] * inv_scale;
            val = std::max(-127.0f, std::min(127.0f, val));
            xq_row[k] = static_cast<int8_t>(std::round(val));
        }
    }
}

// Map ternary weight to index: -1->0, 0->1, +1->2
inline int ternary_to_idx(float w) {
    if (w > 0.5f) return 2;
    if (w < -0.5f) return 0;
    return 1;
}

inline int combined_index_3(int w0, int w1, int w2) {
    return w0 + 3 * w1 + 9 * w2;
}

// Mirror consolidation
inline std::pair<uint8_t, bool> consolidate_index(int idx) {
    if (idx <= 13) {
        return {static_cast<uint8_t>(idx), false};
    } else {
        return {static_cast<uint8_t>(26 - idx), true};
    }
}

/**
 * Pack ternary weights into TL2 format
 * Groups 3 weights, applies mirror consolidation
 * Stores index (4 bits) + sign (1 bit) in uint8
 */
void pack_ternary_tl2_int8(
    torch::Tensor w_tensor,
    torch::Tensor packed_tensor,
    int N, int K
) {
    const float* w = w_tensor.data_ptr<float>();
    uint8_t* packed = packed_tensor.data_ptr<uint8_t>();

    const int K_groups = (K + TL2_GROUP_SIZE - 1) / TL2_GROUP_SIZE;

    #pragma omp parallel for collapse(2)
    for (int kg = 0; kg < K_groups; kg++) {
        for (int n = 0; n < N; n++) {
            const int k_base = kg * TL2_GROUP_SIZE;

            int w0_idx = 1, w1_idx = 1, w2_idx = 1;  // 0 maps to index 1

            if (k_base < K) {
                w0_idx = ternary_to_idx(w[n * K + k_base]);
            }
            if (k_base + 1 < K) {
                w1_idx = ternary_to_idx(w[n * K + k_base + 1]);
            }
            if (k_base + 2 < K) {
                w2_idx = ternary_to_idx(w[n * K + k_base + 2]);
            }

            int idx = combined_index_3(w0_idx, w1_idx, w2_idx);
            auto [consolidated_idx, sign] = consolidate_index(idx);

            packed[kg * N + n] = consolidated_idx | (sign ? 0x10 : 0x00);
        }
    }
}

/**
 * Build 16-entry int8 LUT for 3 int8 activations (TL2)
 *
 * Only first 14 entries are used, rest are padding.
 * idx = w0 + 3*w1 + 9*w2, where w ∈ {0=-1, 1=0, 2=+1}
 */
inline void build_tl2_lut_int8(const int8_t* x, int8_t* lut) {
    const int16_t x0 = x[0], x1 = x[1], x2 = x[2];

    // Compute all 14 sums for indices 0-13
    int16_t sums[16];
    sums[0]  = -x0 - x1 - x2;  // (-1,-1,-1)
    sums[1]  =     - x1 - x2;  // ( 0,-1,-1)
    sums[2]  =  x0 - x1 - x2;  // (+1,-1,-1)
    sums[3]  = -x0      - x2;  // (-1, 0,-1)
    sums[4]  =          - x2;  // ( 0, 0,-1)
    sums[5]  =  x0      - x2;  // (+1, 0,-1)
    sums[6]  = -x0 + x1 - x2;  // (-1,+1,-1)
    sums[7]  =      x1 - x2;   // ( 0,+1,-1)
    sums[8]  =  x0 + x1 - x2;  // (+1,+1,-1)
    sums[9]  = -x0 - x1;       // (-1,-1, 0)
    sums[10] =     - x1;       // ( 0,-1, 0)
    sums[11] =  x0 - x1;       // (+1,-1, 0)
    sums[12] = -x0;            // (-1, 0, 0)
    sums[13] = 0;              // ( 0, 0, 0)
    sums[14] = 0;              // padding
    sums[15] = 0;              // padding

    // Clamp to int8
    for (int i = 0; i < 16; i++) {
        sums[i] = std::max(int16_t(-128), std::min(int16_t(127), sums[i]));
        lut[i] = static_cast<int8_t>(sums[i]);
    }
}

/**
 * TL2 Int8 kernel with PSHUFB
 *
 * Uses PSHUFB for parallel lookups, XOR for sign application.
 */
void matmul_free_tl2_int8(
    torch::Tensor x_int8_tensor,
    torch::Tensor scale_tensor,
    torch::Tensor packed_tensor,
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

    const int K_groups = (K + TL2_GROUP_SIZE - 1) / TL2_GROUP_SIZE;

    omp_set_num_threads(num_threads);

    #pragma omp parallel for schedule(static)
    for (int m = 0; m < M; m++) {
        const int8_t* __restrict__ x_row = x_int8 + m * K;
        float scale = scales[m];

        // Accumulate in int32
        alignas(64) int32_t y_acc[N];
        memset(y_acc, 0, N * sizeof(int32_t));

        // K-first iteration
        for (int kg = 0; kg < K_groups; kg++) {
            const int k_base = kg * TL2_GROUP_SIZE;

            // Get 3 int8 activations
            int8_t x_vals[4] = {0, 0, 0, 0};  // 4 for alignment
            for (int i = 0; i < 3 && k_base + i < K; i++) {
                x_vals[i] = x_row[k_base + i];
            }

            // Build int8 LUT
            alignas(32) int8_t lut[16];
            build_tl2_lut_int8(x_vals, lut);

            // Duplicate LUT for PSHUFB
            __m256i lut_vec = _mm256_broadcastsi128_si256(_mm_load_si128((__m128i*)lut));

            const uint8_t* __restrict__ packed_row = packed + kg * N;

            // Process 32 outputs at a time
            int n = 0;
            for (; n + 31 < N; n += 32) {
                // Load 32 packed values
                __m256i packed_vals = _mm256_loadu_si256((__m256i*)(packed_row + n));

                // Extract indices (lower 4 bits)
                __m256i idx_mask = _mm256_set1_epi8(0x0F);
                __m256i indices = _mm256_and_si256(packed_vals, idx_mask);

                // Extract sign bits (bit 4)
                __m256i sign_mask = _mm256_set1_epi8(0x10);
                __m256i sign_bits = _mm256_and_si256(packed_vals, sign_mask);

                // PSHUFB lookup
                __m256i lut_vals = _mm256_shuffle_epi8(lut_vec, indices);

                // Apply sign: if sign bit set, negate the value
                // sign_bits has 0x10 where sign is set, 0x00 otherwise
                // We need to negate lut_vals where sign_bits != 0

                // Create mask: 0xFF where sign bit set, 0x00 otherwise
                __m256i zero = _mm256_setzero_si256();
                __m256i sign_cmp = _mm256_cmpeq_epi8(sign_bits, zero);  // 0xFF where sign=0
                // sign_cmp is now inverted: 0xFF for no-flip, 0x00 for flip

                // For negation: -x = (~x) + 1, but simpler: use 0 - x
                // But we need conditional: sign ? -val : val
                // Use: result = (val XOR sign_extend) - sign_extend
                // Where sign_extend is 0xFF for negative, 0x00 for positive

                // ~sign_cmp gives 0xFF where we need to negate
                __m256i neg_mask = _mm256_andnot_si256(sign_cmp, _mm256_set1_epi8(0xFF));

                // Negate using XOR and subtract (two's complement)
                __m256i xored = _mm256_xor_si256(lut_vals, neg_mask);
                // Need to add 1 where we negated, but that's complex in int8
                // Simpler: use _mm256_sign_epi8 if available, or compute differently

                // Alternative: compute positive and negative separately, then blend
                __m256i neg_vals = _mm256_sub_epi8(zero, lut_vals);
                __m256i result = _mm256_blendv_epi8(lut_vals, neg_vals, neg_mask);

                // Unpack to int16 for accumulation
                __m256i res_lo = _mm256_cvtepi8_epi16(_mm256_extracti128_si256(result, 0));
                __m256i res_hi = _mm256_cvtepi8_epi16(_mm256_extracti128_si256(result, 1));

                // Extend to int32 and accumulate
                __m256i res_0 = _mm256_cvtepi16_epi32(_mm256_extracti128_si256(res_lo, 0));
                __m256i res_1 = _mm256_cvtepi16_epi32(_mm256_extracti128_si256(res_lo, 1));
                __m256i res_2 = _mm256_cvtepi16_epi32(_mm256_extracti128_si256(res_hi, 0));
                __m256i res_3 = _mm256_cvtepi16_epi32(_mm256_extracti128_si256(res_hi, 1));

                __m256i acc0 = _mm256_loadu_si256((__m256i*)(y_acc + n));
                __m256i acc1 = _mm256_loadu_si256((__m256i*)(y_acc + n + 8));
                __m256i acc2 = _mm256_loadu_si256((__m256i*)(y_acc + n + 16));
                __m256i acc3 = _mm256_loadu_si256((__m256i*)(y_acc + n + 24));

                acc0 = _mm256_add_epi32(acc0, res_0);
                acc1 = _mm256_add_epi32(acc1, res_1);
                acc2 = _mm256_add_epi32(acc2, res_2);
                acc3 = _mm256_add_epi32(acc3, res_3);

                _mm256_storeu_si256((__m256i*)(y_acc + n), acc0);
                _mm256_storeu_si256((__m256i*)(y_acc + n + 8), acc1);
                _mm256_storeu_si256((__m256i*)(y_acc + n + 16), acc2);
                _mm256_storeu_si256((__m256i*)(y_acc + n + 24), acc3);
            }

            // Handle remaining
            for (; n < N; n++) {
                uint8_t p = packed_row[n];
                uint8_t idx = p & 0x0F;
                bool sign = (p & 0x10) != 0;
                int8_t val = lut[idx];
                y_acc[n] += sign ? -static_cast<int32_t>(val) : static_cast<int32_t>(val);
            }
        }

        // Convert to float with scale
        float* y_row = y + m * N;
        __m256 scale_vec = _mm256_set1_ps(scale);

        if (bias != nullptr) {
            int n = 0;
            for (; n + 7 < N; n += 8) {
                __m256i acc = _mm256_loadu_si256((__m256i*)(y_acc + n));
                __m256 acc_f = _mm256_cvtepi32_ps(acc);
                __m256 bias_v = _mm256_loadu_ps(bias + n);
                __m256 result = _mm256_fmadd_ps(acc_f, scale_vec, bias_v);
                _mm256_storeu_ps(y_row + n, result);
            }
            for (; n < N; n++) {
                y_row[n] = static_cast<float>(y_acc[n]) * scale + bias[n];
            }
        } else {
            int n = 0;
            for (; n + 7 < N; n += 8) {
                __m256i acc = _mm256_loadu_si256((__m256i*)(y_acc + n));
                __m256 acc_f = _mm256_cvtepi32_ps(acc);
                __m256 result = _mm256_mul_ps(acc_f, scale_vec);
                _mm256_storeu_ps(y_row + n, result);
            }
            for (; n < N; n++) {
                y_row[n] = static_cast<float>(y_acc[n]) * scale;
            }
        }
    }
}

/**
 * Scalar reference for correctness
 */
void matmul_free_tl2_int8_scalar(
    torch::Tensor x_int8_tensor,
    torch::Tensor scale_tensor,
    torch::Tensor packed_tensor,
    torch::Tensor y_tensor,
    torch::Tensor bias_tensor,
    int M, int N, int K,
    int num_threads
) {
    const int8_t* x_int8 = x_int8_tensor.data_ptr<int8_t>();
    const float* scales = scale_tensor.data_ptr<float>();
    const uint8_t* packed = packed_tensor.data_ptr<uint8_t>();
    float* y = y_tensor.data_ptr<float>();
    const float* bias = bias_tensor.defined() ? bias_tensor.data_ptr<float>() : nullptr;

    const int K_groups = (K + TL2_GROUP_SIZE - 1) / TL2_GROUP_SIZE;

    omp_set_num_threads(num_threads);

    #pragma omp parallel for schedule(static)
    for (int m = 0; m < M; m++) {
        const int8_t* x_row = x_int8 + m * K;
        float scale = scales[m];
        float* y_row = y + m * N;

        // Initialize accumulator
        int32_t y_acc[N];
        memset(y_acc, 0, N * sizeof(int32_t));

        for (int kg = 0; kg < K_groups; kg++) {
            const int k_base = kg * TL2_GROUP_SIZE;

            int8_t x_vals[3] = {0, 0, 0};
            for (int i = 0; i < 3 && k_base + i < K; i++) {
                x_vals[i] = x_row[k_base + i];
            }

            int8_t lut[16];
            build_tl2_lut_int8(x_vals, lut);

            const uint8_t* packed_row = packed + kg * N;

            for (int n = 0; n < N; n++) {
                uint8_t p = packed_row[n];
                uint8_t idx = p & 0x0F;
                bool sign = (p & 0x10) != 0;
                int8_t val = lut[idx];
                y_acc[n] += sign ? -static_cast<int32_t>(val) : static_cast<int32_t>(val);
            }
        }

        // Convert to float
        for (int n = 0; n < N; n++) {
            y_row[n] = static_cast<float>(y_acc[n]) * scale + (bias ? bias[n] : 0.0f);
        }
    }
}

// Python bindings
PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("quantize_activations_int8_tl2", &quantize_activations_int8_tl2,
          "Quantize float32 activations to int8");
    m.def("pack_ternary_tl2_int8", &pack_ternary_tl2_int8,
          "Pack ternary weights to TL2 format");
    m.def("matmul_free_tl2_int8", &matmul_free_tl2_int8,
          "TL2 kernel with int8 and PSHUFB");
    m.def("matmul_free_tl2_int8_scalar", &matmul_free_tl2_int8_scalar,
          "TL2 int8 scalar reference");
}
