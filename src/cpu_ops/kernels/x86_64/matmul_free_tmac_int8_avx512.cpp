/*
 * T-MAC Int8 Ternary MatMul-Free Kernel for x86_64 AVX-512
 *
 * Uses AVX-512BW vpshufb for 64 parallel lookups per instruction (4x SSE).
 * Same pack-and-unpack technique as AVX2 version but with wider vectors.
 */

#include <immintrin.h>
#include <cstring>
#include <cstdint>
#include <cmath>
#include <algorithm>
#include <omp.h>
#include <torch/extension.h>

/**
 * Quantize float activations to int8
 */
void quantize_activations_int8_avx512(
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

/**
 * Pack ternary weights into bit-plane format
 */
void pack_ternary_bitplanes_int8_avx512(
    torch::Tensor w_tensor,
    torch::Tensor sign_plane_tensor,
    torch::Tensor value_plane_tensor,
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
                    sign_nibble |= (1 << i);
                    value_nibble |= (1 << i);
                } else if (val < -0.5f) {
                    sign_nibble |= (1 << i);
                }
            }
            sign_plane[kg * N + n] = sign_nibble;
            value_plane[kg * N + n] = value_nibble;
        }
    }
}

/**
 * Build 16-entry int16 LUT from 4 int8 activations
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
 * Build split low/high byte LUTs for pack-and-unpack
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

    for (int i = 0; i < 16; i++) {
        lut_lo[i] = static_cast<uint8_t>(sums[i] & 0xFF);
        lut_hi[i] = static_cast<uint8_t>((sums[i] >> 8) & 0xFF);
    }
}

/**
 * T-MAC Int8 kernel with AVX-512 PSHUFB
 *
 * Uses _mm512_shuffle_epi8 for 64 parallel lookups (4x SSE version).
 * AVX-512 vpshufb operates on 4 independent 128-bit lanes.
 */
void matmul_free_tmac_int8_avx512(
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

    omp_set_num_threads(num_threads);

    #pragma omp parallel for schedule(static)
    for (int m = 0; m < M; m++) {
        const int8_t* __restrict__ x_row = x_int8 + m * K;
        float scale = scales[m];

        alignas(64) int32_t y_acc[N];
        memset(y_acc, 0, N * sizeof(int32_t));

        for (int kg = 0; kg < K_groups; kg++) {
            const int k_base = kg * 4;

            int8_t x_vals[4] = {0, 0, 0, 0};
            for (int i = 0; i < 4 && k_base + i < K; i++) {
                x_vals[i] = x_row[k_base + i];
            }

            alignas(16) uint8_t lut_lo[16];
            alignas(16) uint8_t lut_hi[16];
            build_lut_16_split(x_vals, lut_lo, lut_hi);

            // Broadcast 16-byte LUT to all 4 lanes of 512-bit register
            __m128i lut_lo_128 = _mm_load_si128((__m128i*)lut_lo);
            __m128i lut_hi_128 = _mm_load_si128((__m128i*)lut_hi);
            __m512i lut_lo_512 = _mm512_broadcast_i32x4(lut_lo_128);
            __m512i lut_hi_512 = _mm512_broadcast_i32x4(lut_hi_128);

            const uint8_t* __restrict__ sign_row = sign_plane + kg * N;
            const uint8_t* __restrict__ value_row = value_plane + kg * N;

            // Process 64 outputs at a time with AVX-512 PSHUFB
            int n = 0;
            for (; n + 63 < N; n += 64) {
                // Load 64 sign and value indices
                __m512i sign_idx = _mm512_loadu_si512((__m512i*)(sign_row + n));
                __m512i value_idx = _mm512_loadu_si512((__m512i*)(value_row + n));

                // Mask to 4 bits
                __m512i mask_0f = _mm512_set1_epi8(0x0F);
                sign_idx = _mm512_and_si512(sign_idx, mask_0f);
                value_idx = _mm512_and_si512(value_idx, mask_0f);

                // AVX-512 PSHUFB lookups (64 lookups each!)
                __m512i sign_lo = _mm512_shuffle_epi8(lut_lo_512, sign_idx);
                __m512i sign_hi = _mm512_shuffle_epi8(lut_hi_512, sign_idx);
                __m512i value_lo = _mm512_shuffle_epi8(lut_lo_512, value_idx);
                __m512i value_hi = _mm512_shuffle_epi8(lut_hi_512, value_idx);

                // Unpack to int16 (within each 128-bit lane)
                // Process in 4 groups of 16 elements each
                for (int lane = 0; lane < 4; lane++) {
                    __m128i s_lo = _mm512_extracti32x4_epi32(sign_lo, lane);
                    __m128i s_hi = _mm512_extracti32x4_epi32(sign_hi, lane);
                    __m128i v_lo = _mm512_extracti32x4_epi32(value_lo, lane);
                    __m128i v_hi = _mm512_extracti32x4_epi32(value_hi, lane);

                    __m128i sign_16_lo = _mm_unpacklo_epi8(s_lo, s_hi);
                    __m128i sign_16_hi = _mm_unpackhi_epi8(s_lo, s_hi);
                    __m128i value_16_lo = _mm_unpacklo_epi8(v_lo, v_hi);
                    __m128i value_16_hi = _mm_unpackhi_epi8(v_lo, v_hi);

                    __m128i two = _mm_set1_epi16(2);
                    __m128i res_16_lo = _mm_sub_epi16(_mm_mullo_epi16(value_16_lo, two), sign_16_lo);
                    __m128i res_16_hi = _mm_sub_epi16(_mm_mullo_epi16(value_16_hi, two), sign_16_hi);

                    __m256i res_32_0 = _mm256_cvtepi16_epi32(res_16_lo);
                    __m256i res_32_1 = _mm256_cvtepi16_epi32(res_16_hi);

                    int offset = n + lane * 16;
                    __m256i acc0 = _mm256_loadu_si256((__m256i*)(y_acc + offset));
                    __m256i acc1 = _mm256_loadu_si256((__m256i*)(y_acc + offset + 8));

                    acc0 = _mm256_add_epi32(acc0, res_32_0);
                    acc1 = _mm256_add_epi32(acc1, res_32_1);

                    _mm256_storeu_si256((__m256i*)(y_acc + offset), acc0);
                    _mm256_storeu_si256((__m256i*)(y_acc + offset + 8), acc1);
                }
            }

            // Handle remaining with 128-bit SSE (same as AVX2 version)
            for (; n + 15 < N; n += 16) {
                __m128i sign_idx = _mm_loadu_si128((__m128i*)(sign_row + n));
                __m128i value_idx = _mm_loadu_si128((__m128i*)(value_row + n));

                __m128i mask_0f = _mm_set1_epi8(0x0F);
                sign_idx = _mm_and_si128(sign_idx, mask_0f);
                value_idx = _mm_and_si128(value_idx, mask_0f);

                __m128i sign_lo = _mm_shuffle_epi8(lut_lo_128, sign_idx);
                __m128i sign_hi = _mm_shuffle_epi8(lut_hi_128, sign_idx);
                __m128i value_lo = _mm_shuffle_epi8(lut_lo_128, value_idx);
                __m128i value_hi = _mm_shuffle_epi8(lut_hi_128, value_idx);

                __m128i sign_16_lo = _mm_unpacklo_epi8(sign_lo, sign_hi);
                __m128i sign_16_hi = _mm_unpackhi_epi8(sign_lo, sign_hi);
                __m128i value_16_lo = _mm_unpacklo_epi8(value_lo, value_hi);
                __m128i value_16_hi = _mm_unpackhi_epi8(value_lo, value_hi);

                __m128i two = _mm_set1_epi16(2);
                __m128i res_16_lo = _mm_sub_epi16(_mm_mullo_epi16(value_16_lo, two), sign_16_lo);
                __m128i res_16_hi = _mm_sub_epi16(_mm_mullo_epi16(value_16_hi, two), sign_16_hi);

                __m256i res_32_0 = _mm256_cvtepi16_epi32(res_16_lo);
                __m256i res_32_1 = _mm256_cvtepi16_epi32(res_16_hi);

                __m256i acc0 = _mm256_loadu_si256((__m256i*)(y_acc + n));
                __m256i acc1 = _mm256_loadu_si256((__m256i*)(y_acc + n + 8));

                acc0 = _mm256_add_epi32(acc0, res_32_0);
                acc1 = _mm256_add_epi32(acc1, res_32_1);

                _mm256_storeu_si256((__m256i*)(y_acc + n), acc0);
                _mm256_storeu_si256((__m256i*)(y_acc + n + 8), acc1);
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
            __m512 scale_vec = _mm512_set1_ps(scale);
            int n = 0;
            for (; n + 15 < N; n += 16) {
                __m512i acc = _mm512_loadu_si512((__m512i*)(y_acc + n));
                __m512 acc_f = _mm512_cvtepi32_ps(acc);
                __m512 result = _mm512_mul_ps(acc_f, scale_vec);
                _mm512_storeu_ps(y_row + n, result);
            }
            for (; n < N; n++) {
                y_row[n] = static_cast<float>(y_acc[n]) * scale;
            }
        }
    }
}

// Python bindings
PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("quantize_activations_int8_avx512", &quantize_activations_int8_avx512,
          "Quantize float32 activations to int8");
    m.def("pack_ternary_bitplanes_int8_avx512", &pack_ternary_bitplanes_int8_avx512,
          "Pack ternary weights to bit-plane format");
    m.def("matmul_free_tmac_int8_avx512", &matmul_free_tmac_int8_avx512,
          "T-MAC int8 kernel with AVX-512 PSHUFB");
}
