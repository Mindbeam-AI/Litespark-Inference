/*
 * T-MAC Int8 Ternary MatMul-Free Kernel for x86_64 AVX2
 *
 * Based on Microsoft T-MAC and BitNet.cpp:
 * - Activations: int8 (per-tensor quantization)
 * - LUT: int8 values
 * - Accumulation: int32
 * - PSHUFB for 32 parallel lookups per instruction
 *
 * Key insight: PSHUFB (_mm256_shuffle_epi8) does 32 parallel int8 lookups!
 * This is why int8 is critical - float32 can't use PSHUFB for computation.
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
 *
 * Per-tensor symmetric quantization:
 *   scale = max(abs(x)) / 127
 *   x_int8 = round(x / scale)
 */
void quantize_activations_int8(
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

        // Find max absolute value
        float max_abs = 0.0f;
        for (int k = 0; k < K; k++) {
            float abs_val = std::abs(x_row[k]);
            if (abs_val > max_abs) max_abs = abs_val;
        }

        // Compute scale
        float scale = max_abs / 127.0f;
        if (scale == 0.0f) scale = 1.0f;  // Avoid division by zero
        scales[m] = scale;

        // Quantize
        float inv_scale = 1.0f / scale;
        for (int k = 0; k < K; k++) {
            float val = x_row[k] * inv_scale;
            // Clamp to [-127, 127]
            val = std::max(-127.0f, std::min(127.0f, val));
            xq_row[k] = static_cast<int8_t>(std::round(val));
        }
    }
}

/**
 * Pack ternary weights into bit-plane format for T-MAC
 *
 * Layout: [K_groups, N] where K_groups = ceil(K/4)
 * Each byte stores 4 weights as a nibble index for 16-entry LUT
 *
 * Weight encoding (bit-plane decomposition):
 *   sign_plane[i] = 1 if weight[i] != 0
 *   value_plane[i] = 1 if weight[i] == +1
 */
void pack_ternary_bitplanes_int8(
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
 * Build 16-entry int8 LUT from 4 int8 activations
 *
 * Each entry is the sum of selected activations.
 * For ternary: result = 2*value_sum - sign_sum
 */
inline void build_lut_16_int8(const int8_t* x, int8_t* lut) {
    const int16_t x0 = x[0], x1 = x[1], x2 = x[2], x3 = x[3];

    // Compute sums (use int16 to avoid overflow, then clamp)
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

    // Clamp to int8 range and store
    for (int i = 0; i < 16; i++) {
        sums[i] = std::max(int16_t(-128), std::min(int16_t(127), sums[i]));
        lut[i] = static_cast<int8_t>(sums[i]);
    }
}

/**
 * T-MAC Int8 kernel with PSHUFB
 *
 * Uses PSHUFB for 32 parallel int8 lookups per instruction!
 * Accumulates in int32 to avoid overflow.
 */
void matmul_free_tmac_int8(
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

    #pragma omp parallel for schedule(static)
    for (int m = 0; m < M; m++) {
        const int8_t* __restrict__ x_row = x_int8 + m * K;
        float scale = scales[m];

        // Accumulate in int32 to avoid overflow
        alignas(64) int32_t y_acc[N];
        memset(y_acc, 0, N * sizeof(int32_t));

        // K-first iteration
        for (int kg = 0; kg < K_groups; kg++) {
            const int k_base = kg * 4;

            // Get 4 int8 activations
            int8_t x_vals[4] = {0, 0, 0, 0};
            for (int i = 0; i < 4 && k_base + i < K; i++) {
                x_vals[i] = x_row[k_base + i];
            }

            // Build int8 LUT
            alignas(32) int8_t lut[16];
            build_lut_16_int8(x_vals, lut);

            // Duplicate LUT for AVX2 PSHUFB (needs same LUT in both 128-bit lanes)
            __m256i lut_vec = _mm256_broadcastsi128_si256(_mm_load_si128((__m128i*)lut));

            const uint8_t* __restrict__ sign_row = sign_plane + kg * N;
            const uint8_t* __restrict__ value_row = value_plane + kg * N;

            // Process 32 outputs at a time with PSHUFB
            int n = 0;
            for (; n + 31 < N; n += 32) {
                // Load 32 sign and value indices
                __m256i sign_idx = _mm256_loadu_si256((__m256i*)(sign_row + n));
                __m256i value_idx = _mm256_loadu_si256((__m256i*)(value_row + n));

                // Mask to 4 bits (indices 0-15)
                __m256i mask = _mm256_set1_epi8(0x0F);
                sign_idx = _mm256_and_si256(sign_idx, mask);
                value_idx = _mm256_and_si256(value_idx, mask);

                // PSHUFB: 32 parallel lookups!
                __m256i sign_vals = _mm256_shuffle_epi8(lut_vec, sign_idx);
                __m256i value_vals = _mm256_shuffle_epi8(lut_vec, value_idx);

                // result = 2*value - sign (as int8, then accumulate to int32)
                // First compute 2*value - sign in int16 to avoid overflow

                // Unpack to int16 for computation
                __m256i sign_lo = _mm256_cvtepi8_epi16(_mm256_extracti128_si256(sign_vals, 0));
                __m256i sign_hi = _mm256_cvtepi8_epi16(_mm256_extracti128_si256(sign_vals, 1));
                __m256i value_lo = _mm256_cvtepi8_epi16(_mm256_extracti128_si256(value_vals, 0));
                __m256i value_hi = _mm256_cvtepi8_epi16(_mm256_extracti128_si256(value_vals, 1));

                // 2*value - sign
                __m256i two = _mm256_set1_epi16(2);
                __m256i result_lo = _mm256_sub_epi16(_mm256_mullo_epi16(two, value_lo), sign_lo);
                __m256i result_hi = _mm256_sub_epi16(_mm256_mullo_epi16(two, value_hi), sign_hi);

                // Accumulate to int32
                // result_lo has 16 int16 values, need to extend to 4 groups of 8 int32
                __m256i res_lo_lo = _mm256_cvtepi16_epi32(_mm256_extracti128_si256(result_lo, 0));
                __m256i res_lo_hi = _mm256_cvtepi16_epi32(_mm256_extracti128_si256(result_lo, 1));
                __m256i res_hi_lo = _mm256_cvtepi16_epi32(_mm256_extracti128_si256(result_hi, 0));
                __m256i res_hi_hi = _mm256_cvtepi16_epi32(_mm256_extracti128_si256(result_hi, 1));

                // Load current accumulators and add
                __m256i acc0 = _mm256_loadu_si256((__m256i*)(y_acc + n));
                __m256i acc1 = _mm256_loadu_si256((__m256i*)(y_acc + n + 8));
                __m256i acc2 = _mm256_loadu_si256((__m256i*)(y_acc + n + 16));
                __m256i acc3 = _mm256_loadu_si256((__m256i*)(y_acc + n + 24));

                acc0 = _mm256_add_epi32(acc0, res_lo_lo);
                acc1 = _mm256_add_epi32(acc1, res_lo_hi);
                acc2 = _mm256_add_epi32(acc2, res_hi_lo);
                acc3 = _mm256_add_epi32(acc3, res_hi_hi);

                _mm256_storeu_si256((__m256i*)(y_acc + n), acc0);
                _mm256_storeu_si256((__m256i*)(y_acc + n + 8), acc1);
                _mm256_storeu_si256((__m256i*)(y_acc + n + 16), acc2);
                _mm256_storeu_si256((__m256i*)(y_acc + n + 24), acc3);
            }

            // Handle remaining outputs
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
            __m256 scale_vec = _mm256_set1_ps(scale);
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
 * Scalar reference implementation for correctness testing
 */
void matmul_free_tmac_int8_scalar(
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

        // Initialize with bias
        for (int n = 0; n < N; n++) {
            y_row[n] = bias ? bias[n] : 0.0f;
        }

        // K-first iteration
        for (int kg = 0; kg < K_groups; kg++) {
            const int k_base = kg * 4;

            // Build LUT
            int8_t x_vals[4] = {0, 0, 0, 0};
            for (int i = 0; i < 4 && k_base + i < K; i++) {
                x_vals[i] = x_row[k_base + i];
            }

            int8_t lut[16];
            build_lut_16_int8(x_vals, lut);

            const uint8_t* sign_row = sign_plane + kg * N;
            const uint8_t* value_row = value_plane + kg * N;

            for (int n = 0; n < N; n++) {
                uint8_t s = sign_row[n] & 0x0F;
                uint8_t v = value_row[n] & 0x0F;
                // Accumulate scaled result
                int16_t result = 2 * static_cast<int16_t>(lut[v]) - static_cast<int16_t>(lut[s]);
                y_row[n] += static_cast<float>(result) * scale;
            }
        }
    }
}

// Python bindings
PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("quantize_activations_int8", &quantize_activations_int8,
          "Quantize float32 activations to int8 with per-row scale");
    m.def("pack_ternary_bitplanes_int8", &pack_ternary_bitplanes_int8,
          "Pack ternary weights to bit-plane format");
    m.def("matmul_free_tmac_int8", &matmul_free_tmac_int8,
          "T-MAC kernel with int8 quantization and PSHUFB");
    m.def("matmul_free_tmac_int8_scalar", &matmul_free_tmac_int8_scalar,
          "T-MAC int8 scalar reference");
}
