/*
 * AVX-VNNI Kernel for Intel Core Ultra / Alder Lake+ / AMD Zen5
 *
 * This is a fallback for CPUs without AVX-512 but with AVX-VNNI support.
 * Uses 256-bit vectors instead of 512-bit, providing ~50% of AVX-512 throughput.
 *
 * Key differences from AVX-512 VNNI:
 * - Uses _mm256_* intrinsics (256-bit) instead of _mm512_* (512-bit)
 * - Processes 32 int8 elements per iteration instead of 64
 * - VEX encoding, no masking registers
 * - K padding aligned to 32 bytes instead of 64
 *
 * Supported CPUs:
 * - Intel Alder Lake (12th gen) and newer
 * - Intel Core Ultra (Meteor Lake, Arrow Lake)
 * - AMD Zen 5
 *
 * Compile with: -mavxvnni -mavx2 -mfma
 */

#include <immintrin.h>
#include <cstring>
#include <cstdint>
#include <cmath>
#include <algorithm>
#include <omp.h>
#include <torch/extension.h>

// K padding for 256-bit (32 bytes = 32 int8 elements)
constexpr int K_PAD_AVX = 32;

/**
 * Horizontal sum of 8 int32 elements in __m256i
 */
inline int32_t hsum_epi32_avx2(__m256i v) {
    // Sum pairs: [a+e, b+f, c+g, d+h, ...]
    __m128i lo = _mm256_castsi256_si128(v);
    __m128i hi = _mm256_extracti128_si256(v, 1);
    __m128i sum = _mm_add_epi32(lo, hi);  // 4 elements

    // Horizontal add
    sum = _mm_hadd_epi32(sum, sum);  // 2 elements
    sum = _mm_hadd_epi32(sum, sum);  // 1 element

    return _mm_extract_epi32(sum, 0);
}

/**
 * Quantize activations to int8 with per-row scaling (AVX2 version)
 */
void quantize_activations_int8_avx(
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

        // Find max absolute value using AVX2
        __m256 max_vec = _mm256_setzero_ps();
        int k = 0;
        for (; k + 7 < K; k += 8) {
            __m256 x_vec = _mm256_loadu_ps(x_row + k);
            __m256 abs_vec = _mm256_andnot_ps(_mm256_set1_ps(-0.0f), x_vec);
            max_vec = _mm256_max_ps(max_vec, abs_vec);
        }

        // Horizontal max
        __m128 lo = _mm256_castps256_ps128(max_vec);
        __m128 hi = _mm256_extractf128_ps(max_vec, 1);
        __m128 max128 = _mm_max_ps(lo, hi);
        max128 = _mm_max_ps(max128, _mm_shuffle_ps(max128, max128, _MM_SHUFFLE(2, 3, 0, 1)));
        max128 = _mm_max_ps(max128, _mm_shuffle_ps(max128, max128, _MM_SHUFFLE(1, 0, 3, 2)));
        float max_abs = _mm_cvtss_f32(max128);

        // Handle remainder
        for (; k < K; k++) {
            float abs_val = std::abs(x_row[k]);
            if (abs_val > max_abs) max_abs = abs_val;
        }

        float scale = max_abs / 127.0f;
        if (scale == 0.0f) scale = 1.0f;
        scales[m] = scale;

        float inv_scale = 1.0f / scale;

        // Quantize using AVX2
        __m256 inv_scale_vec = _mm256_set1_ps(inv_scale);
        __m256 min_val = _mm256_set1_ps(-127.0f);
        __m256 max_val = _mm256_set1_ps(127.0f);

        k = 0;
        for (; k + 7 < K; k += 8) {
            __m256 x_vec = _mm256_loadu_ps(x_row + k);
            __m256 scaled = _mm256_mul_ps(x_vec, inv_scale_vec);
            scaled = _mm256_max_ps(min_val, _mm256_min_ps(max_val, scaled));

            // Round and convert to int32
            __m256i rounded = _mm256_cvtps_epi32(_mm256_round_ps(scaled, _MM_FROUND_TO_NEAREST_INT));

            // Pack to int16 then int8
            __m128i lo_32 = _mm256_castsi256_si128(rounded);
            __m128i hi_32 = _mm256_extracti128_si256(rounded, 1);
            __m128i packed_16 = _mm_packs_epi32(lo_32, hi_32);
            __m128i packed_8 = _mm_packs_epi16(packed_16, packed_16);

            // Store 8 bytes
            _mm_storel_epi64((__m128i*)(xq_row + k), packed_8);
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
 * Pack ternary weights to int8 format for AVX-VNNI
 *
 * Layout: [N, K_padded] int8 where K_padded = ceil(K/32)*32
 */
void pack_weights_avx_int8(
    torch::Tensor w_tensor,        // [N, K] float32 ternary
    torch::Tensor w_int8_tensor,   // [N, K_padded] int8 output
    torch::Tensor w_sum_tensor,    // [N] int32 output
    int N, int K
) {
    const float* w = w_tensor.data_ptr<float>();
    int8_t* w_int8 = w_int8_tensor.data_ptr<int8_t>();
    int32_t* w_sum = w_sum_tensor.data_ptr<int32_t>();

    const int K_padded = ((K + K_PAD_AVX - 1) / K_PAD_AVX) * K_PAD_AVX;

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
 * AVX-VNNI v3 - Main kernel with 4-way register blocking
 *
 * Uses _mm256_dpbusd_epi32 for int8 dot products.
 * Processes 32 elements per iteration (vs 64 for AVX-512).
 */
void matmul_free_avx_vnni_v3(
    torch::Tensor x_int8_tensor,    // [M, K] int8
    torch::Tensor scale_tensor,     // [M] float32
    torch::Tensor w_int8_tensor,    // [N, K_padded] int8
    torch::Tensor w_sum_tensor,     // [N] int32
    torch::Tensor y_tensor,         // [M, N] float32 output
    torch::Tensor bias_tensor,
    int M, int N, int K,
    int num_threads
) {
    const int8_t* __restrict__ x_int8 = x_int8_tensor.data_ptr<int8_t>();
    const float* __restrict__ scales = scale_tensor.data_ptr<float>();
    const int8_t* __restrict__ w_int8 = w_int8_tensor.data_ptr<int8_t>();
    const int32_t* __restrict__ w_sum = w_sum_tensor.data_ptr<int32_t>();
    float* __restrict__ y = y_tensor.data_ptr<float>();
    const float* bias = bias_tensor.defined() ? bias_tensor.data_ptr<float>() : nullptr;

    const int K_padded = ((K + K_PAD_AVX - 1) / K_PAD_AVX) * K_PAD_AVX;

    constexpr int N_TILE = 64;
    constexpr int M_TILE = 32;
    constexpr int N_BLOCK = 4;

    omp_set_num_threads(num_threads);

    // Process M in tiles
    for (int m_tile = 0; m_tile < M; m_tile += M_TILE) {
        const int m_end = std::min(m_tile + M_TILE, M);
        const int m_tile_size = m_end - m_tile;

        // Allocate activation buffer for this M tile
        uint8_t* x_uint8_tile = (uint8_t*)aligned_alloc(32, m_tile_size * K_padded);

        // Convert int8 -> uint8 (add 128)
        #pragma omp parallel for schedule(static)
        for (int m = m_tile; m < m_end; m++) {
            const int m_local = m - m_tile;
            uint8_t* x_uint8 = x_uint8_tile + m_local * K_padded;
            const int8_t* x_row = x_int8 + m * K;

            const __m256i offset_vec = _mm256_set1_epi8((char)128);
            int k = 0;
            for (; k + 31 < K; k += 32) {
                __m256i x_vec = _mm256_loadu_si256((__m256i*)(x_row + k));
                __m256i x_u8 = _mm256_add_epi8(x_vec, offset_vec);
                _mm256_store_si256((__m256i*)(x_uint8 + k), x_u8);
            }
            for (; k < K; k++) {
                x_uint8[k] = static_cast<uint8_t>(static_cast<int16_t>(x_row[k]) + 128);
            }
            for (; k < K_padded; k++) {
                x_uint8[k] = 128;  // Neutral value for uint8
            }
        }

        // Process N in tiles
        for (int n_tile = 0; n_tile < N; n_tile += N_TILE) {
            const int n_end = std::min(n_tile + N_TILE, N);

            #pragma omp parallel for schedule(static)
            for (int m = m_tile; m < m_end; m++) {
                const int m_local = m - m_tile;
                const uint8_t* x_uint8 = x_uint8_tile + m_local * K_padded;
                float scale = scales[m];
                float* y_row = y + m * N;

                // Process with 4-way register blocking
                int n = n_tile;
                for (; n + N_BLOCK - 1 < n_end; n += N_BLOCK) {
                    __m256i acc0 = _mm256_setzero_si256();
                    __m256i acc1 = _mm256_setzero_si256();
                    __m256i acc2 = _mm256_setzero_si256();
                    __m256i acc3 = _mm256_setzero_si256();

                    const int8_t* w0 = w_int8 + (n + 0) * K_padded;
                    const int8_t* w1 = w_int8 + (n + 1) * K_padded;
                    const int8_t* w2 = w_int8 + (n + 2) * K_padded;
                    const int8_t* w3 = w_int8 + (n + 3) * K_padded;

                    // Main loop - 32 elements per iteration
                    for (int kk = 0; kk < K_padded; kk += 32) {
                        __m256i x_vec = _mm256_load_si256((__m256i*)(x_uint8 + kk));

                        // AVX-VNNI: _mm256_dpbusd_epi32
                        acc0 = _mm256_dpbusd_epi32(acc0, x_vec, _mm256_loadu_si256((__m256i*)(w0 + kk)));
                        acc1 = _mm256_dpbusd_epi32(acc1, x_vec, _mm256_loadu_si256((__m256i*)(w1 + kk)));
                        acc2 = _mm256_dpbusd_epi32(acc2, x_vec, _mm256_loadu_si256((__m256i*)(w2 + kk)));
                        acc3 = _mm256_dpbusd_epi32(acc3, x_vec, _mm256_loadu_si256((__m256i*)(w3 + kk)));
                    }

                    // Horizontal sum and offset correction
                    int32_t sum0 = hsum_epi32_avx2(acc0) - 128 * w_sum[n + 0];
                    int32_t sum1 = hsum_epi32_avx2(acc1) - 128 * w_sum[n + 1];
                    int32_t sum2 = hsum_epi32_avx2(acc2) - 128 * w_sum[n + 2];
                    int32_t sum3 = hsum_epi32_avx2(acc3) - 128 * w_sum[n + 3];

                    y_row[n + 0] = static_cast<float>(sum0) * scale + (bias ? bias[n + 0] : 0.0f);
                    y_row[n + 1] = static_cast<float>(sum1) * scale + (bias ? bias[n + 1] : 0.0f);
                    y_row[n + 2] = static_cast<float>(sum2) * scale + (bias ? bias[n + 2] : 0.0f);
                    y_row[n + 3] = static_cast<float>(sum3) * scale + (bias ? bias[n + 3] : 0.0f);
                }

                // Remainder
                for (; n < n_end; n++) {
                    const int8_t* w_row = w_int8 + n * K_padded;
                    __m256i acc = _mm256_setzero_si256();

                    for (int kk = 0; kk < K_padded; kk += 32) {
                        __m256i x_vec = _mm256_load_si256((__m256i*)(x_uint8 + kk));
                        acc = _mm256_dpbusd_epi32(acc, x_vec, _mm256_loadu_si256((__m256i*)(w_row + kk)));
                    }

                    int32_t sum = hsum_epi32_avx2(acc) - 128 * w_sum[n];
                    y_row[n] = static_cast<float>(sum) * scale + (bias ? bias[n] : 0.0f);
                }
            }
        }

        free(x_uint8_tile);
    }
}

/**
 * AVX-VNNI v4 - Large N variant with int32 output
 *
 * Outputs raw int32 for later scaling in Python.
 * Better for large N (>= 1024) where parallelization over N is beneficial.
 */
void matmul_free_avx_vnni_v4_large_n(
    torch::Tensor x_int8_tensor,    // [M, K] int8
    torch::Tensor scale_tensor,     // [M] float32 (unused here, applied in Python)
    torch::Tensor w_int8_tensor,    // [N, K_padded] int8
    torch::Tensor w_sum_tensor,     // [N] int32
    torch::Tensor y_tensor,         // [M, N] int32 output
    int M, int N, int K,
    int num_threads
) {
    const int8_t* __restrict__ x_int8 = x_int8_tensor.data_ptr<int8_t>();
    const int8_t* __restrict__ w_int8 = w_int8_tensor.data_ptr<int8_t>();
    const int32_t* __restrict__ w_sum = w_sum_tensor.data_ptr<int32_t>();
    int32_t* __restrict__ y = y_tensor.data_ptr<int32_t>();

    const int K_padded = ((K + K_PAD_AVX - 1) / K_PAD_AVX) * K_PAD_AVX;

    constexpr int N_BLOCK = 8;
    const int n_blocks = (N + N_BLOCK - 1) / N_BLOCK;

    omp_set_num_threads(num_threads);

    // Precompute uint8 activations
    uint8_t* x_uint8_all = (uint8_t*)aligned_alloc(32, M * K_padded);

    #pragma omp parallel for schedule(static)
    for (int m = 0; m < M; m++) {
        uint8_t* x_uint8 = x_uint8_all + m * K_padded;
        const int8_t* x_row = x_int8 + m * K;

        const __m256i offset_vec = _mm256_set1_epi8((char)128);
        int k = 0;
        for (; k + 31 < K; k += 32) {
            __m256i x_vec = _mm256_loadu_si256((__m256i*)(x_row + k));
            __m256i x_u8 = _mm256_add_epi8(x_vec, offset_vec);
            _mm256_store_si256((__m256i*)(x_uint8 + k), x_u8);
        }
        for (; k < K; k++) {
            x_uint8[k] = static_cast<uint8_t>(static_cast<int16_t>(x_row[k]) + 128);
        }
        for (; k < K_padded; k++) {
            x_uint8[k] = 128;
        }
    }

    // Parallelize over N blocks
    #pragma omp parallel for schedule(dynamic, 4)
    for (int nb = 0; nb < n_blocks; nb++) {
        const int n_start = nb * N_BLOCK;
        const int n_end = std::min(n_start + N_BLOCK, N);
        const int block_size = n_end - n_start;

        for (int m = 0; m < M; m++) {
            const uint8_t* x_uint8 = x_uint8_all + m * K_padded;
            int32_t* y_row = y + m * N;

            if (block_size >= 8) {
                __m256i acc[8];
                for (int i = 0; i < 8; i++) {
                    acc[i] = _mm256_setzero_si256();
                }

                const int8_t* w_ptrs[8];
                for (int i = 0; i < 8; i++) {
                    w_ptrs[i] = w_int8 + (n_start + i) * K_padded;
                }

                for (int kk = 0; kk < K_padded; kk += 32) {
                    __m256i x_vec = _mm256_load_si256((__m256i*)(x_uint8 + kk));

                    for (int i = 0; i < 8; i++) {
                        acc[i] = _mm256_dpbusd_epi32(acc[i], x_vec,
                            _mm256_loadu_si256((__m256i*)(w_ptrs[i] + kk)));
                    }
                }

                for (int i = 0; i < 8; i++) {
                    y_row[n_start + i] = hsum_epi32_avx2(acc[i]) - 128 * w_sum[n_start + i];
                }
            } else {
                // Handle remainder
                for (int n = n_start; n < n_end; n++) {
                    const int8_t* w_row = w_int8 + n * K_padded;
                    __m256i acc = _mm256_setzero_si256();

                    for (int kk = 0; kk < K_padded; kk += 32) {
                        __m256i x_vec = _mm256_load_si256((__m256i*)(x_uint8 + kk));
                        acc = _mm256_dpbusd_epi32(acc, x_vec,
                            _mm256_loadu_si256((__m256i*)(w_row + kk)));
                    }

                    y_row[n] = hsum_epi32_avx2(acc) - 128 * w_sum[n];
                }
            }
        }
    }

    free(x_uint8_all);
}

/**
 * AVX-VNNI v4 Fused - Float input with internal quantization
 *
 * Same as NEON fused kernel: takes float activations, quantizes internally.
 */
void matmul_free_avx_vnni_v4_fused(
    torch::Tensor x_float_tensor,   // [M, K] float32 input
    torch::Tensor w_int8_tensor,    // [N, K_padded] int8
    torch::Tensor w_sum_tensor,     // [N] int32
    torch::Tensor y_tensor,         // [M, N] float32 output
    torch::Tensor bias_tensor,
    float weight_scale,
    int M, int N, int K,
    int num_threads
) {
    const float* __restrict__ x_float = x_float_tensor.data_ptr<float>();
    const int8_t* __restrict__ w_int8 = w_int8_tensor.data_ptr<int8_t>();
    const int32_t* __restrict__ w_sum = w_sum_tensor.data_ptr<int32_t>();
    float* __restrict__ y = y_tensor.data_ptr<float>();
    const float* bias = bias_tensor.defined() ? bias_tensor.data_ptr<float>() : nullptr;

    const int K_padded = ((K + K_PAD_AVX - 1) / K_PAD_AVX) * K_PAD_AVX;

    omp_set_num_threads(num_threads);

    #pragma omp parallel
    {
        // Thread-local buffer for quantized activations
        uint8_t* x_uint8_local = (uint8_t*)aligned_alloc(32, K_padded);

        #pragma omp for schedule(static)
        for (int m = 0; m < M; m++) {
            const float* x_row = x_float + m * K;
            float* y_row = y + m * N;

            // Step 1: Find max absolute value
            __m256 max_vec = _mm256_setzero_ps();
            int k = 0;
            for (; k + 7 < K; k += 8) {
                __m256 x_vec = _mm256_loadu_ps(x_row + k);
                __m256 abs_vec = _mm256_andnot_ps(_mm256_set1_ps(-0.0f), x_vec);
                max_vec = _mm256_max_ps(max_vec, abs_vec);
            }

            // Horizontal max
            __m128 lo = _mm256_castps256_ps128(max_vec);
            __m128 hi = _mm256_extractf128_ps(max_vec, 1);
            __m128 max128 = _mm_max_ps(lo, hi);
            max128 = _mm_max_ps(max128, _mm_shuffle_ps(max128, max128, _MM_SHUFFLE(2, 3, 0, 1)));
            max128 = _mm_max_ps(max128, _mm_shuffle_ps(max128, max128, _MM_SHUFFLE(1, 0, 3, 2)));
            float max_abs = _mm_cvtss_f32(max128);

            for (; k < K; k++) {
                float abs_val = std::abs(x_row[k]);
                if (abs_val > max_abs) max_abs = abs_val;
            }

            float act_scale = max_abs / 127.0f;
            if (act_scale == 0.0f) act_scale = 1.0f;
            float inv_scale = 1.0f / act_scale;

            // Step 2: Quantize to uint8 (with +128 offset)
            __m256 inv_scale_vec = _mm256_set1_ps(inv_scale);
            __m256 offset_f = _mm256_set1_ps(128.0f);
            __m256 min_val = _mm256_set1_ps(1.0f);    // 1 = -127 + 128
            __m256 max_val = _mm256_set1_ps(255.0f);  // 255 = 127 + 128

            k = 0;
            for (; k + 7 < K; k += 8) {
                __m256 x_vec = _mm256_loadu_ps(x_row + k);
                __m256 scaled = _mm256_fmadd_ps(x_vec, inv_scale_vec, offset_f);
                scaled = _mm256_max_ps(min_val, _mm256_min_ps(max_val, scaled));

                __m256i rounded = _mm256_cvtps_epi32(_mm256_round_ps(scaled, _MM_FROUND_TO_NEAREST_INT));

                // Pack to uint8
                __m128i lo_32 = _mm256_castsi256_si128(rounded);
                __m128i hi_32 = _mm256_extracti128_si256(rounded, 1);
                __m128i packed_16 = _mm_packus_epi32(lo_32, hi_32);
                __m128i packed_8 = _mm_packus_epi16(packed_16, packed_16);

                _mm_storel_epi64((__m128i*)(x_uint8_local + k), packed_8);
            }

            for (; k < K; k++) {
                float val = x_row[k] * inv_scale + 128.0f;
                val = std::max(1.0f, std::min(255.0f, val));
                x_uint8_local[k] = static_cast<uint8_t>(std::round(val));
            }

            for (k = K; k < K_padded; k++) {
                x_uint8_local[k] = 128;
            }

            float combined_scale = act_scale * weight_scale;

            // Step 3: Compute matmul with 4-way blocking
            int n = 0;
            for (; n + 3 < N; n += 4) {
                __m256i acc0 = _mm256_setzero_si256();
                __m256i acc1 = _mm256_setzero_si256();
                __m256i acc2 = _mm256_setzero_si256();
                __m256i acc3 = _mm256_setzero_si256();

                const int8_t* w0 = w_int8 + (n + 0) * K_padded;
                const int8_t* w1 = w_int8 + (n + 1) * K_padded;
                const int8_t* w2 = w_int8 + (n + 2) * K_padded;
                const int8_t* w3 = w_int8 + (n + 3) * K_padded;

                for (int kk = 0; kk < K_padded; kk += 32) {
                    __m256i x_vec = _mm256_load_si256((__m256i*)(x_uint8_local + kk));

                    acc0 = _mm256_dpbusd_epi32(acc0, x_vec, _mm256_loadu_si256((__m256i*)(w0 + kk)));
                    acc1 = _mm256_dpbusd_epi32(acc1, x_vec, _mm256_loadu_si256((__m256i*)(w1 + kk)));
                    acc2 = _mm256_dpbusd_epi32(acc2, x_vec, _mm256_loadu_si256((__m256i*)(w2 + kk)));
                    acc3 = _mm256_dpbusd_epi32(acc3, x_vec, _mm256_loadu_si256((__m256i*)(w3 + kk)));
                }

                int32_t sum0 = hsum_epi32_avx2(acc0) - 128 * w_sum[n + 0];
                int32_t sum1 = hsum_epi32_avx2(acc1) - 128 * w_sum[n + 1];
                int32_t sum2 = hsum_epi32_avx2(acc2) - 128 * w_sum[n + 2];
                int32_t sum3 = hsum_epi32_avx2(acc3) - 128 * w_sum[n + 3];

                y_row[n + 0] = static_cast<float>(sum0) * combined_scale + (bias ? bias[n + 0] : 0.0f);
                y_row[n + 1] = static_cast<float>(sum1) * combined_scale + (bias ? bias[n + 1] : 0.0f);
                y_row[n + 2] = static_cast<float>(sum2) * combined_scale + (bias ? bias[n + 2] : 0.0f);
                y_row[n + 3] = static_cast<float>(sum3) * combined_scale + (bias ? bias[n + 3] : 0.0f);
            }

            // Remainder
            for (; n < N; n++) {
                const int8_t* w_row = w_int8 + n * K_padded;
                __m256i acc = _mm256_setzero_si256();

                for (int kk = 0; kk < K_padded; kk += 32) {
                    __m256i x_vec = _mm256_load_si256((__m256i*)(x_uint8_local + kk));
                    acc = _mm256_dpbusd_epi32(acc, x_vec, _mm256_loadu_si256((__m256i*)(w_row + kk)));
                }

                int32_t sum = hsum_epi32_avx2(acc) - 128 * w_sum[n];
                y_row[n] = static_cast<float>(sum) * combined_scale + (bias ? bias[n] : 0.0f);
            }
        }

        free(x_uint8_local);
    }
}

// Python bindings
PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("quantize_activations_int8_avx", &quantize_activations_int8_avx,
          "Quantize activations to int8 (AVX2)");
    m.def("pack_weights_avx_int8", &pack_weights_avx_int8,
          "Pack ternary weights for AVX-VNNI kernel");
    m.def("matmul_free_avx_vnni_v3", &matmul_free_avx_vnni_v3,
          "AVX-VNNI v3: 4-way register blocking");
    m.def("matmul_free_avx_vnni_v4_large_n", &matmul_free_avx_vnni_v4_large_n,
          "AVX-VNNI v4: Large N with int32 output");
    m.def("matmul_free_avx_vnni_v4_fused", &matmul_free_avx_vnni_v4_fused,
          "AVX-VNNI v4 Fused: Float input, internal quantization");
}
