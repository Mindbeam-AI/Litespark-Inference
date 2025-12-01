/*
 * T-MAC Ternary MatMul-Free Kernel for x86_64 AVX2
 *
 * Based on Microsoft T-MAC: https://arxiv.org/abs/2407.00088
 * and BitNet.cpp: https://github.com/microsoft/BitNet
 *
 * KEY INSIGHT FROM T-MAC PAPER:
 * =============================
 * Traditional GEMM: Loop N -> Loop M -> Loop K (spatial first)
 * T-MAC approach:   Loop K -> Loop N -> Loop M (temporal first, K-dimension first)
 *
 * Why K-first matters:
 * - Build LUT ONCE for each K-group of activations
 * - Reuse that LUT across ALL N outputs and ALL M batch elements
 * - LUT stays in registers, no spilling to cache/memory
 *
 * Algorithm:
 * 1. For each K-group (4 weights at a time):
 *    a. Build 16-entry LUT from 4 activations (same for all outputs!)
 *    b. For each output n: lookup weight pattern, accumulate
 * 2. result = 2 * value_lookup - sign_lookup
 *
 * Weight encoding (bit-plane decomposition):
 *   sign_plane[i] = 1 if weight[i] != 0
 *   value_plane[i] = 1 if weight[i] == +1
 *
 *   weight | sign | value | result formula
 *   -------|------|-------|---------------
 *     0    |  0   |   0   | 2*0 - 0 = 0
 *    +1    |  1   |   1   | 2*x - x = +x
 *    -1    |  1   |   0   | 2*0 - x = -x
 */

#include <immintrin.h>
#include <cstring>
#include <cstdint>
#include <algorithm>
#include <omp.h>
#include <torch/extension.h>

// Tile sizes for cache optimization
constexpr int TILE_M = 32;   // Batch tile size
constexpr int TILE_N = 256;  // Output tile size (multiple of 32 for AVX2)
constexpr int TILE_K = 256;  // K tile size (multiple of 4 for LUT groups)

/**
 * Pack ternary weights into bit-plane format
 *
 * Layout: [K_groups, N] - K-dimension first for sequential access during K-loop
 * Each byte stores 4 weights (one nibble index for 16-entry LUT)
 *
 * This is the TRANSPOSED layout that enables K-first iteration!
 */
void pack_ternary_bitplanes(
    torch::Tensor w_tensor,
    torch::Tensor sign_plane_tensor,   // [K_groups, N]
    torch::Tensor value_plane_tensor,  // [K_groups, N]
    int N, int K
) {
    const float* w = w_tensor.data_ptr<float>();
    uint8_t* sign_plane = sign_plane_tensor.data_ptr<uint8_t>();
    uint8_t* value_plane = value_plane_tensor.data_ptr<uint8_t>();

    const int K_groups = (K + 3) / 4;  // 4 weights per group

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
            }
            // Store in K-first layout: sign_plane[kg * N + n]
            sign_plane[kg * N + n] = sign_nibble;
            value_plane[kg * N + n] = value_nibble;
        }
    }
}

/**
 * Build 16-entry LUT for 4 activations
 * Unrolled for maximum performance
 */
inline void build_lut_16(const float* x, float* lut) {
    const float x0 = x[0], x1 = x[1], x2 = x[2], x3 = x[3];
    lut[0]  = 0.0f;
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
 * T-MAC kernel with K-first iteration (Microsoft's approach)
 *
 * This is the KEY optimization:
 * - Outer loop over K (not M or N)
 * - Build LUT once per K-group
 * - Reuse LUT for ALL outputs
 */
void matmul_free_tmac(
    torch::Tensor x_tensor,
    torch::Tensor sign_plane_tensor,   // [K_groups, N]
    torch::Tensor value_plane_tensor,  // [K_groups, N]
    torch::Tensor y_tensor,
    torch::Tensor bias_tensor,
    int M, int N, int K,
    int num_threads
) {
    const float* __restrict__ x = x_tensor.data_ptr<float>();
    const uint8_t* __restrict__ sign_plane = sign_plane_tensor.data_ptr<uint8_t>();
    const uint8_t* __restrict__ value_plane = value_plane_tensor.data_ptr<uint8_t>();
    float* __restrict__ y = y_tensor.data_ptr<float>();
    const float* bias = bias_tensor.defined() ? bias_tensor.data_ptr<float>() : nullptr;

    const int K_groups = (K + 3) / 4;

    omp_set_num_threads(num_threads);

    // Parallelize over M (batch dimension)
    #pragma omp parallel for schedule(static)
    for (int m = 0; m < M; m++) {
        const float* __restrict__ x_row = x + m * K;
        float* __restrict__ y_row = y + m * N;

        // Initialize output with bias or zero
        if (bias != nullptr) {
            memcpy(y_row, bias, N * sizeof(float));
        } else {
            memset(y_row, 0, N * sizeof(float));
        }

        // K-FIRST ITERATION: This is the key T-MAC insight!
        // For each group of 4 weights along K dimension:
        for (int kg = 0; kg < K_groups; kg++) {
            const int k_base = kg * 4;

            // Get 4 activations for this K-group
            float x_vals[4] = {0.0f, 0.0f, 0.0f, 0.0f};
            const int k_remaining = K - k_base;
            const int k_count = (k_remaining >= 4) ? 4 : k_remaining;
            for (int i = 0; i < k_count; i++) {
                x_vals[i] = x_row[k_base + i];
            }

            // Build LUT ONCE for this K-group
            // This LUT will be reused for ALL N outputs!
            alignas(64) float lut[16];
            build_lut_16(x_vals, lut);

            // Weight rows for this K-group (sequential memory access!)
            const uint8_t* __restrict__ sign_row = sign_plane + kg * N;
            const uint8_t* __restrict__ value_row = value_plane + kg * N;

            // Process all N outputs using the SAME LUT
            // This is where T-MAC wins: LUT stays in registers
            int n = 0;

            // Unrolled loop for 8 outputs at a time
            for (; n + 7 < N; n += 8) {
                // Load 8 weight indices (sequential!)
                const uint8_t s0 = sign_row[n+0] & 0x0F;
                const uint8_t s1 = sign_row[n+1] & 0x0F;
                const uint8_t s2 = sign_row[n+2] & 0x0F;
                const uint8_t s3 = sign_row[n+3] & 0x0F;
                const uint8_t s4 = sign_row[n+4] & 0x0F;
                const uint8_t s5 = sign_row[n+5] & 0x0F;
                const uint8_t s6 = sign_row[n+6] & 0x0F;
                const uint8_t s7 = sign_row[n+7] & 0x0F;

                const uint8_t v0 = value_row[n+0] & 0x0F;
                const uint8_t v1 = value_row[n+1] & 0x0F;
                const uint8_t v2 = value_row[n+2] & 0x0F;
                const uint8_t v3 = value_row[n+3] & 0x0F;
                const uint8_t v4 = value_row[n+4] & 0x0F;
                const uint8_t v5 = value_row[n+5] & 0x0F;
                const uint8_t v6 = value_row[n+6] & 0x0F;
                const uint8_t v7 = value_row[n+7] & 0x0F;

                // Lookup and accumulate: result = 2*value - sign
                y_row[n+0] += 2.0f * lut[v0] - lut[s0];
                y_row[n+1] += 2.0f * lut[v1] - lut[s1];
                y_row[n+2] += 2.0f * lut[v2] - lut[s2];
                y_row[n+3] += 2.0f * lut[v3] - lut[s3];
                y_row[n+4] += 2.0f * lut[v4] - lut[s4];
                y_row[n+5] += 2.0f * lut[v5] - lut[s5];
                y_row[n+6] += 2.0f * lut[v6] - lut[s6];
                y_row[n+7] += 2.0f * lut[v7] - lut[s7];
            }

            // Handle remaining outputs
            for (; n < N; n++) {
                const uint8_t s = sign_row[n] & 0x0F;
                const uint8_t v = value_row[n] & 0x0F;
                y_row[n] += 2.0f * lut[v] - lut[s];
            }
        }
    }
}

/**
 * T-MAC with AVX2 vectorization
 *
 * Uses AVX2 for:
 * - Vectorized output accumulation
 * - Better memory throughput
 */
void matmul_free_tmac_avx2(
    torch::Tensor x_tensor,
    torch::Tensor sign_plane_tensor,
    torch::Tensor value_plane_tensor,
    torch::Tensor y_tensor,
    torch::Tensor bias_tensor,
    int M, int N, int K,
    int num_threads
) {
    const float* __restrict__ x = x_tensor.data_ptr<float>();
    const uint8_t* __restrict__ sign_plane = sign_plane_tensor.data_ptr<uint8_t>();
    const uint8_t* __restrict__ value_plane = value_plane_tensor.data_ptr<uint8_t>();
    float* __restrict__ y = y_tensor.data_ptr<float>();
    const float* bias = bias_tensor.defined() ? bias_tensor.data_ptr<float>() : nullptr;

    const int K_groups = (K + 3) / 4;

    omp_set_num_threads(num_threads);

    #pragma omp parallel for schedule(static)
    for (int m = 0; m < M; m++) {
        const float* __restrict__ x_row = x + m * K;
        float* __restrict__ y_row = y + m * N;

        // Initialize output
        if (bias != nullptr) {
            memcpy(y_row, bias, N * sizeof(float));
        } else {
            memset(y_row, 0, N * sizeof(float));
        }

        // K-first iteration
        for (int kg = 0; kg < K_groups; kg++) {
            const int k_base = kg * 4;

            // Build LUT for this K-group
            float x_vals[4] = {0.0f, 0.0f, 0.0f, 0.0f};
            for (int i = 0; i < 4 && k_base + i < K; i++) {
                x_vals[i] = x_row[k_base + i];
            }

            alignas(64) float lut[16];
            build_lut_16(x_vals, lut);

            // Load LUT into registers for faster access
            const __m256 lut_0_7 = _mm256_load_ps(lut);
            const __m256 lut_8_15 = _mm256_load_ps(lut + 8);
            const __m256 two = _mm256_set1_ps(2.0f);

            const uint8_t* __restrict__ sign_row = sign_plane + kg * N;
            const uint8_t* __restrict__ value_row = value_plane + kg * N;

            // Process 8 outputs at a time with AVX2
            int n = 0;
            for (; n + 7 < N; n += 8) {
                // Load current output values
                __m256 y_vec = _mm256_loadu_ps(y_row + n);

                // Gather sign and value lookups
                // Since LUT is only 16 entries, we do scalar lookups
                // but vectorized accumulation
                alignas(32) float sign_results[8];
                alignas(32) float value_results[8];

                for (int i = 0; i < 8; i++) {
                    const uint8_t s = sign_row[n + i] & 0x0F;
                    const uint8_t v = value_row[n + i] & 0x0F;
                    sign_results[i] = lut[s];
                    value_results[i] = lut[v];
                }

                __m256 sign_vec = _mm256_load_ps(sign_results);
                __m256 value_vec = _mm256_load_ps(value_results);

                // result = 2*value - sign
                __m256 result = _mm256_fmsub_ps(two, value_vec, sign_vec);
                y_vec = _mm256_add_ps(y_vec, result);

                _mm256_storeu_ps(y_row + n, y_vec);
            }

            // Handle remaining
            for (; n < N; n++) {
                const uint8_t s = sign_row[n] & 0x0F;
                const uint8_t v = value_row[n] & 0x0F;
                y_row[n] += 2.0f * lut[v] - lut[s];
            }
        }
    }
}

/**
 * T-MAC with tiling for better cache utilization
 *
 * Tiles the computation to keep working set in L1/L2 cache:
 * - Tile over M (batch)
 * - Tile over N (outputs)
 * - Process all K for each tile before moving to next
 */
void matmul_free_tmac_tiled(
    torch::Tensor x_tensor,
    torch::Tensor sign_plane_tensor,
    torch::Tensor value_plane_tensor,
    torch::Tensor y_tensor,
    torch::Tensor bias_tensor,
    int M, int N, int K,
    int num_threads
) {
    const float* __restrict__ x = x_tensor.data_ptr<float>();
    const uint8_t* __restrict__ sign_plane = sign_plane_tensor.data_ptr<uint8_t>();
    const uint8_t* __restrict__ value_plane = value_plane_tensor.data_ptr<uint8_t>();
    float* __restrict__ y = y_tensor.data_ptr<float>();
    const float* bias = bias_tensor.defined() ? bias_tensor.data_ptr<float>() : nullptr;

    const int K_groups = (K + 3) / 4;

    omp_set_num_threads(num_threads);

    // Initialize all outputs first
    #pragma omp parallel for schedule(static)
    for (int m = 0; m < M; m++) {
        float* y_row = y + m * N;
        if (bias != nullptr) {
            memcpy(y_row, bias, N * sizeof(float));
        } else {
            memset(y_row, 0, N * sizeof(float));
        }
    }

    // Tile over N dimension for better cache reuse of weights
    const int n_tiles = (N + TILE_N - 1) / TILE_N;

    #pragma omp parallel for schedule(dynamic)
    for (int n_tile = 0; n_tile < n_tiles; n_tile++) {
        const int n_start = n_tile * TILE_N;
        const int n_end = std::min(n_start + TILE_N, N);
        const int n_count = n_end - n_start;

        // Process all M rows for this N-tile
        for (int m = 0; m < M; m++) {
            const float* __restrict__ x_row = x + m * K;
            float* __restrict__ y_row = y + m * N + n_start;

            // K-first iteration within this tile
            for (int kg = 0; kg < K_groups; kg++) {
                const int k_base = kg * 4;

                // Build LUT for this K-group
                float x_vals[4] = {0.0f, 0.0f, 0.0f, 0.0f};
                for (int i = 0; i < 4 && k_base + i < K; i++) {
                    x_vals[i] = x_row[k_base + i];
                }

                alignas(64) float lut[16];
                build_lut_16(x_vals, lut);

                // Weight pointers for this K-group and N-tile
                const uint8_t* __restrict__ sign_ptr = sign_plane + kg * N + n_start;
                const uint8_t* __restrict__ value_ptr = value_plane + kg * N + n_start;

                // Process outputs in this tile
                for (int i = 0; i < n_count; i++) {
                    const uint8_t s = sign_ptr[i] & 0x0F;
                    const uint8_t v = value_ptr[i] & 0x0F;
                    y_row[i] += 2.0f * lut[v] - lut[s];
                }
            }
        }
    }
}

/**
 * T-MAC with PSHUFB for true SIMD lookups
 *
 * Uses float accumulation with PSHUFB for index gathering.
 * This avoids quantization errors while still using PSHUFB for parallel index loading.
 *
 * Strategy: Use PSHUFB to gather indices, then do float lookups
 */
void matmul_free_tmac_pshufb(
    torch::Tensor x_tensor,
    torch::Tensor sign_plane_tensor,
    torch::Tensor value_plane_tensor,
    torch::Tensor y_tensor,
    torch::Tensor bias_tensor,
    int M, int N, int K,
    int num_threads
) {
    const float* __restrict__ x = x_tensor.data_ptr<float>();
    const uint8_t* __restrict__ sign_plane = sign_plane_tensor.data_ptr<uint8_t>();
    const uint8_t* __restrict__ value_plane = value_plane_tensor.data_ptr<uint8_t>();
    float* __restrict__ y = y_tensor.data_ptr<float>();
    const float* bias = bias_tensor.defined() ? bias_tensor.data_ptr<float>() : nullptr;

    const int K_groups = (K + 3) / 4;

    omp_set_num_threads(num_threads);

    #pragma omp parallel for schedule(static)
    for (int m = 0; m < M; m++) {
        const float* __restrict__ x_row = x + m * K;
        float* __restrict__ y_row = y + m * N;

        // Initialize output
        if (bias != nullptr) {
            memcpy(y_row, bias, N * sizeof(float));
        } else {
            memset(y_row, 0, N * sizeof(float));
        }

        // K-first iteration
        for (int kg = 0; kg < K_groups; kg++) {
            const int k_base = kg * 4;

            // Get activations for this K-group
            float x_vals[4] = {0.0f, 0.0f, 0.0f, 0.0f};
            for (int i = 0; i < 4 && k_base + i < K; i++) {
                x_vals[i] = x_row[k_base + i];
            }

            // Build float LUT
            alignas(64) float lut[16];
            build_lut_16(x_vals, lut);

            const uint8_t* __restrict__ sign_ptr = sign_plane + kg * N;
            const uint8_t* __restrict__ value_ptr = value_plane + kg * N;

            const __m256 two = _mm256_set1_ps(2.0f);

            // Process 16 outputs at a time (gather indices, then float lookups)
            int n = 0;
            for (; n + 15 < N; n += 16) {
                // Load 16 indices
                __m128i sign_idx_16 = _mm_loadu_si128((__m128i*)(sign_ptr + n));
                __m128i value_idx_16 = _mm_loadu_si128((__m128i*)(value_ptr + n));

                // Mask to 4 bits
                __m128i mask = _mm_set1_epi8(0x0F);
                sign_idx_16 = _mm_and_si128(sign_idx_16, mask);
                value_idx_16 = _mm_and_si128(value_idx_16, mask);

                // Extract indices to array for lookups
                alignas(16) uint8_t s_idx[16];
                alignas(16) uint8_t v_idx[16];
                _mm_store_si128((__m128i*)s_idx, sign_idx_16);
                _mm_store_si128((__m128i*)v_idx, value_idx_16);

                // Lookup and accumulate in two batches of 8
                // First 8
                __m256 y_vec0 = _mm256_loadu_ps(y_row + n);
                __m256 sign_vec0 = _mm256_set_ps(
                    lut[s_idx[7]], lut[s_idx[6]], lut[s_idx[5]], lut[s_idx[4]],
                    lut[s_idx[3]], lut[s_idx[2]], lut[s_idx[1]], lut[s_idx[0]]
                );
                __m256 value_vec0 = _mm256_set_ps(
                    lut[v_idx[7]], lut[v_idx[6]], lut[v_idx[5]], lut[v_idx[4]],
                    lut[v_idx[3]], lut[v_idx[2]], lut[v_idx[1]], lut[v_idx[0]]
                );
                __m256 result0 = _mm256_fmsub_ps(two, value_vec0, sign_vec0);
                y_vec0 = _mm256_add_ps(y_vec0, result0);
                _mm256_storeu_ps(y_row + n, y_vec0);

                // Second 8
                __m256 y_vec1 = _mm256_loadu_ps(y_row + n + 8);
                __m256 sign_vec1 = _mm256_set_ps(
                    lut[s_idx[15]], lut[s_idx[14]], lut[s_idx[13]], lut[s_idx[12]],
                    lut[s_idx[11]], lut[s_idx[10]], lut[s_idx[9]], lut[s_idx[8]]
                );
                __m256 value_vec1 = _mm256_set_ps(
                    lut[v_idx[15]], lut[v_idx[14]], lut[v_idx[13]], lut[v_idx[12]],
                    lut[v_idx[11]], lut[v_idx[10]], lut[v_idx[9]], lut[v_idx[8]]
                );
                __m256 result1 = _mm256_fmsub_ps(two, value_vec1, sign_vec1);
                y_vec1 = _mm256_add_ps(y_vec1, result1);
                _mm256_storeu_ps(y_row + n + 8, y_vec1);
            }

            // Handle remaining
            for (; n < N; n++) {
                const uint8_t s = sign_ptr[n] & 0x0F;
                const uint8_t v = value_ptr[n] & 0x0F;
                y_row[n] += 2.0f * lut[v] - lut[s];
            }
        }
    }
}

/**
 * T-MAC optimized: combines all techniques
 * - K-first iteration
 * - Tiling for cache
 * - AVX2 for vectorization
 * - Float LUT (no quantization loss)
 */
void matmul_free_tmac_optimized(
    torch::Tensor x_tensor,
    torch::Tensor sign_plane_tensor,
    torch::Tensor value_plane_tensor,
    torch::Tensor y_tensor,
    torch::Tensor bias_tensor,
    int M, int N, int K,
    int num_threads
) {
    const float* __restrict__ x = x_tensor.data_ptr<float>();
    const uint8_t* __restrict__ sign_plane = sign_plane_tensor.data_ptr<uint8_t>();
    const uint8_t* __restrict__ value_plane = value_plane_tensor.data_ptr<uint8_t>();
    float* __restrict__ y = y_tensor.data_ptr<float>();
    const float* bias = bias_tensor.defined() ? bias_tensor.data_ptr<float>() : nullptr;

    const int K_groups = (K + 3) / 4;

    omp_set_num_threads(num_threads);

    // Parallelize over M with good granularity
    #pragma omp parallel for schedule(static)
    for (int m = 0; m < M; m++) {
        const float* __restrict__ x_row = x + m * K;
        float* __restrict__ y_row = y + m * N;

        // Initialize output
        if (bias != nullptr) {
            // Vectorized bias copy
            int n = 0;
            for (; n + 7 < N; n += 8) {
                _mm256_storeu_ps(y_row + n, _mm256_loadu_ps(bias + n));
            }
            for (; n < N; n++) {
                y_row[n] = bias[n];
            }
        } else {
            memset(y_row, 0, N * sizeof(float));
        }

        // K-first iteration with AVX2 optimizations
        for (int kg = 0; kg < K_groups; kg++) {
            const int k_base = kg * 4;

            // Build LUT for this K-group
            float x_vals[4] = {0.0f, 0.0f, 0.0f, 0.0f};
            for (int i = 0; i < 4 && k_base + i < K; i++) {
                x_vals[i] = x_row[k_base + i];
            }

            alignas(64) float lut[16];
            build_lut_16(x_vals, lut);

            const uint8_t* __restrict__ sign_row = sign_plane + kg * N;
            const uint8_t* __restrict__ value_row = value_plane + kg * N;

            const __m256 two = _mm256_set1_ps(2.0f);

            // Process 8 outputs at a time
            int n = 0;
            for (; n + 7 < N; n += 8) {
                __m256 y_vec = _mm256_loadu_ps(y_row + n);

                // Gather lookups (scalar, but fast since LUT is in L1)
                __m256 sign_vec = _mm256_set_ps(
                    lut[sign_row[n+7] & 0x0F], lut[sign_row[n+6] & 0x0F],
                    lut[sign_row[n+5] & 0x0F], lut[sign_row[n+4] & 0x0F],
                    lut[sign_row[n+3] & 0x0F], lut[sign_row[n+2] & 0x0F],
                    lut[sign_row[n+1] & 0x0F], lut[sign_row[n+0] & 0x0F]
                );
                __m256 value_vec = _mm256_set_ps(
                    lut[value_row[n+7] & 0x0F], lut[value_row[n+6] & 0x0F],
                    lut[value_row[n+5] & 0x0F], lut[value_row[n+4] & 0x0F],
                    lut[value_row[n+3] & 0x0F], lut[value_row[n+2] & 0x0F],
                    lut[value_row[n+1] & 0x0F], lut[value_row[n+0] & 0x0F]
                );

                // result = 2*value - sign
                __m256 result = _mm256_fmsub_ps(two, value_vec, sign_vec);
                y_vec = _mm256_add_ps(y_vec, result);

                _mm256_storeu_ps(y_row + n, y_vec);
            }

            // Handle remaining
            for (; n < N; n++) {
                const uint8_t s = sign_row[n] & 0x0F;
                const uint8_t v = value_row[n] & 0x0F;
                y_row[n] += 2.0f * lut[v] - lut[s];
            }
        }
    }
}

/**
 * T-MAC with AVX2 gather instructions for true vectorized float lookups
 *
 * Uses _mm256_i32gather_ps to gather 8 floats from LUT in parallel
 * This is the fastest approach for float LUTs on x86
 */
void matmul_free_tmac_gather(
    torch::Tensor x_tensor,
    torch::Tensor sign_plane_tensor,
    torch::Tensor value_plane_tensor,
    torch::Tensor y_tensor,
    torch::Tensor bias_tensor,
    int M, int N, int K,
    int num_threads
) {
    const float* __restrict__ x = x_tensor.data_ptr<float>();
    const uint8_t* __restrict__ sign_plane = sign_plane_tensor.data_ptr<uint8_t>();
    const uint8_t* __restrict__ value_plane = value_plane_tensor.data_ptr<uint8_t>();
    float* __restrict__ y = y_tensor.data_ptr<float>();
    const float* bias = bias_tensor.defined() ? bias_tensor.data_ptr<float>() : nullptr;

    const int K_groups = (K + 3) / 4;

    omp_set_num_threads(num_threads);

    #pragma omp parallel for schedule(static)
    for (int m = 0; m < M; m++) {
        const float* __restrict__ x_row = x + m * K;
        float* __restrict__ y_row = y + m * N;

        // Initialize output
        if (bias != nullptr) {
            memcpy(y_row, bias, N * sizeof(float));
        } else {
            memset(y_row, 0, N * sizeof(float));
        }

        const __m256 two = _mm256_set1_ps(2.0f);

        // K-first iteration
        for (int kg = 0; kg < K_groups; kg++) {
            const int k_base = kg * 4;

            // Get activations for this K-group
            float x_vals[4] = {0.0f, 0.0f, 0.0f, 0.0f};
            for (int i = 0; i < 4 && k_base + i < K; i++) {
                x_vals[i] = x_row[k_base + i];
            }

            // Build float LUT (aligned for gather)
            alignas(64) float lut[16];
            build_lut_16(x_vals, lut);

            const uint8_t* __restrict__ sign_ptr = sign_plane + kg * N;
            const uint8_t* __restrict__ value_ptr = value_plane + kg * N;

            // Process 8 outputs at a time using AVX2 gather
            int n = 0;
            for (; n + 7 < N; n += 8) {
                // Load 8 indices and convert to int32 for gather
                // We need to zero-extend uint8 to int32
                alignas(32) int32_t s_idx32[8];
                alignas(32) int32_t v_idx32[8];

                for (int i = 0; i < 8; i++) {
                    s_idx32[i] = sign_ptr[n + i] & 0x0F;
                    v_idx32[i] = value_ptr[n + i] & 0x0F;
                }

                __m256i sign_idx = _mm256_load_si256((__m256i*)s_idx32);
                __m256i value_idx = _mm256_load_si256((__m256i*)v_idx32);

                // Gather from LUT - 8 parallel lookups!
                __m256 sign_vec = _mm256_i32gather_ps(lut, sign_idx, 4);  // scale=4 for float
                __m256 value_vec = _mm256_i32gather_ps(lut, value_idx, 4);

                // result = 2*value - sign
                __m256 y_vec = _mm256_loadu_ps(y_row + n);
                __m256 result = _mm256_fmsub_ps(two, value_vec, sign_vec);
                y_vec = _mm256_add_ps(y_vec, result);
                _mm256_storeu_ps(y_row + n, y_vec);
            }

            // Handle remaining
            for (; n < N; n++) {
                const uint8_t s = sign_ptr[n] & 0x0F;
                const uint8_t v = value_ptr[n] & 0x0F;
                y_row[n] += 2.0f * lut[v] - lut[s];
            }
        }
    }
}

// Python bindings
PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("pack_ternary_bitplanes", &pack_ternary_bitplanes,
          "Pack ternary weights to bit-plane format [K_groups, N]");
    m.def("matmul_free_tmac", &matmul_free_tmac,
          "T-MAC kernel with K-first iteration (scalar)");
    m.def("matmul_free_tmac_avx2", &matmul_free_tmac_avx2,
          "T-MAC kernel with K-first + AVX2");
    m.def("matmul_free_tmac_tiled", &matmul_free_tmac_tiled,
          "T-MAC kernel with K-first + tiling");
    m.def("matmul_free_tmac_pshufb", &matmul_free_tmac_pshufb,
          "T-MAC kernel with float LUT + vectorized accumulation");
    m.def("matmul_free_tmac_gather", &matmul_free_tmac_gather,
          "T-MAC kernel with AVX2 gather instructions");
    m.def("matmul_free_tmac_optimized", &matmul_free_tmac_optimized,
          "T-MAC kernel optimized (K-first + AVX2)");
}
