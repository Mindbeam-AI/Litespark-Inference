/*
 * TL2 Element-wise LUT Kernel for x86_64 AVX2
 *
 * Based on Microsoft BitNet.cpp TL2 approach:
 * https://arxiv.org/abs/2502.11880
 *
 * KEY INSIGHT: Element-wise LUT vs Bit-wise LUT
 * =============================================
 * T-MAC (bit-wise): Groups 4 weights, 2^4 = 16 LUT entries
 *   - Works well for 2-bit/4-bit weights
 *   - "Spatial inefficiencies" for ternary weights
 *
 * TL2 (element-wise): Groups 3 ternary weights, 3^3 = 27 combinations
 *   - Uses "mirror consolidation" to reduce to ~14 entries
 *   - Better suited for ternary {-1, 0, +1} weights
 *   - 1.67 bits per weight (vs 2.0 for T-MAC)
 *
 * Mirror Consolidation:
 * ---------------------
 * For ternary weights, sum(-w0, -w1, -w2) = -sum(w0, w1, w2)
 * So we only store positive sums and apply sign separately.
 *
 * Weight encoding:
 *   3 weights -> index in [0, 26] -> consolidated to [0, 13] + sign bit
 *
 * Algorithm:
 * 1. Group 3 weights together
 * 2. For each group, build 14-entry LUT from 3 activations
 * 3. Lookup using consolidated index, apply sign
 * 4. result = sign ? -lut[idx] : lut[idx]
 */

#include <immintrin.h>
#include <cstring>
#include <cstdint>
#include <algorithm>
#include <omp.h>
#include <torch/extension.h>

// Number of weights per group for TL2
constexpr int TL2_GROUP_SIZE = 3;

// LUT size after mirror consolidation: ceil(27/2) = 14
// But we use 16 for alignment (fits in 128-bit register)
constexpr int TL2_LUT_SIZE = 14;

/**
 * Ternary weight to index mapping for 3 weights
 *
 * Each weight w ∈ {-1, 0, +1} maps to {0, 1, 2}
 * Index = w0_idx + 3*w1_idx + 9*w2_idx ∈ [0, 26]
 *
 * Mirror consolidation:
 * - Indices 0-13 are "positive" (sum >= 0 or first non-zero is positive)
 * - Indices 14-26 map to 26-idx with sign flip
 */

// Map ternary weight to index: -1->0, 0->1, +1->2
inline int ternary_to_idx(float w) {
    if (w > 0.5f) return 2;      // +1
    if (w < -0.5f) return 0;     // -1
    return 1;                     // 0
}

// Combined index for 3 weights
inline int combined_index(int w0, int w1, int w2) {
    return w0 + 3 * w1 + 9 * w2;
}

// Mirror consolidation: returns (consolidated_index, needs_sign_flip)
// The mirror point is at index 13 (middle of 0-26)
inline std::pair<uint8_t, bool> consolidate_index(int idx) {
    if (idx <= 13) {
        return {static_cast<uint8_t>(idx), false};
    } else {
        return {static_cast<uint8_t>(26 - idx), true};
    }
}

/**
 * Pack ternary weights into TL2 format
 *
 * For each group of 3 weights:
 *   - Compute combined index [0-26]
 *   - Apply mirror consolidation -> index [0-13] + sign bit
 *   - Pack into (index: 4 bits, sign: 1 bit) = 5 bits
 *   - Store as uint8 for simplicity (index in lower 4 bits, sign in bit 4)
 *
 * Layout: [K_groups, N] where K_groups = ceil(K/3)
 */
void pack_ternary_tl2(
    torch::Tensor w_tensor,
    torch::Tensor packed_tensor,  // [K_groups, N] uint8
    int N, int K
) {
    const float* w = w_tensor.data_ptr<float>();
    uint8_t* packed = packed_tensor.data_ptr<uint8_t>();

    const int K_groups = (K + TL2_GROUP_SIZE - 1) / TL2_GROUP_SIZE;

    #pragma omp parallel for collapse(2)
    for (int kg = 0; kg < K_groups; kg++) {
        for (int n = 0; n < N; n++) {
            const int k_base = kg * TL2_GROUP_SIZE;

            // Get 3 weights (pad with 0 if at boundary)
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

            int idx = combined_index(w0_idx, w1_idx, w2_idx);
            auto [consolidated_idx, sign] = consolidate_index(idx);

            // Pack: lower 4 bits = index, bit 4 = sign
            packed[kg * N + n] = consolidated_idx | (sign ? 0x10 : 0x00);
        }
    }
}

/**
 * Build 14-entry LUT for 3 activations
 *
 * For indices 0-13 (after consolidation), compute all possible sums.
 * Index encoding (w0, w1, w2 each in {0=−1, 1=0, 2=+1}):
 *   idx = w0 + 3*w1 + 9*w2
 *
 * Consolidated indices 0-13 correspond to original indices 0-13.
 */
inline void build_tl2_lut(const float* x, float* lut) {
    const float x0 = x[0], x1 = x[1], x2 = x[2];

    // Map from weight index {0,1,2} to multiplier {-1,0,+1}
    // w=0 -> -1, w=1 -> 0, w=2 -> +1
    auto weight_mult = [](int w) -> float {
        return static_cast<float>(w - 1);
    };

    // Compute all 14 sums for indices 0-13
    for (int idx = 0; idx < 14; idx++) {
        int w0 = idx % 3;
        int w1 = (idx / 3) % 3;
        int w2 = idx / 9;

        lut[idx] = weight_mult(w0) * x0 + weight_mult(w1) * x1 + weight_mult(w2) * x2;
    }

    // Pad remaining slots
    lut[14] = 0.0f;
    lut[15] = 0.0f;
}

/**
 * TL2 kernel with K-first iteration
 *
 * Same K-first approach as T-MAC, but using element-wise LUT:
 * - Group 3 weights instead of 4
 * - 14-entry LUT instead of 16
 * - Apply sign bit for mirror consolidation
 */
void matmul_free_tl2(
    torch::Tensor x_tensor,
    torch::Tensor packed_tensor,  // [K_groups, N] uint8
    torch::Tensor y_tensor,
    torch::Tensor bias_tensor,
    int M, int N, int K,
    int num_threads
) {
    const float* __restrict__ x = x_tensor.data_ptr<float>();
    const uint8_t* __restrict__ packed = packed_tensor.data_ptr<uint8_t>();
    float* __restrict__ y = y_tensor.data_ptr<float>();
    const float* bias = bias_tensor.defined() ? bias_tensor.data_ptr<float>() : nullptr;

    const int K_groups = (K + TL2_GROUP_SIZE - 1) / TL2_GROUP_SIZE;

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
            const int k_base = kg * TL2_GROUP_SIZE;

            // Get 3 activations for this K-group
            float x_vals[3] = {0.0f, 0.0f, 0.0f};
            for (int i = 0; i < 3 && k_base + i < K; i++) {
                x_vals[i] = x_row[k_base + i];
            }

            // Build LUT for this K-group
            alignas(64) float lut[16];
            build_tl2_lut(x_vals, lut);

            const uint8_t* __restrict__ packed_row = packed + kg * N;

            // Process all N outputs using the same LUT
            for (int n = 0; n < N; n++) {
                uint8_t p = packed_row[n];
                uint8_t idx = p & 0x0F;        // Lower 4 bits = index
                bool sign = (p & 0x10) != 0;   // Bit 4 = sign

                float val = lut[idx];
                y_row[n] += sign ? -val : val;
            }
        }
    }
}

/**
 * TL2 kernel with AVX2 vectorization
 */
void matmul_free_tl2_avx2(
    torch::Tensor x_tensor,
    torch::Tensor packed_tensor,
    torch::Tensor y_tensor,
    torch::Tensor bias_tensor,
    int M, int N, int K,
    int num_threads
) {
    const float* __restrict__ x = x_tensor.data_ptr<float>();
    const uint8_t* __restrict__ packed = packed_tensor.data_ptr<uint8_t>();
    float* __restrict__ y = y_tensor.data_ptr<float>();
    const float* bias = bias_tensor.defined() ? bias_tensor.data_ptr<float>() : nullptr;

    const int K_groups = (K + TL2_GROUP_SIZE - 1) / TL2_GROUP_SIZE;

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
            const int k_base = kg * TL2_GROUP_SIZE;

            // Get 3 activations
            float x_vals[3] = {0.0f, 0.0f, 0.0f};
            for (int i = 0; i < 3 && k_base + i < K; i++) {
                x_vals[i] = x_row[k_base + i];
            }

            // Build LUT
            alignas(64) float lut[16];
            build_tl2_lut(x_vals, lut);

            const uint8_t* __restrict__ packed_row = packed + kg * N;

            // Process 8 outputs at a time with AVX2
            int n = 0;
            for (; n + 7 < N; n += 8) {
                __m256 y_vec = _mm256_loadu_ps(y_row + n);

                // Load 8 packed values
                alignas(32) float results[8];
                for (int i = 0; i < 8; i++) {
                    uint8_t p = packed_row[n + i];
                    uint8_t idx = p & 0x0F;
                    bool sign = (p & 0x10) != 0;
                    results[i] = sign ? -lut[idx] : lut[idx];
                }

                __m256 result_vec = _mm256_load_ps(results);
                y_vec = _mm256_add_ps(y_vec, result_vec);
                _mm256_storeu_ps(y_row + n, y_vec);
            }

            // Handle remaining
            for (; n < N; n++) {
                uint8_t p = packed_row[n];
                uint8_t idx = p & 0x0F;
                bool sign = (p & 0x10) != 0;
                y_row[n] += sign ? -lut[idx] : lut[idx];
            }
        }
    }
}

/**
 * TL2 kernel with AVX2 gather and sign handling
 *
 * Uses gather for LUT lookups and XOR for sign application
 */
void matmul_free_tl2_gather(
    torch::Tensor x_tensor,
    torch::Tensor packed_tensor,
    torch::Tensor y_tensor,
    torch::Tensor bias_tensor,
    int M, int N, int K,
    int num_threads
) {
    const float* __restrict__ x = x_tensor.data_ptr<float>();
    const uint8_t* __restrict__ packed = packed_tensor.data_ptr<uint8_t>();
    float* __restrict__ y = y_tensor.data_ptr<float>();
    const float* bias = bias_tensor.defined() ? bias_tensor.data_ptr<float>() : nullptr;

    const int K_groups = (K + TL2_GROUP_SIZE - 1) / TL2_GROUP_SIZE;

    omp_set_num_threads(num_threads);

    // Sign bit mask for float (bit 31)
    const __m256 sign_mask = _mm256_set1_ps(-0.0f);

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
            const int k_base = kg * TL2_GROUP_SIZE;

            // Get 3 activations
            float x_vals[3] = {0.0f, 0.0f, 0.0f};
            for (int i = 0; i < 3 && k_base + i < K; i++) {
                x_vals[i] = x_row[k_base + i];
            }

            // Build LUT
            alignas(64) float lut[16];
            build_tl2_lut(x_vals, lut);

            const uint8_t* __restrict__ packed_row = packed + kg * N;

            // Process 8 outputs at a time
            int n = 0;
            for (; n + 7 < N; n += 8) {
                // Load 8 packed values and extract indices/signs
                alignas(32) int32_t indices[8];
                alignas(32) float signs[8];

                for (int i = 0; i < 8; i++) {
                    uint8_t p = packed_row[n + i];
                    indices[i] = p & 0x0F;
                    // Create sign mask: 0.0f for positive, -0.0f for negative
                    signs[i] = (p & 0x10) ? -0.0f : 0.0f;
                }

                __m256i idx_vec = _mm256_load_si256((__m256i*)indices);
                __m256 sign_vec = _mm256_load_ps(signs);

                // Gather from LUT
                __m256 lut_vec = _mm256_i32gather_ps(lut, idx_vec, 4);

                // Apply sign using XOR (flip sign bit if needed)
                __m256 result = _mm256_xor_ps(lut_vec, sign_vec);

                // Accumulate
                __m256 y_vec = _mm256_loadu_ps(y_row + n);
                y_vec = _mm256_add_ps(y_vec, result);
                _mm256_storeu_ps(y_row + n, y_vec);
            }

            // Handle remaining
            for (; n < N; n++) {
                uint8_t p = packed_row[n];
                uint8_t idx = p & 0x0F;
                bool sign = (p & 0x10) != 0;
                y_row[n] += sign ? -lut[idx] : lut[idx];
            }
        }
    }
}

/**
 * TL2 optimized: unrolled inner loop for better ILP
 */
void matmul_free_tl2_optimized(
    torch::Tensor x_tensor,
    torch::Tensor packed_tensor,
    torch::Tensor y_tensor,
    torch::Tensor bias_tensor,
    int M, int N, int K,
    int num_threads
) {
    const float* __restrict__ x = x_tensor.data_ptr<float>();
    const uint8_t* __restrict__ packed = packed_tensor.data_ptr<uint8_t>();
    float* __restrict__ y = y_tensor.data_ptr<float>();
    const float* bias = bias_tensor.defined() ? bias_tensor.data_ptr<float>() : nullptr;

    const int K_groups = (K + TL2_GROUP_SIZE - 1) / TL2_GROUP_SIZE;

    omp_set_num_threads(num_threads);

    #pragma omp parallel for schedule(static)
    for (int m = 0; m < M; m++) {
        const float* __restrict__ x_row = x + m * K;
        float* __restrict__ y_row = y + m * N;

        // Initialize output
        if (bias != nullptr) {
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

        // K-first iteration
        for (int kg = 0; kg < K_groups; kg++) {
            const int k_base = kg * TL2_GROUP_SIZE;

            // Get 3 activations
            const float x0 = (k_base < K) ? x_row[k_base] : 0.0f;
            const float x1 = (k_base + 1 < K) ? x_row[k_base + 1] : 0.0f;
            const float x2 = (k_base + 2 < K) ? x_row[k_base + 2] : 0.0f;

            // Build LUT inline
            alignas(64) float lut[16];
            // idx = w0 + 3*w1 + 9*w2, where w ∈ {0=-1, 1=0, 2=+1}
            lut[0]  = -x0 - x1 - x2;  // (-1,-1,-1)
            lut[1]  =    - x1 - x2;   // ( 0,-1,-1)
            lut[2]  =  x0 - x1 - x2;  // (+1,-1,-1)
            lut[3]  = -x0      - x2;  // (-1, 0,-1)
            lut[4]  =        - x2;    // ( 0, 0,-1)
            lut[5]  =  x0    - x2;    // (+1, 0,-1)
            lut[6]  = -x0 + x1 - x2;  // (-1,+1,-1)
            lut[7]  =      x1 - x2;   // ( 0,+1,-1)
            lut[8]  =  x0 + x1 - x2;  // (+1,+1,-1)
            lut[9]  = -x0 - x1;       // (-1,-1, 0)
            lut[10] =    - x1;        // ( 0,-1, 0)
            lut[11] =  x0 - x1;       // (+1,-1, 0)
            lut[12] = -x0;            // (-1, 0, 0)
            lut[13] = 0.0f;           // ( 0, 0, 0)
            lut[14] = 0.0f;           // padding
            lut[15] = 0.0f;           // padding

            const uint8_t* __restrict__ packed_row = packed + kg * N;

            // Process 8 outputs at a time
            int n = 0;
            for (; n + 7 < N; n += 8) {
                __m256 y_vec = _mm256_loadu_ps(y_row + n);

                // Unrolled lookups with sign handling
                const uint8_t p0 = packed_row[n+0], p1 = packed_row[n+1];
                const uint8_t p2 = packed_row[n+2], p3 = packed_row[n+3];
                const uint8_t p4 = packed_row[n+4], p5 = packed_row[n+5];
                const uint8_t p6 = packed_row[n+6], p7 = packed_row[n+7];

                __m256 result = _mm256_set_ps(
                    (p7 & 0x10) ? -lut[p7 & 0x0F] : lut[p7 & 0x0F],
                    (p6 & 0x10) ? -lut[p6 & 0x0F] : lut[p6 & 0x0F],
                    (p5 & 0x10) ? -lut[p5 & 0x0F] : lut[p5 & 0x0F],
                    (p4 & 0x10) ? -lut[p4 & 0x0F] : lut[p4 & 0x0F],
                    (p3 & 0x10) ? -lut[p3 & 0x0F] : lut[p3 & 0x0F],
                    (p2 & 0x10) ? -lut[p2 & 0x0F] : lut[p2 & 0x0F],
                    (p1 & 0x10) ? -lut[p1 & 0x0F] : lut[p1 & 0x0F],
                    (p0 & 0x10) ? -lut[p0 & 0x0F] : lut[p0 & 0x0F]
                );

                y_vec = _mm256_add_ps(y_vec, result);
                _mm256_storeu_ps(y_row + n, y_vec);
            }

            // Handle remaining
            for (; n < N; n++) {
                uint8_t p = packed_row[n];
                uint8_t idx = p & 0x0F;
                bool sign = (p & 0x10) != 0;
                y_row[n] += sign ? -lut[idx] : lut[idx];
            }
        }
    }
}

// Python bindings
PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("pack_ternary_tl2", &pack_ternary_tl2,
          "Pack ternary weights to TL2 format (3 weights per group)");
    m.def("matmul_free_tl2", &matmul_free_tl2,
          "TL2 kernel (scalar)");
    m.def("matmul_free_tl2_avx2", &matmul_free_tl2_avx2,
          "TL2 kernel with AVX2");
    m.def("matmul_free_tl2_gather", &matmul_free_tl2_gather,
          "TL2 kernel with AVX2 gather");
    m.def("matmul_free_tl2_optimized", &matmul_free_tl2_optimized,
          "TL2 kernel optimized");
}
