/*
 * T-MAC VNNI Kernel - Using AVX-512 VNNI for maximum throughput
 *
 * VNNI (Vector Neural Network Instructions) provides:
 * - _mm512_dpbusd_epi32: 64 uint8*int8 multiplies + 16 int32 accumulates per instruction
 * - _mm512_dpbusds_epi32: same with saturation
 *
 * This is how Microsoft achieves 400+ GFLOPS on Intel CPUs.
 *
 * Key insight: Instead of LUT lookups, we can directly compute dot products
 * using VNNI when weights are expanded to int8 format.
 *
 * Strategy 1: Direct VNNI computation (expand weights on the fly)
 * Strategy 2: Use VNNI for partial sum accumulation with LUT
 */

#include <immintrin.h>
#include <cstring>
#include <cstdint>
#include <cmath>
#include <algorithm>
#include <omp.h>
#include <torch/extension.h>

// Cache tile sizes
constexpr int TILE_M = 4;    // Process 4 rows at a time for register reuse
constexpr int TILE_N = 64;   // Process 64 columns at a time
constexpr int TILE_K = 256;  // Process 256 K elements at a time

/**
 * Quantize activations to int8 with per-row scaling
 */
void quantize_activations_int8_vnni(
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
 * Pack ternary weights to int8 format for VNNI
 *
 * Simple format: each weight is stored as int8 (-1, 0, +1)
 * Layout: [N, K] int8
 *
 * This is less compact than bit-planes but enables direct VNNI computation.
 */
void pack_ternary_int8_vnni(
    torch::Tensor w_tensor,      // [N, K] float32 ternary
    torch::Tensor w_int8_tensor, // [N, K] int8 output
    int N, int K
) {
    const float* w = w_tensor.data_ptr<float>();
    int8_t* w_int8 = w_int8_tensor.data_ptr<int8_t>();

    #pragma omp parallel for collapse(2)
    for (int n = 0; n < N; n++) {
        for (int k = 0; k < K; k++) {
            float val = w[n * K + k];
            if (val > 0.5f) {
                w_int8[n * K + k] = 1;
            } else if (val < -0.5f) {
                w_int8[n * K + k] = -1;
            } else {
                w_int8[n * K + k] = 0;
            }
        }
    }
}

/**
 * Pack ternary weights to int8 with K-major layout for better vectorization
 * Layout: [K_padded, N] int8 where K_padded is K rounded up to 64
 */
void pack_ternary_int8_vnni_transposed(
    torch::Tensor w_tensor,      // [N, K] float32 ternary
    torch::Tensor w_int8_tensor, // [K_padded, N] int8 output
    int N, int K
) {
    const float* w = w_tensor.data_ptr<float>();
    int8_t* w_int8 = w_int8_tensor.data_ptr<int8_t>();

    const int K_padded = ((K + 63) / 64) * 64;

    // Zero the output first (for padding)
    memset(w_int8, 0, K_padded * N * sizeof(int8_t));

    #pragma omp parallel for collapse(2)
    for (int k = 0; k < K; k++) {
        for (int n = 0; n < N; n++) {
            float val = w[n * K + k];
            if (val > 0.5f) {
                w_int8[k * N + n] = 1;
            } else if (val < -0.5f) {
                w_int8[k * N + n] = -1;
            } else {
                w_int8[k * N + n] = 0;
            }
        }
    }
}

/**
 * VNNI-based ternary matmul
 *
 * Uses _mm512_dpbusd_epi32 for dot products:
 *   dst = src1 + dot_product(a[uint8], b[int8])
 *
 * For ternary weights, we use int8 weights directly.
 * For activations, we use int8 (signed).
 *
 * Note: dpbusd expects first operand as uint8, second as int8
 * We'll use dpbusds variant or adjust accordingly.
 */
void matmul_free_vnni(
    torch::Tensor x_int8_tensor,  // [M, K] int8
    torch::Tensor scale_tensor,   // [M] float32
    torch::Tensor w_int8_tensor,  // [N, K] int8 ternary weights
    torch::Tensor y_tensor,       // [M, N] float32 output
    torch::Tensor bias_tensor,
    int M, int N, int K,
    int num_threads
) {
    const int8_t* __restrict__ x_int8 = x_int8_tensor.data_ptr<int8_t>();
    const float* __restrict__ scales = scale_tensor.data_ptr<float>();
    const int8_t* __restrict__ w_int8 = w_int8_tensor.data_ptr<int8_t>();
    float* __restrict__ y = y_tensor.data_ptr<float>();
    const float* bias = bias_tensor.defined() ? bias_tensor.data_ptr<float>() : nullptr;

    omp_set_num_threads(num_threads);

    // For VNNI dpbusd, we need uint8 * int8
    // Our activations are int8 (signed), weights are int8 ternary
    // Strategy: add 128 to activations to make them uint8, then adjust result
    //
    // Original: sum(x[i] * w[i]) where x is int8, w is int8
    // With offset: sum((x[i] + 128) * w[i]) - 128 * sum(w[i])
    //            = sum(x[i] * w[i]) + 128 * sum(w[i]) - 128 * sum(w[i])
    //            = sum(x[i] * w[i])
    //
    // So we precompute sum(w[i]) for each output and use it to correct.

    // Actually simpler: use manual dot product with VNNI
    // dpbusd does: dst += sum of (a_uint8[i] * b_int8[i]) for i in 0..3 (per 32-bit element)

    #pragma omp parallel for schedule(static)
    for (int m = 0; m < M; m++) {
        const int8_t* __restrict__ x_row = x_int8 + m * K;
        float scale = scales[m];
        float* y_row = y + m * N;

        for (int n = 0; n < N; n++) {
            const int8_t* __restrict__ w_col = w_int8 + n * K;

            // Compute dot product using VNNI
            // Process 64 elements at a time (16 int32 lanes, 4 int8 per lane)
            __m512i acc = _mm512_setzero_si512();

            int k = 0;
            for (; k + 63 < K; k += 64) {
                // Load 64 activations and weights
                __m512i x_vec = _mm512_loadu_si512((__m512i*)(x_row + k));
                __m512i w_vec = _mm512_loadu_si512((__m512i*)(w_int8 + n * K + k));

                // Convert int8 activations to uint8 by adding 128
                __m512i offset = _mm512_set1_epi8((char)128);
                __m512i x_uint8 = _mm512_add_epi8(x_vec, offset);

                // dpbusd: acc += sum_i(x_uint8[i] * w_int8[i]) for groups of 4
                acc = _mm512_dpbusd_epi32(acc, x_uint8, w_vec);

                // We need to subtract the offset contribution: 128 * sum(w)
                // For each group of 4 weights, we need to subtract 128 * sum(w[0:3])
                // This gets accumulated into acc, so we'll correct at the end
            }

            // Handle remaining elements with scalar
            int32_t sum = 0;

            // Sum the accumulated int32 values
            alignas(64) int32_t acc_arr[16];
            _mm512_storeu_si512((__m512i*)acc_arr, acc);
            for (int i = 0; i < 16; i++) {
                sum += acc_arr[i];
            }

            // Correct for the offset: we added 128 * w for each element
            // Need to compute sum of weights and subtract 128 * sum(w)
            int32_t w_sum = 0;
            for (int kk = 0; kk < (k / 64) * 64; kk++) {
                w_sum += w_col[kk];
            }
            sum -= 128 * w_sum;

            // Handle remaining elements
            for (; k < K; k++) {
                sum += static_cast<int32_t>(x_row[k]) * static_cast<int32_t>(w_col[k]);
            }

            y_row[n] = static_cast<float>(sum) * scale + (bias ? bias[n] : 0.0f);
        }
    }
}

/**
 * Optimized VNNI kernel with tiling and precomputed weight sums
 */
void matmul_free_vnni_tiled(
    torch::Tensor x_int8_tensor,    // [M, K] int8
    torch::Tensor scale_tensor,     // [M] float32
    torch::Tensor w_int8_tensor,    // [K_padded, N] int8 (transposed)
    torch::Tensor w_sum_tensor,     // [N] int32 - precomputed sum of each column
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

    const int K_padded = ((K + 63) / 64) * 64;

    omp_set_num_threads(num_threads);

    #pragma omp parallel for schedule(static)
    for (int m = 0; m < M; m++) {
        const int8_t* __restrict__ x_row = x_int8 + m * K;
        float scale = scales[m];
        float* y_row = y + m * N;

        // Convert activation row to uint8 (add 128)
        alignas(64) uint8_t x_uint8[K_padded];
        for (int k = 0; k < K; k++) {
            x_uint8[k] = static_cast<uint8_t>(static_cast<int16_t>(x_row[k]) + 128);
        }
        for (int k = K; k < K_padded; k++) {
            x_uint8[k] = 128;  // Neutral element (0 + 128)
        }

        // Process N outputs
        int n = 0;
        for (; n + 15 < N; n += 16) {
            // Process 16 outputs at once using register tiling
            __m512i acc[16];
            for (int i = 0; i < 16; i++) {
                acc[i] = _mm512_setzero_si512();
            }

            // Process K in chunks of 64
            for (int k = 0; k < K_padded; k += 64) {
                // Load 64 activation values (shared across all 16 outputs)
                __m512i x_vec = _mm512_loadu_si512((__m512i*)(x_uint8 + k));

                // For each of 16 outputs, load weights and compute
                for (int i = 0; i < 16; i++) {
                    __m512i w_vec = _mm512_loadu_si512((__m512i*)(w_int8 + k * N + (n + i)));
                    // Wait, the transposed layout is [K_padded, N], so w[k, n] = w_int8[k * N + n]
                    // But we're loading 64 consecutive bytes, which would be 64 different n values at same k
                    // That's wrong for this loop structure.

                    // Actually with transposed [K_padded, N] layout, for a single output n:
                    // weights are at positions k*N + n for k = 0..K-1
                    // These are not contiguous!

                    // Need to restructure. For VNNI to work efficiently, we need weights
                    // laid out so consecutive addresses correspond to consecutive k values
                    // for the same n.

                    // Let's use [N, K_padded] layout (non-transposed)
                }
            }
        }

        // Fallback to simple version
        for (; n < N; n++) {
            const int8_t* w_col = w_int8 + n * K_padded;  // Assuming [N, K_padded] layout

            __m512i acc = _mm512_setzero_si512();

            for (int k = 0; k < K_padded; k += 64) {
                __m512i x_vec = _mm512_loadu_si512((__m512i*)(x_uint8 + k));
                __m512i w_vec = _mm512_loadu_si512((__m512i*)(w_col + k));

                acc = _mm512_dpbusd_epi32(acc, x_vec, w_vec);
            }

            // Horizontal sum
            int32_t sum = _mm512_reduce_add_epi32(acc);

            // Correct for offset: subtract 128 * sum(w)
            sum -= 128 * w_sum[n];

            y_row[n] = static_cast<float>(sum) * scale + (bias ? bias[n] : 0.0f);
        }
    }
}

/**
 * Simple but correct VNNI kernel
 *
 * Layout: w_int8 is [N, K_padded] where K_padded = ceil(K/64)*64
 * Each row contains weights for one output, padded to 64-byte alignment
 */
void matmul_free_vnni_simple(
    torch::Tensor x_int8_tensor,    // [M, K] int8
    torch::Tensor scale_tensor,     // [M] float32
    torch::Tensor w_int8_tensor,    // [N, K_padded] int8
    torch::Tensor w_sum_tensor,     // [N] int32 - precomputed sum of each row
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

    const int K_padded = ((K + 63) / 64) * 64;

    omp_set_num_threads(num_threads);

    #pragma omp parallel for schedule(static)
    for (int m = 0; m < M; m++) {
        const int8_t* __restrict__ x_row = x_int8 + m * K;
        float scale = scales[m];
        float* y_row = y + m * N;

        // Convert activation row to uint8 (add 128) and pad
        alignas(64) uint8_t x_uint8[K_padded];
        for (int k = 0; k < K; k++) {
            x_uint8[k] = static_cast<uint8_t>(static_cast<int16_t>(x_row[k]) + 128);
        }
        for (int k = K; k < K_padded; k++) {
            x_uint8[k] = 128;  // 0 + 128 = neutral for dot product
        }

        // Process each output
        for (int n = 0; n < N; n++) {
            const int8_t* __restrict__ w_row = w_int8 + n * K_padded;

            __m512i acc = _mm512_setzero_si512();

            // Main VNNI loop - 64 elements per iteration
            for (int k = 0; k < K_padded; k += 64) {
                __m512i x_vec = _mm512_load_si512((__m512i*)(x_uint8 + k));
                __m512i w_vec = _mm512_loadu_si512((__m512i*)(w_row + k));

                // dpbusd: acc[i] += sum_{j=0}^{3} (x[4i+j] * w[4i+j])
                // Computes 16 partial sums, each from 4 uint8*int8 products
                acc = _mm512_dpbusd_epi32(acc, x_vec, w_vec);
            }

            // Horizontal sum of 16 int32 values
            int32_t sum = _mm512_reduce_add_epi32(acc);

            // Correct for uint8 offset: we computed sum((x+128)*w) = sum(x*w) + 128*sum(w)
            sum -= 128 * w_sum[n];

            y_row[n] = static_cast<float>(sum) * scale + (bias ? bias[n] : 0.0f);
        }
    }
}

/**
 * Pack weights and compute column sums for VNNI kernel
 */
void pack_weights_vnni(
    torch::Tensor w_tensor,        // [N, K] float32 ternary
    torch::Tensor w_int8_tensor,   // [N, K_padded] int8 output
    torch::Tensor w_sum_tensor,    // [N] int32 output
    int N, int K
) {
    const float* w = w_tensor.data_ptr<float>();
    int8_t* w_int8 = w_int8_tensor.data_ptr<int8_t>();
    int32_t* w_sum = w_sum_tensor.data_ptr<int32_t>();

    const int K_padded = ((K + 63) / 64) * 64;

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

// Python bindings
PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("quantize_activations_int8_vnni", &quantize_activations_int8_vnni,
          "Quantize activations to int8");
    m.def("pack_weights_vnni", &pack_weights_vnni,
          "Pack ternary weights for VNNI kernel");
    m.def("matmul_free_vnni_simple", &matmul_free_vnni_simple,
          "VNNI-based ternary matmul (simple)");
}
