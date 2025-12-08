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
 * Fast AVX-512 exponential approximation
 * Uses polynomial approximation for exp(x)
 * Accurate for x in [-88, 88], good enough for softmax
 */
inline __m512 avx512_exp_fast(__m512 x) {
    // Clamp to prevent overflow
    x = _mm512_max_ps(x, _mm512_set1_ps(-88.0f));
    x = _mm512_min_ps(x, _mm512_set1_ps(88.0f));

    // exp(x) = 2^(x * log2(e))
    const __m512 log2e = _mm512_set1_ps(1.44269504088896341f);

    __m512 z = _mm512_mul_ps(x, log2e);

    // Split into integer and fractional parts
    __m512i zi = _mm512_cvtps_epi32(z);
    __m512 zf = _mm512_sub_ps(z, _mm512_cvtepi32_ps(zi));

    // 2^frac using polynomial: 1 + f*ln2 + f^2*ln2^2/2 + ...
    const __m512 c0 = _mm512_set1_ps(1.0f);
    const __m512 c1 = _mm512_set1_ps(0.693147180559945309f);
    const __m512 c2 = _mm512_set1_ps(0.240226506959100712f);
    const __m512 c3 = _mm512_set1_ps(0.0555041086648215799f);

    __m512 poly = _mm512_fmadd_ps(c3, zf, c2);
    poly = _mm512_fmadd_ps(poly, zf, c1);
    poly = _mm512_fmadd_ps(poly, zf, c0);

    // 2^int part via bit manipulation
    __m512i exp_int = _mm512_add_epi32(zi, _mm512_set1_epi32(127));
    exp_int = _mm512_slli_epi32(exp_int, 23);
    __m512 pow2_int = _mm512_castsi512_ps(exp_int);

    return _mm512_mul_ps(pow2_int, poly);
}

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

/**
 * VNNI v2 - Fully optimized kernel
 *
 * Key optimizations:
 * 1. Register blocking (4 outputs at a time)
 * 2. Vectorized uint8 conversion
 * 3. Per-thread activation buffers
 *
 * This version parallelizes across M rows, good for small/medium matrices.
 */
void matmul_free_vnni_v2(
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

    const int K_padded = ((K + 63) / 64) * 64;
    constexpr int N_BLOCK = 4;  // Register blocking

    omp_set_num_threads(num_threads);

    // Allocate per-thread uint8 activation buffers
    uint8_t* x_uint8_buffers = (uint8_t*)aligned_alloc(64, num_threads * K_padded);

    #pragma omp parallel for schedule(static)
    for (int m = 0; m < M; m++) {
        const int tid = omp_get_thread_num();
        uint8_t* x_uint8 = x_uint8_buffers + tid * K_padded;

        const int8_t* __restrict__ x_row = x_int8 + m * K;
        float scale = scales[m];
        float* y_row = y + m * N;

        // Convert activation row to uint8 (vectorized)
        const __m512i offset_vec = _mm512_set1_epi8((char)128);
        int k = 0;
        for (; k + 63 < K; k += 64) {
            __m512i x_vec = _mm512_loadu_si512((__m512i*)(x_row + k));
            __m512i x_u8 = _mm512_add_epi8(x_vec, offset_vec);
            _mm512_store_si512((__m512i*)(x_uint8 + k), x_u8);
        }
        for (; k < K; k++) {
            x_uint8[k] = static_cast<uint8_t>(static_cast<int16_t>(x_row[k]) + 128);
        }
        for (; k < K_padded; k++) {
            x_uint8[k] = 128;
        }

        // Process outputs with register blocking (4 at a time)
        int n = 0;
        for (; n + N_BLOCK - 1 < N; n += N_BLOCK) {
            __m512i acc0 = _mm512_setzero_si512();
            __m512i acc1 = _mm512_setzero_si512();
            __m512i acc2 = _mm512_setzero_si512();
            __m512i acc3 = _mm512_setzero_si512();

            const int8_t* w0 = w_int8 + (n + 0) * K_padded;
            const int8_t* w1 = w_int8 + (n + 1) * K_padded;
            const int8_t* w2 = w_int8 + (n + 2) * K_padded;
            const int8_t* w3 = w_int8 + (n + 3) * K_padded;

            for (int kk = 0; kk < K_padded; kk += 64) {
                __m512i x_vec = _mm512_load_si512((__m512i*)(x_uint8 + kk));

                acc0 = _mm512_dpbusd_epi32(acc0, x_vec, _mm512_loadu_si512((__m512i*)(w0 + kk)));
                acc1 = _mm512_dpbusd_epi32(acc1, x_vec, _mm512_loadu_si512((__m512i*)(w1 + kk)));
                acc2 = _mm512_dpbusd_epi32(acc2, x_vec, _mm512_loadu_si512((__m512i*)(w2 + kk)));
                acc3 = _mm512_dpbusd_epi32(acc3, x_vec, _mm512_loadu_si512((__m512i*)(w3 + kk)));
            }

            int32_t sum0 = _mm512_reduce_add_epi32(acc0) - 128 * w_sum[n + 0];
            int32_t sum1 = _mm512_reduce_add_epi32(acc1) - 128 * w_sum[n + 1];
            int32_t sum2 = _mm512_reduce_add_epi32(acc2) - 128 * w_sum[n + 2];
            int32_t sum3 = _mm512_reduce_add_epi32(acc3) - 128 * w_sum[n + 3];

            y_row[n + 0] = static_cast<float>(sum0) * scale + (bias ? bias[n + 0] : 0.0f);
            y_row[n + 1] = static_cast<float>(sum1) * scale + (bias ? bias[n + 1] : 0.0f);
            y_row[n + 2] = static_cast<float>(sum2) * scale + (bias ? bias[n + 2] : 0.0f);
            y_row[n + 3] = static_cast<float>(sum3) * scale + (bias ? bias[n + 3] : 0.0f);
        }

        // Remainder
        for (; n < N; n++) {
            const int8_t* w_row = w_int8 + n * K_padded;
            __m512i acc = _mm512_setzero_si512();

            for (int kk = 0; kk < K_padded; kk += 64) {
                __m512i x_vec = _mm512_load_si512((__m512i*)(x_uint8 + kk));
                acc = _mm512_dpbusd_epi32(acc, x_vec, _mm512_loadu_si512((__m512i*)(w_row + kk)));
            }

            int32_t sum = _mm512_reduce_add_epi32(acc) - 128 * w_sum[n];
            y_row[n] = static_cast<float>(sum) * scale + (bias ? bias[n] : 0.0f);
        }
    }

    free(x_uint8_buffers);
}

/**
 * VNNI v3 - Cache-optimized tiled kernel for large matrices
 *
 * Based on Microsoft T-MAC's tiling strategy (https://arxiv.org/html/2407.00088v1):
 * - M tiling: Process M_TILE rows at a time
 * - N tiling: Process N_TILE outputs at a time (fit weights in L2 cache)
 * - Precompute uint8 activations for M_TILE rows before iterating N tiles
 *
 * This ensures weight tiles stay in L2 cache while being reused across M rows.
 *
 * Memory layout:
 * - For each M tile: allocate M_TILE * K_padded uint8 buffer
 * - Process all N tiles, reusing the activation buffer
 * - Move to next M tile
 */
void matmul_free_vnni_v3(
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

    const int K_padded = ((K + 63) / 64) * 64;

    // Tile sizes for cache efficiency
    // N_TILE * K_padded should fit in L2 cache (~256KB-1MB per core)
    // For K=2048, K_padded=2048: N_TILE=64 -> 128KB weight tile per N tile
    // M_TILE should be large enough to amortize activation conversion
    constexpr int N_TILE = 64;
    constexpr int M_TILE = 32;
    constexpr int N_BLOCK = 4;  // Register blocking

    omp_set_num_threads(num_threads);

    // Process M in tiles
    for (int m_tile = 0; m_tile < M; m_tile += M_TILE) {
        const int m_end = std::min(m_tile + M_TILE, M);
        const int m_tile_size = m_end - m_tile;

        // Allocate activation buffer for this M tile
        uint8_t* x_uint8_tile = (uint8_t*)aligned_alloc(64, m_tile_size * K_padded);

        // Step 1: Convert all activations in this M tile to uint8 (parallelized)
        #pragma omp parallel for schedule(static)
        for (int m = m_tile; m < m_end; m++) {
            const int m_local = m - m_tile;
            uint8_t* x_uint8 = x_uint8_tile + m_local * K_padded;
            const int8_t* x_row = x_int8 + m * K;

            const __m512i offset_vec = _mm512_set1_epi8((char)128);
            int k = 0;
            for (; k + 63 < K; k += 64) {
                __m512i x_vec = _mm512_loadu_si512((__m512i*)(x_row + k));
                __m512i x_u8 = _mm512_add_epi8(x_vec, offset_vec);
                _mm512_store_si512((__m512i*)(x_uint8 + k), x_u8);
            }
            for (; k < K; k++) {
                x_uint8[k] = static_cast<uint8_t>(static_cast<int16_t>(x_row[k]) + 128);
            }
            for (; k < K_padded; k++) {
                x_uint8[k] = 128;
            }
        }

        // Step 2: Process N in tiles (weights stay in L2 cache)
        for (int n_tile = 0; n_tile < N; n_tile += N_TILE) {
            const int n_end = std::min(n_tile + N_TILE, N);

            // Prefetch weight tile into L2
            for (int n = n_tile; n < n_end; n++) {
                for (int k = 0; k < K_padded; k += 64) {
                    _mm_prefetch((const char*)(w_int8 + n * K_padded + k), _MM_HINT_T1);
                }
            }

            // Process all M rows in this tile against the N tile
            #pragma omp parallel for schedule(static)
            for (int m = m_tile; m < m_end; m++) {
                const int m_local = m - m_tile;
                const uint8_t* x_uint8 = x_uint8_tile + m_local * K_padded;
                float scale = scales[m];
                float* y_row = y + m * N;

                // Process N tile with register blocking
                int n = n_tile;
                for (; n + N_BLOCK - 1 < n_end; n += N_BLOCK) {
                    __m512i acc0 = _mm512_setzero_si512();
                    __m512i acc1 = _mm512_setzero_si512();
                    __m512i acc2 = _mm512_setzero_si512();
                    __m512i acc3 = _mm512_setzero_si512();

                    const int8_t* w0 = w_int8 + (n + 0) * K_padded;
                    const int8_t* w1 = w_int8 + (n + 1) * K_padded;
                    const int8_t* w2 = w_int8 + (n + 2) * K_padded;
                    const int8_t* w3 = w_int8 + (n + 3) * K_padded;

                    for (int kk = 0; kk < K_padded; kk += 64) {
                        __m512i x_vec = _mm512_load_si512((__m512i*)(x_uint8 + kk));

                        acc0 = _mm512_dpbusd_epi32(acc0, x_vec, _mm512_loadu_si512((__m512i*)(w0 + kk)));
                        acc1 = _mm512_dpbusd_epi32(acc1, x_vec, _mm512_loadu_si512((__m512i*)(w1 + kk)));
                        acc2 = _mm512_dpbusd_epi32(acc2, x_vec, _mm512_loadu_si512((__m512i*)(w2 + kk)));
                        acc3 = _mm512_dpbusd_epi32(acc3, x_vec, _mm512_loadu_si512((__m512i*)(w3 + kk)));
                    }

                    int32_t sum0 = _mm512_reduce_add_epi32(acc0) - 128 * w_sum[n + 0];
                    int32_t sum1 = _mm512_reduce_add_epi32(acc1) - 128 * w_sum[n + 1];
                    int32_t sum2 = _mm512_reduce_add_epi32(acc2) - 128 * w_sum[n + 2];
                    int32_t sum3 = _mm512_reduce_add_epi32(acc3) - 128 * w_sum[n + 3];

                    y_row[n + 0] = static_cast<float>(sum0) * scale + (bias ? bias[n + 0] : 0.0f);
                    y_row[n + 1] = static_cast<float>(sum1) * scale + (bias ? bias[n + 1] : 0.0f);
                    y_row[n + 2] = static_cast<float>(sum2) * scale + (bias ? bias[n + 2] : 0.0f);
                    y_row[n + 3] = static_cast<float>(sum3) * scale + (bias ? bias[n + 3] : 0.0f);
                }

                // Remainder
                for (; n < n_end; n++) {
                    const int8_t* w_row = w_int8 + n * K_padded;
                    __m512i acc = _mm512_setzero_si512();

                    for (int kk = 0; kk < K_padded; kk += 64) {
                        __m512i x_vec = _mm512_load_si512((__m512i*)(x_uint8 + kk));
                        acc = _mm512_dpbusd_epi32(acc, x_vec, _mm512_loadu_si512((__m512i*)(w_row + kk)));
                    }

                    int32_t sum = _mm512_reduce_add_epi32(acc) - 128 * w_sum[n];
                    y_row[n] = static_cast<float>(sum) * scale + (bias ? bias[n] : 0.0f);
                }
            }
        }

        free(x_uint8_tile);
    }
}

/**
 * VNNI v3 with int32 output - keeps accumulators in int32 for chaining with activation functions
 *
 * Same as matmul_free_vnni_v3 but outputs int32 instead of float32.
 * This avoids the float conversion overhead when chaining with softmax/swiglu.
 */
void matmul_free_vnni_v3_int32_out(
    torch::Tensor x_int8_tensor,    // [M, K] int8
    torch::Tensor scale_tensor,     // [M] float32 (still needed for later)
    torch::Tensor w_int8_tensor,    // [N, K_padded] int8
    torch::Tensor w_sum_tensor,     // [N] int32
    torch::Tensor y_tensor,         // [M, N] int32 output (NOT float32)
    torch::Tensor bias_tensor,      // ignored for int32 output
    int M, int N, int K,
    int num_threads
) {
    const int8_t* __restrict__ x_int8 = x_int8_tensor.data_ptr<int8_t>();
    const int8_t* __restrict__ w_int8 = w_int8_tensor.data_ptr<int8_t>();
    const int32_t* __restrict__ w_sum = w_sum_tensor.data_ptr<int32_t>();
    int32_t* __restrict__ y = y_tensor.data_ptr<int32_t>();

    const int K_padded = ((K + 63) / 64) * 64;

    constexpr int N_TILE = 64;
    constexpr int M_TILE = 32;
    constexpr int N_BLOCK = 4;

    omp_set_num_threads(num_threads);

    for (int m_tile = 0; m_tile < M; m_tile += M_TILE) {
        const int m_end = std::min(m_tile + M_TILE, M);
        const int m_tile_size = m_end - m_tile;

        uint8_t* x_uint8_tile = (uint8_t*)aligned_alloc(64, m_tile_size * K_padded);

        #pragma omp parallel for schedule(static)
        for (int m = m_tile; m < m_end; m++) {
            const int m_local = m - m_tile;
            uint8_t* x_uint8 = x_uint8_tile + m_local * K_padded;
            const int8_t* x_row = x_int8 + m * K;

            const __m512i offset_vec = _mm512_set1_epi8((char)128);
            int k = 0;
            for (; k + 63 < K; k += 64) {
                __m512i x_vec = _mm512_loadu_si512((__m512i*)(x_row + k));
                __m512i x_u8 = _mm512_add_epi8(x_vec, offset_vec);
                _mm512_store_si512((__m512i*)(x_uint8 + k), x_u8);
            }
            for (; k < K; k++) {
                x_uint8[k] = static_cast<uint8_t>(static_cast<int16_t>(x_row[k]) + 128);
            }
            for (; k < K_padded; k++) {
                x_uint8[k] = 128;
            }
        }

        for (int n_tile = 0; n_tile < N; n_tile += N_TILE) {
            const int n_end = std::min(n_tile + N_TILE, N);

            #pragma omp parallel for schedule(static)
            for (int m = m_tile; m < m_end; m++) {
                const int m_local = m - m_tile;
                const uint8_t* x_uint8 = x_uint8_tile + m_local * K_padded;
                int32_t* y_row = y + m * N;

                int n = n_tile;
                for (; n + N_BLOCK - 1 < n_end; n += N_BLOCK) {
                    __m512i acc0 = _mm512_setzero_si512();
                    __m512i acc1 = _mm512_setzero_si512();
                    __m512i acc2 = _mm512_setzero_si512();
                    __m512i acc3 = _mm512_setzero_si512();

                    const int8_t* w0 = w_int8 + (n + 0) * K_padded;
                    const int8_t* w1 = w_int8 + (n + 1) * K_padded;
                    const int8_t* w2 = w_int8 + (n + 2) * K_padded;
                    const int8_t* w3 = w_int8 + (n + 3) * K_padded;

                    for (int kk = 0; kk < K_padded; kk += 64) {
                        __m512i x_vec = _mm512_load_si512((__m512i*)(x_uint8 + kk));

                        acc0 = _mm512_dpbusd_epi32(acc0, x_vec, _mm512_loadu_si512((__m512i*)(w0 + kk)));
                        acc1 = _mm512_dpbusd_epi32(acc1, x_vec, _mm512_loadu_si512((__m512i*)(w1 + kk)));
                        acc2 = _mm512_dpbusd_epi32(acc2, x_vec, _mm512_loadu_si512((__m512i*)(w2 + kk)));
                        acc3 = _mm512_dpbusd_epi32(acc3, x_vec, _mm512_loadu_si512((__m512i*)(w3 + kk)));
                    }

                    y_row[n + 0] = _mm512_reduce_add_epi32(acc0) - 128 * w_sum[n + 0];
                    y_row[n + 1] = _mm512_reduce_add_epi32(acc1) - 128 * w_sum[n + 1];
                    y_row[n + 2] = _mm512_reduce_add_epi32(acc2) - 128 * w_sum[n + 2];
                    y_row[n + 3] = _mm512_reduce_add_epi32(acc3) - 128 * w_sum[n + 3];
                }

                for (; n < n_end; n++) {
                    const int8_t* w_row = w_int8 + n * K_padded;
                    __m512i acc = _mm512_setzero_si512();

                    for (int kk = 0; kk < K_padded; kk += 64) {
                        __m512i x_vec = _mm512_load_si512((__m512i*)(x_uint8 + kk));
                        acc = _mm512_dpbusd_epi32(acc, x_vec, _mm512_loadu_si512((__m512i*)(w_row + kk)));
                    }

                    y_row[n] = _mm512_reduce_add_epi32(acc) - 128 * w_sum[n];
                }
            }
        }

        free(x_uint8_tile);
    }
}

/**
 * VNNI v3 Fused Softmax - Same cache-optimized tiling as v3 but fuses softmax
 *
 * Key difference from v3:
 * - Accumulates all N outputs for a row into a thread-local buffer
 * - Applies softmax to complete row
 * - Quantizes to int8 output
 *
 * This eliminates the int32 write to memory between matmul and softmax.
 */
void matmul_free_vnni_v3_fused_softmax(
    torch::Tensor x_int8_tensor,    // [M, K] int8
    torch::Tensor scale_tensor,     // [M] float32
    torch::Tensor w_int8_tensor,    // [N, K_padded] int8
    torch::Tensor w_sum_tensor,     // [N] int32
    torch::Tensor y_tensor,         // [M, N] int8 output (softmax applied)
    torch::Tensor y_scale_tensor,   // [M] float32 output scales
    int M, int N, int K,
    int num_threads
) {
    const int8_t* __restrict__ x_int8 = x_int8_tensor.data_ptr<int8_t>();
    const float* __restrict__ scales = scale_tensor.data_ptr<float>();
    const int8_t* __restrict__ w_int8 = w_int8_tensor.data_ptr<int8_t>();
    const int32_t* __restrict__ w_sum = w_sum_tensor.data_ptr<int32_t>();
    int8_t* __restrict__ y = y_tensor.data_ptr<int8_t>();
    float* __restrict__ y_scales = y_scale_tensor.data_ptr<float>();

    const int K_padded = ((K + 63) / 64) * 64;
    const int N_padded = ((N + 15) / 16) * 16;

    constexpr int N_TILE = 64;
    constexpr int M_TILE = 32;
    constexpr int N_BLOCK = 4;

    omp_set_num_threads(num_threads);

    // Process M in tiles
    for (int m_tile = 0; m_tile < M; m_tile += M_TILE) {
        const int m_end = std::min(m_tile + M_TILE, M);
        const int m_tile_size = m_end - m_tile;

        // Allocate activation buffer for this M tile
        uint8_t* x_uint8_tile = (uint8_t*)aligned_alloc(64, m_tile_size * K_padded);

        // Step 1: Convert all activations in this M tile to uint8 (parallelized)
        #pragma omp parallel for schedule(static)
        for (int m = m_tile; m < m_end; m++) {
            const int m_local = m - m_tile;
            uint8_t* x_uint8 = x_uint8_tile + m_local * K_padded;
            const int8_t* x_row = x_int8 + m * K;

            const __m512i offset_vec = _mm512_set1_epi8((char)128);
            int k = 0;
            for (; k + 63 < K; k += 64) {
                __m512i x_vec = _mm512_loadu_si512((__m512i*)(x_row + k));
                __m512i x_u8 = _mm512_add_epi8(x_vec, offset_vec);
                _mm512_store_si512((__m512i*)(x_uint8 + k), x_u8);
            }
            for (; k < K; k++) {
                x_uint8[k] = static_cast<uint8_t>(static_cast<int16_t>(x_row[k]) + 128);
            }
            for (; k < K_padded; k++) {
                x_uint8[k] = 128;
            }
        }

        // Step 2: Compute matmul and apply softmax per row
        // Each thread processes complete rows to enable fused softmax
        #pragma omp parallel
        {
            // Thread-local buffer for accumulating row outputs
            float* row_buffer = (float*)aligned_alloc(64, N_padded * sizeof(float));

            #pragma omp for schedule(static)
            for (int m = m_tile; m < m_end; m++) {
                const int m_local = m - m_tile;
                const uint8_t* x_uint8 = x_uint8_tile + m_local * K_padded;
                float scale = scales[m];
                int8_t* y_row = y + m * N;

                // Compute all N outputs for this row
                for (int n_tile = 0; n_tile < N; n_tile += N_TILE) {
                    const int n_end_tile = std::min(n_tile + N_TILE, N);

                    int n = n_tile;
                    for (; n + N_BLOCK - 1 < n_end_tile; n += N_BLOCK) {
                        __m512i acc0 = _mm512_setzero_si512();
                        __m512i acc1 = _mm512_setzero_si512();
                        __m512i acc2 = _mm512_setzero_si512();
                        __m512i acc3 = _mm512_setzero_si512();

                        const int8_t* w0 = w_int8 + (n + 0) * K_padded;
                        const int8_t* w1 = w_int8 + (n + 1) * K_padded;
                        const int8_t* w2 = w_int8 + (n + 2) * K_padded;
                        const int8_t* w3 = w_int8 + (n + 3) * K_padded;

                        for (int kk = 0; kk < K_padded; kk += 64) {
                            __m512i x_vec = _mm512_load_si512((__m512i*)(x_uint8 + kk));
                            acc0 = _mm512_dpbusd_epi32(acc0, x_vec, _mm512_loadu_si512((__m512i*)(w0 + kk)));
                            acc1 = _mm512_dpbusd_epi32(acc1, x_vec, _mm512_loadu_si512((__m512i*)(w1 + kk)));
                            acc2 = _mm512_dpbusd_epi32(acc2, x_vec, _mm512_loadu_si512((__m512i*)(w2 + kk)));
                            acc3 = _mm512_dpbusd_epi32(acc3, x_vec, _mm512_loadu_si512((__m512i*)(w3 + kk)));
                        }

                        // Convert to float with scale (store in buffer, not memory)
                        row_buffer[n + 0] = static_cast<float>(_mm512_reduce_add_epi32(acc0) - 128 * w_sum[n + 0]) * scale;
                        row_buffer[n + 1] = static_cast<float>(_mm512_reduce_add_epi32(acc1) - 128 * w_sum[n + 1]) * scale;
                        row_buffer[n + 2] = static_cast<float>(_mm512_reduce_add_epi32(acc2) - 128 * w_sum[n + 2]) * scale;
                        row_buffer[n + 3] = static_cast<float>(_mm512_reduce_add_epi32(acc3) - 128 * w_sum[n + 3]) * scale;
                    }

                    // Remainder
                    for (; n < n_end_tile; n++) {
                        const int8_t* w_row = w_int8 + n * K_padded;
                        __m512i acc = _mm512_setzero_si512();
                        for (int kk = 0; kk < K_padded; kk += 64) {
                            __m512i x_vec = _mm512_load_si512((__m512i*)(x_uint8 + kk));
                            acc = _mm512_dpbusd_epi32(acc, x_vec, _mm512_loadu_si512((__m512i*)(w_row + kk)));
                        }
                        row_buffer[n] = static_cast<float>(_mm512_reduce_add_epi32(acc) - 128 * w_sum[n]) * scale;
                    }
                }

                // Pad buffer for SIMD
                for (int n = N; n < N_padded; n++) {
                    row_buffer[n] = -INFINITY;
                }

                // === FUSED SOFTMAX ===

                // Find max (for numerical stability)
                __m512 max_vec = _mm512_set1_ps(-INFINITY);
                for (int n = 0; n < N_padded; n += 16) {
                    __m512 vals = _mm512_load_ps(row_buffer + n);
                    max_vec = _mm512_max_ps(max_vec, vals);
                }
                float max_val = _mm512_reduce_max_ps(max_vec);

                // Compute exp(x - max) and sum
                __m512 sum_vec = _mm512_setzero_ps();
                __m512 max_broadcast = _mm512_set1_ps(max_val);
                for (int n = 0; n < N_padded; n += 16) {
                    __m512 vals = _mm512_load_ps(row_buffer + n);
                    __m512 shifted = _mm512_sub_ps(vals, max_broadcast);
                    __m512 exp_v = avx512_exp_fast(shifted);
                    _mm512_store_ps(row_buffer + n, exp_v);
                    sum_vec = _mm512_add_ps(sum_vec, exp_v);
                }
                float sum = _mm512_reduce_add_ps(sum_vec);

                // Normalize and quantize to int8
                // Softmax output is in [0, 1], scale to [0, 127] for int8
                float inv_sum = 1.0f / sum;
                y_scales[m] = 1.0f / 127.0f;  // Fixed scale for softmax output

                __m512 inv_sum_vec = _mm512_set1_ps(inv_sum);
                __m512 scale_127 = _mm512_set1_ps(127.0f);
                __m512 zero_vec = _mm512_setzero_ps();
                __m512 max_127 = _mm512_set1_ps(127.0f);

                int n = 0;
                for (; n + 15 < N; n += 16) {
                    __m512 exp_v = _mm512_load_ps(row_buffer + n);
                    __m512 prob = _mm512_mul_ps(exp_v, inv_sum_vec);
                    __m512 scaled = _mm512_mul_ps(prob, scale_127);
                    scaled = _mm512_max_ps(scaled, zero_vec);
                    scaled = _mm512_min_ps(scaled, max_127);
                    scaled = _mm512_roundscale_ps(scaled, _MM_FROUND_TO_NEAREST_INT);
                    __m512i int_vals = _mm512_cvtps_epi32(scaled);
                    __m128i packed = _mm512_cvtsepi32_epi8(int_vals);
                    _mm_storeu_si128((__m128i*)(y_row + n), packed);
                }
                for (; n < N; n++) {
                    float prob = row_buffer[n] * inv_sum;
                    int8_t q_val = static_cast<int8_t>(std::min(127.0f, std::round(prob * 127.0f)));
                    y_row[n] = q_val;
                }
            }

            free(row_buffer);
        }

        free(x_uint8_tile);
    }
}

/**
 * VNNI v3 Fused Quantize - Same cache-optimized tiling as v3 but fuses quantization
 *
 * For use with down projection where we just need simple quantization (not softmax).
 */
void matmul_free_vnni_v3_fused_quantize(
    torch::Tensor x_int8_tensor,    // [M, K] int8
    torch::Tensor scale_tensor,     // [M] float32
    torch::Tensor w_int8_tensor,    // [N, K_padded] int8
    torch::Tensor w_sum_tensor,     // [N] int32
    torch::Tensor y_tensor,         // [M, N] int8 output
    torch::Tensor y_scale_tensor,   // [M] float32 output scales
    int M, int N, int K,
    int num_threads
) {
    const int8_t* __restrict__ x_int8 = x_int8_tensor.data_ptr<int8_t>();
    const float* __restrict__ scales = scale_tensor.data_ptr<float>();
    const int8_t* __restrict__ w_int8 = w_int8_tensor.data_ptr<int8_t>();
    const int32_t* __restrict__ w_sum = w_sum_tensor.data_ptr<int32_t>();
    int8_t* __restrict__ y = y_tensor.data_ptr<int8_t>();
    float* __restrict__ y_scales = y_scale_tensor.data_ptr<float>();

    const int K_padded = ((K + 63) / 64) * 64;
    const int N_padded = ((N + 15) / 16) * 16;

    constexpr int N_TILE = 64;
    constexpr int M_TILE = 32;
    constexpr int N_BLOCK = 4;

    omp_set_num_threads(num_threads);

    for (int m_tile = 0; m_tile < M; m_tile += M_TILE) {
        const int m_end = std::min(m_tile + M_TILE, M);
        const int m_tile_size = m_end - m_tile;

        uint8_t* x_uint8_tile = (uint8_t*)aligned_alloc(64, m_tile_size * K_padded);

        #pragma omp parallel for schedule(static)
        for (int m = m_tile; m < m_end; m++) {
            const int m_local = m - m_tile;
            uint8_t* x_uint8 = x_uint8_tile + m_local * K_padded;
            const int8_t* x_row = x_int8 + m * K;

            const __m512i offset_vec = _mm512_set1_epi8((char)128);
            int k = 0;
            for (; k + 63 < K; k += 64) {
                __m512i x_vec = _mm512_loadu_si512((__m512i*)(x_row + k));
                __m512i x_u8 = _mm512_add_epi8(x_vec, offset_vec);
                _mm512_store_si512((__m512i*)(x_uint8 + k), x_u8);
            }
            for (; k < K; k++) {
                x_uint8[k] = static_cast<uint8_t>(static_cast<int16_t>(x_row[k]) + 128);
            }
            for (; k < K_padded; k++) {
                x_uint8[k] = 128;
            }
        }

        #pragma omp parallel
        {
            float* row_buffer = (float*)aligned_alloc(64, N_padded * sizeof(float));

            #pragma omp for schedule(static)
            for (int m = m_tile; m < m_end; m++) {
                const int m_local = m - m_tile;
                const uint8_t* x_uint8 = x_uint8_tile + m_local * K_padded;
                float scale = scales[m];
                int8_t* y_row = y + m * N;

                // Compute all N outputs and find max abs
                float max_abs = 0.0f;

                for (int n_tile = 0; n_tile < N; n_tile += N_TILE) {
                    const int n_end_tile = std::min(n_tile + N_TILE, N);

                    int n = n_tile;
                    for (; n + N_BLOCK - 1 < n_end_tile; n += N_BLOCK) {
                        __m512i acc0 = _mm512_setzero_si512();
                        __m512i acc1 = _mm512_setzero_si512();
                        __m512i acc2 = _mm512_setzero_si512();
                        __m512i acc3 = _mm512_setzero_si512();

                        const int8_t* w0 = w_int8 + (n + 0) * K_padded;
                        const int8_t* w1 = w_int8 + (n + 1) * K_padded;
                        const int8_t* w2 = w_int8 + (n + 2) * K_padded;
                        const int8_t* w3 = w_int8 + (n + 3) * K_padded;

                        for (int kk = 0; kk < K_padded; kk += 64) {
                            __m512i x_vec = _mm512_load_si512((__m512i*)(x_uint8 + kk));
                            acc0 = _mm512_dpbusd_epi32(acc0, x_vec, _mm512_loadu_si512((__m512i*)(w0 + kk)));
                            acc1 = _mm512_dpbusd_epi32(acc1, x_vec, _mm512_loadu_si512((__m512i*)(w1 + kk)));
                            acc2 = _mm512_dpbusd_epi32(acc2, x_vec, _mm512_loadu_si512((__m512i*)(w2 + kk)));
                            acc3 = _mm512_dpbusd_epi32(acc3, x_vec, _mm512_loadu_si512((__m512i*)(w3 + kk)));
                        }

                        float v0 = static_cast<float>(_mm512_reduce_add_epi32(acc0) - 128 * w_sum[n + 0]) * scale;
                        float v1 = static_cast<float>(_mm512_reduce_add_epi32(acc1) - 128 * w_sum[n + 1]) * scale;
                        float v2 = static_cast<float>(_mm512_reduce_add_epi32(acc2) - 128 * w_sum[n + 2]) * scale;
                        float v3 = static_cast<float>(_mm512_reduce_add_epi32(acc3) - 128 * w_sum[n + 3]) * scale;

                        row_buffer[n + 0] = v0;
                        row_buffer[n + 1] = v1;
                        row_buffer[n + 2] = v2;
                        row_buffer[n + 3] = v3;

                        max_abs = std::max(max_abs, std::abs(v0));
                        max_abs = std::max(max_abs, std::abs(v1));
                        max_abs = std::max(max_abs, std::abs(v2));
                        max_abs = std::max(max_abs, std::abs(v3));
                    }

                    for (; n < n_end_tile; n++) {
                        const int8_t* w_row = w_int8 + n * K_padded;
                        __m512i acc = _mm512_setzero_si512();
                        for (int kk = 0; kk < K_padded; kk += 64) {
                            __m512i x_vec = _mm512_load_si512((__m512i*)(x_uint8 + kk));
                            acc = _mm512_dpbusd_epi32(acc, x_vec, _mm512_loadu_si512((__m512i*)(w_row + kk)));
                        }
                        float v = static_cast<float>(_mm512_reduce_add_epi32(acc) - 128 * w_sum[n]) * scale;
                        row_buffer[n] = v;
                        max_abs = std::max(max_abs, std::abs(v));
                    }
                }

                // Compute output scale
                float out_scale = max_abs / 127.0f;
                if (out_scale == 0.0f) out_scale = 1.0f;
                y_scales[m] = out_scale;
                float inv_scale = 1.0f / out_scale;

                // Quantize to int8
                __m512 inv_scale_vec = _mm512_set1_ps(inv_scale);
                __m512 min_val = _mm512_set1_ps(-127.0f);
                __m512 max_val = _mm512_set1_ps(127.0f);

                int n = 0;
                for (; n + 15 < N; n += 16) {
                    __m512 vals = _mm512_loadu_ps(row_buffer + n);
                    vals = _mm512_mul_ps(vals, inv_scale_vec);
                    vals = _mm512_max_ps(vals, min_val);
                    vals = _mm512_min_ps(vals, max_val);
                    vals = _mm512_roundscale_ps(vals, _MM_FROUND_TO_NEAREST_INT);
                    __m512i int_vals = _mm512_cvtps_epi32(vals);
                    __m128i packed = _mm512_cvtsepi32_epi8(int_vals);
                    _mm_storeu_si128((__m128i*)(y_row + n), packed);
                }
                for (; n < N; n++) {
                    float val = row_buffer[n] * inv_scale;
                    val = std::max(-127.0f, std::min(127.0f, val));
                    y_row[n] = static_cast<int8_t>(std::round(val));
                }
            }

            free(row_buffer);
        }

        free(x_uint8_tile);
    }
}

/**
 * VNNI v3 Fused QKV + Softmax - Computes Q, K, V projections reading input once
 *
 * Same cache-optimized tiling as v3 but computes all three projections
 * while only reading the input activation once.
 */
void matmul_free_vnni_v3_fused_qkv_softmax(
    torch::Tensor x_int8_tensor,    // [M, K] int8
    torch::Tensor scale_tensor,     // [M] float32
    torch::Tensor wq_tensor,        // [N, K_padded] int8
    torch::Tensor wq_sum_tensor,    // [N] int32
    torch::Tensor wk_tensor,        // [N, K_padded] int8
    torch::Tensor wk_sum_tensor,    // [N] int32
    torch::Tensor wv_tensor,        // [N, K_padded] int8
    torch::Tensor wv_sum_tensor,    // [N] int32
    torch::Tensor q_tensor,         // [M, N] int8 output
    torch::Tensor q_scale_tensor,   // [M] float32
    torch::Tensor k_tensor,         // [M, N] int8 output
    torch::Tensor k_scale_tensor,   // [M] float32
    torch::Tensor v_tensor,         // [M, N] int8 output
    torch::Tensor v_scale_tensor,   // [M] float32
    int M, int N, int K,
    int num_threads
) {
    const int8_t* __restrict__ x_int8 = x_int8_tensor.data_ptr<int8_t>();
    const float* __restrict__ scales = scale_tensor.data_ptr<float>();
    const int8_t* __restrict__ wq = wq_tensor.data_ptr<int8_t>();
    const int32_t* __restrict__ wq_sum = wq_sum_tensor.data_ptr<int32_t>();
    const int8_t* __restrict__ wk = wk_tensor.data_ptr<int8_t>();
    const int32_t* __restrict__ wk_sum = wk_sum_tensor.data_ptr<int32_t>();
    const int8_t* __restrict__ wv = wv_tensor.data_ptr<int8_t>();
    const int32_t* __restrict__ wv_sum = wv_sum_tensor.data_ptr<int32_t>();
    int8_t* __restrict__ q_out = q_tensor.data_ptr<int8_t>();
    float* __restrict__ q_scales = q_scale_tensor.data_ptr<float>();
    int8_t* __restrict__ k_out = k_tensor.data_ptr<int8_t>();
    float* __restrict__ k_scales = k_scale_tensor.data_ptr<float>();
    int8_t* __restrict__ v_out = v_tensor.data_ptr<int8_t>();
    float* __restrict__ v_scales = v_scale_tensor.data_ptr<float>();

    const int K_padded = ((K + 63) / 64) * 64;
    const int N_padded = ((N + 15) / 16) * 16;

    constexpr int N_TILE = 64;
    constexpr int M_TILE = 32;
    constexpr int N_BLOCK = 4;

    omp_set_num_threads(num_threads);

    // Lambda to apply softmax and quantize a row buffer
    auto apply_softmax_quantize = [&](float* row_buffer, int8_t* y_row, float* y_scale, int row_idx) {
        // Pad buffer
        for (int n = N; n < N_padded; n++) {
            row_buffer[n] = -INFINITY;
        }

        // Find max
        __m512 max_vec = _mm512_set1_ps(-INFINITY);
        for (int n = 0; n < N_padded; n += 16) {
            __m512 vals = _mm512_load_ps(row_buffer + n);
            max_vec = _mm512_max_ps(max_vec, vals);
        }
        float max_val = _mm512_reduce_max_ps(max_vec);

        // Compute exp(x - max) and sum
        __m512 sum_vec = _mm512_setzero_ps();
        __m512 max_broadcast = _mm512_set1_ps(max_val);
        for (int n = 0; n < N_padded; n += 16) {
            __m512 vals = _mm512_load_ps(row_buffer + n);
            __m512 shifted = _mm512_sub_ps(vals, max_broadcast);
            __m512 exp_v = avx512_exp_fast(shifted);
            _mm512_store_ps(row_buffer + n, exp_v);
            sum_vec = _mm512_add_ps(sum_vec, exp_v);
        }
        float sum = _mm512_reduce_add_ps(sum_vec);

        // Normalize and quantize
        float inv_sum = 1.0f / sum;
        y_scale[row_idx] = 1.0f / 127.0f;

        __m512 inv_sum_vec = _mm512_set1_ps(inv_sum);
        __m512 scale_127 = _mm512_set1_ps(127.0f);
        __m512 zero_vec = _mm512_setzero_ps();
        __m512 max_127 = _mm512_set1_ps(127.0f);

        int n = 0;
        for (; n + 15 < N; n += 16) {
            __m512 exp_v = _mm512_load_ps(row_buffer + n);
            __m512 prob = _mm512_mul_ps(exp_v, inv_sum_vec);
            __m512 scaled = _mm512_mul_ps(prob, scale_127);
            scaled = _mm512_max_ps(scaled, zero_vec);
            scaled = _mm512_min_ps(scaled, max_127);
            scaled = _mm512_roundscale_ps(scaled, _MM_FROUND_TO_NEAREST_INT);
            __m512i int_vals = _mm512_cvtps_epi32(scaled);
            __m128i packed = _mm512_cvtsepi32_epi8(int_vals);
            _mm_storeu_si128((__m128i*)(y_row + n), packed);
        }
        for (; n < N; n++) {
            float prob = row_buffer[n] * inv_sum;
            y_row[n] = static_cast<int8_t>(std::min(127.0f, std::round(prob * 127.0f)));
        }
    };

    // Process M in tiles
    for (int m_tile = 0; m_tile < M; m_tile += M_TILE) {
        const int m_end = std::min(m_tile + M_TILE, M);
        const int m_tile_size = m_end - m_tile;

        // Allocate activation buffer for this M tile (shared across Q, K, V)
        uint8_t* x_uint8_tile = (uint8_t*)aligned_alloc(64, m_tile_size * K_padded);

        // Convert activations once
        #pragma omp parallel for schedule(static)
        for (int m = m_tile; m < m_end; m++) {
            const int m_local = m - m_tile;
            uint8_t* x_uint8 = x_uint8_tile + m_local * K_padded;
            const int8_t* x_row = x_int8 + m * K;

            const __m512i offset_vec = _mm512_set1_epi8((char)128);
            int k = 0;
            for (; k + 63 < K; k += 64) {
                __m512i x_vec = _mm512_loadu_si512((__m512i*)(x_row + k));
                __m512i x_u8 = _mm512_add_epi8(x_vec, offset_vec);
                _mm512_store_si512((__m512i*)(x_uint8 + k), x_u8);
            }
            for (; k < K; k++) {
                x_uint8[k] = static_cast<uint8_t>(static_cast<int16_t>(x_row[k]) + 128);
            }
            for (; k < K_padded; k++) {
                x_uint8[k] = 128;
            }
        }

        // Compute Q, K, V for all rows in this M tile
        #pragma omp parallel
        {
            // Thread-local buffers for all three outputs
            float* q_buffer = (float*)aligned_alloc(64, N_padded * sizeof(float));
            float* k_buffer = (float*)aligned_alloc(64, N_padded * sizeof(float));
            float* v_buffer = (float*)aligned_alloc(64, N_padded * sizeof(float));

            #pragma omp for schedule(static)
            for (int m = m_tile; m < m_end; m++) {
                const int m_local = m - m_tile;
                const uint8_t* x_uint8 = x_uint8_tile + m_local * K_padded;
                float scale = scales[m];

                // Compute all N outputs for Q, K, V together
                for (int n_tile = 0; n_tile < N; n_tile += N_TILE) {
                    const int n_end_tile = std::min(n_tile + N_TILE, N);

                    int n = n_tile;
                    for (; n + N_BLOCK - 1 < n_end_tile; n += N_BLOCK) {
                        // Q accumulators
                        __m512i accq0 = _mm512_setzero_si512();
                        __m512i accq1 = _mm512_setzero_si512();
                        __m512i accq2 = _mm512_setzero_si512();
                        __m512i accq3 = _mm512_setzero_si512();
                        // K accumulators
                        __m512i acck0 = _mm512_setzero_si512();
                        __m512i acck1 = _mm512_setzero_si512();
                        __m512i acck2 = _mm512_setzero_si512();
                        __m512i acck3 = _mm512_setzero_si512();
                        // V accumulators
                        __m512i accv0 = _mm512_setzero_si512();
                        __m512i accv1 = _mm512_setzero_si512();
                        __m512i accv2 = _mm512_setzero_si512();
                        __m512i accv3 = _mm512_setzero_si512();

                        const int8_t* wq0 = wq + (n + 0) * K_padded;
                        const int8_t* wq1 = wq + (n + 1) * K_padded;
                        const int8_t* wq2 = wq + (n + 2) * K_padded;
                        const int8_t* wq3 = wq + (n + 3) * K_padded;
                        const int8_t* wk0 = wk + (n + 0) * K_padded;
                        const int8_t* wk1 = wk + (n + 1) * K_padded;
                        const int8_t* wk2 = wk + (n + 2) * K_padded;
                        const int8_t* wk3 = wk + (n + 3) * K_padded;
                        const int8_t* wv0 = wv + (n + 0) * K_padded;
                        const int8_t* wv1 = wv + (n + 1) * K_padded;
                        const int8_t* wv2 = wv + (n + 2) * K_padded;
                        const int8_t* wv3 = wv + (n + 3) * K_padded;

                        for (int kk = 0; kk < K_padded; kk += 64) {
                            __m512i x_vec = _mm512_load_si512((__m512i*)(x_uint8 + kk));
                            // Q
                            accq0 = _mm512_dpbusd_epi32(accq0, x_vec, _mm512_loadu_si512((__m512i*)(wq0 + kk)));
                            accq1 = _mm512_dpbusd_epi32(accq1, x_vec, _mm512_loadu_si512((__m512i*)(wq1 + kk)));
                            accq2 = _mm512_dpbusd_epi32(accq2, x_vec, _mm512_loadu_si512((__m512i*)(wq2 + kk)));
                            accq3 = _mm512_dpbusd_epi32(accq3, x_vec, _mm512_loadu_si512((__m512i*)(wq3 + kk)));
                            // K
                            acck0 = _mm512_dpbusd_epi32(acck0, x_vec, _mm512_loadu_si512((__m512i*)(wk0 + kk)));
                            acck1 = _mm512_dpbusd_epi32(acck1, x_vec, _mm512_loadu_si512((__m512i*)(wk1 + kk)));
                            acck2 = _mm512_dpbusd_epi32(acck2, x_vec, _mm512_loadu_si512((__m512i*)(wk2 + kk)));
                            acck3 = _mm512_dpbusd_epi32(acck3, x_vec, _mm512_loadu_si512((__m512i*)(wk3 + kk)));
                            // V
                            accv0 = _mm512_dpbusd_epi32(accv0, x_vec, _mm512_loadu_si512((__m512i*)(wv0 + kk)));
                            accv1 = _mm512_dpbusd_epi32(accv1, x_vec, _mm512_loadu_si512((__m512i*)(wv1 + kk)));
                            accv2 = _mm512_dpbusd_epi32(accv2, x_vec, _mm512_loadu_si512((__m512i*)(wv2 + kk)));
                            accv3 = _mm512_dpbusd_epi32(accv3, x_vec, _mm512_loadu_si512((__m512i*)(wv3 + kk)));
                        }

                        // Store Q results
                        q_buffer[n + 0] = static_cast<float>(_mm512_reduce_add_epi32(accq0) - 128 * wq_sum[n + 0]) * scale;
                        q_buffer[n + 1] = static_cast<float>(_mm512_reduce_add_epi32(accq1) - 128 * wq_sum[n + 1]) * scale;
                        q_buffer[n + 2] = static_cast<float>(_mm512_reduce_add_epi32(accq2) - 128 * wq_sum[n + 2]) * scale;
                        q_buffer[n + 3] = static_cast<float>(_mm512_reduce_add_epi32(accq3) - 128 * wq_sum[n + 3]) * scale;
                        // Store K results
                        k_buffer[n + 0] = static_cast<float>(_mm512_reduce_add_epi32(acck0) - 128 * wk_sum[n + 0]) * scale;
                        k_buffer[n + 1] = static_cast<float>(_mm512_reduce_add_epi32(acck1) - 128 * wk_sum[n + 1]) * scale;
                        k_buffer[n + 2] = static_cast<float>(_mm512_reduce_add_epi32(acck2) - 128 * wk_sum[n + 2]) * scale;
                        k_buffer[n + 3] = static_cast<float>(_mm512_reduce_add_epi32(acck3) - 128 * wk_sum[n + 3]) * scale;
                        // Store V results
                        v_buffer[n + 0] = static_cast<float>(_mm512_reduce_add_epi32(accv0) - 128 * wv_sum[n + 0]) * scale;
                        v_buffer[n + 1] = static_cast<float>(_mm512_reduce_add_epi32(accv1) - 128 * wv_sum[n + 1]) * scale;
                        v_buffer[n + 2] = static_cast<float>(_mm512_reduce_add_epi32(accv2) - 128 * wv_sum[n + 2]) * scale;
                        v_buffer[n + 3] = static_cast<float>(_mm512_reduce_add_epi32(accv3) - 128 * wv_sum[n + 3]) * scale;
                    }

                    // Remainder
                    for (; n < n_end_tile; n++) {
                        const int8_t* wq_row = wq + n * K_padded;
                        const int8_t* wk_row = wk + n * K_padded;
                        const int8_t* wv_row = wv + n * K_padded;
                        __m512i accq = _mm512_setzero_si512();
                        __m512i acck = _mm512_setzero_si512();
                        __m512i accv = _mm512_setzero_si512();
                        for (int kk = 0; kk < K_padded; kk += 64) {
                            __m512i x_vec = _mm512_load_si512((__m512i*)(x_uint8 + kk));
                            accq = _mm512_dpbusd_epi32(accq, x_vec, _mm512_loadu_si512((__m512i*)(wq_row + kk)));
                            acck = _mm512_dpbusd_epi32(acck, x_vec, _mm512_loadu_si512((__m512i*)(wk_row + kk)));
                            accv = _mm512_dpbusd_epi32(accv, x_vec, _mm512_loadu_si512((__m512i*)(wv_row + kk)));
                        }
                        q_buffer[n] = static_cast<float>(_mm512_reduce_add_epi32(accq) - 128 * wq_sum[n]) * scale;
                        k_buffer[n] = static_cast<float>(_mm512_reduce_add_epi32(acck) - 128 * wk_sum[n]) * scale;
                        v_buffer[n] = static_cast<float>(_mm512_reduce_add_epi32(accv) - 128 * wv_sum[n]) * scale;
                    }
                }

                // Apply softmax and quantize to each output
                apply_softmax_quantize(q_buffer, q_out + m * N, q_scales, m);
                apply_softmax_quantize(k_buffer, k_out + m * N, k_scales, m);
                apply_softmax_quantize(v_buffer, v_out + m * N, v_scales, m);
            }

            free(q_buffer);
            free(k_buffer);
            free(v_buffer);
        }

        free(x_uint8_tile);
    }
}

/**
 * VNNI v4 - Optimized for large N (MLP layers)
 *
 * Key optimizations for N=16384:
 * 1. Parallelize over N instead of M for large N
 * 2. Process 16 outputs at a time (more register blocking)
 * 3. Better work distribution across threads
 * 4. Minimize activation conversion overhead
 *
 * This kernel is specifically designed for MLP up projection where:
 * - M is small (1-128)
 * - N is large (16384)
 * - K is medium (2048)
 */
void matmul_free_vnni_v4_large_n(
    torch::Tensor x_int8_tensor,    // [M, K] int8
    torch::Tensor scale_tensor,     // [M] float32
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

    const int K_padded = ((K + 63) / 64) * 64;

    // For large N, parallelize over N blocks
    constexpr int N_BLOCK = 16;  // Process 16 outputs per iteration
    const int n_blocks = (N + N_BLOCK - 1) / N_BLOCK;

    omp_set_num_threads(num_threads);

    // Precompute all uint8 activations (small cost for small M)
    uint8_t* x_uint8_all = (uint8_t*)aligned_alloc(64, M * K_padded);

    #pragma omp parallel for schedule(static)
    for (int m = 0; m < M; m++) {
        uint8_t* x_uint8 = x_uint8_all + m * K_padded;
        const int8_t* x_row = x_int8 + m * K;

        const __m512i offset_vec = _mm512_set1_epi8((char)128);
        int k = 0;
        for (; k + 63 < K; k += 64) {
            __m512i x_vec = _mm512_loadu_si512((__m512i*)(x_row + k));
            __m512i x_u8 = _mm512_add_epi8(x_vec, offset_vec);
            _mm512_store_si512((__m512i*)(x_uint8 + k), x_u8);
        }
        for (; k < K; k++) {
            x_uint8[k] = static_cast<uint8_t>(static_cast<int16_t>(x_row[k]) + 128);
        }
        for (; k < K_padded; k++) {
            x_uint8[k] = 128;
        }
    }

    // Parallelize over N blocks (better for large N)
    #pragma omp parallel for schedule(dynamic, 4)
    for (int nb = 0; nb < n_blocks; nb++) {
        const int n_start = nb * N_BLOCK;
        const int n_end = std::min(n_start + N_BLOCK, N);
        const int block_size = n_end - n_start;

        // Process all M rows for this N block
        for (int m = 0; m < M; m++) {
            const uint8_t* x_uint8 = x_uint8_all + m * K_padded;
            int32_t* y_row = y + m * N;

            // Process up to 16 outputs at a time with register blocking
            if (block_size >= 16) {
                __m512i acc[16];
                for (int i = 0; i < 16; i++) {
                    acc[i] = _mm512_setzero_si512();
                }

                // Load weight pointers
                const int8_t* w_ptrs[16];
                for (int i = 0; i < 16; i++) {
                    w_ptrs[i] = w_int8 + (n_start + i) * K_padded;
                }

                // Main computation loop
                for (int kk = 0; kk < K_padded; kk += 64) {
                    __m512i x_vec = _mm512_load_si512((__m512i*)(x_uint8 + kk));

                    // Unrolled for 16 outputs
                    acc[0] = _mm512_dpbusd_epi32(acc[0], x_vec, _mm512_loadu_si512((__m512i*)(w_ptrs[0] + kk)));
                    acc[1] = _mm512_dpbusd_epi32(acc[1], x_vec, _mm512_loadu_si512((__m512i*)(w_ptrs[1] + kk)));
                    acc[2] = _mm512_dpbusd_epi32(acc[2], x_vec, _mm512_loadu_si512((__m512i*)(w_ptrs[2] + kk)));
                    acc[3] = _mm512_dpbusd_epi32(acc[3], x_vec, _mm512_loadu_si512((__m512i*)(w_ptrs[3] + kk)));
                    acc[4] = _mm512_dpbusd_epi32(acc[4], x_vec, _mm512_loadu_si512((__m512i*)(w_ptrs[4] + kk)));
                    acc[5] = _mm512_dpbusd_epi32(acc[5], x_vec, _mm512_loadu_si512((__m512i*)(w_ptrs[5] + kk)));
                    acc[6] = _mm512_dpbusd_epi32(acc[6], x_vec, _mm512_loadu_si512((__m512i*)(w_ptrs[6] + kk)));
                    acc[7] = _mm512_dpbusd_epi32(acc[7], x_vec, _mm512_loadu_si512((__m512i*)(w_ptrs[7] + kk)));
                    acc[8] = _mm512_dpbusd_epi32(acc[8], x_vec, _mm512_loadu_si512((__m512i*)(w_ptrs[8] + kk)));
                    acc[9] = _mm512_dpbusd_epi32(acc[9], x_vec, _mm512_loadu_si512((__m512i*)(w_ptrs[9] + kk)));
                    acc[10] = _mm512_dpbusd_epi32(acc[10], x_vec, _mm512_loadu_si512((__m512i*)(w_ptrs[10] + kk)));
                    acc[11] = _mm512_dpbusd_epi32(acc[11], x_vec, _mm512_loadu_si512((__m512i*)(w_ptrs[11] + kk)));
                    acc[12] = _mm512_dpbusd_epi32(acc[12], x_vec, _mm512_loadu_si512((__m512i*)(w_ptrs[12] + kk)));
                    acc[13] = _mm512_dpbusd_epi32(acc[13], x_vec, _mm512_loadu_si512((__m512i*)(w_ptrs[13] + kk)));
                    acc[14] = _mm512_dpbusd_epi32(acc[14], x_vec, _mm512_loadu_si512((__m512i*)(w_ptrs[14] + kk)));
                    acc[15] = _mm512_dpbusd_epi32(acc[15], x_vec, _mm512_loadu_si512((__m512i*)(w_ptrs[15] + kk)));
                }

                // Reduce and store
                for (int i = 0; i < 16; i++) {
                    y_row[n_start + i] = _mm512_reduce_add_epi32(acc[i]) - 128 * w_sum[n_start + i];
                }
            } else {
                // Handle remainder (< 16 outputs)
                for (int n = n_start; n < n_end; n++) {
                    const int8_t* w_row = w_int8 + n * K_padded;
                    __m512i acc = _mm512_setzero_si512();

                    for (int kk = 0; kk < K_padded; kk += 64) {
                        __m512i x_vec = _mm512_load_si512((__m512i*)(x_uint8 + kk));
                        acc = _mm512_dpbusd_epi32(acc, x_vec, _mm512_loadu_si512((__m512i*)(w_row + kk)));
                    }

                    y_row[n] = _mm512_reduce_add_epi32(acc) - 128 * w_sum[n];
                }
            }
        }
    }

    free(x_uint8_all);
}

/**
 * v5 kernel: Optimized for large M with moderate N
 *
 * Parallelizes over M×N_tile space for better thread utilization.
 * For down_matmul: M=128, N=2048, K=8192
 * Total work units: 128 × (2048/16) = 128 × 128 = 16384 tiles
 *
 * This is much better than v3 (serial outer loops) or v4 (serial over M).
 */
void matmul_free_vnni_v5_large_m(
    torch::Tensor x_int8_tensor,    // [M, K] int8
    torch::Tensor scale_tensor,     // [M] float32
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

    const int K_padded = ((K + 63) / 64) * 64;

    // Tile sizes
    constexpr int N_BLOCK = 16;  // Process 16 outputs per N tile
    const int n_tiles = (N + N_BLOCK - 1) / N_BLOCK;
    const int total_tiles = M * n_tiles;

    omp_set_num_threads(num_threads);

    // Precompute all uint8 activations first (parallelize this)
    uint8_t* x_uint8_all = (uint8_t*)aligned_alloc(64, M * K_padded);

    #pragma omp parallel for schedule(static)
    for (int m = 0; m < M; m++) {
        uint8_t* x_uint8 = x_uint8_all + m * K_padded;
        const int8_t* x_row = x_int8 + m * K;

        const __m512i offset_vec = _mm512_set1_epi8((char)128);
        int k = 0;
        for (; k + 63 < K; k += 64) {
            __m512i x_vec = _mm512_loadu_si512((__m512i*)(x_row + k));
            __m512i x_u8 = _mm512_add_epi8(x_vec, offset_vec);
            _mm512_store_si512((__m512i*)(x_uint8 + k), x_u8);
        }
        for (; k < K; k++) {
            x_uint8[k] = static_cast<uint8_t>(static_cast<int16_t>(x_row[k]) + 128);
        }
        for (; k < K_padded; k++) {
            x_uint8[k] = 128;
        }
    }

    // Parallelize over M×N_tiles (flattened)
    #pragma omp parallel for schedule(dynamic, 8)
    for (int tile_idx = 0; tile_idx < total_tiles; tile_idx++) {
        const int m = tile_idx / n_tiles;
        const int n_tile = tile_idx % n_tiles;
        const int n_start = n_tile * N_BLOCK;
        const int n_end = std::min(n_start + N_BLOCK, N);
        const int block_size = n_end - n_start;

        const uint8_t* x_uint8 = x_uint8_all + m * K_padded;
        int32_t* y_row = y + m * N;

        // Process up to 16 outputs at a time with register blocking
        if (block_size >= 16) {
            __m512i acc[16];
            for (int i = 0; i < 16; i++) {
                acc[i] = _mm512_setzero_si512();
            }

            // Load weight pointers
            const int8_t* w_ptrs[16];
            for (int i = 0; i < 16; i++) {
                w_ptrs[i] = w_int8 + (n_start + i) * K_padded;
            }

            // Main computation loop - unrolled for register reuse
            for (int kk = 0; kk < K_padded; kk += 64) {
                __m512i x_vec = _mm512_load_si512((__m512i*)(x_uint8 + kk));

                acc[0] = _mm512_dpbusd_epi32(acc[0], x_vec, _mm512_loadu_si512((__m512i*)(w_ptrs[0] + kk)));
                acc[1] = _mm512_dpbusd_epi32(acc[1], x_vec, _mm512_loadu_si512((__m512i*)(w_ptrs[1] + kk)));
                acc[2] = _mm512_dpbusd_epi32(acc[2], x_vec, _mm512_loadu_si512((__m512i*)(w_ptrs[2] + kk)));
                acc[3] = _mm512_dpbusd_epi32(acc[3], x_vec, _mm512_loadu_si512((__m512i*)(w_ptrs[3] + kk)));
                acc[4] = _mm512_dpbusd_epi32(acc[4], x_vec, _mm512_loadu_si512((__m512i*)(w_ptrs[4] + kk)));
                acc[5] = _mm512_dpbusd_epi32(acc[5], x_vec, _mm512_loadu_si512((__m512i*)(w_ptrs[5] + kk)));
                acc[6] = _mm512_dpbusd_epi32(acc[6], x_vec, _mm512_loadu_si512((__m512i*)(w_ptrs[6] + kk)));
                acc[7] = _mm512_dpbusd_epi32(acc[7], x_vec, _mm512_loadu_si512((__m512i*)(w_ptrs[7] + kk)));
                acc[8] = _mm512_dpbusd_epi32(acc[8], x_vec, _mm512_loadu_si512((__m512i*)(w_ptrs[8] + kk)));
                acc[9] = _mm512_dpbusd_epi32(acc[9], x_vec, _mm512_loadu_si512((__m512i*)(w_ptrs[9] + kk)));
                acc[10] = _mm512_dpbusd_epi32(acc[10], x_vec, _mm512_loadu_si512((__m512i*)(w_ptrs[10] + kk)));
                acc[11] = _mm512_dpbusd_epi32(acc[11], x_vec, _mm512_loadu_si512((__m512i*)(w_ptrs[11] + kk)));
                acc[12] = _mm512_dpbusd_epi32(acc[12], x_vec, _mm512_loadu_si512((__m512i*)(w_ptrs[12] + kk)));
                acc[13] = _mm512_dpbusd_epi32(acc[13], x_vec, _mm512_loadu_si512((__m512i*)(w_ptrs[13] + kk)));
                acc[14] = _mm512_dpbusd_epi32(acc[14], x_vec, _mm512_loadu_si512((__m512i*)(w_ptrs[14] + kk)));
                acc[15] = _mm512_dpbusd_epi32(acc[15], x_vec, _mm512_loadu_si512((__m512i*)(w_ptrs[15] + kk)));
            }

            // Reduce and store
            for (int i = 0; i < 16; i++) {
                y_row[n_start + i] = _mm512_reduce_add_epi32(acc[i]) - 128 * w_sum[n_start + i];
            }
        } else {
            // Handle remainder (< 16 outputs)
            for (int n = n_start; n < n_end; n++) {
                const int8_t* w_row = w_int8 + n * K_padded;
                __m512i acc = _mm512_setzero_si512();

                for (int kk = 0; kk < K_padded; kk += 64) {
                    __m512i x_vec = _mm512_load_si512((__m512i*)(x_uint8 + kk));
                    acc = _mm512_dpbusd_epi32(acc, x_vec, _mm512_loadu_si512((__m512i*)(w_row + kk)));
                }

                y_row[n] = _mm512_reduce_add_epi32(acc) - 128 * w_sum[n];
            }
        }
    }

    free(x_uint8_all);
}

/**
 * Fused MLP: Up projection + SwiGLU + Down projection
 *
 * Combines three operations into one kernel to minimize memory traffic:
 * 1. x @ W_up -> [M, mlp_hidden]
 * 2. SwiGLU -> [M, mlp_hidden/2]
 * 3. result @ W_down -> [M, hidden_dim]
 *
 * Optimized version with:
 * - Heap allocation instead of stack (large arrays)
 * - Proper parallelization over M
 * - Vectorized inner loops with 4-way unrolling
 */
void fused_mlp_int8(
    torch::Tensor x_tensor,         // [M, hidden_dim] int8 input
    torch::Tensor x_scale_tensor,   // [M] float32 input scales
    torch::Tensor w_up_tensor,      // [mlp_hidden, K_padded] int8
    torch::Tensor w_up_sum_tensor,  // [mlp_hidden] int32
    torch::Tensor w_down_tensor,    // [hidden_dim, K_padded_down] int8
    torch::Tensor w_down_sum_tensor,// [hidden_dim] int32
    torch::Tensor y_tensor,         // [M, hidden_dim] int8 output
    torch::Tensor y_scale_tensor,   // [M] float32 output scales
    int M, int hidden_dim, int mlp_hidden,
    int num_threads
) {
    const int8_t* __restrict__ x = x_tensor.data_ptr<int8_t>();
    const float* __restrict__ x_scales = x_scale_tensor.data_ptr<float>();
    const int8_t* __restrict__ w_up = w_up_tensor.data_ptr<int8_t>();
    const int32_t* __restrict__ w_up_sum = w_up_sum_tensor.data_ptr<int32_t>();
    const int8_t* __restrict__ w_down = w_down_tensor.data_ptr<int8_t>();
    const int32_t* __restrict__ w_down_sum = w_down_sum_tensor.data_ptr<int32_t>();
    int8_t* __restrict__ y = y_tensor.data_ptr<int8_t>();
    float* __restrict__ y_scales = y_scale_tensor.data_ptr<float>();

    const int K_up_padded = ((hidden_dim + 63) / 64) * 64;
    const int mlp_intermediate = mlp_hidden / 2;
    const int K_down_padded = ((mlp_intermediate + 63) / 64) * 64;

    omp_set_num_threads(num_threads);

    // Pre-allocate per-thread buffers to avoid repeated allocation
    #pragma omp parallel
    {
        // Thread-local buffers (heap allocated)
        uint8_t* x_uint8 = (uint8_t*)aligned_alloc(64, K_up_padded);
        float* up_float = (float*)aligned_alloc(64, mlp_hidden * sizeof(float));
        float* swiglu_float = (float*)aligned_alloc(64, mlp_intermediate * sizeof(float));
        uint8_t* swiglu_uint8 = (uint8_t*)aligned_alloc(64, K_down_padded);
        float* down_float = (float*)aligned_alloc(64, hidden_dim * sizeof(float));

        #pragma omp for schedule(static)
        for (int m = 0; m < M; m++) {
            const int8_t* x_row = x + m * hidden_dim;
            float x_scale = x_scales[m];
            int8_t* y_row = y + m * hidden_dim;

            // Step 1: Convert input to uint8 (vectorized)
            const __m512i offset_vec = _mm512_set1_epi8((char)128);
            int k = 0;
            for (; k + 63 < hidden_dim; k += 64) {
                __m512i x_vec = _mm512_loadu_si512((__m512i*)(x_row + k));
                __m512i x_u8 = _mm512_add_epi8(x_vec, offset_vec);
                _mm512_store_si512((__m512i*)(x_uint8 + k), x_u8);
            }
            for (; k < hidden_dim; k++) {
                x_uint8[k] = static_cast<uint8_t>(static_cast<int16_t>(x_row[k]) + 128);
            }
            for (; k < K_up_padded; k++) {
                x_uint8[k] = 128;
            }

            // Step 2: Up projection with 4-way unrolling
            int n = 0;
            for (; n + 3 < mlp_hidden; n += 4) {
                const int8_t* w0 = w_up + (n + 0) * K_up_padded;
                const int8_t* w1 = w_up + (n + 1) * K_up_padded;
                const int8_t* w2 = w_up + (n + 2) * K_up_padded;
                const int8_t* w3 = w_up + (n + 3) * K_up_padded;

                __m512i acc0 = _mm512_setzero_si512();
                __m512i acc1 = _mm512_setzero_si512();
                __m512i acc2 = _mm512_setzero_si512();
                __m512i acc3 = _mm512_setzero_si512();

                for (int kk = 0; kk < K_up_padded; kk += 64) {
                    __m512i x_vec = _mm512_load_si512((__m512i*)(x_uint8 + kk));
                    acc0 = _mm512_dpbusd_epi32(acc0, x_vec, _mm512_loadu_si512((__m512i*)(w0 + kk)));
                    acc1 = _mm512_dpbusd_epi32(acc1, x_vec, _mm512_loadu_si512((__m512i*)(w1 + kk)));
                    acc2 = _mm512_dpbusd_epi32(acc2, x_vec, _mm512_loadu_si512((__m512i*)(w2 + kk)));
                    acc3 = _mm512_dpbusd_epi32(acc3, x_vec, _mm512_loadu_si512((__m512i*)(w3 + kk)));
                }

                up_float[n + 0] = static_cast<float>(_mm512_reduce_add_epi32(acc0) - 128 * w_up_sum[n + 0]) * x_scale;
                up_float[n + 1] = static_cast<float>(_mm512_reduce_add_epi32(acc1) - 128 * w_up_sum[n + 1]) * x_scale;
                up_float[n + 2] = static_cast<float>(_mm512_reduce_add_epi32(acc2) - 128 * w_up_sum[n + 2]) * x_scale;
                up_float[n + 3] = static_cast<float>(_mm512_reduce_add_epi32(acc3) - 128 * w_up_sum[n + 3]) * x_scale;
            }
            for (; n < mlp_hidden; n++) {
                const int8_t* w_row = w_up + n * K_up_padded;
                __m512i acc = _mm512_setzero_si512();
                for (int kk = 0; kk < K_up_padded; kk += 64) {
                    __m512i x_vec = _mm512_load_si512((__m512i*)(x_uint8 + kk));
                    acc = _mm512_dpbusd_epi32(acc, x_vec, _mm512_loadu_si512((__m512i*)(w_row + kk)));
                }
                up_float[n] = static_cast<float>(_mm512_reduce_add_epi32(acc) - 128 * w_up_sum[n]) * x_scale;
            }

            // Step 3: SwiGLU activation (vectorized)
            __m512 max_abs_vec = _mm512_setzero_ps();
            n = 0;
            for (; n + 15 < mlp_intermediate; n += 16) {
                __m512 x1 = _mm512_loadu_ps(up_float + n);
                __m512 x2 = _mm512_loadu_ps(up_float + n + mlp_intermediate);

                // sigmoid(x1) = 1 / (1 + exp(-x1))
                __m512 neg_x1 = _mm512_sub_ps(_mm512_setzero_ps(), x1);
                __m512 exp_neg = avx512_exp_fast(neg_x1);
                __m512 sigmoid = _mm512_div_ps(_mm512_set1_ps(1.0f), _mm512_add_ps(_mm512_set1_ps(1.0f), exp_neg));

                // SiLU(x1) = x1 * sigmoid(x1)
                __m512 silu = _mm512_mul_ps(x1, sigmoid);

                // result = SiLU(x1) * x2
                __m512 result = _mm512_mul_ps(silu, x2);
                _mm512_storeu_ps(swiglu_float + n, result);

                max_abs_vec = _mm512_max_ps(max_abs_vec, _mm512_abs_ps(result));
            }
            float max_abs_swiglu = _mm512_reduce_max_ps(max_abs_vec);
            for (; n < mlp_intermediate; n++) {
                float x1 = up_float[n];
                float x2 = up_float[n + mlp_intermediate];
                float sigmoid_x1 = 1.0f / (1.0f + std::exp(-x1));
                float silu_x1 = x1 * sigmoid_x1;
                swiglu_float[n] = silu_x1 * x2;
                max_abs_swiglu = std::max(max_abs_swiglu, std::abs(swiglu_float[n]));
            }

            // Quantize SwiGLU output
            float swiglu_scale = max_abs_swiglu / 127.0f;
            if (swiglu_scale == 0.0f) swiglu_scale = 1.0f;
            float inv_swiglu_scale = 1.0f / swiglu_scale;

            __m512 inv_scale_vec = _mm512_set1_ps(inv_swiglu_scale);
            __m512 min_val = _mm512_set1_ps(-127.0f);
            __m512 max_val = _mm512_set1_ps(127.0f);
            __m512i offset_128 = _mm512_set1_epi32(128);

            n = 0;
            for (; n + 15 < mlp_intermediate; n += 16) {
                __m512 vals = _mm512_loadu_ps(swiglu_float + n);
                vals = _mm512_mul_ps(vals, inv_scale_vec);
                vals = _mm512_max_ps(vals, min_val);
                vals = _mm512_min_ps(vals, max_val);
                vals = _mm512_roundscale_ps(vals, _MM_FROUND_TO_NEAREST_INT);
                __m512i int_vals = _mm512_cvtps_epi32(vals);
                int_vals = _mm512_add_epi32(int_vals, offset_128);
                // Pack to uint8
                __m128i packed = _mm512_cvtepi32_epi8(int_vals);
                _mm_storeu_si128((__m128i*)(swiglu_uint8 + n), packed);
            }
            for (; n < mlp_intermediate; n++) {
                float val = swiglu_float[n] * inv_swiglu_scale;
                val = std::max(-127.0f, std::min(127.0f, val));
                int8_t q = static_cast<int8_t>(std::round(val));
                swiglu_uint8[n] = static_cast<uint8_t>(static_cast<int16_t>(q) + 128);
            }
            for (n = mlp_intermediate; n < K_down_padded; n++) {
                swiglu_uint8[n] = 128;
            }

            // Step 4: Down projection with 4-way unrolling
            __m512 max_abs_down_vec = _mm512_setzero_ps();
            n = 0;
            for (; n + 3 < hidden_dim; n += 4) {
                const int8_t* w0 = w_down + (n + 0) * K_down_padded;
                const int8_t* w1 = w_down + (n + 1) * K_down_padded;
                const int8_t* w2 = w_down + (n + 2) * K_down_padded;
                const int8_t* w3 = w_down + (n + 3) * K_down_padded;

                __m512i acc0 = _mm512_setzero_si512();
                __m512i acc1 = _mm512_setzero_si512();
                __m512i acc2 = _mm512_setzero_si512();
                __m512i acc3 = _mm512_setzero_si512();

                for (int kk = 0; kk < K_down_padded; kk += 64) {
                    __m512i x_vec = _mm512_load_si512((__m512i*)(swiglu_uint8 + kk));
                    acc0 = _mm512_dpbusd_epi32(acc0, x_vec, _mm512_loadu_si512((__m512i*)(w0 + kk)));
                    acc1 = _mm512_dpbusd_epi32(acc1, x_vec, _mm512_loadu_si512((__m512i*)(w1 + kk)));
                    acc2 = _mm512_dpbusd_epi32(acc2, x_vec, _mm512_loadu_si512((__m512i*)(w2 + kk)));
                    acc3 = _mm512_dpbusd_epi32(acc3, x_vec, _mm512_loadu_si512((__m512i*)(w3 + kk)));
                }

                down_float[n + 0] = static_cast<float>(_mm512_reduce_add_epi32(acc0) - 128 * w_down_sum[n + 0]) * swiglu_scale;
                down_float[n + 1] = static_cast<float>(_mm512_reduce_add_epi32(acc1) - 128 * w_down_sum[n + 1]) * swiglu_scale;
                down_float[n + 2] = static_cast<float>(_mm512_reduce_add_epi32(acc2) - 128 * w_down_sum[n + 2]) * swiglu_scale;
                down_float[n + 3] = static_cast<float>(_mm512_reduce_add_epi32(acc3) - 128 * w_down_sum[n + 3]) * swiglu_scale;

                // Track max abs
                __m128 vals = _mm_set_ps(down_float[n+3], down_float[n+2], down_float[n+1], down_float[n+0]);
                __m128 abs_vals = _mm_andnot_ps(_mm_set1_ps(-0.0f), vals);
                // Horizontal max
                float max_of_4 = std::max(std::max(down_float[n+0], down_float[n+1]),
                                          std::max(down_float[n+2], down_float[n+3]));
                max_of_4 = std::abs(max_of_4);
            }
            float max_abs_down = 0.0f;
            for (int i = 0; i < hidden_dim; i++) {
                max_abs_down = std::max(max_abs_down, std::abs(down_float[i]));
            }
            for (; n < hidden_dim; n++) {
                const int8_t* w_row = w_down + n * K_down_padded;
                __m512i acc = _mm512_setzero_si512();
                for (int kk = 0; kk < K_down_padded; kk += 64) {
                    __m512i x_vec = _mm512_load_si512((__m512i*)(swiglu_uint8 + kk));
                    acc = _mm512_dpbusd_epi32(acc, x_vec, _mm512_loadu_si512((__m512i*)(w_row + kk)));
                }
                down_float[n] = static_cast<float>(_mm512_reduce_add_epi32(acc) - 128 * w_down_sum[n]) * swiglu_scale;
                max_abs_down = std::max(max_abs_down, std::abs(down_float[n]));
            }

            // Step 5: Quantize output (vectorized)
            float out_scale = max_abs_down / 127.0f;
            if (out_scale == 0.0f) out_scale = 1.0f;
            y_scales[m] = out_scale;

            float inv_out_scale = 1.0f / out_scale;
            inv_scale_vec = _mm512_set1_ps(inv_out_scale);

            n = 0;
            for (; n + 15 < hidden_dim; n += 16) {
                __m512 vals = _mm512_loadu_ps(down_float + n);
                vals = _mm512_mul_ps(vals, inv_scale_vec);
                vals = _mm512_max_ps(vals, min_val);
                vals = _mm512_min_ps(vals, max_val);
                vals = _mm512_roundscale_ps(vals, _MM_FROUND_TO_NEAREST_INT);
                __m512i int_vals = _mm512_cvtps_epi32(vals);
                __m128i packed = _mm512_cvtsepi32_epi8(int_vals);
                _mm_storeu_si128((__m128i*)(y_row + n), packed);
            }
            for (; n < hidden_dim; n++) {
                float val = down_float[n] * inv_out_scale;
                val = std::max(-127.0f, std::min(127.0f, val));
                y_row[n] = static_cast<int8_t>(std::round(val));
            }
        }

        // Free thread-local buffers
        free(x_uint8);
        free(up_float);
        free(swiglu_float);
        free(swiglu_uint8);
        free(down_float);
    }
}

/**
 * Softmax with int8 output - AVX-512 Vectorized
 *
 * Takes int32 input (from matmul accumulator) and produces int8 output for next layer.
 * Uses per-row scaling for quantization.
 *
 * Algorithm:
 * 1. Convert int32 to float using input scale
 * 2. Compute softmax: exp(x - max) / sum(exp(x - max))
 * 3. Quantize to int8 (softmax output is [0,1], so we scale by 127)
 *
 * AVX-512 optimizations:
 * - Vectorized int32->float conversion
 * - Vectorized max reduction
 * - Vectorized exp using fast approximation
 * - Vectorized sum reduction
 * - Vectorized quantization to int8
 */
void softmax_int8(
    torch::Tensor x_tensor,         // [M, N] int32 input
    torch::Tensor x_scale_tensor,   // [M] float32 input scales
    torch::Tensor y_tensor,         // [M, N] int8 output
    torch::Tensor y_scale_tensor,   // [M] float32 output scales
    int M, int N,
    int num_threads
) {
    const int32_t* __restrict__ x = x_tensor.data_ptr<int32_t>();
    const float* __restrict__ x_scales = x_scale_tensor.data_ptr<float>();
    int8_t* __restrict__ y = y_tensor.data_ptr<int8_t>();
    float* __restrict__ y_scales = y_scale_tensor.data_ptr<float>();

    omp_set_num_threads(num_threads);

    // Pad N to 16 for aligned AVX-512 operations
    const int N_padded = ((N + 15) / 16) * 16;

    #pragma omp parallel for schedule(static)
    for (int m = 0; m < M; m++) {
        const int32_t* x_row = x + m * N;
        int8_t* y_row = y + m * N;
        float x_scale = x_scales[m];
        __m512 scale_vec = _mm512_set1_ps(x_scale);

        // Allocate aligned buffer for float values
        float* float_vals = (float*)aligned_alloc(64, N_padded * sizeof(float));

        // Step 1: Convert int32 to float and find max (vectorized)
        __m512 max_vec = _mm512_set1_ps(-INFINITY);

        int n = 0;
        for (; n + 15 < N; n += 16) {
            __m512i int_vals = _mm512_loadu_si512((__m512i*)(x_row + n));
            __m512 float_v = _mm512_cvtepi32_ps(int_vals);
            float_v = _mm512_mul_ps(float_v, scale_vec);
            _mm512_store_ps(float_vals + n, float_v);
            max_vec = _mm512_max_ps(max_vec, float_v);
        }
        // Handle remainder
        float max_val = _mm512_reduce_max_ps(max_vec);
        for (; n < N; n++) {
            float val = static_cast<float>(x_row[n]) * x_scale;
            float_vals[n] = val;
            max_val = std::max(max_val, val);
        }
        // Pad with -inf so exp gives 0
        for (; n < N_padded; n++) {
            float_vals[n] = -INFINITY;
        }

        // Step 2: Compute exp(x - max) and sum (vectorized)
        __m512 max_broadcast = _mm512_set1_ps(max_val);
        __m512 sum_vec = _mm512_setzero_ps();

        n = 0;
        for (; n + 15 < N_padded; n += 16) {
            __m512 vals = _mm512_load_ps(float_vals + n);
            __m512 shifted = _mm512_sub_ps(vals, max_broadcast);
            __m512 exp_v = avx512_exp_fast(shifted);
            _mm512_store_ps(float_vals + n, exp_v);
            sum_vec = _mm512_add_ps(sum_vec, exp_v);
        }
        float sum = _mm512_reduce_add_ps(sum_vec);

        // Step 3: Normalize and quantize to int8 (vectorized)
        float inv_sum = 1.0f / sum;
        y_scales[m] = 1.0f / 127.0f;  // Output scale: int8 * scale = probability

        __m512 inv_sum_vec = _mm512_set1_ps(inv_sum);
        __m512 scale_127 = _mm512_set1_ps(127.0f);
        __m512 zero_vec = _mm512_setzero_ps();

        n = 0;
        for (; n + 15 < N; n += 16) {
            __m512 exp_v = _mm512_load_ps(float_vals + n);
            __m512 prob = _mm512_mul_ps(exp_v, inv_sum_vec);
            __m512 scaled = _mm512_mul_ps(prob, scale_127);
            // Clamp to [0, 127]
            scaled = _mm512_max_ps(scaled, zero_vec);
            scaled = _mm512_min_ps(scaled, scale_127);
            // Round and convert to int32
            scaled = _mm512_roundscale_ps(scaled, _MM_FROUND_TO_NEAREST_INT);
            __m512i int_vals = _mm512_cvtps_epi32(scaled);
            // Pack to int8 (saturating)
            __m128i packed = _mm512_cvtsepi32_epi8(int_vals);
            _mm_storeu_si128((__m128i*)(y_row + n), packed);
        }
        // Handle remainder
        for (; n < N; n++) {
            float prob = float_vals[n] * inv_sum;
            int8_t q_val = static_cast<int8_t>(std::min(127.0f, std::round(prob * 127.0f)));
            y_row[n] = q_val;
        }

        free(float_vals);
    }
}

/**
 * FUSION 1: Matmul + Softmax
 *
 * Fuses ternary matmul with softmax to avoid writing int32 intermediate.
 * Computes: y_int8 = softmax(x_int8 @ W_ternary)
 *
 * Memory savings: Eliminates M×N int32 write + read
 */
void matmul_softmax_int8(
    torch::Tensor x_int8_tensor,    // [M, K] int8 input
    torch::Tensor x_scale_tensor,   // [M] float32 input scales
    torch::Tensor w_int8_tensor,    // [N, K_padded] int8 ternary weights
    torch::Tensor w_sum_tensor,     // [N] int32 weight sums
    torch::Tensor y_tensor,         // [M, N] int8 output (softmax applied)
    torch::Tensor y_scale_tensor,   // [M] float32 output scales
    int M, int N, int K,
    int num_threads
) {
    const int8_t* __restrict__ x_int8 = x_int8_tensor.data_ptr<int8_t>();
    const float* __restrict__ x_scales = x_scale_tensor.data_ptr<float>();
    const int8_t* __restrict__ w_int8 = w_int8_tensor.data_ptr<int8_t>();
    const int32_t* __restrict__ w_sum = w_sum_tensor.data_ptr<int32_t>();
    int8_t* __restrict__ y = y_tensor.data_ptr<int8_t>();
    float* __restrict__ y_scales = y_scale_tensor.data_ptr<float>();

    const int K_padded = ((K + 63) / 64) * 64;
    const int N_padded = ((N + 15) / 16) * 16;

    omp_set_num_threads(num_threads);

    #pragma omp parallel
    {
        // Thread-local buffers
        uint8_t* x_uint8 = (uint8_t*)aligned_alloc(64, K_padded);
        float* matmul_out = (float*)aligned_alloc(64, N_padded * sizeof(float));

        #pragma omp for schedule(static)
        for (int m = 0; m < M; m++) {
            const int8_t* x_row = x_int8 + m * K;
            float x_scale = x_scales[m];
            int8_t* y_row = y + m * N;

            // Convert input to uint8 (for VNNI)
            const __m512i offset_vec = _mm512_set1_epi8((char)128);
            int k = 0;
            for (; k + 63 < K; k += 64) {
                __m512i x_vec = _mm512_loadu_si512((__m512i*)(x_row + k));
                __m512i x_u8 = _mm512_add_epi8(x_vec, offset_vec);
                _mm512_store_si512((__m512i*)(x_uint8 + k), x_u8);
            }
            for (; k < K; k++) {
                x_uint8[k] = static_cast<uint8_t>(static_cast<int16_t>(x_row[k]) + 128);
            }
            for (; k < K_padded; k++) {
                x_uint8[k] = 128;
            }

            // Compute matmul and convert to float in one pass
            __m512 max_vec = _mm512_set1_ps(-INFINITY);

            int n = 0;
            for (; n + 3 < N; n += 4) {
                const int8_t* w0 = w_int8 + (n + 0) * K_padded;
                const int8_t* w1 = w_int8 + (n + 1) * K_padded;
                const int8_t* w2 = w_int8 + (n + 2) * K_padded;
                const int8_t* w3 = w_int8 + (n + 3) * K_padded;

                __m512i acc0 = _mm512_setzero_si512();
                __m512i acc1 = _mm512_setzero_si512();
                __m512i acc2 = _mm512_setzero_si512();
                __m512i acc3 = _mm512_setzero_si512();

                for (int kk = 0; kk < K_padded; kk += 64) {
                    __m512i x_vec = _mm512_load_si512((__m512i*)(x_uint8 + kk));
                    acc0 = _mm512_dpbusd_epi32(acc0, x_vec, _mm512_loadu_si512((__m512i*)(w0 + kk)));
                    acc1 = _mm512_dpbusd_epi32(acc1, x_vec, _mm512_loadu_si512((__m512i*)(w1 + kk)));
                    acc2 = _mm512_dpbusd_epi32(acc2, x_vec, _mm512_loadu_si512((__m512i*)(w2 + kk)));
                    acc3 = _mm512_dpbusd_epi32(acc3, x_vec, _mm512_loadu_si512((__m512i*)(w3 + kk)));
                }

                // Convert to float immediately (no int32 write to memory)
                float v0 = static_cast<float>(_mm512_reduce_add_epi32(acc0) - 128 * w_sum[n+0]) * x_scale;
                float v1 = static_cast<float>(_mm512_reduce_add_epi32(acc1) - 128 * w_sum[n+1]) * x_scale;
                float v2 = static_cast<float>(_mm512_reduce_add_epi32(acc2) - 128 * w_sum[n+2]) * x_scale;
                float v3 = static_cast<float>(_mm512_reduce_add_epi32(acc3) - 128 * w_sum[n+3]) * x_scale;

                matmul_out[n+0] = v0;
                matmul_out[n+1] = v1;
                matmul_out[n+2] = v2;
                matmul_out[n+3] = v3;

                // Track max for softmax
                __m128 vals = _mm_set_ps(v3, v2, v1, v0);
                max_vec = _mm512_max_ps(max_vec, _mm512_castps128_ps512(vals));
            }
            for (; n < N; n++) {
                const int8_t* w_row = w_int8 + n * K_padded;
                __m512i acc = _mm512_setzero_si512();
                for (int kk = 0; kk < K_padded; kk += 64) {
                    __m512i x_vec = _mm512_load_si512((__m512i*)(x_uint8 + kk));
                    acc = _mm512_dpbusd_epi32(acc, x_vec, _mm512_loadu_si512((__m512i*)(w_row + kk)));
                }
                float v = static_cast<float>(_mm512_reduce_add_epi32(acc) - 128 * w_sum[n]) * x_scale;
                matmul_out[n] = v;
            }
            // Pad with -inf
            for (; n < N_padded; n++) {
                matmul_out[n] = -INFINITY;
            }

            // Find max
            float max_val = -INFINITY;
            for (int i = 0; i < N; i++) {
                max_val = std::max(max_val, matmul_out[i]);
            }

            // Compute exp(x - max) and sum
            __m512 sum_vec = _mm512_setzero_ps();
            __m512 max_broadcast = _mm512_set1_ps(max_val);
            n = 0;
            for (; n + 15 < N_padded; n += 16) {
                __m512 vals = _mm512_load_ps(matmul_out + n);
                __m512 shifted = _mm512_sub_ps(vals, max_broadcast);
                __m512 exp_v = avx512_exp_fast(shifted);
                _mm512_store_ps(matmul_out + n, exp_v);
                sum_vec = _mm512_add_ps(sum_vec, exp_v);
            }
            float sum = _mm512_reduce_add_ps(sum_vec);

            // Normalize and quantize to int8
            float inv_sum = 1.0f / sum;
            y_scales[m] = 1.0f / 127.0f;

            __m512 inv_sum_vec = _mm512_set1_ps(inv_sum);
            __m512 scale_127 = _mm512_set1_ps(127.0f);
            __m512 zero_vec = _mm512_setzero_ps();
            __m512 max_127 = _mm512_set1_ps(127.0f);

            n = 0;
            for (; n + 15 < N; n += 16) {
                __m512 exp_v = _mm512_load_ps(matmul_out + n);
                __m512 prob = _mm512_mul_ps(exp_v, inv_sum_vec);
                __m512 scaled = _mm512_mul_ps(prob, scale_127);
                scaled = _mm512_max_ps(scaled, zero_vec);
                scaled = _mm512_min_ps(scaled, max_127);
                scaled = _mm512_roundscale_ps(scaled, _MM_FROUND_TO_NEAREST_INT);
                __m512i int_vals = _mm512_cvtps_epi32(scaled);
                __m128i packed = _mm512_cvtsepi32_epi8(int_vals);
                _mm_storeu_si128((__m128i*)(y_row + n), packed);
            }
            for (; n < N; n++) {
                float prob = matmul_out[n] * inv_sum;
                int8_t q_val = static_cast<int8_t>(std::min(127.0f, std::round(prob * 127.0f)));
                y_row[n] = q_val;
            }
        }

        free(x_uint8);
        free(matmul_out);
    }
}

/**
 * FUSION 2: QKV Projections (read input once)
 *
 * Computes Q, K, V projections together, reading input x only once.
 * Includes softmax on each output.
 *
 * Memory savings: Read x once instead of 3 times
 */
void matmul_qkv_softmax_int8(
    torch::Tensor x_int8_tensor,    // [M, K] int8 input
    torch::Tensor x_scale_tensor,   // [M] float32 input scales
    torch::Tensor wq_tensor,        // [N, K_padded] int8
    torch::Tensor wq_sum_tensor,    // [N] int32
    torch::Tensor wk_tensor,        // [N, K_padded] int8
    torch::Tensor wk_sum_tensor,    // [N] int32
    torch::Tensor wv_tensor,        // [N, K_padded] int8
    torch::Tensor wv_sum_tensor,    // [N] int32
    torch::Tensor q_tensor,         // [M, N] int8 output
    torch::Tensor q_scale_tensor,   // [M] float32
    torch::Tensor k_tensor,         // [M, N] int8 output
    torch::Tensor k_scale_tensor,   // [M] float32
    torch::Tensor v_tensor,         // [M, N] int8 output
    torch::Tensor v_scale_tensor,   // [M] float32
    int M, int N, int K,
    int num_threads
) {
    const int8_t* __restrict__ x_int8 = x_int8_tensor.data_ptr<int8_t>();
    const float* __restrict__ x_scales = x_scale_tensor.data_ptr<float>();
    const int8_t* __restrict__ wq = wq_tensor.data_ptr<int8_t>();
    const int32_t* __restrict__ wq_sum = wq_sum_tensor.data_ptr<int32_t>();
    const int8_t* __restrict__ wk = wk_tensor.data_ptr<int8_t>();
    const int32_t* __restrict__ wk_sum = wk_sum_tensor.data_ptr<int32_t>();
    const int8_t* __restrict__ wv = wv_tensor.data_ptr<int8_t>();
    const int32_t* __restrict__ wv_sum = wv_sum_tensor.data_ptr<int32_t>();
    int8_t* __restrict__ q_out = q_tensor.data_ptr<int8_t>();
    float* __restrict__ q_scales = q_scale_tensor.data_ptr<float>();
    int8_t* __restrict__ k_out = k_tensor.data_ptr<int8_t>();
    float* __restrict__ k_scales = k_scale_tensor.data_ptr<float>();
    int8_t* __restrict__ v_out = v_tensor.data_ptr<int8_t>();
    float* __restrict__ v_scales = v_scale_tensor.data_ptr<float>();

    const int K_padded = ((K + 63) / 64) * 64;
    const int N_padded = ((N + 15) / 16) * 16;

    omp_set_num_threads(num_threads);

    #pragma omp parallel
    {
        // Thread-local buffers
        uint8_t* x_uint8 = (uint8_t*)aligned_alloc(64, K_padded);
        float* q_float = (float*)aligned_alloc(64, N_padded * sizeof(float));
        float* k_float = (float*)aligned_alloc(64, N_padded * sizeof(float));
        float* v_float = (float*)aligned_alloc(64, N_padded * sizeof(float));

        #pragma omp for schedule(static)
        for (int m = 0; m < M; m++) {
            const int8_t* x_row = x_int8 + m * K;
            float x_scale = x_scales[m];

            // Convert input to uint8 ONCE (shared for Q, K, V)
            const __m512i offset_vec = _mm512_set1_epi8((char)128);
            int k = 0;
            for (; k + 63 < K; k += 64) {
                __m512i x_vec = _mm512_loadu_si512((__m512i*)(x_row + k));
                __m512i x_u8 = _mm512_add_epi8(x_vec, offset_vec);
                _mm512_store_si512((__m512i*)(x_uint8 + k), x_u8);
            }
            for (; k < K; k++) {
                x_uint8[k] = static_cast<uint8_t>(static_cast<int16_t>(x_row[k]) + 128);
            }
            for (; k < K_padded; k++) {
                x_uint8[k] = 128;
            }

            // Compute Q, K, V projections together
            for (int n = 0; n < N; n++) {
                const int8_t* wq_row = wq + n * K_padded;
                const int8_t* wk_row = wk + n * K_padded;
                const int8_t* wv_row = wv + n * K_padded;

                __m512i acc_q = _mm512_setzero_si512();
                __m512i acc_k = _mm512_setzero_si512();
                __m512i acc_v = _mm512_setzero_si512();

                for (int kk = 0; kk < K_padded; kk += 64) {
                    __m512i x_vec = _mm512_load_si512((__m512i*)(x_uint8 + kk));
                    acc_q = _mm512_dpbusd_epi32(acc_q, x_vec, _mm512_loadu_si512((__m512i*)(wq_row + kk)));
                    acc_k = _mm512_dpbusd_epi32(acc_k, x_vec, _mm512_loadu_si512((__m512i*)(wk_row + kk)));
                    acc_v = _mm512_dpbusd_epi32(acc_v, x_vec, _mm512_loadu_si512((__m512i*)(wv_row + kk)));
                }

                q_float[n] = static_cast<float>(_mm512_reduce_add_epi32(acc_q) - 128 * wq_sum[n]) * x_scale;
                k_float[n] = static_cast<float>(_mm512_reduce_add_epi32(acc_k) - 128 * wk_sum[n]) * x_scale;
                v_float[n] = static_cast<float>(_mm512_reduce_add_epi32(acc_v) - 128 * wv_sum[n]) * x_scale;
            }
            // Pad
            for (int n = N; n < N_padded; n++) {
                q_float[n] = -INFINITY;
                k_float[n] = -INFINITY;
                v_float[n] = -INFINITY;
            }

            // Apply softmax to each (Q, K, V)
            auto apply_softmax = [&](float* data, int8_t* out, float* out_scale) {
                // Find max
                float max_val = -INFINITY;
                for (int i = 0; i < N; i++) {
                    max_val = std::max(max_val, data[i]);
                }

                // Compute exp and sum
                float sum = 0.0f;
                __m512 max_vec = _mm512_set1_ps(max_val);
                __m512 sum_vec = _mm512_setzero_ps();

                int n = 0;
                for (; n + 15 < N_padded; n += 16) {
                    __m512 vals = _mm512_load_ps(data + n);
                    __m512 shifted = _mm512_sub_ps(vals, max_vec);
                    __m512 exp_v = avx512_exp_fast(shifted);
                    _mm512_store_ps(data + n, exp_v);
                    sum_vec = _mm512_add_ps(sum_vec, exp_v);
                }
                sum = _mm512_reduce_add_ps(sum_vec);

                // Normalize and quantize
                float inv_sum = 1.0f / sum;
                out_scale[m] = 1.0f / 127.0f;

                __m512 inv_sum_vec = _mm512_set1_ps(inv_sum);
                __m512 scale_127 = _mm512_set1_ps(127.0f);
                __m512 zero_vec = _mm512_setzero_ps();
                __m512 max_127 = _mm512_set1_ps(127.0f);

                n = 0;
                for (; n + 15 < N; n += 16) {
                    __m512 exp_v = _mm512_load_ps(data + n);
                    __m512 prob = _mm512_mul_ps(exp_v, inv_sum_vec);
                    __m512 scaled = _mm512_mul_ps(prob, scale_127);
                    scaled = _mm512_max_ps(scaled, zero_vec);
                    scaled = _mm512_min_ps(scaled, max_127);
                    scaled = _mm512_roundscale_ps(scaled, _MM_FROUND_TO_NEAREST_INT);
                    __m512i int_vals = _mm512_cvtps_epi32(scaled);
                    __m128i packed = _mm512_cvtsepi32_epi8(int_vals);
                    _mm_storeu_si128((__m128i*)(out + m * N + n), packed);
                }
                for (; n < N; n++) {
                    float prob = data[n] * inv_sum;
                    out[m * N + n] = static_cast<int8_t>(std::min(127.0f, std::round(prob * 127.0f)));
                }
            };

            apply_softmax(q_float, q_out, q_scales);
            apply_softmax(k_float, k_out, k_scales);
            apply_softmax(v_float, v_out, v_scales);
        }

        free(x_uint8);
        free(q_float);
        free(k_float);
        free(v_float);
    }
}

/**
 * FUSION 3: Matmul + Quantize
 *
 * Fuses ternary matmul with quantization to avoid writing int32 intermediate.
 * Computes: y_int8 = quantize(x_int8 @ W_ternary)
 *
 * Different from matmul_softmax - this does simple per-row quantization
 * (find max abs, scale to [-127, 127])
 */
void matmul_quantize_int8(
    torch::Tensor x_int8_tensor,    // [M, K] int8 input
    torch::Tensor x_scale_tensor,   // [M] float32 input scales
    torch::Tensor w_int8_tensor,    // [N, K_padded] int8 ternary weights
    torch::Tensor w_sum_tensor,     // [N] int32 weight sums
    torch::Tensor y_tensor,         // [M, N] int8 output
    torch::Tensor y_scale_tensor,   // [M] float32 output scales
    int M, int N, int K,
    int num_threads
) {
    const int8_t* __restrict__ x_int8 = x_int8_tensor.data_ptr<int8_t>();
    const float* __restrict__ x_scales = x_scale_tensor.data_ptr<float>();
    const int8_t* __restrict__ w_int8 = w_int8_tensor.data_ptr<int8_t>();
    const int32_t* __restrict__ w_sum = w_sum_tensor.data_ptr<int32_t>();
    int8_t* __restrict__ y = y_tensor.data_ptr<int8_t>();
    float* __restrict__ y_scales = y_scale_tensor.data_ptr<float>();

    const int K_padded = ((K + 63) / 64) * 64;

    omp_set_num_threads(num_threads);

    #pragma omp parallel
    {
        uint8_t* x_uint8 = (uint8_t*)aligned_alloc(64, K_padded);
        float* matmul_out = (float*)aligned_alloc(64, ((N + 15) / 16) * 16 * sizeof(float));

        #pragma omp for schedule(static)
        for (int m = 0; m < M; m++) {
            const int8_t* x_row = x_int8 + m * K;
            float x_scale = x_scales[m];
            int8_t* y_row = y + m * N;

            // Convert input to uint8
            const __m512i offset_vec = _mm512_set1_epi8((char)128);
            int k = 0;
            for (; k + 63 < K; k += 64) {
                __m512i x_vec = _mm512_loadu_si512((__m512i*)(x_row + k));
                __m512i x_u8 = _mm512_add_epi8(x_vec, offset_vec);
                _mm512_store_si512((__m512i*)(x_uint8 + k), x_u8);
            }
            for (; k < K; k++) {
                x_uint8[k] = static_cast<uint8_t>(static_cast<int16_t>(x_row[k]) + 128);
            }
            for (; k < K_padded; k++) {
                x_uint8[k] = 128;
            }

            // Compute matmul and find max abs in one pass
            float max_abs = 0.0f;

            int n = 0;
            for (; n + 3 < N; n += 4) {
                const int8_t* w0 = w_int8 + (n + 0) * K_padded;
                const int8_t* w1 = w_int8 + (n + 1) * K_padded;
                const int8_t* w2 = w_int8 + (n + 2) * K_padded;
                const int8_t* w3 = w_int8 + (n + 3) * K_padded;

                __m512i acc0 = _mm512_setzero_si512();
                __m512i acc1 = _mm512_setzero_si512();
                __m512i acc2 = _mm512_setzero_si512();
                __m512i acc3 = _mm512_setzero_si512();

                for (int kk = 0; kk < K_padded; kk += 64) {
                    __m512i x_vec = _mm512_load_si512((__m512i*)(x_uint8 + kk));
                    acc0 = _mm512_dpbusd_epi32(acc0, x_vec, _mm512_loadu_si512((__m512i*)(w0 + kk)));
                    acc1 = _mm512_dpbusd_epi32(acc1, x_vec, _mm512_loadu_si512((__m512i*)(w1 + kk)));
                    acc2 = _mm512_dpbusd_epi32(acc2, x_vec, _mm512_loadu_si512((__m512i*)(w2 + kk)));
                    acc3 = _mm512_dpbusd_epi32(acc3, x_vec, _mm512_loadu_si512((__m512i*)(w3 + kk)));
                }

                float v0 = static_cast<float>(_mm512_reduce_add_epi32(acc0) - 128 * w_sum[n+0]) * x_scale;
                float v1 = static_cast<float>(_mm512_reduce_add_epi32(acc1) - 128 * w_sum[n+1]) * x_scale;
                float v2 = static_cast<float>(_mm512_reduce_add_epi32(acc2) - 128 * w_sum[n+2]) * x_scale;
                float v3 = static_cast<float>(_mm512_reduce_add_epi32(acc3) - 128 * w_sum[n+3]) * x_scale;

                matmul_out[n+0] = v0;
                matmul_out[n+1] = v1;
                matmul_out[n+2] = v2;
                matmul_out[n+3] = v3;

                max_abs = std::max(max_abs, std::abs(v0));
                max_abs = std::max(max_abs, std::abs(v1));
                max_abs = std::max(max_abs, std::abs(v2));
                max_abs = std::max(max_abs, std::abs(v3));
            }
            for (; n < N; n++) {
                const int8_t* w_row = w_int8 + n * K_padded;
                __m512i acc = _mm512_setzero_si512();
                for (int kk = 0; kk < K_padded; kk += 64) {
                    __m512i x_vec = _mm512_load_si512((__m512i*)(x_uint8 + kk));
                    acc = _mm512_dpbusd_epi32(acc, x_vec, _mm512_loadu_si512((__m512i*)(w_row + kk)));
                }
                float v = static_cast<float>(_mm512_reduce_add_epi32(acc) - 128 * w_sum[n]) * x_scale;
                matmul_out[n] = v;
                max_abs = std::max(max_abs, std::abs(v));
            }

            // Quantize
            float out_scale = max_abs / 127.0f;
            if (out_scale == 0.0f) out_scale = 1.0f;
            y_scales[m] = out_scale;

            float inv_scale = 1.0f / out_scale;
            __m512 inv_scale_vec = _mm512_set1_ps(inv_scale);
            __m512 min_val = _mm512_set1_ps(-127.0f);
            __m512 max_val = _mm512_set1_ps(127.0f);

            n = 0;
            for (; n + 15 < N; n += 16) {
                __m512 vals = _mm512_loadu_ps(matmul_out + n);
                vals = _mm512_mul_ps(vals, inv_scale_vec);
                vals = _mm512_max_ps(vals, min_val);
                vals = _mm512_min_ps(vals, max_val);
                vals = _mm512_roundscale_ps(vals, _MM_FROUND_TO_NEAREST_INT);
                __m512i int_vals = _mm512_cvtps_epi32(vals);
                __m128i packed = _mm512_cvtsepi32_epi8(int_vals);
                _mm_storeu_si128((__m128i*)(y_row + n), packed);
            }
            for (; n < N; n++) {
                float val = matmul_out[n] * inv_scale;
                val = std::max(-127.0f, std::min(127.0f, val));
                y_row[n] = static_cast<int8_t>(std::round(val));
            }
        }

        free(x_uint8);
        free(matmul_out);
    }
}

/**
 * Fast AVX-512 sigmoid approximation
 * sigmoid(x) = 1 / (1 + exp(-x))
 */
inline __m512 avx512_sigmoid_fast(__m512 x) {
    // sigmoid(x) = 1 / (1 + exp(-x))
    __m512 neg_x = _mm512_sub_ps(_mm512_setzero_ps(), x);
    __m512 exp_neg_x = avx512_exp_fast(neg_x);
    __m512 one = _mm512_set1_ps(1.0f);
    __m512 denom = _mm512_add_ps(one, exp_neg_x);
    return _mm512_div_ps(one, denom);
}

/**
 * SwiGLU activation with int8 output - AVX-512 Vectorized
 *
 * SwiGLU(x) = SiLU(x1) * x2 where x is split in half
 * SiLU(x) = x * sigmoid(x) = x / (1 + exp(-x))
 *
 * Takes int32 input [M, N] where N is the full width (e.g., 16384)
 * Outputs int8 [M, N/2] after SwiGLU (e.g., 8192)
 *
 * AVX-512 optimizations:
 * - Vectorized int32->float conversion
 * - Vectorized sigmoid using fast exp approximation
 * - Vectorized SiLU and multiply
 * - Vectorized max abs reduction
 * - Vectorized quantization to int8
 */
void swiglu_int8(
    torch::Tensor x_tensor,         // [M, N] int32 input (N = 2 * hidden)
    torch::Tensor x_scale_tensor,   // [M] float32 input scales
    torch::Tensor y_tensor,         // [M, N/2] int8 output
    torch::Tensor y_scale_tensor,   // [M] float32 output scales
    int M, int N,
    int num_threads
) {
    const int32_t* __restrict__ x = x_tensor.data_ptr<int32_t>();
    const float* __restrict__ x_scales = x_scale_tensor.data_ptr<float>();
    int8_t* __restrict__ y = y_tensor.data_ptr<int8_t>();
    float* __restrict__ y_scales = y_scale_tensor.data_ptr<float>();

    const int half_N = N / 2;
    const int half_N_padded = ((half_N + 15) / 16) * 16;

    omp_set_num_threads(num_threads);

    #pragma omp parallel for schedule(static)
    for (int m = 0; m < M; m++) {
        const int32_t* x_row = x + m * N;
        int8_t* y_row = y + m * half_N;
        float x_scale = x_scales[m];
        __m512 scale_vec = _mm512_set1_ps(x_scale);

        // Allocate aligned buffer for results
        float* result = (float*)aligned_alloc(64, half_N_padded * sizeof(float));

        // First pass: compute SwiGLU and find max for quantization (vectorized)
        __m512 max_abs_vec = _mm512_setzero_ps();

        int n = 0;
        for (; n + 15 < half_N; n += 16) {
            // Load x1 (first half) and x2 (second half)
            __m512i x1_int = _mm512_loadu_si512((__m512i*)(x_row + n));
            __m512i x2_int = _mm512_loadu_si512((__m512i*)(x_row + n + half_N));

            // Convert to float
            __m512 x1 = _mm512_mul_ps(_mm512_cvtepi32_ps(x1_int), scale_vec);
            __m512 x2 = _mm512_mul_ps(_mm512_cvtepi32_ps(x2_int), scale_vec);

            // SiLU(x1) = x1 * sigmoid(x1)
            __m512 sigmoid_x1 = avx512_sigmoid_fast(x1);
            __m512 silu_x1 = _mm512_mul_ps(x1, sigmoid_x1);

            // SwiGLU = SiLU(x1) * x2
            __m512 swiglu_result = _mm512_mul_ps(silu_x1, x2);
            _mm512_store_ps(result + n, swiglu_result);

            // Track max abs
            __m512 abs_result = _mm512_abs_ps(swiglu_result);
            max_abs_vec = _mm512_max_ps(max_abs_vec, abs_result);
        }

        // Handle remainder (scalar)
        float max_abs = _mm512_reduce_max_ps(max_abs_vec);
        for (; n < half_N; n++) {
            float x1 = static_cast<float>(x_row[n]) * x_scale;
            float x2 = static_cast<float>(x_row[n + half_N]) * x_scale;

            float sigmoid_x1 = 1.0f / (1.0f + std::exp(-x1));
            float silu_x1 = x1 * sigmoid_x1;
            result[n] = silu_x1 * x2;

            max_abs = std::max(max_abs, std::abs(result[n]));
        }

        // Quantize to int8 (vectorized)
        float y_scale = max_abs / 127.0f;
        if (y_scale == 0.0f) y_scale = 1.0f;
        y_scales[m] = y_scale;

        __m512 inv_scale_vec = _mm512_set1_ps(1.0f / y_scale);
        __m512 min_val = _mm512_set1_ps(-127.0f);
        __m512 max_val = _mm512_set1_ps(127.0f);

        n = 0;
        for (; n + 15 < half_N; n += 16) {
            __m512 vals = _mm512_load_ps(result + n);
            vals = _mm512_mul_ps(vals, inv_scale_vec);
            // Clamp to [-127, 127]
            vals = _mm512_max_ps(vals, min_val);
            vals = _mm512_min_ps(vals, max_val);
            // Round and convert to int32
            vals = _mm512_roundscale_ps(vals, _MM_FROUND_TO_NEAREST_INT);
            __m512i int_vals = _mm512_cvtps_epi32(vals);
            // Pack to int8 (saturating)
            __m128i packed = _mm512_cvtsepi32_epi8(int_vals);
            _mm_storeu_si128((__m128i*)(y_row + n), packed);
        }

        // Handle remainder (scalar)
        float inv_scale = 1.0f / y_scale;
        for (; n < half_N; n++) {
            float val = result[n] * inv_scale;
            val = std::max(-127.0f, std::min(127.0f, val));
            y_row[n] = static_cast<int8_t>(std::round(val));
        }

        free(result);
    }
}

/**
 * AVX-512 Vectorized Multi-Head Attention
 *
 * Computes: Attention(Q, K, V) = softmax(Q @ K^T / sqrt(d_k)) @ V
 *
 * Fully vectorized with AVX-512:
 * - Q @ K^T uses vectorized dot products
 * - Softmax uses fast exp approximation
 * - @V uses FMA operations
 */
void attention_int8(
    torch::Tensor q_tensor,         // [M, num_heads * head_dim] int8
    torch::Tensor k_tensor,         // [M, num_heads * head_dim] int8
    torch::Tensor v_tensor,         // [M, num_heads * head_dim] int8
    torch::Tensor q_scale_tensor,   // [M] float32
    torch::Tensor k_scale_tensor,   // [M] float32
    torch::Tensor v_scale_tensor,   // [M] float32
    torch::Tensor out_tensor,       // [M, num_heads * head_dim] int8 output
    torch::Tensor out_scale_tensor, // [M] float32 output scales
    int M, int num_heads, int head_dim,
    float scale,                    // 1/sqrt(head_dim)
    int num_threads
) {
    const int8_t* __restrict__ q = q_tensor.data_ptr<int8_t>();
    const int8_t* __restrict__ k = k_tensor.data_ptr<int8_t>();
    const int8_t* __restrict__ v = v_tensor.data_ptr<int8_t>();
    const float* __restrict__ q_scales = q_scale_tensor.data_ptr<float>();
    const float* __restrict__ k_scales = k_scale_tensor.data_ptr<float>();
    const float* __restrict__ v_scales = v_scale_tensor.data_ptr<float>();
    int8_t* __restrict__ out = out_tensor.data_ptr<int8_t>();
    float* __restrict__ out_scales = out_scale_tensor.data_ptr<float>();

    const int hidden_dim = num_heads * head_dim;
    const int head_dim_padded = ((head_dim + 63) / 64) * 64;

    omp_set_num_threads(num_threads);

    #pragma omp parallel for schedule(static)
    for (int m = 0; m < M; m++) {
        const int8_t* q_row = q + m * hidden_dim;
        float q_scale = q_scales[m];
        int8_t* out_row = out + m * hidden_dim;

        // Allocate per-thread buffers (aligned for AVX-512)
        float* attn_scores = (float*)aligned_alloc(64, ((M * num_heads + 15) / 16) * 16 * sizeof(float));
        float* attn_probs = (float*)aligned_alloc(64, ((M * num_heads + 15) / 16) * 16 * sizeof(float));
        float* attn_out = (float*)aligned_alloc(64, ((hidden_dim + 15) / 16) * 16 * sizeof(float));

        // For each head
        for (int h = 0; h < num_heads; h++) {
            const int8_t* q_head = q_row + h * head_dim;

            // ============================================================
            // Step 1: Q @ K^T with AVX-512 vectorized dot product
            // ============================================================
            for (int n = 0; n < M; n++) {
                const int8_t* k_row = k + n * hidden_dim;
                const int8_t* k_head = k_row + h * head_dim;
                float k_scale = k_scales[n];

                // Vectorized dot product using AVX-512
                __m512i acc = _mm512_setzero_si512();

                int d = 0;
                // Process 64 elements at a time using VNNI-style computation
                for (; d + 63 < head_dim; d += 64) {
                    // Load Q and K as int8
                    __m512i q_vec = _mm512_loadu_si512((__m512i*)(q_head + d));
                    __m512i k_vec = _mm512_loadu_si512((__m512i*)(k_head + d));

                    // Convert to int16 and multiply (avoiding overflow)
                    // Split into low and high 256-bit parts
                    __m256i q_lo = _mm512_extracti32x8_epi32(q_vec, 0);
                    __m256i q_hi = _mm512_extracti32x8_epi32(q_vec, 1);
                    __m256i k_lo = _mm512_extracti32x8_epi32(k_vec, 0);
                    __m256i k_hi = _mm512_extracti32x8_epi32(k_vec, 1);

                    // Extend to int16
                    __m512i q_lo_16 = _mm512_cvtepi8_epi16(q_lo);
                    __m512i q_hi_16 = _mm512_cvtepi8_epi16(q_hi);
                    __m512i k_lo_16 = _mm512_cvtepi8_epi16(k_lo);
                    __m512i k_hi_16 = _mm512_cvtepi8_epi16(k_hi);

                    // Multiply and accumulate
                    __m512i prod_lo = _mm512_madd_epi16(q_lo_16, k_lo_16);
                    __m512i prod_hi = _mm512_madd_epi16(q_hi_16, k_hi_16);

                    acc = _mm512_add_epi32(acc, prod_lo);
                    acc = _mm512_add_epi32(acc, prod_hi);
                }

                // Horizontal sum
                int32_t dot = _mm512_reduce_add_epi32(acc);

                // Handle remainder
                for (; d < head_dim; d++) {
                    dot += static_cast<int32_t>(q_head[d]) * static_cast<int32_t>(k_head[d]);
                }

                // Convert to float with scaling
                float score = static_cast<float>(dot) * q_scale * k_scale * scale;
                attn_scores[n * num_heads + h] = score;
            }

            // ============================================================
            // Step 2: Softmax with AVX-512 vectorized exp
            // ============================================================

            // Find max (vectorized)
            __m512 max_vec = _mm512_set1_ps(-INFINITY);
            int n = 0;
            for (; n + 15 < M; n += 16) {
                // Gather scores for this head across 16 positions
                // Note: scores are interleaved by num_heads
                __m512 scores;
                alignas(64) float temp[16];
                for (int i = 0; i < 16; i++) {
                    temp[i] = attn_scores[(n + i) * num_heads + h];
                }
                scores = _mm512_load_ps(temp);
                max_vec = _mm512_max_ps(max_vec, scores);
            }
            float max_score = _mm512_reduce_max_ps(max_vec);
            for (; n < M; n++) {
                max_score = std::max(max_score, attn_scores[n * num_heads + h]);
            }

            // Compute exp and sum (vectorized)
            __m512 sum_vec = _mm512_setzero_ps();
            __m512 max_broadcast = _mm512_set1_ps(max_score);

            n = 0;
            for (; n + 15 < M; n += 16) {
                alignas(64) float temp[16];
                for (int i = 0; i < 16; i++) {
                    temp[i] = attn_scores[(n + i) * num_heads + h];
                }
                __m512 scores = _mm512_load_ps(temp);
                __m512 shifted = _mm512_sub_ps(scores, max_broadcast);
                __m512 exp_val = avx512_exp_fast(shifted);
                _mm512_store_ps(temp, exp_val);
                for (int i = 0; i < 16; i++) {
                    attn_probs[(n + i) * num_heads + h] = temp[i];
                }
                sum_vec = _mm512_add_ps(sum_vec, exp_val);
            }
            float sum_exp = _mm512_reduce_add_ps(sum_vec);
            for (; n < M; n++) {
                float exp_val = std::exp(attn_scores[n * num_heads + h] - max_score);
                attn_probs[n * num_heads + h] = exp_val;
                sum_exp += exp_val;
            }

            // Normalize
            float inv_sum = 1.0f / sum_exp;
            __m512 inv_sum_vec = _mm512_set1_ps(inv_sum);
            n = 0;
            for (; n + 15 < M; n += 16) {
                alignas(64) float temp[16];
                for (int i = 0; i < 16; i++) {
                    temp[i] = attn_probs[(n + i) * num_heads + h];
                }
                __m512 probs = _mm512_load_ps(temp);
                probs = _mm512_mul_ps(probs, inv_sum_vec);
                _mm512_store_ps(temp, probs);
                for (int i = 0; i < 16; i++) {
                    attn_probs[(n + i) * num_heads + h] = temp[i];
                }
            }
            for (; n < M; n++) {
                attn_probs[n * num_heads + h] *= inv_sum;
            }

            // ============================================================
            // Step 3: attn_probs @ V with vectorized FMA
            // ============================================================
            for (int d = 0; d < head_dim; d++) {
                __m512 acc_vec = _mm512_setzero_ps();

                n = 0;
                for (; n + 15 < M; n += 16) {
                    // Gather attention probs
                    alignas(64) float prob_temp[16];
                    alignas(64) float v_temp[16];
                    for (int i = 0; i < 16; i++) {
                        prob_temp[i] = attn_probs[(n + i) * num_heads + h];
                        const int8_t* v_row = v + (n + i) * hidden_dim;
                        v_temp[i] = static_cast<float>(v_row[h * head_dim + d]) * v_scales[n + i];
                    }
                    __m512 prob_vec = _mm512_load_ps(prob_temp);
                    __m512 v_vec = _mm512_load_ps(v_temp);
                    acc_vec = _mm512_fmadd_ps(prob_vec, v_vec, acc_vec);
                }

                float sum = _mm512_reduce_add_ps(acc_vec);
                for (; n < M; n++) {
                    const int8_t* v_row = v + n * hidden_dim;
                    float v_val = static_cast<float>(v_row[h * head_dim + d]) * v_scales[n];
                    sum += attn_probs[n * num_heads + h] * v_val;
                }
                attn_out[h * head_dim + d] = sum;
            }
        }

        // ============================================================
        // Step 4: Quantize output to int8 (vectorized)
        // ============================================================
        __m512 max_abs_vec = _mm512_setzero_ps();
        int i = 0;
        for (; i + 15 < hidden_dim; i += 16) {
            __m512 vals = _mm512_loadu_ps(attn_out + i);
            __m512 abs_vals = _mm512_abs_ps(vals);
            max_abs_vec = _mm512_max_ps(max_abs_vec, abs_vals);
        }
        float max_abs = _mm512_reduce_max_ps(max_abs_vec);
        for (; i < hidden_dim; i++) {
            max_abs = std::max(max_abs, std::abs(attn_out[i]));
        }

        float out_scale_val = max_abs / 127.0f;
        if (out_scale_val == 0.0f) out_scale_val = 1.0f;
        out_scales[m] = out_scale_val;

        __m512 inv_scale_vec = _mm512_set1_ps(1.0f / out_scale_val);
        __m512 min_val = _mm512_set1_ps(-127.0f);
        __m512 max_val = _mm512_set1_ps(127.0f);

        i = 0;
        for (; i + 15 < hidden_dim; i += 16) {
            __m512 vals = _mm512_loadu_ps(attn_out + i);
            vals = _mm512_mul_ps(vals, inv_scale_vec);
            vals = _mm512_max_ps(vals, min_val);
            vals = _mm512_min_ps(vals, max_val);
            vals = _mm512_roundscale_ps(vals, _MM_FROUND_TO_NEAREST_INT);
            __m512i int_vals = _mm512_cvtps_epi32(vals);

            // Pack to int8
            __m128i packed = _mm512_cvtsepi32_epi8(int_vals);
            _mm_storeu_si128((__m128i*)(out_row + i), packed);
        }
        for (; i < hidden_dim; i++) {
            float val = attn_out[i] / out_scale_val;
            val = std::max(-127.0f, std::min(127.0f, val));
            out_row[i] = static_cast<int8_t>(std::round(val));
        }

        free(attn_scores);
        free(attn_probs);
        free(attn_out);
    }
}

/**
 * Flash Attention Style Tiled Attention - AVX-512 Optimized
 *
 * Key optimizations over standard attention:
 * 1. Tiles the computation to avoid O(M²) memory
 * 2. Computes attention in chunks, accumulating online softmax
 * 3. Better cache utilization by processing blocks at a time
 * 4. Parallelizes over M×heads for large M (better thread utilization)
 *
 * Online softmax algorithm:
 * - Process keys in blocks
 * - Maintain running max and sum
 * - Rescale previous accumulator when max changes
 */
void attention_int8_flash(
    torch::Tensor q_tensor,         // [M, num_heads * head_dim] int8
    torch::Tensor k_tensor,         // [M, num_heads * head_dim] int8
    torch::Tensor v_tensor,         // [M, num_heads * head_dim] int8
    torch::Tensor q_scale_tensor,   // [M] float32
    torch::Tensor k_scale_tensor,   // [M] float32
    torch::Tensor v_scale_tensor,   // [M] float32
    torch::Tensor out_tensor,       // [M, num_heads * head_dim] int8 output
    torch::Tensor out_scale_tensor, // [M] float32 output scales
    int M, int num_heads, int head_dim,
    float scale,                    // 1/sqrt(head_dim)
    int num_threads
) {
    const int8_t* __restrict__ q = q_tensor.data_ptr<int8_t>();
    const int8_t* __restrict__ k = k_tensor.data_ptr<int8_t>();
    const int8_t* __restrict__ v = v_tensor.data_ptr<int8_t>();
    const float* __restrict__ q_scales = q_scale_tensor.data_ptr<float>();
    const float* __restrict__ k_scales = k_scale_tensor.data_ptr<float>();
    const float* __restrict__ v_scales = v_scale_tensor.data_ptr<float>();
    int8_t* __restrict__ out = out_tensor.data_ptr<int8_t>();
    float* __restrict__ out_scales = out_scale_tensor.data_ptr<float>();

    const int hidden_dim = num_heads * head_dim;

    // Block size for Flash Attention style tiling
    // Should fit in L1/L2 cache
    constexpr int BLOCK_SIZE = 32;

    omp_set_num_threads(num_threads);

    // Parallelize over M (query rows) - each thread handles one query
    #pragma omp parallel for schedule(static)
    for (int m = 0; m < M; m++) {
        const int8_t* q_row = q + m * hidden_dim;
        float q_scale_val = q_scales[m];
        int8_t* out_row = out + m * hidden_dim;

        // Per-head output accumulator (float)
        float* attn_out = (float*)aligned_alloc(64, hidden_dim * sizeof(float));
        memset(attn_out, 0, hidden_dim * sizeof(float));

        // Process each head
        for (int h = 0; h < num_heads; h++) {
            const int8_t* q_head = q_row + h * head_dim;

            // Online softmax state per head
            float running_max = -INFINITY;
            float running_sum = 0.0f;

            // Output accumulator for this head
            float* head_out = attn_out + h * head_dim;
            for (int d = 0; d < head_dim; d++) {
                head_out[d] = 0.0f;
            }

            // Process keys in blocks (Flash Attention style)
            for (int k_start = 0; k_start < M; k_start += BLOCK_SIZE) {
                int k_end = std::min(k_start + BLOCK_SIZE, M);
                int block_size = k_end - k_start;

                // Compute scores for this block
                alignas(64) float scores[BLOCK_SIZE];
                float block_max = -INFINITY;

                for (int n = 0; n < block_size; n++) {
                    int k_idx = k_start + n;
                    const int8_t* k_row = k + k_idx * hidden_dim;
                    const int8_t* k_head = k_row + h * head_dim;
                    float k_scale_val = k_scales[k_idx];

                    // Dot product Q @ K^T
                    __m512i acc = _mm512_setzero_si512();
                    int d = 0;
                    for (; d + 31 < head_dim; d += 32) {
                        __m256i q_vec = _mm256_loadu_si256((__m256i*)(q_head + d));
                        __m256i k_vec = _mm256_loadu_si256((__m256i*)(k_head + d));

                        __m512i q_16 = _mm512_cvtepi8_epi16(q_vec);
                        __m512i k_16 = _mm512_cvtepi8_epi16(k_vec);

                        __m512i prod = _mm512_madd_epi16(q_16, k_16);
                        acc = _mm512_add_epi32(acc, prod);
                    }

                    int32_t dot = _mm512_reduce_add_epi32(acc);
                    for (; d < head_dim; d++) {
                        dot += static_cast<int32_t>(q_head[d]) * static_cast<int32_t>(k_head[d]);
                    }

                    scores[n] = static_cast<float>(dot) * q_scale_val * k_scale_val * scale;
                    block_max = std::max(block_max, scores[n]);
                }

                // Online softmax update
                float new_max = std::max(running_max, block_max);

                // Rescale previous accumulator if max changed
                if (running_max != -INFINITY && new_max > running_max) {
                    float rescale = std::exp(running_max - new_max);
                    running_sum *= rescale;
                    for (int d = 0; d < head_dim; d++) {
                        head_out[d] *= rescale;
                    }
                }

                // Compute exp(scores - new_max) and accumulate
                float block_sum = 0.0f;
                alignas(64) float probs[BLOCK_SIZE];

                for (int n = 0; n < block_size; n++) {
                    probs[n] = std::exp(scores[n] - new_max);
                    block_sum += probs[n];
                }

                // Accumulate weighted V
                for (int n = 0; n < block_size; n++) {
                    int k_idx = k_start + n;
                    const int8_t* v_row = v + k_idx * hidden_dim;
                    const int8_t* v_head = v_row + h * head_dim;
                    float v_scale_val = v_scales[k_idx];
                    float prob = probs[n];

                    // Vectorized V accumulation
                    int d = 0;
                    __m512 prob_vec = _mm512_set1_ps(prob);
                    __m512 v_scale_vec = _mm512_set1_ps(v_scale_val);

                    for (; d + 15 < head_dim; d += 16) {
                        // Load 16 int8 values
                        __m128i v_int8 = _mm_loadu_si128((__m128i*)(v_head + d));
                        __m512i v_int32 = _mm512_cvtepi8_epi32(v_int8);
                        __m512 v_float = _mm512_cvtepi32_ps(v_int32);
                        v_float = _mm512_mul_ps(v_float, v_scale_vec);

                        // Load current accumulator
                        __m512 acc = _mm512_loadu_ps(head_out + d);
                        acc = _mm512_fmadd_ps(prob_vec, v_float, acc);
                        _mm512_storeu_ps(head_out + d, acc);
                    }

                    for (; d < head_dim; d++) {
                        head_out[d] += prob * static_cast<float>(v_head[d]) * v_scale_val;
                    }
                }

                running_max = new_max;
                running_sum += block_sum;
            }

            // Normalize by sum
            float inv_sum = 1.0f / running_sum;
            for (int d = 0; d < head_dim; d++) {
                head_out[d] *= inv_sum;
            }
        }

        // Quantize output to int8
        float max_abs = 0.0f;
        for (int i = 0; i < hidden_dim; i++) {
            max_abs = std::max(max_abs, std::abs(attn_out[i]));
        }

        float out_scale_val = max_abs / 127.0f;
        if (out_scale_val == 0.0f) out_scale_val = 1.0f;
        out_scales[m] = out_scale_val;

        float inv_scale = 1.0f / out_scale_val;
        for (int i = 0; i < hidden_dim; i++) {
            float val = attn_out[i] * inv_scale;
            val = std::max(-127.0f, std::min(127.0f, val));
            out_row[i] = static_cast<int8_t>(std::round(val));
        }

        free(attn_out);
    }
}

/**
 * Attention kernel optimized for large M (parallelizes over M×heads)
 *
 * For large M (e.g., 128) with many heads (e.g., 32), parallelizing over just M
 * wastes thread potential. With 16 threads and M=128, each thread handles 8 queries.
 * But if we parallelize over M×heads (128×32=4096), we have much better work distribution.
 *
 * Key insight: Each (m, h) pair is independent and can be computed in parallel.
 * We just need to coordinate the final output write.
 */
void attention_int8_v2(
    torch::Tensor q_tensor,         // [M, num_heads * head_dim] int8
    torch::Tensor k_tensor,         // [M, num_heads * head_dim] int8
    torch::Tensor v_tensor,         // [M, num_heads * head_dim] int8
    torch::Tensor q_scale_tensor,   // [M] float32
    torch::Tensor k_scale_tensor,   // [M] float32
    torch::Tensor v_scale_tensor,   // [M] float32
    torch::Tensor out_tensor,       // [M, num_heads * head_dim] int8 output
    torch::Tensor out_scale_tensor, // [M] float32 output scales
    int M, int num_heads, int head_dim,
    float scale,                    // 1/sqrt(head_dim)
    int num_threads
) {
    const int8_t* __restrict__ q = q_tensor.data_ptr<int8_t>();
    const int8_t* __restrict__ k = k_tensor.data_ptr<int8_t>();
    const int8_t* __restrict__ v = v_tensor.data_ptr<int8_t>();
    const float* __restrict__ q_scales = q_scale_tensor.data_ptr<float>();
    const float* __restrict__ k_scales = k_scale_tensor.data_ptr<float>();
    const float* __restrict__ v_scales = v_scale_tensor.data_ptr<float>();
    int8_t* __restrict__ out = out_tensor.data_ptr<int8_t>();
    float* __restrict__ out_scales = out_scale_tensor.data_ptr<float>();

    const int hidden_dim = num_heads * head_dim;
    const int total_work = M * num_heads;
    constexpr int BLOCK_SIZE = 32;

    omp_set_num_threads(num_threads);

    // Allocate intermediate float output buffer (we'll quantize at the end per row)
    float* attn_out_all = (float*)aligned_alloc(64, M * hidden_dim * sizeof(float));
    memset(attn_out_all, 0, M * hidden_dim * sizeof(float));

    // Parallelize over M×heads
    #pragma omp parallel for schedule(dynamic, 4)
    for (int work_idx = 0; work_idx < total_work; work_idx++) {
        int m = work_idx / num_heads;
        int h = work_idx % num_heads;

        const int8_t* q_row = q + m * hidden_dim;
        const int8_t* q_head = q_row + h * head_dim;
        float q_scale_val = q_scales[m];

        // Output location for this head
        float* head_out = attn_out_all + m * hidden_dim + h * head_dim;

        // Online softmax state
        float running_max = -INFINITY;
        float running_sum = 0.0f;

        // Initialize head output
        for (int d = 0; d < head_dim; d++) {
            head_out[d] = 0.0f;
        }

        // Process keys in blocks (Flash Attention style)
        for (int k_start = 0; k_start < M; k_start += BLOCK_SIZE) {
            int k_end = std::min(k_start + BLOCK_SIZE, M);
            int block_size = k_end - k_start;

            // Compute scores for this block
            alignas(64) float scores[BLOCK_SIZE];
            float block_max = -INFINITY;

            for (int n = 0; n < block_size; n++) {
                int k_idx = k_start + n;
                const int8_t* k_row = k + k_idx * hidden_dim;
                const int8_t* k_head = k_row + h * head_dim;
                float k_scale_val = k_scales[k_idx];

                // Dot product Q @ K^T using AVX-512
                __m512i acc = _mm512_setzero_si512();
                int d = 0;
                for (; d + 31 < head_dim; d += 32) {
                    __m256i q_vec = _mm256_loadu_si256((__m256i*)(q_head + d));
                    __m256i k_vec = _mm256_loadu_si256((__m256i*)(k_head + d));

                    __m512i q_16 = _mm512_cvtepi8_epi16(q_vec);
                    __m512i k_16 = _mm512_cvtepi8_epi16(k_vec);

                    __m512i prod = _mm512_madd_epi16(q_16, k_16);
                    acc = _mm512_add_epi32(acc, prod);
                }

                int32_t dot = _mm512_reduce_add_epi32(acc);
                for (; d < head_dim; d++) {
                    dot += static_cast<int32_t>(q_head[d]) * static_cast<int32_t>(k_head[d]);
                }

                scores[n] = static_cast<float>(dot) * q_scale_val * k_scale_val * scale;
                block_max = std::max(block_max, scores[n]);
            }

            // Online softmax update
            float new_max = std::max(running_max, block_max);

            // Rescale previous accumulator if max changed
            if (running_max != -INFINITY && new_max > running_max) {
                float rescale = std::exp(running_max - new_max);
                running_sum *= rescale;
                __m512 rescale_vec = _mm512_set1_ps(rescale);
                int d = 0;
                for (; d + 15 < head_dim; d += 16) {
                    __m512 vals = _mm512_loadu_ps(head_out + d);
                    vals = _mm512_mul_ps(vals, rescale_vec);
                    _mm512_storeu_ps(head_out + d, vals);
                }
                for (; d < head_dim; d++) {
                    head_out[d] *= rescale;
                }
            }

            // Compute exp(scores - new_max) and accumulate
            float block_sum = 0.0f;
            alignas(64) float probs[BLOCK_SIZE];

            for (int n = 0; n < block_size; n++) {
                probs[n] = std::exp(scores[n] - new_max);
                block_sum += probs[n];
            }

            // Accumulate weighted V (vectorized)
            for (int n = 0; n < block_size; n++) {
                int k_idx = k_start + n;
                const int8_t* v_row = v + k_idx * hidden_dim;
                const int8_t* v_head = v_row + h * head_dim;
                float v_scale_val = v_scales[k_idx];
                float prob = probs[n];

                __m512 prob_vec = _mm512_set1_ps(prob);
                __m512 v_scale_vec = _mm512_set1_ps(v_scale_val);

                int d = 0;
                for (; d + 15 < head_dim; d += 16) {
                    __m128i v_int8 = _mm_loadu_si128((__m128i*)(v_head + d));
                    __m512i v_int32 = _mm512_cvtepi8_epi32(v_int8);
                    __m512 v_float = _mm512_cvtepi32_ps(v_int32);
                    v_float = _mm512_mul_ps(v_float, v_scale_vec);

                    __m512 acc_val = _mm512_loadu_ps(head_out + d);
                    acc_val = _mm512_fmadd_ps(prob_vec, v_float, acc_val);
                    _mm512_storeu_ps(head_out + d, acc_val);
                }

                for (; d < head_dim; d++) {
                    head_out[d] += prob * static_cast<float>(v_head[d]) * v_scale_val;
                }
            }

            running_max = new_max;
            running_sum += block_sum;
        }

        // Normalize by sum
        float inv_sum = 1.0f / running_sum;
        __m512 inv_sum_vec = _mm512_set1_ps(inv_sum);
        int d = 0;
        for (; d + 15 < head_dim; d += 16) {
            __m512 vals = _mm512_loadu_ps(head_out + d);
            vals = _mm512_mul_ps(vals, inv_sum_vec);
            _mm512_storeu_ps(head_out + d, vals);
        }
        for (; d < head_dim; d++) {
            head_out[d] *= inv_sum;
        }
    }

    // Quantize output per row (parallelize over M)
    #pragma omp parallel for schedule(static)
    for (int m = 0; m < M; m++) {
        float* attn_out = attn_out_all + m * hidden_dim;
        int8_t* out_row = out + m * hidden_dim;

        // Find max abs using AVX-512
        __m512 max_abs_vec = _mm512_setzero_ps();
        int i = 0;
        for (; i + 15 < hidden_dim; i += 16) {
            __m512 vals = _mm512_loadu_ps(attn_out + i);
            __m512 abs_vals = _mm512_abs_ps(vals);
            max_abs_vec = _mm512_max_ps(max_abs_vec, abs_vals);
        }
        float max_abs = _mm512_reduce_max_ps(max_abs_vec);
        for (; i < hidden_dim; i++) {
            max_abs = std::max(max_abs, std::abs(attn_out[i]));
        }

        float out_scale_val = max_abs / 127.0f;
        if (out_scale_val == 0.0f) out_scale_val = 1.0f;
        out_scales[m] = out_scale_val;

        // Quantize
        __m512 inv_scale_vec = _mm512_set1_ps(1.0f / out_scale_val);
        __m512 min_val = _mm512_set1_ps(-127.0f);
        __m512 max_val = _mm512_set1_ps(127.0f);

        i = 0;
        for (; i + 15 < hidden_dim; i += 16) {
            __m512 vals = _mm512_loadu_ps(attn_out + i);
            vals = _mm512_mul_ps(vals, inv_scale_vec);
            vals = _mm512_max_ps(vals, min_val);
            vals = _mm512_min_ps(vals, max_val);
            vals = _mm512_roundscale_ps(vals, _MM_FROUND_TO_NEAREST_INT);
            __m512i int_vals = _mm512_cvtps_epi32(vals);
            __m128i packed = _mm512_cvtsepi32_epi8(int_vals);
            _mm_storeu_si128((__m128i*)(out_row + i), packed);
        }
        for (; i < hidden_dim; i++) {
            float val = attn_out[i] / out_scale_val;
            val = std::max(-127.0f, std::min(127.0f, val));
            out_row[i] = static_cast<int8_t>(std::round(val));
        }
    }

    free(attn_out_all);
}

/**
 * Fused QKV Projection + Attention
 *
 * Combines Q, K, V projections with attention in a single kernel.
 * Avoids intermediate materialization of Q, K, V tensors.
 *
 * Input: x [M, hidden_dim] int8
 * Output: y [M, hidden_dim] int8
 */
void fused_qkv_attention(
    torch::Tensor x_tensor,         // [M, hidden_dim] int8 input
    torch::Tensor x_scale_tensor,   // [M] float32 input scales
    torch::Tensor wq_tensor,        // [hidden_dim, K_padded] int8 Q weights
    torch::Tensor wq_sum_tensor,    // [hidden_dim] int32 Q weight sums
    torch::Tensor wk_tensor,        // [hidden_dim, K_padded] int8 K weights
    torch::Tensor wk_sum_tensor,    // [hidden_dim] int32 K weight sums
    torch::Tensor wv_tensor,        // [hidden_dim, K_padded] int8 V weights
    torch::Tensor wv_sum_tensor,    // [hidden_dim] int32 V weight sums
    torch::Tensor out_tensor,       // [M, hidden_dim] int8 output
    torch::Tensor out_scale_tensor, // [M] float32 output scales
    int M, int hidden_dim, int num_heads, int head_dim,
    float scale,
    int num_threads
) {
    const int8_t* __restrict__ x = x_tensor.data_ptr<int8_t>();
    const float* __restrict__ x_scales = x_scale_tensor.data_ptr<float>();
    const int8_t* __restrict__ wq = wq_tensor.data_ptr<int8_t>();
    const int32_t* __restrict__ wq_sum = wq_sum_tensor.data_ptr<int32_t>();
    const int8_t* __restrict__ wk = wk_tensor.data_ptr<int8_t>();
    const int32_t* __restrict__ wk_sum = wk_sum_tensor.data_ptr<int32_t>();
    const int8_t* __restrict__ wv = wv_tensor.data_ptr<int8_t>();
    const int32_t* __restrict__ wv_sum = wv_sum_tensor.data_ptr<int32_t>();
    int8_t* __restrict__ out = out_tensor.data_ptr<int8_t>();
    float* __restrict__ out_scales = out_scale_tensor.data_ptr<float>();

    const int K_padded = ((hidden_dim + 63) / 64) * 64;
    constexpr int BLOCK_SIZE = 32;

    omp_set_num_threads(num_threads);

    // Precompute all Q, K, V projections in parallel first
    // This is more cache-friendly than computing on-the-fly
    float* Q_float = (float*)aligned_alloc(64, M * hidden_dim * sizeof(float));
    float* K_float = (float*)aligned_alloc(64, M * hidden_dim * sizeof(float));
    float* V_float = (float*)aligned_alloc(64, M * hidden_dim * sizeof(float));

    // Compute Q, K, V projections
    #pragma omp parallel for schedule(static)
    for (int m = 0; m < M; m++) {
        const int8_t* x_row = x + m * hidden_dim;
        float x_scale = x_scales[m];

        // Convert x to uint8 for VNNI
        alignas(64) uint8_t x_uint8[K_padded];
        for (int k = 0; k < hidden_dim; k++) {
            x_uint8[k] = static_cast<uint8_t>(static_cast<int16_t>(x_row[k]) + 128);
        }
        for (int k = hidden_dim; k < K_padded; k++) {
            x_uint8[k] = 128;
        }

        // Compute Q row
        for (int n = 0; n < hidden_dim; n++) {
            const int8_t* w_row = wq + n * K_padded;
            __m512i acc = _mm512_setzero_si512();
            for (int kk = 0; kk < K_padded; kk += 64) {
                __m512i x_vec = _mm512_load_si512((__m512i*)(x_uint8 + kk));
                __m512i w_vec = _mm512_loadu_si512((__m512i*)(w_row + kk));
                acc = _mm512_dpbusd_epi32(acc, x_vec, w_vec);
            }
            int32_t sum = _mm512_reduce_add_epi32(acc) - 128 * wq_sum[n];
            Q_float[m * hidden_dim + n] = static_cast<float>(sum) * x_scale;
        }

        // Compute K row
        for (int n = 0; n < hidden_dim; n++) {
            const int8_t* w_row = wk + n * K_padded;
            __m512i acc = _mm512_setzero_si512();
            for (int kk = 0; kk < K_padded; kk += 64) {
                __m512i x_vec = _mm512_load_si512((__m512i*)(x_uint8 + kk));
                __m512i w_vec = _mm512_loadu_si512((__m512i*)(w_row + kk));
                acc = _mm512_dpbusd_epi32(acc, x_vec, w_vec);
            }
            int32_t sum = _mm512_reduce_add_epi32(acc) - 128 * wk_sum[n];
            K_float[m * hidden_dim + n] = static_cast<float>(sum) * x_scale;
        }

        // Compute V row
        for (int n = 0; n < hidden_dim; n++) {
            const int8_t* w_row = wv + n * K_padded;
            __m512i acc = _mm512_setzero_si512();
            for (int kk = 0; kk < K_padded; kk += 64) {
                __m512i x_vec = _mm512_load_si512((__m512i*)(x_uint8 + kk));
                __m512i w_vec = _mm512_loadu_si512((__m512i*)(w_row + kk));
                acc = _mm512_dpbusd_epi32(acc, x_vec, w_vec);
            }
            int32_t sum = _mm512_reduce_add_epi32(acc) - 128 * wv_sum[n];
            V_float[m * hidden_dim + n] = static_cast<float>(sum) * x_scale;
        }
    }

    // Now compute attention using the float Q, K, V
    #pragma omp parallel for schedule(static)
    for (int m = 0; m < M; m++) {
        const float* q_row = Q_float + m * hidden_dim;
        int8_t* out_row = out + m * hidden_dim;
        float* attn_out = (float*)aligned_alloc(64, hidden_dim * sizeof(float));

        for (int h = 0; h < num_heads; h++) {
            const float* q_head = q_row + h * head_dim;
            float* head_out = attn_out + h * head_dim;

            float running_max = -INFINITY;
            float running_sum = 0.0f;
            for (int d = 0; d < head_dim; d++) head_out[d] = 0.0f;

            for (int k_start = 0; k_start < M; k_start += BLOCK_SIZE) {
                int k_end = std::min(k_start + BLOCK_SIZE, M);
                int block_size = k_end - k_start;

                alignas(64) float scores[BLOCK_SIZE];
                float block_max = -INFINITY;

                for (int n = 0; n < block_size; n++) {
                    int k_idx = k_start + n;
                    const float* k_head = K_float + k_idx * hidden_dim + h * head_dim;

                    // Float dot product (already dequantized)
                    __m512 acc = _mm512_setzero_ps();
                    int d = 0;
                    for (; d + 15 < head_dim; d += 16) {
                        __m512 q_vec = _mm512_loadu_ps(q_head + d);
                        __m512 k_vec = _mm512_loadu_ps(k_head + d);
                        acc = _mm512_fmadd_ps(q_vec, k_vec, acc);
                    }
                    float dot = _mm512_reduce_add_ps(acc);
                    for (; d < head_dim; d++) {
                        dot += q_head[d] * k_head[d];
                    }

                    scores[n] = dot * scale;
                    block_max = std::max(block_max, scores[n]);
                }

                float new_max = std::max(running_max, block_max);
                if (running_max != -INFINITY && new_max > running_max) {
                    float rescale = std::exp(running_max - new_max);
                    running_sum *= rescale;
                    for (int d = 0; d < head_dim; d++) head_out[d] *= rescale;
                }

                float block_sum = 0.0f;
                for (int n = 0; n < block_size; n++) {
                    float prob = std::exp(scores[n] - new_max);
                    block_sum += prob;

                    int k_idx = k_start + n;
                    const float* v_head = V_float + k_idx * hidden_dim + h * head_dim;

                    __m512 prob_vec = _mm512_set1_ps(prob);
                    int d = 0;
                    for (; d + 15 < head_dim; d += 16) {
                        __m512 v_vec = _mm512_loadu_ps(v_head + d);
                        __m512 out_vec = _mm512_loadu_ps(head_out + d);
                        out_vec = _mm512_fmadd_ps(prob_vec, v_vec, out_vec);
                        _mm512_storeu_ps(head_out + d, out_vec);
                    }
                    for (; d < head_dim; d++) {
                        head_out[d] += prob * v_head[d];
                    }
                }

                running_max = new_max;
                running_sum += block_sum;
            }

            float inv_sum = 1.0f / running_sum;
            for (int d = 0; d < head_dim; d++) head_out[d] *= inv_sum;
        }

        // Quantize output
        float max_abs = 0.0f;
        for (int i = 0; i < hidden_dim; i++) {
            max_abs = std::max(max_abs, std::abs(attn_out[i]));
        }

        float out_scale_val = max_abs / 127.0f;
        if (out_scale_val == 0.0f) out_scale_val = 1.0f;
        out_scales[m] = out_scale_val;

        float inv_scale = 1.0f / out_scale_val;
        for (int i = 0; i < hidden_dim; i++) {
            float val = attn_out[i] * inv_scale;
            val = std::max(-127.0f, std::min(127.0f, val));
            out_row[i] = static_cast<int8_t>(std::round(val));
        }

        free(attn_out);
    }

    free(Q_float);
    free(K_float);
    free(V_float);
}

/**
 * Fused residual add + quantize - AVX-512 Vectorized
 *
 * Computes: y = x1 + x2, then quantizes to int8
 * Both inputs are int8 with their own scales.
 */
void fused_residual_quantize(
    torch::Tensor x1_tensor,        // [M, N] int8
    torch::Tensor x1_scale_tensor,  // [M] float32
    torch::Tensor x2_tensor,        // [M, N] int8
    torch::Tensor x2_scale_tensor,  // [M] float32
    torch::Tensor y_tensor,         // [M, N] int8 output
    torch::Tensor y_scale_tensor,   // [M] float32 output scales
    int M, int N,
    int num_threads
) {
    const int8_t* __restrict__ x1 = x1_tensor.data_ptr<int8_t>();
    const float* __restrict__ x1_scales = x1_scale_tensor.data_ptr<float>();
    const int8_t* __restrict__ x2 = x2_tensor.data_ptr<int8_t>();
    const float* __restrict__ x2_scales = x2_scale_tensor.data_ptr<float>();
    int8_t* __restrict__ y = y_tensor.data_ptr<int8_t>();
    float* __restrict__ y_scales = y_scale_tensor.data_ptr<float>();

    const int N_padded = ((N + 15) / 16) * 16;

    omp_set_num_threads(num_threads);

    #pragma omp parallel for schedule(static)
    for (int m = 0; m < M; m++) {
        const int8_t* x1_row = x1 + m * N;
        const int8_t* x2_row = x2 + m * N;
        int8_t* y_row = y + m * N;

        float x1_scale = x1_scales[m];
        float x2_scale = x2_scales[m];

        __m512 x1_scale_vec = _mm512_set1_ps(x1_scale);
        __m512 x2_scale_vec = _mm512_set1_ps(x2_scale);

        // First pass: compute sum and find max (vectorized)
        float* temp = (float*)aligned_alloc(64, N_padded * sizeof(float));
        __m512 max_abs_vec = _mm512_setzero_ps();

        int n = 0;
        for (; n + 15 < N; n += 16) {
            // Load 16 int8 values from each input
            __m128i x1_int8 = _mm_loadu_si128((__m128i*)(x1_row + n));
            __m128i x2_int8 = _mm_loadu_si128((__m128i*)(x2_row + n));

            // Convert to int32
            __m512i x1_int32 = _mm512_cvtepi8_epi32(x1_int8);
            __m512i x2_int32 = _mm512_cvtepi8_epi32(x2_int8);

            // Convert to float and scale
            __m512 x1_float = _mm512_mul_ps(_mm512_cvtepi32_ps(x1_int32), x1_scale_vec);
            __m512 x2_float = _mm512_mul_ps(_mm512_cvtepi32_ps(x2_int32), x2_scale_vec);

            // Add
            __m512 sum = _mm512_add_ps(x1_float, x2_float);
            _mm512_store_ps(temp + n, sum);

            // Track max abs
            __m512 abs_sum = _mm512_abs_ps(sum);
            max_abs_vec = _mm512_max_ps(max_abs_vec, abs_sum);
        }

        float max_abs = _mm512_reduce_max_ps(max_abs_vec);

        // Handle remainder
        for (; n < N; n++) {
            float val = static_cast<float>(x1_row[n]) * x1_scale +
                       static_cast<float>(x2_row[n]) * x2_scale;
            temp[n] = val;
            max_abs = std::max(max_abs, std::abs(val));
        }

        // Quantize (vectorized)
        float y_scale = max_abs / 127.0f;
        if (y_scale == 0.0f) y_scale = 1.0f;
        y_scales[m] = y_scale;

        __m512 inv_scale_vec = _mm512_set1_ps(1.0f / y_scale);
        __m512 min_val = _mm512_set1_ps(-127.0f);
        __m512 max_val = _mm512_set1_ps(127.0f);

        n = 0;
        for (; n + 15 < N; n += 16) {
            __m512 vals = _mm512_load_ps(temp + n);
            vals = _mm512_mul_ps(vals, inv_scale_vec);
            vals = _mm512_max_ps(vals, min_val);
            vals = _mm512_min_ps(vals, max_val);
            vals = _mm512_roundscale_ps(vals, _MM_FROUND_TO_NEAREST_INT);
            __m512i int_vals = _mm512_cvtps_epi32(vals);
            __m128i packed = _mm512_cvtsepi32_epi8(int_vals);
            _mm_storeu_si128((__m128i*)(y_row + n), packed);
        }

        float inv_scale = 1.0f / y_scale;
        for (; n < N; n++) {
            float val = temp[n] * inv_scale;
            val = std::max(-127.0f, std::min(127.0f, val));
            y_row[n] = static_cast<int8_t>(std::round(val));
        }

        free(temp);
    }
}

/**
 * Quantize int32 matmul output to int8
 *
 * Takes int32 accumulator output and input scale, produces int8 output.
 */
void quantize_int32_to_int8(
    torch::Tensor x_tensor,         // [M, N] int32 input
    torch::Tensor x_scale_tensor,   // [M] float32 input scales (from activations)
    torch::Tensor y_tensor,         // [M, N] int8 output
    torch::Tensor y_scale_tensor,   // [M] float32 output scales
    int M, int N,
    int num_threads
) {
    const int32_t* __restrict__ x = x_tensor.data_ptr<int32_t>();
    const float* __restrict__ x_scales = x_scale_tensor.data_ptr<float>();
    int8_t* __restrict__ y = y_tensor.data_ptr<int8_t>();
    float* __restrict__ y_scales = y_scale_tensor.data_ptr<float>();

    omp_set_num_threads(num_threads);

    #pragma omp parallel for schedule(static)
    for (int m = 0; m < M; m++) {
        const int32_t* x_row = x + m * N;
        int8_t* y_row = y + m * N;
        float x_scale = x_scales[m];

        // First pass: convert to float and find max
        float max_abs = 0.0f;
        for (int n = 0; n < N; n++) {
            float val = static_cast<float>(x_row[n]) * x_scale;
            float abs_val = std::abs(val);
            if (abs_val > max_abs) max_abs = abs_val;
        }

        // Compute output scale
        float y_scale = max_abs / 127.0f;
        if (y_scale == 0.0f) y_scale = 1.0f;
        y_scales[m] = y_scale;

        // Second pass: quantize
        float inv_scale = 1.0f / y_scale;
        for (int n = 0; n < N; n++) {
            float val = static_cast<float>(x_row[n]) * x_scale * inv_scale;
            val = std::max(-127.0f, std::min(127.0f, val));
            y_row[n] = static_cast<int8_t>(std::round(val));
        }
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
    m.def("matmul_free_vnni_v2", &matmul_free_vnni_v2,
          "VNNI v2 optimized");
    m.def("matmul_free_vnni_v3", &matmul_free_vnni_v3,
          "VNNI v3 cache-optimized tiled");
    m.def("matmul_free_vnni_v3_int32_out", &matmul_free_vnni_v3_int32_out,
          "VNNI v3 with int32 output");
    m.def("softmax_int8", &softmax_int8,
          "Softmax with int8 output");
    m.def("swiglu_int8", &swiglu_int8,
          "SwiGLU activation with int8 output");
    m.def("attention_int8", &attention_int8,
          "Multi-head attention in int8");
    m.def("fused_residual_quantize", &fused_residual_quantize,
          "Fused residual add + quantize");
    m.def("quantize_int32_to_int8", &quantize_int32_to_int8,
          "Quantize int32 to int8");
    m.def("attention_int8_flash", &attention_int8_flash,
          "Flash Attention style tiled attention");
    m.def("attention_int8_v2", &attention_int8_v2,
          "Attention optimized for large M (parallelizes over M×heads)");
    m.def("fused_qkv_attention", &fused_qkv_attention,
          "Fused QKV projection + attention");
    m.def("matmul_free_vnni_v4_large_n", &matmul_free_vnni_v4_large_n,
          "VNNI v4 optimized for large N");
    m.def("matmul_free_vnni_v5_large_m", &matmul_free_vnni_v5_large_m,
          "VNNI v5 optimized for large M (parallelizes over M×N tiles)");
    m.def("fused_mlp_int8", &fused_mlp_int8,
          "Fused MLP: up + swiglu + down");
    m.def("matmul_softmax_int8", &matmul_softmax_int8,
          "Fused matmul + softmax (never writes int32 to memory)");
    m.def("matmul_qkv_softmax_int8", &matmul_qkv_softmax_int8,
          "Fused Q,K,V projection + softmax (reads input once)");
    m.def("matmul_quantize_int8", &matmul_quantize_int8,
          "Fused matmul + quantize (for down projection)");
    m.def("matmul_free_vnni_v3_fused_softmax", &matmul_free_vnni_v3_fused_softmax,
          "VNNI v3 with fused softmax (same tiling as v3)");
    m.def("matmul_free_vnni_v3_fused_quantize", &matmul_free_vnni_v3_fused_quantize,
          "VNNI v3 with fused quantize (same tiling as v3)");
    m.def("matmul_free_vnni_v3_fused_qkv_softmax", &matmul_free_vnni_v3_fused_qkv_softmax,
          "VNNI v3 fused Q,K,V projection + softmax (reads input once)");
}
