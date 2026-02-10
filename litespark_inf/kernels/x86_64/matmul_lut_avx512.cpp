/*
 * LUT-Based Ternary Matmul Kernel for AVX-512
 * 
 * Key insight from T-MAC paper:
 * - Group 2 ternary weights together -> 9 possible sums (3^2)
 * - Precompute LUT from activations
 * - Use vpshufb for O(1) lookup per group
 * 
 * This kernel is optimized for M=1 (token generation) where:
 * - LUT construction cost is amortized across all N outputs
 * - Memory bandwidth is the bottleneck
 * 
 * Weight format: 2-bit packed (4 weights per byte)
 * Encoding: -1=0b00, 0=0b01, +1=0b10
 */

#include <immintrin.h>
#include <cstdint>
#include <cstring>
#include <cmath>
#include <algorithm>
#include <omp.h>
#include <torch/extension.h>

// For 2 ternary weights: 3^2 = 9 combinations
// LUT index = (w0+1)*3 + (w1+1) where w in {-1,0,1}
// Index 0: w0=-1, w1=-1 -> -a0-a1
// Index 1: w0=-1, w1=0  -> -a0
// Index 2: w0=-1, w1=1  -> -a0+a1
// Index 3: w0=0,  w1=-1 -> -a1
// Index 4: w0=0,  w1=0  -> 0
// Index 5: w0=0,  w1=1  -> a1
// Index 6: w0=1,  w1=-1 -> a0-a1
// Index 7: w0=1,  w1=0  -> a0
// Index 8: w0=1,  w1=1  -> a0+a1

/**
 * Build LUT for 2-weight groups from int8 activations
 * Input: activations [K] int8
 * Output: lut [K/2][16] int16 (16 entries for vpshufb, only 9 used)
 */
inline void build_lut_2weight(const int8_t* __restrict__ act, int16_t* __restrict__ lut, int K) {
    for (int k = 0; k < K; k += 2) {
        int16_t a0 = act[k];
        int16_t a1 = act[k + 1];
        
        int16_t* lut_k = lut + (k / 2) * 16;
        lut_k[0] = -a0 - a1;  // (-1,-1)
        lut_k[1] = -a0;       // (-1, 0)
        lut_k[2] = -a0 + a1;  // (-1,+1)
        lut_k[3] = -a1;       // ( 0,-1)
        lut_k[4] = 0;         // ( 0, 0)
        lut_k[5] = a1;        // ( 0,+1)
        lut_k[6] = a0 - a1;   // (+1,-1)
        lut_k[7] = a0;        // (+1, 0)
        lut_k[8] = a0 + a1;   // (+1,+1)
        // Pad remaining entries
        lut_k[9] = lut_k[10] = lut_k[11] = lut_k[12] = lut_k[13] = lut_k[14] = lut_k[15] = 0;
    }
}

/**
 * Build LUT using AVX-512 - process 32 activation pairs at once
 */
inline void build_lut_2weight_avx512(const int8_t* __restrict__ act, int16_t* __restrict__ lut, int K) {
    // Process 64 activations (32 pairs) at a time
    int k = 0;
    for (; k + 63 < K; k += 64) {
        // Load 64 int8 activations
        __m512i a_vec = _mm512_loadu_si512((__m512i*)(act + k));
        
        // Deinterleave even/odd elements
        // a0 = act[0], act[2], act[4]... a1 = act[1], act[3], act[5]...
        __m512i idx_even = _mm512_set_epi8(
            62,60,58,56,54,52,50,48,46,44,42,40,38,36,34,32,
            30,28,26,24,22,20,18,16,14,12,10,8,6,4,2,0,
            62,60,58,56,54,52,50,48,46,44,42,40,38,36,34,32,
            30,28,26,24,22,20,18,16,14,12,10,8,6,4,2,0
        );
        __m512i idx_odd = _mm512_set_epi8(
            63,61,59,57,55,53,51,49,47,45,43,41,39,37,35,33,
            31,29,27,25,23,21,19,17,15,13,11,9,7,5,3,1,
            63,61,59,57,55,53,51,49,47,45,43,41,39,37,35,33,
            31,29,27,25,23,21,19,17,15,13,11,9,7,5,3,1
        );
        
        // For now, fall back to scalar for LUT building (complex deinterleave)
        // The hot path is the lookup, not LUT construction
        for (int kk = 0; kk < 64; kk += 2) {
            int16_t a0 = act[k + kk];
            int16_t a1 = act[k + kk + 1];
            int16_t* lut_k = lut + ((k + kk) / 2) * 16;
            lut_k[0] = -a0 - a1;
            lut_k[1] = -a0;
            lut_k[2] = -a0 + a1;
            lut_k[3] = -a1;
            lut_k[4] = 0;
            lut_k[5] = a1;
            lut_k[6] = a0 - a1;
            lut_k[7] = a0;
            lut_k[8] = a0 + a1;
            lut_k[9] = lut_k[10] = lut_k[11] = lut_k[12] = lut_k[13] = lut_k[14] = lut_k[15] = 0;
        }
    }
    // Handle remainder
    for (; k < K; k += 2) {
        int16_t a0 = act[k];
        int16_t a1 = (k + 1 < K) ? act[k + 1] : 0;
        int16_t* lut_k = lut + (k / 2) * 16;
        lut_k[0] = -a0 - a1;
        lut_k[1] = -a0;
        lut_k[2] = -a0 + a1;
        lut_k[3] = -a1;
        lut_k[4] = 0;
        lut_k[5] = a1;
        lut_k[6] = a0 - a1;
        lut_k[7] = a0;
        lut_k[8] = a0 + a1;
        lut_k[9] = lut_k[10] = lut_k[11] = lut_k[12] = lut_k[13] = lut_k[14] = lut_k[15] = 0;
    }
}

/**
 * Pack ternary weights to 4-bit index format
 * Input: weights [N, K] int8 with values {-1, 0, 1}
 * Output: packed [N, K/2] uint8, each byte = (w0+1)*3 + (w1+1)
 */
void pack_weights_lut(
    torch::Tensor w_tensor,      // [N, K] int8
    torch::Tensor packed_tensor, // [N, K/2] uint8 output
    int N, int K
) {
    const int8_t* w = w_tensor.data_ptr<int8_t>();
    uint8_t* packed = packed_tensor.data_ptr<uint8_t>();

    #pragma omp parallel for
    for (int n = 0; n < N; n++) {
        const int8_t* w_row = w + n * K;
        uint8_t* p_row = packed + n * (K / 2);

        for (int k = 0; k < K; k += 2) {
            int8_t w0 = w_row[k];
            int8_t w1 = (k + 1 < K) ? w_row[k + 1] : 0;
            // Map {-1,0,1} to {0,1,2}
            uint8_t idx = (uint8_t)((w0 + 1) * 3 + (w1 + 1));
            p_row[k / 2] = idx;
        }
    }
}

/**
 * LUT-based matmul for M=1 (token generation)
 *
 * For each output n:
 *   y[n] = sum_k LUT[k/2][weight_idx[n,k/2]]
 *
 * Key optimization: LUT is built once, reused for all N outputs
 */
void matmul_lut_m1(
    torch::Tensor x_int8_tensor,     // [1, K] int8 activations
    torch::Tensor scale_tensor,      // [1] float32 scale
    torch::Tensor w_packed_tensor,   // [N, K/2] uint8 packed weights
    torch::Tensor y_tensor,          // [1, N] float32 output
    torch::Tensor bias_tensor,
    int N, int K,
    int num_threads
) {
    const int8_t* x = x_int8_tensor.data_ptr<int8_t>();
    float scale = scale_tensor.data_ptr<float>()[0];
    const uint8_t* w_packed = w_packed_tensor.data_ptr<uint8_t>();
    float* y = y_tensor.data_ptr<float>();
    const float* bias = bias_tensor.defined() ? bias_tensor.data_ptr<float>() : nullptr;

    const int K_pairs = K / 2;

    // Allocate LUT: [K/2][16] int16
    int16_t* lut = (int16_t*)aligned_alloc(64, K_pairs * 16 * sizeof(int16_t));

    // Build LUT from activations (single-threaded, fast)
    build_lut_2weight_avx512(x, lut, K);

    omp_set_num_threads(num_threads);

    // Parallel over N outputs
    #pragma omp parallel for schedule(static)
    for (int n = 0; n < N; n++) {
        const uint8_t* w_row = w_packed + n * K_pairs;
        int32_t sum = 0;

        // Accumulate using LUT lookups
        // Process 32 pairs at a time using AVX-512
        int kp = 0;
        __m512i acc = _mm512_setzero_si512();

        for (; kp + 31 < K_pairs; kp += 32) {
            // Load 32 weight indices
            __m256i idx_vec = _mm256_loadu_si256((__m256i*)(w_row + kp));

            // For each of 32 indices, lookup in corresponding LUT
            // LUT is [K/2][16], so lut[(kp+i)*16 + idx[i]]
            // This is complex vectorization - use scalar for correctness first
            for (int i = 0; i < 32; i++) {
                uint8_t idx = w_row[kp + i];
                sum += lut[(kp + i) * 16 + idx];
            }
        }

        // Handle remainder
        for (; kp < K_pairs; kp++) {
            uint8_t idx = w_row[kp];
            sum += lut[kp * 16 + idx];
        }

        y[n] = static_cast<float>(sum) * scale + (bias ? bias[n] : 0.0f);
    }

    free(lut);
}

/**
 * LUT-based matmul with AVX-512 vpshufb optimization
 * Key: Reorganize LUT for efficient SIMD lookup
 *
 * Instead of LUT[K/2][16], use LUT[16][K/2] (transposed)
 * Then vpshufb can lookup 64 values at once
 */
void matmul_lut_m1_avx512(
    torch::Tensor x_int8_tensor,     // [1, K] int8 activations
    torch::Tensor scale_tensor,      // [1] float32 scale
    torch::Tensor w_packed_tensor,   // [N, K/2] uint8 packed weights
    torch::Tensor y_tensor,          // [1, N] float32 output
    torch::Tensor bias_tensor,
    int N, int K,
    int num_threads
) {
    const int8_t* x = x_int8_tensor.data_ptr<int8_t>();
    float scale = scale_tensor.data_ptr<float>()[0];
    const uint8_t* w_packed = w_packed_tensor.data_ptr<uint8_t>();
    float* y = y_tensor.data_ptr<float>();
    const float* bias = bias_tensor.defined() ? bias_tensor.data_ptr<float>() : nullptr;

    const int K_pairs = K / 2;
    const int K_pairs_padded = ((K_pairs + 63) / 64) * 64;

    // Build transposed LUT: [16][K_pairs_padded] int8 (saturate to int8 range)
    // For efficiency, we use int8 LUT values (saturated)
    int8_t* lut_t = (int8_t*)aligned_alloc(64, 16 * K_pairs_padded);
    memset(lut_t, 0, 16 * K_pairs_padded);

    // Build LUT entries
    for (int kp = 0; kp < K_pairs; kp++) {
        int16_t a0 = x[kp * 2];
        int16_t a1 = x[kp * 2 + 1];

        // Compute all 9 possible sums and saturate to int8
        int16_t vals[9] = {
            (int16_t)(-a0 - a1),  // idx 0
            (int16_t)(-a0),       // idx 1
            (int16_t)(-a0 + a1),  // idx 2
            (int16_t)(-a1),       // idx 3
            0,                     // idx 4
            a1,                    // idx 5
            (int16_t)(a0 - a1),   // idx 6
            a0,                    // idx 7
            (int16_t)(a0 + a1)    // idx 8
        };

        for (int i = 0; i < 9; i++) {
            // Saturate to int8 range
            int16_t v = vals[i];
            if (v > 127) v = 127;
            if (v < -128) v = -128;
            lut_t[i * K_pairs_padded + kp] = (int8_t)v;
        }
    }

    omp_set_num_threads(num_threads);

    // Parallel over N outputs
    #pragma omp parallel for schedule(static)
    for (int n = 0; n < N; n++) {
        const uint8_t* w_row = w_packed + n * K_pairs;
        __m512i acc = _mm512_setzero_si512();

        // Process 64 weight indices at a time
        int kp = 0;
        for (; kp + 63 < K_pairs; kp += 64) {
            // Load 64 weight indices
            __m512i idx = _mm512_loadu_si512((__m512i*)(w_row + kp));

            // For vpshufb, indices must be in low 4 bits of each byte
            // Our indices are 0-8, so this is fine

            // Load LUT rows and do shuffle
            // We need to gather from lut_t based on indices
            // This is tricky - vpshufb works on 16-byte lanes

            // Simpler approach: horizontal sum with gather
            // For now, use scalar accumulation (optimize later)
            for (int i = 0; i < 64; i++) {
                uint8_t idx_val = w_row[kp + i];
                acc = _mm512_add_epi32(acc,
                    _mm512_set1_epi32(lut_t[idx_val * K_pairs_padded + kp + i]));
            }
        }

        int32_t sum = _mm512_reduce_add_epi32(acc);

        // Handle remainder
        for (; kp < K_pairs; kp++) {
            uint8_t idx = w_row[kp];
            sum += lut_t[idx * K_pairs_padded + kp];
        }

        y[n] = static_cast<float>(sum) * scale + (bias ? bias[n] : 0.0f);
    }

    free(lut_t);
}

/**
 * Hybrid kernel: LUT for M<=4, VNNI for M>4
 */
void matmul_hybrid(
    torch::Tensor x_int8_tensor,
    torch::Tensor scale_tensor,
    torch::Tensor w_int8_tensor,     // [N, K] int8 for VNNI
    torch::Tensor w_packed_tensor,   // [N, K/2] uint8 for LUT (optional)
    torch::Tensor w_sum_tensor,
    torch::Tensor y_tensor,
    torch::Tensor bias_tensor,
    int M, int N, int K,
    int num_threads
) {
    if (M <= 4 && w_packed_tensor.defined()) {
        // Use LUT for small M
        for (int m = 0; m < M; m++) {
            // Create views for single row
            auto x_row = x_int8_tensor.index({m});
            auto scale_row = scale_tensor.index({m});
            auto y_row = y_tensor.index({m});

            matmul_lut_m1(
                x_row.unsqueeze(0),
                scale_row.unsqueeze(0),
                w_packed_tensor,
                y_row.unsqueeze(0),
                bias_tensor, N, K, num_threads
            );
        }
    } else {
        // Use VNNI for large M - call existing v3 kernel
        // (Implementation would call matmul_free_vnni_v3 here)
    }
}

// Python bindings
PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("pack_weights_lut", &pack_weights_lut,
          "Pack ternary weights for LUT kernel");
    m.def("matmul_lut_m1", &matmul_lut_m1,
          "LUT-based matmul for M=1 (token generation)");
    m.def("matmul_lut_m1_avx512", &matmul_lut_m1_avx512,
          "LUT-based matmul with AVX-512 optimization");
    m.def("matmul_hybrid", &matmul_hybrid,
          "Hybrid LUT/VNNI kernel");
}

