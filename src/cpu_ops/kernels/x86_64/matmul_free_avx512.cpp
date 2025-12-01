/*
 * AVX-512 Optimized MatMul-Free Kernel with 2-bit Packed Weights
 *
 * Key optimizations:
 * - AVX-512: 16-wide SIMD (vs 8-wide AVX2)
 * - 2-bit packed weights: 16x memory reduction
 * - Lookup table approach: decode 2-bit to mask in single operation
 *
 * Weight encoding: 00 = 0, 01 = +1, 11 = -1 (10 unused)
 * Packed format: 16 weights per uint32_t
 */

#include <immintrin.h>
#include <cstring>
#include <cstdint>
#include <algorithm>
#include <omp.h>
#include <torch/extension.h>

// Cross-platform CPUID support
#ifdef _MSC_VER
    #include <intrin.h>
#else
    #include <cpuid.h>
#endif

/**
 * Check if AVX-512F is supported on this CPU
 */
bool has_avx512_support() {
#ifdef _MSC_VER
    int cpuinfo[4];
    __cpuid(cpuinfo, 7);
    return (cpuinfo[1] & (1 << 16)) != 0;  // AVX-512F bit (EBX bit 16)
#else
    unsigned int eax, ebx, ecx, edx;
    if (__get_cpuid_count(7, 0, &eax, &ebx, &ecx, &edx)) {
        return (ebx & (1 << 16)) != 0;  // AVX-512F bit (EBX bit 16)
    }
    return false;
#endif
}

/**
 * Horizontal sum of __m512 vector
 */
inline float hsum_avx512(__m512 v) {
    // Reduce 512 -> 256 -> 128 -> scalar
    __m256 lo = _mm512_castps512_ps256(v);
    __m256 hi = _mm512_extractf32x8_ps(v, 1);
    __m256 sum256 = _mm256_add_ps(lo, hi);

    __m128 lo128 = _mm256_castps256_ps128(sum256);
    __m128 hi128 = _mm256_extractf128_ps(sum256, 1);
    __m128 sum128 = _mm_add_ps(lo128, hi128);

    __m128 shuf = _mm_movehdup_ps(sum128);
    __m128 sums = _mm_add_ps(sum128, shuf);
    shuf = _mm_movehl_ps(shuf, sums);
    sums = _mm_add_ss(sums, shuf);
    return _mm_cvtss_f32(sums);
}

/**
 * Pack ternary weights into 2-bit format
 *
 * Encoding: 00 = 0, 01 = +1, 11 = -1
 * 16 weights packed into each uint32_t
 *
 * @param w_tensor Float32 ternary weights [N, K]
 * @param w_packed_tensor Output packed weights [N, K_packed] where K_packed = ceil(K/16)
 * @param N Number of output features
 * @param K Number of input features
 */
void pack_weights_2bit_avx512(
    torch::Tensor w_tensor,
    torch::Tensor w_packed_tensor,
    int N, int K
) {
    const float* w = w_tensor.data_ptr<float>();
    uint32_t* w_packed = reinterpret_cast<uint32_t*>(w_packed_tensor.data_ptr<uint8_t>());

    const int K_packed = (K + 15) / 16;

    #pragma omp parallel for
    for (int n = 0; n < N; n++) {
        const float* w_row = w + n * K;
        uint32_t* packed_row = w_packed + n * K_packed;

        for (int kp = 0; kp < K_packed; kp++) {
            uint32_t packed = 0;
            for (int i = 0; i < 16 && (kp * 16 + i) < K; i++) {
                float val = w_row[kp * 16 + i];
                uint32_t code;
                if (val > 0.5f) {
                    code = 0x1;  // +1
                } else if (val < -0.5f) {
                    code = 0x3;  // -1
                } else {
                    code = 0x0;  // 0
                }
                packed |= (code << (i * 2));
            }
            packed_row[kp] = packed;
        }
    }
}

/**
 * Fast mask extraction from 2-bit packed weights using SIMD
 *
 * Given a uint32_t with 16 x 2-bit codes (encoding: 00=0, 01=+1, 11=-1):
 * - Extract bit0 of each 2-bit code -> if code & 1 is set, weight is non-zero
 * - Extract bit1 of each 2-bit code -> if code & 2 is set, weight is negative
 *
 * pos_mask = (bit0 set) AND (bit1 not set) = code == 01
 * neg_mask = (bit0 set) AND (bit1 set) = code == 11
 */
inline void extract_masks_fast(uint32_t packed, __mmask16& pos_mask, __mmask16& neg_mask) {
    // Extract even bits (bit0 of each 2-bit code) and odd bits (bit1)
    // packed format: b31b30 b29b28 ... b3b2 b1b0
    // bit0 positions: 0, 2, 4, 6, ... 30 (even)
    // bit1 positions: 1, 3, 5, 7, ... 31 (odd)

    // Use PEXT to extract specific bits (BMI2 instruction)
    // Even bits mask: 0x55555555 = 0101...0101
    // Odd bits mask:  0xAAAAAAAA = 1010...1010

    uint32_t bit0_vals = _pext_u32(packed, 0x55555555);  // Extract bit0 of each code
    uint32_t bit1_vals = _pext_u32(packed, 0xAAAAAAAA);  // Extract bit1 of each code

    // pos_mask: bit0=1 AND bit1=0 (code == 01)
    // neg_mask: bit0=1 AND bit1=1 (code == 11)
    pos_mask = (__mmask16)(bit0_vals & ~bit1_vals);
    neg_mask = (__mmask16)(bit0_vals & bit1_vals);
}

/**
 * AVX-512 MatMul-free kernel with 2-bit packed weights
 * VERSION 1: Basic implementation (for comparison)
 */
void matmul_free_avx512_packed_v1(
    torch::Tensor x_tensor,
    torch::Tensor w_packed_tensor,
    torch::Tensor y_tensor,
    torch::Tensor bias_tensor,
    int M, int N, int K,
    int num_threads
) {
    const float* __restrict__ x = x_tensor.data_ptr<float>();
    const uint32_t* __restrict__ w_packed = reinterpret_cast<const uint32_t*>(
        w_packed_tensor.data_ptr<uint8_t>());
    float* __restrict__ y = y_tensor.data_ptr<float>();
    const float* bias = bias_tensor.defined() ? bias_tensor.data_ptr<float>() : nullptr;

    const int K_packed = (K + 15) / 16;

    omp_set_num_threads(num_threads);

    #pragma omp parallel for schedule(static)
    for (int m = 0; m < M; m++) {
        const float* __restrict__ x_row = x + m * K;
        float* __restrict__ y_row = y + m * N;

        int n = 0;
        for (; n <= N - 4; n += 4) {
            const uint32_t* __restrict__ wp0 = w_packed + (n + 0) * K_packed;
            const uint32_t* __restrict__ wp1 = w_packed + (n + 1) * K_packed;
            const uint32_t* __restrict__ wp2 = w_packed + (n + 2) * K_packed;
            const uint32_t* __restrict__ wp3 = w_packed + (n + 3) * K_packed;

            __m512 sum_pos0 = _mm512_setzero_ps();
            __m512 sum_neg0 = _mm512_setzero_ps();
            __m512 sum_pos1 = _mm512_setzero_ps();
            __m512 sum_neg1 = _mm512_setzero_ps();
            __m512 sum_pos2 = _mm512_setzero_ps();
            __m512 sum_neg2 = _mm512_setzero_ps();
            __m512 sum_pos3 = _mm512_setzero_ps();
            __m512 sum_neg3 = _mm512_setzero_ps();

            for (int kp = 0; kp < K_packed; kp++) {
                __m512 x_vals = _mm512_loadu_ps(x_row + kp * 16);

                __mmask16 pos_mask0, neg_mask0;
                __mmask16 pos_mask1, neg_mask1;
                __mmask16 pos_mask2, neg_mask2;
                __mmask16 pos_mask3, neg_mask3;

                extract_masks_fast(wp0[kp], pos_mask0, neg_mask0);
                extract_masks_fast(wp1[kp], pos_mask1, neg_mask1);
                extract_masks_fast(wp2[kp], pos_mask2, neg_mask2);
                extract_masks_fast(wp3[kp], pos_mask3, neg_mask3);

                sum_pos0 = _mm512_mask_add_ps(sum_pos0, pos_mask0, sum_pos0, x_vals);
                sum_neg0 = _mm512_mask_add_ps(sum_neg0, neg_mask0, sum_neg0, x_vals);
                sum_pos1 = _mm512_mask_add_ps(sum_pos1, pos_mask1, sum_pos1, x_vals);
                sum_neg1 = _mm512_mask_add_ps(sum_neg1, neg_mask1, sum_neg1, x_vals);
                sum_pos2 = _mm512_mask_add_ps(sum_pos2, pos_mask2, sum_pos2, x_vals);
                sum_neg2 = _mm512_mask_add_ps(sum_neg2, neg_mask2, sum_neg2, x_vals);
                sum_pos3 = _mm512_mask_add_ps(sum_pos3, pos_mask3, sum_pos3, x_vals);
                sum_neg3 = _mm512_mask_add_ps(sum_neg3, neg_mask3, sum_neg3, x_vals);
            }

            float r0 = hsum_avx512(sum_pos0) - hsum_avx512(sum_neg0);
            float r1 = hsum_avx512(sum_pos1) - hsum_avx512(sum_neg1);
            float r2 = hsum_avx512(sum_pos2) - hsum_avx512(sum_neg2);
            float r3 = hsum_avx512(sum_pos3) - hsum_avx512(sum_neg3);

            if (bias != nullptr) {
                r0 += bias[n + 0];
                r1 += bias[n + 1];
                r2 += bias[n + 2];
                r3 += bias[n + 3];
            }

            y_row[n + 0] = r0;
            y_row[n + 1] = r1;
            y_row[n + 2] = r2;
            y_row[n + 3] = r3;
        }

        for (; n < N; n++) {
            const uint32_t* __restrict__ wp = w_packed + n * K_packed;
            __m512 sum_pos = _mm512_setzero_ps();
            __m512 sum_neg = _mm512_setzero_ps();

            for (int kp = 0; kp < K_packed; kp++) {
                __m512 x_vals = _mm512_loadu_ps(x_row + kp * 16);
                __mmask16 pos_mask, neg_mask;
                extract_masks_fast(wp[kp], pos_mask, neg_mask);
                sum_pos = _mm512_mask_add_ps(sum_pos, pos_mask, sum_pos, x_vals);
                sum_neg = _mm512_mask_add_ps(sum_neg, neg_mask, sum_neg, x_vals);
            }

            float result = hsum_avx512(sum_pos) - hsum_avx512(sum_neg);
            if (bias != nullptr) result += bias[n];
            y_row[n] = result;
        }
    }
}

/**
 * AVX-512 MatMul-free kernel with 2-bit packed weights
 * VERSION 2: Single accumulator approach (subtract neg from pos inline)
 *
 * Key insight: Instead of two accumulators (pos/neg) per output,
 * use one accumulator and do: acc += x (pos) or acc -= x (neg)
 * This halves register pressure and reduces hsum calls.
 */
void matmul_free_avx512_packed(
    torch::Tensor x_tensor,
    torch::Tensor w_packed_tensor,
    torch::Tensor y_tensor,
    torch::Tensor bias_tensor,
    int M, int N, int K,
    int num_threads
) {
    const float* __restrict__ x = x_tensor.data_ptr<float>();
    const uint32_t* __restrict__ w_packed = reinterpret_cast<const uint32_t*>(
        w_packed_tensor.data_ptr<uint8_t>());
    float* __restrict__ y = y_tensor.data_ptr<float>();
    const float* bias = bias_tensor.defined() ? bias_tensor.data_ptr<float>() : nullptr;

    const int K_packed = (K + 15) / 16;

    omp_set_num_threads(num_threads);

    #pragma omp parallel for schedule(static)
    for (int m = 0; m < M; m++) {
        const float* __restrict__ x_row = x + m * K;
        float* __restrict__ y_row = y + m * N;

        // Process 6 outputs at a time (sweet spot for register usage)
        // 6 accumulators = 6 ZMM, leaves room for x_vals and temporaries
        int n = 0;
        for (; n <= N - 6; n += 6) {
            const uint32_t* __restrict__ wp0 = w_packed + (n + 0) * K_packed;
            const uint32_t* __restrict__ wp1 = w_packed + (n + 1) * K_packed;
            const uint32_t* __restrict__ wp2 = w_packed + (n + 2) * K_packed;
            const uint32_t* __restrict__ wp3 = w_packed + (n + 3) * K_packed;
            const uint32_t* __restrict__ wp4 = w_packed + (n + 4) * K_packed;
            const uint32_t* __restrict__ wp5 = w_packed + (n + 5) * K_packed;

            // Single accumulator per output (not pos/neg split)
            __m512 acc0 = _mm512_setzero_ps();
            __m512 acc1 = _mm512_setzero_ps();
            __m512 acc2 = _mm512_setzero_ps();
            __m512 acc3 = _mm512_setzero_ps();
            __m512 acc4 = _mm512_setzero_ps();
            __m512 acc5 = _mm512_setzero_ps();

            for (int kp = 0; kp < K_packed; kp++) {
                __m512 x_vals = _mm512_loadu_ps(x_row + kp * 16);
                __m512 neg_x = _mm512_sub_ps(_mm512_setzero_ps(), x_vals);  // -x

                __mmask16 pm0, nm0, pm1, nm1, pm2, nm2;
                __mmask16 pm3, nm3, pm4, nm4, pm5, nm5;

                extract_masks_fast(wp0[kp], pm0, nm0);
                extract_masks_fast(wp1[kp], pm1, nm1);
                extract_masks_fast(wp2[kp], pm2, nm2);
                extract_masks_fast(wp3[kp], pm3, nm3);
                extract_masks_fast(wp4[kp], pm4, nm4);
                extract_masks_fast(wp5[kp], pm5, nm5);

                // Add +x where positive, add -x where negative
                acc0 = _mm512_mask_add_ps(acc0, pm0, acc0, x_vals);
                acc0 = _mm512_mask_add_ps(acc0, nm0, acc0, neg_x);
                acc1 = _mm512_mask_add_ps(acc1, pm1, acc1, x_vals);
                acc1 = _mm512_mask_add_ps(acc1, nm1, acc1, neg_x);
                acc2 = _mm512_mask_add_ps(acc2, pm2, acc2, x_vals);
                acc2 = _mm512_mask_add_ps(acc2, nm2, acc2, neg_x);
                acc3 = _mm512_mask_add_ps(acc3, pm3, acc3, x_vals);
                acc3 = _mm512_mask_add_ps(acc3, nm3, acc3, neg_x);
                acc4 = _mm512_mask_add_ps(acc4, pm4, acc4, x_vals);
                acc4 = _mm512_mask_add_ps(acc4, nm4, acc4, neg_x);
                acc5 = _mm512_mask_add_ps(acc5, pm5, acc5, x_vals);
                acc5 = _mm512_mask_add_ps(acc5, nm5, acc5, neg_x);
            }

            // Only 6 hsum calls instead of 12
            float r0 = hsum_avx512(acc0);
            float r1 = hsum_avx512(acc1);
            float r2 = hsum_avx512(acc2);
            float r3 = hsum_avx512(acc3);
            float r4 = hsum_avx512(acc4);
            float r5 = hsum_avx512(acc5);

            if (bias != nullptr) {
                r0 += bias[n + 0]; r1 += bias[n + 1];
                r2 += bias[n + 2]; r3 += bias[n + 3];
                r4 += bias[n + 4]; r5 += bias[n + 5];
            }

            y_row[n + 0] = r0; y_row[n + 1] = r1;
            y_row[n + 2] = r2; y_row[n + 3] = r3;
            y_row[n + 4] = r4; y_row[n + 5] = r5;
        }

        // Handle remaining outputs
        for (; n < N; n++) {
            const uint32_t* __restrict__ wp = w_packed + n * K_packed;
            __m512 acc = _mm512_setzero_ps();

            for (int kp = 0; kp < K_packed; kp++) {
                __m512 x_vals = _mm512_loadu_ps(x_row + kp * 16);
                __m512 neg_x = _mm512_sub_ps(_mm512_setzero_ps(), x_vals);

                __mmask16 pos_mask, neg_mask;
                extract_masks_fast(wp[kp], pos_mask, neg_mask);

                acc = _mm512_mask_add_ps(acc, pos_mask, acc, x_vals);
                acc = _mm512_mask_add_ps(acc, neg_mask, acc, neg_x);
            }

            float result = hsum_avx512(acc);
            if (bias != nullptr) result += bias[n];
            y_row[n] = result;
        }
    }
}

/**
 * Alternative: AVX-512 with unpacked float32 weights (simpler, for comparison)
 */
void matmul_free_avx512_float(
    torch::Tensor x_tensor,
    torch::Tensor w_tensor,
    torch::Tensor y_tensor,
    torch::Tensor bias_tensor,
    int M, int N, int K,
    int num_threads
) {
    const float* __restrict__ x = x_tensor.data_ptr<float>();
    const float* __restrict__ w = w_tensor.data_ptr<float>();
    float* __restrict__ y = y_tensor.data_ptr<float>();
    const float* bias = bias_tensor.defined() ? bias_tensor.data_ptr<float>() : nullptr;

    const int K_SIMD = 16;  // AVX-512 width
    const int K_SIMD_END = (K / K_SIMD) * K_SIMD;

    omp_set_num_threads(num_threads);

    const __m512 zero = _mm512_setzero_ps();

    #pragma omp parallel for schedule(static)
    for (int m = 0; m < M; m++) {
        const float* __restrict__ x_row = x + m * K;
        float* __restrict__ y_row = y + m * N;

        // Process 4 outputs at a time
        int n = 0;
        for (; n <= N - 4; n += 4) {
            const float* __restrict__ w0 = w + (n + 0) * K;
            const float* __restrict__ w1 = w + (n + 1) * K;
            const float* __restrict__ w2 = w + (n + 2) * K;
            const float* __restrict__ w3 = w + (n + 3) * K;

            __m512 sum_pos0 = _mm512_setzero_ps();
            __m512 sum_neg0 = _mm512_setzero_ps();
            __m512 sum_pos1 = _mm512_setzero_ps();
            __m512 sum_neg1 = _mm512_setzero_ps();
            __m512 sum_pos2 = _mm512_setzero_ps();
            __m512 sum_neg2 = _mm512_setzero_ps();
            __m512 sum_pos3 = _mm512_setzero_ps();
            __m512 sum_neg3 = _mm512_setzero_ps();

            for (int k = 0; k < K_SIMD_END; k += K_SIMD) {
                __m512 x_vals = _mm512_loadu_ps(x_row + k);

                // Output 0
                __m512 wv0 = _mm512_loadu_ps(w0 + k);
                __mmask16 mp0 = _mm512_cmp_ps_mask(wv0, zero, _CMP_GT_OQ);
                __mmask16 mn0 = _mm512_cmp_ps_mask(wv0, zero, _CMP_LT_OQ);
                sum_pos0 = _mm512_mask_add_ps(sum_pos0, mp0, sum_pos0, x_vals);
                sum_neg0 = _mm512_mask_add_ps(sum_neg0, mn0, sum_neg0, x_vals);

                // Output 1
                __m512 wv1 = _mm512_loadu_ps(w1 + k);
                __mmask16 mp1 = _mm512_cmp_ps_mask(wv1, zero, _CMP_GT_OQ);
                __mmask16 mn1 = _mm512_cmp_ps_mask(wv1, zero, _CMP_LT_OQ);
                sum_pos1 = _mm512_mask_add_ps(sum_pos1, mp1, sum_pos1, x_vals);
                sum_neg1 = _mm512_mask_add_ps(sum_neg1, mn1, sum_neg1, x_vals);

                // Output 2
                __m512 wv2 = _mm512_loadu_ps(w2 + k);
                __mmask16 mp2 = _mm512_cmp_ps_mask(wv2, zero, _CMP_GT_OQ);
                __mmask16 mn2 = _mm512_cmp_ps_mask(wv2, zero, _CMP_LT_OQ);
                sum_pos2 = _mm512_mask_add_ps(sum_pos2, mp2, sum_pos2, x_vals);
                sum_neg2 = _mm512_mask_add_ps(sum_neg2, mn2, sum_neg2, x_vals);

                // Output 3
                __m512 wv3 = _mm512_loadu_ps(w3 + k);
                __mmask16 mp3 = _mm512_cmp_ps_mask(wv3, zero, _CMP_GT_OQ);
                __mmask16 mn3 = _mm512_cmp_ps_mask(wv3, zero, _CMP_LT_OQ);
                sum_pos3 = _mm512_mask_add_ps(sum_pos3, mp3, sum_pos3, x_vals);
                sum_neg3 = _mm512_mask_add_ps(sum_neg3, mn3, sum_neg3, x_vals);
            }

            // Horizontal reduce
            float r0 = hsum_avx512(sum_pos0) - hsum_avx512(sum_neg0);
            float r1 = hsum_avx512(sum_pos1) - hsum_avx512(sum_neg1);
            float r2 = hsum_avx512(sum_pos2) - hsum_avx512(sum_neg2);
            float r3 = hsum_avx512(sum_pos3) - hsum_avx512(sum_neg3);

            // Scalar tail
            for (int k = K_SIMD_END; k < K; k++) {
                float xv = x_row[k];
                if (w0[k] > 0.0f) r0 += xv; else if (w0[k] < 0.0f) r0 -= xv;
                if (w1[k] > 0.0f) r1 += xv; else if (w1[k] < 0.0f) r1 -= xv;
                if (w2[k] > 0.0f) r2 += xv; else if (w2[k] < 0.0f) r2 -= xv;
                if (w3[k] > 0.0f) r3 += xv; else if (w3[k] < 0.0f) r3 -= xv;
            }

            if (bias != nullptr) {
                r0 += bias[n + 0];
                r1 += bias[n + 1];
                r2 += bias[n + 2];
                r3 += bias[n + 3];
            }

            y_row[n + 0] = r0;
            y_row[n + 1] = r1;
            y_row[n + 2] = r2;
            y_row[n + 3] = r3;
        }

        // Handle remaining outputs
        for (; n < N; n++) {
            const float* __restrict__ w_row = w + n * K;

            __m512 sum_pos = _mm512_setzero_ps();
            __m512 sum_neg = _mm512_setzero_ps();

            for (int k = 0; k < K_SIMD_END; k += K_SIMD) {
                __m512 x_vals = _mm512_loadu_ps(x_row + k);
                __m512 w_vals = _mm512_loadu_ps(w_row + k);

                __mmask16 mp = _mm512_cmp_ps_mask(w_vals, zero, _CMP_GT_OQ);
                __mmask16 mn = _mm512_cmp_ps_mask(w_vals, zero, _CMP_LT_OQ);

                sum_pos = _mm512_mask_add_ps(sum_pos, mp, sum_pos, x_vals);
                sum_neg = _mm512_mask_add_ps(sum_neg, mn, sum_neg, x_vals);
            }

            float result = hsum_avx512(sum_pos) - hsum_avx512(sum_neg);

            for (int k = K_SIMD_END; k < K; k++) {
                float xv = x_row[k];
                float wv = w_row[k];
                if (wv > 0.0f) result += xv;
                else if (wv < 0.0f) result -= xv;
            }

            if (bias != nullptr) {
                result += bias[n];
            }
            y_row[n] = result;
        }
    }
}

// Python bindings
PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("has_avx512_support", &has_avx512_support, "Check AVX-512 support");
    m.def("pack_weights_2bit_avx512", &pack_weights_2bit_avx512,
          "Pack ternary weights to 2-bit format for AVX-512");
    m.def("matmul_free_avx512_packed", &matmul_free_avx512_packed,
          "AVX-512 MatMul-free with 2-bit packed weights (optimized v2)");
    m.def("matmul_free_avx512_packed_v1", &matmul_free_avx512_packed_v1,
          "AVX-512 MatMul-free with 2-bit packed weights (basic v1)");
    m.def("matmul_free_avx512_float", &matmul_free_avx512_float,
          "AVX-512 MatMul-free with float32 weights");
}
