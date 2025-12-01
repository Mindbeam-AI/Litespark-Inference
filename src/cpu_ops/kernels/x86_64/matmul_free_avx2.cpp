/*
 * AVX2-Optimized MatMul-Free Kernel for x86_64
 *
 * Implements true MatMul-free operations using only addition/subtraction
 * for ternary weights {-1, 0, 1}, optimized with AVX2 instructions.
 */

#include <immintrin.h>
#include <cstring>
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
 * Horizontal sum of __m256 vector
 */
inline float hsum_avx(__m256 v) {
    __m128 lo = _mm256_castps256_ps128(v);
    __m128 hi = _mm256_extractf128_ps(v, 1);
    __m128 sum = _mm_add_ps(lo, hi);
    __m128 shuf = _mm_movehdup_ps(sum);
    __m128 sums = _mm_add_ps(sum, shuf);
    shuf = _mm_movehl_ps(shuf, sums);
    sums = _mm_add_ss(sums, shuf);
    return _mm_cvtss_f32(sums);
}

/**
 * AVX2-optimized MatMul-free kernel with K-blocking for cache efficiency
 *
 * Computes: y[m,n] = sum(x[m,k] where w[n,k]==1) - sum(x[m,k] where w[n,k]==-1)
 *
 * Key optimizations:
 * - 4-way output blocking (minimal register pressure)
 * - K-blocking for L1/L2 cache efficiency
 * - Process multiple K values per inner loop iteration
 * - _mm256_and_ps with mask for fast predication
 *
 * @param x_tensor Input matrix [M, K] (row-major)
 * @param w_tensor Ternary weight matrix [N, K] (row-major, values in {-1, 0, 1})
 * @param y_tensor Output matrix [M, N] (row-major)
 * @param bias_tensor Optional bias vector [N]
 * @param M Number of input rows
 * @param N Number of output features
 * @param K Number of input features
 * @param num_threads Number of OpenMP threads to use
 */
void matmul_free_avx2(
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

    // Tuned parameters for Intel Ice Lake
    const int K_SIMD = 8;          // AVX2 width
    const int N_BLOCK = 4;         // Output blocking - reduced for more registers per output
    const int K_BLOCK = 32;        // Process 32 K values (4 AVX2 loads) per inner iteration
    const int K_SIMD_END = (K / K_SIMD) * K_SIMD;

    omp_set_num_threads(num_threads);

    // Parallelize over output rows
    #pragma omp parallel for schedule(static)
    for (int m = 0; m < M; m++) {
        const float* __restrict__ x_row = x + m * K;
        float* __restrict__ y_row = y + m * N;

        // Process N_BLOCK outputs together
        int n = 0;
        for (; n <= N - N_BLOCK; n += N_BLOCK) {
            // 4 accumulators for pos/neg (8 total YMM registers)
            __m256 sum_pos0 = _mm256_setzero_ps();
            __m256 sum_pos1 = _mm256_setzero_ps();
            __m256 sum_pos2 = _mm256_setzero_ps();
            __m256 sum_pos3 = _mm256_setzero_ps();
            __m256 sum_neg0 = _mm256_setzero_ps();
            __m256 sum_neg1 = _mm256_setzero_ps();
            __m256 sum_neg2 = _mm256_setzero_ps();
            __m256 sum_neg3 = _mm256_setzero_ps();

            const __m256 zero = _mm256_setzero_ps();

            // Pointers to weight rows
            const float* __restrict__ w0 = w + (n + 0) * K;
            const float* __restrict__ w1 = w + (n + 1) * K;
            const float* __restrict__ w2 = w + (n + 2) * K;
            const float* __restrict__ w3 = w + (n + 3) * K;

            // Main loop: process K_BLOCK elements at a time for better cache use
            int k = 0;
            for (; k <= K_SIMD_END - K_BLOCK; k += K_BLOCK) {
                // Prefetch next K_BLOCK
                _mm_prefetch((const char*)(x_row + k + K_BLOCK), _MM_HINT_T0);
                _mm_prefetch((const char*)(w0 + k + K_BLOCK), _MM_HINT_T0);
                _mm_prefetch((const char*)(w1 + k + K_BLOCK), _MM_HINT_T0);
                _mm_prefetch((const char*)(w2 + k + K_BLOCK), _MM_HINT_T0);
                _mm_prefetch((const char*)(w3 + k + K_BLOCK), _MM_HINT_T0);

                // Unroll 4x to process K_BLOCK (32) elements
                for (int kk = 0; kk < K_BLOCK; kk += K_SIMD) {
                    const __m256 x_vals = _mm256_loadu_ps(x_row + k + kk);

                    // Output 0
                    __m256 wv0 = _mm256_loadu_ps(w0 + k + kk);
                    __m256 mp0 = _mm256_cmp_ps(wv0, zero, _CMP_GT_OQ);
                    __m256 mn0 = _mm256_cmp_ps(wv0, zero, _CMP_LT_OQ);
                    sum_pos0 = _mm256_add_ps(sum_pos0, _mm256_and_ps(x_vals, mp0));
                    sum_neg0 = _mm256_add_ps(sum_neg0, _mm256_and_ps(x_vals, mn0));

                    // Output 1
                    __m256 wv1 = _mm256_loadu_ps(w1 + k + kk);
                    __m256 mp1 = _mm256_cmp_ps(wv1, zero, _CMP_GT_OQ);
                    __m256 mn1 = _mm256_cmp_ps(wv1, zero, _CMP_LT_OQ);
                    sum_pos1 = _mm256_add_ps(sum_pos1, _mm256_and_ps(x_vals, mp1));
                    sum_neg1 = _mm256_add_ps(sum_neg1, _mm256_and_ps(x_vals, mn1));

                    // Output 2
                    __m256 wv2 = _mm256_loadu_ps(w2 + k + kk);
                    __m256 mp2 = _mm256_cmp_ps(wv2, zero, _CMP_GT_OQ);
                    __m256 mn2 = _mm256_cmp_ps(wv2, zero, _CMP_LT_OQ);
                    sum_pos2 = _mm256_add_ps(sum_pos2, _mm256_and_ps(x_vals, mp2));
                    sum_neg2 = _mm256_add_ps(sum_neg2, _mm256_and_ps(x_vals, mn2));

                    // Output 3
                    __m256 wv3 = _mm256_loadu_ps(w3 + k + kk);
                    __m256 mp3 = _mm256_cmp_ps(wv3, zero, _CMP_GT_OQ);
                    __m256 mn3 = _mm256_cmp_ps(wv3, zero, _CMP_LT_OQ);
                    sum_pos3 = _mm256_add_ps(sum_pos3, _mm256_and_ps(x_vals, mp3));
                    sum_neg3 = _mm256_add_ps(sum_neg3, _mm256_and_ps(x_vals, mn3));
                }
            }

            // Handle remaining SIMD iterations
            for (; k < K_SIMD_END; k += K_SIMD) {
                const __m256 x_vals = _mm256_loadu_ps(x_row + k);

                __m256 wv0 = _mm256_loadu_ps(w0 + k);
                sum_pos0 = _mm256_add_ps(sum_pos0, _mm256_and_ps(x_vals, _mm256_cmp_ps(wv0, zero, _CMP_GT_OQ)));
                sum_neg0 = _mm256_add_ps(sum_neg0, _mm256_and_ps(x_vals, _mm256_cmp_ps(wv0, zero, _CMP_LT_OQ)));

                __m256 wv1 = _mm256_loadu_ps(w1 + k);
                sum_pos1 = _mm256_add_ps(sum_pos1, _mm256_and_ps(x_vals, _mm256_cmp_ps(wv1, zero, _CMP_GT_OQ)));
                sum_neg1 = _mm256_add_ps(sum_neg1, _mm256_and_ps(x_vals, _mm256_cmp_ps(wv1, zero, _CMP_LT_OQ)));

                __m256 wv2 = _mm256_loadu_ps(w2 + k);
                sum_pos2 = _mm256_add_ps(sum_pos2, _mm256_and_ps(x_vals, _mm256_cmp_ps(wv2, zero, _CMP_GT_OQ)));
                sum_neg2 = _mm256_add_ps(sum_neg2, _mm256_and_ps(x_vals, _mm256_cmp_ps(wv2, zero, _CMP_LT_OQ)));

                __m256 wv3 = _mm256_loadu_ps(w3 + k);
                sum_pos3 = _mm256_add_ps(sum_pos3, _mm256_and_ps(x_vals, _mm256_cmp_ps(wv3, zero, _CMP_GT_OQ)));
                sum_neg3 = _mm256_add_ps(sum_neg3, _mm256_and_ps(x_vals, _mm256_cmp_ps(wv3, zero, _CMP_LT_OQ)));
            }

            // Horizontal reduce and store
            float r0 = hsum_avx(sum_pos0) - hsum_avx(sum_neg0);
            float r1 = hsum_avx(sum_pos1) - hsum_avx(sum_neg1);
            float r2 = hsum_avx(sum_pos2) - hsum_avx(sum_neg2);
            float r3 = hsum_avx(sum_pos3) - hsum_avx(sum_neg3);

            // Scalar tail for K
            for (int kk = K_SIMD_END; kk < K; kk++) {
                float xv = x_row[kk];
                if (w0[kk] > 0.0f) r0 += xv; else if (w0[kk] < 0.0f) r0 -= xv;
                if (w1[kk] > 0.0f) r1 += xv; else if (w1[kk] < 0.0f) r1 -= xv;
                if (w2[kk] > 0.0f) r2 += xv; else if (w2[kk] < 0.0f) r2 -= xv;
                if (w3[kk] > 0.0f) r3 += xv; else if (w3[kk] < 0.0f) r3 -= xv;
            }

            // Add bias and store
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

        // Handle remaining outputs one at a time
        for (; n < N; n++) {
            const float* __restrict__ w_row = w + n * K;

            __m256 sum_pos_vec = _mm256_setzero_ps();
            __m256 sum_neg_vec = _mm256_setzero_ps();
            const __m256 zero = _mm256_setzero_ps();

            for (int k = 0; k < K_SIMD_END; k += K_SIMD) {
                const __m256 x_vals = _mm256_loadu_ps(x_row + k);
                const __m256 w_vals = _mm256_loadu_ps(w_row + k);

                sum_pos_vec = _mm256_add_ps(sum_pos_vec, _mm256_and_ps(x_vals, _mm256_cmp_ps(w_vals, zero, _CMP_GT_OQ)));
                sum_neg_vec = _mm256_add_ps(sum_neg_vec, _mm256_and_ps(x_vals, _mm256_cmp_ps(w_vals, zero, _CMP_LT_OQ)));
            }

            float result = hsum_avx(sum_pos_vec) - hsum_avx(sum_neg_vec);

            for (int k = K_SIMD_END; k < K; k++) {
                float x_val = x_row[k];
                float w_val = w_row[k];
                if (w_val > 0.0f) result += x_val;
                else if (w_val < 0.0f) result -= x_val;
            }

            if (bias != nullptr) {
                result += bias[n];
            }
            y_row[n] = result;
        }
    }
}

/**
 * Check if AVX2 is supported on this CPU
 */
bool has_avx2_support() {
#ifdef _MSC_VER
    // Windows/MSVC
    int cpuinfo[4];
    __cpuid(cpuinfo, 7);
    return (cpuinfo[1] & (1 << 5)) != 0;  // Check AVX2 bit (EBX bit 5)
#else
    // Linux/macOS with GCC/Clang
    unsigned int eax, ebx, ecx, edx;
    if (__get_cpuid_count(7, 0, &eax, &ebx, &ecx, &edx)) {
        return (ebx & (1 << 5)) != 0;  // Check AVX2 bit (EBX bit 5)
    }
    return false;
#endif
}

// Note: PYBIND11_MODULE is defined in matmul_free_generic.cpp which serves as the main entry point
// The generic module conditionally exports AVX2 functions when HAS_AVX2_SUPPORT is defined
