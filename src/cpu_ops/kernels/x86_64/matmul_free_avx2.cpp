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
 * AVX2-optimized MatMul-free kernel with output blocking
 *
 * Computes: y[m,n] = sum(x[m,k] where w[n,k]==1) - sum(x[m,k] where w[n,k]==-1)
 *
 * Key optimizations:
 * - 8-way output blocking (reduced from 16 to avoid register spilling)
 * - _mm256_and_ps with mask for faster predication than blendv
 * - Prefetching for memory latency hiding
 * - Sequential memory access patterns
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
    const float* x = x_tensor.data_ptr<float>();
    const float* w = w_tensor.data_ptr<float>();
    float* y = y_tensor.data_ptr<float>();
    const float* bias = bias_tensor.defined() ? bias_tensor.data_ptr<float>() : nullptr;

    // AVX2: 8-wide SIMD, 8-way output blocking (reduced to avoid register pressure)
    const int K_SIMD = 8;
    const int K_SIMD_END = (K / K_SIMD) * K_SIMD;
    const int N_BLOCK = 8;  // Reduced from 16 to fit in registers
    const int K_PREFETCH = 128;  // Prefetch distance

    omp_set_num_threads(num_threads);

    // Parallelize over output rows with dynamic scheduling
    #pragma omp parallel for schedule(dynamic, 4)
    for (int m = 0; m < M; m++) {
        const float* x_row = x + m * K;
        float* y_row = y + m * N;

        // Process N_BLOCK outputs together for better instruction-level parallelism
        int n = 0;
        for (; n <= N - N_BLOCK; n += N_BLOCK) {
            // 8 separate positive/negative accumulators (16 total - fits in YMM registers)
            __m256 sum_pos[N_BLOCK], sum_neg[N_BLOCK];

            #pragma unroll
            for (int i = 0; i < N_BLOCK; i++) {
                sum_pos[i] = _mm256_setzero_ps();
                sum_neg[i] = _mm256_setzero_ps();
            }

            const __m256 zero = _mm256_setzero_ps();

            // SIMD loop over K dimension
            for (int k = 0; k < K_SIMD_END; k += K_SIMD) {
                // Prefetch future data
                if (k + K_PREFETCH < K) {
                    _mm_prefetch((const char*)(x_row + k + K_PREFETCH), _MM_HINT_T0);
                }

                // Load input once per K iteration
                const __m256 x_vals = _mm256_loadu_ps(x_row + k);

                // Process all N_BLOCK outputs with this input - manually unrolled
                #pragma unroll
                for (int i = 0; i < N_BLOCK; i++) {
                    // Load 8 weight values
                    const __m256 w_vals = _mm256_loadu_ps(w + (n + i) * K + k);

                    // Create masks for positive and negative weights
                    // Use _mm256_and_ps instead of blendv - faster on most Intel CPUs
                    const __m256 mask_pos = _mm256_cmp_ps(w_vals, zero, _CMP_GT_OQ);  // w > 0
                    const __m256 mask_neg = _mm256_cmp_ps(w_vals, zero, _CMP_LT_OQ);  // w < 0

                    // TRUE matmul-free: conditional accumulation using AND with mask
                    // _mm256_and_ps is 1 cycle latency vs blendv's 2 cycles
                    sum_pos[i] = _mm256_add_ps(sum_pos[i], _mm256_and_ps(x_vals, mask_pos));
                    sum_neg[i] = _mm256_add_ps(sum_neg[i], _mm256_and_ps(x_vals, mask_neg));
                }
            }

            // Reduce and write results for this block
            for (int i = 0; i < N_BLOCK; i++) {
                float sum_p = hsum_avx(sum_pos[i]);
                float sum_n = hsum_avx(sum_neg[i]);

                // Handle scalar tail
                for (int k = K_SIMD_END; k < K; k++) {
                    float x_val = x_row[k];
                    float w_val = w[(n + i) * K + k];

                    if (w_val > 0.0f) {
                        sum_p += x_val;
                    } else if (w_val < 0.0f) {
                        sum_n += x_val;
                    }
                }

                float result = sum_p - sum_n;
                if (bias != nullptr) {
                    result += bias[n + i];
                }
                y_row[n + i] = result;
            }
        }

        // Handle remaining outputs (scalar tail)
        for (; n < N; n++) {
            const float* w_row = w + n * K;

            __m256 sum_pos_vec = _mm256_setzero_ps();
            __m256 sum_neg_vec = _mm256_setzero_ps();
            const __m256 zero = _mm256_setzero_ps();

            for (int k = 0; k < K_SIMD_END; k += K_SIMD) {
                const __m256 x_vals = _mm256_loadu_ps(x_row + k);
                const __m256 w_vals = _mm256_loadu_ps(w_row + k);

                const __m256 mask_pos = _mm256_cmp_ps(w_vals, zero, _CMP_GT_OQ);
                const __m256 mask_neg = _mm256_cmp_ps(w_vals, zero, _CMP_LT_OQ);

                sum_pos_vec = _mm256_add_ps(sum_pos_vec, _mm256_and_ps(x_vals, mask_pos));
                sum_neg_vec = _mm256_add_ps(sum_neg_vec, _mm256_and_ps(x_vals, mask_neg));
            }

            float sum_pos = hsum_avx(sum_pos_vec);
            float sum_neg = hsum_avx(sum_neg_vec);

            for (int k = K_SIMD_END; k < K; k++) {
                float x_val = x_row[k];
                float w_val = w_row[k];

                if (w_val > 0.0f) {
                    sum_pos += x_val;
                } else if (w_val < 0.0f) {
                    sum_neg += x_val;
                }
            }

            float result = sum_pos - sum_neg;
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
