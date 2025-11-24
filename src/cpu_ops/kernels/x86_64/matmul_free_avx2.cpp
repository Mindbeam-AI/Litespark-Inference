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

extern "C" {

/**
 * AVX2-optimized MatMul-free kernel
 * 
 * Computes: y[m,n] = sum(x[m,k] where w[n,k]==1) - sum(x[m,k] where w[n,k]==-1)
 * 
 * @param x Input matrix [M, K] (row-major)
 * @param w Ternary weight matrix [N, K] (row-major, values in {-1, 0, 1})
 * @param y Output matrix [M, N] (row-major)
 * @param bias Optional bias vector [N]
 * @param M Number of input rows
 * @param N Number of output features
 * @param K Number of input features
 * @param num_threads Number of OpenMP threads to use
 */
void matmul_free_avx2(
    const float* x,
    const float* w,
    float* y,
    const float* bias,
    int M, int N, int K,
    int num_threads
) {
    // Set number of threads
    omp_set_num_threads(num_threads);
    
    // Process 8 output features at a time (AVX2 width)
    const int N_BLOCK = 8;
    const int K_BLOCK = 32;  // Process K in blocks for better cache usage
    
    #pragma omp parallel for
    for (int m = 0; m < M; m++) {
        for (int n = 0; n < N; n += N_BLOCK) {
            // Handle remaining features if N is not divisible by N_BLOCK
            int n_end = std::min(n + N_BLOCK, N);
            int n_width = n_end - n;
            
            // Initialize accumulators for positive and negative sums
            __m256 sum_pos[1] = {_mm256_setzero_ps()};
            __m256 sum_neg[1] = {_mm256_setzero_ps()};
            
            // Process K dimension in blocks
            for (int k = 0; k < K; k += K_BLOCK) {
                int k_end = std::min(k + K_BLOCK, K);
                
                for (int kk = k; kk < k_end; kk++) {
                    // Load input value (broadcast to all lanes)
                    __m256 x_val = _mm256_set1_ps(x[m * K + kk]);
                    
                    // Load weights for current output features
                    __m256 w_vals;
                    if (n_width == N_BLOCK) {
                        w_vals = _mm256_loadu_ps(&w[n * K + kk * N + n]);
                    } else {
                        // Handle partial load for remaining features
                        float w_temp[8] = {0};
                        for (int i = 0; i < n_width; i++) {
                            w_temp[i] = w[(n + i) * K + kk];
                        }
                        w_vals = _mm256_loadu_ps(w_temp);
                    }
                    
                    // Create masks for positive and negative weights
                    __m256 zero = _mm256_setzero_ps();
                    __m256 mask_pos = _mm256_cmp_ps(w_vals, zero, _CMP_GT_OQ);
                    __m256 mask_neg = _mm256_cmp_ps(w_vals, zero, _CMP_LT_OQ);
                    
                    // Accumulate: add where weight > 0, subtract where weight < 0
                    sum_pos[0] = _mm256_fmadd_ps(_mm256_and_ps(mask_pos, x_val), 
                                                _mm256_set1_ps(1.0f), sum_pos[0]);
                    sum_neg[0] = _mm256_fmadd_ps(_mm256_and_ps(mask_neg, x_val), 
                                                _mm256_set1_ps(1.0f), sum_neg[0]);
                }
            }
            
            // Compute final result: sum_pos - sum_neg
            __m256 result = _mm256_sub_ps(sum_pos[0], sum_neg[0]);
            
            // Add bias if provided
            if (bias != nullptr) {
                __m256 bias_vals;
                if (n_width == N_BLOCK) {
                    bias_vals = _mm256_loadu_ps(&bias[n]);
                } else {
                    float bias_temp[8] = {0};
                    for (int i = 0; i < n_width; i++) {
                        bias_temp[i] = bias[n + i];
                    }
                    bias_vals = _mm256_loadu_ps(bias_temp);
                }
                result = _mm256_add_ps(result, bias_vals);
            }
            
            // Store result
            if (n_width == N_BLOCK) {
                _mm256_storeu_ps(&y[m * N + n], result);
            } else {
                float result_temp[8];
                _mm256_storeu_ps(result_temp, result);
                for (int i = 0; i < n_width; i++) {
                    y[m * N + n + i] = result_temp[i];
                }
            }
        }
    }
}

/**
 * Check if AVX2 is supported on this CPU
 */
bool has_avx2_support() {
    // Simple CPUID check for AVX2
    int cpuinfo[4];
    __cpuid(cpuinfo, 7);
    return (cpuinfo[1] & (1 << 5)) != 0;  // Check AVX2 bit
}

} // extern "C"
