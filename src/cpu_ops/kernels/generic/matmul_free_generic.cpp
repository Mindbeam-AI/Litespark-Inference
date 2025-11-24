/*
 * Generic MatMul-Free Kernel (Fallback Implementation)
 * 
 * Implements true MatMul-free operations using only addition/subtraction
 * for ternary weights {-1, 0, 1}, without SIMD optimizations.
 * 
 * This serves as a fallback for architectures without AVX2/NEON support.
 */

#include <cstring>
#include <algorithm>
#include <cmath>
#include <omp.h>

extern "C" {

/**
 * Generic MatMul-free kernel (no SIMD)
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
void matmul_free_generic(
    const float* x,
    const float* w,
    float* y,
    const float* bias,
    int M, int N, int K,
    int num_threads
) {
    // Set number of threads
    omp_set_num_threads(num_threads);
    
    // Block sizes for cache efficiency
    const int M_BLOCK = 32;
    const int N_BLOCK = 64;
    const int K_BLOCK = 128;
    
    #pragma omp parallel for collapse(2)
    for (int mb = 0; mb < M; mb += M_BLOCK) {
        for (int nb = 0; nb < N; nb += N_BLOCK) {
            int m_end = std::min(mb + M_BLOCK, M);
            int n_end = std::min(nb + N_BLOCK, N);
            
            for (int m = mb; m < m_end; m++) {
                for (int n = nb; n < n_end; n++) {
                    float sum_pos = 0.0f;
                    float sum_neg = 0.0f;
                    
                    // Process K dimension in blocks for better cache usage
                    for (int kb = 0; kb < K; kb += K_BLOCK) {
                        int k_end = std::min(kb + K_BLOCK, K);
                        
                        for (int k = kb; k < k_end; k++) {
                            float x_val = x[m * K + k];
                            float w_val = w[n * K + k];
                            
                            // True MatMul-free: only add/subtract operations
                            if (w_val > 0.5f) {
                                sum_pos += x_val;
                            } else if (w_val < -0.5f) {
                                sum_neg += x_val;
                            }
                            // If w_val is ~0, do nothing (skip multiplication)
                        }
                    }
                    
                    // Compute result: sum_pos - sum_neg
                    float result = sum_pos - sum_neg;
                    
                    // Add bias if provided
                    if (bias != nullptr) {
                        result += bias[n];
                    }
                    
                    y[m * N + n] = result;
                }
            }
        }
    }
}

/**
 * Optimized version with manual loop unrolling
 */
void matmul_free_generic_unrolled(
    const float* x,
    const float* w,
    float* y,
    const float* bias,
    int M, int N, int K,
    int num_threads
) {
    omp_set_num_threads(num_threads);
    
    #pragma omp parallel for
    for (int m = 0; m < M; m++) {
        for (int n = 0; n < N; n++) {
            float sum_pos = 0.0f;
            float sum_neg = 0.0f;
            
            // Unroll K loop by 4 for better instruction-level parallelism
            int k = 0;
            for (; k < K - 3; k += 4) {
                float x0 = x[m * K + k + 0];
                float x1 = x[m * K + k + 1];
                float x2 = x[m * K + k + 2];
                float x3 = x[m * K + k + 3];
                
                float w0 = w[n * K + k + 0];
                float w1 = w[n * K + k + 1];
                float w2 = w[n * K + k + 2];
                float w3 = w[n * K + k + 3];
                
                // Process 4 elements at once
                if (w0 > 0.5f) sum_pos += x0; else if (w0 < -0.5f) sum_neg += x0;
                if (w1 > 0.5f) sum_pos += x1; else if (w1 < -0.5f) sum_neg += x1;
                if (w2 > 0.5f) sum_pos += x2; else if (w2 < -0.5f) sum_neg += x2;
                if (w3 > 0.5f) sum_pos += x3; else if (w3 < -0.5f) sum_neg += x3;
            }
            
            // Handle remaining elements
            for (; k < K; k++) {
                float x_val = x[m * K + k];
                float w_val = w[n * K + k];
                
                if (w_val > 0.5f) {
                    sum_pos += x_val;
                } else if (w_val < -0.5f) {
                    sum_neg += x_val;
                }
            }
            
            // Compute result
            float result = sum_pos - sum_neg;
            if (bias != nullptr) {
                result += bias[n];
            }
            
            y[m * N + n] = result;
        }
    }
}

/**
 * Simple feature detection (always returns true for generic)
 */
bool has_generic_support() {
    return true;
}

/**
 * Get optimal number of threads for the current system
 */
int get_optimal_threads() {
    int max_threads = omp_get_max_threads();
    // Use all available threads, but cap at reasonable limit
    return std::min(max_threads, 16);
}

} // extern "C"
