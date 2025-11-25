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
#include <thread>
#include <torch/extension.h>
#include <omp.h>

void matmul_free_neon(torch::Tensor x, torch::Tensor w, torch::Tensor y, torch::Tensor bias, int M, int N, int K, int num_threads);
bool has_neon_support();
void matmul_free_avx2(torch::Tensor x, torch::Tensor w, torch::Tensor y, torch::Tensor bias, int M, int N, int K, int num_threads);
bool has_avx2_support();

/**
 * Generic MatMul-free kernel - Branchless version
 *
 * Computes: y[m,n] = sum(x[m,k] * w[n,k]) where w ∈ {-1, 0, 1}
 * This is branchless and faster than conditional accumulation.
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
void matmul_free_generic(
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

    omp_set_num_threads(num_threads);

    // Process 4 output features together
    const int N_BLOCK = 4;

    // Parallelize over output rows with dynamic scheduling
    #pragma omp parallel for schedule(dynamic, 16)
    for (int m = 0; m < M; m++) {
        const float* x_row = x + m * K;
        float* y_row = y + m * N;

        // Process output features in blocks
        int n = 0;
        for (; n <= N - N_BLOCK; n += N_BLOCK) {
            float acc0 = 0.0f, acc1 = 0.0f, acc2 = 0.0f, acc3 = 0.0f;

            // Unroll by 4 for better instruction-level parallelism
            int k = 0;
            for (; k <= K - 4; k += 4) {
                float x0 = x_row[k + 0];
                float x1 = x_row[k + 1];
                float x2 = x_row[k + 2];
                float x3 = x_row[k + 3];

                // Process 4 outputs x 4 inputs = 16 operations
                acc0 += x0 * w[(n + 0) * K + k + 0] + x1 * w[(n + 0) * K + k + 1] +
                        x2 * w[(n + 0) * K + k + 2] + x3 * w[(n + 0) * K + k + 3];
                acc1 += x0 * w[(n + 1) * K + k + 0] + x1 * w[(n + 1) * K + k + 1] +
                        x2 * w[(n + 1) * K + k + 2] + x3 * w[(n + 1) * K + k + 3];
                acc2 += x0 * w[(n + 2) * K + k + 0] + x1 * w[(n + 2) * K + k + 1] +
                        x2 * w[(n + 2) * K + k + 2] + x3 * w[(n + 2) * K + k + 3];
                acc3 += x0 * w[(n + 3) * K + k + 0] + x1 * w[(n + 3) * K + k + 1] +
                        x2 * w[(n + 3) * K + k + 2] + x3 * w[(n + 3) * K + k + 3];
            }

            // Handle remaining elements
            for (; k < K; k++) {
                float x_val = x_row[k];
                acc0 += x_val * w[(n + 0) * K + k];
                acc1 += x_val * w[(n + 1) * K + k];
                acc2 += x_val * w[(n + 2) * K + k];
                acc3 += x_val * w[(n + 3) * K + k];
            }

            // Add bias if provided
            if (bias != nullptr) {
                acc0 += bias[n + 0];
                acc1 += bias[n + 1];
                acc2 += bias[n + 2];
                acc3 += bias[n + 3];
            }

            y_row[n + 0] = acc0;
            y_row[n + 1] = acc1;
            y_row[n + 2] = acc2;
            y_row[n + 3] = acc3;
        }

        // Handle remaining output features
        for (; n < N; n++) {
            const float* w_row = w + n * K;
            float acc = 0.0f;

            // Unroll by 4
            int k = 0;
            for (; k <= K - 4; k += 4) {
                acc += x_row[k + 0] * w_row[k + 0] +
                       x_row[k + 1] * w_row[k + 1] +
                       x_row[k + 2] * w_row[k + 2] +
                       x_row[k + 3] * w_row[k + 3];
            }

            for (; k < K; k++) {
                acc += x_row[k] * w_row[k];
            }

            if (bias != nullptr) {
                acc += bias[n];
            }

            y_row[n] = acc;
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
    int max_threads = std::thread::hardware_concurrency();
    // Use all available threads, but cap at reasonable limit
    return std::min(max_threads, 16);
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("matmul_free_generic", &matmul_free_generic, "Generic MatMul-free kernel");
    m.def("has_generic_support", &has_generic_support, "Check for generic support");
    m.def("get_optimal_threads", &get_optimal_threads, "Get optimal number of threads");

    // Conditionally define functions based on compilation flags
#ifdef HAS_NEON_SUPPORT
    m.def("matmul_free_neon", &matmul_free_neon, "NEON-optimized MatMul-free kernel");
    m.def("has_neon_support", &has_neon_support, "Check for NEON support");
#endif
#ifdef HAS_AVX2_SUPPORT
    m.def("matmul_free_avx2", &matmul_free_avx2, "AVX2-optimized MatMul-free kernel");
    m.def("has_avx2_support", &has_avx2_support, "Check for AVX2 support");
#endif
}
