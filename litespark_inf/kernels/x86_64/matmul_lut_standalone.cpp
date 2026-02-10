/*
 * Standalone LUT-Based Ternary Matmul Kernel - No PyTorch Dependency
 * Optimized for tg128 (M=1 token generation)
 * 
 * Compile:
 *   g++ -O3 -march=native -mavx512f -mavx512bw -mavx512vbmi -fopenmp -shared -fPIC matmul_lut_standalone.cpp -o libmatmul_lut.so
 * 
 * Test:
 *   g++ -O3 -march=native -mavx512f -mavx512bw -fopenmp -DTEST_MAIN matmul_lut_standalone.cpp -o test_lut && ./test_lut
 */

#include <immintrin.h>
#include <cstdint>
#include <cstdlib>
#include <cstring>
#include <omp.h>

extern "C" {

// Weight packing: 4 ternary weights per byte using 2-bit encoding
// w[0..3] packed as: (w0+1) | ((w1+1)<<2) | ((w2+1)<<4) | ((w3+1)<<6)
void pack_weights_4bit(const int8_t* w, uint8_t* packed, int N, int K) {
    #pragma omp parallel for
    for (int n = 0; n < N; n++) {
        const int8_t* row = w + n * K;
        uint8_t* out = packed + n * (K / 4);
        for (int k = 0; k < K; k += 4) {
            uint8_t p = (uint8_t)((row[k] + 1) | ((row[k+1] + 1) << 2) | 
                                  ((row[k+2] + 1) << 4) | ((row[k+3] + 1) << 6));
            out[k / 4] = p;
        }
    }
}

// LUT for 4 weights: 3^4 = 81 combinations, pad to 128 for alignment
#define LUT4_SIZE 128

// Build LUT for groups of 4 activations
// lut[k/4 * 128 + idx] = sum of selected activations
void build_lut4(const int8_t* act, int16_t* lut, int K) {
    for (int k = 0; k < K; k += 4) {
        int16_t a0 = act[k], a1 = act[k+1], a2 = act[k+2], a3 = act[k+3];
        int16_t* L = lut + (k / 4) * LUT4_SIZE;
        
        // Compute all 81 possible sums: each weight in {-1, 0, +1}
        for (int w0 = -1; w0 <= 1; w0++) {
            for (int w1 = -1; w1 <= 1; w1++) {
                for (int w2 = -1; w2 <= 1; w2++) {
                    for (int w3 = -1; w3 <= 1; w3++) {
                        int idx = (w0+1) | ((w1+1)<<2) | ((w2+1)<<4) | ((w3+1)<<6);
                        L[idx] = (int16_t)(w0*a0 + w1*a1 + w2*a2 + w3*a3);
                    }
                }
            }
        }
    }
}

// Main matmul: y[n] = sum_k LUT[k/4][packed_w[n, k/4]] for M=1
// x: [K] int8, packed_w: [N, K/4] uint8, y: [N] float
void matmul_lut_m1(
    const int8_t* x, float x_scale,
    const uint8_t* packed_w,
    float* y, const float* bias,
    int N, int K, int num_threads
) {
    const int K4 = K / 4;

    // Allocate LUT with proper size calculation
    size_t lut_size = (size_t)K4 * LUT4_SIZE * sizeof(int16_t);
    int16_t* lut = (int16_t*)aligned_alloc(64, lut_size);
    if (!lut) return;
    memset(lut, 0, lut_size);

    // Build LUT (single-threaded, fast)
    build_lut4(x, lut, K);

    omp_set_num_threads(num_threads);

    // Parallel accumulation over N outputs
    #pragma omp parallel for schedule(static)
    for (int n = 0; n < N; n++) {
        const uint8_t* w_row = packed_w + n * K4;
        int32_t sum = 0;

        // Process 32 groups at a time (32 bytes = 128 weights)
        int k4 = 0;
        for (; k4 + 31 < K4; k4 += 32) {
            // Unroll and accumulate
            for (int i = 0; i < 32; i++) {
                uint8_t idx = w_row[k4 + i];
                sum += lut[(k4 + i) * LUT4_SIZE + idx];
            }
        }

        // Remainder
        for (; k4 < K4; k4++) {
            sum += lut[k4 * LUT4_SIZE + w_row[k4]];
        }

        y[n] = (float)sum * x_scale + (bias ? bias[n] : 0.0f);
    }

    free(lut);
}

} // extern "C"

#ifdef TEST_MAIN
#include <cstdio>
#include <chrono>

int main() {
    const int N = 8192, K = 8192, num_threads = 8;
    
    int8_t* x = (int8_t*)aligned_alloc(64, K);
    int8_t* w = (int8_t*)aligned_alloc(64, N * K);
    uint8_t* packed_w = (uint8_t*)aligned_alloc(64, N * (K/4));
    float* y = (float*)aligned_alloc(64, N * sizeof(float));
    
    // Init random weights and activations
    for (int i = 0; i < K; i++) x[i] = (rand() % 256) - 128;
    for (int i = 0; i < N * K; i++) w[i] = (rand() % 3) - 1;
    
    pack_weights_4bit(w, packed_w, N, K);
    
    // Warmup
    for (int i = 0; i < 3; i++) matmul_lut_m1(x, 1.0f, packed_w, y, nullptr, N, K, num_threads);
    
    // Benchmark
    const int runs = 100;
    auto start = std::chrono::high_resolution_clock::now();
    for (int i = 0; i < runs; i++) {
        matmul_lut_m1(x, 1.0f, packed_w, y, nullptr, N, K, num_threads);
    }
    auto end = std::chrono::high_resolution_clock::now();
    double ms = std::chrono::duration<double, std::milli>(end - start).count() / runs;
    
    printf("N=%d K=%d threads=%d: %.3f ms/matmul (%.2f GOPS)\n", 
           N, K, num_threads, ms, (double)N*K*2/ms/1e6);
    
    free(x); free(w); free(packed_w); free(y);
    return 0;
}
#endif

