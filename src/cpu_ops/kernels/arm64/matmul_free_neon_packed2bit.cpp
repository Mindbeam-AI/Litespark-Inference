/*
 * 2-bit Packed MatMul-Free Kernel - Optimized for Memory Efficiency
 *
 * TRUE MatMul-free with 2-bit weight packing
 * Format: 00=0, 01=+1, 10=-1, 11=unused
 * 16 weights per uint32 = 16x memory reduction!
 *
 * Trade-off: Slightly slower than unpacked, but MASSIVE memory savings
 */

#include <arm_neon.h>
#include <cstring>
#include <algorithm>
#include <torch/extension.h>
#include <omp.h>

/**
 * Pack float ternary weights to 2-bit format
 * 16 weights per uint32: 00=0, 01=+1, 10=-1
 */
void pack_weights_2bit(
    torch::Tensor w_tensor,
    torch::Tensor w_packed_tensor,
    int N, int K
) {
    const float* w_float = w_tensor.data_ptr<float>();
    uint32_t* w_packed = w_packed_tensor.data_ptr<uint32_t>();

    int K_packed = (K + 15) / 16;  // Round up to 16

    #pragma omp parallel for
    for (int n = 0; n < N; n++) {
        for (int k_pack = 0; k_pack < K_packed; k_pack++) {
            uint32_t packed = 0;

            for (int i = 0; i < 16; i++) {
                int k = k_pack * 16 + i;
                if (k < K) {
                    float w_val = w_float[n * K + k];
                    uint32_t bits;

                    if (w_val > 0.5f) {
                        bits = 1;  // 01 = +1
                    } else if (w_val < -0.5f) {
                        bits = 2;  // 10 = -1
                    } else {
                        bits = 0;  // 00 = 0
                    }

                    packed |= (bits << (i * 2));
                }
            }

            w_packed[n * K_packed + k_pack] = packed;
        }
    }
}

/**
 * Extract single 2-bit value
 */
inline int8_t extract_2bit(uint32_t packed, int index) {
    uint32_t bits = (packed >> (index * 2)) & 0x3;
    if (bits == 1) return 1;
    if (bits == 2) return -1;
    return 0;
}

/**
 * MatMul-free with 2-bit packed weights
 * Optimized for MEMORY, not speed
 */
void matmul_free_neon_packed2bit(
    torch::Tensor x_tensor,
    torch::Tensor w_packed_tensor,
    torch::Tensor y_tensor,
    torch::Tensor bias_tensor,
    int M, int N, int K,
    int num_threads
) {
    const float* x = x_tensor.data_ptr<float>();
    const uint32_t* w_packed = w_packed_tensor.data_ptr<uint32_t>();
    float* y = y_tensor.data_ptr<float>();
    const float* bias = bias_tensor.defined() ? bias_tensor.data_ptr<float>() : nullptr;

    const int K_packed = (K + 15) / 16;
    const int N_BLOCK = 8;  // Conservative blocking for packed version

    omp_set_num_threads(num_threads);

    #pragma omp parallel for schedule(dynamic, 16)
    for (int m = 0; m < M; m++) {
        const float* x_row = x + m * K;
        float* y_row = y + m * N;

        // Process N_BLOCK outputs together
        int n = 0;
        for (; n <= N - N_BLOCK; n += N_BLOCK) {
            float acc[N_BLOCK] = {0};

            // Process packed weights (16 at a time)
            for (int k_pack = 0; k_pack < K_packed; k_pack++) {
                // Load packed weights for N_BLOCK outputs
                uint32_t packed[N_BLOCK];
                for (int i = 0; i < N_BLOCK; i++) {
                    packed[i] = w_packed[(n + i) * K_packed + k_pack];
                }

                // Process 16 weights from packed format
                int k_base = k_pack * 16;
                for (int i = 0; i < 16 && (k_base + i) < K; i++) {
                    float x_val = x_row[k_base + i];

                    // Extract weights and accumulate
                    for (int j = 0; j < N_BLOCK; j++) {
                        int8_t w_val = extract_2bit(packed[j], i);
                        if (w_val == 1) acc[j] += x_val;
                        else if (w_val == -1) acc[j] -= x_val;
                    }
                }
            }

            // Add bias and store
            for (int i = 0; i < N_BLOCK; i++) {
                if (bias != nullptr) {
                    acc[i] += bias[n + i];
                }
                y_row[n + i] = acc[i];
            }
        }

        // Handle remaining outputs
        for (; n < N; n++) {
            float acc = 0.0f;

            for (int k_pack = 0; k_pack < K_packed; k_pack++) {
                uint32_t packed = w_packed[n * K_packed + k_pack];
                int k_base = k_pack * 16;

                for (int i = 0; i < 16 && (k_base + i) < K; i++) {
                    float x_val = x_row[k_base + i];
                    int8_t w_val = extract_2bit(packed, i);

                    if (w_val == 1) acc += x_val;
                    else if (w_val == -1) acc -= x_val;
                }
            }

            if (bias != nullptr) {
                acc += bias[n];
            }

            y_row[n] = acc;
        }
    }
}

/**
 * Get memory footprint of packed weights
 */
size_t get_packed_memory_bytes(int N, int K) {
    int K_packed = (K + 15) / 16;
    return N * K_packed * sizeof(uint32_t);
}

size_t get_unpacked_memory_bytes(int N, int K) {
    return N * K * sizeof(float);
}

float get_memory_reduction(int N, int K) {
    return (float)get_unpacked_memory_bytes(N, K) / (float)get_packed_memory_bytes(N, K);
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("pack_weights_2bit", &pack_weights_2bit, "Pack weights to 2-bit");
    m.def("matmul_free_neon_packed2bit", &matmul_free_neon_packed2bit, "MatMul-free with 2-bit packing");
    m.def("get_packed_memory_bytes", &get_packed_memory_bytes, "Get packed weight memory");
    m.def("get_unpacked_memory_bytes", &get_unpacked_memory_bytes, "Get unpacked weight memory");
    m.def("get_memory_reduction", &get_memory_reduction, "Get memory reduction factor");
}
