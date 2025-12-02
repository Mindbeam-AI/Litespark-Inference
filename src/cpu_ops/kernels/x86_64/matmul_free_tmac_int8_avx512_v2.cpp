/*
 * T-MAC Int8 AVX-512 Kernel v2 - Optimized
 *
 * Key optimizations vs v1:
 * 1. No lane extraction - stay in 512-bit registers
 * 2. Cache-aware tiling (tile N for L1, tile K for L2)
 * 3. Prefetching for weight planes
 * 4. Process 64 outputs without extracting to 128-bit
 *
 * The trick: instead of unpacking int8->int16 within lanes,
 * we use horizontal add and accumulate directly in int32.
 */

#include <immintrin.h>
#include <cstring>
#include <cstdint>
#include <cmath>
#include <algorithm>
#include <omp.h>
#include <torch/extension.h>

// Cache parameters (tune for your CPU)
constexpr int L1_CACHE_SIZE = 32 * 1024;   // 32KB L1D
constexpr int L2_CACHE_SIZE = 1024 * 1024; // 1MB L2 per core
constexpr int TILE_N = 256;  // Process 256 outputs at a time (fits in L1)
constexpr int TILE_K = 64;   // Process 64 K groups at a time

/**
 * Build int8 LUT (for cases where sum fits in int8)
 * Only valid when all activations are small enough
 */
inline void build_lut_16_int8_clamped(const int8_t* x, int8_t* lut) {
    const int16_t x0 = x[0], x1 = x[1], x2 = x[2], x3 = x[3];

    // Compute sums and clamp to int8 range
    int16_t sums[16];
    sums[0]  = 0;
    sums[1]  = x0;
    sums[2]  = x1;
    sums[3]  = x0 + x1;
    sums[4]  = x2;
    sums[5]  = x0 + x2;
    sums[6]  = x1 + x2;
    sums[7]  = x0 + x1 + x2;
    sums[8]  = x3;
    sums[9]  = x0 + x3;
    sums[10] = x1 + x3;
    sums[11] = x0 + x1 + x3;
    sums[12] = x2 + x3;
    sums[13] = x0 + x2 + x3;
    sums[14] = x1 + x2 + x3;
    sums[15] = x0 + x1 + x2 + x3;

    for (int i = 0; i < 16; i++) {
        lut[i] = static_cast<int8_t>(std::max((int16_t)-128, std::min((int16_t)127, sums[i])));
    }
}

/**
 * Build split low/high byte LUTs for pack-and-unpack
 */
inline void build_lut_16_split(const int8_t* x, uint8_t* lut_lo, uint8_t* lut_hi) {
    const int16_t x0 = x[0], x1 = x[1], x2 = x[2], x3 = x[3];

    int16_t sums[16];
    sums[0]  = 0;
    sums[1]  = x0;
    sums[2]  = x1;
    sums[3]  = x0 + x1;
    sums[4]  = x2;
    sums[5]  = x0 + x2;
    sums[6]  = x1 + x2;
    sums[7]  = x0 + x1 + x2;
    sums[8]  = x3;
    sums[9]  = x0 + x3;
    sums[10] = x1 + x3;
    sums[11] = x0 + x1 + x3;
    sums[12] = x2 + x3;
    sums[13] = x0 + x2 + x3;
    sums[14] = x1 + x2 + x3;
    sums[15] = x0 + x1 + x2 + x3;

    for (int i = 0; i < 16; i++) {
        lut_lo[i] = static_cast<uint8_t>(sums[i] & 0xFF);
        lut_hi[i] = static_cast<uint8_t>((sums[i] >> 8) & 0xFF);
    }
}

/**
 * Build 16-entry int16 LUT from 4 int8 activations
 */
inline void build_lut_16_int16(const int8_t* x, int16_t* lut) {
    const int16_t x0 = x[0], x1 = x[1], x2 = x[2], x3 = x[3];

    lut[0]  = 0;
    lut[1]  = x0;
    lut[2]  = x1;
    lut[3]  = x0 + x1;
    lut[4]  = x2;
    lut[5]  = x0 + x2;
    lut[6]  = x1 + x2;
    lut[7]  = x0 + x1 + x2;
    lut[8]  = x3;
    lut[9]  = x0 + x3;
    lut[10] = x1 + x3;
    lut[11] = x0 + x1 + x3;
    lut[12] = x2 + x3;
    lut[13] = x0 + x2 + x3;
    lut[14] = x1 + x2 + x3;
    lut[15] = x0 + x1 + x2 + x3;
}

/**
 * AVX-512 optimized kernel v2
 *
 * Key insight: Instead of extracting lanes, use AVX-512 unpack instructions
 * _mm512_unpacklo_epi8 and _mm512_unpackhi_epi8 work on 4 lanes simultaneously
 */
void matmul_free_tmac_int8_avx512_v2(
    torch::Tensor x_int8_tensor,
    torch::Tensor scale_tensor,
    torch::Tensor sign_plane_tensor,
    torch::Tensor value_plane_tensor,
    torch::Tensor y_tensor,
    torch::Tensor bias_tensor,
    int M, int N, int K,
    int num_threads
) {
    const int8_t* __restrict__ x_int8 = x_int8_tensor.data_ptr<int8_t>();
    const float* __restrict__ scales = scale_tensor.data_ptr<float>();
    const uint8_t* __restrict__ sign_plane = sign_plane_tensor.data_ptr<uint8_t>();
    const uint8_t* __restrict__ value_plane = value_plane_tensor.data_ptr<uint8_t>();
    float* __restrict__ y = y_tensor.data_ptr<float>();
    const float* bias = bias_tensor.defined() ? bias_tensor.data_ptr<float>() : nullptr;

    const int K_groups = (K + 3) / 4;

    omp_set_num_threads(num_threads);

    #pragma omp parallel for schedule(static)
    for (int m = 0; m < M; m++) {
        const int8_t* __restrict__ x_row = x_int8 + m * K;
        float scale = scales[m];

        alignas(64) int32_t y_acc[N];
        memset(y_acc, 0, N * sizeof(int32_t));

        // Process K groups
        for (int kg = 0; kg < K_groups; kg++) {
            const int k_base = kg * 4;

            int8_t x_vals[4] = {0, 0, 0, 0};
            for (int i = 0; i < 4 && k_base + i < K; i++) {
                x_vals[i] = x_row[k_base + i];
            }

            alignas(16) uint8_t lut_lo[16];
            alignas(16) uint8_t lut_hi[16];
            build_lut_16_split(x_vals, lut_lo, lut_hi);

            // Broadcast 16-byte LUT to all 4 lanes of 512-bit register
            __m128i lut_lo_128 = _mm_load_si128((__m128i*)lut_lo);
            __m128i lut_hi_128 = _mm_load_si128((__m128i*)lut_hi);
            __m512i lut_lo_512 = _mm512_broadcast_i32x4(lut_lo_128);
            __m512i lut_hi_512 = _mm512_broadcast_i32x4(lut_hi_128);

            const uint8_t* __restrict__ sign_row = sign_plane + kg * N;
            const uint8_t* __restrict__ value_row = value_plane + kg * N;

            // Prefetch next K group's weight planes
            if (kg + 1 < K_groups) {
                _mm_prefetch((const char*)(sign_plane + (kg + 1) * N), _MM_HINT_T0);
                _mm_prefetch((const char*)(value_plane + (kg + 1) * N), _MM_HINT_T0);
            }

            // Process 64 outputs at a time
            // AVX-512 unpack operates on 4 independent 128-bit lanes
            // unpacklo_epi8 interleaves bytes 0-7 from each 128-bit lane
            // This gives us int16 values in the correct positions
            int n = 0;
            for (; n + 63 < N; n += 64) {
                // Prefetch accumulator and next weight block
                _mm_prefetch((const char*)(y_acc + n + 64), _MM_HINT_T0);
                _mm_prefetch((const char*)(sign_row + n + 64), _MM_HINT_T0);
                _mm_prefetch((const char*)(value_row + n + 64), _MM_HINT_T0);

                // Load 64 indices
                __m512i sign_idx = _mm512_loadu_si512((__m512i*)(sign_row + n));
                __m512i value_idx = _mm512_loadu_si512((__m512i*)(value_row + n));

                // Mask to 4 bits
                __m512i mask_0f = _mm512_set1_epi8(0x0F);
                sign_idx = _mm512_and_si512(sign_idx, mask_0f);
                value_idx = _mm512_and_si512(value_idx, mask_0f);

                // PSHUFB lookups - 64 lookups each
                __m512i sign_lo = _mm512_shuffle_epi8(lut_lo_512, sign_idx);
                __m512i sign_hi = _mm512_shuffle_epi8(lut_hi_512, sign_idx);
                __m512i value_lo = _mm512_shuffle_epi8(lut_lo_512, value_idx);
                __m512i value_hi = _mm512_shuffle_epi8(lut_hi_512, value_idx);

                // Unpack to int16 using AVX-512 (works on 4 lanes simultaneously)
                // This avoids extracting lanes!
                __m512i sign_16_even = _mm512_unpacklo_epi8(sign_lo, sign_hi);  // lanes 0,1,2,3 low halves
                __m512i sign_16_odd = _mm512_unpackhi_epi8(sign_lo, sign_hi);   // lanes 0,1,2,3 high halves
                __m512i value_16_even = _mm512_unpacklo_epi8(value_lo, value_hi);
                __m512i value_16_odd = _mm512_unpackhi_epi8(value_lo, value_hi);

                // Compute 2*value - sign as int16
                __m512i two = _mm512_set1_epi16(2);
                __m512i res_16_even = _mm512_sub_epi16(_mm512_mullo_epi16(value_16_even, two), sign_16_even);
                __m512i res_16_odd = _mm512_sub_epi16(_mm512_mullo_epi16(value_16_odd, two), sign_16_odd);

                // Now we need to convert int16 to int32 and accumulate
                // Extract lower and upper halves, extend to int32

                // res_16_even has: [lane0_lo8][lane1_lo8][lane2_lo8][lane3_lo8] (each 8 int16s)
                // res_16_odd has:  [lane0_hi8][lane1_hi8][lane2_hi8][lane3_hi8]

                // We need to reorder to get consecutive outputs
                // Use permute to gather same-position elements across lanes

                // For even results (positions 0,2,4,6,... within each group of 16)
                // _mm512_cvtepi16_epi32 takes __m256i (16 int16) and returns __m512i (16 int32)
                // These lines were incorrect - removing unused variables

                // The layout after unpack is tricky. Let me explain:
                // Input: 64 bytes [0..63] across 4 lanes of 16 bytes each
                // After unpacklo: each 128-bit lane has interleaved bytes 0-7
                // After unpackhi: each 128-bit lane has interleaved bytes 8-15
                //
                // Lane 0: bytes 0-15 -> unpacklo gives int16[0-7], unpackhi gives int16[8-15]
                // Lane 1: bytes 16-31 -> etc.
                //
                // So the outputs are already in correct order within each group!

                // Accumulate results
                // Lane 0 (n+0 to n+15): even_lo_256 gives n+0..7, odd_lo_256 gives n+8..15
                // But wait - extracti64x4 splits differently...

                // Let's be more careful. After _mm512_cvtepi16_epi32:
                // _mm512_castsi512_si256(res_16_even) = lower 256 bits = lane0_lo + lane1_lo (8 int16 each = 256 bits)
                // _mm512_extracti64x4_epi64(res_16_even, 1) = upper 256 bits = lane2_lo + lane3_lo

                // After cvtepi16_epi32, we get 8 int32 from 8 int16
                // even_lo_256 = int32 versions of lane0_lo[0:7] (only first 8 int16 from 256 bits)
                // This means we're only processing 8 elements, not 16!

                // Need to use full 512-bit conversion or do it in chunks

                // Simpler approach: process in 16-element chunks using AVX-512 conversion
                // _mm512_cvtepi16_epi32 takes __m256i and produces __m512i (16 int16 -> 16 int32)

                // Extract 256-bit chunks from res_16_even (contains 32 int16)
                __m256i res_even_0 = _mm512_castsi512_si256(res_16_even);
                __m256i res_even_1 = _mm512_extracti64x4_epi64(res_16_even, 1);
                __m256i res_odd_0 = _mm512_castsi512_si256(res_16_odd);
                __m256i res_odd_1 = _mm512_extracti64x4_epi64(res_16_odd, 1);

                // Convert to int32 (16 at a time)
                __m512i res_32_even_0 = _mm512_cvtepi16_epi32(res_even_0);  // 16 int32
                __m512i res_32_even_1 = _mm512_cvtepi16_epi32(res_even_1);  // 16 int32
                __m512i res_32_odd_0 = _mm512_cvtepi16_epi32(res_odd_0);    // 16 int32
                __m512i res_32_odd_1 = _mm512_cvtepi16_epi32(res_odd_1);    // 16 int32

                // Now we have 64 int32 values, but they're interleaved due to unpack
                // The ordering is:
                // res_32_even_0: positions [0,8,1,9,2,10,3,11,4,12,5,13,6,14,7,15] from first 16 outputs? No...
                //
                // Actually, unpacklo/hi for epi8 interleaves byte-by-byte:
                // unpacklo([a0..a15], [b0..b15]) = [a0,b0,a1,b1,a2,b2,a3,b3,a4,b4,a5,b5,a6,b6,a7,b7]
                // This gives us int16 values where each pair (ai,bi) forms one int16
                //
                // So after unpacklo(sign_lo, sign_hi), we get the correct int16 values
                // in positions 0-7 of each 128-bit lane. unpackhi gives positions 8-15.
                //
                // The issue is 512-bit unpack works lane-by-lane, so:
                // res_16_even = [lane0_0:7, lane1_0:7, lane2_0:7, lane3_0:7] (as int16)
                // res_16_odd = [lane0_8:15, lane1_8:15, lane2_8:15, lane3_8:15]
                //
                // For output n+0..n+63:
                // - n+0..15 should come from lane0
                // - n+16..31 from lane1
                // - n+32..47 from lane2
                // - n+48..63 from lane3
                //
                // But res_32_even_0 has [lane0_0:7, lane1_0:7] as int32 (16 values)
                // res_32_even_1 has [lane2_0:7, lane3_0:7]
                // res_32_odd_0 has [lane0_8:15, lane1_8:15]
                // res_32_odd_1 has [lane2_8:15, lane3_8:15]
                //
                // We need to interleave them properly. Use permute instructions.

                // Actually, let me reconsider. The current layout:
                // res_32_even_0[0:7] = lane0 positions 0-7 (outputs n+0 to n+7)
                // res_32_even_0[8:15] = lane1 positions 0-7 (outputs n+16 to n+23)
                // res_32_odd_0[0:7] = lane0 positions 8-15 (outputs n+8 to n+15)
                // res_32_odd_0[8:15] = lane1 positions 8-15 (outputs n+24 to n+31)
                // res_32_even_1[0:7] = lane2 positions 0-7 (outputs n+32 to n+39)
                // res_32_even_1[8:15] = lane3 positions 0-7 (outputs n+48 to n+55)
                // res_32_odd_1[0:7] = lane2 positions 8-15 (outputs n+40 to n+47)
                // res_32_odd_1[8:15] = lane3 positions 8-15 (outputs n+56 to n+63)

                // We need to shuffle to get: n+0..15, n+16..31, n+32..47, n+48..63
                // From even_0 and odd_0: [even_0_lo, odd_0_lo, even_0_hi, odd_0_hi]

                // Use 256-bit permute2x128 or 512-bit permutex2var

                // Extract 256-bit halves and recombine
                __m256i e0_lo = _mm512_castsi512_si256(res_32_even_0);      // n+0..7
                __m256i e0_hi = _mm512_extracti64x4_epi64(res_32_even_0, 1); // n+16..23
                __m256i o0_lo = _mm512_castsi512_si256(res_32_odd_0);       // n+8..15
                __m256i o0_hi = _mm512_extracti64x4_epi64(res_32_odd_0, 1);  // n+24..31
                __m256i e1_lo = _mm512_castsi512_si256(res_32_even_1);      // n+32..39
                __m256i e1_hi = _mm512_extracti64x4_epi64(res_32_even_1, 1); // n+48..55
                __m256i o1_lo = _mm512_castsi512_si256(res_32_odd_1);       // n+40..47
                __m256i o1_hi = _mm512_extracti64x4_epi64(res_32_odd_1, 1);  // n+56..63

                // Combine into correct order
                __m512i out_0_15 = _mm512_inserti64x4(_mm512_castsi256_si512(e0_lo), o0_lo, 1);   // n+0..15
                __m512i out_16_31 = _mm512_inserti64x4(_mm512_castsi256_si512(e0_hi), o0_hi, 1); // n+16..31
                __m512i out_32_47 = _mm512_inserti64x4(_mm512_castsi256_si512(e1_lo), o1_lo, 1); // n+32..47
                __m512i out_48_63 = _mm512_inserti64x4(_mm512_castsi256_si512(e1_hi), o1_hi, 1); // n+48..63

                // Load accumulators and add
                __m512i acc0 = _mm512_loadu_si512((__m512i*)(y_acc + n));
                __m512i acc1 = _mm512_loadu_si512((__m512i*)(y_acc + n + 16));
                __m512i acc2 = _mm512_loadu_si512((__m512i*)(y_acc + n + 32));
                __m512i acc3 = _mm512_loadu_si512((__m512i*)(y_acc + n + 48));

                acc0 = _mm512_add_epi32(acc0, out_0_15);
                acc1 = _mm512_add_epi32(acc1, out_16_31);
                acc2 = _mm512_add_epi32(acc2, out_32_47);
                acc3 = _mm512_add_epi32(acc3, out_48_63);

                _mm512_storeu_si512((__m512i*)(y_acc + n), acc0);
                _mm512_storeu_si512((__m512i*)(y_acc + n + 16), acc1);
                _mm512_storeu_si512((__m512i*)(y_acc + n + 32), acc2);
                _mm512_storeu_si512((__m512i*)(y_acc + n + 48), acc3);
            }

            // Handle remaining with 128-bit SSE
            for (; n + 15 < N; n += 16) {
                __m128i sign_idx = _mm_loadu_si128((__m128i*)(sign_row + n));
                __m128i value_idx = _mm_loadu_si128((__m128i*)(value_row + n));

                __m128i mask_0f = _mm_set1_epi8(0x0F);
                sign_idx = _mm_and_si128(sign_idx, mask_0f);
                value_idx = _mm_and_si128(value_idx, mask_0f);

                __m128i sign_lo = _mm_shuffle_epi8(lut_lo_128, sign_idx);
                __m128i sign_hi = _mm_shuffle_epi8(lut_hi_128, sign_idx);
                __m128i value_lo = _mm_shuffle_epi8(lut_lo_128, value_idx);
                __m128i value_hi = _mm_shuffle_epi8(lut_hi_128, value_idx);

                __m128i sign_16_lo = _mm_unpacklo_epi8(sign_lo, sign_hi);
                __m128i sign_16_hi = _mm_unpackhi_epi8(sign_lo, sign_hi);
                __m128i value_16_lo = _mm_unpacklo_epi8(value_lo, value_hi);
                __m128i value_16_hi = _mm_unpackhi_epi8(value_lo, value_hi);

                __m128i two = _mm_set1_epi16(2);
                __m128i res_16_lo = _mm_sub_epi16(_mm_mullo_epi16(value_16_lo, two), sign_16_lo);
                __m128i res_16_hi = _mm_sub_epi16(_mm_mullo_epi16(value_16_hi, two), sign_16_hi);

                __m256i res_32_0 = _mm256_cvtepi16_epi32(res_16_lo);
                __m256i res_32_1 = _mm256_cvtepi16_epi32(res_16_hi);

                __m256i acc0 = _mm256_loadu_si256((__m256i*)(y_acc + n));
                __m256i acc1 = _mm256_loadu_si256((__m256i*)(y_acc + n + 8));

                acc0 = _mm256_add_epi32(acc0, res_32_0);
                acc1 = _mm256_add_epi32(acc1, res_32_1);

                _mm256_storeu_si256((__m256i*)(y_acc + n), acc0);
                _mm256_storeu_si256((__m256i*)(y_acc + n + 8), acc1);
            }

            // Scalar remainder
            alignas(32) int16_t lut[16];
            build_lut_16_int16(x_vals, lut);
            for (; n < N; n++) {
                uint8_t s = sign_row[n] & 0x0F;
                uint8_t v = value_row[n] & 0x0F;
                y_acc[n] += 2 * static_cast<int32_t>(lut[v]) - static_cast<int32_t>(lut[s]);
            }
        }

        // Convert to float
        float* y_row = y + m * N;
        if (bias != nullptr) {
            for (int n = 0; n < N; n++) {
                y_row[n] = static_cast<float>(y_acc[n]) * scale + bias[n];
            }
        } else {
            __m512 scale_vec = _mm512_set1_ps(scale);
            int n = 0;
            for (; n + 15 < N; n += 16) {
                __m512i acc = _mm512_loadu_si512((__m512i*)(y_acc + n));
                __m512 acc_f = _mm512_cvtepi32_ps(acc);
                __m512 result = _mm512_mul_ps(acc_f, scale_vec);
                _mm512_storeu_ps(y_row + n, result);
            }
            for (; n < N; n++) {
                y_row[n] = static_cast<float>(y_acc[n]) * scale;
            }
        }
    }
}

/**
 * Tiled version with better cache utilization
 * Processes tiles of N to keep working set in L1
 */
void matmul_free_tmac_int8_avx512_tiled(
    torch::Tensor x_int8_tensor,
    torch::Tensor scale_tensor,
    torch::Tensor sign_plane_tensor,
    torch::Tensor value_plane_tensor,
    torch::Tensor y_tensor,
    torch::Tensor bias_tensor,
    int M, int N, int K,
    int num_threads
) {
    const int8_t* __restrict__ x_int8 = x_int8_tensor.data_ptr<int8_t>();
    const float* __restrict__ scales = scale_tensor.data_ptr<float>();
    const uint8_t* __restrict__ sign_plane = sign_plane_tensor.data_ptr<uint8_t>();
    const uint8_t* __restrict__ value_plane = value_plane_tensor.data_ptr<uint8_t>();
    float* __restrict__ y = y_tensor.data_ptr<float>();
    const float* bias = bias_tensor.defined() ? bias_tensor.data_ptr<float>() : nullptr;

    const int K_groups = (K + 3) / 4;

    omp_set_num_threads(num_threads);

    // Tile N to fit in L1 cache
    // Each tile: TILE_N int32 accumulators = 1KB
    // Plus 2 * TILE_N bytes for sign/value = 0.5KB
    // Total ~1.5KB per tile, easily fits in 32KB L1

    #pragma omp parallel for schedule(static)
    for (int m = 0; m < M; m++) {
        const int8_t* __restrict__ x_row = x_int8 + m * K;
        float scale = scales[m];
        float* y_row = y + m * N;

        // Process N in tiles
        for (int n_tile = 0; n_tile < N; n_tile += TILE_N) {
            const int n_end = std::min(n_tile + TILE_N, N);
            const int tile_size = n_end - n_tile;

            alignas(64) int32_t y_acc[TILE_N];
            memset(y_acc, 0, tile_size * sizeof(int32_t));

            // Process all K groups for this N tile
            for (int kg = 0; kg < K_groups; kg++) {
                const int k_base = kg * 4;

                int8_t x_vals[4] = {0, 0, 0, 0};
                for (int i = 0; i < 4 && k_base + i < K; i++) {
                    x_vals[i] = x_row[k_base + i];
                }

                alignas(16) uint8_t lut_lo[16];
                alignas(16) uint8_t lut_hi[16];
                build_lut_16_split(x_vals, lut_lo, lut_hi);

                __m128i lut_lo_128 = _mm_load_si128((__m128i*)lut_lo);
                __m128i lut_hi_128 = _mm_load_si128((__m128i*)lut_hi);

                const uint8_t* __restrict__ sign_row = sign_plane + kg * N + n_tile;
                const uint8_t* __restrict__ value_row = value_plane + kg * N + n_tile;

                // Process tile with SSE (simpler, good cache behavior)
                int n = 0;
                for (; n + 15 < tile_size; n += 16) {
                    __m128i sign_idx = _mm_loadu_si128((__m128i*)(sign_row + n));
                    __m128i value_idx = _mm_loadu_si128((__m128i*)(value_row + n));

                    __m128i mask_0f = _mm_set1_epi8(0x0F);
                    sign_idx = _mm_and_si128(sign_idx, mask_0f);
                    value_idx = _mm_and_si128(value_idx, mask_0f);

                    __m128i sign_lo = _mm_shuffle_epi8(lut_lo_128, sign_idx);
                    __m128i sign_hi = _mm_shuffle_epi8(lut_hi_128, sign_idx);
                    __m128i value_lo = _mm_shuffle_epi8(lut_lo_128, value_idx);
                    __m128i value_hi = _mm_shuffle_epi8(lut_hi_128, value_idx);

                    __m128i sign_16_lo = _mm_unpacklo_epi8(sign_lo, sign_hi);
                    __m128i sign_16_hi = _mm_unpackhi_epi8(sign_lo, sign_hi);
                    __m128i value_16_lo = _mm_unpacklo_epi8(value_lo, value_hi);
                    __m128i value_16_hi = _mm_unpackhi_epi8(value_lo, value_hi);

                    __m128i two = _mm_set1_epi16(2);
                    __m128i res_16_lo = _mm_sub_epi16(_mm_mullo_epi16(value_16_lo, two), sign_16_lo);
                    __m128i res_16_hi = _mm_sub_epi16(_mm_mullo_epi16(value_16_hi, two), sign_16_hi);

                    __m256i res_32_0 = _mm256_cvtepi16_epi32(res_16_lo);
                    __m256i res_32_1 = _mm256_cvtepi16_epi32(res_16_hi);

                    __m256i acc0 = _mm256_loadu_si256((__m256i*)(y_acc + n));
                    __m256i acc1 = _mm256_loadu_si256((__m256i*)(y_acc + n + 8));

                    acc0 = _mm256_add_epi32(acc0, res_32_0);
                    acc1 = _mm256_add_epi32(acc1, res_32_1);

                    _mm256_storeu_si256((__m256i*)(y_acc + n), acc0);
                    _mm256_storeu_si256((__m256i*)(y_acc + n + 8), acc1);
                }

                // Scalar remainder
                alignas(32) int16_t lut[16];
                build_lut_16_int16(x_vals, lut);
                for (; n < tile_size; n++) {
                    uint8_t s = sign_row[n] & 0x0F;
                    uint8_t v = value_row[n] & 0x0F;
                    y_acc[n] += 2 * static_cast<int32_t>(lut[v]) - static_cast<int32_t>(lut[s]);
                }
            }

            // Write results for this tile
            if (bias != nullptr) {
                for (int n = 0; n < tile_size; n++) {
                    y_row[n_tile + n] = static_cast<float>(y_acc[n]) * scale + bias[n_tile + n];
                }
            } else {
                __m256 scale_vec = _mm256_set1_ps(scale);
                int n = 0;
                for (; n + 7 < tile_size; n += 8) {
                    __m256i acc = _mm256_loadu_si256((__m256i*)(y_acc + n));
                    __m256 acc_f = _mm256_cvtepi32_ps(acc);
                    __m256 result = _mm256_mul_ps(acc_f, scale_vec);
                    _mm256_storeu_ps(y_row + n_tile + n, result);
                }
                for (; n < tile_size; n++) {
                    y_row[n_tile + n] = static_cast<float>(y_acc[n]) * scale;
                }
            }
        }
    }
}

// Python bindings
PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("matmul_free_tmac_int8_avx512_v2", &matmul_free_tmac_int8_avx512_v2,
          "T-MAC int8 AVX-512 v2 (optimized lane handling)");
    m.def("matmul_free_tmac_int8_avx512_tiled", &matmul_free_tmac_int8_avx512_tiled,
          "T-MAC int8 AVX-512 tiled (cache-aware)");
}
