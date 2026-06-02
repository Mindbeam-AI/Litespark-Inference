/*
 * AVX2 scalar-loop fallback for litespark torchless on x86_64 CPUs that
 * lack AVX-512 (e.g. AMD Zen 3 Threadripper/Ryzen 5xxx, older Xeons).
 *
 * Mirrors every LSPK_API symbol that matmul_lut_avx512_extern_c.cpp
 * exports, but with the `_avx2` arch tag. The kernels are intentionally
 * written as plain C++ loops -- no intrinsics -- so the compiler can
 * auto-vectorize them with -O3 -mavx2 -mfma using 256-bit ymm registers.
 * That's 2-4x slower than hand-tuned VNNI on the same data, but it
 * runs correctly anywhere with AVX2 + FMA (Haswell+, Zen 1+).
 *
 * Selection: kernel.py picks "_matmul_lut_avx2*" vs "_matmul_lut_avx512*"
 * at runtime based on /proc/cpuinfo (or __builtin_cpu_supports on glibc).
 *
 * Skipped vs the AVX-512 source:
 *   - AMX paths (matmul_amx_int8_mT, transpose_packed_to_amx_vnni,
 *     amx_request_permission). Those are Sapphire-Rapids-only and the
 *     AVX-512 source already gates them on LITESPARK_USE_AMX; not needed
 *     here.
 *
 * Contract for every symbol matches the AVX-512 source byte-for-byte;
 * only the implementation is scalar.
 */

#include <cstddef>
#include <cstdint>
#include <cstring>
#include <cmath>
#include <immintrin.h>

#if defined(_OPENMP)
#include <omp.h>
#endif

#if defined(_WIN32)
  #define LSPK_API __declspec(dllexport)
#else
  #define LSPK_API __attribute__((visibility("default")))
#endif

namespace {

// bf16 (high half of fp32) -> fp32.
static inline float bf16_to_fp32(uint16_t b) {
    uint32_t bits = static_cast<uint32_t>(b) << 16;
    float f;
    std::memcpy(&f, &bits, sizeof(float));
    return f;
}

// Per-tensor absmax fp32 -> int8 with saturating-to-[-127,127] rounding.
// Mirrors quantize_activation_avx512 exactly.
static inline float _quantize_absmax(
    const float* __restrict__ x, int8_t* __restrict__ out, int K
) {
    float absmax = 0.0f;
    for (int k = 0; k < K; ++k) {
        float a = std::fabs(x[k]);
        if (a > absmax) absmax = a;
    }
    if (absmax < 1e-5f) absmax = 1e-5f;
    const float scale = absmax / 127.0f;
    const float inv_scale = 1.0f / scale;
    for (int k = 0; k < K; ++k) {
        float v = x[k] * inv_scale;
        int32_t iv = static_cast<int32_t>(std::lrintf(v));
        if (iv >  127) iv =  127;
        if (iv < -127) iv = -127;
        out[k] = static_cast<int8_t>(iv);
    }
    return scale;
}

// RMSNorm core with bf16 gamma; optionally writes intermediate fp32
// (used by the fused rmsnorm_quantize variant).
static inline void _rmsnorm_bf16(
    const float* __restrict__ x, const uint16_t* __restrict__ gamma,
    float* __restrict__ out, int K, float eps
) {
    float ssq = 0.0f;
    for (int k = 0; k < K; ++k) ssq += x[k] * x[k];
    const float rrms = 1.0f / std::sqrt(ssq / static_cast<float>(K) + eps);
    for (int k = 0; k < K; ++k) {
        out[k] = x[k] * rrms * bf16_to_fp32(gamma[k]);
    }
}

static inline void _rmsnorm_fp32(
    const float* __restrict__ x, const float* __restrict__ gamma,
    float* __restrict__ out, int K, float eps
) {
    float ssq = 0.0f;
    for (int k = 0; k < K; ++k) ssq += x[k] * x[k];
    const float rrms = 1.0f / std::sqrt(ssq / static_cast<float>(K) + eps);
    for (int k = 0; k < K; ++k) out[k] = x[k] * rrms * gamma[k];
}

static inline float _rmsnorm_quantize_bf16_core(
    const float* __restrict__ x, const uint16_t* __restrict__ gamma,
    int8_t* __restrict__ out_i8, int K, float eps, float* __restrict__ tmp_fp32
) {
    _rmsnorm_bf16(x, gamma, tmp_fp32, K, eps);
    return _quantize_absmax(tmp_fp32, out_i8, K);
}

// ---------------------------------------------------------------------------
// AVX2 helpers for the matmul hot paths.
// ---------------------------------------------------------------------------

// Horizontal sum of an __m256i (8 int32 lanes).
static inline int32_t hsum_i32_x8(__m256i v) {
    __m128i lo = _mm256_castsi256_si128(v);
    __m128i hi = _mm256_extracti128_si256(v, 1);
    __m128i s  = _mm_add_epi32(lo, hi);
    s = _mm_add_epi32(s, _mm_shuffle_epi32(s, _MM_SHUFFLE(1, 0, 3, 2)));
    s = _mm_add_epi32(s, _mm_shuffle_epi32(s, _MM_SHUFFLE(2, 3, 0, 1)));
    return _mm_cvtsi128_si32(s);
}

// Unpack 16 packed bytes (4 2-bit weights each) into 64 contiguous unsigned
// nibbles in [0, 2]. Output is written to a 64-byte buffer.
//
// Method: extract the 4 2-bit streams via 16-bit shifts + AND, then
// interleave with VPUNPCK* to recover the per-position nibble order.
// All ops are within 128-bit lanes; no lane-crossing needed.
static inline void unpack_64_nibbles(const uint8_t* __restrict__ packed16,
                                     uint8_t*       __restrict__ out64) {
    const __m128i p    = _mm_loadu_si128(reinterpret_cast<const __m128i*>(packed16));
    const __m128i mask = _mm_set1_epi8(0x3);
    // (b >> 2i) & 0x3 for i in {0,1,2,3}. _mm_srli_epi16 spills bits across
    // adjacent byte pairs but the post-AND with 0x3 cleans that up.
    const __m128i s0 = _mm_and_si128(p, mask);
    const __m128i s1 = _mm_and_si128(_mm_srli_epi16(p, 2), mask);
    const __m128i s2 = _mm_and_si128(_mm_srli_epi16(p, 4), mask);
    const __m128i s3 = _mm_and_si128(_mm_srli_epi16(p, 6), mask);
    // Interleave bytes: (s0,s1) -> 16 byte-pairs, (s2,s3) -> 16 byte-pairs.
    const __m128i s01_lo = _mm_unpacklo_epi8(s0, s1);
    const __m128i s01_hi = _mm_unpackhi_epi8(s0, s1);
    const __m128i s23_lo = _mm_unpacklo_epi8(s2, s3);
    const __m128i s23_hi = _mm_unpackhi_epi8(s2, s3);
    // Interleave 16-bit words -> 32-bit quads of {n_{4k},n_{4k+1},n_{4k+2},n_{4k+3}}.
    const __m128i q0 = _mm_unpacklo_epi16(s01_lo, s23_lo);  // nibbles  0..15
    const __m128i q1 = _mm_unpackhi_epi16(s01_lo, s23_lo);  // nibbles 16..31
    const __m128i q2 = _mm_unpacklo_epi16(s01_hi, s23_hi);  // nibbles 32..47
    const __m128i q3 = _mm_unpackhi_epi16(s01_hi, s23_hi);  // nibbles 48..63
    _mm_storeu_si128(reinterpret_cast<__m128i*>(out64 +  0), q0);
    _mm_storeu_si128(reinterpret_cast<__m128i*>(out64 + 16), q1);
    _mm_storeu_si128(reinterpret_cast<__m128i*>(out64 + 32), q2);
    _mm_storeu_si128(reinterpret_cast<__m128i*>(out64 + 48), q3);
}

} // namespace

extern "C" {

// ============================================================================
// Ternary packed matmul, M=1 (token generation).
//
// y[n] = (x_scale * w_scale) * sum_k w_{n,k} * x_k
// where w in {-1, 0, +1} packed 4-per-byte as an unsigned-shifted nibble
// (encoded value in {0, 1, 2} per 2-bit slot, decode = enc - 1).
//
// Identical contract to matmul_lut_avx512_m1.
// ============================================================================
LSPK_API void matmul_lut_avx2_m1(
    const int8_t*  __restrict__ x,
    const uint8_t* __restrict__ packed_w,
    float w_scale, float x_scale,
    float* __restrict__ y,
    int N, int K
) {
    const int Kb = K >> 2;
    const float out_scale = x_scale * w_scale;

    // sum_x is reused once per matmul to undo the +1 bias on the
    // unsigned-nibble dot product. Vectorized via VPMADDUBSW with all-ones.
    int32_t sum_x = 0;
    {
        const __m256i ones_u8 = _mm256_set1_epi8(1);
        const __m256i ones_s16 = _mm256_set1_epi16(1);
        __m256i sx = _mm256_setzero_si256();
        int k = 0;
        for (; k + 32 <= K; k += 32) {
            const __m256i a = _mm256_loadu_si256(reinterpret_cast<const __m256i*>(x + k));
            // |a| up to 127 so signed*unsigned-1 fits in int16; saturation does not trigger.
            const __m256i p16 = _mm256_maddubs_epi16(ones_u8, a);
            sx = _mm256_add_epi32(sx, _mm256_madd_epi16(p16, ones_s16));
        }
        sum_x = hsum_i32_x8(sx);
        for (; k < K; ++k) sum_x += static_cast<int32_t>(x[k]);
    }

    const __m256i ones16 = _mm256_set1_epi16(1);

#pragma omp parallel for if(N >= 64) schedule(static)
    for (int n = 0; n < N; ++n) {
        const uint8_t* row = packed_w + static_cast<ptrdiff_t>(n) * Kb;
        // Two parallel accumulators so the OOO core can keep multiple
        // VPMADDUBSW/VPMADDWD pairs in flight.
        __m256i acc0 = _mm256_setzero_si256();
        __m256i acc1 = _mm256_setzero_si256();

        int kb = 0;
        // Each iteration consumes 16 packed bytes (64 weights / activations).
        alignas(32) uint8_t nibbles[64];
        for (; kb + 16 <= Kb; kb += 16) {
            unpack_64_nibbles(row + kb, nibbles);
            const __m256i n0 = _mm256_load_si256(reinterpret_cast<const __m256i*>(nibbles +  0));
            const __m256i n1 = _mm256_load_si256(reinterpret_cast<const __m256i*>(nibbles + 32));
            const __m256i a0 = _mm256_loadu_si256(reinterpret_cast<const __m256i*>(x + (kb << 2) +  0));
            const __m256i a1 = _mm256_loadu_si256(reinterpret_cast<const __m256i*>(x + (kb << 2) + 32));
            // u8 (n in [0,2]) * s8 (a in [-127,127]) -> s16, sum of 2 adjacent.
            // Then s16 -> s32, sum of 2 adjacent. Net: dot product over 4 lanes.
            acc0 = _mm256_add_epi32(acc0,
                _mm256_madd_epi16(_mm256_maddubs_epi16(n0, a0), ones16));
            acc1 = _mm256_add_epi32(acc1,
                _mm256_madd_epi16(_mm256_maddubs_epi16(n1, a1), ones16));
        }
        int32_t sum = hsum_i32_x8(_mm256_add_epi32(acc0, acc1));

        // Scalar tail for the remaining packed bytes (Kb % 16 != 0).
        for (; kb < Kb; ++kb) {
            const uint8_t b = row[kb];
            const int kk = kb << 2;
            sum += static_cast<int32_t>((b >> 0) & 0x3) * static_cast<int32_t>(x[kk + 0]);
            sum += static_cast<int32_t>((b >> 2) & 0x3) * static_cast<int32_t>(x[kk + 1]);
            sum += static_cast<int32_t>((b >> 4) & 0x3) * static_cast<int32_t>(x[kk + 2]);
            sum += static_cast<int32_t>((b >> 6) & 0x3) * static_cast<int32_t>(x[kk + 3]);
        }
        y[n] = static_cast<float>(sum - sum_x) * out_scale;
    }
}

// ============================================================================
// Ternary packed matmul, batched M=T (prefill). T <= 64.
// ============================================================================
#define LSPK_AVX2_MAX_T 64

LSPK_API void matmul_lut_avx2_mT(
    const int8_t*  __restrict__ x,
    const float*   __restrict__ x_scales,
    const uint8_t* __restrict__ packed_w,
    float w_scale,
    float* __restrict__ y,
    int T, int N, int K
) {
    if (T <= 0 || T > LSPK_AVX2_MAX_T) return;
    const int Kb = K >> 2;

    // Per-token sum(x) via VPMADDUBSW with all-ones (same trick as m1).
    int32_t sum_x[LSPK_AVX2_MAX_T];
    {
        const __m256i ones_u8  = _mm256_set1_epi8(1);
        const __m256i ones_s16 = _mm256_set1_epi16(1);
        for (int t = 0; t < T; ++t) {
            __m256i sx = _mm256_setzero_si256();
            const int8_t* xt = x + static_cast<ptrdiff_t>(t) * K;
            int k = 0;
            for (; k + 32 <= K; k += 32) {
                const __m256i a = _mm256_loadu_si256(reinterpret_cast<const __m256i*>(xt + k));
                const __m256i p16 = _mm256_maddubs_epi16(ones_u8, a);
                sx = _mm256_add_epi32(sx, _mm256_madd_epi16(p16, ones_s16));
            }
            int32_t s = hsum_i32_x8(sx);
            for (; k < K; ++k) s += static_cast<int32_t>(xt[k]);
            sum_x[t] = s;
        }
    }

    const __m256i ones16 = _mm256_set1_epi16(1);

#pragma omp parallel for if(N >= 64) schedule(static)
    for (int n = 0; n < N; ++n) {
        const uint8_t* row = packed_w + static_cast<ptrdiff_t>(n) * Kb;
        __m256i acc[LSPK_AVX2_MAX_T];
        for (int t = 0; t < T; ++t) acc[t] = _mm256_setzero_si256();

        int kb = 0;
        alignas(32) uint8_t nibbles[64];
        // Unpack 16 packed bytes ONCE per kb-iteration; reuse across T tokens.
        // This is where T > 1 wins big over T calls to the m1 kernel.
        for (; kb + 16 <= Kb; kb += 16) {
            unpack_64_nibbles(row + kb, nibbles);
            const __m256i n0 = _mm256_load_si256(reinterpret_cast<const __m256i*>(nibbles +  0));
            const __m256i n1 = _mm256_load_si256(reinterpret_cast<const __m256i*>(nibbles + 32));
            for (int t = 0; t < T; ++t) {
                const int8_t* xt = x + static_cast<ptrdiff_t>(t) * K + (kb << 2);
                const __m256i a0 = _mm256_loadu_si256(reinterpret_cast<const __m256i*>(xt +  0));
                const __m256i a1 = _mm256_loadu_si256(reinterpret_cast<const __m256i*>(xt + 32));
                acc[t] = _mm256_add_epi32(acc[t],
                    _mm256_madd_epi16(_mm256_maddubs_epi16(n0, a0), ones16));
                acc[t] = _mm256_add_epi32(acc[t],
                    _mm256_madd_epi16(_mm256_maddubs_epi16(n1, a1), ones16));
            }
        }

        for (int t = 0; t < T; ++t) {
            int32_t sum = hsum_i32_x8(acc[t]);
            const int8_t* xt = x + static_cast<ptrdiff_t>(t) * K;
            // Scalar tail
            for (int kbt = kb; kbt < Kb; ++kbt) {
                const uint8_t b = row[kbt];
                const int kk = kbt << 2;
                sum += static_cast<int32_t>((b >> 0) & 0x3) * static_cast<int32_t>(xt[kk + 0]);
                sum += static_cast<int32_t>((b >> 2) & 0x3) * static_cast<int32_t>(xt[kk + 1]);
                sum += static_cast<int32_t>((b >> 4) & 0x3) * static_cast<int32_t>(xt[kk + 2]);
                sum += static_cast<int32_t>((b >> 6) & 0x3) * static_cast<int32_t>(xt[kk + 3]);
            }
            const float s = x_scales[t] * w_scale;
            y[static_cast<ptrdiff_t>(t) * N + n] =
                static_cast<float>(sum - sum_x[t]) * s;
        }
    }
}

// ============================================================================
// Pre-unpacked weights matmul (one int8 weight per byte). Optional symbol;
// kernel.py loads it conditionally.
// ============================================================================
LSPK_API void matmul_unpacked_avx2_m1(
    const int8_t*  __restrict__ x,
    const uint8_t* __restrict__ w_u8,
    float w_scale, float x_scale,
    float* __restrict__ y,
    int N, int K
) {
    const float out_scale = x_scale * w_scale;
    const __m256i ones_u8  = _mm256_set1_epi8(1);
    const __m256i ones_s16 = _mm256_set1_epi16(1);

    // sum_x via the same VPMADDUBSW(all-ones, x) -> VPMADDWD(*, all-ones) chain.
    int32_t sum_x = 0;
    {
        __m256i sx = _mm256_setzero_si256();
        int k = 0;
        for (; k + 32 <= K; k += 32) {
            const __m256i a = _mm256_loadu_si256(reinterpret_cast<const __m256i*>(x + k));
            sx = _mm256_add_epi32(sx,
                _mm256_madd_epi16(_mm256_maddubs_epi16(ones_u8, a), ones_s16));
        }
        sum_x = hsum_i32_x8(sx);
        for (; k < K; ++k) sum_x += static_cast<int32_t>(x[k]);
    }

#pragma omp parallel for if(N >= 64) schedule(static)
    for (int n = 0; n < N; ++n) {
        const uint8_t* row = w_u8 + static_cast<ptrdiff_t>(n) * K;
        __m256i acc0 = _mm256_setzero_si256();
        __m256i acc1 = _mm256_setzero_si256();

        int k = 0;
        // 4-way unrolled by lane-width (32 u8 per ymm), two accumulators for ILP.
        for (; k + 64 <= K; k += 64) {
            const __m256i w0 = _mm256_loadu_si256(reinterpret_cast<const __m256i*>(row + k +  0));
            const __m256i w1 = _mm256_loadu_si256(reinterpret_cast<const __m256i*>(row + k + 32));
            const __m256i a0 = _mm256_loadu_si256(reinterpret_cast<const __m256i*>(x   + k +  0));
            const __m256i a1 = _mm256_loadu_si256(reinterpret_cast<const __m256i*>(x   + k + 32));
            acc0 = _mm256_add_epi32(acc0,
                _mm256_madd_epi16(_mm256_maddubs_epi16(w0, a0), ones_s16));
            acc1 = _mm256_add_epi32(acc1,
                _mm256_madd_epi16(_mm256_maddubs_epi16(w1, a1), ones_s16));
        }
        // 32-byte tail
        for (; k + 32 <= K; k += 32) {
            const __m256i w = _mm256_loadu_si256(reinterpret_cast<const __m256i*>(row + k));
            const __m256i a = _mm256_loadu_si256(reinterpret_cast<const __m256i*>(x   + k));
            acc0 = _mm256_add_epi32(acc0,
                _mm256_madd_epi16(_mm256_maddubs_epi16(w, a), ones_s16));
        }
        int32_t sum = hsum_i32_x8(_mm256_add_epi32(acc0, acc1));
        for (; k < K; ++k) sum += static_cast<int32_t>(row[k]) * static_cast<int32_t>(x[k]);

        y[n] = static_cast<float>(sum - sum_x) * out_scale;
    }
}

// ============================================================================
// Activation quantization (fp32 -> int8, per-tensor absmax).
// ============================================================================
LSPK_API float quantize_activation_avx2(
    const float* __restrict__ x_fp32,
    int8_t*      __restrict__ x_int8_out,
    int K
) {
    return _quantize_absmax(x_fp32, x_int8_out, K);
}

// ============================================================================
// LM head: logits[v] = sum_h w[v,h] * x[h]
// Three weight dtypes: bf16, int8 (per-row scale), int4 (2 nibbles/byte).
// ============================================================================
LSPK_API void lm_head_bf16_fp32_avx2(
    const uint16_t* __restrict__ emb,
    const float*    __restrict__ x,
    float*          __restrict__ logits,
    int V, int H
) {
#pragma omp parallel for if(V >= 64) schedule(static)
    for (int v = 0; v < V; ++v) {
        const uint16_t* row = emb + static_cast<ptrdiff_t>(v) * H;
        float acc = 0.0f;
        for (int h = 0; h < H; ++h) acc += bf16_to_fp32(row[h]) * x[h];
        logits[v] = acc;
    }
}

LSPK_API void lm_head_int8_fp32_avx2(
    const int8_t* __restrict__ emb,
    const float*  __restrict__ emb_scale,
    const float*  __restrict__ x,
    float*        __restrict__ logits,
    int V, int H
) {
#pragma omp parallel for if(V >= 64) schedule(static)
    for (int v = 0; v < V; ++v) {
        const int8_t* row = emb + static_cast<ptrdiff_t>(v) * H;
        float acc = 0.0f;
        for (int h = 0; h < H; ++h) acc += static_cast<float>(row[h]) * x[h];
        logits[v] = emb_scale[v] * acc;
    }
}

LSPK_API void lm_head_int4_fp32_avx2(
    const uint8_t* __restrict__ emb,
    const float*   __restrict__ emb_scale,
    const float*   __restrict__ x,
    float*         __restrict__ logits,
    int V, int H
) {
    // Encoding: 2 nibbles per byte; signed via the (nib XOR 8) - 8 trick.
    // Lower nibble first, upper nibble second.
#pragma omp parallel for if(V >= 64) schedule(static)
    for (int v = 0; v < V; ++v) {
        const uint8_t* row = emb + static_cast<ptrdiff_t>(v) * (H / 2);
        float acc = 0.0f;
        const int Hpairs = H >> 1;
        for (int p = 0; p < Hpairs; ++p) {
            const uint8_t byte = row[p];
            // signed-nibble decode: (u XOR 8) - 8 maps {0..7, 8..15} to
            // {0..7, -8..-1}, matching the packed encoding (q in [-7,+7]).
            const int lo_n = static_cast<int>((byte & 0x0F) ^ 0x08) - 8;
            const int hi_n = static_cast<int>(((byte >> 4) & 0x0F) ^ 0x08) - 8;
            acc += static_cast<float>(lo_n) * x[p * 2 + 0];
            acc += static_cast<float>(hi_n) * x[p * 2 + 1];
        }
        logits[v] = emb_scale[v] * acc;
    }
}

// ============================================================================
// RMSNorm (bf16 gamma + fp32 gamma variants).
// ============================================================================
LSPK_API void rmsnorm_bf16gamma_fp32_avx2(
    const float*    __restrict__ x,
    const uint16_t* __restrict__ gamma,
    float*          __restrict__ out,
    int K, float eps
) {
    _rmsnorm_bf16(x, gamma, out, K, eps);
}

LSPK_API void rmsnorm_fp32gamma_fp32_avx2(
    const float* __restrict__ x,
    const float* __restrict__ gamma,
    float*       __restrict__ out,
    int K, float eps
) {
    _rmsnorm_fp32(x, gamma, out, K, eps);
}

// ============================================================================
// Fused MLP gate: out = max(gate, 0)^2 * up.
// ============================================================================
LSPK_API void relu2_mul_fp32_avx2(
    const float* __restrict__ gate,
    const float* __restrict__ up,
    float*       __restrict__ out,
    int K
) {
    for (int k = 0; k < K; ++k) {
        float g = gate[k];
        if (g < 0.0f) g = 0.0f;
        out[k] = g * g * up[k];
    }
}

// ============================================================================
// In-place add: a += b.
// ============================================================================
LSPK_API void add_inplace_fp32_avx2(
    float*       __restrict__ a,
    const float* __restrict__ b,
    int K
) {
    for (int k = 0; k < K; ++k) a[k] += b[k];
}

// ============================================================================
// Fused RMSNorm+quantize variants.
// ============================================================================
LSPK_API float rmsnorm_quantize_bf16gamma_avx2(
    const float*    __restrict__ x,
    const uint16_t* __restrict__ gamma,
    int8_t*         __restrict__ x_int8_out,
    float*          __restrict__ tmp_fp32,
    int K, float eps
) {
    return _rmsnorm_quantize_bf16_core(x, gamma, x_int8_out, K, eps, tmp_fp32);
}

LSPK_API void rmsnorm_quantize_bf16gamma_batched_avx2(
    const float*    __restrict__ x,
    const uint16_t* __restrict__ gamma,
    int8_t*         __restrict__ x_int8_out,
    float*          __restrict__ scales_out,
    float*          __restrict__ tmp_fp32,
    int T, int K, float eps
) {
#pragma omp parallel for if(T >= 2) schedule(static)
    for (int t = 0; t < T; ++t) {
        scales_out[t] = _rmsnorm_quantize_bf16_core(
            x + static_cast<ptrdiff_t>(t) * K,
            gamma,
            x_int8_out + static_cast<ptrdiff_t>(t) * K,
            K, eps,
            tmp_fp32 + static_cast<ptrdiff_t>(t) * K);
    }
}

// ============================================================================
// Batched helpers (per-T parallel calls to the per-row versions above).
// ============================================================================
LSPK_API void rmsnorm_bf16gamma_fp32_batched_avx2(
    const float*    __restrict__ x,
    const uint16_t* __restrict__ gamma,
    float*          __restrict__ out,
    int T, int K, float eps
) {
#pragma omp parallel for if(T >= 2) schedule(static)
    for (int t = 0; t < T; ++t) {
        _rmsnorm_bf16(x + static_cast<ptrdiff_t>(t) * K, gamma,
                      out + static_cast<ptrdiff_t>(t) * K, K, eps);
    }
}

LSPK_API void rmsnorm_fp32gamma_fp32_batched_avx2(
    const float* __restrict__ x,
    const float* __restrict__ gamma,
    float*       __restrict__ out,
    int T, int K, float eps
) {
#pragma omp parallel for if(T >= 2) schedule(static)
    for (int t = 0; t < T; ++t) {
        _rmsnorm_fp32(x + static_cast<ptrdiff_t>(t) * K, gamma,
                      out + static_cast<ptrdiff_t>(t) * K, K, eps);
    }
}

LSPK_API void quantize_activation_batched_avx2(
    const float* __restrict__ x,
    int8_t*      __restrict__ x_int8,
    float*       __restrict__ out_scales,
    int T, int K
) {
#pragma omp parallel for if(T >= 2) schedule(static)
    for (int t = 0; t < T; ++t) {
        out_scales[t] = _quantize_absmax(
            x + static_cast<ptrdiff_t>(t) * K,
            x_int8 + static_cast<ptrdiff_t>(t) * K, K);
    }
}

LSPK_API void relu2_mul_fp32_batched_avx2(
    const float* __restrict__ gate,
    const float* __restrict__ up,
    float*       __restrict__ out,
    int T, int K
) {
#pragma omp parallel for if(T >= 2) schedule(static)
    for (int t = 0; t < T; ++t) {
        const float* g = gate + static_cast<ptrdiff_t>(t) * K;
        const float* u = up   + static_cast<ptrdiff_t>(t) * K;
        float*       o = out  + static_cast<ptrdiff_t>(t) * K;
        for (int k = 0; k < K; ++k) {
            float gv = g[k];
            if (gv < 0.0f) gv = 0.0f;
            o[k] = gv * gv * u[k];
        }
    }
}

// ============================================================================
// Informational probes.
// ============================================================================
LSPK_API int matmul_lut_avx2_has_omp(void) {
#if defined(_OPENMP)
    return 1;
#else
    return 0;
#endif
}

LSPK_API int matmul_lut_avx2_max_threads(void) {
#if defined(_OPENMP)
    return omp_get_max_threads();
#else
    return 1;
#endif
}

}  // extern "C"

#if defined(_WIN32)
#include <Python.h>
extern "C" {
PyMODINIT_FUNC PyInit__matmul_lut_avx2(void) {
    static PyMethodDef methods[] = { {NULL, NULL, 0, NULL} };
    static PyModuleDef def = {
        PyModuleDef_HEAD_INIT, "_matmul_lut_avx2", NULL, -1, methods,
        NULL, NULL, NULL, NULL,
    };
    return PyModule_Create(&def);
}
}
#endif
