// Torchless extern "C" boundary for ternary matmul on x86_64 (AVX-512).
//
// Mirror of arm64/matmul_lut_neon_extern_c.cpp. Same contracts, same packed
// weight format -- so litespark_inference/torchless/runtime.py and
// litespark_inference/torchless/kernel.py can dispatch by architecture
// without changing call sites.
//
// Required ISA: AVX-512F + AVX-512BW + AVX-512VNNI (VPDPBUSD).
// Targets:
//   - Linux/Windows on Ice Lake/Cascade Lake (Intel) and Zen 4+ (AMD)
//   - Sapphire Rapids / Granite Rapids (servers)
//   - Skylake-X is NOT supported (no VNNI). Fall back to the torch-backed
//     path on those hosts (LITESPARK_FORCE_TORCH=1).
//
// Phase 1 sketch: not yet performance-tuned. Built for correctness +
// minimal surface so the torchless runtime works on x86 hosts. Phase 2
// can specialize the LM head matmuls and add an AVX2 fallback.

#include <immintrin.h>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <cstring>

#if defined(_OPENMP)
#include <omp.h>
#endif


// Horizontal sum of an AVX-512 int32 register.
static inline int32_t hsum_i32_x16(__m512i v) {
    return _mm512_reduce_add_epi32(v);
}

// Horizontal sum of an AVX-512 fp32 register.
static inline float hsum_f32_x16(__m512 v) {
    return _mm512_reduce_add_ps(v);
}

// Decode 16 packed bytes (64 ternary weights) into a 64-lane int8 register
// in natural order. Each byte encodes 4 ternary weights as 2 bits each:
//   w_i = ((b >> (2*i)) & 0x3) - 1   in {-1, 0, +1}
//
// Reads packed[0..15] and writes 64 int8 lanes in [0..63].
static inline __m512i unpack_64_ternary(__m128i packed16) {
    // Broadcast each of the 16 input bytes into 4 consecutive output lanes,
    // then shift+mask to extract the 4 nibble-pairs from each byte. The
    // simplest correct implementation uses a shuffle to spread bytes:
    //   out = shuffle([b0,b0,b0,b0, b1,b1,b1,b1, ..., b15,b15,b15,b15])
    // then per-lane right-shift by [0,2,4,6, 0,2,4,6, ...] and mask 0x3.
    static const __m512i spread = _mm512_set_epi8(
        15,15,15,15, 14,14,14,14, 13,13,13,13, 12,12,12,12,
        11,11,11,11, 10,10,10,10,  9, 9, 9, 9,  8, 8, 8, 8,
         7, 7, 7, 7,  6, 6, 6, 6,  5, 5, 5, 5,  4, 4, 4, 4,
         3, 3, 3, 3,  2, 2, 2, 2,  1, 1, 1, 1,  0, 0, 0, 0
    );
    static const __m512i shifts = _mm512_set1_epi32(0x06040200);  // bytes [0,2,4,6]
    static const __m512i mask03 = _mm512_set1_epi8(0x03);
    static const __m512i one    = _mm512_set1_epi8(1);

    const __m512i broadcast = _mm512_broadcast_i32x4(_mm512_castsi512_si128(_mm512_castsi128_si512(packed16)));
    // Spread bytes: each input byte appears 4 times in a row.
    const __m512i spread_bytes = _mm512_shuffle_epi8(broadcast, spread);
    // Shift each lane right by the appropriate amount. AVX-512 has a
    // per-element variable shift for 32-bit lanes (vpsrlvd) but we want
    // per-byte. Use the fact that 4 consecutive bytes map to shifts
    // {0,2,4,6}: pack into a 32-bit lane and use vpsrlvd with broadcast.
    //
    // Easier path: do it in two halves with vpmultishiftqb (AVX-512VBMI),
    // but we're avoiding VBMI to keep the ISA bar at AVX-512F+BW+VNNI.
    // So we do four parallel shifts via blend/and/or:
    const __m512i s0 =                                  spread_bytes;
    const __m512i s1 = _mm512_srli_epi16(spread_bytes, 2);
    const __m512i s2 = _mm512_srli_epi16(spread_bytes, 4);
    const __m512i s3 = _mm512_srli_epi16(spread_bytes, 6);

    // Select among s0..s3 per-byte based on lane index mod 4.
    // Build a mask: lane i takes shift_i = (i & 3).
    // Easiest: do 4 separate AND+OR with mask patterns.
    static const __m512i pattern0 = _mm512_set1_epi32(0x000000FF);  // lanes 0
    static const __m512i pattern1 = _mm512_set1_epi32(0x0000FF00);  // lanes 1
    static const __m512i pattern2 = _mm512_set1_epi32(0x00FF0000);  // lanes 2
    static const __m512i pattern3 = _mm512_set1_epi32(0xFF000000);  // lanes 3

    const __m512i p0 = _mm512_and_si512(s0, pattern0);
    const __m512i p1 = _mm512_and_si512(s1, pattern1);
    const __m512i p2 = _mm512_and_si512(s2, pattern2);
    const __m512i p3 = _mm512_and_si512(s3, pattern3);
    const __m512i merged = _mm512_or_si512(_mm512_or_si512(p0, p1), _mm512_or_si512(p2, p3));

    // Mask to 2 bits, then subtract 1 to get signed {-1, 0, +1}.
    return _mm512_sub_epi8(_mm512_and_si512(merged, mask03), one);
}


extern "C" {

// y[n] = x_scale * w_scale * sum_k (decoded packed weight) * x[k]
//
// Same contract as matmul_lut_neon_m1 in arm64/matmul_lut_neon_extern_c.cpp.
//
// x:        int8 [K]
// packed_w: uint8 [N, K/4]  (K must be a multiple of 4)
// w_scale:  fp32 scalar (per-tensor)
// x_scale:  fp32 scalar
// y:        fp32 [N]  (output, caller-allocated)
void matmul_lut_avx512_m1(
    const int8_t*  __restrict__ x,
    const uint8_t* __restrict__ packed_w,
    float w_scale,
    float x_scale,
    float* __restrict__ y,
    int N,
    int K
) {
    const int Kb = K >> 2;  // K / 4 (packed bytes per row)
    const float out_scale = x_scale * w_scale;

#pragma omp parallel for if(N >= 64) schedule(static)
    for (int n = 0; n < N; ++n) {
        const uint8_t* row = packed_w + static_cast<ptrdiff_t>(n) * Kb;
        __m512i acc = _mm512_setzero_si512();

        int kb = 0;
        // Fast path: 16 packed bytes -> 64 ternary weights -> one VNNI dot.
        for (; kb + 16 <= Kb; kb += 16) {
            const __m128i packed = _mm_loadu_si128(reinterpret_cast<const __m128i*>(row + kb));
            const __m512i w64 = unpack_64_ternary(packed);
            const __m512i x64 = _mm512_loadu_si512(reinterpret_cast<const __m512i*>(x + (kb << 2)));
            // VPDPBUSD: a is unsigned i8, b is signed i8. Our weights are
            // already signed; activations are signed too. Bias the weights
            // up by +1 (range [0,2]) to use VPDPBUSD safely, then subtract
            // sum(x) at the end. Cleaner alternative: use VPDPBSSD if
            // AVX-VNNI-INT8 is available; here we stick to baseline VNNI.
            //
            // Simpler correctness-first impl: widen w64 to int16, then
            // multiply-accumulate with widened x. ~2x slower than VNNI but
            // we'll tune in phase 2.
            const __m256i w_lo = _mm512_extracti64x4_epi64(w64, 0);
            const __m256i w_hi = _mm512_extracti64x4_epi64(w64, 1);
            const __m256i x_lo = _mm512_extracti64x4_epi64(x64, 0);
            const __m256i x_hi = _mm512_extracti64x4_epi64(x64, 1);
            // Promote int8 -> int16 (signed).
            const __m512i w16_lo = _mm512_cvtepi8_epi16(w_lo);
            const __m512i w16_hi = _mm512_cvtepi8_epi16(w_hi);
            const __m512i x16_lo = _mm512_cvtepi8_epi16(x_lo);
            const __m512i x16_hi = _mm512_cvtepi8_epi16(x_hi);
            // 16-bit multiply-add into int32 accumulator (VPMADDWD).
            acc = _mm512_add_epi32(acc, _mm512_madd_epi16(w16_lo, x16_lo));
            acc = _mm512_add_epi32(acc, _mm512_madd_epi16(w16_hi, x16_hi));
        }

        int32_t sum = hsum_i32_x16(acc);

        // Tail: byte at a time.
        for (; kb < Kb; ++kb) {
            const uint8_t b = row[kb];
            const int k = kb << 2;
            sum += (static_cast<int32_t>((b >> 0) & 0x3) - 1) * static_cast<int32_t>(x[k + 0]);
            sum += (static_cast<int32_t>((b >> 2) & 0x3) - 1) * static_cast<int32_t>(x[k + 1]);
            sum += (static_cast<int32_t>((b >> 4) & 0x3) - 1) * static_cast<int32_t>(x[k + 2]);
            sum += (static_cast<int32_t>((b >> 6) & 0x3) - 1) * static_cast<int32_t>(x[k + 3]);
        }

        y[n] = static_cast<float>(sum) * out_scale;
    }
}

// Per-tensor absmax quantization of an fp32 activation vector to int8.
float quantize_activation_avx512(
    const float* __restrict__ x_fp32,
    int8_t*      __restrict__ x_int8_out,
    int K
) {
    // Pass 1: absmax.
    __m512 absmax_v = _mm512_setzero_ps();
    const __m512 sign_mask = _mm512_castsi512_ps(_mm512_set1_epi32(0x7FFFFFFF));
    int k = 0;
    for (; k + 16 <= K; k += 16) {
        const __m512 v = _mm512_loadu_ps(x_fp32 + k);
        absmax_v = _mm512_max_ps(absmax_v, _mm512_and_ps(v, sign_mask));
    }
    float absmax = _mm512_reduce_max_ps(absmax_v);
    for (; k < K; ++k) {
        const float a = std::fabs(x_fp32[k]);
        if (a > absmax) absmax = a;
    }
    if (absmax < 1e-5f) absmax = 1e-5f;
    const float scale = absmax / 127.0f;
    const float inv_scale = 1.0f / scale;

    // Pass 2: quantize.
    const __m512 inv_v = _mm512_set1_ps(inv_scale);
    k = 0;
    for (; k + 16 <= K; k += 16) {
        const __m512 v = _mm512_mul_ps(_mm512_loadu_ps(x_fp32 + k), inv_v);
        // Convert to int32 with current rounding mode (default = nearest-even).
        const __m512i i32 = _mm512_cvtps_epi32(v);
        // Saturating narrow int32 -> int8.
        const __m128i i8 = _mm512_cvtsepi32_epi8(i32);
        _mm_storeu_si128(reinterpret_cast<__m128i*>(x_int8_out + k), i8);
    }
    for (; k < K; ++k) {
        float v = x_fp32[k] * inv_scale;
        int32_t iv = static_cast<int32_t>(std::round(v));
        if (iv >  127) iv =  127;
        if (iv < -127) iv = -127;
        x_int8_out[k] = static_cast<int8_t>(iv);
    }
    return scale;
}

// LM head matmul: logits[v] = sum_h bf16_to_fp32(emb[v, h]) * x[h]
// emb is uint16-viewed bf16. bf16 -> fp32 = shift bits into upper half.
void lm_head_bf16_fp32_avx512(
    const uint16_t* __restrict__ emb,
    const float*    __restrict__ x,
    float*          __restrict__ logits,
    int V, int H
) {
#pragma omp parallel for if(V >= 64) schedule(static)
    for (int v = 0; v < V; ++v) {
        const uint16_t* row = emb + static_cast<ptrdiff_t>(v) * H;
        __m512 acc = _mm512_setzero_ps();
        int h = 0;
        for (; h + 16 <= H; h += 16) {
            // bf16 -> fp32: zero-extend u16 to u32, then shift left by 16.
            const __m256i b16 = _mm256_loadu_si256(reinterpret_cast<const __m256i*>(row + h));
            const __m512i b32 = _mm512_cvtepu16_epi32(b16);
            const __m512  w   = _mm512_castsi512_ps(_mm512_slli_epi32(b32, 16));
            const __m512  xv  = _mm512_loadu_ps(x + h);
            acc = _mm512_fmadd_ps(w, xv, acc);
        }
        float tail = 0.0f;
        for (; h < H; ++h) {
            uint32_t bits = static_cast<uint32_t>(row[h]) << 16;
            float w;
            std::memcpy(&w, &bits, sizeof(float));
            tail += w * x[h];
        }
        logits[v] = tail + hsum_f32_x16(acc);
    }
}

// LM head matmul with per-row int8 quantized embeddings.
void lm_head_int8_fp32_avx512(
    const int8_t* __restrict__ emb,
    const float*  __restrict__ emb_scale,
    const float*  __restrict__ x,
    float*        __restrict__ logits,
    int V, int H
) {
#pragma omp parallel for if(V >= 64) schedule(static)
    for (int v = 0; v < V; ++v) {
        const int8_t* row = emb + static_cast<ptrdiff_t>(v) * H;
        __m512 acc = _mm512_setzero_ps();
        int h = 0;
        for (; h + 16 <= H; h += 16) {
            const __m128i b8 = _mm_loadu_si128(reinterpret_cast<const __m128i*>(row + h));
            // int8 -> int32 -> fp32 (signed widen).
            const __m512i b32 = _mm512_cvtepi8_epi32(b8);
            const __m512  w   = _mm512_cvtepi32_ps(b32);
            const __m512  xv  = _mm512_loadu_ps(x + h);
            acc = _mm512_fmadd_ps(w, xv, acc);
        }
        float tail = 0.0f;
        for (; h < H; ++h) tail += static_cast<float>(row[h]) * x[h];
        logits[v] = emb_scale[v] * (tail + hsum_f32_x16(acc));
    }
}

// LM head matmul with per-row int4 quantized embeddings (2 nibbles/byte).
void lm_head_int4_fp32_avx512(
    const uint8_t* __restrict__ emb,
    const float*   __restrict__ emb_scale,
    const float*   __restrict__ x,
    float*         __restrict__ logits,
    int V, int H
) {
#pragma omp parallel for if(V >= 64) schedule(static)
    for (int v = 0; v < V; ++v) {
        const uint8_t* row = emb + static_cast<ptrdiff_t>(v) * (H / 2);
        __m512 acc = _mm512_setzero_ps();
        int h = 0;
        // Inner loop: 8 packed bytes -> 16 int4 -> 16 fp32 -> 16 FMAs.
        for (; h + 16 <= H; h += 16) {
            // Load 8 packed bytes into the low half of a 128-bit reg.
            const __m128i p8 = _mm_loadl_epi64(reinterpret_cast<const __m128i*>(row + (h >> 1)));
            // Sign-extend low nibble: (b << 4) >> 4 (arithmetic).
            const __m128i lo = _mm_srai_epi16(_mm_slli_epi16(p8, 12), 12);  // 16-bit lanes
            // Sign-extend high nibble: b >> 4 (arithmetic).
            const __m128i hi = _mm_srai_epi16(p8, 4);
            // Above is wrong-resolution: int4 is byte-level. Re-do at byte:
            // We want each byte b -> two signed int8s: lo = (b<<4)>>4 (signed),
            // hi = b>>4 (signed). SSE doesn't have a byte-arith-shift, so
            // emulate via word arithmetic and mask.
            // Cleaner: process via 16-bit math, see scalar tail for the
            // canonical formula. Phase 2: vectorize this block properly.
            (void)lo; (void)hi;
            float partial = 0.0f;
            for (int j = 0; j < 16; j += 2) {
                const uint8_t b = row[(h + j) >> 1];
                const int8_t  lo_s = static_cast<int8_t>(static_cast<int8_t>(b << 4) >> 4);
                const int8_t  hi_s = static_cast<int8_t>(b) >> 4;
                partial += static_cast<float>(lo_s) * x[h + j];
                partial += static_cast<float>(hi_s) * x[h + j + 1];
            }
            // Fold partial into acc as a single-element add (cheap).
            acc = _mm512_add_ps(acc, _mm512_set_ps(0,0,0,0, 0,0,0,0, 0,0,0,0, 0,0,0,partial));
        }
        float tail = 0.0f;
        for (; h < H; h += 2) {
            const uint8_t b = row[h >> 1];
            const int8_t  lo = static_cast<int8_t>(static_cast<int8_t>(b << 4) >> 4);
            const int8_t  hi = static_cast<int8_t>(b) >> 4;
            tail += static_cast<float>(lo) * x[h];
            if (h + 1 < H) tail += static_cast<float>(hi) * x[h + 1];
        }
        logits[v] = emb_scale[v] * (tail + hsum_f32_x16(acc));
    }
}

// RMSNorm with bf16 gamma.
void rmsnorm_bf16gamma_fp32_avx512(
    const float*    __restrict__ x,
    const uint16_t* __restrict__ gamma,
    float*          __restrict__ out,
    int K,
    float eps
) {
    __m512 ssq = _mm512_setzero_ps();
    int k = 0;
    for (; k + 16 <= K; k += 16) {
        const __m512 v = _mm512_loadu_ps(x + k);
        ssq = _mm512_fmadd_ps(v, v, ssq);
    }
    float sum_sq = hsum_f32_x16(ssq);
    for (; k < K; ++k) sum_sq += x[k] * x[k];
    const float rrms = 1.0f / std::sqrt(sum_sq / static_cast<float>(K) + eps);

    const __m512 rrms_v = _mm512_set1_ps(rrms);
    k = 0;
    for (; k + 16 <= K; k += 16) {
        const __m512 v   = _mm512_loadu_ps(x + k);
        const __m256i g16 = _mm256_loadu_si256(reinterpret_cast<const __m256i*>(gamma + k));
        const __m512i g32 = _mm512_cvtepu16_epi32(g16);
        const __m512  g   = _mm512_castsi512_ps(_mm512_slli_epi32(g32, 16));
        _mm512_storeu_ps(out + k, _mm512_mul_ps(_mm512_mul_ps(v, rrms_v), g));
    }
    for (; k < K; ++k) {
        uint32_t gbits = static_cast<uint32_t>(gamma[k]) << 16;
        float g;
        std::memcpy(&g, &gbits, sizeof(float));
        out[k] = x[k] * rrms * g;
    }
}

// RMSNorm with fp32 gamma.
void rmsnorm_fp32gamma_fp32_avx512(
    const float* __restrict__ x,
    const float* __restrict__ gamma,
    float*       __restrict__ out,
    int K,
    float eps
) {
    __m512 ssq = _mm512_setzero_ps();
    int k = 0;
    for (; k + 16 <= K; k += 16) {
        const __m512 v = _mm512_loadu_ps(x + k);
        ssq = _mm512_fmadd_ps(v, v, ssq);
    }
    float sum_sq = hsum_f32_x16(ssq);
    for (; k < K; ++k) sum_sq += x[k] * x[k];
    const float rrms = 1.0f / std::sqrt(sum_sq / static_cast<float>(K) + eps);

    const __m512 rrms_v = _mm512_set1_ps(rrms);
    k = 0;
    for (; k + 16 <= K; k += 16) {
        _mm512_storeu_ps(out + k,
            _mm512_mul_ps(_mm512_mul_ps(_mm512_loadu_ps(x + k), rrms_v),
                          _mm512_loadu_ps(gamma + k)));
    }
    for (; k < K; ++k) out[k] = x[k] * rrms * gamma[k];
}

// Fused BitNet MLP gate: out = max(gate, 0)^2 * up.
void relu2_mul_fp32_avx512(
    const float* __restrict__ gate,
    const float* __restrict__ up,
    float*       __restrict__ out,
    int K
) {
    const __m512 zero = _mm512_setzero_ps();
    int k = 0;
    for (; k + 16 <= K; k += 16) {
        const __m512 g = _mm512_max_ps(_mm512_loadu_ps(gate + k), zero);
        _mm512_storeu_ps(out + k, _mm512_mul_ps(_mm512_mul_ps(g, g), _mm512_loadu_ps(up + k)));
    }
    for (; k < K; ++k) {
        float g = gate[k];
        if (g < 0.0f) g = 0.0f;
        out[k] = g * g * up[k];
    }
}

// In-place add: a += b.
void add_inplace_fp32_avx512(
    float*       __restrict__ a,
    const float* __restrict__ b,
    int K
) {
    int k = 0;
    for (; k + 16 <= K; k += 16) {
        _mm512_storeu_ps(a + k, _mm512_add_ps(_mm512_loadu_ps(a + k), _mm512_loadu_ps(b + k)));
    }
    for (; k < K; ++k) a[k] += b[k];
}

// Informational probes (mirror the NEON ones).
int matmul_lut_avx512_has_omp(void) {
#if defined(_OPENMP)
    return 1;
#else
    return 0;
#endif
}

int matmul_lut_avx512_max_threads(void) {
#if defined(_OPENMP)
    return omp_get_max_threads();
#else
    return 0;
#endif
}

}  // extern "C"
