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

// MSVC accepts `__restrict` (no trailing underscores) but not the GCC/Clang
// `__restrict__` form. Map down so the function signatures parse on both.
#if defined(_MSC_VER) && !defined(__clang__)
#  define __restrict__ __restrict
#endif

// Export every entry point for ctypes loading. On ELF (Linux/macOS) gcc/clang
// expose all extern "C" symbols by default; on Windows MSVC hides them
// unless dllexport is set. Default visibility is fine for ELF.
#if defined(_WIN32)
#  define LSPK_API __declspec(dllexport)
#else
#  define LSPK_API
#endif


// Horizontal sum of an AVX-512 int32 register.
static inline int32_t hsum_i32_x16(__m512i v) {
    return _mm512_reduce_add_epi32(v);
}

// Horizontal sum of an AVX-512 fp32 register.
static inline float hsum_f32_x16(__m512 v) {
    return _mm512_reduce_add_ps(v);
}

// Decode 16 packed bytes (64 ternary weights) as UNSIGNED nibbles in [0, 2]
// (the raw on-disk encoding before the canonical -1 bias). The VNNI matmul
// uses this and applies the -1 bias on the output side via a single
// sum(x) subtraction per row -- VPDPBUSD is unsigned*signed, so feeding
// it the unbiased nibbles keeps the operand types matching.
//
// Reads packed[0..15] and returns 64 u8 lanes in natural [w0..w63] order.
static inline __m512i unpack_64_unsigned(__m128i packed16) {
    static const __m512i spread = _mm512_set_epi8(
        15,15,15,15, 14,14,14,14, 13,13,13,13, 12,12,12,12,
        11,11,11,11, 10,10,10,10,  9, 9, 9, 9,  8, 8, 8, 8,
         7, 7, 7, 7,  6, 6, 6, 6,  5, 5, 5, 5,  4, 4, 4, 4,
         3, 3, 3, 3,  2, 2, 2, 2,  1, 1, 1, 1,  0, 0, 0, 0
    );
    static const __m512i mask03 = _mm512_set1_epi8(0x03);
    static const __m512i pattern0 = _mm512_set1_epi32(0x000000FF);
    static const __m512i pattern1 = _mm512_set1_epi32(0x0000FF00);
    static const __m512i pattern2 = _mm512_set1_epi32(0x00FF0000);
    static const __m512i pattern3 = _mm512_set1_epi32(0xFF000000);

    const __m512i broadcast = _mm512_broadcast_i32x4(
        _mm512_castsi512_si128(_mm512_castsi128_si512(packed16)));
    const __m512i spread_bytes = _mm512_shuffle_epi8(broadcast, spread);
    const __m512i s1 = _mm512_srli_epi16(spread_bytes, 2);
    const __m512i s2 = _mm512_srli_epi16(spread_bytes, 4);
    const __m512i s3 = _mm512_srli_epi16(spread_bytes, 6);

    const __m512i p0 = _mm512_and_si512(spread_bytes, pattern0);
    const __m512i p1 = _mm512_and_si512(s1,           pattern1);
    const __m512i p2 = _mm512_and_si512(s2,           pattern2);
    const __m512i p3 = _mm512_and_si512(s3,           pattern3);
    const __m512i merged = _mm512_or_si512(
        _mm512_or_si512(p0, p1), _mm512_or_si512(p2, p3));

    return _mm512_and_si512(merged, mask03);
}


extern "C" {

// y[n] = x_scale * w_scale * sum_k (decoded packed weight) * x[k]
//
// Same contract as matmul_lut_neon_m1 in arm64/matmul_lut_neon_extern_c.cpp.
//
// VNNI fast path: VPDPBUSD takes (uint8, int8) -> int32. Our packed nibbles
// are uint8 in [0, 2] (the unbiased encoding) and our activations are int8.
// We compute sum_k n_k * x_k via VPDPBUSD and then subtract sum_k x_k once
// per row to recover sum_k (n_k - 1) * x_k = sum_k w_k * x_k.
//
// x:        int8 [K]
// packed_w: uint8 [N, K/4]  (K must be a multiple of 4)
// w_scale:  fp32 scalar (per-tensor)
// x_scale:  fp32 scalar
// y:        fp32 [N]  (output, caller-allocated)
LSPK_API void matmul_lut_avx512_m1(
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

    // Compute sum(x) once via VPDPBUSD with a vector of unsigned ones.
    // Used to undo the +1 bias on the unsigned-nibble dot product below.
    const __m512i ones_u8 = _mm512_set1_epi8(1);
    __m512i sx_v = _mm512_setzero_si512();
    int k = 0;
    for (; k + 64 <= K; k += 64) {
        const __m512i x64 = _mm512_loadu_si512(reinterpret_cast<const __m512i*>(x + k));
        sx_v = _mm512_dpbusd_epi32(sx_v, ones_u8, x64);
    }
    int32_t sum_x = _mm512_reduce_add_epi32(sx_v);
    for (; k < K; ++k) sum_x += static_cast<int32_t>(x[k]);

#pragma omp parallel for if(N >= 64) schedule(static)
    for (int n = 0; n < N; ++n) {
        const uint8_t* row = packed_w + static_cast<ptrdiff_t>(n) * Kb;
        __m512i acc = _mm512_setzero_si512();

        int kb = 0;
        // Fast path: 16 packed bytes -> 64 unsigned nibbles -> one VPDPBUSD.
        // VPDPBUSD: src + dot4(uint8, int8) per int32 lane. 64 i8 lanes ->
        // 16 int32 lanes, each accumulating 4 products. Inner loop is
        // load + unpack + load + 1 VNNI = ~6 instructions for 64 MACs.
        for (; kb + 16 <= Kb; kb += 16) {
            const __m128i packed = _mm_loadu_si128(reinterpret_cast<const __m128i*>(row + kb));
            const __m512i n64 = unpack_64_unsigned(packed);
            const __m512i x64 = _mm512_loadu_si512(reinterpret_cast<const __m512i*>(x + (kb << 2)));
            acc = _mm512_dpbusd_epi32(acc, n64, x64);
        }

        // Both the fast loop and the tail accumulate sum_k n_k * x_k with
        // n in [0, 2]. We subtract the full sum_x at the end to convert
        // to sum_k (n_k - 1) * x_k = sum_k w_k * x_k.
        int32_t sum = _mm512_reduce_add_epi32(acc);

        // Tail: byte at a time, unbiased (n in [0, 2], no -1 here).
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

// Per-tensor absmax quantization of an fp32 activation vector to int8.
LSPK_API float quantize_activation_avx512(
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
LSPK_API void lm_head_bf16_fp32_avx512(
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
LSPK_API void lm_head_int8_fp32_avx512(
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
//
// Inner loop: 16 packed bytes -> 32 signed nibbles -> 32 fp32 -> 2 FMA-16s.
// Sign-extend each nibble via the XOR-bias trick:
//   nib_unsigned in [0, 15];
//   nib_signed   = (nib XOR 8) - 8     (operates per-byte, 8-bit lanes)
// Maps {0..7, 8..15} -> {0..7, -8..-1}, which is the standard 4-bit two's
// complement decode and matches our packed encoding (q & 0x0F, q in [-7,+7]).
LSPK_API void lm_head_int4_fp32_avx512(
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

        const __m128i mask_lo = _mm_set1_epi8(0x0F);
        const __m128i bias    = _mm_set1_epi8(0x08);

        int h = 0;
        for (; h + 32 <= H; h += 32) {
            // Load 16 packed bytes (32 nibbles).
            const __m128i p16 = _mm_loadu_si128(
                reinterpret_cast<const __m128i*>(row + (h >> 1)));

            // Low nibble of each byte -> position 2i, high nibble -> 2i+1.
            // For high nibbles, srli_epi16 by 4 then per-byte mask cleans
            // up the cross-byte bleed.
            const __m128i low_nib  = _mm_and_si128(p16, mask_lo);
            const __m128i high_nib = _mm_and_si128(_mm_srli_epi16(p16, 4), mask_lo);

            // Sign-extend nibbles in [0,15] -> int8 in [-8, 7].
            const __m128i low_s  = _mm_sub_epi8(_mm_xor_si128(low_nib,  bias), bias);
            const __m128i high_s = _mm_sub_epi8(_mm_xor_si128(high_nib, bias), bias);

            // Interleave to natural order [l0,h0, l1,h1, ..., l15,h15].
            const __m128i first16  = _mm_unpacklo_epi8(low_s, high_s);
            const __m128i second16 = _mm_unpackhi_epi8(low_s, high_s);

            // Widen int8 -> int32 -> fp32, FMA against x.
            const __m512 w_a = _mm512_cvtepi32_ps(_mm512_cvtepi8_epi32(first16));
            const __m512 w_b = _mm512_cvtepi32_ps(_mm512_cvtepi8_epi32(second16));
            acc = _mm512_fmadd_ps(w_a, _mm512_loadu_ps(x + h),      acc);
            acc = _mm512_fmadd_ps(w_b, _mm512_loadu_ps(x + h + 16), acc);
        }
        // Tail: 2 nibbles at a time.
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
LSPK_API void rmsnorm_bf16gamma_fp32_avx512(
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
LSPK_API void rmsnorm_fp32gamma_fp32_avx512(
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
LSPK_API void relu2_mul_fp32_avx512(
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
LSPK_API void add_inplace_fp32_avx512(
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
LSPK_API int matmul_lut_avx512_has_omp(void) {
#if defined(_OPENMP)
    return 1;
#else
    return 0;
#endif
}

LSPK_API int matmul_lut_avx512_max_threads(void) {
#if defined(_OPENMP)
    return omp_get_max_threads();
#else
    return 0;
#endif
}

}  // extern "C"

// MSVC's link.exe insists on a `PyInit_<module>` export when building a .pyd.
// We don't actually use the .pyd as a Python C extension -- it's loaded via
// ctypes from kernel.py -- but we need to satisfy the linker. The stub
// returns a minimal empty module so `import` would technically work too.
//
// Linux/clang doesn't require this and the symbol is harmless if exported.
#if defined(_WIN32)
#include <Python.h>
extern "C" {
PyMODINIT_FUNC PyInit__matmul_lut_avx512(void) {
    static PyMethodDef methods[] = { {NULL, NULL, 0, NULL} };
    static PyModuleDef def = {
        PyModuleDef_HEAD_INIT, "_matmul_lut_avx512", NULL, -1, methods,
        NULL, NULL, NULL, NULL,
    };
    return PyModule_Create(&def);
}
}  // extern "C"
#endif
