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
//
// Two implementations:
//   * VBMI path (Cannon Lake+, Ice Lake+, Zen 4+, Sapphire Rapids+):
//     vpmultishiftqb does all 4 nibble-pair shifts in one instruction
//     and runs on port 5, leaving port 0 free for VPDPBUSD.
//   * Fallback (AVX-512 BW only): 4 separate shifts + ANDs + ORs.
//
// Both end with a single mask to 2 bits. Selected by setup.py via
// -DLITESPARK_USE_VBMI; we use an explicit macro because MSVC doesn't
// predefine __AVX512VBMI__ even when /arch:AVX512 is set.
#if defined(LITESPARK_USE_VBMI) || defined(__AVX512VBMI__)
static inline __m512i unpack_64_unsigned(__m128i packed16) {
    // Per-qword: each output byte i takes bits [shift_i+7 .. shift_i] from
    // the qword. After spread, each qword contains [b0,b0,b0,b0,b1,b1,b1,b1]
    // (two distinct input bytes, each replicated 4x). We want output:
    //   byte 0 = b0 bits[1:0]   -> qword bits[1:0]   -> shift  0
    //   byte 1 = b0 bits[3:2]   -> qword bits[3:2]   -> shift  2
    //   byte 2 = b0 bits[5:4]   -> qword bits[5:4]   -> shift  4
    //   byte 3 = b0 bits[7:6]   -> qword bits[7:6]   -> shift  6
    //   byte 4 = b1 bits[1:0]   -> qword bits[33:32] -> shift 32
    //   byte 5 = b1 bits[3:2]   -> qword bits[35:34] -> shift 34
    //   byte 6 = b1 bits[5:4]   -> qword bits[37:36] -> shift 36
    //   byte 7 = b1 bits[7:6]   -> qword bits[39:38] -> shift 38
    // Mask 0x03 trims the upper 6 bits of each extracted byte.
    static const __m512i spread = _mm512_set_epi8(
        15,15,15,15, 14,14,14,14, 13,13,13,13, 12,12,12,12,
        11,11,11,11, 10,10,10,10,  9, 9, 9, 9,  8, 8, 8, 8,
         7, 7, 7, 7,  6, 6, 6, 6,  5, 5, 5, 5,  4, 4, 4, 4,
         3, 3, 3, 3,  2, 2, 2, 2,  1, 1, 1, 1,  0, 0, 0, 0
    );
    static const __m512i shifts = _mm512_set1_epi64(0x2624222006040200ULL);
    static const __m512i mask03 = _mm512_set1_epi8(0x03);

    const __m512i broadcast = _mm512_broadcast_i32x4(
        _mm512_castsi512_si128(_mm512_castsi128_si512(packed16)));
    const __m512i spread_bytes = _mm512_shuffle_epi8(broadcast, spread);
    const __m512i shifted = _mm512_multishift_epi64_epi8(shifts, spread_bytes);
    return _mm512_and_si512(shifted, mask03);
}
#else
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
#endif


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

        // Four parallel accumulators so the OOO core can have multiple
        // VPDPBUSDs in flight (~5 cycle latency, 1/cycle throughput on
        // Ice Lake+; with one acc the dependency chain serializes them).
        __m512i acc0 = _mm512_setzero_si512();
        __m512i acc1 = _mm512_setzero_si512();
        __m512i acc2 = _mm512_setzero_si512();
        __m512i acc3 = _mm512_setzero_si512();

        int kb = 0;
        // 4-way unrolled fast path: 64 packed bytes / 256 unsigned nibbles
        // / 256 activations / 4 VPDPBUSDs per iter.
        for (; kb + 64 <= Kb; kb += 64) {
            const __m128i p0 = _mm_loadu_si128(reinterpret_cast<const __m128i*>(row + kb +  0));
            const __m128i p1 = _mm_loadu_si128(reinterpret_cast<const __m128i*>(row + kb + 16));
            const __m128i p2 = _mm_loadu_si128(reinterpret_cast<const __m128i*>(row + kb + 32));
            const __m128i p3 = _mm_loadu_si128(reinterpret_cast<const __m128i*>(row + kb + 48));
            const __m512i n0 = unpack_64_unsigned(p0);
            const __m512i n1 = unpack_64_unsigned(p1);
            const __m512i n2 = unpack_64_unsigned(p2);
            const __m512i n3 = unpack_64_unsigned(p3);
            const int kx = kb << 2;
            const __m512i x0 = _mm512_loadu_si512(reinterpret_cast<const __m512i*>(x + kx +   0));
            const __m512i x1 = _mm512_loadu_si512(reinterpret_cast<const __m512i*>(x + kx +  64));
            const __m512i x2 = _mm512_loadu_si512(reinterpret_cast<const __m512i*>(x + kx + 128));
            const __m512i x3 = _mm512_loadu_si512(reinterpret_cast<const __m512i*>(x + kx + 192));
            acc0 = _mm512_dpbusd_epi32(acc0, n0, x0);
            acc1 = _mm512_dpbusd_epi32(acc1, n1, x1);
            acc2 = _mm512_dpbusd_epi32(acc2, n2, x2);
            acc3 = _mm512_dpbusd_epi32(acc3, n3, x3);
        }
        // 16-byte tail (Kb%64 != 0): keep using acc0.
        for (; kb + 16 <= Kb; kb += 16) {
            const __m128i packed = _mm_loadu_si128(reinterpret_cast<const __m128i*>(row + kb));
            const __m512i n64 = unpack_64_unsigned(packed);
            const __m512i x64 = _mm512_loadu_si512(reinterpret_cast<const __m512i*>(x + (kb << 2)));
            acc0 = _mm512_dpbusd_epi32(acc0, n64, x64);
        }

        // Reduce four accumulators -> one int32. We subtract the full
        // sum_x at the end to convert sum_k n_k*x_k (n in [0,2]) into
        // sum_k (n_k - 1)*x_k = sum_k w_k*x_k.
        const __m512i acc = _mm512_add_epi32(
            _mm512_add_epi32(acc0, acc1),
            _mm512_add_epi32(acc2, acc3));
        int32_t sum = _mm512_reduce_add_epi32(acc);

        // Byte tail: unbiased (n in [0, 2], no -1 here).
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

// Batched (M=T tokens) matmul with the same packed weights. Used by the
// prefill path so a T-token prompt costs one matmul kernel call per
// projection instead of T calls. Per output row we unpack the weights
// ONCE and then issue T VPDPBUSDs against the T per-token activations,
// amortizing the unpack and the weight read across the whole batch.
//
// Layout:
//   x:        int8 [T, K]   row-major; x[t, k] at x[t*K + k]
//   x_scales: fp32 [T]
//   packed_w: uint8 [N, K/4] (same as the M=1 kernel)
//   w_scale:  fp32 (per-tensor)
//   y:        fp32 [T, N]   row-major
//
// T cap is 64 -- bigger batches should chunk on the caller side. Each
// active token holds one int32 accumulator in zmm; with T=64 the
// compiler will spill some, but correctness is preserved and the win
// over T M=1 calls is the shared unpack and the shared L1 weight read,
// not register-resident accumulators.
#define LSPK_MAX_T 64

LSPK_API void matmul_lut_avx512_mT(
    const int8_t*  __restrict__ x,
    const float*   __restrict__ x_scales,
    const uint8_t* __restrict__ packed_w,
    float w_scale,
    float* __restrict__ y,
    int T, int N, int K
) {
    if (T <= 0 || T > LSPK_MAX_T) return;
    const int Kb = K >> 2;

    // Per-token sum(x) for the +1 bias correction (unsigned-nibble dot).
    int32_t sum_x[LSPK_MAX_T];
    {
        const __m512i ones_u8 = _mm512_set1_epi8(1);
        for (int t = 0; t < T; ++t) {
            __m512i sx_v = _mm512_setzero_si512();
            const int8_t* xt = x + static_cast<ptrdiff_t>(t) * K;
            int k = 0;
            for (; k + 64 <= K; k += 64) {
                const __m512i x64 = _mm512_loadu_si512(reinterpret_cast<const __m512i*>(xt + k));
                sx_v = _mm512_dpbusd_epi32(sx_v, ones_u8, x64);
            }
            int32_t s = _mm512_reduce_add_epi32(sx_v);
            for (; k < K; ++k) s += static_cast<int32_t>(xt[k]);
            sum_x[t] = s;
        }
    }

#pragma omp parallel for if(N >= 64) schedule(static)
    for (int n = 0; n < N; ++n) {
        const uint8_t* row = packed_w + static_cast<ptrdiff_t>(n) * Kb;

        __m512i acc[LSPK_MAX_T];
        for (int t = 0; t < T; ++t) acc[t] = _mm512_setzero_si512();

        int kb = 0;
        // Inner loop: 16 packed bytes / 64 unsigned nibbles unpacked
        // ONCE, then T VPDPBUSDs against the T per-token activation
        // chunks. T independent accumulators break the dep chain.
        for (; kb + 16 <= Kb; kb += 16) {
            const __m128i packed = _mm_loadu_si128(
                reinterpret_cast<const __m128i*>(row + kb));
            const __m512i n_chunk = unpack_64_unsigned(packed);
            const int kx = kb << 2;
            for (int t = 0; t < T; ++t) {
                const __m512i x_chunk = _mm512_loadu_si512(
                    reinterpret_cast<const __m512i*>(x + static_cast<ptrdiff_t>(t) * K + kx));
                acc[t] = _mm512_dpbusd_epi32(acc[t], n_chunk, x_chunk);
            }
        }

        // Reduce per-token accumulators -> int32 partial sums.
        int32_t partial[LSPK_MAX_T];
        for (int t = 0; t < T; ++t) {
            partial[t] = _mm512_reduce_add_epi32(acc[t]);
        }

        // Tail: byte at a time, unbiased.
        for (; kb < Kb; ++kb) {
            const uint8_t b = row[kb];
            const int kx = kb << 2;
            const int32_t n0 = (b >> 0) & 0x3;
            const int32_t n1 = (b >> 2) & 0x3;
            const int32_t n2 = (b >> 4) & 0x3;
            const int32_t n3 = (b >> 6) & 0x3;
            for (int t = 0; t < T; ++t) {
                const int8_t* xt = x + static_cast<ptrdiff_t>(t) * K;
                partial[t] += n0 * static_cast<int32_t>(xt[kx + 0])
                           +  n1 * static_cast<int32_t>(xt[kx + 1])
                           +  n2 * static_cast<int32_t>(xt[kx + 2])
                           +  n3 * static_cast<int32_t>(xt[kx + 3]);
            }
        }

        for (int t = 0; t < T; ++t) {
            const float scale_t = x_scales[t] * w_scale;
            y[static_cast<ptrdiff_t>(t) * N + n] =
                static_cast<float>(partial[t] - sum_x[t]) * scale_t;
        }
    }
}


// Same matmul as above but consumes PRE-UNPACKED weights (one u8 per
// weight in [0,1,2] -- the same encoding the packed format would expand
// to but stored 4x bigger). Saves the entire port-5 unpack chain at the
// cost of 4x weight memory. Useful on hosts with plenty of RAM where
// compute, not bandwidth, is the bottleneck.
//
// Weights are unsigned ([0,1,2]); we recover the canonical (n-1) bias
// the same way as the packed kernel: subtract sum_x once per row.
//
// x:        int8 [K]
// w_u8:     uint8 [N, K]   (pre-unpacked, K must be a multiple of 64)
// w_scale:  fp32 (per-tensor absmean)
// x_scale:  fp32
// y:        fp32 [N]       (caller-allocated)
LSPK_API void matmul_unpacked_avx512_m1(
    const int8_t*  __restrict__ x,
    const uint8_t* __restrict__ w_u8,
    float w_scale,
    float x_scale,
    float* __restrict__ y,
    int N,
    int K
) {
    const float out_scale = x_scale * w_scale;

    // sum(x) once per matmul, used to undo the +1 unsigned bias on w.
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
        const uint8_t* row = w_u8 + static_cast<ptrdiff_t>(n) * K;
        __m512i acc0 = _mm512_setzero_si512();
        __m512i acc1 = _mm512_setzero_si512();
        __m512i acc2 = _mm512_setzero_si512();
        __m512i acc3 = _mm512_setzero_si512();

        int kk = 0;
        // 4-way unrolled: 256 weights / 256 activations / 4 VPDPBUSDs per iter.
        // No unpack -- just two loads per VPDPBUSD.
        for (; kk + 256 <= K; kk += 256) {
            const __m512i n0 = _mm512_loadu_si512(reinterpret_cast<const __m512i*>(row + kk +   0));
            const __m512i n1 = _mm512_loadu_si512(reinterpret_cast<const __m512i*>(row + kk +  64));
            const __m512i n2 = _mm512_loadu_si512(reinterpret_cast<const __m512i*>(row + kk + 128));
            const __m512i n3 = _mm512_loadu_si512(reinterpret_cast<const __m512i*>(row + kk + 192));
            const __m512i x0 = _mm512_loadu_si512(reinterpret_cast<const __m512i*>(x + kk +   0));
            const __m512i x1 = _mm512_loadu_si512(reinterpret_cast<const __m512i*>(x + kk +  64));
            const __m512i x2 = _mm512_loadu_si512(reinterpret_cast<const __m512i*>(x + kk + 128));
            const __m512i x3 = _mm512_loadu_si512(reinterpret_cast<const __m512i*>(x + kk + 192));
            acc0 = _mm512_dpbusd_epi32(acc0, n0, x0);
            acc1 = _mm512_dpbusd_epi32(acc1, n1, x1);
            acc2 = _mm512_dpbusd_epi32(acc2, n2, x2);
            acc3 = _mm512_dpbusd_epi32(acc3, n3, x3);
        }
        // 64-byte tail
        for (; kk + 64 <= K; kk += 64) {
            const __m512i n0 = _mm512_loadu_si512(reinterpret_cast<const __m512i*>(row + kk));
            const __m512i x0 = _mm512_loadu_si512(reinterpret_cast<const __m512i*>(x   + kk));
            acc0 = _mm512_dpbusd_epi32(acc0, n0, x0);
        }
        const __m512i acc = _mm512_add_epi32(
            _mm512_add_epi32(acc0, acc1),
            _mm512_add_epi32(acc2, acc3));
        int32_t sum = _mm512_reduce_add_epi32(acc);

        // Scalar tail (K % 64 != 0)
        for (; kk < K; ++kk) {
            sum += static_cast<int32_t>(row[kk]) * static_cast<int32_t>(x[kk]);
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

// ---------------------------------------------------------------------------
// AMX matmul (Sapphire Rapids+, optional via -DLITESPARK_USE_AMX).
//
// Uses TDPBSSD (signed * signed int8) with weights pre-transposed at load
// time into AMX-VNNI layout: for a [N, K] int8 weight in {-1, 0, +1}, the
// AMX layout is [K/4, N, 4] -- i.e. each row of the layout holds 4 K-
// adjacent weights for one of N output channels, packed as int8 into a
// 4-byte group. TILELOADD reads a (rows=16, cols=64) tile in one shot.
//
// Tile config for our matmul:
//   tile 0 = A  : M_tile rows  * 64 cols i8     (T tokens, 64 K elements)
//   tile 1 = B  : 16 rows      * 64 cols i8     (16 N channels, 64 K in VNNI groups)
//   tile 2 = C  : M_tile rows  * 64 cols i32    (16 N channels, int32 result)
//
// Unsigned-bias / sum_x trick is not needed since TDPBSSD takes signed
// operands directly. We just feed the [-1, 0, +1] weights as int8.
//
// M_tile capped at 16 (AMX hard limit). Caller chunks T > 16 outside.
#if defined(LITESPARK_USE_AMX)
#include <immintrin.h>

// Linux requires explicit opt-in to use AMX from userspace. Without this
// arch_prctl call, the first TILELOADD/TDPBSSD raises SIGILL even on a
// CPU that has the feature. Returns 1 on success (AMX usable), 0 if the
// kernel refused (older kernel without dynamic xstate, or non-Linux).
#if defined(__linux__)
#include <sys/syscall.h>
#include <unistd.h>
#include <errno.h>
#ifndef ARCH_GET_XCOMP_PERM
#define ARCH_GET_XCOMP_PERM 0x1022
#endif
#ifndef ARCH_REQ_XCOMP_PERM
#define ARCH_REQ_XCOMP_PERM 0x1023
#endif
#ifndef XFEATURE_XTILEDATA
#define XFEATURE_XTILEDATA 18
#endif
#endif

LSPK_API int amx_request_permission(void) {
#if defined(__linux__)
    long r = syscall(SYS_arch_prctl, ARCH_REQ_XCOMP_PERM, XFEATURE_XTILEDATA);
    return r == 0 ? 1 : 0;
#else
    return 1;  // not Linux: assume the kernel doesn't gate AMX
#endif
}

struct __attribute__((packed)) amx_tile_config {
    uint8_t  palette_id;
    uint8_t  start_row;
    uint8_t  reserved[14];
    uint16_t colsb[16];
    uint8_t  rows[16];
};

static inline void _lspk_amx_load_cfg(int M_tile) {
    struct amx_tile_config cfg;
    std::memset(&cfg, 0, sizeof(cfg));
    cfg.palette_id = 1;
    cfg.start_row  = 0;
    // Tile 0: A   (M_tile rows, 64 cols i8)
    cfg.colsb[0] = 64;
    cfg.rows[0]  = static_cast<uint8_t>(M_tile);
    // Tile 1: B   (16 rows,     64 cols i8 -- one row = one N channel)
    cfg.colsb[1] = 64;
    cfg.rows[1]  = 16;
    // Tile 2: C   (M_tile rows, 64 cols i32 = 16 N int32s)
    cfg.colsb[2] = 64;
    cfg.rows[2]  = static_cast<uint8_t>(M_tile);
    _tile_loadconfig(&cfg);
}

// AMX-VNNI ternary matmul for M=T (T <= 16).
//
// x:        int8 [T, K]   (row-major)
// x_scales: fp32 [T]
// w_amx:    int8 [K/4, N, 4]   (pre-transposed; w[k_group][n][b] holds
//                               weight n at K-offset k_group*4 + b)
// w_scale:  fp32 (per-tensor)
// y:        fp32 [T, N]
LSPK_API void matmul_amx_int8_mT(
    const int8_t* __restrict__ x,
    const float*  __restrict__ x_scales,
    const int8_t* __restrict__ w_amx,
    float w_scale,
    float*        __restrict__ y,
    int T, int N, int K
) {
    if (T <= 0 || T > 16) return;        // caller chunks
    if ((K & 0x3F) != 0) return;          // K must be a multiple of 64
    if ((N & 0x0F) != 0) return;          // N must be a multiple of 16
    const int K_groups = K >> 2;          // 4 K-elements per VNNI group
    const int b_row_stride = N * 4;       // bytes per row of w_amx[k_group]

    _lspk_amx_load_cfg(T);

#pragma omp parallel for if(N >= 64) schedule(static)
    for (int n = 0; n < N; n += 16) {
        // Each thread owns its own tile state -- AMX is per-logical-CPU.
        // Re-load config in each thread to be safe.
        _lspk_amx_load_cfg(T);

        _tile_zero(2);
        for (int k = 0; k < K; k += 64) {
            _tile_loadd(0, x + k, K);
            // B tile: rows = 16 N channels, cols = 64 K bytes (VNNI grouped).
            // Source layout w_amx[k/4 + 0..15][n][0..3]:
            //   row r of the tile = w_amx[k_group+r] starting at column n*4
            // That row stride is N*4 bytes (b_row_stride).
            const int8_t* b_ptr = w_amx
                + static_cast<ptrdiff_t>(k >> 2) * static_cast<ptrdiff_t>(b_row_stride)
                + static_cast<ptrdiff_t>(n) * 4;
            _tile_loadd(1, b_ptr, b_row_stride);
            _tile_dpbssd(2, 0, 1);
        }
        // Spill the int32 tile and write fp32 result.
        alignas(64) int32_t buf[16 * 16];
        _tile_stored(2, buf, 16 * sizeof(int32_t));
        for (int t = 0; t < T; ++t) {
            const float scale_t = x_scales[t] * w_scale;
            for (int nn = 0; nn < 16; ++nn) {
                y[static_cast<ptrdiff_t>(t) * N + n + nn] =
                    static_cast<float>(buf[t * 16 + nn]) * scale_t;
            }
        }
    }

    _tile_release();
}

// One-shot helper: transpose a [N, K/4] packed ternary matrix into
// [K/4, N, 4] int8 AMX-VNNI layout. Called at load time.
//
// w_packed: uint8 [N, K/4]   (4 ternary weights per byte, encoding
//                             (w0+1) | (w1+1)<<2 | (w2+1)<<4 | (w3+1)<<6)
// w_amx:    int8 [K/4, N, 4]  (output, caller-allocated, 4*N*K/4 bytes)
LSPK_API void transpose_packed_to_amx_vnni(
    const uint8_t* __restrict__ w_packed,
    int8_t*        __restrict__ w_amx,
    int N, int K
) {
    const int Kb = K >> 2;
#pragma omp parallel for schedule(static)
    for (int kb = 0; kb < Kb; ++kb) {
        for (int n = 0; n < N; ++n) {
            const uint8_t b = w_packed[static_cast<ptrdiff_t>(n) * Kb + kb];
            int8_t* dst = w_amx
                + static_cast<ptrdiff_t>(kb) * N * 4
                + static_cast<ptrdiff_t>(n) * 4;
            dst[0] = static_cast<int8_t>((b >> 0) & 0x3) - 1;
            dst[1] = static_cast<int8_t>((b >> 2) & 0x3) - 1;
            dst[2] = static_cast<int8_t>((b >> 4) & 0x3) - 1;
            dst[3] = static_cast<int8_t>((b >> 6) & 0x3) - 1;
        }
    }
}
#endif  // LITESPARK_USE_AMX


// Batched (T-token) versions of the per-row helpers used by prefill.
// Each one processes T independent K-vectors in a single C call so the
// ctypes round-trip and any thread-launch overhead amortize across T.

LSPK_API void rmsnorm_bf16gamma_fp32_batched_avx512(
    const float*    __restrict__ x,      // [T, K]
    const uint16_t* __restrict__ gamma,  // [K]
    float*          __restrict__ out,    // [T, K]
    int T, int K, float eps
) {
#pragma omp parallel for if(T >= 2) schedule(static)
    for (int t = 0; t < T; ++t) {
        rmsnorm_bf16gamma_fp32_avx512(x + (ptrdiff_t)t * K, gamma, out + (ptrdiff_t)t * K, K, eps);
    }
}

LSPK_API void rmsnorm_fp32gamma_fp32_batched_avx512(
    const float* __restrict__ x,      // [T, K]
    const float* __restrict__ gamma,  // [K]
    float*       __restrict__ out,    // [T, K]
    int T, int K, float eps
) {
#pragma omp parallel for if(T >= 2) schedule(static)
    for (int t = 0; t < T; ++t) {
        rmsnorm_fp32gamma_fp32_avx512(x + (ptrdiff_t)t * K, gamma, out + (ptrdiff_t)t * K, K, eps);
    }
}

LSPK_API void quantize_activation_batched_avx512(
    const float* __restrict__ x,         // [T, K]
    int8_t*      __restrict__ x_int8,    // [T, K]
    float*       __restrict__ out_scales,// [T]
    int T, int K
) {
#pragma omp parallel for if(T >= 2) schedule(static)
    for (int t = 0; t < T; ++t) {
        out_scales[t] = quantize_activation_avx512(
            x + (ptrdiff_t)t * K, x_int8 + (ptrdiff_t)t * K, K);
    }
}

LSPK_API void relu2_mul_fp32_batched_avx512(
    const float* __restrict__ gate, // [T, K]
    const float* __restrict__ up,   // [T, K]
    float*       __restrict__ out,  // [T, K]
    int T, int K
) {
#pragma omp parallel for if(T >= 2) schedule(static)
    for (int t = 0; t < T; ++t) {
        relu2_mul_fp32_avx512(
            gate + (ptrdiff_t)t * K, up + (ptrdiff_t)t * K,
            out + (ptrdiff_t)t * K, K);
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
