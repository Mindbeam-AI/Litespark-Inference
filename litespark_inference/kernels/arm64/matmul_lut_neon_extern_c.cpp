// Torchless extern "C" boundary for ternary matmul on ARM64.
//
// M=1 path (token generation): consumes 2-bit packed ternary weights in the
// same format as kernels/x86_64/matmul_lut_standalone.cpp and Python-side
// litespark_inference.torchless.pack.pack_ternary_4_per_byte:
//
//   packed byte = (w0+1) | ((w1+1)<<2) | ((w2+1)<<4) | ((w3+1)<<6)
//
// Fast inner loop: each iteration consumes 16 packed bytes (= 64 ternary
// weights), unpacks them with NEON shifts + zips into four int8x16_t
// vectors in natural element order [w0..w15], [w16..w31], [w32..w47],
// [w48..w63], then issues four vdotq_s32 against the matching activation
// vectors. That's 64 multiply-accumulates per inner iteration with only
// ~15 NEON instructions of unpack, vs ~64 instructions of scalar unpack
// previously. OpenMP parallelises over output rows when N >= 64.

#include <arm_neon.h>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <cstring>

#include <Accelerate/Accelerate.h>

#if defined(_OPENMP)
#include <omp.h>
#endif


// Unpack 16 packed bytes (64 ternary weights) into four int8x16_t vectors.
//   packed[i] = w[4i+0]<<0 | w[4i+1]<<2 | w[4i+2]<<4 | w[4i+3]<<6  (w in {0,1,2})
// After unpack:
//   *w0 = [w[0]-1 ..  w[15]-1]
//   *w1 = [w[16]-1 .. w[31]-1]
//   *w2 = [w[32]-1 .. w[47]-1]
//   *w3 = [w[48]-1 .. w[63]-1]
// Derivation:
//   a = packed & 0x3                       = [w[0], w[4], w[8], ..., w[60]]
//   b = (packed >> 2) & 0x3                = [w[1], w[5], ..., w[61]]
//   c = (packed >> 4) & 0x3                = [w[2], w[6], ..., w[62]]
//   d = (packed >> 6)                      = [w[3], w[7], ..., w[63]]
//   vzip1q_u8(a, b) low lanes -> [w0,w1,w4,w5,w8,w9,w12,w13,w16,w17,...]
//   reinterpret as u16 and vzip1q_u16 with cd_lo (similarly constructed)
//   gives [(w0,w1),(w2,w3),(w4,w5),...] = natural order.
static inline void unpack_16_bytes(
    uint8x16_t packed,
    int8x16_t* w0, int8x16_t* w1, int8x16_t* w2, int8x16_t* w3
) {
    const uint8x16_t mask = vdupq_n_u8(0x03);
    const int8x16_t  one  = vdupq_n_s8(1);

    const uint8x16_t a = vandq_u8(packed, mask);
    const uint8x16_t b = vandq_u8(vshrq_n_u8(packed, 2), mask);
    const uint8x16_t c = vandq_u8(vshrq_n_u8(packed, 4), mask);
    const uint8x16_t d = vshrq_n_u8(packed, 6);

    const uint8x16_t ab_lo = vzip1q_u8(a, b);
    const uint8x16_t ab_hi = vzip2q_u8(a, b);
    const uint8x16_t cd_lo = vzip1q_u8(c, d);
    const uint8x16_t cd_hi = vzip2q_u8(c, d);

    const uint16x8_t ab_lo_u16 = vreinterpretq_u16_u8(ab_lo);
    const uint16x8_t ab_hi_u16 = vreinterpretq_u16_u8(ab_hi);
    const uint16x8_t cd_lo_u16 = vreinterpretq_u16_u8(cd_lo);
    const uint16x8_t cd_hi_u16 = vreinterpretq_u16_u8(cd_hi);

    const uint16x8_t z0 = vzip1q_u16(ab_lo_u16, cd_lo_u16);
    const uint16x8_t z1 = vzip2q_u16(ab_lo_u16, cd_lo_u16);
    const uint16x8_t z2 = vzip1q_u16(ab_hi_u16, cd_hi_u16);
    const uint16x8_t z3 = vzip2q_u16(ab_hi_u16, cd_hi_u16);

    *w0 = vsubq_s8(vreinterpretq_s8_u16(z0), one);
    *w1 = vsubq_s8(vreinterpretq_s8_u16(z1), one);
    *w2 = vsubq_s8(vreinterpretq_s8_u16(z2), one);
    *w3 = vsubq_s8(vreinterpretq_s8_u16(z3), one);
}


extern "C" {

// y[n] = x_scale * w_scale * sum_k (decoded packed weight) * x[k]
//
// BitNet b1.58 uses a single per-tensor w_scale (weight.abs().mean()), not
// per-row -- matches prepare_ternary_weight in models.py and
// litespark_inference.torchless.quantize.quantize_ternary_absmean.
//
// x:        int8 [K]
// packed_w: uint8 [N, K/4]  (K must be a multiple of 4)
// w_scale:  fp32 scalar (per-tensor)
// x_scale:  fp32 scalar
// y:        fp32 [N]  (output, caller-allocated)
void matmul_lut_neon_m1(
    const int8_t*  __restrict__ x,
    const uint8_t* __restrict__ packed_w,
    float w_scale,
    float x_scale,
    float* __restrict__ y,
    int N,
    int K
) {
    const int Kb = K >> 2;  // K / 4
    const float out_scale = x_scale * w_scale;

#pragma omp parallel for if(N >= 64) schedule(static)
    for (int n = 0; n < N; ++n) {
        const uint8_t* row = packed_w + static_cast<ptrdiff_t>(n) * Kb;
        int32x4_t acc_v = vdupq_n_s32(0);

        int kb = 0;

        // Fast path: 16 packed bytes (64 weights, 64 activations) per iter.
        for (; kb + 16 <= Kb; kb += 16) {
            const uint8x16_t packed = vld1q_u8(row + kb);
            int8x16_t w0, w1, w2, w3;
            unpack_16_bytes(packed, &w0, &w1, &w2, &w3);

            const int k = kb << 2;
            const int8x16_t x0 = vld1q_s8(x + k + 0);
            const int8x16_t x1 = vld1q_s8(x + k + 16);
            const int8x16_t x2 = vld1q_s8(x + k + 32);
            const int8x16_t x3 = vld1q_s8(x + k + 48);

            acc_v = vdotq_s32(acc_v, w0, x0);
            acc_v = vdotq_s32(acc_v, w1, x1);
            acc_v = vdotq_s32(acc_v, w2, x2);
            acc_v = vdotq_s32(acc_v, w3, x3);
        }

        // Medium path: 4 packed bytes (16 weights, 1 SDOT).
        for (; kb + 4 <= Kb; kb += 4) {
            uint32_t p4;
            std::memcpy(&p4, row + kb, sizeof(p4));

            alignas(16) int8_t unp[16];
            for (int i = 0; i < 4; ++i) {
                const uint8_t b = static_cast<uint8_t>((p4 >> (i * 8)) & 0xFFu);
                unp[i * 4 + 0] = static_cast<int8_t>((b >> 0) & 0x3) - 1;
                unp[i * 4 + 1] = static_cast<int8_t>((b >> 2) & 0x3) - 1;
                unp[i * 4 + 2] = static_cast<int8_t>((b >> 4) & 0x3) - 1;
                unp[i * 4 + 3] = static_cast<int8_t>((b >> 6) & 0x3) - 1;
            }
            const int8x16_t w_vec = vld1q_s8(unp);
            const int8x16_t x_vec = vld1q_s8(x + kb * 4);
            acc_v = vdotq_s32(acc_v, w_vec, x_vec);
        }

        int32_t acc = vaddvq_s32(acc_v);

        // Tail: individual packed bytes (for weird Kb not divisible by 4).
        for (; kb < Kb; ++kb) {
            const uint8_t b = row[kb];
            const int k = kb << 2;
            acc += (static_cast<int32_t>((b >> 0) & 0x3) - 1) * static_cast<int32_t>(x[k + 0]);
            acc += (static_cast<int32_t>((b >> 2) & 0x3) - 1) * static_cast<int32_t>(x[k + 1]);
            acc += (static_cast<int32_t>((b >> 4) & 0x3) - 1) * static_cast<int32_t>(x[k + 2]);
            acc += (static_cast<int32_t>((b >> 6) & 0x3) - 1) * static_cast<int32_t>(x[k + 3]);
        }

        y[n] = static_cast<float>(acc) * out_scale;
    }
}

void matmul_accelerate_f32_neon_m1(
    const float* __restrict__ x,
    const float* __restrict__ w_float32,
    float*       __restrict__ y,
    int N,
    int K
) {
    cblas_sgemm(CblasRowMajor, CblasNoTrans, CblasTrans,
        1, N, K,
        1.0f, x, K,
        w_float32, K,
        0.0f, y, N);
}

// Batched ternary matmul for M>1 (prompt prefill).
//
// Same packed-weight format and arithmetic as matmul_lut_neon_m1, but it
// multiplies one weight matrix against M token activations at once. The key
// win over looping the M=1 kernel M times: each 16-byte packed block is
// unpacked ONCE per row and reused across a register-resident tile of MR
// tokens, and the whole weight matrix is streamed ~ceil(M/MR) times instead
// of M times. Unpack cost, weight memory traffic, and the per-call ctypes
// boundary all drop by ~MR, which is what makes prefill (TTFT) cheap.
//
// x:        int8 [M, K]   (row-major; one quantized activation vector per row)
// packed_w: uint8 [N, K/4]
// w_scale:  fp32 scalar (per-tensor, as in the M=1 path)
// x_scale:  fp32 [M]      (per-token activation scale)
// y:        fp32 [M, N]   (row-major output, caller-allocated)
void matmul_lut_neon_prefill(
    const int8_t*  __restrict__ x,
    const uint8_t* __restrict__ packed_w,
    float w_scale,
    const float*   __restrict__ x_scale,
    float* __restrict__ y,
    int M,
    int N,
    int K
) {
    const int Kb = K >> 2;  // K / 4
    constexpr int MR = 8;   // token tile held in registers

#pragma omp parallel for if(N >= 64) schedule(static)
    for (int n = 0; n < N; ++n) {
        const uint8_t* row = packed_w + static_cast<ptrdiff_t>(n) * Kb;

        // Full tiles of MR tokens: unpack each weight block once, fan out
        // across the MR activations.
        int m0 = 0;
        for (; m0 + MR <= M; m0 += MR) {
            int32x4_t acc[MR];
            for (int i = 0; i < MR; ++i) acc[i] = vdupq_n_s32(0);

            int kb = 0;
            for (; kb + 16 <= Kb; kb += 16) {
                const uint8x16_t packed = vld1q_u8(row + kb);
                int8x16_t w0, w1, w2, w3;
                unpack_16_bytes(packed, &w0, &w1, &w2, &w3);
                const int k = kb << 2;
                for (int i = 0; i < MR; ++i) {
                    const int8_t* xi = x + static_cast<ptrdiff_t>(m0 + i) * K + k;
                    acc[i] = vdotq_s32(acc[i], w0, vld1q_s8(xi + 0));
                    acc[i] = vdotq_s32(acc[i], w1, vld1q_s8(xi + 16));
                    acc[i] = vdotq_s32(acc[i], w2, vld1q_s8(xi + 32));
                    acc[i] = vdotq_s32(acc[i], w3, vld1q_s8(xi + 48));
                }
            }
            for (; kb + 4 <= Kb; kb += 4) {
                uint32_t p4;
                std::memcpy(&p4, row + kb, sizeof(p4));
                alignas(16) int8_t unp[16];
                for (int j = 0; j < 4; ++j) {
                    const uint8_t b = static_cast<uint8_t>((p4 >> (j * 8)) & 0xFFu);
                    unp[j * 4 + 0] = static_cast<int8_t>((b >> 0) & 0x3) - 1;
                    unp[j * 4 + 1] = static_cast<int8_t>((b >> 2) & 0x3) - 1;
                    unp[j * 4 + 2] = static_cast<int8_t>((b >> 4) & 0x3) - 1;
                    unp[j * 4 + 3] = static_cast<int8_t>((b >> 6) & 0x3) - 1;
                }
                const int8x16_t w_vec = vld1q_s8(unp);
                const int k = kb << 2;
                for (int i = 0; i < MR; ++i) {
                    acc[i] = vdotq_s32(acc[i], w_vec,
                                       vld1q_s8(x + static_cast<ptrdiff_t>(m0 + i) * K + k));
                }
            }

            int32_t tail_acc[MR];
            for (int i = 0; i < MR; ++i) tail_acc[i] = vaddvq_s32(acc[i]);
            for (; kb < Kb; ++kb) {
                const uint8_t b = row[kb];
                const int k = kb << 2;
                const int32_t w_0 = static_cast<int32_t>((b >> 0) & 0x3) - 1;
                const int32_t w_1 = static_cast<int32_t>((b >> 2) & 0x3) - 1;
                const int32_t w_2 = static_cast<int32_t>((b >> 4) & 0x3) - 1;
                const int32_t w_3 = static_cast<int32_t>((b >> 6) & 0x3) - 1;
                for (int i = 0; i < MR; ++i) {
                    const int8_t* xi = x + static_cast<ptrdiff_t>(m0 + i) * K + k;
                    tail_acc[i] += w_0 * xi[0] + w_1 * xi[1] + w_2 * xi[2] + w_3 * xi[3];
                }
            }
            for (int i = 0; i < MR; ++i) {
                const int m = m0 + i;
                y[static_cast<ptrdiff_t>(m) * N + n] =
                    static_cast<float>(tail_acc[i]) * (x_scale[m] * w_scale);
            }
        }

        // Remainder tokens (M % MR): fall back to the single-token path.
        for (int m = m0; m < M; ++m) {
            const int8_t* xm = x + static_cast<ptrdiff_t>(m) * K;
            int32x4_t acc_v = vdupq_n_s32(0);
            int kb = 0;
            for (; kb + 16 <= Kb; kb += 16) {
                const uint8x16_t packed = vld1q_u8(row + kb);
                int8x16_t w0, w1, w2, w3;
                unpack_16_bytes(packed, &w0, &w1, &w2, &w3);
                const int k = kb << 2;
                acc_v = vdotq_s32(acc_v, w0, vld1q_s8(xm + k + 0));
                acc_v = vdotq_s32(acc_v, w1, vld1q_s8(xm + k + 16));
                acc_v = vdotq_s32(acc_v, w2, vld1q_s8(xm + k + 32));
                acc_v = vdotq_s32(acc_v, w3, vld1q_s8(xm + k + 48));
            }
            for (; kb + 4 <= Kb; kb += 4) {
                uint32_t p4;
                std::memcpy(&p4, row + kb, sizeof(p4));
                alignas(16) int8_t unp[16];
                for (int j = 0; j < 4; ++j) {
                    const uint8_t b = static_cast<uint8_t>((p4 >> (j * 8)) & 0xFFu);
                    unp[j * 4 + 0] = static_cast<int8_t>((b >> 0) & 0x3) - 1;
                    unp[j * 4 + 1] = static_cast<int8_t>((b >> 2) & 0x3) - 1;
                    unp[j * 4 + 2] = static_cast<int8_t>((b >> 4) & 0x3) - 1;
                    unp[j * 4 + 3] = static_cast<int8_t>((b >> 6) & 0x3) - 1;
                }
                acc_v = vdotq_s32(acc_v, vld1q_s8(unp), vld1q_s8(xm + (kb << 2)));
            }
            int32_t acc = vaddvq_s32(acc_v);
            for (; kb < Kb; ++kb) {
                const uint8_t b = row[kb];
                const int k = kb << 2;
                acc += (static_cast<int32_t>((b >> 0) & 0x3) - 1) * static_cast<int32_t>(xm[k + 0]);
                acc += (static_cast<int32_t>((b >> 2) & 0x3) - 1) * static_cast<int32_t>(xm[k + 1]);
                acc += (static_cast<int32_t>((b >> 4) & 0x3) - 1) * static_cast<int32_t>(xm[k + 2]);
                acc += (static_cast<int32_t>((b >> 6) & 0x3) - 1) * static_cast<int32_t>(xm[k + 3]);
            }
            y[static_cast<ptrdiff_t>(m) * N + n] =
                static_cast<float>(acc) * (x_scale[m] * w_scale);
        }
    }
}

// Per-tensor absmax quantization of an fp32 activation vector to int8.
// Returns the scale (dequantized = int8 * scale). Caller provides the
// int8 output buffer, sized >= K. Two passes over x_fp32, both vectorized.
//
// Used by the torchless runtime to share a single quantized activation
// across q/k/v and gate/up projections, and to avoid numpy-side scans.
float quantize_activation_neon(
    const float* __restrict__ x_fp32,
    int8_t*      __restrict__ x_int8_out,
    int K
) {
    // ---- Pass 1: absmax ----
    float32x4_t max_v = vdupq_n_f32(0.0f);
    int k = 0;
    for (; k + 16 <= K; k += 16) {
        max_v = vmaxq_f32(max_v, vabsq_f32(vld1q_f32(x_fp32 + k +  0)));
        max_v = vmaxq_f32(max_v, vabsq_f32(vld1q_f32(x_fp32 + k +  4)));
        max_v = vmaxq_f32(max_v, vabsq_f32(vld1q_f32(x_fp32 + k +  8)));
        max_v = vmaxq_f32(max_v, vabsq_f32(vld1q_f32(x_fp32 + k + 12)));
    }
    for (; k + 4 <= K; k += 4) {
        max_v = vmaxq_f32(max_v, vabsq_f32(vld1q_f32(x_fp32 + k)));
    }
    float absmax = vmaxvq_f32(max_v);
    for (; k < K; ++k) {
        const float a = std::fabs(x_fp32[k]);
        if (a > absmax) absmax = a;
    }
    if (absmax < 1e-5f) absmax = 1e-5f;
    const float scale = absmax / 127.0f;
    const float inv_scale = 1.0f / scale;

    // ---- Pass 2: quantize ----
    const float32x4_t inv_v = vdupq_n_f32(inv_scale);
    k = 0;
    for (; k + 16 <= K; k += 16) {
        const float32x4_t v0 = vmulq_f32(vld1q_f32(x_fp32 + k +  0), inv_v);
        const float32x4_t v1 = vmulq_f32(vld1q_f32(x_fp32 + k +  4), inv_v);
        const float32x4_t v2 = vmulq_f32(vld1q_f32(x_fp32 + k +  8), inv_v);
        const float32x4_t v3 = vmulq_f32(vld1q_f32(x_fp32 + k + 12), inv_v);

        // vcvtaq_s32_f32: convert to int32 with round-half-away-from-zero.
        // Not identical to numpy's ties-to-even, but the logit-parity test
        // confirms the drift stays within noise.
        const int32x4_t i0 = vcvtaq_s32_f32(v0);
        const int32x4_t i1 = vcvtaq_s32_f32(v1);
        const int32x4_t i2 = vcvtaq_s32_f32(v2);
        const int32x4_t i3 = vcvtaq_s32_f32(v3);

        // Saturating narrow int32 -> int16 -> int8 (clips to [-128, 127]).
        const int16x8_t s01 = vcombine_s16(vqmovn_s32(i0), vqmovn_s32(i1));
        const int16x8_t s23 = vcombine_s16(vqmovn_s32(i2), vqmovn_s32(i3));
        const int8x16_t s   = vcombine_s8(vqmovn_s16(s01), vqmovn_s16(s23));
        vst1q_s8(x_int8_out + k, s);
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
//
// emb is stored as uint16 holding raw bf16 bits. bf16 -> fp32 is just
// "shift the bits into the upper half of a uint32 and reinterpret as float";
// we do that with vshll_n_u16 + vreinterpretq_f32_u32 and then use a
// single FMA lane against the fp32 activation. No fp32 copy of the
// embedding is needed, which saves ~1.25 GB RSS on bitnet-2b.
//
// emb:    uint16 [V, H]  (bf16 bits; H must be >= 8 and a multiple of 8)
// x:      fp32   [H]
// logits: fp32   [V]     (caller-allocated)
void lm_head_bf16_fp32_neon(
    const uint16_t* __restrict__ emb,
    const float*    __restrict__ x,
    float*          __restrict__ logits,
    int V, int H
) {
#pragma omp parallel for if(V >= 64) schedule(static)
    for (int v = 0; v < V; ++v) {
        const uint16_t* row = emb + static_cast<ptrdiff_t>(v) * H;
        float32x4_t acc0 = vdupq_n_f32(0.0f);
        float32x4_t acc1 = vdupq_n_f32(0.0f);

        int h = 0;
        for (; h + 8 <= H; h += 8) {
            const uint16x8_t b8 = vld1q_u16(row + h);
            // bf16 -> fp32: place the 16 bf16 bits into the upper half of
            // each uint32 lane, reinterpret as float32. Exact upcast.
            const uint32x4_t lo_u = vshll_n_u16(vget_low_u16(b8),  16);
            const uint32x4_t hi_u = vshll_n_u16(vget_high_u16(b8), 16);
            const float32x4_t w_lo = vreinterpretq_f32_u32(lo_u);
            const float32x4_t w_hi = vreinterpretq_f32_u32(hi_u);
            const float32x4_t x_lo = vld1q_f32(x + h);
            const float32x4_t x_hi = vld1q_f32(x + h + 4);
            acc0 = vfmaq_f32(acc0, w_lo, x_lo);
            acc1 = vfmaq_f32(acc1, w_hi, x_hi);
        }
        float tail = 0.0f;
        for (; h < H; ++h) {
            uint32_t bits = static_cast<uint32_t>(row[h]) << 16;
            float w;
            std::memcpy(&w, &bits, sizeof(float));
            tail += w * x[h];
        }
        logits[v] = tail + vaddvq_f32(acc0) + vaddvq_f32(acc1);
    }
}

// LM head matmul with per-row int8 quantized embeddings.
//     logits[v] = emb_scale[v] * sum_h emb_int8[v, h] * x[h]
//
// Embedding is stored at ~half the size of the bf16 variant (1 byte/weight
// vs 2) with a per-row fp32 scale. Saves ~313 MB on BitNet-2B's
// [128256, 2560] table. Used by the torchless runtime as the low-memory
// LM head path; the bf16 kernel above stays available for callers that
// want bit-identical logits with the torch-backed reference.
//
// emb:        int8 [V, H]   (per-row absmax quant of the bf16 embedding)
// emb_scale:  fp32 [V]      (dequant factor per row)
// x:          fp32 [H]      (final-norm output)
// logits:     fp32 [V]      (caller-allocated)
void lm_head_int8_fp32_neon(
    const int8_t* __restrict__ emb,
    const float*  __restrict__ emb_scale,
    const float*  __restrict__ x,
    float*        __restrict__ logits,
    int V, int H
) {
#pragma omp parallel for if(V >= 64) schedule(static)
    for (int v = 0; v < V; ++v) {
        const int8_t* row = emb + static_cast<ptrdiff_t>(v) * H;
        float32x4_t acc0 = vdupq_n_f32(0.0f);
        float32x4_t acc1 = vdupq_n_f32(0.0f);
        float32x4_t acc2 = vdupq_n_f32(0.0f);
        float32x4_t acc3 = vdupq_n_f32(0.0f);

        int h = 0;
        for (; h + 16 <= H; h += 16) {
            const int8x16_t b = vld1q_s8(row + h);
            // Widen int8x16 -> int16x8 -> int32x4 (four lanes) -> float32x4.
            const int16x8_t lo16 = vmovl_s8(vget_low_s8(b));
            const int16x8_t hi16 = vmovl_s8(vget_high_s8(b));
            const float32x4_t f0 = vcvtq_f32_s32(vmovl_s16(vget_low_s16(lo16)));
            const float32x4_t f1 = vcvtq_f32_s32(vmovl_s16(vget_high_s16(lo16)));
            const float32x4_t f2 = vcvtq_f32_s32(vmovl_s16(vget_low_s16(hi16)));
            const float32x4_t f3 = vcvtq_f32_s32(vmovl_s16(vget_high_s16(hi16)));
            acc0 = vfmaq_f32(acc0, f0, vld1q_f32(x + h + 0));
            acc1 = vfmaq_f32(acc1, f1, vld1q_f32(x + h + 4));
            acc2 = vfmaq_f32(acc2, f2, vld1q_f32(x + h + 8));
            acc3 = vfmaq_f32(acc3, f3, vld1q_f32(x + h + 12));
        }
        float tail = 0.0f;
        for (; h < H; ++h) {
            tail += static_cast<float>(row[h]) * x[h];
        }
        const float sum = tail
            + vaddvq_f32(acc0) + vaddvq_f32(acc1)
            + vaddvq_f32(acc2) + vaddvq_f32(acc3);
        logits[v] = emb_scale[v] * sum;
    }
}

// LM head matmul with per-row int4 quantized embeddings (2 nibbles/byte).
//     logits[v] = emb_scale[v] * sum_h nibble_decode(emb[v, h]) * x[h]
//
// Each packed byte holds two signed 4-bit values in the range [-7, +7]
// (absmax quant, per row). Layout: low nibble = position 2i, high nibble
// = position 2i+1. Sign-extension is done by shifting the nibble up to
// the top of a byte and then arithmetic-shifting back down.
//
// emb:        uint8 [V, H/2]  (2 int4 per byte; H must be even)
// emb_scale:  fp32 [V]
// x:          fp32 [H]
// logits:     fp32 [V]
void lm_head_int4_fp32_neon(
    const uint8_t* __restrict__ emb,
    const float*   __restrict__ emb_scale,
    const float*   __restrict__ x,
    float*         __restrict__ logits,
    int V, int H
) {
#pragma omp parallel for if(V >= 64) schedule(static)
    for (int v = 0; v < V; ++v) {
        const uint8_t* row = emb + static_cast<ptrdiff_t>(v) * (H / 2);
        float32x4_t acc0 = vdupq_n_f32(0.0f);
        float32x4_t acc1 = vdupq_n_f32(0.0f);
        float32x4_t acc2 = vdupq_n_f32(0.0f);
        float32x4_t acc3 = vdupq_n_f32(0.0f);

        int h = 0;
        // Inner loop: 8 packed bytes -> 16 int4 -> 16 fp32 -> 16 FMAs.
        for (; h + 16 <= H; h += 16) {
            const int8x8_t p8 = vld1_s8(reinterpret_cast<const int8_t*>(row + (h >> 1)));
            // Sign-extend low nibble: (p << 4) >> 4 arithmetic.
            const int8x8_t lo = vshr_n_s8(vshl_n_s8(p8, 4), 4);
            // Sign-extend high nibble: p >> 4 arithmetic.
            const int8x8_t hi = vshr_n_s8(p8, 4);
            // Interleave lo/hi to natural order [lo0,hi0,lo1,hi1,...].
            const int8x8x2_t zipped = vzip_s8(lo, hi);
            const int8x16_t w = vcombine_s8(zipped.val[0], zipped.val[1]);

            const int16x8_t lo16 = vmovl_s8(vget_low_s8(w));
            const int16x8_t hi16 = vmovl_s8(vget_high_s8(w));
            const float32x4_t f0 = vcvtq_f32_s32(vmovl_s16(vget_low_s16(lo16)));
            const float32x4_t f1 = vcvtq_f32_s32(vmovl_s16(vget_high_s16(lo16)));
            const float32x4_t f2 = vcvtq_f32_s32(vmovl_s16(vget_low_s16(hi16)));
            const float32x4_t f3 = vcvtq_f32_s32(vmovl_s16(vget_high_s16(hi16)));
            acc0 = vfmaq_f32(acc0, f0, vld1q_f32(x + h + 0));
            acc1 = vfmaq_f32(acc1, f1, vld1q_f32(x + h + 4));
            acc2 = vfmaq_f32(acc2, f2, vld1q_f32(x + h + 8));
            acc3 = vfmaq_f32(acc3, f3, vld1q_f32(x + h + 12));
        }
        float tail = 0.0f;
        for (; h < H; h += 2) {
            const uint8_t b = row[h >> 1];
            const int8_t lo = static_cast<int8_t>(static_cast<int8_t>(b << 4) >> 4);
            const int8_t hi = static_cast<int8_t>(b) >> 4;
            tail += static_cast<float>(lo) * x[h];
            if (h + 1 < H) tail += static_cast<float>(hi) * x[h + 1];
        }
        const float sum = tail
            + vaddvq_f32(acc0) + vaddvq_f32(acc1)
            + vaddvq_f32(acc2) + vaddvq_f32(acc3);
        logits[v] = emb_scale[v] * sum;
    }
}

// RMSNorm: out = x * (1 / sqrt(mean(x^2) + eps)) * gamma
// gamma stored as bf16 (uint16 holding bf16 bits), x and out are fp32.
// All three arrays are [K]. Two-pass: scan sum-of-squares, then apply.
//
// Replaces a numpy rmsnorm path that created ~3 full-size temporaries
// per call. Called 4x per layer x 30 layers per forward = 120 times
// per token; eliminating the numpy churn removes hundreds of small
// allocations per forward and meaningfully shrinks the allocator pool
// high-water-mark.
void rmsnorm_bf16gamma_fp32_neon(
    const float*    __restrict__ x,
    const uint16_t* __restrict__ gamma,
    float*          __restrict__ out,
    int K,
    float eps
) {
    // Pass 1: sum of x^2.
    float32x4_t s0 = vdupq_n_f32(0.0f);
    float32x4_t s1 = vdupq_n_f32(0.0f);
    int k = 0;
    for (; k + 8 <= K; k += 8) {
        const float32x4_t a = vld1q_f32(x + k + 0);
        const float32x4_t b = vld1q_f32(x + k + 4);
        s0 = vfmaq_f32(s0, a, a);
        s1 = vfmaq_f32(s1, b, b);
    }
    float sum_sq = vaddvq_f32(s0) + vaddvq_f32(s1);
    for (; k < K; ++k) sum_sq += x[k] * x[k];
    const float mean_sq = sum_sq / static_cast<float>(K);
    const float rrms    = 1.0f / std::sqrt(mean_sq + eps);

    // Pass 2: out = x * rrms * bf16_to_fp32(gamma).
    const float32x4_t rrms_v = vdupq_n_f32(rrms);
    k = 0;
    for (; k + 8 <= K; k += 8) {
        const float32x4_t a = vld1q_f32(x + k + 0);
        const float32x4_t b = vld1q_f32(x + k + 4);
        const uint16x8_t g8 = vld1q_u16(gamma + k);
        const uint32x4_t glo = vshll_n_u16(vget_low_u16(g8),  16);
        const uint32x4_t ghi = vshll_n_u16(vget_high_u16(g8), 16);
        const float32x4_t g_a = vreinterpretq_f32_u32(glo);
        const float32x4_t g_b = vreinterpretq_f32_u32(ghi);
        vst1q_f32(out + k + 0, vmulq_f32(vmulq_f32(a, rrms_v), g_a));
        vst1q_f32(out + k + 4, vmulq_f32(vmulq_f32(b, rrms_v), g_b));
    }
    for (; k < K; ++k) {
        uint32_t gbits = static_cast<uint32_t>(gamma[k]) << 16;
        float g;
        std::memcpy(&g, &gbits, sizeof(float));
        out[k] = x[k] * rrms * g;
    }
}

// Same as above but gamma is already fp32 (for already-upcast norms).
void rmsnorm_fp32gamma_fp32_neon(
    const float* __restrict__ x,
    const float* __restrict__ gamma,
    float*       __restrict__ out,
    int K,
    float eps
) {
    float32x4_t s0 = vdupq_n_f32(0.0f);
    float32x4_t s1 = vdupq_n_f32(0.0f);
    int k = 0;
    for (; k + 8 <= K; k += 8) {
        const float32x4_t a = vld1q_f32(x + k + 0);
        const float32x4_t b = vld1q_f32(x + k + 4);
        s0 = vfmaq_f32(s0, a, a);
        s1 = vfmaq_f32(s1, b, b);
    }
    float sum_sq = vaddvq_f32(s0) + vaddvq_f32(s1);
    for (; k < K; ++k) sum_sq += x[k] * x[k];
    const float rrms = 1.0f / std::sqrt(sum_sq / static_cast<float>(K) + eps);
    const float32x4_t rrms_v = vdupq_n_f32(rrms);
    k = 0;
    for (; k + 8 <= K; k += 8) {
        vst1q_f32(out + k + 0,
            vmulq_f32(vmulq_f32(vld1q_f32(x + k + 0), rrms_v), vld1q_f32(gamma + k + 0)));
        vst1q_f32(out + k + 4,
            vmulq_f32(vmulq_f32(vld1q_f32(x + k + 4), rrms_v), vld1q_f32(gamma + k + 4)));
    }
    for (; k < K; ++k) out[k] = x[k] * rrms * gamma[k];
}

// Fused BitNet MLP gate path: out = max(gate, 0)^2 * up.
// Replaces `inter = relu2(gate) * up` in the numpy path (two full-size
// temporaries per call, called once per layer per forward).
void relu2_mul_fp32_neon(
    const float* __restrict__ gate,
    const float* __restrict__ up,
    float*       __restrict__ out,
    int K
) {
    const float32x4_t zero = vdupq_n_f32(0.0f);
    int k = 0;
    for (; k + 4 <= K; k += 4) {
        const float32x4_t g = vmaxq_f32(vld1q_f32(gate + k), zero);
        vst1q_f32(out + k, vmulq_f32(vmulq_f32(g, g), vld1q_f32(up + k)));
    }
    for (; k < K; ++k) {
        float g = gate[k];
        if (g < 0.0f) g = 0.0f;
        out[k] = g * g * up[k];
    }
}

// In-place add: a += b. Fp32 vectors.
void add_inplace_fp32_neon(
    float*       __restrict__ a,
    const float* __restrict__ b,
    int K
) {
    int k = 0;
    for (; k + 8 <= K; k += 8) {
        vst1q_f32(a + k + 0, vaddq_f32(vld1q_f32(a + k + 0), vld1q_f32(b + k + 0)));
        vst1q_f32(a + k + 4, vaddq_f32(vld1q_f32(a + k + 4), vld1q_f32(b + k + 4)));
    }
    for (; k < K; ++k) a[k] += b[k];
}

// Informational: returns 1 if the binary was compiled with OpenMP, 0 otherwise.
int matmul_lut_neon_has_omp(void) {
#if defined(_OPENMP)
    return 1;
#else
    return 0;
#endif
}

// Informational: active OpenMP threads (0 if not built with OMP).
int matmul_lut_neon_max_threads(void) {
#if defined(_OPENMP)
    return omp_get_max_threads();
#else
    return 0;
#endif
}

int matmul_lut_neon_has_accelerate(void) {
#if defined(__APPLE__)
    return 1;
#else
    return 0;
#endif
}

}  // extern "C"
