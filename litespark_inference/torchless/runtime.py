"""
Torchless BitNet forward pass + greedy generation.

One-token-at-a-time (M=1) path: processes tokens sequentially, calling the
extern "C" NEON packed ternary matmul for every projection. Prompt prefill
happens by looping forward_one over each prompt token; prefill is therefore
O(T) per-token cost rather than the batched O(T) matmul a proper prefill
kernel would give, but it's correct and keeps this commit focused.

BitNet-2B architecture (from microsoft/bitnet-b1.58-2B-4T-bf16):
  - hidden_size = 2560, num_layers = 30
  - num_heads = 20, num_kv_heads = 5 (GQA ratio 4:1)
  - intermediate_size = 6912
  - RMSNorm on input + post-attention, plus BitNet sub-norms after attn
    output and after the gated MLP, before the projection.
  - LM head tied to input embedding.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Optional

import numpy as np

from .kernel import (
    add_inplace as _add_inplace,
    has_amx as _has_amx,
    has_batched_helpers as _has_batched_helpers,
    has_batched_matmul as _has_batched_matmul,
    has_fused_rmsnorm_quantize as _has_fused_rmsq,
    lm_head_bf16 as _lm_head_bf16,
    lm_head_int4 as _lm_head_int4,
    lm_head_int8 as _lm_head_int8,
    matmul_amx_mT as _matmul_amx_mT,
    matmul_amx_mT_padded as _matmul_amx_mT_padded,
    matmul_packed_m1,
    matmul_packed_mT as _matmul_packed_mT,
    quantize_activation as _quant_neon,
    quantize_activation_batched as _quant_batched,
    relu2_mul_into as _relu2_mul,
    relu2_mul_into_batched as _relu2_mul_batched,
    rmsnorm_into as _rmsnorm_neon,
    rmsnorm_into_batched as _rmsnorm_batched,
    rmsnorm_quantize_into as _rmsq_fused,
    rmsnorm_quantize_batched as _rmsq_fused_batched,
)
from .model import PackedBitNetModel, PackedFalconModel, PackedProjection
from .ops import (
    apply_rope,
    bf16_u16_to_fp32,
    relu2,
    rmsnorm,
    rope_tables,
    softmax,
)


@dataclass
class InferState:
    """KV cache + RoPE tables + reusable scratch for a single inference session."""

    cache_k: np.ndarray   # [num_layers, T_max, num_kv_heads, head_dim] float32
    cache_v: np.ndarray   # [num_layers, T_max, num_kv_heads, head_dim] float32
    rope_cos: np.ndarray  # [T_max, head_dim/2] float32
    rope_sin: np.ndarray  # [T_max, head_dim/2] float32
    t_max: int
    pos: int = 0

    # Reusable scratch buffers so the per-projection hot path stops allocating
    # numpy arrays. Sizes are set by init_state() from the model config.
    sc_int8_hidden: np.ndarray = None     # [hidden_size] int8 (q/k/v/o input)
    sc_int8_inter: np.ndarray = None      # [intermediate_size] int8 (down input)
    sc_q: np.ndarray = None               # [hidden_size] fp32 (q/o/down outputs)
    sc_k: np.ndarray = None               # [num_kv_heads*head_dim] fp32
    sc_v: np.ndarray = None               # [num_kv_heads*head_dim] fp32
    sc_inter_a: np.ndarray = None         # [intermediate_size] fp32 (gate)
    sc_inter_b: np.ndarray = None         # [intermediate_size] fp32 (up)
    sc_hidden_a: np.ndarray = None        # [hidden_size] fp32 (generic)
    sc_hidden_b: np.ndarray = None        # [hidden_size] fp32 (generic)
    sc_hidden_norm: np.ndarray = None     # [hidden_size] fp32 (rmsnorm output)
    sc_inter_norm: np.ndarray = None      # [intermediate_size] fp32 (rmsnorm output)
    sc_residual: np.ndarray = None        # [hidden_size] fp32 (x in the forward loop)
    sc_logits: np.ndarray = None          # [vocab_size] fp32 (LM head output)


def init_state(model: PackedBitNetModel, t_max: int) -> InferState:
    c = model.config
    D = c.hidden_size // c.num_heads
    cache_k = np.zeros((c.num_layers, t_max, c.num_kv_heads, D), dtype=np.float32)
    cache_v = np.zeros((c.num_layers, t_max, c.num_kv_heads, D), dtype=np.float32)
    cos, sin = rope_tables(D, t_max, c.rope_theta)
    kv_dim = c.num_kv_heads * D
    return InferState(
        cache_k=cache_k, cache_v=cache_v,
        rope_cos=cos, rope_sin=sin,
        t_max=t_max, pos=0,
        sc_int8_hidden=np.empty(c.hidden_size, dtype=np.int8),
        sc_int8_inter=np.empty(c.intermediate_size, dtype=np.int8),
        sc_q=np.empty(c.hidden_size, dtype=np.float32),
        sc_k=np.empty(kv_dim, dtype=np.float32),
        sc_v=np.empty(kv_dim, dtype=np.float32),
        sc_inter_a=np.empty(c.intermediate_size, dtype=np.float32),
        sc_inter_b=np.empty(c.intermediate_size, dtype=np.float32),
        sc_hidden_a=np.empty(c.hidden_size, dtype=np.float32),
        sc_hidden_b=np.empty(c.hidden_size, dtype=np.float32),
        sc_hidden_norm=np.empty(c.hidden_size, dtype=np.float32),
        sc_inter_norm=np.empty(c.intermediate_size, dtype=np.float32),
        sc_residual=np.empty(c.hidden_size, dtype=np.float32),
        sc_logits=np.empty(c.vocab_size, dtype=np.float32),
    )


def _call_matmul(
    x_int8: np.ndarray, x_scale: float, proj: PackedProjection,
    out: np.ndarray,
) -> np.ndarray:
    """Pre-quantized matmul. Uses the unpacked-weights path when the loader
    pre-decoded weights (LITESPARK_PREUNPACK=1) and the active kernel
    supports it; otherwise falls back to the packed-weights matmul."""
    if proj.w_unpacked is not None:
        from .kernel import matmul_unpacked_m1
        return matmul_unpacked_m1(x_int8, proj.w_unpacked, proj.scale, x_scale, out=out)
    return matmul_packed_m1(x_int8, proj.w_packed, proj.scale, x_scale, out=out)


def _lm_head(
    model: PackedBitNetModel, x_fp32: np.ndarray, out_logits: np.ndarray,
) -> np.ndarray:
    """
    Tied LM head. Dispatches to the int4/int8 path when the model was
    loaded with a quantized embedding, or the bf16 NEON matmul when the
    embedding is kept in full precision.
    """
    if model.embed_int4 is not None:
        return _lm_head_int4(
            model.embed_int4, model.embed_scale, x_fp32, out_logits,
            H=model.config.hidden_size,
        )
    if model.embed_int8 is not None:
        return _lm_head_int8(
            model.embed_int8, model.embed_scale, x_fp32, out_logits,
        )
    return _lm_head_bf16(model.embed_tokens, x_fp32, out_logits)


def _embed_lookup(model: PackedBitNetModel, token_id: int) -> np.ndarray:
    """Return the fp32 embedding row for `token_id`, regardless of storage."""
    if model.embed_int4 is not None:
        # Unpack the 2-nibble row into int8 {-7..7}, then dequantize.
        H_half = model.embed_int4.shape[1]
        packed = model.embed_int4[token_id].astype(np.int8)
        low  = (packed.astype(np.int8) << 4) >> 4      # sign-extended low nibble
        high = packed.astype(np.int8) >> 4             # sign-extended high nibble
        row = np.empty(H_half * 2, dtype=np.float32)
        row[0::2] = low.astype(np.float32)
        row[1::2] = high.astype(np.float32)
        row *= model.embed_scale[token_id]
        return row
    if model.embed_int8 is not None:
        row = model.embed_int8[token_id].astype(np.float32)
        row *= model.embed_scale[token_id]
        return row
    return bf16_u16_to_fp32(model.embed_tokens[token_id:token_id + 1])[0].copy()


def forward_one(model: PackedBitNetModel, state: InferState, token_id: int) -> np.ndarray:
    """
    Run the forward pass for a single token at position state.pos, update the
    KV cache, advance state.pos, and return the vocab-sized logits.
    """
    c = model.config
    H = c.hidden_size
    D = H // c.num_heads
    Q = c.num_heads
    KV = c.num_kv_heads
    GQA = Q // KV

    if state.pos >= state.t_max:
        raise RuntimeError(
            f"inference state exhausted: pos={state.pos} >= t_max={state.t_max}"
        )

    # Embedding lookup -> residual buffer (avoid allocating a new fp32 each forward)
    x = state.sc_residual                            # [H] fp32, preallocated
    np.copyto(x, _embed_lookup(model, token_id))

    cos_row = state.rope_cos[state.pos]
    sin_row = state.rope_sin[state.pos]
    inv_sqrt_d = 1.0 / float(np.sqrt(D))

    h_i8 = state.sc_int8_hidden    # int8 scratch for q/k/v/o inputs
    i_i8 = state.sc_int8_inter     # int8 scratch for down_proj input
    h_fp = state.sc_hidden_norm    # fp32 scratch for hidden-sized rmsnorm outputs
    i_fp = state.sc_inter_norm     # fp32 scratch for inter-sized rmsnorm outputs

    use_fused = _has_fused_rmsq()

    def _rmsq1(src, gamma, fp_scratch, i8_out):
        """M=1 RMSNorm + int8 quant. Fused single-pass kernel when bf16."""
        if use_fused and gamma.dtype == np.uint16:
            return _rmsq_fused(src, gamma, i8_out, fp_scratch, eps=c.rms_norm_eps)
        _rmsnorm_neon(src, gamma, fp_scratch, eps=c.rms_norm_eps)
        return _quant_neon(fp_scratch, i8_out)

    for li, layer in enumerate(model.layers):
        # ---- Attention ----
        h_scale = _rmsq1(x, layer.input_norm, h_fp, h_i8)
        q_flat = _call_matmul(h_i8, h_scale, layer.q_proj, state.sc_q)
        k_flat = _call_matmul(h_i8, h_scale, layer.k_proj, state.sc_k)
        v_flat = _call_matmul(h_i8, h_scale, layer.v_proj, state.sc_v)

        q = apply_rope(q_flat.reshape(Q, D), cos_row, sin_row)
        k = apply_rope(k_flat.reshape(KV, D), cos_row, sin_row)

        state.cache_k[li, state.pos] = k
        state.cache_v[li, state.pos] = v_flat.reshape(KV, D)

        k_hist = state.cache_k[li, :state.pos + 1]   # [T, KV, D]
        v_hist = state.cache_v[li, :state.pos + 1]

        # GQA scores: for each (kv_head, group in [0, GQA)), t in [0, pos]:
        #     score[kv_h, g, t] = dot(q[kv_h, g, :], k_hist[t, kv_h, :])
        q_grouped = q.reshape(KV, GQA, D)            # [KV, GQA, D]
        k_perm = k_hist.transpose(1, 0, 2)           # [KV, T, D]
        scores = np.matmul(q_grouped, k_perm.transpose(0, 2, 1))  # [KV, GQA, T]
        scores *= inv_sqrt_d
        weights = softmax(scores, axis=-1)

        v_perm = v_hist.transpose(1, 0, 2)           # [KV, T, D]
        attn = np.matmul(weights, v_perm)            # [KV, GQA, D]
        attn_flat = np.ascontiguousarray(attn.reshape(Q * D))

        attn_scale = _rmsq1(attn_flat, layer.attn_sub_norm, h_fp, h_i8)
        _call_matmul(h_i8, attn_scale, layer.o_proj, state.sc_hidden_a)
        _add_inplace(x, state.sc_hidden_a)           # x += o_out

        # ---- MLP ----
        h_scale = _rmsq1(x, layer.post_attn_norm, h_fp, h_i8)
        _call_matmul(h_i8, h_scale, layer.gate_proj, state.sc_inter_a)
        _call_matmul(h_i8, h_scale, layer.up_proj,   state.sc_inter_b)
        # inter = relu2(gate) * up   -> overwrite sc_inter_a in-place.
        _relu2_mul(state.sc_inter_a, state.sc_inter_b, state.sc_inter_a)
        inter_scale = _rmsq1(state.sc_inter_a, layer.ffn_sub_norm, i_fp, i_i8)
        _call_matmul(i_i8, inter_scale, layer.down_proj, state.sc_hidden_b)
        _add_inplace(x, state.sc_hidden_b)           # x += down_out

    _rmsnorm_neon(x, model.final_norm, h_fp, eps=c.rms_norm_eps)
    logits = _lm_head(model, h_fp, state.sc_logits)
    # NOTE: do NOT madvise DONTNEED here per-forward -- re-faulting the
    # 626 MB embedding costs ~10 ms and wipes ~25% of throughput. The
    # embedding stays resident between LM head calls; callers who care
    # about between-session RSS can invoke model.release_embedding_pages()
    # explicitly.
    state.pos += 1
    return logits


def _forward_prefill_hidden(
    model: PackedBitNetModel, state: InferState, token_ids: list[int],
) -> "np.ndarray | None":
    """Internal: run the layer loop for T tokens and return the post-final-
    rmsnorm hidden state [T, H] for ALL positions, or None on the M=1
    fallback path. Caller is responsible for the LM head."""
    T = len(token_ids)
    if T == 0:
        raise ValueError("token_ids must be non-empty")
    if T == 1 or not _has_batched_matmul():
        return None  # caller handles M=1 path

    c = model.config
    H = c.hidden_size
    I_dim = c.intermediate_size
    D = H // c.num_heads
    Q = c.num_heads
    KV = c.num_kv_heads
    GQA = Q // KV
    inv_sqrt_d = 1.0 / float(np.sqrt(D))

    if state.pos + T > state.t_max:
        raise RuntimeError(
            f"prefill would exceed t_max: pos={state.pos} T={T} t_max={state.t_max}"
        )

    # Build [T, H] residual stream from the embedding lookup.
    x = np.empty((T, H), dtype=np.float32)
    for t, tid in enumerate(token_ids):
        x[t] = _embed_lookup(model, int(tid))

    # Per-T scratch (sized once; reused across layers).
    h_fp     = np.empty((T, H),     dtype=np.float32)
    h_i8     = np.empty((T, H),     dtype=np.int8)
    h_scales = np.empty(T,          dtype=np.float32)
    i_fp     = np.empty((T, I_dim), dtype=np.float32)
    i_i8     = np.empty((T, I_dim), dtype=np.int8)
    i_scales = np.empty(T,          dtype=np.float32)
    attn_out = np.empty((T, H),     dtype=np.float32)
    inter    = np.empty((T, I_dim), dtype=np.float32)

    base_pos = state.pos
    use_batched_helpers = _has_batched_helpers()
    use_fused_rmsq = _has_fused_rmsq()
    use_amx = _has_amx() and any(getattr(L.q_proj, "w_amx", None) is not None for L in model.layers)

    def _matmul_proj(x_int8_all, x_scales_all, proj):
        """Dispatch matmul on (T, kernel availability):

          - 8 <= T <= 16 with AMX:        phase 9 AMX (1 tile, no padding)
          - T in {32, 48, 64} with AMX:   phase 10 AMX (multi-tile B reuse)
          - otherwise:                    VNNI M=T

        We only take the phase 10 path on exact 16-multiples because the
        zero-padding alloc + trim costs we measured for non-multiples
        (e.g. T=35 -> pad to 48) eat the kernel speedup. A pre-allocated
        scratch buffer in InferState would fix that; left as future work.
        For T<8 the per-call AMX overhead loses to linear-in-T VNNI; for
        T>64 we'd need >4 C tiles which exceeds the AMX 8-tile budget.
        """
        T_local = x_int8_all.shape[0]
        w_amx = getattr(proj, "w_amx", None)
        if use_amx and w_amx is not None:
            if 8 <= T_local <= 16:
                return _matmul_amx_mT(x_int8_all, x_scales_all, w_amx, proj.scale)
            if T_local in (32, 48, 64):
                return _matmul_amx_mT_padded(x_int8_all, x_scales_all, w_amx, proj.scale)
        return _matmul_packed_mT(x_int8_all, x_scales_all, proj.w_packed, proj.scale)

    def _rmsq(src, gamma, fp_out, i8_out, scales_out):
        """RMSNorm + per-token absmax int8 quantize over the T-batch.

        Prefer the fused single-pass kernel when available AND gamma is
        bf16 (uint16); fall back to the two separate calls otherwise.
        """
        if use_fused_rmsq and gamma.dtype == np.uint16:
            _rmsq_fused_batched(src, gamma, i8_out, scales_out, fp_out, eps=c.rms_norm_eps)
            return
        if use_batched_helpers:
            _rmsnorm_batched(src, gamma, fp_out, eps=c.rms_norm_eps)
            _quant_batched(fp_out, i8_out, scales_out)
        else:
            for t in range(T):
                _rmsnorm_neon(src[t], gamma, fp_out[t], eps=c.rms_norm_eps)
                scales_out[t] = _quant_neon(fp_out[t], i8_out[t])

    for li, layer in enumerate(model.layers):
        # ---- Attention block ----
        _rmsq(x, layer.input_norm, h_fp, h_i8, h_scales)

        q_all = _matmul_proj(h_i8, h_scales, layer.q_proj)
        k_all = _matmul_proj(h_i8, h_scales, layer.k_proj)
        v_all = _matmul_proj(h_i8, h_scales, layer.v_proj)

        # Apply RoPE per token, write KV cache.
        for t in range(T):
            pos = base_pos + t
            cos_row = state.rope_cos[pos]
            sin_row = state.rope_sin[pos]
            q_t = apply_rope(q_all[t].reshape(Q, D), cos_row, sin_row)
            k_t = apply_rope(k_all[t].reshape(KV, D), cos_row, sin_row)
            q_all[t] = q_t.reshape(-1)
            state.cache_k[li, pos] = k_t
            state.cache_v[li, pos] = v_all[t].reshape(KV, D)

        # Causal attention, per token.
        for t in range(T):
            pos = base_pos + t
            q = q_all[t].reshape(Q, D)
            k_hist = state.cache_k[li, :pos + 1]   # [pos+1, KV, D]
            v_hist = state.cache_v[li, :pos + 1]
            q_grouped = q.reshape(KV, GQA, D)
            k_perm = k_hist.transpose(1, 0, 2)
            scores = np.matmul(q_grouped, k_perm.transpose(0, 2, 1))
            scores *= inv_sqrt_d
            weights = softmax(scores, axis=-1)
            v_perm = v_hist.transpose(1, 0, 2)
            attn = np.matmul(weights, v_perm)
            attn_out[t] = np.ascontiguousarray(attn.reshape(Q * D))

        _rmsq(attn_out, layer.attn_sub_norm, h_fp, h_i8, h_scales)
        o_all = _matmul_proj(h_i8, h_scales, layer.o_proj)
        x += o_all                                  # residual: x[T, H] += o[T, H]

        # ---- MLP block ----
        _rmsq(x, layer.post_attn_norm, h_fp, h_i8, h_scales)
        gate_all = _matmul_proj(h_i8, h_scales, layer.gate_proj)
        up_all   = _matmul_proj(h_i8, h_scales, layer.up_proj)

        if use_batched_helpers:
            _relu2_mul_batched(gate_all, up_all, inter)
        else:
            for t in range(T):
                _relu2_mul(gate_all[t], up_all[t], inter[t])
        _rmsq(inter, layer.ffn_sub_norm, i_fp, i_i8, i_scales)
        down_all = _matmul_proj(i_i8, i_scales, layer.down_proj)
        x += down_all                               # residual

    state.pos = base_pos + T

    # Final RMSNorm over all T positions, then return hidden state.
    final_all = np.empty((T, H), dtype=np.float32)
    if use_batched_helpers:
        _rmsnorm_batched(x, model.final_norm, final_all, eps=c.rms_norm_eps)
    else:
        for t in range(T):
            _rmsnorm_neon(x[t], model.final_norm, final_all[t], eps=c.rms_norm_eps)
    return final_all


def forward_prefill(
    model: PackedBitNetModel, state: InferState, token_ids: list[int],
) -> np.ndarray:
    """
    Run a forward pass for T prompt tokens at once and return the final-
    position logits (the only thing decode needs from prefill).

    Falls back to forward_one * T on platforms without the batched
    matmul kernel (e.g. ARM NEON build today).
    """
    final_all = _forward_prefill_hidden(model, state, list(token_ids))
    if final_all is None:
        # M=1 fallback (T=1 or no batched kernel)
        logits = None
        for tid in token_ids:
            logits = forward_one(model, state, int(tid))
        return logits  # type: ignore[return-value]
    return _lm_head(model, np.ascontiguousarray(final_all[-1]), state.sc_logits)


def forward_prefill_all_logits(
    model: PackedBitNetModel, state: InferState, token_ids: list[int],
) -> np.ndarray:
    """Same as forward_prefill but returns logits for ALL T positions.

    Used by speculative-decode validation: we need the model's prediction
    after every speculatively-fed token to decide which ones to accept.

    Returns array of shape [T, vocab_size] fp32.
    """
    T = len(token_ids)
    if T == 0:
        raise ValueError("token_ids must be non-empty")

    final_all = _forward_prefill_hidden(model, state, list(token_ids))
    if final_all is None:
        # M=1 fallback: forward_one each token, collect each logits.
        out = np.empty((T, model.config.vocab_size), dtype=np.float32)
        for t, tid in enumerate(token_ids):
            out[t] = forward_one(model, state, int(tid))
        return out

    # Batched path: T LM heads over the per-position post-final-norm
    # hidden states. Each LM head is independent; iterate in Python --
    # for T <= 16 the LM head cost is small relative to the layer work
    # we just did.
    out = np.empty((T, model.config.vocab_size), dtype=np.float32)
    for t in range(T):
        h = np.ascontiguousarray(final_all[t])
        _lm_head(model, h, out[t])
    return out


def _lookup_speculation(
    context: list[int], match_n: int, k: int,
) -> list[int]:
    """Prompt-lookup speculation (Saxena 2023): scan `context` from end to
    find an n-gram whose tail matches the most recent `match_n` tokens of
    context. Return the next `k` tokens after the longest such match.
    Empty list if no match.

    Tries n in [match_n, match_n-1, ..., 2] in order, returning the
    candidates from the FIRST n that finds a match. Latest match wins
    (search backwards).
    """
    L = len(context)
    if L < 2 or k <= 0:
        return []
    for n in range(min(match_n, L - 1), 1, -1):
        needle = context[L - n:]
        # Search backwards (i is the start index of an n-gram in context).
        # Skip the trivial match at the end (i = L - n).
        for i in range(L - n - 1, -1, -1):
            if context[i:i + n] == needle:
                # Take the next k tokens after the match.
                start = i + n
                end = min(start + k, L)
                cand = context[start:end]
                if cand:
                    return cand
    return []


def generate_speculative(
    model: PackedBitNetModel,
    prompt_token_ids: list[int],
    max_new_tokens: int,
    spec_k: int = 4,
    match_n: int = 4,
    eos_id: Optional[int] = None,
    state: Optional[InferState] = None,
    return_stats: bool = False,
):
    """
    Greedy speculative decoding via prompt-lookup speculation.

    Args:
        model:             loaded PackedBitNetModel
        prompt_token_ids:  prompt tokens
        max_new_tokens:    cap on generated tokens
        spec_k:            number of tokens to speculate per step
        match_n:           n-gram length to match against context
        eos_id:            optional EOS to stop on (skipped if None)
        state:             existing InferState or None to allocate
        return_stats:      if True, also return a stats dict

    Returns the list of generated token IDs, or (tokens, stats) if
    return_stats. Output is bit-identical to greedy decode.
    """
    if not _has_batched_matmul():
        # No batched kernel -> just do plain greedy.
        toks = generate(model, prompt_token_ids, max_new_tokens, state=state)
        if return_stats:
            return toks, {"specdec_supported": False, "steps": len(toks),
                          "tokens_per_step": 1.0, "acceptance_rate": 0.0}
        return toks

    t_max = len(prompt_token_ids) + max_new_tokens + spec_k + 2
    if state is None:
        state = init_state(model, t_max)
    elif state.t_max < t_max:
        raise ValueError(f"state.t_max={state.t_max} < required {t_max}")

    # Prefill -> last position's logits predicts t0.
    logits_next = forward_prefill(model, state, list(prompt_token_ids))

    generated: list[int] = []
    context = list(prompt_token_ids)
    accept_total = 0
    step_count = 0
    matched_steps = 0

    while len(generated) < max_new_tokens:
        # First-token argmax from previously-cached logits.
        next_token = int(np.argmax(logits_next))
        if eos_id is not None and next_token == eos_id:
            break

        # Try speculation
        spec = _lookup_speculation(context + [next_token], match_n, spec_k)

        if not spec:
            # No match -> regular forward_one for next step.
            generated.append(next_token)
            context.append(next_token)
            step_count += 1
            if len(generated) >= max_new_tokens:
                break
            logits_next = forward_one(model, state, next_token)
            continue

        # Validation: feed [next_token, spec[0..K-2]] (K tokens),
        # collect K logits. logits[i] predicts what comes AFTER input[i].
        K_actual = min(spec_k, len(spec))
        # Cap to avoid running off the t_max budget.
        rem = max_new_tokens - len(generated)
        K_actual = min(K_actual, rem)
        if K_actual <= 0:
            break

        val_input = [next_token] + spec[:K_actual - 1]
        start_pos = state.pos
        all_logits = forward_prefill_all_logits(model, state, val_input)
        # state.pos now = start_pos + K_actual

        # Walk the predictions: logits[i] predicts the token after val_input[i].
        #   logits[0]   should predict spec[0]
        #   logits[1]   should predict spec[1]
        #   ...
        #   logits[K-2] should predict spec[K-2]
        #   logits[K-1] is the always-accepted bonus token after spec[K-2].
        accepted = 0
        for i in range(K_actual - 1):
            pred = int(np.argmax(all_logits[i]))
            if pred == spec[i]:
                accepted += 1
            else:
                bonus = pred  # mismatch: bonus is the model's actual prediction
                break
        else:
            # All speculated tokens matched their predictions; bonus is
            # the model's prediction after spec[K-2].
            bonus = int(np.argmax(all_logits[K_actual - 1]))

        # Tokens this step: next_token + spec[0..accepted-1] + bonus
        emit = [next_token] + spec[:accepted] + [bonus]
        # Roll back state.pos. Valid kv positions: start_pos (next_token)
        # ... start_pos + accepted (spec[accepted-1]). Invalid above that.
        # New state.pos should be start_pos + accepted + 1 (next free
        # slot after next_token + accepted spec tokens).
        state.pos = start_pos + accepted + 1

        # Check eos / max-new-tokens cap incrementally.
        for tok in emit:
            if eos_id is not None and tok == eos_id:
                generated.append(tok)
                context.append(tok)
                logits_next = None  # signal halt below
                break
            generated.append(tok)
            context.append(tok)
            if len(generated) >= max_new_tokens:
                logits_next = None
                break
        if logits_next is None:
            break

        # Need logits_next predicting AFTER bonus. Bonus's k/v is not yet
        # in cache (it replaced spec[accepted] in the output). forward_one
        # writes it and returns the prediction.
        logits_next = forward_one(model, state, bonus)

        accept_total += accepted
        matched_steps += 1
        step_count += 1

    if return_stats:
        stats = {
            "specdec_supported": True,
            "steps": step_count,
            "tokens": len(generated),
            "tokens_per_step": (len(generated) / step_count) if step_count else 0.0,
            "acceptance_rate": (accept_total / (matched_steps * (spec_k - 1)))
                               if matched_steps and spec_k > 1 else 0.0,
            "matched_steps": matched_steps,
        }
        return generated, stats
    return generated


def generate(
    model: PackedBitNetModel,
    prompt_token_ids: list[int],
    max_new_tokens: int,
    state: Optional[InferState] = None,
) -> list[int]:
    """
    Greedy (argmax) decoding from a prompt.

    Returns the list of generated token IDs (length == max_new_tokens).
    """
    t_max = len(prompt_token_ids) + max_new_tokens
    if state is None:
        state = init_state(model, t_max)
    elif state.t_max < t_max:
        raise ValueError(
            f"provided state has t_max={state.t_max}, need at least {t_max}"
        )

    # Prefill (batched if available)
    logits = forward_prefill(model, state, list(prompt_token_ids))

    # Greedy decode
    generated: list[int] = []
    for _ in range(max_new_tokens):
        next_id = int(np.argmax(logits))
        generated.append(next_id)
        logits = forward_one(model, state, next_id)
    return generated


def _falcon_lm_head(model: PackedFalconModel, x_fp32: np.ndarray, out_logits: np.ndarray) -> np.ndarray:
    if model.lm_head_int4 is not None:
        return _lm_head_int4(model.lm_head_int4, model.lm_head_scale, x_fp32, out_logits, H=model.config.hidden_size,)
    if model.lm_head_int8 is not None:
        return _lm_head_int8(model.lm_head_int8, model.lm_head_scale, x_fp32, out_logits)
    return _lm_head_bf16(model.lm_head_tokens, x_fp32, out_logits)


def _falcon_embed_lookup(model: PackedFalconModel, token_id: int) -> np.ndarray:
    if model.embed_int4 is not None:
        H_half = model.embed_int4.shape[1]
        packed = model.embed_int4[token_id].astype(np.int8)
        low = (packed.astype(np.int8) << 4) >> 4
        high = packed.astype(np.int8) >> 4
        row = np.empty(H_half * 2, dtype=np.float32)
        row[0::2] = low.astype(np.float32)
        row[1::2] = high.astype(np.float32)
        row *= model.embed_scale[token_id]
        return row
    if model.embed_int8 is not None:
        row = model.embed_int8[token_id].astype(np.float32)
        row *= model.embed_scale[token_id]
        return row
    return bf16_u16_to_fp32(model.embed_tokens[token_id:token_id + 1])[0].copy()


def _silu_mul_into(gate: np.ndarray, up: np.ndarray, out: np.ndarray) -> np.ndarray:
    np.negative(gate, out=out)
    np.exp(out, out=out)
    out += 1.0
    np.divide(gate, out, out=out)
    out *= up
    return out


def falcon_forward_one(model: PackedFalconModel, state: InferState, token_id: int) -> np.ndarray:
    c = model.config
    H = c.hidden_size
    D = H // c.num_heads
    Q = c.num_heads
    KV = c.num_kv_heads
    GQA = Q // KV

    if state.pos >= state.t_max:
        raise RuntimeError(f"inference state exhausted: pos={state.pos} >= t_max={state.t_max}")

    x = state.sc_residual
    np.copyto(x, _falcon_embed_lookup(model, token_id))

    cos_row = state.rope_cos[state.pos]
    sin_row = state.rope_sin[state.pos]
    inv_sqrt_d = 1.0 / float(np.sqrt(D))

    h_i8 = state.sc_int8_hidden
    i_i8 = state.sc_int8_inter
    h_fp = state.sc_hidden_norm
    i_fp = state.sc_inter_norm

    for li, layer in enumerate(model.layers):
        _rmsnorm_neon(x, layer.input_norm, h_fp, eps=c.rms_norm_eps)
        h_scale = _quant_neon(h_fp, h_i8)
        q_flat = _call_matmul(h_i8, h_scale, layer.q_proj, state.sc_q)
        k_flat = _call_matmul(h_i8, h_scale, layer.k_proj, state.sc_k)
        v_flat = _call_matmul(h_i8, h_scale, layer.v_proj, state.sc_v)

        q = apply_rope(q_flat.reshape(Q, D), cos_row, sin_row)
        k = apply_rope(k_flat.reshape(KV, D), cos_row, sin_row)

        state.cache_k[li, state.pos] = k
        state.cache_v[li, state.pos] = v_flat.reshape(KV, D)
        k_hist = state.cache_k[li, :state.pos + 1]
        v_hist = state.cache_v[li, :state.pos + 1]

        q_grouped = q.reshape(KV, GQA, D)
        k_perm = k_hist.transpose(1, 0, 2)
        scores = np.matmul(q_grouped, k_perm.transpose(0, 2, 1))
        scores *= inv_sqrt_d
        weights = softmax(scores, axis=-1)
        v_perm = v_hist.transpose(1, 0, 2)
        attn = np.matmul(weights, v_perm)
        attn_flat = np.ascontiguousarray(attn.reshape(Q * D))

        attn_scale = _quant_neon(attn_flat, h_i8)
        _call_matmul(h_i8, attn_scale, layer.o_proj, state.sc_hidden_a)
        _add_inplace(x, state.sc_hidden_a)

        _rmsnorm_neon(x, layer.post_attn_norm, h_fp, eps=c.rms_norm_eps)
        h_scale = _quant_neon(h_fp, h_i8)
        _call_matmul(h_i8, h_scale, layer.gate_proj, state.sc_inter_a)
        _call_matmul(h_i8, h_scale, layer.up_proj, state.sc_inter_b)
        _silu_mul_into(state.sc_inter_a, state.sc_inter_b, i_fp)
        inter_scale = _quant_neon(i_fp, i_i8)
        _call_matmul(i_i8, inter_scale, layer.down_proj, state.sc_hidden_b)
        _add_inplace(x, state.sc_hidden_b)

    _rmsnorm_neon(x, model.final_norm, h_fp, eps=c.rms_norm_eps)
    logits = _falcon_lm_head(model, h_fp, state.sc_logits)
    state.pos += 1
    return logits


def falcon_forward_prefill(model: PackedFalconModel, state: InferState, token_ids: list[int]) -> np.ndarray:
    logits = None
    for token_id in token_ids:
        logits = falcon_forward_one(model, state, int(token_id))
    return logits


def falcon_generate(model: PackedFalconModel, token_ids: list[int], max_new_tokens: int) -> list[int]:
    state = init_state(model, t_max=len(token_ids) + max_new_tokens + 1)
    logits = falcon_forward_prefill(model, state, token_ids)
    out: list[int] = []
    eos = set(model.config.eos_token_ids)
    for _ in range(max_new_tokens):
        next_id = int(np.argmax(logits))
        if next_id in eos:
            break
        out.append(next_id)
        logits = falcon_forward_one(model, state, next_id)
    return out
