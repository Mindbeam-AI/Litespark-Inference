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
    lm_head_bf16 as _lm_head_bf16,
    lm_head_int4 as _lm_head_int4,
    lm_head_int8 as _lm_head_int8,
    matmul_packed_m1,
    matmul_packed_prefill as _matmul_prefill,
    quantize_activation as _quant_neon,
    relu2_mul_into as _relu2_mul,
    rmsnorm_into as _rmsnorm_neon,
)
from .model import PackedBitNetModel, PackedProjection
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
    """Pre-quantized matmul call into a caller-owned output buffer."""
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

    for li, layer in enumerate(model.layers):
        # ---- Attention ----
        _rmsnorm_neon(x, layer.input_norm, h_fp, eps=c.rms_norm_eps)
        h_scale = _quant_neon(h_fp, h_i8)
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

        _rmsnorm_neon(attn_flat, layer.attn_sub_norm, h_fp, eps=c.rms_norm_eps)
        attn_scale = _quant_neon(h_fp, h_i8)
        _call_matmul(h_i8, attn_scale, layer.o_proj, state.sc_hidden_a)
        _add_inplace(x, state.sc_hidden_a)           # x += o_out

        # ---- MLP ----
        _rmsnorm_neon(x, layer.post_attn_norm, h_fp, eps=c.rms_norm_eps)
        h_scale = _quant_neon(h_fp, h_i8)
        _call_matmul(h_i8, h_scale, layer.gate_proj, state.sc_inter_a)
        _call_matmul(h_i8, h_scale, layer.up_proj,   state.sc_inter_b)
        # inter = relu2(gate) * up   -> overwrite sc_inter_a in-place.
        _relu2_mul(state.sc_inter_a, state.sc_inter_b, state.sc_inter_a)
        _rmsnorm_neon(state.sc_inter_a, layer.ffn_sub_norm, i_fp, eps=c.rms_norm_eps)
        inter_scale = _quant_neon(i_fp, i_i8)
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


def _rmsnorm_quant_rows(
    src: np.ndarray, gamma: np.ndarray, eps: float,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Per-row RMSNorm + int8 quantization for a [T, K] batch, reusing the exact
    NEON kernels the M=1 path uses (so prefill numerics match generation).

    Returns (normed [T, K] fp32, q [T, K] int8, scales [T] fp32).
    """
    T, K = src.shape
    normed = np.empty((T, K), dtype=np.float32)
    q = np.empty((T, K), dtype=np.int8)
    scales = np.empty(T, dtype=np.float32)
    for t in range(T):
        _rmsnorm_neon(np.ascontiguousarray(src[t]), gamma, normed[t], eps=eps)
        scales[t] = _quant_neon(normed[t], q[t])
    return normed, q, scales


def forward_prefill(
    model: PackedBitNetModel, state: InferState, token_ids: list[int],
) -> np.ndarray:
    """
    Batched prompt prefill: run all `token_ids` through the stack at once,
    populating the KV cache for positions [state.pos, state.pos + T) and
    returning the vocab-sized logits for the LAST prompt token.

    Equivalent, token for token, to looping forward_one over token_ids -- but
    the seven ternary projections per layer are issued as single batched
    matmuls (matmul_packed_prefill) instead of T separate M=1 calls, so the
    weight matrices are unpacked and streamed once per prompt rather than once
    per token. RMSNorm / quantization / attention stay per-row so the result
    matches the M=1 path numerically.
    """
    c = model.config
    H = c.hidden_size
    D = H // c.num_heads
    Q = c.num_heads
    KV = c.num_kv_heads
    GQA = Q // KV
    eps = c.rms_norm_eps

    T = len(token_ids)
    if T == 0:
        raise ValueError("token_ids must be non-empty")
    start = state.pos
    if start + T > state.t_max:
        raise RuntimeError(
            f"inference state exhausted: pos={start} + T={T} > t_max={state.t_max}"
        )

    # Embedding lookup for the whole prompt -> [T, H] fp32 residual stream.
    X = np.empty((T, H), dtype=np.float32)
    for t, tid in enumerate(token_ids):
        X[t] = _embed_lookup(model, int(tid))

    cos = state.rope_cos[start:start + T][:, None, :]   # [T, 1, D/2]
    sin = state.rope_sin[start:start + T][:, None, :]
    inv_sqrt_d = 1.0 / float(np.sqrt(D))

    # Causal mask over absolute positions: token t (abs start+t) sees p <= start+t.
    P = start + T
    abs_pos = start + np.arange(T)
    disallowed = np.arange(P)[None, :] > abs_pos[:, None]   # [T, P] True == masked

    for li, layer in enumerate(model.layers):
        # ---- Attention ----
        _, h_i8, h_scale = _rmsnorm_quant_rows(X, layer.input_norm, eps)
        q = _matmul_prefill(h_i8, layer.q_proj.w_packed, layer.q_proj.scale, h_scale)
        k = _matmul_prefill(h_i8, layer.k_proj.w_packed, layer.k_proj.scale, h_scale)
        v = _matmul_prefill(h_i8, layer.v_proj.w_packed, layer.v_proj.scale, h_scale)

        q = apply_rope(q.reshape(T, Q, D), cos, sin)
        k = apply_rope(k.reshape(T, KV, D), cos, sin)
        v = v.reshape(T, KV, D)

        state.cache_k[li, start:start + T] = k
        state.cache_v[li, start:start + T] = v
        k_hist = state.cache_k[li, :P]   # [P, KV, D]
        v_hist = state.cache_v[li, :P]

        # GQA causal attention for all T query positions at once.
        q_grouped = q.reshape(T, KV, GQA, D)
        scores = np.einsum('tkgd,pkd->tkgp', q_grouped, k_hist) * inv_sqrt_d
        scores = np.where(disallowed[:, None, None, :], -np.inf, scores)
        weights = softmax(scores, axis=-1)
        attn = np.einsum('tkgp,pkd->tkgd', weights, v_hist)
        attn_flat = np.ascontiguousarray(attn.reshape(T, Q * D))

        _, a_i8, a_scale = _rmsnorm_quant_rows(attn_flat, layer.attn_sub_norm, eps)
        o = _matmul_prefill(a_i8, layer.o_proj.w_packed, layer.o_proj.scale, a_scale)
        X += o

        # ---- MLP ----
        _, p_i8, p_scale = _rmsnorm_quant_rows(X, layer.post_attn_norm, eps)
        gate = _matmul_prefill(p_i8, layer.gate_proj.w_packed, layer.gate_proj.scale, p_scale)
        up = _matmul_prefill(p_i8, layer.up_proj.w_packed, layer.up_proj.scale, p_scale)
        inter = np.empty_like(gate)
        for t in range(T):
            _relu2_mul(gate[t], up[t], inter[t])
        _, i_i8, i_scale = _rmsnorm_quant_rows(inter, layer.ffn_sub_norm, eps)
        down = _matmul_prefill(i_i8, layer.down_proj.w_packed, layer.down_proj.scale, i_scale)
        X += down

    state.pos = start + T

    # Only the last prompt token's logits are needed to start decoding.
    h_fp = state.sc_hidden_norm
    _rmsnorm_neon(np.ascontiguousarray(X[T - 1]), model.final_norm, h_fp, eps=eps)
    return _lm_head(model, h_fp, state.sc_logits)


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

    # Prefill. The batched path issues one matmul per projection for the whole
    # prompt; a single-token prompt has nothing to batch, so use the M=1 path.
    if not prompt_token_ids:
        raise ValueError("prompt_token_ids must be non-empty")
    if len(prompt_token_ids) == 1:
        logits = forward_one(model, state, int(prompt_token_ids[0]))
    else:
        logits = forward_prefill(model, state, [int(t) for t in prompt_token_ids])

    # Greedy decode
    generated: list[int] = []
    for _ in range(max_new_tokens):
        next_id = int(np.argmax(logits))
        generated.append(next_id)
        logits = forward_one(model, state, next_id)
    return generated
