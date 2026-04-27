"""
Per-phase timing breakdown for the torchless forward pass.

Wraps each section of runtime.forward_one with perf_counter so we can see
where the 100 ms / token actually goes, and decide whether to go after the
LM head, the numpy primitives, or pull the whole forward into C.

Reports median per-forward timings across N_ITERS warm forwards.
"""

from __future__ import annotations

import statistics
import sys
import time
from collections import defaultdict

import numpy as np


def main() -> int:
    from litespark_inference.torchless import load_bitnet_2b
    from litespark_inference.torchless.kernel import matmul_packed_m1
    from litespark_inference.torchless.kernel import quantize_activation as _quant_neon
    from litespark_inference.torchless.ops import (
        apply_rope, bf16_u16_to_fp32, relu2, rmsnorm, rope_tables, softmax,
    )
    from litespark_inference.torchless.runtime import init_state

    print("Loading model...", flush=True)
    t0 = time.perf_counter()
    model = load_bitnet_2b()
    print(f"  {time.perf_counter() - t0:.1f}s", flush=True)

    c = model.config
    H = c.hidden_size
    D = H // c.num_heads
    Q = c.num_heads
    KV = c.num_kv_heads
    GQA = Q // KV
    kv_dim = KV * D

    # Prime the state with a small prefill so the KV cache has real data.
    PROMPT = [1, 100, 200, 300]
    N_ITERS = 20
    state = init_state(model, t_max=len(PROMPT) + N_ITERS + 2)

    # Do the prefill and first generate the regular way, so the perf of the
    # instrumented loop below is measured against a realistic KV cache length.
    from litespark_inference.torchless.runtime import forward_one
    for tid in PROMPT:
        logits = forward_one(model, state, int(tid))

    # Now run N_ITERS instrumented forwards, simulating greedy decoding.
    timings = defaultdict(list)

    for _ in range(N_ITERS):
        next_tok = int(np.argmax(logits))

        t_tok = time.perf_counter()

        # --- embedding lookup ---
        t = time.perf_counter()
        x = bf16_u16_to_fp32(model.embed_tokens[next_tok:next_tok + 1])[0].copy()
        timings["embed"].append(time.perf_counter() - t)

        cos_row = state.rope_cos[state.pos]
        sin_row = state.rope_sin[state.pos]
        inv_sqrt_d = 1.0 / float(np.sqrt(D))

        h_i8 = state.sc_int8_hidden
        i_i8 = state.sc_int8_inter

        # --- per-layer ---
        t_layers = time.perf_counter()
        layer_rms_total = 0.0
        layer_quant_total = 0.0
        layer_matmul_total = 0.0
        layer_rope_total = 0.0
        layer_attn_total = 0.0
        layer_mlp_misc_total = 0.0

        for li, layer in enumerate(model.layers):
            # RMSNorm before attn
            t = time.perf_counter()
            h = rmsnorm(x, layer.input_norm, eps=c.rms_norm_eps)
            layer_rms_total += time.perf_counter() - t

            # Quantize h once for q/k/v
            t = time.perf_counter()
            h_cont = np.ascontiguousarray(h)
            h_scale = _quant_neon(h_cont, h_i8)
            layer_quant_total += time.perf_counter() - t

            # q/k/v matmuls
            t = time.perf_counter()
            q_flat = matmul_packed_m1(h_i8, layer.q_proj.w_packed, layer.q_proj.scale, h_scale, out=state.sc_q)
            k_flat = matmul_packed_m1(h_i8, layer.k_proj.w_packed, layer.k_proj.scale, h_scale, out=state.sc_k)
            v_flat = matmul_packed_m1(h_i8, layer.v_proj.w_packed, layer.v_proj.scale, h_scale, out=state.sc_v)
            layer_matmul_total += time.perf_counter() - t

            # RoPE
            t = time.perf_counter()
            q = apply_rope(q_flat.reshape(Q, D), cos_row, sin_row)
            k = apply_rope(k_flat.reshape(KV, D), cos_row, sin_row)
            layer_rope_total += time.perf_counter() - t

            state.cache_k[li, state.pos] = k
            state.cache_v[li, state.pos] = v_flat.reshape(KV, D)

            # Attention (softmax + einsum-ish)
            t = time.perf_counter()
            k_hist = state.cache_k[li, :state.pos + 1]
            v_hist = state.cache_v[li, :state.pos + 1]
            q_grouped = q.reshape(KV, GQA, D)
            k_perm = k_hist.transpose(1, 0, 2)
            scores = np.matmul(q_grouped, k_perm.transpose(0, 2, 1))
            scores *= inv_sqrt_d
            weights = softmax(scores, axis=-1)
            v_perm = v_hist.transpose(1, 0, 2)
            attn = np.matmul(weights, v_perm).reshape(Q * D)
            layer_attn_total += time.perf_counter() - t

            # attn sub-norm + o_proj
            t = time.perf_counter()
            attn = rmsnorm(attn, layer.attn_sub_norm, eps=c.rms_norm_eps)
            layer_rms_total += time.perf_counter() - t

            t = time.perf_counter()
            a_cont = np.ascontiguousarray(attn)
            a_scale = _quant_neon(a_cont, h_i8)
            layer_quant_total += time.perf_counter() - t

            t = time.perf_counter()
            o_out = matmul_packed_m1(h_i8, layer.o_proj.w_packed, layer.o_proj.scale, a_scale, out=state.sc_hidden_a)
            layer_matmul_total += time.perf_counter() - t

            x = x + o_out

            # MLP path
            t = time.perf_counter()
            h = rmsnorm(x, layer.post_attn_norm, eps=c.rms_norm_eps)
            layer_rms_total += time.perf_counter() - t

            t = time.perf_counter()
            h_cont = np.ascontiguousarray(h)
            h_scale = _quant_neon(h_cont, h_i8)
            layer_quant_total += time.perf_counter() - t

            t = time.perf_counter()
            gate = matmul_packed_m1(h_i8, layer.gate_proj.w_packed, layer.gate_proj.scale, h_scale, out=state.sc_inter_a)
            up = matmul_packed_m1(h_i8, layer.up_proj.w_packed, layer.up_proj.scale, h_scale, out=state.sc_inter_b)
            layer_matmul_total += time.perf_counter() - t

            t = time.perf_counter()
            inter = relu2(gate) * up
            layer_mlp_misc_total += time.perf_counter() - t

            t = time.perf_counter()
            inter = rmsnorm(inter, layer.ffn_sub_norm, eps=c.rms_norm_eps)
            layer_rms_total += time.perf_counter() - t

            t = time.perf_counter()
            i_cont = np.ascontiguousarray(inter)
            i_scale = _quant_neon(i_cont, i_i8)
            layer_quant_total += time.perf_counter() - t

            t = time.perf_counter()
            down_out = matmul_packed_m1(i_i8, layer.down_proj.w_packed, layer.down_proj.scale, i_scale, out=state.sc_hidden_b)
            layer_matmul_total += time.perf_counter() - t

            x = x + down_out

        timings["30x_rmsnorm_sum"].append(layer_rms_total)
        timings["30x_quant_sum"].append(layer_quant_total)
        timings["30x_matmul_sum"].append(layer_matmul_total)
        timings["30x_rope_sum"].append(layer_rope_total)
        timings["30x_attn_sum"].append(layer_attn_total)
        timings["30x_mlp_misc_sum"].append(layer_mlp_misc_total)
        timings["all_layers"].append(time.perf_counter() - t_layers)

        # Final norm + LM head
        t = time.perf_counter()
        x = rmsnorm(x, model.final_norm, eps=c.rms_norm_eps)
        timings["final_norm"].append(time.perf_counter() - t)

        t = time.perf_counter()
        from litespark_inference.torchless.kernel import lm_head_bf16
        x_cont = np.ascontiguousarray(x)
        logits = lm_head_bf16(model.embed_tokens, x_cont, state.sc_logits)
        timings["lm_head"].append(time.perf_counter() - t)

        state.pos += 1
        timings["total"].append(time.perf_counter() - t_tok)

    def fmt(name, xs):
        med = statistics.median(xs) * 1000
        return f"  {name:<26} {med:>7.2f} ms"

    print()
    total_med = statistics.median(timings["total"]) * 1000
    print(f"Per-token breakdown (median of {N_ITERS} forwards), total {total_med:.2f} ms "
          f"({1000/total_med:.2f} tok/s):")
    print(fmt("embed lookup", timings["embed"]))
    print(fmt("30x rmsnorm (x4/layer)", timings["30x_rmsnorm_sum"]))
    print(fmt("30x quant", timings["30x_quant_sum"]))
    print(fmt("30x matmul (7/layer)", timings["30x_matmul_sum"]))
    print(fmt("30x rope", timings["30x_rope_sum"]))
    print(fmt("30x attention", timings["30x_attn_sum"]))
    print(fmt("30x mlp misc (relu2*up)", timings["30x_mlp_misc_sum"]))
    print(fmt("  (sum of 30x parts)", [
        sum(vals) for vals in zip(
            timings["30x_rmsnorm_sum"],
            timings["30x_quant_sum"],
            timings["30x_matmul_sum"],
            timings["30x_rope_sum"],
            timings["30x_attn_sum"],
            timings["30x_mlp_misc_sum"],
        )
    ]))
    print(fmt("all_layers total", timings["all_layers"]))
    print(fmt("final_norm", timings["final_norm"]))
    print(fmt("lm_head", timings["lm_head"]))
    print(fmt("-> forward total", timings["total"]))
    return 0


if __name__ == "__main__":
    sys.exit(main())
