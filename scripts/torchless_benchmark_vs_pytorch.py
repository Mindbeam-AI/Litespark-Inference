"""
Torchless Litespark vs PyTorch baseline, pp128 + tg128 methodology.

Mirrors the workload of benchmark_kernel.py --inference --pytorch so numbers
are comparable to what that script prints, but runs the torchless Litespark
path (numpy + extern "C" NEON kernels, no torch) on whichever `embed_dtype`
you ask for ("bf16" | "int8" | "int4").

Because torch + transformers pulls in a large runtime and is slow to import,
we run each backend in its own subprocess and aggregate the numbers at the
end. This also makes peak RSS and allocator state honest per-backend.

Usage:
    python scripts/torchless_benchmark_vs_pytorch.py
    python scripts/torchless_benchmark_vs_pytorch.py --embed-dtype int4
    python scripts/torchless_benchmark_vs_pytorch.py --skip-torch
"""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import time
from pathlib import Path


PROMPT = "The quick brown fox jumps over the lazy dog. " * 20
PP_RUNS = 3
TG_TOKENS = 128
TG_RUNS = 1


# Python payload run inside the PyTorch-baseline subprocess.
_TORCH_PAYLOAD = r'''
import json, time, os, gc, resource
os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

PROMPT = %(prompt_repr)s
PP_RUNS = %(pp_runs)d
TG_TOKENS = %(tg_tokens)d
TG_RUNS = %(tg_runs)d

torch.set_num_threads(1)
def maxrss_mb():
    return resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / (1024 * 1024)

HF_NAME = "microsoft/bitnet-b1.58-2B-4T-bf16"

rss_before = maxrss_mb()
# Mirror the fallback pattern from benchmark_kernel.py:run_pytorch_baseline.
# trust_remote_code=True tries to fetch a configuration_bitnet.py that the
# HF cache here does not have; LlamaForCausalLM can load the same weights
# fine, which is what benchmark_kernel.py falls back to.
try:
    model = AutoModelForCausalLM.from_pretrained(HF_NAME, trust_remote_code=True)
except (ValueError, OSError, EnvironmentError):
    from transformers import LlamaForCausalLM
    model = LlamaForCausalLM.from_pretrained(HF_NAME)
model.eval()
try:
    tok = AutoTokenizer.from_pretrained(HF_NAME, trust_remote_code=True)
except (ValueError, OSError, EnvironmentError):
    tok = AutoTokenizer.from_pretrained(HF_NAME)
if tok.pad_token is None:
    tok.pad_token = tok.eos_token
rss_after_load = maxrss_mb()

input_ids = tok.encode(PROMPT, return_tensors="pt", max_length=128, truncation=True)
attn = torch.ones_like(input_ids)
prompt_tokens = input_ids.shape[1]

# pp (prompt processing) - cold run, no warmup (matches benchmark_kernel.py)
pp_times = []
with torch.no_grad():
    for _ in range(PP_RUNS):
        t0 = time.perf_counter()
        _ = model(input_ids, attention_mask=attn)
        pp_times.append((time.perf_counter() - t0) * 1000)

# tg (token generation)
tg_times = []
with torch.no_grad():
    for _ in range(TG_RUNS):
        t0 = time.perf_counter()
        _ = model.generate(
            input_ids, attention_mask=attn,
            max_new_tokens=TG_TOKENS, do_sample=False,
            pad_token_id=tok.eos_token_id,
        )
        tg_times.append(time.perf_counter() - t0)

params_bytes = sum(p.nbytes for p in model.parameters())
print(json.dumps({
    "backend": "pytorch-bf16",
    "prompt_tokens": int(prompt_tokens),
    "pp_times_ms": pp_times,
    "tg_seconds": tg_times,
    "tg_tokens": TG_TOKENS,
    "rss_before_mb": rss_before,
    "rss_after_load_mb": rss_after_load,
    "rss_peak_mb": maxrss_mb(),
    "params_mb": params_bytes / (1024 * 1024),
}))
'''


# Python payload for the torchless subprocess.
_TORCHLESS_PAYLOAD = r'''
import json, time, os, gc, resource
from litespark_inference.torchless import load_bitnet_2b, load_tokenizer
from litespark_inference.torchless.runtime import forward_one, generate, init_state

PROMPT = %(prompt_repr)s
PP_RUNS = %(pp_runs)d
TG_TOKENS = %(tg_tokens)d
TG_RUNS = %(tg_runs)d
EMBED_DTYPE = %(embed_dtype_repr)s

def maxrss_mb():
    return resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / (1024 * 1024)

rss_before = maxrss_mb()
model = load_bitnet_2b(embed_dtype=EMBED_DTYPE)
tok = load_tokenizer()
rss_after_load = maxrss_mb()

ids = tok.encode(PROMPT)[:128]
prompt_tokens = len(ids)

# Warmup once so the first prefill doesnt include dylib-load / jit costs
state = init_state(model, t_max=prompt_tokens + 4)
for tid in ids:
    _ = forward_one(model, state, int(tid))

# pp: measure full-prefill wall time (forward_one per prompt token).
pp_times = []
for _ in range(PP_RUNS):
    state = init_state(model, t_max=prompt_tokens + 1)
    t0 = time.perf_counter()
    for tid in ids:
        _ = forward_one(model, state, int(tid))
    pp_times.append((time.perf_counter() - t0) * 1000)

# tg: measure end-to-end generate time for TG_TOKENS.
tg_times = []
for _ in range(TG_RUNS):
    t0 = time.perf_counter()
    _ = generate(model, ids, max_new_tokens=TG_TOKENS)
    tg_times.append(time.perf_counter() - t0)

# Tensor-byte accounting
try:
    tb = model.tensor_bytes()
except Exception:
    tb = {}
print(json.dumps({
    "backend": "torchless-" + EMBED_DTYPE,
    "prompt_tokens": int(prompt_tokens),
    "pp_times_ms": pp_times,
    "tg_seconds": tg_times,
    "tg_tokens": TG_TOKENS,
    "rss_before_mb": rss_before,
    "rss_after_load_mb": rss_after_load,
    "rss_peak_mb": maxrss_mb(),
    "tensor_bytes": tb,
}))
'''


def _run_payload(python_src: str) -> dict:
    """Run a Python snippet in a fresh subprocess, return its JSON stdout."""
    venv_python = sys.executable
    result = subprocess.run(
        [venv_python, "-u", "-c", python_src],
        capture_output=True, text=True,
    )
    if result.returncode != 0:
        sys.stderr.write(f"subprocess failed (rc={result.returncode}):\n{result.stderr}\n")
        raise SystemExit(result.returncode)
    # Some transformers banners may print to stderr; we want the last valid
    # JSON object on stdout.
    for line in reversed(result.stdout.strip().splitlines()):
        line = line.strip()
        if line.startswith("{"):
            return json.loads(line)
    raise RuntimeError(f"no JSON in stdout:\n{result.stdout}")


def _summarize(name: str, r: dict) -> None:
    import statistics as st
    pp_mean = st.mean(r["pp_times_ms"])
    pp_std = st.stdev(r["pp_times_ms"]) if len(r["pp_times_ms"]) > 1 else 0.0
    tg_mean = st.mean(r["tg_seconds"])
    tg_throughput = r["tg_tokens"] / tg_mean
    pp_throughput = r["prompt_tokens"] / (pp_mean / 1000.0)
    print(f"\n=== {name} ===")
    print(f"  prompt tokens : {r['prompt_tokens']}")
    print(f"  Memory")
    print(f"    rss_before_mb    : {r['rss_before_mb']:8.1f} MB")
    print(f"    rss_after_load   : {r['rss_after_load_mb']:8.1f} MB")
    print(f"    rss_peak         : {r['rss_peak_mb']:8.1f} MB")
    if "params_mb" in r:
        print(f"    params (sum)     : {r['params_mb']:8.1f} MB")
    if "tensor_bytes" in r and r["tensor_bytes"]:
        tb = r["tensor_bytes"]
        MB = 1024 * 1024
        print(f"    packed weights   : {tb.get('packed_weights',0)/MB:8.1f} MB")
        print(f"    embedding        : {tb.get('embedding',0)/MB:8.1f} MB")
        print(f"    total (incl emb) : {tb.get('total_incl_embedding',0)/MB:8.1f} MB")
    print(f"  Prompt processing (pp{r['prompt_tokens']})")
    print(f"    time/run  : {pp_mean:8.1f} +- {pp_std:.1f} ms")
    print(f"    throughput: {pp_throughput:8.2f} tok/s")
    print(f"  Token generation (tg{r['tg_tokens']})")
    print(f"    time/run  : {tg_mean*1000:8.1f} ms  ({r['tg_tokens']} tokens)")
    print(f"    throughput: {tg_throughput:8.2f} tok/s")


def _compare(pt: dict, tl: dict) -> None:
    import statistics as st
    pt_pp = st.mean(pt["pp_times_ms"])
    tl_pp = st.mean(tl["pp_times_ms"])
    pt_tg = st.mean(pt["tg_seconds"])
    tl_tg = st.mean(tl["tg_seconds"])
    pt_tg_tps = pt["tg_tokens"] / pt_tg
    tl_tg_tps = tl["tg_tokens"] / tl_tg
    print("\n=== COMPARISON: Torchless Litespark vs PyTorch ===")
    print(f"  {'Metric':<28} {'PyTorch':>12} {'Torchless':>12} {'Speedup':>10}")
    print(f"  {'-'*60}")
    print(f"  {'Peak RSS (MB)':<28} {pt['rss_peak_mb']:>12,.0f} {tl['rss_peak_mb']:>12,.0f} {pt['rss_peak_mb']/tl['rss_peak_mb']:>9.1f}x")
    print(f"  {'TTFT-ish pp time (ms)':<28} {pt_pp:>12,.1f} {tl_pp:>12,.1f} {pt_pp/tl_pp:>9.1f}x")
    print(f"  {'tg throughput (tok/s)':<28} {pt_tg_tps:>12.2f} {tl_tg_tps:>12.2f} {tl_tg_tps/pt_tg_tps:>9.1f}x")
    print(f"  {'-'*60}")


def main() -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--embed-dtype", choices=["bf16", "int8", "int4"], default="int4")
    p.add_argument("--skip-torch", action="store_true",
                   help="Only run the torchless benchmark (skip the slow PyTorch baseline).")
    args = p.parse_args()

    print(f"Running torchless vs pytorch benchmark (embed_dtype={args.embed_dtype})")

    torchless_src = _TORCHLESS_PAYLOAD % {
        "prompt_repr": repr(PROMPT),
        "pp_runs": PP_RUNS,
        "tg_tokens": TG_TOKENS,
        "tg_runs": TG_RUNS,
        "embed_dtype_repr": repr(args.embed_dtype),
    }
    t0 = time.perf_counter()
    tl = _run_payload(torchless_src)
    print(f"  [torchless run took {time.perf_counter() - t0:.1f}s wall]")
    _summarize(f"Torchless Litespark ({args.embed_dtype})", tl)

    if args.skip_torch:
        print("\n(skipped pytorch baseline; pass without --skip-torch for full comparison)")
        return 0

    torch_src = _TORCH_PAYLOAD % {
        "prompt_repr": repr(PROMPT),
        "pp_runs": PP_RUNS,
        "tg_tokens": TG_TOKENS,
        "tg_runs": TG_RUNS,
    }
    t0 = time.perf_counter()
    pt = _run_payload(torch_src)
    print(f"  [pytorch baseline took {time.perf_counter() - t0:.1f}s wall]")
    _summarize("PyTorch baseline (bf16)", pt)

    _compare(pt, tl)
    return 0


if __name__ == "__main__":
    sys.exit(main())
