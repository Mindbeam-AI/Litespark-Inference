"""
Torchless BitNet end-to-end benchmark harness.

Runs prefill + generation tok/s on a small fixed prompt set, with and
without speculative decoding, and prints a CSV-friendly table.

Designed to be invoked once per kernel-config (LITESPARK_AMX,
LITESPARK_PREUNPACK, etc.) so a wrapper script can collect the matrix.

Usage:
    python benchmarks/bench_torchless.py
    python benchmarks/bench_torchless.py --runs 5 --max-new-tokens 64
    python benchmarks/bench_torchless.py --speculative --spec-k 8

The output is one CSV row per (config, prompt) plus a summary.
"""

from __future__ import annotations

import argparse
import csv
import os
import platform
import sys
import time

DEFAULT_PROMPTS = [
    # short / generic
    ("short", "The capital of France is"),
    # math-y, repetitive token patterns
    ("math",  "Question: What is 2 + 2? Answer: 2 + 2 equals"),
    # code, low-acceptance for prompt-lookup
    ("code",  "def fibonacci(n):\n    if n < 2:\n        return n\n    return"),
    # structured, repeats prompt phrasing
    ("repeat", "The quick brown fox jumps over the lazy dog. The quick brown fox"),
]


def _rss_mb() -> float:
    try:
        with open(f"/proc/{os.getpid()}/status") as f:
            for line in f:
                if line.startswith("VmRSS:"):
                    return int(line.split()[1]) / 1024.0
    except FileNotFoundError:
        pass
    return -1.0


def main() -> int:
    p = argparse.ArgumentParser(prog="bench_torchless")
    p.add_argument("--runs", type=int, default=3,
                   help="repetitions per (prompt, mode); median is reported")
    p.add_argument("--max-new-tokens", type=int, default=32)
    p.add_argument("--speculative", action="store_true",
                   help="also run a speculative-decoding pass per prompt")
    p.add_argument("--spec-k", type=int, default=4)
    p.add_argument("--match-n", type=int, default=4)
    p.add_argument("--csv", default="-",
                   help="output CSV file path; '-' for stdout")
    p.add_argument("--label", default="",
                   help="optional run label embedded in every CSV row")
    args = p.parse_args()

    print("[bench] loading model + tokenizer...", flush=True, file=sys.stderr)
    from litespark_inference.torchless import BitNet
    bn = BitNet.from_pretrained()

    rss_after_load = _rss_mb()
    paths = bn.kernel_paths()
    cpu = platform.processor() or platform.machine()
    print(f"[bench] cpu={cpu} rss_after_load={rss_after_load:.0f} MB", file=sys.stderr)
    print(f"[bench] kernel paths: {paths}", file=sys.stderr)

    out = sys.stdout if args.csv == "-" else open(args.csv, "w", newline="")
    writer = csv.writer(out)
    writer.writerow([
        "label", "cpu", "instance_size", "rss_mb",
        "prompt_id", "mode", "spec_k", "max_new_tokens",
        "prompt_tokens", "generated_tokens",
        "prefill_tok_per_s", "gen_tok_per_s",
        "spec_acceptance_rate", "spec_tokens_per_step",
    ])

    instance_size = (
        f"{os.cpu_count()}vcpu/"
        f"{platform.system()}/"
        f"{platform.machine()}"
    )
    label = args.label

    summary_rows = []
    for prompt_id, prompt in DEFAULT_PROMPTS:
        for mode in ("greedy",) + (("spec",) if args.speculative else ()):
            speculative = (mode == "spec")
            # Warm one untimed run to amortize per-prompt cache.
            _ = bn.generate(prompt, args.max_new_tokens, speculative=speculative,
                            spec_k=args.spec_k, match_n=args.match_n,
                            return_full=True)
            for run_i in range(args.runs):
                r = bn.generate(prompt, args.max_new_tokens, speculative=speculative,
                                spec_k=args.spec_k, match_n=args.match_n,
                                return_full=True)
                spec_accept = ""
                spec_per_step = ""
                if speculative and r.spec_stats:
                    spec_accept = f"{r.spec_stats.get('acceptance_rate', 0.0):.3f}"
                    spec_per_step = f"{r.spec_stats.get('tokens_per_step', 0.0):.3f}"
                writer.writerow([
                    label, cpu, instance_size, f"{rss_after_load:.0f}",
                    prompt_id, mode, args.spec_k, args.max_new_tokens,
                    r.prompt_tokens, r.generated_tokens,
                    f"{r.prefill_tok_per_s:.2f}", f"{r.gen_tok_per_s:.2f}",
                    spec_accept, spec_per_step,
                ])
                summary_rows.append((prompt_id, mode, r.gen_tok_per_s,
                                     r.prefill_tok_per_s))
            out.flush()
    if args.csv != "-":
        out.close()

    # Stderr summary table
    print("\n[bench] summary (median across runs):", file=sys.stderr)
    by_key = {}
    for pid, mode, g, pre in summary_rows:
        by_key.setdefault((pid, mode), {"gen": [], "pre": []})
        by_key[(pid, mode)]["gen"].append(g)
        by_key[(pid, mode)]["pre"].append(pre)
    print(f"  {'prompt':<8} {'mode':<8} {'prefill':>10} {'gen':>8}", file=sys.stderr)
    import statistics as _st
    for (pid, mode), v in by_key.items():
        print(f"  {pid:<8} {mode:<8} {_st.median(v['pre']):>10.1f} "
              f"{_st.median(v['gen']):>8.2f}", file=sys.stderr)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
