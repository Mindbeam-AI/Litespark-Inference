"""Matched pp128/tg128-style benchmark for the Nemotron CPU variants.

Uses the public benchmark's repeated fox prompt, a full prefill warmup,
fresh caches for every run, and fixed-length greedy decode ignoring EOS.
This measures throughput, not response quality. Run variants sequentially.
"""
import argparse
import hashlib
import json
from pathlib import Path
import platform
import resource
import statistics
import time

import torch
from transformers import AutoTokenizer

from litespark_inference.nemotron_cpu import load_nemotron


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", required=True)
    parser.add_argument("--dense-int4")
    parser.add_argument("--threads", type=int, default=10)
    parser.add_argument("--prompt-tokens", type=int, default=128)
    parser.add_argument("--decode-steps", type=int, default=128)
    parser.add_argument("--runs", type=int, default=2)
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args()
    if min(args.prompt_tokens, args.decode_steps, args.runs) < 1:
        parser.error("Token counts and runs must be positive")
    started = time.perf_counter()
    model = load_nemotron(args.model, args.threads, int4_overlay=args.dense_int4)
    tokenizer = AutoTokenizer.from_pretrained(args.model, local_files_only=True, trust_remote_code=True)
    prompt = "The quick brown fox jumps over the lazy dog. " * max(20, args.prompt_tokens)
    ids = tokenizer.encode(prompt, add_special_tokens=False)[:args.prompt_tokens]
    tokens = torch.tensor([ids])
    report = {"variant": "ternary_dense_int4" if args.dense_int4 else "ternary_dense_bf16",
        "platform": platform.platform(), "threads": args.threads,
        "load_seconds": time.perf_counter()-started, "prompt_tokens": len(ids),
        "prompt_ids_sha256": hashlib.sha256(json.dumps(ids).encode()).hexdigest(),
        "decode_steps": args.decode_steps, "ignore_eos_for_fixed_workload": True,
        "warmup": "one complete prefill", "runs": []}
    with torch.inference_mode():
        start = time.perf_counter()
        warmup = model(input_ids=tokens, use_cache=True, logits_to_keep=1)
        report["warmup_seconds"] = time.perf_counter()-start
        del warmup
        print(json.dumps({"warmup_seconds": report["warmup_seconds"]}), flush=True)
        for run in range(args.runs):
            start = time.perf_counter()
            result = model(input_ids=tokens, use_cache=True, logits_to_keep=1)
            prefill = time.perf_counter()-start
            usage = resource.getrusage(resource.RUSAGE_SELF)
            start = time.perf_counter()
            generated = []
            for step in range(args.decode_steps):
                token = int(result.logits[0,-1].argmax())
                generated.append(token)
                result = model(input_ids=torch.tensor([[token]]), past_key_values=result.past_key_values,
                               use_cache=True, logits_to_keep=1)
                if not torch.isfinite(result.logits).all():
                    raise RuntimeError("Nonfinite benchmark logits")
                if (step+1) % 32 == 0:
                    print(json.dumps({"run": run, "decode_steps": step+1,
                                      "seconds": time.perf_counter()-start}), flush=True)
            elapsed = time.perf_counter()-start
            report["runs"].append({"prefill_seconds": prefill, "decode_seconds": elapsed,
                "decode_tokens_per_second": args.decode_steps/elapsed, "token_ids": generated,
                "page_faults": resource.getrusage(resource.RUSAGE_SELF).ru_majflt-usage.ru_majflt})
            report["peak_rss_bytes"] = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
            args.output.write_text(json.dumps(report, indent=2)+"\n")
            print(json.dumps({k:v for k,v in report["runs"][-1].items() if k != "token_ids"}), flush=True)
            del result
    report["decode_tokens_per_second"] = args.decode_steps/statistics.mean(r["decode_seconds"] for r in report["runs"])
    report["prefill_seconds"] = statistics.mean(r["prefill_seconds"] for r in report["runs"])
    args.output.write_text(json.dumps(report, indent=2)+"\n")
    print(json.dumps({k:v for k,v in report.items() if k != "runs"}, indent=2), flush=True)


if __name__ == "__main__":
    main()
