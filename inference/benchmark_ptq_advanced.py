#!/usr/bin/env python3
"""
Advanced PTQ benchmark runner.

Runs both original methods and experimental v2/new methods side-by-side.
Metrics: TTFT/throughput (short) and perplexity (WikiText-2, configurable samples).

This script reuses benchmarking utilities from benchmark_ptq.py but extends the
method set with newer variants:
  - pt2_v2, quip_v2, awq_v2, smoothquant_v2, pt2_ssr_v2, mixed_attention_v2
  - butterfly (learned orthogonal transform + PT² backend)
  - cqe_plus (SVD-clamp preconditioning + PT² backend)
"""

import argparse
from pathlib import Path
import gc

from benchmark_ptq import (
    BenchmarkConfig,
    benchmark_model_short,
    evaluate_perplexity,
    load_fp16_model,
)
from auto_quantize import load_auto_quantized_model
from ptq_models import load_ptq_model
from w8a8_models import load_w8a8_model, load_w8a8_ternary_model


ADV_METHODS = [
    # originals
    'absmean', 'percentile', 'optimal', 'gptq', 'smoothquant', 'awq',
    'pt2', 'pt2_calibrated', 'pt2_ssr', 'quip', 'omniquant',
    'auto_accuracy', 'auto_balanced', 'w8a8', 'w8a8_ternary',
    'mixed_attention', '2bit',
    # new/v2
    'pt2_v2', 'quip_v2', 'awq_v2', 'smoothquant_v2', 'pt2_ssr_v2',
    'mixed_attention_v2', 'butterfly', 'cqe_plus',
    # experimental new
    'butterfly_learned', 'pt2_distill', 'pt2_outlier',
    'ptqtp', 'pt2_faithful', 'ttq_kd',
]


def load_model_for_method(model_name: str, method: str, use_cache: bool, force_requantize: bool):
    """Dispatch to appropriate loader with special cases for new methods."""
    if method == 'fp16':
        return load_fp16_model(model_name)
    if method == 'w8a8':
        return load_w8a8_model(model_name)
    if method == 'w8a8_ternary':
        return load_w8a8_ternary_model(model_name)
    if method.startswith('auto_'):
        strategy = method.replace('auto_', '')
        return load_auto_quantized_model(
            model_name,
            strategy=strategy,
            use_cache=use_cache,
            force_requantize=force_requantize
        )[:2]
    if method == 'mixed_attention_v2':
        # Keep Q/K higher precision, quantize the rest
        return load_ptq_model(
            model_name,
            method='pt2_calibrated',
            skip_patterns=['q_proj', 'k_proj'],
            use_cache=use_cache,
            force_requantize=force_requantize
        )
    if method == 'mixed_attention':
        return load_ptq_model(
            model_name,
            method='pt2_calibrated',
            skip_patterns=['q_proj', 'k_proj', 'v_proj', 'o_proj'],
            use_cache=use_cache,
            force_requantize=force_requantize
        )
    # Default: PTQ loader with method string
    return load_ptq_model(
        model_name,
        method=method,
        use_cache=use_cache,
        force_requantize=force_requantize
    )


def main():
    parser = argparse.ArgumentParser(description="Advanced PTQ benchmark (ternary)")
    parser.add_argument('--model', type=str, required=True, help='HF model name')
    parser.add_argument('--methods', nargs='+', default=['pt2', 'pt2_v2', 'pt2_ssr', 'pt2_ssr_v2', 'butterfly'],
                        choices=ADV_METHODS + ['fp16'])
    parser.add_argument('--compare-all', action='store_true',
                        help='Benchmark all known methods (fp16 + originals + new)')
    parser.add_argument('--ppl-samples', type=int, default=50, help='WikiText-2 samples for perplexity')
    parser.add_argument('--no-cache', action='store_true', help='Do not load/save PTQ cache')
    parser.add_argument('--force-requantize', action='store_true', help='Ignore cache and requantize')
    parser.add_argument('--short-only', action='store_true', help='Skip long benchmarks (only short + ppl)')
    parser.add_argument('--save-summary', type=str, default=None, help='Path to save txt summary')
    parser.add_argument('--save-plot', type=str, default=None, help='Path to save perplexity bar plot')
    args = parser.parse_args()

    if args.compare_all:
        methods = ['fp16'] + ADV_METHODS
    else:
        methods = args.methods

    use_cache = not args.no_cache
    force_requantize = args.force_requantize

    config = BenchmarkConfig()
    config.ppl_max_samples = args.ppl_samples

    results = {'model': args.model, 'methods': {}}

    for method in methods:
        print(f"\n{'='*70}")
        print(f"Method: {method}")
        print("="*70)

        try:
            model, tokenizer = load_model_for_method(
                args.model, method,
                use_cache=use_cache,
                force_requantize=force_requantize
            )
        except Exception as e:
            print(f"  [!] Failed to load method {method}: {e}")
            continue

        # Short benchmark
        short_stats = benchmark_model_short(model, tokenizer, method, config)

        # Perplexity
        ppl, ppl_stats = evaluate_perplexity(
            model, tokenizer,
            max_samples=args.ppl_samples,
            verbose=True
        )

        results['methods'][method] = {
            'short': short_stats,
            'ppl': ppl,
            'ppl_stats': ppl_stats,
        }

        # Cleanup to free memory
        del model
        gc.collect()

    # Print summary
    print("\nSummary:")
    for m, r in results['methods'].items():
        ttft = r['short'].get('mean_ttft_ms', float('nan'))
        tokps = r['short'].get('tokens_per_sec', float('nan'))
        ppl = r['ppl']
        print(f"  {m:20s}  TTFT={ttft:.2f} ms   Tok/s={tokps:.2f}   PPL={ppl:.2f}")

    if args.save_summary:
        out = Path(args.save_summary)
        out.parent.mkdir(parents=True, exist_ok=True)
        with open(out, 'w') as f:
            for m, r in results['methods'].items():
                ttft = r['short'].get('mean_ttft_ms', float('nan'))
                tokps = r['short'].get('tokens_per_sec', float('nan'))
                ppl = r['ppl']
                f.write(f"{m:20s} TTFT={ttft:.2f} Tok/s={tokps:.2f} PPL={ppl:.2f}\n")
        print(f"Saved summary to {out}")

    if args.save_plot:
        try:
            import matplotlib
            matplotlib.use('Agg')
            import matplotlib.pyplot as plt
            labels = list(results['methods'].keys())
            ppls = [results['methods'][m]['ppl'] for m in labels]
            plt.figure(figsize=(10, 5))
            plt.bar(labels, ppls)
            plt.xticks(rotation=45, ha='right')
            plt.ylabel("Perplexity (lower is better)")
            plt.tight_layout()
            out = Path(args.save_plot)
            out.parent.mkdir(parents=True, exist_ok=True)
            plt.savefig(out, dpi=150)
            plt.close()
            print(f"Saved plot to {out}")
        except ImportError:
            print("matplotlib not available; skipping plot.")


if __name__ == '__main__':
    main()
