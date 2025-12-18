#!/usr/bin/env python3
"""
Convert a W8A8 quantized model into ternary using advanced methods
without touching the original W8A8 path. Supports multiple methods and
benchmarks (TTFT, Tok/s, perplexity).

Usage (from inference/):
  python3 w8a8_to_ternary.py --model TinyLlama/TinyLlama-1.1B-Chat-v1.0 \\
      --methods pt2_v2 butterfly_learned pt2_distill pt2_outlier \\
      --calibration-samples 32 --ppl-samples 50 --no-cache --force-requantize
"""

import argparse
import gc
from typing import Dict, Optional
from pathlib import Path

import torch
import torch.nn as nn
from ptq_quantize import (
    prepare_ternary_weight_ptq,
    collect_calibration_data,
    QuantizationStats,
)
from ptq_models import PTQTernaryLinear, PTQModel, get_kernel, save_quantized_model
from w8a8_models import load_w8a8_model, W8A8Linear
from benchmark_ptq import (
    BenchmarkConfig,
    benchmark_model_short,
    evaluate_perplexity,
)


ADV_DEFAULT_METHODS = [
    'pt2_v2',
    'butterfly_learned',
    'pt2_distill',
    'pt2_outlier',
    'pt2_ssr_v2',
    'cqe_plus',
]


def reconstruct_weight_from_w8a8(layer: W8A8Linear) -> torch.Tensor:
    """Return float weight [N, K] from W8A8Linear buffers."""
    N, K = layer.out_features, layer.in_features
    return layer.w_int8[:, :K].float() * layer.scale_w.unsqueeze(1)


def convert_w8a8_layer_to_ternary(
    layer: W8A8Linear,
    method: str,
    calibration_activations: Optional[list] = None,
    **kwargs,
) -> (PTQTernaryLinear, QuantizationStats):
    """Create a PTQTernaryLinear from a W8A8Linear."""
    weight = reconstruct_weight_from_w8a8(layer)
    bias = layer.bias.clone() if layer.bias is not None else None

    w_int8, w_sum, scale, stats, extra_info = prepare_ternary_weight_ptq(
        weight,
        method=method,
        calibration_activations=calibration_activations,
        **kwargs,
    )

    ptq_linear = PTQTernaryLinear(
        layer.in_features,
        layer.out_features,
        bias=bias is not None,
        num_threads=layer.num_threads,
    )
    ptq_linear.w_int8 = w_int8
    ptq_linear.w_sum = w_sum
    ptq_linear.scale = scale
    ptq_linear.quant_stats = stats
    if bias is not None:
        ptq_linear.bias = bias

    # Attach extras
    if extra_info:
        if 'salient_indices' in extra_info:
            ptq_linear.ssr_info = extra_info
            ptq_linear._prepare_ssr_correction(weight, extra_info)
        if 'U' in extra_info:
            ptq_linear.quip_info = extra_info
            ptq_linear.U = extra_info['U'].to(weight.device)
        if 'lorc_L' in extra_info:
            ptq_linear.lorc_L = extra_info['lorc_L'].to(weight.device)
            ptq_linear.lorc_R = extra_info['lorc_R'].to(weight.device)
        if 'scale_w' in extra_info:
            ptq_linear.scale_w = extra_info['scale_w'].to(weight.device)
        if 'outlier_indices' in extra_info:
            ptq_linear.outlier_indices = extra_info['outlier_indices'].to(weight.device)
        if 'outlier_w_int8' in extra_info:
            ptq_linear.outlier_w_int8 = extra_info['outlier_w_int8'].to(weight.device)
        if 'outlier_scale' in extra_info:
            ptq_linear.outlier_scale = extra_info['outlier_scale'].to(weight.device)

    return ptq_linear, stats


def convert_model(
    model: nn.Module,
    method: str,
    calibration_data: Optional[Dict[str, list]],
    **kwargs,
) -> Dict[str, QuantizationStats]:
    """Replace all W8A8Linear layers with PTQTernaryLinear using given method."""
    stats = {}
    for full_name, module in list(model.named_modules()):
        if isinstance(module, W8A8Linear):
            # Get parent and attribute
            parts = full_name.rsplit('.', 1)
            if len(parts) == 1:
                parent = model
                attr_name = parts[0]
            else:
                parent_name, attr_name = parts
                parent = model.get_submodule(parent_name)

            layer_calib = None
            if calibration_data is not None and full_name in calibration_data:
                layer_calib = calibration_data[full_name]

            ptq_linear, layer_stats = convert_w8a8_layer_to_ternary(
                module,
                method=method,
                calibration_activations=layer_calib,
                **kwargs,
            )
            setattr(parent, attr_name, ptq_linear)
            stats[full_name] = layer_stats
            print(f"  Converted {full_name}: sparsity={layer_stats.sparsity:.1%}, MSE={layer_stats.mse:.2e}")
    return stats


def run_benchmarks(model, tokenizer, method: str, ppl_samples: int):
    config = BenchmarkConfig()
    config.ppl_max_samples = ppl_samples
    short_stats = benchmark_model_short(model, tokenizer, method, config)
    ppl, ppl_stats = evaluate_perplexity(model, tokenizer, max_samples=ppl_samples, verbose=True)
    return short_stats, ppl, ppl_stats


def main():
    parser = argparse.ArgumentParser(description="Convert W8A8 model to ternary with advanced methods")
    parser.add_argument('--model', type=str, required=True, help='HF model name')
    parser.add_argument('--methods', nargs='+', default=ADV_DEFAULT_METHODS, help='Methods to try')
    parser.add_argument('--calibration-samples', type=int, default=32, help='Calibration samples for calibrated methods')
    parser.add_argument('--ppl-samples', type=int, default=50, help='Samples for perplexity eval')
    parser.add_argument('--no-cache', action='store_true', help='Skip saving cache')
    parser.add_argument('--force-requantize', action='store_true', help='Ignore any caches in loaders')
    args = parser.parse_args()

    # Load W8A8 model
    print(f"Loading W8A8 model {args.model} ...")
    w8a8_model, tokenizer = load_w8a8_model(args.model)

    # Collect calibration data if needed
    calibration_data = None
    if args.calibration_samples > 0:
        print(f"Collecting calibration data ({args.calibration_samples} samples)...")
        calibration_data = collect_calibration_data(
            w8a8_model,
            tokenizer,
            num_samples=args.calibration_samples,
            max_length=512,
        )

    # Preload kernel (may be fallback on macOS)
    _, kernel_type = get_kernel()
    if kernel_type == 'fallback':
        print("  [Note] Using PyTorch fallback kernels.")

    results = {}
    for method in args.methods:
        print(f"\n=== Converting with method: {method} ===")
        # Reload fresh W8A8 each time to avoid cross-contamination
        model, _ = load_w8a8_model(args.model)

        # Convert
        stats = convert_model(
            model,
            method=method,
            calibration_data=calibration_data,
        )

        # Wrap
        ptq_model = PTQModel(model, None)
        ptq_model.quant_stats = stats
        ptq_model.eval()

        # Benchmarks
        short_stats, ppl, ppl_stats = run_benchmarks(ptq_model, tokenizer, method, args.ppl_samples)
        results[method] = {
            'short': short_stats,
            'ppl': ppl,
            'ppl_stats': ppl_stats,
        }

        # Cache saved models if desired
        if not args.no_cache:
            cache_path = Path(__file__).parent / 'ptq_cache' / f"{args.model.replace('/', '_')}_{method}.pt"
            save_quantized_model(model, cache_path, stats)

        # Cleanup
        del model
        gc.collect()

    print("\nSummary (W8A8 -> ternary):")
    for m, r in results.items():
        ttft = r['short'].get('ttft_ms', float('nan'))
        tokps = r['short'].get('tokens_per_sec', float('nan'))
        ppl = r['ppl']
        print(f"  {m:20s} TTFT={ttft:.2f} ms   Tok/s={tokps:.2f}   PPL={ppl:.2f}")


if __name__ == '__main__':
    main()
