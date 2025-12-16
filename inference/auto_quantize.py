#!/usr/bin/env python3
"""
Automated Mixed-Strategy Quantization

This module provides a framework to automatically select the best quantization
method on a per-layer basis, optimizing for the best quality-performance tradeoff.

The strategy:
1. Profile each layer with multiple quantization methods
2. Select the method that minimizes reconstruction error
3. Optionally apply LoRC compensation to high-error layers

Usage:
    from auto_quantize import load_auto_quantized_model

    model, tokenizer = load_auto_quantized_model(
        'TinyLlama/TinyLlama-1.1B-Chat-v1.0',
        strategy='accuracy',  # or 'balanced', 'speed'
    )
"""

import torch
import torch.nn as nn
from typing import Dict, List, Optional, Tuple, Any
from dataclasses import dataclass
from enum import Enum
import time

from ptq_quantize import (
    prepare_ternary_weight_ptq,
    QuantizationStats,
    collect_calibration_data,
)
from ptq_models import (
    PTQTernaryLinear,
    PTQModel,
    get_kernel,
    get_cache_path,
    save_quantized_model,
    load_quantized_model,
)
from transformers import AutoModelForCausalLM, AutoTokenizer, AutoConfig


class Strategy(Enum):
    ACCURACY = "accuracy"      # Best quality, slowest quantization
    BALANCED = "balanced"      # Good quality, reasonable speed
    SPEED = "speed"            # Fast quantization, acceptable quality


@dataclass
class LayerQuantConfig:
    """Configuration for a single layer's quantization."""
    method: str
    use_lorc: bool = False
    lorc_rank: int = 4
    mse: float = 0.0
    sparsity: float = 0.0


@dataclass
class AutoQuantConfig:
    """Global configuration for auto-quantization."""
    strategy: Strategy = Strategy.BALANCED
    candidate_methods: List[str] = None
    lorc_threshold: float = 0.01  # Apply LoRC if MSE > threshold
    lorc_rank: int = 4
    calibration_samples: int = 32
    verbose: bool = True

    def __post_init__(self):
        if self.candidate_methods is None:
            if self.strategy == Strategy.ACCURACY:
                # Try all methods, pick best
                self.candidate_methods = [
                    'pt2_calibrated', 'pt2', 'omniquant', 'quip', 'optimal'
                ]
            elif self.strategy == Strategy.BALANCED:
                # Fast methods only
                self.candidate_methods = ['pt2_calibrated', 'pt2', 'optimal']
            else:  # SPEED
                # Fastest methods
                self.candidate_methods = ['absmean', 'optimal']


def evaluate_method_for_layer(
    weight: torch.Tensor,
    method: str,
    calibration_activations: Optional[list] = None,
    num_eval_samples: int = 10,
) -> Tuple[float, float, QuantizationStats]:
    """
    Evaluate a quantization method on a single layer.

    Returns:
        (mse, reconstruction_error, stats)
    """
    # Quantize
    w_int8, w_sum, scale, stats, extra_info = prepare_ternary_weight_ptq(
        weight,
        method=method,
        calibration_activations=calibration_activations,
    )

    # Reconstruct for error calculation
    # w_int8 contains packed ternary in columns, extract and reconstruct
    N, K = weight.shape
    K_padded = w_int8.shape[1]

    # Simple reconstruction: use the quantized weight
    w_reconstructed = w_int8[:, :K].float() * scale

    # Calculate reconstruction error
    mse = ((weight.float() - w_reconstructed) ** 2).mean().item()

    # If we have calibration data, calculate output error
    output_error = 0.0
    if calibration_activations is not None and len(calibration_activations) > 0:
        for i, act in enumerate(calibration_activations[:num_eval_samples]):
            if act.dim() == 3:
                act = act.view(-1, act.shape[-1])
            y_orig = act.float() @ weight.T
            y_quant = act.float() @ w_reconstructed.T
            output_error += ((y_orig - y_quant) ** 2).mean().item()
        output_error /= min(len(calibration_activations), num_eval_samples)

    return mse, output_error, stats


def select_best_method(
    weight: torch.Tensor,
    candidate_methods: List[str],
    calibration_activations: Optional[list] = None,
    verbose: bool = False,
) -> Tuple[str, float]:
    """
    Select the best quantization method for a layer.

    Returns:
        (best_method, best_mse)
    """
    best_method = candidate_methods[0]
    best_mse = float('inf')

    for method in candidate_methods:
        try:
            mse, output_error, stats = evaluate_method_for_layer(
                weight, method, calibration_activations
            )

            # Use output error if available, else MSE
            score = output_error if output_error > 0 else mse

            if verbose:
                print(f"      {method}: MSE={mse:.2e}, output_error={output_error:.2e}")

            if score < best_mse:
                best_mse = score
                best_method = method

        except Exception as e:
            if verbose:
                print(f"      {method}: FAILED ({e})")
            continue

    return best_method, best_mse


def auto_quantize_model(
    model: nn.Module,
    config: AutoQuantConfig,
    calibration_data: Optional[Dict[str, list]] = None,
    skip_layers: Optional[List[str]] = None,
    num_threads: int = None,
) -> Tuple[Dict[str, LayerQuantConfig], Dict[str, QuantizationStats]]:
    """
    Automatically quantize all linear layers with per-layer method selection.

    Returns:
        (layer_configs, quant_stats)
    """
    skip_layers = skip_layers or []
    layer_configs = {}
    quant_stats = {}

    def should_skip(name: str) -> bool:
        return any(pattern in name for pattern in skip_layers)

    # Collect all linear layers
    linear_layers = []
    for name, module in model.named_modules():
        if isinstance(module, nn.Linear) and not should_skip(name):
            linear_layers.append((name, module))

    if config.verbose:
        print(f"\nAuto-quantizing {len(linear_layers)} layers...")
        print(f"Strategy: {config.strategy.value}")
        print(f"Candidate methods: {config.candidate_methods}")

    for idx, (name, module) in enumerate(linear_layers):
        weight = module.weight.data.float()

        if config.verbose:
            print(f"\n  [{idx+1}/{len(linear_layers)}] {name}")
            print(f"    Shape: {list(weight.shape)}")

        # Get calibration data for this layer
        layer_calib = None
        if calibration_data is not None and name in calibration_data:
            layer_calib = calibration_data[name]

        # Select best method
        if len(config.candidate_methods) > 1:
            best_method, best_mse = select_best_method(
                weight,
                config.candidate_methods,
                layer_calib,
                verbose=config.verbose,
            )
        else:
            best_method = config.candidate_methods[0]
            best_mse = 0.0

        # Decide whether to use LoRC
        use_lorc = best_mse > config.lorc_threshold

        if config.verbose:
            print(f"    -> Selected: {best_method}" +
                  (f" + LoRC(rank={config.lorc_rank})" if use_lorc else ""))

        # Store config
        layer_configs[name] = LayerQuantConfig(
            method=best_method,
            use_lorc=use_lorc,
            lorc_rank=config.lorc_rank if use_lorc else 0,
            mse=best_mse,
        )

        # Actually quantize with selected method
        ptq_linear = PTQTernaryLinear(
            module.in_features,
            module.out_features,
            bias=module.bias is not None,
            num_threads=num_threads,
        )

        quant_kwargs = {
            'calibration_activations': layer_calib,
            'use_lorc': use_lorc,
            'lorc_rank': config.lorc_rank,
        }

        ptq_linear.quantize_from_linear(module, method=best_method, **quant_kwargs)

        # Replace in model
        parts = name.rsplit('.', 1)
        if len(parts) == 1:
            parent = model
            attr_name = parts[0]
        else:
            parent_name, attr_name = parts
            parent = model.get_submodule(parent_name)

        setattr(parent, attr_name, ptq_linear)
        quant_stats[name] = ptq_linear.quant_stats
        layer_configs[name].sparsity = ptq_linear.quant_stats.sparsity

    return layer_configs, quant_stats


def load_auto_quantized_model(
    model_name: str,
    strategy: str = 'balanced',
    candidate_methods: List[str] = None,
    calibration_samples: int = 32,
    lorc_threshold: float = 0.01,
    lorc_rank: int = 4,
    skip_lm_head: bool = True,
    skip_embeddings: bool = True,
    num_threads: int = None,
    verbose: bool = True,
    use_cache: bool = True,
    cache_dir: Optional[str] = None,
    force_requantize: bool = False,
) -> Tuple[PTQModel, Any, Dict[str, LayerQuantConfig]]:
    """
    Load and auto-quantize a HuggingFace model.

    Args:
        model_name: HuggingFace model name
        strategy: 'accuracy', 'balanced', or 'speed'
        candidate_methods: Override default methods for strategy
        calibration_samples: Number of calibration samples
        lorc_threshold: MSE threshold for applying LoRC
        lorc_rank: Rank for LoRC matrices
        skip_lm_head: Don't quantize language model head
        skip_embeddings: Don't quantize embeddings
        num_threads: Number of threads for kernels
        verbose: Print progress
        use_cache: Cache quantized model
        cache_dir: Cache directory
        force_requantize: Ignore cache

    Returns:
        (model, tokenizer, layer_configs)
    """
    print(f"Loading model: {model_name}")

    # Create config
    config = AutoQuantConfig(
        strategy=Strategy(strategy),
        candidate_methods=candidate_methods,
        lorc_threshold=lorc_threshold,
        lorc_rank=lorc_rank,
        calibration_samples=calibration_samples,
        verbose=verbose,
    )

    # Load HF model
    hf_config = AutoConfig.from_pretrained(model_name, trust_remote_code=True)

    print("  Loading weights...")
    hf_model = AutoModelForCausalLM.from_pretrained(
        model_name,
        torch_dtype=torch.float32,
        device_map=None,
        trust_remote_code=True,
    )

    tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    # Skip layers
    skip_layers = []
    if skip_lm_head:
        skip_layers.extend(['lm_head', 'output'])
    if skip_embeddings:
        skip_layers.extend(['embed', 'wte', 'wpe'])

    # Check cache
    cache_method = f"auto_{strategy}"
    cache_path = get_cache_path(model_name, cache_method, cache_dir)

    if use_cache and not force_requantize and cache_path.exists():
        quant_stats, loaded = load_quantized_model(
            hf_model, cache_path, num_threads, skip_layers
        )
        if loaded:
            model = PTQModel(hf_model, hf_config)
            model.quant_stats = quant_stats
            model.eval()
            print(f"  Loaded from cache: {cache_path}")
            return model, tokenizer, {}

    # Collect calibration data
    print(f"\nCollecting calibration data ({calibration_samples} samples)...")
    calibration_data = collect_calibration_data(
        hf_model, tokenizer,
        num_samples=calibration_samples,
        max_length=512,
    )

    # Auto-quantize
    _, kernel_type = get_kernel()
    if kernel_type == 'fallback':
        print("  [Note] Using PyTorch fallback")

    start_time = time.time()
    layer_configs, quant_stats = auto_quantize_model(
        hf_model,
        config,
        calibration_data,
        skip_layers,
        num_threads,
    )
    elapsed = time.time() - start_time

    # Save cache
    if use_cache:
        save_quantized_model(hf_model, cache_path, quant_stats)

    # Wrap model
    model = PTQModel(hf_model, hf_config)
    model.quant_stats = quant_stats
    model.eval()

    # Print summary
    if verbose:
        print(f"\n{'='*60}")
        print("AUTO-QUANTIZATION SUMMARY")
        print(f"{'='*60}")
        print(f"Time: {elapsed:.1f}s")
        print(f"Strategy: {strategy}")

        # Count methods used
        method_counts = {}
        lorc_count = 0
        for lc in layer_configs.values():
            method_counts[lc.method] = method_counts.get(lc.method, 0) + 1
            if lc.use_lorc:
                lorc_count += 1

        print(f"\nMethods used:")
        for method, count in sorted(method_counts.items(), key=lambda x: -x[1]):
            print(f"  {method}: {count} layers")

        if lorc_count > 0:
            print(f"\nLoRC applied to {lorc_count} layers")

        avg_sparsity = sum(lc.sparsity for lc in layer_configs.values()) / len(layer_configs)
        print(f"\nAverage sparsity: {avg_sparsity:.1%}")

    return model, tokenizer, layer_configs


# ============================================================================
# CLI
# ============================================================================

if __name__ == '__main__':
    import argparse

    parser = argparse.ArgumentParser(description='Auto-quantize a model')
    parser.add_argument('--model', type=str, default='TinyLlama/TinyLlama-1.1B-Chat-v1.0',
                        help='HuggingFace model name')
    parser.add_argument('--strategy', type=str, default='balanced',
                        choices=['accuracy', 'balanced', 'speed'],
                        help='Quantization strategy')
    parser.add_argument('--calibration-samples', type=int, default=32,
                        help='Number of calibration samples')
    parser.add_argument('--no-cache', action='store_true',
                        help='Disable caching')
    parser.add_argument('--force-requantize', action='store_true',
                        help='Force re-quantization')
    args = parser.parse_args()

    print("=" * 70)
    print("AUTO-QUANTIZATION TEST")
    print("=" * 70)

    model, tokenizer, layer_configs = load_auto_quantized_model(
        args.model,
        strategy=args.strategy,
        calibration_samples=args.calibration_samples,
        use_cache=not args.no_cache,
        force_requantize=args.force_requantize,
    )

    # Test generation
    prompt = "The meaning of life is"
    print(f"\nPrompt: {prompt}")

    input_ids = tokenizer.encode(prompt, return_tensors='pt')

    print("Generating...")
    with torch.no_grad():
        output_ids = model.generate(input_ids, max_new_tokens=30, do_sample=False)

    output_text = tokenizer.decode(output_ids[0], skip_special_tokens=True)
    print(f"Output: {output_text}")
