#!/usr/bin/env python3
"""
Post-Training Quantization (PTQ) for Regular Models

This module provides functions to quantize regular FP16/FP32 models to ternary
format {-1, 0, +1} for use with our optimized CPU kernels.

Unlike BitNet (which is trained with ternary constraints), PTQ applies
quantization after training, which typically results in higher perplexity
but allows you to quantize ANY model.
"""

import torch
import torch.nn as nn
from typing import Tuple, Dict, Optional
from dataclasses import dataclass


@dataclass
class QuantizationStats:
    """Statistics from quantization process."""
    original_nonzero: int
    quantized_nonzero: int
    sparsity: float
    scale: float
    mse: float  # Mean squared error from quantization


def quantize_to_ternary_absmean(
    weight: torch.Tensor,
    threshold_ratio: float = 0.5
) -> Tuple[torch.Tensor, float, QuantizationStats]:
    """
    Quantize weight tensor to ternary {-1, 0, +1} using absolute mean scaling.

    This is the standard BitNet quantization method:
    - scale = mean(|W|)
    - W_ternary = round(W / scale).clamp(-1, 1)
    - Values within threshold_ratio * scale of zero become 0

    Args:
        weight: Float tensor to quantize
        threshold_ratio: Values within this ratio of scale become 0 (default 0.5)

    Returns:
        w_ternary: Ternary tensor {-1, 0, +1} as int8
        scale: Scaling factor to reconstruct approximate weights
        stats: Quantization statistics
    """
    # Compute scale
    abs_mean = weight.abs().mean()
    if abs_mean < 1e-10:
        # All zeros
        return torch.zeros_like(weight, dtype=torch.int8), 1.0, QuantizationStats(
            original_nonzero=0, quantized_nonzero=0, sparsity=1.0, scale=1.0, mse=0.0
        )

    scale = abs_mean.item()

    # Normalize
    w_normalized = weight / scale

    # Apply threshold for zeros
    threshold = threshold_ratio
    w_ternary = torch.zeros_like(weight, dtype=torch.int8)
    w_ternary[w_normalized > threshold] = 1
    w_ternary[w_normalized < -threshold] = -1

    # Compute statistics
    original_nonzero = (weight != 0).sum().item()
    quantized_nonzero = (w_ternary != 0).sum().item()
    sparsity = 1.0 - (quantized_nonzero / weight.numel())

    # MSE for reconstruction quality
    w_reconstructed = w_ternary.float() * scale
    mse = ((weight - w_reconstructed) ** 2).mean().item()

    stats = QuantizationStats(
        original_nonzero=original_nonzero,
        quantized_nonzero=quantized_nonzero,
        sparsity=sparsity,
        scale=scale,
        mse=mse
    )

    return w_ternary, scale, stats


def quantize_to_ternary_percentile(
    weight: torch.Tensor,
    zero_percentile: float = 33.0
) -> Tuple[torch.Tensor, float, QuantizationStats]:
    """
    Quantize weight tensor to ternary using percentile-based thresholds.

    This method sets the threshold such that approximately `zero_percentile`%
    of weights become zero, potentially preserving more important weights.

    Args:
        weight: Float tensor to quantize
        zero_percentile: Percentage of weights to zero out (default 33%)

    Returns:
        w_ternary: Ternary tensor {-1, 0, +1} as int8
        scale: Scaling factor
        stats: Quantization statistics
    """
    abs_weight = weight.abs()

    # Find threshold at given percentile
    threshold = torch.quantile(abs_weight.flatten(), zero_percentile / 100.0).item()

    # Scale is mean of non-zeroed weights
    mask = abs_weight > threshold
    if mask.sum() == 0:
        scale = 1.0
    else:
        scale = abs_weight[mask].mean().item()

    # Quantize
    w_ternary = torch.zeros_like(weight, dtype=torch.int8)
    w_ternary[weight > threshold] = 1
    w_ternary[weight < -threshold] = -1

    # Statistics
    original_nonzero = (weight != 0).sum().item()
    quantized_nonzero = (w_ternary != 0).sum().item()
    sparsity = 1.0 - (quantized_nonzero / weight.numel())

    w_reconstructed = w_ternary.float() * scale
    mse = ((weight - w_reconstructed) ** 2).mean().item()

    stats = QuantizationStats(
        original_nonzero=original_nonzero,
        quantized_nonzero=quantized_nonzero,
        sparsity=sparsity,
        scale=scale,
        mse=mse
    )

    return w_ternary, scale, stats


def quantize_to_ternary_optimal(
    weight: torch.Tensor,
    target_sparsity: float = 0.33
) -> Tuple[torch.Tensor, float, QuantizationStats]:
    """
    Quantize with optimal scale to minimize MSE for given sparsity target.

    This method searches for the best scale factor that minimizes
    reconstruction error while achieving approximately the target sparsity.

    Args:
        weight: Float tensor to quantize
        target_sparsity: Target fraction of zeros (default 0.33)

    Returns:
        w_ternary: Ternary tensor
        scale: Optimal scaling factor
        stats: Quantization statistics
    """
    abs_weight = weight.abs()

    # Find threshold for target sparsity
    threshold = torch.quantile(abs_weight.flatten(), target_sparsity).item()

    # Create mask of non-zero elements
    pos_mask = weight > threshold
    neg_mask = weight < -threshold
    nonzero_mask = pos_mask | neg_mask

    if nonzero_mask.sum() == 0:
        return torch.zeros_like(weight, dtype=torch.int8), 1.0, QuantizationStats(
            original_nonzero=(weight != 0).sum().item(),
            quantized_nonzero=0,
            sparsity=1.0,
            scale=1.0,
            mse=(weight ** 2).mean().item()
        )

    # Optimal scale minimizes MSE: scale = mean(|W| for nonzero W_ternary)
    # For ternary, this simplifies to mean of absolute values of kept weights
    scale = abs_weight[nonzero_mask].mean().item()

    # Quantize
    w_ternary = torch.zeros_like(weight, dtype=torch.int8)
    w_ternary[pos_mask] = 1
    w_ternary[neg_mask] = -1

    # Statistics
    original_nonzero = (weight != 0).sum().item()
    quantized_nonzero = nonzero_mask.sum().item()
    sparsity = 1.0 - (quantized_nonzero / weight.numel())

    w_reconstructed = w_ternary.float() * scale
    mse = ((weight - w_reconstructed) ** 2).mean().item()

    stats = QuantizationStats(
        original_nonzero=original_nonzero,
        quantized_nonzero=quantized_nonzero,
        sparsity=sparsity,
        scale=scale,
        mse=mse
    )

    return w_ternary, scale, stats


def prepare_ternary_weight_ptq(
    weight: torch.Tensor,
    method: str = 'absmean',
    **kwargs
) -> Tuple[torch.Tensor, torch.Tensor, float, QuantizationStats]:
    """
    Prepare weight for our ternary kernels using PTQ.

    Converts weight to the format expected by our VNNI/Graviton kernels:
    - w_int8: [N, K_padded] int8 tensor with values {-1, 0, +1}
    - w_sum: [N] int32 tensor with row sums
    - scale: float scaling factor

    Args:
        weight: [N, K] float tensor (out_features, in_features)
        method: 'absmean', 'percentile', or 'optimal'
        **kwargs: Additional arguments for the quantization method

    Returns:
        w_int8: Padded int8 ternary weights
        w_sum: Row sums for bias correction
        scale: Scaling factor
        stats: Quantization statistics
    """
    N, K = weight.shape
    K_padded = ((K + 63) // 64) * 64

    # Quantize based on method
    if method == 'absmean':
        w_ternary, scale, stats = quantize_to_ternary_absmean(weight, **kwargs)
    elif method == 'percentile':
        w_ternary, scale, stats = quantize_to_ternary_percentile(weight, **kwargs)
    elif method == 'optimal':
        w_ternary, scale, stats = quantize_to_ternary_optimal(weight, **kwargs)
    else:
        raise ValueError(f"Unknown quantization method: {method}")

    # Pad to K_padded
    if K_padded > K:
        w_int8 = torch.zeros(N, K_padded, dtype=torch.int8)
        w_int8[:, :K] = w_ternary
    else:
        w_int8 = w_ternary.contiguous()

    # Compute row sums (for bias correction in kernel)
    w_sum = w_int8.sum(dim=1, dtype=torch.int32)

    return w_int8, w_sum, scale, stats


def analyze_model_weights(model: nn.Module) -> Dict[str, Dict]:
    """
    Analyze weight distributions in a model before quantization.

    Args:
        model: PyTorch model

    Returns:
        Dictionary with weight statistics per layer
    """
    stats = {}

    for name, module in model.named_modules():
        if isinstance(module, nn.Linear):
            weight = module.weight.data

            stats[name] = {
                'shape': tuple(weight.shape),
                'mean': weight.mean().item(),
                'std': weight.std().item(),
                'abs_mean': weight.abs().mean().item(),
                'min': weight.min().item(),
                'max': weight.max().item(),
                'sparsity': (weight == 0).float().mean().item(),
                'near_zero_10pct': (weight.abs() < weight.abs().mean() * 0.1).float().mean().item(),
            }

    return stats


def estimate_quantization_quality(
    model: nn.Module,
    method: str = 'absmean',
    **kwargs
) -> Dict[str, QuantizationStats]:
    """
    Estimate quantization quality for each layer without modifying the model.

    Args:
        model: PyTorch model
        method: Quantization method
        **kwargs: Method-specific arguments

    Returns:
        Dictionary mapping layer names to QuantizationStats
    """
    results = {}

    for name, module in model.named_modules():
        if isinstance(module, nn.Linear):
            weight = module.weight.data

            if method == 'absmean':
                _, _, stats = quantize_to_ternary_absmean(weight, **kwargs)
            elif method == 'percentile':
                _, _, stats = quantize_to_ternary_percentile(weight, **kwargs)
            elif method == 'optimal':
                _, _, stats = quantize_to_ternary_optimal(weight, **kwargs)
            else:
                raise ValueError(f"Unknown method: {method}")

            results[name] = stats

    return results


if __name__ == '__main__':
    # Test quantization methods
    print("Testing PTQ Quantization Methods")
    print("=" * 60)

    # Create test weight (simulating a typical LLM weight distribution)
    torch.manual_seed(42)
    weight = torch.randn(1024, 1024) * 0.02  # Typical std for LLM weights

    print(f"\nOriginal weight stats:")
    print(f"  Shape: {weight.shape}")
    print(f"  Mean: {weight.mean():.6f}")
    print(f"  Std: {weight.std():.6f}")
    print(f"  Abs mean: {weight.abs().mean():.6f}")

    # Test each method
    methods = [
        ('absmean', {'threshold_ratio': 0.5}),
        ('percentile', {'zero_percentile': 33.0}),
        ('optimal', {'target_sparsity': 0.33}),
    ]

    for method, kwargs in methods:
        print(f"\n{method.upper()} method:")
        w_int8, w_sum, scale, stats = prepare_ternary_weight_ptq(weight, method, **kwargs)

        print(f"  Scale: {scale:.6f}")
        print(f"  Sparsity: {stats.sparsity:.2%}")
        print(f"  MSE: {stats.mse:.8f}")
        print(f"  RMSE: {stats.mse ** 0.5:.6f}")

        # Reconstruction quality
        w_reconstructed = w_int8[:, :1024].float() * scale
        rel_error = ((weight - w_reconstructed).abs() / (weight.abs() + 1e-10)).mean()
        print(f"  Relative Error: {rel_error:.2%}")

        # Value distribution
        unique, counts = torch.unique(w_int8, return_counts=True)
        print(f"  Value distribution:")
        for v, c in zip(unique.tolist(), counts.tolist()):
            print(f"    {v:+d}: {c / w_int8.numel():.2%}")
