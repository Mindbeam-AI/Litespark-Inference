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


def quantize_to_ternary_gptq(
    weight: torch.Tensor,
    H: Optional[torch.Tensor] = None,
    blocksize: int = 128,
    target_sparsity: float = 0.33
) -> Tuple[torch.Tensor, float, QuantizationStats]:
    """
    GPTQ-style ternary quantization with error compensation.

    Uses Hessian information to quantize weights column-by-column,
    updating remaining weights to compensate for quantization error.

    This is a simplified version - full GPTQ uses calibration data
    to compute the Hessian. Here we use weight magnitude as proxy.

    Args:
        weight: Float tensor to quantize [out_features, in_features]
        H: Optional Hessian diagonal (if None, use weight magnitude)
        blocksize: Process this many columns at a time
        target_sparsity: Target fraction of zeros

    Returns:
        w_ternary: Ternary tensor {-1, 0, +1}
        scale: Scaling factor
        stats: Quantization statistics
    """
    N, K = weight.shape
    w = weight.clone()

    # Compute importance scores (Hessian diagonal approximation)
    if H is None:
        # Use squared weight magnitude as importance proxy
        H = (w ** 2).sum(dim=0) + 1e-10

    # Find threshold for target sparsity
    abs_weight = w.abs()
    threshold = torch.quantile(abs_weight.flatten(), target_sparsity).item()

    # Compute optimal scale from weights that will be kept
    mask = abs_weight > threshold
    if mask.sum() == 0:
        scale = 1.0
    else:
        scale = abs_weight[mask].mean().item()

    # Initialize output
    w_ternary = torch.zeros_like(w, dtype=torch.int8)

    # Process in blocks for efficiency
    for start in range(0, K, blocksize):
        end = min(start + blocksize, K)

        for j in range(start, end):
            col = w[:, j]

            # Quantize this column
            q = torch.zeros(N, dtype=torch.int8)
            q[col > threshold] = 1
            q[col < -threshold] = -1

            w_ternary[:, j] = q

            # Compute quantization error
            q_float = q.float() * scale
            error = col - q_float

            # Distribute error to remaining columns (simplified GPTQ update)
            if j < K - 1:
                # Weight the error distribution by inverse Hessian
                remaining_H = H[j+1:end]
                if remaining_H.sum() > 0:
                    # Distribute error proportionally to remaining columns
                    weights_dist = remaining_H / remaining_H.sum()
                    for k, wd in enumerate(weights_dist):
                        if j + 1 + k < end:
                            w[:, j + 1 + k] += error * wd.item() * 0.1  # Damped update

    # Compute statistics
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


def quantize_to_ternary_smoothquant(
    weight: torch.Tensor,
    act_scales: Optional[torch.Tensor] = None,
    alpha: float = 0.5,
    target_sparsity: float = 0.33
) -> Tuple[torch.Tensor, float, QuantizationStats]:
    """
    SmoothQuant-style ternary quantization.

    Migrates quantization difficulty from activations to weights by
    scaling weights based on activation magnitude. This makes weight
    distribution more uniform and easier to quantize.

    Args:
        weight: Float tensor to quantize [out_features, in_features]
        act_scales: Per-input-channel activation scales (if None, use weight stats)
        alpha: Smoothing factor (0=all on weights, 1=all on activations)
        target_sparsity: Target fraction of zeros

    Returns:
        w_ternary: Ternary tensor {-1, 0, +1}
        scale: Scaling factor
        stats: Quantization statistics
    """
    N, K = weight.shape

    # If no activation scales provided, estimate from weight magnitude
    if act_scales is None:
        # Use per-column weight magnitude as proxy for activation importance
        act_scales = weight.abs().mean(dim=0) + 1e-10

    # Compute weight scales (per-column max)
    w_scales = weight.abs().max(dim=0).values + 1e-10

    # Compute smoothing scales: s = act_scales^alpha / w_scales^(1-alpha)
    smooth_scales = (act_scales ** alpha) / (w_scales ** (1 - alpha))

    # Apply smoothing to weights (equivalent to dividing activations by smooth_scales)
    w_smooth = weight * smooth_scales.unsqueeze(0)

    # Now quantize the smoothed weights
    abs_weight = w_smooth.abs()
    threshold = torch.quantile(abs_weight.flatten(), target_sparsity).item()

    # Compute scale from smoothed weights
    mask = abs_weight > threshold
    if mask.sum() == 0:
        scale = 1.0
    else:
        scale = abs_weight[mask].mean().item()

    # Quantize
    w_ternary = torch.zeros_like(w_smooth, dtype=torch.int8)
    w_ternary[w_smooth > threshold] = 1
    w_ternary[w_smooth < -threshold] = -1

    # Compute statistics (compare against original weight)
    original_nonzero = (weight != 0).sum().item()
    quantized_nonzero = (w_ternary != 0).sum().item()
    sparsity = 1.0 - (quantized_nonzero / weight.numel())

    # For MSE, account for the smoothing transform
    # Reconstructed = w_ternary * scale / smooth_scales
    w_reconstructed = (w_ternary.float() * scale) / smooth_scales.unsqueeze(0)
    mse = ((weight - w_reconstructed) ** 2).mean().item()

    stats = QuantizationStats(
        original_nonzero=original_nonzero,
        quantized_nonzero=quantized_nonzero,
        sparsity=sparsity,
        scale=scale,
        mse=mse
    )

    return w_ternary, scale, stats


def iterative_ternary_fitting(
    weight: torch.Tensor,
    max_iters: int = 10,
    verbose: bool = False
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """
    Iterative Ternary Fitting (ITF) from PT²-LLM paper.

    Alternates between:
    1. Optimal scale (α) and shift (μ) computation given fixed T
    2. Optimal ternary assignment given fixed α, μ

    Args:
        weight: [N, K] weight matrix
        max_iters: Maximum iterations (typically converges in <10)
        verbose: Print convergence info

    Returns:
        T: Ternary weights {-1, 0, +1} as int8
        alpha: Per-row scales [N]
        mu: Per-row shifts [N]
    """
    N, K = weight.shape
    W = weight.float()

    # Initialize with simple absolute mean scaling
    alpha = W.abs().mean(dim=1, keepdim=True).clamp(min=1e-10)  # [N, 1]
    mu = torch.zeros(N, 1, device=W.device, dtype=W.dtype)

    # Initial ternary assignment
    Z = (W - mu) / alpha
    T = Z.round().clamp(-1, 1).to(torch.int8)

    for iteration in range(max_iters):
        T_old = T.clone()
        T_float = T.float()

        # Step 1: Optimize α, μ given fixed T
        # Minimize ||W - (α*T + μ)||² over α, μ for each row
        # Closed-form solution from normal equations

        T_sq_sum = (T_float ** 2).sum(dim=1, keepdim=True)      # [N, 1]
        T_sum = T_float.sum(dim=1, keepdim=True)                 # [N, 1]
        WT_sum = (W * T_float).sum(dim=1, keepdim=True)          # [N, 1]
        W_sum = W.sum(dim=1, keepdim=True)                       # [N, 1]

        # Solve 2x2 linear system for each row:
        # [K, T_sum] [μ]   [W_sum]
        # [T_sum, T²_sum] [α] = [WT_sum]

        denom = K * T_sq_sum - T_sum ** 2 + 1e-10

        # α = (K * WT_sum - T_sum * W_sum) / denom
        # μ = (T²_sum * W_sum - T_sum * WT_sum) / denom
        alpha_new = (K * WT_sum - T_sum * W_sum) / denom
        mu_new = (T_sq_sum * W_sum - T_sum * WT_sum) / denom

        # Ensure positive scale
        alpha = alpha_new.clamp(min=1e-10)
        mu = mu_new

        # Step 2: Update T given fixed α, μ
        Z = (W - mu) / alpha
        T = Z.round().clamp(-1, 1).to(torch.int8)

        # Check convergence
        changed = (T != T_old).sum().item()
        if verbose:
            mse = ((W - (T.float() * alpha + mu)) ** 2).mean().item()
            print(f"  ITF iter {iteration}: changed={changed}, MSE={mse:.6e}")

        if changed == 0:
            if verbose:
                print(f"  ITF converged at iteration {iteration}")
            break

    return T, alpha.squeeze(1), mu.squeeze(1)


def activation_aware_alignment(
    weight: torch.Tensor,
    T: torch.Tensor,
    calibration_activations: list,
    verbose: bool = False
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Activation-aware Grid Alignment (AGA) from PT²-LLM paper.

    Refines scale and shift using activation statistics to minimize
    output error rather than just weight reconstruction error.

    Args:
        weight: [N, K] original weight matrix
        T: [N, K] ternary weights from ITF
        calibration_activations: List of [batch, K] activation tensors
        verbose: Print debug info

    Returns:
        alpha: Refined per-row scales [N]
        mu: Refined per-row shifts [N]
    """
    N, K = weight.shape
    W = weight.float()
    T_float = T.float()

    # Compute activation covariance diagonal: S[k] = sum_samples(X[:,k]^2)
    S = torch.zeros(K, device=W.device, dtype=W.dtype)
    num_samples = 0

    for X in calibration_activations:
        # X: [batch, K]
        S += (X.float() ** 2).sum(dim=0)
        num_samples += X.shape[0]

    if num_samples == 0:
        # Fall back to uniform weighting
        S = torch.ones(K, device=W.device, dtype=W.dtype)
    else:
        S = S / num_samples + 1e-10

    # Weighted sums for each row
    # d = sum(S) = total weight
    d = S.sum()

    # v[i] = sum_k(T[i,k] * S[k])
    v = (T_float * S.unsqueeze(0)).sum(dim=1)  # [N]

    # T²S[i] = sum_k(T[i,k]² * S[k])
    T_sq_S = ((T_float ** 2) * S.unsqueeze(0)).sum(dim=1)  # [N]

    # WTS[i] = sum_k(W[i,k] * T[i,k] * S[k])
    WT_S = (W * T_float * S.unsqueeze(0)).sum(dim=1)  # [N]

    # WS[i] = sum_k(W[i,k] * S[k])
    W_S = (W * S.unsqueeze(0)).sum(dim=1)  # [N]

    # Solve weighted normal equations
    denom = d * T_sq_S - v ** 2 + 1e-10
    alpha = (d * WT_S - v * W_S) / denom
    mu = (T_sq_S * W_S - v * WT_S) / denom

    alpha = alpha.clamp(min=1e-10)

    if verbose:
        W_recon = T_float * alpha.unsqueeze(1) + mu.unsqueeze(1)
        mse = ((W - W_recon) ** 2).mean().item()
        print(f"  AGA refinement: MSE={mse:.6e}")

    return alpha, mu


def quantize_to_ternary_pt2(
    weight: torch.Tensor,
    calibration_activations: Optional[list] = None,
    max_iters: int = 10,
    verbose: bool = False
) -> Tuple[torch.Tensor, float, QuantizationStats]:
    """
    PT²-LLM quantization: ITF + AGA.

    This is the state-of-the-art PTQ method for ternary quantization,
    achieving ~2x perplexity degradation (vs 1000x+ for naive methods).

    Args:
        weight: Float tensor to quantize [N, K]
        calibration_activations: Optional list of activation tensors for AGA
        max_iters: Max ITF iterations
        verbose: Print progress

    Returns:
        w_ternary: Ternary tensor {-1, 0, +1}
        scale: Average scale (for compatibility)
        stats: Quantization statistics
    """
    # Step 1: Iterative Ternary Fitting
    T, alpha, mu = iterative_ternary_fitting(weight, max_iters, verbose)

    # Step 2: Activation-aware Grid Alignment (if calibration data provided)
    if calibration_activations is not None and len(calibration_activations) > 0:
        alpha, mu = activation_aware_alignment(weight, T, calibration_activations, verbose)

    # Compute statistics
    W = weight.float()
    T_float = T.float()
    W_reconstructed = T_float * alpha.unsqueeze(1) + mu.unsqueeze(1)

    original_nonzero = (weight != 0).sum().item()
    quantized_nonzero = (T != 0).sum().item()
    sparsity = 1.0 - (quantized_nonzero / weight.numel())
    mse = ((W - W_reconstructed) ** 2).mean().item()

    # Use mean scale for compatibility with our kernel format
    # Note: Full PT²-LLM uses per-row scales, but our kernels use single scale
    avg_scale = alpha.mean().item()

    stats = QuantizationStats(
        original_nonzero=original_nonzero,
        quantized_nonzero=quantized_nonzero,
        sparsity=sparsity,
        scale=avg_scale,
        mse=mse
    )

    return T, avg_scale, stats


def quantize_to_ternary_awq(
    weight: torch.Tensor,
    act_scales: Optional[torch.Tensor] = None,
    top_k_percent: float = 1.0,
    target_sparsity: float = 0.33
) -> Tuple[torch.Tensor, float, QuantizationStats]:
    """
    AWQ-style (Activation-aware Weight Quantization) ternary quantization.

    Preserves weights in channels with high activation magnitude by
    using per-channel scaling that protects important channels.

    Args:
        weight: Float tensor to quantize [out_features, in_features]
        act_scales: Per-input-channel activation scales (importance)
        top_k_percent: Protect top k% of channels more carefully
        target_sparsity: Target fraction of zeros

    Returns:
        w_ternary: Ternary tensor {-1, 0, +1}
        scale: Scaling factor
        stats: Quantization statistics
    """
    N, K = weight.shape

    # If no activation scales, use weight magnitude as proxy
    if act_scales is None:
        act_scales = weight.abs().mean(dim=0)

    # Find top-k important channels
    k = max(1, int(K * top_k_percent / 100))
    _, top_indices = torch.topk(act_scales, k)

    # Create importance mask
    importance = torch.ones(K)
    importance[top_indices] = 2.0  # Double importance for top channels

    # Compute per-channel thresholds (lower threshold for important channels)
    abs_weight = weight.abs()

    # Base threshold from target sparsity
    base_threshold = torch.quantile(abs_weight.flatten(), target_sparsity).item()

    # Adjust threshold per channel based on importance
    # Important channels get lower threshold (keep more weights)
    channel_thresholds = base_threshold / importance

    # Quantize with per-channel thresholds
    w_ternary = torch.zeros_like(weight, dtype=torch.int8)
    for j in range(K):
        col = weight[:, j]
        thresh = channel_thresholds[j].item()
        w_ternary[col > thresh, j] = 1
        w_ternary[col < -thresh, j] = -1

    # Compute scale from kept weights
    mask = w_ternary != 0
    if mask.sum() == 0:
        scale = 1.0
    else:
        scale = weight.abs()[mask].mean().item()

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
    elif method == 'gptq':
        w_ternary, scale, stats = quantize_to_ternary_gptq(weight, **kwargs)
    elif method == 'smoothquant':
        w_ternary, scale, stats = quantize_to_ternary_smoothquant(weight, **kwargs)
    elif method == 'awq':
        w_ternary, scale, stats = quantize_to_ternary_awq(weight, **kwargs)
    elif method == 'pt2':
        w_ternary, scale, stats = quantize_to_ternary_pt2(weight, **kwargs)
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
