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


def quantize_to_ternary_perchannel(
    weight: torch.Tensor,
    threshold_ratio: float = 0.5
) -> Tuple[torch.Tensor, torch.Tensor, QuantizationStats]:
    """
    Quantize weight tensor to ternary {-1, 0, +1} with PER-CHANNEL (per-row) scaling.

    This is the key method for PTQ - each output channel gets its own scale factor,
    which dramatically improves quantization quality compared to per-tensor scaling.

    For weight matrix W[N, K] (out_features, in_features):
    - scale[n] = mean(|W[n, :]|) for each row n
    - W_ternary[n, k] = round(W[n,k] / scale[n]).clamp(-1, 1)

    Args:
        weight: [N, K] Float tensor to quantize
        threshold_ratio: Values within this ratio of scale become 0 (default 0.5)

    Returns:
        w_ternary: [N, K] Ternary tensor {-1, 0, +1} as int8
        scale_w: [N] Per-channel scaling factors
        stats: Quantization statistics (uses mean scale for summary)
    """
    N, K = weight.shape

    # Compute per-channel (per-row) scale
    # scale[n] = mean(|W[n, :]|)
    abs_row_mean = weight.abs().mean(dim=1)  # [N]

    # Handle zero rows
    abs_row_mean = torch.clamp(abs_row_mean, min=1e-10)
    scale_w = abs_row_mean  # [N]

    # Normalize each row by its scale
    # W_normalized[n, k] = W[n, k] / scale[n]
    w_normalized = weight / scale_w.unsqueeze(1)  # [N, K]

    # Apply threshold for zeros
    threshold = threshold_ratio
    w_ternary = torch.zeros_like(weight, dtype=torch.int8)
    w_ternary[w_normalized > threshold] = 1
    w_ternary[w_normalized < -threshold] = -1

    # Compute statistics
    original_nonzero = (weight != 0).sum().item()
    quantized_nonzero = (w_ternary != 0).sum().item()
    sparsity = 1.0 - (quantized_nonzero / weight.numel())

    # MSE for reconstruction quality (using per-channel scales)
    w_reconstructed = w_ternary.float() * scale_w.unsqueeze(1)
    mse = ((weight - w_reconstructed) ** 2).mean().item()

    # Use mean scale for stats summary
    mean_scale = scale_w.mean().item()

    stats = QuantizationStats(
        original_nonzero=original_nonzero,
        quantized_nonzero=quantized_nonzero,
        sparsity=sparsity,
        scale=mean_scale,
        mse=mse
    )

    return w_ternary, scale_w, stats


def quantize_to_ternary_perchannel_hessian(
    weight: torch.Tensor,
    calibration_activations: Optional[list] = None,
    threshold_ratio: float = 0.5,
    damping: float = 0.01
) -> Tuple[torch.Tensor, torch.Tensor, QuantizationStats]:
    """
    Per-channel ternary quantization with Hessian-weighted rounding.

    Uses calibration data to compute importance of each column:
    H_diag[k] = sum over samples of X[:, k]^2

    When rounding, we weight the decision by column importance.

    Args:
        weight: [N, K] Float tensor to quantize
        calibration_activations: List of [batch, K] activation tensors
        threshold_ratio: Values within this ratio of scale become 0
        damping: Regularization for Hessian diagonal

    Returns:
        w_ternary: [N, K] Ternary tensor {-1, 0, +1} as int8
        scale_w: [N] Per-channel scaling factors
        stats: Quantization statistics
    """
    N, K = weight.shape

    # Compute Hessian diagonal from calibration data
    if calibration_activations is not None and len(calibration_activations) > 0:
        H_diag = torch.zeros(K, dtype=torch.float32, device=weight.device)
        n_samples = 0
        for X in calibration_activations:
            if X.dim() == 3:  # [batch, seq, K]
                X = X.reshape(-1, K)
            H_diag += (X ** 2).sum(dim=0)
            n_samples += X.shape[0]
        H_diag = H_diag / max(n_samples, 1)
        H_diag = H_diag + damping * H_diag.mean()  # Damping
    else:
        # No calibration data - use uniform importance
        H_diag = torch.ones(K, dtype=torch.float32, device=weight.device)

    # Normalize importance weights
    H_diag = H_diag / H_diag.max()

    # Compute per-channel scale using importance-weighted mean
    # More important columns contribute more to the scale
    weighted_abs = weight.abs() * H_diag.unsqueeze(0)  # [N, K]
    scale_w = weighted_abs.sum(dim=1) / H_diag.sum()  # [N]
    scale_w = torch.clamp(scale_w, min=1e-10)

    # Normalize
    w_normalized = weight / scale_w.unsqueeze(1)  # [N, K]

    # Hessian-weighted rounding
    # For values near threshold, use Hessian to decide
    threshold = threshold_ratio
    w_ternary = torch.zeros_like(weight, dtype=torch.int8)

    # Clear positive/negative
    w_ternary[w_normalized > threshold] = 1
    w_ternary[w_normalized < -threshold] = -1

    # For borderline values, use Hessian-weighted decision
    # If rounding to 1/-1 would cause large error on important columns, round to 0
    borderline_pos = (w_normalized > 0) & (w_normalized <= threshold)
    borderline_neg = (w_normalized < 0) & (w_normalized >= -threshold)

    # Error if we round to 1 vs 0: |w_norm - 1|^2 * H vs |w_norm - 0|^2 * H
    # Round to 1 if: |w_norm - 1| < |w_norm| (i.e., w_norm > 0.5)
    # With Hessian: weight this decision by importance

    # For positive borderline: round to 1 if w_normalized > 0.5 and high importance
    err_to_1 = ((w_normalized - 1.0) ** 2) * H_diag.unsqueeze(0)
    err_to_0 = (w_normalized ** 2) * H_diag.unsqueeze(0)

    should_round_up = err_to_1 < err_to_0
    w_ternary[borderline_pos & should_round_up] = 1

    err_to_neg1 = ((w_normalized + 1.0) ** 2) * H_diag.unsqueeze(0)
    should_round_down = err_to_neg1 < err_to_0
    w_ternary[borderline_neg & should_round_down] = -1

    # Statistics
    original_nonzero = (weight != 0).sum().item()
    quantized_nonzero = (w_ternary != 0).sum().item()
    sparsity = 1.0 - (quantized_nonzero / weight.numel())

    w_reconstructed = w_ternary.float() * scale_w.unsqueeze(1)
    mse = ((weight - w_reconstructed) ** 2).mean().item()
    mean_scale = scale_w.mean().item()

    stats = QuantizationStats(
        original_nonzero=original_nonzero,
        quantized_nonzero=quantized_nonzero,
        sparsity=sparsity,
        scale=mean_scale,
        mse=mse
    )

    return w_ternary, scale_w, stats


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
    target_sparsity: float = 0.33,
    **kwargs  # Accept extra args like calibration_activations
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


def salient_channel_reorder(
    weight: torch.Tensor,
    act_scales: Optional[torch.Tensor] = None,
    salient_percentile: float = 1.0,
    verbose: bool = False
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """
    Salient channel Sparse Reordering (SSR) from PT²-LLM paper.

    Identifies salient (outlier) channels based on activation magnitude
    and reorders them to allow separate handling during quantization.

    Args:
        weight: [N, K] weight matrix
        act_scales: Per-channel activation scales (if None, use weight magnitude)
        salient_percentile: Top percentile of channels to mark as salient
        verbose: Print debug info

    Returns:
        reorder_indices: Indices to reorder columns (salient first)
        inverse_indices: Indices to restore original order
        salient_mask: Boolean mask marking salient channels
    """
    N, K = weight.shape

    # Compute channel importance
    if act_scales is not None:
        importance = act_scales
    else:
        # Use column-wise weight magnitude as proxy
        importance = weight.abs().mean(dim=0)

    # Find threshold for salient channels
    num_salient = max(1, int(K * salient_percentile / 100))
    threshold = torch.topk(importance, num_salient).values[-1]

    # Create salient mask
    salient_mask = importance >= threshold

    # Create reorder indices: salient channels first, then rest
    salient_indices = torch.where(salient_mask)[0]
    non_salient_indices = torch.where(~salient_mask)[0]
    reorder_indices = torch.cat([salient_indices, non_salient_indices])

    # Create inverse indices to restore original order
    inverse_indices = torch.zeros(K, dtype=torch.long)
    for new_idx, old_idx in enumerate(reorder_indices):
        inverse_indices[old_idx] = new_idx

    if verbose:
        print(f"  SSR: {salient_mask.sum().item()} salient channels ({salient_percentile:.1f}%)")
        print(f"  Salient importance: {importance[salient_mask].mean():.4f}")
        print(f"  Non-salient importance: {importance[~salient_mask].mean():.4f}")

    return reorder_indices, inverse_indices, salient_mask


def quantize_to_ternary_pt2_with_ssr(
    weight: torch.Tensor,
    calibration_activations: Optional[list] = None,
    max_iters: int = 10,
    salient_percentile: float = 1.0,
    salient_bits: int = 8,
    verbose: bool = False
) -> Tuple[torch.Tensor, float, QuantizationStats, Optional[dict]]:
    """
    PT²-LLM with SSR: Handles salient channels separately.

    Salient channels (outliers) can cause significant quantization error.
    SSR keeps these in higher precision or handles them separately.

    Args:
        weight: [N, K] weight matrix
        calibration_activations: Optional activation data for AGA
        max_iters: ITF iterations
        salient_percentile: Percentile of channels to treat as salient
        salient_bits: Bit-width for salient channels (8 = int8, -1 = keep fp16)
        verbose: Print progress

    Returns:
        w_ternary: Ternary weights (with salient channels handled specially)
        scale: Average scale
        stats: Quantization statistics
        ssr_info: Dict with SSR metadata (salient weights, indices, etc.)
    """
    N, K = weight.shape
    W = weight.float()

    # Step 1: Compute activation scales if we have calibration data
    if calibration_activations is not None and len(calibration_activations) > 0:
        act_scales = torch.zeros(K, device=W.device, dtype=W.dtype)
        num_samples = 0
        for X in calibration_activations:
            act_scales += X.float().abs().mean(dim=0)
            num_samples += 1
        if num_samples > 0:
            act_scales = act_scales / num_samples
    else:
        act_scales = None

    # Step 2: Identify and reorder salient channels
    reorder_idx, inverse_idx, salient_mask = salient_channel_reorder(
        W, act_scales, salient_percentile, verbose
    )

    num_salient = salient_mask.sum().item()

    # Step 3: Reorder weight columns
    W_reordered = W[:, reorder_idx]

    # Step 4: Split into salient and non-salient
    W_salient = W_reordered[:, :num_salient]  # [N, num_salient]
    W_nonsalient = W_reordered[:, num_salient:]  # [N, K - num_salient]

    # Step 5: Quantize non-salient channels with ITF
    if W_nonsalient.shape[1] > 0:
        T_nonsalient, alpha_nonsalient, mu_nonsalient = iterative_ternary_fitting(
            W_nonsalient, max_iters, verbose
        )

        # Apply AGA if we have calibration data
        if calibration_activations is not None and len(calibration_activations) > 0:
            # Reorder activations too
            cal_reordered = [X[:, reorder_idx][:, num_salient:] for X in calibration_activations]
            alpha_nonsalient, mu_nonsalient = activation_aware_alignment(
                W_nonsalient, T_nonsalient, cal_reordered, verbose
            )
    else:
        T_nonsalient = torch.zeros(N, 0, dtype=torch.int8)
        alpha_nonsalient = torch.zeros(N)
        mu_nonsalient = torch.zeros(N)

    # Step 6: Handle salient channels (keep higher precision)
    if num_salient > 0:
        if salient_bits == 8:
            # Quantize to int8 (better than ternary)
            max_abs = W_salient.abs().max(dim=1, keepdim=True).values.clamp(min=1e-10)
            W_salient_int8 = (W_salient / max_abs * 127).round().clamp(-127, 127).to(torch.int8)
            salient_scale = max_abs.squeeze(1)
        else:
            # Keep as fp16
            W_salient_int8 = None
            salient_scale = None

    # Step 7: Combine ternary weights (salient channels get 0 in ternary)
    # For our kernel, we'll convert salient to their "best ternary approximation"
    # but store actual values in ssr_info for hybrid inference
    if num_salient > 0:
        # Approximate salient with ternary for pure-ternary inference
        T_salient = ((W_salient / W_salient.abs().mean(dim=1, keepdim=True).clamp(min=1e-10))
                     .round().clamp(-1, 1).to(torch.int8))
    else:
        T_salient = torch.zeros(N, 0, dtype=torch.int8)

    # Step 8: Reassemble in reordered space, then restore original order
    T_reordered = torch.cat([T_salient, T_nonsalient], dim=1)
    T = T_reordered[:, inverse_idx]

    # Compute combined scale (for ternary-only path)
    avg_scale = alpha_nonsalient.mean().item() if W_nonsalient.shape[1] > 0 else 1.0

    # Compute reconstruction for statistics
    W_recon = T.float() * avg_scale

    # Statistics
    original_nonzero = (weight != 0).sum().item()
    quantized_nonzero = (T != 0).sum().item()
    sparsity = 1.0 - (quantized_nonzero / weight.numel())
    mse = ((W - W_recon) ** 2).mean().item()

    stats = QuantizationStats(
        original_nonzero=original_nonzero,
        quantized_nonzero=quantized_nonzero,
        sparsity=sparsity,
        scale=avg_scale,
        mse=mse
    )

    # SSR info for hybrid inference
    ssr_info = {
        'num_salient': num_salient,
        'salient_indices': reorder_idx[:num_salient].clone() if num_salient > 0 else None,
        'salient_weights_int8': W_salient_int8 if num_salient > 0 and salient_bits == 8 else None,
        'salient_scale': salient_scale if num_salient > 0 and salient_bits == 8 else None,
        'salient_weights_fp16': W_salient.half() if num_salient > 0 and salient_bits != 8 else None,
        'reorder_indices': reorder_idx,
        'inverse_indices': inverse_idx,
    }

    return T, avg_scale, stats, ssr_info


def collect_calibration_data(
    model: nn.Module,
    tokenizer,
    num_samples: int = 32,
    max_length: int = 512,
    dataset_name: str = 'wikitext'
) -> Dict[str, list]:
    """
    Collect calibration activations by running samples through the model.

    Args:
        model: HuggingFace model (before quantization)
        tokenizer: Tokenizer for the model
        num_samples: Number of calibration samples
        max_length: Maximum sequence length
        dataset_name: Dataset to use for calibration

    Returns:
        Dict mapping layer names to lists of activation tensors
    """
    try:
        from datasets import load_dataset
    except ImportError:
        print("  [!] datasets not installed, skipping calibration")
        return {}

    # Load calibration dataset
    if dataset_name == 'wikitext':
        try:
            dataset = load_dataset('wikitext', 'wikitext-2-raw-v1', split='train')
        except:
            print("  [!] Failed to load WikiText, using random data")
            return {}
    else:
        return {}

    calibration_data = {}
    hooks = []

    def make_hook(name):
        def hook(module, input, output):
            if name not in calibration_data:
                calibration_data[name] = []
            # Store input activations (what goes INTO the linear layer)
            inp = input[0].detach()
            if inp.dim() == 3:
                inp = inp.view(-1, inp.shape[-1])
            calibration_data[name].append(inp[:64])  # Limit to 64 tokens per sample
        return hook

    # Register hooks on all linear layers
    for name, module in model.named_modules():
        if isinstance(module, nn.Linear):
            hooks.append(module.register_forward_hook(make_hook(name)))

    # Run calibration samples
    model.eval()
    samples_collected = 0

    print(f"  Collecting calibration data ({num_samples} samples)...")

    with torch.no_grad():
        for i, sample in enumerate(dataset):
            if samples_collected >= num_samples:
                break

            text = sample['text']
            if not text.strip() or len(text) < 50:
                continue

            try:
                encodings = tokenizer(
                    text,
                    return_tensors='pt',
                    truncation=True,
                    max_length=max_length
                )
                input_ids = encodings['input_ids']

                if input_ids.shape[1] < 10:
                    continue

                _ = model(input_ids)
                samples_collected += 1

                if samples_collected % 10 == 0:
                    print(f"    Collected {samples_collected}/{num_samples} samples")

            except Exception as e:
                continue

    # Remove hooks
    for hook in hooks:
        hook.remove()

    print(f"  Collected activations for {len(calibration_data)} layers")

    return calibration_data


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


def _hadamard_matrix(n: int) -> torch.Tensor:
    """
    Generate a Hadamard matrix of size n x n using Sylvester's construction.
    n must be a power of 2.
    """
    if n == 1:
        return torch.tensor([[1.0]])

    # Check if n is a power of 2
    if n & (n - 1) != 0:
        raise ValueError(f"Hadamard matrix size must be a power of 2, got {n}")

    # Recursive Sylvester construction
    H_half = _hadamard_matrix(n // 2)
    H = torch.cat([
        torch.cat([H_half, H_half], dim=1),
        torch.cat([H_half, -H_half], dim=1)
    ], dim=0)
    return H


def get_randomized_hadamard_transform(
    dim: int,
    device: torch.device,
    dtype: torch.dtype
) -> torch.Tensor:
    """
    Generate a randomized Hadamard transform matrix U of size [dim, dim].
    U = D1 * H * D2 * H * D3, where H is Hadamard and Di are random diagonal matrices.
    This creates a random orthogonal matrix.
    """
    # Find the next power of 2 >= dim
    n = 1
    while n < dim:
        n *= 2

    # Generate Hadamard matrix (pure PyTorch, no scipy needed)
    H_full = _hadamard_matrix(n)
    # Normalize
    H_full = H_full / (n ** 0.5)
    # Truncate to dim x dim if needed
    H = H_full[:dim, :dim].to(device=device, dtype=dtype)

    # Create random diagonal matrices (random signs)
    D1 = torch.diag((torch.randint(0, 2, (dim,), device=device).float() * 2 - 1).to(dtype))
    D2 = torch.diag((torch.randint(0, 2, (dim,), device=device).float() * 2 - 1).to(dtype))

    # Combine to form the orthogonal matrix
    U = D1 @ H @ D2 @ H
    return U


def quantize_to_ternary_quip(
    weight: torch.Tensor,
    **kwargs
) -> Tuple[torch.Tensor, float, QuantizationStats, Dict]:
    """
    Simplified QuIP-style ternary quantization using a one-sided transform.

    Applies an incoherence transform to the columns of the weights (W' = W @ U.T)
    before quantizing them with a standard high-quality method (PT²).

    The transform matrix U needs to be applied to the output during inference.

    Args:
        weight: Float tensor to quantize [N, K]
        **kwargs: Arguments for the underlying quantizer (e.g., pt2)

    Returns:
        w_ternary: Ternary tensor of the *transformed* weights
        scale: Scaling factor for the *transformed* weights
        stats: Quantization statistics
        quip_info: Dictionary containing the transform matrix U
    """
    N, K = weight.shape
    W = weight.float()

    # We transform the columns of the weight matrix
    U = get_randomized_hadamard_transform(K, W.device, W.dtype)
    W_transformed = W @ U.T
    
    if kwargs.get('verbose', False):
        print(f"  Applied QuIP one-sided incoherence transform.")

    # Quantize the transformed weights using a robust method like PT2
    w_ternary, scale, stats = quantize_to_ternary_pt2(W_transformed, **kwargs)

    quip_info = {
        'U': U,
        'original_shape': (N, K)
    }

    # The returned ternary weight is for the *transformed* matrix
    return w_ternary, scale, stats, quip_info


# ----------------------------------------------------------------------------
# Learned / experimental variants
# ----------------------------------------------------------------------------

def get_butterfly_transform(
    dim: int,
    device: torch.device,
    dtype: torch.dtype
) -> torch.Tensor:
    """
    Construct an orthogonal transform using a QR factorization of a random
    Gaussian matrix (cheap surrogate for learnable butterfly/QuaRot).
    """
    rnd = torch.randn(dim, dim, device=device, dtype=dtype)
    q, _ = torch.linalg.qr(rnd, mode='reduced')
    return q


def quantize_to_ternary_butterfly(
    weight: torch.Tensor,
    **kwargs
) -> Tuple[torch.Tensor, float, QuantizationStats, Dict]:
    """
    Butterfly/QuaRot-style ternary quantization:
    - Build an orthogonal transform U (QR on random matrix)
    - Transform columns: W' = W @ U.T
    - Quantize W' with PT² backend
    """
    N, K = weight.shape
    U = get_butterfly_transform(K, weight.device, weight.dtype)
    W_rot = weight.float() @ U.T

    if kwargs.get('verbose', False):
        print("  Applying butterfly/QR orthogonal transform before PT².")

    w_ternary, scale, stats = quantize_to_ternary_pt2(W_rot, **kwargs)
    info = {'U': U, 'original_shape': (N, K)}
    return w_ternary, scale, stats, info


def quantize_to_ternary_cqe_plus(
    weight: torch.Tensor,
    sv_clip_percentile: float = 99.0,
    **kwargs
) -> Tuple[torch.Tensor, float, QuantizationStats]:
    """
    Cross-layer singular-value preconditioning (CQE+/SVD clamp surrogate):
    - SVD on weight, clamp singular values to a percentile to tame outliers
    - Recompose and quantize with PT²
    """
    W = weight.float()
    U, S, Vh = torch.linalg.svd(W, full_matrices=False)
    if sv_clip_percentile <= 0 or sv_clip_percentile > 100:
        sv_clip = S.max()
    else:
        sv_clip = torch.quantile(S, sv_clip_percentile / 100.0)
    S_clamped = torch.clamp(S, max=sv_clip)
    W_pre = (U * S_clamped.unsqueeze(0)) @ Vh

    if kwargs.get('verbose', False):
        print(f"  SVD clamp: clip@{sv_clip_percentile:.1f}pct -> {sv_clip.item():.4f}")

    return quantize_to_ternary_pt2(W_pre, **kwargs)


def quantize_to_ternary_pt2_v2(
    weight: torch.Tensor,
    calibration_activations: Optional[list] = None,
    max_iters: int = 10,
    verbose: bool = False
) -> Tuple[torch.Tensor, float, QuantizationStats, Dict]:
    """
    PT² with asymmetric per-row scales and lightweight bias correction.
    - Start from PT² ternary (T, alpha, mu)
    - Derive separate positive/negative scales per row (a,b)
    - Return average scale for kernel; stash a,b for reconstruction
    """
    T, alpha, mu = iterative_ternary_fitting(weight, max_iters, verbose)

    T_float = T.float()
    W = weight.float()
    pos_mask = T_float > 0
    neg_mask = T_float < 0

    # Per-row asymmetric scales
    pos_scale = (W * pos_mask).sum(dim=1) / (pos_mask.sum(dim=1).clamp(min=1))
    neg_scale = (W * neg_mask).sum(dim=1).abs() / (neg_mask.sum(dim=1).clamp(min=1))
    asym_scale = (pos_scale + neg_scale).clamp(min=1e-8) / 2.0

    # Reconstruct with asym scales and bias mu
    W_recon = T_float * asym_scale.unsqueeze(1) + mu.unsqueeze(1)

    if calibration_activations is not None and len(calibration_activations) > 0:
        # Optional AGA refinement
        alpha_refined, mu_refined = activation_aware_alignment(W, T, calibration_activations, verbose)
        W_recon = T_float * alpha_refined.unsqueeze(1) + mu_refined.unsqueeze(1)
        asym_scale = alpha_refined

    original_nonzero = (weight != 0).sum().item()
    quantized_nonzero = (T != 0).sum().item()
    sparsity = 1.0 - (quantized_nonzero / weight.numel())
    mse = ((W - W_recon) ** 2).mean().item()

    avg_scale = asym_scale.mean().item()
    stats = QuantizationStats(
        original_nonzero=original_nonzero,
        quantized_nonzero=quantized_nonzero,
        sparsity=sparsity,
        scale=avg_scale,
        mse=mse
    )

    extra = {
        'asym_pos_scale': pos_scale,
        'asym_neg_scale': neg_scale,
        'bias_mu': mu,
    }

    return T, avg_scale, stats, extra


def quantize_to_ternary_awq_v2(
    weight: torch.Tensor,
    group_size: int = 64,
    target_sparsity: float = 0.33,
    **kwargs
) -> Tuple[torch.Tensor, float, QuantizationStats]:
    """
    AWQ-style ternary with group-wise (or head-wise) scaling.
    Columns are grouped (default 64); important groups get lower thresholds.
    """
    N, K = weight.shape
    num_groups = max(1, K // group_size)
    # Group importance by mean magnitude
    imp = weight.abs().view(N, num_groups, -1).mean(dim=2).mean(dim=0)
    # Normalize importance
    imp = imp / (imp.max() + 1e-8)

    base_threshold = torch.quantile(weight.abs().flatten(), target_sparsity).item()
    thresholds = base_threshold / (0.5 + imp)  # important groups get lower threshold

    w_ternary = torch.zeros_like(weight, dtype=torch.int8)
    for g in range(num_groups):
        start = g * group_size
        end = min((g + 1) * group_size, K)
        thr = thresholds[g].item()
        block = weight[:, start:end]
        w_ternary[:, start:end][block > thr] = 1
        w_ternary[:, start:end][block < -thr] = -1

    mask = w_ternary != 0
    scale = weight.abs()[mask].mean().item() if mask.sum() > 0 else 1.0

    w_reconstructed = w_ternary.float() * scale
    mse = ((weight - w_reconstructed) ** 2).mean().item()
    stats = QuantizationStats(
        original_nonzero=(weight != 0).sum().item(),
        quantized_nonzero=mask.sum().item(),
        sparsity=1.0 - (mask.sum().item() / weight.numel()),
        scale=scale,
        mse=mse
    )
    return w_ternary, scale, stats


def quantize_to_ternary_smoothquant_v2(
    weight: torch.Tensor,
    act_scales: Optional[torch.Tensor] = None,
    threshold_percentile: float = 70.0,
    stochastic_round: bool = False,
    **kwargs
) -> Tuple[torch.Tensor, float, QuantizationStats]:
    """
    SmoothQuant variant with percentile thresholding and optional stochastic rounding.
    """
    N, K = weight.shape
    if act_scales is None:
        act_scales = weight.abs().mean(dim=0) + 1e-10
    w_scales = weight.abs().max(dim=0).values + 1e-10
    smooth_scales = (act_scales ** 0.5) / (w_scales ** 0.5)
    w_smooth = weight * smooth_scales.unsqueeze(0)

    abs_weight = w_smooth.abs()
    threshold = torch.quantile(abs_weight.flatten(), threshold_percentile / 100.0).item()

    if stochastic_round:
        noise = torch.empty_like(w_smooth).uniform_(-0.5, 0.5) * threshold * 0.05
        w_smooth = w_smooth + noise

    mask = abs_weight > threshold
    scale = abs_weight[mask].mean().item() if mask.sum() > 0 else 1.0

    w_ternary = torch.zeros_like(weight, dtype=torch.int8)
    w_ternary[w_smooth > threshold] = 1
    w_ternary[w_smooth < -threshold] = -1

    w_reconstructed = (w_ternary.float() * scale) / smooth_scales.unsqueeze(0)
    mse = ((weight - w_reconstructed) ** 2).mean().item()

    stats = QuantizationStats(
        original_nonzero=(weight != 0).sum().item(),
        quantized_nonzero=(w_ternary != 0).sum().item(),
        sparsity=1.0 - ((w_ternary != 0).sum().item() / weight.numel()),
        scale=scale,
        mse=mse
    )
    return w_ternary, scale, stats


def quantize_to_ternary_pt2_ssr_v2(
    weight: torch.Tensor,
    salient_percentile: float = 0.5,
    salient_bits: int = 8,
    **kwargs
) -> Tuple[torch.Tensor, float, QuantizationStats, Optional[dict]]:
    """
    Enhanced SSR: smaller salient set + optional int8 stash for salient columns.
    """
    T, scale, stats, ssr_info = quantize_to_ternary_pt2_with_ssr(
        weight,
        salient_percentile=salient_percentile,
        salient_bits=salient_bits,
        **kwargs
    )
    if ssr_info and ssr_info.get('salient_weights_int8') is None and ssr_info.get('num_salient', 0) > 0:
        # Build int8 copy for salient columns
        W_sal = weight[:, ssr_info['salient_indices']].float()
        max_abs = W_sal.abs().max(dim=1, keepdim=True).values.clamp(min=1e-10)
        W_int8 = (W_sal / max_abs * 127).round().clamp(-127, 127).to(torch.int8)
        ssr_info['salient_weights_int8'] = W_int8
        ssr_info['salient_scale'] = max_abs.squeeze(1)
    return T, scale, stats, ssr_info

import torch.optim as optim


def quantize_to_ternary_omniquant(
    weight: torch.Tensor,
    calibration_activations: Optional[list] = None,
    **kwargs
) -> Tuple[torch.Tensor, float, QuantizationStats, Optional[Dict]]:
    """
    OmniQuant-style ternary quantization.

    Uses a brief optimization phase with calibration data to learn:
    1. LET: A channel-wise scaling factor to smooth activation outliers.
    2. LWC: A clipping threshold to tame weight outliers.

    These learned parameters are then used to pre-process the weights before
    quantizing with a robust backend like PT².

    Args:
        weight: Float tensor to quantize [N, K]
        calibration_activations: List of activation tensors for optimization
        **kwargs: Arguments for the backend quantizer (e.g., pt2)

    Returns:
        w_ternary: Ternary tensor
        scale: Scaling factor
        stats: Quantization statistics
        extra_info: None (OmniQuant is self-contained)
    """
    N, K = weight.shape
    W = weight.float()

    if calibration_activations is None or len(calibration_activations) == 0:
        print("  [!] OmniQuant requires calibration data. Falling back to pt2.")
        w_ternary, scale, stats = quantize_to_ternary_pt2(weight, **kwargs)
        return w_ternary, scale, stats, None

    # Concatenate calibration data into a single tensor
    X_calib = torch.cat([x.to(W.device) for x in calibration_activations], dim=0)

    # --- Learnable Parameters ---
    # LET: Learnable Equivalent Transformation scales (similar to SmoothQuant)
    s = torch.ones(K, device=W.device, requires_grad=True)
    # LWC: Learnable Weight Clipping threshold
    clip_val = torch.tensor([W.abs().max()], device=W.device, requires_grad=True)

    optimizer = optim.AdamW([s, clip_val], lr=1e-3, weight_decay=1e-5)
    num_iters = 50
    
    if kwargs.get('verbose', False):
        print(f"  Running OmniQuant optimization for {num_iters} iterations...")

    for i in range(num_iters):
        optimizer.zero_grad()

        # Apply LET to weights
        W_transformed = W * s.unsqueeze(0)
        
        # Apply LWC
        W_clipped = W_transformed.clamp(-clip_val, clip_val)
        
        # Quantize the processed weight (using a simple, fast method for the loop)
        w_q, scale, _ = quantize_to_ternary_optimal(W_clipped, target_sparsity=0.5)

        # Reconstruct the weight
        W_recon = w_q.float() * scale
        
        # Inversely apply LET to the reconstructed weight for loss calculation
        W_final_recon = W_recon / s.unsqueeze(0)

        # Calculate reconstruction loss on a sample of calibration data
        sample_indices = torch.randperm(X_calib.shape[0])[:32]
        X_sample = X_calib[sample_indices]
        
        Y_orig = X_sample @ W.T
        Y_quant = X_sample @ W_final_recon.T
        
        loss = ((Y_orig - Y_quant)**2).mean()
        
        loss.backward()
        optimizer.step()

        # Ensure clip_val remains positive
        with torch.no_grad():
            clip_val.clamp_(min=1e-6)

        if kwargs.get('verbose', False) and (i % 10 == 0):
            print(f"    Iter {i}: Loss={loss.item():.6e}, Clip={clip_val.item():.4f}, s_mean={s.mean().item():.4f}")

    # --- Final Quantization ---
    # Apply the learned transforms
    W_transformed = W * s.detach().unsqueeze(0)
    W_clipped = W_transformed.clamp(-clip_val.detach(), clip_val.detach())

    if kwargs.get('verbose', False):
        print(f"  OmniQuant optimization finished.")
        print(f"  Final Clip Value: {clip_val.item():.4f}")
        print(f"  Final Scale `s` mean: {s.mean().item():.4f} (std: {s.std().item():.4f})")

    # Use the best quantizer on the processed weights
    w_ternary, scale, stats = quantize_to_ternary_pt2(W_clipped, **kwargs)
    
    # Recalculate MSE against original weight, accounting for LET
    W_recon_final = (w_ternary.float() * scale) / s.detach().unsqueeze(0)
    final_mse = ((W - W_recon_final) ** 2).mean().item()
    stats.mse = final_mse

    return w_ternary, scale, stats, None


# ============================================================================
# New Advanced Quantization Methods
# ============================================================================

def quantize_to_2bit(
    weight: torch.Tensor,
    levels: Tuple[float, float, float, float] = (-1.5, -0.5, 0.5, 1.5),
) -> Tuple[torch.Tensor, float, QuantizationStats]:
    """
    Quantize weight tensor to 2-bit with 4 levels.

    Default levels are {-1.5, -0.5, +0.5, +1.5} which gives better granularity
    than ternary {-1, 0, +1}.

    The quantized weights are stored as int8 with values {-2, -1, +1, +2}
    (we avoid 0 to distinguish from ternary).

    Args:
        weight: Float tensor to quantize
        levels: The 4 quantization levels (should be symmetric around 0)

    Returns:
        w_2bit: 2-bit quantized tensor as int8 {-2, -1, +1, +2}
        scale: Scaling factor
        stats: Quantization statistics
    """
    # Compute scale based on weight magnitude
    abs_mean = weight.abs().mean()
    if abs_mean < 1e-10:
        return torch.zeros_like(weight, dtype=torch.int8), 1.0, QuantizationStats(
            original_nonzero=0, quantized_nonzero=0, sparsity=1.0, scale=1.0, mse=0.0
        )

    # Scale to map weight distribution to quantization levels
    # We use the 75th percentile to set the scale
    p75 = torch.quantile(weight.abs().flatten(), 0.75).item()
    scale = p75 / 1.0  # Map 75th percentile to middle levels (±0.5, ±1.5)
    if scale < 1e-10:
        scale = abs_mean.item()

    # Normalize weights
    w_normalized = weight / scale

    # Quantize to 4 levels: -1.5, -0.5, +0.5, +1.5 -> stored as -2, -1, +1, +2
    w_2bit = torch.zeros_like(weight, dtype=torch.int8)

    # Thresholds at -1.0, 0.0, +1.0
    w_2bit[w_normalized <= -1.0] = -2  # maps to -1.5
    w_2bit[(w_normalized > -1.0) & (w_normalized <= 0.0)] = -1  # maps to -0.5
    w_2bit[(w_normalized > 0.0) & (w_normalized < 1.0)] = 1  # maps to +0.5
    w_2bit[w_normalized >= 1.0] = 2  # maps to +1.5

    # For reconstruction: -2 -> -1.5, -1 -> -0.5, +1 -> +0.5, +2 -> +1.5
    # w_reconstructed = (w_2bit.float() - 0.5 * sign(w_2bit)) * scale
    # Simpler: w_2bit * 0.5 * (1 + 0.5/abs(w_2bit)) but that's complex
    # Just use: (-2 -> -1.5, -1 -> -0.5, 1 -> 0.5, 2 -> 1.5)
    level_map = {-2: -1.5, -1: -0.5, 1: 0.5, 2: 1.5, 0: 0.0}
    w_reconstructed = torch.zeros_like(weight, dtype=torch.float32)
    for k, v in level_map.items():
        w_reconstructed[w_2bit == k] = v * scale

    # Statistics
    original_nonzero = (weight != 0).sum().item()
    quantized_nonzero = (w_2bit != 0).sum().item()  # All are nonzero for 2-bit
    sparsity = 1.0 - (quantized_nonzero / weight.numel())
    mse = ((weight - w_reconstructed) ** 2).mean().item()

    stats = QuantizationStats(
        original_nonzero=original_nonzero,
        quantized_nonzero=quantized_nonzero,
        sparsity=sparsity,
        scale=scale,
        mse=mse
    )

    return w_2bit, scale, stats


def quantize_to_ternary_gptq_calibrated(
    weight: torch.Tensor,
    calibration_activations: Optional[list] = None,
    blocksize: int = 128,
    target_sparsity: float = 0.33,
    damping: float = 0.01,
    verbose: bool = False
) -> Tuple[torch.Tensor, float, QuantizationStats]:
    """
    GPTQ-style ternary quantization with REAL Hessian from calibration data.

    This is the proper GPTQ algorithm that uses X.T @ X from calibration
    data as the Hessian, not just weight magnitude.

    Args:
        weight: Float tensor to quantize [N, K]
        calibration_activations: List of activation tensors for Hessian
        blocksize: Process this many columns at a time
        target_sparsity: Target fraction of zeros
        damping: Damping factor for Hessian inverse
        verbose: Print debug info

    Returns:
        w_ternary: Ternary tensor {-1, 0, +1}
        scale: Scaling factor
        stats: Quantization statistics
    """
    N, K = weight.shape
    w = weight.clone().float()

    # Compute Hessian from calibration data: H = X.T @ X
    if calibration_activations is not None and len(calibration_activations) > 0:
        H = torch.zeros(K, K, dtype=torch.float32, device=weight.device)
        nsamples = 0
        for X in calibration_activations:
            if X.dim() == 3:
                X = X.view(-1, X.shape[-1])
            X = X.float()
            H += X.T @ X
            nsamples += X.shape[0]
        H /= nsamples

        # Add damping for numerical stability
        damp = damping * torch.diag(H).mean()
        H += damp * torch.eye(K, device=H.device)

        if verbose:
            print(f"  GPTQ: Computed Hessian from {len(calibration_activations)} samples")
    else:
        # Fallback: use diagonal approximation from weight magnitude
        H = torch.diag((w ** 2).sum(dim=0) + 1e-10)
        if verbose:
            print(f"  GPTQ: No calibration data, using weight magnitude proxy")

    # Find threshold and scale
    abs_weight = w.abs()
    threshold = torch.quantile(abs_weight.flatten(), target_sparsity).item()
    mask = abs_weight > threshold
    scale = abs_weight[mask].mean().item() if mask.sum() > 0 else 1.0

    # Initialize output
    w_ternary = torch.zeros_like(w, dtype=torch.int8)

    # Process column by column with Hessian-based error compensation
    for start in range(0, K, blocksize):
        end = min(start + blocksize, K)

        # Get block of Hessian
        H_block = H[start:end, start:end]

        # Cholesky decomposition for stable inverse
        try:
            L = torch.linalg.cholesky(H_block)
            H_inv_block = torch.cholesky_inverse(L)
        except:
            # Fallback to pseudo-inverse
            H_inv_block = torch.linalg.pinv(H_block)

        for j in range(start, end):
            col = w[:, j]

            # Quantize this column
            q = torch.zeros(N, dtype=torch.int8, device=weight.device)
            q[col > threshold] = 1
            q[col < -threshold] = -1
            w_ternary[:, j] = q

            # Compute quantization error
            q_float = q.float() * scale
            error = col - q_float

            # GPTQ update: distribute error to remaining columns
            # w[:, j+1:end] += error.unsqueeze(1) @ H_inv[j, j+1:end].unsqueeze(0) / H_inv[j, j]
            if j < end - 1:
                j_local = j - start
                h_inv_jj = H_inv_block[j_local, j_local]
                if h_inv_jj > 1e-10:
                    h_inv_row = H_inv_block[j_local, j_local+1:]
                    update = error.unsqueeze(1) @ h_inv_row.unsqueeze(0) / h_inv_jj
                    w[:, j+1:end] -= update

    # Compute final statistics
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

    if verbose:
        print(f"  GPTQ calibrated: MSE={mse:.2e}, sparsity={sparsity:.1%}")

    return w_ternary, scale, stats


def compute_lorc(
    W: torch.Tensor,
    W_q_recon: torch.Tensor,
    rank: int = 4
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Compute Low-Rank Compensation (LoRC) matrices.

    Approximates the quantization error matrix E = W - W_q_recon with
    two low-rank matrices, L and R, such that E ≈ L @ R.

    For the forward pass, we need: y_corrected = y + x @ E.T
    where x is [M, K], E is [N, K], and y is [M, N].

    So we compute: x @ E.T = x @ (L @ R).T = x @ R.T @ L.T

    We return L_out = R.T [K, rank] and R_out = L.T [rank, N]
    so forward pass computes: (x @ L_out) @ R_out = x @ R.T @ L.T = x @ E.T

    Args:
        W: Original full-precision weight matrix [N, K].
        W_q_recon: Reconstructed quantized weight matrix [N, K].
        rank: The rank for the low-rank approximation.

    Returns:
        L_out: [K, rank] matrix for forward pass
        R_out: [rank, N] matrix for forward pass
    """
    error = W - W_q_recon  # [N, K]

    # Using SVD to find the best low-rank approximation
    # error = U @ S @ Vh where U is [N, N], S is [min(N,K)], Vh is [K, K]
    U, S, Vh = torch.linalg.svd(error, full_matrices=False)

    # Keep the top 'rank' singular values
    U_r = U[:, :rank]      # [N, rank]
    S_r = torch.diag(S[:rank])  # [rank, rank]
    Vh_r = Vh[:rank, :]    # [rank, K]

    # Original decomposition: E ≈ (U_r @ S_r) @ Vh_r = L @ R
    # L = U_r @ S_r  [N, rank]
    # R = Vh_r       [rank, K]

    # For forward: x @ E.T = x @ R.T @ L.T
    # So L_out = R.T = Vh_r.T  [K, rank]
    # And R_out = L.T = (U_r @ S_r).T = S_r @ U_r.T  [rank, N]

    L_out = Vh_r.T  # [K, rank]
    R_out = S_r @ U_r.T  # [rank, N]

    return L_out, R_out


def prepare_ternary_weight_ptq(
    weight: torch.Tensor,
    method: str = 'absmean',
    use_lorc: bool = False,
    lorc_rank: int = 4,
    **kwargs
) -> Tuple[torch.Tensor, torch.Tensor, float, QuantizationStats, Optional[Dict]]:
    """
    Prepare weight for our ternary kernels using PTQ.

    Converts weight to the format expected by our VNNI/Graviton kernels:
    - w_int8: [N, K_padded] int8 tensor with values {-1, 0, +1}
    - w_sum: [N] int32 tensor with row sums
    - scale: float scaling factor

    Args:
        weight: [N, K] float tensor (out_features, in_features)
        method: Quantization method to use.
        use_lorc: If True, compute and return LoRC matrices.
        lorc_rank: The rank for the LoRC approximation.
        **kwargs: Additional arguments for the quantization method.

    Returns:
        w_int8: Padded int8 ternary weights.
        w_sum: Row sums for bias correction.
        scale: Scaling factor.
        stats: Quantization statistics.
        extra_info: Dictionary with method-specific data (e.g., for QuIP, SSR, or LoRC).
    """
    N, K = weight.shape
    
    extra_info = {}
    w_ternary, scale, stats = None, None, None

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
    elif method == 'pt2_calibrated':
        w_ternary, scale, stats = quantize_to_ternary_pt2(weight, **kwargs)
    elif method == 'pt2_ssr':
        w_ternary, scale, stats, ssr_info = quantize_to_ternary_pt2_with_ssr(weight, **kwargs)
        if ssr_info: extra_info.update(ssr_info)
    elif method == 'quip':
        w_ternary, scale, stats, quip_info = quantize_to_ternary_quip(weight, **kwargs)
        if quip_info: extra_info.update(quip_info)
    elif method == 'omniquant':
        w_ternary, scale, stats, omni_info = quantize_to_ternary_omniquant(weight, **kwargs)
        if omni_info: extra_info.update(omni_info)
    elif method == '2bit':
        w_ternary, scale, stats = quantize_to_2bit(weight, **kwargs)
    elif method == 'gptq_calibrated':
        w_ternary, scale, stats = quantize_to_ternary_gptq_calibrated(weight, **kwargs)
    elif method == 'perchannel':
        # Per-channel scaling - scale is a tensor [N] not a float
        w_ternary, scale, stats = quantize_to_ternary_perchannel(weight, **kwargs)
        extra_info['scale_w'] = scale  # Store per-channel scales
        scale = scale.mean().item()  # Use mean for backward compat
    elif method == 'perchannel_hessian':
        # Per-channel with Hessian-weighted rounding
        w_ternary, scale, stats = quantize_to_ternary_perchannel_hessian(weight, **kwargs)
        extra_info['scale_w'] = scale  # Store per-channel scales
        scale = scale.mean().item()  # Use mean for backward compat
    elif method == 'pt2_v2':
        w_ternary, scale, stats, extra = quantize_to_ternary_pt2_v2(weight, **kwargs)
        extra_info.update(extra)
    elif method == 'quip_v2':
        w_ternary, scale, stats, quip_info = quantize_to_ternary_butterfly(weight, **kwargs)
        if quip_info:
            extra_info.update(quip_info)
    elif method == 'awq_v2':
        w_ternary, scale, stats = quantize_to_ternary_awq_v2(weight, **kwargs)
    elif method == 'smoothquant_v2':
        w_ternary, scale, stats = quantize_to_ternary_smoothquant_v2(weight, **kwargs)
    elif method == 'pt2_ssr_v2':
        w_ternary, scale, stats, ssr_info = quantize_to_ternary_pt2_ssr_v2(weight, **kwargs)
        if ssr_info: extra_info.update(ssr_info)
    elif method == 'butterfly':
        w_ternary, scale, stats, info = quantize_to_ternary_butterfly(weight, **kwargs)
        if info: extra_info.update(info)
    elif method == 'cqe_plus':
        w_ternary, scale, stats = quantize_to_ternary_cqe_plus(weight, **kwargs)
    else:
        raise ValueError(f"Unknown quantization method: {method}")

    # --- LoRC Calculation ---
    if use_lorc:
        # Reconstruct the quantized weight to compute the error
        # Note: This reconstruction needs to be accurate for the given method
        W_q_recon = (w_ternary.float() * scale)
        if method == 'omniquant':
            # OmniQuant includes LET, which needs to be inverted
            s = kwargs.get('let_s_inv') # Assume s is returned somehow
            if s is not None:
                W_q_recon = W_q_recon / s.unsqueeze(0)

        L, R = compute_lorc(weight, W_q_recon, rank=lorc_rank)
        extra_info['lorc_L'] = L
        extra_info['lorc_R'] = R
        if kwargs.get('verbose', False):
            print(f"  Computed LoRC matrices with rank={lorc_rank}")


    # Determine padded dimension from the ternary weight shape
    N_out, K_out = w_ternary.shape
    K_padded = ((K_out + 63) // 64) * 64

    # Pad to K_padded
    if K_padded > K_out:
        w_int8 = torch.zeros(N_out, K_padded, dtype=torch.int8)
        w_int8[:, :K_out] = w_ternary
    else:
        w_int8 = w_ternary.contiguous()

    # Compute row sums (for bias correction in kernel)
    w_sum = w_int8.sum(dim=1, dtype=torch.int32)

    return w_int8, w_sum, scale, stats, extra_info or None




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
        w_int8, w_sum, scale, stats, extra_info = prepare_ternary_weight_ptq(weight, method, **kwargs)

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
