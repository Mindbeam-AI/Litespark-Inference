"""
BitNet b1.58 per-tensor absmean ternary quantization, numpy-only.

Mirrors models.py:prepare_ternary_weight:

    scale_inv = 1 / weight.abs().mean()
    w_ternary = round(weight * scale_inv).clip(-1, 1).to(int8)
    output    = matmul(x, w_ternary) * scale   # scale = absmean, scalar

Per-tensor (single scalar) scale -- NOT per-row -- matches the BitNet b1.58
convention and the shipped torch-backed loader.
"""

from __future__ import annotations

import numpy as np


def bf16_u16_to_fp32(a_u16: np.ndarray) -> np.ndarray:
    """Reinterpret bf16 stored as uint16 -> fp32 (bit-exact upcast)."""
    return (a_u16.astype(np.uint32) << 16).view(np.float32)


def quantize_ternary_absmean(
    w_bf16_u16: np.ndarray, row_block: int = 128
) -> tuple[np.ndarray, float]:
    """
    Quantize a bf16 (uint16) weight matrix [N, K] to ternary with a single
    per-tensor absmean scale.

    Returns:
        w_int8: [N, K] int8 in {-1, 0, +1}
        scale:  float (the absmean; callers multiply matmul output by this)

    Two-pass row-blocked to bound transient memory at ~row_block*K*4 bytes
    instead of the full N*K*4 fp32 matrix (matters for the large projections:
    a 6912x2560 layer is 67 MB fp32 vs 2.5 MB at row_block=128).
    """
    assert w_bf16_u16.dtype == np.uint16
    N, K = w_bf16_u16.shape
    total = N * K

    # Pass 1: accumulate abs sum over row blocks.
    abs_sum = 0.0
    for r0 in range(0, N, row_block):
        r1 = min(r0 + row_block, N)
        block_fp32 = bf16_u16_to_fp32(w_bf16_u16[r0:r1])
        abs_sum += float(np.abs(block_fp32).sum())
    absmean = abs_sum / total
    if absmean <= 0.0:
        absmean = 1.0
    scale_inv = 1.0 / absmean

    # Pass 2: quantize with the global scale.
    w_int8 = np.empty((N, K), dtype=np.int8)
    for r0 in range(0, N, row_block):
        r1 = min(r0 + row_block, N)
        block_fp32 = bf16_u16_to_fp32(w_bf16_u16[r0:r1])
        w_int8[r0:r1] = np.round(block_fp32 * scale_inv).clip(-1, 1).astype(np.int8)
    return w_int8, absmean
