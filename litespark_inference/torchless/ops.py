"""
Pure-numpy numerical primitives for the torchless BitNet runtime.

RMSNorm, RoPE, softmax, per-token activation quantization, ReLU-squared. All
operations keep output dtype float32; bf16 inputs are upcast on the fly.
"""

from __future__ import annotations

import numpy as np

from .quantize import bf16_u16_to_fp32


def _as_fp32(a: np.ndarray) -> np.ndarray:
    if a.dtype == np.uint16:
        return bf16_u16_to_fp32(a)
    if a.dtype == np.float32:
        return a
    return a.astype(np.float32)


def rmsnorm(x: np.ndarray, gamma: np.ndarray, eps: float = 1e-5) -> np.ndarray:
    """
    Root-mean-square norm:  x / sqrt(mean(x^2) + eps) * gamma

    x:     [..., D] float32
    gamma: [D] either bf16 (uint16) or float32
    """
    rms = np.sqrt((x * x).mean(axis=-1, keepdims=True) + eps)
    return (x / rms) * _as_fp32(gamma)


def softmax(x: np.ndarray, axis: int = -1) -> np.ndarray:
    m = x.max(axis=axis, keepdims=True)
    e = np.exp(x - m)
    return e / e.sum(axis=axis, keepdims=True)


def rope_tables(head_dim: int, max_pos: int, theta: float) -> tuple[np.ndarray, np.ndarray]:
    """
    Precompute cos/sin for rotary position embeddings.

    Returns: (cos [max_pos, D/2], sin [max_pos, D/2]) as float32.
    """
    half = head_dim // 2
    inv_freq = (1.0 / (theta ** (np.arange(0, head_dim, 2, dtype=np.float32) / head_dim)))
    positions = np.arange(max_pos, dtype=np.float32)[:, None]   # [T, 1]
    freqs = positions * inv_freq[None, :]                        # [T, D/2]
    return np.cos(freqs).astype(np.float32), np.sin(freqs).astype(np.float32)


def apply_rope(x: np.ndarray, cos_row: np.ndarray, sin_row: np.ndarray) -> np.ndarray:
    """
    Llama-style split-half RoPE:
        x = [x1, x2]  (halves along last dim)
        out = [x1*cos - x2*sin, x1*sin + x2*cos]

    x:       [..., D] float32
    cos/sin: [D/2] for one position
    """
    half = x.shape[-1] // 2
    x1 = x[..., :half]
    x2 = x[..., half:]
    out1 = x1 * cos_row - x2 * sin_row
    out2 = x1 * sin_row + x2 * cos_row
    return np.concatenate([out1, out2], axis=-1)


def quantize_activation_int8(x_fp32: np.ndarray) -> tuple[np.ndarray, float]:
    """
    Per-tensor absmax activation quantization for a 1-D vector (M=1 path).

    Returns:
        x_int8: [K] int8
        scale:  float (fp32 dequant factor; dequantized = x_int8 * scale)
    """
    assert x_fp32.ndim == 1
    abs_max = float(np.abs(x_fp32).max())
    scale = max(abs_max, 1e-5) / 127.0
    x_int8 = np.round(x_fp32 / scale).clip(-127, 127).astype(np.int8)
    return x_int8, scale


def relu2(x: np.ndarray) -> np.ndarray:
    """BitNet's squared ReLU activation: max(x, 0) ** 2."""
    y = np.maximum(x, 0.0)
    return y * y
