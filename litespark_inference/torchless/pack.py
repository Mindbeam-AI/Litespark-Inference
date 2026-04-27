"""
2-bit packing for ternary weights (4 weights per byte).

Encoding matches kernels/x86_64/matmul_lut_standalone.cpp:22:
    (w0+1) | ((w1+1)<<2) | ((w2+1)<<4) | ((w3+1)<<6)
where w in {-1, 0, +1} -> {0, 1, 2}.

Any NEON port of that kernel can consume the same packed bytes directly.
"""

from __future__ import annotations

import numpy as np


def pack_ternary_4_per_byte(w_int8: np.ndarray) -> np.ndarray:
    """
    Pack [N, K] int8 in {-1, 0, +1} to [N, ceil(K/4)] uint8.

    If K is not a multiple of 4, zero-pad on the right. Implemented with
    in-place OR-ing into a single output buffer to keep the transient
    footprint down to ~N*K/4 bytes (one output + two small per-stride
    shifted intermediates) instead of ~4*N*K bytes from a chain of
    `|` / `<<` operations that each materialize a full-size temporary.
    """
    assert w_int8.dtype == np.int8
    N, K = w_int8.shape
    if K % 4:
        pad = 4 - (K % 4)
        w_int8 = np.concatenate([w_int8, np.zeros((N, pad), dtype=np.int8)], axis=1)
        K = w_int8.shape[1]
    Kb = K // 4
    # Reinterpret int8 bytes as uint8 (view, not copy). For int8 in {-1,0,+1}
    # the uint8 view is {255, 0, 1}; modular +1 yields {0, 1, 2} with no
    # upcast and no full-size intermediate copy.
    u = w_int8.view(np.uint8)
    out = np.empty((N, Kb), dtype=np.uint8)
    tmp = np.empty((N, Kb), dtype=np.uint8)
    # Stride across each 4-column stripe, add 1 (uint8 wrap), combine into out.
    np.add(u[:, 0::4], np.uint8(1), out=out, casting="unsafe")
    for shift, col in ((2, 1), (4, 2), (6, 3)):
        np.add(u[:, col::4], np.uint8(1), out=tmp, casting="unsafe")
        tmp <<= shift
        out |= tmp
    return out


def unpack_4_per_byte(w_packed: np.ndarray, K: int) -> np.ndarray:
    """
    Inverse of pack_ternary_4_per_byte. Mostly for tests; production kernels
    consume the packed bytes directly.
    """
    N, Kb = w_packed.shape
    assert Kb * 4 >= K
    g = np.stack([
        (w_packed >> 0) & 0x3,
        (w_packed >> 2) & 0x3,
        (w_packed >> 4) & 0x3,
        (w_packed >> 6) & 0x3,
    ], axis=-1)
    w_int8 = g.reshape(N, Kb * 4).astype(np.int8) - 1
    return w_int8[:, :K]
