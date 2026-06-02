"""
Correctness test for the torchless extern "C" NEON matmul kernel.

Verifies that matmul_packed_m1 matches a pure-numpy reference on random
ternary weights and random int8 activations, at the shapes used by BitNet-2B
projections.
"""

from __future__ import annotations

import sys

import numpy as np

from litespark_inference.torchless.kernel import (
    matmul_packed_m1,
    matmul_packed_prefill,
)
from litespark_inference.torchless.pack import pack_ternary_4_per_byte


def numpy_reference(
    x_int8: np.ndarray, w_int8: np.ndarray, x_scale: float, w_scale: float
) -> np.ndarray:
    acc = w_int8.astype(np.int32) @ x_int8.astype(np.int32)
    return acc.astype(np.float32) * x_scale * w_scale


def run_shape(N: int, K: int, seed: int) -> None:
    rng = np.random.default_rng(seed)
    w_int8 = rng.integers(-1, 2, size=(N, K), dtype=np.int8, endpoint=False)
    x_int8 = rng.integers(-127, 128, size=K, dtype=np.int8, endpoint=False)
    w_scale = float(rng.random() * 0.01 + 0.001)
    x_scale = float(rng.random() * 0.1 + 0.001)

    y_ref = numpy_reference(x_int8, w_int8, x_scale, w_scale)
    packed = pack_ternary_4_per_byte(w_int8)
    y = matmul_packed_m1(x_int8, packed, w_scale, x_scale)

    abs_err = np.abs(y - y_ref)
    rel_err = abs_err / (np.abs(y_ref) + 1e-9)
    max_abs = abs_err.max()
    max_rel = rel_err.max()
    print(f"  N={N:<5} K={K:<5} max_abs={max_abs:.4g}  max_rel={max_rel:.4g}")

    # The arithmetic is bit-exact up to fp32 rounding in the final scale mul,
    # so tolerance is tight.
    if max_abs > 1e-3:
        raise SystemExit(f"kernel output mismatched numpy reference: max_abs={max_abs}")


def run_prefill_shape(M: int, N: int, K: int, seed: int) -> None:
    """matmul_packed_prefill must equal numpy AND a loop of matmul_packed_m1."""
    rng = np.random.default_rng(seed)
    w_int8 = rng.integers(-1, 2, size=(N, K), dtype=np.int8, endpoint=False)
    x_int8 = rng.integers(-127, 128, size=(M, K), dtype=np.int8, endpoint=False)
    w_scale = float(rng.random() * 0.01 + 0.001)
    x_scale = (rng.random(M).astype(np.float32) * 0.1 + 0.001)

    packed = pack_ternary_4_per_byte(w_int8)

    # numpy reference: per-row int matmul scaled by per-row x_scale * w_scale.
    acc = x_int8.astype(np.int32) @ w_int8.astype(np.int32).T   # [M, N]
    y_ref = acc.astype(np.float32) * (x_scale[:, None] * w_scale)

    y_batched = matmul_packed_prefill(x_int8, packed, w_scale, x_scale)

    # Looping the M=1 path must reproduce the batched result bit-for-bit.
    y_loop = np.stack([
        matmul_packed_m1(x_int8[m], packed, w_scale, float(x_scale[m]))
        for m in range(M)
    ])

    max_abs_ref = float(np.abs(y_batched - y_ref).max())
    max_abs_loop = float(np.abs(y_batched - y_loop).max())
    print(f"  M={M:<4} N={N:<5} K={K:<5} "
          f"max_abs(vs numpy)={max_abs_ref:.4g}  max_abs(vs m1-loop)={max_abs_loop:.4g}")
    if max_abs_ref > 1e-3:
        raise SystemExit(f"prefill mismatched numpy reference: max_abs={max_abs_ref}")
    if max_abs_loop != 0.0:
        raise SystemExit(f"prefill not bit-identical to m1 loop: max_abs={max_abs_loop}")


def main() -> int:
    # Shapes covering the BitNet-2B projections (N_out, K_in):
    shapes = [
        (64, 64),        # tiny sanity
        (256, 2560),     # trivial
        (2560, 2560),    # q_proj / o_proj
        (640, 2560),     # k_proj / v_proj
        (6912, 2560),    # gate_proj / up_proj
        (2560, 6912),    # down_proj
    ]
    print("Torchless kernel correctness test (M=1):")
    for seed, (N, K) in enumerate(shapes, start=1):
        run_shape(N, K, seed=seed)

    # Batched prefill, including M values that exercise the MR=8 register tile
    # plus a remainder (e.g. M=13 -> one full tile of 8 + 5 leftover tokens).
    print("Batched prefill correctness test (M>1):")
    for seed, (M, N, K) in enumerate(
        [(8, 2560, 2560), (13, 640, 2560), (128, 6912, 2560), (128, 2560, 6912)],
        start=100,
    ):
        run_prefill_shape(M, N, K, seed=seed)
    print("OK")
    return 0


if __name__ == "__main__":
    sys.exit(main())
