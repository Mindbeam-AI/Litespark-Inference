"""
Correctness test for the torchless extern "C" NEON matmul kernel.

Verifies that matmul_packed_m1 matches a pure-numpy reference on random
ternary weights and random int8 activations, at the shapes used by BitNet-2B
projections.
"""

from __future__ import annotations

import sys

import numpy as np

from litespark_inference.torchless.kernel import matmul_packed_m1
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
    print("Torchless kernel correctness test:")
    for seed, (N, K) in enumerate(shapes, start=1):
        run_shape(N, K, seed=seed)
    print("OK")
    return 0


if __name__ == "__main__":
    sys.exit(main())
