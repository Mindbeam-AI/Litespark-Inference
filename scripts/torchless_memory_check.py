"""
Measure live RSS and tensor-byte budget of the torchless BitNet-2B loader.

Run:
    /usr/bin/time -l python scripts/torchless_memory_check.py

This script deliberately does not import torch / transformers. It will raise
if either ends up imported transitively.
"""

from __future__ import annotations

import gc
import os
import subprocess
import sys


def current_rss_mb() -> float:
    r = subprocess.run(
        ["ps", "-o", "rss=", "-p", str(os.getpid())],
        capture_output=True, text=True, timeout=5,
    )
    return int(r.stdout.strip()) / 1024


def fmt_mb(x: float) -> str:
    return f"{x:,.1f} MB"


def main() -> None:
    if "torch" in sys.modules:
        raise RuntimeError("torch imported before main() - POC invariant violated")

    gc.collect()
    rss0 = current_rss_mb()
    print(f"[0] python baseline RSS                = {fmt_mb(rss0)}", flush=True)

    from litespark_inference.torchless import load_bitnet_2b

    gc.collect()
    rss_imports = current_rss_mb()
    print(f"[1] after torchless imports            = {fmt_mb(rss_imports)}", flush=True)

    model = load_bitnet_2b()
    gc.collect()

    # macOS: ask the allocator to return freed pages before we read RSS
    try:
        import ctypes
        libc = ctypes.CDLL("libSystem.dylib")
        libc.malloc_zone_pressure_relief.restype = ctypes.c_size_t
        libc.malloc_zone_pressure_relief.argtypes = [ctypes.c_void_p, ctypes.c_size_t]
        libc.malloc_zone_pressure_relief(None, 0)
    except Exception:
        pass

    rss_loaded = current_rss_mb()

    tb = model.tensor_bytes()
    MB = 1024 * 1024
    print(f"[2] loaded all weights, post-gc", flush=True)
    print(f"    current RSS                        = {fmt_mb(rss_loaded)}", flush=True)
    print(f"    RSS delta vs post-imports          = {fmt_mb(rss_loaded - rss_imports)}", flush=True)
    print(f"", flush=True)
    print(f"    packed ternary weights (2-bit)      = {fmt_mb(tb['packed_weights'] / MB)}", flush=True)
    print(f"    w_sum (int32)                        = {fmt_mb(tb['w_sum'] / MB)}", flush=True)
    print(f"    scales (fp32)                        = {fmt_mb(tb['scales'] / MB)}", flush=True)
    print(f"    per-layer norms                      = {fmt_mb(tb['per_layer_norms'] / MB)}", flush=True)
    print(f"    final norm                           = {fmt_mb(tb['final_norm'] / MB)}", flush=True)
    print(f"    embedding (bf16 / uint16)            = {fmt_mb(tb['embedding'] / MB)}", flush=True)
    print(f"    --------------------------------------", flush=True)
    print(f"    tensor bytes, NO embedding           = {fmt_mb(tb['total_excl_embedding'] / MB)}", flush=True)
    print(f"    tensor bytes, WITH embedding         = {fmt_mb(tb['total_incl_embedding'] / MB)}", flush=True)

    if "torch" in sys.modules:
        raise RuntimeError("torch got imported transitively - torchless invariant violated")


if __name__ == "__main__":
    main()
