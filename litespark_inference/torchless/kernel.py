"""
ctypes bindings for the torchless extern "C" matmul kernel.

Build path (preferred): the per-arch C++ source under
  litespark_inference/kernels/arm64/matmul_lut_neon_extern_c.cpp     (NEON)
  litespark_inference/kernels/x86_64/matmul_lut_avx512_extern_c.cpp  (AVX-512)
is compiled at install time by setup.py into a Python-extension-shaped
shared library that lands next to this file. We locate it by globbing
for any `_matmul_lut_{neon,avx512}*.{so,dylib}` artefact.

Fallback: if the shared library is missing (e.g. someone cloned the repo
without `pip install`-ing it, or copied just the package directory), we
JIT-compile via clang and write the dylib next to this file. The JIT path
is purely a developer convenience -- a `pip install` produces the
extension during install and never invokes the JIT.

The C-side symbols are arch-suffixed (`*_neon` / `*_avx512`) so an
accidental cross-arch load surfaces immediately. The Python-side wrappers
below expose arch-neutral names that runtime.py imports.

Deliberately does not import torch.
"""

from __future__ import annotations

import ctypes
import os
import platform
import subprocess
import sys
from pathlib import Path
from typing import Optional

import numpy as np


_HERE = Path(__file__).resolve().parent

_MACHINE = platform.machine().lower()
_IS_ARM = _MACHINE in ("arm64", "aarch64")
_IS_X86 = _MACHINE in ("x86_64", "amd64")

if _IS_ARM:
    _ARCH_TAG = "neon"
    _SRC = _HERE.parent / "kernels" / "arm64" / "matmul_lut_neon_extern_c.cpp"
elif _IS_X86:
    _ARCH_TAG = "avx512"
    _SRC = _HERE.parent / "kernels" / "x86_64" / "matmul_lut_avx512_extern_c.cpp"
else:
    _ARCH_TAG = "unknown"
    _SRC = _HERE  # nonexistent; load() will fail gracefully

_LIB_PREFIX = f"_matmul_lut_{_ARCH_TAG}"

_lib: Optional[ctypes.CDLL] = None


class KernelBuildError(RuntimeError):
    pass


def _find_built_extension() -> Optional[Path]:
    """Look for a pre-built extension produced by setup.py build_ext.

    setuptools writes the artefact with a Python ABI tag, e.g.
    `_matmul_lut_neon.cpython-311-darwin.so` or
    `_matmul_lut_avx512.cpython-311-x86_64-linux-gnu.so`. We accept any
    file matching the per-arch prefix, falling back to a legacy bare-name
    dylib that the JIT path produces.
    """
    candidates: list[Path] = []
    for pattern in (f"{_LIB_PREFIX}*.so", f"{_LIB_PREFIX}*.dylib", f"{_LIB_PREFIX}*.pyd"):
        candidates.extend(_HERE.glob(pattern))
    if not candidates:
        return None
    # Prefer the Python-ABI-tagged one (setup.py output) over a bare dylib
    # (legacy JIT output) if both exist.
    candidates.sort(key=lambda p: ("cpython" not in p.name, p.name))
    return candidates[0]


_LIB_SUFFIX = {"Darwin": ".dylib", "Linux": ".so", "Windows": ".dll"}.get(
    platform.system(), ".so"
)
# Path the JIT fallback writes to. The pip-installed extension will have a
# different (ABI-tagged) name and will be preferred by _find_built_extension.
_LIB_PATH = _HERE / f"{_LIB_PREFIX}{_LIB_SUFFIX}"


_HOMEBREW_LIBOMP = Path("/opt/homebrew/opt/libomp")


def _omp_flags() -> list[str]:
    """Return compiler/linker flags to enable OpenMP, or [] if not found.

    Apple clang needs Homebrew's libomp; gcc/Linux clang usually find it via
    plain -fopenmp. The Homebrew-libomp path is best-effort: if it's absent
    we silently compile without OMP (the #pragma omp line becomes a no-op).
    """
    if platform.system() == "Darwin" and _HOMEBREW_LIBOMP.exists():
        return [
            "-Xpreprocessor", "-fopenmp",
            f"-I{_HOMEBREW_LIBOMP / 'include'}",
            f"-L{_HOMEBREW_LIBOMP / 'lib'}",
            f"-Wl,-rpath,{_HOMEBREW_LIBOMP / 'lib'}",
            "-lomp",
        ]
    # Non-macOS: trust -fopenmp if available. A CalledProcessError will
    # surface upstream if the host has no libomp, and we fall back cleanly
    # in _compile().
    if platform.system() != "Darwin":
        return ["-fopenmp"]
    return []


def _compile() -> None:
    """Compile the extern "C" kernel into a shared library via clang/gcc."""
    compiler = os.environ.get("CXX", "clang++")
    base = [
        compiler,
        "-O3",
        "-std=c++17",
        "-fPIC",
        "-shared",
        "-Wall",
        "-Wno-unused-function",
    ]
    if _IS_ARM:
        # Apple clang accepts -mcpu=native; Linux clang/gcc want -march.
        # NEON + SDOT ("dotprod") are required by the SIMD path.
        base += ["-mcpu=native"] if platform.system() == "Darwin" else [
            "-march=armv8.2-a+dotprod",
        ]
    elif _IS_X86:
        # Match setup.py: AVX-512F + BW + DQ + VNNI + VBMI + FMA.
        base += [
            "-mavx512f", "-mavx512bw", "-mavx512dq",
            "-mavx512vnni", "-mavx512vbmi", "-mfma",
        ]

    omp = _omp_flags()
    cmd_omp = base + omp + [str(_SRC), "-o", str(_LIB_PATH)]
    cmd_noomp = base + [str(_SRC), "-o", str(_LIB_PATH)]

    for cmd in (cmd_omp, cmd_noomp) if omp else (cmd_noomp,):
        try:
            subprocess.run(cmd, check=True, capture_output=True, text=True)
            return
        except subprocess.CalledProcessError as e:
            last_err = e
            continue
    raise KernelBuildError(
        f"kernel build failed:\n  cmd: {' '.join(cmd_noomp)}\n"
        f"  stderr: {last_err.stderr}\n  stdout: {last_err.stdout}"
    ) from last_err


def _load() -> ctypes.CDLL:
    global _lib
    if _lib is not None:
        return _lib
    built = _find_built_extension()
    if built is not None and (
        not _SRC.exists() or built.stat().st_mtime >= _SRC.stat().st_mtime
    ):
        # Pre-built extension is up to date with the source. Use it.
        lib_path = built
    else:
        # Either nothing built yet, or the built artefact is older than
        # the source (developer edited the .cpp without re-running
        # `pip install -e .`). Fall back to the JIT path.
        if built is None:
            print(
                "[litespark torchless] note: no pre-built kernel extension "
                "found; falling back to JIT clang compile. Re-run "
                "`pip install -e .` (or `pip install .`) to build the "
                "extension once and avoid this every import.",
                file=sys.stderr,
            )
        _compile()
        lib_path = _LIB_PATH
    lib = ctypes.CDLL(str(lib_path))

    # Resolve arch-tagged C symbols and re-attach under arch-neutral names.
    # Use numpy.ctypeslib.ndpointer for array args -- it lets callers pass
    # numpy arrays directly (no per-call .ctypes.data_as allocation), and
    # enforces dtype + C-contiguity at type-check time so wrappers can
    # skip np.ascontiguousarray on the hot path.
    from numpy import ctypeslib as _npct
    _F32 = _npct.ndpointer(dtype=np.float32, flags="C_CONTIGUOUS")
    _I8  = _npct.ndpointer(dtype=np.int8,    flags="C_CONTIGUOUS")
    _U8  = _npct.ndpointer(dtype=np.uint8,   flags="C_CONTIGUOUS")
    _U16 = _npct.ndpointer(dtype=np.uint16,  flags="C_CONTIGUOUS")

    def _alias(neutral_name: str, c_name: str, argtypes, restype):
        fn = getattr(lib, c_name)
        fn.argtypes = argtypes
        fn.restype = restype
        setattr(lib, neutral_name, fn)

    T = _ARCH_TAG
    _alias("matmul_lut_m1", f"matmul_lut_{T}_m1",
           [_I8, _U8, ctypes.c_float, ctypes.c_float, _F32, ctypes.c_int, ctypes.c_int],
           None)
    _alias("quantize_activation", f"quantize_activation_{T}",
           [_F32, _I8, ctypes.c_int],
           ctypes.c_float)
    _alias("lm_head_bf16_fp32", f"lm_head_bf16_fp32_{T}",
           [_U16, _F32, _F32, ctypes.c_int, ctypes.c_int],
           None)
    _alias("lm_head_int8_fp32", f"lm_head_int8_fp32_{T}",
           [_I8, _F32, _F32, _F32, ctypes.c_int, ctypes.c_int],
           None)
    _alias("lm_head_int4_fp32", f"lm_head_int4_fp32_{T}",
           [_U8, _F32, _F32, _F32, ctypes.c_int, ctypes.c_int],
           None)
    _alias("rmsnorm_bf16gamma_fp32", f"rmsnorm_bf16gamma_fp32_{T}",
           [_F32, _U16, _F32, ctypes.c_int, ctypes.c_float],
           None)
    _alias("rmsnorm_fp32gamma_fp32", f"rmsnorm_fp32gamma_fp32_{T}",
           [_F32, _F32, _F32, ctypes.c_int, ctypes.c_float],
           None)
    _alias("relu2_mul_fp32", f"relu2_mul_fp32_{T}",
           [_F32, _F32, _F32, ctypes.c_int],
           None)
    _alias("add_inplace_fp32", f"add_inplace_fp32_{T}",
           [_F32, _F32, ctypes.c_int],
           None)
    _alias("has_omp_probe", f"matmul_lut_{T}_has_omp", [], ctypes.c_int)
    _alias("max_threads_probe", f"matmul_lut_{T}_max_threads", [], ctypes.c_int)

    _lib = lib
    return lib


def has_omp() -> bool:
    return bool(_load().has_omp_probe())


def max_threads() -> int:
    return int(_load().max_threads_probe())


# All wrappers below skip per-call dtype/shape asserts and just hand the
# arrays to ctypes -- ndpointer-typed argtypes raise TypeError on mismatch
# and enforce C-contiguity, so the ascontiguousarray + assert calls that
# used to live in each wrapper are now redundant (and were measured at
# ~5% of forward-pass time).


def lm_head_int4(
    emb_packed: np.ndarray,
    emb_scale: np.ndarray,
    x_fp32: np.ndarray,
    out_fp32: np.ndarray,
    H: int,
) -> np.ndarray:
    """Tied LM head via int4xfp32 matmul (2 nibbles/byte; H must be even)."""
    V = emb_packed.shape[0]
    _load().lm_head_int4_fp32(emb_packed, emb_scale, x_fp32, out_fp32, V, H)
    return out_fp32


def lm_head_int8(
    emb_int8: np.ndarray,
    emb_scale: np.ndarray,
    x_fp32: np.ndarray,
    out_fp32: np.ndarray,
) -> np.ndarray:
    """Tied LM head via int8xfp32 matmul with per-row fp32 scale."""
    V, H = emb_int8.shape
    _load().lm_head_int8_fp32(emb_int8, emb_scale, x_fp32, out_fp32, V, H)
    return out_fp32


def lm_head_bf16(
    emb_u16: np.ndarray,
    x_fp32: np.ndarray,
    out_fp32: np.ndarray,
) -> np.ndarray:
    """Tied LM head via bf16xfp32 matmul (emb_u16 holds bf16 bits)."""
    V, H = emb_u16.shape
    _load().lm_head_bf16_fp32(emb_u16, x_fp32, out_fp32, V, H)
    return out_fp32


def rmsnorm_into(
    x_fp32: np.ndarray,
    gamma: np.ndarray,
    out_fp32: np.ndarray,
    eps: float = 1e-5,
) -> np.ndarray:
    """RMSNorm x -> out (caller-owned). gamma may be uint16 (bf16) or fp32."""
    K = x_fp32.shape[0]
    lib = _load()
    if gamma.dtype == np.uint16:
        lib.rmsnorm_bf16gamma_fp32(x_fp32, gamma, out_fp32, K, eps)
    elif gamma.dtype == np.float32:
        lib.rmsnorm_fp32gamma_fp32(x_fp32, gamma, out_fp32, K, eps)
    else:
        raise TypeError(f"gamma dtype must be uint16 or float32, got {gamma.dtype}")
    return out_fp32


def relu2_mul_into(
    gate_fp32: np.ndarray,
    up_fp32: np.ndarray,
    out_fp32: np.ndarray,
) -> np.ndarray:
    """out = max(gate, 0)^2 * up (caller-owned out)."""
    _load().relu2_mul_fp32(gate_fp32, up_fp32, out_fp32, gate_fp32.shape[0])
    return out_fp32


def add_inplace(a_fp32: np.ndarray, b_fp32: np.ndarray) -> np.ndarray:
    """a += b in-place (both fp32 1-D)."""
    _load().add_inplace_fp32(a_fp32, b_fp32, a_fp32.shape[0])
    return a_fp32


def quantize_activation(x_fp32: np.ndarray, out_int8: np.ndarray) -> float:
    """Per-tensor absmax int8 quantization in-place. Returns the scale."""
    return float(_load().quantize_activation(x_fp32, out_int8, x_fp32.shape[0]))


def matmul_packed_m1(
    x_int8: np.ndarray,
    packed_w: np.ndarray,
    w_scale: float,
    x_scale: float,
    out: Optional[np.ndarray] = None,
) -> np.ndarray:
    """
    Ternary matmul for M=1 (token generation), consuming 2-bit packed weights.

    Args:
        x_int8:   [K] int8 (pre-quantized activation).
        packed_w: [N, K/4] uint8 (output of pack_ternary_4_per_byte).
        w_scale:  scalar float (BitNet b1.58 per-tensor absmean).
        x_scale:  scalar activation scale.
        out:      Optional [N] float32 buffer. If None, one is allocated.

    Returns:
        [N] float32 output.
    """
    N, Kb = packed_w.shape
    K = Kb * 4
    if out is None:
        out = np.empty(N, dtype=np.float32)
    _load().matmul_lut_m1(x_int8, packed_w, w_scale, x_scale, out, N, K)
    return out


def ensure_built() -> Path:
    """Make sure the kernel is loadable; return the path that was loaded."""
    _load()
    built = _find_built_extension()
    return built if built is not None else _LIB_PATH
