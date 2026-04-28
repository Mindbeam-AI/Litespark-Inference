"""
Build the torchless extern "C" matmul kernel as a Python extension at install
time, so users get a pre-built shared library on `pip install` and no
clang invocation happens on first inference.

Per-arch:
  arm64/aarch64 -> kernels/arm64/matmul_lut_neon_extern_c.cpp     (NEON SDOT)
  x86_64/AMD64  -> kernels/x86_64/matmul_lut_avx512_extern_c.cpp  (AVX-512)
  other         -> no extension built; torch-backed path remains usable

Project metadata (name, version, dependencies, console scripts, package
discovery, package-data) lives in pyproject.toml. setuptools merges that
metadata with the Extension defined here.

The extension uses extern "C" (no Python C API) but we still build it via
setuptools' Extension machinery so the resulting .so/.dylib lands in
litespark_inference/torchless/ with the right Python ABI tag, gets
installed automatically, and is included in wheels.
litespark_inference.torchless.kernel loads it via ctypes.
"""

from __future__ import annotations

import platform
from pathlib import Path
from typing import Optional

from setuptools import Extension, setup
from setuptools.command.build_ext import build_ext


_ARM_KERNEL_SRC = Path("litespark_inference/kernels/arm64/matmul_lut_neon_extern_c.cpp")
_X86_KERNEL_SRC = Path("litespark_inference/kernels/x86_64/matmul_lut_avx512_extern_c.cpp")


def _torchless_kernel_src() -> Optional[Path]:
    """Pick the source file matching the current architecture, or None."""
    machine = platform.machine().lower()
    if machine in ("arm64", "aarch64") and _ARM_KERNEL_SRC.exists():
        return _ARM_KERNEL_SRC
    if machine in ("x86_64", "amd64") and _X86_KERNEL_SRC.exists():
        return _X86_KERNEL_SRC
    return None


def _torchless_ext_name() -> str:
    """Python module name for the torchless extension (per-arch)."""
    machine = platform.machine().lower()
    if machine in ("arm64", "aarch64"):
        return "litespark_inference.torchless._matmul_lut_neon"
    if machine in ("x86_64", "amd64"):
        return "litespark_inference.torchless._matmul_lut_avx512"
    return "litespark_inference.torchless._matmul_lut_unknown"


def _platform_compile_args(compiler_type: str) -> tuple[list[str], list[str]]:
    """Return (extra_compile_args, extra_link_args) for current arch + compiler.

    `compiler_type` is setuptools' compiler family name: 'unix' (gcc/clang/
    Apple clang), 'msvc' (cl.exe), or 'mingw32'. The flag dialects are
    different enough that we branch.

    Apple Silicon (unix): -mcpu=native (covers NEON SDOT on M-series). For
    OpenMP we link Homebrew's libomp explicitly because Apple's toolchain
    doesn't ship one.

    Linux ARM64 (unix): -march=armv8.2-a+dotprod (Graviton 2/3/4 et al).
    OpenMP via system libomp / libgomp.

    x86_64 unix (Linux/Mac-Intel): -mavx512f/bw/dq/vnni + -mfma.
    x86_64 msvc (Windows): /arch:AVX512 enables F+CD+BW+DQ+VL on Skylake-X+.
    VNNI/FMA intrinsics work without explicit flags under MSVC (they're
    gated by header guards, not arch flags).
    """
    sys_name = platform.system()
    machine = platform.machine().lower()
    is_msvc = compiler_type == "msvc"
    compile_args: list[str] = []
    link_args: list[str] = []

    # Baseline warning + optimization flags
    if is_msvc:
        compile_args += ["/O2", "/std:c++17", "/EHsc"]
    else:
        compile_args += ["-O3", "-std=c++17", "-Wall", "-Wno-unused-function"]

    # Architecture-specific ISA flags
    if machine in ("arm64", "aarch64"):
        if is_msvc:
            # Windows-on-ARM via MSVC isn't a target for the torchless
            # path today; the kernel uses GCC/Clang NEON intrinsics.
            pass
        elif sys_name == "Darwin":
            compile_args += ["-mcpu=native"]
        else:
            compile_args += ["-march=armv8.2-a+dotprod"]
    elif machine in ("x86_64", "amd64"):
        if is_msvc:
            # /arch:AVX512 enables F+CD+BW+DQ+VL on Skylake-X+. MSVC's
            # intrinsic headers gate VBMI2 / VNNI by their own __cpuid
            # checks, so we don't need a separate flag for those.
            compile_args += ["/arch:AVX512"]
        else:
            # AVX-512F + BW + DQ for the float/byte ops; VNNI for the
            # ternary matmul VPDPBUSD; VBMI for vpmultishiftqb-based
            # nibble unpack (~3x fewer ops than the F+BW fallback) and
            # crucially runs on port 5, freeing port 0 for VPDPBUSD.
            # All available on Ice Lake+, Sapphire Rapids+, AMD Zen 4+.
            # Cascade Lake (no VBMI) falls back via the #ifdef in the kernel.
            compile_args += [
                "-mavx512f", "-mavx512bw", "-mavx512dq",
                "-mavx512vnni", "-mavx512vbmi", "-mfma",
            ]

    # OpenMP
    if sys_name == "Darwin":
        # Homebrew installs libomp at /opt/homebrew (arm64) or /usr/local
        # (intel). Try both.
        candidates = [
            Path("/opt/homebrew/opt/libomp"),
            Path("/usr/local/opt/libomp"),
        ]
        omp_root = next((p for p in candidates if p.exists()), None)
        if omp_root is not None:
            compile_args += [
                "-Xpreprocessor", "-fopenmp",
                f"-I{omp_root / 'include'}",
            ]
            link_args += [
                f"-L{omp_root / 'lib'}",
                f"-Wl,-rpath,{omp_root / 'lib'}",
                "-lomp",
            ]
        else:
            print(
                "[litespark setup] note: Homebrew libomp not found; "
                "building the torchless kernel without OpenMP. "
                "Install with `brew install libomp` for multi-threaded "
                "kernels on macOS."
            )
    elif sys_name == "Linux":
        compile_args += ["-fopenmp"]
        link_args += ["-fopenmp"]
    elif sys_name == "Windows" and is_msvc:
        compile_args += ["/openmp"]

    return compile_args, link_args


class _TorchlessExtensionBuild(build_ext):
    """Per-platform compile/link flags applied at build time."""

    def build_extension(self, ext: Extension) -> None:
        compile_args, link_args = _platform_compile_args(self.compiler.compiler_type)
        ext.extra_compile_args = list(ext.extra_compile_args or []) + compile_args
        ext.extra_link_args = list(ext.extra_link_args or []) + link_args
        super().build_extension(ext)


_src = _torchless_kernel_src()
_ext_modules = (
    [
        Extension(
            name=_torchless_ext_name(),
            sources=[str(_src)],
            language="c++",
        )
    ]
    if _src is not None
    else []
)


setup(
    ext_modules=_ext_modules,
    cmdclass={"build_ext": _TorchlessExtensionBuild},
)
