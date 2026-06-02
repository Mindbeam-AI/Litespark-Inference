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

import os
import platform
from pathlib import Path
from typing import Optional

from setuptools import Extension, setup
from setuptools.command.build_ext import build_ext


_ARM_KERNEL_SRC = Path("litespark_inference/kernels/arm64/matmul_lut_neon_extern_c.cpp")
_X86_AVX512_KERNEL_SRC = Path("litespark_inference/kernels/x86_64/matmul_lut_avx512_extern_c.cpp")
_X86_AVX2_KERNEL_SRC = Path("litespark_inference/kernels/x86_64/matmul_lut_avx2_extern_c.cpp")


def _host_has_avx512() -> bool:
    """Detect whether the host CPU supports the AVX-512 set this kernel uses.

    Linux: read /proc/cpuinfo for `avx512f`. macOS Intel / Windows: assume
    AVX-512 (the Intel boxes shipping with this software family have
    Skylake-X+, and Apple killed Intel macOS support; this branch is
    rarely hit). Override with LITESPARK_KERNEL_TIER=avx512|avx2.
    """
    override = os.environ.get("LITESPARK_KERNEL_TIER", "").strip().lower()
    if override in ("avx512", "avx2"):
        return override == "avx512"
    if platform.system() == "Linux":
        try:
            with open("/proc/cpuinfo") as f:
                return "avx512f" in f.read()
        except OSError:
            return False
    return True  # non-Linux x86: assume AVX-512 (override above if not)


def _x86_kernel_src() -> Path:
    return _X86_AVX512_KERNEL_SRC if _host_has_avx512() else _X86_AVX2_KERNEL_SRC


def _x86_arch_tag() -> str:
    return "avx512" if _host_has_avx512() else "avx2"


def _torchless_kernel_src() -> Optional[Path]:
    """Pick the source file matching the current architecture, or None."""
    machine = platform.machine().lower()
    if machine in ("arm64", "aarch64") and _ARM_KERNEL_SRC.exists():
        return _ARM_KERNEL_SRC
    if machine in ("x86_64", "amd64"):
        src = _x86_kernel_src()
        if src.exists():
            return src
    return None


def _torchless_ext_name() -> str:
    """Python module name for the torchless extension (per-arch)."""
    machine = platform.machine().lower()
    if machine in ("arm64", "aarch64"):
        return "litespark_inference.torchless._matmul_lut_neon"
    if machine in ("x86_64", "amd64"):
        return f"litespark_inference.torchless._matmul_lut_{_x86_arch_tag()}"
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
        # LITESPARK_USE_VBMI selects the vpmultishiftqb fast path in
        # the kernel. Explicit macro because MSVC doesn't predefine
        # __AVX512VBMI__ even when /arch:AVX512 is set.
        compile_args += ["-DLITESPARK_USE_VBMI=1"] if is_msvc else []
        # LITESPARK_USE_AMX gates the AMX TMUL kernel on. Off by default
        # because AMX needs Sapphire Rapids+ -- enable via
        # LITESPARK_BUILD_AMX=1 at install time on a machine that can
        # actually run it. Older boxes will SIGILL on first AMX load
        # otherwise.
        if os.environ.get("LITESPARK_BUILD_AMX", "0") == "1":
            compile_args += ["-DLITESPARK_USE_AMX=1"]
            if not is_msvc:
                compile_args += ["-mamx-tile", "-mamx-int8"]
        if is_msvc:
            # /arch:AVX512 enables F+CD+BW+DQ+VL on Skylake-X+. MSVC's
            # intrinsic headers gate VBMI / VNNI by their own __cpuid
            # checks, so we don't need a separate flag for those.
            # MSVC currently always builds the AVX-512 source (no AVX2
            # fallback wired through MSVC); if you need that, add it here.
            compile_args += ["/arch:AVX512"]
        elif _host_has_avx512():
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
        else:
            # AVX2 + FMA fallback for hosts without AVX-512 (e.g. AMD Zen
            # 2/3, pre-Skylake-X Intel). The avx2 source is plain scalar
            # C++ written to auto-vectorize under -O3 with these flags.
            compile_args += ["-mavx2", "-mfma"]

    # OpenMP
    if sys_name == "Darwin":
        link_args += ["-framework", "Accelerate"]
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
