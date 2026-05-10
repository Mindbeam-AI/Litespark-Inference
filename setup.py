"""
Build the torchless extern "C" NEON kernel as a Python extension at install
time, so users get a pre-built shared library on `pip install` and no
clang invocation happens on first inference.

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

from setuptools import Extension, setup
from setuptools.command.build_ext import build_ext


_KERNEL_SRC = Path("litespark_inference/kernels/arm64/matmul_lut_neon_extern_c.cpp")

# The NEON kernel source unconditionally includes <arm_neon.h>, so it can only
# be compiled on ARM64. On x86_64 we skip the extension entirely; the package
# still installs cleanly and the runtime falls back to the x86_64 kernels under
# litespark_inference/kernels/x86_64/ (or to torch/transformers when those are
# unavailable). Without this gate, `pip install` on Linux/macOS x86 fails with
# "'arm_neon.h' file not found".
_IS_ARM64 = platform.machine().lower() in ("arm64", "aarch64")


def _platform_compile_args() -> tuple[list[str], list[str]]:
    """Return (extra_compile_args, extra_link_args) for the current platform.

    Apple Silicon: -mcpu=native (covers NEON SDOT on M-series). For OpenMP
    we link Homebrew's libomp explicitly because Apple's toolchain doesn't
    ship one.

    Linux ARM64: -march=armv8.2-a+dotprod (Graviton 2/3/4 et al). OpenMP
    is the system libomp / libgomp, picked up by -fopenmp on the linker
    line.

    Other platforms: best-effort generic flags. The matmul kernel only
    targets arm64 today.
    """
    sys_name = platform.system()
    machine = platform.machine().lower()
    compile_args = ["-O3", "-std=c++17", "-Wall", "-Wno-unused-function"]
    link_args: list[str] = []

    # Architecture
    if machine in ("arm64", "aarch64"):
        if sys_name == "Darwin":
            compile_args += ["-mcpu=native"]
        else:
            compile_args += ["-march=armv8.2-a+dotprod"]

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

    return compile_args, link_args


class _NEONExtensionBuild(build_ext):
    """Per-platform compile/link flags applied at build time."""

    def build_extension(self, ext: Extension) -> None:
        compile_args, link_args = _platform_compile_args()
        ext.extra_compile_args = list(ext.extra_compile_args or []) + compile_args
        ext.extra_link_args = list(ext.extra_link_args or []) + link_args
        super().build_extension(ext)


_torchless_ext = Extension(
    name="litespark_inference.torchless._matmul_lut_neon",
    sources=[str(_KERNEL_SRC)],
    language="c++",
)


setup(
    ext_modules=[_torchless_ext] if (_KERNEL_SRC.exists() and _IS_ARM64) else [],
    cmdclass={"build_ext": _NEONExtensionBuild},
)
