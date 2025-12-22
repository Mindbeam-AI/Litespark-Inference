"""
Architecture detection and kernel loading for BitNet CPU inference.

Supports:
- Apple Silicon (NEON SDOT)
- x86_64 with AVX-512 VNNI (Intel Ice Lake+, AMD Zen4+)
- x86_64 with AVX-VNNI on AVX2 (Intel Core Ultra 7/9, 12th gen+)
"""

import platform
import os
from pathlib import Path
from typing import Dict, Optional

import torch
from torch.utils.cpp_extension import load


def detect_x86_features() -> Dict[str, bool]:
    """Detect x86 CPU features (AVX-512 vs AVX-VNNI)."""
    features = {
        'avx512f': False,
        'avx512vnni': False,
        'avx_vnni': False,  # AVX-VNNI on AVX2 (Intel 12th gen+, Core Ultra)
    }

    system = platform.system()

    if system == 'Linux':
        try:
            with open('/proc/cpuinfo') as f:
                cpuinfo = f.read().lower()
                features['avx512f'] = 'avx512f' in cpuinfo
                features['avx512vnni'] = 'avx512vnni' in cpuinfo or 'avx512_vnni' in cpuinfo
                features['avx_vnni'] = 'avx_vnni' in cpuinfo
        except:
            pass
    elif system == 'Darwin':
        # macOS: use sysctl
        import subprocess
        try:
            result = subprocess.run(
                ['sysctl', '-a'],
                capture_output=True, text=True, timeout=5
            )
            sysctl_output = result.stdout.lower()
            features['avx512f'] = 'hw.optional.avx512f: 1' in sysctl_output
            features['avx512vnni'] = 'hw.optional.avx512vnni: 1' in sysctl_output
            features['avx_vnni'] = 'hw.optional.avx_vnni: 1' in sysctl_output
        except:
            pass
    elif system == 'Windows':
        # Windows: assume AVX-512 if x86_64 (conservative)
        features['avx512vnni'] = True

    return features


def get_arch_info() -> Dict:
    """Get architecture information for kernel selection."""
    machine = platform.machine().lower()

    info = {
        'machine': machine,
        'is_x86_64': machine in ['x86_64', 'amd64'],
        'is_arm64': machine in ['aarch64', 'arm64'],
        'is_apple_silicon': False,
        'kernel_type': None,
        'x86_features': None,
    }

    if info['is_x86_64']:
        info['x86_features'] = detect_x86_features()

        if info['x86_features']['avx512vnni']:
            info['kernel_type'] = 'vnni'  # Full AVX-512 VNNI
        elif info['x86_features']['avx_vnni']:
            info['kernel_type'] = 'avx_vnni'  # AVX-VNNI on AVX2 (Intel Ultra)
        else:
            # Fallback: assume AVX-512 VNNI
            info['kernel_type'] = 'vnni'

    elif info['is_arm64']:
        info['is_apple_silicon'] = platform.system() == 'Darwin'
        if info['is_apple_silicon']:
            info['kernel_type'] = 'neon'
        else:
            raise RuntimeError("Only Apple Silicon ARM64 is supported for local release")

    return info


# Cached kernel instance
_kernel = None
_kernel_type = None


def get_kernel():
    """Load the appropriate kernel for this architecture (cached)."""
    global _kernel, _kernel_type

    if _kernel is None:
        arch_info = get_arch_info()
        kernel_dir = Path(__file__).parent / 'kernels'

        if arch_info['is_x86_64']:
            if arch_info['kernel_type'] == 'avx_vnni':
                # Intel Core Ultra with AVX-VNNI (256-bit)
                kernel_path = kernel_dir / 'x86_64' / 'matmul_free_avx_vnni.cpp'

                _kernel = load(
                    name='matmul_free_avx_vnni',
                    sources=[str(kernel_path)],
                    extra_cflags=['-O3', '-march=native', '-fopenmp', '-mavx2', '-mavxvnni'],
                    extra_ldflags=['-fopenmp'],
                    verbose=False
                )
                _kernel_type = 'avx_vnni'
            else:
                # Full AVX-512 VNNI (Intel Ice Lake+, AMD Zen4+)
                kernel_path = kernel_dir / 'x86_64' / 'matmul_free_tmac_vnni.cpp'

                _kernel = load(
                    name='matmul_free_vnni',
                    sources=[str(kernel_path)],
                    extra_cflags=['-O3', '-march=native', '-fopenmp', '-mavx512f', '-mavx512bw', '-mavx512vnni'],
                    extra_ldflags=['-fopenmp'],
                    verbose=False
                )
                _kernel_type = 'vnni'

        elif arch_info['is_arm64'] and arch_info['is_apple_silicon']:
            # Apple Silicon with NEON SDOT
            kernel_path = kernel_dir / 'arm64' / 'matmul_free_neon_int8.cpp'

            # macOS with Apple Clang needs special OpenMP flags
            libomp_paths = [
                '/opt/homebrew/opt/libomp',  # Apple Silicon homebrew
                '/usr/local/opt/libomp',     # Intel Mac homebrew
            ]
            libomp_path = None
            for p in libomp_paths:
                if os.path.exists(p):
                    libomp_path = p
                    break

            if libomp_path:
                extra_cflags = [
                    '-O3', '-march=native',
                    '-Xclang', '-fopenmp',
                    f'-I{libomp_path}/include',
                ]
                extra_ldflags = [
                    f'-L{libomp_path}/lib',
                    '-lomp',
                    '-framework', 'Accelerate',
                ]
            else:
                # No OpenMP available, compile without it
                extra_cflags = ['-O3', '-march=native', '-DDISABLE_OPENMP']
                extra_ldflags = ['-framework', 'Accelerate']

            _kernel = load(
                name='matmul_free_neon',
                sources=[str(kernel_path)],
                extra_cflags=extra_cflags,
                extra_ldflags=extra_ldflags,
                verbose=False
            )
            _kernel_type = 'neon'
        else:
            raise RuntimeError(f"Unsupported architecture: {arch_info['machine']}")

    return _kernel


def get_kernel_type() -> str:
    """Get the kernel type being used."""
    global _kernel_type
    if _kernel_type is None:
        get_kernel()
    return _kernel_type


def get_k_padding() -> int:
    """Get K padding alignment based on architecture."""
    kernel_type = get_kernel_type()
    if kernel_type == 'vnni':
        return 64  # AVX-512 = 512 bits = 64 bytes
    elif kernel_type == 'avx_vnni':
        return 32  # AVX2 = 256 bits = 32 bytes
    else:
        return 16  # NEON = 128 bits = 16 bytes
