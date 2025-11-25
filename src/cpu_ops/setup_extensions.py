"""
PyTorch C++ Extension Setup for CPU-Optimized MatMul-Free Operations

This module handles the compilation and loading of optimized C++ kernels
for different CPU architectures.
"""

import os
import platform
import torch
from torch.utils.cpp_extension import load, CppExtension
from pathlib import Path
from .simd import detect_simd_support

# Get the directory containing this file
CURRENT_DIR = Path(__file__).parent
KERNELS_DIR = CURRENT_DIR / "kernels"


def get_compiler_flags():
    """Get appropriate compiler flags for the current platform."""
    flags = ['-O3']

    system = platform.system()
    machine = platform.machine().lower()

    simd_caps = detect_simd_support()
    if simd_caps.architecture == 'x86_64' and simd_caps.has_avx2:
        flags.append('-DHAS_AVX2_SUPPORT')
    elif simd_caps.architecture == 'arm64' and simd_caps.has_neon:
        flags.append('-DHAS_NEON_SUPPORT')

    if system == 'Darwin':
        # macOS specific flags - use Xpreprocessor for OpenMP with clang
        flags.extend(['-Xpreprocessor', '-fopenmp', '-std=c++17', '-mmacosx-version-min=10.9'])

        if 'arm' in machine:
            # Apple Silicon
            flags.extend(['-mcpu=apple-m1', '-mtune=native'])
        else:
            # Intel Mac
            flags.extend(['-march=native', '-mtune=native'])

    elif system == 'Linux':
        # Linux specific flags - standard OpenMP flag
        flags.extend(['-fopenmp', '-std=c++17', '-fPIC'])

        if 'x86' in machine or 'amd64' in machine:
            # x86_64 Linux
            flags.extend(['-march=native', '-mtune=native'])
        elif 'arm' in machine or 'aarch64' in machine:
            # ARM64 Linux
            flags.extend(['-mcpu=native', '-mtune=native'])

    return flags


def get_source_files():
    """Get the appropriate source files based on CPU architecture."""
    simd_caps = detect_simd_support()
    sources = []
    
    if simd_caps.architecture == 'x86_64':
        if simd_caps.has_avx2:
            sources.append(str(KERNELS_DIR / "x86_64" / "matmul_free_avx2.cpp"))
        # TODO: Add AVX-512 support
        # if simd_caps.has_avx512:
        #     sources.append(str(KERNELS_DIR / "x86_64" / "matmul_free_avx512.cpp"))
            
    elif simd_caps.architecture == 'arm64':
        if simd_caps.has_neon:
            sources.append(str(KERNELS_DIR / "arm64" / "matmul_free_neon.cpp"))
    
    # Always include generic fallback
    sources.append(str(KERNELS_DIR / "generic" / "matmul_free_generic.cpp"))
    
    return sources


def get_linker_flags():
    """Get appropriate linker flags for the current platform."""
    system = platform.system()

    if system == 'Darwin':
        # macOS: Link against libomp from Homebrew
        flags = ['-L/opt/homebrew/lib', '-L/usr/local/lib', '-lomp']

    elif system == 'Linux':
        # Linux: Standard OpenMP linking
        flags = ['-fopenmp', '-lgomp']
    else:
        flags = []

    return flags


def compile_cpu_kernels():
    """
    Compile CPU-optimized kernels as PyTorch C++ extension.
    
    Returns:
        Compiled extension module or None if compilation fails
    """
    try:
        sources = get_source_files()
        
        if not sources:
            print("Warning: No optimized kernels available for this architecture")
            return None
        
        # Check if source files exist
        missing_files = [f for f in sources if not os.path.exists(f)]
        if missing_files:
            print(f"Warning: Missing source files: {missing_files}")
            return None
        
        print(f"Compiling CPU kernels from: {sources}")
        
        # Compile the extension
        cpu_kernels = load(
            name="cpu_kernels",
            sources=sources,
            extra_cflags=get_compiler_flags(),
            extra_ldflags=get_linker_flags(),
            verbose=True
        )
        
        print("✅ CPU kernels compiled successfully!")
        return cpu_kernels
        
    except Exception as e:
        print(f"❌ Failed to compile CPU kernels: {e}")
        print("Falling back to Python implementation")
        return None


def load_cpu_kernels():
    """
    Load compiled CPU kernels with caching.
    
    Returns:
        Compiled extension module or None if not available
    """
    # Try to load from cache first
    try:
        import cpu_kernels
        return cpu_kernels
    except ImportError:
        pass
    
    # Compile if not cached
    return compile_cpu_kernels()


# Module-level kernel loading
_cpu_kernels = None

def get_cpu_kernels():
    """Get the compiled CPU kernels (cached)."""
    global _cpu_kernels
    if _cpu_kernels is None:
        _cpu_kernels = load_cpu_kernels()
    return _cpu_kernels


def has_compiled_kernels():
    """Check if compiled kernels are available."""
    return get_cpu_kernels() is not None


if __name__ == "__main__":
    # Test compilation
    print("Testing CPU kernel compilation...")
    kernels = compile_cpu_kernels()
    if kernels:
        print("✅ Compilation test passed!")
    else:
        print("❌ Compilation test failed!")
