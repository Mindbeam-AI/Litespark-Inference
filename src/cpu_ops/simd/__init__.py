"""
SIMD Detection and Utilities for CPU Optimization

Provides cross-platform SIMD capability detection and utilities
for x86_64 (AVX2, AVX-512) and ARM64 (NEON) architectures.
"""

import platform
import subprocess
import os
from typing import Dict, List, Optional
from dataclasses import dataclass


@dataclass
class SIMDCapabilities:
    """Container for SIMD capability information."""
    architecture: str
    has_sse: bool = False
    has_avx: bool = False
    has_avx2: bool = False
    has_avx512: bool = False
    has_neon: bool = False
    vector_width: int = 4  # Default to 4 floats (128-bit)
    optimal_block_size: int = 64


def detect_x86_features() -> Dict[str, bool]:
    """Detect x86_64 SIMD features using cpuinfo or /proc/cpuinfo."""
    features = {
        'sse': False,
        'avx': False,
        'avx2': False,
        'avx512': False
    }
    
    try:
        if platform.system() == 'Linux':
            # Read /proc/cpuinfo on Linux
            with open('/proc/cpuinfo', 'r') as f:
                cpuinfo = f.read().lower()
                features['sse'] = 'sse' in cpuinfo
                features['avx'] = 'avx' in cpuinfo
                features['avx2'] = 'avx2' in cpuinfo
                features['avx512'] = 'avx512' in cpuinfo
                
        elif platform.system() == 'Darwin':
            # Use sysctl on macOS (for Intel Macs)
            try:
                result = subprocess.run(['sysctl', '-a'], capture_output=True, text=True)
                sysctl_output = result.stdout.lower()
                features['sse'] = 'sse' in sysctl_output
                features['avx'] = 'avx' in sysctl_output
                features['avx2'] = 'avx2' in sysctl_output
                features['avx512'] = 'avx512' in sysctl_output
            except:
                pass
                
        elif platform.system() == 'Windows':
            # TODO: Implement Windows CPUID detection
            pass
            
    except Exception as e:
        print(f"Warning: Could not detect x86 features: {e}")
    
    return features


def detect_arm_features() -> Dict[str, bool]:
    """Detect ARM64 SIMD features."""
    features = {
        'neon': False,
        'sve': False  # Scalable Vector Extension (future ARM feature)
    }
    
    try:
        if platform.system() == 'Darwin':
            # Apple Silicon always has NEON
            machine = platform.machine().lower()
            if 'arm' in machine or 'aarch64' in machine:
                features['neon'] = True
                
        elif platform.system() == 'Linux':
            # Check /proc/cpuinfo for ARM features
            try:
                with open('/proc/cpuinfo', 'r') as f:
                    cpuinfo = f.read().lower()
                    features['neon'] = 'neon' in cpuinfo or 'asimd' in cpuinfo
                    features['sve'] = 'sve' in cpuinfo
            except:
                pass
                
    except Exception as e:
        print(f"Warning: Could not detect ARM features: {e}")
    
    return features


def detect_simd_support() -> SIMDCapabilities:
    """
    Detect SIMD capabilities for the current CPU.
    
    Returns:
        SIMDCapabilities object with detected features
    """
    machine = platform.machine().lower()
    system = platform.system()
    
    if 'x86' in machine or 'amd64' in machine or 'i386' in machine:
        # x86_64 architecture
        features = detect_x86_features()
        
        caps = SIMDCapabilities(
            architecture='x86_64',
            has_sse=features['sse'],
            has_avx=features['avx'],
            has_avx2=features['avx2'],
            has_avx512=features['avx512']
        )
        
        # Set optimal parameters based on capabilities
        if caps.has_avx512:
            caps.vector_width = 16  # 512-bit / 32-bit = 16 floats
            caps.optimal_block_size = 128
        elif caps.has_avx2:
            caps.vector_width = 8   # 256-bit / 32-bit = 8 floats
            caps.optimal_block_size = 64
        elif caps.has_avx:
            caps.vector_width = 8   # 256-bit / 32-bit = 8 floats
            caps.optimal_block_size = 64
        else:
            caps.vector_width = 4   # 128-bit / 32-bit = 4 floats
            caps.optimal_block_size = 32
            
    elif 'arm' in machine or 'aarch64' in machine:
        # ARM64 architecture
        features = detect_arm_features()
        
        caps = SIMDCapabilities(
            architecture='arm64',
            has_neon=features['neon']
        )
        
        if caps.has_neon:
            caps.vector_width = 4   # NEON: 128-bit / 32-bit = 4 floats
            caps.optimal_block_size = 64
        else:
            caps.vector_width = 1   # Scalar fallback
            caps.optimal_block_size = 16
            
    else:
        # Unknown architecture - use generic fallback
        caps = SIMDCapabilities(
            architecture='generic',
            vector_width=1,
            optimal_block_size=16
        )
    
    return caps


def get_simd_width() -> int:
    """Get the optimal SIMD vector width for the current CPU."""
    return detect_simd_support().vector_width


def get_optimal_block_size() -> int:
    """Get the optimal block size for the current CPU."""
    return detect_simd_support().optimal_block_size


# Cache the detection result to avoid repeated system calls
_cached_simd_caps: Optional[SIMDCapabilities] = None

def get_simd_capabilities() -> SIMDCapabilities:
    """Get cached SIMD capabilities."""
    global _cached_simd_caps
    if _cached_simd_caps is None:
        _cached_simd_caps = detect_simd_support()
    return _cached_simd_caps
