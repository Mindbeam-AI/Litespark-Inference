#!/usr/bin/env python3
"""
CPU-Optimized Inference Script for MatMul-Free Language Models

This script demonstrates CPU-only inference using optimized MatMul-free operations
with support for multiple CPU architectures including Apple Silicon.
"""

import argparse
import time
import torch
import yaml
from pathlib import Path
import psutil
import platform

# Import CPU-optimized components
from src.cpu_ops import (
    get_cpu_info, 
    get_optimal_implementation,
    detect_simd_support
)

# Import model components (will be CPU-optimized versions)
from src.models.hgrn_bit.configuration_hgrn_bit import HGRNBitConfig
# TODO: Create CPU-optimized model class
# from src.models.cpu_hgrn_bit import CPUHGRNBitForCausalLM


def load_cpu_config(config_path: str, model_name: str) -> dict:
    """Load CPU-optimized model configuration."""
    with open(config_path, 'r') as f:
        configs = yaml.safe_load(f)
    
    if model_name not in configs:
        available = list(configs.keys())
        raise ValueError(f"Model '{model_name}' not found. Available: {available}")
    
    return configs[model_name]


def print_system_info():
    """Print detailed system information for optimization."""
    print("=" * 80)
    print("SYSTEM INFORMATION")
    print("=" * 80)
    
    # Basic system info
    cpu_info = get_cpu_info()
    simd_caps = detect_simd_support()
    
    print(f"Platform: {cpu_info['system']} {cpu_info['architecture']}")
    print(f"Processor: {cpu_info['processor']}")
    print(f"CPU Cores: {cpu_info['cpu_count']}")
    print(f"Architecture: {simd_caps.architecture}")
    
    # SIMD capabilities
    print(f"\nSIMD Capabilities:")
    if simd_caps.architecture == 'x86_64':
        print(f"  SSE: {simd_caps.has_sse}")
        print(f"  AVX: {simd_caps.has_avx}")
        print(f"  AVX2: {simd_caps.has_avx2}")
        print(f"  AVX-512: {simd_caps.has_avx512}")
    elif simd_caps.architecture == 'arm64':
        print(f"  NEON: {simd_caps.has_neon}")
        if cpu_info.get('is_apple_silicon'):
            print(f"  Apple Silicon: Yes")
    
    print(f"  Vector Width: {simd_caps.vector_width} floats")
    print(f"  Optimal Block Size: {simd_caps.optimal_block_size}")
    
    # Memory info
    memory = psutil.virtual_memory()
    print(f"\nMemory:")
    print(f"  Total: {memory.total / (1024**3):.1f} GB")
    print(f"  Available: {memory.available / (1024**3):.1f} GB")
    print(f"  Usage: {memory.percent:.1f}%")
    
    # Optimal implementation
    impl = get_optimal_implementation()
    print(f"\nOptimal Implementation: {impl}")
    
    print("=" * 80)


def benchmark_cpu_operations(model_config: dict, num_runs: int = 10):
    """Benchmark CPU MatMul-free operations."""
    print("\nBENCHMARKING CPU OPERATIONS")
    print("-" * 40)
    
    # Test different sizes
    test_sizes = [
        (1, 1024),      # Single token
        (32, 1024),     # Small batch
        (128, 1024),    # Medium batch
    ]
    
    hidden_size = model_config['hidden_size']
    
    for batch_size, seq_len in test_sizes:
        print(f"\nTesting: batch_size={batch_size}, seq_len={seq_len}, hidden_size={hidden_size}")
        
        # Create test tensors
        x = torch.randn(batch_size * seq_len, hidden_size)
        w = torch.randn(hidden_size, hidden_size)
        
        # Quantize weights to ternary
        from src.cpu_ops.matmul_free_cpu import weight_quant_cpu
        w_ternary = weight_quant_cpu(w)
        
        # Benchmark MatMul-free operation
        from src.cpu_ops.matmul_free_cpu import matmul_free_cpu
        
        # Warmup
        for _ in range(3):
            _ = matmul_free_cpu(x, w_ternary)
        
        # Benchmark
        times = []
        for _ in range(num_runs):
            start_time = time.perf_counter()
            result = matmul_free_cpu(x, w_ternary)
            end_time = time.perf_counter()
            times.append(end_time - start_time)
        
        avg_time = sum(times) / len(times)
        throughput = (batch_size * seq_len * hidden_size * hidden_size) / avg_time / 1e9
        
        print(f"  Average time: {avg_time*1000:.2f} ms")
        print(f"  Throughput: {throughput:.2f} GFLOPS")
        print(f"  Memory usage: {result.numel() * 4 / 1024**2:.1f} MB")


def run_inference_demo(model_config: dict):
    """Run a simple inference demonstration."""
    print("\nINFERENCE DEMONSTRATION")
    print("-" * 40)
    
    # TODO: Implement when CPU-optimized model is ready
    print("CPU-optimized model inference coming soon...")
    print("This will demonstrate:")
    print("- Loading pre-trained MatMul-free weights")
    print("- CPU-optimized text generation")
    print("- Memory usage optimization")
    print("- Cross-platform performance")


def main():
    parser = argparse.ArgumentParser(description="CPU-Optimized MatMul-Free Inference")
    
    parser.add_argument(
        "--config", 
        type=str, 
        default="configs/cpu_models.yaml",
        help="Path to CPU model configuration file"
    )
    parser.add_argument(
        "--model", 
        type=str, 
        default="cpu_small_370M",
        help="Model configuration to use"
    )
    parser.add_argument(
        "--benchmark", 
        action="store_true",
        help="Run CPU operation benchmarks"
    )
    parser.add_argument(
        "--demo", 
        action="store_true",
        help="Run inference demonstration"
    )
    parser.add_argument(
        "--num-runs", 
        type=int, 
        default=10,
        help="Number of benchmark runs"
    )
    
    args = parser.parse_args()
    
    # Print system information
    print_system_info()
    
    # Load model configuration
    try:
        model_config = load_cpu_config(args.config, args.model)
        print(f"\nLoaded model config: {args.model}")
        print(f"Hidden size: {model_config['hidden_size']}")
        print(f"Layers: {model_config['num_hidden_layers']}")
        print(f"CPU optimized: {model_config.get('cpu_optimized', False)}")
    except Exception as e:
        print(f"Error loading config: {e}")
        return 1
    
    # Run benchmarks if requested
    if args.benchmark:
        try:
            benchmark_cpu_operations(model_config, args.num_runs)
        except Exception as e:
            print(f"Benchmark failed: {e}")
    
    # Run demo if requested
    if args.demo:
        try:
            run_inference_demo(model_config)
        except Exception as e:
            print(f"Demo failed: {e}")
    
    print("\n✅ CPU inference script completed!")
    return 0


if __name__ == "__main__":
    exit(main())
