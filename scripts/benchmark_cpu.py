#!/usr/bin/env python3
"""
Comprehensive CPU Benchmarking for MatMul-Free Operations

Compares MatMul-free operations against traditional PyTorch CPU operations
across different architectures and model sizes.
"""

import argparse
import time
import torch
import numpy as np
import matplotlib.pyplot as plt
import json
from pathlib import Path
import psutil
import platform
from typing import Dict, List, Tuple

from src.cpu_ops import get_cpu_info, detect_simd_support
from src.cpu_ops.matmul_free_cpu import matmul_free_cpu, weight_quant_cpu


class CPUBenchmark:
    """Comprehensive CPU benchmarking suite."""
    
    def __init__(self, output_dir: str = "benchmark_results"):
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(exist_ok=True)
        
        self.cpu_info = get_cpu_info()
        self.simd_caps = detect_simd_support()
        self.results = {}
        
    def benchmark_single_operation(
        self, 
        M: int, 
        N: int, 
        K: int, 
        num_runs: int = 100
    ) -> Dict[str, float]:
        """Benchmark a single MatMul operation size."""
        
        # Create test data
        x = torch.randn(M, K, dtype=torch.float32)
        w_full = torch.randn(N, K, dtype=torch.float32)
        w_ternary = weight_quant_cpu(w_full)
        
        results = {}
        
        # Benchmark 1: Traditional PyTorch F.linear
        print(f"  Testing PyTorch F.linear ({M}x{K} @ {K}x{N})...")
        times_pytorch = []
        
        # Warmup
        for _ in range(5):
            _ = torch.nn.functional.linear(x, w_full)
        
        # Benchmark
        for _ in range(num_runs):
            start = time.perf_counter()
            result_pytorch = torch.nn.functional.linear(x, w_full)
            end = time.perf_counter()
            times_pytorch.append(end - start)
        
        results['pytorch_time'] = np.mean(times_pytorch)
        results['pytorch_std'] = np.std(times_pytorch)
        
        # Benchmark 2: MatMul-free CPU operation
        print(f"  Testing MatMul-free CPU ({M}x{K} @ {K}x{N})...")
        times_matmul_free = []
        
        # Warmup
        for _ in range(5):
            _ = matmul_free_cpu(x, w_ternary)
        
        # Benchmark
        for _ in range(num_runs):
            start = time.perf_counter()
            result_matmul_free = matmul_free_cpu(x, w_ternary)
            end = time.perf_counter()
            times_matmul_free.append(end - start)
        
        results['matmul_free_time'] = np.mean(times_matmul_free)
        results['matmul_free_std'] = np.std(times_matmul_free)
        
        # Calculate metrics
        results['speedup'] = results['pytorch_time'] / results['matmul_free_time']
        results['pytorch_gflops'] = (2 * M * N * K) / results['pytorch_time'] / 1e9
        results['matmul_free_gflops'] = (M * N * K) / results['matmul_free_time'] / 1e9  # Only add/subtract
        
        # Memory usage (approximate)
        pytorch_memory = (M * K + N * K + M * N) * 4  # float32
        matmul_free_memory = (M * K + N * K * 0.25 + M * N) * 4  # ternary weights = 2 bits
        results['memory_reduction'] = pytorch_memory / matmul_free_memory
        
        return results
    
    def benchmark_model_sizes(self) -> Dict[str, Dict]:
        """Benchmark different model sizes."""
        
        # Test configurations: (M, N, K) representing different scenarios
        test_configs = {
            'single_token': (1, 1024, 1024),      # Single token inference
            'small_batch': (32, 1024, 1024),      # Small batch
            'medium_batch': (128, 1024, 1024),    # Medium batch
            'large_batch': (512, 1024, 1024),     # Large batch
            'wide_layer': (128, 4096, 1024),      # Wide output layer
            'deep_layer': (128, 1024, 4096),      # Deep input layer
            'square_large': (256, 2048, 2048),    # Large square matrix
        }
        
        results = {}
        
        for config_name, (M, N, K) in test_configs.items():
            print(f"\nBenchmarking {config_name}: {M}x{K} @ {K}x{N}")
            
            try:
                config_results = self.benchmark_single_operation(M, N, K, num_runs=50)
                results[config_name] = {
                    'dimensions': (M, N, K),
                    **config_results
                }
                
                print(f"  PyTorch: {config_results['pytorch_time']*1000:.2f}ms "
                      f"({config_results['pytorch_gflops']:.1f} GFLOPS)")
                print(f"  MatMul-free: {config_results['matmul_free_time']*1000:.2f}ms "
                      f"({config_results['matmul_free_gflops']:.1f} GFLOPS)")
                print(f"  Speedup: {config_results['speedup']:.2f}x")
                print(f"  Memory reduction: {config_results['memory_reduction']:.1f}x")
                
            except Exception as e:
                print(f"  Error: {e}")
                results[config_name] = {'error': str(e)}
        
        return results
    
    def benchmark_threading(self) -> Dict[str, float]:
        """Benchmark different threading configurations."""
        print("\nBenchmarking threading performance...")
        
        # Test with different thread counts
        max_threads = psutil.cpu_count()
        thread_counts = [1, 2, 4, max_threads // 2, max_threads]
        thread_counts = [t for t in thread_counts if t <= max_threads and t > 0]
        
        M, N, K = 256, 1024, 1024
        x = torch.randn(M, K)
        w = weight_quant_cpu(torch.randn(N, K))
        
        results = {}
        
        for num_threads in thread_counts:
            print(f"  Testing with {num_threads} threads...")
            
            # Set thread count (this would be passed to C++ kernels)
            torch.set_num_threads(num_threads)
            
            times = []
            for _ in range(20):
                start = time.perf_counter()
                _ = matmul_free_cpu(x, w)
                end = time.perf_counter()
                times.append(end - start)
            
            avg_time = np.mean(times)
            results[f'{num_threads}_threads'] = avg_time
            
            print(f"    Average time: {avg_time*1000:.2f}ms")
        
        return results
    
    def save_results(self, filename: str = None):
        """Save benchmark results to JSON file."""
        if filename is None:
            timestamp = time.strftime("%Y%m%d_%H%M%S")
            arch = self.simd_caps.architecture
            filename = f"cpu_benchmark_{arch}_{timestamp}.json"
        
        filepath = self.output_dir / filename
        
        # Prepare results with system info
        full_results = {
            'system_info': {
                'cpu_info': self.cpu_info,
                'simd_capabilities': {
                    'architecture': self.simd_caps.architecture,
                    'has_sse': self.simd_caps.has_sse,
                    'has_avx': self.simd_caps.has_avx,
                    'has_avx2': self.simd_caps.has_avx2,
                    'has_avx512': self.simd_caps.has_avx512,
                    'has_neon': self.simd_caps.has_neon,
                    'vector_width': self.simd_caps.vector_width,
                    'optimal_block_size': self.simd_caps.optimal_block_size,
                },
                'memory_gb': psutil.virtual_memory().total / (1024**3),
                'timestamp': time.strftime("%Y-%m-%d %H:%M:%S")
            },
            'benchmark_results': self.results
        }
        
        with open(filepath, 'w') as f:
            json.dump(full_results, f, indent=2)
        
        print(f"\n✅ Results saved to: {filepath}")
        return filepath
    
    def plot_results(self):
        """Create visualization plots of benchmark results."""
        if 'model_sizes' not in self.results:
            print("No model size results to plot")
            return
        
        model_results = self.results['model_sizes']
        
        # Extract data for plotting
        configs = []
        speedups = []
        memory_reductions = []
        
        for config_name, results in model_results.items():
            if 'error' not in results:
                configs.append(config_name)
                speedups.append(results['speedup'])
                memory_reductions.append(results['memory_reduction'])
        
        # Create plots
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))
        
        # Speedup plot
        bars1 = ax1.bar(configs, speedups, color='skyblue', alpha=0.7)
        ax1.set_title('MatMul-Free vs PyTorch CPU Speedup')
        ax1.set_ylabel('Speedup (x)')
        ax1.set_xlabel('Test Configuration')
        ax1.tick_params(axis='x', rotation=45)
        ax1.axhline(y=1.0, color='red', linestyle='--', alpha=0.5, label='No speedup')
        ax1.legend()
        
        # Add value labels on bars
        for bar, speedup in zip(bars1, speedups):
            height = bar.get_height()
            ax1.text(bar.get_x() + bar.get_width()/2., height + 0.05,
                    f'{speedup:.1f}x', ha='center', va='bottom')
        
        # Memory reduction plot
        bars2 = ax2.bar(configs, memory_reductions, color='lightgreen', alpha=0.7)
        ax2.set_title('Memory Usage Reduction')
        ax2.set_ylabel('Memory Reduction (x)')
        ax2.set_xlabel('Test Configuration')
        ax2.tick_params(axis='x', rotation=45)
        
        # Add value labels on bars
        for bar, reduction in zip(bars2, memory_reductions):
            height = bar.get_height()
            ax2.text(bar.get_x() + bar.get_width()/2., height + 0.1,
                    f'{reduction:.1f}x', ha='center', va='bottom')
        
        plt.tight_layout()
        
        # Save plot
        plot_path = self.output_dir / f"cpu_benchmark_{self.simd_caps.architecture}.png"
        plt.savefig(plot_path, dpi=300, bbox_inches='tight')
        print(f"📊 Plot saved to: {plot_path}")
        
        plt.show()
    
    def run_full_benchmark(self):
        """Run the complete benchmark suite."""
        print("🚀 Starting comprehensive CPU benchmark...")
        print(f"Architecture: {self.simd_caps.architecture}")
        print(f"Vector width: {self.simd_caps.vector_width}")
        
        # Run benchmarks
        self.results['model_sizes'] = self.benchmark_model_sizes()
        self.results['threading'] = self.benchmark_threading()
        
        # Save and visualize results
        self.save_results()
        self.plot_results()
        
        print("\n🎉 Benchmark completed!")


def main():
    parser = argparse.ArgumentParser(description="CPU MatMul-Free Benchmark")
    parser.add_argument("--output-dir", default="benchmark_results", help="Output directory")
    parser.add_argument("--quick", action="store_true", help="Run quick benchmark (fewer iterations)")
    
    args = parser.parse_args()
    
    benchmark = CPUBenchmark(args.output_dir)
    benchmark.run_full_benchmark()


if __name__ == "__main__":
    main()
