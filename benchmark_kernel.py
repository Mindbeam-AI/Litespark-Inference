#!/usr/bin/env python3
"""
Kernel Benchmark Script for Litespark-Inf

Reproduces the benchmark format from Microsoft BitNet src/README.md
Tests raw kernel performance on specific matrix sizes with timing statistics.
"""

import torch
import time
import platform
import os
import sys
import json
import statistics
from dataclasses import dataclass
from typing import List, Tuple, Dict
import argparse


@dataclass
class BenchmarkResult:
    matrix_size: str
    mean_ms: float
    std_ms: float
    num_runs: int


def get_cpu_info() -> Dict:
    """Get CPU information."""
    info = {
        'model': 'Unknown',
        'cores': os.cpu_count(),
        'arch': platform.machine(),
    }

    system = platform.system()
    if system == 'Linux':
        try:
            with open('/proc/cpuinfo') as f:
                for line in f:
                    if 'model name' in line:
                        info['model'] = line.split(':')[1].strip()
                        break
        except:
            pass
    elif system == 'Darwin':
        import subprocess
        try:
            result = subprocess.run(['sysctl', '-n', 'machdep.cpu.brand_string'],
                                    capture_output=True, text=True)
            info['model'] = result.stdout.strip()
        except:
            pass

    return info


def get_simd_features() -> Dict:
    """Detect SIMD features."""
    features = {
        'avx512f': False,
        'avx512vnni': False,
        'avx_vnni': False,
        'neon': False,
    }

    system = platform.system()
    machine = platform.machine().lower()

    if machine in ['aarch64', 'arm64']:
        features['neon'] = True
    elif system == 'Linux':
        try:
            with open('/proc/cpuinfo') as f:
                cpuinfo = f.read().lower()
                features['avx512f'] = 'avx512f' in cpuinfo
                features['avx512vnni'] = 'avx512vnni' in cpuinfo or 'avx512_vnni' in cpuinfo
                features['avx_vnni'] = 'avx_vnni' in cpuinfo
        except:
            pass
    elif system == 'Darwin':
        # Apple Silicon
        if machine == 'arm64':
            features['neon'] = True

    return features


def get_current_rss_mb() -> float:
    """Get current process RSS (resident set size) in MB."""
    system = platform.system()
    if system == 'Darwin':
        import subprocess
        try:
            result = subprocess.run(
                ['ps', '-o', 'rss=', '-p', str(os.getpid())],
                capture_output=True, text=True, timeout=5,
            )
            return int(result.stdout.strip()) / 1024
        except (subprocess.TimeoutExpired, ValueError):
            pass
    elif system == 'Linux':
        try:
            with open('/proc/self/status') as f:
                for line in f:
                    if line.startswith('VmRSS:'):
                        return int(line.split()[1]) / 1024
        except FileNotFoundError:
            pass
    import resource
    if system == 'Darwin':
        return resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / (1024 * 1024)
    return resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024


def load_kernel():
    """Load the appropriate kernel."""
    from litespark_inference.models import get_kernel, get_kernel_type
    kernel = get_kernel()
    kernel_type = get_kernel_type()
    return kernel, kernel_type


def benchmark_matmul(kernel, kernel_type: str, M: int, K: int, N: int,
                     num_threads: int, num_warmup: int = 5, num_runs: int = 20) -> BenchmarkResult:
    """
    Benchmark a single matrix multiplication.

    Args:
        kernel: The loaded kernel module
        kernel_type: Type of kernel (vnni, avx_vnni, neon)
        M: Batch/rows dimension
        K: Inner dimension
        N: Output dimension
        num_threads: Number of threads to use
        num_warmup: Warmup iterations
        num_runs: Benchmark iterations

    Returns:
        BenchmarkResult with timing statistics
    """
    # Determine padding based on kernel type
    if kernel_type == 'vnni':
        k_pad = 64
    elif kernel_type == 'avx_vnni':
        k_pad = 32
    else:
        k_pad = 16

    K_padded = ((K + k_pad - 1) // k_pad) * k_pad

    # Create test tensors
    # Activation: int8 quantized input
    x_int8 = torch.randint(-127, 127, (M, K_padded), dtype=torch.int8)
    x_scale = torch.rand(M, dtype=torch.float32) * 0.1

    # Weight: ternary int8 (-1, 0, +1)
    w_int8 = torch.randint(-1, 2, (N, K_padded), dtype=torch.int8)
    w_sum = w_int8.sum(dim=1, dtype=torch.int32)

    # Output buffer
    y_int32 = torch.zeros(M, N, dtype=torch.int32)

    matrix_size = f"[{M}, {K}] × [{K}, {N}]"

    # Select kernel function based on type
    if kernel_type == 'vnni':
        kernel_fn = kernel.matmul_free_vnni_v4_large_n
    elif kernel_type == 'avx_vnni':
        kernel_fn = kernel.matmul_free_avx_vnni_v4_large_n
    elif kernel_type in ('neon', 'graviton', 'neon_i8mm', 'sve', 'sme2'):
        # ARM kernels: use fused kernel with float input
        _prefix = {
            'neon': 'matmul_free_neon_sdot',
            'graviton': 'matmul_free_neon_sdot',
            'neon_i8mm': 'matmul_free_neon_i8mm',
            'sve': 'matmul_free_sve',
            'sme2': 'matmul_free_sme2',
        }[kernel_type]
        fused_suffix = 'v4_fused_wp' if (M <= 4 and num_threads > 1) else 'v4_fused'
        fused_fn = getattr(kernel, f'{_prefix}_{fused_suffix}')

        x_float = torch.randn(M, K, dtype=torch.float32)
        y_float = torch.zeros(M, N, dtype=torch.float32)
        bias = torch.Tensor()
        scale = 1.0

        # Warmup
        for _ in range(num_warmup):
            fused_fn(
                x_float.contiguous(), w_int8, w_sum, y_float, bias, scale,
                M, N, K, num_threads
            )

        # Benchmark
        times = []
        for _ in range(num_runs):
            y_float.zero_()
            start = time.perf_counter()
            fused_fn(
                x_float.contiguous(), w_int8, w_sum, y_float, bias, scale,
                M, N, K, num_threads
            )
            times.append((time.perf_counter() - start) * 1000)

        mean_ms = statistics.mean(times)
        std_ms = statistics.stdev(times) if len(times) > 1 else 0.0
        return BenchmarkResult(matrix_size, mean_ms, std_ms, num_runs)

    # Warmup for VNNI kernels
    for _ in range(num_warmup):
        y_int32.zero_()
        kernel_fn(x_int8, x_scale, w_int8, w_sum, y_int32, M, N, K, num_threads)

    # Benchmark
    times = []
    for _ in range(num_runs):
        y_int32.zero_()
        start = time.perf_counter()
        kernel_fn(x_int8, x_scale, w_int8, w_sum, y_int32, M, N, K, num_threads)
        times.append((time.perf_counter() - start) * 1000)

    mean_ms = statistics.mean(times)
    std_ms = statistics.stdev(times) if len(times) > 1 else 0.0

    return BenchmarkResult(matrix_size, mean_ms, std_ms, num_runs)


def run_full_benchmark(num_threads: int = 1, num_runs: int = 20) -> Dict:
    """
    Run the complete benchmark suite matching BitNet src/README.md format.

    Tests matrix sizes:
    - [1, 2048] × [2048, 2048]
    - [32, 2048] × [2048, 2048]
    - [128, 2048] × [2048, 2048]
    - [256, 2048] × [2048, 2048]
    - [512, 2048] × [2048, 2048]
    - [2048, 2048] × [2048, 2048]
    - [128, 2048] × [2048, 8192]
    - [128, 8192] × [8192, 2048]
    """

    # Matrix sizes to test (M, K, N)
    matrix_sizes = [
        (1, 2048, 2048),
        (32, 2048, 2048),
        (128, 2048, 2048),
        (256, 2048, 2048),
        (512, 2048, 2048),
        (2048, 2048, 2048),
        (128, 2048, 8192),
        (128, 8192, 2048),
    ]

    print("Loading kernel...")
    kernel, kernel_type = load_kernel()

    cpu_info = get_cpu_info()
    simd_features = get_simd_features()

    results = {
        'system': {
            'cpu_model': cpu_info['model'],
            'cpu_cores': cpu_info['cores'],
            'architecture': cpu_info['arch'],
            'simd_features': simd_features,
            'kernel_type': kernel_type,
        },
        'config': {
            'num_threads': num_threads,
            'num_runs': num_runs,
        },
        'matrix_benchmarks': []
    }

    print(f"\n{'='*70}")
    print("KERNEL PERFORMANCE BENCHMARK")
    print(f"{'='*70}")
    print(f"CPU: {cpu_info['model']}")
    print(f"Architecture: {cpu_info['arch']}")
    print(f"Kernel: {kernel_type}")
    print(f"Threads: {num_threads}")
    print(f"Runs per test: {num_runs}")
    print(f"{'='*70}\n")

    print(f"{'Matrix Size':<35} {'Time (ms)':<20}")
    print("-" * 55)

    for M, K, N in matrix_sizes:
        result = benchmark_matmul(kernel, kernel_type, M, K, N, num_threads, num_runs=num_runs)
        print(f"{result.matrix_size:<35} {result.mean_ms:.3f}±{result.std_ms:.3f}")

        results['matrix_benchmarks'].append({
            'matrix_size': result.matrix_size,
            'M': M,
            'K': K,
            'N': N,
            'mean_ms': result.mean_ms,
            'std_ms': result.std_ms,
        })

    print("-" * 55)

    return results


def run_thread_scaling_benchmark(thread_counts: List[int] = None, num_runs: int = 20) -> Dict:
    """
    Run benchmark at different thread counts to show scaling.
    """
    if thread_counts is None:
        thread_counts = [1, 2, 4, 8]

    print(f"\n{'='*70}")
    print("THREAD SCALING BENCHMARK")
    print(f"{'='*70}\n")

    # Test matrix: [128, 2048] × [2048, 2048] (typical workload)
    M, K, N = 128, 2048, 2048

    kernel, kernel_type = load_kernel()

    results = {
        'matrix_size': f"[{M}, {K}] × [{K}, {N}]",
        'thread_scaling': []
    }

    print(f"Matrix: [{M}, {K}] × [{K}, {N}]")
    print(f"\n{'Threads':<10} {'Time (ms)':<20} {'Speedup':<10}")
    print("-" * 40)

    baseline_time = None
    for threads in thread_counts:
        result = benchmark_matmul(kernel, kernel_type, M, K, N, threads, num_runs=num_runs)

        if baseline_time is None:
            baseline_time = result.mean_ms
            speedup = 1.0
        else:
            speedup = baseline_time / result.mean_ms

        print(f"{threads:<10} {result.mean_ms:.3f}±{result.std_ms:.3f}      {speedup:.2f}x")

        results['thread_scaling'].append({
            'threads': threads,
            'mean_ms': result.mean_ms,
            'std_ms': result.std_ms,
            'speedup': speedup,
        })

    print("-" * 40)

    return results


def run_inference_benchmark(model_name: str = 'bitnet-2b', num_threads: int = 4, num_tokens: int = 128) -> Dict:
    """
    Run end-to-end inference benchmark (pp128 + tg128 style).
    """
    from litespark_inference.models import load_ternary_model, get_kernel_type
    import gc

    print(f"\n{'='*70}")
    print("INFERENCE BENCHMARK (pp128 + tg128)")
    print(f"{'='*70}\n")

    # Load model
    print(f"Loading model: {model_name}")
    gc.collect()
    mem_before = get_current_rss_mb()
    model, tokenizer = load_ternary_model(model_name, num_threads=num_threads, mode='neon')
    gc.collect()
    mem_after = get_current_rss_mb()
    memory_mb_rss = mem_after - mem_before
    memory_mb_params = sum(p.nbytes for p in model.parameters()) / (1024 * 1024)
    memory_mb = memory_mb_rss if memory_mb_rss > memory_mb_params else memory_mb_params

    kernel_type = get_kernel_type()

    # Test prompt (approximately 128 tokens)
    prompt = "The quick brown fox jumps over the lazy dog. " * 20
    input_ids = tokenizer.encode(prompt, return_tensors='pt', max_length=128, truncation=True)
    actual_prompt_tokens = input_ids.shape[1]

    print(f"Kernel: {kernel_type}")
    print(f"Threads: {num_threads}")
    print(f"Memory: {memory_mb:.0f} MB")
    print(f"Prompt tokens: {actual_prompt_tokens}")
    print(f"Generate tokens: {num_tokens}")

    # Warmup
    print("\nWarming up...")
    with torch.no_grad():
        for _ in range(2):
            _ = model(input_ids)

    # Prompt processing (pp) benchmark
    print("\nPrompt Processing (pp):")
    pp_times = []
    with torch.no_grad():
        for _ in range(5):
            start = time.perf_counter()
            _ = model(input_ids)
            pp_times.append((time.perf_counter() - start) * 1000)

    pp_mean = statistics.mean(pp_times)
    pp_std = statistics.stdev(pp_times) if len(pp_times) > 1 else 0.0
    pp_throughput = actual_prompt_tokens / (pp_mean / 1000)
    print(f"  Time: {pp_mean:.2f}±{pp_std:.2f} ms")
    print(f"  Throughput: {pp_throughput:.2f} tokens/sec")

    # Token generation (tg) benchmark
    print(f"\nToken Generation (tg{num_tokens}):")
    tg_times = []
    with torch.no_grad():
        for _ in range(3):
            start = time.perf_counter()
            _ = model.generate(input_ids, max_new_tokens=num_tokens, temperature=0)
            tg_times.append(time.perf_counter() - start)

    tg_mean = statistics.mean(tg_times)
    tg_std = statistics.stdev(tg_times) if len(tg_times) > 1 else 0.0
    tg_throughput = num_tokens / tg_mean
    print(f"  Time: {tg_mean*1000:.2f}±{tg_std*1000:.2f} ms")
    print(f"  Throughput: {tg_throughput:.2f} tokens/sec")

    results = {
        'model': model_name,
        'kernel': kernel_type,
        'threads': num_threads,
        'memory_mb': memory_mb,
        'prompt_processing': {
            'tokens': actual_prompt_tokens,
            'mean_ms': pp_mean,
            'std_ms': pp_std,
            'throughput_tps': pp_throughput,
        },
        'token_generation': {
            'tokens': num_tokens,
            'mean_ms': tg_mean * 1000,
            'std_ms': tg_std * 1000,
            'throughput_tps': tg_throughput,
        }
    }

    del model, tokenizer
    gc.collect()

    return results


def run_pytorch_baseline(model_name: str = 'bitnet-2b', num_threads: int = 4,
                         num_tokens: int = 128, max_gen_tokens: int = 32) -> Dict:
    """
    Run PyTorch baseline inference benchmark for comparison.

    Uses standard HuggingFace AutoModelForCausalLM with the model's native
    dtype.  Generation is capped at max_gen_tokens to keep runtime reasonable.
    """
    from transformers import AutoModelForCausalLM, AutoTokenizer
    import transformers
    import gc

    transformers.logging.set_verbosity_error()

    # Map model keys to original HuggingFace repo names for the PyTorch
    # baseline.  We can't use AVAILABLE_MODELS because dev may resolve
    # bitnet-2b to a local converted checkpoint that AutoModelForCausalLM
    # can't load.
    _PYTORCH_HF_REPOS = {
        'bitnet-2b': 'microsoft/bitnet-b1.58-2B-4T-bf16',
        'falcon-edge-1b': 'tiiuae/Falcon-E-1B-Base',
        'falcon-edge-1b-instruct': 'tiiuae/Falcon-E-1B-Instruct',
        'falcon-edge-3b': 'tiiuae/Falcon-E-3B-Base',
        'falcon-edge-3b-instruct': 'tiiuae/Falcon-E-3B-Instruct',
    }

    hf_name = _PYTORCH_HF_REPOS.get(model_name)
    if hf_name is None:
        print(f"No PyTorch baseline available for model: {model_name}")
        return None

    print(f"\n{'='*70}")
    print("PYTORCH BASELINE BENCHMARK")
    print(f"{'='*70}\n")

    print(f"Model: {model_name} ({hf_name})")
    print(f"Threads: {num_threads}")

    torch.set_num_threads(num_threads)

    # Load model in its native dtype and measure memory
    print("\nLoading PyTorch model...")
    gc.collect()
    mem_before = get_current_rss_mb()

    try:
        try:
            model = AutoModelForCausalLM.from_pretrained(
                hf_name, trust_remote_code=True
            )
        except (ValueError, OSError, EnvironmentError):
            from transformers import LlamaForCausalLM
            model = LlamaForCausalLM.from_pretrained(hf_name)
        model.eval()
        tokenizer = AutoTokenizer.from_pretrained(hf_name, trust_remote_code=True)
    except Exception as e:
        print(f"  Failed to load PyTorch model: {e}")
        print("  Skipping PyTorch baseline.")
        return None

    gc.collect()
    mem_after = get_current_rss_mb()
    memory_mb_rss = mem_after - mem_before
    memory_mb_params = sum(p.nbytes for p in model.parameters()) / (1024 * 1024)
    memory_mb = memory_mb_rss if memory_mb_rss > memory_mb_params else memory_mb_params

    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    print(f"Memory: {memory_mb:.0f} MB")

    # Prepare input (same prompt as litespark benchmark)
    prompt = "The quick brown fox jumps over the lazy dog. " * 20
    input_ids = tokenizer.encode(prompt, return_tensors='pt', max_length=128, truncation=True)
    attention_mask = torch.ones_like(input_ids)
    actual_prompt_tokens = input_ids.shape[1]
    gen_tokens = min(num_tokens, max_gen_tokens)

    print(f"Prompt tokens: {actual_prompt_tokens}")
    print(f"Generate tokens: {gen_tokens}")

    # Prompt processing (TTFT) benchmark — no warmup, measuring cold-start
    # latency which is what a user actually experiences on first inference.
    print("\nPrompt Processing (TTFT):")
    pp_times = []
    with torch.no_grad():
        for _ in range(3):
            start = time.perf_counter()
            _ = model(input_ids, attention_mask=attention_mask)
            pp_times.append((time.perf_counter() - start) * 1000)

    pp_mean = statistics.mean(pp_times)
    pp_std = statistics.stdev(pp_times) if len(pp_times) > 1 else 0.0
    pp_throughput = actual_prompt_tokens / (pp_mean / 1000)
    print(f"  Time: {pp_mean:.2f}±{pp_std:.2f} ms")
    print(f"  Throughput: {pp_throughput:.2f} tokens/sec")

    # Token generation benchmark
    print(f"\nToken Generation (tg{gen_tokens}):")
    tg_times = []
    with torch.no_grad():
        start = time.perf_counter()
        _ = model.generate(
            input_ids, attention_mask=attention_mask,
            max_new_tokens=gen_tokens, do_sample=False,
        )
        tg_times.append(time.perf_counter() - start)

    tg_mean = statistics.mean(tg_times)
    tg_throughput = gen_tokens / tg_mean
    print(f"  Time: {tg_mean*1000:.2f} ms")
    print(f"  Throughput: {tg_throughput:.2f} tokens/sec")

    results = {
        'model': model_name,
        'hf_name': hf_name,
        'threads': num_threads,
        'memory_mb': memory_mb,
        'prompt_processing': {
            'tokens': actual_prompt_tokens,
            'mean_ms': pp_mean,
            'std_ms': pp_std,
            'throughput_tps': pp_throughput,
        },
        'token_generation': {
            'tokens': gen_tokens,
            'mean_ms': tg_mean * 1000,
            'throughput_tps': tg_throughput,
        }
    }

    del model, tokenizer
    gc.collect()

    return results


def print_comparison_table(litespark_results: Dict, pytorch_results: Dict) -> None:
    """Print a side-by-side comparison table of Litespark vs PyTorch."""
    ls = litespark_results
    pt = pytorch_results

    ls_mem = ls['memory_mb']
    pt_mem = pt['memory_mb']
    ls_ttft = ls['prompt_processing']['mean_ms']
    pt_ttft = pt['prompt_processing']['mean_ms']
    ls_tps = ls['token_generation']['throughput_tps']
    pt_tps = pt['token_generation']['throughput_tps']

    mem_speedup = pt_mem / ls_mem if ls_mem > 0 else 0
    ttft_speedup = pt_ttft / ls_ttft if ls_ttft > 0 else 0
    tps_speedup = ls_tps / pt_tps if pt_tps > 0 else 0

    print(f"\n{'='*70}")
    print("COMPARISON: Litespark vs PyTorch")
    print(f"{'='*70}\n")

    print(f"{'Metric':<22} {'PyTorch':>12} {'Litespark':>12} {'Speedup':>10}")
    print("-" * 58)
    print(f"{'Memory (MB)':<22} {pt_mem:>12,.0f} {ls_mem:>12,.0f} {mem_speedup:>9.1f}x")
    print(f"{'TTFT (ms)':<22} {pt_ttft:>12,.1f} {ls_ttft:>12,.1f} {ttft_speedup:>9.1f}x")
    print(f"{'Throughput (tok/s)':<22} {pt_tps:>12.2f} {ls_tps:>12.2f} {tps_speedup:>9.1f}x")
    print("-" * 58)


def main():
    parser = argparse.ArgumentParser(description='Litespark-Inf Kernel Benchmark')
    parser.add_argument('--threads', '-t', type=int, default=1,
                        help='Number of threads (default: 1)')
    parser.add_argument('--runs', '-r', type=int, default=20,
                        help='Number of runs per benchmark (default: 20)')
    parser.add_argument('--scaling', action='store_true',
                        help='Run thread scaling benchmark')
    parser.add_argument('--inference', action='store_true',
                        help='Run inference benchmark (pp128 + tg128)')
    parser.add_argument('--all', action='store_true',
                        help='Run all benchmarks')
    parser.add_argument('--output', '-o', type=str,
                        help='Output JSON file for results')
    parser.add_argument('--model', '-m', type=str, default='bitnet-2b',
                        help='Model name for inference benchmark (default: bitnet-2b)')
    parser.add_argument('--pytorch', action='store_true',
                        help='Include PyTorch baseline comparison (requires extra RAM, slow)')

    args = parser.parse_args()

    all_results = {
        'system': {},
        'benchmarks': {}
    }

    # Get system info
    cpu_info = get_cpu_info()
    simd_features = get_simd_features()
    all_results['system'] = {
        'cpu_model': cpu_info['model'],
        'cpu_cores': cpu_info['cores'],
        'architecture': cpu_info['arch'],
        'simd_features': simd_features,
    }

    # Run matrix benchmark (always)
    matrix_results = run_full_benchmark(num_threads=args.threads, num_runs=args.runs)
    all_results['benchmarks']['matrix'] = matrix_results

    # Run thread scaling if requested
    if args.scaling or args.all:
        max_threads = min(os.cpu_count() or 8, 16)
        thread_counts = [1, 2, 4, 8]
        if max_threads > 8:
            thread_counts.extend([12, 16])
        thread_counts = [t for t in thread_counts if t <= max_threads]

        scaling_results = run_thread_scaling_benchmark(thread_counts, num_runs=args.runs)
        all_results['benchmarks']['thread_scaling'] = scaling_results

    # Run inference benchmark if requested.
    # When --pytorch is set, run PyTorch first so it doesn't benefit from
    # warm caches left behind by the Litespark benchmark.
    if args.inference or args.all:
        pytorch_results = None
        if args.pytorch:
            pytorch_results = run_pytorch_baseline(
                model_name=args.model, num_threads=args.threads
            )
            if pytorch_results is not None:
                all_results['benchmarks']['pytorch_baseline'] = pytorch_results

        inference_results = run_inference_benchmark(model_name=args.model, num_threads=args.threads)
        all_results['benchmarks']['inference'] = inference_results

        if pytorch_results is not None:
            print_comparison_table(inference_results, pytorch_results)

    elif args.pytorch:
        print("Warning: --pytorch requires --inference or --all to run.")

    # Save results if output file specified
    if args.output:
        with open(args.output, 'w') as f:
            json.dump(all_results, f, indent=2)
        print(f"\nResults saved to: {args.output}")

    print("\n" + "="*70)
    print("BENCHMARK COMPLETE")
    print("="*70)


if __name__ == '__main__':
    main()
