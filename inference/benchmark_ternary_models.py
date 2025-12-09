#!/usr/bin/env python3
"""
Benchmark Pre-trained Ternary Models - Complete Inference Test

Compares our optimized kernels (VNNI/Graviton) vs PyTorch FP32 baseline.

Metrics collected:
- TTFT (Time-to-First-Token): Critical for interactive applications
- Tokens/sec: Generation throughput
- Perplexity: Model quality on WikiText-2
- Memory usage
- Batch size scaling (M=1, 8, 32, 128)

Supports:
- x86_64 with AVX-512 VNNI
- ARM64 with NEON SDOT (Graviton)
"""

import torch
import torch.nn.functional as F
import numpy as np
import time
import gc
import platform
import argparse
import json
from pathlib import Path
from typing import Dict, List, Optional, Tuple
from dataclasses import dataclass

from ternary_models import load_ternary_model, list_models, AVAILABLE_MODELS, get_arch_info


# ============================================================================
# Memory Tracking
# ============================================================================

def get_memory_mb() -> float:
    """Get process RSS in MB."""
    try:
        with open('/proc/self/status', 'r') as f:
            for line in f:
                if line.startswith('VmRSS:'):
                    return int(line.split()[1]) / 1024.0
    except:
        pass
    try:
        import resource
        usage = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
        if platform.system() == 'Darwin':
            return usage / (1024 * 1024)
        return usage / 1024
    except:
        return 0.0


# ============================================================================
# TTFT Benchmark
# ============================================================================

def benchmark_ttft(
    model,
    tokenizer,
    prompts: List[str],
    warmup: int = 3,
    num_runs: int = 5
) -> Dict:
    """
    Benchmark Time-to-First-Token (TTFT).

    TTFT = time from prompt submission to first output token.
    This is the prefill latency - critical for interactive applications.
    """
    # Warmup
    for _ in range(warmup):
        input_ids = tokenizer.encode(prompts[0], return_tensors='pt')
        with torch.no_grad():
            _ = model(input_ids)

    # Collect measurements
    measurements = []

    for run in range(num_runs):
        for prompt in prompts:
            input_ids = tokenizer.encode(prompt, return_tensors='pt')
            prompt_tokens = input_ids.shape[1]

            gc.collect()

            start = time.perf_counter()
            with torch.no_grad():
                logits = model(input_ids)
                # Get first token (completes TTFT)
                _ = logits[:, -1, :].argmax(dim=-1)
            end = time.perf_counter()

            ttft_ms = (end - start) * 1000
            measurements.append({
                'ttft_ms': ttft_ms,
                'prompt_tokens': prompt_tokens,
                'prompt': prompt[:50]
            })

    all_ttft = [m['ttft_ms'] for m in measurements]
    all_ttft_sorted = sorted(all_ttft)

    return {
        'mean_ms': np.mean(all_ttft),
        'std_ms': np.std(all_ttft),
        'min_ms': min(all_ttft),
        'max_ms': max(all_ttft),
        'p50_ms': all_ttft_sorted[len(all_ttft) // 2],
        'p95_ms': all_ttft_sorted[int(len(all_ttft) * 0.95)],
        'p99_ms': all_ttft_sorted[min(int(len(all_ttft) * 0.99), len(all_ttft) - 1)],
        'num_samples': len(all_ttft),
        'by_prompt_length': measurements[:len(prompts)]  # First run per prompt
    }


# ============================================================================
# Throughput Benchmark
# ============================================================================

def benchmark_throughput(
    model,
    tokenizer,
    prompt: str,
    num_tokens: int = 50,
    warmup: int = 3,
    num_runs: int = 5
) -> Dict:
    """
    Benchmark token generation throughput.

    Measures tokens/sec during autoregressive generation.
    """
    input_ids = tokenizer.encode(prompt, return_tensors='pt')

    # Warmup
    for _ in range(warmup):
        ids = input_ids.clone()
        with torch.no_grad():
            for _ in range(5):
                logits = model(ids)
                next_token = logits[:, -1, :].argmax(dim=-1, keepdim=True)
                ids = torch.cat([ids, next_token], dim=1)

    # Benchmark
    results = []
    for run in range(num_runs):
        ids = input_ids.clone()
        gc.collect()

        # Measure TTFT separately
        start_ttft = time.perf_counter()
        with torch.no_grad():
            logits = model(ids)
            next_token = logits[:, -1, :].argmax(dim=-1, keepdim=True)
            ids = torch.cat([ids, next_token], dim=1)
        ttft = time.perf_counter() - start_ttft

        # Measure remaining tokens
        start_gen = time.perf_counter()
        with torch.no_grad():
            for _ in range(num_tokens - 1):
                logits = model(ids)
                next_token = logits[:, -1, :].argmax(dim=-1, keepdim=True)
                ids = torch.cat([ids, next_token], dim=1)
        gen_time = time.perf_counter() - start_gen

        total_time = ttft + gen_time
        # Throughput excludes TTFT (measures decode speed)
        tokens_per_sec = (num_tokens - 1) / gen_time if gen_time > 0 else 0

        results.append({
            'ttft_ms': ttft * 1000,
            'gen_time_ms': gen_time * 1000,
            'total_time_ms': total_time * 1000,
            'tokens_per_sec': tokens_per_sec,
            'num_tokens': num_tokens
        })

    return {
        'mean_tokens_per_sec': np.mean([r['tokens_per_sec'] for r in results]),
        'mean_ttft_ms': np.mean([r['ttft_ms'] for r in results]),
        'mean_total_ms': np.mean([r['total_time_ms'] for r in results]),
        'std_tokens_per_sec': np.std([r['tokens_per_sec'] for r in results]),
        'num_tokens': num_tokens,
        'runs': results
    }


# ============================================================================
# Batch Size Benchmark
# ============================================================================

def benchmark_batch_sizes(
    model,
    tokenizer,
    prompt: str,
    batch_sizes: List[int] = [1, 8, 32, 128],
    warmup: int = 2,
    num_runs: int = 3
) -> Dict:
    """
    Benchmark performance across different batch sizes.

    Tests M=1 (interactive), M=8 (small batch), M=32 (medium), M=128 (throughput).
    """
    input_ids = tokenizer.encode(prompt, return_tensors='pt')
    seq_len = input_ids.shape[1]

    results = {}

    for batch_size in batch_sizes:
        # Create batched input
        batched_ids = input_ids.repeat(batch_size, 1)

        # Warmup
        for _ in range(warmup):
            with torch.no_grad():
                _ = model(batched_ids)

        # Benchmark forward pass (prefill)
        times = []
        for _ in range(num_runs):
            gc.collect()
            start = time.perf_counter()
            with torch.no_grad():
                logits = model(batched_ids)
                _ = logits[:, -1, :].argmax(dim=-1)
            end = time.perf_counter()
            times.append((end - start) * 1000)

        mean_time = np.mean(times)
        results[f'M={batch_size}'] = {
            'batch_size': batch_size,
            'seq_len': seq_len,
            'mean_time_ms': mean_time,
            'std_time_ms': np.std(times),
            'throughput_samples_per_sec': batch_size / (mean_time / 1000),
            'time_per_sample_ms': mean_time / batch_size
        }

    return results


# ============================================================================
# Perplexity Benchmark
# ============================================================================

def benchmark_perplexity(
    model,
    tokenizer,
    texts: List[str] = None,
    max_length: int = 512,
    num_samples: int = 50
) -> Dict:
    """
    Compute perplexity on WikiText-2 validation set.
    """
    # Load WikiText-2 if no texts provided
    if texts is None:
        try:
            from datasets import load_dataset
            dataset = load_dataset('wikitext', 'wikitext-2-raw-v1', split='validation')
            texts = [t for t in dataset['text'] if len(t.strip()) > 100][:num_samples]
        except Exception as e:
            print(f"  Warning: Could not load WikiText-2: {e}")
            texts = [
                "The quick brown fox jumps over the lazy dog. " * 20,
                "Machine learning is transforming the world of technology. " * 20,
            ]

    total_loss = 0.0
    total_tokens = 0

    for text in texts:
        encodings = tokenizer(text, return_tensors='pt', truncation=True, max_length=max_length)
        input_ids = encodings.input_ids

        if input_ids.shape[1] < 2:
            continue

        with torch.no_grad():
            logits = model(input_ids)
            shift_logits = logits[..., :-1, :].contiguous()
            shift_labels = input_ids[..., 1:].contiguous()
            loss = F.cross_entropy(
                shift_logits.view(-1, shift_logits.size(-1)),
                shift_labels.view(-1)
            )

        num_tokens = input_ids.shape[1] - 1
        total_loss += loss.item() * num_tokens
        total_tokens += num_tokens

    if total_tokens == 0:
        return {'perplexity': float('inf'), 'num_tokens': 0}

    perplexity = np.exp(total_loss / total_tokens)
    return {
        'perplexity': perplexity,
        'num_tokens': total_tokens,
        'num_samples': len(texts)
    }


# ============================================================================
# PyTorch Baseline Comparison
# ============================================================================

def load_pytorch_baseline(model_key: str):
    """Load original HuggingFace model for comparison."""
    from transformers import AutoModelForCausalLM, AutoTokenizer

    hf_name = AVAILABLE_MODELS[model_key][0]

    print(f"  Loading PyTorch baseline: {hf_name}")
    model = AutoModelForCausalLM.from_pretrained(hf_name, trust_remote_code=True)
    model.eval()

    tokenizer = AutoTokenizer.from_pretrained(hf_name, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    return model, tokenizer


def compare_vs_baseline(
    ternary_model,
    baseline_model,
    tokenizer,
    prompts: List[str],
    num_tokens: int = 30,
    num_runs: int = 3
) -> Dict:
    """
    Compare our ternary kernels vs PyTorch FP32 baseline.
    """
    results = {
        'ternary': {},
        'pytorch_fp32': {},
        'speedup': {}
    }

    # Benchmark ternary model
    print("  Benchmarking ternary model...")
    ternary_ttft = benchmark_ttft(ternary_model, tokenizer, prompts, num_runs=num_runs)
    ternary_throughput = benchmark_throughput(ternary_model, tokenizer, prompts[1], num_tokens=num_tokens, num_runs=num_runs)

    results['ternary'] = {
        'ttft_mean_ms': ternary_ttft['mean_ms'],
        'ttft_p95_ms': ternary_ttft['p95_ms'],
        'tokens_per_sec': ternary_throughput['mean_tokens_per_sec']
    }

    # Benchmark PyTorch baseline
    print("  Benchmarking PyTorch FP32 baseline...")
    baseline_ttft = benchmark_ttft(baseline_model, tokenizer, prompts, num_runs=num_runs)
    baseline_throughput = benchmark_throughput(baseline_model, tokenizer, prompts[1], num_tokens=num_tokens, num_runs=num_runs)

    results['pytorch_fp32'] = {
        'ttft_mean_ms': baseline_ttft['mean_ms'],
        'ttft_p95_ms': baseline_ttft['p95_ms'],
        'tokens_per_sec': baseline_throughput['mean_tokens_per_sec']
    }

    # Compute speedups
    results['speedup'] = {
        'ttft': baseline_ttft['mean_ms'] / ternary_ttft['mean_ms'],
        'throughput': ternary_throughput['mean_tokens_per_sec'] / baseline_throughput['mean_tokens_per_sec']
    }

    return results


# ============================================================================
# Sample Generation
# ============================================================================

def generate_sample(model, tokenizer, prompt: str, max_tokens: int = 50) -> Tuple[str, float]:
    """Generate sample text and return output + time."""
    input_ids = tokenizer.encode(prompt, return_tensors='pt')

    start = time.perf_counter()
    with torch.no_grad():
        for _ in range(max_tokens):
            logits = model(input_ids)
            next_token = logits[:, -1, :].argmax(dim=-1, keepdim=True)
            input_ids = torch.cat([input_ids, next_token], dim=1)

            if tokenizer.eos_token_id and next_token.item() == tokenizer.eos_token_id:
                break
    elapsed = time.perf_counter() - start

    output = tokenizer.decode(input_ids[0], skip_special_tokens=True)
    return output, elapsed * 1000


# ============================================================================
# Main Benchmark
# ============================================================================

def run_full_benchmark(
    model_key: str,
    num_threads: int = None,
    max_tokens: int = 50,
    num_runs: int = 5,
    batch_sizes: List[int] = [1, 8, 32],
    compare_baseline: bool = True,
    skip_perplexity: bool = False,
    output_file: str = None
):
    """
    Run complete benchmark suite for a ternary model.
    """
    arch_info = get_arch_info()

    print("=" * 80)
    print(f"TERNARY MODEL INFERENCE BENCHMARK")
    print("=" * 80)
    print(f"Model: {model_key}")
    print(f"Platform: {platform.system()} {platform.machine()}")
    print(f"Kernel: {arch_info['kernel_type'].upper()}")
    print(f"PyTorch: {torch.__version__}")
    print(f"Threads: {num_threads or torch.get_num_threads()}")
    print("=" * 80)

    # Load ternary model
    print("\n[1/7] Loading ternary model...")
    model, tokenizer = load_ternary_model(model_key, num_threads)
    mem_model = get_memory_mb()
    print(f"  Model loaded. Memory: {mem_model:.1f} MB")

    # Test prompts
    prompts = [
        "The",
        "The future of AI is",
        "In a world where technology has advanced beyond imagination,",
        "Machine learning and artificial intelligence are transforming industries worldwide",
    ]

    results = {
        'model': model_key,
        'platform': f"{platform.system()} {platform.machine()}",
        'kernel': arch_info['kernel_type'],
        'pytorch_version': torch.__version__,
        'num_threads': num_threads or torch.get_num_threads(),
        'memory_mb': mem_model,
    }

    # TTFT Benchmark
    print("\n[2/7] Benchmarking Time-to-First-Token (TTFT)...")
    ttft_results = benchmark_ttft(model, tokenizer, prompts, num_runs=num_runs)
    results['ttft'] = ttft_results

    print(f"  Mean TTFT:  {ttft_results['mean_ms']:.2f} ms")
    print(f"  P50 TTFT:   {ttft_results['p50_ms']:.2f} ms")
    print(f"  P95 TTFT:   {ttft_results['p95_ms']:.2f} ms")
    print(f"  P99 TTFT:   {ttft_results['p99_ms']:.2f} ms")

    print("\n  TTFT by prompt length:")
    for m in ttft_results['by_prompt_length']:
        print(f"    {m['prompt_tokens']:3d} tokens -> {m['ttft_ms']:.2f} ms")

    # Throughput Benchmark
    print(f"\n[3/7] Benchmarking throughput ({max_tokens} tokens)...")
    throughput_results = benchmark_throughput(model, tokenizer, prompts[1], num_tokens=max_tokens, num_runs=num_runs)
    results['throughput'] = throughput_results

    print(f"  Tokens/sec: {throughput_results['mean_tokens_per_sec']:.1f} +/- {throughput_results['std_tokens_per_sec']:.1f}")
    print(f"  Mean TTFT:  {throughput_results['mean_ttft_ms']:.2f} ms")
    print(f"  Mean total: {throughput_results['mean_total_ms']:.2f} ms")

    # Batch Size Benchmark
    print(f"\n[4/7] Benchmarking batch sizes {batch_sizes}...")
    batch_results = benchmark_batch_sizes(model, tokenizer, prompts[1], batch_sizes=batch_sizes, num_runs=num_runs)
    results['batch_sizes'] = batch_results

    print(f"\n  {'Batch':<10} {'Time (ms)':<12} {'Throughput':<15} {'Per-sample':<12}")
    print(f"  {'-'*50}")
    for key, data in batch_results.items():
        print(f"  {key:<10} {data['mean_time_ms']:<12.2f} {data['throughput_samples_per_sec']:<15.1f} {data['time_per_sample_ms']:<12.2f}")

    # Perplexity Benchmark
    if not skip_perplexity:
        print("\n[5/7] Benchmarking perplexity on WikiText-2...")
        ppl_results = benchmark_perplexity(model, tokenizer)
        results['perplexity'] = ppl_results
        print(f"  Perplexity: {ppl_results['perplexity']:.2f}")
        print(f"  Tokens evaluated: {ppl_results['num_tokens']}")
    else:
        print("\n[5/7] Perplexity benchmark skipped")
        results['perplexity'] = {'perplexity': float('nan'), 'num_tokens': 0}

    # PyTorch Baseline Comparison
    if compare_baseline:
        print("\n[6/7] Comparing vs PyTorch FP32 baseline...")
        try:
            baseline_model, _ = load_pytorch_baseline(model_key)
            comparison = compare_vs_baseline(model, baseline_model, tokenizer, prompts, num_tokens=max_tokens, num_runs=num_runs)
            results['vs_baseline'] = comparison

            print(f"\n  {'Metric':<20} {'Ternary':<15} {'PyTorch FP32':<15} {'Speedup':<10}")
            print(f"  {'-'*60}")
            print(f"  {'TTFT (ms)':<20} {comparison['ternary']['ttft_mean_ms']:<15.2f} {comparison['pytorch_fp32']['ttft_mean_ms']:<15.2f} {comparison['speedup']['ttft']:<10.2f}x")
            print(f"  {'Tokens/sec':<20} {comparison['ternary']['tokens_per_sec']:<15.1f} {comparison['pytorch_fp32']['tokens_per_sec']:<15.1f} {comparison['speedup']['throughput']:<10.2f}x")

            del baseline_model
            gc.collect()
        except Exception as e:
            print(f"  Warning: Could not load baseline: {e}")
            results['vs_baseline'] = None
    else:
        print("\n[6/7] Baseline comparison skipped")
        results['vs_baseline'] = None

    # Sample Generation
    print("\n[7/7] Sample generation...")
    sample_prompt = "The future of artificial intelligence is"
    output, gen_time = generate_sample(model, tokenizer, sample_prompt, max_tokens=50)
    results['sample'] = {
        'prompt': sample_prompt,
        'output': output,
        'time_ms': gen_time
    }
    print(f"  Prompt: {sample_prompt}")
    print(f"  Output: {output}")
    print(f"  Time: {gen_time:.1f} ms")

    # Summary
    print("\n" + "=" * 80)
    print("SUMMARY")
    print("=" * 80)
    print(f"Model:        {model_key}")
    print(f"Kernel:       {arch_info['kernel_type'].upper()}")
    print(f"Memory:       {mem_model:.1f} MB")
    print(f"TTFT:         {ttft_results['mean_ms']:.2f} ms (P95: {ttft_results['p95_ms']:.2f} ms)")
    print(f"Throughput:   {throughput_results['mean_tokens_per_sec']:.1f} tokens/sec")
    if not skip_perplexity:
        print(f"Perplexity:   {results['perplexity']['perplexity']:.2f}")
    if results['vs_baseline']:
        print(f"Speedup:      {results['vs_baseline']['speedup']['ttft']:.2f}x TTFT, {results['vs_baseline']['speedup']['throughput']:.2f}x throughput")
    print("=" * 80)

    # Save results
    if output_file:
        with open(output_file, 'w') as f:
            # Convert numpy types to Python types for JSON
            def convert(obj):
                if isinstance(obj, np.floating):
                    return float(obj)
                if isinstance(obj, np.integer):
                    return int(obj)
                if isinstance(obj, dict):
                    return {k: convert(v) for k, v in obj.items()}
                if isinstance(obj, list):
                    return [convert(v) for v in obj]
                return obj

            json.dump(convert(results), f, indent=2)
        print(f"\nResults saved to: {output_file}")

    return results


# ============================================================================
# CLI
# ============================================================================

if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        description='Complete ternary model inference benchmark',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # List available models
  python benchmark_ternary_models.py --list

  # Run full benchmark on smallest model
  python benchmark_ternary_models.py --model mmfreelm-370m

  # Run benchmark and save results
  python benchmark_ternary_models.py --model mmfreelm-370m --save results.json

  # Quick benchmark without perplexity and baseline comparison
  python benchmark_ternary_models.py --model mmfreelm-370m --skip-ppl --no-baseline

  # Benchmark with specific batch sizes
  python benchmark_ternary_models.py --model bitnet-3b --batch-sizes 1,8,32,128
"""
    )

    parser.add_argument('--model', type=str, default=None,
                        help='Model to benchmark (see --list for options)')
    parser.add_argument('--list', action='store_true',
                        help='List available models')
    parser.add_argument('--threads', type=int, default=None,
                        help='Number of threads')
    parser.add_argument('--max-tokens', type=int, default=50,
                        help='Max tokens to generate (default: 50)')
    parser.add_argument('--runs', type=int, default=5,
                        help='Number of benchmark runs (default: 5)')
    parser.add_argument('--batch-sizes', type=str, default='1,8,32',
                        help='Comma-separated batch sizes (default: 1,8,32)')
    parser.add_argument('--skip-ppl', action='store_true',
                        help='Skip perplexity benchmark')
    parser.add_argument('--no-baseline', action='store_true',
                        help='Skip PyTorch baseline comparison')
    parser.add_argument('--save', type=str, default=None,
                        help='Save results to JSON file')

    args = parser.parse_args()

    if args.list:
        list_models()
    elif args.model:
        batch_sizes = [int(x) for x in args.batch_sizes.split(',')]
        run_full_benchmark(
            model_key=args.model,
            num_threads=args.threads,
            max_tokens=args.max_tokens,
            num_runs=args.runs,
            batch_sizes=batch_sizes,
            compare_baseline=not args.no_baseline,
            skip_perplexity=args.skip_ppl,
            output_file=args.save
        )
    else:
        print("No model specified. Use --list to see available models.")
        print("Example: python benchmark_ternary_models.py --model mmfreelm-370m")
